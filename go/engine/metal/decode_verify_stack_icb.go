// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"sync/atomic"
	"time"

	core "dappco.re/go"
	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/metal"
	"github.com/tmc/apple/objc"
)

// decode_verify_stack_icb.go — the WHOLE-layer-stack MTP verify ICB. The
// per-layer verify-tail lane (decode_verify_icb.go) measured break-even because
// per-layer executeCommands (~33/pass) plus per-layer barrier drains cost what
// the recorded ops save; the chained per-token decode ICB's ~3µs/op economics
// come from ONE executeCommands per ~700-op stream. This lane records the
// verify fold's interior layers [1, nLayers-2] WHOLE — attention halves and
// tails — into one ICB and replays them with a single execute per verify
// block, between the live layer-0 and last-layer encodes.
//
// The edges stay live for the same reason the tail lane excluded them, widened
// to whole layers: layer 0 may read direct no-copy input views and the last
// layer may scatter to caller views and takes the per-row epilogue tier —
// binds whose identity and shape change call-to-call. Keeping them live means
// the lane changes NOTHING about the fold's I/O behaviour, and the recorded
// interior binds only session-lifetime slabs and caches, so a width's
// recording survives K wobble (the pinned I/O scratch reallocates per K; the
// fold slabs only grow).
//
// Record==live by construction: the first eligible pass at a width K runs the
// live fold encodes exactly as before, and each interior op mirrors into an
// ICB command through the same emit* bodies (dispatch_sink.go's
// one-math-two-targets discipline). Anything the mirror cannot record fails
// the WHOLE recording and the lane declines for that key — the live fold is
// never altered. The live↔ICB boundaries share one serial encoder, ordered by
// tracked-resource hazards + UseResource — the tail lane's shipped contract.
//
// Per-pass variation lives OUTSIDE the recorded commands, resolving the
// staged-sliding blocker the tail lane named:
//   - RoPE positions: the fold's packed offsets buffer (offPacked) is already
//     host-written per pass; the recorded binds point at it unchanged.
//   - SDPA live lengths: each recorded SDPA binds a dedicated 4-byte length
//     buffer the host rewrites per pass (the arch replay's nGlobalBuf shape).
//   - Cache landing/read offsets: pos-dependent binds (a global layer's
//     basePos row, a sliding layer's per-row pos%window slot, a q8 owner's
//     code+scale rows) register as rebind entries replayed through
//     setICBKernelBufferAtCommandIndexFast — prepareStepRebind's mechanism.
//   - The PLE slab: its pinned upload buffer reallocates when K changes, so
//     its binds register as rebinds re-pointed at the current slab each pass.
//
// Shape phases are part of the validity key: the sliding landing flips from
// direct to staged at basePos+K > window, and the globals' multi-query SDPA
// holds only below the 2-pass knee. A phase flip mismatches the key and the
// width re-records once under the new shape; the deep phase (globals past the
// knee) declines to the live fold rather than record the blocks-laddered
// 2-pass pair.
//
// Ordering: every recorded command but the pass's first carries a SetBarrier
// (the arch recorder's discipline), except the proven sibling overlaps — the
// K/V projections after Q barriered their shared input, and up after gate.
// Barriers are a superset of the live encoder's tracked-resource hazards, so
// the recorded serialisation is a valid schedule of the live pass and the
// per-element maths is unchanged.
//
// Scope (declines run the live fold, never an error): recorded-ICB quant
// sessions, dense or PLE archs, every layer on the batched-rows fold. MoE
// layers, TurboQuant caches, bidirectional row caps, prefill chunks, attention
// sinks, SiLU folds, runtime-dim SDPA and the deep 2-pass phase all decline.
// The banked per-layer tail lane keeps priority when armed (LTHN_VERIFY_ICB=1).

// verifyStackICBDisabled: the lane is OPT-IN (LTHN_VERIFY_STACK_ICB=1) until
// the ENGINE's own re-engagement bistability is fixed — the real-checkpoint
// parity run (TestRealE2BVerifyStackICBTokensMatchLive) flips a near-tied
// token (~1 in 2, 2480 vs 496) with the lane force-disabled in BOTH arms
// (TestRealE2BVerifyStackKVDiff LTHN_KVDIFF_BOTH_LIVE=1: 4/10, both
// directions), KV cache bytes byte-identical, heavy host-side logging
// suppressing it — a timing-sensitive nondeterminism in the LIVE MTP path
// around the plain-stretch boundary (LTHN_MTP_REENGAGE=0 → 10/10 stable),
// not a replay defect. The lane's replay is byte-faithful; the parity gate
// cannot hold against a bistable reference, so the lane waits opt-in until
// the engine flake is root-caused and the gate holds under -count=10.
//
// RECEIPT (e2b 4-bit + bf16 assistant, K=6 blocks, same prompt): the interior
// records as ~656 commands and replays with ONE execute; traced GPU per verify
// pass 9.4ms live → 8.3ms replay (stack.replay segment 7.6ms), verify-forward
// wall 10.4-11.2ms live → 10.0-10.9ms replay (~-0.4ms) after the sibling
// fence relaxations, decode tok/s unchanged (171-173 both lanes at 512
// tokens). The verify fold is NOT encode-bound at this shape: the wall is
// intrinsic batched-op time + the serial dependency chain + ~2ms of host
// wrapper (embeds, PLE slab build) outside the fold, so removing the
// per-segment encode moves little. The payoff levers are op-count reduction
// (fusion) inside the recorded interior, not execute granularity.
var verifyStackICBDisabled = os.Getenv("LTHN_VERIFY_STACK_ICB") != "1"

// verifyStackICBDisabledForTest forces the live fold encodes — the A/B lever
// for the replay parity tests. Production never sets it.
var verifyStackICBDisabledForTest bool

// verifyStackReplays counts whole-stack replay executes — the engagement
// counter for the parity tests (a parity that never engaged proves nothing).
var verifyStackReplays atomic.Int64

// verifyStackKey identifies the buffer world and shape phase a whole-stack
// recording baked in. Every slab is an identity the recorded commands bound (a
// K-growth reallocation mismatches and the width re-records once); staged and
// multiQ are the basePos-derived shape decisions that alter which ops encode.
// The PLE slab is deliberately absent — its binds rebind per pass.
type verifyStackKey struct {
	k, dModel, layers     int
	staged, multiQ        bool
	inPacked, outPacked   uintptr
	offPacked             uintptr
	// rowsDigest folds EVERY per-row buffer identity (inRows, outRows, offBuf
	// across all K rows, in order) into the key: the recording bakes each
	// row's binds, and a probe-K resize can churn rows 1..K-1 while row 0
	// survives — a single-row key then replays against stale row buffers.
	rowsDigest uintptr
	hSlab, mlpNormSlab    uintptr
	gateSlab, upSlab      uintptr
	gatedSlab, downSlab   uintptr
	attnNormSlab, qSlab   uintptr
	attnSlab, attnOutSlab uintptr
	kStage, vStage        uintptr
}

// vsRebindKind selects the per-pass offset formula for a rebound bind.
type vsRebindKind uint8

const (
	// vsRebindGlobalRow: off = basePos·stride (a global owner's landing row —
	// bf16 rows, q8 code rows and q8 scale rows differ only by stride).
	vsRebindGlobalRow vsRebindKind = iota
	// vsRebindSlideSlot: off = ((basePos+row) % slideW)·stride (a sliding
	// owner's per-row ring slot — the staged landing's write target).
	vsRebindSlideSlot
	// vsRebindPLESlab: re-point the bind at the CURRENT pass's PLE slab buffer
	// at the recorded base offset (the pinned slab reallocates when K changes).
	vsRebindPLESlab
)

type vsRebind struct {
	cmdIdx  uint
	bindIdx uint
	buf     metal.MTLBuffer
	kind    vsRebindKind
	stride  int
	row     int
	slideW  int
	baseOff uint
}

// vsScalarKind selects the per-pass value written into a dynamic length buffer.
type vsScalarKind uint8

const (
	// vsNTotal: basePos+K — the multi-query SDPA's total live length.
	vsNTotal vsScalarKind = iota
	// vsNRow: min(basePos+row+1, capW) — a per-row SDPA's live length (capW>0
	// is the sliding window; the lane records no uncapped per-row SDPA).
	vsNRow
)

type vsDynScalar struct {
	buf  metal.MTLBuffer
	ptr  *int32
	kind vsScalarKind
	row  int
	capW int
}

// verifyStackICB is the finished recording: the whole pass as one command
// range, the resident set, the per-pass rebinds and dynamic lengths, and the
// validity key.
type verifyStackICB struct {
	icb         metal.MTLIndirectCommandBuffer
	rng         foundation.NSRange
	resident    []metal.MTLResource
	residentIDs []objc.ID
	rebinds     []vsRebind
	dyn         []vsDynScalar
	key         verifyStackKey
	// pleSlabID is the slab identity the recording bound; a differing current
	// slab is re-pointed by the PLE rebinds and declared resident additionally.
	pleSlabID uintptr
	// residentScratch backs the declare list when the current PLE slab differs
	// from the recorded one (resident + the live slab, rebuilt per pass).
	residentScratch    []metal.MTLResource
	residentIDsScratch []objc.ID
}

// prepare writes this pass's dynamic lengths and re-points the pos-dependent
// binds for basePos. pleSlab is the CURRENT pass's slab buffer (nil on
// non-PLE archs). Must run before the execute that replays the range.
func (v *verifyStackICB) prepare(basePos int, pleSlab metal.MTLBuffer) {
	for i := range v.dyn {
		d := &v.dyn[i]
		switch d.kind {
		case vsNTotal:
			*d.ptr = int32(basePos + v.key.k)
		case vsNRow:
			n := basePos + d.row + 1
			if d.capW > 0 && n > d.capW {
				n = d.capW
			}
			*d.ptr = int32(n)
		}
	}
	for i := range v.rebinds {
		rb := &v.rebinds[i]
		switch rb.kind {
		case vsRebindGlobalRow:
			setICBKernelBufferAtCommandIndexFast(v.icb, rb.cmdIdx, rb.buf, uint(basePos*rb.stride), rb.bindIdx)
		case vsRebindSlideSlot:
			slot := (basePos + rb.row) % rb.slideW
			setICBKernelBufferAtCommandIndexFast(v.icb, rb.cmdIdx, rb.buf, uint(slot*rb.stride), rb.bindIdx)
		case vsRebindPLESlab:
			if pleSlab != nil {
				setICBKernelBufferAtCommandIndexFast(v.icb, rb.cmdIdx, pleSlab, rb.baseOff, rb.bindIdx)
			}
		}
	}
}

// declareList returns the resident set to declare on the replay encoder,
// appending the current PLE slab when it differs from the recorded one.
func (v *verifyStackICB) declareList(pleSlab metal.MTLBuffer) ([]metal.MTLResource, []objc.ID) {
	if pleSlab == nil || bufID(pleSlab) == v.pleSlabID {
		return v.resident, v.residentIDs
	}
	n := len(v.resident) + 1
	if cap(v.residentScratch) < n {
		v.residentScratch = make([]metal.MTLResource, n)
		v.residentIDsScratch = make([]objc.ID, n)
	}
	v.residentScratch = v.residentScratch[:n]
	v.residentIDsScratch = v.residentIDsScratch[:n]
	copy(v.residentScratch, v.resident)
	copy(v.residentIDsScratch, v.residentIDs)
	v.residentScratch[n-1] = pleSlab
	v.residentIDsScratch[n-1] = pleSlab.GetID()
	return v.residentScratch, v.residentIDsScratch
}

type vsCapturedBind struct {
	buf uintptr
	off uint
	idx uint
}

// verifyStackRecorder mirrors one live fold pass into ICB commands. Any op it
// cannot record fails the WHOLE recording — a partially recorded pass would
// replay wrong bytes, never a fallback point.
type verifyStackRecorder struct {
	icb          metal.MTLIndirectCommandBuffer
	maxCmds      int
	used         int
	failed       bool
	nLayers      int
	layerEntries int // entry-rms count — finish requires one per interior layer
	inRange      bool
	resident     []metal.MTLResource
	residentSeen map[uintptr]struct{}
	rebinds      []vsRebind
	dyn          []vsDynScalar
	lastBinds    []vsCapturedBind
	overlap      bool // one-shot: next command records WITHOUT a barrier
	key          verifyStackKey
	pleSlabID    uintptr
}

// setLayer arms the mirror for interior layers only — the live edges (layer 0
// and the last layer) never record, so their I/O-view binds and per-row
// epilogue tier stay exactly the live fold's.
func (r *verifyStackRecorder) setLayer(li int) {
	if r != nil {
		r.inRange = li >= 1 && li <= r.nLayers-2
	}
}

// newVerifyStackRecorder sizes the ICB for the widest pass shape: per layer
// the attention half (entry rms, three projections, rope rows, landing +
// per-row rope/norm/SDPA) plus the tail (o-proj, residuals, MLP, PLE chain,
// scalar) bounded by 26+3K commands.
func newVerifyStackRecorder(nLayers, k int, key verifyStackKey, pleSlab metal.MTLBuffer) *verifyStackRecorder {
	maxCmds := nLayers*(26+3*k) + 32
	icbDesc := metal.NewMTLIndirectCommandBufferDescriptor()
	icbDesc.SetCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch | metal.MTLIndirectCommandTypeConcurrentDispatchThreads)
	icbDesc.SetInheritBuffers(false)
	icbDesc.SetInheritPipelineState(false)
	icbDesc.SetMaxKernelBufferBindCount(16)
	icb := device.NewIndirectCommandBufferWithDescriptorMaxCommandCountOptions(icbDesc, uint(maxCmds), metal.MTLResourceStorageModeShared)
	if icb == nil {
		return nil
	}
	return &verifyStackRecorder{
		icb:          icb,
		maxCmds:      maxCmds,
		nLayers:      nLayers,
		resident:     make([]metal.MTLResource, 0, nLayers*48+64),
		residentSeen: make(map[uintptr]struct{}, nLayers*48+64),
		key:          key,
		pleSlabID:    bufID(pleSlab),
	}
}

func (r *verifyStackRecorder) active() bool { return r != nil && !r.failed && r.inRange }

func (r *verifyStackRecorder) fail() {
	if r != nil {
		r.failed = true
	}
}

// overlapNext records the next command WITHOUT a barrier — for an independent
// sibling of a producer whose first consumer already barriered it (K/V
// projections after Q, up after gate — the arch recorder's proven set).
// LTHN_VERIFY_STACK_ALLBARRIERS=1 defeats every relaxation (the race-hunt
// instrument: full-barrier recordings isolate whether a claimed-disjoint
// sibling pair actually collides).
func (r *verifyStackRecorder) overlapNext() {
	if r != nil && !verifyStackAllBarriers {
		r.overlap = true
	}
}

var verifyStackAllBarriers = os.Getenv("LTHN_VERIFY_STACK_ALLBARRIERS") == "1"

var verifyStackRowHashArmed = os.Getenv("LTHN_VERIFY_STACK_ROWHASH") == "1"

func (r *verifyStackRecorder) addResident(b metal.MTLBuffer) {
	if b == nil {
		return
	}
	id := uintptr(b.GetID())
	if id == 0 {
		return
	}
	if _, ok := r.residentSeen[id]; ok {
		return
	}
	r.residentSeen[id] = struct{}{}
	r.resident = append(r.resident, b)
}

func (r *verifyStackRecorder) captureBind(buf metal.MTLBuffer, off, idx uint) {
	r.addResident(buf)
	r.lastBinds = append(r.lastBinds, vsCapturedBind{buf: bufID(buf), off: off, idx: idx})
}

// nextCmd allocates the next ICB command. Every command but the pass's first
// carries a SetBarrier unless overlapNext armed the sibling relaxation.
func (r *verifyStackRecorder) nextCmd() (metal.MTLIndirectComputeCommand, bool) {
	if r.failed {
		return nil, false
	}
	if r.used >= r.maxCmds {
		r.fail()
		return nil, false
	}
	c := indirectComputeCommandAtIndexFast(r.icb, uint(r.used))
	if r.used > 0 && !r.overlap {
		setICBBarrier(c)
	}
	r.overlap = false
	r.used++
	r.lastBinds = r.lastBinds[:0]
	return c, true
}

// markRebindLast registers every bind of the JUST-recorded command matching
// (buf, off) as a per-pass rebind. An op whose expected pos-dependent bind is
// absent fails the recording — the drift guard against emit-ABI changes.
func (r *verifyStackRecorder) markRebindLast(buf metal.MTLBuffer, off uint, kind vsRebindKind, stride, row, slideW int) {
	if r == nil || r.failed {
		return
	}
	id := bufID(buf)
	found := false
	for _, b := range r.lastBinds {
		if b.buf == id && b.off == off {
			r.rebinds = append(r.rebinds, vsRebind{
				cmdIdx: uint(r.used - 1), bindIdx: b.idx, buf: buf,
				kind: kind, stride: stride, row: row, slideW: slideW, baseOff: off,
			})
			found = true
		}
	}
	if !found {
		r.fail()
	}
}

// markPLESlabLast registers the just-recorded command's PLE slab bind for
// per-pass re-pointing (the pinned slab reallocates whenever K changes).
func (r *verifyStackRecorder) markPLESlabLast(pleSlab metal.MTLBuffer, off uint) {
	if r == nil || r.failed || pleSlab == nil {
		return
	}
	id := bufID(pleSlab)
	found := false
	for _, b := range r.lastBinds {
		if b.buf == id && b.off == off {
			r.rebinds = append(r.rebinds, vsRebind{
				cmdIdx: uint(r.used - 1), bindIdx: b.idx, kind: vsRebindPLESlab, baseOff: off,
			})
			found = true
		}
	}
	if !found {
		r.fail()
	}
}

// dynI32 allocates a dedicated per-pass length buffer (the arch replay's
// nGlobalBuf shape: the host rewrites the value, the recorded bind is fixed).
func (r *verifyStackRecorder) dynI32(kind vsScalarKind, row, capW int) metal.MTLBuffer {
	if r == nil || r.failed {
		return nil
	}
	b := device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared)
	if b == nil {
		r.fail()
		return nil
	}
	r.addResident(b)
	r.dyn = append(r.dyn, vsDynScalar{buf: b, ptr: (*int32)(b.Contents()), kind: kind, row: row, capW: capW})
	return b
}

// layerEntry counts a layer's entry rms — finish requires exactly one per
// layer, the whole-stack completeness guard.
func (r *verifyStackRecorder) layerEntry() {
	if r != nil {
		r.layerEntries++
	}
}

// finish returns the finished recording, or nil when it failed or is
// incomplete (fewer entry-rms records than interior layers means a layer took
// an unmirrored path).
func (r *verifyStackRecorder) finish() *verifyStackICB {
	if r == nil {
		return nil
	}
	if os.Getenv("LTHN_VERIFY_STACK_DEBUG") != "" {
		nativeTraceLog(core.Sprintf("verify-stack record finish: k=%d failed=%v used=%d entries=%d/%d\n", r.key.k, r.failed, r.used, r.layerEntries, r.nLayers-2))
	}
	if r.failed || r.used == 0 || r.layerEntries != r.nLayers-2 {
		return nil
	}
	return &verifyStackICB{
		icb:         r.icb,
		rng:         foundation.NSRange{Location: 0, Length: uint(r.used)},
		resident:    r.resident,
		residentIDs: resourceIDsForFastUse(nil, r.resident),
		rebinds:     r.rebinds,
		dyn:         r.dyn,
		key:         r.key,
		pleSlabID:   r.pleSlabID,
	}
}

// vsRecordSink records into an ICB command (fastICBSink semantics), captures
// every buffer bind for markRebindLast, and accumulates the resident set.
type vsRecordSink struct {
	cmd metal.MTLIndirectComputeCommand
	rec *verifyStackRecorder
}

func (s vsRecordSink) setPSO(pso metal.MTLComputePipelineState) { setICBPSO(s.cmd, pso) }
func (s vsRecordSink) setBuf(buf metal.MTLBuffer, off, idx uint) {
	s.rec.captureBind(buf, off, idx)
	setICBKernelBuffer(s.cmd, buf, off, idx)
}
func (s vsRecordSink) setI32(v int32, idx uint) {
	b := scalarI32(v)
	s.rec.addResident(b)
	setICBKernelBuffer(s.cmd, b, 0, idx)
}
func (s vsRecordSink) setI64(v int64, idx uint) {
	b := scalarI64(v)
	s.rec.addResident(b)
	setICBKernelBuffer(s.cmd, b, 0, idx)
}
func (s vsRecordSink) setF32(v float32, idx uint) {
	b := scalarF32(v)
	s.rec.addResident(b)
	setICBKernelBuffer(s.cmd, b, 0, idx)
}
func (s vsRecordSink) dispatchThreads(grid, group metal.MTLSize) {
	concurrentDispatchThreads(s.cmd, grid, group)
}
func (s vsRecordSink) dispatchThreadgroups(grid, group metal.MTLSize) {
	concurrentDispatchThreadgroups(s.cmd, grid, group)
}

// ---- ICB-capable pipelines for the attention-half kernels the tail lane
// never recorded (custom library; supportIndirectCommandBuffers required) ----

func customPipelineICB(name string) (metal.MTLComputePipelineState, error) {
	key := name + "|icb"
	icbPSOMu.Lock()
	defer icbPSOMu.Unlock()
	if pso, ok := icbPSOCache[key]; ok {
		return pso, nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		return nil, core.NewError("native.customPipelineICB: custom library unavailable for " + name)
	}
	fn := customLibrary.NewFunctionWithName(name)
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.customPipelineICB: kernel " + name + " not found")
	}
	desc := metal.NewMTLComputePipelineDescriptor()
	desc.SetComputeFunction(fn)
	desc.SetSupportIndirectCommandBuffers(true)
	pso, err := device.NewComputePipelineStateWithDescriptorOptionsReflectionError(desc, 0, nil)
	if err != nil {
		return nil, core.E("native.customPipelineICB", name, err)
	}
	icbPSOCache[key] = pso
	return pso, nil
}

func qkNormRopeRowsPipelineICB() (metal.MTLComputePipelineState, error) {
	return customPipelineICB("lthn_qknorm_rope_rows_bf16")
}

func sdpaMultiQPipelineICBForHeadDim(headDim int) (metal.MTLComputePipelineState, error) {
	return customPipelineICB(core.Sprintf("lthn_sdpa_multiq_bf16_%d", headDim))
}

func sdpaMultiQQ8PipelineICBForHeadDim(headDim int) (metal.MTLComputePipelineState, error) {
	return customPipelineICB(core.Sprintf("lthn_sdpa_multiq_q8_bf16_%d", headDim))
}

func kvQ8StoreRowsPipelineICBVS() (metal.MTLComputePipelineState, error) {
	return customPipelineICB("lthn_kv_q8_store_rows_bf16")
}

// ---- op mirrors: each records exactly the dispatch its live enc* twin
// encoded, through the shared emit* body where one exists ----

func (r *verifyStackRecorder) recRMSRows(x, w, out metal.MTLBuffer, xOff, wOff, outOff uint, rows, axisSize int, eps float32) {
	if r == nil || r.failed {
		return
	}
	pso, err := pipelineForICB(rmsKernelBF16(axisSize))
	if err != nil {
		r.fail()
		return
	}
	c, ok := r.nextCmd()
	if !ok {
		return
	}
	emitRMSNormRows(vsRecordSink{c, r}, pso, x, w, out, xOff, wOff, outOff, axisSize, eps, rows, rmsThreadgroup(axisSize, pso))
}

func (r *verifyStackRecorder) recAdd(a metal.MTLBuffer, aOff uint, b metal.MTLBuffer, bOff uint, out metal.MTLBuffer, outOff uint, n int) {
	if r == nil || r.failed {
		return
	}
	pso, err := pipelineForICB("vv_Addbfloat16")
	if err != nil {
		r.fail()
		return
	}
	c, ok := r.nextCmd()
	if !ok {
		return
	}
	emitBinary(vsRecordSink{c, r}, pso, a, aOff, b, bOff, out, outOff, n)
}

// recResidualRowsMaybeNorm mirrors encResidualRowsMaybeNorm: one add, or
// norm-rows + add.
func (r *verifyStackRecorder) recResidualRowsMaybeNorm(x metal.MTLBuffer, xOff uint, v metal.MTLBuffer, vOff uint, scratch, out metal.MTLBuffer, outOff uint, norm bufView, rows, dModel int, eps float32) {
	if r == nil || r.failed {
		return
	}
	if norm.buf == nil {
		r.recAdd(x, xOff, v, vOff, out, outOff, rows*dModel)
		return
	}
	r.recRMSRows(v, norm.buf, scratch, vOff, norm.off, 0, rows, dModel, eps)
	r.recAdd(x, xOff, scratch, 0, out, outOff, rows*dModel)
}

func (r *verifyStackRecorder) recGeluGateMul(gate, up, out metal.MTLBuffer, gateOff, upOff, outOff uint, n int) {
	if r == nil || r.failed {
		return
	}
	pso, err := geluPipelineICB()
	if err != nil {
		r.fail()
		return
	}
	c, ok := r.nextCmd()
	if !ok {
		return
	}
	emitBinary(vsRecordSink{c, r}, pso, gate, gateOff, up, upOff, out, outOff, n)
}

func (r *verifyStackRecorder) recMulRows(a, b, out metal.MTLBuffer, aOff, bOff, outOff uint, rows, rowLen int) {
	if r == nil || r.failed {
		return
	}
	pso, err := mulRowsPipelineICB()
	if err != nil {
		r.fail()
		return
	}
	c, ok := r.nextCmd()
	if !ok {
		return
	}
	emitMulRows(vsRecordSink{c, r}, pso, a, b, out, aOff, bOff, outOff, rows, rowLen)
}

func (r *verifyStackRecorder) recRMSNormResidualRows(x, weight, res, out metal.MTLBuffer, xOff, wOff, resOff, outOff uint, rows, axisSize int, eps float32) {
	if r == nil || r.failed {
		return
	}
	pso, err := rmsResidualPipelineICB(axisSize)
	if err != nil {
		r.fail()
		return
	}
	c, ok := r.nextCmd()
	if !ok {
		return
	}
	emitRMSNormResidualRows(vsRecordSink{c, r}, pso, x, weight, res, out, xOff, wOff, resOff, outOff, rows, axisSize, eps, rmsThreadgroup(axisSize, pso))
}

func (r *verifyStackRecorder) recPLEGateGeluRows(gatePacked, gateScales, gateBiases bufView, x metal.MTLBuffer, xOff uint, ple metal.MTLBuffer, pleOff uint, gated metal.MTLBuffer, gatedOff uint, rows, dModel, pliDim, gs, bits int) {
	if r == nil || r.failed {
		return
	}
	pso, ok := lthnPLEGateGeluPipelineICB(bits)
	if !ok {
		r.fail()
		return
	}
	c, okc := r.nextCmd()
	if !okc {
		return
	}
	emitPLEGateGeluRows(vsRecordSink{c, r}, pso, gatePacked, gateScales, gateBiases, x, xOff, ple, pleOff, gated, gatedOff, rows, dModel, pliDim, gs)
}

func (r *verifyStackRecorder) recQMMT(wq, scales, biases, x, out metal.MTLBuffer, wqOff, scalesOff, biasesOff, xOff, outOff uint, m, outDim, inDim, gs, bits int) {
	if r == nil || r.failed {
		return
	}
	pso, err := pipelineForICB(qmmTKernelName(outDim, gs, bits))
	if err != nil {
		r.fail()
		return
	}
	c, ok := r.nextCmd()
	if !ok {
		return
	}
	emitQMMT(vsRecordSink{c, r}, pso, wq, wqOff, scales, scalesOff, biases, biasesOff, x, xOff, out, outOff, m, outDim, inDim)
}

func (r *verifyStackRecorder) recGemvBatched(mat metal.MTLBuffer, matOff uint, vec metal.MTLBuffer, vecOff uint, out metal.MTLBuffer, outOff uint, outDim, inDim, batch int) {
	if r == nil || r.failed {
		return
	}
	if batch >= steelGEMMMinRows {
		r.fail() // the steel-GEMM route never applies at verify K; do not record it blind
		return
	}
	bm, bn, sm, sn, tm, tn := gemvTiles(inDim, outDim)
	pso, err := pipelineForICB(gemvKernelName("bfloat16", bm, bn, sm, sn, tm, tn))
	if err != nil {
		r.fail()
		return
	}
	c, ok := r.nextCmd()
	if !ok {
		return
	}
	emitGemvBatchedVecAt(vsRecordSink{c, r}, pso, mat, matOff, vec, vecOff, out, outOff, inDim, outDim, batch, bm, bn, sm, tm)
}

// recProjectRows mirrors projector.projectRows for the concrete projector
// types the dense fold drives. Unknown projector types fail the recording.
func (r *verifyStackRecorder) recProjectRows(proj projector, in, out metal.MTLBuffer, inOff, outOff uint, rows int, p projIndex) {
	if r == nil || r.failed {
		return
	}
	switch m := proj.(type) {
	case qmvProjector:
		r.recQMVProjectRows(m, in, out, inOff, outOff, rows, p)
	case bf16Projector:
		w, outDim, inDim, ok := m.weightDims(p)
		if !ok || w.buf == nil {
			r.fail()
			return
		}
		r.recGemvBatched(w.buf, w.off, in, inOff, out, outOff, outDim, inDim, rows)
	default:
		r.fail()
	}
}

func (r *verifyStackRecorder) recQMVProjectRows(m qmvProjector, in, out metal.MTLBuffer, inOff, outOff uint, rows int, p projIndex) {
	w, outDim, inDim, ok := m.weightDims(p)
	if !ok || !w.present() {
		r.fail()
		return
	}
	if w.dense() {
		r.recGemvBatched(w.wq.buf, w.wq.off, in, inOff, out, outOff, outDim, inDim, rows)
		return
	}
	gs, bits := m.groupSize, m.bits
	if w.bits > 0 {
		gs, bits = w.gs, w.bits
	}
	if plan, ok := qmvRowsPlanFor(rows, outDim, inDim, gs, bits); ok {
		if plan.tiled {
			pso, ok := lthnQMVRowsPipelineICB(plan.tiledKey)
			if !ok {
				r.fail()
				return
			}
			c, okc := r.nextCmd()
			if !okc {
				return
			}
			emitQMVRowsTiled(vsRecordSink{c, r}, pso, w.wq.buf, w.wq.off, w.scales.buf, w.scales.off, w.biases.buf, w.biases.off, in, inOff, out, outOff, inDim, outDim)
			return
		}
		pso, ok := lthnGatherQMVPipelineICB(plan.gatherKey)
		if !ok {
			r.fail()
			return
		}
		lhs, rhs, okb := qmvRowsIndexBuffers()
		if !okb {
			r.fail()
			return
		}
		c, okc := r.nextCmd()
		if !okc {
			return
		}
		emitLthnGatherQMVRoutes(vsRecordSink{c, r}, pso, in, inOff, w.wq.buf, w.wq.off, w.scales.buf, w.scales.off, w.biases.buf, w.biases.off, lhs, rhs, 0, out, outOff, outDim, inDim, gs, bits, 0, rows)
		return
	}
	// the live path's qmm_t fallback (K%gs==0 checked by rowsCapable upfront)
	if inDim <= 0 || gs <= 0 || inDim%gs != 0 {
		r.fail()
		return
	}
	r.recQMMT(w.wq.buf, w.scales.buf, w.biases.buf, in, out, w.wq.off, w.scales.off, w.biases.off, inOff, outOff, rows, outDim, inDim, gs, bits)
}

// recQKNormRopeRows mirrors encQKNormRopeRows — the batched fused per-head
// QK-norm + RoPE (positions from the packed offsets buffer, host-written per
// pass, so the recorded bind replays unchanged).
func (r *verifyStackRecorder) recQKNormRopeRows(x, w, out metal.MTLBuffer, xOff, wOff, outOff uint, xRowStride, outRowStride int, offBuf, periods metal.MTLBuffer, rows, nHeads, headDim, rotaryDim int, base, scale, eps float32) {
	if r == nil || r.failed {
		return
	}
	pso, err := qkNormRopeRowsPipelineICB()
	if err != nil {
		r.fail()
		return
	}
	rd := headDim
	if rotaryDim > 0 && rotaryDim < headDim {
		rd = rotaryDim
	}
	c, ok := r.nextCmd()
	if !ok {
		return
	}
	sink := vsRecordSink{c, r}
	sink.setPSO(pso)
	sink.setBuf(x, xOff, 0)
	sink.setBuf(w, wOff, 1)
	sink.setBuf(out, outOff, 2)
	sink.setF32(eps, 3)
	sink.setI32(int32(headDim), 4)
	sink.setI32(int32(rd), 5)
	sink.setF32(scale, 6)
	sink.setBuf(offBuf, 0, 7)
	sink.setF32(log2F32(base), 8)
	if periods != nil {
		sink.setBuf(periods, 0, 9)
		sink.setI32(1, 10)
	} else {
		sink.setBuf(qkRopeDummyBuf(), 0, 9)
		sink.setI32(0, 10)
	}
	sink.setI32(int32(xRowStride), 11)
	sink.setI32(int32(outRowStride), 12)
	sink.dispatchThreads(
		metal.MTLSize{Width: uint(nHeads * headDim), Height: uint(rows), Depth: 1},
		metal.MTLSize{Width: uint(headDim), Height: 1, Depth: 1},
	)
}

// recQKNormRopeAt mirrors the per-row fused QK-norm + RoPE (the staged sliding
// landing writes each row into its ring slot).
func (r *verifyStackRecorder) recQKNormRopeAt(x, w, out metal.MTLBuffer, xOff, wOff, outOff uint, offBuf metal.MTLBuffer, offOff uint, periods metal.MTLBuffer, nHeads, headDim, rotaryDim int, base, scale, eps float32) {
	if r == nil || r.failed {
		return
	}
	pso, err := qkNormRopePipelineICB()
	if err != nil {
		r.fail()
		return
	}
	rd := headDim
	if rotaryDim > 0 && rotaryDim < headDim {
		rd = rotaryDim
	}
	c, ok := r.nextCmd()
	if !ok {
		return
	}
	emitQKNormRopeAt(vsRecordSink{c, r}, pso, x, w, out, xOff, wOff, outOff, offBuf, offOff, periods, qkRopeDummyBuf(), nHeads, headDim, rd, eps, scale, log2F32(base))
}

// recKVQ8StoreRows mirrors encKVQ8StoreRows — the q8 owner's staged K-row
// quantise landing (the cache and scale binds carry the batch-base offsets and
// register as per-pass rebinds at the call site).
func (r *verifyStackRecorder) recKVQ8StoreRows(stage, cache metal.MTLBuffer, cacheOff uint, scales metal.MTLBuffer, scaleOff uint, rows, kvDim int) {
	if r == nil || r.failed {
		return
	}
	pso, err := kvQ8StoreRowsPipelineICBVS()
	if err != nil {
		r.fail()
		return
	}
	if kvDim <= 0 || kvDim%kvQ8GroupSize != 0 {
		r.fail()
		return
	}
	c, ok := r.nextCmd()
	if !ok {
		return
	}
	sink := vsRecordSink{c, r}
	sink.setPSO(pso)
	sink.setBuf(stage, 0, 0)
	sink.setBuf(cache, cacheOff, 1)
	sink.setBuf(scales, scaleOff, 2)
	sink.setI32(int32(kvDim), 3)
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: uint(kvDim / kvQ8GroupSize), Height: uint(rows), Depth: 1},
		metal.MTLSize{Width: 32, Height: 1, Depth: 1},
	)
}

// recSDPARow mirrors the per-row single-pass SDPA a staged sliding row runs.
// The live length binds a dynamic buffer (min(basePos+row+1, capW) per pass);
// the lane never records an uncapped or 2-pass-eligible row — a window at or
// past the knee fails the recording.
func (r *verifyStackRecorder) recSDPARow(headDim int, q metal.MTLBuffer, qOff uint, k, v, out metal.MTLBuffer, outOff uint, row, capW, nHeads, nKVHeads int, kHeadStride, kSeqStride, vHeadStride, vSeqStride int64, scale float32) {
	if r == nil || r.failed {
		return
	}
	if capW <= 0 || capW >= sdpa2PassMinKV {
		r.fail()
		return
	}
	pso, err := sdpaVectorPipelineICBForHeadDim(headDim)
	if err != nil {
		r.fail()
		return
	}
	nBuf := r.dynI32(vsNRow, row, capW)
	if nBuf == nil {
		return
	}
	c, ok := r.nextCmd()
	if !ok {
		return
	}
	emitSDPAAt(vsRecordSink{c, r}, pso, q, qOff, k, v, out, outOff, 0, nBuf, nHeads, nKVHeads, 0, kHeadStride, kSeqStride, vHeadStride, vSeqStride, scale)
}

// recSDPAMultiQ mirrors encSDPAMultiQCausal with the total live length in a
// dynamic buffer (basePos+K per pass).
func (r *verifyStackRecorder) recSDPAMultiQ(headDim int, q, k, v, out metal.MTLBuffer, nHeads, nKVHeads, kRows int, kHeadStride, kSeqStride, vHeadStride, vSeqStride int64, scale float32) {
	if r == nil || r.failed {
		return
	}
	pso, err := sdpaMultiQPipelineICBForHeadDim(headDim)
	if err != nil {
		r.fail()
		return
	}
	nBuf := r.dynI32(vsNTotal, 0, 0)
	if nBuf == nil {
		return
	}
	c, ok := r.nextCmd()
	if !ok {
		return
	}
	sink := vsRecordSink{c, r}
	sink.setPSO(pso)
	sink.setBuf(q, 0, 0)
	sink.setBuf(k, 0, 1)
	sink.setBuf(v, 0, 2)
	sink.setBuf(out, 0, 3)
	sink.setI32(int32(nHeads/nKVHeads), 4)
	sink.setBuf(nBuf, 0, 5)
	sink.setI64(kHeadStride, 6)
	sink.setI64(kSeqStride, 7)
	sink.setI64(vHeadStride, 8)
	sink.setI64(vSeqStride, 9)
	sink.setF32(scale, 10)
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: uint(nHeads), Height: uint(kRows), Depth: 1},
		metal.MTLSize{Width: 1024, Height: 1, Depth: 1},
	)
}

// recSDPAMultiQQ8 mirrors encSDPAMultiQCausalQ8 (scale planes at 11/12).
func (r *verifyStackRecorder) recSDPAMultiQQ8(headDim int, q, k, v, out, kScales, vScales metal.MTLBuffer, nHeads, nKVHeads, kRows int, kHeadStride, kSeqStride, vHeadStride, vSeqStride int64, scale float32) {
	if r == nil || r.failed {
		return
	}
	pso, err := sdpaMultiQQ8PipelineICBForHeadDim(headDim)
	if err != nil {
		r.fail()
		return
	}
	nBuf := r.dynI32(vsNTotal, 0, 0)
	if nBuf == nil {
		return
	}
	c, ok := r.nextCmd()
	if !ok {
		return
	}
	sink := vsRecordSink{c, r}
	sink.setPSO(pso)
	sink.setBuf(q, 0, 0)
	sink.setBuf(k, 0, 1)
	sink.setBuf(v, 0, 2)
	sink.setBuf(out, 0, 3)
	sink.setI32(int32(nHeads/nKVHeads), 4)
	sink.setBuf(nBuf, 0, 5)
	sink.setI64(kHeadStride, 6)
	sink.setI64(kSeqStride, 7)
	sink.setI64(vHeadStride, 8)
	sink.setI64(vSeqStride, 9)
	sink.setF32(scale, 10)
	sink.setBuf(kScales, 0, 11)
	sink.setBuf(vScales, 0, 12)
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: uint(nHeads), Height: uint(kRows), Depth: 1},
		metal.MTLSize{Width: 1024, Height: 1, Depth: 1},
	)
}

// ---- orchestration (called from the batched fold) ----

// verifyStackLaneShape reports whether this call has the session shape the
// whole-stack lane serves. Structural only — fold prerequisites and per-op
// recordability are decided by the recorder itself. The lane never touches the
// pass's I/O (the edges stay live), so every fold entry the MTP verify takes
// qualifies.
func (s *archDecodeState) verifyStackLaneShape(K int, icbSession bool) bool {
	if verifyStackICBDisabled || verifyStackICBDisabledForTest {
		return false
	}
	// the banked per-layer tail lane keeps the pass whenever it is ENABLED
	// (LTHN_VERIFY_ICB=1), including its own force-live A/B control — the
	// stack lane must never stand in for that lane's live reference.
	if !verifyTailICBDisabled {
		return false
	}
	if !s.verifyFoldSmallK || K <= 1 || K > batchedDenseICBMaxRows {
		return false
	}
	if !icbSession {
		return false // v1 serves the recorded quant session (the MTP pair target)
	}
	if len(s.specs) < 4 {
		return false // a recordable interior needs at least two layers
	}
	if s.trace || s.rowAttnCaps != nil || s.prefillSkipToLayer > 0 {
		return false
	}
	if s.prefillEmbedDevice != nil || s.prefillPLESlabDevice != nil {
		return false
	}
	if s.icb != nil && s.icb.hasKVTQ() {
		return false
	}
	for li := range s.specs {
		if s.specs[li].MoE {
			return false // the MoE router's host top-k cannot record
		}
	}
	return true
}

// verifyStackKeyFor derives this pass's validity key: the slab world plus the
// basePos-dependent shape phase (staged sliding landing, multi-query globals).
func (s *archDecodeState) verifyStackKeyFor(K, basePos int, inRows, outRows, offRows []metal.MTLBuffer, inPacked, outPacked, offPacked, hSlab, mlpNormSlab, gateSlab, upSlab, gatedSlab, downSlab, attnNormSlab, qSlab, attnSlab, attnOutSlab, kStage, vStage metal.MTLBuffer) verifyStackKey {
	digest := uintptr(2166136261)
	fold := func(rows []metal.MTLBuffer) {
		for i := 0; i < K && i < len(rows); i++ {
			digest = (digest ^ bufID(rows[i])) * 16777619
		}
	}
	fold(inRows)
	fold(outRows)
	fold(offRows)
	return verifyStackKey{
		rowsDigest: digest,
		k: K, dModel: s.dModel, layers: len(s.specs),
		staged:   s.slidingWindow > 0 && basePos+K > s.slidingWindow,
		multiQ:   basePos+K < sdpa2PassMinKV,
		inPacked: bufID(inPacked), outPacked: bufID(outPacked), offPacked: bufID(offPacked),
		hSlab: bufID(hSlab), mlpNormSlab: bufID(mlpNormSlab),
		gateSlab: bufID(gateSlab), upSlab: bufID(upSlab),
		gatedSlab: bufID(gatedSlab), downSlab: bufID(downSlab),
		attnNormSlab: bufID(attnNormSlab), qSlab: bufID(qSlab),
		attnSlab: bufID(attnSlab), attnOutSlab: bufID(attnOutSlab),
		kStage: bufID(kStage), vStage: bufID(vStage),
	}
}

// executeInto replays the recorded interior into the pass's live encoder: the
// host prologue (row setup, position writes, PLE slab upload) and the live
// layer-0 encodes have already run, so this rebinds for basePos, declares
// residency once and executes the whole interior as one range. The live↔ICB
// ordering rides the encoder's tracked-resource hazards, exactly as the tail
// lane's per-layer executes did.
func (v *verifyStackICB) executeInto(enc metal.MTLComputeCommandEncoderObject, basePos int, pleSlab metal.MTLBuffer) {
	debug := os.Getenv("LTHN_VERIFY_STACK_DEBUG") != ""
	t0 := time.Now()
	v.prepare(basePos, pleSlab)
	t1 := time.Now()
	resident, residentIDs := v.declareList(pleSlab)
	useResourcesIDsFastObject(enc, resident, residentIDs, metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
	t2 := time.Now()
	executeCommandsInBufferWithRangeObjectFast(enc, v.icb, v.rng)
	if debug {
		nativeTraceLog(core.Sprintf("verify-stack replay: k=%d basePos=%d staged=%v cmds=%d rebinds=%d dyn=%d resident=%d prep=%.2fms declare=%.2fms\n",
			v.key.k, basePos, v.key.staged, v.rng.Length, len(v.rebinds), len(v.dyn), len(resident),
			float64(t1.Sub(t0).Microseconds())/1000, float64(t2.Sub(t1).Microseconds())/1000))
	}
	verifyStackReplays.Add(1)
}

// log2F32 is the rope base transform both fused QK-norm+RoPE forms take.
func log2F32(base float32) float32 {
	return float32(math.Log2(float64(base)))
}
