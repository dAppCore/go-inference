// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"sync/atomic"

	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/metal"
	"github.com/tmc/apple/objc"
)

// decode_verify_icb.go — the MTP verify-tail ICB (#372's remaining lever): the
// verify fold's per-layer tail from the O-projection through the layer scalar is
// POSITION-INDEPENDENT — every buffer it binds is a retained slab whose identity
// is stable across verify calls at a fixed K — so it records ONCE into an
// indirect command buffer and replays per layer between the live attention
// encodes. The attention half (rope positions, SDPA live length, K/V landing
// offsets) stays live. Arithmetic from the wall/GPU split probe: a recorded op
// drains ~3µs GPU-side vs ~8-11µs re-encoded; the tail is ~14 ops × (nLayers-2),
// the bulk of the verify's ~9.7ms wall on e2b.
//
// Record==live by construction: the FIRST eligible verify pass runs the live
// tail encodes exactly as before, and a recorder mirrors each op into ICB
// commands through the same emit* bodies (dispatch_sink.go's one-math-two-targets
// discipline). From the second pass on, each recorded layer's tail is ONE
// executeCommands range. Layers 0 (input rows may be direct no-copy views) and
// L-1 (output rows may scatter to caller views) always stay live — the recorded
// set is [1, L-2], ~94% of the tail.
//
// Ordering: in-range dependencies carry recorded SetBarriers (every command but
// the range's first); the live↔ICB boundaries on one serial encoder are ordered
// by tracked-resource hazards + UseResource — the contract the chained decode
// lane (embed gather → ICB body → head → argmax, one encoder) ships on.
//
// Validity: a key of the recorded buffer identities (ping-pong rows, fold slabs,
// PLE slab) plus K/dModel. A verify with a different K (a truncated final draft
// block) or a reallocated slab simply misses the key and runs live — the ICB is
// kept for the next matching block. LTHN_VERIFY_ICB=0 kills the lane (the
// reproducibility anchor); the recording also disarms wholesale if any op in
// range cannot record (missing ICB pipeline, unexpected branch shape).

// verifyTailICBDisabled: the lane is OPT-IN (LTHN_VERIFY_ICB=1) — the live A/B
// measured it break-even, not a win (see the receipt below), so production
// keeps the proven live tail encodes and the mechanism stays for the
// whole-layer follow-up.
//
// RECEIPT (e2b 4-bit + bf16 assistant, K=6 blocks, same prompt):
// live verify fwd 10.3-10.9ms; tail-replay 10.8-11.2ms after the declare-once
// residency fix (12.1-12.2 before it), recording pass +2ms. The per-layer
// executeCommands (~33/pass) + the recorded all-prior barrier drains
// (~13/layer) cost what the recorded ops save — the chained decode lane's
// ~3µs/op economics come from ONE execute per ~700-op token, not one per
// 14-op layer tail. The winning shape is recording the WHOLE layer stack
// (attention + tail, one execute per verify pass, pos/N rebinds) — blocked on
// the staged-sliding landing's per-pass slot offsets, priced separately.
var verifyTailICBDisabled = os.Getenv("LTHN_VERIFY_ICB") != "1"

// verifyTailICBDisabledForTest forces the live tail encodes — the A/B lever for
// the replay parity tests. Production never sets it.
var verifyTailICBDisabledForTest bool

// verifyTailReplays counts recorded-range executions — the engagement counter
// for the parity tests (a parity that never engaged the lane proves nothing).
var verifyTailReplays atomic.Int64

// verifyTailKey identifies the buffer world a verify-tail recording baked in.
// Every field is an identity the recorded commands bound; a mismatch at replay
// time means the recording no longer describes this call — run live.
type verifyTailKey struct {
	k, dModel           int
	inPacked, outPacked uintptr // the ping-pong row slabs (denseBatch.rows)
	hSlab, mlpNormSlab  uintptr // the MLP-fold slabs (denseBatch.mlpFold)
	gateSlab, upSlab    uintptr
	gatedSlab, downSlab uintptr
	attnNormSlab        uintptr // the attention-fold slabs the tail touches
	attnSlab, attnOut   uintptr
	pleSlab             uintptr // 0 for non-PLE archs
}

func bufID(b metal.MTLBuffer) uintptr {
	if b == nil {
		return 0
	}
	return uintptr(b.GetID())
}

// verifyTailICB is the finished recording: the ICB, one command range per layer
// (Length 0 = layer not recorded → live), the resident set the replay encoder
// must declare, and the validity key.
type verifyTailICB struct {
	icb         metal.MTLIndirectCommandBuffer
	ranges      []foundation.NSRange
	resident    []metal.MTLResource
	residentIDs []objc.ID
	key         verifyTailKey
	// declaredEnc is the encoder the resident set was last declared on — one
	// UseResource per encoder, not per execute (see execute).
	declaredEnc uintptr
}

// replayable reports whether layer li has a recorded range under a matching key.
func (v *verifyTailICB) replayable(li int, key verifyTailKey) bool {
	return v != nil && v.icb != nil && li >= 0 && li < len(v.ranges) &&
		v.ranges[li].Length > 0 && v.key == key
}

// execute replays layer li's recorded tail into the live encoder. Residency is
// declared ONCE per encoder (the production fine-grained replay's shape — one
// UseResource, many range executes): a per-execute declaration re-processed
// ~70 resources × nLayers per verify pass and cost more than the replay saved
// (the first live A/B measured the lane SLOWER by ~1.5ms/verify). declaredEnc
// tracks the encoder identity — the GPU-trace checkpoints rotate encoders
// mid-pass, and a rotated encoder needs its own declaration.
func (v *verifyTailICB) execute(enc metal.MTLComputeCommandEncoderObject, li int) {
	if id := uintptr(enc.GetID()); id != v.declaredEnc {
		useResourcesIDsFastObject(enc, v.resident, v.residentIDs, metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
		v.declaredEnc = id
	}
	executeCommandsInBufferWithRangeObjectFast(enc, v.icb, v.ranges[li])
	verifyTailReplays.Add(1)
}

// verifyTailRecorder mirrors one live verify pass's tail encodes into ICB
// commands. Any op it cannot record fails the WHOLE recording (failed=true) —
// a partially-recorded model would split the win for permanent key complexity.
type verifyTailRecorder struct {
	icb          metal.MTLIndirectCommandBuffer
	maxCmds      int
	used         int
	ranges       []foundation.NSRange
	resident     []metal.MTLResource
	residentSeen map[uintptr]struct{}
	curLayer     int // -1 outside a layer
	layerStart   int
	failed       bool
	key          verifyTailKey
}

// newVerifyTailRecorder sizes the ICB for the worst-case tail (16 commands per
// layer covers the widest branch: composed PLE chain + rms/add pairs + scalar).
func newVerifyTailRecorder(nLayers int, key verifyTailKey) *verifyTailRecorder {
	maxCmds := nLayers * 16
	icbDesc := metal.NewMTLIndirectCommandBufferDescriptor()
	icbDesc.SetCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch | metal.MTLIndirectCommandTypeConcurrentDispatchThreads)
	icbDesc.SetInheritBuffers(false)
	icbDesc.SetInheritPipelineState(false)
	icbDesc.SetMaxKernelBufferBindCount(16)
	icb := device.NewIndirectCommandBufferWithDescriptorMaxCommandCountOptions(icbDesc, uint(maxCmds), metal.MTLResourceStorageModeShared)
	if icb == nil {
		return nil
	}
	return &verifyTailRecorder{
		icb:          icb,
		maxCmds:      maxCmds,
		ranges:       make([]foundation.NSRange, nLayers),
		resident:     make([]metal.MTLResource, 0, 64),
		residentSeen: make(map[uintptr]struct{}, 64),
		curLayer:     -1,
		key:          key,
	}
}

func (r *verifyTailRecorder) fail() {
	if r != nil {
		r.failed = true
	}
}

// recording reports whether layer li's tail ops should mirror into the ICB.
func (r *verifyTailRecorder) recording(li int) bool {
	return r != nil && !r.failed && r.curLayer == li
}

// beginLayer opens layer li's command range. Layers outside [1, nLayers-2] are
// never recorded (input-view / output-scatter variability lives at the edges).
func (r *verifyTailRecorder) beginLayer(li int) {
	if r == nil || r.failed || li < 1 || li >= len(r.ranges)-1 {
		return
	}
	r.curLayer = li
	r.layerStart = r.used
}

// endLayer closes layer li's range. A layer that recorded nothing keeps
// Length 0 (live at replay).
func (r *verifyTailRecorder) endLayer(li int) {
	if r == nil || r.curLayer != li {
		return
	}
	if !r.failed && r.used > r.layerStart {
		r.ranges[li] = foundation.NSRange{Location: uint(r.layerStart), Length: uint(r.used - r.layerStart)}
	}
	r.curLayer = -1
}

// finish returns the finished ICB, or nil when the recording failed or captured
// nothing.
func (r *verifyTailRecorder) finish() *verifyTailICB {
	if r == nil || r.failed || r.used == 0 {
		return nil
	}
	return &verifyTailICB{
		icb:         r.icb,
		ranges:      r.ranges,
		resident:    r.resident,
		residentIDs: resourceIDsForFastUse(nil, r.resident),
		key:         r.key,
	}
}

func (r *verifyTailRecorder) addResident(b metal.MTLBuffer) {
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

// nextCmd allocates the next ICB command. Every command but the first of a
// layer's range carries a SetBarrier — the recorded in-range dependency chain
// (the production recorder's pattern); the range's first command is ordered
// against the preceding live attention dispatches by the encoder's tracked-
// resource hazards.
func (r *verifyTailRecorder) nextCmd() (metal.MTLIndirectComputeCommand, bool) {
	if r.failed {
		return nil, false
	}
	if r.used >= r.maxCmds {
		r.fail()
		return nil, false
	}
	c := indirectComputeCommandAtIndexFast(r.icb, uint(r.used))
	if r.used > r.layerStart {
		setICBBarrier(c)
	}
	r.used++
	return c, true
}

// vtRecordSink is the recorder's dispatchSink: it records into an ICB command
// (fastICBSink semantics) AND accumulates every bound buffer — including the
// memoised scalar buffers — into the recorder's resident set.
type vtRecordSink struct {
	cmd metal.MTLIndirectComputeCommand
	rec *verifyTailRecorder
}

func (s vtRecordSink) setPSO(pso metal.MTLComputePipelineState) { setICBPSO(s.cmd, pso) }
func (s vtRecordSink) setBuf(buf metal.MTLBuffer, off, idx uint) {
	s.rec.addResident(buf)
	setICBKernelBuffer(s.cmd, buf, off, idx)
}
func (s vtRecordSink) setI32(v int32, idx uint) {
	b := scalarI32(v)
	s.rec.addResident(b)
	setICBKernelBuffer(s.cmd, b, 0, idx)
}
func (s vtRecordSink) setI64(v int64, idx uint) {
	b := scalarI64(v)
	s.rec.addResident(b)
	setICBKernelBuffer(s.cmd, b, 0, idx)
}
func (s vtRecordSink) setF32(v float32, idx uint) {
	b := scalarF32(v)
	s.rec.addResident(b)
	setICBKernelBuffer(s.cmd, b, 0, idx)
}
func (s vtRecordSink) dispatchThreads(grid, group metal.MTLSize) {
	concurrentDispatchThreads(s.cmd, grid, group)
}
func (s vtRecordSink) dispatchThreadgroups(grid, group metal.MTLSize) {
	concurrentDispatchThreadgroups(s.cmd, grid, group)
}

// ---- op mirrors: each records exactly the dispatch its live enc* twin encoded ----

// recordProjectRows mirrors projector.projectRows for the concrete projector
// types the dense fold drives. Unknown projector types fail the recording.
func (r *verifyTailRecorder) recordProjectRows(proj projector, in, out metal.MTLBuffer, inOff, outOff uint, rows int, p projIndex) {
	if r.failed {
		return
	}
	switch m := proj.(type) {
	case qmvProjector:
		r.recordQMVProjectRows(m, in, out, inOff, outOff, rows, p)
	case bf16Projector:
		w, outDim, inDim, ok := m.weightDims(p)
		if !ok || w.buf == nil {
			r.fail()
			return
		}
		r.recordGemvBatched(w.buf, w.off, in, inOff, out, outOff, outDim, inDim, rows)
	default:
		r.fail()
	}
}

func (r *verifyTailRecorder) recordQMVProjectRows(m qmvProjector, in, out metal.MTLBuffer, inOff, outOff uint, rows int, p projIndex) {
	w, outDim, inDim, ok := m.weightDims(p)
	if !ok || !w.present() {
		r.fail()
		return
	}
	if w.dense() {
		r.recordGemvBatched(w.wq.buf, w.wq.off, in, inOff, out, outOff, outDim, inDim, rows)
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
			c, ok := r.nextCmd()
			if !ok {
				return
			}
			emitQMVRowsTiled(vtRecordSink{c, r}, pso, w.wq.buf, w.wq.off, w.scales.buf, w.scales.off, w.biases.buf, w.biases.off, in, inOff, out, outOff, inDim, outDim)
			return
		}
		pso, ok := lthnGatherQMVPipelineICB(plan.gatherKey)
		if !ok {
			r.fail()
			return
		}
		lhs, rhs, ok := qmvRowsIndexBuffers()
		if !ok {
			r.fail()
			return
		}
		c, ok := r.nextCmd()
		if !ok {
			return
		}
		emitLthnGatherQMVRoutes(vtRecordSink{c, r}, pso, in, inOff, w.wq.buf, w.wq.off, w.scales.buf, w.scales.off, w.biases.buf, w.biases.off, lhs, rhs, 0, out, outOff, outDim, inDim, gs, bits, 0, rows)
		return
	}
	// the live path's qmm_t fallback (K%gs==0 checked by rowsCapable upfront)
	if inDim <= 0 || gs <= 0 || inDim%gs != 0 {
		r.fail()
		return
	}
	r.recordQMMT(w.wq.buf, w.scales.buf, w.biases.buf, in, out, w.wq.off, w.scales.off, w.biases.off, inOff, outOff, rows, outDim, inDim, gs, bits)
}

func (r *verifyTailRecorder) recordGemvBatched(mat metal.MTLBuffer, matOff uint, vec metal.MTLBuffer, vecOff uint, out metal.MTLBuffer, outOff uint, outDim, inDim, batch int) {
	if r.failed {
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
	emitGemvBatchedVecAt(vtRecordSink{c, r}, pso, mat, matOff, vec, vecOff, out, outOff, inDim, outDim, batch, bm, bn, sm, tm)
}

func (r *verifyTailRecorder) recordQMMT(wq, scales, biases, x, out metal.MTLBuffer, wqOff, scalesOff, biasesOff, xOff, outOff uint, m, outDim, inDim, gs, bits int) {
	if r.failed {
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
	emitQMMT(vtRecordSink{c, r}, pso, wq, wqOff, scales, scalesOff, biases, biasesOff, x, xOff, out, outOff, m, outDim, inDim)
}

func (r *verifyTailRecorder) recordRMSRows(x, w, out metal.MTLBuffer, xOff, wOff, outOff uint, rows, axisSize int, eps float32) {
	if r.failed {
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
	emitRMSNormRows(vtRecordSink{c, r}, pso, x, w, out, xOff, wOff, outOff, axisSize, eps, rows, rmsThreadgroup(axisSize, pso))
}

func (r *verifyTailRecorder) recordAdd(a metal.MTLBuffer, aOff uint, b metal.MTLBuffer, bOff uint, out metal.MTLBuffer, outOff uint, n int) {
	if r.failed {
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
	emitBinary(vtRecordSink{c, r}, pso, a, aOff, b, bOff, out, outOff, n)
}

// recordResidualRowsMaybeNorm mirrors encResidualRowsMaybeNorm: one add, or
// norm-rows + add.
func (r *verifyTailRecorder) recordResidualRowsMaybeNorm(x metal.MTLBuffer, xOff uint, v metal.MTLBuffer, vOff uint, scratch, out metal.MTLBuffer, outOff uint, norm bufView, rows, dModel int, eps float32) {
	if r.failed {
		return
	}
	if norm.buf == nil {
		r.recordAdd(x, xOff, v, vOff, out, outOff, rows*dModel)
		return
	}
	r.recordRMSRows(v, norm.buf, scratch, vOff, norm.off, 0, rows, dModel, eps)
	r.recordAdd(x, xOff, scratch, 0, out, outOff, rows*dModel)
}

func (r *verifyTailRecorder) recordGeluGateMul(gate, up, out metal.MTLBuffer, gateOff, upOff, outOff uint, n int) {
	if r.failed {
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
	emitBinary(vtRecordSink{c, r}, pso, gate, gateOff, up, upOff, out, outOff, n)
}

func (r *verifyTailRecorder) recordPLEGateGeluRows(gatePacked, gateScales, gateBiases bufView, x metal.MTLBuffer, xOff uint, ple metal.MTLBuffer, pleOff uint, gated metal.MTLBuffer, gatedOff uint, rows, dModel, pliDim, gs, bits int) {
	if r.failed {
		return
	}
	pso, ok := lthnPLEGateGeluPipelineICB(bits)
	if !ok {
		r.fail()
		return
	}
	c, ok := r.nextCmd()
	if !ok {
		return
	}
	emitPLEGateGeluRows(vtRecordSink{c, r}, pso, gatePacked, gateScales, gateBiases, x, xOff, ple, pleOff, gated, gatedOff, rows, dModel, pliDim, gs)
}

func (r *verifyTailRecorder) recordRMSNormResidualRows(x, weight, res, out metal.MTLBuffer, xOff, wOff, resOff, outOff uint, rows, axisSize int, eps float32) {
	if r.failed {
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
	emitRMSNormResidualRows(vtRecordSink{c, r}, pso, x, weight, res, out, xOff, wOff, resOff, outOff, rows, axisSize, eps, rmsThreadgroup(axisSize, pso))
}

func (r *verifyTailRecorder) recordMulRows(a, b, out metal.MTLBuffer, aOff, bOff, outOff uint, rows, rowLen int) {
	if r.failed {
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
	emitMulRows(vtRecordSink{c, r}, pso, a, b, out, aOff, bOff, outOff, rows, rowLen)
}
