// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/arch/Qwen/qwen3"
	"dappco.re/go/inference/model/composed"
	"github.com/tmc/apple/metal"
)

// composed_chain_backend.go — whole-token chaining for the composed lane (#26): every layer's
// encodes land on ONE retained command buffer, the hidden state ping-pongs between two device
// buffers, and the host sees exactly one upload (Begin) and one wait+readback (End) per forward.
// The per-layer folds this chains are the proven single-CB layer bodies; encoder boundaries
// within the shared CB order the layers (a Metal queue executes a CB's encoders serially), so no
// cross-layer barriers are needed. The CB is RETAINED across the hook calls (the lane_set
// pool-boundary trap: an autoreleased cb dies when a step's pool drains, and a later wait hangs).
type composedChainCtx struct {
	cb       metal.MTLCommandBufferObject
	rec      *composedChainRecording // non-nil = RECORDING mode: bodies land in the ICB, cb is unused
	hA, hB   *pinnedNoCopyBytes
	curIsA   bool
	L, D     int
	gdSc     *gatedDeltaQuantLayerScratch
	gdKey    gatedDeltaQuantLayerKey
	attnPins []*pinnedNoCopyBytes
	// posBuf is the per-token POSITION every position-dependent attention command binds
	// (qprep/kprep/vappend/sdpa read `constant int&` from it): the live path writes it per
	// forward; a replayed recording bumps the recording's own copy. One buffer per ctx —
	// every layer of one token shares the same position.
	posBuf metal.MTLBuffer
	posPtr *int32
	// attnSc pools the attention layers' staging per geometry — every same-shaped attn layer
	// of the chain shares one set (encoder/barrier ordering serialises reuse), replacing the
	// per-call pin churn. A recording owns its scratch for its lifetime.
	attnSc map[attnChainKey]*attnChainLayerScratch
	// moeSc pools the MoE FFN-tail body slabs per geometry — the router/gather/combine + shared
	// staging, shared by every same-shaped MoE layer of the chain (the MoE tail is live-only, so a
	// recording ctx never populates this).
	moeSc map[moeChainKey]*moeChainScratch
	// head fold (#18): the terminal RMSNorm + LM head encoded onto this chain's CB by
	// ComposedChainHeadDevice. headSc holds the staging + the logits pin ComposedChainTakeLogits
	// reads after End's wait; returned to its pool at take (or at End when never taken).
	headSc    *chainHeadScratch
	headVocab int
}

// layerTarget hands a chain body its dispatch target: recording mode shares the recording
// target (no encoders — ICB commands); live mode opens a fresh encoder on the chain CB, and
// done() closes it (the encoder boundary that orders this layer after the previous one).
func (c *composedChainCtx) layerTarget() (t *chainTarget, done func(), err error) {
	if c.rec != nil {
		t = &chainTarget{rec: c.rec}
		// a layer boundary is a dependency edge in the recorded stream
		t.pending = true
		return t, func() {}, nil
	}
	enc := computeCommandEncoderFast(c.cb)
	t = &chainTarget{enc: enc}
	return t, func() { endEncodingFast(enc) }, nil
}

// attnChainKey identifies one attention-layer staging geometry.
type attnChainKey struct{ L, D, qCols, kvDim, mixCols, FF int }

// attnChainLayerScratch is the pooled staging for one attention chain layer shape.
type attnChainLayerScratch struct {
	normed, nBF                *pinnedNoCopyBytes
	qRaw, qRawBF               *pinnedNoCopyBytes
	kRaw, kRawBF, vRaw, vRawBF *pinnedNoCopyBytes
	qPrep, gateBuf, attnOut    *pinnedNoCopyBytes
	attnBF, mix, mixBF         *pinnedNoCopyBytes
	gFF, gFFBF, uFF, uFFBF     *pinnedNoCopyBytes
	sFF                        *pinnedNoCopyBytes
}

var attnChainLayerPools sync.Map // attnChainKey -> *sync.Pool

func getAttnChainLayerScratch(k attnChainKey) (*attnChainLayerScratch, error) {
	poolAny, ok := attnChainLayerPools.Load(k)
	if !ok {
		poolAny, _ = attnChainLayerPools.LoadOrStore(k, &sync.Pool{})
	}
	if v := poolAny.(*sync.Pool).Get(); v != nil {
		return v.(*attnChainLayerScratch), nil
	}
	sc := &attnChainLayerScratch{}
	var err error
	alloc := func(n int) *pinnedNoCopyBytes {
		if err != nil {
			return nil
		}
		var b *pinnedNoCopyBytes
		b, err = newPinnedNoCopyBytes(n)
		return b
	}
	sc.normed = alloc(k.L * k.D * 4)
	sc.nBF = alloc(k.L * max(k.D, k.FF) * bf16Size)
	sc.qRaw = alloc(k.L * k.qCols * 4)
	sc.qRawBF = alloc(k.L * k.qCols * bf16Size)
	sc.kRaw = alloc(k.L * k.kvDim * 4)
	sc.kRawBF = alloc(k.L * k.kvDim * bf16Size)
	sc.vRaw = alloc(k.L * k.kvDim * 4)
	sc.vRawBF = alloc(k.L * k.kvDim * bf16Size)
	sc.qPrep = alloc(k.L * k.mixCols * 4)
	sc.gateBuf = alloc(k.L * k.mixCols * 4)
	sc.attnOut = alloc(k.L * k.mixCols * 4)
	sc.attnBF = alloc(k.L * k.mixCols * bf16Size)
	sc.mix = alloc(k.L * k.D * 4)
	sc.mixBF = alloc(k.L * k.D * bf16Size)
	sc.gFF = alloc(k.L * k.FF * 4)
	sc.gFFBF = alloc(k.L * k.FF * bf16Size)
	sc.uFF = alloc(k.L * k.FF * 4)
	sc.uFFBF = alloc(k.L * k.FF * bf16Size)
	sc.sFF = alloc(k.L * k.FF * 4)
	if err != nil {
		return nil, err
	}
	return sc, nil
}

func putAttnChainLayerScratch(k attnChainKey, sc *attnChainLayerScratch) {
	if sc == nil {
		return
	}
	if v, ok := attnChainLayerPools.Load(k); ok {
		v.(*sync.Pool).Put(sc)
	}
}

// attnScratchFor resolves the shared attention staging for this geometry (one per ctx per
// geometry — same-shaped layers share; ordering serialises reuse).
func (c *composedChainCtx) attnScratchFor(k attnChainKey) (*attnChainLayerScratch, error) {
	if c.attnSc == nil {
		c.attnSc = make(map[attnChainKey]*attnChainLayerScratch, 2)
	}
	if sc, ok := c.attnSc[k]; ok {
		return sc, nil
	}
	sc, err := getAttnChainLayerScratch(k)
	if err != nil {
		return nil, err
	}
	c.attnSc[k] = sc
	return sc, nil
}

// releaseScratch returns the ctx's pooled staging (gd + attn + head) — the live End path.
// A RECORDING ctx keeps everything: the recorded commands bind these buffers for the
// recording's lifetime (composedChainRecording.release drops them).
func (c *composedChainCtx) releaseScratch() {
	if c.gdSc != nil {
		putGatedDeltaQuantLayerScratch(c.gdKey, c.gdSc)
		c.gdSc = nil
	}
	for k, sc := range c.attnSc {
		putAttnChainLayerScratch(k, sc)
	}
	c.attnSc = nil
	for k, sc := range c.moeSc {
		putMoEChainLayerScratch(k, sc)
	}
	c.moeSc = nil
}

// moeChainKey identifies one MoE FFN-tail body geometry.
type moeChainKey struct{ D, nE, topK, expertDFF, sharedFF int }

var moeChainLayerPools sync.Map // moeChainKey -> *sync.Pool

func getMoEChainLayerScratch(k moeChainKey) (*moeChainScratch, error) {
	poolAny, ok := moeChainLayerPools.Load(k)
	if !ok {
		poolAny, _ = moeChainLayerPools.LoadOrStore(k, &sync.Pool{})
	}
	if v := poolAny.(*sync.Pool).Get(); v != nil {
		return v.(*moeChainScratch), nil
	}
	return newMoEChainScratch(k.D, k.nE, k.topK, k.expertDFF, k.sharedFF)
}

func putMoEChainLayerScratch(k moeChainKey, sc *moeChainScratch) {
	if sc == nil {
		return
	}
	if v, ok := moeChainLayerPools.Load(k); ok {
		v.(*sync.Pool).Put(sc)
	}
}

// moeScratchFor resolves the shared MoE-tail staging for this geometry (one per ctx per geometry —
// same-shaped layers share; encoder/barrier ordering serialises reuse).
func (c *composedChainCtx) moeScratchFor(k moeChainKey) (*moeChainScratch, error) {
	if c.moeSc == nil {
		c.moeSc = make(map[moeChainKey]*moeChainScratch, 1)
	}
	if sc, ok := c.moeSc[k]; ok {
		return sc, nil
	}
	sc, err := getMoEChainLayerScratch(k)
	if err != nil {
		return nil, err
	}
	c.moeSc[k] = sc
	return sc, nil
}

func (c *composedChainCtx) cur() *pinnedNoCopyBytes {
	if c.curIsA {
		return c.hA
	}
	return c.hB
}

func (c *composedChainCtx) next() *pinnedNoCopyBytes {
	if c.curIsA {
		return c.hB
	}
	return c.hA
}

// ComposedChainBeginDevice opens a chained forward: uploads h [L,D] once and hands back the
// opaque context every chained layer encodes into.
func ComposedChainBeginDevice(h []float32, L, D int) (any, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if L <= 0 || len(h) != L*D {
		return nil, core.NewError("native.ComposedChainBeginDevice: size mismatch")
	}
	hA, err := newPinnedNoCopyBytes(L * D * 4)
	if err != nil {
		return nil, err
	}
	hB, err := newPinnedNoCopyBytes(L * D * 4)
	if err != nil {
		return nil, err
	}
	copy(hA.bytes, float32Bytes(h))
	posBuf := device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared)
	var cb metal.MTLCommandBufferObject
	withAutoreleasePool(func() {
		cb = commandBufferFast(queue)
		cb.Retain() // survives the per-step autorelease pools until End releases it
	})
	return &composedChainCtx{cb: cb, hA: hA, hB: hB, curIsA: true, L: L, D: D,
		posBuf: posBuf, posPtr: (*int32)(posBuf.Contents())}, nil
}

// ComposedChainEndDevice commits the chained forward, waits once, and returns the final hidden.
func ComposedChainEndDevice(ctxAny any) ([]float32, error) {
	ctx, ok := ctxAny.(*composedChainCtx)
	if !ok || ctx.cb.GetID() == 0 {
		return nil, core.NewError("native.ComposedChainEndDevice: not a chain context")
	}
	var y []float32
	withAutoreleasePool(func() {
		commitCommandBufferFast(ctx.cb)
		waitUntilCompletedFast(ctx.cb)
		y = make([]float32, ctx.L*ctx.D)
		copy(y, unsafe.Slice((*float32)(unsafe.Pointer(&ctx.cur().bytes[0])), ctx.L*ctx.D))
		ctx.cb.Release()
		ctx.cb = metal.MTLCommandBufferObject{}
	})
	ctx.releaseScratch()
	return y, nil
}

// ComposedChainRecordBegin opens a RECORDING pass (#18 CB recording): the same per-layer chain
// walk drives the returned ctx, but every dispatch lands in an MTLIndirectCommandBuffer instead
// of a live encoder — nothing executes. ComposedChainRecordEnd hands back the finished
// recording; composedChainReplay then re-issues it per token with one executeCommandsInBuffer.
func ComposedChainRecordBegin(L, D, nLayers int) (any, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if L != 1 {
		return nil, core.NewError("native.ComposedChainRecordBegin: only the L=1 decode stream records")
	}
	hA, err := newPinnedNoCopyBytes(L * D * 4)
	if err != nil {
		return nil, err
	}
	hB, err := newPinnedNoCopyBytes(L * D * 4)
	if err != nil {
		return nil, err
	}
	posBuf := device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared)
	maxCmd := uint(nLayers*64 + 96)
	var icb metal.MTLIndirectCommandBuffer
	withAutoreleasePool(func() {
		icbDesc := metal.NewMTLIndirectCommandBufferDescriptor()
		icbDesc.SetCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch | metal.MTLIndirectCommandTypeConcurrentDispatchThreads)
		icbDesc.SetInheritBuffers(false)
		icbDesc.SetInheritPipelineState(false)
		icbDesc.SetMaxKernelBufferBindCount(16)
		icb = device.NewIndirectCommandBufferWithDescriptorMaxCommandCountOptions(icbDesc, maxCmd, metal.MTLResourceStorageModeShared)
	})
	if icb == nil {
		return nil, core.NewError("native.ComposedChainRecordBegin: ICB allocation failed")
	}
	rec := &composedChainRecording{
		icb: icb, maxCmd: maxCmd,
		posBuf: posBuf, posPtr: (*int32)(posBuf.Contents()),
		hA: hA, hB: hB, D: D,
		resourceSet: make(map[uintptr]struct{}, nLayers*24),
	}
	rec.track(posBuf)
	rec.pins = append(rec.pins, hA, hB)
	rec.track(hA.buf)
	rec.track(hB.buf)
	return &composedChainCtx{rec: rec, hA: hA, hB: hB, curIsA: true, L: L, D: D,
		posBuf: posBuf, posPtr: rec.posPtr}, nil
}

// ComposedChainRecordEnd finalises a recording pass: the final-hidden pin is whichever
// ping-pong the last layer wrote, the ctx's scratch ownership moves to the recording (its
// commands bind those buffers for the recording's lifetime), and the recording is handed back.
func ComposedChainRecordEnd(ctxAny any) (any, error) {
	ctx, ok := ctxAny.(*composedChainCtx)
	if !ok || ctx.rec == nil {
		return nil, core.NewError("native.ComposedChainRecordEnd: not a recording context")
	}
	rec := ctx.rec
	if rec.count == 0 {
		return nil, core.NewError("native.ComposedChainRecordEnd: nothing recorded")
	}
	rec.final = ctx.cur()
	// scratch moves to the recording: gd staging by handle (returned to its pool on release),
	// attn staging + head pins by ownership of the pins slice.
	rec.gdScratch, rec.gdKey, rec.hasGD = ctx.gdSc, ctx.gdKey, ctx.gdSc != nil
	ctx.gdSc = nil
	for _, sc := range ctx.attnSc {
		rec.pins = append(rec.pins,
			sc.normed, sc.nBF, sc.qRaw, sc.qRawBF, sc.kRaw, sc.kRawBF, sc.vRaw, sc.vRawBF,
			sc.qPrep, sc.gateBuf, sc.attnOut, sc.attnBF, sc.mix, sc.mixBF,
			sc.gFF, sc.gFFBF, sc.uFF, sc.uFFBF, sc.sFF)
	}
	ctx.attnSc = nil
	if ctx.headSc != nil { // the head fold's staging + logits pin belong to the recording too
		rec.pins = append(rec.pins, ctx.headSc.normed, ctx.headSc.nBF, ctx.headSc.logitsBF, ctx.headSc.logits)
		rec.logitsPin = ctx.headSc.logits
		rec.headVocab = ctx.headVocab
		ctx.headSc = nil
	}
	return rec, nil
}

// ComposedChainReplayDevice re-issues a recorded chain for one token — the model-side hook.
// The position comes from the recording's own attention states (they all sit at the same
// position; a desync or a capacity/realloc mismatch declines with ok=false and the caller
// re-records after the live pass).
func ComposedChainReplayDevice(recAny any, h []float32) (y []float32, logits []float32, ok bool, err error) {
	rec, isRec := recAny.(*composedChainRecording)
	if !isRec {
		return nil, nil, false, core.NewError("native.ComposedChainReplayDevice: not a recording")
	}
	if len(h) != rec.D {
		return nil, nil, false, core.NewError("native.ComposedChainReplayDevice: hidden size mismatch")
	}
	pos := 0
	if len(rec.attnStates) > 0 {
		pos = rec.attnStates[0].n
	}
	return composedChainReplay(rec, h, pos)
}

// ComposedChainRecordingRelease drops a recording (stale after a cache realloc / geometry
// change): pooled gd staging returns, pins and the ICB release with the reference.
func ComposedChainRecordingRelease(recAny any) {
	if rec, ok := recAny.(*composedChainRecording); ok {
		rec.release()
	}
}

// chainHeadScratch is the head fold's staging: the last row's normed output (f32 + its bf16
// cast), the logits (bf16 staging + f32 result pin). Pooled by (D, Vocab).
type chainHeadScratch struct {
	normed, nBF, logitsBF, logits *pinnedNoCopyBytes
}

type chainHeadKey struct{ D, Vocab int }

var chainHeadPools sync.Map // chainHeadKey -> *sync.Pool

func getChainHeadScratch(D, Vocab int) (*chainHeadScratch, error) {
	key := chainHeadKey{D, Vocab}
	poolAny, ok := chainHeadPools.Load(key)
	if !ok {
		poolAny, _ = chainHeadPools.LoadOrStore(key, &sync.Pool{})
	}
	if v := poolAny.(*sync.Pool).Get(); v != nil {
		return v.(*chainHeadScratch), nil
	}
	sc := &chainHeadScratch{}
	var err error
	alloc := func(n int) *pinnedNoCopyBytes {
		if err != nil {
			return nil
		}
		var b *pinnedNoCopyBytes
		b, err = newPinnedNoCopyBytes(n)
		return b
	}
	sc.normed = alloc(D * 4)
	sc.nBF = alloc(D * bf16Size)
	sc.logitsBF = alloc(Vocab * bf16Size)
	sc.logits = alloc(Vocab * 4)
	if err != nil {
		return nil, err
	}
	return sc, nil
}

func putChainHeadScratch(D, Vocab int, sc *chainHeadScratch) {
	if v, ok := chainHeadPools.Load(chainHeadKey{D, Vocab}); ok {
		v.(*sync.Pool).Put(sc)
	}
}

// ComposedChainHeadDevice encodes the model's terminal stage onto the chain's retained command
// buffer (#18 head fold): RMSNorm(cur[L-1,:], normF) → LM head projection (bf16-resident or
// packed form via the same f32-activation emitters the chain layers use) → f32 logits, all
// ordered after the last layer by the encoder boundary. The logits land in a pinned buffer
// ComposedChainTakeLogits reads AFTER ComposedChainEndDevice's wait. The head stages write only
// this scratch — never the hidden pins — so an encode error leaves the chain itself intact and
// the caller falls back to the separate head exactly as before.
func ComposedChainHeadDevice(ctxAny any, normF []float32, headB *model.BF16Weight, headQ *model.QuantWeight, D, Vocab int, eps float32) error {
	ctx, ok := ctxAny.(*composedChainCtx)
	if !ok || (ctx.rec == nil && ctx.cb.GetID() == 0) {
		return core.NewError("native.ComposedChainHeadDevice: not a chain context")
	}
	if ctx.headSc != nil {
		return core.NewError("native.ComposedChainHeadDevice: head already encoded on this chain")
	}
	if len(normF) != D {
		return core.NewError("native.ComposedChainHeadDevice: normF size mismatch")
	}
	if headQ != nil {
		if !quantGeometryOK(headQ, Vocab, D) {
			return core.NewError("native.ComposedChainHeadDevice: unsupported packed head geometry")
		}
	} else if !bf16GeometryOK(headB, Vocab, D) {
		return core.NewError("native.ComposedChainHeadDevice: unsupported bf16 head geometry")
	}
	rmsName := "rmsfloat32"
	if D > rmsLoopedLimit {
		rmsName = "rms_loopedfloat32"
	}
	sc, err := getChainHeadScratch(D, Vocab)
	if err != nil {
		return err
	}
	var encErr error
	withAutoreleasePool(func() {
		normFBuf := residentFloat32(normF)
		lastRowOff := uint((ctx.L - 1) * ctx.D * 4)
		t, done, terr := ctx.layerTarget()
		if terr != nil {
			encErr = terr
			return
		}
		defer done()
		psoRMS, perr := t.pso(rmsName)
		if perr != nil {
			encErr = perr
			return
		}
		emitRMSNormAt(t.cmd(), psoRMS, ctx.cur().buf, normFBuf, sc.normed.buf, lastRowOff, 0, 0, D, eps, rmsThreadgroup(D, psoRMS))
		t.barrier()
		if headQ != nil {
			encErr = chainProjQuantF32(t, headQ, sc.normed.buf, sc.nBF.buf, sc.logitsBF.buf, sc.logits.buf, 1, Vocab, D)
		} else {
			encErr = chainProjBF16F32(t, headB, sc.normed.buf, sc.nBF.buf, sc.logitsBF.buf, sc.logits.buf, 1, Vocab, D)
		}
		if encErr == nil && t.err != nil {
			encErr = t.err
		}
	})
	if encErr != nil {
		putChainHeadScratch(D, Vocab, sc)
		return encErr
	}
	ctx.headSc, ctx.headVocab = sc, Vocab
	return nil
}

// ComposedChainTakeLogits hands back the head-fold logits after ComposedChainEndDevice's wait
// (nil when no head was encoded) and returns the staging to its pool.
func ComposedChainTakeLogits(ctxAny any) []float32 {
	ctx, ok := ctxAny.(*composedChainCtx)
	if !ok || ctx.headSc == nil {
		return nil
	}
	logits := make([]float32, ctx.headVocab)
	copy(logits, unsafe.Slice((*float32)(unsafe.Pointer(&ctx.headSc.logits.bytes[0])), ctx.headVocab))
	putChainHeadScratch(ctx.D, ctx.headVocab, ctx.headSc)
	ctx.headSc = nil
	return logits
}

// gatedDeltaBF16ChainLayerDevice encodes one dense bf16 gated-delta layer onto the chain — the
// gatedDeltaBF16LayerRun body against the context's ping-pong hidden, sharing one pooled scratch
// across every gd layer of the chain (encoder boundaries serialise their reuse).
func gatedDeltaBF16ChainLayerDevice(ctxAny any, sc *qwen3.GatedDeltaScratch, inputNorm []float32, w *qwen3.GatedDeltaWeights, cfg qwen3.GatedDeltaConfig, postNorm []float32, gate, up, down *model.BF16Weight, priorConv, priorDelta []float32, FF int, eps float32) error {
	ctx, ok := ctxAny.(*composedChainCtx)
	if !ok {
		return core.NewError("native.gatedDeltaBF16ChainLayerDevice: not a chain context")
	}
	if ctx.rec != nil {
		// the bf16 chain bodies are not target-converted yet — a bf16 layer declines the
		// recording pass (the model keeps the re-encode chain; only all-quant stacks record).
		return core.NewError("native.gatedDeltaBF16ChainLayerDevice: bf16 layers do not record")
	}
	h, _ := sc.Device.(*gatedDeltaDeviceState)
	if h == nil {
		if !gatedDeltaBlockUsable(cfg.HeadDim, cfg.HeadDim, cfg.KeyHeads, cfg.ValueHeads, cfg.ConvKernel) {
			return core.NewError("native.gatedDeltaBF16ChainLayerDevice: geometry not servable")
		}
		nh, err := newGatedDeltaDeviceState(cfg.KeyHeads, cfg.ValueHeads, cfg.HeadDim, cfg.HeadDim, cfg.ConvKernel, 1)
		if err != nil {
			return err
		}
		h = nh
	}
	L, D := ctx.L, ctx.D
	convDim, vDim := h.convDim, h.Hv*h.Dv
	if !bf16GeometryOK(w.InProjQKVB, convDim, D) || !bf16GeometryOK(w.InProjZB, vDim, D) ||
		!bf16GeometryOK(w.InProjAB, h.Hv, D) || !bf16GeometryOK(w.InProjBB, h.Hv, D) ||
		!bf16GeometryOK(w.OutProjB, D, vDim) ||
		!bf16GeometryOK(gate, FF, D) || !bf16GeometryOK(up, FF, D) || !bf16GeometryOK(down, D, FF) {
		return core.NewError("native.gatedDeltaBF16ChainLayerDevice: unsupported bf16 geometry")
	}
	if !h.valid {
		h.prime(priorConv, priorDelta)
	}
	key := gatedDeltaQuantLayerKey{L: L, D: D, FF: FF, Hk: h.Hk, Hv: h.Hv, Dk: h.Dk, K: h.K}
	if ctx.gdSc == nil {
		gsc, err := getGatedDeltaQuantLayerScratch(key)
		if err != nil {
			return err
		}
		ctx.gdSc, ctx.gdKey = gsc, key
	} else if ctx.gdKey != key {
		return core.NewError("native.gatedDeltaBF16ChainLayerDevice: mixed gd geometries in one chain")
	}
	gsc := ctx.gdSc
	rmsName := "rmsfloat32"
	if D > rmsLoopedLimit {
		rmsName = "rms_loopedfloat32"
	}
	psoRMS, err := pipelineFor(rmsName)
	if err != nil {
		return err
	}
	var encErr error
	withAutoreleasePool(func() {
		wConv := residentFloat32(w.ConvWeight)
		wBias := wConv
		hasBias := 0
		if w.ConvBias != nil {
			wBias = residentFloat32(w.ConvBias)
			hasBias = 1
		}
		enc := computeCommandEncoderFast(ctx.cb)
		fail := func(err error) { encErr = err; endEncodingFast(enc) }
		emitRMSNormRows(encSink{enc}, psoRMS, ctx.cur().buf, residentFloat32(inputNorm), gsc.normed1.buf, 0, 0, 0, D, eps, L, rmsThreadgroup(D, psoRMS))
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		if err := encNarrowF32ToBF16(enc, gsc.normed1.buf, gsc.n1BF.buf, L*D); err != nil {
			fail(err)
			return
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		if err := encProjBF16From(enc, w.InProjQKVB, gsc.n1BF.buf, gsc.qkvBF.buf, gsc.qkv.buf, L, convDim, D); err != nil {
			fail(err)
			return
		}
		if err := encProjBF16From(enc, w.InProjZB, gsc.n1BF.buf, gsc.zBF.buf, gsc.z.buf, L, vDim, D); err != nil {
			fail(err)
			return
		}
		if err := encProjBF16From(enc, w.InProjAB, gsc.n1BF.buf, gsc.aBF.buf, gsc.a.buf, L, h.Hv, D); err != nil {
			fail(err)
			return
		}
		if err := encProjBF16From(enc, w.InProjBB, gsc.n1BF.buf, gsc.bBF.buf, gsc.b.buf, L, h.Hv, D); err != nil {
			fail(err)
			return
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		if err := encGatedDeltaBlockStages(enc, h, gdBlockStageBufs{
			qkv: gsc.qkv.buf, z: gsc.z.buf, a: gsc.a.buf, b: gsc.b.buf,
			qN: gsc.qN.buf, kN: gsc.kN.buf, vN: gsc.vN.buf, g: gsc.g.buf, beta: gsc.beta.buf,
			gated: gsc.gated.buf,
		}, wConv, wBias, hasBias, residentFloat32(w.ALog), residentFloat32(w.DtBias), residentFloat32(w.Norm), L); err != nil {
			fail(err)
			return
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		if err := encProjBF16F32(enc, w.OutProjB, gsc.gated.buf, gsc.gatedBF.buf, gsc.mixBF.buf, gsc.mix.buf, L, D, vDim); err != nil {
			fail(err)
			return
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		if err := encResidualNormMLPBF16Tail(enc, quantTailBufs{
			h: ctx.cur().buf, mix: gsc.mix.buf, normed: gsc.normed2.buf, nBF: gsc.n2BF.buf,
			g: gsc.gFF.buf, gBF: gsc.gFFBF.buf, u: gsc.uFF.buf, uBF: gsc.uFFBF.buf, s: gsc.sFF.buf, out: ctx.next().buf,
		}, residentFloat32(postNorm), gate, up, down, L, D, FF, eps); err != nil {
			fail(err)
			return
		}
		endEncodingFast(enc)
	})
	if encErr != nil {
		return encErr
	}
	sc.Device = h
	ctx.curIsA = !ctx.curIsA
	return nil
}

// gatedDeltaQuantChainLayerDevice encodes one packed gated-delta layer onto the chain — the
// gatedDeltaQuantLayerRun body (lthn_gated_delta.go) against the context's ping-pong hidden,
// sharing the chain's one pooled gd scratch (the same buffer shapes the bf16 chain step uses —
// both weight forms cast to bf16 at the qmv/gemv boundary). The quant twin of
// gatedDeltaBF16ChainLayerDevice.
func gatedDeltaQuantChainLayerDevice(ctxAny any, sc *qwen3.GatedDeltaScratch, inputNorm []float32, w *qwen3.GatedDeltaWeights, cfg qwen3.GatedDeltaConfig, postNorm []float32, gate, up, down *model.QuantWeight, priorConv, priorDelta []float32, FF int, eps float32) error {
	ctx, ok := ctxAny.(*composedChainCtx)
	if !ok {
		return core.NewError("native.gatedDeltaQuantChainLayerDevice: not a chain context")
	}
	h, _ := sc.Device.(*gatedDeltaDeviceState)
	if h == nil {
		if !gatedDeltaBlockUsable(cfg.HeadDim, cfg.HeadDim, cfg.KeyHeads, cfg.ValueHeads, cfg.ConvKernel) {
			return core.NewError("native.gatedDeltaQuantChainLayerDevice: geometry not servable")
		}
		nh, err := newGatedDeltaDeviceState(cfg.KeyHeads, cfg.ValueHeads, cfg.HeadDim, cfg.HeadDim, cfg.ConvKernel, 1)
		if err != nil {
			return err
		}
		h = nh
	}
	L, D := ctx.L, ctx.D
	convDim, vDim := h.convDim, h.Hv*h.Dv
	if w.InProjQKVQ == nil || !quantGeometryOK(w.InProjQKVQ, convDim, D) ||
		w.InProjZQ == nil || !quantGeometryOK(w.InProjZQ, vDim, D) ||
		w.InProjAQ == nil || !quantGeometryOK(w.InProjAQ, h.Hv, D) ||
		w.InProjBQ == nil || !quantGeometryOK(w.InProjBQ, h.Hv, D) ||
		w.OutProjQ == nil || !quantGeometryOK(w.OutProjQ, D, vDim) ||
		!quantGeometryOK(gate, FF, D) || !quantGeometryOK(up, FF, D) || !quantGeometryOK(down, D, FF) {
		return core.NewError("native.gatedDeltaQuantChainLayerDevice: unsupported quant geometry")
	}
	if !h.valid {
		h.prime(priorConv, priorDelta)
	}
	key := gatedDeltaQuantLayerKey{L: L, D: D, FF: FF, Hk: h.Hk, Hv: h.Hv, Dk: h.Dk, K: h.K}
	if ctx.gdSc == nil {
		gsc, err := getGatedDeltaQuantLayerScratch(key)
		if err != nil {
			return err
		}
		ctx.gdSc, ctx.gdKey = gsc, key
	} else if ctx.gdKey != key {
		return core.NewError("native.gatedDeltaQuantChainLayerDevice: mixed gd geometries in one chain")
	}
	gsc := ctx.gdSc
	rmsName := "rmsfloat32"
	if D > rmsLoopedLimit {
		rmsName = "rms_loopedfloat32"
	}
	var encErr error
	withAutoreleasePool(func() {
		wConv := residentFloat32(w.ConvWeight)
		wBias := wConv
		hasBias := 0
		if w.ConvBias != nil {
			wBias = residentFloat32(w.ConvBias)
			hasBias = 1
		}
		t, done, terr := ctx.layerTarget()
		if terr != nil {
			encErr = terr
			return
		}
		defer done()
		fail := func(err error) { encErr = err }
		psoRMS, perr := t.pso(rmsName)
		if perr != nil {
			fail(perr)
			return
		}
		emitRMSNormRows(t.cmd(), psoRMS, ctx.cur().buf, residentFloat32(inputNorm), gsc.normed1.buf, 0, 0, 0, D, eps, L, rmsThreadgroup(D, psoRMS))
		t.barrier()
		if err := chainNarrowF32ToBF16(t, gsc.normed1.buf, gsc.n1BF.buf, L*D); err != nil {
			fail(err)
			return
		}
		t.barrier()
		if err := chainProjQuantBF16In(t, w.InProjQKVQ, gsc.n1BF.buf, gsc.qkvBF.buf, gsc.qkv.buf, L, convDim, D); err != nil {
			fail(err)
			return
		}
		if err := chainProjQuantBF16In(t, w.InProjZQ, gsc.n1BF.buf, gsc.zBF.buf, gsc.z.buf, L, vDim, D); err != nil {
			fail(err)
			return
		}
		if err := chainProjQuantBF16In(t, w.InProjAQ, gsc.n1BF.buf, gsc.aBF.buf, gsc.a.buf, L, h.Hv, D); err != nil {
			fail(err)
			return
		}
		if err := chainProjQuantBF16In(t, w.InProjBQ, gsc.n1BF.buf, gsc.bBF.buf, gsc.b.buf, L, h.Hv, D); err != nil {
			fail(err)
			return
		}
		t.barrier()
		if err := chainGatedDeltaBlockStages(t, h, gdBlockStageBufs{
			qkv: gsc.qkv.buf, z: gsc.z.buf, a: gsc.a.buf, b: gsc.b.buf,
			qN: gsc.qN.buf, kN: gsc.kN.buf, vN: gsc.vN.buf, g: gsc.g.buf, beta: gsc.beta.buf,
			gated: gsc.gated.buf,
		}, wConv, wBias, hasBias, residentFloat32(w.ALog), residentFloat32(w.DtBias), residentFloat32(w.Norm), L); err != nil {
			fail(err)
			return
		}
		t.barrier()
		if err := chainProjQuantF32(t, w.OutProjQ, gsc.gated.buf, gsc.gatedBF.buf, gsc.mixBF.buf, gsc.mix.buf, L, D, vDim); err != nil {
			fail(err)
			return
		}
		t.barrier()
		if err := chainResidualNormMLPQuantTail(t, quantTailBufs{
			h: ctx.cur().buf, mix: gsc.mix.buf, normed: gsc.normed2.buf, nBF: gsc.n2BF.buf,
			g: gsc.gFF.buf, gBF: gsc.gFFBF.buf, u: gsc.uFF.buf, uBF: gsc.uFFBF.buf, s: gsc.sFF.buf, out: ctx.next().buf,
		}, residentFloat32(postNorm), gate, up, down, L, D, FF, eps); err != nil {
			fail(err)
			return
		}
		if t.err != nil {
			fail(t.err)
		}
	})
	if encErr != nil {
		return encErr
	}
	sc.Device = h
	ctx.curIsA = !ctx.curIsA
	return nil
}

// attnBF16ChainLayerDevice encodes one dense bf16 attention layer (device-KV) onto the chain —
// the AttnBF16FullLayerDevice body against the ping-pong hidden.
func attnBF16ChainLayerDevice(ctxAny, dev any, inputNorm []float32, qw, kw, vw, ow *model.BF16Weight, qNormW, kNormW, postNorm []float32, gate, up, down *model.BF16Weight, priorK, priorV []float32, H, KVH, HD, RD, pos0, window, gated, qkNorm, FF int, eps, theta float32) (any, error) {
	ctx, ok := ctxAny.(*composedChainCtx)
	if !ok {
		return dev, core.NewError("native.attnBF16ChainLayerDevice: not a chain context")
	}
	if ctx.rec != nil {
		// not target-converted yet — a bf16 layer declines the recording pass (see the gd twin).
		return dev, core.NewError("native.attnBF16ChainLayerDevice: bf16 layers do not record")
	}
	L, D := ctx.L, ctx.D
	if !attnCoreUsable(H, KVH, HD, RD) {
		return dev, core.NewError("native.attnBF16ChainLayerDevice: core not servable")
	}
	qCols := H * HD
	if gated != 0 {
		qCols = 2 * H * HD
	}
	mixCols := H * HD
	if !bf16GeometryOK(qw, qCols, D) || !bf16GeometryOK(kw, KVH*HD, D) || !bf16GeometryOK(vw, KVH*HD, D) ||
		!bf16GeometryOK(ow, D, mixCols) || !bf16GeometryOK(gate, FF, D) || !bf16GeometryOK(up, FF, D) || !bf16GeometryOK(down, D, FF) {
		return dev, core.NewError("native.attnBF16ChainLayerDevice: size/geometry mismatch")
	}
	h, _ := dev.(*attnKVDeviceState)
	if h == nil {
		h = &attnKVDeviceState{KVH: KVH, HD: HD}
		if err := h.ensureCap(pos0 + L); err != nil {
			return dev, err
		}
		if pos0 > 0 {
			if len(priorK) != pos0*KVH*HD || len(priorV) != pos0*KVH*HD {
				return dev, core.NewError("native.attnBF16ChainLayerDevice: prior state size mismatch")
			}
			copy(h.kBuf.bytes, float32Bytes(priorK))
			copy(h.vBuf.bytes, float32Bytes(priorV))
		}
		h.n = pos0
	} else if err := h.ensureCap(pos0 + L); err != nil {
		return h, err
	}
	if h.n != pos0 {
		return h, core.NewError("native.attnBF16ChainLayerDevice: position desync with resident cache")
	}
	var encErr error
	withAutoreleasePool(func() {
		alloc := func(nBytes int) *pinnedNoCopyBytes {
			if encErr != nil {
				return nil
			}
			b, err := newPinnedNoCopyBytes(nBytes)
			if err != nil {
				encErr = err
			}
			ctx.attnPins = append(ctx.attnPins, b)
			return b
		}
		normed := alloc(L * D * 4)
		nBF := alloc(L * max(D, FF) * bf16Size)
		qRaw := alloc(L * qCols * 4)
		qRawBF := alloc(L * qCols * bf16Size)
		kRaw := alloc(L * KVH * HD * 4)
		kRawBF := alloc(L * KVH * HD * bf16Size)
		vRaw := alloc(L * KVH * HD * 4)
		vRawBF := alloc(L * KVH * HD * bf16Size)
		qPrep := alloc(L * H * HD * 4)
		gateBuf := alloc(L * H * HD * 4)
		attnOut := alloc(L * H * HD * 4)
		attnBF := alloc(L * mixCols * bf16Size)
		mix := alloc(L * D * 4)
		mixBF := alloc(L * D * bf16Size)
		gFF := alloc(L * FF * 4)
		gFFBF := alloc(L * FF * bf16Size)
		uFF := alloc(L * FF * 4)
		uFFBF := alloc(L * FF * bf16Size)
		sFF := alloc(L * FF * 4)
		if encErr != nil {
			return
		}
		inNormBuf := residentFloat32(inputNorm)
		postNormBuf := residentFloat32(postNorm)
		normQ, normK := inNormBuf, inNormBuf
		if qkNorm == 1 {
			normQ = residentFloat32(qNormW)
			normK = residentFloat32(kNormW)
		}
		rmsName := "rmsfloat32"
		if D > rmsLoopedLimit {
			rmsName = "rms_loopedfloat32"
		}
		psoRMS, perr := pipelineFor(rmsName)
		if perr != nil {
			encErr = perr
			return
		}
		enc := computeCommandEncoderFast(ctx.cb)
		fail := func(err error) { encErr = err; endEncodingFast(enc) }
		emitRMSNormRows(encSink{enc}, psoRMS, ctx.cur().buf, inNormBuf, normed.buf, 0, 0, 0, D, eps, L, rmsThreadgroup(D, psoRMS))
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		if err := encNarrowF32ToBF16(enc, normed.buf, nBF.buf, L*D); err != nil {
			fail(err)
			return
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		if err := encProjBF16From(enc, qw, nBF.buf, qRawBF.buf, qRaw.buf, L, qCols, D); err != nil {
			fail(err)
			return
		}
		if err := encProjBF16From(enc, kw, nBF.buf, kRawBF.buf, kRaw.buf, L, KVH*HD, D); err != nil {
			fail(err)
			return
		}
		if err := encProjBF16From(enc, vw, nBF.buf, vRawBF.buf, vRaw.buf, L, KVH*HD, D); err != nil {
			fail(err)
			return
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		if err := encAttnQPrep(enc, qRaw.buf, normQ, qPrep.buf, gateBuf.buf, L, H, HD, RD, gated, qkNorm, eps, theta, pos0); err != nil {
			fail(err)
			return
		}
		if err := encAttnKPrep(enc, kRaw.buf, normK, h.kBuf.buf, L, KVH, HD, RD, qkNorm, eps, theta, pos0); err != nil {
			fail(err)
			return
		}
		endEncodingFast(enc)
		blit := blitCommandEncoderFast(ctx.cb)
		blit.CopyFromBufferSourceOffsetToBufferDestinationOffsetSize(vRaw.buf, 0, h.vBuf.buf, uint(pos0*KVH*HD*4), uint(L*KVH*HD*4))
		endBlitEncodingFast(blit)
		enc = computeCommandEncoderFast(ctx.cb)
		if err := encAttnSDPA(enc, qPrep.buf, h.kBuf.buf, h.vBuf.buf, attnOut.buf, L, H, KVH, HD, pos0, window); err != nil {
			fail(err)
			return
		}
		if gated != 0 {
			memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
			if err := encAttnGateSilu(enc, attnOut.buf, gateBuf.buf, L*H*HD); err != nil {
				fail(err)
				return
			}
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		if err := encNarrowF32ToBF16(enc, attnOut.buf, attnBF.buf, L*mixCols); err != nil {
			fail(err)
			return
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		if err := encProjBF16From(enc, ow, attnBF.buf, mixBF.buf, mix.buf, L, D, mixCols); err != nil {
			fail(err)
			return
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		if err := encResidualNormMLPBF16Tail(enc, quantTailBufs{
			h: ctx.cur().buf, mix: mix.buf, normed: normed.buf, nBF: nBF.buf,
			g: gFF.buf, gBF: gFFBF.buf, u: uFF.buf, uBF: uFFBF.buf, s: sFF.buf, out: ctx.next().buf,
		}, postNormBuf, gate, up, down, L, D, FF, eps); err != nil {
			fail(err)
			return
		}
		endEncodingFast(enc)
	})
	if encErr != nil {
		return h, encErr
	}
	h.n = pos0 + L
	ctx.curIsA = !ctx.curIsA
	return h, nil
}

// attnQuantChainLayerDevice encodes one packed dense-attention layer (device-KV) onto the chain —
// the AttnQuantFullLayerDevice body against the ping-pong hidden, the packed twin of
// attnBF16ChainLayerDevice.
func attnQuantChainLayerDevice(ctxAny, dev any, inputNorm []float32, qw, kw, vw, ow *model.QuantWeight, qNormW, kNormW, postNorm []float32, gate, up, down *model.QuantWeight, priorK, priorV []float32, H, KVH, HD, RD, pos0, window, gated, qkNorm, FF int, eps, theta float32) (any, error) {
	ctx, ok := ctxAny.(*composedChainCtx)
	if !ok {
		return dev, core.NewError("native.attnQuantChainLayerDevice: not a chain context")
	}
	L, D := ctx.L, ctx.D
	if !attnCoreUsable(H, KVH, HD, RD) {
		return dev, core.NewError("native.attnQuantChainLayerDevice: core not servable")
	}
	qCols := H * HD
	if gated != 0 {
		qCols = 2 * H * HD
	}
	mixCols := H * HD
	if !quantGeometryOK(qw, qCols, D) || !quantGeometryOK(kw, KVH*HD, D) || !quantGeometryOK(vw, KVH*HD, D) ||
		!quantGeometryOK(ow, D, mixCols) || !quantGeometryOK(gate, FF, D) || !quantGeometryOK(up, FF, D) || !quantGeometryOK(down, D, FF) {
		return dev, core.NewError("native.attnQuantChainLayerDevice: size/geometry mismatch")
	}
	h, _ := dev.(*attnKVDeviceState)
	if h == nil {
		h = &attnKVDeviceState{KVH: KVH, HD: HD}
		if err := h.ensureCap(pos0 + L); err != nil {
			return dev, err
		}
		if pos0 > 0 {
			if len(priorK) != pos0*KVH*HD || len(priorV) != pos0*KVH*HD {
				return dev, core.NewError("native.attnQuantChainLayerDevice: prior state size mismatch")
			}
			copy(h.kBuf.bytes, float32Bytes(priorK))
			copy(h.vBuf.bytes, float32Bytes(priorV))
		}
		h.n = pos0
	} else if err := h.ensureCap(pos0 + L); err != nil {
		return h, err
	}
	if h.n != pos0 {
		return h, core.NewError("native.attnQuantChainLayerDevice: position desync with resident cache")
	}
	sc, err := ctx.attnScratchFor(attnChainKey{L: L, D: D, qCols: qCols, kvDim: KVH * HD, mixCols: mixCols, FF: FF})
	if err != nil {
		return h, err
	}
	*ctx.posPtr = int32(pos0) // every layer of this token shares the position (idempotent)
	var encErr error
	withAutoreleasePool(func() {
		inNormBuf := residentFloat32(inputNorm)
		postNormBuf := residentFloat32(postNorm)
		normQ, normK := inNormBuf, inNormBuf
		if qkNorm == 1 {
			normQ = residentFloat32(qNormW)
			normK = residentFloat32(kNormW)
		}
		rmsName := "rmsfloat32"
		if D > rmsLoopedLimit {
			rmsName = "rms_loopedfloat32"
		}
		t, done, terr := ctx.layerTarget()
		if terr != nil {
			encErr = terr
			return
		}
		defer done()
		fail := func(err error) { encErr = err }
		psoRMS, perr := t.pso(rmsName)
		if perr != nil {
			fail(perr)
			return
		}
		emitRMSNormRows(t.cmd(), psoRMS, ctx.cur().buf, inNormBuf, sc.normed.buf, 0, 0, 0, D, eps, L, rmsThreadgroup(D, psoRMS))
		t.barrier()
		if err := chainNarrowF32ToBF16(t, sc.normed.buf, sc.nBF.buf, L*D); err != nil {
			fail(err)
			return
		}
		t.barrier()
		if err := chainProjQuantBF16In(t, qw, sc.nBF.buf, sc.qRawBF.buf, sc.qRaw.buf, L, qCols, D); err != nil {
			fail(err)
			return
		}
		if err := chainProjQuantBF16In(t, kw, sc.nBF.buf, sc.kRawBF.buf, sc.kRaw.buf, L, KVH*HD, D); err != nil {
			fail(err)
			return
		}
		if err := chainProjQuantBF16In(t, vw, sc.nBF.buf, sc.vRawBF.buf, sc.vRaw.buf, L, KVH*HD, D); err != nil {
			fail(err)
			return
		}
		t.barrier()
		if err := chainAttnQPrep(t, sc.qRaw.buf, normQ, sc.qPrep.buf, sc.gateBuf.buf, ctx.posBuf, L, H, HD, RD, gated, qkNorm, eps, theta, pos0); err != nil {
			fail(err)
			return
		}
		if err := chainAttnKPrep(t, sc.kRaw.buf, normK, h.kBuf.buf, ctx.posBuf, L, KVH, HD, RD, qkNorm, eps, theta, pos0); err != nil {
			fail(err)
			return
		}
		// V lands through the position-indexed compute append (the blit's ICB-recordable twin).
		if err := chainAttnVAppend(t, sc.vRaw.buf, h.vBuf.buf, ctx.posBuf, L, KVH*HD, pos0); err != nil {
			fail(err)
			return
		}
		t.barrier()
		if err := chainAttnSDPA(t, sc.qPrep.buf, h.kBuf.buf, h.vBuf.buf, sc.attnOut.buf, ctx.posBuf, L, H, KVH, HD, pos0, window); err != nil {
			fail(err)
			return
		}
		if gated != 0 {
			t.barrier()
			if err := chainAttnGateSilu(t, sc.attnOut.buf, sc.gateBuf.buf, L*H*HD); err != nil {
				fail(err)
				return
			}
		}
		t.barrier()
		if err := chainNarrowF32ToBF16(t, sc.attnOut.buf, sc.attnBF.buf, L*mixCols); err != nil {
			fail(err)
			return
		}
		t.barrier()
		if err := chainProjQuantBF16In(t, ow, sc.attnBF.buf, sc.mixBF.buf, sc.mix.buf, L, D, mixCols); err != nil {
			fail(err)
			return
		}
		t.barrier()
		if err := chainResidualNormMLPQuantTail(t, quantTailBufs{
			h: ctx.cur().buf, mix: sc.mix.buf, normed: sc.normed.buf, nBF: sc.nBF.buf,
			g: sc.gFF.buf, gBF: sc.gFFBF.buf, u: sc.uFF.buf, uBF: sc.uFFBF.buf, s: sc.sFF.buf, out: ctx.next().buf,
		}, postNormBuf, gate, up, down, L, D, FF, eps); err != nil {
			fail(err)
			return
		}
		if t.err != nil {
			fail(t.err)
		}
	})
	if encErr != nil {
		return h, encErr
	}
	if ctx.rec != nil {
		// recording pass: nothing executed — the cache position must not advance, and the
		// recording binds THIS state's pins (replay checks identity + capacity).
		ctx.rec.attnStates = append(ctx.rec.attnStates, h)
		ctx.rec.attnKPins = append(ctx.rec.attnKPins, uintptr(unsafe.Pointer(h.kBuf)))
		ctx.rec.attnVPins = append(ctx.rec.attnVPins, uintptr(unsafe.Pointer(h.vBuf)))
	} else {
		h.n = pos0 + L
	}
	ctx.curIsA = !ctx.curIsA
	return h, nil
}

// composed_chain_moe wiring: bind the MoE-tail chain hooks (declared in the composed lib) to the
// native whole-layer bodies + the recordability probe. Separate from composed_bf16_backend's init so
// the MoE lane is self-contained (AX-8: composed declares, the backend binds).
func init() {
	composed.AttnQuantChainMoELayerDevice = attnQuantChainMoELayerDevice
	composed.GatedDeltaQuantChainMoELayerDevice = gatedDeltaQuantChainMoELayerDevice
	composed.MoEChainRecordable = moeChainRecordable
}

// attnQuantChainMoELayerDevice encodes one packed dense-attention layer whose FFN is a MoE onto the
// chain — attnQuantChainLayerDevice's mixer body verbatim, then the MoE tail
// (chainResidualNormMoEQuantTail) instead of the dense SwiGLU tail. LIVE-only: the MoE tail's custom
// router/combine/scale kernels have no ICB-specialised pipeline, so a recording pass declines here
// and the model keeps the re-encode chain (still one command buffer per token).
func attnQuantChainMoELayerDevice(ctxAny, dev any, inputNorm []float32, qw, kw, vw, ow *model.QuantWeight, qNormW, kNormW, postNorm []float32, moe *composed.MoEMLP, priorK, priorV []float32, H, KVH, HD, RD, pos0, window, gated, qkNorm int, eps, theta float32) (any, error) {
	ctx, ok := ctxAny.(*composedChainCtx)
	if !ok {
		return dev, core.NewError("native.attnQuantChainMoELayerDevice: not a chain context")
	}
	if ctx.rec != nil {
		return dev, core.NewError("native.attnQuantChainMoELayerDevice: MoE layers do not record")
	}
	moeW, ok := resolveMoEChainWeights(moe)
	if !ok {
		return dev, core.NewError("native.attnQuantChainMoELayerDevice: MoE weights not chainable")
	}
	L, D := ctx.L, ctx.D
	if !attnCoreUsable(H, KVH, HD, RD) {
		return dev, core.NewError("native.attnQuantChainMoELayerDevice: core not servable")
	}
	qCols := H * HD
	if gated != 0 {
		qCols = 2 * H * HD
	}
	mixCols := H * HD
	if !quantGeometryOK(qw, qCols, D) || !quantGeometryOK(kw, KVH*HD, D) || !quantGeometryOK(vw, KVH*HD, D) ||
		!quantGeometryOK(ow, D, mixCols) {
		return dev, core.NewError("native.attnQuantChainMoELayerDevice: size/geometry mismatch")
	}
	h, _ := dev.(*attnKVDeviceState)
	if h == nil {
		h = &attnKVDeviceState{KVH: KVH, HD: HD}
		if err := h.ensureCap(pos0 + L); err != nil {
			return dev, err
		}
		if pos0 > 0 {
			if len(priorK) != pos0*KVH*HD || len(priorV) != pos0*KVH*HD {
				return dev, core.NewError("native.attnQuantChainMoELayerDevice: prior state size mismatch")
			}
			copy(h.kBuf.bytes, float32Bytes(priorK))
			copy(h.vBuf.bytes, float32Bytes(priorV))
		}
		h.n = pos0
	} else if err := h.ensureCap(pos0 + L); err != nil {
		return h, err
	}
	if h.n != pos0 {
		return h, core.NewError("native.attnQuantChainMoELayerDevice: position desync with resident cache")
	}
	sc, err := ctx.attnScratchFor(attnChainKey{L: L, D: D, qCols: qCols, kvDim: KVH * HD, mixCols: mixCols, FF: D})
	if err != nil {
		return h, err
	}
	moeSc, err := ctx.moeScratchFor(moeChainKey{D: D, nE: moeW.numExperts, topK: moeW.topK, expertDFF: moeW.expertDFF, sharedFF: moeW.sharedFF})
	if err != nil {
		return h, err
	}
	*ctx.posPtr = int32(pos0) // every layer of this token shares the position (idempotent)
	var encErr error
	withAutoreleasePool(func() {
		inNormBuf := residentFloat32(inputNorm)
		postNormBuf := residentFloat32(postNorm)
		normQ, normK := inNormBuf, inNormBuf
		if qkNorm == 1 {
			normQ = residentFloat32(qNormW)
			normK = residentFloat32(kNormW)
		}
		rmsName := "rmsfloat32"
		if D > rmsLoopedLimit {
			rmsName = "rms_loopedfloat32"
		}
		t, done, terr := ctx.layerTarget()
		if terr != nil {
			encErr = terr
			return
		}
		defer done()
		fail := func(err error) { encErr = err }
		psoRMS, perr := t.pso(rmsName)
		if perr != nil {
			fail(perr)
			return
		}
		emitRMSNormRows(t.cmd(), psoRMS, ctx.cur().buf, inNormBuf, sc.normed.buf, 0, 0, 0, D, eps, L, rmsThreadgroup(D, psoRMS))
		t.barrier()
		if err := chainNarrowF32ToBF16(t, sc.normed.buf, sc.nBF.buf, L*D); err != nil {
			fail(err)
			return
		}
		t.barrier()
		if err := chainProjQuantBF16In(t, qw, sc.nBF.buf, sc.qRawBF.buf, sc.qRaw.buf, L, qCols, D); err != nil {
			fail(err)
			return
		}
		if err := chainProjQuantBF16In(t, kw, sc.nBF.buf, sc.kRawBF.buf, sc.kRaw.buf, L, KVH*HD, D); err != nil {
			fail(err)
			return
		}
		if err := chainProjQuantBF16In(t, vw, sc.nBF.buf, sc.vRawBF.buf, sc.vRaw.buf, L, KVH*HD, D); err != nil {
			fail(err)
			return
		}
		t.barrier()
		if err := chainAttnQPrep(t, sc.qRaw.buf, normQ, sc.qPrep.buf, sc.gateBuf.buf, ctx.posBuf, L, H, HD, RD, gated, qkNorm, eps, theta, pos0); err != nil {
			fail(err)
			return
		}
		if err := chainAttnKPrep(t, sc.kRaw.buf, normK, h.kBuf.buf, ctx.posBuf, L, KVH, HD, RD, qkNorm, eps, theta, pos0); err != nil {
			fail(err)
			return
		}
		if err := chainAttnVAppend(t, sc.vRaw.buf, h.vBuf.buf, ctx.posBuf, L, KVH*HD, pos0); err != nil {
			fail(err)
			return
		}
		t.barrier()
		if err := chainAttnSDPA(t, sc.qPrep.buf, h.kBuf.buf, h.vBuf.buf, sc.attnOut.buf, ctx.posBuf, L, H, KVH, HD, pos0, window); err != nil {
			fail(err)
			return
		}
		if gated != 0 {
			t.barrier()
			if err := chainAttnGateSilu(t, sc.attnOut.buf, sc.gateBuf.buf, L*H*HD); err != nil {
				fail(err)
				return
			}
		}
		t.barrier()
		if err := chainNarrowF32ToBF16(t, sc.attnOut.buf, sc.attnBF.buf, L*mixCols); err != nil {
			fail(err)
			return
		}
		t.barrier()
		if err := chainProjQuantBF16In(t, ow, sc.attnBF.buf, sc.mixBF.buf, sc.mix.buf, L, D, mixCols); err != nil {
			fail(err)
			return
		}
		t.barrier()
		if err := chainResidualNormMoEQuantTail(t, moeTailBufs{
			h: ctx.cur().buf, mix: sc.mix.buf, normed: sc.normed.buf, nBF: sc.nBF.buf, out: ctx.next().buf, sc: moeSc,
		}, postNormBuf, moeW, L, D, eps); err != nil {
			fail(err)
			return
		}
		if t.err != nil {
			fail(t.err)
		}
	})
	if encErr != nil {
		return h, encErr
	}
	h.n = pos0 + L
	ctx.curIsA = !ctx.curIsA
	return h, nil
}

// gatedDeltaQuantChainMoELayerDevice is attnQuantChainMoELayerDevice's gated-delta twin:
// gatedDeltaQuantChainLayerDevice's mixer body verbatim, then the MoE tail. LIVE-only (declines
// recording), same as the attn MoE layer.
func gatedDeltaQuantChainMoELayerDevice(ctxAny any, sc *qwen3.GatedDeltaScratch, inputNorm []float32, w *qwen3.GatedDeltaWeights, cfg qwen3.GatedDeltaConfig, postNorm []float32, moe *composed.MoEMLP, priorConv, priorDelta []float32, eps float32) error {
	ctx, ok := ctxAny.(*composedChainCtx)
	if !ok {
		return core.NewError("native.gatedDeltaQuantChainMoELayerDevice: not a chain context")
	}
	if ctx.rec != nil {
		return core.NewError("native.gatedDeltaQuantChainMoELayerDevice: MoE layers do not record")
	}
	moeW, ok := resolveMoEChainWeights(moe)
	if !ok {
		return core.NewError("native.gatedDeltaQuantChainMoELayerDevice: MoE weights not chainable")
	}
	h, _ := sc.Device.(*gatedDeltaDeviceState)
	if h == nil {
		if !gatedDeltaBlockUsable(cfg.HeadDim, cfg.HeadDim, cfg.KeyHeads, cfg.ValueHeads, cfg.ConvKernel) {
			return core.NewError("native.gatedDeltaQuantChainMoELayerDevice: geometry not servable")
		}
		nh, err := newGatedDeltaDeviceState(cfg.KeyHeads, cfg.ValueHeads, cfg.HeadDim, cfg.HeadDim, cfg.ConvKernel, 1)
		if err != nil {
			return err
		}
		h = nh
	}
	L, D := ctx.L, ctx.D
	convDim, vDim := h.convDim, h.Hv*h.Dv
	if w.InProjQKVQ == nil || !quantGeometryOK(w.InProjQKVQ, convDim, D) ||
		w.InProjZQ == nil || !quantGeometryOK(w.InProjZQ, vDim, D) ||
		w.InProjAQ == nil || !quantGeometryOK(w.InProjAQ, h.Hv, D) ||
		w.InProjBQ == nil || !quantGeometryOK(w.InProjBQ, h.Hv, D) ||
		w.OutProjQ == nil || !quantGeometryOK(w.OutProjQ, D, vDim) {
		return core.NewError("native.gatedDeltaQuantChainMoELayerDevice: unsupported quant geometry")
	}
	if !h.valid {
		h.prime(priorConv, priorDelta)
	}
	key := gatedDeltaQuantLayerKey{L: L, D: D, FF: D, Hk: h.Hk, Hv: h.Hv, Dk: h.Dk, K: h.K}
	if ctx.gdSc == nil {
		gsc, err := getGatedDeltaQuantLayerScratch(key)
		if err != nil {
			return err
		}
		ctx.gdSc, ctx.gdKey = gsc, key
	} else if ctx.gdKey != key {
		return core.NewError("native.gatedDeltaQuantChainMoELayerDevice: mixed gd geometries in one chain")
	}
	gsc := ctx.gdSc
	moeSc, err := ctx.moeScratchFor(moeChainKey{D: D, nE: moeW.numExperts, topK: moeW.topK, expertDFF: moeW.expertDFF, sharedFF: moeW.sharedFF})
	if err != nil {
		return err
	}
	rmsName := "rmsfloat32"
	if D > rmsLoopedLimit {
		rmsName = "rms_loopedfloat32"
	}
	var encErr error
	withAutoreleasePool(func() {
		wConv := residentFloat32(w.ConvWeight)
		wBias := wConv
		hasBias := 0
		if w.ConvBias != nil {
			wBias = residentFloat32(w.ConvBias)
			hasBias = 1
		}
		t, done, terr := ctx.layerTarget()
		if terr != nil {
			encErr = terr
			return
		}
		defer done()
		fail := func(err error) { encErr = err }
		psoRMS, perr := t.pso(rmsName)
		if perr != nil {
			fail(perr)
			return
		}
		emitRMSNormRows(t.cmd(), psoRMS, ctx.cur().buf, residentFloat32(inputNorm), gsc.normed1.buf, 0, 0, 0, D, eps, L, rmsThreadgroup(D, psoRMS))
		t.barrier()
		if err := chainNarrowF32ToBF16(t, gsc.normed1.buf, gsc.n1BF.buf, L*D); err != nil {
			fail(err)
			return
		}
		t.barrier()
		if err := chainProjQuantBF16In(t, w.InProjQKVQ, gsc.n1BF.buf, gsc.qkvBF.buf, gsc.qkv.buf, L, convDim, D); err != nil {
			fail(err)
			return
		}
		if err := chainProjQuantBF16In(t, w.InProjZQ, gsc.n1BF.buf, gsc.zBF.buf, gsc.z.buf, L, vDim, D); err != nil {
			fail(err)
			return
		}
		if err := chainProjQuantBF16In(t, w.InProjAQ, gsc.n1BF.buf, gsc.aBF.buf, gsc.a.buf, L, h.Hv, D); err != nil {
			fail(err)
			return
		}
		if err := chainProjQuantBF16In(t, w.InProjBQ, gsc.n1BF.buf, gsc.bBF.buf, gsc.b.buf, L, h.Hv, D); err != nil {
			fail(err)
			return
		}
		t.barrier()
		if err := chainGatedDeltaBlockStages(t, h, gdBlockStageBufs{
			qkv: gsc.qkv.buf, z: gsc.z.buf, a: gsc.a.buf, b: gsc.b.buf,
			qN: gsc.qN.buf, kN: gsc.kN.buf, vN: gsc.vN.buf, g: gsc.g.buf, beta: gsc.beta.buf,
			gated: gsc.gated.buf,
		}, wConv, wBias, hasBias, residentFloat32(w.ALog), residentFloat32(w.DtBias), residentFloat32(w.Norm), L); err != nil {
			fail(err)
			return
		}
		t.barrier()
		if err := chainProjQuantF32(t, w.OutProjQ, gsc.gated.buf, gsc.gatedBF.buf, gsc.mixBF.buf, gsc.mix.buf, L, D, vDim); err != nil {
			fail(err)
			return
		}
		t.barrier()
		if err := chainResidualNormMoEQuantTail(t, moeTailBufs{
			h: ctx.cur().buf, mix: gsc.mix.buf, normed: gsc.normed2.buf, nBF: gsc.n2BF.buf, out: ctx.next().buf, sc: moeSc,
		}, residentFloat32(postNorm), moeW, L, D, eps); err != nil {
			fail(err)
			return
		}
		if t.err != nil {
			fail(t.err)
		}
	})
	if encErr != nil {
		return encErr
	}
	sc.Device = h
	ctx.curIsA = !ctx.curIsA
	return nil
}
