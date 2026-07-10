// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"runtime"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"github.com/tmc/apple/metal"
)

// attnScaleOf is the SDPA scale the model DECLARES (the engine applies it, never
// assumes): gemma4 = 1.0 (its per-head QK-norm is the scaling), standard transformers
// = 1/√headDim. Falls back to 1/√headDim for a hand-built Arch that predates the
// declared field (AttnScale == 0), so existing paths are byte-identical.
func attnScaleOf(arch model.Arch) float32 {
	if arch.AttnScale != 0 {
		return arch.AttnScale
	}
	return float32(1.0 / math.Sqrt(float64(arch.HeadDim)))
}

// embedScaleOf is the token-embedding multiplier the model DECLARES (the engine applies
// it, never assumes): gemma-family = √hidden, llama-family = 1.0. Falls back to √hidden
// for a hand-built Arch that predates the declared field (EmbedScale == 0), so existing
// paths are byte-identical.
func embedScaleOf(arch model.Arch) float32 {
	if arch.EmbedScale != 0 {
		return arch.EmbedScale
	}
	if arch.Hidden <= 0 {
		return 0
	}
	return float32(math.Sqrt(float64(arch.Hidden)))
}

// headDimOf / kvHeadsOf are a layer's RESOLVED attention geometry: gemma4 full_attention
// layers use a larger head_dim (global_head_dim) and may differ in KV heads, declared per
// layer on the spec (pkg/model/gemma4). They fall back to the uniform arch value for a spec
// that predates the per-type resolution (a hand-built Arch), so existing uniform paths are
// byte-identical.
func headDimOf(spec model.LayerSpec, fallback int) int {
	if spec.HeadDim > 0 {
		return spec.HeadDim
	}
	return fallback
}

func kvHeadsOf(spec model.LayerSpec, fallback int) int {
	if spec.KVHeads > 0 {
		return spec.KVHeads
	}
	return fallback
}

// encAttnHalfShared is the KV-SHARING attention half: a layer that shares another
// layer's KV cache projects ONLY its query (from its own input) and attends over
// the owner's cache — no K/V projection, no K-RoPE, no cache write. attendK/attendV
// are the owner's seq-major caches; the window N=pos+1 is the owner's live length
// (the owner wrote row pos earlier this token). Writes x + Wo·attn -> h.
func encAttnHalfShared(
	enc metal.MTLComputeCommandEncoder,
	x, attendK, attendV, offBuf, h metal.MTLBuffer,
	attnNormW, postAttnNorm, qNorm bufView,
	sc attnScratch, proj projector,
	dModel, nHeads, nKVHeads, headDim, pos, slideW, rotaryDim int, base, scale, eps float32,
	ropeFreqs metal.MTLBuffer,
) error {
	kvDim := nKVHeads * headDim
	if err := encRMSNormBF16(enc, x, attnNormW.buf, sc.normed, attnNormW.off, dModel, eps); err != nil {
		return err
	}
	if err := proj.project(enc, sc.normed, sc.q, 0, projQ); err != nil {
		return err
	}
	if gpuHasGeluKernel() && qNorm.buf != nil {
		// fused: sc.q = RoPE(RMSNorm(sc.q, qNorm)) in one op — lockstep with the ICB setQKNormRope
		if err := encQKNormRope(enc, sc.q, qNorm.buf, sc.q, 0, qNorm.off, 0, offBuf, ropeFreqs, nHeads, headDim, rotaryDim, base, scale, eps); err != nil {
			return err
		}
	} else {
		if qNorm.buf != nil { // gemma4 per-head QK-norm before RoPE (sharers project only Q)
			if err := encRMSNormRowsBF16(enc, sc.q, qNorm.buf, sc.q, 0, qNorm.off, 0, nHeads, headDim, eps); err != nil {
				return err
			}
		}
		// RoPE Q in place so partial rotary's untouched tail keeps the projected value.
		if err := encRopeDecode(enc, sc.q, sc.q, 0, 0, offBuf, ropeFreqs, nHeads, headDim, rotaryDim, base, scale); err != nil {
			return err
		}
	}
	// attend the OWNER's cache (no write): the whole seq-major cache (global) or the whole live ring
	// (sliding, slideW>0) — n live rows from offset 0, matching the owner's ring write in encAttnHalfKV.
	n := pos + 1
	if slideW > 0 && n > slideW {
		n = slideW
	}
	if err := encSDPADecode(enc, sc, sc.q, attendK, attendV, sc.attn,
		nHeads, nKVHeads, headDim, n,
		int64(headDim), int64(kvDim), int64(headDim), int64(kvDim), scale, 0); err != nil {
		return err
	}
	if err := proj.project(enc, sc.attn, sc.attnOut, 0, projO); err != nil {
		return err
	}
	return encResidualMaybeNorm(enc, x, sc.attnOut, sc.normed, h, postAttnNorm, dModel, eps)
}

// encAttnHalfSharedInputAt is encAttnHalfShared with the layer input bound at xOff and the
// per-row position bound at offOff — the batched dense prefill's row shape (mirrors
// encAttnHalfKVInputAt). Row i attends the owner's cache capped at its own live length; the
// owner's rows for this batch were encoded earlier in the same command buffer (lower layer
// index), and Metal's hazard tracking orders the cross-row write→read exactly as the
// sequential per-token chain would.
func encAttnHalfSharedInputAt(
	enc metal.MTLComputeCommandEncoder,
	x metal.MTLBuffer, xOff uint, attendK, attendV, offBuf, h metal.MTLBuffer, hOff, offOff uint,
	attnNormW, postAttnNorm, qNorm bufView,
	sc attnScratch, proj projector,
	dModel, nHeads, nKVHeads, headDim, pos, slideW, rotaryDim int, base, scale, eps float32,
	ropeFreqs metal.MTLBuffer,
) error {
	kvDim := nKVHeads * headDim
	// entry rms via the size-specialised single-row kernel at the row's offset — the batched
	// interleave's rows must norm bit-identically to the sequential step (the generic rows
	// kernel reduces in a different order and drifts the whole layer by ulps).
	if err := encRMSNormBF16At(enc, x, attnNormW.buf, sc.normed, xOff, attnNormW.off, 0, dModel, eps); err != nil {
		return err
	}
	if err := proj.project(enc, sc.normed, sc.q, 0, projQ); err != nil {
		return err
	}
	if gpuHasGeluKernel() && qNorm.buf != nil {
		// fused: sc.q = RoPE(RMSNorm(sc.q, qNorm)) in one op — lockstep with the ICB setQKNormRope
		if err := encQKNormRopeAt(enc, sc.q, qNorm.buf, sc.q, 0, qNorm.off, 0, offBuf, offOff, ropeFreqs, nHeads, headDim, rotaryDim, base, scale, eps); err != nil {
			return err
		}
	} else {
		if qNorm.buf != nil { // gemma4 per-head QK-norm before RoPE (sharers project only Q)
			if err := encRMSNormRowsBF16(enc, sc.q, qNorm.buf, sc.q, 0, qNorm.off, 0, nHeads, headDim, eps); err != nil {
				return err
			}
		}
		if err := encRopeDecodeAt(enc, sc.q, sc.q, 0, 0, offBuf, offOff, ropeFreqs, nHeads, headDim, rotaryDim, base, scale); err != nil {
			return err
		}
	}
	// attend the OWNER's cache (no write): n live rows, matching encAttnHalfShared.
	n := pos + 1
	if slideW > 0 && n > slideW {
		n = slideW
	}
	if err := encSDPADecode(enc, sc, sc.q, attendK, attendV, sc.attn,
		nHeads, nKVHeads, headDim, n,
		int64(headDim), int64(kvDim), int64(headDim), int64(kvDim), scale, 0); err != nil {
		return err
	}
	if err := proj.project(enc, sc.attn, sc.attnOut, 0, projO); err != nil {
		return err
	}
	return encResidualMaybeNormAt(enc, x, xOff, sc.attnOut, 0, sc.normed, h, hOff, postAttnNorm, dModel, eps)
}

// archLayerBufs holds one layer's resident buffers for runArchDecode: bf16 norms +
// the (bf16 or 4-bit qmv) projector + the growing KV caches. kCache/vCache are nil for
// sharer layers (they attend the owner's); mnw and the projector's MLP weights are
// unbound for MoE layers (MoEBlockBF16 owns that FFN).
type archLayerBufs struct {
	anw, mnw                 bufView
	postAttnNorm, postFFNorm bufView         // gemma4 post-attn/post-FF norms (nil buf = skip)
	qNorm, kNorm             bufView         // gemma4 per-head QK-norm (nil buf = skip)
	layerScalar              metal.MTLBuffer // gemma4 per-layer output scalar, broadcast to dModel (synthesised, nil = skip)
	kCache, vCache           metal.MTLBuffer
	kCachePtr, vCachePtr     *byte
	proj                     projector
	dFF                      int // this layer's FFN width (gemma4 E2B/E4B vary it per layer)
}

func (lb *archLayerBufs) cacheKVContents() {
	if lb == nil {
		return
	}
	if lb.kCache != nil {
		lb.kCachePtr = (*byte)(lb.kCache.Contents())
	}
	if lb.vCache != nil {
		lb.vCachePtr = (*byte)(lb.vCache.Contents())
	}
}

// archDecodeState holds the resident buffers of an arch decode — the per-layer weights/
// caches (lb), shared scratch, and the position buffer — so a single token can be stepped
// repeatedly over a PERSISTENT, growing KV cache. Both the whole-sequence runArchDecode and
// the incremental generation loop build one (inside a withAutoreleasePool) and call
// stepToken per token; the caches in lb persist across calls within that pool, which is
// what turns the O(N²) re-decode into O(1)/token incremental decode.
type archDecodeState struct {
	specs        []model.LayerSpec
	lb           []archLayerBufs
	moeWeights   []*MoELayerWeights
	pagedKV      []*devicePagedKVCache
	asc          attnScratch
	msc          mlpScratch
	coreScratch  *archDecodeCoreScratch
	hBuf, xA, xB metal.MTLBuffer
	denseBatch   denseBatchScratch
	offBuf       metal.MTLBuffer
	offPtr       *int32
	hBufPtr      *byte
	xAPtr, xBPtr *byte
	// verifyFoldSmallK lets the MTP verify take the batched fold on a
	// recorded-ICB session BELOW batchedDenseICBMaxRows (the per-row
	// interleave reads every quant weight K times — K× a plain decode step,
	// which erased the speculative win on dense targets). Scoped to the
	// assistant verify only (set around verifyAssistantDraftHiddens); the
	// fold is the same token-identity tier the prompt-scale qmm already
	// trades at, and the routing is deterministic (every verify folds), so
	// live and restored sessions write the same lane's bytes.
	verifyFoldSmallK bool
	// verifyTail is the recorded MTP verify-tail ICB (#372): the fold's
	// pos-independent per-layer tail (O-projection → layer scalar) recorded once
	// during the first eligible verify and replayed per layer thereafter.
	// verifyTailRec is live only during the recording pass; verifyTailTried
	// stops a failed recording from re-arming every block.
	verifyTail      *verifyTailICB
	verifyTailRec   *verifyTailRecorder
	verifyTailTried bool
	// rowAttnCaps, when non-nil, overrides each batch row's visible attention
	// length (absolute kv rows) — the bidirectional image-span prefill
	// (gemma4_unified): span rows see through to their span end. Legal only on
	// the batched-rope attention fold, where the WHOLE chunk's K/V lands
	// before any SDPA reads; the pass hard-errors rather than fall to a lane
	// that would silently evaluate the span causally. Transient — set around
	// one chunk by prefillRetainedEmbeddingsBidirChunk.
	rowAttnCaps []int32
	ropeFreqs   metal.MTLBuffer // resident periods (1/inv_freq) for YaRN long-context rope; nil = base-derived rope
	// gemma4 global (proportional+partial) rope: the period spectrum over the FULL head dim
	// (metal's gemma4ProportionalFreqs) for GlobalAttention layers, so rope pairs (d, d+globalHeadDim/2)
	// over the whole head — NOT (d, d+rotaryDim/2). nil ⇒ no proportional global layers.
	globalRopeFreqs metal.MTLBuffer
	globalHeadDim   int             // the full head dim global layers rope over (passed as rotaryDim to the freqs path)
	valueNormOnes   metal.MTLBuffer // gemma4 value-norm: [maxHeadDim] ones weight for the no-scale per-head RMSNorm on V; nil = no value-norm (Mistral)

	dModel, nHeads, nKVHeads, headDim, dFF, slidingWindow, maxLen int
	rotaryDim, rotaryDimLocal                                     int     // partial-rotary dims (global / sliding); == headDim is full
	base, localBase, scale, eps                                   float32 // localBase = sliding-layer RoPE theta

	// gemma4 per-layer-input tower (E2B/E4B): when ple is non-nil, each layer's output is gated
	// by PerLayerInputGateQuant before layer_scalar, fed its pliDim slice of perLayerInput (the
	// PerLayerInputs tensor, set per token). nil = no PLE tower (dense models — byte-identical).
	ple                  []pleLayer
	perLayerInput        []byte // [numLayers·pliDim] bf16, set before each token's stepToken
	perLayerInputBuf     metal.MTLBuffer
	perLayerInputLen     int
	pliDim               int
	hostScratch          []byte // reusable dModel bf16 host handoff for tests and non-buffer host-orchestrated branches
	hostPinnedScratch    *pinnedNoCopyBytes
	inputEmbScratch      *pinnedNoCopyBytes
	inputEmbCandidate    uintptr
	inputEmbCandidateLen int
	inputEmbCandidateHit int
	pleGateScratch       *perLayerInputGateScratch
	pleInputScratch      *pinnedNoCopyBytes
	pleSlabScratch       *pinnedNoCopyBytes // batched dense prefill: K tokens' PLE tensors in one pinned slab

	// gemma4 4-bit MoE (26B-A4B): moeQuant[li] != nil runs MoEBlockQuant for that layer's FFN
	// (host-orchestrated like the bf16 MoE). nil entries use the dense MLP / bf16 moeWeights.
	moeQuant []*MoEQuantLayerWeights
	// moeOwnedScratch is the state's OWN MoE block scratch (lazily built at the first quant MoE
	// layer): single-flight by construction, so the fully-device MoE path can recycle it across
	// layers without a per-layer completion wait — the queue's commit order is the only
	// synchroniser (the pooled path must wait, because pools cross sessions). Only long-lived
	// SESSION states own one (moeScratchOwnable, set by the session builders): a standalone
	// forward's state dies per call, so owning would re-allocate the scratch every forward —
	// it keeps the per-layer pool round-trip and its wait instead.
	moeScratchOwnable     bool
	moeOwnedScratch       *moeBlockBF16Scratch
	moeRouterOwnedScratch *routerDeviceScratch

	// trace (LTHN_NATIVE_TRACE): when set, stepToken flushes + reads back each layer's output
	// hidden and logs the per-token worst max-abs + NaN layer — the decode-degradation probe.
	trace bool

	// gpuProf, when armed (tests only — nil in production), splits stepToken's per-token encoder
	// at the attn/moe family seams with timestamp counter sampling: the per-family GPU time table.
	gpuProf *gpuCounterProfiler

	// chainTail, when set (the session's chained live decode — nil otherwise), encodes the head
	// argmax + the next token's input production into the step's OWN command buffer right before
	// its commit: one cb and one wait per token, no host embed/argmax between. hidden is the
	// final post-stack buffer. Skipped when this token's encoding broke the cb (MoE break-out /
	// test probes) — the session detects the miss and finishes that token serially.
	chainTail func(enc metal.MTLComputeCommandEncoderObject, hidden metal.MTLBuffer) error

	// chainSkipWait, with chainTail set, makes the step COMMIT its command buffer and return
	// immediately — no wait, no host readback — leaving the committed cb in chainPendingCB.
	// The submit-ahead decode uses this to encode token N+1 while N still runs; the caller
	// owns the wait.
	chainSkipWait  bool
	chainPendingCB metal.MTLCommandBufferObject

	// icb, when non-nil, is the recorded arch ICB the session replays per token (the encode-bypass)
	// instead of re-encoding via stepToken. Set at session build when icbEligible (no MoE, no trace,
	// uniform head geometry + simple uniform rope — the ICB core's assumptions). It holds its OWN
	// maxLen-linear caches (NOT the state's lb ring caches), so an ICB session decodes EVERY token
	// (prefill + decode) through it. nil ⇒ stepToken.
	icb *archICBReplay
}

func (s *archDecodeState) hostHiddenScratch(dModel int) []byte {
	n := dModel * bf16Size
	if cap(s.hostScratch) < n {
		s.hostScratch = make([]byte, n)
	}
	return s.hostScratch[:n]
}

func (s *archDecodeState) hostHiddenPinnedScratch(dModel int) ([]byte, metal.MTLBuffer, error) {
	if s == nil {
		return nil, nil, core.NewError("native.archDecodeState.hostHiddenPinnedScratch: state is nil")
	}
	n := dModel * bf16Size
	if n <= 0 {
		return nil, nil, core.NewError("native.archDecodeState.hostHiddenPinnedScratch: hidden size must be > 0")
	}
	if s.coreScratch != nil {
		p, err := s.coreScratch.hostPinnedScratch(n)
		if err != nil {
			return nil, nil, err
		}
		if p != nil {
			s.hostPinnedScratch = p
			return p.bytes, p.buf, nil
		}
	}
	if s.hostPinnedScratch == nil || len(s.hostPinnedScratch.bytes) != n {
		if s.hostPinnedScratch != nil {
			s.hostPinnedScratch.Close()
			s.hostPinnedScratch = nil
		}
		var err error
		s.hostPinnedScratch, err = newPinnedNoCopyBytes(n)
		if err != nil {
			return nil, nil, err
		}
	}
	return s.hostPinnedScratch.bytes, s.hostPinnedScratch.buf, nil
}

func (s *archDecodeState) perLayerInputGateScratch() *perLayerInputGateScratch {
	if s.pleGateScratch == nil || s.pleGateScratch.dModel != s.dModel || s.pleGateScratch.pliDim != s.pliDim {
		if s.pleGateScratch != nil {
			s.pleGateScratch.Close()
		}
		s.pleGateScratch = newPerLayerInputGateScratch(s.dModel, s.pliDim)
	}
	return s.pleGateScratch
}

func (s *archDecodeState) inputEmbBuffer(inputEmb []byte, dModel int) (metal.MTLBuffer, bool) {
	if s == nil || len(inputEmb) != dModel*bf16Size || len(inputEmb) == 0 {
		return nil, false
	}
	if s.inputEmbScratch != nil && len(s.inputEmbScratch.bytes) == len(inputEmb) && &s.inputEmbScratch.bytes[0] == &inputEmb[0] {
		return s.inputEmbScratch.buf, true
	}
	if s.inputEmbScratch != nil {
		s.inputEmbScratch.Close()
		s.inputEmbScratch = nil
	}
	if isMappedShardBytes(inputEmb) {
		return nil, false
	}
	pinner := pinGoBytes(inputEmb)
	if pinner == nil {
		return nil, false
	}
	buf := newNoCopyBuffer(unsafe.Pointer(&inputEmb[0]), uint(len(inputEmb)))
	if buf == nil || buf.GetID() == 0 {
		pinner.Unpin()
		return nil, false
	}
	s.inputEmbScratch = &pinnedNoCopyBytes{bytes: inputEmb, buf: buf, pinner: pinner}
	runtime.SetFinalizer(s.inputEmbScratch, (*pinnedNoCopyBytes).Close)
	return buf, true
}

func (s *archDecodeState) stableInputEmbBuffer(inputEmb []byte, dModel int) (metal.MTLBuffer, bool) {
	if s == nil || len(inputEmb) != dModel*bf16Size || len(inputEmb) == 0 {
		return nil, false
	}
	if s.inputEmbScratch != nil && len(s.inputEmbScratch.bytes) == len(inputEmb) && &s.inputEmbScratch.bytes[0] == &inputEmb[0] {
		return s.inputEmbScratch.buf, true
	}
	ptr := uintptr(unsafe.Pointer(&inputEmb[0]))
	if s.inputEmbCandidate != ptr || s.inputEmbCandidateLen != len(inputEmb) {
		s.inputEmbCandidate = ptr
		s.inputEmbCandidateLen = len(inputEmb)
		s.inputEmbCandidateHit = 1
		return nil, false
	}
	s.inputEmbCandidateHit++
	if s.inputEmbCandidateHit < 3 {
		return nil, false
	}
	return s.inputEmbBuffer(inputEmb, dModel)
}

func (s *archDecodeState) hostPLEInputBuffer(want int) (metal.MTLBuffer, error) {
	if s == nil {
		return nil, core.NewError("native.archDecodeState.hostPLEInputBuffer: state is nil")
	}
	if len(s.perLayerInput) != want {
		return nil, core.NewError("native.archDecodeState.hostPLEInputBuffer: PLE tensor size mismatch")
	}
	if want <= 0 {
		return nil, core.NewError("native.archDecodeState.hostPLEInputBuffer: PLE tensor must be non-empty")
	}
	if s.pleInputScratch != nil && len(s.pleInputScratch.bytes) == want && &s.pleInputScratch.bytes[0] == &s.perLayerInput[0] {
		return s.pleInputScratch.buf, nil
	}
	if s.pleInputScratch != nil {
		s.pleInputScratch.Close()
		s.pleInputScratch = nil
	}
	if !isMappedShardBytes(s.perLayerInput) {
		pinner := pinGoBytes(s.perLayerInput)
		if pinner != nil {
			buf := newNoCopyBuffer(unsafe.Pointer(&s.perLayerInput[0]), uint(want))
			if buf != nil && buf.GetID() != 0 {
				s.pleInputScratch = &pinnedNoCopyBytes{bytes: s.perLayerInput, buf: buf, pinner: pinner}
				runtime.SetFinalizer(s.pleInputScratch, (*pinnedNoCopyBytes).Close)
				return buf, nil
			}
			pinner.Unpin()
		}
	}
	var err error
	s.pleInputScratch, err = newPinnedNoCopyBytes(want)
	if err != nil {
		return nil, err
	}
	return s.pleInputScratch.copyBuffer(s.perLayerInput)
}

// pleSlabBuffer pins a batched-prefill PLE slab (K tokens × numLayers·pliDim bf16, token-major)
// into a reusable device buffer. The copy is K·plDim bytes — trivial against the per-token host
// round-trips the batched path exists to remove.
func (s *archDecodeState) pleSlabBuffer(slab []byte) (metal.MTLBuffer, error) {
	if s == nil {
		return nil, core.NewError("native.archDecodeState.pleSlabBuffer: state is nil")
	}
	if len(slab) == 0 {
		return nil, core.NewError("native.archDecodeState.pleSlabBuffer: empty PLE slab")
	}
	if s.pleSlabScratch != nil && len(s.pleSlabScratch.bytes) != len(slab) {
		s.pleSlabScratch.Close()
		s.pleSlabScratch = nil
	}
	if s.pleSlabScratch == nil {
		scratch, err := newPinnedNoCopyBytes(len(slab))
		if err != nil {
			return nil, err
		}
		s.pleSlabScratch = scratch
	}
	return s.pleSlabScratch.copyBuffer(slab)
}

func (s *archDecodeState) Close() {
	if s == nil {
		return
	}
	if s.pleGateScratch != nil {
		s.pleGateScratch.Close()
		s.pleGateScratch = nil
	}
	if s.pleSlabScratch != nil {
		s.pleSlabScratch.Close()
		s.pleSlabScratch = nil
	}
	if s.pleInputScratch != nil {
		s.pleInputScratch.Close()
		s.pleInputScratch = nil
	}
	if s.inputEmbScratch != nil {
		s.inputEmbScratch.Close()
		s.inputEmbScratch = nil
	}
	s.denseBatch.Close()
	for _, cache := range s.pagedKV {
		if cache != nil {
			cache.Close()
		}
	}
	s.pagedKV = nil
	s.inputEmbCandidate = 0
	s.inputEmbCandidateLen = 0
	s.inputEmbCandidateHit = 0
	if s.hostPinnedScratch != nil && (s.coreScratch == nil || s.hostPinnedScratch != s.coreScratch.hostPinned) {
		s.hostPinnedScratch.Close()
	}
	s.hostPinnedScratch = nil
	if s.coreScratch != nil {
		putArchDecodeCoreScratch(s.coreScratch)
		s.coreScratch = nil
	}
}

func (s *archDecodeState) bufferPtr(buf metal.MTLBuffer) *byte {
	if s == nil || buf == nil {
		return nil
	}
	switch buf {
	case s.hBuf:
		if s.hBufPtr != nil {
			return s.hBufPtr
		}
	case s.xA:
		if s.xAPtr != nil {
			return s.xAPtr
		}
	case s.xB:
		if s.xBPtr != nil {
			return s.xBPtr
		}
	}
	return (*byte)(buf.Contents())
}

func (s *archDecodeState) initDevicePagedKV(pageSize int) error {
	return s.initDevicePagedKVWithPrealloc(pageSize, false)
}

func (s *archDecodeState) initDevicePagedKVWithPrealloc(pageSize int, prealloc bool) error {
	if s == nil {
		return core.NewError("native.archDecodeState.initDevicePagedKV: nil state")
	}
	for _, cache := range s.pagedKV {
		if cache != nil {
			cache.Close()
		}
	}
	if len(s.specs) == 0 {
		s.pagedKV = nil
		return nil
	}
	pages := make([]*devicePagedKVCache, len(s.specs))
	for li, spec := range s.specs {
		if !spec.OwnsCache() {
			continue
		}
		cacheMax := s.maxLen
		ring := false
		if s.slidingWindow > 0 && s.slidingWindow < s.maxLen && spec.Attention != model.GlobalAttention {
			cacheMax = s.slidingWindow
			ring = true
		}
		lkv, lhd := kvHeadsOf(spec, s.nKVHeads), headDimOf(spec, s.headDim)
		cache, err := newDevicePagedKVCache(lkv, lhd, cacheMax, pageSize)
		if err != nil {
			for _, prior := range pages {
				if prior != nil {
					prior.Close()
				}
			}
			return err
		}
		cache.ring = ring
		if kvQ8Enabled && s.nHeads == 2*lkv && lhd <= 256 && (lkv*lhd)%kvQ8GroupSize == 0 {
			// q8 pages exist only for gqa2 geometry (the only q8 SDPA kernels).
			// Layers outside it keep bf16 pages — mixed modes are fine, every
			// landing and read site branches per cache. The all-miss case is
			// caught below: a requested quantised cache never silently
			// downgrades wholesale.
			cache.quantQ8 = true
		}
		if prealloc {
			if err := cache.preallocPages(); err != nil {
				for _, prior := range pages {
					if prior != nil {
						prior.Close()
					}
				}
				cache.Close()
				return err
			}
		}
		pages[li] = cache
	}
	if kvQ8Requested {
		// EXPLICIT request only: the default (kvQ8Enabled) silently runs bf16 on
		// non-gqa2 geometry, but an explicit LTHN_KV_Q8=1 that can't apply is an error.
		anyQ8 := false
		for _, cache := range pages {
			if cache != nil && cache.quantQ8 {
				anyQ8 = true
				break
			}
		}
		if !anyQ8 {
			for _, cache := range pages {
				if cache != nil {
					cache.Close()
				}
			}
			return core.NewError("native.initDevicePagedKV: LTHN_KV_Q8=1 set but no layer has gqa2 geometry (nHeads == 2*kvHeads, headDim <= 256)")
		}
	}
	s.pagedKV = pages
	return nil
}

func (s *archDecodeState) layerPagedKV(li int) *devicePagedKVCache {
	if s == nil || li < 0 || li >= len(s.pagedKV) {
		return nil
	}
	return s.pagedKV[li]
}

func (s *archDecodeState) hasDevicePagedKV() bool {
	if s == nil {
		return false
	}
	for _, cache := range s.pagedKV {
		if cache != nil {
			return true
		}
	}
	return false
}

func (s *archDecodeState) resetDevicePagedAttentionScratch() {
	if s == nil {
		return
	}
	for _, cache := range s.pagedKV {
		cache.resetAttentionScratchCursor()
	}
}

func (s *archDecodeState) reloadDevicePagedKVFromLinear(position int) error {
	if s == nil || !s.hasDevicePagedKV() {
		return nil
	}
	for li, spec := range s.specs {
		cache := s.layerPagedKV(li)
		if cache == nil || !spec.OwnsCache() {
			continue
		}
		if li >= len(s.lb) || s.lb[li].kCache == nil || s.lb[li].vCache == nil {
			return core.NewError("native.archDecodeState.reloadDevicePagedKVFromLinear: missing linear cache")
		}
		lkv, lhd := kvHeadsOf(spec, s.nKVHeads), headDimOf(spec, s.headDim)
		rowBytes := lkv * lhd * bf16Size
		if rowBytes <= 0 {
			return core.NewError("native.archDecodeState.reloadDevicePagedKVFromLinear: invalid row bytes")
		}
		cacheBytes := int(bufferLengthFast(s.lb[li].kCache))
		if cacheBytes%rowBytes != 0 || int(bufferLengthFast(s.lb[li].vCache)) != cacheBytes {
			return core.NewError("native.archDecodeState.reloadDevicePagedKVFromLinear: cache size mismatch")
		}
		rows := cacheBytes / rowBytes
		tokens := max(min(position, rows), 0)
		s.lb[li].cacheKVContents()
		if err := cache.loadLinearSnapshot(unsafe.Slice(s.lb[li].kCachePtr, cacheBytes), unsafe.Slice(s.lb[li].vCachePtr, cacheBytes), tokens); err != nil {
			return err
		}
		if cache.ring {
			cache.offset = position
			cache.length = tokens
			cache.linearSynced = tokens
		}
	}
	return nil
}

func (s *archDecodeState) syncLinearKVFromDevicePaged(position int) error {
	if s == nil || !s.hasDevicePagedKV() {
		return nil
	}
	if position < 0 {
		return core.NewError("native.archDecodeState.syncLinearKVFromDevicePaged: negative position")
	}
	for li, spec := range s.specs {
		cache := s.layerPagedKV(li)
		if cache == nil || !spec.OwnsCache() {
			continue
		}
		if position < cache.length {
			if err := cache.truncate(position); err != nil {
				return err
			}
		}
		if cache.ring {
			if li >= len(s.lb) || s.lb[li].kCache == nil || s.lb[li].vCache == nil {
				return core.NewError("native.archDecodeState.syncLinearKVFromDevicePaged: missing linear cache")
			}
			lkv, lhd := kvHeadsOf(spec, s.nKVHeads), headDimOf(spec, s.headDim)
			rowBytes := lkv * lhd * bf16Size
			if rowBytes <= 0 {
				return core.NewError("native.archDecodeState.syncLinearKVFromDevicePaged: invalid row bytes")
			}
			rows := cache.length
			if rows > cache.maxSize && cache.maxSize > 0 {
				rows = cache.maxSize
			}
			if rows <= 0 {
				continue
			}
			n := rows * rowBytes
			cacheBytes := int(bufferLengthFast(s.lb[li].kCache))
			if n > cacheBytes || int(bufferLengthFast(s.lb[li].vCache)) != cacheBytes {
				return core.NewError("native.archDecodeState.syncLinearKVFromDevicePaged: cache size mismatch")
			}
			_, _, kPtr, vPtr, err := cache.linearSnapshot(rows)
			if err != nil {
				return err
			}
			s.lb[li].cacheKVContents()
			copy(unsafe.Slice(s.lb[li].kCachePtr, n), unsafe.Slice(kPtr, n))
			copy(unsafe.Slice(s.lb[li].vCachePtr, n), unsafe.Slice(vPtr, n))
			cache.linearSynced = rows
			continue
		}
		if position > cache.length {
			return core.NewError("native.archDecodeState.syncLinearKVFromDevicePaged: page cache shorter than position")
		}
		start := min(cache.linearSynced, position)
		if start == position {
			continue
		}
		if li >= len(s.lb) || s.lb[li].kCache == nil || s.lb[li].vCache == nil {
			return core.NewError("native.archDecodeState.syncLinearKVFromDevicePaged: missing linear cache")
		}
		lkv, lhd := kvHeadsOf(spec, s.nKVHeads), headDimOf(spec, s.headDim)
		rowBytes := lkv * lhd * bf16Size
		if rowBytes <= 0 {
			return core.NewError("native.archDecodeState.syncLinearKVFromDevicePaged: invalid row bytes")
		}
		startBytes := start * rowBytes
		n := position * rowBytes
		cacheBytes := int(bufferLengthFast(s.lb[li].kCache))
		if n > cacheBytes || int(bufferLengthFast(s.lb[li].vCache)) != cacheBytes {
			return core.NewError("native.archDecodeState.syncLinearKVFromDevicePaged: cache size mismatch")
		}
		_, _, kPtr, vPtr, err := cache.linearSnapshot(position)
		if err != nil {
			return err
		}
		s.lb[li].cacheKVContents()
		copy(unsafe.Slice(s.lb[li].kCachePtr, n)[startBytes:], unsafe.Slice(kPtr, n)[startBytes:])
		copy(unsafe.Slice(s.lb[li].vCachePtr, n)[startBytes:], unsafe.Slice(vPtr, n)[startBytes:])
		cache.linearSynced = position
	}
	return nil
}

func (s *archDecodeState) truncateDevicePagedKV(position int) error {
	if s == nil || !s.hasDevicePagedKV() {
		return nil
	}
	for _, cache := range s.pagedKV {
		if cache == nil {
			continue
		}
		if err := cache.truncate(position); err != nil {
			return err
		}
	}
	return nil
}

func (s *archDecodeState) bufferBytes(buf metal.MTLBuffer, n int) []byte {
	return unsafe.Slice(s.bufferPtr(buf), n)
}

// pleLayer is one layer's per-layer-input gate weights: the 4-bit gate + projection and the
// bf16 post-norm. A nil postNorm marks a layer with no gate (so a mixed model is fine).
type pleLayer struct {
	gate, proj      QuantWeight
	postNorm        []byte
	groupSize, bits int
}

// ArchPLEBF16 is the token-id-aware PLE payload for a bf16 whole-sequence arch decode.
// TokenIDs line up with the input embeddings passed to DecodeForwardArch/ICB; the PLE
// tensor is computed as PerLayerInputs(id, inputEmbedding) before each token is decoded.
type ArchPLEBF16 struct {
	TokenIDs           []int32
	EmbedPerLayer      []byte
	PerLayerModelProjW []byte
	PerLayerProjNormW  []byte
	VocabPLI, PliDim   int
}

// ArchPLEQuant is the token-id-aware PLE payload for a quant whole-sequence arch decode.
// The embed-per-layer and optional model projection triples are the bookend weights
// consumed by PerLayerInputs; the per-layer gate/projection weights live on qlayers.
type ArchPLEQuant struct {
	TokenIDs []int32

	EmbedPerLayer, EmbedPerLayerScales, EmbedPerLayerBiases              []byte
	PerLayerModelProjW, PerLayerModelProjScales, PerLayerModelProjBiases []byte
	PerLayerProjNormW                                                    []byte

	VocabPLI, PliDim        int
	GroupSize, Bits         int
	ProjGroupSize, ProjBits int
}

type archDecodePLEInputs struct {
	tokenIDs      []int32
	compute       func(id int32, emb []byte) ([]byte, error)
	computeBuffer func(id int32, emb []byte, embBuf metal.MTLBuffer) (int, metal.MTLBuffer, []byte, error)
	scratch       *plHostScratch
	buffer        metal.MTLBuffer
}

func (p *archDecodePLEInputs) Close() {
	if p == nil {
		return
	}
	if p.scratch != nil {
		p.scratch.Close()
	}
	p.scratch = nil
	p.buffer = nil
}

func (p *archDecodePLEInputs) ensureScratch(plDim, dModel int, projScale float32) (*plHostScratch, error) {
	if p == nil {
		return nil, core.NewError("native.archDecodePLEInputs.ensureScratch: runtime is nil")
	}
	if p.scratch == nil {
		scratch, err := newPLHostScratch(plDim, dModel, projScale)
		if err != nil {
			return nil, err
		}
		p.scratch = scratch
		return scratch, nil
	}
	if p.scratch.plDim != plDim || p.scratch.dModel != dModel {
		return nil, core.NewError("native.archDecodePLEInputs.ensureScratch: scratch dimension mismatch")
	}
	return p.scratch, nil
}

func singleArchPLEBF16(fn string, ple []ArchPLEBF16) (*ArchPLEBF16, error) {
	if len(ple) == 0 {
		return nil, nil
	}
	if len(ple) > 1 {
		return nil, core.NewError(fn + ": at most one PLE payload is supported")
	}
	return &ple[0], nil
}

func singleArchPLEQuant(fn string, ple []ArchPLEQuant) (*ArchPLEQuant, error) {
	if len(ple) == 0 {
		return nil, nil
	}
	if len(ple) > 1 {
		return nil, core.NewError(fn + ": at most one PLE payload is supported")
	}
	return &ple[0], nil
}

func archPLEBF16Runtime(fn string, p *ArchPLEBF16, nLayers, T, dModel int, eps float32) (*archDecodePLEInputs, int, error) {
	if p == nil {
		return nil, 0, nil
	}
	if len(p.TokenIDs) != T {
		return nil, 0, core.NewError(fn + ": PLE token id count must equal inputs")
	}
	if p.VocabPLI <= 0 || p.PliDim <= 0 {
		return nil, 0, core.NewError(fn + ": PLE vocab and hidden dims must be > 0")
	}
	if len(p.PerLayerProjNormW) != p.PliDim*bf16Size {
		return nil, 0, core.NewError(fn + ": PLE projection norm must be pliDim bf16 bytes")
	}
	rt := &archDecodePLEInputs{tokenIDs: p.TokenIDs}
	var projView bufView
	plDim := nLayers * p.PliDim
	projScale := float32(1.0 / math.Sqrt(float64(dModel)))
	ensureResident := func() (*plHostScratch, error) {
		if projView.buf == nil {
			projView = bf16WeightView(p.PerLayerModelProjW, bufView{})
		}
		return rt.ensureScratch(plDim, dModel, projScale)
	}
	rt.compute = func(id int32, emb []byte) ([]byte, error) {
		var scratch *plHostScratch
		if len(p.PerLayerModelProjW) > 0 {
			var err error
			scratch, err = ensureResident()
			if err != nil {
				return nil, err
			}
		}
		out, err := PerLayerInputs(p.EmbedPerLayer, nil, nil, p.PerLayerModelProjW, nil, nil, p.PerLayerProjNormW, id, emb, p.VocabPLI, nLayers, p.PliDim, dModel, 0, 0, 0, 0, eps, projView, scratch)
		if err != nil {
			rt.buffer = nil
			return nil, err
		}
		if scratch != nil {
			rt.buffer = scratch.out
		} else {
			rt.buffer = nil
		}
		return out, nil
	}
	rt.computeBuffer = func(id int32, emb []byte, embBuf metal.MTLBuffer) (int, metal.MTLBuffer, []byte, error) {
		if len(p.PerLayerModelProjW) == 0 {
			out, err := rt.compute(id, emb)
			return len(out), nil, out, err
		}
		scratch, err := ensureResident()
		if err != nil {
			rt.buffer = nil
			return 0, nil, nil, err
		}
		var buf metal.MTLBuffer
		var n int
		if embBuf != nil {
			buf, n, err = perLayerInputsResidentMetalBuffer(p.EmbedPerLayer, nil, nil, p.PerLayerModelProjW, p.PerLayerProjNormW, id, embBuf, p.VocabPLI, nLayers, p.PliDim, dModel, 0, 0, eps, projView, scratch)
		} else {
			buf, n, err = perLayerInputsResidentBuffer(p.EmbedPerLayer, nil, nil, p.PerLayerModelProjW, p.PerLayerProjNormW, id, emb, p.VocabPLI, nLayers, p.PliDim, dModel, 0, 0, eps, projView, scratch)
		}
		if err != nil {
			rt.buffer = nil
			return 0, nil, nil, err
		}
		rt.buffer = buf
		return n, buf, nil, nil
	}
	return rt, p.PliDim, nil
}

func archPLEQuantRuntime(fn string, p *ArchPLEQuant, nLayers, T, dModel int, eps float32) (*archDecodePLEInputs, int, error) {
	if p == nil {
		return nil, 0, nil
	}
	if len(p.TokenIDs) != T {
		return nil, 0, core.NewError(fn + ": PLE token id count must equal inputs")
	}
	if p.VocabPLI <= 0 || p.PliDim <= 0 || p.GroupSize <= 0 || p.Bits <= 0 {
		return nil, 0, core.NewError(fn + ": PLE quant geometry must be set")
	}
	if len(p.PerLayerProjNormW) != p.PliDim*bf16Size {
		return nil, 0, core.NewError(fn + ": PLE projection norm must be pliDim bf16 bytes")
	}
	rt := &archDecodePLEInputs{tokenIDs: p.TokenIDs}
	var projView bufView
	plDim := nLayers * p.PliDim
	projScale := float32(1.0 / math.Sqrt(float64(dModel)))
	ensureScratch := func() (*plHostScratch, error) {
		return rt.ensureScratch(plDim, dModel, projScale)
	}
	ensureResident := func() (*plHostScratch, error) {
		if projView.buf == nil {
			projView = bf16WeightView(p.PerLayerModelProjW, bufView{})
		}
		return ensureScratch()
	}
	rt.compute = func(id int32, emb []byte) ([]byte, error) {
		var scratch *plHostScratch
		if len(p.PerLayerModelProjW) > 0 {
			var err error
			if len(p.PerLayerModelProjScales) == 0 {
				scratch, err = ensureResident()
			} else {
				scratch, err = ensureScratch()
			}
			if err != nil {
				return nil, err
			}
		}
		out, err := PerLayerInputs(p.EmbedPerLayer, p.EmbedPerLayerScales, p.EmbedPerLayerBiases, p.PerLayerModelProjW, p.PerLayerModelProjScales, p.PerLayerModelProjBiases, p.PerLayerProjNormW, id, emb, p.VocabPLI, nLayers, p.PliDim, dModel, p.GroupSize, p.Bits, p.ProjGroupSize, p.ProjBits, eps, projView, scratch)
		if err != nil {
			rt.buffer = nil
			return nil, err
		}
		if scratch != nil {
			rt.buffer = scratch.out
		} else {
			rt.buffer = nil
		}
		return out, nil
	}
	rt.computeBuffer = func(id int32, emb []byte, embBuf metal.MTLBuffer) (int, metal.MTLBuffer, []byte, error) {
		if len(p.PerLayerModelProjW) == 0 {
			out, err := rt.compute(id, emb)
			return len(out), nil, out, err
		}
		var scratch *plHostScratch
		var err error
		if len(p.PerLayerModelProjScales) == 0 {
			scratch, err = ensureResident()
		} else {
			scratch, err = ensureScratch()
		}
		if err != nil {
			rt.buffer = nil
			return 0, nil, nil, err
		}
		var buf metal.MTLBuffer
		var n int
		if len(p.PerLayerModelProjScales) != 0 {
			proj := QuantWeight{Packed: p.PerLayerModelProjW, Scales: p.PerLayerModelProjScales, Biases: p.PerLayerModelProjBiases}
			if embBuf != nil {
				buf, n, err = perLayerInputsQuantResidentMetalBuffer(p.EmbedPerLayer, p.EmbedPerLayerScales, p.EmbedPerLayerBiases, proj, p.PerLayerProjNormW, id, embBuf, p.VocabPLI, nLayers, p.PliDim, dModel, p.GroupSize, p.Bits, p.ProjGroupSize, p.ProjBits, eps, scratch)
			} else {
				buf, n, err = perLayerInputsQuantResidentBuffer(p.EmbedPerLayer, p.EmbedPerLayerScales, p.EmbedPerLayerBiases, proj, p.PerLayerProjNormW, id, emb, p.VocabPLI, nLayers, p.PliDim, dModel, p.GroupSize, p.Bits, p.ProjGroupSize, p.ProjBits, eps, scratch)
			}
		} else if embBuf != nil {
			buf, n, err = perLayerInputsResidentMetalBuffer(p.EmbedPerLayer, p.EmbedPerLayerScales, p.EmbedPerLayerBiases, p.PerLayerModelProjW, p.PerLayerProjNormW, id, embBuf, p.VocabPLI, nLayers, p.PliDim, dModel, p.GroupSize, p.Bits, eps, projView, scratch)
		} else {
			buf, n, err = perLayerInputsResidentBuffer(p.EmbedPerLayer, p.EmbedPerLayerScales, p.EmbedPerLayerBiases, p.PerLayerModelProjW, p.PerLayerProjNormW, id, emb, p.VocabPLI, nLayers, p.PliDim, dModel, p.GroupSize, p.Bits, eps, projView, scratch)
		}
		if err != nil {
			rt.buffer = nil
			return 0, nil, nil, err
		}
		rt.buffer = buf
		return n, buf, nil, nil
	}
	return rt, p.PliDim, nil
}

func quantWeightBytesOK(w QuantWeight, outDim, inDim, groupSize, bits int) bool {
	return inDim%groupSize == 0 &&
		len(w.Packed) == outDim*inDim*bits/8 &&
		len(w.Scales) == outDim*(inDim/groupSize)*bf16Size &&
		len(w.Biases) == outDim*(inDim/groupSize)*bf16Size
}

func bf16PLELayers(fn string, layers []DecodeLayerWeights, dModel, pliDim int) ([]pleLayer, error) {
	ple := make([]pleLayer, len(layers))
	for li := range layers {
		w := layers[li]
		if len(w.PerLayerGate) != pliDim*dModel*bf16Size ||
			len(w.PerLayerProjection) != dModel*pliDim*bf16Size ||
			len(w.PostPerLayerInputNormW) != dModel*bf16Size {
			return nil, core.NewError(core.Sprintf("%s: PLE bf16 layer %d weight size mismatch", fn, li))
		}
		ple[li] = pleLayer{
			gate:     QuantWeight{Packed: w.PerLayerGate},
			proj:     QuantWeight{Packed: w.PerLayerProjection},
			postNorm: w.PostPerLayerInputNormW,
		}
	}
	return ple, nil
}

func quantPLELayers(fn string, qlayers []QuantizedLayerWeights, dModel, pliDim, groupSize, bits int) ([]pleLayer, error) {
	ple := make([]pleLayer, len(qlayers))
	for li := range qlayers {
		w := qlayers[li]
		// dense-or-quant per weight: a sidecar-less PLE gate/projection is a dense bf16 matrix
		// (the bf16 arch ICB recorder wraps bf16 weights as sidecar-less QuantWeights), so it
		// validates by byte size alone — no affine geometry to require.
		if !quantWeightProjectionShapeOK(w.PerLayerGate, pliDim, dModel, groupSize, bits) ||
			!quantWeightProjectionShapeOK(w.PerLayerProjection, dModel, pliDim, groupSize, bits) ||
			len(w.PostPerLayerInputNormW) != dModel*bf16Size {
			return nil, core.NewError(core.Sprintf("%s: PLE quant layer %d weight size mismatch", fn, li))
		}
		ple[li] = pleLayer{
			gate: w.PerLayerGate, proj: w.PerLayerProjection,
			postNorm: w.PostPerLayerInputNormW, groupSize: groupSize, bits: bits,
		}
	}
	return ple, nil
}

// newArchDecodeState builds the shared scratch + position buffer over the caller's
// per-layer buffers. MUST be called inside a withAutoreleasePool.
func newArchDecodeState(specs []model.LayerSpec, lb []archLayerBufs, moeWeights []*MoELayerWeights, dModel, nHeads, nKVHeads, headDim, dFF, slidingWindow, rotaryDim, rotaryDimLocal int, base, localBase, scale, eps float32, valueNorm bool, maxLen int) archDecodeState {
	// scratch must fit the LARGEST layer's q/kv (gemma4 full_attention layers use a
	// bigger head_dim than sliding) — the shared scratch is reused across all layers.
	maxQDim, maxKvDim, maxHeadDim := nHeads*headDim, nKVHeads*headDim, headDim
	for _, sp := range specs {
		lhd, lkv := headDimOf(sp, headDim), kvHeadsOf(sp, nKVHeads)
		if q := nHeads * lhd; q > maxQDim {
			maxQDim = q
		}
		if kv := lkv * lhd; kv > maxKvDim {
			maxKvDim = kv
		}
		if lhd > maxHeadDim {
			maxHeadDim = lhd
		}
	}
	// per-layer FFN width (gemma4 E2B/E4B MatFormer): the shared MLP scratch must fit the WIDEST layer.
	maxDFF := dFF
	for i := range lb {
		if lb[i].dFF > maxDFF {
			maxDFF = lb[i].dFF
		}
	}
	// gemma4 value-norm weight: ones of the largest head_dim, shared across heads + layers
	// (the per-head value RMSNorm reads axisSize=headDim of it). nil ⇒ no value-norm.
	var valueNormOnes metal.MTLBuffer
	if valueNorm {
		valueNormOnes = bf16ConstBuffer(maxHeadDim, 1.0)
	}
	// gemma4 global proportional+partial rope spectrum (see proportionalRopePeriods): built once
	// for GlobalAttention layers so their rope pairs over the FULL head dim. Sliding (full rotary)
	// keeps the base-derived path.
	var globalRopeFreqs metal.MTLBuffer
	globalHeadDim := 0
	for _, sp := range specs {
		if sp.Attention == model.GlobalAttention {
			globalHeadDim = headDimOf(sp, headDim)
			break
		}
	}
	if globalHeadDim > 0 && rotaryDim > 0 && rotaryDim < globalHeadDim {
		periods := globalRopePeriodsFromFolded(globalHeadDim, rotaryDim, base)
		globalRopeFreqs = cachedRawRopePeriodsBuffer(periods)
	}
	coreScratch := getArchDecodeCoreScratch(dModel, maxQDim, maxKvDim, nHeads, maxLen, maxDFF)
	return archDecodeState{
		specs: specs, lb: lb, moeWeights: moeWeights,
		globalRopeFreqs: globalRopeFreqs, globalHeadDim: globalHeadDim,
		asc: coreScratch.asc, msc: coreScratch.msc,
		coreScratch:    coreScratch,
		hBuf:           coreScratch.hBuf,
		xA:             coreScratch.xA,
		xB:             coreScratch.xB,
		offBuf:         coreScratch.offBuf,
		offPtr:         coreScratch.offPtr,
		hBufPtr:        coreScratch.hBufPtr,
		xAPtr:          coreScratch.xAPtr,
		xBPtr:          coreScratch.xBPtr,
		valueNormOnes:  valueNormOnes,
		dModel:         dModel,
		nHeads:         nHeads,
		nKVHeads:       nKVHeads,
		headDim:        headDim,
		dFF:            dFF,
		slidingWindow:  slidingWindow,
		maxLen:         maxLen,
		rotaryDim:      rotaryDim,
		rotaryDimLocal: rotaryDimLocal,
		base:           base, localBase: localBase, scale: scale, eps: eps,
		trace: nativeTraceEnabled(),
	}
}

// bufMaxAbsNaN reads a dModel-length bf16 buffer back to host and returns the largest finite
// absolute value plus the count of NaN/Inf-scale elements — the per-layer trace signal. A
// blow-up or NaN, and the token/layer it first appears at, localise where a decode degrades.
// Debug-path only (the readback forces a commit+wait).
func bufMaxAbsNaN(buf metal.MTLBuffer, dModel int) (maxAbs float32, bad int) {
	b := unsafe.Slice((*byte)(buf.Contents()), dModel*bf16Size)
	for i := range dModel {
		v := bf16ToF32(b[i*bf16Size], b[i*bf16Size+1])
		if v != v || v > 3.0e38 || v < -3.0e38 { // NaN or Inf-scale
			bad++
			continue
		}
		if v < 0 {
			v = -v
		}
		if v > maxAbs {
			maxAbs = v
		}
	}
	return maxAbs, bad
}

// captureLayerHiddens, when set by the cross-engine test, makes stepToken append each
// layer's output hidden (dModel bf16 bytes) to capturedLayerHiddens — the native half of
// the per-layer cross-engine diff. Reset capturedLayerHiddens to nil before the step.
var (
	captureLayerHiddens  bool
	capturedLayerHiddens [][]byte
	capturedAttnHiddens  [][]byte // post-attention hidden (x + Wo·attn) per layer — isolates attention from MLP
)

// stepToken decodes ONE token (its embedding) at sequence position pos, writing this
// token's K/V into the growing cache, and returns its output hidden state. The projector
// seam keeps it weight-representation-agnostic (bf16 / 4-bit qmv); it honours owner/sharer
// KV-sharing, sliding-window, the gemma4 norms, and MoE (the mid-token command-buffer flush
// because the router does host top-k). The caches persist across calls, so successive
// positions extend the same sequence. MUST be called inside a withAutoreleasePool.
func (s *archDecodeState) stepToken(inputEmb []byte, pos int) ([]byte, error) {
	return s.stepTokenResultWithInput(inputEmb, pos, true, true)
}

func (s *archDecodeState) stepTokenInto(inputEmb []byte, pos int, dst []byte) ([]byte, error) {
	return s.stepTokenResultWithInputInto(inputEmb, pos, true, true, dst)
}

func (s *archDecodeState) stepTokenNoResult(inputEmb []byte, pos int) error {
	_, err := s.stepTokenResultWithInput(inputEmb, pos, false, true)
	return err
}

func (s *archDecodeState) stepTokenResult(inputEmb []byte, pos int, readResult bool) ([]byte, error) {
	return s.stepTokenResultWithInput(inputEmb, pos, readResult, true)
}

func (s *archDecodeState) stepTokenLoaded(inputEmb []byte, pos int) ([]byte, error) {
	return s.stepTokenResultWithInput(inputEmb, pos, true, false)
}

func (s *archDecodeState) stepTokenResultWithInput(inputEmb []byte, pos int, readResult, copyInput bool) ([]byte, error) {
	return s.stepTokenResultWithInputInto(inputEmb, pos, readResult, copyInput, nil)
}

func (s *archDecodeState) stepTokenResultWithInputInto(inputEmb []byte, pos int, readResult, copyInput bool, dst []byte) ([]byte, error) {
	*s.offPtr = int32(pos)
	inputBuf := s.xA
	if copyInput {
		if buf, ok := s.stableInputEmbBuffer(inputEmb, s.dModel); ok {
			inputBuf = buf
		} else {
			copy(s.bufferBytes(s.xA, s.dModel*bf16Size), inputEmb)
		}
	}
	var pleInputBuf metal.MTLBuffer
	if len(s.ple) > 0 {
		want := len(s.specs) * s.pliDim * bf16Size
		got := len(s.perLayerInput)
		if s.perLayerInputBuf != nil {
			got = s.perLayerInputLen
		}
		if got != want {
			return nil, core.NewError("native.archDecodeState.stepToken: PLE tensor size mismatch")
		}
		if s.perLayerInputBuf != nil {
			pleInputBuf = s.perLayerInputBuf
		} else {
			var err error
			pleInputBuf, err = s.hostPLEInputBuffer(want)
			if err != nil {
				return nil, err
			}
		}
	}
	cb := commandBufferFast(queue)
	var enc metal.MTLComputeCommandEncoderObject
	if s.gpuProf != nil {
		enc = s.gpuProf.encoderFor(cb, "attn")
	} else {
		enc = computeCommandEncoderFast(cb)
	}
	s.resetDevicePagedAttentionScratch()
	in, out := inputBuf, s.xB
	if inputBuf != s.xA {
		out = s.xA
	}
	var trWorstAbs float32
	trWorstLayer, trFirstBad, trBadLayers := -1, -1, 0
	cbBroken := false // a mid-token cb swap (MoE break-out / test probes) invalidates chainTail
	// encConc: enc is an OPEN CONCURRENT encoder carried between passes (#341
	// phase 1.5) — the attn pass, the MoE block and the per-layer scalar ride ONE
	// encoder per layer stack with barriers at the true edges instead of paying
	// ~4 encoder seams per layer. Serial-only stages normalise it back first.
	encConc := false
	toSerial := func() {
		if !encConc {
			return
		}
		endEncodingFast(enc)
		enc = computeCommandEncoderFast(cb)
		encConc = false
	}
	for li := 0; li < len(s.specs); li++ {
		if li > 0 {
			enc = s.profSeam(cb, enc, "attn")
		}
		// per-attention-type head geometry (gemma4 full layers use the larger global head_dim);
		// the SDPA scale stays s.scale — the model DECLARED it (gemma4 1.0, not 1/√headDim).
		lhd, lkv := headDimOf(s.specs[li], s.headDim), kvHeadsOf(s.specs[li], s.nKVHeads)
		// sliding layers window the SDPA AND use the local RoPE theta + rotary dim; global use the
		// global. gemma4 global rope is proportional + PARTIAL: drive the freqs path over the FULL
		// head (rotDim=lhd) with the Inf-padded spectrum so it pairs (d, d+headDim/2) — the base
		// path's (d, d+rotaryDim/2) pairing is wrong for partial rotary (see globalRopeFreqs).
		slideW, rbase, rotDim := 0, s.base, s.rotaryDim
		layerRopeFreqs := s.ropeFreqs
		if s.specs[li].Attention == model.SlidingAttention {
			slideW, rbase, rotDim = s.slidingWindow, s.localBase, s.rotaryDimLocal
		} else if s.globalRopeFreqs != nil {
			layerRopeFreqs, rotDim = s.globalRopeFreqs, lhd
		}
		if s.specs[li].OwnsCache() {
			if cache := s.layerPagedKV(li); cache != nil {
				var aerr error
				if enc, encConc, aerr = encAttnHalfKVPaged(enc, cb, s.gpuProf, encConc, in, cache, s.offBuf, s.hBuf, 0, s.lb[li].anw, s.lb[li].postAttnNorm, s.lb[li].qNorm, s.lb[li].kNorm, s.valueNormOnes, s.asc, s.lb[li].proj, s.dModel, s.nHeads, lkv, lhd, pos, slideW, rotDim, rbase, s.scale, s.eps, layerRopeFreqs); aerr != nil {
					endEncodingFast(enc)
					return nil, aerr
				}
			} else {
				toSerial() // the plain KV attention half is a serial emitter
				if err := encAttnHalfKV(enc, in, s.lb[li].kCache, s.lb[li].vCache, s.offBuf, s.hBuf, s.lb[li].anw, s.lb[li].postAttnNorm, s.lb[li].qNorm, s.lb[li].kNorm, s.valueNormOnes, s.asc, s.lb[li].proj, s.dModel, s.nHeads, lkv, lhd, pos, slideW, rotDim, rbase, s.scale, s.eps, layerRopeFreqs); err != nil {
					endEncodingFast(enc)
					return nil, err
				}
			}
		} else {
			toSerial() // the shared-KV attention halves are serial emitters
			own := s.specs[li].KVShareFrom
			if cache := s.layerPagedKV(own); cache != nil {
				if err := encAttnHalfSharedPaged(enc, in, cache, s.offBuf, s.hBuf, 0, s.lb[li].anw, s.lb[li].postAttnNorm, s.lb[li].qNorm, s.asc, s.lb[li].proj, s.dModel, s.nHeads, lkv, lhd, pos, slideW, rotDim, rbase, s.scale, s.eps, layerRopeFreqs); err != nil {
					endEncodingFast(enc)
					return nil, err
				}
			} else if err := encAttnHalfShared(enc, in, s.lb[own].kCache, s.lb[own].vCache, s.offBuf, s.hBuf, s.lb[li].anw, s.lb[li].postAttnNorm, s.lb[li].qNorm, s.asc, s.lb[li].proj, s.dModel, s.nHeads, lkv, lhd, pos, slideW, rotDim, rbase, s.scale, s.eps, layerRopeFreqs); err != nil {
				endEncodingFast(enc)
				return nil, err
			}
		}
		if captureLayerHiddens { // post-attention hidden (x + Wo·attn) — isolates attention from MLP
			endEncodingFast(enc)
			encConc = false
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			capturedAttnHiddens = append(capturedAttnHiddens, append([]byte(nil), s.bufferBytes(s.hBuf, s.dModel*bf16Size)...))
			cb = commandBufferFast(queue)
			enc = computeCommandEncoderFast(cb)
			cbBroken = true
		}
		var moeQ *MoEQuantLayerWeights
		if li < len(s.moeQuant) {
			moeQ = s.moeQuant[li]
		}
		if moeW := s.moeWeights[li]; moeQ != nil || moeW != nil {
			enc = s.profSeam(cb, enc, "moe.router")
			// Fully-encoded lane first (session-owned scratches, device router + gathered
			// experts): the WHOLE block encodes into the LIVE encoder — no command-buffer
			// break at all. Declines fall to the break-out flow below.
			handledMoE := false
			if moeQ != nil && s.moeScratchOwnable && quantMoEDeviceRouterBuffersUsable(*moeQ, s.dModel) && routerTopKUsable(moeQ.NumExperts, moeQ.TopK) {
				var err error
				if s.moeOwnedScratch == nil {
					s.moeOwnedScratch, err = getMoEBlockBF16Scratch(s.dModel, s.dFF, moeQ.ExpertDFF, moeQ.TopK)
					if err != nil {
						return nil, err
					}
				}
				if s.moeRouterOwnedScratch == nil {
					s.moeRouterOwnedScratch, err = getRouterDeviceScratch(s.dModel, moeQ.NumExperts, moeQ.TopK)
					if err != nil {
						return nil, err
					}
				}
				enc, encConc, handledMoE, err = encMoEBlockQuantDevice(enc, cb, s.gpuProf, encConc, s.moeRouterOwnedScratch, s.moeOwnedScratch, s.hBuf, out, *moeQ, s.dModel, s.dFF, s.eps)
				if err != nil {
					endEncodingFast(enc)
					return nil, err
				}
			}
			if !handledMoE {
				// The MoE stages run their own command buffers over h's SHARED buffer. The quant
				// happy path (device router + device index + buffer output) never reads the host
				// bytes, so the live buffer commits WITHOUT a wait — the queue orders the MoE
				// stages after it, and the expert stage's own completion wait (which every scratch
				// lifetime already assumes) remains the layer's single sync. The old shape waited
				// AND host-read h here: two full GPU round-trips per MoE layer × ~48 layers was
				// the 26B decode running at a sixth of its cgo rate.
				endEncodingFast(enc)
				encConc = false
				commitCommandBufferFast(cb)
				var err error
				if moeQ != nil && quantMoEDeviceRouterBuffersUsable(*moeQ, s.dModel) && routerTopKUsable(moeQ.NumExperts, moeQ.TopK) {
					if s.moeScratchOwnable && s.moeOwnedScratch == nil {
						s.moeOwnedScratch, err = getMoEBlockBF16Scratch(s.dModel, s.dFF, moeQ.ExpertDFF, moeQ.TopK)
						if err != nil {
							return nil, err
						}
					}
					err = moeBlockQuantWithBufferOutputInPool(nil, s.hBuf, out, *moeQ, s.dModel, s.dFF, s.eps, s.moeOwnedScratch)
				} else if moeQ != nil {
					// device router unavailable (older metallib / exotic geometry): the host
					// router needs the bytes, so this lane keeps the completion wait.
					waitUntilCompletedFast(cb)
					err = moeBlockQuantWithBufferOutputInPool(s.bufferBytes(s.hBuf, s.dModel*bf16Size), s.hBuf, out, *moeQ, s.dModel, s.dFF, s.eps, nil)
				} else {
					// bf16 MoE keeps the host handoff (its block still reads host bytes).
					waitUntilCompletedFast(cb)
					err = moeBlockBF16WithBufferOutputInPool(s.bufferBytes(s.hBuf, s.dModel*bf16Size), s.hBuf, out, *moeW, s.dModel, s.dFF, s.eps)
				}
				if err != nil {
					return nil, err
				}
				cb = commandBufferFast(queue)
				enc = computeCommandEncoderFast(cb)
				cbBroken = true
			}
		} else {
			toSerial()   // the dense MLP half is a serial emitter
			lff := s.dFF // per-layer FFN width (gemma4 E2B/E4B); falls back to the arch default
			if s.lb[li].dFF > 0 {
				lff = s.lb[li].dFF
			}
			if err := encMLPHalfBF16(enc, s.hBuf, out, s.lb[li].mnw, s.lb[li].postFFNorm, s.msc, s.lb[li].proj, s.dModel, lff, s.eps); err != nil {
				endEncodingFast(enc)
				return nil, err
			}
		}
		// gemma4 per-layer-input gate (E2B/E4B): keep the gate chain in the live command buffer.
		// The per-token PLE tensor is pinned once at step entry, and each layer binds its pliDim row
		// by byte offset. Applied to the layer output before the per-layer scalar.
		if len(s.ple) > li && len(s.ple[li].postNorm) > 0 {
			toSerial() // the PLE gate helpers are multi-dispatch serial emitters
			pl := s.ple[li]
			if len(pl.postNorm) != s.dModel*bf16Size {
				endEncodingFast(enc)
				return nil, core.NewError("native.archDecodeState.stepToken: PLE post norm size mismatch")
			}
			pliOff := uint(li * s.pliDim * bf16Size)
			sc := s.perLayerInputGateScratch()
			if pl.bits == 0 { // bf16 PLE gate (the quant path sets bits 4/8 ⇒ the qmv)
				if len(pl.gate.Packed) != s.pliDim*s.dModel*bf16Size || len(pl.proj.Packed) != s.dModel*s.pliDim*bf16Size {
					endEncodingFast(enc)
					return nil, core.NewError("native.archDecodeState.stepToken: PLE bf16 weight size mismatch")
				}
				if err := encPerLayerInputGateBF16Scratch(enc, sc, out, residentBytes(pl.gate.Packed), pleInputBuf, residentBytes(pl.proj.Packed), residentBytes(pl.postNorm), out, pliOff, s.dModel, s.pliDim, s.eps); err != nil {
					endEncodingFast(enc)
					return nil, err
				}
			} else {
				gateGroupSize, gateBits, err := validatePerLayerInputGateQuantWeight("gate", pl.gate, s.pliDim, s.dModel, pl.groupSize, pl.bits)
				if err != nil {
					endEncodingFast(enc)
					return nil, err
				}
				projGroupSize, projBits, err := validatePerLayerInputGateQuantWeight("projection", pl.proj, s.dModel, s.pliDim, pl.groupSize, pl.bits)
				if err != nil {
					endEncodingFast(enc)
					return nil, err
				}
				gatePacked, gateScales, gateBiases := quantWeightViews(pl.gate)
				projPacked, projScales, projBiases := quantWeightViews(pl.proj)
				if err := encPerLayerInputGateQuantScratch(enc, sc, out, gatePacked, gateScales, gateBiases, pleInputBuf, projPacked, projScales, projBiases, residentBytes(pl.postNorm), out, pliOff, s.dModel, s.pliDim, gateGroupSize, gateBits, projGroupSize, projBits, s.eps); err != nil {
					endEncodingFast(enc)
					return nil, err
				}
			}
		}
		// gemma4 per-layer output scalar: multiply the layer's hidden before the next layer.
		if s.lb[li].layerScalar != nil {
			if encConc {
				// single kernel — rides the carried concurrent encoder behind a barrier
				// (the next pass's entry barrier orders ITS writes in turn).
				memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
			}
			if err := encMulBF16(enc, out, s.lb[li].layerScalar, out, s.dModel); err != nil {
				endEncodingFast(enc)
				return nil, err
			}
		}
		if layerSpanProbeForTest != nil { // probe-only per-layer GPU spans (test hook, nil in production)
			endEncodingFast(enc)
			encConc = false
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			layerSpanProbeForTest[li] += int64(float64(cb.GPUEndTime()-cb.GPUStartTime()) * 1e9)
			cb = commandBufferFast(queue)
			enc = computeCommandEncoderFast(cb)
			cbBroken = true
		}
		if s.trace { // per-layer diagnostic: flush, read this layer's output hidden, accumulate
			endEncodingFast(enc)
			encConc = false
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			ma, bad := bufMaxAbsNaN(out, s.dModel)
			if bad > 0 {
				trBadLayers++
				if trFirstBad < 0 {
					trFirstBad = li
				}
			}
			if ma > trWorstAbs {
				trWorstAbs, trWorstLayer = ma, li
			}
			cb = commandBufferFast(queue)
			enc = computeCommandEncoderFast(cb)
			cbBroken = true
		}
		if captureLayerHiddens { // cross-engine per-layer diff: store this layer's output hidden
			endEncodingFast(enc)
			encConc = false
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			capturedLayerHiddens = append(capturedLayerHiddens, append([]byte(nil), s.bufferBytes(out, s.dModel*bf16Size)...))
			cb = commandBufferFast(queue)
			enc = computeCommandEncoderFast(cb)
			cbBroken = true
		}
		if in == inputBuf && inputBuf != s.xA {
			in, out = out, s.xB
		} else {
			in, out = out, in
		}
	}
	toSerial() // the chain tail and everything after expect the tracked serial encoder
	if s.chainTail != nil && !cbBroken {
		// The tail encodes the head argmax + the NEXT token's input production into THIS
		// command buffer; the commit + wait below stay the token's single sync. When the
		// tail writes the next input into a buffer the hidden currently occupies, encoder
		// order keeps the GPU side correct — but the HOST readResult copy below reads the
		// post-overwrite bytes, so a chained caller must ignore res.
		if terr := s.chainTail(enc, in); terr != nil {
			endEncodingFast(enc)
			return nil, terr
		}
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	if s.chainSkipWait && s.chainTail != nil {
		// submit-ahead: the caller waits on this cb (and reads the head scratch) later.
		s.chainPendingCB = cb
		return nil, nil
	}
	waitUntilCompletedFast(cb)
	if pieceTimingOn { // diagnostic: the step CB's true GPU execution span vs its wall
		chainedGPUSpanNs += int64(float64(cb.GPUEndTime()-cb.GPUStartTime()) * 1e9)
	}
	var res []byte
	if readResult {
		n := s.dModel * bf16Size
		if dst != nil {
			if len(dst) != n {
				return nil, core.NewError("native.archDecodeState.stepToken: destination must be hidden bf16 bytes")
			}
			res = dst
		} else {
			res = make([]byte, n)
		}
		copy(res, s.bufferBytes(in, s.dModel*bf16Size))
	}
	if s.trace {
		wt := "-"
		if trWorstLayer >= 0 {
			wt = "sliding"
			if s.specs[trWorstLayer].Attention == model.GlobalAttention {
				wt = "GLOBAL"
			}
		}
		fm, fb := bufMaxAbsNaN(in, s.dModel)
		var ieAbs float32 // input-embedding magnitude — flags a bad token-embed (e.g. a control token's 4-bit dequant)
		for i := 0; i+1 < len(inputEmb); i += 2 {
			if v := bf16ToF32(inputEmb[i], inputEmb[i+1]); v > ieAbs {
				ieAbs = v
			} else if -v > ieAbs {
				ieAbs = -v
			}
		}
		nativeTraceLog(core.Sprintf("native-trace tok=%d inEmbAbs=%.4g worstAbs=%.4g@L%d(%s) badLayers=%d firstBad=L%d finalAbs=%.4g finalBad=%d\n",
			pos, ieAbs, trWorstAbs, trWorstLayer, wt, trBadLayers, trFirstBad, fm, fb))
	}
	return res, nil
}

// runArchDecode is the whole-sequence arch decode: it builds a state and steps each input
// token at its position over a fresh growing cache. See archDecodeState/stepToken — the
// bf16 (DecodeForwardArch) and 4-bit qmv (DecodeForwardArchQuant) forwards share this. MUST
// be called inside a withAutoreleasePool.
func runArchDecode(
	inputs [][]byte, specs []model.LayerSpec, lb []archLayerBufs, moeWeights []*MoELayerWeights,
	dModel, nHeads, nKVHeads, headDim, dFF, slidingWindow, rotaryDim, rotaryDimLocal int, base, localBase, scale, eps float32, valueNorm bool, maxLen int,
) ([][]byte, error) {
	s := newArchDecodeState(specs, lb, moeWeights, dModel, nHeads, nKVHeads, headDim, dFF, slidingWindow, rotaryDim, rotaryDimLocal, base, localBase, scale, eps, valueNorm, maxLen)
	defer s.Close()
	return runArchDecodeState(inputs, &s, nil)
}

func runArchDecodeState(inputs [][]byte, s *archDecodeState, ple *archDecodePLEInputs) ([][]byte, error) {
	return runArchDecodeStateInto(nil, inputs, s, ple, false)
}

func runArchDecodeStateInto(outputs [][]byte, inputs [][]byte, s *archDecodeState, ple *archDecodePLEInputs, useCallerOut bool) ([][]byte, error) {
	if ple != nil {
		defer ple.Close()
	}
	outLen := s.dModel * bf16Size
	if cap(outputs) < len(inputs) {
		outputs = make([][]byte, len(inputs))
	} else {
		outputs = outputs[:len(inputs)]
	}
	for t := range outputs {
		if useCallerOut && cap(outputs[t]) >= outLen {
			outputs[t] = outputs[t][:outLen]
			continue
		}
		outputs[t] = make([]byte, outLen)
	}
	for t := range inputs {
		inputLoaded := false
		if ple != nil {
			want := len(s.specs) * s.pliDim * bf16Size
			if ple.computeBuffer != nil {
				copy(s.bufferBytes(s.xA, s.dModel*bf16Size), inputs[t])
				inputLoaded = true
				n, buf, host, err := ple.computeBuffer(ple.tokenIDs[t], inputs[t], s.xA)
				if err != nil {
					return nil, err
				}
				if n != want {
					return nil, core.NewError("native.runArchDecodeState: PLE tensor size mismatch")
				}
				if buf == nil && len(host) != want {
					return nil, core.NewError("native.runArchDecodeState: PLE tensor size mismatch")
				}
				s.perLayerInput = host
				s.perLayerInputBuf = buf
				s.perLayerInputLen = n
			} else {
				pli, err := ple.compute(ple.tokenIDs[t], inputs[t])
				if err != nil {
					return nil, err
				}
				if len(pli) != want {
					return nil, core.NewError("native.runArchDecodeState: PLE tensor size mismatch")
				}
				s.perLayerInput = pli
				s.perLayerInputBuf = ple.buffer
				s.perLayerInputLen = len(pli)
			}
			if s.perLayerInputBuf == nil && len(s.perLayerInput) != want {
				return nil, core.NewError("native.runArchDecodeState: PLE tensor size mismatch")
			}
		}
		var out []byte
		var err error
		if inputLoaded {
			out, err = s.stepTokenResultWithInputInto(inputs[t], t, true, false, outputs[t])
		} else {
			out, err = s.stepTokenInto(inputs[t], t, outputs[t])
		}
		if err != nil {
			return nil, err
		}
		outputs[t] = out
	}
	return outputs, nil
}

// DecodeForwardArch is the bf16 arch-driven decode forward: it runs a decode DRIVEN by
// a declared gemma4 arch (specs, one LayerSpec per layer) rather than treating every
// layer uniformly. It honours the full cache-topology (owner/sharer KV), the per-layer
// attention type (sliding window), and MoE layers (the dual-branch MoEBlockBF16). With
// an all-owner, all-global, dense arch it equals DecodeForward byte-for-byte (gated).
// bf16 re-encode path (one commit+wait/token; MoE layers flush mid-token). The 4-bit
// variant DecodeForwardArchQuant shares the loop (runArchDecode) via the projector seam.
func DecodeForwardArch(
	inputs [][]byte, layers []DecodeLayerWeights, specs []model.LayerSpec,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow int,
	base, scale, eps float32, valueNorm bool,
	pleArgs ...ArchPLEBF16,
) ([][]byte, error) {
	return decodeForwardArchInto(nil, inputs, layers, specs, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow, base, scale, eps, valueNorm, false, pleArgs...)
}

// DecodeForwardArchInto is DecodeForwardArch with caller-owned per-token output
// storage. Output slices with enough capacity are reused for the final hidden
// readback from each token.
func DecodeForwardArchInto(
	outputs [][]byte, inputs [][]byte, layers []DecodeLayerWeights, specs []model.LayerSpec,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow int,
	base, scale, eps float32, valueNorm bool,
	pleArgs ...ArchPLEBF16,
) ([][]byte, error) {
	return decodeForwardArchInto(outputs, inputs, layers, specs, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow, base, scale, eps, valueNorm, true, pleArgs...)
}

func decodeForwardArchInto(
	outputs [][]byte, inputs [][]byte, layers []DecodeLayerWeights, specs []model.LayerSpec,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow int,
	base, scale, eps float32, valueNorm bool,
	useCallerOut bool,
	pleArgs ...ArchPLEBF16,
) ([][]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	nLayers, T := len(layers), len(inputs)
	if nLayers == 0 || T == 0 {
		return nil, core.NewError("native.DecodeForwardArch: need layers and inputs")
	}
	if len(specs) != nLayers {
		return nil, core.NewError("native.DecodeForwardArch: specs length must equal layers")
	}
	if T > maxLen {
		return nil, core.NewError("native.DecodeForwardArch: more tokens than maxLen cache rows")
	}
	for i := range inputs {
		if len(inputs[i]) != dModel*bf16Size {
			return nil, core.NewError("native.DecodeForwardArch: each input must be dModel bf16 bytes")
		}
	}
	for li := range specs {
		o := specs[li].KVShareFrom
		if o < 0 || o > li || (o != li && !specs[o].OwnsCache()) {
			return nil, core.NewError("native.DecodeForwardArch: KVShareFrom must reference an earlier owner layer")
		}
		if specs[li].MoE != (layers[li].MoE != nil) {
			return nil, core.NewError("native.DecodeForwardArch: spec.MoE must match the presence of layer MoE weights")
		}
	}
	plePayload, err := singleArchPLEBF16("native.DecodeForwardArch", pleArgs)
	if err != nil {
		return nil, err
	}
	pleRuntime, pliDim, err := archPLEBF16Runtime("native.DecodeForwardArch", plePayload, nLayers, T, dModel, eps)
	if err != nil {
		return nil, err
	}
	if pleRuntime != nil {
		defer pleRuntime.Close()
	}
	var pleLayers []pleLayer
	if pleRuntime != nil {
		pleLayers, err = bf16PLELayers("native.DecodeForwardArch", layers, dModel, pliDim)
		if err != nil {
			return nil, err
		}
	}

	setup := getArchBF16LayerBufScratch(nLayers)
	defer putArchBF16LayerBufScratch(setup)
	withAutoreleasePool(func() {
		lb, moeWeights, berr := buildBF16ArchLayerBufsIntoScratch(setup, layers, specs, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow, nil)
		if berr != nil {
			err = berr
			return
		}
		if pleRuntime != nil {
			state := newArchDecodeState(specs, lb, moeWeights, dModel, nHeads, nKVHeads, headDim, dFF, slidingWindow, headDim, headDim, base, base, scale, eps, valueNorm, maxLen)
			defer state.Close()
			state.ple, state.pliDim = pleLayers, pliDim
			outputs, err = runArchDecodeStateInto(outputs, inputs, &state, pleRuntime, useCallerOut)
			return
		}
		state := newArchDecodeState(specs, lb, moeWeights, dModel, nHeads, nKVHeads, headDim, dFF, slidingWindow, headDim, headDim, base, base, scale, eps, valueNorm, maxLen)
		defer state.Close()
		outputs, err = runArchDecodeStateInto(outputs, inputs, &state, nil, useCallerOut)
	})
	return outputs, err
}

// buildBF16ArchLayerBufs builds the per-layer resident buffers for a bf16 arch decode:
// bf16 norms + the bf16 projector + the growing KV caches (owner layers only), and the
// per-layer MoE weights (moeWeights[li] != nil ⟺ a MoE layer, whose dense MLP norm +
// gate/up/down stay unbound — MoEBlockBF16 owns that FFN). Shared by the whole-sequence
// forward and the incremental generation loop.
//
// sb is the zero-copy weight source: when non-nil, every weight is bound as a no-copy view into
// the shared shard mmap at its byte offset (no upload, no second resident copy); when nil (the
// in-memory weight bytes of DecodeForwardArch or a session built from a parsed blob), each weight
// is uploaded into a fresh owned buffer at offset 0 — byte-identical, just a heap+GPU copy. A
// non-nil sb errors if a weight is not a view into its mapping (a programming error). MUST be
// called inside a withAutoreleasePool.
func buildBF16ArchLayerBufs(layers []DecodeLayerWeights, specs []model.LayerSpec, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow int, sb *shardBuffers) ([]archLayerBufs, []*MoELayerWeights, error) {
	nLayers := len(layers)
	lb := make([]archLayerBufs, nLayers)
	moeWeights := make([]*MoELayerWeights, nLayers)
	return buildBF16ArchLayerBufsInto(lb, moeWeights, layers, specs, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow, sb)
}

func buildBF16ArchLayerBufsInto(lb []archLayerBufs, moeWeights []*MoELayerWeights, layers []DecodeLayerWeights, specs []model.LayerSpec, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow int, sb *shardBuffers) ([]archLayerBufs, []*MoELayerWeights, error) {
	return buildBF16ArchLayerBufsInternal(lb, moeWeights, nil, layers, specs, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow, sb)
}

func buildBF16ArchLayerBufsIntoScratch(setup *archBF16LayerBufScratch, layers []DecodeLayerWeights, specs []model.LayerSpec, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow int, sb *shardBuffers) ([]archLayerBufs, []*MoELayerWeights, error) {
	if setup == nil {
		return buildBF16ArchLayerBufs(layers, specs, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow, sb)
	}
	setup.reset(len(layers))
	return buildBF16ArchLayerBufsInternal(setup.lb, setup.moeWeights, setup, layers, specs, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow, sb)
}

func buildBF16ArchLayerBufsInternal(lb []archLayerBufs, moeWeights []*MoELayerWeights, setup *archBF16LayerBufScratch, layers []DecodeLayerWeights, specs []model.LayerSpec, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow int, sb *shardBuffers) ([]archLayerBufs, []*MoELayerWeights, error) {
	nLayers := len(layers)
	if cap(lb) < nLayers {
		lb = make([]archLayerBufs, nLayers)
	} else {
		lb = lb[:nLayers]
		clear(lb)
	}
	if cap(moeWeights) < nLayers {
		moeWeights = make([]*MoELayerWeights, nLayers)
	} else {
		moeWeights = moeWeights[:nLayers]
		clear(moeWeights)
	}
	var ferr error
	// view resolves a required weight: a no-copy shard view (sb != nil) or an uploaded copy.
	view := func(b []byte) bufView {
		if sb != nil {
			return sb.mustBufFor(b, &ferr)
		}
		return bufView{buf: residentBytes(b)}
	}
	// viewOrNil is view for an optional weight (absent ⇒ zero bufView, the "skip" sentinel).
	viewOrNil := func(b []byte) bufView {
		if len(b) == 0 {
			return bufView{}
		}
		return view(b)
	}
	for li := range layers {
		w := layers[li]
		// per-attention-type geometry: gemma4 full_attention layers use a larger head_dim
		// (global_head_dim), so the projection dims + KV-cache row size are per layer.
		lhd, lkv := headDimOf(specs[li], headDim), kvHeadsOf(specs[li], nKVHeads)
		qDim, kvDim := nHeads*lhd, lkv*lhd
		// sliding layers RING at slidingWindow rows (they only ever attend the last slidingWindow), so
		// they need slidingWindow rows of cache, not maxLen — the full-context KV memory fix. Global
		// (full_attention) layers attend everything, so they keep maxLen. min() keeps short contexts
		// (maxLen ≤ window) at maxLen (no benefit, no wrap). encAttnHalfKV does the matching ring write.
		cacheLen := maxLen
		if slidingWindow > 0 && slidingWindow < maxLen && specs[li].Attention != model.GlobalAttention {
			cacheLen = slidingWindow
		}
		cacheBytes := uint(cacheLen * kvDim * bf16Size)
		lb[li].anw = view(w.AttnNormW)
		lb[li].postAttnNorm = viewOrNil(w.PostAttnNormW)
		lb[li].postFFNorm = viewOrNil(w.PostFFNormW)
		lb[li].qNorm = viewOrNil(w.QNormW)
		lb[li].kNorm = viewOrNil(w.KNormW)
		lb[li].layerScalar = layerScalarBuf(w.LayerScalarW, dModel) // synthesised broadcast (not a shard view)
		if specs[li].OwnsCache() {
			if setup != nil {
				lb[li].kCache, lb[li].vCache, lb[li].kCachePtr, lb[li].vCachePtr = setup.kvCache(li, cacheBytes)
			} else {
				lb[li].kCache = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
				lb[li].vCache = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
				lb[li].cacheKVContents()
			}
		}
		lFF := dFF // per-layer FFN width — gemma4 E2B/E4B MatFormer varies it (6144/12288); 0 ⇒ arch default
		if w.DFF > 0 {
			lFF = w.DFF
		}
		lb[li].dFF = lFF
		// KV-shared layers project only Q (they attend an owner's cache) and carry no
		// k/v weights — bind them optionally so the uploaded-copy path (sb == nil)
		// tolerates their absence exactly like the no-copy shard path already does.
		wK, wV := viewOrNil(w.WK), viewOrNil(w.WV)
		if specs[li].OwnsCache() {
			wK = view(w.WK)
		}
		p := bf16Projector{
			wQ: view(w.WQ), wK: wK, wV: wV, wO: view(w.WO),
			dModel: dModel, qDim: qDim, kvDim: kvDim, dFF: lFF,
		}
		if layers[li].MoE == nil {
			lb[li].mnw = view(w.MLPNormW)
			p.wGate = view(w.WGate)
			p.wUp = view(w.WUp)
			p.wDown = view(w.WDown)
		} else {
			moeWeights[li] = layers[li].MoE
		}
		lb[li].proj = p
	}
	return lb, moeWeights, ferr
}

// layerScalarBuf broadcasts a gemma4 per-layer output scalar (shape [1] bf16) to a dModel-length
// bf16 buffer for the per-layer output multiply, or nil when absent. The [1]→dModel fill matches
// metal.Mul(hidden, scalar) (broadcast); bf16→f32→bf16 round-trips the scalar value exactly.
func layerScalarBuf(scalarW []byte, dModel int) metal.MTLBuffer {
	if len(scalarW) != bf16Size {
		return nil
	}
	return bf16ConstBuffer(dModel, bf16ToF32(scalarW[0], scalarW[1]))
}

// valueNormOnesBuf is the gemma4 value-norm weight: a [headDim] bf16 ones vector so the
// proven RMSNorm-rows kernel computes the no-scale per-head RMSNorm on V (metal's
// RMSNormNoScale). Returns nil when off (non-gemma4) ⇒ the decode skips value-norm.
// MUST be called inside a withAutoreleasePool.
//
// headDim MUST be the LARGEST per-layer head dim (maxHeadDimOf), not the base/uniform one:
// gemma4 E2B global layers use head_dim 512 vs sliding 256, and the value-norm op reads
// axisSize=hdOf(li) (512 on a global layer). A buffer sized at the base 256 makes that read
// run off the end of the ones vector → the upper half of every global head's V is normed by
// garbage weights, diverging from the host path at the first global layer (proven by the
// q4 ICB per-layer localiser). The re-encode arch path already sizes it at maxHeadDim in
// newArchDecodeState; the ICB wrappers must do the same.
func valueNormOnesBuf(on bool, headDim int) metal.MTLBuffer {
	if !on {
		return nil
	}
	return bf16ConstBuffer(headDim, 1.0)
}

// maxHeadDimOf returns the largest per-layer head dim over specs (falling back to the base
// headDim) — the size the shared value-norm ones vector + any per-head-dim scratch must use so
// a wider global layer's read stays in bounds. Mirrors newArchDecodeState's maxHeadDim.
func maxHeadDimOf(specs []model.LayerSpec, headDim int) int {
	m := headDim
	for _, sp := range specs {
		if hd := headDimOf(sp, headDim); hd > m {
			m = hd
		}
	}
	return m
}
