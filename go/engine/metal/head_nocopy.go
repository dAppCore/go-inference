// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"runtime"
	"slices"
	"sync"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"github.com/tmc/apple/metal"
)

// This file is the resident LM head — the fix for the per-token serve memory balloon. The head
// runs once per generated token over the (tied) [vocab × dModel] weight: final RMSNorm, the output
// projection (bf16 gemv or 4-bit qmv), then the optional logit soft-cap. LMHeadBF16/LMHeadQuant
// upload that whole weight into a FRESH Metal buffer EVERY token (sharedBytes inside QMVBF16 /
// MatVecBF16), an owned copy the autorelease pool never frees → resident memory grows ~weight-size
// per token (the ~503 MB tied embedding at 12B = the ~59 GB serve balloon). headEncoder binds the
// head weight ONCE and reuses it every token: zero per-token upload, zero growth.
//
// HOW the weight is bound, by dtype:
//   - bf16: a no-copy view into the shared shard mmap, or the caller's resident in-memory backing
//     when no shard mapping exists (the gemv reads the shard buffer reliably — proven byte-identical
//     in the full session).
//   - 4-bit: uploaded ONCE into a retained owned buffer at session build, then reused. The 4-bit
//     affine_qmv reading a NO-COPY view of the shard mmap is unreliable when other quant buffers
//     coexist in the session (NaN — the same class of issue that keeps the quant LAYER weights on
//     the copy path); a single owned upload sidesteps it AND still kills the balloon (one upload,
//     not one per token). It costs ONE resident copy of the head weight — not the per-token growth.
// Either way the per-token cost is just the dModel-length activation upload; the weight is resident.

// headEncoder is a resident LM head, built once. For bf16 the weight is bound as a no-copy shard
// view; for 4-bit it is an owned buffer uploaded once at build (held resident on this struct). Both
// avoid the per-token weight upload that caused the balloon. encode() allocates only the tiny
// per-call scratch/output; direct greedy reuses tiny scratch buffers through a concurrency-safe
// pool. nil (no shardBuffers, or an unresolved weight) signals the caller to fall back to the
// per-token upload head.
type headEncoder struct {
	finalNorm bufView // bf16 final-norm, no-copy shard view (a tiny vector — always reliable)
	weight    bufView // bf16 no-copy shard view, OR the 4-bit packed weight uploaded once (off 0)
	// quant triple companions (4-bit head only): scales/biases uploaded once. nil buf for bf16.
	scales, biases  bufView
	softCapScale    bufView
	invSoftCapScale bufView
	quant           bool
	groupSize, bits int
	dModel, vocab   int
	eps, softCap    float32
	greedyScratch   sync.Pool
	topKScratch     sync.Pool
	hiddenScratch   headHiddenScratchPool
	// greedyRowsScratch pools the K-row direct-greedy buffers (normed rows,
	// per-row tile bests, out tokens) for the fused verify head.
	greedyRowsScratch sync.Pool
}

// headGreedyRowsScratch is one K-row fused-greedy invocation's GPU scratch.
// logitsRows is allocated only for the quant head (the bf16 head scores
// weight tiles directly and never materialises logits).
type headGreedyRowsScratch struct {
	kCap, tileCap, dModelCap, vocabCap         int
	normed, tileValues, tileIndices, outTokens metal.MTLBuffer
	logitsRows                                 metal.MTLBuffer
}

func (h *headEncoder) getGreedyRowsScratch(k, tileCount int) *headGreedyRowsScratch {
	return h.getGreedyRowsScratchLogits(k, tileCount, false)
}

func (h *headEncoder) getGreedyRowsScratchLogits(k, tileCount int, needLogits bool) *headGreedyRowsScratch {
	if v := h.greedyRowsScratch.Get(); v != nil {
		s := v.(*headGreedyRowsScratch)
		if s.kCap >= k && s.tileCap >= tileCount && s.dModelCap >= h.dModel && (!needLogits || s.vocabCap >= h.vocab) {
			return s
		}
	}
	s := &headGreedyRowsScratch{kCap: k, tileCap: tileCount, dModelCap: h.dModel}
	s.normed = device.NewBufferWithLengthOptions(uint(k*h.dModel*bf16Size), metal.MTLResourceStorageModeShared)
	s.tileValues = device.NewBufferWithLengthOptions(uint(k*tileCount*4), metal.MTLResourceStorageModeShared)
	s.tileIndices = device.NewBufferWithLengthOptions(uint(k*tileCount*4), metal.MTLResourceStorageModeShared)
	s.outTokens = device.NewBufferWithLengthOptions(uint(k*4), metal.MTLResourceStorageModeShared)
	if s.normed == nil || s.tileValues == nil || s.tileIndices == nil || s.outTokens == nil {
		return nil
	}
	if needLogits {
		s.logitsRows = device.NewBufferWithLengthOptions(uint(k*h.vocab*bf16Size), metal.MTLResourceStorageModeShared)
		if s.logitsRows == nil {
			return nil
		}
		s.vocabCap = h.vocab
	}
	return s
}

func (h *headEncoder) putGreedyRowsScratch(s *headGreedyRowsScratch) {
	if s != nil && s.normed != nil && s.tileValues != nil && s.tileIndices != nil && s.outTokens != nil {
		h.greedyRowsScratch.Put(s)
	}
}

type headHiddenScratchPool struct {
	core.Pool[any]
}

type headGreedyScratch struct {
	tileCapacity            int
	tileValues, tileIndices metal.MTLBuffer
	outToken                metal.MTLBuffer
	outTokenPtr             *int32
	sampleParams            metal.MTLBuffer
	sampleParamsPtr         *logitsSampleKernelParams
	dModelCapacity          int
	normed                  metal.MTLBuffer
	vocabCapacity           int
	logits                  metal.MTLBuffer
	logitsPtr               *byte
	logitsOutView           metal.MTLBuffer
	logitsOutViewPtr        uintptr
	logitsOutViewLen        int
	logitsOutViewPinned     *pinnedNoCopyBytes
	softcapA, softcapB      metal.MTLBuffer
	suppressCapacity        int
	suppress                metal.MTLBuffer
	suppressPtr             *int32
	suppressPinned          *pinnedNoCopyBytes
	historyCapacity         int
	history                 metal.MTLBuffer
	historyPtr              *int32
	historyPinned           *pinnedNoCopyBytes
}

type headHiddenScratch struct {
	n      int
	pinned *pinnedNoCopyBytes
	view   cachedNoCopyBytesView
}

type headTopKScratch struct {
	candidateCapacity     int
	candidateValues       metal.MTLBuffer
	candidateIndices      metal.MTLBuffer
	topKCapacity          int
	topValues, topIndices metal.MTLBuffer
	topValuesPtr          *float32
	topIndicesPtr         *int32
	outToken              metal.MTLBuffer
	outTokenPtr           *int32
	sampleParams          metal.MTLBuffer
	sampleParamsPtr       *topKSampleKernelParams
	dModelCapacity        int
	normed                metal.MTLBuffer
	vocabCapacity         int
	logits                metal.MTLBuffer
	suppressCapacity      int
	suppress              metal.MTLBuffer
	suppressPtr           *int32
	suppressPinned        *pinnedNoCopyBytes
	historyCapacity       int
	history               metal.MTLBuffer
	historyPtr            *int32
	historyPinned         *pinnedNoCopyBytes
}

type topKSampleKernelParams struct {
	n           int32
	topK        int32
	temperature float32
	topP        float32
	minP        float32
	draw        float32
}

type logitsSampleKernelParams struct {
	vocab         int32
	suppressCount int32
	historyCount  int32
	topK          int32
	temperature   float32
	topP          float32
	minP          float32
	draw          float32
	repeatPenalty float32
}

// newHeadEncoder builds the resident head: it resolves the final norm to a no-copy shard view when
// a shard mapping is available, otherwise it binds owned resident buffers for in-memory sessions.
// BF16 directory heads use no-copy shard views; 4-bit heads use a one-time owned upload (packed +
// scales + biases) because qmv over the shared mmap is unreliable in-session. Returns nil only when
// required weights are missing or an expected shard view cannot be resolved. MUST be called inside a
// withAutoreleasePool (the owned buffers are objc-retained, so they survive it).
func newHeadEncoder(sb *shardBuffers, finalNormW, weight, scales, biases []byte, dModel, vocab, groupSize, bits int, eps, softCap float32, quant bool) (*headEncoder, error) {
	h := &headEncoder{
		quant:     quant,
		groupSize: groupSize, bits: bits, dModel: dModel, vocab: vocab, eps: eps, softCap: softCap,
	}
	if quant {
		// Upload-once owned buffers — weight + scales + biases AND the final norm: a handful of
		// tensors, one upload, no per-token balloon. (An older version of this comment blamed an
		// "in-session aliasing issue" — that ghost was safetensors OFFSET ALIGNMENT, solved by
		// bufForAligned's per-tensor fallback; the upload here stays because it is simply the
		// cheapest correct shape for four tensors, not because views are hazardous.)
		if len(finalNormW) == 0 || len(weight) == 0 || len(scales) == 0 || len(biases) == 0 {
			return nil, nil
		}
		h.finalNorm = copyView(finalNormW)
		h.weight = copyView(weight)
		h.scales = copyView(scales)
		h.biases = copyView(biases)
		h.initSoftcapBuffers()
		return h, nil
	}
	if len(finalNormW) == 0 || len(weight) == 0 {
		return nil, nil
	}
	if sb == nil {
		h.finalNorm = bufView{buf: residentBytes(finalNormW)}
		h.weight = bufView{buf: residentBytes(weight)}
		h.initSoftcapBuffers()
		return h, nil
	}
	// bf16: no-copy shard views. The direct-head kernels read the weight as bfloat4 (the K-row
	// verify head's unroll), so the view demands 8-byte alignment — safetensors guarantees NONE
	// (measured: 248/281 bf16 tensors in e2b-bf16 sit ≡2/4/6 mod 8; the tied embedding only
	// escapes by being the file's first tensor). A misaligned weight falls back to a private
	// copy of just that tensor inside bufForAligned rather than feeding the vec loads garbage.
	fn, err := sb.bufFor(finalNormW)
	if err != nil || fn.buf == nil {
		return nil, nil
	}
	w, err := sb.bufForAligned(weight, 8)
	if err != nil || w.buf == nil {
		return nil, nil
	}
	h.finalNorm = fn
	h.weight = w
	h.initSoftcapBuffers()
	return h, nil
}

func (h *headEncoder) initSoftcapBuffers() {
	if h.softCap <= 0 {
		return
	}
	h.invSoftCapScale = bufView{buf: bf16ConstBuffer(1, 1/h.softCap)}
	h.softCapScale = bufView{buf: bf16ConstBuffer(1, h.softCap)}
}

func newHeadHiddenScratch(n int) (*headHiddenScratch, error) {
	pinned, err := newPinnedNoCopyBytes(n)
	if err != nil {
		return nil, err
	}
	return &headHiddenScratch{n: n, pinned: pinned}, nil
}

func (s *headHiddenScratch) Close() {
	if s == nil {
		return
	}
	if s.pinned != nil {
		s.pinned.Close()
		s.pinned = nil
	}
	s.view.Close()
	s.n = 0
}

func (h *headEncoder) getHiddenScratch(n int) (*headHiddenScratch, error) {
	if v := h.hiddenScratch.Get(); v != nil {
		s := v.(*headHiddenScratch)
		if s != nil && s.n == n && s.pinned != nil && s.pinned.buf != nil {
			return s, nil
		}
		if s != nil {
			s.Close()
		}
	}
	return newHeadHiddenScratch(n)
}

func (h *headEncoder) putHiddenScratch(s *headHiddenScratch) {
	if s != nil && s.n > 0 && s.pinned != nil && s.pinned.buf != nil {
		h.hiddenScratch.Put(s)
	}
}

func (h *headEncoder) hiddenBuffer(hidden []byte) (*headHiddenScratch, metal.MTLBuffer, error) {
	scratch, err := h.getHiddenScratch(len(hidden))
	if err != nil {
		if initErr != nil {
			return nil, nil, err
		}
		return nil, sharedBytes(hidden), nil
	}
	var buf metal.MTLBuffer
	var ok bool
	if len(hidden) >= 64 {
		buf, ok = scratch.view.buffer(hidden)
	}
	if !ok {
		buf, err = scratch.pinned.copyBuffer(hidden)
		if err != nil {
			scratch.Close()
			return nil, sharedBytes(hidden), nil
		}
	}
	return scratch, buf, nil
}

// encode runs the head for one hidden state (dModel bf16 bytes) and returns vocab bf16 logits,
// binding the RESIDENT head weight — NO per-token weight upload (the whole point: the ~503 MB
// tied embedding is bound once, not re-uploaded). Same RMSNorm and gemv/qmv kernel + ABI as
// LMHeadBF16/LMHeadQuant; sampled softcap stays on the BF16 kernel route instead of looping on
// the host. The
// per-call scratch/output are freshly allocated (small, transient), so encode holds no shared
// mutable state and is concurrency-safe.
func (h *headEncoder) encode(hidden []byte, skipSoftcap bool) ([]byte, error) {
	return h.encodeInto(hidden, skipSoftcap, nil)
}

func (h *headEncoder) encodeInto(hidden []byte, skipSoftcap bool, out []byte) ([]byte, error) {
	if len(hidden) != h.dModel*bf16Size {
		return nil, core.NewError("native.headEncoder.encode: hidden must be dModel bf16 bytes")
	}
	if cap(out) < h.vocab*bf16Size {
		out = make([]byte, h.vocab*bf16Size)
	} else {
		out = out[:h.vocab*bf16Size]
	}
	var encErr error
	if pool, ok := beginAutoreleasePoolRaw(); ok {
		defer endAutoreleasePoolRaw(pool)
		var hiddenScratch *headHiddenScratch
		var hiddenBuf metal.MTLBuffer
		hiddenScratch, hiddenBuf, encErr = h.hiddenBuffer(hidden) // the only host staging: the dModel-length activation, not the weight
		if encErr == nil {
			encErr = h.encodeBufferIntoPool(hiddenBuf, skipSoftcap, out)
			h.putHiddenScratch(hiddenScratch)
		}
	} else {
		withAutoreleasePool(func() {
			var hiddenScratch *headHiddenScratch
			var hiddenBuf metal.MTLBuffer
			hiddenScratch, hiddenBuf, encErr = h.hiddenBuffer(hidden) // the only host staging: the dModel-length activation, not the weight
			if encErr != nil {
				return
			}
			defer h.putHiddenScratch(hiddenScratch)
			encErr = h.encodeBufferIntoPool(hiddenBuf, skipSoftcap, out)
		})
	}
	if encErr != nil {
		return nil, encErr
	}
	return out, nil
}

func (h *headEncoder) encodeBufferInto(hiddenBuf metal.MTLBuffer, skipSoftcap bool, out []byte) ([]byte, error) {
	if hiddenBuf == nil {
		return nil, core.NewError("native.headEncoder.encode: missing hidden buffer")
	}
	if cap(out) < h.vocab*bf16Size {
		out = make([]byte, h.vocab*bf16Size)
	} else {
		out = out[:h.vocab*bf16Size]
	}
	var encErr error
	if pool, ok := beginAutoreleasePoolRaw(); ok {
		defer endAutoreleasePoolRaw(pool)
		encErr = h.encodeBufferIntoPool(hiddenBuf, skipSoftcap, out)
	} else {
		withAutoreleasePool(func() {
			encErr = h.encodeBufferIntoPool(hiddenBuf, skipSoftcap, out)
		})
	}
	if encErr != nil {
		return nil, encErr
	}
	return out, nil
}

func (h *headEncoder) encodeBufferIntoPool(hiddenBuf metal.MTLBuffer, skipSoftcap bool, out []byte) error {
	scratch := h.getGreedyScratch(1, true, h.softCap > 0 && !skipSoftcap && h.vocab > 0)
	normed := scratch.normed
	logits := scratch.logits
	outLen := h.vocab * bf16Size
	directOut := false
	if len(out) >= outLen {
		out = out[:outLen]
	}
	if len(out) == outLen {
		tmp, ok := scratch.logitsOutputView(out)
		if ok {
			logits = tmp
			directOut = true
		}
	}
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	if err := h.emitLogitsChainAt(enc, hiddenBuf, 0, skipSoftcap, scratch, normed, logits); err != nil {
		endEncodingFast(enc)
		h.putGreedyScratch(scratch)
		return err
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	if !directOut {
		copy(out, unsafe.Slice(scratch.logitsPtr, h.vocab*bf16Size))
	}
	h.putGreedyScratch(scratch)
	return nil
}

// emitLogitsChainAt encodes the full-vocab logits chain — final RMSNorm, the
// lm_head matvec, and the optional softcap triple — into the CALLER's live
// encoder, against the given hidden buffer/offset and the scratch's staging
// buffers. Factored from encodeBufferIntoPool so a batched caller can encode
// K rows into ONE command buffer with the IDENTICAL per-row kernels (the
// per-lane and batched logits stay byte-identical by construction).
func (h *headEncoder) emitLogitsChainAt(enc metal.MTLComputeCommandEncoderObject, hiddenBuf metal.MTLBuffer, hiddenOff uint, skipSoftcap bool, scratch *headGreedyScratch, normed, logits metal.MTLBuffer) error {
	sink := encObjectSink{enc: enc}
	rmsPSO, err := pipelineFor(rmsKernelBF16(h.dModel))
	if err != nil {
		return err
	}
	if hiddenOff == 0 {
		emitRMSNorm(sink, rmsPSO, hiddenBuf, h.finalNorm.buf, normed, h.finalNorm.off, h.dModel, h.eps, rmsThreadgroup(h.dModel, rmsPSO))
	} else {
		if err := encRMSNormRowsBF16(enc, hiddenBuf, h.finalNorm.buf, normed, hiddenOff, h.finalNorm.off, 0, 1, h.dModel, h.eps); err != nil {
			return err
		}
	}
	if h.quant {
		qmvPSO, err := pipelineFor(qmvBF16KernelName(h.vocab, h.dModel, h.groupSize, h.bits))
		if err != nil {
			return err
		}
		emitQMV(sink, qmvPSO, h.weight.buf, h.weight.off, h.scales.buf, h.scales.off, h.biases.buf, h.biases.off, normed, logits, 0, h.dModel, h.vocab)
	} else {
		bm, bn, sm, sn, tm, tn := gemvTiles(h.dModel, h.vocab)
		gemvPSO, err := pipelineFor(gemvKernelName("bfloat16", bm, bn, sm, sn, tm, tn))
		if err != nil {
			return err
		}
		emitGemv(sink, gemvPSO, h.weight.buf, h.weight.off, normed, logits, 0, h.dModel, h.vocab, bm, bn, sm, tm)
	}
	if h.softCap > 0 && !skipSoftcap && h.vocab > 0 {
		invBytes := bf16ScalarBytes(1 / h.softCap)
		invScale := h.invSoftCapScale
		if invScale.buf == nil {
			invScale = bufView{buf: bf16ConstBuffer(1, 1/h.softCap)}
		}
		if err := encScaleBF16Object(enc, logits, invScale.buf, scratch.softcapA, invScale.off, invBytes[:], h.vocab); err != nil {
			return err
		}
		if err := encTanhBF16Object(enc, scratch.softcapA, scratch.softcapB, h.vocab); err != nil {
			return err
		}
		capBytes := bf16ScalarBytes(h.softCap)
		capScale := h.softCapScale
		if capScale.buf == nil {
			capScale = bufView{buf: bf16ConstBuffer(1, h.softCap)}
		}
		if err := encScaleBF16Object(enc, scratch.softcapB, capScale.buf, logits, capScale.off, capBytes[:], h.vocab); err != nil {
			return err
		}
	}
	return nil
}

// encodeLogitsRowsInPool encodes K independent full-vocab logits chains — one
// per hidden buffer — into a SINGLE command buffer with a single wait,
// copying each row's logits into outs[i]. The per-row kernels are exactly
// encodeBufferIntoPool's (emitLogitsChainAt), so each row's bytes equal the
// per-lane path's; only the submission count changes (K commit+wait
// round-trips become one — the sampled Phase-1 tax the lane probe measured).
func (h *headEncoder) encodeLogitsRowsInPool(hiddenBufs []metal.MTLBuffer, skipSoftcap bool, outs [][]byte) error {
	k := len(hiddenBufs)
	if k == 0 || len(outs) != k {
		return core.NewError("native.headEncoder.encodeLogitsRows: mismatched batch")
	}
	needSoftcap := h.softCap > 0 && !skipSoftcap && h.vocab > 0
	scratches := make([]*headGreedyScratch, 0, k)
	release := func() {
		for _, sc := range scratches {
			h.putGreedyScratch(sc)
		}
	}
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	for i := range hiddenBufs {
		scratch := h.getGreedyScratch(1, true, needSoftcap)
		scratches = append(scratches, scratch)
		if err := h.emitLogitsChainAt(enc, hiddenBufs[i], 0, skipSoftcap, scratch, scratch.normed, scratch.logits); err != nil {
			endEncodingFast(enc)
			release()
			return err
		}
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	outLen := h.vocab * bf16Size
	for i, scratch := range scratches {
		if len(outs[i]) != outLen {
			release()
			return core.NewError("native.headEncoder.encodeLogitsRows: out row must be vocab bf16 bytes")
		}
		copy(outs[i], unsafe.Slice(scratch.logitsPtr, outLen))
	}
	release()
	return nil
}

func newHeadGreedyScratch(tileCapacity, dModel, vocab int, needLogits, needSoftcap bool) *headGreedyScratch {
	s := &headGreedyScratch{
		tileCapacity: tileCapacity,
		tileValues:   device.NewBufferWithLengthOptions(uint(tileCapacity*4), metal.MTLResourceStorageModeShared),
		tileIndices:  device.NewBufferWithLengthOptions(uint(tileCapacity*4), metal.MTLResourceStorageModeShared),
		outToken:     device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared),
		sampleParams: device.NewBufferWithLengthOptions(uint(unsafe.Sizeof(logitsSampleKernelParams{})), metal.MTLResourceStorageModeShared),
	}
	s.outTokenPtr = (*int32)(s.outToken.Contents())
	s.sampleParamsPtr = (*logitsSampleKernelParams)(s.sampleParams.Contents())
	if dModel > 0 {
		s.dModelCapacity = dModel
		s.normed = scratchBF16(dModel)
	}
	if needLogits && vocab > 0 {
		s.vocabCapacity = vocab
		s.logits = scratchBF16(vocab)
		s.logitsPtr = (*byte)(s.logits.Contents())
	}
	if needSoftcap && vocab > 0 {
		s.softcapA = scratchBF16(vocab)
		s.softcapB = scratchBF16(vocab)
	}
	return s
}

func (h *headEncoder) getGreedyScratch(tileCount int, needLogits, needSoftcap bool) *headGreedyScratch {
	if v := h.greedyScratch.Get(); v != nil {
		s := v.(*headGreedyScratch)
		hasTiles := s.tileCapacity >= tileCount && s.tileValues != nil && s.tileIndices != nil && s.outToken != nil && s.outTokenPtr != nil
		hasNormed := s.dModelCapacity >= h.dModel && s.normed != nil
		hasParams := s.sampleParams != nil && s.sampleParamsPtr != nil
		hasLogits := !needLogits || (s.vocabCapacity >= h.vocab && s.logits != nil && s.logitsPtr != nil)
		hasSoftcap := !needSoftcap || (s.vocabCapacity >= h.vocab && s.softcapA != nil && s.softcapB != nil)
		if hasTiles && hasNormed && hasParams && hasLogits && hasSoftcap {
			return s
		}
	}
	return newHeadGreedyScratch(tileCount, h.dModel, h.vocab, needLogits, needSoftcap)
}

func (h *headEncoder) putGreedyScratch(s *headGreedyScratch) {
	if s != nil && s.tileValues != nil && s.tileIndices != nil && s.outToken != nil && s.outTokenPtr != nil && s.sampleParams != nil && s.sampleParamsPtr != nil && s.normed != nil {
		h.greedyScratch.Put(s)
	}
}

func (s *headGreedyScratch) closeLogitsOutputView() {
	if s == nil {
		return
	}
	if s.logitsOutViewPinned != nil {
		s.logitsOutViewPinned.Close()
	}
	s.logitsOutViewPtr = 0
	s.logitsOutViewLen = 0
	s.logitsOutView = nil
	s.logitsOutViewPinned = nil
}

func (s *headGreedyScratch) logitsOutputView(out []byte) (metal.MTLBuffer, bool) {
	if s == nil || len(out) == 0 {
		return nil, false
	}
	ptr := uintptr(unsafe.Pointer(&out[0]))
	if s.logitsOutView != nil && s.logitsOutViewPtr == ptr && s.logitsOutViewLen == len(out) {
		return s.logitsOutView, true
	}
	s.closeLogitsOutputView()
	if buf, ok := registeredPinnedNoCopyBytes(out); ok {
		s.logitsOutViewPtr = ptr
		s.logitsOutViewLen = len(out)
		s.logitsOutView = buf
		s.logitsOutViewPinned = nil
		return buf, true
	}
	buf, pinner, noCopy := residentNoCopyBytes(out)
	if !noCopy {
		if pinner != nil {
			pinner.Unpin()
		}
		return nil, false
	}
	pinned := &pinnedNoCopyBytes{bytes: out, buf: buf, pinner: pinner}
	runtime.SetFinalizer(pinned, (*pinnedNoCopyBytes).Close)
	s.logitsOutViewPtr = ptr
	s.logitsOutViewLen = len(out)
	s.logitsOutView = buf
	s.logitsOutViewPinned = pinned
	return buf, true
}

func (h *headEncoder) directGreedyUsable() bool {
	if h == nil || h.finalNorm.buf == nil || h.weight.buf == nil {
		return false
	}
	if h.quant {
		return h.scales.buf != nil && h.biases.buf != nil && qmvLogitsArgmaxUsable(h.dModel, h.vocab, h.groupSize, h.bits)
	}
	return bf16LMHeadArgmaxUsable(h.dModel, h.vocab)
}

func (s *headGreedyScratch) suppressBuffer(ids []int32) metal.MTLBuffer {
	if len(ids) == 0 {
		return nil
	}
	if s.suppress == nil || s.suppressCapacity < len(ids) {
		if s.suppressPinned != nil {
			s.suppressPinned.Close()
			s.suppressPinned = nil
		}
		s.suppressCapacity = len(ids)
		if pinned, err := newPinnedNoCopyBytes(len(ids) * 4); err == nil {
			s.suppressPinned = pinned
			s.suppress = pinned.buf
			s.suppressPtr = (*int32)(unsafe.Pointer(&pinned.bytes[0]))
		} else {
			s.suppress = device.NewBufferWithLengthOptions(uint(len(ids)*4), metal.MTLResourceStorageModeShared)
			s.suppressPtr = (*int32)(s.suppress.Contents())
		}
	}
	copy(unsafe.Slice(s.suppressPtr, len(ids)), ids)
	return s.suppress
}

func (s *headGreedyScratch) historyBuffer(ids []int32) metal.MTLBuffer {
	if len(ids) == 0 {
		return nil
	}
	if s.history == nil || s.historyCapacity < len(ids) {
		if s.historyPinned != nil {
			s.historyPinned.Close()
			s.historyPinned = nil
		}
		s.historyCapacity = len(ids)
		if pinned, err := newPinnedNoCopyBytes(len(ids) * 4); err == nil {
			s.historyPinned = pinned
			s.history = pinned.buf
			s.historyPtr = (*int32)(unsafe.Pointer(&pinned.bytes[0]))
		} else {
			s.history = device.NewBufferWithLengthOptions(uint(len(ids)*4), metal.MTLResourceStorageModeShared)
			s.historyPtr = (*int32)(s.history.Contents())
		}
	}
	copy(unsafe.Slice(s.historyPtr, len(ids)), ids)
	return s.history
}

func (s *headGreedyScratch) logitsSampleParamsBuffer(params model.SampleParams, draw float32, vocab int, suppressCount int, historyCount int) metal.MTLBuffer {
	p := s.sampleParamsPtr
	*p = logitsSampleKernelParams{
		vocab:         int32(vocab),
		suppressCount: int32(suppressCount),
		historyCount:  int32(historyCount),
		topK:          int32(logitsSampleKernelTopK(params, vocab)),
		temperature:   params.Temperature,
		topP:          params.TopP,
		minP:          params.MinP,
		draw:          draw,
		repeatPenalty: params.RepeatPenalty,
	}
	return s.sampleParams
}

func (s *headGreedyScratch) token() int32 {
	return *s.outTokenPtr
}

func newHeadTopKScratch(candidateCount, topK, dModel, vocab int, needLogits bool) *headTopKScratch {
	s := &headTopKScratch{
		candidateCapacity: candidateCount,
		candidateValues:   device.NewBufferWithLengthOptions(uint(candidateCount*4), metal.MTLResourceStorageModeShared),
		candidateIndices:  device.NewBufferWithLengthOptions(uint(candidateCount*4), metal.MTLResourceStorageModeShared),
		topKCapacity:      topK,
		topValues:         device.NewBufferWithLengthOptions(uint(topK*4), metal.MTLResourceStorageModeShared),
		topIndices:        device.NewBufferWithLengthOptions(uint(topK*4), metal.MTLResourceStorageModeShared),
		outToken:          device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared),
		sampleParams:      device.NewBufferWithLengthOptions(uint(unsafe.Sizeof(topKSampleKernelParams{})), metal.MTLResourceStorageModeShared),
	}
	s.topValuesPtr = (*float32)(s.topValues.Contents())
	s.topIndicesPtr = (*int32)(s.topIndices.Contents())
	s.outTokenPtr = (*int32)(s.outToken.Contents())
	s.sampleParamsPtr = (*topKSampleKernelParams)(s.sampleParams.Contents())
	if dModel > 0 {
		s.dModelCapacity = dModel
		s.normed = scratchBF16(dModel)
	}
	if needLogits && vocab > 0 {
		s.vocabCapacity = vocab
		s.logits = scratchBF16(vocab)
	}
	return s
}

func (h *headEncoder) getTopKScratch(candidateCount, topK int, needLogits bool) *headTopKScratch {
	if v := h.topKScratch.Get(); v != nil {
		s := v.(*headTopKScratch)
		hasCandidates := s.candidateCapacity >= candidateCount && s.candidateValues != nil && s.candidateIndices != nil
		hasTopK := s.topKCapacity >= topK && s.topValues != nil && s.topValuesPtr != nil && s.topIndices != nil && s.topIndicesPtr != nil && s.outToken != nil && s.outTokenPtr != nil && s.sampleParams != nil && s.sampleParamsPtr != nil
		hasNormed := s.dModelCapacity >= h.dModel && s.normed != nil
		hasLogits := !needLogits || (s.vocabCapacity >= h.vocab && s.logits != nil)
		if hasCandidates && hasTopK && hasNormed && hasLogits {
			return s
		}
	}
	return newHeadTopKScratch(candidateCount, topK, h.dModel, h.vocab, needLogits)
}

func (h *headEncoder) putTopKScratch(s *headTopKScratch) {
	if s != nil && s.candidateValues != nil && s.candidateIndices != nil && s.topValues != nil && s.topValuesPtr != nil && s.topIndices != nil && s.topIndicesPtr != nil && s.outToken != nil && s.outTokenPtr != nil && s.sampleParams != nil && s.sampleParamsPtr != nil && s.normed != nil {
		h.topKScratch.Put(s)
	}
}

func (s *headTopKScratch) sampleParamsBuffer(params model.SampleParams, draw float32, candidateCount int) metal.MTLBuffer {
	p := s.sampleParamsPtr
	*p = topKSampleKernelParams{
		n:           int32(candidateCount),
		topK:        int32(params.TopK),
		temperature: params.Temperature,
		topP:        params.TopP,
		minP:        params.MinP,
		draw:        draw,
	}
	return s.sampleParams
}

func (s *headTopKScratch) token() int32 {
	return *s.outTokenPtr
}

func (s *headTopKScratch) suppressBuffer(ids []int32) metal.MTLBuffer {
	if len(ids) == 0 {
		return nil
	}
	if s.suppress == nil || s.suppressCapacity < len(ids) {
		if s.suppressPinned != nil {
			s.suppressPinned.Close()
			s.suppressPinned = nil
		}
		s.suppressCapacity = len(ids)
		if pinned, err := newPinnedNoCopyBytes(len(ids) * 4); err == nil {
			s.suppressPinned = pinned
			s.suppress = pinned.buf
			s.suppressPtr = (*int32)(unsafe.Pointer(&pinned.bytes[0]))
		} else {
			s.suppress = device.NewBufferWithLengthOptions(uint(len(ids)*4), metal.MTLResourceStorageModeShared)
			s.suppressPtr = (*int32)(s.suppress.Contents())
		}
	}
	copy(unsafe.Slice(s.suppressPtr, len(ids)), ids)
	return s.suppress
}

func (s *headTopKScratch) historyBuffer(ids []int32) metal.MTLBuffer {
	if len(ids) == 0 {
		return nil
	}
	if s.history == nil || s.historyCapacity < len(ids) {
		if s.historyPinned != nil {
			s.historyPinned.Close()
			s.historyPinned = nil
		}
		s.historyCapacity = len(ids)
		if pinned, err := newPinnedNoCopyBytes(len(ids) * 4); err == nil {
			s.historyPinned = pinned
			s.history = pinned.buf
			s.historyPtr = (*int32)(unsafe.Pointer(&pinned.bytes[0]))
		} else {
			s.history = device.NewBufferWithLengthOptions(uint(len(ids)*4), metal.MTLResourceStorageModeShared)
			s.historyPtr = (*int32)(s.history.Contents())
		}
	}
	copy(unsafe.Slice(s.historyPtr, len(ids)), ids)
	return s.history
}

func tokenSuppressed(id int, suppress []int32) bool {
	return slices.Contains(suppress, int32(id))
}

func greedyBF16Suppressed(logits []byte, vocab int, suppress []int32) (int32, error) {
	if len(suppress) == 0 {
		return model.Greedy(logits, vocab)
	}
	if len(logits) != vocab*bf16Size {
		return 0, core.NewError("native.greedyBF16Suppressed: logits must be vocab bf16 bytes")
	}
	best := -1
	var bestV float32
	for i := range vocab {
		if tokenSuppressed(i, suppress) {
			continue
		}
		v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1])
		if best < 0 || v > bestV {
			best, bestV = i, v
		}
	}
	if best < 0 {
		return 0, core.NewError("native.greedyBF16Suppressed: all vocab ids are suppressed")
	}
	return int32(best), nil
}

// greedy is the direct-token counterpart to pkg/metal's direct greedy/q4 LM-head
// top-k features, narrowed to the production greedy case. It runs final RMSNorm
// and head argmax in one command buffer, masks suppressed ids before argmax,
// and copies back only the selected token. ok=false means this head/geometry
// cannot use the custom kernel, so callers keep the existing full-logits path.
// encodeGreedy encodes finalRMSNorm(hiddenBuf) + LMHead + tiled argmax into `enc` WITHOUT committing —
// the caller owns the command buffer, so a decode step can chain its replay onto the SAME buffer and pay
// one sync/token instead of two. The returned scratch owns the GPU token buffer and its cached contents
// pointer; callers must read scratch.token() only after the command buffer completes. ok=false ⇒ the head
// can't do a direct GPU argmax (caller falls back to the logits path).
func (h *headEncoder) encodeGreedy(enc metal.MTLComputeCommandEncoder, hiddenBuf metal.MTLBuffer, suppress []int32) (scratch *headGreedyScratch, ok bool, err error) {
	return h.encodeGreedyAt(enc, hiddenBuf, 0, suppress)
}

func (h *headEncoder) encodeGreedyAt(enc metal.MTLComputeCommandEncoder, hiddenBuf metal.MTLBuffer, hiddenOff uint, suppress []int32) (scratch *headGreedyScratch, ok bool, err error) {
	if !h.directGreedyUsable() {
		return nil, false, nil
	}
	rowsPerTile := bf16LMHeadArgmaxRowsPerTile
	needLogits := false
	if h.quant {
		rowsPerTile = bf16LogitsArgmaxRowsPerTile
		needLogits = true
	}
	tileCount := (h.vocab + rowsPerTile - 1) / rowsPerTile
	scratch = h.getGreedyScratch(tileCount, needLogits, false)
	normed := scratch.normed
	suppressBuf := scratch.suppressBuffer(suppress)
	if hiddenOff == 0 {
		err = encRMSNormBF16(enc, hiddenBuf, h.finalNorm.buf, normed, h.finalNorm.off, h.dModel, h.eps)
	} else {
		err = encRMSNormRowsBF16(enc, hiddenBuf, h.finalNorm.buf, normed, hiddenOff, h.finalNorm.off, 0, 1, h.dModel, h.eps)
	}
	if err != nil {
		return scratch, true, err
	}
	if h.quant {
		logits := scratch.logits
		if err = encQMVBF16(enc, h.weight.buf, h.scales.buf, h.biases.buf, normed, logits,
			h.weight.off, h.scales.off, h.biases.off, 0, h.vocab, h.dModel, h.groupSize, h.bits); err != nil {
			return scratch, true, err
		}
		if err = encBF16LogitsArgmaxTilesBF16(enc, logits, scratch.tileValues, scratch.tileIndices, suppressBuf, h.vocab, len(suppress)); err != nil {
			return scratch, true, err
		}
	} else {
		if err = encBF16LMHeadArgmaxTilesBF16(enc, normed, h.weight.buf, scratch.tileValues, scratch.tileIndices, suppressBuf, 0, h.weight.off, h.dModel, h.vocab, len(suppress)); err != nil {
			return scratch, true, err
		}
	}
	if err = encArgmaxMergeF32(enc, scratch.tileValues, scratch.tileIndices, scratch.outToken, tileCount); err != nil {
		return scratch, true, err
	}
	return scratch, true, nil
}

func (h *headEncoder) greedy(hidden []byte, suppress []int32) (token int32, ok bool, err error) {
	withAutoreleasePool(func() {
		token, ok, err = h.greedyInPool(hidden, suppress)
	})
	return token, ok, err
}

func (h *headEncoder) greedyInPool(hidden []byte, suppress []int32) (token int32, ok bool, err error) {
	if len(hidden) != h.dModel*bf16Size {
		return 0, true, core.NewError("native.headEncoder.greedy: hidden must be dModel bf16 bytes")
	}
	token = -1
	hiddenScratch, hiddenBuf, err := h.hiddenBuffer(hidden)
	if err != nil {
		return 0, true, err
	}
	defer h.putHiddenScratch(hiddenScratch)
	return h.greedyBufferInPool(hiddenBuf, suppress)
}

func (h *headEncoder) greedyBufferInPool(hiddenBuf metal.MTLBuffer, suppress []int32) (token int32, ok bool, err error) {
	return h.greedyBufferAtInPool(hiddenBuf, 0, suppress)
}

func (h *headEncoder) greedyBufferAtInPool(hiddenBuf metal.MTLBuffer, hiddenOff uint, suppress []int32) (token int32, ok bool, err error) {
	if hiddenBuf == nil {
		return 0, true, core.NewError("native.headEncoder.greedy: missing hidden buffer")
	}
	if !h.hiddenBufferOffsetInRange(hiddenBuf, hiddenOff) {
		return 0, true, core.NewError("native.headEncoder.greedy: hidden offset is out of range")
	}
	token = -1
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	var scratch *headGreedyScratch
	scratch, ok, err = h.encodeGreedyAt(enc, hiddenBuf, hiddenOff, suppress)
	if !ok || err != nil {
		endEncodingFast(enc)
		if scratch != nil {
			h.putGreedyScratch(scratch)
		}
		if err != nil {
			return 0, true, err
		}
		return 0, false, nil
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	token = scratch.token()
	if (token < 0 || int(token) >= h.vocab) && argmaxDebugEnabled() {
		diag := h.argmaxInvalidDiag(scratch, hiddenBuf, hiddenOff)
		h.putGreedyScratch(scratch)
		return 0, true, core.NewError(core.Sprintf("native.headEncoder.greedy: direct argmax returned invalid token %d for vocab %d%s", token, h.vocab, diag))
	}
	h.putGreedyScratch(scratch)
	if !ok {
		return 0, false, nil
	}
	if token < 0 || int(token) >= h.vocab {
		return 0, true, core.NewError(core.Sprintf("native.headEncoder.greedy: direct argmax returned invalid token %d for vocab %d", token, h.vocab))
	}
	return token, true, nil
}

// argmaxDebugEnabled gates the invalid-token forensic dump (the ~52K long-context
// corruption hunt): when the direct argmax returns an out-of-range token, dump per-stage
// NaN counts and extrema (hidden → normed → logits → tiles) so a failing run localises
// which stage the garbage entered at. Enable with LTHN_DEBUG_ARGMAX=1.
func argmaxDebugEnabled() bool { return os.Getenv("LTHN_DEBUG_ARGMAX") != "" }

// bf16NaNScanBytes counts bf16 NaNs in a host byte slice, reporting the first NaN's
// element index (-1 when none) — the host-side sibling of bf16BufStats.
func bf16NaNScanBytes(b []byte) (count, firstIdx int) {
	firstIdx = -1
	for i := 0; i+1 < len(b); i += 2 {
		h := uint16(b[i]) | uint16(b[i+1])<<8
		if h&0x7F80 == 0x7F80 && h&0x007F != 0 {
			if firstIdx < 0 {
				firstIdx = i / 2
			}
			count++
		}
	}
	return count, firstIdx
}

// bf16BufStats scans n bf16 elements at a buffer offset: NaN/±Inf counts, finite min/max,
// and the first NaN's element index (-1 when none).
func bf16BufStats(buf metal.MTLBuffer, off uint, n int) (nan, inf int, minV, maxV float32, firstNaN int) {
	firstNaN = -1
	if buf == nil || n <= 0 {
		return 0, 0, 0, 0, -1
	}
	b := unsafe.Slice((*byte)(unsafe.Add(buf.Contents(), uintptr(off))), n*bf16Size)
	first := true
	for i := 0; i < n; i++ {
		h := uint16(b[i*2]) | uint16(b[i*2+1])<<8
		if h&0x7F80 == 0x7F80 {
			if h&0x007F != 0 {
				nan++
				if firstNaN < 0 {
					firstNaN = i
				}
			} else {
				inf++
			}
			continue
		}
		v := bf16ToF32(b[i*2], b[i*2+1])
		if first {
			minV, maxV, first = v, v, false
			continue
		}
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
	}
	return nan, inf, minV, maxV, firstNaN
}

// argmaxInvalidDiag reads the greedy scratch's shared-storage stages after an invalid
// direct-argmax token and reports where the garbage entered: the input hidden, the
// RMS-normed hidden, the (quant-path) logits, and the per-tile argmax values/indices.
func (h *headEncoder) argmaxInvalidDiag(scratch *headGreedyScratch, hiddenBuf metal.MTLBuffer, hiddenOff uint) string {
	rowsPerTile := bf16LMHeadArgmaxRowsPerTile
	if h.quant {
		rowsPerTile = bf16LogitsArgmaxRowsPerTile
	}
	tileCount := (h.vocab + rowsPerTile - 1) / rowsPerTile
	hn, hi, hmin, hmax, hf := bf16BufStats(hiddenBuf, hiddenOff, h.dModel)
	nn, ni, nmin, nmax, nf := bf16BufStats(scratch.normed, 0, h.dModel)
	d := core.Sprintf("\n  argmax-diag: hidden NaN=%d Inf=%d min=%.4g max=%.4g firstNaN=%d\n  normed NaN=%d Inf=%d min=%.4g max=%.4g firstNaN=%d",
		hn, hi, hmin, hmax, hf, nn, ni, nmin, nmax, nf)
	if h.quant && scratch.logits != nil {
		ln, li, lmin, lmax, lf := bf16BufStats(scratch.logits, 0, h.vocab)
		d += core.Sprintf("\n  logits NaN=%d Inf=%d min=%.4g max=%.4g firstNaN=%d", ln, li, lmin, lmax, lf)
	}
	if scratch.tileValues != nil && scratch.tileIndices != nil {
		tv := unsafe.Slice((*float32)(scratch.tileValues.Contents()), tileCount)
		ti := unsafe.Slice((*int32)(scratch.tileIndices.Contents()), tileCount)
		tnan, tneg := 0, 0
		var tmin, tmax float32
		firstBad := -1
		for i := 0; i < tileCount; i++ {
			v := tv[i]
			if v != v {
				tnan++
				if firstBad < 0 {
					firstBad = i
				}
				continue
			}
			if i == 0 || v < tmin {
				tmin = v
			}
			if i == 0 || v > tmax {
				tmax = v
			}
			if ti[i] < 0 || int(ti[i]) >= h.vocab {
				tneg++
				if firstBad < 0 {
					firstBad = i
				}
			}
		}
		d += core.Sprintf("\n  tiles=%d NaN=%d badIdx=%d min=%.4g max=%.4g firstBad=%d  tile0=(%.4g,%d) tileLast=(%.4g,%d)",
			tileCount, tnan, tneg, tmin, tmax, firstBad, tv[0], ti[0], tv[tileCount-1], ti[tileCount-1])
	}
	return d
}

func (h *headEncoder) logitsSampleUsable() bool {
	if h.finalNorm.buf == nil || h.weight.buf == nil || !logitsSampleBF16Usable(h.vocab) {
		return false
	}
	if h.quant {
		return h.scales.buf != nil && h.biases.buf != nil
	}
	return true
}

func (h *headEncoder) logitsBufferSampleUsable() bool {
	return h != nil && logitsSampleBF16Usable(h.vocab)
}

func (h *headEncoder) sampleLogitsToken(hidden []byte, params model.SampleParams, draw float32, history []int32) (token int32, ok bool, err error) {
	withAutoreleasePool(func() {
		token, ok, err = h.sampleLogitsTokenInPool(hidden, params, draw, history)
	})
	return token, ok, err
}

func (h *headEncoder) sampleLogitsTokenInPool(hidden []byte, params model.SampleParams, draw float32, history []int32) (token int32, ok bool, err error) {
	if len(hidden) != h.dModel*bf16Size {
		return 0, true, core.NewError("native.headEncoder.sampleLogitsToken: hidden must be dModel bf16 bytes")
	}
	if !h.logitsSampleUsable() {
		return 0, false, nil
	}
	token = -1
	hiddenScratch, hiddenBuf, err := h.hiddenBuffer(hidden)
	if err != nil {
		return 0, true, err
	}
	defer h.putHiddenScratch(hiddenScratch)
	return h.sampleLogitsTokenBufferInPool(hiddenBuf, params, draw, history)
}

func (h *headEncoder) sampleLogitsTokenBufferInPool(hiddenBuf metal.MTLBuffer, params model.SampleParams, draw float32, history []int32) (token int32, ok bool, err error) {
	return h.sampleLogitsTokenBufferAtInPool(hiddenBuf, 0, params, draw, history)
}

func (h *headEncoder) sampleLogitsTokenBufferAtInPool(hiddenBuf metal.MTLBuffer, hiddenOff uint, params model.SampleParams, draw float32, history []int32) (token int32, ok bool, err error) {
	if hiddenBuf == nil {
		return 0, true, core.NewError("native.headEncoder.sampleLogitsToken: missing hidden buffer")
	}
	if !h.hiddenBufferOffsetInRange(hiddenBuf, hiddenOff) {
		return 0, true, core.NewError("native.headEncoder.sampleLogitsToken: hidden offset is out of range")
	}
	if !h.logitsSampleUsable() {
		return 0, false, nil
	}
	token = -1
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	scratch, ok, err := h.encodeLogitsSampleObjectAt(enc, hiddenBuf, hiddenOff, params, draw, history)
	if !ok || err != nil {
		endEncodingFast(enc)
		if scratch != nil {
			h.putGreedyScratch(scratch)
		}
		if err != nil {
			return 0, true, err
		}
		return 0, false, nil
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	token = scratch.token()
	h.putGreedyScratch(scratch)
	if token < 0 || int(token) >= h.vocab {
		return 0, true, core.NewError(core.Sprintf("native.headEncoder.sampleLogitsToken: sampled invalid token %d for vocab %d", token, h.vocab))
	}
	return token, true, nil
}

func (h *headEncoder) sampleLogitsBufferInPool(logitsBuf metal.MTLBuffer, params model.SampleParams, draw float32, history []int32) (token int32, ok bool, err error) {
	if logitsBuf == nil {
		return 0, true, core.NewError("native.headEncoder.sampleLogitsBuffer: missing logits buffer")
	}
	if !h.logitsBufferSampleUsable() {
		return 0, false, nil
	}
	token = -1
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	scratch := h.getGreedyScratch(1, false, false)
	suppressBuf := scratch.suppressBuffer(params.SuppressTokens)
	historyBuf := scratch.historyBuffer(history)
	err = encLogitsSampleBF16Object(enc, logitsBuf, suppressBuf, historyBuf, scratch.outToken, scratch.logitsSampleParamsBuffer(params, draw, h.vocab, len(params.SuppressTokens), len(history)))
	if err != nil {
		endEncodingFast(enc)
		h.putGreedyScratch(scratch)
		return 0, true, err
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	token = scratch.token()
	h.putGreedyScratch(scratch)
	if token < 0 || int(token) >= h.vocab {
		return 0, true, core.NewError(core.Sprintf("native.headEncoder.sampleLogitsBuffer: sampled invalid token %d for vocab %d", token, h.vocab))
	}
	return token, true, nil
}

func (h *headEncoder) hiddenBufferOffsetInRange(hiddenBuf metal.MTLBuffer, hiddenOff uint) bool {
	if h == nil || hiddenBuf == nil || h.dModel <= 0 {
		return false
	}
	rowBytes := uint(h.dModel * bf16Size)
	n := bufferLengthFast(hiddenBuf)
	return hiddenOff <= n && rowBytes <= n-hiddenOff
}

func (h *headEncoder) encodeFinalNormObject(enc metal.MTLComputeCommandEncoderObject, hiddenBuf metal.MTLBuffer, hiddenOff uint, normed metal.MTLBuffer) error {
	if hiddenOff == 0 {
		sink := encObjectSink{enc: enc}
		rmsPSO, err := pipelineFor(rmsKernelBF16(h.dModel))
		if err != nil {
			return err
		}
		emitRMSNorm(sink, rmsPSO, hiddenBuf, h.finalNorm.buf, normed, h.finalNorm.off, h.dModel, h.eps, rmsThreadgroup(h.dModel, rmsPSO))
		return nil
	}
	return encRMSNormRowsBF16Object(enc, hiddenBuf, h.finalNorm.buf, normed, hiddenOff, h.finalNorm.off, 0, 1, h.dModel, h.eps)
}

func (h *headEncoder) encodeLogitsSample(enc metal.MTLComputeCommandEncoder, hiddenBuf metal.MTLBuffer, params model.SampleParams, draw float32, history []int32) (scratch *headGreedyScratch, ok bool, err error) {
	return h.encodeLogitsSampleAt(enc, hiddenBuf, 0, params, draw, history)
}

func (h *headEncoder) encodeLogitsSampleAt(enc metal.MTLComputeCommandEncoder, hiddenBuf metal.MTLBuffer, hiddenOff uint, params model.SampleParams, draw float32, history []int32) (scratch *headGreedyScratch, ok bool, err error) {
	if !h.logitsSampleUsable() {
		return nil, false, nil
	}
	scratch = h.getGreedyScratch(1, true, h.softCap > 0 && h.vocab > 0)
	normed := scratch.normed
	logits := scratch.logits
	if hiddenOff == 0 {
		err = encRMSNormBF16(enc, hiddenBuf, h.finalNorm.buf, normed, h.finalNorm.off, h.dModel, h.eps)
	} else {
		err = encRMSNormRowsBF16(enc, hiddenBuf, h.finalNorm.buf, normed, hiddenOff, h.finalNorm.off, 0, 1, h.dModel, h.eps)
	}
	if err != nil {
		return scratch, true, err
	}
	if h.quant {
		if err = encQMVBF16(enc, h.weight.buf, h.scales.buf, h.biases.buf, normed, logits,
			h.weight.off, h.scales.off, h.biases.off, 0, h.vocab, h.dModel, h.groupSize, h.bits); err != nil {
			return scratch, true, err
		}
	} else {
		if err = encGemvBF16To(enc, h.weight.buf, normed, logits, h.weight.off, 0, h.vocab, h.dModel); err != nil {
			return scratch, true, err
		}
	}
	if h.softCap > 0 && h.vocab > 0 {
		invBytes := bf16ScalarBytes(1 / h.softCap)
		invScale := h.invSoftCapScale
		if invScale.buf == nil {
			invScale = bufView{buf: bf16ConstBuffer(1, 1/h.softCap)}
		}
		if err = encScaleBF16(enc, logits, invScale.buf, scratch.softcapA, invScale.off, invBytes[:], h.vocab); err != nil {
			return scratch, true, err
		}
		if err = encTanhBF16(enc, scratch.softcapA, scratch.softcapB, h.vocab); err != nil {
			return scratch, true, err
		}
		capBytes := bf16ScalarBytes(h.softCap)
		capScale := h.softCapScale
		if capScale.buf == nil {
			capScale = bufView{buf: bf16ConstBuffer(1, h.softCap)}
		}
		if err = encScaleBF16(enc, scratch.softcapB, capScale.buf, logits, capScale.off, capBytes[:], h.vocab); err != nil {
			return scratch, true, err
		}
	}
	suppressBuf := scratch.suppressBuffer(params.SuppressTokens)
	historyBuf := scratch.historyBuffer(history)
	if err = encLogitsSampleBF16(enc, logits, suppressBuf, historyBuf, scratch.outToken, scratch.logitsSampleParamsBuffer(params, draw, h.vocab, len(params.SuppressTokens), len(history))); err != nil {
		return scratch, true, err
	}
	return scratch, true, nil
}

func (h *headEncoder) encodeLogitsSampleObject(enc metal.MTLComputeCommandEncoderObject, hiddenBuf metal.MTLBuffer, params model.SampleParams, draw float32, history []int32) (scratch *headGreedyScratch, ok bool, err error) {
	return h.encodeLogitsSampleObjectAt(enc, hiddenBuf, 0, params, draw, history)
}

func (h *headEncoder) encodeLogitsSampleObjectAt(enc metal.MTLComputeCommandEncoderObject, hiddenBuf metal.MTLBuffer, hiddenOff uint, params model.SampleParams, draw float32, history []int32) (scratch *headGreedyScratch, ok bool, err error) {
	if !h.logitsSampleUsable() {
		return nil, false, nil
	}
	scratch = h.getGreedyScratch(1, true, h.softCap > 0 && h.vocab > 0)
	normed := scratch.normed
	logits := scratch.logits
	sink := encObjectSink{enc: enc}
	if err = h.encodeFinalNormObject(enc, hiddenBuf, hiddenOff, normed); err != nil {
		return scratch, true, err
	}
	if h.quant {
		qmvPSO, err := pipelineFor(qmvBF16KernelName(h.vocab, h.dModel, h.groupSize, h.bits))
		if err != nil {
			return scratch, true, err
		}
		emitQMV(sink, qmvPSO, h.weight.buf, h.weight.off, h.scales.buf, h.scales.off, h.biases.buf, h.biases.off, normed, logits, 0, h.dModel, h.vocab)
	} else {
		bm, bn, sm, sn, tm, tn := gemvTiles(h.dModel, h.vocab)
		gemvPSO, err := pipelineFor(gemvKernelName("bfloat16", bm, bn, sm, sn, tm, tn))
		if err != nil {
			return scratch, true, err
		}
		emitGemv(sink, gemvPSO, h.weight.buf, h.weight.off, normed, logits, 0, h.dModel, h.vocab, bm, bn, sm, tm)
	}
	if h.softCap > 0 && h.vocab > 0 {
		invBytes := bf16ScalarBytes(1 / h.softCap)
		invScale := h.invSoftCapScale
		if invScale.buf == nil {
			invScale = bufView{buf: bf16ConstBuffer(1, 1/h.softCap)}
		}
		if err = encScaleBF16Object(enc, logits, invScale.buf, scratch.softcapA, invScale.off, invBytes[:], h.vocab); err != nil {
			return scratch, true, err
		}
		if err = encTanhBF16Object(enc, scratch.softcapA, scratch.softcapB, h.vocab); err != nil {
			return scratch, true, err
		}
		capBytes := bf16ScalarBytes(h.softCap)
		capScale := h.softCapScale
		if capScale.buf == nil {
			capScale = bufView{buf: bf16ConstBuffer(1, h.softCap)}
		}
		if err = encScaleBF16Object(enc, scratch.softcapB, capScale.buf, logits, capScale.off, capBytes[:], h.vocab); err != nil {
			return scratch, true, err
		}
	}
	suppressBuf := scratch.suppressBuffer(params.SuppressTokens)
	historyBuf := scratch.historyBuffer(history)
	if err = encLogitsSampleBF16Object(enc, logits, suppressBuf, historyBuf, scratch.outToken, scratch.logitsSampleParamsBuffer(params, draw, h.vocab, len(params.SuppressTokens), len(history))); err != nil {
		return scratch, true, err
	}
	return scratch, true, nil
}

func (h *headEncoder) sampleTopKCandidates(hidden []byte, topK int, suppress []int32) (logits []byte, ids []int32, ok bool, err error) {
	return h.sampleTopKCandidatesInto(hidden, topK, suppress, nil, nil, false)
}

func (h *headEncoder) sampleTopKCandidatesFusedQ4(hidden []byte, topK int, suppress []int32) (logits []byte, ids []int32, ok bool, err error) {
	return h.sampleTopKCandidatesInto(hidden, topK, suppress, nil, nil, true)
}

func (h *headEncoder) topKSampleUsable(topK int) bool {
	if h.finalNorm.buf == nil || h.weight.buf == nil || !topKSampleUsable(topK) {
		return false
	}
	if h.quant {
		if h.scales.buf == nil || h.biases.buf == nil {
			return false
		}
		return q4LMHeadTopKUsable(h.dModel, h.vocab, h.groupSize, h.bits, topK) ||
			qmvLogitsTopKUsable(h.dModel, h.vocab, h.groupSize, h.bits, topK)
	}
	return bf16LMHeadTopKUsable(h.dModel, h.vocab, topK)
}

func (h *headEncoder) preferFusedQ4TopK(topK int) bool {
	return h != nil && h.quant && q4LMHeadTopKUsable(h.dModel, h.vocab, h.groupSize, h.bits, topK)
}

func (h *headEncoder) sampleTopKToken(hidden []byte, params model.SampleParams, draw float32, history []int32) (token int32, ok bool, err error) {
	withAutoreleasePool(func() {
		token, ok, err = h.sampleTopKTokenInPool(hidden, params, draw, history)
	})
	return token, ok, err
}

func (h *headEncoder) sampleTopKTokenInPool(hidden []byte, params model.SampleParams, draw float32, history []int32) (token int32, ok bool, err error) {
	if len(hidden) != h.dModel*bf16Size {
		return 0, true, core.NewError("native.headEncoder.sampleTopKToken: hidden must be dModel bf16 bytes")
	}
	if !h.topKSampleUsable(params.TopK) {
		return 0, false, nil
	}
	hiddenScratch, hiddenBuf, err := h.hiddenBuffer(hidden)
	if err != nil {
		return 0, true, err
	}
	defer h.putHiddenScratch(hiddenScratch)
	return h.sampleTopKTokenBufferInPool(hiddenBuf, params, draw, history)
}

func (h *headEncoder) sampleTopKTokenBufferInPool(hiddenBuf metal.MTLBuffer, params model.SampleParams, draw float32, history []int32) (token int32, ok bool, err error) {
	return h.sampleTopKTokenBufferAtInPool(hiddenBuf, 0, params, draw, history)
}

func (h *headEncoder) sampleTopKTokenBufferAtInPool(hiddenBuf metal.MTLBuffer, hiddenOff uint, params model.SampleParams, draw float32, history []int32) (token int32, ok bool, err error) {
	if hiddenBuf == nil {
		return 0, true, core.NewError("native.headEncoder.sampleTopKToken: missing hidden buffer")
	}
	if !h.hiddenBufferOffsetInRange(hiddenBuf, hiddenOff) {
		return 0, true, core.NewError("native.headEncoder.sampleTopKToken: hidden offset is out of range")
	}
	if !h.topKSampleUsable(params.TopK) {
		return 0, false, nil
	}
	token = -1
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	var scratch *headTopKScratch
	scratch, ok, err = h.encodeTopKSampleAtFast(enc, hiddenBuf, hiddenOff, params, draw, history)
	if !ok || err != nil {
		endEncodingFast(enc)
		if scratch != nil {
			h.putTopKScratch(scratch)
		}
		if err != nil {
			return 0, true, err
		}
		return 0, false, nil
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	token = scratch.token()
	h.putTopKScratch(scratch)
	if token < 0 || int(token) >= h.vocab {
		return 0, true, core.NewError(core.Sprintf("native.headEncoder.sampleTopKToken: sampled invalid token %d for vocab %d", token, h.vocab))
	}
	return token, true, nil
}

func (h *headEncoder) sampleTopKCandidatesInto(hidden []byte, topK int, suppress []int32, outLogits []byte, outIDs []int32, preferFusedQ4 bool) (logits []byte, ids []int32, ok bool, err error) {
	return h.sampleTopKCandidatesWithHistoryInto(hidden, topK, suppress, nil, 1, outLogits, outIDs, preferFusedQ4)
}

func (h *headEncoder) sampleTopKCandidatesWithHistoryInto(hidden []byte, topK int, suppress []int32, history []int32, repeatPenalty float32, outLogits []byte, outIDs []int32, preferFusedQ4 bool) (logits []byte, ids []int32, ok bool, err error) {
	if len(hidden) != h.dModel*bf16Size {
		return nil, nil, true, core.NewError("native.headEncoder.sampleTopKCandidates: hidden must be dModel bf16 bytes")
	}
	var scratch *headTopKScratch
	var encErr error
	withAutoreleasePool(func() {
		var hiddenScratch *headHiddenScratch
		var hiddenBuf metal.MTLBuffer
		hiddenScratch, hiddenBuf, encErr = h.hiddenBuffer(hidden)
		if encErr != nil {
			return
		}
		defer h.putHiddenScratch(hiddenScratch)
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		scratch, ok, encErr = h.encodeTopKCandidatesWithHistoryObject(enc, hiddenBuf, topK, suppress, history, repeatPenalty, preferFusedQ4)
		if !ok || encErr != nil {
			endEncodingFast(enc)
			if scratch != nil {
				h.putTopKScratch(scratch)
				scratch = nil
			}
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
	})
	if encErr != nil {
		return nil, nil, true, encErr
	}
	if !ok {
		return nil, nil, false, nil
	}
	defer h.putTopKScratch(scratch)
	return h.readTopKCandidatesInto(scratch, topK, outLogits, outIDs)
}

func (h *headEncoder) sampleTopKCandidatesBufferInto(hiddenBuf metal.MTLBuffer, topK int, suppress []int32, outLogits []byte, outIDs []int32, preferFusedQ4 bool) (logits []byte, ids []int32, ok bool, err error) {
	return h.sampleTopKCandidatesBufferWithHistoryInto(hiddenBuf, topK, suppress, nil, 1, outLogits, outIDs, preferFusedQ4)
}

func (h *headEncoder) sampleTopKCandidatesBufferWithHistoryInto(hiddenBuf metal.MTLBuffer, topK int, suppress []int32, history []int32, repeatPenalty float32, outLogits []byte, outIDs []int32, preferFusedQ4 bool) (logits []byte, ids []int32, ok bool, err error) {
	if hiddenBuf == nil {
		return nil, nil, true, core.NewError("native.headEncoder.sampleTopKCandidates: missing hidden buffer")
	}
	var scratch *headTopKScratch
	var encErr error
	withAutoreleasePool(func() {
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		scratch, ok, encErr = h.encodeTopKCandidatesWithHistoryObject(enc, hiddenBuf, topK, suppress, history, repeatPenalty, preferFusedQ4)
		if !ok || encErr != nil {
			endEncodingFast(enc)
			if scratch != nil {
				h.putTopKScratch(scratch)
				scratch = nil
			}
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
	})
	if encErr != nil {
		return nil, nil, true, encErr
	}
	if !ok {
		return nil, nil, false, nil
	}
	defer h.putTopKScratch(scratch)
	return h.readTopKCandidatesInto(scratch, topK, outLogits, outIDs)
}

func (h *headEncoder) encodeTopKSampleFast(enc metal.MTLComputeCommandEncoderObject, hiddenBuf metal.MTLBuffer, params model.SampleParams, draw float32, history []int32) (scratch *headTopKScratch, ok bool, err error) {
	return h.encodeTopKSampleAtFast(enc, hiddenBuf, 0, params, draw, history)
}

func (h *headEncoder) encodeTopKSampleAtFast(enc metal.MTLComputeCommandEncoderObject, hiddenBuf metal.MTLBuffer, hiddenOff uint, params model.SampleParams, draw float32, history []int32) (scratch *headTopKScratch, ok bool, err error) {
	preferFusedQ4 := h.preferFusedQ4TopK(params.TopK)
	return h.encodeTopKSampleObjectAt(enc, hiddenBuf, hiddenOff, params, draw, history, preferFusedQ4)
}

func (h *headEncoder) encodeTopKSample(enc metal.MTLComputeCommandEncoder, hiddenBuf metal.MTLBuffer, params model.SampleParams, draw float32, history []int32, preferFusedQ4 bool) (scratch *headTopKScratch, ok bool, err error) {
	var candidateCount int
	scratch, candidateCount, ok, err = h.encodeTopKCandidateRows(enc, hiddenBuf, params.TopK, params.SuppressTokens, history, params.RepeatPenalty, preferFusedQ4)
	if !ok || err != nil {
		return scratch, ok, err
	}
	if err = encTopKMergeSampleF32(enc, scratch.candidateValues, scratch.candidateIndices, scratch.outToken, scratch.sampleParamsBuffer(params, draw, candidateCount)); err != nil {
		return scratch, true, err
	}
	return scratch, true, nil
}

func (h *headEncoder) encodeTopKSampleObject(enc metal.MTLComputeCommandEncoderObject, hiddenBuf metal.MTLBuffer, params model.SampleParams, draw float32, history []int32, preferFusedQ4 bool) (scratch *headTopKScratch, ok bool, err error) {
	return h.encodeTopKSampleObjectAt(enc, hiddenBuf, 0, params, draw, history, preferFusedQ4)
}

func (h *headEncoder) encodeTopKSampleObjectAt(enc metal.MTLComputeCommandEncoderObject, hiddenBuf metal.MTLBuffer, hiddenOff uint, params model.SampleParams, draw float32, history []int32, preferFusedQ4 bool) (scratch *headTopKScratch, ok bool, err error) {
	var candidateCount int
	scratch, candidateCount, ok, err = h.encodeTopKCandidateRowsObjectAt(enc, hiddenBuf, hiddenOff, params.TopK, params.SuppressTokens, history, params.RepeatPenalty, preferFusedQ4)
	if !ok || err != nil {
		return scratch, ok, err
	}
	if err = encTopKMergeSampleF32Object(enc, scratch.candidateValues, scratch.candidateIndices, scratch.outToken, scratch.sampleParamsBuffer(params, draw, candidateCount)); err != nil {
		return scratch, true, err
	}
	return scratch, true, nil
}

func (h *headEncoder) encodeTopKCandidatesWithHistoryFast(enc metal.MTLComputeCommandEncoderObject, hiddenBuf metal.MTLBuffer, topK int, suppress []int32, history []int32, repeatPenalty float32) (scratch *headTopKScratch, ok bool, err error) {
	preferFusedQ4 := h.preferFusedQ4TopK(topK)
	return h.encodeTopKCandidatesWithHistoryObject(enc, hiddenBuf, topK, suppress, history, repeatPenalty, preferFusedQ4)
}

func (h *headEncoder) encodeTopKCandidates(enc metal.MTLComputeCommandEncoder, hiddenBuf metal.MTLBuffer, topK int, suppress []int32, preferFusedQ4 bool) (scratch *headTopKScratch, ok bool, err error) {
	return h.encodeTopKCandidatesWithHistory(enc, hiddenBuf, topK, suppress, nil, 1, preferFusedQ4)
}

func (h *headEncoder) encodeTopKCandidatesWithHistory(enc metal.MTLComputeCommandEncoder, hiddenBuf metal.MTLBuffer, topK int, suppress []int32, history []int32, repeatPenalty float32, preferFusedQ4 bool) (scratch *headTopKScratch, ok bool, err error) {
	var candidateCount int
	scratch, candidateCount, ok, err = h.encodeTopKCandidateRows(enc, hiddenBuf, topK, suppress, history, repeatPenalty, preferFusedQ4)
	if !ok || err != nil {
		return scratch, ok, err
	}
	if err = encTopKMergeF32(enc, scratch.candidateValues, scratch.candidateIndices, scratch.topValues, scratch.topIndices, candidateCount, topK); err != nil {
		return scratch, true, err
	}
	return scratch, true, nil
}

func (h *headEncoder) encodeTopKCandidatesWithHistoryObject(enc metal.MTLComputeCommandEncoderObject, hiddenBuf metal.MTLBuffer, topK int, suppress []int32, history []int32, repeatPenalty float32, preferFusedQ4 bool) (scratch *headTopKScratch, ok bool, err error) {
	var candidateCount int
	scratch, candidateCount, ok, err = h.encodeTopKCandidateRowsObject(enc, hiddenBuf, topK, suppress, history, repeatPenalty, preferFusedQ4)
	if !ok || err != nil {
		return scratch, ok, err
	}
	if err = encTopKMergeF32Object(enc, scratch.candidateValues, scratch.candidateIndices, scratch.topValues, scratch.topIndices, candidateCount, topK); err != nil {
		return scratch, true, err
	}
	return scratch, true, nil
}

func (h *headEncoder) encodeTopKCandidateRowsObject(enc metal.MTLComputeCommandEncoderObject, hiddenBuf metal.MTLBuffer, topK int, suppress []int32, history []int32, repeatPenalty float32, preferFusedQ4 bool) (scratch *headTopKScratch, candidateCount int, ok bool, err error) {
	return h.encodeTopKCandidateRowsObjectAt(enc, hiddenBuf, 0, topK, suppress, history, repeatPenalty, preferFusedQ4)
}

func (h *headEncoder) encodeTopKCandidateRowsObjectAt(enc metal.MTLComputeCommandEncoderObject, hiddenBuf metal.MTLBuffer, hiddenOff uint, topK int, suppress []int32, history []int32, repeatPenalty float32, preferFusedQ4 bool) (scratch *headTopKScratch, candidateCount int, ok bool, err error) {
	if h.finalNorm.buf == nil || h.weight.buf == nil || topK <= 0 || topK > headSampleTopKMaxK || topK > h.vocab {
		return nil, 0, false, nil
	}
	needLogits := false
	fusedQuantTopK := false
	fusedCandidatesPerTile := topK
	candidateCount = h.vocab
	if h.quant {
		if h.scales.buf == nil || h.biases.buf == nil {
			return nil, 0, false, nil
		}
		q4Usable := q4LMHeadTopKUsable(h.dModel, h.vocab, h.groupSize, h.bits, topK)
		fusedQuantTopK = preferFusedQ4 && q4Usable
		needLogits = true
		if fusedQuantTopK {
			needLogits = false
			fusedCandidatesPerTile = q4LMHeadTopKCandidatesPerTile(topK)
			candidateCount = q4LMHeadTopKCandidateCount(h.vocab, topK)
		} else {
			qmvUsable := qmvLogitsTopKUsable(h.dModel, h.vocab, h.groupSize, h.bits, topK)
			if qmvUsable {
				candidateCount = ((h.vocab + bf16LogitsArgmaxRowsPerTile - 1) / bf16LogitsArgmaxRowsPerTile) * topK
			} else if q4Usable {
				needLogits = false
				fusedQuantTopK = true
				fusedCandidatesPerTile = q4LMHeadTopKCandidatesPerTile(topK)
				candidateCount = q4LMHeadTopKCandidateCount(h.vocab, topK)
			} else {
				return nil, 0, false, nil
			}
		}
	} else {
		if !bf16LMHeadTopKUsable(h.dModel, h.vocab, topK) {
			return nil, 0, false, nil
		}
		candidateCount = ((h.vocab + bf16LMHeadArgmaxRowsPerTile - 1) / bf16LMHeadArgmaxRowsPerTile) * bf16LMHeadArgmaxRowsPerTile
	}
	scratch = h.getTopKScratch(candidateCount, topK, needLogits)
	normed := scratch.normed
	suppressBuf := scratch.suppressBuffer(suppress)
	historyBuf := scratch.historyBuffer(history)
	historyCount := len(history)
	sink := encObjectSink{enc: enc}
	if err = h.encodeFinalNormObject(enc, hiddenBuf, hiddenOff, normed); err != nil {
		return scratch, candidateCount, true, err
	}
	if h.quant {
		if fusedQuantTopK {
			if err = encQ4LMHeadTopKTilesBF16Object(enc, normed, h.weight.buf, h.scales.buf, h.biases.buf,
				scratch.candidateValues, scratch.candidateIndices, suppressBuf, historyBuf,
				0, h.weight.off, h.scales.off, h.biases.off,
				h.dModel, h.vocab, h.groupSize, len(suppress), historyCount, topK, fusedCandidatesPerTile, repeatPenalty, h.softCap); err != nil {
				return scratch, candidateCount, true, err
			}
		} else {
			qmvPSO, err := pipelineFor(qmvBF16KernelName(h.vocab, h.dModel, h.groupSize, h.bits))
			if err != nil {
				return scratch, candidateCount, true, err
			}
			emitQMV(sink, qmvPSO, h.weight.buf, h.weight.off, h.scales.buf, h.scales.off, h.biases.buf, h.biases.off, normed, scratch.logits, 0, h.dModel, h.vocab)
			if err = encBF16LogitsTopKTilesBF16Object(enc, scratch.logits, scratch.candidateValues, scratch.candidateIndices, suppressBuf, historyBuf, h.vocab, len(suppress), historyCount, topK, repeatPenalty, h.softCap); err != nil {
				return scratch, candidateCount, true, err
			}
		}
	} else {
		if err = encBF16LMHeadCandidatesBF16Object(enc, normed, h.weight.buf, scratch.candidateValues, scratch.candidateIndices, suppressBuf, historyBuf, 0, h.weight.off, h.dModel, h.vocab, len(suppress), historyCount, repeatPenalty, h.softCap); err != nil {
			return scratch, candidateCount, true, err
		}
	}
	return scratch, candidateCount, true, nil
}

func (h *headEncoder) encodeTopKCandidateRows(enc metal.MTLComputeCommandEncoder, hiddenBuf metal.MTLBuffer, topK int, suppress []int32, history []int32, repeatPenalty float32, preferFusedQ4 bool) (scratch *headTopKScratch, candidateCount int, ok bool, err error) {
	if h.finalNorm.buf == nil || h.weight.buf == nil || topK <= 0 || topK > headSampleTopKMaxK || topK > h.vocab {
		return nil, 0, false, nil
	}
	needLogits := false
	fusedQuantTopK := false
	fusedCandidatesPerTile := topK
	candidateCount = h.vocab
	if h.quant {
		if h.scales.buf == nil || h.biases.buf == nil {
			return nil, 0, false, nil
		}
		q4Usable := q4LMHeadTopKUsable(h.dModel, h.vocab, h.groupSize, h.bits, topK)
		if preferFusedQ4 && q4Usable {
			fusedQuantTopK = true
			fusedCandidatesPerTile = q4LMHeadTopKCandidatesPerTile(topK)
			candidateCount = q4LMHeadTopKCandidateCount(h.vocab, topK)
		} else {
			qmvUsable := qmvLogitsTopKUsable(h.dModel, h.vocab, h.groupSize, h.bits, topK)
			if qmvUsable {
				needLogits = true
				candidateCount = ((h.vocab + bf16LogitsArgmaxRowsPerTile - 1) / bf16LogitsArgmaxRowsPerTile) * topK
			} else if q4Usable {
				fusedQuantTopK = true
				fusedCandidatesPerTile = q4LMHeadTopKCandidatesPerTile(topK)
				candidateCount = q4LMHeadTopKCandidateCount(h.vocab, topK)
			} else {
				return nil, 0, false, nil
			}
		}
	} else {
		if !bf16LMHeadTopKUsable(h.dModel, h.vocab, topK) {
			return nil, 0, false, nil
		}
		candidateCount = ((h.vocab + bf16LMHeadArgmaxRowsPerTile - 1) / bf16LMHeadArgmaxRowsPerTile) * bf16LMHeadArgmaxRowsPerTile
	}

	scratch = h.getTopKScratch(candidateCount, topK, needLogits)
	normed := scratch.normed
	suppressBuf := scratch.suppressBuffer(suppress)
	historyBuf := scratch.historyBuffer(history)
	historyCount := len(history)
	if err = encRMSNormBF16(enc, hiddenBuf, h.finalNorm.buf, normed, h.finalNorm.off, h.dModel, h.eps); err != nil {
		return scratch, candidateCount, true, err
	}
	if h.quant {
		if fusedQuantTopK {
			if err = encQ4LMHeadTopKTilesBF16(enc, normed, h.weight.buf, h.scales.buf, h.biases.buf,
				scratch.candidateValues, scratch.candidateIndices, suppressBuf, historyBuf,
				0, h.weight.off, h.scales.off, h.biases.off,
				h.dModel, h.vocab, h.groupSize, len(suppress), historyCount, topK, fusedCandidatesPerTile, repeatPenalty, h.softCap); err != nil {
				return scratch, candidateCount, true, err
			}
		} else {
			if err = encQMVBF16(enc, h.weight.buf, h.scales.buf, h.biases.buf, normed, scratch.logits,
				h.weight.off, h.scales.off, h.biases.off, 0, h.vocab, h.dModel, h.groupSize, h.bits); err != nil {
				return scratch, candidateCount, true, err
			}
			if err = encBF16LogitsTopKTilesBF16(enc, scratch.logits, scratch.candidateValues, scratch.candidateIndices, suppressBuf, historyBuf, h.vocab, len(suppress), historyCount, topK, repeatPenalty, h.softCap); err != nil {
				return scratch, candidateCount, true, err
			}
		}
	} else {
		if err = encBF16LMHeadCandidatesBF16(enc, normed, h.weight.buf, scratch.candidateValues, scratch.candidateIndices, suppressBuf, historyBuf, 0, h.weight.off, h.dModel, h.vocab, len(suppress), historyCount, repeatPenalty, h.softCap); err != nil {
			return scratch, candidateCount, true, err
		}
	}
	return scratch, candidateCount, true, nil
}

func (h *headEncoder) readTopKCandidates(scratch *headTopKScratch, topK int) (logits []byte, ids []int32, ok bool, err error) {
	return h.readTopKCandidatesInto(scratch, topK, nil, nil)
}

func (h *headEncoder) readTopKCandidatesInto(scratch *headTopKScratch, topK int, outLogits []byte, outIDs []int32) (logits []byte, ids []int32, ok bool, err error) {
	if scratch == nil || scratch.topValues == nil || scratch.topValuesPtr == nil || scratch.topIndices == nil || scratch.topIndicesPtr == nil {
		return nil, nil, true, core.NewError("native.headEncoder.sampleTopKCandidates: missing top-k scratch")
	}
	if cap(outLogits) < topK*bf16Size {
		outLogits = make([]byte, topK*bf16Size)
	} else {
		outLogits = outLogits[:topK*bf16Size]
	}
	if cap(outIDs) < topK {
		outIDs = make([]int32, 0, topK)
	} else {
		outIDs = outIDs[:0]
	}
	values := unsafe.Slice(scratch.topValuesPtr, topK)
	topIDs := unsafe.Slice(scratch.topIndicesPtr, topK)
	valid := 0
	for i, id := range topIDs {
		if id < 0 || int(id) >= h.vocab {
			continue
		}
		b := f32ToBF16(values[i])
		outLogits[valid*bf16Size] = byte(b)
		outLogits[valid*bf16Size+1] = byte(b >> 8)
		outIDs = append(outIDs, id)
		valid++
	}
	if len(outIDs) == 0 {
		return nil, nil, true, core.NewError("native.headEncoder.sampleTopKCandidates: no unsuppressed candidates")
	}
	return outLogits[:valid*bf16Size], outIDs, true, nil
}

// greedyRowsBufferInPool encodes K fused lm_head+argmax chains — one per
// hidden row at rowStride intervals inside rowsBuf — into a SINGLE command
// buffer with a single wait, writing the argmax token ids into out. The MTP
// verify previously ran this per row (norm+head+argmax+commit+wait each,
// ~2ms/row of mostly synchronisation); one buffer collapses the tax to one
// wait. ok=false defers to the caller's per-row fallback (direct greedy not
// usable for this head).
func (h *headEncoder) greedyRowsBufferInPool(rowsBuf metal.MTLBuffer, rowStride uint, k int, suppress []int32, out []int32) (bool, error) {
	if h == nil || rowsBuf == nil || k <= 0 || len(out) < k {
		return false, core.NewError("native.headEncoder.greedyRows: invalid batch")
	}
	if !h.hiddenBufferOffsetInRange(rowsBuf, uint(k-1)*rowStride) {
		return false, core.NewError("native.headEncoder.greedyRows: rows exceed the hidden buffer")
	}
	// bf16 heads with no suppression take the fused K-row kernel: one weight
	// pass scores every vocab tile against all K rows (the per-row path re-read
	// the full lm_head weight K times — its dominant cost). Falls through to
	// the per-row encodes when the kernel is unavailable (older metallib).
	if !h.quant && len(suppress) == 0 && k > 1 && k <= 8 && int(rowStride) == h.dModel*bf16Size {
		if handled, err := h.greedyRowsFusedInPool(rowsBuf, k, out); handled || err != nil {
			return handled && err == nil, err
		}
	}
	// quant heads batch the same way through the qmm_t the verify fold already
	// runs on (token-identity tier): ONE weight pass produces all K logits
	// rows, then per-row tile argmax + one rows merge — the per-row path's K
	// full qmv weight sweeps were the verify's rows-head cost (#359).
	if h.quant && len(suppress) == 0 && k > 1 && k <= 8 && int(rowStride) == h.dModel*bf16Size {
		if handled, err := h.greedyRowsQuantFusedInPool(rowsBuf, k, out); handled || err != nil {
			return handled && err == nil, err
		}
	}
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	scratches := make([]*headGreedyScratch, 0, k)
	release := func() {
		for _, s := range scratches {
			h.putGreedyScratch(s)
		}
	}
	for i := range k {
		scratch, ok, err := h.encodeGreedyAt(enc, rowsBuf, uint(i)*rowStride, suppress)
		if scratch != nil {
			scratches = append(scratches, scratch)
		}
		if err != nil || !ok {
			endEncodingFast(enc)
			release()
			return false, err
		}
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	for i, scratch := range scratches {
		token := scratch.token()
		if token < 0 || int(token) >= h.vocab {
			release()
			return false, core.NewError(core.Sprintf("native.headEncoder.greedyRows: row %d argmax returned invalid token %d for vocab %d", i, token, h.vocab))
		}
		out[i] = token
	}
	release()
	return true, nil
}

// greedyRowsQuantFusedInPool is the quant head's K-row fused greedy: one
// RMSNorm over all K rows, ONE qmm_t sweeping the quant lm_head once to
// produce all K logits rows (the same kernel + accuracy tier the MTP verify
// fold runs its projections on), then a per-row logits tile argmax and one
// rows merge — a single command buffer. handled=false defers to the per-row
// encodes (qmm pipeline unavailable, or scratch allocation failed).
func (h *headEncoder) greedyRowsQuantFusedInPool(rowsBuf metal.MTLBuffer, k int, out []int32) (handled bool, err error) {
	if _, perr := pipelineFor(qmmTKernelName(h.vocab, h.groupSize, h.bits)); perr != nil {
		return false, nil
	}
	tileCount := (h.vocab + bf16LogitsArgmaxRowsPerTile - 1) / bf16LogitsArgmaxRowsPerTile
	s := h.getGreedyRowsScratchLogits(k, tileCount, true)
	if s == nil {
		return false, nil
	}
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	fail := func(ferr error) (bool, error) {
		endEncodingFast(enc)
		h.putGreedyRowsScratch(s)
		return true, ferr
	}
	if ferr := encRMSNormRowsBF16(enc, rowsBuf, h.finalNorm.buf, s.normed, 0, h.finalNorm.off, 0, k, h.dModel, h.eps); ferr != nil {
		return fail(ferr)
	}
	if ferr := encQMMTBF16At(enc, h.weight.buf, h.scales.buf, h.biases.buf, s.normed, s.logitsRows,
		h.weight.off, h.scales.off, h.biases.off, 0, 0, k, h.vocab, h.dModel, h.groupSize, h.bits); ferr != nil {
		return fail(ferr)
	}
	for i := range k {
		if ferr := encBF16LogitsArgmaxTilesBF16At(enc, s.logitsRows, s.tileValues, s.tileIndices, nil,
			uint(i*h.vocab*bf16Size), uint(i*tileCount*4), uint(i*tileCount*4), h.vocab, 0); ferr != nil {
			return fail(ferr)
		}
	}
	if ferr := encArgmaxMergeRowsF32(enc, s.tileValues, s.tileIndices, s.outTokens, tileCount, k); ferr != nil {
		return fail(ferr)
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	toks := unsafe.Slice((*int32)(s.outTokens.Contents()), k)
	for i := range k {
		token := toks[i]
		if token < 0 || int(token) >= h.vocab {
			h.putGreedyRowsScratch(s)
			return true, core.NewError(core.Sprintf("native.headEncoder.greedyRowsQuantFused: row %d argmax returned invalid token %d for vocab %d", i, token, h.vocab))
		}
		out[i] = token
	}
	h.putGreedyRowsScratch(s)
	return true, nil
}

// greedyRowsFusedInPool runs the K-row fused greedy head: one RMSNorm over
// all K rows, ONE weight-pass lm_head+argmax kernel scoring every vocab tile
// against all K rows, and K per-row tile merges — all in a single command
// buffer. handled=false defers to the per-row encodes (kernel missing from an
// older metallib).
func (h *headEncoder) greedyRowsFusedInPool(rowsBuf metal.MTLBuffer, k int, out []int32) (handled bool, err error) {
	if _, perr := bf16LMHeadArgmaxTilesRowsPipeline(k); perr != nil {
		return false, nil
	}
	tileCount := (h.vocab + bf16LMHeadArgmaxRowsPerTile - 1) / bf16LMHeadArgmaxRowsPerTile
	s := h.getGreedyRowsScratch(k, tileCount)
	if s == nil {
		return false, nil
	}
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	fail := func(ferr error) (bool, error) {
		endEncodingFast(enc)
		h.putGreedyRowsScratch(s)
		return true, ferr
	}
	if ferr := encRMSNormRowsBF16(enc, rowsBuf, h.finalNorm.buf, s.normed, 0, h.finalNorm.off, 0, k, h.dModel, h.eps); ferr != nil {
		return fail(ferr)
	}
	if ferr := encBF16LMHeadArgmaxTilesRowsBF16(enc, s.normed, h.weight.buf, s.tileValues, s.tileIndices, nil, 0, h.weight.off, h.dModel, h.vocab, 0, k, tileCount); ferr != nil {
		return fail(ferr)
	}
	if ferr := encArgmaxMergeRowsF32(enc, s.tileValues, s.tileIndices, s.outTokens, tileCount, k); ferr != nil {
		return fail(ferr)
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	toks := unsafe.Slice((*int32)(s.outTokens.Contents()), k)
	for i := range k {
		token := toks[i]
		if token < 0 || int(token) >= h.vocab {
			h.putGreedyRowsScratch(s)
			return true, core.NewError(core.Sprintf("native.headEncoder.greedyRowsFused: row %d argmax returned invalid token %d for vocab %d", i, token, h.vocab))
		}
		out[i] = token
	}
	h.putGreedyRowsScratch(s)
	return true, nil
}
