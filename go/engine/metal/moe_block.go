// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"runtime"
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// MoELayerWeights holds the bf16 weights AND the MoE-specific shape of one gemma4 MoE
// feed-forward block: the five independent RMSNorm weights, the local dense MLP, the
// router, and the experts. Norm weights are dModel bf16. RouterNormWScaled is the
// router's own norm weight ALREADY scaled by RootSize (folded at load like metal's
// cached ScaleScaled — see MoERouter). PerExpertScale is optional (nil to skip). The
// local MLP runs at the model-wide dFF; the experts run at ExpertDFF (gemma4 gives
// them a distinct MoEIntermediateSize). The MoE-specific dims (NumExperts/TopK/
// ExpertDFF) live here so a MoE layer is self-describing — model-wide dModel/dFF/eps
// stay executor parameters shared by dense and MoE layers alike.
type MoELayerWeights struct {
	NumExperts, TopK, ExpertDFF int // MoE shape (model-wide dModel/dFF/eps are args)

	PreFFNormW   []byte // local MLP input norm
	PreFFNorm2W  []byte // expert-branch input norm
	PostFFNorm1W []byte // post local-MLP norm
	PostFFNorm2W []byte // post-expert norm
	PostFFNormW  []byte // final combined-branch norm

	WGate, WUp, WDown []byte // local dense MLP (dFF)

	RouterNormWScaled []byte // router internal norm (pre-scaled by RootSize)
	RouterW           []byte // [NumExperts × dModel] expert-score projection
	PerExpertScale    []byte // [NumExperts] optional (nil to skip)

	ExpGateW, ExpUpW, ExpDownW []byte // experts ([NumExperts × …] at ExpertDFF)
}

// mlpTransformBF16 is the gemma SwiGLU MLP transform on an ALREADY-normed input:
// WDown·(gelu(WGate·x)·(WUp·x)) — no input norm, no residual (the MoE block applies
// those around it). Structurally one expert's computation; composed from the
// parity-proven bf16 ops encoded as one resident sequence. The per-token input is
// transient; the local dense weights are fixed per layer and stay resident like the
// selected expert weights.
func mlpTransformBF16(x, wGate, wUp, wDown []byte, dModel, dFF int) ([]byte, error) {
	return mlpTransformBF16Into(nil, x, wGate, wUp, wDown, dModel, dFF)
}

func mlpTransformBF16Into(out []byte, x, wGate, wUp, wDown []byte, dModel, dFF int) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != dModel*bf16Size {
		return nil, core.NewError("native.mlpTransformBF16: x must be dModel bf16 bytes")
	}
	if len(wGate) != dFF*dModel*bf16Size || len(wUp) != dFF*dModel*bf16Size {
		return nil, core.NewError("native.mlpTransformBF16: wGate/wUp must be dFF*dModel bf16 bytes")
	}
	if len(wDown) != dModel*dFF*bf16Size {
		return nil, core.NewError("native.mlpTransformBF16: wDown must be dModel*dFF bf16 bytes")
	}
	outLen := dModel * bf16Size
	if cap(out) < outLen {
		out = make([]byte, outLen)
	} else {
		out = out[:outLen]
	}
	if dModel == 0 || dFF == 0 {
		clear(out)
		return out, nil
	}

	var encErr error
	withAutoreleasePool(func() {
		scratch, err := getMLPTransformScratch(dModel, dFF)
		if err != nil {
			encErr = err
			return
		}
		defer putMLPTransformScratch(scratch)
		xBuf, ok := scratch.inputView(x)
		if !ok {
			xBuf, err = scratch.x.copyBuffer(x)
			if err != nil {
				encErr = err
				return
			}
		}
		wgBuf, wuBuf, wdBuf := residentBytes(wGate), residentBytes(wUp), residentBytes(wDown)
		msc := scratch.mlp
		outBuf := msc.down
		directOut := false
		if tmp, ok := scratch.outputView(out); ok {
			outBuf = tmp
			directOut = true
		}

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		if encErr = encGemvBF16(enc, wgBuf, xBuf, msc.gate, dFF, dModel); encErr != nil {
			endEncodingFast(enc)
			return
		}
		if encErr = encGemvBF16(enc, wuBuf, xBuf, msc.up, dFF, dModel); encErr != nil {
			endEncodingFast(enc)
			return
		}
		if encErr = encGeluGateMul(enc, msc.gate, msc.up, msc.gated, msc, dFF); encErr != nil {
			endEncodingFast(enc)
			return
		}
		if encErr = encGemvBF16(enc, wdBuf, msc.gated, outBuf, dModel, dFF); encErr != nil {
			endEncodingFast(enc)
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, unsafe.Slice((*byte)(msc.down.Contents()), len(out)))
		}
	})
	return out, encErr
}

type moeBlockPostCombineScratch struct {
	dModel                       int
	h, h1, h2, out               *pinnedNoCopyBytes
	hPinned, h1Pinned, h2Pinned  *pinnedNoCopyBytes
	h1Normed, h2Normed, combined metal.MTLBuffer
	ffResidual                   metal.MTLBuffer
}

type scratchLIFOPool[T any] struct {
	mu    sync.Mutex
	items []T
}

func (p *scratchLIFOPool[T]) Get() T {
	p.mu.Lock()
	defer p.mu.Unlock()
	n := len(p.items)
	if n == 0 {
		var zero T
		return zero
	}
	item := p.items[n-1]
	var zero T
	p.items[n-1] = zero
	p.items = p.items[:n-1]
	return item
}

func (p *scratchLIFOPool[T]) Put(item T) {
	p.mu.Lock()
	p.items = append(p.items, item)
	p.mu.Unlock()
}

var moeBlockPostCombineScratchPools sync.Map

func newMoEBlockPostCombineScratch(dModel int) (*moeBlockPostCombineScratch, error) {
	size := dModel * bf16Size
	h, err := newPinnedNoCopyBytes(size)
	if err != nil {
		return nil, err
	}
	h1, err := newPinnedNoCopyBytes(size)
	if err != nil {
		h.Close()
		return nil, err
	}
	h2, err := newPinnedNoCopyBytes(size)
	if err != nil {
		h.Close()
		h1.Close()
		return nil, err
	}
	out, err := newPinnedNoCopyBytes(size)
	if err != nil {
		h.Close()
		h1.Close()
		h2.Close()
		return nil, err
	}
	return &moeBlockPostCombineScratch{
		dModel:     dModel,
		h:          h,
		h1:         h1,
		h2:         h2,
		out:        out,
		h1Normed:   scratchBF16(dModel),
		h2Normed:   scratchBF16(dModel),
		combined:   scratchBF16(dModel),
		ffResidual: scratchBF16(dModel),
	}, nil
}

func getMoEBlockPostCombineScratch(dModel int) (*moeBlockPostCombineScratch, error) {
	pool := moeBlockPostCombineScratchPoolFor(dModel)
	if s := pool.Get(); s != nil {
		if s != nil &&
			s.dModel == dModel &&
			s.h != nil && s.h.buf != nil &&
			s.h1 != nil && s.h1.buf != nil &&
			s.h2 != nil && s.h2.buf != nil &&
			s.out != nil && s.out.buf != nil &&
			s.h1Normed != nil &&
			s.h2Normed != nil &&
			s.combined != nil &&
			s.ffResidual != nil {
			return s, nil
		}
		s.Close()
	}
	return newMoEBlockPostCombineScratch(dModel)
}

func moeBlockPostCombineScratchPoolFor(dModel int) *scratchLIFOPool[*moeBlockPostCombineScratch] {
	if v, ok := moeBlockPostCombineScratchPools.Load(dModel); ok {
		return v.(*scratchLIFOPool[*moeBlockPostCombineScratch])
	}
	pool := &scratchLIFOPool[*moeBlockPostCombineScratch]{}
	if v, loaded := moeBlockPostCombineScratchPools.LoadOrStore(dModel, pool); loaded {
		return v.(*scratchLIFOPool[*moeBlockPostCombineScratch])
	}
	return pool
}

func putMoEBlockPostCombineScratch(s *moeBlockPostCombineScratch) {
	if s != nil &&
		s.h != nil && s.h.buf != nil &&
		s.h1 != nil && s.h1.buf != nil &&
		s.h2 != nil && s.h2.buf != nil &&
		s.out != nil && s.out.buf != nil &&
		s.h1Normed != nil &&
		s.h2Normed != nil &&
		s.combined != nil &&
		s.ffResidual != nil {
		moeBlockPostCombineScratchPoolFor(s.dModel).Put(s)
	}
}

func (s *moeBlockPostCombineScratch) Close() {
	if s == nil {
		return
	}
	if s.h != nil {
		s.h.Close()
		s.h = nil
	}
	if s.hPinned != nil {
		s.hPinned.Close()
		s.hPinned = nil
	}
	if s.h1 != nil {
		s.h1.Close()
		s.h1 = nil
	}
	if s.h1Pinned != nil {
		s.h1Pinned.Close()
		s.h1Pinned = nil
	}
	if s.h2 != nil {
		s.h2.Close()
		s.h2 = nil
	}
	if s.h2Pinned != nil {
		s.h2Pinned.Close()
		s.h2Pinned = nil
	}
	if s.out != nil {
		s.out.Close()
		s.out = nil
	}
	s.dModel = 0
}

func postCombineInputView(slot **pinnedNoCopyBytes, x []byte) (metal.MTLBuffer, bool) {
	if len(x) == 0 {
		return nil, false
	}
	if pinned := *slot; pinned != nil && len(pinned.bytes) == len(x) && &pinned.bytes[0] == &x[0] {
		return pinned.buf, true
	}
	if *slot != nil {
		(*slot).Close()
		*slot = nil
	}
	if buf, ok := registeredPinnedNoCopyBytes(x); ok {
		return buf, true
	}
	buf, pinner, noCopy := residentNoCopyBytes(x)
	if !noCopy {
		if pinner != nil {
			pinner.Unpin()
		}
		return nil, false
	}
	pinned := &pinnedNoCopyBytes{bytes: x, buf: buf, pinner: pinner}
	runtime.SetFinalizer(pinned, (*pinnedNoCopyBytes).Close)
	*slot = pinned
	return buf, true
}

func (s *moeBlockPostCombineScratch) residualView(h []byte) (metal.MTLBuffer, bool) {
	if s == nil {
		return nil, false
	}
	return postCombineInputView(&s.hPinned, h)
}

func (s *moeBlockPostCombineScratch) branch1View(h1 []byte) (metal.MTLBuffer, bool) {
	if s == nil {
		return nil, false
	}
	return postCombineInputView(&s.h1Pinned, h1)
}

func (s *moeBlockPostCombineScratch) branch2View(h2 []byte) (metal.MTLBuffer, bool) {
	if s == nil {
		return nil, false
	}
	return postCombineInputView(&s.h2Pinned, h2)
}

func moeBlockPostCombineBF16(h, h1, h2 []byte, post1 []byte, post1View bufView, post2 []byte, post2View bufView, post []byte, postView bufView, dModel int, eps float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	size := dModel * bf16Size
	if len(h) != size || len(h1) != size || len(h2) != size {
		return nil, core.NewError("native.moeBlockPostCombineBF16: h/h1/h2 must be dModel bf16 bytes")
	}
	if len(post1) != size || len(post2) != size || len(post) != size {
		return nil, core.NewError("native.moeBlockPostCombineBF16: post norm weights must be dModel bf16 bytes")
	}
	out := make([]byte, size)
	if dModel == 0 {
		return out, nil
	}
	post1Buf := bf16WeightView(post1, post1View)
	post2Buf := bf16WeightView(post2, post2View)
	postBuf := bf16WeightView(post, postView)

	var encErr error
	withAutoreleasePool(func() {
		scratch, err := getMoEBlockPostCombineScratch(dModel)
		if err != nil {
			encErr = err
			return
		}
		defer putMoEBlockPostCombineScratch(scratch)
		hBuf, ok := scratch.residualView(h)
		if !ok {
			hBuf, err = scratch.h.copyBuffer(h)
			if err != nil {
				encErr = err
				return
			}
		}
		h1Buf, ok := scratch.branch1View(h1)
		if !ok {
			h1Buf, err = scratch.h1.copyBuffer(h1)
			if err != nil {
				encErr = err
				return
			}
		}
		h2Buf, ok := scratch.branch2View(h2)
		if !ok {
			h2Buf, err = scratch.h2.copyBuffer(h2)
			if err != nil {
				encErr = err
				return
			}
		}

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		if encErr = encRMSNormBF16(enc, h1Buf, post1Buf.buf, scratch.h1Normed, post1Buf.off, dModel, eps); encErr != nil {
			endEncodingFast(enc)
			return
		}
		if encErr = encRMSNormBF16(enc, h2Buf, post2Buf.buf, scratch.h2Normed, post2Buf.off, dModel, eps); encErr != nil {
			endEncodingFast(enc)
			return
		}
		if encErr = encAddBF16(enc, scratch.h1Normed, scratch.h2Normed, scratch.combined, dModel); encErr != nil {
			endEncodingFast(enc)
			return
		}
		if encErr = encRMSNormBF16(enc, scratch.combined, postBuf.buf, scratch.ffResidual, postBuf.off, dModel, eps); encErr != nil {
			endEncodingFast(enc)
			return
		}
		if encErr = encAddBF16(enc, hBuf, scratch.ffResidual, scratch.out.buf, dModel); encErr != nil {
			endEncodingFast(enc)
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		copy(out, scratch.out.bytes[:size])
	})
	return out, encErr
}

type moeBlockBF16Scratch struct {
	dModel, dFF, expertDFF, topK int
	h, weights, idx, out         *pinnedNoCopyBytes
	hPinned                      *pinnedNoCopyBytes
	weightsPinned                *pinnedNoCopyBytes
	idxPinned                    *pinnedNoCopyBytes
	outPinned                    *pinnedNoCopyBytes
	mlp                          mlpScratch
	localIn, expertIn            metal.MTLBuffer
	localOut                     metal.MTLBuffer
	expertScaled, expertAcc      metal.MTLBuffer
	localNormed, expertNormed    metal.MTLBuffer
	combined, ffResidual         metal.MTLBuffer
	localMegaGated               metal.MTLBuffer
	localMegaArrive              metal.MTLBuffer
	localMegaArrivePtr           *uint32
	// all-routes expert slabs (the encoded lane's single-dispatch expert projections):
	// [topK × expertDFF] gate/up/gated + [topK × dModel] down, plus the constant index
	// buffers the MLX gather batch dimension consumes (zeros for the shared-x lhs, iota
	// for the per-route gated rows).
	expertGateAll, expertUpAll, expertGatedAll, expertDownAll metal.MTLBuffer
	routeZeros, routeIota                                     metal.MTLBuffer
}

// ensureAllRoutesScratch sizes the all-routes expert slabs (idempotent; the scratch's
// dims are fixed at construction so a single build serves every layer).
func (s *moeBlockBF16Scratch) ensureAllRoutesScratch() error {
	if s.expertGateAll != nil {
		return nil
	}
	if s.topK <= 0 || s.expertDFF <= 0 || s.dModel <= 0 {
		return core.NewError("native.moeBlockScratch: all-routes scratch needs topK/expertDFF/dModel")
	}
	ffBytes := uint(s.topK * s.expertDFF * bf16Size)
	s.expertGateAll = device.NewBufferWithLengthOptions(ffBytes, metal.MTLResourceStorageModeShared)
	s.expertUpAll = device.NewBufferWithLengthOptions(ffBytes, metal.MTLResourceStorageModeShared)
	s.expertGatedAll = device.NewBufferWithLengthOptions(ffBytes, metal.MTLResourceStorageModeShared)
	s.expertDownAll = device.NewBufferWithLengthOptions(uint(s.topK*s.dModel*bf16Size), metal.MTLResourceStorageModeShared)
	zeros := make([]int32, s.topK)
	iota := make([]int32, s.topK)
	for i := range iota {
		iota[i] = int32(i)
	}
	s.routeZeros = device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&zeros[0]), uint(s.topK*4), metal.MTLResourceStorageModeShared)
	s.routeIota = device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&iota[0]), uint(s.topK*4), metal.MTLResourceStorageModeShared)
	if s.expertGateAll == nil || s.expertUpAll == nil || s.expertGatedAll == nil || s.expertDownAll == nil || s.routeZeros == nil || s.routeIota == nil {
		s.expertGateAll, s.expertUpAll, s.expertGatedAll, s.expertDownAll = nil, nil, nil, nil
		s.routeZeros, s.routeIota = nil, nil
		return core.NewError("native.moeBlockScratch: all-routes scratch unavailable")
	}
	return nil
}

type moeBlockBF16ScratchKey struct {
	dModel, dFF, expertDFF, topK int
}

var moeBlockBF16ScratchPools sync.Map

func newMoEBlockBF16Scratch(dModel, dFF, expertDFF, topK int) (*moeBlockBF16Scratch, error) {
	size := dModel * bf16Size
	h, err := newPinnedNoCopyBytes(size)
	if err != nil {
		return nil, err
	}
	weightsSize := topK * bf16Size
	if weightsSize <= 0 {
		weightsSize = bf16Size
	}
	weights, err := newPinnedNoCopyBytes(weightsSize)
	if err != nil {
		h.Close()
		return nil, err
	}
	idxSize := topK * 4
	if idxSize <= 0 {
		idxSize = 4
	}
	idx, err := newPinnedNoCopyBytes(idxSize)
	if err != nil {
		h.Close()
		weights.Close()
		return nil, err
	}
	out, err := newPinnedNoCopyBytes(size)
	if err != nil {
		h.Close()
		weights.Close()
		idx.Close()
		return nil, err
	}
	scratchDFF := max(expertDFF, dFF)
	return &moeBlockBF16Scratch{
		dModel:       dModel,
		dFF:          dFF,
		expertDFF:    expertDFF,
		topK:         topK,
		h:            h,
		weights:      weights,
		idx:          idx,
		out:          out,
		mlp:          newMLPScratch(dModel, scratchDFF),
		localIn:      scratchBF16(dModel),
		expertIn:     scratchBF16(dModel),
		localOut:     scratchBF16(dModel),
		expertScaled: scratchBF16(dModel),
		expertAcc:    scratchBF16(dModel),
		localNormed:  scratchBF16(dModel),
		expertNormed: scratchBF16(dModel),
		combined:     scratchBF16(dModel),
		ffResidual:   scratchBF16(dModel),
	}, nil
}

func moeBlockBF16ScratchPoolFor(dModel, dFF, expertDFF, topK int) *scratchLIFOPool[*moeBlockBF16Scratch] {
	key := moeBlockBF16ScratchKey{dModel: dModel, dFF: dFF, expertDFF: expertDFF, topK: topK}
	if v, ok := moeBlockBF16ScratchPools.Load(key); ok {
		return v.(*scratchLIFOPool[*moeBlockBF16Scratch])
	}
	pool := &scratchLIFOPool[*moeBlockBF16Scratch]{}
	if v, loaded := moeBlockBF16ScratchPools.LoadOrStore(key, pool); loaded {
		return v.(*scratchLIFOPool[*moeBlockBF16Scratch])
	}
	return pool
}

func getMoEBlockBF16Scratch(dModel, dFF, expertDFF, topK int) (*moeBlockBF16Scratch, error) {
	pool := moeBlockBF16ScratchPoolFor(dModel, dFF, expertDFF, topK)
	if s := pool.Get(); s != nil {
		wantWeights := topK * bf16Size
		if wantWeights <= 0 {
			wantWeights = bf16Size
		}
		wantIdx := topK * 4
		if wantIdx <= 0 {
			wantIdx = 4
		}
		if s != nil &&
			s.dModel == dModel &&
			s.dFF == dFF &&
			s.expertDFF == expertDFF &&
			s.topK == topK &&
			s.h != nil && s.h.buf != nil &&
			s.weights != nil && s.weights.buf != nil && len(s.weights.bytes) == wantWeights &&
			s.idx != nil && s.idx.buf != nil && len(s.idx.bytes) == wantIdx &&
			s.out != nil && s.out.buf != nil &&
			s.mlp.gate != nil &&
			s.mlp.up != nil &&
			s.mlp.gated != nil &&
			s.mlp.down != nil &&
			s.localIn != nil &&
			s.expertIn != nil &&
			s.localOut != nil &&
			s.expertScaled != nil &&
			s.expertAcc != nil &&
			s.localNormed != nil &&
			s.expertNormed != nil &&
			s.combined != nil &&
			s.ffResidual != nil {
			return s, nil
		}
		s.Close()
	}
	return newMoEBlockBF16Scratch(dModel, dFF, expertDFF, topK)
}

func putMoEBlockBF16Scratch(s *moeBlockBF16Scratch) {
	if s != nil &&
		s.h != nil && s.h.buf != nil &&
		s.weights != nil && s.weights.buf != nil &&
		s.idx != nil && s.idx.buf != nil &&
		s.out != nil && s.out.buf != nil &&
		s.mlp.gate != nil &&
		s.mlp.up != nil &&
		s.mlp.gated != nil &&
		s.mlp.down != nil &&
		s.localIn != nil &&
		s.expertIn != nil &&
		s.localOut != nil &&
		s.expertScaled != nil &&
		s.expertAcc != nil &&
		s.localNormed != nil &&
		s.expertNormed != nil &&
		s.combined != nil &&
		s.ffResidual != nil {
		moeBlockBF16ScratchPoolFor(s.dModel, s.dFF, s.expertDFF, s.topK).Put(s)
	}
}

func (s *moeBlockBF16Scratch) ensureLocalMegaScratch() error {
	if s.localMegaGated != nil && s.localMegaArrive != nil && s.localMegaArrivePtr != nil {
		return nil
	}
	s.localMegaGated = device.NewBufferWithLengthOptions(uint(s.dFF*4), metal.MTLResourceStorageModeShared)
	s.localMegaArrive = device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared)
	if s.localMegaGated == nil || s.localMegaGated.GetID() == 0 || s.localMegaArrive == nil || s.localMegaArrive.GetID() == 0 {
		s.localMegaGated = nil
		s.localMegaArrive = nil
		s.localMegaArrivePtr = nil
		return core.NewError("native.moeBlockScratch: local megakernel scratch unavailable")
	}
	s.localMegaArrivePtr = (*uint32)(s.localMegaArrive.Contents())
	return nil
}

func (s *moeBlockBF16Scratch) Close() {
	if s == nil {
		return
	}
	if s.h != nil {
		s.h.Close()
		s.h = nil
	}
	if s.hPinned != nil {
		s.hPinned.Close()
		s.hPinned = nil
	}
	if s.weights != nil {
		s.weights.Close()
		s.weights = nil
	}
	if s.weightsPinned != nil {
		s.weightsPinned.Close()
		s.weightsPinned = nil
	}
	if s.idx != nil {
		s.idx.Close()
		s.idx = nil
	}
	if s.idxPinned != nil {
		s.idxPinned.Close()
		s.idxPinned = nil
	}
	if s.out != nil {
		s.out.Close()
		s.out = nil
	}
	if s.outPinned != nil {
		s.outPinned.Close()
		s.outPinned = nil
	}
	s.localMegaGated = nil
	s.localMegaArrive = nil
	s.localMegaArrivePtr = nil
	s.dModel, s.dFF, s.expertDFF, s.topK = 0, 0, 0, 0
}

func (s *moeBlockBF16Scratch) inputView(h []byte) (metal.MTLBuffer, bool) {
	if s == nil || len(h) == 0 {
		return nil, false
	}
	if s.hPinned != nil && len(s.hPinned.bytes) == len(h) && &s.hPinned.bytes[0] == &h[0] {
		return s.hPinned.buf, true
	}
	if s.hPinned != nil {
		s.hPinned.Close()
		s.hPinned = nil
	}
	if buf, ok := registeredPinnedNoCopyBytes(h); ok {
		return buf, true
	}
	buf, pinner, noCopy := residentNoCopyBytes(h)
	if !noCopy {
		if pinner != nil {
			pinner.Unpin()
		}
		return nil, false
	}
	pinned := &pinnedNoCopyBytes{bytes: h, buf: buf, pinner: pinner}
	runtime.SetFinalizer(pinned, (*pinnedNoCopyBytes).Close)
	s.hPinned = pinned
	return buf, true
}

func (s *moeBlockBF16Scratch) weightsView(weights []byte) (metal.MTLBuffer, bool) {
	if s == nil || len(weights) == 0 {
		return nil, false
	}
	if s.weightsPinned != nil && len(s.weightsPinned.bytes) == len(weights) && &s.weightsPinned.bytes[0] == &weights[0] {
		return s.weightsPinned.buf, true
	}
	if s.weightsPinned != nil {
		s.weightsPinned.Close()
		s.weightsPinned = nil
	}
	if buf, ok := registeredPinnedNoCopyBytes(weights); ok {
		return buf, true
	}
	buf, pinner, noCopy := residentNoCopyBytes(weights)
	if !noCopy {
		if pinner != nil {
			pinner.Unpin()
		}
		return nil, false
	}
	pinned := &pinnedNoCopyBytes{bytes: weights, buf: buf, pinner: pinner}
	runtime.SetFinalizer(pinned, (*pinnedNoCopyBytes).Close)
	s.weightsPinned = pinned
	return buf, true
}

func (s *moeBlockBF16Scratch) indexView(idx []int32) (metal.MTLBuffer, bool) {
	if s == nil || len(idx) == 0 {
		return nil, false
	}
	idxBytes := unsafe.Slice((*byte)(unsafe.Pointer(&idx[0])), len(idx)*4)
	if s.idxPinned != nil && len(s.idxPinned.bytes) == len(idxBytes) && &s.idxPinned.bytes[0] == &idxBytes[0] {
		return s.idxPinned.buf, true
	}
	if s.idxPinned != nil {
		return nil, false
	}
	if buf, ok := registeredPinnedNoCopyBytes(idxBytes); ok {
		runtime.KeepAlive(idx)
		return buf, true
	}
	buf, pinner, noCopy := residentNoCopyBytes(idxBytes)
	if !noCopy {
		if pinner != nil {
			pinner.Unpin()
		}
		return nil, false
	}
	pinned := &pinnedNoCopyBytes{bytes: idxBytes, buf: buf, pinner: pinner}
	runtime.SetFinalizer(pinned, (*pinnedNoCopyBytes).Close)
	s.idxPinned = pinned
	runtime.KeepAlive(idx)
	return buf, true
}

func (s *moeBlockBF16Scratch) outputView(out []byte) (metal.MTLBuffer, bool) {
	if s == nil || len(out) == 0 {
		return nil, false
	}
	if s.outPinned != nil && len(s.outPinned.bytes) == len(out) && &s.outPinned.bytes[0] == &out[0] {
		return s.outPinned.buf, true
	}
	if s.outPinned != nil {
		s.outPinned.Close()
		s.outPinned = nil
	}
	if buf, ok := registeredPinnedNoCopyBytes(out); ok {
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
	s.outPinned = pinned
	return buf, true
}

func moeBlockBF16AfterRouter(h []byte, idx []int32, weights []byte, weightBuf metal.MTLBuffer, w MoELayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockBF16AfterRouterWithBuffer(h, nil, idx, weights, weightBuf, w, dModel, dFF, eps)
}

func moeBlockBF16AfterRouterWithBuffer(h []byte, hBuf metal.MTLBuffer, idx []int32, weights []byte, weightBuf metal.MTLBuffer, w MoELayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockBF16AfterRouterWithBufferPooled(h, hBuf, nil, nil, idx, weights, weightBuf, w, dModel, dFF, eps, true, false)
}

func moeBlockBF16AfterRouterWithBufferInPool(h []byte, hBuf metal.MTLBuffer, idx []int32, weights []byte, weightBuf metal.MTLBuffer, w MoELayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockBF16AfterRouterWithBufferPooled(h, hBuf, nil, nil, idx, weights, weightBuf, w, dModel, dFF, eps, false, false)
}

func moeBlockBF16AfterRouterWithBufferInto(out []byte, h []byte, hBuf metal.MTLBuffer, idx []int32, weights []byte, weightBuf metal.MTLBuffer, w MoELayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockBF16AfterRouterWithBufferPooled(h, hBuf, out, nil, idx, weights, weightBuf, w, dModel, dFF, eps, true, true)
}

func moeBlockBF16AfterRouterWithBufferIntoInPool(out []byte, h []byte, hBuf metal.MTLBuffer, idx []int32, weights []byte, weightBuf metal.MTLBuffer, w MoELayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockBF16AfterRouterWithBufferPooled(h, hBuf, out, nil, idx, weights, weightBuf, w, dModel, dFF, eps, false, true)
}

func moeBlockBF16AfterRouterWithBufferOutputInPool(h []byte, hBuf, outputBuf metal.MTLBuffer, idx []int32, weights []byte, weightBuf metal.MTLBuffer, w MoELayerWeights, dModel, dFF int, eps float32) error {
	if outputBuf == nil {
		return core.NewError("native.moeBlockBF16AfterRouter: output buffer is nil")
	}
	_, err := moeBlockBF16AfterRouterWithBufferPooled(h, hBuf, nil, outputBuf, idx, weights, weightBuf, w, dModel, dFF, eps, false, false)
	return err
}

func moeBlockBF16AfterRouterWithBufferPooled(h []byte, hBuf metal.MTLBuffer, out []byte, outputBuf metal.MTLBuffer, idx []int32, weights []byte, weightBuf metal.MTLBuffer, w MoELayerWeights, dModel, dFF int, eps float32, useAutoreleasePool bool, useCallerOut bool) ([]byte, error) {
	expertDFF, numExperts, topK := w.ExpertDFF, w.NumExperts, w.TopK
	size := dModel * bf16Size
	if len(h) != size {
		return nil, core.NewError("native.moeBlockBF16AfterRouter: h must be dModel bf16 bytes")
	}
	if len(idx) != topK || len(weights) != topK*bf16Size {
		return nil, core.NewError("native.moeBlockBF16AfterRouter: idx/weights length must equal topK")
	}
	if len(w.PreFFNormW) != size || len(w.PreFFNorm2W) != size || len(w.PostFFNorm1W) != size || len(w.PostFFNorm2W) != size || len(w.PostFFNormW) != size {
		return nil, core.NewError("native.moeBlockBF16AfterRouter: norm weights must be dModel bf16 bytes")
	}
	if len(w.WGate) != dFF*dModel*bf16Size || len(w.WUp) != dFF*dModel*bf16Size {
		return nil, core.NewError("native.moeBlockBF16AfterRouter: local gate/up weights must be dFF*dModel bf16 bytes")
	}
	if len(w.WDown) != dModel*dFF*bf16Size {
		return nil, core.NewError("native.moeBlockBF16AfterRouter: local down weight must be dModel*dFF bf16 bytes")
	}
	gateSz, downSz := expertDFF*dModel*bf16Size, dModel*expertDFF*bf16Size
	if len(w.ExpGateW) != numExperts*gateSz || len(w.ExpUpW) != numExperts*gateSz || len(w.ExpDownW) != numExperts*downSz {
		return nil, core.NewError("native.moeBlockBF16AfterRouter: expert weight size mismatch")
	}
	for i := range idx {
		if idx[i] < 0 || int(idx[i]) >= numExperts {
			return nil, core.NewError("native.moeBlockBF16AfterRouter: expert index out of range")
		}
	}
	bufferOut := outputBuf != nil
	callerOut := !bufferOut && useCallerOut && cap(out) >= size
	if bufferOut {
		out = nil
	} else if callerOut {
		out = out[:size]
	} else {
		out = make([]byte, size)
	}
	if dModel == 0 || dFF == 0 || expertDFF == 0 {
		if bufferOut && size > 0 {
			clear(unsafe.Slice((*byte)(outputBuf.Contents()), size))
			return nil, nil
		}
		if !bufferOut {
			clear(out)
		}
		return out, nil
	}

	pre1Buf := bf16WeightView(w.PreFFNormW, bufView{})
	pre2Buf := bf16WeightView(w.PreFFNorm2W, bufView{})
	post1Buf := bf16WeightView(w.PostFFNorm1W, bufView{})
	post2Buf := bf16WeightView(w.PostFFNorm2W, bufView{})
	postBuf := bf16WeightView(w.PostFFNormW, bufView{})
	localGate, localUp, localDown := residentBytes(w.WGate), residentBytes(w.WUp), residentBytes(w.WDown)
	expertGate, expertUp, expertDown := residentBytes(w.ExpGateW), residentBytes(w.ExpUpW), residentBytes(w.ExpDownW)
	localInBM, localInBN, localInSM, localInSN, localInTM, localInTN := gemvTiles(dModel, dFF)
	localInPSO, err := pipelineFor(gemvKernelName("bfloat16", localInBM, localInBN, localInSM, localInSN, localInTM, localInTN))
	if err != nil {
		return nil, err
	}
	localDownBM, localDownBN, localDownSM, localDownSN, localDownTM, localDownTN := gemvTiles(dFF, dModel)
	localDownPSO, err := pipelineFor(gemvKernelName("bfloat16", localDownBM, localDownBN, localDownSM, localDownSN, localDownTM, localDownTN))
	if err != nil {
		return nil, err
	}
	expertInBM, expertInBN, expertInSM, expertInSN, expertInTM, expertInTN := gemvTiles(dModel, expertDFF)
	expertInPSO, err := pipelineFor(gemvKernelName("bfloat16", expertInBM, expertInBN, expertInSM, expertInSN, expertInTM, expertInTN))
	if err != nil {
		return nil, err
	}
	expertDownBM, expertDownBN, expertDownSM, expertDownSN, expertDownTM, expertDownTN := gemvTiles(expertDFF, dModel)
	expertDownPSO, err := pipelineFor(gemvKernelName("bfloat16", expertDownBM, expertDownBN, expertDownSM, expertDownSN, expertDownTM, expertDownTN))
	if err != nil {
		return nil, err
	}
	rmsPSO, err := pipelineFor(rmsKernelBF16(dModel))
	if err != nil {
		return nil, err
	}
	rmsTG := rmsThreadgroup(dModel, rmsPSO)
	addPSO, err := pipelineFor("vv_Addbfloat16")
	if err != nil {
		return nil, err
	}
	var geluPSO metal.MTLComputePipelineState
	useFusedGelu := gpuHasGeluKernel()
	if useFusedGelu {
		geluPSO, err = geluPipeline()
		if err != nil {
			return nil, err
		}
	}
	scalePSO, scaleErr := bf16MulScalarPipeline()

	var encErr error
	run := func() {
		scratch, err := getMoEBlockBF16Scratch(dModel, dFF, expertDFF, topK)
		if err != nil {
			encErr = err
			return
		}
		defer putMoEBlockBF16Scratch(scratch)
		inputBuf := hBuf
		if inputBuf == nil {
			var ok bool
			inputBuf, ok = scratch.inputView(h)
			if !ok {
				inputBuf, err = scratch.h.copyBuffer(h)
				if err != nil {
					encErr = err
					return
				}
			}
		}
		weightsBuf := weightBuf
		if topK > 0 {
			if weightsBuf == nil {
				var ok bool
				weightsBuf, ok = scratch.weightsView(weights)
				if !ok {
					weightsBuf, err = scratch.weights.copyBuffer(weights)
					if err != nil {
						encErr = err
						return
					}
				}
			}
		} else {
			clear(unsafe.Slice((*byte)(scratch.expertAcc.Contents()), size))
		}
		msc := scratch.mlp
		finalOutBuf := scratch.out.buf
		directOut := false
		if bufferOut {
			finalOutBuf = outputBuf
			directOut = true
		} else if callerOut {
			if tmp, ok := scratch.outputView(out); ok {
				finalOutBuf = tmp
				directOut = true
			}
		}

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		sink := encSink{enc}
		emitRMS := func(x, weight, out metal.MTLBuffer, wOff uint) {
			emitRMSNorm(sink, rmsPSO, x, weight, out, wOff, dModel, eps, rmsTG)
		}
		emitLocalInGemv := func(mat, vec, out metal.MTLBuffer, matOff uint) {
			emitGemv(sink, localInPSO, mat, matOff, vec, out, 0, dModel, dFF, localInBM, localInBN, localInSM, localInTM)
		}
		emitLocalDownGemv := func(mat, vec, out metal.MTLBuffer) {
			emitGemv(sink, localDownPSO, mat, 0, vec, out, 0, dFF, dModel, localDownBM, localDownBN, localDownSM, localDownTM)
		}
		emitExpertInGemv := func(mat, vec, out metal.MTLBuffer, matOff uint) {
			emitGemv(sink, expertInPSO, mat, matOff, vec, out, 0, dModel, expertDFF, expertInBM, expertInBN, expertInSM, expertInTM)
		}
		emitExpertDownGemv := func(mat, vec, out metal.MTLBuffer, matOff uint) {
			emitGemv(sink, expertDownPSO, mat, matOff, vec, out, 0, expertDFF, dModel, expertDownBM, expertDownBN, expertDownSM, expertDownTM)
		}
		emitGelu := func(gate, up, out metal.MTLBuffer, n int) error {
			if useFusedGelu {
				emitBinary(sink, geluPSO, gate, 0, up, 0, out, 0, n)
				return nil
			}
			return encGeluGateMul(enc, gate, up, out, msc, n)
		}
		emitScale := func(in, scalar, out metal.MTLBuffer, scalarOffset uint, scalarBytes []byte, n int) error {
			if scaleErr != nil {
				return encScaleBF16(enc, in, scalar, out, scalarOffset, scalarBytes, n)
			}
			sink.setPSO(scalePSO)
			sink.setBuf(in, 0, 0)
			sink.setBuf(scalar, scalarOffset, 1)
			sink.setBuf(out, 0, 2)
			sink.setI32(int32(n), 3)
			group := min(uint(n), uint(256))
			sink.dispatchThreads(
				metal.MTLSize{Width: uint(n), Height: 1, Depth: 1},
				metal.MTLSize{Width: group, Height: 1, Depth: 1},
			)
			return nil
		}
		emitAdd := func(a, b, out metal.MTLBuffer) {
			emitBinary(sink, addPSO, a, 0, b, 0, out, 0, dModel)
		}
		emitRMS(inputBuf, pre1Buf.buf, scratch.localIn, pre1Buf.off)
		emitLocalInGemv(localGate, scratch.localIn, msc.gate, 0)
		emitLocalInGemv(localUp, scratch.localIn, msc.up, 0)
		if encErr = emitGelu(msc.gate, msc.up, msc.gated, dFF); encErr != nil {
			endEncodingFast(enc)
			return
		}
		emitLocalDownGemv(localDown, msc.gated, scratch.localOut)
		emitRMS(inputBuf, pre2Buf.buf, scratch.expertIn, pre2Buf.off)
		for i := range topK {
			e := int(idx[i])
			gateOff, downOff := uint(e*gateSz), uint(e*downSz)
			emitExpertInGemv(expertGate, scratch.expertIn, msc.gate, gateOff)
			emitExpertInGemv(expertUp, scratch.expertIn, msc.up, gateOff)
			if encErr = emitGelu(msc.gate, msc.up, msc.gated, expertDFF); encErr != nil {
				endEncodingFast(enc)
				return
			}
			emitExpertDownGemv(expertDown, msc.gated, msc.down, downOff)
			if i == 0 {
				if encErr = emitScale(msc.down, weightsBuf, scratch.expertAcc, uint(i*bf16Size), weights[i*bf16Size:(i+1)*bf16Size], dModel); encErr != nil {
					endEncodingFast(enc)
					return
				}
			} else {
				if encErr = emitScale(msc.down, weightsBuf, scratch.expertScaled, uint(i*bf16Size), weights[i*bf16Size:(i+1)*bf16Size], dModel); encErr != nil {
					endEncodingFast(enc)
					return
				}
				emitAdd(scratch.expertAcc, scratch.expertScaled, scratch.expertAcc)
			}
		}
		emitRMS(scratch.localOut, post1Buf.buf, scratch.localNormed, post1Buf.off)
		emitRMS(scratch.expertAcc, post2Buf.buf, scratch.expertNormed, post2Buf.off)
		emitAdd(scratch.localNormed, scratch.expertNormed, scratch.combined)
		emitRMS(scratch.combined, postBuf.buf, scratch.ffResidual, postBuf.off)
		emitAdd(inputBuf, scratch.ffResidual, finalOutBuf)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, scratch.out.bytes[:size])
		}
	}
	if useAutoreleasePool {
		withAutoreleasePool(run)
	} else {
		run()
	}
	return out, encErr
}

// MoEBlockBF16 runs the dual-branch feed-forward of a gemma4 MoE layer on the
// post-attention residual h and returns h + ffResidual. BOTH branches run: the local
// dense MLP on rms(h, PreFFNorm), and the expert branch (router → topK experts) on
// rms(h, PreFFNorm2). Each branch output is independently normed (PostFFNorm1 /
// PostFFNorm2), summed, post-normed (PostFFNorm), then added back to the residual
// once. Mirrors pkg/metal/model/gemma4 decoder_layer.go's MoE branch op-for-op.
//
// The router operates on the RAW residual h (it applies its own internal norm); the
// experts operate on the separately-normed h2In. The router runs host top-k (see
// MoERouter) so this block is not a single command buffer; everything else is the
// parity-proven bf16 ops composed. Byte-for-byte against an independent reference
// that rebuilds both branches from primitives (TestMoEBlock). The per-layer-input
// gate, the LayerScalar, and the FFN-memory augmenter are out of scope (later
// slices / nil for standard gemma4) — this block ends at residual + ffResidual.
// NumExperts/TopK/ExpertDFF come from w; dModel/dFF/eps are the model-wide args.
func MoEBlockBF16(h []byte, w MoELayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockBF16WithBuffer(h, nil, w, dModel, dFF, eps)
}

func MoEBlockBF16Into(out []byte, h []byte, w MoELayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockBF16WithBufferInto(out, h, nil, w, dModel, dFF, eps)
}

func moeBlockBF16WithBuffer(h []byte, hBuf metal.MTLBuffer, w MoELayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockBF16WithBufferPooled(h, hBuf, w, dModel, dFF, eps, true)
}

func moeBlockBF16WithBufferInto(out []byte, h []byte, hBuf metal.MTLBuffer, w MoELayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockBF16WithBufferPooledInto(out, h, hBuf, w, dModel, dFF, eps, true, true)
}

func moeBlockBF16WithBufferInPool(h []byte, hBuf metal.MTLBuffer, w MoELayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockBF16WithBufferPooled(h, hBuf, w, dModel, dFF, eps, false)
}

func moeBlockBF16WithBufferOutputInPool(h []byte, hBuf, outputBuf metal.MTLBuffer, w MoELayerWeights, dModel, dFF int, eps float32) error {
	if outputBuf == nil {
		return core.NewError("native.MoEBlockBF16: output buffer is nil")
	}
	if err := ensureInit(); err != nil {
		return err
	}
	if len(h) != dModel*bf16Size {
		return core.NewError("native.MoEBlockBF16: h must be dModel bf16 bytes")
	}
	numExperts, topK := w.NumExperts, w.TopK

	if idx, weights, weightBuf, routerScratch, ok, err := moeRouterBF16DeviceTopKNoCopyWithBufferInPool(h, hBuf, w.RouterNormWScaled, w.RouterW, w.PerExpertScale, numExperts, topK, dModel, eps); ok || err != nil {
		if err != nil {
			return err
		}
		err = moeBlockBF16AfterRouterWithBufferOutputInPool(h, hBuf, outputBuf, idx, weights, weightBuf, w, dModel, dFF, eps)
		putRouterDeviceScratch(routerScratch)
		return err
	}
	idx, weights, err := MoERouter(h, w.RouterNormWScaled, w.RouterW, w.PerExpertScale, numExperts, topK, dModel, eps)
	if err != nil {
		return err
	}
	return moeBlockBF16AfterRouterWithBufferOutputInPool(h, hBuf, outputBuf, idx, weights, nil, w, dModel, dFF, eps)
}

func moeBlockBF16WithBufferPooled(h []byte, hBuf metal.MTLBuffer, w MoELayerWeights, dModel, dFF int, eps float32, useAutoreleasePool bool) ([]byte, error) {
	return moeBlockBF16WithBufferPooledInto(nil, h, hBuf, w, dModel, dFF, eps, useAutoreleasePool, false)
}

func moeBlockBF16WithBufferPooledInto(out []byte, h []byte, hBuf metal.MTLBuffer, w MoELayerWeights, dModel, dFF int, eps float32, useAutoreleasePool bool, useCallerOut bool) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(h) != dModel*bf16Size {
		return nil, core.NewError("native.MoEBlockBF16: h must be dModel bf16 bytes")
	}
	numExperts, topK := w.NumExperts, w.TopK

	if useAutoreleasePool {
		var blockOut []byte
		var blockErr error
		withAutoreleasePool(func() {
			blockOut, blockErr = moeBlockBF16WithBufferPooledInto(out, h, hBuf, w, dModel, dFF, eps, false, useCallerOut)
		})
		return blockOut, blockErr
	}

	// router decision on the raw residual (the router applies its own norm).
	if idx, weights, weightBuf, routerScratch, ok, err := moeRouterBF16DeviceTopKNoCopyWithBufferInPool(h, hBuf, w.RouterNormWScaled, w.RouterW, w.PerExpertScale, numExperts, topK, dModel, eps); ok || err != nil {
		if err != nil {
			return nil, err
		}
		var blockOut []byte
		if useCallerOut {
			blockOut, err = moeBlockBF16AfterRouterWithBufferIntoInPool(out, h, hBuf, idx, weights, weightBuf, w, dModel, dFF, eps)
		} else {
			blockOut, err = moeBlockBF16AfterRouterWithBufferInPool(h, hBuf, idx, weights, weightBuf, w, dModel, dFF, eps)
		}
		putRouterDeviceScratch(routerScratch)
		return blockOut, err
	}
	idx, weights, err := MoERouter(h, w.RouterNormWScaled, w.RouterW, w.PerExpertScale, numExperts, topK, dModel, eps)
	if err != nil {
		return nil, err
	}
	if useCallerOut {
		return moeBlockBF16AfterRouterWithBufferIntoInPool(out, h, hBuf, idx, weights, nil, w, dModel, dFF, eps)
	}
	return moeBlockBF16AfterRouterWithBufferInPool(h, hBuf, idx, weights, nil, w, dModel, dFF, eps)
}

// MoEQuantLayerWeights is MoELayerWeights for a 4-bit MoE layer (gemma4 26B-A4B): the local
// dense MLP, the router score projection, and the batched SwitchGLU experts are all affine-
// quantised; the five norms stay bf16. RouterNormWScaled is the router norm pre-folded by
// RootSize (as MoERouter expects); PerExpertScale is optional. Local dFF (IntermediateSize) and
// expert dFF (ExpertDFF / MoEIntermediateSize) differ, as in the bf16 block.
type MoEQuantLayerWeights struct {
	NumExperts, TopK, ExpertDFF int
	// per-component quant (mixed-precision QAT: gemma4 26B-A4B keeps the experts 4-bit but the
	// local MLP + router 8-bit). Uniform packs set all three the same.
	ExpertGroupSize, ExpertBits int
	LocalGroupSize, LocalBits   int
	RouterGroupSize, RouterBits int

	PreFFNormW, PreFFNorm2W                 []byte
	PostFFNorm1W, PostFFNorm2W, PostFFNormW []byte
	preFFNormView, preFFNorm2View           bufView
	postFFNorm1View, postFFNorm2View        bufView
	postFFNormView                          bufView

	LocalGate, LocalUp, LocalDown QuantWeight // local dense MLP (dFF)

	RouterNormWScaled  []byte
	Router             QuantWeight // [NumExperts × dModel] expert-score projection
	PerExpertScale     []byte      // [NumExperts] (nil to skip)
	routerNormView     bufView
	perExpertScaleView bufView

	ExpGate, ExpUp, ExpGateUp, ExpDown QuantWeight // batched SwitchGLU experts (ExpertDFF)
}

type mlpTransformScratch struct {
	dModel, dFF int
	x           *pinnedNoCopyBytes
	mlp         mlpScratch
	inViewPtr   uintptr
	inViewLen   int
	inView      metal.MTLBuffer
	inPinned    *pinnedNoCopyBytes
	outViewPtr  uintptr
	outViewLen  int
	outView     metal.MTLBuffer
	outPinned   *pinnedNoCopyBytes
}

type mlpTransformScratchKey struct {
	dModel, dFF int
}

var mlpTransformScratchPools sync.Map

func newMLPTransformScratch(dModel, dFF int) (*mlpTransformScratch, error) {
	x, err := newPinnedNoCopyBytes(dModel * bf16Size)
	if err != nil {
		return nil, err
	}
	return &mlpTransformScratch{
		dModel: dModel,
		dFF:    dFF,
		x:      x,
		mlp:    newMLPScratch(dModel, dFF),
	}, nil
}

func mlpTransformScratchPoolFor(dModel, dFF int) *scratchLIFOPool[*mlpTransformScratch] {
	key := mlpTransformScratchKey{dModel: dModel, dFF: dFF}
	if v, ok := mlpTransformScratchPools.Load(key); ok {
		return v.(*scratchLIFOPool[*mlpTransformScratch])
	}
	pool := &scratchLIFOPool[*mlpTransformScratch]{}
	if v, loaded := mlpTransformScratchPools.LoadOrStore(key, pool); loaded {
		return v.(*scratchLIFOPool[*mlpTransformScratch])
	}
	return pool
}

func getMLPTransformScratch(dModel, dFF int) (*mlpTransformScratch, error) {
	pool := mlpTransformScratchPoolFor(dModel, dFF)
	if s := pool.Get(); s != nil {
		if s != nil &&
			s.dModel == dModel &&
			s.dFF == dFF &&
			s.x != nil &&
			s.x.buf != nil &&
			s.mlp.gate != nil &&
			s.mlp.up != nil &&
			s.mlp.gated != nil &&
			s.mlp.down != nil {
			return s, nil
		}
		s.Close()
	}
	return newMLPTransformScratch(dModel, dFF)
}

func putMLPTransformScratch(s *mlpTransformScratch) {
	if s != nil && s.x != nil && s.x.buf != nil && s.mlp.gate != nil && s.mlp.up != nil && s.mlp.gated != nil && s.mlp.down != nil {
		mlpTransformScratchPoolFor(s.dModel, s.dFF).Put(s)
	}
}

func (s *mlpTransformScratch) Close() {
	if s == nil {
		return
	}
	if s.x != nil {
		s.x.Close()
		s.x = nil
	}
	s.closeInputView()
	s.closeOutputView()
	s.dModel, s.dFF = 0, 0
}

func (s *mlpTransformScratch) closeInputView() {
	if s == nil {
		return
	}
	if s.inPinned != nil {
		s.inPinned.Close()
	}
	s.inViewPtr = 0
	s.inViewLen = 0
	s.inView = nil
	s.inPinned = nil
}

func (s *mlpTransformScratch) closeOutputView() {
	if s == nil {
		return
	}
	if s.outPinned != nil {
		s.outPinned.Close()
	}
	s.outViewPtr = 0
	s.outViewLen = 0
	s.outView = nil
	s.outPinned = nil
}

func (s *mlpTransformScratch) inputView(x []byte) (metal.MTLBuffer, bool) {
	if s == nil || len(x) == 0 {
		return nil, false
	}
	ptr := uintptr(unsafe.Pointer(&x[0]))
	if s.inView != nil && s.inViewPtr == ptr && s.inViewLen == len(x) {
		return s.inView, true
	}
	s.closeInputView()
	if buf, ok := registeredPinnedNoCopyBytes(x); ok {
		s.inViewPtr = ptr
		s.inViewLen = len(x)
		s.inView = buf
		s.inPinned = nil
		return buf, true
	}
	buf, pinner, noCopy := residentNoCopyBytes(x)
	if !noCopy {
		if pinner != nil {
			pinner.Unpin()
		}
		return nil, false
	}
	pinned := &pinnedNoCopyBytes{bytes: x, buf: buf, pinner: pinner}
	runtime.SetFinalizer(pinned, (*pinnedNoCopyBytes).Close)
	s.inViewPtr = ptr
	s.inViewLen = len(x)
	s.inView = buf
	s.inPinned = pinned
	return buf, true
}

func (s *mlpTransformScratch) outputView(out []byte) (metal.MTLBuffer, bool) {
	if s == nil || len(out) == 0 {
		return nil, false
	}
	ptr := uintptr(unsafe.Pointer(&out[0]))
	if s.outView != nil && s.outViewPtr == ptr && s.outViewLen == len(out) {
		return s.outView, true
	}
	s.closeOutputView()
	if buf, ok := registeredPinnedNoCopyBytes(out); ok {
		s.outViewPtr = ptr
		s.outViewLen = len(out)
		s.outView = buf
		s.outPinned = nil
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
	s.outViewPtr = ptr
	s.outViewLen = len(out)
	s.outView = buf
	s.outPinned = pinned
	return buf, true
}

type mlpTransformMegaScratch struct {
	dModel, dFF        int
	x                  *pinnedNoCopyBytes
	gated, out, arrive metal.MTLBuffer
	outBytes           []byte
	arrivePtr          *uint32
	inViewPtr          uintptr
	inViewLen          int
	inView             metal.MTLBuffer
	inPinned           *pinnedNoCopyBytes
	outViewPtr         uintptr
	outViewLen         int
	outView            metal.MTLBuffer
	outPinned          *pinnedNoCopyBytes
}

var mlpTransformMegaScratchPools sync.Map

func newMLPTransformMegaScratch(dModel, dFF int) (*mlpTransformMegaScratch, error) {
	x, err := newPinnedNoCopyBytes(dModel * bf16Size)
	if err != nil {
		return nil, err
	}
	gated := device.NewBufferWithLengthOptions(uint(dFF*4), metal.MTLResourceStorageModeShared)
	out := device.NewBufferWithLengthOptions(uint(dModel*bf16Size), metal.MTLResourceStorageModeShared)
	arrive := device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared)
	return &mlpTransformMegaScratch{
		dModel:    dModel,
		dFF:       dFF,
		x:         x,
		gated:     gated,
		out:       out,
		arrive:    arrive,
		outBytes:  unsafe.Slice((*byte)(out.Contents()), dModel*bf16Size),
		arrivePtr: (*uint32)(arrive.Contents()),
	}, nil
}

func mlpTransformMegaScratchPoolFor(dModel, dFF int) *scratchLIFOPool[*mlpTransformMegaScratch] {
	key := mlpTransformScratchKey{dModel: dModel, dFF: dFF}
	if v, ok := mlpTransformMegaScratchPools.Load(key); ok {
		return v.(*scratchLIFOPool[*mlpTransformMegaScratch])
	}
	pool := &scratchLIFOPool[*mlpTransformMegaScratch]{}
	if v, loaded := mlpTransformMegaScratchPools.LoadOrStore(key, pool); loaded {
		return v.(*scratchLIFOPool[*mlpTransformMegaScratch])
	}
	return pool
}

func getMLPTransformMegaScratch(dModel, dFF int) (*mlpTransformMegaScratch, error) {
	pool := mlpTransformMegaScratchPoolFor(dModel, dFF)
	if s := pool.Get(); s != nil {
		if s != nil && s.dModel == dModel && s.dFF == dFF && s.x != nil && s.x.buf != nil && s.gated != nil && s.out != nil && s.arrive != nil && len(s.outBytes) == dModel*bf16Size && s.arrivePtr != nil {
			return s, nil
		}
		s.Close()
	}
	return newMLPTransformMegaScratch(dModel, dFF)
}

func putMLPTransformMegaScratch(s *mlpTransformMegaScratch) {
	if s != nil && s.x != nil && s.x.buf != nil && s.gated != nil && s.out != nil && s.arrive != nil && len(s.outBytes) == s.dModel*bf16Size && s.arrivePtr != nil {
		mlpTransformMegaScratchPoolFor(s.dModel, s.dFF).Put(s)
	}
}

func (s *mlpTransformMegaScratch) Close() {
	if s == nil {
		return
	}
	if s.x != nil {
		s.x.Close()
		s.x = nil
	}
	s.gated = nil
	s.out = nil
	s.arrive = nil
	s.outBytes = nil
	s.arrivePtr = nil
	s.closeInputView()
	s.closeOutputView()
	s.dModel, s.dFF = 0, 0
}

func (s *mlpTransformMegaScratch) closeInputView() {
	if s == nil {
		return
	}
	if s.inPinned != nil {
		s.inPinned.Close()
	}
	s.inViewPtr = 0
	s.inViewLen = 0
	s.inView = nil
	s.inPinned = nil
}

func (s *mlpTransformMegaScratch) closeOutputView() {
	if s == nil {
		return
	}
	if s.outPinned != nil {
		s.outPinned.Close()
	}
	s.outViewPtr = 0
	s.outViewLen = 0
	s.outView = nil
	s.outPinned = nil
}

func (s *mlpTransformMegaScratch) inputView(x []byte) (metal.MTLBuffer, bool) {
	if s == nil || len(x) == 0 {
		return nil, false
	}
	ptr := uintptr(unsafe.Pointer(&x[0]))
	if s.inView != nil && s.inViewPtr == ptr && s.inViewLen == len(x) {
		return s.inView, true
	}
	s.closeInputView()
	if buf, ok := registeredPinnedNoCopyBytes(x); ok {
		s.inViewPtr = ptr
		s.inViewLen = len(x)
		s.inView = buf
		s.inPinned = nil
		return buf, true
	}
	buf, pinner, noCopy := residentNoCopyBytes(x)
	if !noCopy {
		if pinner != nil {
			pinner.Unpin()
		}
		return nil, false
	}
	pinned := &pinnedNoCopyBytes{bytes: x, buf: buf, pinner: pinner}
	runtime.SetFinalizer(pinned, (*pinnedNoCopyBytes).Close)
	s.inViewPtr = ptr
	s.inViewLen = len(x)
	s.inView = buf
	s.inPinned = pinned
	return buf, true
}

func (s *mlpTransformMegaScratch) outputView(out []byte) (metal.MTLBuffer, bool) {
	if s == nil || len(out) == 0 {
		return nil, false
	}
	ptr := uintptr(unsafe.Pointer(&out[0]))
	if s.outView != nil && s.outViewPtr == ptr && s.outViewLen == len(out) {
		return s.outView, true
	}
	s.closeOutputView()
	if buf, ok := registeredPinnedNoCopyBytes(out); ok {
		s.outViewPtr = ptr
		s.outViewLen = len(out)
		s.outView = buf
		s.outPinned = nil
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
	s.outViewPtr = ptr
	s.outViewLen = len(out)
	s.outView = buf
	s.outPinned = pinned
	return buf, true
}

type quantMLPProjView struct {
	packed, scales, biases bufView
	groupSize, bits        int
}

func ffnMegaDefaultGeometry(dModel, dFF int) bool {
	return dModel >= 256 && dFF >= 512
}

var (
	moeArriveZeroOnce sync.Once
	moeArriveZeroBuf  metal.MTLBuffer
)

// moeArriveZeroBuffer returns a shared 4-byte zero buffer — the copy source that resets the FFN
// megakernel's grid-barrier arrive counter INSIDE the encoder (no host write, no wait).
func moeArriveZeroBuffer() metal.MTLBuffer {
	moeArriveZeroOnce.Do(func() {
		moeArriveZeroBuf = device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared)
	})
	return moeArriveZeroBuf
}

// ffnMegaKernelCompatible is the KERNEL truth: the widths the megakernel is specialised for
// (4/8-bit byte-aligned codes via ffnMegaPipelineBits, parity-proven at both) with all three
// projections agreeing so one PSO serves the dispatch.
func ffnMegaKernelCompatible(gate, up, down quantMLPProjView, dModel, dFF int) bool {
	return (gate.bits == 4 || gate.bits == 8) && up.bits == gate.bits && down.bits == gate.bits &&
		gate.groupSize == up.groupSize && gate.groupSize == down.groupSize &&
		gate.groupSize > 0 && dModel%gate.groupSize == 0 && dFF%gate.groupSize == 0
}

// ffnMegaSupported is the ROUTING policy: 8-bit stays unrouted by receipt — on the 26B-A4B's
// b8 local MLP the mega measured 26.9 tok/s vs the qmv trio's 51.9 on the fully-encoded decode
// (#338); its per-byte scalar gemv loses to the steel qmv far more than the 4->1 dispatch
// saving returns. Route a new width only with a geometry receipt that says otherwise.
func ffnMegaSupported(gate, up, down quantMLPProjView, dModel, dFF int) bool {
	return gate.bits == 4 && ffnMegaKernelCompatible(gate, up, down, dModel, dFF)
}

func emitFFNMega[S dispatchSink](sink S, pso metal.MTLComputePipelineState, x metal.MTLBuffer, xOff uint, gate, up, down quantMLPProjView, gated, out metal.MTLBuffer, outOff uint, arrive metal.MTLBuffer, dModel, dFF int) {
	sink.setPSO(pso)
	sink.setBuf(x, xOff, 0)
	sink.setBuf(gate.packed.buf, gate.packed.off, 1)
	sink.setBuf(gate.scales.buf, gate.scales.off, 2)
	sink.setBuf(gate.biases.buf, gate.biases.off, 3)
	sink.setBuf(up.packed.buf, up.packed.off, 4)
	sink.setBuf(up.scales.buf, up.scales.off, 5)
	sink.setBuf(up.biases.buf, up.biases.off, 6)
	sink.setBuf(down.packed.buf, down.packed.off, 7)
	sink.setBuf(down.scales.buf, down.scales.off, 8)
	sink.setBuf(down.biases.buf, down.biases.off, 9)
	sink.setBuf(gated, 0, 10)
	sink.setBuf(out, outOff, 11)
	sink.setBuf(arrive, 0, 12)
	sink.setI32(int32(dModel), 13)
	sink.setI32(int32(dFF), 14)
	sink.setI32(int32(gate.groupSize), 15)
	sink.setI32(ffnMegaNumThreadgroups, 16)
	sink.setI32(ffnMegaMaxSpinIterations, 17)
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: ffnMegaNumThreadgroups, Height: 1, Depth: 1},
		metal.MTLSize{Width: ffnMegaThreadsPerGroup, Height: 1, Depth: 1},
	)
}

// encMoEBlockQuantDevice encodes the WHOLE MoE block — device router top-K, the local MLP, the
// gathered expert MLPs and the norm/combine tail — into the CALLER's live encoder: zero command
// buffers, zero waits, zero host bytes. This is the fully-encoded decode lane (26B-A4B): the
// break-out flow it replaces cost ~3 command buffers per layer per token. handled=false declines
// (non-gather geometry, missing kernels, mega-local, host-only shapes) to the proven break-out
// path, byte-identically. The caller owns BOTH scratches' single-flight lifetime (session-owned;
// recycling across layers/tokens is GPU-GPU in commit order) and pre-validated the router
// geometry (routerTopKUsable + quantMoEDeviceRouterBuffersUsable).
//
// The device-relevant prelude is deliberately a sibling of moeBlockQuantAfterRouterWith
// DeviceIndexBufferPooled's — keep the two in step when weight layouts change.
func encMoEBlockQuantDevice(enc metal.MTLComputeCommandEncoderObject, cb metal.MTLCommandBufferObject, prof *gpuCounterProfiler, routerScratch *routerDeviceScratch, scratch *moeBlockBF16Scratch, hBuf, outputBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel, dFF int, eps float32) (metal.MTLComputeCommandEncoderObject, bool, error) {
	expertDFF, numExperts, topK := w.ExpertDFF, w.NumExperts, w.TopK
	size := dModel * bf16Size
	if hBuf == nil || outputBuf == nil || routerScratch == nil || scratch == nil || topK <= 0 || dModel == 0 || dFF == 0 || expertDFF == 0 {
		return enc, false, nil
	}
	if len(w.PreFFNormW) != size || len(w.PreFFNorm2W) != size || len(w.PostFFNorm1W) != size || len(w.PostFFNorm2W) != size || len(w.PostFFNormW) != size {
		return enc, false, nil
	}
	localGatePacked, localGateScales, localGateBiases, localGateGroupSize, localGateBits, err := quantWeightViewsForShape("native.encMoEBlockQuantDevice: local gate", w.LocalGate, dFF, dModel, w.LocalGroupSize, w.LocalBits)
	if err != nil {
		return enc, false, nil
	}
	localUpPacked, localUpScales, localUpBiases, localUpGroupSize, localUpBits, err := quantWeightViewsForShape("native.encMoEBlockQuantDevice: local up", w.LocalUp, dFF, dModel, w.LocalGroupSize, w.LocalBits)
	if err != nil {
		return enc, false, nil
	}
	localDownPacked, localDownScales, localDownBiases, localDownGroupSize, localDownBits, err := quantWeightViewsForShape("native.encMoEBlockQuantDevice: local down", w.LocalDown, dModel, dFF, w.LocalGroupSize, w.LocalBits)
	if err != nil {
		return enc, false, nil
	}
	// (the break-out path may run the local MLP through the megakernel; this lane always
	// encodes the qmv trio instead — the megakernel's arrive counter is a host write, and
	// three chained qmvs inside the live encoder still beat a broken-out mega dispatch)
	fusedExperts := len(w.ExpGateUp.Packed) > 0
	var expGateUpPacked, expGateUpScales, expGateUpBiases bufView
	var expGatePacked, expGateScales, expGateBiases bufView
	var expUpPacked, expUpScales, expUpBiases bufView
	var expDownPacked, expDownScales, expDownBiases bufView
	var expGateGroupSize, expGateBits, expUpGroupSize, expUpBits, expGateUpGroupSize, expGateUpBits, expDownGroupSize, expDownBits int
	if fusedExperts {
		expGateUpPacked, expGateUpScales, expGateUpBiases, expGateUpGroupSize, expGateUpBits, err = quantWeightViewsForShape("native.encMoEBlockQuantDevice: expert gate_up", w.ExpGateUp, numExperts*2*expertDFF, dModel, w.ExpertGroupSize, w.ExpertBits)
		if err != nil {
			return enc, false, nil
		}
	} else {
		expGatePacked, expGateScales, expGateBiases, expGateGroupSize, expGateBits, err = quantWeightViewsForShape("native.encMoEBlockQuantDevice: expert gate", w.ExpGate, numExperts*expertDFF, dModel, w.ExpertGroupSize, w.ExpertBits)
		if err != nil {
			return enc, false, nil
		}
		expUpPacked, expUpScales, expUpBiases, expUpGroupSize, expUpBits, err = quantWeightViewsForShape("native.encMoEBlockQuantDevice: expert up", w.ExpUp, numExperts*expertDFF, dModel, w.ExpertGroupSize, w.ExpertBits)
		if err != nil {
			return enc, false, nil
		}
	}
	expDownPacked, expDownScales, expDownBiases, expDownGroupSize, expDownBits, err = quantWeightViewsForShape("native.encMoEBlockQuantDevice: expert down", w.ExpDown, numExperts*dModel, expertDFF, w.ExpertGroupSize, w.ExpertBits)
	if err != nil {
		return enc, false, nil
	}
	if !affineBitsSupported(expDownBits) {
		return enc, false, nil
	}
	inGroup, inBits, inRows := expGateGroupSize, expGateBits, expertDFF
	if fusedExperts {
		if !affineBitsSupported(expGateUpBits) {
			return enc, false, nil
		}
		inGroup, inBits, inRows = expGateUpGroupSize, expGateUpBits, 2*expertDFF
	} else if !affineBitsSupported(expGateBits) || expGateBits != expUpBits || expGateGroupSize != expUpGroupSize {
		// gate + up share one gather PSO — width and group size must agree between them.
		return enc, false, nil
	}
	gatherExpertInPSO, err := gatherQMVBF16SteelPipeline(expertDFF, dModel, inGroup, inBits)
	if err != nil {
		return enc, false, nil
	}
	gatherExpertDownPSO, err := gatherQMVBF16SteelPipeline(dModel, expertDFF, expDownGroupSize, expDownBits)
	if err != nil {
		return enc, false, nil
	}
	gatherExpertInMeta, err := gatherQMVBF16Metadata(numExperts, expertDFF, dModel, inGroup, inBits, inRows)
	if err != nil {
		return enc, false, nil
	}
	gatherExpertDownMeta, err := gatherQMVBF16Metadata(numExperts, dModel, expertDFF, expDownGroupSize, expDownBits, dModel)
	if err != nil {
		return enc, false, nil
	}
	localGatePSO, err := pipelineFor(qmvBF16KernelName(dFF, dModel, localGateGroupSize, localGateBits))
	if err != nil {
		return enc, false, nil
	}
	localUpPSO, err := pipelineFor(qmvBF16KernelName(dFF, dModel, localUpGroupSize, localUpBits))
	if err != nil {
		return enc, false, nil
	}
	localDownPSO, err := pipelineFor(qmvBF16KernelName(dModel, dFF, localDownGroupSize, localDownBits))
	if err != nil {
		return enc, false, nil
	}
	rmsPSO, err := pipelineFor(rmsKernelBF16(dModel))
	if err != nil {
		return enc, false, nil
	}
	rmsTG := rmsThreadgroup(dModel, rmsPSO)
	addPSO, err := pipelineFor("vv_Addbfloat16")
	if err != nil {
		return enc, false, nil
	}
	if !gpuHasGeluKernel() {
		return enc, false, nil
	}
	geluPSO, err := geluPipeline()
	if err != nil {
		return enc, false, nil
	}
	scalePSO, err := bf16MulScalarPipeline()
	if err != nil {
		return enc, false, nil
	}
	pre1Buf := bf16WeightView(w.PreFFNormW, w.preFFNormView)
	pre2Buf := bf16WeightView(w.PreFFNorm2W, w.preFFNorm2View)
	post1Buf := bf16WeightView(w.PostFFNorm1W, w.postFFNorm1View)
	post2Buf := bf16WeightView(w.PostFFNorm2W, w.postFFNorm2View)
	postBuf := bf16WeightView(w.PostFFNormW, w.postFFNormView)

	// stage 1: the device router — idx/weights land in routerScratch on device.
	var rerr error
	if enc, rerr = encMoERouterQuantTopK(enc, cb, prof, routerScratch, hBuf, w.RouterNormWScaled, w.routerNormView, w.Router, w.PerExpertScale, w.perExpertScaleView, numExperts, topK, dModel, w.RouterGroupSize, w.RouterBits, eps); rerr != nil {
		return enc, false, nil
	}
	routeIdxBuf, weightsBuf := routerScratch.idxBuf, routerScratch.weightBuf
	msc := scratch.mlp
	sink := encSink{enc}
	// seam splits the live encoder at a stage boundary when the GPU profiler is armed (tests
	// only — prof nil in production): enc AND sink are rebuilt so every emit closure below
	// follows the new encoder.
	seam := func(label string) {
		if prof == nil {
			return
		}
		endEncodingFast(enc)
		enc = prof.encoderFor(cb, label)
		sink = encSink{enc}
	}

	// stage 2: the block body — the break-out run()'s device subset, same kernels, same order.
	emitRMS := func(x, weight, out metal.MTLBuffer, wOff uint) {
		emitRMSNorm(sink, rmsPSO, x, weight, out, wOff, dModel, eps, rmsTG)
	}
	emitQ := func(pso metal.MTLComputePipelineState, wq, scales, biases bufView, x, out metal.MTLBuffer, inDim, outDim int) {
		emitQMV(sink, pso, wq.buf, wq.off, scales.buf, scales.off, biases.buf, biases.off, x, out, 0, inDim, outDim)
	}
	emitGatherQ := func(pso metal.MTLComputePipelineState, meta *gatherQMVBF16Meta, wq, scales, biases bufView, x, out metal.MTLBuffer, route, inDim, outDim, groupSize, bits, rowBase int) {
		emitGatherQMVBF16Steel(sink, pso, meta, x, wq.buf, wq.off, scales.buf, scales.off, biases.buf, biases.off, routeIdxBuf, uint(route*4), out, 0, outDim, inDim, groupSize, bits, rowBase)
	}
	emitScaleAt := func(in metal.MTLBuffer, scalarOff uint, out metal.MTLBuffer, n int) {
		sink.setPSO(scalePSO)
		sink.setBuf(in, 0, 0)
		sink.setBuf(weightsBuf, scalarOff, 1)
		sink.setBuf(out, 0, 2)
		sink.setI32(int32(n), 3)
		group := min(uint(n), uint(256))
		sink.dispatchThreads(
			metal.MTLSize{Width: uint(n), Height: 1, Depth: 1},
			metal.MTLSize{Width: group, Height: 1, Depth: 1},
		)
	}
	emitScaleFromAt := func(in metal.MTLBuffer, inOff, scalarOff uint, out metal.MTLBuffer, n int) {
		sink.setPSO(scalePSO)
		sink.setBuf(in, inOff, 0)
		sink.setBuf(weightsBuf, scalarOff, 1)
		sink.setBuf(out, 0, 2)
		sink.setI32(int32(n), 3)
		group := min(uint(n), uint(256))
		sink.dispatchThreads(
			metal.MTLSize{Width: uint(n), Height: 1, Depth: 1},
			metal.MTLSize{Width: group, Height: 1, Depth: 1},
		)
	}
	emitAdd := func(a, b, out metal.MTLBuffer) {
		emitBinary(sink, addPSO, a, 0, b, 0, out, 0, dModel)
	}
	seam("moe.local")
	emitRMS(hBuf, pre1Buf.buf, scratch.localIn, pre1Buf.off)
	// local MLP: the FFN megakernel when its geometry holds — its grid barrier needs the
	// arrive counter zeroed per dispatch, which the break-out path did with a HOST write;
	// here a 2-element bf16 copy from a device zero buffer resets it INSIDE the encoder
	// (hazard-ordered before the mega's atomics), keeping the whole lane host-free. The
	// qmv trio stays as the fallback for unsupported geometry.
	localGateViewQ := quantMLPProjView{packed: localGatePacked, scales: localGateScales, biases: localGateBiases, groupSize: localGateGroupSize, bits: localGateBits}
	localUpViewQ := quantMLPProjView{packed: localUpPacked, scales: localUpScales, biases: localUpBiases, groupSize: localUpGroupSize, bits: localUpBits}
	localDownViewQ := quantMLPProjView{packed: localDownPacked, scales: localDownScales, biases: localDownBiases, groupSize: localDownGroupSize, bits: localDownBits}
	megaLocal := ffnMegaDefaultGeometry(dModel, dFF) && ffnMegaSupported(localGateViewQ, localUpViewQ, localDownViewQ, dModel, dFF)
	var localMegaPSO metal.MTLComputePipelineState
	if megaLocal {
		if localMegaPSO, err = ffnMegaPipelineBits(localGateViewQ.bits); err != nil {
			megaLocal = false
		}
	}
	if megaLocal {
		if err = scratch.ensureLocalMegaScratch(); err != nil {
			megaLocal = false
		}
	}
	if megaLocal {
		if cerr := encCopyBF16Contig(enc, moeArriveZeroBuffer(), scratch.localMegaArrive, 0, 0, 2); cerr != nil {
			megaLocal = false
		}
	}
	if megaLocal {
		emitFFNMega(sink, localMegaPSO, scratch.localIn, 0, localGateViewQ, localUpViewQ, localDownViewQ, scratch.localMegaGated, scratch.localOut, 0, scratch.localMegaArrive, dModel, dFF)
	} else {
		emitQ(localGatePSO, localGatePacked, localGateScales, localGateBiases, scratch.localIn, msc.gate, dModel, dFF)
		emitQ(localUpPSO, localUpPacked, localUpScales, localUpBiases, scratch.localIn, msc.up, dModel, dFF)
		emitBinary(sink, geluPSO, msc.gate, 0, msc.up, 0, msc.gated, 0, dFF)
		emitQ(localDownPSO, localDownPacked, localDownScales, localDownBiases, msc.gated, scratch.localOut, dFF, dModel)
	}
	seam("moe.expert")
	emitRMS(hBuf, pre2Buf.buf, scratch.expertIn, pre2Buf.off)
	// expert MLPs, ALL routes per dispatch: the MLX gather batch dimension carries every
	// selected expert (grid.z = topK, rhs = the router's device idxBuf), collapsing the
	// old per-route loop's topK×3 matvec dispatches to 3. gate/up share the one normed
	// input (lhs zeros, stride 0); down walks each route's gated row (lhs iota, stride 1).
	// The weighted-sum tail stays per-route (small elementwise ops). Falls back to the
	// per-route loop when the slabs or metadata are unavailable.
	allRoutes := scratch.ensureAllRoutesScratch() == nil
	var inKeyShared, downKeyBatched gatherQMVAllRoutesMetaKey
	var inAllMeta, downAllMeta *gatherQMVBF16Meta
	if allRoutes {
		inGroupSize, inBitsSel, inRowsSel := expGateGroupSize, expGateBits, expertDFF
		if fusedExperts {
			inGroupSize, inBitsSel, inRowsSel = expGateUpGroupSize, expGateUpBits, 2*expertDFF
		}
		inKeyShared = gatherQMVAllRoutesMetaKey{numExperts: numExperts, outDim: expertDFF, inDim: dModel, groupSize: inGroupSize, bits: inBitsSel, expertRows: inRowsSel, routes: topK, batchedX: false}
		downKeyBatched = gatherQMVAllRoutesMetaKey{numExperts: numExperts, outDim: dModel, inDim: expertDFF, groupSize: expDownGroupSize, bits: expDownBits, expertRows: dModel, routes: topK, batchedX: true}
		if inAllMeta, err = gatherQMVAllRoutesMetadata(numExperts, expertDFF, dModel, inGroupSize, inBitsSel, inRowsSel, topK, false); err != nil {
			allRoutes = false
		} else if downAllMeta, err = gatherQMVAllRoutesMetadata(numExperts, dModel, expertDFF, expDownGroupSize, expDownBits, dModel, topK, true); err != nil {
			allRoutes = false
		}
	}
	if allRoutes {
		if fusedExperts {
			emitGatherQMVAllRoutes(sink, gatherExpertInPSO, inAllMeta, inKeyShared, scratch.expertIn, 0, expGateUpPacked.buf, expGateUpPacked.off, expGateUpScales.buf, expGateUpScales.off, expGateUpBiases.buf, expGateUpBiases.off, scratch.routeZeros, routeIdxBuf, 0, scratch.expertGateAll, 0, expertDFF, dModel, expGateUpGroupSize, expGateUpBits, 0, topK)
			emitGatherQMVAllRoutes(sink, gatherExpertInPSO, inAllMeta, inKeyShared, scratch.expertIn, 0, expGateUpPacked.buf, expGateUpPacked.off, expGateUpScales.buf, expGateUpScales.off, expGateUpBiases.buf, expGateUpBiases.off, scratch.routeZeros, routeIdxBuf, 0, scratch.expertUpAll, 0, expertDFF, dModel, expGateUpGroupSize, expGateUpBits, expertDFF, topK)
		} else {
			emitGatherQMVAllRoutes(sink, gatherExpertInPSO, inAllMeta, inKeyShared, scratch.expertIn, 0, expGatePacked.buf, expGatePacked.off, expGateScales.buf, expGateScales.off, expGateBiases.buf, expGateBiases.off, scratch.routeZeros, routeIdxBuf, 0, scratch.expertGateAll, 0, expertDFF, dModel, expGateGroupSize, expGateBits, 0, topK)
			emitGatherQMVAllRoutes(sink, gatherExpertInPSO, inAllMeta, inKeyShared, scratch.expertIn, 0, expUpPacked.buf, expUpPacked.off, expUpScales.buf, expUpScales.off, expUpBiases.buf, expUpBiases.off, scratch.routeZeros, routeIdxBuf, 0, scratch.expertUpAll, 0, expertDFF, dModel, expUpGroupSize, expUpBits, 0, topK)
		}
		emitBinary(sink, geluPSO, scratch.expertGateAll, 0, scratch.expertUpAll, 0, scratch.expertGatedAll, 0, topK*expertDFF)
		emitGatherQMVAllRoutes(sink, gatherExpertDownPSO, downAllMeta, downKeyBatched, scratch.expertGatedAll, 0, expDownPacked.buf, expDownPacked.off, expDownScales.buf, expDownScales.off, expDownBiases.buf, expDownBiases.off, scratch.routeIota, routeIdxBuf, 0, scratch.expertDownAll, 0, dModel, expertDFF, expDownGroupSize, expDownBits, 0, topK)
		seam("moe.tail")
		for i := range topK {
			if i == 0 {
				emitScaleFromAt(scratch.expertDownAll, uint(i*dModel*bf16Size), uint(i*bf16Size), scratch.expertAcc, dModel)
			} else {
				emitScaleFromAt(scratch.expertDownAll, uint(i*dModel*bf16Size), uint(i*bf16Size), scratch.expertScaled, dModel)
				emitAdd(scratch.expertAcc, scratch.expertScaled, scratch.expertAcc)
			}
		}
	} else {
		for i := range topK {
			if fusedExperts {
				emitGatherQ(gatherExpertInPSO, gatherExpertInMeta, expGateUpPacked, expGateUpScales, expGateUpBiases, scratch.expertIn, msc.gate, i, dModel, expertDFF, expGateUpGroupSize, expGateUpBits, 0)
				emitGatherQ(gatherExpertInPSO, gatherExpertInMeta, expGateUpPacked, expGateUpScales, expGateUpBiases, scratch.expertIn, msc.up, i, dModel, expertDFF, expGateUpGroupSize, expGateUpBits, expertDFF)
			} else {
				emitGatherQ(gatherExpertInPSO, gatherExpertInMeta, expGatePacked, expGateScales, expGateBiases, scratch.expertIn, msc.gate, i, dModel, expertDFF, expGateGroupSize, expGateBits, 0)
				emitGatherQ(gatherExpertInPSO, gatherExpertInMeta, expUpPacked, expUpScales, expUpBiases, scratch.expertIn, msc.up, i, dModel, expertDFF, expUpGroupSize, expUpBits, 0)
			}
			emitBinary(sink, geluPSO, msc.gate, 0, msc.up, 0, msc.gated, 0, expertDFF)
			emitGatherQ(gatherExpertDownPSO, gatherExpertDownMeta, expDownPacked, expDownScales, expDownBiases, msc.gated, msc.down, i, expertDFF, dModel, expDownGroupSize, expDownBits, 0)
			if i == 0 {
				emitScaleAt(msc.down, uint(i*bf16Size), scratch.expertAcc, dModel)
			} else {
				emitScaleAt(msc.down, uint(i*bf16Size), scratch.expertScaled, dModel)
				emitAdd(scratch.expertAcc, scratch.expertScaled, scratch.expertAcc)
			}
		}
	}
	emitRMS(scratch.localOut, post1Buf.buf, scratch.localNormed, post1Buf.off)
	emitRMS(scratch.expertAcc, post2Buf.buf, scratch.expertNormed, post2Buf.off)
	emitAdd(scratch.localNormed, scratch.expertNormed, scratch.combined)
	emitRMS(scratch.combined, postBuf.buf, scratch.ffResidual, postBuf.off)
	emitAdd(hBuf, scratch.ffResidual, outputBuf)
	return enc, true, nil
}

func quantWeightViewsForShape(fn string, w QuantWeight, outDim, inDim, groupSize, bits int) (bufView, bufView, bufView, int, int, error) {
	groupSize, bits = quantWeightGeometryForShape(w, outDim, inDim, groupSize, bits)
	if groupSize <= 0 || bits <= 0 || inDim%groupSize != 0 {
		return bufView{}, bufView{}, bufView{}, 0, 0, core.NewError(fn + ": invalid quant geometry")
	}
	wantPacked := outDim * inDim * bits / 8
	wantScales := outDim * (inDim / groupSize) * bf16Size
	if len(w.Packed) != wantPacked || len(w.Scales) != wantScales || len(w.Biases) != wantScales {
		return bufView{}, bufView{}, bufView{}, 0, 0, core.NewError(fn + ": quant weight size mismatch")
	}
	packed, scales, biases := quantWeightViews(w)
	return packed, scales, biases, groupSize, bits, nil
}

func moeBlockQuantAfterRouter(h []byte, idx []int32, weights []byte, weightBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockQuantAfterRouterWithBuffer(h, nil, idx, weights, weightBuf, w, dModel, dFF, eps)
}

func moeBlockQuantAfterRouterWithBuffer(h []byte, hBuf metal.MTLBuffer, idx []int32, weights []byte, weightBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockQuantAfterRouterWithBufferPooled(h, hBuf, nil, nil, idx, weights, weightBuf, w, dModel, dFF, eps, true, false)
}

func moeBlockQuantAfterRouterWithBufferInPool(h []byte, hBuf metal.MTLBuffer, idx []int32, weights []byte, weightBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockQuantAfterRouterWithBufferPooled(h, hBuf, nil, nil, idx, weights, weightBuf, w, dModel, dFF, eps, false, false)
}

func moeBlockQuantAfterRouterWithBufferInto(out []byte, h []byte, hBuf metal.MTLBuffer, idx []int32, weights []byte, weightBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockQuantAfterRouterWithBufferPooled(h, hBuf, out, nil, idx, weights, weightBuf, w, dModel, dFF, eps, true, true)
}

func moeBlockQuantAfterRouterWithBufferIntoInPool(out []byte, h []byte, hBuf metal.MTLBuffer, idx []int32, weights []byte, weightBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockQuantAfterRouterWithBufferPooled(h, hBuf, out, nil, idx, weights, weightBuf, w, dModel, dFF, eps, false, true)
}

func moeBlockQuantAfterRouterWithBufferOutputInPool(h []byte, hBuf, outputBuf metal.MTLBuffer, idx []int32, weights []byte, weightBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel, dFF int, eps float32) error {
	if outputBuf == nil {
		return core.NewError("native.moeBlockQuantAfterRouter: output buffer is nil")
	}
	_, err := moeBlockQuantAfterRouterWithBufferPooled(h, hBuf, nil, outputBuf, idx, weights, weightBuf, w, dModel, dFF, eps, false, false)
	return err
}

func moeBlockQuantAfterRouterWithBufferPooled(h []byte, hBuf metal.MTLBuffer, out []byte, outputBuf metal.MTLBuffer, idx []int32, weights []byte, weightBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel, dFF int, eps float32, useAutoreleasePool bool, useCallerOut bool) ([]byte, error) {
	return moeBlockQuantAfterRouterWithDeviceIndexBufferPooled(h, hBuf, out, outputBuf, idx, nil, weights, weightBuf, w, dModel, dFF, eps, useAutoreleasePool, useCallerOut, nil)
}

func moeBlockQuantAfterRouterWithDeviceIndexBufferOutputInPool(h []byte, hBuf, outputBuf metal.MTLBuffer, idx []int32, idxBuf metal.MTLBuffer, weights []byte, weightBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel, dFF int, eps float32, ownedScratch *moeBlockBF16Scratch) error {
	if outputBuf == nil {
		return core.NewError("native.moeBlockQuantAfterRouter: output buffer is nil")
	}
	_, err := moeBlockQuantAfterRouterWithDeviceIndexBufferPooled(h, hBuf, nil, outputBuf, idx, idxBuf, weights, weightBuf, w, dModel, dFF, eps, false, false, ownedScratch)
	return err
}

func moeBlockQuantAfterRouterWithDeviceIndexBufferPooled(h []byte, hBuf metal.MTLBuffer, out []byte, outputBuf metal.MTLBuffer, idx []int32, idxBuf metal.MTLBuffer, weights []byte, weightBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel, dFF int, eps float32, useAutoreleasePool bool, useCallerOut bool, ownedScratch *moeBlockBF16Scratch) ([]byte, error) {
	expertDFF, numExperts, topK := w.ExpertDFF, w.NumExperts, w.TopK
	size := dModel * bf16Size
	if hBuf == nil && h == nil {
		return nil, core.NewError("native.moeBlockQuantAfterRouter: h bytes or hBuf required")
	}
	if h != nil && len(h) != size {
		return nil, core.NewError("native.moeBlockQuantAfterRouter: h must be dModel bf16 bytes")
	}
	idxOnDevice := idxBuf != nil
	weightsOnDevice := weightBuf != nil
	if (!idxOnDevice && len(idx) != topK) || (idxOnDevice && idx != nil && len(idx) != topK) || (!weightsOnDevice && len(weights) != topK*bf16Size) || (weightsOnDevice && weights != nil && len(weights) != topK*bf16Size) {
		return nil, core.NewError("native.moeBlockQuantAfterRouter: idx/weights length must equal topK")
	}
	if len(w.PreFFNormW) != size || len(w.PreFFNorm2W) != size || len(w.PostFFNorm1W) != size || len(w.PostFFNorm2W) != size || len(w.PostFFNormW) != size {
		return nil, core.NewError("native.moeBlockQuantAfterRouter: norm weights must be dModel bf16 bytes")
	}
	localGatePacked, localGateScales, localGateBiases, localGateGroupSize, localGateBits, err := quantWeightViewsForShape("native.moeBlockQuantAfterRouter: local gate", w.LocalGate, dFF, dModel, w.LocalGroupSize, w.LocalBits)
	if err != nil {
		return nil, err
	}
	localUpPacked, localUpScales, localUpBiases, localUpGroupSize, localUpBits, err := quantWeightViewsForShape("native.moeBlockQuantAfterRouter: local up", w.LocalUp, dFF, dModel, w.LocalGroupSize, w.LocalBits)
	if err != nil {
		return nil, err
	}
	localDownPacked, localDownScales, localDownBiases, localDownGroupSize, localDownBits, err := quantWeightViewsForShape("native.moeBlockQuantAfterRouter: local down", w.LocalDown, dModel, dFF, w.LocalGroupSize, w.LocalBits)
	if err != nil {
		return nil, err
	}
	localGateView := quantMLPProjView{packed: localGatePacked, scales: localGateScales, biases: localGateBiases, groupSize: localGateGroupSize, bits: localGateBits}
	localUpView := quantMLPProjView{packed: localUpPacked, scales: localUpScales, biases: localUpBiases, groupSize: localUpGroupSize, bits: localUpBits}
	localDownView := quantMLPProjView{packed: localDownPacked, scales: localDownScales, biases: localDownBiases, groupSize: localDownGroupSize, bits: localDownBits}

	fusedExperts := len(w.ExpGateUp.Packed) > 0
	expertGatePackedPer, expertGateScalePer := 0, 0
	expertDownPackedPer, expertDownScalePer := 0, 0
	var expGatePacked, expGateScales, expGateBiases bufView
	var expUpPacked, expUpScales, expUpBiases bufView
	var expGateUpPacked, expGateUpScales, expGateUpBiases bufView
	var expDownPacked, expDownScales, expDownBiases bufView
	var expGateGroupSize, expGateBits, expUpGroupSize, expUpBits, expGateUpGroupSize, expGateUpBits, expDownGroupSize, expDownBits int
	if fusedExperts {
		expGateUpPacked, expGateUpScales, expGateUpBiases, expGateUpGroupSize, expGateUpBits, err = quantWeightViewsForShape("native.moeBlockQuantAfterRouter: expert gate_up", w.ExpGateUp, numExperts*2*expertDFF, dModel, w.ExpertGroupSize, w.ExpertBits)
		if err != nil {
			return nil, err
		}
		expDownPacked, expDownScales, expDownBiases, expDownGroupSize, expDownBits, err = quantWeightViewsForShape("native.moeBlockQuantAfterRouter: expert down", w.ExpDown, numExperts*dModel, expertDFF, w.ExpertGroupSize, w.ExpertBits)
		if err != nil {
			return nil, err
		}
	} else {
		expGatePacked, expGateScales, expGateBiases, expGateGroupSize, expGateBits, err = quantWeightViewsForShape("native.moeBlockQuantAfterRouter: expert gate", w.ExpGate, numExperts*expertDFF, dModel, w.ExpertGroupSize, w.ExpertBits)
		if err != nil {
			return nil, err
		}
		expUpPacked, expUpScales, expUpBiases, expUpGroupSize, expUpBits, err = quantWeightViewsForShape("native.moeBlockQuantAfterRouter: expert up", w.ExpUp, numExperts*expertDFF, dModel, w.ExpertGroupSize, w.ExpertBits)
		if err != nil {
			return nil, err
		}
		expDownPacked, expDownScales, expDownBiases, expDownGroupSize, expDownBits, err = quantWeightViewsForShape("native.moeBlockQuantAfterRouter: expert down", w.ExpDown, numExperts*dModel, expertDFF, w.ExpertGroupSize, w.ExpertBits)
		if err != nil {
			return nil, err
		}
	}
	if expGateGroupSize > 0 {
		expertGatePackedPer = expertDFF * dModel * expGateBits / 8
		expertGateScalePer = expertDFF * (dModel / expGateGroupSize) * bf16Size
	}
	if expGateUpGroupSize > 0 {
		expertGatePackedPer = expertDFF * dModel * expGateUpBits / 8
		expertGateScalePer = expertDFF * (dModel / expGateUpGroupSize) * bf16Size
	}
	if expDownGroupSize > 0 {
		expertDownPackedPer = dModel * expertDFF * expDownBits / 8
		expertDownScalePer = dModel * (expertDFF / expDownGroupSize) * bf16Size
	}
	if !idxOnDevice {
		for i := range idx {
			if idx[i] < 0 || int(idx[i]) >= numExperts {
				return nil, core.NewError("native.moeBlockQuantAfterRouter: expert index out of range")
			}
		}
	}

	bufferOut := outputBuf != nil
	callerOut := !bufferOut && useCallerOut && cap(out) >= size
	if bufferOut {
		out = nil
	} else if callerOut {
		out = out[:size]
	} else {
		out = make([]byte, size)
	}
	if dModel == 0 || dFF == 0 || expertDFF == 0 {
		if bufferOut && size > 0 {
			clear(unsafe.Slice((*byte)(outputBuf.Contents()), size))
			return nil, nil
		}
		if !bufferOut {
			clear(out)
		}
		return out, nil
	}
	qmvPSO := func(outDim, inDim, groupSize, bits int) (metal.MTLComputePipelineState, error) {
		return pipelineFor(qmvBF16KernelName(outDim, inDim, groupSize, bits))
	}
	useLocalMega := ffnMegaDefaultGeometry(dModel, dFF) && ffnMegaSupported(localGateView, localUpView, localDownView, dModel, dFF)
	var localMegaPSO metal.MTLComputePipelineState
	if useLocalMega {
		localMegaPSO, err = ffnMegaPipelineBits(localGateView.bits)
		if err != nil {
			useLocalMega = false
		}
	}
	var localGatePSO, localUpPSO, localDownPSO metal.MTLComputePipelineState
	if !useLocalMega {
		localGatePSO, err = qmvPSO(dFF, dModel, localGateGroupSize, localGateBits)
		if err != nil {
			return nil, err
		}
		localUpPSO, err = qmvPSO(dFF, dModel, localUpGroupSize, localUpBits)
		if err != nil {
			return nil, err
		}
		localDownPSO, err = qmvPSO(dModel, dFF, localDownGroupSize, localDownBits)
		if err != nil {
			return nil, err
		}
	}
	hostIdxAvailable := len(idx) == topK
	useGatherExperts := (idxBuf != nil || hostIdxAvailable) && topK > 0 && affineBitsSupported(expDownBits)
	if fusedExperts {
		useGatherExperts = useGatherExperts && affineBitsSupported(expGateUpBits)
	} else {
		useGatherExperts = useGatherExperts && affineBitsSupported(expGateBits) && expGateBits == expUpBits && expGateGroupSize == expUpGroupSize
	}
	var gatherExpertInPSO, gatherExpertDownPSO metal.MTLComputePipelineState
	var gatherExpertInMeta, gatherExpertDownMeta *gatherQMVBF16Meta
	if useGatherExperts {
		inGroup, inBits := expGateGroupSize, expGateBits
		inRows := expertDFF
		if fusedExperts {
			inGroup, inBits = expGateUpGroupSize, expGateUpBits
			inRows = 2 * expertDFF
		}
		gatherExpertInPSO, err = gatherQMVBF16SteelPipeline(expertDFF, dModel, inGroup, inBits)
		if err == nil {
			gatherExpertDownPSO, err = gatherQMVBF16SteelPipeline(dModel, expertDFF, expDownGroupSize, expDownBits)
		}
		if err == nil {
			gatherExpertInMeta, err = gatherQMVBF16Metadata(numExperts, expertDFF, dModel, inGroup, inBits, inRows)
		}
		if err == nil {
			gatherExpertDownMeta, err = gatherQMVBF16Metadata(numExperts, dModel, expertDFF, expDownGroupSize, expDownBits, dModel)
		}
		if err != nil {
			useGatherExperts = false
		}
	}
	if !useGatherExperts && len(idx) != topK {
		return nil, core.NewError("native.moeBlockQuantAfterRouter: host idx required when gathered device expert routing is unavailable")
	}
	var expGatePSO, expUpPSO, expGateUpPSO, expDownPSO metal.MTLComputePipelineState
	if !useGatherExperts {
		if fusedExperts {
			expGateUpPSO, err = qmvPSO(expertDFF, dModel, expGateUpGroupSize, expGateUpBits)
			if err != nil {
				return nil, err
			}
		} else {
			expGatePSO, err = qmvPSO(expertDFF, dModel, expGateGroupSize, expGateBits)
			if err != nil {
				return nil, err
			}
			expUpPSO, err = qmvPSO(expertDFF, dModel, expUpGroupSize, expUpBits)
			if err != nil {
				return nil, err
			}
		}
		expDownPSO, err = qmvPSO(dModel, expertDFF, expDownGroupSize, expDownBits)
		if err != nil {
			return nil, err
		}
	}
	rmsPSO, err := pipelineFor(rmsKernelBF16(dModel))
	if err != nil {
		return nil, err
	}
	rmsTG := rmsThreadgroup(dModel, rmsPSO)
	addPSO, err := pipelineFor("vv_Addbfloat16")
	if err != nil {
		return nil, err
	}
	var geluPSO metal.MTLComputePipelineState
	useFusedGelu := gpuHasGeluKernel()
	if useFusedGelu {
		geluPSO, err = geluPipeline()
		if err != nil {
			return nil, err
		}
	}
	scalePSO, scaleErr := bf16MulScalarPipeline()
	if scaleErr != nil && len(weights) != topK*bf16Size {
		return nil, core.NewError("native.moeBlockQuantAfterRouter: host weights required when device scalar scaling is unavailable")
	}
	pre1Buf := bf16WeightView(w.PreFFNormW, w.preFFNormView)
	pre2Buf := bf16WeightView(w.PreFFNorm2W, w.preFFNorm2View)
	post1Buf := bf16WeightView(w.PostFFNorm1W, w.postFFNorm1View)
	post2Buf := bf16WeightView(w.PostFFNorm2W, w.postFFNorm2View)
	postBuf := bf16WeightView(w.PostFFNormW, w.postFFNormView)

	var encErr error
	run := func() {
		var err error
		scratch := ownedScratch
		if scratch == nil {
			scratch, err = getMoEBlockBF16Scratch(dModel, dFF, expertDFF, topK)
			if err != nil {
				encErr = err
				return
			}
			defer putMoEBlockBF16Scratch(scratch)
		}
		routeIdxBuf := idxBuf
		if useGatherExperts && routeIdxBuf == nil {
			var ok bool
			routeIdxBuf, ok = scratch.indexView(idx)
			if !ok {
				idxBytes := unsafe.Slice((*byte)(unsafe.Pointer(&idx[0])), len(idx)*4)
				routeIdxBuf, err = scratch.idx.copyBuffer(idxBytes)
				runtime.KeepAlive(idx)
				if err != nil {
					encErr = err
					return
				}
			}
		}
		inputBuf := hBuf
		if inputBuf == nil {
			var ok bool
			inputBuf, ok = scratch.inputView(h)
			if !ok {
				inputBuf, err = scratch.h.copyBuffer(h)
				if err != nil {
					encErr = err
					return
				}
			}
		}
		weightsBuf := weightBuf
		if topK > 0 {
			if weightsBuf == nil {
				var ok bool
				weightsBuf, ok = scratch.weightsView(weights)
				if !ok {
					weightsBuf, err = scratch.weights.copyBuffer(weights)
					if err != nil {
						encErr = err
						return
					}
				}
			}
		} else {
			clear(unsafe.Slice((*byte)(scratch.expertAcc.Contents()), size))
		}
		msc := scratch.mlp
		if useLocalMega {
			if err = scratch.ensureLocalMegaScratch(); err != nil {
				encErr = err
				return
			}
			*scratch.localMegaArrivePtr = 0
		}
		finalOutBuf := scratch.out.buf
		directOut := false
		if bufferOut {
			finalOutBuf = outputBuf
			directOut = true
		} else if callerOut {
			if tmp, ok := scratch.outputView(out); ok {
				finalOutBuf = tmp
				directOut = true
			}
		}

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		sink := encSink{enc}
		emitRMS := func(x, weight, out metal.MTLBuffer, wOff uint) {
			emitRMSNorm(sink, rmsPSO, x, weight, out, wOff, dModel, eps, rmsTG)
		}
		emitQ := func(pso metal.MTLComputePipelineState, wq, scales, biases, x, out metal.MTLBuffer, wqOff, scalesOff, biasesOff, outOff uint, inDim, outDim int) {
			emitQMV(sink, pso, wq, wqOff, scales, scalesOff, biases, biasesOff, x, out, outOff, inDim, outDim)
		}
		emitGatherQ := func(pso metal.MTLComputePipelineState, meta *gatherQMVBF16Meta, wq, scales, biases, x, out metal.MTLBuffer, wqOff, scalesOff, biasesOff uint, route int, inDim, outDim, groupSize, bits, rowBase int) {
			emitGatherQMVBF16Steel(sink, pso, meta, x, wq, wqOff, scales, scalesOff, biases, biasesOff, routeIdxBuf, uint(route*4), out, 0, outDim, inDim, groupSize, bits, rowBase)
		}
		emitGelu := func(gate, up, out metal.MTLBuffer, n int) error {
			if useFusedGelu {
				emitBinary(sink, geluPSO, gate, 0, up, 0, out, 0, n)
				return nil
			}
			return encGeluGateMul(enc, gate, up, out, msc, n)
		}
		emitScale := func(in, scalar, out metal.MTLBuffer, scalarOffset uint, scalarBytes []byte, n int) error {
			if scaleErr != nil {
				return encScaleBF16(enc, in, scalar, out, scalarOffset, scalarBytes, n)
			}
			sink.setPSO(scalePSO)
			sink.setBuf(in, 0, 0)
			sink.setBuf(scalar, scalarOffset, 1)
			sink.setBuf(out, 0, 2)
			sink.setI32(int32(n), 3)
			group := min(uint(n), uint(256))
			sink.dispatchThreads(
				metal.MTLSize{Width: uint(n), Height: 1, Depth: 1},
				metal.MTLSize{Width: group, Height: 1, Depth: 1},
			)
			return nil
		}
		emitAdd := func(a, b, out metal.MTLBuffer) {
			emitBinary(sink, addPSO, a, 0, b, 0, out, 0, dModel)
		}
		emitRMS(inputBuf, pre1Buf.buf, scratch.localIn, pre1Buf.off)
		if useLocalMega {
			emitFFNMega(sink, localMegaPSO, scratch.localIn, 0, localGateView, localUpView, localDownView, scratch.localMegaGated, scratch.localOut, 0, scratch.localMegaArrive, dModel, dFF)
		} else {
			emitQ(localGatePSO, localGatePacked.buf, localGateScales.buf, localGateBiases.buf, scratch.localIn, msc.gate, localGatePacked.off, localGateScales.off, localGateBiases.off, 0, dModel, dFF)
			emitQ(localUpPSO, localUpPacked.buf, localUpScales.buf, localUpBiases.buf, scratch.localIn, msc.up, localUpPacked.off, localUpScales.off, localUpBiases.off, 0, dModel, dFF)
			if encErr = emitGelu(msc.gate, msc.up, msc.gated, dFF); encErr != nil {
				endEncodingFast(enc)
				return
			}
			emitQ(localDownPSO, localDownPacked.buf, localDownScales.buf, localDownBiases.buf, msc.gated, scratch.localOut, localDownPacked.off, localDownScales.off, localDownBiases.off, 0, dFF, dModel)
		}
		emitRMS(inputBuf, pre2Buf.buf, scratch.expertIn, pre2Buf.off)
		for i := range topK {
			if useGatherExperts {
				if fusedExperts {
					emitGatherQ(gatherExpertInPSO, gatherExpertInMeta, expGateUpPacked.buf, expGateUpScales.buf, expGateUpBiases.buf, scratch.expertIn, msc.gate, expGateUpPacked.off, expGateUpScales.off, expGateUpBiases.off, i, dModel, expertDFF, expGateUpGroupSize, expGateUpBits, 0)
					emitGatherQ(gatherExpertInPSO, gatherExpertInMeta, expGateUpPacked.buf, expGateUpScales.buf, expGateUpBiases.buf, scratch.expertIn, msc.up, expGateUpPacked.off, expGateUpScales.off, expGateUpBiases.off, i, dModel, expertDFF, expGateUpGroupSize, expGateUpBits, expertDFF)
				} else {
					emitGatherQ(gatherExpertInPSO, gatherExpertInMeta, expGatePacked.buf, expGateScales.buf, expGateBiases.buf, scratch.expertIn, msc.gate, expGatePacked.off, expGateScales.off, expGateBiases.off, i, dModel, expertDFF, expGateGroupSize, expGateBits, 0)
					emitGatherQ(gatherExpertInPSO, gatherExpertInMeta, expUpPacked.buf, expUpScales.buf, expUpBiases.buf, scratch.expertIn, msc.up, expUpPacked.off, expUpScales.off, expUpBiases.off, i, dModel, expertDFF, expUpGroupSize, expUpBits, 0)
				}
			} else {
				e := int(idx[i])
				if fusedExperts {
					gatePackedOff, gateScaleOff := uint(e*2*expertGatePackedPer), uint(e*2*expertGateScalePer)
					upPackedOff, upScaleOff := gatePackedOff+uint(expertGatePackedPer), gateScaleOff+uint(expertGateScalePer)
					emitQ(expGateUpPSO, expGateUpPacked.buf, expGateUpScales.buf, expGateUpBiases.buf, scratch.expertIn, msc.gate, expGateUpPacked.off+gatePackedOff, expGateUpScales.off+gateScaleOff, expGateUpBiases.off+gateScaleOff, 0, dModel, expertDFF)
					emitQ(expGateUpPSO, expGateUpPacked.buf, expGateUpScales.buf, expGateUpBiases.buf, scratch.expertIn, msc.up, expGateUpPacked.off+upPackedOff, expGateUpScales.off+upScaleOff, expGateUpBiases.off+upScaleOff, 0, dModel, expertDFF)
				} else {
					gatePackedOff, gateScaleOff := uint(e*expertGatePackedPer), uint(e*expertGateScalePer)
					emitQ(expGatePSO, expGatePacked.buf, expGateScales.buf, expGateBiases.buf, scratch.expertIn, msc.gate, expGatePacked.off+gatePackedOff, expGateScales.off+gateScaleOff, expGateBiases.off+gateScaleOff, 0, dModel, expertDFF)
					emitQ(expUpPSO, expUpPacked.buf, expUpScales.buf, expUpBiases.buf, scratch.expertIn, msc.up, expUpPacked.off+gatePackedOff, expUpScales.off+gateScaleOff, expUpBiases.off+gateScaleOff, 0, dModel, expertDFF)
				}
			}
			if encErr = emitGelu(msc.gate, msc.up, msc.gated, expertDFF); encErr != nil {
				endEncodingFast(enc)
				return
			}
			if useGatherExperts {
				emitGatherQ(gatherExpertDownPSO, gatherExpertDownMeta, expDownPacked.buf, expDownScales.buf, expDownBiases.buf, msc.gated, msc.down, expDownPacked.off, expDownScales.off, expDownBiases.off, i, expertDFF, dModel, expDownGroupSize, expDownBits, 0)
			} else {
				e := int(idx[i])
				downPackedOff, downScaleOff := uint(e*expertDownPackedPer), uint(e*expertDownScalePer)
				emitQ(expDownPSO, expDownPacked.buf, expDownScales.buf, expDownBiases.buf, msc.gated, msc.down, expDownPacked.off+downPackedOff, expDownScales.off+downScaleOff, expDownBiases.off+downScaleOff, 0, expertDFF, dModel)
			}
			var weightBytes []byte
			if len(weights) >= (i+1)*bf16Size {
				weightBytes = weights[i*bf16Size : (i+1)*bf16Size]
			}
			if i == 0 {
				if encErr = emitScale(msc.down, weightsBuf, scratch.expertAcc, uint(i*bf16Size), weightBytes, dModel); encErr != nil {
					endEncodingFast(enc)
					return
				}
			} else {
				if encErr = emitScale(msc.down, weightsBuf, scratch.expertScaled, uint(i*bf16Size), weightBytes, dModel); encErr != nil {
					endEncodingFast(enc)
					return
				}
				emitAdd(scratch.expertAcc, scratch.expertScaled, scratch.expertAcc)
			}
		}
		emitRMS(scratch.localOut, post1Buf.buf, scratch.localNormed, post1Buf.off)
		emitRMS(scratch.expertAcc, post2Buf.buf, scratch.expertNormed, post2Buf.off)
		emitAdd(scratch.localNormed, scratch.expertNormed, scratch.combined)
		emitRMS(scratch.combined, postBuf.buf, scratch.ffResidual, postBuf.off)
		emitAdd(inputBuf, scratch.ffResidual, finalOutBuf)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		// The decode loop's fully-device path (session-OWNED scratch, device h/idx/weights,
		// buffer output, no host clears or mega arrive-counter) skips the completion wait:
		// the queue orders every later consumer, the owner guarantees single-flight scratch
		// reuse (GPU-GPU, commit-ordered), and no host bytes are read after this point. Every
		// other shape keeps the wait — pooled scratch lifetimes and host readbacks assume it.
		skipWait := ownedScratch != nil && bufferOut && hBuf != nil && idxBuf != nil && weightBuf != nil &&
			topK > 0 && !useLocalMega && scaleErr == nil
		if !skipWait {
			waitUntilCompletedFast(cb)
		}
		if !directOut {
			copy(out, scratch.out.bytes[:size])
		}
	}
	if useAutoreleasePool {
		withAutoreleasePool(run)
	} else {
		run()
	}
	return out, encErr
}

// mlpTransformQuant is mlpTransformBF16 for a 4-bit MLP: gate/up (dModel→dFF) and down
// (dFF→dModel) via resident quant QMVBF16, with the SwiGLU activation between — no
// residual. The local quant weights are fixed per layer, so their packed/scales/biases
// buffers follow the same resident route as selected quant expert slices.
func mlpTransformQuant(x []byte, gate, up, down QuantWeight, dModel, dFF, groupSize, bits int) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != dModel*bf16Size {
		return nil, core.NewError("native.mlpTransformQuant: x must be dModel bf16 bytes")
	}
	if dModel == 0 || dFF == 0 {
		return make([]byte, dModel*bf16Size), nil
	}
	gateView, upView, downView, err := mlpTransformQuantViews("native.mlpTransformQuant", gate, up, down, dModel, dFF, groupSize, bits)
	if err != nil {
		return nil, err
	}
	if ffnMegaDefaultGeometry(dModel, dFF) {
		if out, ok, err := mlpTransformQuantMegaWithViews(x, gateView, upView, downView, dModel, dFF); ok || err != nil {
			return out, err
		}
	}
	return mlpTransformQuantComposedWithViews(x, gateView, upView, downView, dModel, dFF)
}

func mlpTransformQuantComposed(x []byte, gate, up, down QuantWeight, dModel, dFF, groupSize, bits int) ([]byte, error) {
	return mlpTransformQuantComposedIntoInternal(nil, x, gate, up, down, dModel, dFF, groupSize, bits, false)
}

func mlpTransformQuantComposedInto(out []byte, x []byte, gate, up, down QuantWeight, dModel, dFF, groupSize, bits int) ([]byte, error) {
	return mlpTransformQuantComposedIntoInternal(out, x, gate, up, down, dModel, dFF, groupSize, bits, true)
}

func mlpTransformQuantComposedIntoInternal(out []byte, x []byte, gate, up, down QuantWeight, dModel, dFF, groupSize, bits int, useCallerOut bool) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != dModel*bf16Size {
		return nil, core.NewError("native.mlpTransformQuant: x must be dModel bf16 bytes")
	}
	outLen := dModel * bf16Size
	callerOut := useCallerOut && cap(out) >= outLen
	if callerOut {
		out = out[:outLen]
	} else {
		out = make([]byte, outLen)
	}
	if dModel == 0 || dFF == 0 {
		clear(out)
		return out, nil
	}
	gateView, upView, downView, err := mlpTransformQuantViews("native.mlpTransformQuant", gate, up, down, dModel, dFF, groupSize, bits)
	if err != nil {
		return nil, err
	}
	return mlpTransformQuantComposedWithViewsInto(out, x, gateView, upView, downView, dModel, dFF, callerOut)
}

func mlpTransformQuantMega(x []byte, gate, up, down QuantWeight, dModel, dFF, groupSize, bits int) ([]byte, error) {
	return mlpTransformQuantMegaIntoInternal(nil, x, gate, up, down, dModel, dFF, groupSize, bits, false)
}

func mlpTransformQuantMegaInto(out []byte, x []byte, gate, up, down QuantWeight, dModel, dFF, groupSize, bits int) ([]byte, error) {
	return mlpTransformQuantMegaIntoInternal(out, x, gate, up, down, dModel, dFF, groupSize, bits, true)
}

func mlpTransformQuantMegaIntoInternal(out []byte, x []byte, gate, up, down QuantWeight, dModel, dFF, groupSize, bits int, useCallerOut bool) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != dModel*bf16Size {
		return nil, core.NewError("native.mlpTransformQuantMega: x must be dModel bf16 bytes")
	}
	outLen := dModel * bf16Size
	callerOut := useCallerOut && cap(out) >= outLen
	if callerOut {
		out = out[:outLen]
	} else {
		out = make([]byte, outLen)
	}
	if dModel == 0 || dFF == 0 {
		clear(out)
		return out, nil
	}
	gateView, upView, downView, err := mlpTransformQuantViews("native.mlpTransformQuantMega", gate, up, down, dModel, dFF, groupSize, bits)
	if err != nil {
		return nil, err
	}
	out, ok, err := mlpTransformQuantMegaWithViewsInto(out, x, gateView, upView, downView, dModel, dFF, callerOut)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, core.NewError("native.mlpTransformQuantMega: unsupported quant geometry or megakernel unavailable")
	}
	return out, nil
}

func mlpTransformQuantViews(fn string, gate, up, down QuantWeight, dModel, dFF, groupSize, bits int) (quantMLPProjView, quantMLPProjView, quantMLPProjView, error) {
	gatePacked, gateScales, gateBiases, gateGroupSize, gateBits, err := quantWeightViewsForShape(fn+": gate", gate, dFF, dModel, groupSize, bits)
	if err != nil {
		return quantMLPProjView{}, quantMLPProjView{}, quantMLPProjView{}, err
	}
	upPacked, upScales, upBiases, upGroupSize, upBits, err := quantWeightViewsForShape(fn+": up", up, dFF, dModel, groupSize, bits)
	if err != nil {
		return quantMLPProjView{}, quantMLPProjView{}, quantMLPProjView{}, err
	}
	downPacked, downScales, downBiases, downGroupSize, downBits, err := quantWeightViewsForShape(fn+": down", down, dModel, dFF, groupSize, bits)
	if err != nil {
		return quantMLPProjView{}, quantMLPProjView{}, quantMLPProjView{}, err
	}
	return quantMLPProjView{packed: gatePacked, scales: gateScales, biases: gateBiases, groupSize: gateGroupSize, bits: gateBits},
		quantMLPProjView{packed: upPacked, scales: upScales, biases: upBiases, groupSize: upGroupSize, bits: upBits},
		quantMLPProjView{packed: downPacked, scales: downScales, biases: downBiases, groupSize: downGroupSize, bits: downBits},
		nil
}

func mlpTransformQuantMegaWithViews(x []byte, gate, up, down quantMLPProjView, dModel, dFF int) ([]byte, bool, error) {
	return mlpTransformQuantMegaWithViewsInto(nil, x, gate, up, down, dModel, dFF, false)
}

func mlpTransformQuantMegaWithViewsInto(out []byte, x []byte, gate, up, down quantMLPProjView, dModel, dFF int, useCallerOut bool) ([]byte, bool, error) {
	if !ffnMegaSupported(gate, up, down, dModel, dFF) {
		return nil, false, nil
	}
	return mlpTransformQuantMegaRun(out, x, gate, up, down, dModel, dFF, useCallerOut)
}

// mlpTransformQuantMegaRun executes the megakernel for any KERNEL-compatible width, bypassing
// the routing receipt in ffnMegaSupported — the width parity tests drive 8-bit through here.
func mlpTransformQuantMegaRun(out []byte, x []byte, gate, up, down quantMLPProjView, dModel, dFF int, useCallerOut bool) ([]byte, bool, error) {
	if !ffnMegaKernelCompatible(gate, up, down, dModel, dFF) {
		return nil, false, nil
	}
	pso, err := ffnMegaPipelineBits(gate.bits)
	if err != nil {
		return nil, false, nil
	}
	outLen := dModel * bf16Size
	callerOut := useCallerOut && cap(out) >= outLen
	if callerOut {
		out = out[:outLen]
	} else {
		out = make([]byte, outLen)
	}

	var encErr error
	withAutoreleasePool(func() {
		scratch, err := getMLPTransformMegaScratch(dModel, dFF)
		if err != nil {
			encErr = err
			return
		}
		defer putMLPTransformMegaScratch(scratch)
		xBuf, ok := scratch.inputView(x)
		if !ok {
			xBuf, err = scratch.x.copyBuffer(x)
			if err != nil {
				encErr = err
				return
			}
		}
		*scratch.arrivePtr = 0
		outBuf := scratch.out
		directOut := false
		if callerOut {
			if tmp, ok := scratch.outputView(out); ok {
				outBuf = tmp
				directOut = true
			}
		}
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		sink := encSink{enc}
		emitFFNMega(sink, pso, xBuf, 0, gate, up, down, scratch.gated, outBuf, 0, scratch.arrive, dModel, dFF)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, scratch.outBytes[:len(out)])
		}
	})
	return out, true, encErr
}

func mlpTransformQuantComposedWithViews(x []byte, gate, up, down quantMLPProjView, dModel, dFF int) ([]byte, error) {
	return mlpTransformQuantComposedWithViewsInto(nil, x, gate, up, down, dModel, dFF, false)
}

func mlpTransformQuantComposedWithViewsInto(out []byte, x []byte, gate, up, down quantMLPProjView, dModel, dFF int, useCallerOut bool) ([]byte, error) {
	outLen := dModel * bf16Size
	callerOut := useCallerOut && cap(out) >= outLen
	if callerOut {
		out = out[:outLen]
	} else {
		out = make([]byte, outLen)
	}
	gatePSO, err := pipelineFor(qmvBF16KernelName(dFF, dModel, gate.groupSize, gate.bits))
	if err != nil {
		return nil, err
	}
	upPSO, err := pipelineFor(qmvBF16KernelName(dFF, dModel, up.groupSize, up.bits))
	if err != nil {
		return nil, err
	}
	downPSO, err := pipelineFor(qmvBF16KernelName(dModel, dFF, down.groupSize, down.bits))
	if err != nil {
		return nil, err
	}
	var geluPSO metal.MTLComputePipelineState
	useFusedGelu := gpuHasGeluKernel()
	if useFusedGelu {
		geluPSO, err = geluPipeline()
		if err != nil {
			return nil, err
		}
	}

	var encErr error
	withAutoreleasePool(func() {
		scratch, err := getMLPTransformScratch(dModel, dFF)
		if err != nil {
			encErr = err
			return
		}
		defer putMLPTransformScratch(scratch)
		xBuf, ok := scratch.inputView(x)
		if !ok {
			xBuf, err = scratch.x.copyBuffer(x)
			if err != nil {
				encErr = err
				return
			}
		}
		msc := scratch.mlp
		outBuf := msc.down
		directOut := false
		if callerOut {
			if tmp, ok := scratch.outputView(out); ok {
				outBuf = tmp
				directOut = true
			}
		}

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		sink := encSink{enc}
		emitQMV(sink, gatePSO, gate.packed.buf, gate.packed.off, gate.scales.buf, gate.scales.off, gate.biases.buf, gate.biases.off, xBuf, msc.gate, 0, dModel, dFF)
		emitQMV(sink, upPSO, up.packed.buf, up.packed.off, up.scales.buf, up.scales.off, up.biases.buf, up.biases.off, xBuf, msc.up, 0, dModel, dFF)
		if useFusedGelu {
			emitBinary(sink, geluPSO, msc.gate, 0, msc.up, 0, msc.gated, 0, dFF)
		} else {
			encErr = encGeluGateMul(enc, msc.gate, msc.up, msc.gated, msc, dFF)
		}
		if encErr != nil {
			endEncodingFast(enc)
			return
		}
		emitQMV(sink, downPSO, down.packed.buf, down.packed.off, down.scales.buf, down.scales.off, down.biases.buf, down.biases.off, msc.gated, outBuf, 0, dFF, dModel)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, unsafe.Slice((*byte)(msc.down.Contents()), len(out)))
		}
	})
	return out, encErr
}

// MoEBlockQuant is MoEBlockBF16 for a 4-bit MoE layer — the same dual-branch feed-forward
// (local dense MLP + router→topK experts, each independently normed, summed, post-normed,
// residual added once), with QMVBF16 / MoERouterQuant / MoEExpertsQuant in place of the bf16
// ops. The router runs on the raw residual; the local MLP uses dFF, the experts ExpertDFF.
func MoEBlockQuant(h []byte, w MoEQuantLayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockQuantWithBuffer(h, nil, w, dModel, dFF, eps)
}

func MoEBlockQuantInto(out []byte, h []byte, w MoEQuantLayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockQuantWithBufferInto(out, h, nil, w, dModel, dFF, eps)
}

func moeBlockQuantWithBuffer(h []byte, hBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockQuantWithBufferPooled(h, hBuf, w, dModel, dFF, eps, true)
}

func moeBlockQuantWithBufferInto(out []byte, h []byte, hBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockQuantWithBufferPooledInto(out, h, hBuf, w, dModel, dFF, eps, true, true)
}

func moeBlockQuantWithBufferInPool(h []byte, hBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel, dFF int, eps float32) ([]byte, error) {
	return moeBlockQuantWithBufferPooled(h, hBuf, w, dModel, dFF, eps, false)
}

func moeBlockQuantWithBufferOutputInPool(h []byte, hBuf, outputBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel, dFF int, eps float32, ownedScratch *moeBlockBF16Scratch) error {
	if outputBuf == nil {
		return core.NewError("native.MoEBlockQuant: output buffer is nil")
	}
	if err := ensureInit(); err != nil {
		return err
	}
	// h may be nil when hBuf carries the hidden (the decode loop's no-wait handoff: the live
	// command buffer commits without a completion wait and the MoE stages queue behind it —
	// nothing on the happy path reads host bytes). Host-bytes callers still validate.
	if hBuf == nil && h == nil {
		return core.NewError("native.MoEBlockQuant: h bytes or hBuf required")
	}
	if h != nil && len(h) != dModel*bf16Size {
		return core.NewError("native.MoEBlockQuant: h must be dModel bf16 bytes")
	}
	numExperts, topK := w.NumExperts, w.TopK

	if quantMoEDeviceRouterBuffersUsable(w, dModel) {
		weightBuf, routerScratch, ok, err := moeRouterQuantDeviceTopKBuffersWithBufferInPool(h, hBuf, w.RouterNormWScaled, w.routerNormView, w.Router, w.PerExpertScale, w.perExpertScaleView, numExperts, topK, dModel, w.RouterGroupSize, w.RouterBits, eps)
		if ok || err != nil {
			if err != nil {
				return err
			}
			var idxBuf metal.MTLBuffer
			if routerScratch != nil {
				idxBuf = routerScratch.idxBuf
			}
			err = moeBlockQuantAfterRouterWithDeviceIndexBufferOutputInPool(h, hBuf, outputBuf, nil, idxBuf, nil, weightBuf, w, dModel, dFF, eps, ownedScratch)
			putRouterDeviceScratch(routerScratch)
			return err
		}
	}
	if idx, weights, weightBuf, routerScratch, ok, err := moeRouterQuantDeviceTopKNoCopyWithBufferInPool(h, hBuf, w.RouterNormWScaled, w.routerNormView, w.Router, w.PerExpertScale, w.perExpertScaleView, numExperts, topK, dModel, w.RouterGroupSize, w.RouterBits, eps); ok || err != nil {
		if err != nil {
			return err
		}
		var idxBuf metal.MTLBuffer
		if routerScratch != nil {
			idxBuf = routerScratch.idxBuf
		}
		idxView, weightView := quantMoEHostRouterViewsForDeviceBuffers(idx, weights, idxBuf, weightBuf, w, dModel)
		err = moeBlockQuantAfterRouterWithDeviceIndexBufferOutputInPool(h, hBuf, outputBuf, idxView, idxBuf, weightView, weightBuf, w, dModel, dFF, eps, ownedScratch)
		putRouterDeviceScratch(routerScratch)
		return err
	}
	idx, weights, err := moeRouterQuantWithViews(h, w.RouterNormWScaled, w.routerNormView, w.Router, w.PerExpertScale, w.perExpertScaleView, numExperts, topK, dModel, w.RouterGroupSize, w.RouterBits, eps)
	if err != nil {
		return err
	}
	return moeBlockQuantAfterRouterWithBufferOutputInPool(h, hBuf, outputBuf, idx, weights, nil, w, dModel, dFF, eps)
}

func moeBlockQuantWithBufferPooled(h []byte, hBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel, dFF int, eps float32, useAutoreleasePool bool) ([]byte, error) {
	return moeBlockQuantWithBufferPooledInto(nil, h, hBuf, w, dModel, dFF, eps, useAutoreleasePool, false)
}

func moeBlockQuantWithBufferPooledInto(out []byte, h []byte, hBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel, dFF int, eps float32, useAutoreleasePool bool, useCallerOut bool) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(h) != dModel*bf16Size {
		return nil, core.NewError("native.MoEBlockQuant: h must be dModel bf16 bytes")
	}
	numExperts, topK := w.NumExperts, w.TopK

	if useAutoreleasePool {
		var blockOut []byte
		var blockErr error
		withAutoreleasePool(func() {
			blockOut, blockErr = moeBlockQuantWithBufferPooledInto(out, h, hBuf, w, dModel, dFF, eps, false, useCallerOut)
		})
		return blockOut, blockErr
	}

	if quantMoEDeviceRouterBuffersUsable(w, dModel) {
		weightBuf, routerScratch, ok, err := moeRouterQuantDeviceTopKBuffersWithBufferInPool(h, hBuf, w.RouterNormWScaled, w.routerNormView, w.Router, w.PerExpertScale, w.perExpertScaleView, numExperts, topK, dModel, w.RouterGroupSize, w.RouterBits, eps)
		if ok || err != nil {
			if err != nil {
				return nil, err
			}
			var idxBuf metal.MTLBuffer
			if routerScratch != nil {
				idxBuf = routerScratch.idxBuf
			}
			blockOut, err := moeBlockQuantAfterRouterWithDeviceIndexBufferPooled(h, hBuf, out, nil, nil, idxBuf, nil, weightBuf, w, dModel, dFF, eps, false, useCallerOut, nil)
			putRouterDeviceScratch(routerScratch)
			return blockOut, err
		}
	}
	if idx, weights, weightBuf, routerScratch, ok, err := moeRouterQuantDeviceTopKNoCopyWithBufferInPool(h, hBuf, w.RouterNormWScaled, w.routerNormView, w.Router, w.PerExpertScale, w.perExpertScaleView, numExperts, topK, dModel, w.RouterGroupSize, w.RouterBits, eps); ok || err != nil {
		if err != nil {
			return nil, err
		}
		var idxBuf metal.MTLBuffer
		if routerScratch != nil {
			idxBuf = routerScratch.idxBuf
		}
		idxView, weightView := quantMoEHostRouterViewsForDeviceBuffers(idx, weights, idxBuf, weightBuf, w, dModel)
		blockOut, err := moeBlockQuantAfterRouterWithDeviceIndexBufferPooled(h, hBuf, out, nil, idxView, idxBuf, weightView, weightBuf, w, dModel, dFF, eps, false, useCallerOut, nil)
		putRouterDeviceScratch(routerScratch)
		return blockOut, err
	}
	idx, weights, err := moeRouterQuantWithViews(h, w.RouterNormWScaled, w.routerNormView, w.Router, w.PerExpertScale, w.perExpertScaleView, numExperts, topK, dModel, w.RouterGroupSize, w.RouterBits, eps)
	if err != nil {
		return nil, err
	}
	if useCallerOut {
		return moeBlockQuantAfterRouterWithBufferIntoInPool(out, h, hBuf, idx, weights, nil, w, dModel, dFF, eps)
	}
	return moeBlockQuantAfterRouterWithBufferInPool(h, hBuf, idx, weights, nil, w, dModel, dFF, eps)
}

func quantMoEHostRouterViewsForDeviceBuffers(idx []int32, weights []byte, idxBuf, weightBuf metal.MTLBuffer, w MoEQuantLayerWeights, dModel int) ([]int32, []byte) {
	if idxBuf == nil || weightBuf == nil || !quantMoEDeviceRouterBuffersUsable(w, dModel) {
		return idx, weights
	}
	return nil, nil
}

func quantMoEDeviceRouterBuffersUsable(w MoEQuantLayerWeights, dModel int) bool {
	if w.TopK <= 0 || w.NumExperts <= 0 || w.ExpertDFF <= 0 {
		return false
	}
	if _, err := bf16MulScalarPipeline(); err != nil {
		return false
	}
	expertDFF, numExperts := w.ExpertDFF, w.NumExperts
	downGroup, downBits := quantWeightGeometryForShape(w.ExpDown, numExperts*dModel, expertDFF, w.ExpertGroupSize, w.ExpertBits)
	if downGroup <= 0 || !affineBitsSupported(downBits) || expertDFF%downGroup != 0 {
		return false
	}
	if len(w.ExpGateUp.Packed) > 0 {
		gateUpGroup, gateUpBits := quantWeightGeometryForShape(w.ExpGateUp, numExperts*2*expertDFF, dModel, w.ExpertGroupSize, w.ExpertBits)
		return gateUpGroup > 0 && affineBitsSupported(gateUpBits) && dModel%gateUpGroup == 0
	}
	gateGroup, gateBits := quantWeightGeometryForShape(w.ExpGate, numExperts*expertDFF, dModel, w.ExpertGroupSize, w.ExpertBits)
	upGroup, upBits := quantWeightGeometryForShape(w.ExpUp, numExperts*expertDFF, dModel, w.ExpertGroupSize, w.ExpertBits)
	return gateGroup > 0 && upGroup > 0 && affineBitsSupported(gateBits) && gateBits == upBits && dModel%gateGroup == 0 && dModel%upGroup == 0 && gateGroup == upGroup
}
