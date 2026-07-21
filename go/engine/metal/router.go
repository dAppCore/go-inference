// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"runtime"
	"slices"
	"sync"
	"sync/atomic"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// bf16ToF32 decodes one little-endian bf16 (2 bytes: lo, hi) to float32 — the
// inverse of f32ToBF16, for reading a device result back to the host.
func bf16ToF32(lo, hi byte) float32 {
	return math.Float32frombits(uint32(uint16(lo)|uint16(hi)<<8) << 16)
}

type routerDeviceScratch struct {
	dModelCapacity     int
	numExpertsCapacity int
	topKCapacity       int
	x                  *pinnedNoCopyBytes
	xPinned            *pinnedNoCopyBytes
	normedBuf          metal.MTLBuffer
	scoresBuf          metal.MTLBuffer
	idxBuf             metal.MTLBuffer
	idxPtr             *int32
	weightBuf          metal.MTLBuffer
	weightPtr          *byte
}

type routerDeviceScratchKey struct {
	dModel, numExperts, topK int
}

type routerDeviceScratchPool struct {
	core.Pool[*routerDeviceScratch]
}

var routerDeviceScratchPools sync.Map

type routerQuantHostScratch struct {
	dModel, numExperts          int
	normed, scores              *pinnedNoCopyBytes
	selectScores, selectSoftmax []float32
	selectIdx                   []int32
	selectWeights               []byte
}

type routerHostScratchKey struct {
	dModel, numExperts int
}

var routerQuantHostScratchPools sync.Map

func routerQuantHostScratchPoolFor(key routerHostScratchKey) *sync.Pool {
	if v, ok := routerQuantHostScratchPools.Load(key); ok {
		return v.(*sync.Pool)
	}
	pool := new(sync.Pool)
	if v, loaded := routerQuantHostScratchPools.LoadOrStore(key, pool); loaded {
		return v.(*sync.Pool)
	}
	return pool
}

func routerQuantHostScratchReady(s *routerQuantHostScratch, key routerHostScratchKey) bool {
	return s != nil &&
		s.dModel == key.dModel &&
		s.numExperts == key.numExperts &&
		s.normed != nil &&
		s.normed.buf != nil &&
		len(s.normed.bytes) == key.dModel*bf16Size &&
		s.scores != nil &&
		s.scores.buf != nil &&
		len(s.scores.bytes) == key.numExperts*bf16Size
}

func newRouterQuantHostScratch(dModel, numExperts int) (*routerQuantHostScratch, error) {
	if dModel <= 0 || numExperts <= 0 {
		return nil, core.NewError("native.newRouterQuantHostScratch: invalid dimensions")
	}
	normed, err := newPinnedNoCopyBytes(dModel * bf16Size)
	if err != nil {
		return nil, err
	}
	scores, err := newPinnedNoCopyBytes(numExperts * bf16Size)
	if err != nil {
		normed.Close()
		return nil, err
	}
	return &routerQuantHostScratch{dModel: dModel, numExperts: numExperts, normed: normed, scores: scores}, nil
}

func (s *routerQuantHostScratch) Close() {
	if s == nil {
		return
	}
	if s.normed != nil {
		s.normed.Close()
		s.normed = nil
	}
	if s.scores != nil {
		s.scores.Close()
		s.scores = nil
	}
	s.selectScores = nil
	s.selectSoftmax = nil
	s.selectIdx = nil
	s.selectWeights = nil
	s.dModel, s.numExperts = 0, 0
}

func getRouterQuantHostScratch(dModel, numExperts int) (*routerQuantHostScratch, error) {
	key := routerHostScratchKey{dModel: dModel, numExperts: numExperts}
	pool := routerQuantHostScratchPoolFor(key)
	if v := pool.Get(); v != nil {
		s := v.(*routerQuantHostScratch)
		if routerQuantHostScratchReady(s, key) {
			return s, nil
		}
		s.Close()
	}
	return newRouterQuantHostScratch(dModel, numExperts)
}

func putRouterQuantHostScratch(s *routerQuantHostScratch) {
	if s == nil {
		return
	}
	key := routerHostScratchKey{dModel: s.dModel, numExperts: s.numExperts}
	if routerQuantHostScratchReady(s, key) {
		routerQuantHostScratchPoolFor(key).Put(s)
	}
}

type routerHostScratch = routerQuantHostScratch

func newRouterHostScratch(dModel, numExperts int) (*routerHostScratch, error) {
	return newRouterQuantHostScratch(dModel, numExperts)
}

func getRouterHostScratch(dModel, numExperts int) (*routerHostScratch, error) {
	return getRouterQuantHostScratch(dModel, numExperts)
}

func putRouterHostScratch(s *routerHostScratch) {
	putRouterQuantHostScratch(s)
}

func newRouterDeviceScratch(dModel, numExperts, topK int) (*routerDeviceScratch, error) {
	x, err := newPinnedNoCopyBytes(dModel * bf16Size)
	if err != nil {
		return nil, err
	}
	s := &routerDeviceScratch{
		dModelCapacity:     dModel,
		numExpertsCapacity: numExperts,
		topKCapacity:       topK,
		x:                  x,
		normedBuf:          scratchBF16(dModel),
		scoresBuf:          scratchBF16(numExperts),
		idxBuf:             device.NewBufferWithLengthOptions(uint(topK*4), metal.MTLResourceStorageModeShared),
		weightBuf:          scratchBF16(topK),
	}
	s.idxPtr = (*int32)(s.idxBuf.Contents())
	s.weightPtr = (*byte)(s.weightBuf.Contents())
	return s, nil
}

func (s *routerDeviceScratch) Close() {
	if s == nil {
		return
	}
	if s.x != nil {
		s.x.Close()
		s.x = nil
	}
	if s.xPinned != nil {
		s.xPinned.Close()
		s.xPinned = nil
	}
}

func (s *routerDeviceScratch) inputView(x []byte) (metal.MTLBuffer, bool) {
	if s == nil || len(x) == 0 {
		return nil, false
	}
	if s.xPinned != nil && len(s.xPinned.bytes) == len(x) && &s.xPinned.bytes[0] == &x[0] {
		return s.xPinned.buf, true
	}
	if s.xPinned != nil {
		s.xPinned.Close()
		s.xPinned = nil
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
	s.xPinned = pinned
	return buf, true
}

func getRouterDeviceScratch(dModel, numExperts, topK int) (*routerDeviceScratch, error) {
	pool := routerDeviceScratchPoolFor(dModel, numExperts, topK)
	for {
		s := pool.Get()
		if s == nil {
			break
		}
		ok := s != nil &&
			s.dModelCapacity == dModel &&
			s.numExpertsCapacity == numExperts &&
			s.topKCapacity == topK &&
			s.x != nil &&
			s.x.buf != nil &&
			len(s.x.bytes) == dModel*bf16Size &&
			s.normedBuf != nil &&
			s.scoresBuf != nil &&
			s.idxBuf != nil &&
			s.idxPtr != nil &&
			s.weightBuf != nil &&
			s.weightPtr != nil
		if ok {
			return s, nil
		}
		s.Close()
	}
	return newRouterDeviceScratch(dModel, numExperts, topK)
}

func routerDeviceScratchPoolFor(dModel, numExperts, topK int) *routerDeviceScratchPool {
	key := routerDeviceScratchKey{dModel: dModel, numExperts: numExperts, topK: topK}
	if v, ok := routerDeviceScratchPools.Load(key); ok {
		return v.(*routerDeviceScratchPool)
	}
	pool := &routerDeviceScratchPool{}
	if v, loaded := routerDeviceScratchPools.LoadOrStore(key, pool); loaded {
		return v.(*routerDeviceScratchPool)
	}
	return pool
}

func putRouterDeviceScratch(s *routerDeviceScratch) {
	if s != nil && s.x != nil && s.x.buf != nil && s.normedBuf != nil && s.scoresBuf != nil && s.idxBuf != nil && s.idxPtr != nil && s.weightBuf != nil && s.weightPtr != nil {
		routerDeviceScratchPoolFor(s.dModelCapacity, s.numExpertsCapacity, s.topKCapacity).Put(s)
	}
}

// topKByScore returns the indices of the topK highest scores, highest first,
// with ties broken by lower index. It deliberately selects only the requested
// experts instead of sorting the full expert list, matching the router hot path's
// top-k shape.
func topKByScore(scores []float32, topK int) []int32 {
	return topKByScoreInto(scores, topK, nil)
}

func topKByScoreInto(scores []float32, topK int, out []int32) []int32 {
	if cap(out) < topK {
		out = make([]int32, topK)
	} else {
		out = out[:topK]
	}
	for slot := range topK {
		best := -1
		for i, score := range scores {
			if selectedExpert(out[:slot], int32(i)) {
				continue
			}
			if best < 0 || score > scores[best] {
				best = i
			}
		}
		out[slot] = int32(best)
	}
	return out
}

func selectedExpert(selected []int32, expert int32) bool {
	return slices.Contains(selected, expert)
}

// softmaxAt returns softmax over the scores at idx (max-subtracted for stability),
// in idx order, as float32.
func softmaxAt(scores []float32, idx []int32) []float32 {
	return softmaxAtInto(scores, idx, nil)
}

func softmaxAtInto(scores []float32, idx []int32, w []float32) []float32 {
	maxS := float32(math.Inf(-1))
	for _, e := range idx {
		if scores[e] > maxS {
			maxS = scores[e]
		}
	}
	if cap(w) < len(idx) {
		w = make([]float32, len(idx))
	} else {
		w = w[:len(idx)]
	}
	var sum float32
	for i, e := range idx {
		w[i] = float32(math.Exp(float64(scores[e] - maxS)))
		sum += w[i]
	}
	for i := range w {
		w[i] /= sum
	}
	return w
}

// softmaxAllThenGatherInto computes softmax over ALL of scores (every expert, not just the selected
// idx), then gathers the probabilities at idx, in idx order, WITHOUT renormalising the gathered
// subset to sum to one — OLMoE's router order (#65): softmax over every routed expert BEFORE the
// top-k gather, no renormalisation of the gathered weights (norm_topk_prob=false), so the combine
// weights returned do NOT sum to 1. This is NOT the same computation as softmaxAtInto: renormalising
// softmax-over-all's gathered subset to sum to one is mathematically IDENTICAL to softmaxing over
// just that subset (the max-subtraction and normalisation constant cancel), so softmaxAtInto already
// computes that value — cheaper — for a NormaliseMoETopK=true arch; this function is only for the
// false case, where the two orders diverge. Mutates scores in place (turns logits into un-normalised
// then normalised probabilities) — callers must be done reading raw scores before calling this
// (routerSelectWithScratch's topKByScoreInto has already selected idx from them).
func softmaxAllThenGatherInto(scores []float32, idx []int32, w []float32) []float32 {
	maxS := float32(math.Inf(-1))
	for _, s := range scores {
		if s > maxS {
			maxS = s
		}
	}
	var sum float32
	for i, s := range scores {
		e := float32(math.Exp(float64(s - maxS)))
		scores[i] = e
		sum += e
	}
	if cap(w) < len(idx) {
		w = make([]float32, len(idx))
	} else {
		w = w[:len(idx)]
	}
	for i, e := range idx {
		w[i] = scores[e] / sum
	}
	return w
}

// MoERouter runs the gemma4 MoE router: it RMS-norms x with the pre-scaled router
// norm weight, projects to per-expert scores, selects the topK highest-scoring
// experts and softmaxes their scores — optionally multiplying each by its per-expert
// scale. Returns (idx, weights) ready to feed MoEExperts.
//
// normWScaled is the router norm weight ALREADY scaled by RootSize (= dModel^-0.5),
// folded once at load exactly like the metal model caches ScaleScaled = Scale·RootSize
// — so this sub-slice needs no on-device scalar-mul. perExpertScale (numExperts bf16)
// is optional; pass nil to skip it. routerW is [numExperts × dModel] row-major bf16
// (each expert is a row), x is dModel bf16; idx is topK int32, weights topK bf16.
//
// The hot path keeps RMSNorm, score projection, top-k, softmax, and optional
// per-expert scaling in one command buffer via the native router top-k kernel,
// mirroring pkg/metal's NativeMoERouterTopK feature. The host selector remains
// only for shapes the copied kernel does not support, such as topK > 32.
//
// The routing decision is order-INVARIANT: each selected expert's weight is
// independent of the order idx is returned in (softmax is over the selected scores;
// the downstream combine is a commutative weighted sum). The parity gate therefore
// compares expert→weight maps, not positional sequences.
//
// normalise is the arch's declared NormaliseMoETopK (#65, MoELayerWeights.NormaliseTopK): true keeps
// this function's shipping shape (top-k select, softmax over just the K selected scores — the
// mathematically identical, cheaper form of softmax-over-all + renormalise-the-gathered-K; see
// softmaxAllThenGatherInto's doc); false selects the SAME top-k but weighs them by softmax over ALL
// numExperts, gathered WITHOUT renormalising (OLMoE's shape). gemma4 always passes true, so its path
// through this function is byte-unchanged.
func MoERouter(x, normWScaled, routerW, perExpertScale []byte, numExperts, topK, dModel int, eps float32, normalise bool) ([]int32, []byte, error) {
	if err := ensureInit(); err != nil {
		return nil, nil, err
	}
	if len(x) != dModel*bf16Size {
		return nil, nil, core.NewError("native.MoERouter: x must be dModel bf16 bytes")
	}
	if len(normWScaled) != dModel*bf16Size {
		return nil, nil, core.NewError("native.MoERouter: normWScaled must be dModel bf16 bytes")
	}
	if len(routerW) != numExperts*dModel*bf16Size {
		return nil, nil, core.NewError("native.MoERouter: routerW must be numExperts*dModel bf16 bytes")
	}
	if perExpertScale != nil && len(perExpertScale) != numExperts*bf16Size {
		return nil, nil, core.NewError("native.MoERouter: perExpertScale must be numExperts bf16 bytes (or nil)")
	}
	if topK <= 0 || topK > numExperts {
		return nil, nil, core.NewError("native.MoERouter: topK must be in 1..numExperts")
	}

	if idx, weights, ok, err := moeRouterBF16DeviceTopK(x, normWScaled, routerW, perExpertScale, numExperts, topK, dModel, eps, normalise); ok || err != nil {
		return idx, weights, err
	}

	scratch, err := getRouterHostScratch(dModel, numExperts)
	if err != nil {
		return nil, nil, err
	}
	defer putRouterHostScratch(scratch)
	return moeRouterBF16HostSelectWithScratch(x, normWScaled, routerW, perExpertScale, numExperts, topK, dModel, eps, scratch, normalise)
}

func moeRouterBF16HostSelectWithScratch(x, normWScaled, routerW, perExpertScale []byte, numExperts, topK, dModel int, eps float32, scratch *routerHostScratch, normalise bool) ([]int32, []byte, error) {
	if scratch == nil || scratch.normed == nil || scratch.scores == nil {
		return nil, nil, core.NewError("native.moeRouterBF16HostSelectWithScratch: scratch is required")
	}
	if scratch.dModel != dModel || scratch.numExperts != numExperts {
		return nil, nil, core.NewError("native.moeRouterBF16HostSelectWithScratch: scratch dimension mismatch")
	}
	normed := scratch.normed.bytes[:dModel*bf16Size]
	scoresB := scratch.scores.bytes[:numExperts*bf16Size]
	var err error
	normed, err = RMSNormBF16Into(normed, x, normWScaled, 1, dModel, eps)
	if err != nil {
		return nil, nil, err
	}
	scoresB, err = matVecBF16ResidentInto(scoresB, routerW, normed, numExperts, dModel)
	if err != nil {
		return nil, nil, err
	}
	idx, weights := routerSelectWithScratch(scoresB, perExpertScale, numExperts, topK, scratch, normalise)
	return idx, weights, nil
}

func moeRouterBF16DeviceTopK(x, normWScaled, routerW, perExpertScale []byte, numExperts, topK, dModel int, eps float32, normalise bool) ([]int32, []byte, bool, error) {
	idxView, weightView, _, scratch, ok, err := moeRouterBF16DeviceTopKNoCopy(x, normWScaled, routerW, perExpertScale, numExperts, topK, dModel, eps, normalise)
	if !ok || err != nil {
		return nil, nil, ok, err
	}
	defer putRouterDeviceScratch(scratch)
	idx, weights := copyRouterTopKViews(idxView, weightView)
	return idx, weights, true, nil
}

func moeRouterBF16DeviceTopKNoCopy(x, normWScaled, routerW, perExpertScale []byte, numExperts, topK, dModel int, eps float32, normalise bool) ([]int32, []byte, metal.MTLBuffer, *routerDeviceScratch, bool, error) {
	return moeRouterBF16DeviceTopKNoCopyWithBuffer(x, nil, normWScaled, routerW, perExpertScale, numExperts, topK, dModel, eps, normalise)
}

func moeRouterBF16DeviceTopKNoCopyWithBuffer(x []byte, xBuf metal.MTLBuffer, normWScaled, routerW, perExpertScale []byte, numExperts, topK, dModel int, eps float32, normalise bool) ([]int32, []byte, metal.MTLBuffer, *routerDeviceScratch, bool, error) {
	return moeRouterBF16DeviceTopKNoCopyWithBufferPooled(x, xBuf, normWScaled, routerW, perExpertScale, numExperts, topK, dModel, eps, normalise, true)
}

func moeRouterBF16DeviceTopKNoCopyWithBufferInPool(x []byte, xBuf metal.MTLBuffer, normWScaled, routerW, perExpertScale []byte, numExperts, topK, dModel int, eps float32, normalise bool) ([]int32, []byte, metal.MTLBuffer, *routerDeviceScratch, bool, error) {
	return moeRouterBF16DeviceTopKNoCopyWithBufferPooled(x, xBuf, normWScaled, routerW, perExpertScale, numExperts, topK, dModel, eps, normalise, false)
}

// moeRouterBF16DeviceTopKNoCopyWithBufferPooled declines (ok=false) when normalise is false: the GPU
// topK kernel (lthn_router_topk_impl.h) always renormalises the gathered K to sum to one — the ONLY
// order a fixed kernel can implement without a metallib rebuild — so a NormaliseMoETopK=false arch
// (#65, OLMoE) must fall through to the host path (moeRouterBF16HostSelectWithScratch), which
// computes the correct softmax-over-all-then-gather-no-renormalise order.
func moeRouterBF16DeviceTopKNoCopyWithBufferPooled(x []byte, xBuf metal.MTLBuffer, normWScaled, routerW, perExpertScale []byte, numExperts, topK, dModel int, eps float32, normalise, useAutoreleasePool bool) ([]int32, []byte, metal.MTLBuffer, *routerDeviceScratch, bool, error) {
	if !routerTopKUsable(numExperts, topK) || !normalise {
		return nil, nil, nil, nil, false, nil
	}
	if len(x) != dModel*bf16Size {
		return nil, nil, nil, nil, true, core.NewError("native.MoERouter: x must be dModel bf16 bytes")
	}
	if len(normWScaled) != dModel*bf16Size {
		return nil, nil, nil, nil, true, core.NewError("native.MoERouter: normWScaled must be dModel bf16 bytes")
	}
	if len(routerW) != numExperts*dModel*bf16Size {
		return nil, nil, nil, nil, true, core.NewError("native.MoERouter: routerW must be numExperts*dModel bf16 bytes")
	}
	if perExpertScale != nil && len(perExpertScale) != numExperts*bf16Size {
		return nil, nil, nil, nil, true, core.NewError("native.MoERouter: perExpertScale must be numExperts bf16 bytes (or nil)")
	}
	rmsPSO, err := pipelineFor(rmsKernelBF16(dModel))
	if err != nil {
		return nil, nil, nil, nil, true, err
	}
	rmsTG := rmsThreadgroup(dModel, rmsPSO)
	routerBM, routerBN, routerSM, routerSN, routerTM, routerTN := gemvTiles(dModel, numExperts)
	routerPSO, err := pipelineFor(gemvKernelName("bfloat16", routerBM, routerBN, routerSM, routerSN, routerTM, routerTN))
	if err != nil {
		return nil, nil, nil, nil, true, err
	}
	topKPSO, err := routerTopKPipelineK(topK)
	if err != nil {
		return nil, nil, nil, nil, true, err
	}
	normBuf := residentBytes(normWScaled)
	routerBuf := residentBytes(routerW)
	var scaleBuf metal.MTLBuffer
	if perExpertScale != nil {
		scaleBuf = residentBytes(perExpertScale)
	}
	var idx []int32
	var weights []byte
	var weightBuf metal.MTLBuffer
	var resultScratch *routerDeviceScratch
	var encErr error
	run := func() {
		scratch, err := getRouterDeviceScratch(dModel, numExperts, topK)
		if err != nil {
			encErr = err
			return
		}
		inputBuf := xBuf
		if inputBuf == nil {
			var ok bool
			inputBuf, ok = scratch.inputView(x)
			if !ok {
				inputBuf, err = scratch.x.copyPrefixBuffer(x)
				if err != nil {
					putRouterDeviceScratch(scratch)
					encErr = err
					return
				}
			}
		}
		normedBuf := scratch.normedBuf
		scoresBuf := scratch.scoresBuf

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		sink := encSink{enc}
		emitRMSNorm(sink, rmsPSO, inputBuf, normBuf, normedBuf, 0, dModel, eps, rmsTG)
		emitGemv(sink, routerPSO, routerBuf, 0, normedBuf, scoresBuf, 0, dModel, numExperts, routerBM, routerBN, routerSM, routerTM)
		scaleBind := scaleBuf
		scaleFlag := int32(0)
		if perExpertScale != nil {
			scaleFlag = 1
		} else {
			scaleBind = scoresBuf
		}
		sink.setPSO(topKPSO)
		sink.setBuf(scoresBuf, 0, 0)
		sink.setBuf(scaleBind, 0, 1)
		sink.setBuf(scratch.idxBuf, 0, 2)
		sink.setBuf(scratch.weightBuf, 0, 3)
		sink.setI32(int32(numExperts), 4)
		sink.setI32(int32(topK), 5)
		sink.setI32(scaleFlag, 6)
		// the kernel is a single 32-lane simdgroup (lane >= 32 exits immediately)
		sink.dispatchThreads(
			metal.MTLSize{Width: 32, Height: 1, Depth: 1},
			metal.MTLSize{Width: 32, Height: 1, Depth: 1},
		)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		idx = unsafe.Slice(scratch.idxPtr, topK)
		weights = unsafe.Slice(scratch.weightPtr, topK*bf16Size)
		weightBuf = scratch.weightBuf
		resultScratch = scratch
	}
	if useAutoreleasePool {
		withAutoreleasePool(run)
	} else {
		run()
	}
	if encErr != nil {
		return nil, nil, nil, nil, true, encErr
	}
	return idx, weights, weightBuf, resultScratch, true, nil
}

func matVecBF16Resident(mat, vec []byte, outDim, inDim int) ([]byte, error) {
	return matVecBF16ResidentInto(nil, mat, vec, outDim, inDim)
}

func matVecBF16ResidentInto(out []byte, mat, vec []byte, outDim, inDim int) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(mat) != outDim*inDim*bf16Size {
		return nil, core.NewError("native.matVecBF16Resident: mat must be outDim*inDim bf16 bytes")
	}
	if len(vec) != inDim*bf16Size {
		return nil, core.NewError("native.matVecBF16Resident: vec must be inDim bf16 bytes")
	}
	if outDim == 0 || inDim == 0 {
		outLen := outDim * bf16Size
		if cap(out) < outLen {
			return make([]byte, outLen), nil
		}
		return out[:outLen], nil
	}
	return MatVecBF16BufInto(out, bufView{buf: residentBytes(mat)}, vec, outDim, inDim)
}

// routerSelect performs the host top-k + softmax (+ optional per-expert scale) over the raw
// per-expert scores (numExperts bf16) — the routing decision shared by MoERouter and
// MoERouterQuant (they differ only in how the scores are projected: bf16 gemv vs 4-bit qmv).
//
// normalise selects the combine-weight policy (#65, model.Arch.NormaliseMoETopK): true softmaxes
// over just the selected top-K scores (sums to 1 — mixtral/gemma4's shape, and the mathematically
// identical, cheaper form of softmax-over-all + renormalise); false softmaxes over ALL numExperts and
// gathers the top-K WITHOUT renormalising (OLMoE's shape — see softmaxAllThenGatherInto).
func routerSelect(scoresB, perExpertScale []byte, numExperts, topK int, normalise bool) ([]int32, []byte) {
	return routerSelectWithScratch(scoresB, perExpertScale, numExperts, topK, nil, normalise)
}

func routerSelectWithScratch(scoresB, perExpertScale []byte, numExperts, topK int, scratch *routerQuantHostScratch, normalise bool) ([]int32, []byte) {
	var scores []float32
	if scratch != nil {
		if cap(scratch.selectScores) < numExperts {
			scratch.selectScores = make([]float32, numExperts)
		} else {
			scratch.selectScores = scratch.selectScores[:numExperts]
		}
		scores = scratch.selectScores
	} else {
		scores = make([]float32, numExperts)
	}
	for e := range numExperts {
		scores[e] = bf16ToF32(scoresB[e*bf16Size], scoresB[e*bf16Size+1])
	}
	var idx []int32
	if scratch != nil {
		scratch.selectIdx = topKByScoreInto(scores, topK, scratch.selectIdx)
		idx = scratch.selectIdx
	} else {
		idx = topKByScore(scores, topK)
	}
	var w []float32
	if normalise {
		if scratch != nil {
			scratch.selectSoftmax = softmaxAtInto(scores, idx, scratch.selectSoftmax)
			w = scratch.selectSoftmax
		} else {
			w = softmaxAt(scores, idx)
		}
	} else {
		// #65: softmax over ALL experts, gather the selected K WITHOUT renormalising.
		if scratch != nil {
			scratch.selectSoftmax = softmaxAllThenGatherInto(scores, idx, scratch.selectSoftmax)
			w = scratch.selectSoftmax
		} else {
			w = softmaxAllThenGatherInto(scores, idx, nil)
		}
	}
	if perExpertScale != nil {
		for i, e := range idx {
			w[i] *= bf16ToF32(perExpertScale[int(e)*bf16Size], perExpertScale[int(e)*bf16Size+1])
		}
	}
	needWeights := topK * bf16Size
	var weights []byte
	if scratch != nil {
		if cap(scratch.selectWeights) < needWeights {
			scratch.selectWeights = make([]byte, needWeights)
		} else {
			scratch.selectWeights = scratch.selectWeights[:needWeights]
		}
		weights = scratch.selectWeights
	} else {
		weights = make([]byte, needWeights)
	}
	for i, v := range w {
		h := f32ToBF16(v)
		weights[i*bf16Size] = byte(h)
		weights[i*bf16Size+1] = byte(h >> 8)
	}
	return idx, weights
}

// MoERouterQuant is MoERouter with a quantised expert-score projection (gemma4
// 26B-A4B's router.proj is affine-quantised). RMS-norm, resident QMV score
// projection, top-k, softmax, and optional scale use the same device router
// top-k path as MoERouter when the copied kernel supports the shape.
//
// normalise is MoERouter's combine-weight policy (#65, model.Arch.NormaliseMoETopK) — see its doc.
func MoERouterQuant(x, normWScaled []byte, routerProj QuantWeight, perExpertScale []byte, numExperts, topK, dModel, groupSize, bits int, eps float32, normalise bool) ([]int32, []byte, error) {
	return moeRouterQuantWithViews(x, normWScaled, bufView{}, routerProj, perExpertScale, bufView{}, numExperts, topK, dModel, groupSize, bits, eps, normalise)
}

func moeRouterQuantWithViews(x, normWScaled []byte, normView bufView, routerProj QuantWeight, perExpertScale []byte, perExpertScaleView bufView, numExperts, topK, dModel, groupSize, bits int, eps float32, normalise bool) ([]int32, []byte, error) {
	if err := ensureInit(); err != nil {
		return nil, nil, err
	}
	if len(x) != dModel*bf16Size {
		return nil, nil, core.NewError("native.MoERouterQuant: x must be dModel bf16 bytes")
	}
	if len(normWScaled) != dModel*bf16Size {
		return nil, nil, core.NewError("native.MoERouterQuant: normWScaled must be dModel bf16 bytes")
	}
	if topK <= 0 || topK > numExperts {
		return nil, nil, core.NewError("native.MoERouterQuant: topK must be in 1..numExperts")
	}
	if perExpertScale != nil && len(perExpertScale) != numExperts*bf16Size {
		return nil, nil, core.NewError("native.MoERouterQuant: perExpertScale must be numExperts bf16 bytes (or nil)")
	}
	groupSize, bits = quantWeightGeometryForShape(routerProj, numExperts, dModel, groupSize, bits)
	if groupSize <= 0 || dModel%groupSize != 0 {
		return nil, nil, core.NewError("native.MoERouterQuant: groupSize must divide dModel")
	}
	wantPacked, wantSB := numExperts*dModel*bits/8, numExperts*(dModel/groupSize)*bf16Size
	if len(routerProj.Packed) != wantPacked || len(routerProj.Scales) != wantSB || len(routerProj.Biases) != wantSB {
		return nil, nil, core.NewError("native.MoERouterQuant: routerProj size mismatch vs numExperts×dModel")
	}

	if idx, weights, ok, err := moeRouterQuantDeviceTopK(x, normWScaled, normView, routerProj, perExpertScale, perExpertScaleView, numExperts, topK, dModel, groupSize, bits, eps, normalise); ok || err != nil {
		return idx, weights, err
	}

	scratch, err := getRouterQuantHostScratch(dModel, numExperts)
	if err != nil {
		return nil, nil, err
	}
	defer putRouterQuantHostScratch(scratch)
	return moeRouterQuantHostSelectWithScratch(x, normWScaled, normView, routerProj, perExpertScale, numExperts, topK, dModel, groupSize, bits, eps, scratch, normalise)
}

func moeRouterQuantHostSelectWithScratch(x, normWScaled []byte, normView bufView, routerProj QuantWeight, perExpertScale []byte, numExperts, topK, dModel, groupSize, bits int, eps float32, scratch *routerQuantHostScratch, normalise bool) ([]int32, []byte, error) {
	if scratch == nil || scratch.normed == nil || scratch.scores == nil {
		return nil, nil, core.NewError("native.moeRouterQuantHostSelectWithScratch: scratch is required")
	}
	if scratch.dModel != dModel || scratch.numExperts != numExperts {
		return nil, nil, core.NewError("native.moeRouterQuantHostSelectWithScratch: scratch dimension mismatch")
	}
	normed := scratch.normed.bytes[:dModel*bf16Size]
	scoresB := scratch.scores.bytes[:numExperts*bf16Size]
	var err error
	normed, err = rmsNormBF16ViewInto(normed, x, normWScaled, normView, 1, dModel, eps)
	if err != nil {
		return nil, nil, err
	}
	scoresB, err = qmvBF16ResidentInto(scoresB, normed, routerProj, numExperts, dModel, groupSize, bits)
	if err != nil {
		return nil, nil, err
	}
	idx, weights := routerSelectWithScratch(scoresB, perExpertScale, numExperts, topK, scratch, normalise)
	return idx, weights, nil
}

func moeRouterQuantDeviceTopK(x, normWScaled []byte, normView bufView, routerProj QuantWeight, perExpertScale []byte, perExpertScaleView bufView, numExperts, topK, dModel, groupSize, bits int, eps float32, normalise bool) ([]int32, []byte, bool, error) {
	idxView, weightView, _, scratch, ok, err := moeRouterQuantDeviceTopKNoCopy(x, normWScaled, normView, routerProj, perExpertScale, perExpertScaleView, numExperts, topK, dModel, groupSize, bits, eps, normalise)
	if !ok || err != nil {
		return nil, nil, ok, err
	}
	defer putRouterDeviceScratch(scratch)
	idx, weights := copyRouterTopKViews(idxView, weightView)
	return idx, weights, true, nil
}

func moeRouterQuantDeviceTopKNoCopy(x, normWScaled []byte, normView bufView, routerProj QuantWeight, perExpertScale []byte, perExpertScaleView bufView, numExperts, topK, dModel, groupSize, bits int, eps float32, normalise bool) ([]int32, []byte, metal.MTLBuffer, *routerDeviceScratch, bool, error) {
	return moeRouterQuantDeviceTopKNoCopyWithBuffer(x, nil, normWScaled, normView, routerProj, perExpertScale, perExpertScaleView, numExperts, topK, dModel, groupSize, bits, eps, normalise)
}

func moeRouterQuantDeviceTopKNoCopyWithBuffer(x []byte, xBuf metal.MTLBuffer, normWScaled []byte, normView bufView, routerProj QuantWeight, perExpertScale []byte, perExpertScaleView bufView, numExperts, topK, dModel, groupSize, bits int, eps float32, normalise bool) ([]int32, []byte, metal.MTLBuffer, *routerDeviceScratch, bool, error) {
	return moeRouterQuantDeviceTopKNoCopyWithBufferPooled(x, xBuf, normWScaled, normView, routerProj, perExpertScale, perExpertScaleView, numExperts, topK, dModel, groupSize, bits, eps, normalise, true)
}

// moeRouterQuantDeviceTopKNoCopyWithBufferInPool's signature is FROZEN (called directly by
// mtp_rows_moe.go's MTP verify lane): it cannot take a normalise parameter. It hardcodes true — the
// GPU topK kernel's only order (always renormalise the gathered K) — so the MTP verify path (which
// predates #65 and has never routed a NormaliseMoETopK=false arch) is byte-unchanged. A future MTP
// caller for such an arch needs its own entry point; this one must keep returning exactly what it
// always has.
func moeRouterQuantDeviceTopKNoCopyWithBufferInPool(x []byte, xBuf metal.MTLBuffer, normWScaled []byte, normView bufView, routerProj QuantWeight, perExpertScale []byte, perExpertScaleView bufView, numExperts, topK, dModel, groupSize, bits int, eps float32) ([]int32, []byte, metal.MTLBuffer, *routerDeviceScratch, bool, error) {
	return moeRouterQuantDeviceTopKNoCopyWithBufferPooled(x, xBuf, normWScaled, normView, routerProj, perExpertScale, perExpertScaleView, numExperts, topK, dModel, groupSize, bits, eps, true, false)
}

func moeRouterQuantDeviceTopKNoCopyWithBufferPooled(x []byte, xBuf metal.MTLBuffer, normWScaled []byte, normView bufView, routerProj QuantWeight, perExpertScale []byte, perExpertScaleView bufView, numExperts, topK, dModel, groupSize, bits int, eps float32, normalise, useAutoreleasePool bool) ([]int32, []byte, metal.MTLBuffer, *routerDeviceScratch, bool, error) {
	return moeRouterQuantDeviceTopKWithBufferPooled(x, xBuf, normWScaled, normView, routerProj, perExpertScale, perExpertScaleView, numExperts, topK, dModel, groupSize, bits, eps, normalise, useAutoreleasePool, true)
}

func moeRouterQuantDeviceTopKBuffersWithBufferInPool(x []byte, xBuf metal.MTLBuffer, normWScaled []byte, normView bufView, routerProj QuantWeight, perExpertScale []byte, perExpertScaleView bufView, numExperts, topK, dModel, groupSize, bits int, eps float32, normalise bool) (metal.MTLBuffer, *routerDeviceScratch, bool, error) {
	_, _, weightBuf, scratch, ok, err := moeRouterQuantDeviceTopKWithBufferPooled(x, xBuf, normWScaled, normView, routerProj, perExpertScale, perExpertScaleView, numExperts, topK, dModel, groupSize, bits, eps, normalise, false, false)
	return weightBuf, scratch, ok, err
}

// encMoERouterQuantTopKRow encodes ONE row's router — RMS → score QMV → the
// 32-lane topK select — into the caller's encoder, byte-for-byte the dispatch
// sequence the single-row primitive commits privately. This is a SUBMISSION
// seam only (#53: the gate's reduction is never batched — K rows are K
// independent instances of this exact per-row sequence, whatever command
// buffer they share); results land in the row's own scratch idx/weight
// buffers, readable after the caller's wait.
func encMoERouterQuantTopKRow(enc metal.MTLComputeCommandEncoderObject, scratch *routerDeviceScratch, inputBuf metal.MTLBuffer, normBuf bufView, wBuf, scalesBuf, biasesBuf bufView, scaleBuf metal.MTLBuffer, scaleOff uint, hasPerExpertScale bool, rmsPSO, qmvPSO, routerPSO metal.MTLComputePipelineState, rmsTG uint, dModel, numExperts, topK int, eps float32) {
	sink := encSink{enc}
	emitRMSNorm(sink, rmsPSO, inputBuf, normBuf.buf, scratch.normedBuf, normBuf.off, dModel, eps, rmsTG)
	encMoERouterQuantTopKTail(enc, scratch, wBuf, scalesBuf, biasesBuf, scaleBuf, scaleOff, hasPerExpertScale, qmvPSO, routerPSO, dModel, numExperts, topK, scratch.idxBuf, 0, scratch.weightBuf, 0)
}

// encMoERouterQuantTopKRowTo is encMoERouterQuantTopKRow with the topK select's
// idx/weight outputs redirected — the device-routed verify lane (#53) points K
// rows' selects at row offsets of ONE shared pair buffer, which the expert
// gathers then read WITHOUT the indices ever returning to the host.
func encMoERouterQuantTopKRowTo(enc metal.MTLComputeCommandEncoderObject, scratch *routerDeviceScratch, inputBuf metal.MTLBuffer, inputOff uint, normBuf bufView, wBuf, scalesBuf, biasesBuf bufView, scaleBuf metal.MTLBuffer, scaleOff uint, hasPerExpertScale bool, rmsPSO, qmvPSO, routerPSO metal.MTLComputePipelineState, rmsTG uint, dModel, numExperts, topK int, eps float32, idxOut metal.MTLBuffer, idxOff uint, weightOut metal.MTLBuffer, weightOff uint) {
	sink := encSink{enc}
	emitRMSNormRows(sink, rmsPSO, inputBuf, normBuf.buf, scratch.normedBuf, inputOff, normBuf.off, 0, dModel, eps, 1, rmsTG)
	encMoERouterQuantTopKTail(enc, scratch, wBuf, scalesBuf, biasesBuf, scaleBuf, scaleOff, hasPerExpertScale, qmvPSO, routerPSO, dModel, numExperts, topK, idxOut, idxOff, weightOut, weightOff)
}

// encMoERouterQuantTopKTail encodes the score QMV + the 32-lane topK select —
// the shared tail of the single-row primitive and the row-offset To-variant.
func encMoERouterQuantTopKTail(enc metal.MTLComputeCommandEncoderObject, scratch *routerDeviceScratch, wBuf, scalesBuf, biasesBuf bufView, scaleBuf metal.MTLBuffer, scaleOff uint, hasPerExpertScale bool, qmvPSO, routerPSO metal.MTLComputePipelineState, dModel, numExperts, topK int, idxOut metal.MTLBuffer, idxOff uint, weightOut metal.MTLBuffer, weightOff uint) {
	sink := encSink{enc}
	emitQMV(sink, qmvPSO, wBuf.buf, wBuf.off, scalesBuf.buf, scalesBuf.off, biasesBuf.buf, biasesBuf.off, scratch.normedBuf, scratch.scoresBuf, 0, dModel, numExperts)
	if scaleBuf == nil {
		scaleBuf = scratch.scoresBuf
	}
	scaleFlag := int32(0)
	if hasPerExpertScale {
		scaleFlag = 1
	}
	sink.setPSO(routerPSO)
	sink.setBuf(scratch.scoresBuf, 0, 0)
	sink.setBuf(scaleBuf, scaleOff, 1)
	sink.setBuf(idxOut, idxOff, 2)
	sink.setBuf(weightOut, weightOff, 3)
	sink.setI32(int32(numExperts), 4)
	sink.setI32(int32(topK), 5)
	sink.setI32(scaleFlag, 6)
	// the kernel is a single 32-lane simdgroup (lane >= 32 exits immediately)
	sink.dispatchThreads(
		metal.MTLSize{Width: 32, Height: 1, Depth: 1},
		metal.MTLSize{Width: 32, Height: 1, Depth: 1},
	)
}

// moeRouterQuantDeviceTopKWithBufferPooled declines (ok=false) when normalise is false — see
// moeRouterBF16DeviceTopKNoCopyWithBufferPooled's identical note: the GPU topK kernel cannot
// implement softmax-over-all-then-gather-no-renormalise, so a NormaliseMoETopK=false arch (#65) must
// fall through to the host path (moeRouterQuantHostSelectWithScratch).
func moeRouterQuantDeviceTopKWithBufferPooled(x []byte, xBuf metal.MTLBuffer, normWScaled []byte, normView bufView, routerProj QuantWeight, perExpertScale []byte, perExpertScaleView bufView, numExperts, topK, dModel, groupSize, bits int, eps float32, normalise, useAutoreleasePool, returnHostViews bool) ([]int32, []byte, metal.MTLBuffer, *routerDeviceScratch, bool, error) {
	if !routerTopKUsable(numExperts, topK) || !normalise {
		return nil, nil, nil, nil, false, nil
	}
	// x may be nil when xBuf carries the hidden (the decode loop's no-wait handoff — the
	// input fallback below never fires with a buffer supplied).
	if xBuf == nil && x == nil {
		return nil, nil, nil, nil, true, core.NewError("native.MoERouterQuant: x bytes or xBuf required")
	}
	if x != nil && len(x) != dModel*bf16Size {
		return nil, nil, nil, nil, true, core.NewError("native.MoERouterQuant: x must be dModel bf16 bytes")
	}
	if len(normWScaled) != dModel*bf16Size {
		return nil, nil, nil, nil, true, core.NewError("native.MoERouterQuant: normWScaled must be dModel bf16 bytes")
	}
	if perExpertScale != nil && len(perExpertScale) != numExperts*bf16Size {
		return nil, nil, nil, nil, true, core.NewError("native.MoERouterQuant: perExpertScale must be numExperts bf16 bytes (or nil)")
	}
	groupSize, bits = quantWeightGeometryForShape(routerProj, numExperts, dModel, groupSize, bits)
	if groupSize <= 0 || dModel%groupSize != 0 {
		return nil, nil, nil, nil, true, core.NewError("native.MoERouterQuant: groupSize must divide dModel")
	}
	wantPacked, wantSB := numExperts*dModel*bits/8, numExperts*(dModel/groupSize)*bf16Size
	if len(routerProj.Packed) != wantPacked || len(routerProj.Scales) != wantSB || len(routerProj.Biases) != wantSB {
		return nil, nil, nil, nil, true, core.NewError("native.MoERouterQuant: routerProj size mismatch vs numExperts×dModel")
	}
	rmsPSO, err := pipelineFor(rmsKernelBF16(dModel))
	if err != nil {
		return nil, nil, nil, nil, true, err
	}
	rmsTG := rmsThreadgroup(dModel, rmsPSO)
	qmvPSO, err := pipelineFor(qmvBF16KernelName(numExperts, dModel, groupSize, bits))
	if err != nil {
		return nil, nil, nil, nil, true, err
	}
	routerPSO, err := routerTopKPipelineK(topK)
	if err != nil {
		return nil, nil, nil, nil, true, err
	}
	var idx []int32
	var weights []byte
	var weightBuf metal.MTLBuffer
	var resultScratch *routerDeviceScratch
	var encErr error
	run := func() {
		scratch, err := getRouterDeviceScratch(dModel, numExperts, topK)
		if err != nil {
			encErr = err
			return
		}
		inputBuf := xBuf
		if inputBuf == nil {
			var ok bool
			inputBuf, ok = scratch.inputView(x)
			if !ok {
				inputBuf, err = scratch.x.copyPrefixBuffer(x)
				if err != nil {
					putRouterDeviceScratch(scratch)
					encErr = err
					return
				}
			}
		}
		normBuf := bf16WeightView(normWScaled, normView)
		wBuf, scalesBuf, biasesBuf := quantWeightViews(routerProj)
		var scaleBuf metal.MTLBuffer
		var scaleOff uint
		if perExpertScale != nil {
			scaleView := bf16WeightView(perExpertScale, perExpertScaleView)
			scaleBuf, scaleOff = scaleView.buf, scaleView.off
		}
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		encMoERouterQuantTopKRow(enc, scratch, inputBuf, normBuf, wBuf, scalesBuf, biasesBuf, scaleBuf, scaleOff, perExpertScale != nil, rmsPSO, qmvPSO, routerPSO, rmsTG, dModel, numExperts, topK, eps)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		if returnHostViews {
			waitUntilCompletedFast(cb)
			idx = unsafe.Slice(scratch.idxPtr, topK)
			weights = unsafe.Slice(scratch.weightPtr, topK*bf16Size)
		}
		weightBuf = scratch.weightBuf
		resultScratch = scratch
	}
	if useAutoreleasePool {
		withAutoreleasePool(run)
	} else {
		run()
	}
	if encErr != nil {
		return nil, nil, nil, nil, true, encErr
	}
	return idx, weights, weightBuf, resultScratch, true, nil
}

func routerTopKUsable(numExperts, topK int) bool {
	if topK <= 0 || topK > numExperts || topK > routerTopKMaxK {
		return false
	}
	_, err := routerTopKPipelineK(topK)
	return err == nil
}

// routerFusedDispatches counts routers that ran the single-dispatch fused kernel
// instead of the 3-dispatch chain (#340). Engagement counter for the A/B tests —
// a compare that never engaged the lane proves nothing.
var routerFusedDispatches atomic.Int64

type routerFusedKey struct {
	groupSize, bits, topK int
}

var (
	routerFusedPSOMu    sync.Mutex
	routerFusedPSOCache = map[routerFusedKey]metal.MTLComputePipelineState{}
)

// routerFusedPipeline resolves (and caches, including failures) the fused router
// kernel for a quant geometry: top-k bakes as the shared lthn_router_topk_k
// function constant on the group_size/bits template instance. A miss — custom
// library absent, or a gs/bits pair outside the instantiated set — caches nil so
// the caller falls back to the 3-dispatch chain without re-probing.
func routerFusedPipeline(groupSize, bits, topK int) (metal.MTLComputePipelineState, bool) {
	if topK <= 0 || topK > routerTopKMaxK {
		return nil, false
	}
	key := routerFusedKey{groupSize: groupSize, bits: bits, topK: topK}
	routerFusedPSOMu.Lock()
	defer routerFusedPSOMu.Unlock()
	if pso, ok := routerFusedPSOCache[key]; ok {
		return pso, pso != nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		routerFusedPSOCache[key] = nil
		return nil, false
	}
	name := core.Sprintf("lthn_moe_router_fused_bfloat16_t_gs_%d_b_%d", groupSize, bits)
	fc := metal.NewMTLFunctionConstantValues()
	kk := uint32(topK)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&kk), metal.MTLDataTypeUInt, 0)
	fn, err := customLibrary.NewFunctionWithNameConstantValuesError(name, fc)
	if err != nil || fn == nil || fn.GetID() == 0 {
		routerFusedPSOCache[key] = nil
		return nil, false
	}
	pso, perr := device.NewComputePipelineStateWithFunctionError(fn)
	if perr != nil {
		routerFusedPSOCache[key] = nil
		return nil, false
	}
	routerFusedPSOCache[key] = pso
	return pso, true
}

// routerEncodePlan holds the router's resolved PSOs + weight views so the three stages can
// emit independently — the serial path runs them back to back; the concurrent MoE pass
// interleaves them with the local/expert stages between dependency barriers. When the
// fused single-dispatch router is available (fusedPSO non-nil) emitFused replaces all
// three stages.
type routerEncodePlan struct {
	scratch                  *routerDeviceScratch
	inputBuf                 metal.MTLBuffer
	rmsPSO, qmvPSO, topkPSO  metal.MTLComputePipelineState
	fusedPSO                 metal.MTLComputePipelineState
	fusedTG                  uint
	normBuf                  bufView
	wBuf, scalesBuf, biasBuf bufView
	scaleBuf                 metal.MTLBuffer
	scaleOff                 uint
	scaleFlag                int32
	numExperts, topK, dModel int
	eps                      float32
}

func buildRouterEncodePlan(scratch *routerDeviceScratch, inputBuf metal.MTLBuffer, normWScaled []byte, normView bufView, routerProj QuantWeight, perExpertScale []byte, perExpertScaleView bufView, numExperts, topK, dModel, groupSize, bits int, eps float32) (*routerEncodePlan, error) {
	groupSize, bits = quantWeightGeometryForShape(routerProj, numExperts, dModel, groupSize, bits)
	rmsPSO, err := pipelineFor(rmsKernelBF16(dModel))
	if err != nil {
		return nil, err
	}
	qmvPSO, err := pipelineFor(qmvBF16KernelName(numExperts, dModel, groupSize, bits))
	if err != nil {
		return nil, err
	}
	topkPSO, err := routerTopKPipelineK(topK)
	if err != nil {
		return nil, err
	}
	p := &routerEncodePlan{
		scratch: scratch, inputBuf: inputBuf,
		rmsPSO: rmsPSO, qmvPSO: qmvPSO, topkPSO: topkPSO,
		normBuf:    bf16WeightView(normWScaled, normView),
		numExperts: numExperts, topK: topK, dModel: dModel, eps: eps,
	}
	p.wBuf, p.scalesBuf, p.biasBuf = quantWeightViews(routerProj)
	if perExpertScale != nil {
		scaleView := bf16WeightView(perExpertScale, perExpertScaleView)
		p.scaleBuf, p.scaleOff = scaleView.buf, scaleView.off
		p.scaleFlag = 1
	} else {
		p.scaleBuf = scratch.scoresBuf
	}
	// The fused single-dispatch router (#340): launched at the rms single-row
	// kernel's own threadgroup shape so its phase-1 reduction tree — and
	// therefore the routing decision — is byte-identical to the chain. Only the
	// single-row rms shape qualifies, and the pipeline must admit that many
	// threads per threadgroup (register pressure from the qmv phase can cap it).
	if dModel <= rmsLoopedLimit {
		if pso, ok := routerFusedPipeline(groupSize, bits, topK); ok {
			tg := rmsThreadgroup(dModel, pso)
			if uint(pso.MaxTotalThreadsPerThreadgroup()) >= uint(tg) {
				p.fusedPSO, p.fusedTG = pso, uint(tg)
			}
		}
	}
	return p, nil
}

// emitFused encodes the whole router as ONE dispatch when the fused kernel is
// available AND opted in (routerFusedEnabled — off by the 125.3-vs-140.6 tok/s
// receipt), returning whether it engaged — the caller falls back to the
// emitRMS/emitQMV/emitTopK chain on false. Same scratch, same output buffers,
// byte-identical routing (see the kernel's replication notes).
func (p *routerEncodePlan) emitFused(sink encSink) bool {
	if p.fusedPSO == nil || !routerFusedEnabled {
		return false
	}
	routerFusedDispatches.Add(1)
	sink.setPSO(p.fusedPSO)
	sink.setBuf(p.inputBuf, 0, 0)
	sink.setBuf(p.normBuf.buf, p.normBuf.off, 1)
	sink.setBuf(p.scratch.normedBuf, 0, 2)
	sink.setBuf(p.wBuf.buf, p.wBuf.off, 3)
	sink.setBuf(p.scalesBuf.buf, p.scalesBuf.off, 4)
	sink.setBuf(p.biasBuf.buf, p.biasBuf.off, 5)
	sink.setBuf(p.scratch.scoresBuf, 0, 6)
	sink.setBuf(p.scaleBuf, p.scaleOff, 7)
	sink.setBuf(p.scratch.idxBuf, 0, 8)
	sink.setBuf(p.scratch.weightBuf, 0, 9)
	sink.setF32(p.eps, 10)
	sink.setI32(int32(p.dModel), 11)
	sink.setI32(int32(p.numExperts), 12)
	sink.setI32(p.scaleFlag, 13)
	sink.setI32(int32(p.topK), 14)
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: 1, Height: 1, Depth: 1},
		metal.MTLSize{Width: p.fusedTG, Height: 1, Depth: 1},
	)
	return true
}

func (p *routerEncodePlan) emitRMS(sink encSink) {
	emitRMSNorm(sink, p.rmsPSO, p.inputBuf, p.normBuf.buf, p.scratch.normedBuf, p.normBuf.off, p.dModel, p.eps, rmsThreadgroup(p.dModel, p.rmsPSO))
}

func (p *routerEncodePlan) emitQMV(sink encSink) {
	emitQMV(sink, p.qmvPSO, p.wBuf.buf, p.wBuf.off, p.scalesBuf.buf, p.scalesBuf.off, p.biasBuf.buf, p.biasBuf.off, p.scratch.normedBuf, p.scratch.scoresBuf, 0, p.dModel, p.numExperts)
}

func (p *routerEncodePlan) emitTopK(sink encSink) {
	sink.setPSO(p.topkPSO)
	sink.setBuf(p.scratch.scoresBuf, 0, 0)
	sink.setBuf(p.scaleBuf, p.scaleOff, 1)
	sink.setBuf(p.scratch.idxBuf, 0, 2)
	sink.setBuf(p.scratch.weightBuf, 0, 3)
	sink.setI32(int32(p.numExperts), 4)
	sink.setI32(int32(p.topK), 5)
	sink.setI32(p.scaleFlag, 6)
	// the kernel is a single 32-lane simdgroup (lane >= 32 exits immediately)
	sink.dispatchThreads(
		metal.MTLSize{Width: 32, Height: 1, Depth: 1},
		metal.MTLSize{Width: 32, Height: 1, Depth: 1},
	)
}
func copyRouterTopKOutput(scratch *routerDeviceScratch, topK int) ([]int32, []byte) {
	return copyRouterTopKViews(unsafe.Slice(scratch.idxPtr, topK), unsafe.Slice(scratch.weightPtr, topK*bf16Size))
}

func copyRouterTopKViews(idxView []int32, weightView []byte) ([]int32, []byte) {
	idx := make([]int32, len(idxView))
	weights := make([]byte, len(weightView))
	copy(idx, idxView)
	copy(weights, weightView)
	return idx, weights
}
