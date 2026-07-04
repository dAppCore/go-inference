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

// sdpaPSOCache memoises the sdpa_vector pipeline keyed by kernel name. The decode
// path is always no-mask / non-causal / non-transposed / no-sinks, so the six
// function constants are fixed to false; if other combinations are added later,
// fold them into the key.
var (
	sdpaPSOMu                    sync.Mutex
	sdpaPSOCache                 = map[string]metal.MTLComputePipelineState{}
	sdpaVectorHeadDimPSOCache    = map[int]metal.MTLComputePipelineState{}
	sdpaVector2Pass1HeadDimCache = map[sdpa2Pass1Key]metal.MTLComputePipelineState{}
	sdpaVector2Pass2HeadDimCache = map[int]metal.MTLComputePipelineState{}
	sdpaBF16ScratchPools         sync.Map
	errSDPABF16ScratchDim        = core.NewError("native.sdpaBF16Scratch: dimension mismatch")
)

type sdpa2Pass1Key struct {
	headDim int
	blocks  int32
}

type sdpaBF16Scratch struct {
	qBytes, kBytes, vBytes, outBytes int
	q, k, v, out                     *pinnedNoCopyBytes
	qView, kView, vView              cachedNoCopyBytesView
	p2PartialBytes, p2SumBytes       int
	p2Partials, p2Sums, p2Maxs       metal.MTLBuffer
	outViewPtr                       uintptr
	outViewLen                       int
	outView                          metal.MTLBuffer
	outViewPinned                    *pinnedNoCopyBytes
}

type sdpaBF16ScratchKey struct {
	qBytes, kBytes, vBytes, outBytes int
}

type sdpaBF16ScratchPool struct {
	mu    sync.Mutex
	items []*sdpaBF16Scratch
}

func sdpaBF16ScratchPoolFor(key sdpaBF16ScratchKey) *sdpaBF16ScratchPool {
	if v, ok := sdpaBF16ScratchPools.Load(key); ok {
		return v.(*sdpaBF16ScratchPool)
	}
	pool := new(sdpaBF16ScratchPool)
	if v, loaded := sdpaBF16ScratchPools.LoadOrStore(key, pool); loaded {
		return v.(*sdpaBF16ScratchPool)
	}
	return pool
}

func (p *sdpaBF16ScratchPool) Get() *sdpaBF16Scratch {
	p.mu.Lock()
	defer p.mu.Unlock()
	n := len(p.items)
	if n == 0 {
		return nil
	}
	s := p.items[n-1]
	p.items[n-1] = nil
	p.items = p.items[:n-1]
	return s
}

func (p *sdpaBF16ScratchPool) Put(s *sdpaBF16Scratch) {
	if s == nil {
		return
	}
	p.mu.Lock()
	p.items = append(p.items, s)
	p.mu.Unlock()
}

func sdpaBF16ScratchReady(s *sdpaBF16Scratch, key sdpaBF16ScratchKey) bool {
	return s != nil &&
		s.qBytes == key.qBytes && s.kBytes == key.kBytes && s.vBytes == key.vBytes && s.outBytes == key.outBytes &&
		s.q != nil && s.k != nil && s.v != nil && s.out != nil
}

func newSDPABF16Scratch(qBytes, kBytes, vBytes, outBytes int) (*sdpaBF16Scratch, error) {
	if qBytes <= 0 || kBytes <= 0 || vBytes <= 0 || outBytes <= 0 {
		return nil, core.NewError("native.newSDPABF16Scratch: invalid dimensions")
	}
	q, err := newPinnedNoCopyBytes(qBytes)
	if err != nil {
		return nil, err
	}
	k, err := newPinnedNoCopyBytes(kBytes)
	if err != nil {
		q.Close()
		return nil, err
	}
	v, err := newPinnedNoCopyBytes(vBytes)
	if err != nil {
		q.Close()
		k.Close()
		return nil, err
	}
	out, err := newPinnedNoCopyBytes(outBytes)
	if err != nil {
		q.Close()
		k.Close()
		v.Close()
		return nil, err
	}
	return &sdpaBF16Scratch{
		qBytes: qBytes, kBytes: kBytes, vBytes: vBytes, outBytes: outBytes,
		q: q, k: k, v: v, out: out,
	}, nil
}

func getSDPABF16Scratch(qBytes, kBytes, vBytes, outBytes int) (*sdpaBF16Scratch, error) {
	key := sdpaBF16ScratchKey{qBytes: qBytes, kBytes: kBytes, vBytes: vBytes, outBytes: outBytes}
	pool := sdpaBF16ScratchPoolFor(key)
	if s := pool.Get(); s != nil {
		if sdpaBF16ScratchReady(s, key) {
			return s, nil
		}
		s.Close()
	}
	return newSDPABF16Scratch(qBytes, kBytes, vBytes, outBytes)
}

func putSDPABF16Scratch(s *sdpaBF16Scratch) {
	if s == nil {
		return
	}
	key := sdpaBF16ScratchKey{qBytes: s.qBytes, kBytes: s.kBytes, vBytes: s.vBytes, outBytes: s.outBytes}
	if sdpaBF16ScratchReady(s, key) {
		sdpaBF16ScratchPoolFor(key).Put(s)
	}
}

func (s *sdpaBF16Scratch) Close() {
	if s == nil {
		return
	}
	if s.q != nil {
		s.q.Close()
		s.q = nil
	}
	if s.k != nil {
		s.k.Close()
		s.k = nil
	}
	if s.v != nil {
		s.v.Close()
		s.v = nil
	}
	if s.out != nil {
		s.out.Close()
		s.out = nil
	}
	s.qView.Close()
	s.kView.Close()
	s.vView.Close()
	s.closeOutputView()
	s.qBytes, s.kBytes, s.vBytes, s.outBytes = 0, 0, 0, 0
	s.p2Partials, s.p2Sums, s.p2Maxs = nil, nil, nil
	s.p2PartialBytes, s.p2SumBytes = 0, 0
}

func (s *sdpaBF16Scratch) closeOutputView() {
	if s == nil {
		return
	}
	if s.outViewPinned != nil {
		s.outViewPinned.Close()
	}
	s.outViewPtr = 0
	s.outViewLen = 0
	s.outView = nil
	s.outViewPinned = nil
}

func (s *sdpaBF16Scratch) outputView(out []byte) (metal.MTLBuffer, bool) {
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
		s.outViewPinned = nil
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
	s.outViewPinned = pinned
	return buf, true
}

func (s *sdpaBF16Scratch) buffers(qb, kb, vb []byte) (metal.MTLBuffer, metal.MTLBuffer, metal.MTLBuffer, metal.MTLBuffer, error) {
	if s == nil || s.q == nil || s.k == nil || s.v == nil || s.out == nil {
		return nil, nil, nil, nil, core.NewError("native.sdpaBF16Scratch.buffers: scratch is nil")
	}
	if len(qb) != s.qBytes || len(kb) != s.kBytes || len(vb) != s.vBytes || len(s.out.bytes) != s.outBytes {
		return nil, nil, nil, nil, errSDPABF16ScratchDim
	}
	qBuf, ok := s.qView.buffer(qb)
	if !ok {
		var err error
		qBuf, err = s.q.copyBuffer(qb)
		if err != nil {
			return nil, nil, nil, nil, err
		}
	}
	kBuf, ok := s.kView.buffer(kb)
	if !ok {
		var err error
		kBuf, err = s.k.copyBuffer(kb)
		if err != nil {
			return nil, nil, nil, nil, err
		}
	}
	vBuf, ok := s.vView.buffer(vb)
	if !ok {
		var err error
		vBuf, err = s.v.copyBuffer(vb)
		if err != nil {
			return nil, nil, nil, nil, err
		}
	}
	return qBuf, kBuf, vBuf, s.out.buf, nil
}

func (s *sdpaBF16Scratch) twoPassBuffers(nbh int, blocks int32, headDim int) (metal.MTLBuffer, metal.MTLBuffer, metal.MTLBuffer, error) {
	if s == nil {
		return nil, nil, nil, core.NewError("native.sdpaBF16Scratch.twoPassBuffers: scratch is nil")
	}
	if nbh <= 0 || blocks <= 0 || headDim <= 0 {
		return nil, nil, nil, core.NewError("native.sdpaBF16Scratch.twoPassBuffers: invalid dimensions")
	}
	partialBytes := nbh * int(blocks) * headDim * bf16Size
	sumBytes := nbh * int(blocks) * 4
	if s.p2Partials != nil && s.p2Sums != nil && s.p2Maxs != nil &&
		s.p2PartialBytes == partialBytes && s.p2SumBytes == sumBytes {
		return s.p2Partials, s.p2Sums, s.p2Maxs, nil
	}
	s.p2Partials = device.NewBufferWithLengthOptions(uint(partialBytes), metal.MTLResourceStorageModeShared)
	s.p2Sums = device.NewBufferWithLengthOptions(uint(sumBytes), metal.MTLResourceStorageModeShared)
	s.p2Maxs = device.NewBufferWithLengthOptions(uint(sumBytes), metal.MTLResourceStorageModeShared)
	if s.p2Partials == nil || s.p2Sums == nil || s.p2Maxs == nil ||
		s.p2Partials.GetID() == 0 || s.p2Sums.GetID() == 0 || s.p2Maxs.GetID() == 0 {
		s.p2Partials, s.p2Sums, s.p2Maxs = nil, nil, nil
		s.p2PartialBytes, s.p2SumBytes = 0, 0
		return nil, nil, nil, core.NewError("native.sdpaBF16Scratch.twoPassBuffers: failed to create intermediates")
	}
	s.p2PartialBytes, s.p2SumBytes = partialBytes, sumBytes
	return s.p2Partials, s.p2Sums, s.p2Maxs, nil
}

// sdpaVectorPipeline builds (and caches) the sdpa_vector kernel with MLX's six
// attention function constants all false (no mask, query not transposed, not
// causal, no bool/float mask, no sinks) — the decode-time configuration.
func sdpaVectorPipeline(name string) (metal.MTLComputePipelineState, error) {
	sdpaPSOMu.Lock()
	defer sdpaPSOMu.Unlock()
	if pso, ok := sdpaPSOCache[name]; ok {
		return pso, nil
	}
	if library == nil || library.GetID() == 0 {
		return nil, core.NewError("native.sdpaVectorPipeline: library unavailable for " + name)
	}
	fc := metal.NewMTLFunctionConstantValues()
	off := uint8(0)
	// indices: has_mask(20) query_transposed(21) do_causal(22) bool_mask(23)
	// float_mask(24) has_sinks(25)
	for _, idx := range []uint{20, 21, 22, 23, 24, 25} {
		fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&off), metal.MTLDataTypeBool, idx)
	}
	fn, err := library.NewFunctionWithNameConstantValuesError(name, fc)
	if err != nil {
		return nil, core.E("native.sdpaVectorPipeline", name, err)
	}
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.sdpaVectorPipeline: kernel " + name + " not found")
	}
	pso, err := device.NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		return nil, core.E("native.sdpaVectorPipeline", "pipeline "+name, err)
	}
	sdpaPSOCache[name] = pso
	return pso, nil
}

func sdpaVectorPipelineForHeadDim(headDim int) (metal.MTLComputePipelineState, error) {
	sdpaPSOMu.Lock()
	if pso, ok := sdpaVectorHeadDimPSOCache[headDim]; ok {
		sdpaPSOMu.Unlock()
		return pso, nil
	}
	sdpaPSOMu.Unlock()

	pso, err := sdpaVectorPipeline(core.Sprintf("sdpa_vector_bfloat16_t_%d_%d", headDim, headDim))
	if err != nil {
		return nil, err
	}

	sdpaPSOMu.Lock()
	if existing, ok := sdpaVectorHeadDimPSOCache[headDim]; ok {
		sdpaPSOMu.Unlock()
		return existing, nil
	}
	sdpaVectorHeadDimPSOCache[headDim] = pso
	sdpaPSOMu.Unlock()
	return pso, nil
}

// sdpaVector2Pass1Pipeline builds (and caches) the sdpa_vector_2pass_1 kernel —
// attention function constants 20..25 false (decode-time: no mask/transpose/
// causal/sinks) PLUS function constant 26 = blocks (the cache-split count). blocks
// is baked into the pipeline because the kernel indexes the intermediate by it; the
// PSO is keyed by name+blocks so a new block count is a fresh pipeline, not a clash.
func sdpaVector2Pass1Pipeline(name string, blocks int32) (metal.MTLComputePipelineState, error) {
	key := core.Sprintf("%s:b%d", name, blocks)
	sdpaPSOMu.Lock()
	defer sdpaPSOMu.Unlock()
	if pso, ok := sdpaPSOCache[key]; ok {
		return pso, nil
	}
	if library == nil || library.GetID() == 0 {
		return nil, core.NewError("native.sdpaVector2Pass1Pipeline: library unavailable for " + name)
	}
	fc := metal.NewMTLFunctionConstantValues()
	off := uint8(0)
	for _, idx := range []uint{20, 21, 22, 23, 24, 25} { // has_mask query_transposed do_causal bool_mask float_mask has_sinks
		fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&off), metal.MTLDataTypeBool, idx)
	}
	blk := blocks
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&blk), metal.MTLDataTypeInt, 26) // blocks
	fn, err := library.NewFunctionWithNameConstantValuesError(name, fc)
	if err != nil {
		return nil, core.E("native.sdpaVector2Pass1Pipeline", name, err)
	}
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.sdpaVector2Pass1Pipeline: kernel " + name + " not found")
	}
	pso, err := device.NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		return nil, core.E("native.sdpaVector2Pass1Pipeline", "pipeline "+name, err)
	}
	sdpaPSOCache[key] = pso
	return pso, nil
}

// sdpaVector2Pass2Pipeline builds (and caches) the sdpa_vector_2pass_2 combine
// kernel. It carries no function constants (MLX builds it plain) — blocks arrives
// as a runtime buffer — so a name-keyed lookup suffices.
func sdpaVector2Pass2Pipeline(name string) (metal.MTLComputePipelineState, error) {
	sdpaPSOMu.Lock()
	defer sdpaPSOMu.Unlock()
	if pso, ok := sdpaPSOCache[name]; ok {
		return pso, nil
	}
	if library == nil || library.GetID() == 0 {
		return nil, core.NewError("native.sdpaVector2Pass2Pipeline: library unavailable for " + name)
	}
	fn := library.NewFunctionWithName(name)
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.sdpaVector2Pass2Pipeline: kernel " + name + " not found")
	}
	pso, err := device.NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		return nil, core.E("native.sdpaVector2Pass2Pipeline", "pipeline "+name, err)
	}
	sdpaPSOCache[name] = pso
	return pso, nil
}

func sdpaVector2Pass1PipelineForHeadDim(headDim int, blocks int32) (metal.MTLComputePipelineState, error) {
	key := sdpa2Pass1Key{headDim: headDim, blocks: blocks}
	sdpaPSOMu.Lock()
	if pso, ok := sdpaVector2Pass1HeadDimCache[key]; ok {
		sdpaPSOMu.Unlock()
		return pso, nil
	}
	sdpaPSOMu.Unlock()

	pso, err := sdpaVector2Pass1Pipeline(core.Sprintf("sdpa_vector_2pass_1_bfloat16_t_%d_%d", headDim, headDim), blocks)
	if err != nil {
		return nil, err
	}

	sdpaPSOMu.Lock()
	if existing, ok := sdpaVector2Pass1HeadDimCache[key]; ok {
		sdpaPSOMu.Unlock()
		return existing, nil
	}
	sdpaVector2Pass1HeadDimCache[key] = pso
	sdpaPSOMu.Unlock()
	return pso, nil
}

func sdpaVector2Pass2PipelineForHeadDim(headDim int) (metal.MTLComputePipelineState, error) {
	sdpaPSOMu.Lock()
	if pso, ok := sdpaVector2Pass2HeadDimCache[headDim]; ok {
		sdpaPSOMu.Unlock()
		return pso, nil
	}
	sdpaPSOMu.Unlock()

	pso, err := sdpaVector2Pass2Pipeline(core.Sprintf("sdpa_vector_2pass_2_bfloat16_t_%d", headDim))
	if err != nil {
		return nil, err
	}

	sdpaPSOMu.Lock()
	if existing, ok := sdpaVector2Pass2HeadDimCache[headDim]; ok {
		sdpaPSOMu.Unlock()
		return existing, nil
	}
	sdpaVector2Pass2HeadDimCache[headDim] = pso
	sdpaPSOMu.Unlock()
	return pso, nil
}

// sdpa2PassDisabledForTest forces the decode SDPA back onto the single-pass kernel
// even when the 2-pass intermediates are present — a measurement/parity lever so a
// test can A/B the same live path with and without the long-context kernel.
var sdpa2PassDisabledForTest bool

// sdpa2PassMinKV is the attended-window length at which decode attention switches
// from single-pass sdpa_vector (one threadgroup per head reduces the whole cache) to
// the 2-pass kernels (the reduction fans over `blocks` threadgroups). Below the knee
// single-pass wins (no intermediate round-trip); at/above it the 2-pass saturation
// pays off — the single-pass kvLen<1024 guidance, made the routing threshold.
const sdpa2PassMinKV = 1024

// sdpa2PassBlocks picks the cache-split count for a kvLen — the number of
// threadgroups that share the softmax reduction. Single-pass uses one threadgroup
// per (b·head) and stalls past ~1024 because that one group reduces the whole
// cache; 2-pass fans the reduction over `blocks` groups, so saturation grows with
// context. Must stay a multiple of BN=32 (the pass-2 combine loops blocks/32).
// The ladder mirrors MLX's own heuristic (more blocks as N climbs).
func sdpa2PassBlocks(kvLen int) int32 {
	switch {
	case kvLen <= 8192:
		return 64
	case kvLen <= 32768:
		return 128
	case kvLen <= 65536:
		return 256
	default:
		return 512
	}
}

// SDPA2Pass computes single-query scaled-dot-product attention over a contiguous KV
// cache via MLX's TWO-pass sdpa_vector kernels — the long-context path. Pass 1
// (sdpa_vector_2pass_1) splits the cache into `blocks` segments across threadgroups,
// each emitting a partial weighted-V sum + that segment's online-softmax (sum, max)
// into intermediate buffers; pass 2 (sdpa_vector_2pass_2) merges the per-block
// partials back into one head output. Same inputs/outputs and byte ABI intent as
// SDPA (raw bf16, q (b,nHeads,1,headDim), k/v (b,nKVHeads,kvLen,headDim) → out
// (b,nHeads,1,headDim)) — but it keeps scaling past kvLen~1024 where SDPA's single
// threadgroup-per-head reduction degrades. Token-identical to SDPA (online softmax,
// same maths); validated cosine~1 vs a host float reference in sdpa_2pass_test.go.
func SDPA2Pass(qb, kb, vb []byte, b, nHeads, nKVHeads, headDim, kvLen int, scale float32) ([]byte, error) {
	return SDPA2PassInto(nil, qb, kb, vb, b, nHeads, nKVHeads, headDim, kvLen, scale)
}

func SDPA2PassInto(out []byte, qb, kb, vb []byte, b, nHeads, nKVHeads, headDim, kvLen int, scale float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if nKVHeads == 0 || nHeads%nKVHeads != 0 {
		return nil, core.NewError("native.SDPA2Pass: nHeads must be a multiple of nKVHeads")
	}
	blocks := sdpa2PassBlocks(kvLen)
	pso1, err := sdpaVector2Pass1PipelineForHeadDim(headDim, blocks)
	if err != nil {
		return nil, err
	}
	pso2, err := sdpaVector2Pass2PipelineForHeadDim(headDim)
	if err != nil {
		return nil, err
	}

	const bf16Size = 2
	outLen := b * nHeads * headDim * bf16Size
	callerOut := cap(out) >= outLen
	if !callerOut {
		out = make([]byte, outLen)
	} else {
		out = out[:outLen]
	}
	var encErr error
	withAutoreleasePool(func() {
		scratch, err := getSDPABF16Scratch(len(qb), len(kb), len(vb), outLen)
		if err != nil {
			encErr = err
			return
		}
		defer putSDPABF16Scratch(scratch)
		qBuf, kBuf, vBuf, outBuf, err := scratch.buffers(qb, kb, vb)
		if err != nil {
			encErr = err
			return
		}
		directOut := false
		if callerOut {
			if tmp, ok := scratch.outputView(out); ok {
				outBuf = tmp
				directOut = true
			}
		}
		// intermediates: partials [b·nHeads·blocks·headDim] bf16, sums/maxs [b·nHeads·blocks] float32.
		nbh := b * nHeads
		partials, sums, maxs, err := scratch.twoPassBuffers(nbh, blocks, headDim)
		if err != nil {
			encErr = err
			return
		}

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		sink := encSink{enc}
		emitSDPA2Pass1(sink, pso1, qBuf, kBuf, vBuf, partials, sums, maxs, 0, b, nHeads, nKVHeads, kvLen, int(blocks), int64(kvLen*headDim), int64(headDim), int64(kvLen*headDim), int64(headDim), scale)
		emitSDPA2Pass2(sink, pso2, partials, sums, maxs, outBuf, b, nHeads, int(blocks))
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)

		if !directOut {
			copy(out, scratch.out.bytes[:outLen])
		}
	})
	if encErr != nil {
		return nil, encErr
	}
	return out, nil
}

// SDPA computes single-query scaled-dot-product attention (the decode path) over
// a contiguous KV cache, driving MLX's sdpa_vector kernel directly (no cgo).
// Inputs are raw bfloat16 bytes — the only dtype the decode attention kernel is
// compiled for — laid out as q (b, nHeads, 1, headDim), k/v (b, nKVHeads, kvLen,
// headDim); the result is the bfloat16 output bytes, shape (b, nHeads, 1,
// headDim). nHeads/nKVHeads gives the GQA factor. Buffer ABI: q(0) k(1) v(2)
// out(3) gqa_factor(4) N(5) k_head_stride(6) k_seq_stride(7) v_head_stride(8)
// v_seq_stride(9) scale(10), strides in elements; one threadgroup per (b·head).
// No mask / not causal. Byte-for-byte parity with pkg/metal.ScaledDotProductAttention
// is gated in parity_test.go.
//
// kvLen must stay under 1024 to keep MLX on the single-pass kernel (the 2-pass
// kernel accumulates the softmax differently); decode against a longer cache is
// the sdpa_vector_2pass follow-up.
func SDPA(qb, kb, vb []byte, b, nHeads, nKVHeads, headDim, kvLen int, scale float32) ([]byte, error) {
	return SDPAInto(nil, qb, kb, vb, b, nHeads, nKVHeads, headDim, kvLen, scale)
}

func SDPAInto(out []byte, qb, kb, vb []byte, b, nHeads, nKVHeads, headDim, kvLen int, scale float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if nKVHeads == 0 || nHeads%nKVHeads != 0 {
		return nil, core.NewError("native.SDPA: nHeads must be a multiple of nKVHeads")
	}
	pso, err := sdpaVectorPipelineForHeadDim(headDim)
	if err != nil {
		return nil, err
	}

	const bf16Size = 2
	outLen := b * nHeads * headDim * bf16Size
	callerOut := cap(out) >= outLen
	if !callerOut {
		out = make([]byte, outLen)
	} else {
		out = out[:outLen]
	}
	var encErr error
	withAutoreleasePool(func() {
		scratch, err := getSDPABF16Scratch(len(qb), len(kb), len(vb), outLen)
		if err != nil {
			encErr = err
			return
		}
		defer putSDPABF16Scratch(scratch)
		qBuf, kBuf, vBuf, outBuf, err := scratch.buffers(qb, kb, vb)
		if err != nil {
			encErr = err
			return
		}
		directOut := false
		if callerOut {
			if tmp, ok := scratch.outputView(out); ok {
				outBuf = tmp
				directOut = true
			}
		}

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		emitSDPA(encSink{enc}, pso, qBuf, kBuf, vBuf, outBuf, 0, nil, b*nHeads, b*nKVHeads, kvLen, int64(kvLen*headDim), int64(headDim), int64(kvLen*headDim), int64(headDim), scale)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)

		if !directOut {
			copy(out, scratch.out.bytes[:outLen])
		}
	})
	if encErr != nil {
		return nil, encErr
	}
	return out, nil
}
