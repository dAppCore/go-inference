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

// matmul_steel.go drives MLX's fused steel GEMM directly (no cgo) so a float32 matmul is
// BYTE-IDENTICAL to pkg/metal.Matmul. The bf16 gemv-loop (MatRowsBF16) matches metal.Matmul because
// the bf16 rounding absorbs the accumulation-order difference; float32 has no such rounding, so the
// Conformer audio attention — which runs in float32 — needs the same tiled GEMM metal dispatches.
// This wraps the steel_gemm_fused kernel (the no-axpby, no-batch, contiguous A·B path) with the
// default large-device tiling bm64 bn64 bk16 wm2 wn2.

var (
	steelPSOMu    sync.Mutex
	steelPSOCache = map[string]metal.MTLComputePipelineState{}

	steelPipelineKeyMu    sync.Mutex
	steelPipelineKeyCache = map[steelPipelineKeyParts]string{}

	steelSplitKNameMu    sync.Mutex
	steelSplitKNameCache = map[steelSplitKNameKey]string{}

	matMulF32SteelScratchPools  sync.Map
	errMatMulF32SteelScratchDim = core.NewError("native.matMulF32SteelScratch: dimension mismatch")

	matMulF32SplitKParamsScratchPools sync.Map
	errMatMulF32SplitKParamsDim       = core.NewError("native.matMulF32SplitKParamsScratch: dimension mismatch")

	matMulF32SplitKAccumScratchPools sync.Map
)

type steelPipelineKeyParts struct {
	name string
	bits uint8
}

type steelSplitKNameKey struct {
	bm, bn, bk, wm, wn  int
	mnAligned, kAligned bool
}

func steelBoolBits(hasBatch, useOutSource, doAxpby, alignM, alignN, alignK bool) uint8 {
	var bits uint8
	if hasBatch {
		bits |= 1 << 0
	}
	if useOutSource {
		bits |= 1 << 1
	}
	if doAxpby {
		bits |= 1 << 2
	}
	if alignM {
		bits |= 1 << 3
	}
	if alignN {
		bits |= 1 << 4
	}
	if alignK {
		bits |= 1 << 5
	}
	return bits
}

func steelPipelineKey(name string, hasBatch, useOutSource, doAxpby, alignM, alignN, alignK bool) string {
	parts := steelPipelineKeyParts{
		name: name,
		bits: steelBoolBits(hasBatch, useOutSource, doAxpby, alignM, alignN, alignK),
	}
	steelPipelineKeyMu.Lock()
	if key, ok := steelPipelineKeyCache[parts]; ok {
		steelPipelineKeyMu.Unlock()
		return key
	}
	suffix := [6]byte{'f', 'f', 'f', 'f', 'f', 'f'}
	if hasBatch {
		suffix[0] = 't'
	}
	if useOutSource {
		suffix[1] = 't'
	}
	if doAxpby {
		suffix[2] = 't'
	}
	if alignM {
		suffix[3] = 't'
	}
	if alignN {
		suffix[4] = 't'
	}
	if alignK {
		suffix[5] = 't'
	}
	key := name + "|" + string(suffix[:])
	steelPipelineKeyCache[parts] = key
	steelPipelineKeyMu.Unlock()
	return key
}

func steelSplitKKernelName(bm, bn, bk, wm, wn int, mnAligned, kAligned bool) string {
	key := steelSplitKNameKey{bm: bm, bn: bn, bk: bk, wm: wm, wn: wn, mnAligned: mnAligned, kAligned: kAligned}
	steelSplitKNameMu.Lock()
	if name, ok := steelSplitKNameCache[key]; ok {
		steelSplitKNameMu.Unlock()
		return name
	}
	al := func(b bool) string {
		if b {
			return "t"
		}
		return "n"
	}
	name := core.Sprintf("steel_gemm_splitk_nt_float32_float32_bm%d_bn%d_bk%d_wm%d_wn%d_MN_%saligned_K_%saligned",
		bm, bn, bk, wm, wn, al(mnAligned), al(kAligned))
	steelSplitKNameCache[key] = name
	steelSplitKNameMu.Unlock()
	return name
}

// steelGemmPipeline builds (and caches) the steel_gemm_fused float32 kernel specialised by MLX's six
// boolean function constants (has_batch 10, use_out_source 100, do_axpby 110, align_M 200, align_N
// 201, align_K 202) — the same set mlx-c sets, so the dispatched kernel is identical.
func steelGemmPipeline(name string, hasBatch, useOutSource, doAxpby, alignM, alignN, alignK bool) (metal.MTLComputePipelineState, error) {
	key := steelPipelineKey(name, hasBatch, useOutSource, doAxpby, alignM, alignN, alignK)
	steelPSOMu.Lock()
	defer steelPSOMu.Unlock()
	if pso, ok := steelPSOCache[key]; ok {
		return pso, nil
	}
	if library == nil || library.GetID() == 0 {
		return nil, core.NewError("native.steelGemmPipeline: library unavailable for " + name)
	}
	fc := metal.NewMTLFunctionConstantValues()
	set := func(v bool, idx uint) {
		b := uint8(0)
		if v {
			b = 1
		}
		fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&b), metal.MTLDataTypeBool, idx)
	}
	set(hasBatch, 10)
	set(useOutSource, 100)
	set(doAxpby, 110)
	set(alignM, 200)
	set(alignN, 201)
	set(alignK, 202)

	fn, err := library.NewFunctionWithNameConstantValuesError(name, fc)
	if err != nil {
		return nil, core.E("native.steelGemmPipeline", name, err)
	}
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.steelGemmPipeline: kernel " + name + " not found")
	}
	pso, err := device.NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		return nil, core.E("native.steelGemmPipeline", "pipeline "+name, err)
	}
	steelPSOCache[key] = pso
	return pso, nil
}

// steelTiling is one tiling/kernel choice. MLX picks tiling by device + dtype + transpose
// (GEMM_TPARAM_MACRO). This Mac is a small-device arch ('g'/'p'): float32 nn falls through to the
// default 64/64/16/2/2; float32 nt is 64/32/32/2/2 (matmul.cpp). Mismatching the tiling — or the
// nn/nt kernel — changes the accumulation order and breaks f32 byte-parity (nt≠nn at some shapes).
type steelTiling struct {
	bm, bn, bk, wm, wn int
	name               string
}

type matMulF32SteelScratch struct {
	M, K, N       int
	bm, bn, bk    int
	ldb           int
	a, out        *pinnedNoCopyBytes
	aView         cachedNoCopyBytesView
	params        *pinnedNoCopyBytes
	paramsFilled  bool
	outViewPtr    uintptr
	outViewLen    int
	outView       metal.MTLBuffer
	outViewPinned *pinnedNoCopyBytes
}

type matMulF32SteelScratchKey struct {
	M, K, N, ldb int
	bm, bn, bk   int
}

func newMatMulF32SteelScratch(M, K, N, ldb int, t steelTiling) (*matMulF32SteelScratch, error) {
	if M <= 0 || K <= 0 || N <= 0 {
		return nil, core.NewError("native.newMatMulF32SteelScratch: invalid dimensions")
	}
	a, err := newPinnedNoCopyBytes(M * K * 4)
	if err != nil {
		return nil, err
	}
	out, err := newPinnedNoCopyBytes(M * N * 4)
	if err != nil {
		a.Close()
		return nil, err
	}
	params, err := newPinnedNoCopyBytes(72)
	if err != nil {
		a.Close()
		out.Close()
		return nil, err
	}
	return &matMulF32SteelScratch{
		M: M, K: K, N: N,
		bm: t.bm, bn: t.bn, bk: t.bk, ldb: ldb,
		a: a, out: out, params: params,
	}, nil
}

func matMulF32SteelScratchPoolFor(M, K, N, ldb int, t steelTiling) *sync.Pool {
	key := matMulF32SteelScratchKey{M: M, K: K, N: N, ldb: ldb, bm: t.bm, bn: t.bn, bk: t.bk}
	if v, ok := matMulF32SteelScratchPools.Load(key); ok {
		return v.(*sync.Pool)
	}
	pool := &sync.Pool{}
	if v, loaded := matMulF32SteelScratchPools.LoadOrStore(key, pool); loaded {
		return v.(*sync.Pool)
	}
	return pool
}

func getMatMulF32SteelScratch(M, K, N, ldb int, t steelTiling) (*matMulF32SteelScratch, error) {
	pool := matMulF32SteelScratchPoolFor(M, K, N, ldb, t)
	if v := pool.Get(); v != nil {
		s := v.(*matMulF32SteelScratch)
		if s.M == M && s.K == K && s.N == N && s.ldb == ldb &&
			s.bm == t.bm && s.bn == t.bn && s.bk == t.bk &&
			s.a != nil && s.out != nil && s.params != nil {
			return s, nil
		}
		s.Close()
	}
	return newMatMulF32SteelScratch(M, K, N, ldb, t)
}

func putMatMulF32SteelScratch(s *matMulF32SteelScratch) {
	if s != nil {
		t := steelTiling{bm: s.bm, bn: s.bn, bk: s.bk}
		matMulF32SteelScratchPoolFor(s.M, s.K, s.N, s.ldb, t).Put(s)
	}
}

func (s *matMulF32SteelScratch) Close() {
	if s == nil {
		return
	}
	if s.a != nil {
		s.a.Close()
		s.a = nil
	}
	if s.out != nil {
		s.out.Close()
		s.out = nil
	}
	if s.params != nil {
		s.params.Close()
		s.params = nil
	}
	s.aView.Close()
	s.closeOutputView()
	s.M, s.K, s.N = 0, 0, 0
	s.bm, s.bn, s.bk, s.ldb = 0, 0, 0, 0
	s.paramsFilled = false
}

func (s *matMulF32SteelScratch) closeOutputView() {
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

func (s *matMulF32SteelScratch) outputView(out []float32) (metal.MTLBuffer, bool) {
	if s == nil || len(out) == 0 {
		return nil, false
	}
	ptr := uintptr(unsafe.Pointer(&out[0]))
	if s.outView != nil && s.outViewPtr == ptr && s.outViewLen == len(out) {
		return s.outView, true
	}
	s.closeOutputView()
	outBytes := float32Bytes(out)
	if buf, ok := registeredPinnedNoCopyBytes(outBytes); ok {
		s.outViewPtr = ptr
		s.outViewLen = len(out)
		s.outView = buf
		s.outViewPinned = nil
		return buf, true
	}
	buf, pinner, noCopy := residentNoCopyBytes(outBytes)
	if !noCopy {
		if pinner != nil {
			pinner.Unpin()
		}
		return nil, false
	}
	pinned := &pinnedNoCopyBytes{bytes: outBytes, buf: buf, pinner: pinner}
	runtime.SetFinalizer(pinned, (*pinnedNoCopyBytes).Close)
	s.outViewPtr = ptr
	s.outViewLen = len(out)
	s.outView = buf
	s.outViewPinned = pinned
	return buf, true
}

func (s *matMulF32SteelScratch) buffers(a []float32) (metal.MTLBuffer, metal.MTLBuffer, metal.MTLBuffer, error) {
	if s == nil || s.a == nil || s.out == nil || s.params == nil {
		return nil, nil, nil, core.NewError("native.matMulF32SteelScratch.buffers: scratch is nil")
	}
	if len(a) != s.M*s.K || len(s.out.bytes) != s.M*s.N*4 || len(s.params.bytes) != 72 {
		return nil, nil, nil, errMatMulF32SteelScratchDim
	}
	aBytes := float32Bytes(a)
	aBuf, ok := s.aView.buffer(aBytes)
	if !ok {
		var err error
		aBuf, err = s.a.copyBuffer(aBytes)
		if err != nil {
			return nil, nil, nil, err
		}
	}
	if !s.paramsFilled {
		tn, tm := (s.N+s.bn-1)/s.bn, (s.M+s.bm-1)/s.bm
		fillMatMulF32SteelParams(s.params.bytes, s.M, s.K, s.N, s.ldb, tn, tm, s.K/s.bk)
		s.paramsFilled = true
	}
	return aBuf, s.out.buf, s.params.buf, nil
}

func fillMatMulF32SteelParams(params []byte, M, K, N, ldb, tilesN, tilesM, kIterations int) {
	for i := range params {
		params[i] = 0
	}
	putI32 := func(off int, v int32) {
		params[off], params[off+1], params[off+2], params[off+3] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
	}
	putI32(0, int32(M))
	putI32(4, int32(N))
	putI32(8, int32(K))
	putI32(12, int32(K))   // lda
	putI32(16, int32(ldb)) // ldb (N for nn, K for nt)
	putI32(20, int32(N))   // ldd
	putI32(24, int32(tilesN))
	putI32(28, int32(tilesM))
	putI32(56, 0)
	putI32(60, int32(kIterations)) // gemm_k_iterations_aligned
	putI32(64, 1)                  // batch_ndim
}

type matMulF32SplitKParamsScratch struct {
	M, K, N               int
	tilesN, tilesM        int
	partitions, stride    int
	partSize, kIterations int
	params                *pinnedNoCopyBytes
	paramsFilled          bool
}

type matMulF32SplitKParamsScratchKey struct {
	M, K, N               int
	tilesN, tilesM        int
	partitions, stride    int
	partSize, kIterations int
}

type matMulF32SplitKParamsScratchPool struct {
	core.Pool[*matMulF32SplitKParamsScratch]
}

func newMatMulF32SplitKParamsScratch(M, K, N, tilesN, tilesM, partitions, stride, partSize, kIterations int) (*matMulF32SplitKParamsScratch, error) {
	if M <= 0 || K <= 0 || N <= 0 || tilesN <= 0 || tilesM <= 0 || partitions <= 0 || stride <= 0 || partSize <= 0 || kIterations <= 0 {
		return nil, core.NewError("native.newMatMulF32SplitKParamsScratch: invalid dimensions")
	}
	params, err := newPinnedNoCopyBytes(52)
	if err != nil {
		return nil, err
	}
	return &matMulF32SplitKParamsScratch{
		M: M, K: K, N: N,
		tilesN: tilesN, tilesM: tilesM,
		partitions: partitions, stride: stride, partSize: partSize, kIterations: kIterations,
		params: params,
	}, nil
}

func matMulF32SplitKParamsScratchPoolFor(M, K, N, tilesN, tilesM, partitions, stride, partSize, kIterations int) *matMulF32SplitKParamsScratchPool {
	key := matMulF32SplitKParamsScratchKey{
		M: M, K: K, N: N,
		tilesN: tilesN, tilesM: tilesM,
		partitions: partitions, stride: stride,
		partSize: partSize, kIterations: kIterations,
	}
	if v, ok := matMulF32SplitKParamsScratchPools.Load(key); ok {
		return v.(*matMulF32SplitKParamsScratchPool)
	}
	pool := &matMulF32SplitKParamsScratchPool{}
	if v, loaded := matMulF32SplitKParamsScratchPools.LoadOrStore(key, pool); loaded {
		return v.(*matMulF32SplitKParamsScratchPool)
	}
	return pool
}

func getMatMulF32SplitKParamsScratch(M, K, N, tilesN, tilesM, partitions, stride, partSize, kIterations int) (*matMulF32SplitKParamsScratch, error) {
	pool := matMulF32SplitKParamsScratchPoolFor(M, K, N, tilesN, tilesM, partitions, stride, partSize, kIterations)
	if s := pool.Get(); s != nil {
		if s.M == M && s.K == K && s.N == N &&
			s.tilesN == tilesN && s.tilesM == tilesM &&
			s.partitions == partitions && s.stride == stride &&
			s.partSize == partSize && s.kIterations == kIterations &&
			s.params != nil {
			return s, nil
		}
		s.Close()
	}
	return newMatMulF32SplitKParamsScratch(M, K, N, tilesN, tilesM, partitions, stride, partSize, kIterations)
}

func putMatMulF32SplitKParamsScratch(s *matMulF32SplitKParamsScratch) {
	if s != nil && s.M > 0 && s.K > 0 && s.N > 0 && s.tilesN > 0 && s.tilesM > 0 &&
		s.partitions > 0 && s.stride > 0 && s.partSize > 0 && s.kIterations > 0 && s.params != nil {
		matMulF32SplitKParamsScratchPoolFor(s.M, s.K, s.N, s.tilesN, s.tilesM, s.partitions, s.stride, s.partSize, s.kIterations).Put(s)
	}
}

func (s *matMulF32SplitKParamsScratch) Close() {
	if s == nil {
		return
	}
	if s.params != nil {
		s.params.Close()
		s.params = nil
	}
	s.M, s.K, s.N = 0, 0, 0
	s.tilesN, s.tilesM = 0, 0
	s.partitions, s.stride = 0, 0
	s.partSize, s.kIterations = 0, 0
	s.paramsFilled = false
}

func (s *matMulF32SplitKParamsScratch) buffer() (metal.MTLBuffer, error) {
	if s == nil || s.params == nil {
		return nil, core.NewError("native.matMulF32SplitKParamsScratch.buffer: scratch is nil")
	}
	if len(s.params.bytes) != 52 {
		return nil, errMatMulF32SplitKParamsDim
	}
	if !s.paramsFilled {
		fillMatMulF32SplitKParams(s.params.bytes, s.M, s.K, s.N, s.tilesN, s.tilesM, s.partitions, s.stride, s.partSize, s.kIterations)
		s.paramsFilled = true
	}
	return s.params.buf, nil
}

func fillMatMulF32SplitKParams(params []byte, M, K, N, tilesN, tilesM, partitions, stride, partSize, kIterations int) {
	for i := range params {
		params[i] = 0
	}
	putI32 := func(off, v int) {
		params[off], params[off+1], params[off+2], params[off+3] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
	}
	putI32(0, M)
	putI32(4, N)
	putI32(8, K)
	putI32(12, K) // lda
	putI32(16, K) // ldb (nt)
	putI32(20, N) // ldc
	putI32(24, tilesN)
	putI32(28, tilesM)
	putI32(32, partitions)
	putI32(36, stride)
	putI32(40, partSize)
	putI32(44, 0) // swizzle_log
	putI32(48, kIterations)
}

type matMulF32SplitKAccumScratch struct {
	M, N, partitions int
	split            *pinnedNoCopyBytes
}

type matMulF32SplitKAccumScratchKey struct {
	M, N, partitions int
}

type matMulF32SplitKAccumScratchPool struct {
	core.Pool[*matMulF32SplitKAccumScratch]
}

func newMatMulF32SplitKAccumScratch(M, N, partitions int) (*matMulF32SplitKAccumScratch, error) {
	if M <= 0 || N <= 0 || partitions <= 0 {
		return nil, core.NewError("native.newMatMulF32SplitKAccumScratch: invalid dimensions")
	}
	split, err := newPinnedNoCopyBytes(partitions * M * N * 4)
	if err != nil {
		return nil, err
	}
	return &matMulF32SplitKAccumScratch{M: M, N: N, partitions: partitions, split: split}, nil
}

func matMulF32SplitKAccumScratchPoolFor(M, N, partitions int) *matMulF32SplitKAccumScratchPool {
	key := matMulF32SplitKAccumScratchKey{M: M, N: N, partitions: partitions}
	if v, ok := matMulF32SplitKAccumScratchPools.Load(key); ok {
		return v.(*matMulF32SplitKAccumScratchPool)
	}
	pool := &matMulF32SplitKAccumScratchPool{}
	if v, loaded := matMulF32SplitKAccumScratchPools.LoadOrStore(key, pool); loaded {
		return v.(*matMulF32SplitKAccumScratchPool)
	}
	return pool
}

func getMatMulF32SplitKAccumScratch(M, N, partitions int) (*matMulF32SplitKAccumScratch, error) {
	pool := matMulF32SplitKAccumScratchPoolFor(M, N, partitions)
	if s := pool.Get(); s != nil {
		if s.M == M && s.N == N && s.partitions == partitions && s.split != nil {
			return s, nil
		}
		s.Close()
	}
	return newMatMulF32SplitKAccumScratch(M, N, partitions)
}

func putMatMulF32SplitKAccumScratch(s *matMulF32SplitKAccumScratch) {
	if s != nil && s.M > 0 && s.N > 0 && s.partitions > 0 && s.split != nil {
		matMulF32SplitKAccumScratchPoolFor(s.M, s.N, s.partitions).Put(s)
	}
}

func (s *matMulF32SplitKAccumScratch) Close() {
	if s == nil {
		return
	}
	if s.split != nil {
		s.split.Close()
		s.split = nil
	}
	s.M, s.N, s.partitions = 0, 0, 0
}

func (s *matMulF32SplitKAccumScratch) buffer() (metal.MTLBuffer, error) {
	if s == nil || s.split == nil || s.split.buf == nil {
		return nil, core.NewError("native.matMulF32SplitKAccumScratch.buffer: scratch is nil")
	}
	if len(s.split.bytes) != s.partitions*s.M*s.N*4 {
		return nil, core.NewError("native.matMulF32SplitKAccumScratch.buffer: dimension mismatch")
	}
	return s.split.buf, nil
}

var (
	steelNN = steelTiling{64, 64, 16, 2, 2, "steel_gemm_fused_nn_float32_float32_bm64_bn64_bk16_wm2_wn2"}
	steelNT = steelTiling{64, 32, 32, 2, 2, "steel_gemm_fused_nt_float32_float32_bm64_bn32_bk32_wm2_wn2"}
)

// MatMulF32 computes out[M,N] = a[M,K] @ b[K,N] (row-major contiguous f32) through MLX's fused steel
// GEMM — BYTE-IDENTICAL to pkg/metal.Matmul on the same f32 arrays. nn, no output source, no axpby.
func MatMulF32(a, b []float32, M, K, N int) ([]float32, error) {
	return MatMulF32Into(nil, a, b, M, K, N)
}

// MatMulF32Into is MatMulF32 with caller-owned output storage when cap(out) >= M*N.
func MatMulF32Into(out, a, b []float32, M, K, N int) ([]float32, error) {
	return matMulF32Into(out, a, b, M, K, N, true)
}

func matMulF32Into(out, a, b []float32, M, K, N int, directOutput bool) ([]float32, error) {
	outLen := M * N
	callerOut := cap(out) >= outLen
	if !callerOut {
		out = make([]float32, outLen)
	} else {
		out = out[:outLen]
	}
	if err := matMulF32CoreInto(out, a, b, M, K, N, steelNN, false, directOutput && callerOut); err != nil {
		return nil, err
	}
	return out, nil
}

// MatMulF32NT computes out[M,N] = a[M,K] @ b[N,K]ᵀ (b stored row-major [N,K]) — BYTE-IDENTICAL to
// metal.Matmul(a, Transpose(b)). It replicates MLX's dispatch (matmul.cpp): for f32 without TF32,
// use_nax is false, so small-M·N-with-large-K routes to SIMD split-K (a different accumulation than
// the fused kernel); everything else uses the fused nt kernel. The Conformer relative-key projection
// (Matmul(PosEmbed, Transpose(W)), M=PosCount tiny, K=hidden large) is exactly a split-K case — the
// nn or fused nt kernel diverges ~1 ULP there.
func MatMulF32NT(a, b []float32, M, K, N int) ([]float32, error) {
	return MatMulF32NTInto(nil, a, b, M, K, N)
}

// MatMulF32NTInto is MatMulF32NT with caller-owned output storage when cap(out) >= M*N.
func MatMulF32NTInto(out, a, b []float32, M, K, N int) ([]float32, error) {
	return matMulF32NTIntoPublic(out, a, b, M, K, N, true)
}

func matMulF32NTIntoPublic(out, a, b []float32, M, K, N int, directOutput bool) ([]float32, error) {
	outLen := M * N
	callerOut := cap(out) >= outLen
	if !callerOut {
		out = make([]float32, outLen)
	} else {
		out = out[:outLen]
	}
	if err := matMulF32NTInto(out, a, b, M, K, N, directOutput && callerOut); err != nil {
		return nil, err
	}
	return out, nil
}

func matMulF32NTInto(out, a, b []float32, M, K, N int, directOutput bool) error {
	dtm, dtn, dtk := (M+15)/16, (N+15)/16, K/16
	maxMN := M
	if N > maxMN {
		maxMN = N
	}
	// Case 1 (matmul.cpp): !use_nax && batch==1 && _tm·_tn ≤ threshold && _tk ≥ 8 && K ≥ max(M,N).
	// threshold is device-dependent: 1024 (small device 'g'/'p' — this Mac, confirmed by the nn
	// tiling) / 2048 ('s'/'d'). relK's _tm·_tn is far below either, so the audio tower is unaffected;
	// a shape with _tm·_tn ∈ (1024, 2048] on a bigger Apple GPU would need the device's real threshold
	// (the byte-parity test would catch the mismatch).
	const splitKThreshold = 1024
	if dtm*dtn <= splitKThreshold && dtk >= 8 && K >= maxMN {
		return matMulF32SplitKNTInto(out, a, b, M, K, N, directOutput)
	}
	return matMulF32CoreInto(out, a, b, M, K, N, steelNT, true, directOutput)
}

func nextPow2(n int) int {
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

// getBlockDims mirrors mlx's get_block_dims (utils): largest per-axis powers of two whose log-sum ≤ 10.
func getBlockDims(d0, d1, d2 int) (uint, uint, uint) {
	pows := [3]int{}
	sum := 0
	for {
		presum := sum
		if d0 >= 1<<(pows[0]+1) {
			pows[0]++
			sum++
		}
		if sum == 10 {
			break
		}
		if d1 >= 1<<(pows[1]+1) {
			pows[1]++
			sum++
		}
		if sum == 10 {
			break
		}
		if d2 >= 1<<(pows[2]+1) {
			pows[2]++
			sum++
		}
		if sum == 10 || presum == sum {
			break
		}
	}
	return uint(1 << pows[0]), uint(1 << pows[1]), uint(1 << pows[2])
}

// matMulF32SplitKNT runs MLX's non-NAX SIMD split-K (steel_gemm_splitk + accum), nt — byte-identical
// to metal.Matmul on a split-K-dispatched shape. K is partitioned; each partition writes a partial
// GEMM into C_split[p], then the accum kernel sums the partitions into out. b is [N,K].
func matMulF32SplitKNT(a, b []float32, M, K, N int) ([]float32, error) {
	out := make([]float32, M*N)
	if err := matMulF32SplitKNTInto(out, a, b, M, K, N, false); err != nil {
		return nil, err
	}
	return out, nil
}

func matMulF32SplitKNTInto(out, a, b []float32, M, K, N int, directOutput bool) error {
	if err := ensureInit(); err != nil {
		return err
	}
	if len(a) != M*K || len(b) != K*N || len(out) != M*N {
		return core.NewError("native.matMulF32SplitKNT: size mismatch")
	}
	if M == 0 || N == 0 || K == 0 {
		return nil
	}
	bm, bn, bk, wm, wn := 16, 32, 16, 2, 2
	if M >= 40 {
		bm = 32
	}
	if N < 40 {
		bn = 16
	}
	ptm, ptn, ptk := (M+31)/32, (N+31)/32, K/16
	partitions := nextPow2(ptk / (ptm * ptn))
	if partitions < 2 {
		partitions = 2
	}
	if partitions > 32 {
		partitions = 32
	}
	stride := M * N
	kIters := (K / bk) / partitions
	partSize := kIters * bk
	mnAligned := M%bm == 0 && N%bn == 0
	kAligned := K%bk == 0
	gemmName := steelSplitKKernelName(bm, bn, bk, wm, wn, mnAligned, kAligned)
	gemmPSO, err := pipelineFor(gemmName)
	if err != nil {
		return err
	}
	accumPSO, err := pipelineFor("steel_gemm_splitk_accum_float32_float32")
	if err != nil {
		return err
	}
	tn, tm := (N+bn-1)/bn, (M+bm-1)/bm

	bd0, bd1, bd2 := getBlockDims(N, M, 1)
	var encErr error
	withAutoreleasePool(func() {
		ioScratch, err := getQMVFloatScratch(M*N, M*K)
		if err != nil {
			encErr = err
			return
		}
		defer putQMVFloatScratch(ioScratch)
		aBuf, outBuf, err := ioScratch.buffers(a)
		if err != nil {
			encErr = err
			return
		}
		directOut := false
		if directOutput {
			if tmp, ok := ioScratch.outputView(out); ok {
				outBuf = tmp
				directOut = true
			}
		}
		bBuf := residentFloat32(b)
		accScratch, err := getMatMulF32SplitKAccumScratch(M, N, partitions)
		if err != nil {
			encErr = err
			return
		}
		defer putMatMulF32SplitKAccumScratch(accScratch)
		cSplit, err := accScratch.buffer()
		if err != nil {
			encErr = err
			return
		}
		paramsScratch, err := getMatMulF32SplitKParamsScratch(M, K, N, tn, tm, partitions, stride, partSize, kIters)
		if err != nil {
			encErr = err
			return
		}
		defer putMatMulF32SplitKParamsScratch(paramsScratch)
		paramsBuf, err := paramsScratch.buffer()
		if err != nil {
			encErr = err
			return
		}
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		emitSteelSplitKGemm(encSink{enc}, gemmPSO, aBuf, bBuf, cSplit, paramsBuf, tn, tm, partitions, uint(wn), uint(wm))
		endEncodingFast(enc)

		acc := computeCommandEncoderFast(cb)
		emitSteelSplitKAccum(encSink{acc}, accumPSO, cSplit, outBuf, partitions, stride, M, N, bd0, bd1, bd2)
		endEncodingFast(acc)

		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, unsafe.Slice((*float32)(unsafe.Pointer(&ioScratch.out.bytes[0])), M*N))
		}
	})
	return encErr
}

// matMulF32Core drives one steel GEMM. b is [K,N] when !transposeB, [N,K] when transposeB (the kernel
// transposes it). lda is always K; ldb is N for nn, K for nt; ldd is N.
func matMulF32Core(a, b []float32, M, K, N int, t steelTiling, transposeB bool) ([]float32, error) {
	out := make([]float32, M*N)
	if err := matMulF32CoreInto(out, a, b, M, K, N, t, transposeB, false); err != nil {
		return nil, err
	}
	return out, nil
}

func matMulF32CoreInto(out, a, b []float32, M, K, N int, t steelTiling, transposeB, directOutput bool) error {
	if err := ensureInit(); err != nil {
		return err
	}
	if len(a) != M*K || len(b) != K*N || len(out) != M*N {
		return core.NewError("native.matMulF32Core: size mismatch")
	}
	if M == 0 || N == 0 || K == 0 {
		return nil
	}
	alignM, alignN, alignK := M%t.bm == 0, N%t.bn == 0, K%t.bk == 0
	pso, err := steelGemmPipeline(t.name, false, false, false, alignM, alignN, alignK)
	if err != nil {
		return err
	}
	tn, tm := (N+t.bn-1)/t.bn, (M+t.bm-1)/t.bm
	ldb := N
	if transposeB {
		ldb = K
	}

	var encErr error
	withAutoreleasePool(func() {
		ioScratch, err := getMatMulF32SteelScratch(M, K, N, ldb, t)
		if err != nil {
			encErr = err
			return
		}
		defer putMatMulF32SteelScratch(ioScratch)
		aBuf, outBuf, paramsBuf, err := ioScratch.buffers(a)
		if err != nil {
			encErr = err
			return
		}
		directOut := false
		if directOutput {
			if tmp, ok := ioScratch.outputView(out); ok {
				outBuf = tmp
				directOut = true
			}
		}
		bBuf := residentFloat32(b)
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		emitSteelGemm(encSink{enc}, pso, aBuf, bBuf, outBuf, paramsBuf, tn, tm, uint(t.wn), uint(t.wm))
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, unsafe.Slice((*float32)(unsafe.Pointer(&ioScratch.out.bytes[0])), M*N))
		}
	})
	return encErr
}
