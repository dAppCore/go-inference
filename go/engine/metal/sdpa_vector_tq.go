// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/kv/turboquant"
	"github.com/tmc/apple/metal"
)

// sdpa_vector_tq.go — the TurboQuant live-KV decode SDPA lane (campaign #41
// S3, kernels/lthn_tq_kv.metal): pipeline resolvers, the dispatchSink emitters
// the recorded arch ICB and the host test drivers share, and the Π/centroid
// residency the whole lane keys on. Structure mirrors sdpa_vector_q8.go — the
// q8 pair is the structural template; the payload format is kv/turboquant's
// Q_mse wire layout (turboquant_device.go is the format authority: packed
// LSB-first centroid indices, ceil(d·bits/8) bytes per row per head, plus one
// f32 norm γ per row per head).
//
// This file (not turboquant_device.go) imports kv/turboquant: the session
// wiring is where Π/centroids come from — RotationMatrix(tqKVSeed, d) and
// Centroids(d, bits) narrowed to f32, resident once per (d) / (d, bits) and
// shared by every store/read dispatch. The S2 device kernels stay
// value-neutral exactly as documented there.

// tqKVSeed fixes the rotation for the live-KV lane. The cache never leaves the
// session (snapshots decline in v1), so the only contract is store == read —
// one process-wide constant.
const tqKVSeed = 41

// tqKVHeadDims are the instantiated SDPA head dims (lthn_sdpa_vector_tq_bf16_*).
// 512 is the gemma4 dense family's global_head_dim; 128/256 cover the common
// GQA geometries (and the kernel-gate matrix).
func tqKVHeadDimOK(d int) bool { return d == 128 || d == 256 || d == 512 }

// tqKVBitsOK are the instantiated store/read bit widths — the S1/S2 set.
func tqKVBitsOK(bits int) bool { return bits == 2 || bits == 3 || bits == 4 }

// tqKVGeometryOK is the qualification gate a GLOBAL owner layer must pass for
// the TurboQuant live cache: both bit widths instantiated and the head dim on
// the SDPA instantiation list (the store/rot kernels cap at 512 too).
func tqKVGeometryOK(kBits, vBits, headDim int) bool {
	return tqKVBitsOK(kBits) && tqKVBitsOK(vBits) && tqKVHeadDimOK(headDim)
}

// --- Π / centroid residency -------------------------------------------------------------------

var (
	tqKVPiMu        sync.Mutex
	tqKVPiCache     = map[int]metal.MTLBuffer{}    // headDim -> resident f32 [d,d]
	tqKVCentCache   = map[[2]int]metal.MTLBuffer{} // {headDim, bits} -> resident f32 [1<<bits]
	tqKVCentF32Mu   sync.Mutex
	tqKVCentF32Memo = map[[2]int][]float32{}
	tqKVPiF32Memo   = map[int][]float32{}
)

// tqKVPiF32 is the f32 rotation for headDim d (RotationMatrix(tqKVSeed, d)
// narrowed) — the exact values BOTH the device kernels and the host references
// use, memoised host-side for the test oracles.
func tqKVPiF32(d int) []float32 {
	tqKVCentF32Mu.Lock()
	defer tqKVCentF32Mu.Unlock()
	if pi, ok := tqKVPiF32Memo[d]; ok {
		return pi
	}
	pi64 := turboquant.RotationMatrix(tqKVSeed, d)
	pi := make([]float32, len(pi64))
	for i, v := range pi64 {
		pi[i] = float32(v)
	}
	tqKVPiF32Memo[d] = pi
	return pi
}

// tqKVCentroidsF32 is the f32 Lloyd-Max table for (d, bits), memoised like the
// rotation.
func tqKVCentroidsF32(d, bits int) []float32 {
	tqKVCentF32Mu.Lock()
	defer tqKVCentF32Mu.Unlock()
	key := [2]int{d, bits}
	if c, ok := tqKVCentF32Memo[key]; ok {
		return c
	}
	c64 := turboquant.Centroids(d, bits)
	c := make([]float32, len(c64))
	for i, v := range c64 {
		c[i] = float32(v)
	}
	tqKVCentF32Memo[key] = c
	return c
}

// tqKVPiBuffer returns the resident device rotation for headDim d, uploaded
// once per process and shared by every session (Π is seed+d determined).
func tqKVPiBuffer(d int) metal.MTLBuffer {
	tqKVPiMu.Lock()
	defer tqKVPiMu.Unlock()
	if b, ok := tqKVPiCache[d]; ok {
		return b
	}
	b := residentFloat32(tqKVPiF32(d))
	tqKVPiCache[d] = b
	return b
}

// tqKVCentroidsBuffer returns the resident centroid table for (d, bits).
func tqKVCentroidsBuffer(d, bits int) metal.MTLBuffer {
	tqKVPiMu.Lock()
	defer tqKVPiMu.Unlock()
	key := [2]int{d, bits}
	if b, ok := tqKVCentCache[key]; ok {
		return b
	}
	b := residentFloat32(tqKVCentroidsF32(d, bits))
	tqKVCentCache[key] = b
	return b
}

// --- pipeline resolvers -----------------------------------------------------------------------

var (
	tqKVPSOMu          sync.Mutex
	tqKVStorePSOCache  = map[int]metal.MTLComputePipelineState{}
	tqRotRowsPSOCache  = map[bool]metal.MTLComputePipelineState{}
	sdpaVectorTQPSOs   = map[[3]int]metal.MTLComputePipelineState{}
	sdpaV2P1TQPSOCache = map[[4]int]metal.MTLComputePipelineState{}
)

// tqKVStorePipeline resolves lthn_tq_kv_store_bf16_bN for bits ∈ {2,3,4}.
func tqKVStorePipeline(bits int) (metal.MTLComputePipelineState, error) {
	if !tqKVBitsOK(bits) {
		return nil, core.NewError("native.tqKVStorePipeline: unsupported bit width (want 2, 3, or 4)")
	}
	tqKVPSOMu.Lock()
	defer tqKVPSOMu.Unlock()
	if pso, ok := tqKVStorePSOCache[bits]; ok {
		if pso == nil {
			return nil, core.NewError("native.tqKVStorePipeline: kernel unavailable")
		}
		return pso, nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		return nil, core.NewError("native.tqKVStorePipeline: custom library unavailable")
	}
	name := core.Sprintf("lthn_tq_kv_store_bf16_b%d", bits)
	fn := customLibrary.NewFunctionWithName(name)
	if fn == nil || fn.GetID() == 0 {
		tqKVStorePSOCache[bits] = nil
		return nil, core.NewError("native.tqKVStorePipeline: kernel " + name + " not found")
	}
	pso, err := device.NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		tqKVStorePSOCache[bits] = nil
		return nil, err
	}
	tqKVStorePSOCache[bits] = pso
	return pso, nil
}

// tqRotRowsPipeline resolves lthn_tq_rot_rows_bf16 (Π·x) or
// lthn_tq_unrot_rows_bf16 (Πᵀ·x).
func tqRotRowsPipeline(transpose bool) (metal.MTLComputePipelineState, error) {
	tqKVPSOMu.Lock()
	defer tqKVPSOMu.Unlock()
	if pso, ok := tqRotRowsPSOCache[transpose]; ok {
		if pso == nil {
			return nil, core.NewError("native.tqRotRowsPipeline: kernel unavailable")
		}
		return pso, nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		return nil, core.NewError("native.tqRotRowsPipeline: custom library unavailable")
	}
	name := "lthn_tq_rot_rows_bf16"
	if transpose {
		name = "lthn_tq_unrot_rows_bf16"
	}
	fn := customLibrary.NewFunctionWithName(name)
	if fn == nil || fn.GetID() == 0 {
		tqRotRowsPSOCache[transpose] = nil
		return nil, core.NewError("native.tqRotRowsPipeline: kernel " + name + " not found")
	}
	pso, err := device.NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		tqRotRowsPSOCache[transpose] = nil
		return nil, err
	}
	tqRotRowsPSOCache[transpose] = pso
	return pso, nil
}

// tqSDPAFunctionConstants builds the (blocks, kBits, vBits) constant set the
// TQ SDPA kernels specialise on — fc 26 (2-pass only), 27, 28.
func tqSDPAFunctionConstants(blocks int32, kBits, vBits int32, withBlocks bool) metal.MTLFunctionConstantValues {
	fc := metal.NewMTLFunctionConstantValues()
	if withBlocks {
		blk := blocks
		fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&blk), metal.MTLDataTypeInt, 26)
	}
	kb, vb := kBits, vBits
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&kb), metal.MTLDataTypeInt, 27)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&vb), metal.MTLDataTypeInt, 28)
	return fc
}

// sdpaVectorTQPipeline resolves (and caches) the single-pass TQ decode SDPA
// for (headDim, kBits, vBits).
func sdpaVectorTQPipeline(headDim, kBits, vBits int) (metal.MTLComputePipelineState, error) {
	if !tqKVGeometryOK(kBits, vBits, headDim) {
		return nil, core.NewError("native.sdpaVectorTQPipeline: unsupported (headDim, kBits, vBits)")
	}
	key := [3]int{headDim, kBits, vBits}
	tqKVPSOMu.Lock()
	defer tqKVPSOMu.Unlock()
	if pso, ok := sdpaVectorTQPSOs[key]; ok {
		if pso == nil {
			return nil, core.NewError("native.sdpaVectorTQPipeline: kernel unavailable")
		}
		return pso, nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		return nil, core.NewError("native.sdpaVectorTQPipeline: custom library unavailable")
	}
	name := core.Sprintf("lthn_sdpa_vector_tq_bf16_%d", headDim)
	fn, err := customLibrary.NewFunctionWithNameConstantValuesError(name, tqSDPAFunctionConstants(0, int32(kBits), int32(vBits), false))
	if err != nil || fn == nil || fn.GetID() == 0 {
		sdpaVectorTQPSOs[key] = nil
		return nil, core.E("native.sdpaVectorTQPipeline", name, err)
	}
	pso, perr := device.NewComputePipelineStateWithFunctionError(fn)
	if perr != nil {
		sdpaVectorTQPSOs[key] = nil
		return nil, perr
	}
	sdpaVectorTQPSOs[key] = pso
	return pso, nil
}

// sdpaVector2Pass1TQPipeline resolves the TQ pass-1 with the block count baked
// as function constant 26 (the MLX 2-pass convention) beside the bit widths.
// Pass 2 is MLX's sdpa_vector_2pass_2 unchanged (it never touches K/V).
func sdpaVector2Pass1TQPipeline(headDim, kBits, vBits int, blocks int32) (metal.MTLComputePipelineState, error) {
	if !tqKVGeometryOK(kBits, vBits, headDim) {
		return nil, core.NewError("native.sdpaVector2Pass1TQPipeline: unsupported (headDim, kBits, vBits)")
	}
	key := [4]int{headDim, kBits, vBits, int(blocks)}
	tqKVPSOMu.Lock()
	defer tqKVPSOMu.Unlock()
	if pso, ok := sdpaV2P1TQPSOCache[key]; ok {
		if pso == nil {
			return nil, core.NewError("native.sdpaVector2Pass1TQPipeline: kernel unavailable")
		}
		return pso, nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		return nil, core.NewError("native.sdpaVector2Pass1TQPipeline: custom library unavailable")
	}
	name := core.Sprintf("lthn_sdpa_vector_2pass_1_tq_bf16_%d", headDim)
	fn, err := customLibrary.NewFunctionWithNameConstantValuesError(name, tqSDPAFunctionConstants(blocks, int32(kBits), int32(vBits), true))
	if err != nil || fn == nil || fn.GetID() == 0 {
		sdpaV2P1TQPSOCache[key] = nil
		return nil, core.E("native.sdpaVector2Pass1TQPipeline", name, err)
	}
	pso, perr := device.NewComputePipelineStateWithFunctionError(fn)
	if perr != nil {
		sdpaV2P1TQPSOCache[key] = nil
		return nil, perr
	}
	sdpaV2P1TQPSOCache[key] = pso
	return pso, nil
}

// --- emitters (shared by the live encoder and the recorded ICB via dispatchSink) --------------

// tqKVStoreThreads is the store/rot kernels' fixed threadgroup width
// (LTHN_TQ_KV_CAP — sized to the widest instantiated head dim, 512).
const tqKVStoreThreads = 512

// emitTQKVStore records the staging→cache TurboQuant quantise hop through any
// sink: row(bf16 staging)=0, pi=1, centroids=2, codes=3 (rebound per token to
// the cache row), gammas=4 (rebound per token), d=5. One threadgroup per head.
func emitTQKVStore[S dispatchSink](sink S, pso metal.MTLComputePipelineState, row, pi, centroids metal.MTLBuffer, codes metal.MTLBuffer, codesOff uint, gammas metal.MTLBuffer, gammasOff uint, heads, headDim int) {
	sink.setPSO(pso)
	sink.setBuf(row, 0, 0)
	sink.setBuf(pi, 0, 1)
	sink.setBuf(centroids, 0, 2)
	sink.setBuf(codes, codesOff, 3)
	sink.setBuf(gammas, gammasOff, 4)
	sink.setI32(int32(headDim), 5)
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: uint(heads), Height: 1, Depth: 1},
		metal.MTLSize{Width: tqKVStoreThreads, Height: 1, Depth: 1},
	)
}

// emitTQRotRows records y = Π·x (transpose=false: the once-per-step q
// pre-rotation) or y = Πᵀ·x (transpose=true: the once-per-step output
// unrotation) over `rows` bf16 rows of dimension d — in=0, pi=1, out=2, d=3.
func emitTQRotRows[S dispatchSink](sink S, pso metal.MTLComputePipelineState, in, pi, out metal.MTLBuffer, rows, d int) {
	sink.setPSO(pso)
	sink.setBuf(in, 0, 0)
	sink.setBuf(pi, 0, 1)
	sink.setBuf(out, 0, 2)
	sink.setI32(int32(d), 3)
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: uint(rows), Height: 1, Depth: 1},
		metal.MTLSize{Width: tqKVStoreThreads, Height: 1, Depth: 1},
	)
}

// emitSDPAVectorTQ records the single-pass TQ decode SDPA through any sink.
// q must be PRE-ROTATED (emitTQRotRows) and out lands in ROTATED space (the
// unrotation op follows). Strides are BYTES of the code planes: head stride =
// ceil(headDim·bits/8), seq stride = kvHeads·headStride; the kernel derives
// the γ-plane strides from their ratio. ABI: q=0 k=1 v=2 out=3 gqa=4
// N=5 (nBuf when non-nil — the ICB's rebindable length), strides 6..9,
// scale=10, kGammas=11, vGammas=12, kCentroids=13, vCentroids=14. One
// threadgroup per q head, 32×32 threads — the MLX sdpa_vector dispatch.
func emitSDPAVectorTQ[S dispatchSink](sink S, pso metal.MTLComputePipelineState, q, k, v, out metal.MTLBuffer, kGammas, vGammas, kCentroids, vCentroids metal.MTLBuffer, nBuf metal.MTLBuffer, nHeads, nKVHeads, n int, kHeadStride, kSeqStride, vHeadStride, vSeqStride int64, scale float32) {
	sink.setPSO(pso)
	sink.setBuf(q, 0, 0)
	sink.setBuf(k, 0, 1)
	sink.setBuf(v, 0, 2)
	sink.setBuf(out, 0, 3)
	sink.setI32(int32(nHeads/nKVHeads), 4)
	if nBuf != nil {
		sink.setBuf(nBuf, 0, 5)
	} else {
		sink.setI32(int32(n), 5)
	}
	sink.setI64(kHeadStride, 6)
	sink.setI64(kSeqStride, 7)
	sink.setI64(vHeadStride, 8)
	sink.setI64(vSeqStride, 9)
	sink.setF32(scale, 10)
	sink.setBuf(kGammas, 0, 11)
	sink.setBuf(vGammas, 0, 12)
	sink.setBuf(kCentroids, 0, 13)
	sink.setBuf(vCentroids, 0, 14)
	sink.dispatchThreadgroups(metal.MTLSize{Width: uint(nHeads), Height: 1, Depth: 1}, metal.MTLSize{Width: 1024, Height: 1, Depth: 1})
}

// emitSDPAVector2Pass1TQ records the TQ pass-1 through any sink — the MLX
// 2-pass ABI with the γ/centroid planes appended (kCentroids at 6, the MLX
// ABI's free slot; N at 7 as a buffer or inline; strides 8..11; scale 12;
// kGammas 13; vGammas 14; vCentroids 15 — NEVER 16: the recorded arch ICB is
// built with maxKernelBufferBindCount 16, so an index-16 bind records as a
// silent no-op and the kernel reads garbage) and the decode grid
// (nKVHeads, 1, blocks) of (32, gqa, 1). Pass 2 is MLX's sdpa_vector_2pass_2
// unchanged; its merged output stays in rotated space and the recorded
// unrotation op lands it.
func emitSDPAVector2Pass1TQ[S dispatchSink](sink S, pso metal.MTLComputePipelineState, q, k, v, partials metal.MTLBuffer, sums, maxs metal.MTLBuffer, kGammas, vGammas, kCentroids, vCentroids metal.MTLBuffer, nBuf metal.MTLBuffer, nHeads, nKVHeads, n, blocks int, kHeadStride, kSeqStride, vHeadStride, vSeqStride int64, scale float32) {
	sink.setPSO(pso)
	sink.setBuf(q, 0, 0)
	sink.setBuf(k, 0, 1)
	sink.setBuf(v, 0, 2)
	sink.setBuf(partials, 0, 3)
	sink.setBuf(sums, 0, 4)
	sink.setBuf(maxs, 0, 5)
	sink.setBuf(kCentroids, 0, 6)
	if nBuf != nil {
		sink.setBuf(nBuf, 0, 7)
	} else {
		sink.setI32(int32(n), 7)
	}
	sink.setI64(kHeadStride, 8)
	sink.setI64(kSeqStride, 9)
	sink.setI64(vHeadStride, 10)
	sink.setI64(vSeqStride, 11)
	sink.setF32(scale, 12)
	sink.setBuf(kGammas, 0, 13)
	sink.setBuf(vGammas, 0, 14)
	sink.setBuf(vCentroids, 0, 15)
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: uint(nKVHeads), Height: 1, Depth: uint(blocks)},
		metal.MTLSize{Width: 32, Height: uint(nHeads / nKVHeads), Depth: 1},
	)
}

// --- host round-trip drivers (the kernel-gate tests + pre-integration probes) -----------------

// TurboQuantKVStoreDevice quantises `heads` contiguous bf16 rows (headDim
// elements each — ONE cache row's staging) through lthn_tq_kv_store_bf16_bN,
// returning the packed codes ([heads × ceil(headDim·bits/8)] bytes) and γ
// ([heads] f32). Π/centroids are the lane's own residency (tqKVSeed) — the
// exact values the live session uses.
func TurboQuantKVStoreDevice(rows []byte, heads, headDim, bits int) ([]byte, []float32, error) {
	if err := ensureInit(); err != nil {
		return nil, nil, err
	}
	if !tqKVBitsOK(bits) || headDim <= 0 || headDim > tqKVStoreThreads || heads <= 0 {
		return nil, nil, core.NewError("native.TurboQuantKVStoreDevice: unsupported geometry")
	}
	if len(rows) != heads*headDim*bf16Size {
		return nil, nil, core.NewError("native.TurboQuantKVStoreDevice: staging size mismatch")
	}
	pso, err := tqKVStorePipeline(bits)
	if err != nil {
		return nil, nil, err
	}
	bytesPerHead := tqBytesPerRow(bits, headDim)
	codes := make([]byte, heads*bytesPerHead)
	gammas := make([]float32, heads)
	var encErr error
	withAutoreleasePool(func() {
		stage, cerr := newPinnedNoCopyBytes(len(rows))
		if cerr != nil {
			encErr = cerr
			return
		}
		rowBuf, uerr := stage.copyBuffer(rows)
		if uerr != nil {
			encErr = uerr
			return
		}
		codesBuf := device.NewBufferWithLengthOptions(uint(len(codes)), metal.MTLResourceStorageModeShared)
		gammasBuf := device.NewBufferWithLengthOptions(uint(heads*4), metal.MTLResourceStorageModeShared)
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		emitTQKVStore(encSink{enc}, pso, rowBuf, tqKVPiBuffer(headDim), tqKVCentroidsBuffer(headDim, bits), codesBuf, 0, gammasBuf, 0, heads, headDim)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		copy(codes, unsafe.Slice((*byte)(codesBuf.Contents()), len(codes)))
		copy(gammas, unsafe.Slice((*float32)(gammasBuf.Contents()), heads))
		releaseDeviceBuffers(codesBuf, gammasBuf)
	})
	if encErr != nil {
		return nil, nil, encErr
	}
	return codes, gammas, nil
}

// TurboQuantSDPADevice runs the full TQ decode read chain over host slices —
// q pre-rotation, the single-pass OR 2-pass code-reading SDPA, and the output
// unrotation, exactly the op sequence the recorded ICB replays per token.
// q is [nHeads×headDim] bf16; kCodes/vCodes are [n × nKVHeads × bytesPerHead]
// packed rows; kGammas/vGammas are [n × nKVHeads] f32. twoPass selects the
// 2-pass pair (pass 2 = MLX's sdpa_vector_2pass_2, unchanged).
func TurboQuantSDPADevice(q []byte, kCodes, vCodes []byte, kGammas, vGammas []float32, nHeads, nKVHeads, headDim, n, kBits, vBits int, scale float32, twoPass bool) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if !tqKVGeometryOK(kBits, vBits, headDim) || nHeads <= 0 || nKVHeads <= 0 || nHeads%nKVHeads != 0 || n <= 0 {
		return nil, core.NewError("native.TurboQuantSDPADevice: unsupported geometry")
	}
	kBytesPerHead := tqBytesPerRow(kBits, headDim)
	vBytesPerHead := tqBytesPerRow(vBits, headDim)
	if len(q) != nHeads*headDim*bf16Size || len(kCodes) != n*nKVHeads*kBytesPerHead ||
		len(vCodes) != n*nKVHeads*vBytesPerHead || len(kGammas) != n*nKVHeads || len(vGammas) != n*nKVHeads {
		return nil, core.NewError("native.TurboQuantSDPADevice: size mismatch")
	}
	rotPSO, err := tqRotRowsPipeline(false)
	if err != nil {
		return nil, err
	}
	unrotPSO, err := tqRotRowsPipeline(true)
	if err != nil {
		return nil, err
	}
	out := make([]byte, nHeads*headDim*bf16Size)
	var encErr error
	withAutoreleasePool(func() {
		upload := func(b []byte) metal.MTLBuffer {
			if encErr != nil || len(b) == 0 {
				return nil
			}
			buf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&b[0]), uint(len(b)), metal.MTLResourceStorageModeShared)
			if buf == nil {
				encErr = core.NewError("native.TurboQuantSDPADevice: buffer allocation failed")
			}
			return buf
		}
		qBuf := upload(q)
		kBuf := upload(kCodes)
		vBuf := upload(vCodes)
		kgBuf := residentFloat32(kGammas)
		vgBuf := residentFloat32(vGammas)
		if encErr != nil {
			return
		}
		qRot := device.NewBufferWithLengthOptions(uint(len(q)), metal.MTLResourceStorageModeShared)
		oRot := device.NewBufferWithLengthOptions(uint(len(out)), metal.MTLResourceStorageModeShared)
		outBuf := device.NewBufferWithLengthOptions(uint(len(out)), metal.MTLResourceStorageModeShared)
		pi := tqKVPiBuffer(headDim)
		kCent := tqKVCentroidsBuffer(headDim, kBits)
		vCent := tqKVCentroidsBuffer(headDim, vBits)

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		sink := encSink{enc}
		emitTQRotRows(sink, rotPSO, qBuf, pi, qRot, nHeads, headDim)
		if twoPass {
			blocks := sdpa2PassBlocks(n, nKVHeads)
			pso1, perr := sdpaVector2Pass1TQPipeline(headDim, kBits, vBits, blocks)
			if perr != nil {
				encErr = perr
			}
			pso2, p2err := sdpaVector2Pass2PipelineForHeadDim(headDim)
			if p2err != nil {
				encErr = p2err
			}
			if encErr == nil {
				partials := device.NewBufferWithLengthOptions(uint(nHeads*int(blocks)*headDim*4), metal.MTLResourceStorageModeShared)
				sums := device.NewBufferWithLengthOptions(uint(nHeads*int(blocks)*4), metal.MTLResourceStorageModeShared)
				maxs := device.NewBufferWithLengthOptions(uint(nHeads*int(blocks)*4), metal.MTLResourceStorageModeShared)
				emitSDPAVector2Pass1TQ(sink, pso1, qRot, kBuf, vBuf, partials, sums, maxs, kgBuf, vgBuf, kCent, vCent, nil,
					nHeads, nKVHeads, n, int(blocks), int64(kBytesPerHead), int64(nKVHeads*kBytesPerHead), int64(vBytesPerHead), int64(nKVHeads*vBytesPerHead), scale)
				emitSDPA2Pass2At(sink, pso2, partials, sums, maxs, oRot, 0, 1, nHeads, int(blocks))
			}
		} else {
			pso, serr := sdpaVectorTQPipeline(headDim, kBits, vBits)
			if serr != nil {
				encErr = serr
			} else {
				emitSDPAVectorTQ(sink, pso, qRot, kBuf, vBuf, oRot, kgBuf, vgBuf, kCent, vCent, nil,
					nHeads, nKVHeads, n, int64(kBytesPerHead), int64(nKVHeads*kBytesPerHead), int64(vBytesPerHead), int64(nKVHeads*vBytesPerHead), scale)
			}
		}
		if encErr == nil {
			emitTQRotRows(sink, unrotPSO, oRot, pi, outBuf, nHeads, headDim)
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if encErr == nil {
			copy(out, unsafe.Slice((*byte)(outBuf.Contents()), len(out)))
		}
	})
	if encErr != nil {
		return nil, encErr
	}
	return out, nil
}
