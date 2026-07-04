// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// lthn_kernels.go is the native engine's own custom-kernel mechanism: kernels MLX's static metallib
// does not have, compiled from kernels/*.metal into a sibling lthn_kernels.metallib that device.go
// loads beside MLX's (customLibrary). The first such kernel is the fused gelu (kernels/
// lthn_gelu_gate_mul.metal). This is the foundation for any fused/novel op the native wants — fused
// activations, the "compute fp32, store bf16" path, future LEK/MTP kernels — independent of whether
// any one of them is wired into the serve decode.

// gpuHasGeluKernel reports whether the fused gelu kernel is available (the custom kernels metallib
// loaded). The composed bf16 chain is the production path; this is the fused capability beside it.
func gpuHasGeluKernel() bool { return customLibraryLoaded }

var (
	geluPSOOnce sync.Once
	geluPSO     metal.MTLComputePipelineState
	geluPSOErr  error

	ffnMegaPSOOnce sync.Once
	ffnMegaPSO     metal.MTLComputePipelineState
	ffnMegaPSOErr  error
)

// geluPipeline builds (once) the fused gelu pipeline from the custom kernels library.
func geluPipeline() (metal.MTLComputePipelineState, error) {
	geluPSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			geluPSOErr = core.NewError("native.geluPipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_gelu_gate_mul_bf16")
		if fn == nil || fn.GetID() == 0 {
			geluPSOErr = core.NewError("native.geluPipeline: kernel lthn_gelu_gate_mul_bf16 not found")
			return
		}
		geluPSO, geluPSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return geluPSO, geluPSOErr
}

const (
	ffnMegaNumThreadgroups   = 64
	ffnMegaThreadsPerGroup   = 128
	ffnMegaMaxSpinIterations = 1_000_000
)

func ffnMegaPipeline() (metal.MTLComputePipelineState, error) {
	ffnMegaPSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			ffnMegaPSOErr = core.NewError("native.ffnMegaPipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_ffn_megakernel")
		if fn == nil || fn.GetID() == 0 {
			ffnMegaPSOErr = core.NewError("native.ffnMegaPipeline: kernel lthn_ffn_megakernel not found")
			return
		}
		ffnMegaPSO, ffnMegaPSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return ffnMegaPSO, ffnMegaPSOErr
}

var (
	mulRowsPSOOnce sync.Once
	mulRowsPSO     metal.MTLComputePipelineState
	mulRowsPSOErr  error
)

func mulRowsPipeline() (metal.MTLComputePipelineState, error) {
	mulRowsPSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			mulRowsPSOErr = core.NewError("native.mulRowsPipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_mul_rows_bf16")
		if fn == nil || fn.GetID() == 0 {
			mulRowsPSOErr = core.NewError("native.mulRowsPipeline: kernel lthn_mul_rows_bf16 not found")
			return
		}
		mulRowsPSO, mulRowsPSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return mulRowsPSO, mulRowsPSOErr
}

// gpuHasMulRowsKernel reports whether the broadcast rows-multiply kernel is loadable — the batched
// epilogue's gate for folding the K per-row layer-scalar dispatches into one.
func gpuHasMulRowsKernel() bool {
	pso, err := mulRowsPipeline()
	return err == nil && pso != nil && pso.GetID() != 0
}

// encMulRowsBF16 encodes out row r = a row r · b — ONE b row of rowLen broadcast across `rows`
// contiguous a rows — in one dispatch: the batched pass's per-layer output scalar applied to all
// K rows at once. Per-element float math identical to `rows` per-row vv_mul dispatches.
func encMulRowsBF16(enc metal.MTLComputeCommandEncoder, a, b, out metal.MTLBuffer, aOff, bOff, outOff uint, rows, rowLen int) error {
	pso, err := mulRowsPipeline()
	if err != nil {
		return err
	}
	n := rows * rowLen
	sink := encSink{enc}
	sink.setPSO(pso)
	sink.setBuf(a, aOff, 0)
	sink.setBuf(b, bOff, 1)
	sink.setBuf(out, outOff, 2)
	sink.setI32(int32(n), 3)
	sink.setI32(int32(rowLen), 4)
	sink.dispatchThreads(
		metal.MTLSize{Width: uint(n), Height: 1, Depth: 1},
		metal.MTLSize{Width: uint(elemGroupTG(n)), Height: 1, Depth: 1},
	)
	return nil
}

// encGeluGateMulFused encodes gelu(gate)·up via the fused kernel — one dispatch, fp32-internal, one
// bf16 rounding (see the kernel comment for why this differs from the composed production path).
// gate/up/out are contiguous bf16 buffers of n elements. Guard with gpuHasGeluKernel before calling.
func encGeluGateMulFused(enc metal.MTLComputeCommandEncoder, gate, up, out metal.MTLBuffer, n int) error {
	return encGeluGateMulFusedTo(enc, gate, up, out, 0, 0, 0, n)
}

func encGeluGateMulFusedTo(enc metal.MTLComputeCommandEncoder, gate, up, out metal.MTLBuffer, gateOff, upOff, outOff uint, n int) error {
	pso, err := geluPipeline()
	if err != nil {
		return err
	}
	// the fused gelu(gate)·up shares the binary-op ABI (in0=0, in1=1, out=2, count=3) — one shared
	// emitBinary body with vv_Add/vv_Multiply and the ICB recorder's gelu op, just a different pipeline.
	emitBinary(encSink{enc}, pso, gate, gateOff, up, upOff, out, outOff, n)
	return nil
}

// geluGateMulFused is the one-shot host wrapper around the fused kernel — gate/up bf16 bytes in,
// bf16 bytes out. The diagnostic + bench exercise it; the decode stays on the composed chain.
func geluGateMulFused(gate, up []byte, n int) ([]byte, error) {
	out := make([]byte, n*bf16Size)
	if err := geluGateMulFusedInto(out, gate, up, n, false); err != nil {
		return nil, err
	}
	return out, nil
}

func geluGateMulFusedInto(out, gate, up []byte, n int, directOutput bool) error {
	var encErr error
	withAutoreleasePool(func() {
		ioScratch, err := getBinaryByteScratch(n * bf16Size)
		if err != nil {
			encErr = err
			return
		}
		defer putBinaryByteScratch(ioScratch)
		gBuf, uBuf, oBuf, err := ioScratch.buffers(gate, up)
		if err != nil {
			encErr = err
			return
		}
		directOut := false
		if directOutput {
			tmp, ok := ioScratch.outputView(out)
			if ok {
				oBuf = tmp
				directOut = true
			}
		}
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		if encErr = encGeluGateMulFused(enc, gBuf, uBuf, oBuf, n); encErr != nil {
			endEncodingFast(enc)
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, ioScratch.out.bytes[:n*bf16Size])
		}
	})
	return encErr
}

var (
	bf16MulScalarPSOOnce sync.Once
	bf16MulScalarPSO     metal.MTLComputePipelineState
	bf16MulScalarPSOErr  error
)

func bf16MulScalarPipeline() (metal.MTLComputePipelineState, error) {
	bf16MulScalarPSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			bf16MulScalarPSOErr = core.NewError("native.bf16MulScalarPipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_bf16_mul_scalar")
		if fn == nil || fn.GetID() == 0 {
			bf16MulScalarPSOErr = core.NewError("native.bf16MulScalarPipeline: kernel lthn_bf16_mul_scalar not found")
			return
		}
		bf16MulScalarPSO, bf16MulScalarPSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return bf16MulScalarPSO, bf16MulScalarPSOErr
}

func encMulScalarBF16(enc metal.MTLComputeCommandEncoder, in, scalar, out metal.MTLBuffer, scalarOffset uint, n int) error {
	if n < 0 {
		return core.NewError("native.encMulScalarBF16: n must be >= 0")
	}
	if n == 0 {
		return nil
	}
	pso, err := bf16MulScalarPipeline()
	if err != nil {
		return err
	}
	sink := encSink{enc}
	sink.setPSO(pso)
	sink.setBuf(in, 0, 0)
	sink.setBuf(scalar, scalarOffset, 1)
	sink.setBuf(out, 0, 2)
	sink.setI32(int32(n), 3)
	group := uint(256)
	if uint(n) < group {
		group = uint(n)
	}
	sink.dispatchThreads(
		metal.MTLSize{Width: uint(n), Height: 1, Depth: 1},
		metal.MTLSize{Width: group, Height: 1, Depth: 1},
	)
	return nil
}

func encMulScalarBF16Object(enc metal.MTLComputeCommandEncoderObject, in, scalar, out metal.MTLBuffer, scalarOffset uint, n int) error {
	if n < 0 {
		return core.NewError("native.encMulScalarBF16: n must be >= 0")
	}
	if n == 0 {
		return nil
	}
	pso, err := bf16MulScalarPipeline()
	if err != nil {
		return err
	}
	sink := encObjectSink{enc: enc}
	sink.setPSO(pso)
	sink.setBuf(in, 0, 0)
	sink.setBuf(scalar, scalarOffset, 1)
	sink.setBuf(out, 0, 2)
	sink.setI32(int32(n), 3)
	group := uint(256)
	if uint(n) < group {
		group = uint(n)
	}
	sink.dispatchThreads(
		metal.MTLSize{Width: uint(n), Height: 1, Depth: 1},
		metal.MTLSize{Width: group, Height: 1, Depth: 1},
	)
	return nil
}

func bf16ScalarBytes(v float32) [bf16Size]byte {
	h := f32ToBF16(v)
	return [bf16Size]byte{byte(h), byte(h >> 8)}
}

func encScaleBF16(enc metal.MTLComputeCommandEncoder, in, scalar, out metal.MTLBuffer, scalarOffset uint, scalarBytes []byte, n int) error {
	if err := encMulScalarBF16(enc, in, scalar, out, scalarOffset, n); err == nil {
		return nil
	}
	return encMulBF16(enc, in, sharedBytes(scalarFillBF16(scalarBytes, n)), out, n)
}

func encScaleBF16Object(enc metal.MTLComputeCommandEncoderObject, in, scalar, out metal.MTLBuffer, scalarOffset uint, scalarBytes []byte, n int) error {
	if err := encMulScalarBF16Object(enc, in, scalar, out, scalarOffset, n); err == nil {
		return nil
	}
	return encMulBF16(enc, in, sharedBytes(scalarFillBF16(scalarBytes, n)), out, n)
}

// MulScalarBF16 multiplies each bf16 element in in by one bf16 scalar. When the
// native custom kernels are available it binds the scalar directly, avoiding the
// dense broadcast vector that pkg/metal's scalar bridge also avoids. Without the
// sibling custom metallib it falls back to the existing dense-vector multiply so
// the public operation still works.
func MulScalarBF16(in, scalar []byte) ([]byte, error) {
	out := make([]byte, len(in))
	if err := mulScalarBF16Into(out, in, scalar, false); err != nil {
		return nil, err
	}
	return out, nil
}

func MulScalarBF16Into(out, in, scalar []byte) error {
	return mulScalarBF16Into(out, in, scalar, true)
}

func mulScalarBF16Into(out, in, scalar []byte, directOutput bool) error {
	if err := ensureInit(); err != nil {
		return err
	}
	if len(in)%bf16Size != 0 {
		return core.NewError("native.MulScalarBF16Into: input byte length must be a multiple of 2")
	}
	if len(scalar) != bf16Size {
		return core.NewError("native.MulScalarBF16Into: scalar must be one bf16 value")
	}
	if len(out) != len(in) {
		return core.NewError("native.MulScalarBF16Into: out must be the same byte length as in")
	}
	n := len(in) / bf16Size
	if n == 0 {
		return nil
	}
	var encErr error
	var setupErr error
	withAutoreleasePool(func() {
		scratch, err := getQMVBF16Scratch(n, n)
		if err != nil {
			setupErr = err
			return
		}
		defer putQMVBF16Scratch(scratch)
		inBuf, outBuf, err := scratch.buffers(in)
		if err != nil {
			setupErr = err
			return
		}
		directOut := false
		if directOutput {
			tmp, ok := scratch.outputView(out)
			if ok {
				outBuf = tmp
				directOut = true
			}
		}
		scalarBuf := bf16ConstBuffer(1, bf16ToF32(scalar[0], scalar[1]))
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		if encErr = encMulScalarBF16(enc, inBuf, scalarBuf, outBuf, 0, n); encErr != nil {
			endEncodingFast(enc)
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, scratch.out.bytes[:len(out)])
		}
	})
	if setupErr != nil {
		return setupErr
	}
	if encErr == nil {
		return nil
	}
	fallback, err := MulBF16(in, scalarFillBF16(scalar, n))
	if err != nil {
		return err
	}
	copy(out, fallback)
	return nil
}

const routerTopKMaxK = 32

var (
	routerTopKPSOOnce sync.Once
	routerTopKPSO     metal.MTLComputePipelineState
	routerTopKPSOErr  error
)

func routerTopKPipeline() (metal.MTLComputePipelineState, error) {
	routerTopKPSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			routerTopKPSOErr = core.NewError("native.routerTopKPipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_moe_router_topk_bf16")
		if fn == nil || fn.GetID() == 0 {
			routerTopKPSOErr = core.NewError("native.routerTopKPipeline: kernel lthn_moe_router_topk_bf16 not found")
			return
		}
		routerTopKPSO, routerTopKPSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return routerTopKPSO, routerTopKPSOErr
}

type routerTopKScratch struct {
	numExperts, topK      int
	scoresView, scaleView cachedNoCopyBytesView
	idxBuf, weightBuf     metal.MTLBuffer
}

type routerTopKScratchKey struct {
	numExperts, topK int
}

var routerTopKScratchPools sync.Map

func routerTopKScratchPoolFor(numExperts, topK int) *scratchLIFOPool[*routerTopKScratch] {
	key := routerTopKScratchKey{numExperts: numExperts, topK: topK}
	if v, ok := routerTopKScratchPools.Load(key); ok {
		return v.(*scratchLIFOPool[*routerTopKScratch])
	}
	p := &scratchLIFOPool[*routerTopKScratch]{}
	actual, _ := routerTopKScratchPools.LoadOrStore(key, p)
	return actual.(*scratchLIFOPool[*routerTopKScratch])
}

func newRouterTopKScratch(numExperts, topK int) *routerTopKScratch {
	return &routerTopKScratch{
		numExperts: numExperts,
		topK:       topK,
		idxBuf:     device.NewBufferWithLengthOptions(uint(topK*4), metal.MTLResourceStorageModeShared),
		weightBuf:  scratchBF16(topK),
	}
}

func getRouterTopKScratch(numExperts, topK int) *routerTopKScratch {
	p := routerTopKScratchPoolFor(numExperts, topK)
	if s := p.Get(); s != nil && s.numExperts == numExperts && s.topK == topK && s.idxBuf != nil && s.weightBuf != nil {
		return s
	}
	return newRouterTopKScratch(numExperts, topK)
}

func putRouterTopKScratch(s *routerTopKScratch) {
	if s != nil && s.numExperts > 0 && s.topK > 0 && s.idxBuf != nil && s.weightBuf != nil {
		routerTopKScratchPoolFor(s.numExperts, s.topK).Put(s)
	}
}

func encRouterTopKBF16(enc metal.MTLComputeCommandEncoder, scores, perExpertScale, topIndices, topWeights metal.MTLBuffer, scaleOff uint, numExperts, topK int, hasScale bool) error {
	if topK <= 0 || topK > numExperts || topK > routerTopKMaxK {
		return core.NewError("native.encRouterTopKBF16: topK must be in 1..numExperts and <= 32")
	}
	pso, err := routerTopKPipeline()
	if err != nil {
		return err
	}
	if perExpertScale == nil {
		perExpertScale = scores
	}
	scaleFlag := int32(0)
	if hasScale {
		scaleFlag = 1
	}
	sink := encSink{enc}
	sink.setPSO(pso)
	sink.setBuf(scores, 0, 0)
	sink.setBuf(perExpertScale, scaleOff, 1)
	sink.setBuf(topIndices, 0, 2)
	sink.setBuf(topWeights, 0, 3)
	sink.setI32(int32(numExperts), 4)
	sink.setI32(int32(topK), 5)
	sink.setI32(scaleFlag, 6)
	sink.dispatchThreads(
		metal.MTLSize{Width: 256, Height: 1, Depth: 1},
		metal.MTLSize{Width: 256, Height: 1, Depth: 1},
	)
	return nil
}

func routerTopKBF16(scoresB, perExpertScale []byte, numExperts, topK int) ([]int32, []byte, error) {
	if err := ensureInit(); err != nil {
		return nil, nil, err
	}
	if len(scoresB) != numExperts*bf16Size {
		return nil, nil, core.NewError("native.routerTopKBF16: scores must be numExperts bf16 bytes")
	}
	if perExpertScale != nil && len(perExpertScale) != numExperts*bf16Size {
		return nil, nil, core.NewError("native.routerTopKBF16: perExpertScale must be numExperts bf16 bytes or nil")
	}
	if topK <= 0 || topK > numExperts || topK > routerTopKMaxK {
		return nil, nil, core.NewError("native.routerTopKBF16: topK must be in 1..numExperts and <= 32")
	}
	idx := make([]int32, topK)
	weights := make([]byte, topK*bf16Size)
	var encErr error
	withAutoreleasePool(func() {
		scratch := getRouterTopKScratch(numExperts, topK)
		defer putRouterTopKScratch(scratch)
		scoresBuf, ok := scratch.scoresView.bufferAfterStable(scoresB, 2)
		if !ok {
			scoresBuf = sharedBytes(scoresB)
		}
		scaleBuf := metal.MTLBuffer(nil)
		if perExpertScale != nil {
			var ok bool
			scaleBuf, ok = scratch.scaleView.bufferAfterStable(perExpertScale, 2)
			if !ok {
				scaleBuf = sharedBytes(perExpertScale)
			}
		}
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		if encErr = encRouterTopKBF16(enc, scoresBuf, scaleBuf, scratch.idxBuf, scratch.weightBuf, 0, numExperts, topK, perExpertScale != nil); encErr != nil {
			endEncodingFast(enc)
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		copy(idx, unsafe.Slice((*int32)(scratch.idxBuf.Contents()), topK))
		copy(weights, unsafe.Slice((*byte)(scratch.weightBuf.Contents()), topK*bf16Size))
	})
	if encErr != nil {
		return nil, nil, encErr
	}
	return idx, weights, nil
}

const bf16LMHeadArgmaxRowsPerTile = 8
const bf16LogitsArgmaxRowsPerTile = 256
const (
	headSampleTopKMaxK         = 64
	q4LMHeadTopKBlockSize      = 512
	q4LMHeadTopKSimdgroups     = 4
	q4LMHeadTopKSubtiles       = 8
	q4LMHeadTopKResultsPerSIMD = 4
	q4LMHeadTopKRowsPerTile    = q4LMHeadTopKSimdgroups * q4LMHeadTopKSubtiles * q4LMHeadTopKResultsPerSIMD
	q4LMHeadTopKPackedPerInt32 = 8
)

var (
	bf16LMHeadArgmaxTilesPSOOnce sync.Once
	bf16LMHeadArgmaxTilesPSO     metal.MTLComputePipelineState
	bf16LMHeadArgmaxTilesPSOErr  error
	bf16LogitsArgmaxTilesPSOOnce sync.Once
	bf16LogitsArgmaxTilesPSO     metal.MTLComputePipelineState
	bf16LogitsArgmaxTilesPSOErr  error
	argmaxMergeF32PSOOnce        sync.Once
	argmaxMergeF32PSO            metal.MTLComputePipelineState
	argmaxMergeF32PSOErr         error
	bf16LMHeadCandidatesPSOOnce  sync.Once
	bf16LMHeadCandidatesPSO      metal.MTLComputePipelineState
	bf16LMHeadCandidatesPSOErr   error
	bf16LogitsCandidatesPSOOnce  sync.Once
	bf16LogitsCandidatesPSO      metal.MTLComputePipelineState
	bf16LogitsCandidatesPSOErr   error
	bf16LogitsTopKTilesPSOOnce   sync.Once
	bf16LogitsTopKTilesPSO       metal.MTLComputePipelineState
	bf16LogitsTopKTilesPSOErr    error
	q4LMHeadTopKTilesPSOOnce     sync.Once
	q4LMHeadTopKTilesPSO         metal.MTLComputePipelineState
	q4LMHeadTopKTilesPSOErr      error
	topKMergeF32PSOOnce          sync.Once
	topKMergeF32PSO              metal.MTLComputePipelineState
	topKMergeF32PSOErr           error
	topKMergeSampleF32PSOOnce    sync.Once
	topKMergeSampleF32PSO        metal.MTLComputePipelineState
	topKMergeSampleF32PSOErr     error
	logitsSampleBF16PSOOnce      sync.Once
	logitsSampleBF16PSO          metal.MTLComputePipelineState
	logitsSampleBF16PSOErr       error
)

func bf16LMHeadArgmaxTilesPipeline() (metal.MTLComputePipelineState, error) {
	bf16LMHeadArgmaxTilesPSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			bf16LMHeadArgmaxTilesPSOErr = core.NewError("native.bf16LMHeadArgmaxTilesPipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_bf16_lm_head_argmax_tiles_bf16")
		if fn == nil || fn.GetID() == 0 {
			bf16LMHeadArgmaxTilesPSOErr = core.NewError("native.bf16LMHeadArgmaxTilesPipeline: kernel lthn_bf16_lm_head_argmax_tiles_bf16 not found")
			return
		}
		bf16LMHeadArgmaxTilesPSO, bf16LMHeadArgmaxTilesPSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return bf16LMHeadArgmaxTilesPSO, bf16LMHeadArgmaxTilesPSOErr
}

func bf16LogitsArgmaxTilesPipeline() (metal.MTLComputePipelineState, error) {
	bf16LogitsArgmaxTilesPSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			bf16LogitsArgmaxTilesPSOErr = core.NewError("native.bf16LogitsArgmaxTilesPipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_bf16_logits_argmax_tiles_bf16")
		if fn == nil || fn.GetID() == 0 {
			bf16LogitsArgmaxTilesPSOErr = core.NewError("native.bf16LogitsArgmaxTilesPipeline: kernel lthn_bf16_logits_argmax_tiles_bf16 not found")
			return
		}
		bf16LogitsArgmaxTilesPSO, bf16LogitsArgmaxTilesPSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return bf16LogitsArgmaxTilesPSO, bf16LogitsArgmaxTilesPSOErr
}

func argmaxMergeF32Pipeline() (metal.MTLComputePipelineState, error) {
	argmaxMergeF32PSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			argmaxMergeF32PSOErr = core.NewError("native.argmaxMergeF32Pipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_argmax_merge_f32")
		if fn == nil || fn.GetID() == 0 {
			argmaxMergeF32PSOErr = core.NewError("native.argmaxMergeF32Pipeline: kernel lthn_argmax_merge_f32 not found")
			return
		}
		argmaxMergeF32PSO, argmaxMergeF32PSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return argmaxMergeF32PSO, argmaxMergeF32PSOErr
}

func bf16LMHeadCandidatesPipeline() (metal.MTLComputePipelineState, error) {
	bf16LMHeadCandidatesPSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			bf16LMHeadCandidatesPSOErr = core.NewError("native.bf16LMHeadCandidatesPipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_bf16_lm_head_candidates_bf16")
		if fn == nil || fn.GetID() == 0 {
			bf16LMHeadCandidatesPSOErr = core.NewError("native.bf16LMHeadCandidatesPipeline: kernel lthn_bf16_lm_head_candidates_bf16 not found")
			return
		}
		bf16LMHeadCandidatesPSO, bf16LMHeadCandidatesPSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return bf16LMHeadCandidatesPSO, bf16LMHeadCandidatesPSOErr
}

func bf16LogitsCandidatesPipeline() (metal.MTLComputePipelineState, error) {
	bf16LogitsCandidatesPSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			bf16LogitsCandidatesPSOErr = core.NewError("native.bf16LogitsCandidatesPipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_bf16_logits_candidates_bf16")
		if fn == nil || fn.GetID() == 0 {
			bf16LogitsCandidatesPSOErr = core.NewError("native.bf16LogitsCandidatesPipeline: kernel lthn_bf16_logits_candidates_bf16 not found")
			return
		}
		bf16LogitsCandidatesPSO, bf16LogitsCandidatesPSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return bf16LogitsCandidatesPSO, bf16LogitsCandidatesPSOErr
}

func bf16LogitsTopKTilesPipeline() (metal.MTLComputePipelineState, error) {
	bf16LogitsTopKTilesPSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			bf16LogitsTopKTilesPSOErr = core.NewError("native.bf16LogitsTopKTilesPipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_bf16_logits_topk_tiles_bf16")
		if fn == nil || fn.GetID() == 0 {
			bf16LogitsTopKTilesPSOErr = core.NewError("native.bf16LogitsTopKTilesPipeline: kernel lthn_bf16_logits_topk_tiles_bf16 not found")
			return
		}
		bf16LogitsTopKTilesPSO, bf16LogitsTopKTilesPSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return bf16LogitsTopKTilesPSO, bf16LogitsTopKTilesPSOErr
}

func q4LMHeadTopKTilesPipeline() (metal.MTLComputePipelineState, error) {
	q4LMHeadTopKTilesPSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			q4LMHeadTopKTilesPSOErr = core.NewError("native.q4LMHeadTopKTilesPipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_q4_lm_head_topk_tiles_bf16")
		if fn == nil || fn.GetID() == 0 {
			q4LMHeadTopKTilesPSOErr = core.NewError("native.q4LMHeadTopKTilesPipeline: kernel lthn_q4_lm_head_topk_tiles_bf16 not found")
			return
		}
		q4LMHeadTopKTilesPSO, q4LMHeadTopKTilesPSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return q4LMHeadTopKTilesPSO, q4LMHeadTopKTilesPSOErr
}

func topKMergeF32Pipeline() (metal.MTLComputePipelineState, error) {
	topKMergeF32PSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			topKMergeF32PSOErr = core.NewError("native.topKMergeF32Pipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_topk_merge_f32")
		if fn == nil || fn.GetID() == 0 {
			topKMergeF32PSOErr = core.NewError("native.topKMergeF32Pipeline: kernel lthn_topk_merge_f32 not found")
			return
		}
		topKMergeF32PSO, topKMergeF32PSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return topKMergeF32PSO, topKMergeF32PSOErr
}

func topKMergeSampleF32Pipeline() (metal.MTLComputePipelineState, error) {
	topKMergeSampleF32PSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			topKMergeSampleF32PSOErr = core.NewError("native.topKMergeSampleF32Pipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_topk_merge_sample_f32")
		if fn == nil || fn.GetID() == 0 {
			topKMergeSampleF32PSOErr = core.NewError("native.topKMergeSampleF32Pipeline: kernel lthn_topk_merge_sample_f32 not found")
			return
		}
		topKMergeSampleF32PSO, topKMergeSampleF32PSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return topKMergeSampleF32PSO, topKMergeSampleF32PSOErr
}

func logitsSampleBF16Pipeline() (metal.MTLComputePipelineState, error) {
	logitsSampleBF16PSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			logitsSampleBF16PSOErr = core.NewError("native.logitsSampleBF16Pipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_logits_sample_bf16")
		if fn == nil || fn.GetID() == 0 {
			logitsSampleBF16PSOErr = core.NewError("native.logitsSampleBF16Pipeline: kernel lthn_logits_sample_bf16 not found")
			return
		}
		logitsSampleBF16PSO, logitsSampleBF16PSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return logitsSampleBF16PSO, logitsSampleBF16PSOErr
}

func bf16LMHeadArgmaxUsable(dModel, vocab int) bool {
	if dModel <= 0 || vocab <= 0 {
		return false
	}
	if _, err := bf16LMHeadArgmaxTilesPipeline(); err != nil {
		return false
	}
	if _, err := argmaxMergeF32Pipeline(); err != nil {
		return false
	}
	return true
}

func qmvLogitsArgmaxUsable(dModel, vocab, groupSize, bits int) bool {
	if dModel <= 0 || vocab <= 0 || bits != 4 {
		return false
	}
	if groupSize != 32 && groupSize != 64 && groupSize != 128 {
		return false
	}
	if dModel%groupSize != 0 {
		return false
	}
	if _, err := pipelineFor(qmvBF16KernelName(vocab, dModel, groupSize, bits)); err != nil {
		return false
	}
	if _, err := bf16LogitsArgmaxTilesPipeline(); err != nil {
		return false
	}
	if _, err := argmaxMergeF32Pipeline(); err != nil {
		return false
	}
	return true
}

func bf16LMHeadTopKUsable(dModel, vocab, topK int) bool {
	if dModel <= 0 || vocab <= 0 || topK <= 0 || topK > headSampleTopKMaxK || topK > vocab {
		return false
	}
	if _, err := bf16LMHeadCandidatesPipeline(); err != nil {
		return false
	}
	if _, err := topKMergeF32Pipeline(); err != nil {
		return false
	}
	return true
}

func qmvLogitsTopKUsable(dModel, vocab, groupSize, bits, topK int) bool {
	if dModel <= 0 || vocab <= 0 || bits != 4 || topK <= 0 || topK > headSampleTopKMaxK || topK > vocab {
		return false
	}
	if groupSize != 32 && groupSize != 64 && groupSize != 128 {
		return false
	}
	if dModel%groupSize != 0 {
		return false
	}
	if _, err := pipelineFor(qmvBF16KernelName(vocab, dModel, groupSize, bits)); err != nil {
		return false
	}
	if _, err := bf16LogitsTopKTilesPipeline(); err != nil {
		return false
	}
	if _, err := topKMergeF32Pipeline(); err != nil {
		return false
	}
	return true
}

func q4LMHeadTopKUsable(dModel, vocab, groupSize, bits, topK int) bool {
	if dModel <= 0 || vocab <= 0 || bits != 4 || topK <= 0 || topK > headSampleTopKMaxK || topK > vocab {
		return false
	}
	if groupSize != 32 && groupSize != 64 && groupSize != 128 {
		return false
	}
	if dModel%groupSize != 0 || dModel%q4LMHeadTopKBlockSize != 0 {
		return false
	}
	if _, err := q4LMHeadTopKTilesPipeline(); err != nil {
		return false
	}
	if _, err := topKMergeF32Pipeline(); err != nil {
		return false
	}
	return true
}

func topKSampleUsable(topK int) bool {
	if topK <= 0 || topK > headSampleTopKMaxK {
		return false
	}
	if _, err := topKMergeSampleF32Pipeline(); err != nil {
		return false
	}
	return true
}

func logitsSampleBF16Usable(vocab int) bool {
	if vocab <= 0 {
		return false
	}
	if _, err := logitsSampleBF16Pipeline(); err != nil {
		return false
	}
	return true
}

func q4LMHeadTopKCandidateCount(vocab, topK int) int {
	perTile := topK
	if q4LMHeadTopKRowsPerTile < perTile {
		perTile = q4LMHeadTopKRowsPerTile
	}
	tileCount := (vocab + q4LMHeadTopKRowsPerTile - 1) / q4LMHeadTopKRowsPerTile
	return tileCount * perTile
}

func q4LMHeadTopKCandidatesPerTile(topK int) int {
	if q4LMHeadTopKRowsPerTile < topK {
		return q4LMHeadTopKRowsPerTile
	}
	return topK
}

func encBF16LogitsArgmaxTilesBF16(
	enc metal.MTLComputeCommandEncoder,
	logits, tileValues, tileIndices, suppress metal.MTLBuffer,
	vocab, suppressCount int,
) error {
	if vocab <= 0 {
		return core.NewError("native.encBF16LogitsArgmaxTilesBF16: invalid logits geometry")
	}
	pso, err := bf16LogitsArgmaxTilesPipeline()
	if err != nil {
		return err
	}
	tileCount := (vocab + bf16LogitsArgmaxRowsPerTile - 1) / bf16LogitsArgmaxRowsPerTile
	setPSO(enc, pso)
	setBuf(enc, logits, 0, 0)
	setBuf(enc, tileValues, 0, 1)
	setBuf(enc, tileIndices, 0, 2)
	setEncInt32(enc, int32(vocab), 3)
	if suppress == nil {
		suppress = logits
	}
	setBuf(enc, suppress, 0, 4)
	setEncInt32(enc, int32(suppressCount), 5)
	dispatchThreadgroups(enc,
		metal.MTLSize{Width: uint(tileCount), Height: 1, Depth: 1},
		metal.MTLSize{Width: 32, Height: 1, Depth: 1},
	)
	return nil
}

func encBF16LMHeadArgmaxTilesBF16(
	enc metal.MTLComputeCommandEncoder,
	x, weight, tileValues, tileIndices, suppress metal.MTLBuffer,
	xOff, weightOff uint,
	dModel, vocab, suppressCount int,
) error {
	if dModel <= 0 || vocab <= 0 {
		return core.NewError("native.encBF16LMHeadArgmaxTilesBF16: invalid head geometry")
	}
	pso, err := bf16LMHeadArgmaxTilesPipeline()
	if err != nil {
		return err
	}
	tileCount := (vocab + bf16LMHeadArgmaxRowsPerTile - 1) / bf16LMHeadArgmaxRowsPerTile
	setPSO(enc, pso)
	setBuf(enc, x, xOff, 0)
	setBuf(enc, weight, weightOff, 1)
	setBuf(enc, tileValues, 0, 2)
	setBuf(enc, tileIndices, 0, 3)
	setEncInt32(enc, int32(dModel), 4)
	setEncInt32(enc, int32(vocab), 5)
	if suppress == nil {
		suppress = x
	}
	setBuf(enc, suppress, 0, 6)
	setEncInt32(enc, int32(suppressCount), 7)
	dispatchThreadgroups(enc,
		metal.MTLSize{Width: uint(tileCount), Height: 1, Depth: 1},
		metal.MTLSize{Width: 32, Height: bf16LMHeadArgmaxRowsPerTile, Depth: 1},
	)
	return nil
}

func encArgmaxMergeF32(enc metal.MTLComputeCommandEncoder, values, indices, out metal.MTLBuffer, n int) error {
	if n <= 0 {
		return core.NewError("native.encArgmaxMergeF32: n must be > 0")
	}
	pso, err := argmaxMergeF32Pipeline()
	if err != nil {
		return err
	}
	setPSO(enc, pso)
	setBuf(enc, values, 0, 0)
	setBuf(enc, indices, 0, 1)
	setBuf(enc, out, 0, 2)
	setEncInt32(enc, int32(n), 3)
	dispatchThreads(enc,
		metal.MTLSize{Width: 32, Height: 1, Depth: 1},
		metal.MTLSize{Width: 32, Height: 1, Depth: 1},
	)
	return nil
}

func encBF16LMHeadCandidatesBF16(
	enc metal.MTLComputeCommandEncoder,
	x, weight, values, indices, suppress, history metal.MTLBuffer,
	xOff, weightOff uint,
	dModel, vocab, suppressCount, historyCount int,
	repeatPenalty, softCap float32,
) error {
	if dModel <= 0 || vocab <= 0 {
		return core.NewError("native.encBF16LMHeadCandidatesBF16: invalid head geometry")
	}
	pso, err := bf16LMHeadCandidatesPipeline()
	if err != nil {
		return err
	}
	tileCount := (vocab + bf16LMHeadArgmaxRowsPerTile - 1) / bf16LMHeadArgmaxRowsPerTile
	setPSO(enc, pso)
	setBuf(enc, x, xOff, 0)
	setBuf(enc, weight, weightOff, 1)
	setBuf(enc, values, 0, 2)
	setBuf(enc, indices, 0, 3)
	setEncInt32(enc, int32(dModel), 4)
	setEncInt32(enc, int32(vocab), 5)
	if suppress == nil {
		suppress = x
	}
	setBuf(enc, suppress, 0, 6)
	setEncInt32(enc, int32(suppressCount), 7)
	if history == nil {
		history = x
	}
	setBuf(enc, history, 0, 8)
	setEncInt32(enc, int32(historyCount), 9)
	setEncFloat32(enc, repeatPenalty, 10)
	setEncFloat32(enc, softCap, 11)
	dispatchThreadgroups(enc,
		metal.MTLSize{Width: uint(tileCount), Height: 1, Depth: 1},
		metal.MTLSize{Width: 32, Height: bf16LMHeadArgmaxRowsPerTile, Depth: 1},
	)
	return nil
}

func encBF16LMHeadCandidatesBF16Object(
	enc metal.MTLComputeCommandEncoderObject,
	x, weight, values, indices, suppress, history metal.MTLBuffer,
	xOff, weightOff uint,
	dModel, vocab, suppressCount, historyCount int,
	repeatPenalty, softCap float32,
) error {
	if dModel <= 0 || vocab <= 0 {
		return core.NewError("native.encBF16LMHeadCandidatesBF16: invalid head geometry")
	}
	pso, err := bf16LMHeadCandidatesPipeline()
	if err != nil {
		return err
	}
	tileCount := (vocab + bf16LMHeadArgmaxRowsPerTile - 1) / bf16LMHeadArgmaxRowsPerTile
	sink := encObjectSink{enc: enc}
	sink.setPSO(pso)
	sink.setBuf(x, xOff, 0)
	sink.setBuf(weight, weightOff, 1)
	sink.setBuf(values, 0, 2)
	sink.setBuf(indices, 0, 3)
	sink.setI32(int32(dModel), 4)
	sink.setI32(int32(vocab), 5)
	if suppress == nil {
		suppress = x
	}
	sink.setBuf(suppress, 0, 6)
	sink.setI32(int32(suppressCount), 7)
	if history == nil {
		history = x
	}
	sink.setBuf(history, 0, 8)
	sink.setI32(int32(historyCount), 9)
	sink.setF32(repeatPenalty, 10)
	sink.setF32(softCap, 11)
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: uint(tileCount), Height: 1, Depth: 1},
		metal.MTLSize{Width: 32, Height: bf16LMHeadArgmaxRowsPerTile, Depth: 1},
	)
	return nil
}

func encBF16LogitsCandidatesBF16(
	enc metal.MTLComputeCommandEncoder,
	logits, values, indices, suppress metal.MTLBuffer,
	vocab, suppressCount int,
	softCap float32,
) error {
	if vocab <= 0 {
		return core.NewError("native.encBF16LogitsCandidatesBF16: invalid logits geometry")
	}
	pso, err := bf16LogitsCandidatesPipeline()
	if err != nil {
		return err
	}
	setPSO(enc, pso)
	setBuf(enc, logits, 0, 0)
	setBuf(enc, values, 0, 1)
	setBuf(enc, indices, 0, 2)
	setEncInt32(enc, int32(vocab), 3)
	if suppress == nil {
		suppress = logits
	}
	setBuf(enc, suppress, 0, 4)
	setEncInt32(enc, int32(suppressCount), 5)
	setEncFloat32(enc, softCap, 6)
	group := uint(256)
	if uint(vocab) < group {
		group = uint(vocab)
	}
	dispatchThreads(enc,
		metal.MTLSize{Width: uint(vocab), Height: 1, Depth: 1},
		metal.MTLSize{Width: group, Height: 1, Depth: 1},
	)
	return nil
}

func encBF16LogitsTopKTilesBF16(
	enc metal.MTLComputeCommandEncoder,
	logits, values, indices, suppress, history metal.MTLBuffer,
	vocab, suppressCount, historyCount, topK int,
	repeatPenalty, softCap float32,
) error {
	if vocab <= 0 {
		return core.NewError("native.encBF16LogitsTopKTilesBF16: invalid logits geometry")
	}
	if topK <= 0 || topK > headSampleTopKMaxK {
		return core.NewError("native.encBF16LogitsTopKTilesBF16: topK must be in 1..64")
	}
	pso, err := bf16LogitsTopKTilesPipeline()
	if err != nil {
		return err
	}
	tileCount := (vocab + bf16LogitsArgmaxRowsPerTile - 1) / bf16LogitsArgmaxRowsPerTile
	setPSO(enc, pso)
	setBuf(enc, logits, 0, 0)
	setBuf(enc, values, 0, 1)
	setBuf(enc, indices, 0, 2)
	setEncInt32(enc, int32(vocab), 3)
	if suppress == nil {
		suppress = logits
	}
	setBuf(enc, suppress, 0, 4)
	setEncInt32(enc, int32(suppressCount), 5)
	if history == nil {
		history = logits
	}
	setBuf(enc, history, 0, 6)
	setEncInt32(enc, int32(historyCount), 7)
	setEncFloat32(enc, repeatPenalty, 8)
	setEncFloat32(enc, softCap, 9)
	setEncInt32(enc, int32(topK), 10)
	dispatchThreadgroups(enc,
		metal.MTLSize{Width: uint(tileCount), Height: 1, Depth: 1},
		metal.MTLSize{Width: 32, Height: 1, Depth: 1},
	)
	return nil
}

func encBF16LogitsTopKTilesBF16Object(
	enc metal.MTLComputeCommandEncoderObject,
	logits, values, indices, suppress, history metal.MTLBuffer,
	vocab, suppressCount, historyCount, topK int,
	repeatPenalty, softCap float32,
) error {
	if vocab <= 0 {
		return core.NewError("native.encBF16LogitsTopKTilesBF16: invalid logits geometry")
	}
	if topK <= 0 || topK > headSampleTopKMaxK {
		return core.NewError("native.encBF16LogitsTopKTilesBF16: topK must be in 1..64")
	}
	pso, err := bf16LogitsTopKTilesPipeline()
	if err != nil {
		return err
	}
	tileCount := (vocab + bf16LogitsArgmaxRowsPerTile - 1) / bf16LogitsArgmaxRowsPerTile
	sink := encObjectSink{enc: enc}
	sink.setPSO(pso)
	sink.setBuf(logits, 0, 0)
	sink.setBuf(values, 0, 1)
	sink.setBuf(indices, 0, 2)
	sink.setI32(int32(vocab), 3)
	if suppress == nil {
		suppress = logits
	}
	sink.setBuf(suppress, 0, 4)
	sink.setI32(int32(suppressCount), 5)
	if history == nil {
		history = logits
	}
	sink.setBuf(history, 0, 6)
	sink.setI32(int32(historyCount), 7)
	sink.setF32(repeatPenalty, 8)
	sink.setF32(softCap, 9)
	sink.setI32(int32(topK), 10)
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: uint(tileCount), Height: 1, Depth: 1},
		metal.MTLSize{Width: 32, Height: 1, Depth: 1},
	)
	return nil
}

func encQ4LMHeadTopKTilesBF16(
	enc metal.MTLComputeCommandEncoder,
	x, weight, scales, biases, values, indices, suppress, history metal.MTLBuffer,
	xOff, weightOff, scalesOff, biasesOff uint,
	dModel, vocab, groupSize, suppressCount, historyCount, topK, candidatesPerTile int,
	repeatPenalty, softCap float32,
) error {
	if dModel <= 0 || vocab <= 0 {
		return core.NewError("native.encQ4LMHeadTopKTilesBF16: invalid head geometry")
	}
	if topK <= 0 || topK > headSampleTopKMaxK {
		return core.NewError("native.encQ4LMHeadTopKTilesBF16: topK must be in 1..64")
	}
	if candidatesPerTile <= 0 || candidatesPerTile > topK || candidatesPerTile > q4LMHeadTopKRowsPerTile {
		return core.NewError("native.encQ4LMHeadTopKTilesBF16: invalid candidatesPerTile")
	}
	if groupSize != 32 && groupSize != 64 && groupSize != 128 {
		return core.NewError("native.encQ4LMHeadTopKTilesBF16: groupSize must be 32, 64, or 128")
	}
	if dModel%groupSize != 0 || dModel%q4LMHeadTopKBlockSize != 0 {
		return core.NewError("native.encQ4LMHeadTopKTilesBF16: dModel must be a multiple of groupSize and 512")
	}
	pso, err := q4LMHeadTopKTilesPipeline()
	if err != nil {
		return err
	}
	tileCount := (vocab + q4LMHeadTopKRowsPerTile - 1) / q4LMHeadTopKRowsPerTile
	setPSO(enc, pso)
	setBuf(enc, x, xOff, 0)
	setBuf(enc, weight, weightOff, 1)
	setBuf(enc, scales, scalesOff, 2)
	setBuf(enc, biases, biasesOff, 3)
	setBuf(enc, values, 0, 4)
	setBuf(enc, indices, 0, 5)
	setEncInt32(enc, int32(dModel), 6)
	setEncInt32(enc, int32(vocab), 7)
	setEncInt32(enc, int32(groupSize), 8)
	if suppress == nil {
		suppress = x
	}
	setBuf(enc, suppress, 0, 9)
	setEncInt32(enc, int32(suppressCount), 10)
	if history == nil {
		history = x
	}
	setBuf(enc, history, 0, 11)
	setEncInt32(enc, int32(historyCount), 12)
	setEncFloat32(enc, repeatPenalty, 13)
	setEncFloat32(enc, softCap, 14)
	setEncInt32(enc, int32(topK), 15)
	setEncInt32(enc, int32(candidatesPerTile), 16)
	dispatchThreadgroups(enc,
		metal.MTLSize{Width: uint(tileCount), Height: 1, Depth: 1},
		metal.MTLSize{Width: 32, Height: q4LMHeadTopKSimdgroups, Depth: 1},
	)
	return nil
}

func encQ4LMHeadTopKTilesBF16Object(
	enc metal.MTLComputeCommandEncoderObject,
	x, weight, scales, biases, values, indices, suppress, history metal.MTLBuffer,
	xOff, weightOff, scalesOff, biasesOff uint,
	dModel, vocab, groupSize, suppressCount, historyCount, topK, candidatesPerTile int,
	repeatPenalty, softCap float32,
) error {
	if dModel <= 0 || vocab <= 0 {
		return core.NewError("native.encQ4LMHeadTopKTilesBF16: invalid head geometry")
	}
	if topK <= 0 || topK > headSampleTopKMaxK {
		return core.NewError("native.encQ4LMHeadTopKTilesBF16: topK must be in 1..64")
	}
	if candidatesPerTile <= 0 || candidatesPerTile > topK || candidatesPerTile > q4LMHeadTopKRowsPerTile {
		return core.NewError("native.encQ4LMHeadTopKTilesBF16: invalid candidatesPerTile")
	}
	if groupSize != 32 && groupSize != 64 && groupSize != 128 {
		return core.NewError("native.encQ4LMHeadTopKTilesBF16: groupSize must be 32, 64, or 128")
	}
	if dModel%groupSize != 0 || dModel%q4LMHeadTopKBlockSize != 0 {
		return core.NewError("native.encQ4LMHeadTopKTilesBF16: dModel must be a multiple of groupSize and 512")
	}
	pso, err := q4LMHeadTopKTilesPipeline()
	if err != nil {
		return err
	}
	if suppress == nil {
		suppress = x
	}
	if history == nil {
		history = x
	}
	tileCount := (vocab + q4LMHeadTopKRowsPerTile - 1) / q4LMHeadTopKRowsPerTile
	sink := encObjectSink{enc: enc}
	sink.setPSO(pso)
	sink.setBuf(x, xOff, 0)
	sink.setBuf(weight, weightOff, 1)
	sink.setBuf(scales, scalesOff, 2)
	sink.setBuf(biases, biasesOff, 3)
	sink.setBuf(values, 0, 4)
	sink.setBuf(indices, 0, 5)
	sink.setI32(int32(dModel), 6)
	sink.setI32(int32(vocab), 7)
	sink.setI32(int32(groupSize), 8)
	sink.setBuf(suppress, 0, 9)
	sink.setI32(int32(suppressCount), 10)
	sink.setBuf(history, 0, 11)
	sink.setI32(int32(historyCount), 12)
	sink.setF32(repeatPenalty, 13)
	sink.setF32(softCap, 14)
	sink.setI32(int32(topK), 15)
	sink.setI32(int32(candidatesPerTile), 16)
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: uint(tileCount), Height: 1, Depth: 1},
		metal.MTLSize{Width: 32, Height: q4LMHeadTopKSimdgroups, Depth: 1},
	)
	return nil
}

func encTopKMergeF32(enc metal.MTLComputeCommandEncoder, values, indices, outValues, outIndices metal.MTLBuffer, n, topK int) error {
	if n <= 0 {
		return core.NewError("native.encTopKMergeF32: n must be > 0")
	}
	if topK <= 0 || topK > headSampleTopKMaxK {
		return core.NewError("native.encTopKMergeF32: topK must be in 1..64")
	}
	pso, err := topKMergeF32Pipeline()
	if err != nil {
		return err
	}
	setPSO(enc, pso)
	setBuf(enc, values, 0, 0)
	setBuf(enc, indices, 0, 1)
	setBuf(enc, outValues, 0, 2)
	setBuf(enc, outIndices, 0, 3)
	setEncInt32(enc, int32(n), 4)
	setEncInt32(enc, int32(topK), 5)
	dispatchThreads(enc,
		metal.MTLSize{Width: 32, Height: 1, Depth: 1},
		metal.MTLSize{Width: 32, Height: 1, Depth: 1},
	)
	return nil
}

func encTopKMergeF32Object(enc metal.MTLComputeCommandEncoderObject, values, indices, outValues, outIndices metal.MTLBuffer, n, topK int) error {
	if n <= 0 {
		return core.NewError("native.encTopKMergeF32: n must be > 0")
	}
	if topK <= 0 || topK > headSampleTopKMaxK {
		return core.NewError("native.encTopKMergeF32: topK must be in 1..64")
	}
	pso, err := topKMergeF32Pipeline()
	if err != nil {
		return err
	}
	sink := encObjectSink{enc: enc}
	sink.setPSO(pso)
	sink.setBuf(values, 0, 0)
	sink.setBuf(indices, 0, 1)
	sink.setBuf(outValues, 0, 2)
	sink.setBuf(outIndices, 0, 3)
	sink.setI32(int32(n), 4)
	sink.setI32(int32(topK), 5)
	sink.dispatchThreads(
		metal.MTLSize{Width: 32, Height: 1, Depth: 1},
		metal.MTLSize{Width: 32, Height: 1, Depth: 1},
	)
	return nil
}

func encTopKMergeSampleF32(enc metal.MTLComputeCommandEncoder, values, indices, out, params metal.MTLBuffer) error {
	if params == nil {
		return core.NewError("native.encTopKMergeSampleF32: missing params buffer")
	}
	pso, err := topKMergeSampleF32Pipeline()
	if err != nil {
		return err
	}
	setPSO(enc, pso)
	setBuf(enc, values, 0, 0)
	setBuf(enc, indices, 0, 1)
	setBuf(enc, out, 0, 2)
	setBuf(enc, params, 0, 3)
	dispatchThreads(enc,
		metal.MTLSize{Width: 32, Height: 1, Depth: 1},
		metal.MTLSize{Width: 32, Height: 1, Depth: 1},
	)
	return nil
}

func encTopKMergeSampleF32Object(enc metal.MTLComputeCommandEncoderObject, values, indices, out, params metal.MTLBuffer) error {
	if params == nil {
		return core.NewError("native.encTopKMergeSampleF32: missing params buffer")
	}
	pso, err := topKMergeSampleF32Pipeline()
	if err != nil {
		return err
	}
	sink := encObjectSink{enc: enc}
	sink.setPSO(pso)
	sink.setBuf(values, 0, 0)
	sink.setBuf(indices, 0, 1)
	sink.setBuf(out, 0, 2)
	sink.setBuf(params, 0, 3)
	sink.dispatchThreads(
		metal.MTLSize{Width: 32, Height: 1, Depth: 1},
		metal.MTLSize{Width: 32, Height: 1, Depth: 1},
	)
	return nil
}

func encLogitsSampleBF16(enc metal.MTLComputeCommandEncoder, logits, suppress, history, out, params metal.MTLBuffer) error {
	if params == nil {
		return core.NewError("native.encLogitsSampleBF16: missing params buffer")
	}
	pso, err := logitsSampleBF16Pipeline()
	if err != nil {
		return err
	}
	if suppress == nil {
		suppress = logits
	}
	if history == nil {
		history = logits
	}
	setPSO(enc, pso)
	setBuf(enc, logits, 0, 0)
	setBuf(enc, suppress, 0, 1)
	setBuf(enc, history, 0, 2)
	setBuf(enc, out, 0, 3)
	setBuf(enc, params, 0, 4)
	dispatchThreads(enc,
		metal.MTLSize{Width: 256, Height: 1, Depth: 1},
		metal.MTLSize{Width: 256, Height: 1, Depth: 1},
	)
	return nil
}

func encLogitsSampleBF16Object(enc metal.MTLComputeCommandEncoderObject, logits, suppress, history, out, params metal.MTLBuffer) error {
	if params == nil {
		return core.NewError("native.encLogitsSampleBF16: missing params buffer")
	}
	pso, err := logitsSampleBF16Pipeline()
	if err != nil {
		return err
	}
	if suppress == nil {
		suppress = logits
	}
	if history == nil {
		history = logits
	}
	sink := encObjectSink{enc: enc}
	sink.setPSO(pso)
	sink.setBuf(logits, 0, 0)
	sink.setBuf(suppress, 0, 1)
	sink.setBuf(history, 0, 2)
	sink.setBuf(out, 0, 3)
	sink.setBuf(params, 0, 4)
	sink.dispatchThreads(
		metal.MTLSize{Width: 256, Height: 1, Depth: 1},
		metal.MTLSize{Width: 256, Height: 1, Depth: 1},
	)
	return nil
}
