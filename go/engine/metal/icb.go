// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/metal"
	"github.com/tmc/apple/objc"
)

// Indirect Command Buffer (ICB) replay: record a fixed op sequence ONCE, then
// re-issue it per step with a single executeCommandsInBuffer call instead of
// re-encoding every op on the host. This is the encode-bypass lever — a decode
// step's command sequence is fixed across tokens, so recording it once skips the
// per-token host re-encode. Two things differ from the regular encode path:
//   - ICB commands bind only BUFFERS, never inline setBytes — so every scalar
//     parameter becomes a tiny persistent buffer (scalarI32/scalarI64/scalarF32);
//   - the ICB has no automatic hazard tracking, so dependent commands need an
//     explicit SetBarrier, and the replay encoder must mark every referenced
//     buffer resident with UseResource.
//
// NormProjectICB de-risks the mechanism on a real dependent 2-op sequence
// (rms→gemv) before it scales to the full DecodeLayer.

// icbPSOCache memoises ICB-capable pipelines (built with
// supportIndirectCommandBuffers=true, required for a kernel to run inside an ICB).
var (
	icbPSOMu                     sync.Mutex
	icbPSOCache                  = map[string]metal.MTLComputePipelineState{}
	sdpaVectorICBHeadDimPSOCache = map[int]metal.MTLComputePipelineState{}
)

// pipelineForICB builds (and caches) an ICB-capable pipeline for a metallib
// kernel — same kernel as pipelineFor, but the descriptor sets
// supportIndirectCommandBuffers so it can be recorded into an indirect command.
func pipelineForICB(name string) (metal.MTLComputePipelineState, error) {
	icbPSOMu.Lock()
	defer icbPSOMu.Unlock()
	if pso, ok := icbPSOCache[name]; ok {
		return pso, nil
	}
	if library == nil || library.GetID() == 0 {
		return nil, core.NewError("native.pipelineForICB: library unavailable for " + name)
	}
	fn := library.NewFunctionWithName(name)
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.pipelineForICB: kernel " + name + " not found")
	}
	desc := metal.NewMTLComputePipelineDescriptor()
	desc.SetComputeFunction(fn)
	desc.SetSupportIndirectCommandBuffers(true)
	pso, err := device.NewComputePipelineStateWithDescriptorOptionsReflectionError(desc, 0, nil)
	if err != nil {
		return nil, core.E("native.pipelineForICB", name, err)
	}
	icbPSOCache[name] = pso
	return pso, nil
}

// geluPipelineICB builds (and caches) the ICB-capable fused gelu pipeline from the
// custom kernels library (pipelineForICB resolves from the main metallib; the fused
// gelu lives in customLibrary). Used by the ICB decode sites when gpuHasGeluKernel.
func geluPipelineICB() (metal.MTLComputePipelineState, error) {
	const key = "lthn_gelu_gate_mul_bf16|icb"
	icbPSOMu.Lock()
	defer icbPSOMu.Unlock()
	if pso, ok := icbPSOCache[key]; ok {
		return pso, nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		return nil, core.NewError("native.geluPipelineICB: custom library unavailable")
	}
	fn := customLibrary.NewFunctionWithName("lthn_gelu_gate_mul_bf16")
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.geluPipelineICB: kernel lthn_gelu_gate_mul_bf16 not found")
	}
	desc := metal.NewMTLComputePipelineDescriptor()
	desc.SetComputeFunction(fn)
	desc.SetSupportIndirectCommandBuffers(true)
	pso, err := device.NewComputePipelineStateWithDescriptorOptionsReflectionError(desc, 0, nil)
	if err != nil {
		return nil, core.E("native.geluPipelineICB", "pipeline", err)
	}
	icbPSOCache[key] = pso
	return pso, nil
}

// qkNormRopePipelineICB builds (and caches) the ICB-capable fused per-head QK-norm + RoPE pipeline
// (lthn_qknorm_rope_bf16). Lockstep with the re-encode encQKNormRope (same kernel) so the two stay
// byte-equal; ~1 ULP from the old composed rms-rows→rope path (see the kernel comment).
func qkNormRopePipelineICB() (metal.MTLComputePipelineState, error) {
	const key = "lthn_qknorm_rope_bf16|icb"
	icbPSOMu.Lock()
	defer icbPSOMu.Unlock()
	if pso, ok := icbPSOCache[key]; ok {
		return pso, nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		return nil, core.NewError("native.qkNormRopePipelineICB: custom library unavailable")
	}
	fn := customLibrary.NewFunctionWithName("lthn_qknorm_rope_bf16")
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.qkNormRopePipelineICB: kernel lthn_qknorm_rope_bf16 not found")
	}
	desc := metal.NewMTLComputePipelineDescriptor()
	desc.SetComputeFunction(fn)
	desc.SetSupportIndirectCommandBuffers(true)
	pso, err := device.NewComputePipelineStateWithDescriptorOptionsReflectionError(desc, 0, nil)
	if err != nil {
		return nil, core.E("native.qkNormRopePipelineICB", "pipeline", err)
	}
	icbPSOCache[key] = pso
	return pso, nil
}

// rmsResidualPipelineICB builds (and caches) the ICB-capable fused residual-RMSNorm pipeline
// (lthn_rmsnorm_residual_bf16: out = res + rmsnorm(x, w)). supportIndirectCommandBuffers is required —
// without it the kernel faults when recorded into an ICB command. Same kernel as RMSNormResidualBF16.
// rmsResidualPipelineICB selects the fused post-norm tail kernel by axis, exactly as
// rmsNormResidualPipelineFor does for the live encoder: single-row up to rmsLoopedLimit,
// the grid-striding looped variant beyond (gemma4 31B hidden 5376 — the #348 tail-drop).
func rmsResidualPipelineICB(axisSize int) (metal.MTLComputePipelineState, error) {
	name := "lthn_rmsnorm_residual_bf16"
	if axisSize > rmsLoopedLimit {
		name = "lthn_rmsnorm_residual_looped_bf16"
	}
	key := name + "|icb"
	icbPSOMu.Lock()
	defer icbPSOMu.Unlock()
	if pso, ok := icbPSOCache[key]; ok {
		return pso, nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		return nil, core.NewError("native.rmsResidualPipelineICB: custom library unavailable")
	}
	fn := customLibrary.NewFunctionWithName(name)
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.rmsResidualPipelineICB: kernel " + name + " not found — rebuild lthn_kernels.metallib")
	}
	desc := metal.NewMTLComputePipelineDescriptor()
	desc.SetComputeFunction(fn)
	desc.SetSupportIndirectCommandBuffers(true)
	pso, err := device.NewComputePipelineStateWithDescriptorOptionsReflectionError(desc, 0, nil)
	if err != nil {
		return nil, core.E("native.rmsResidualPipelineICB", "pipeline", err)
	}
	icbPSOCache[key] = pso
	return pso, nil
}

// squareICB records the contiguous Square kernel once into an ICB and replays it
// — the smallest real ICB (one op, in/out + a scalar count as a buffer) to isolate
// the basic mechanism (ICB-capable PSO + scalar-as-buffer + residency + execute)
// from the multi-op barrier path. Returns in[i]² if the ICB executes correctly.
func squareICB(in []float32) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	pso, err := pipelineForICB("v_Squarefloat32float32")
	if err != nil {
		return nil, err
	}
	n := len(in)
	out := make([]float32, n)
	withAutoreleasePool(func() {
		inBuf := shared(in)
		outBuf := scratch(n)
		sizeBuf := scalarI32(int32(n))

		icbDesc := metal.NewMTLIndirectCommandBufferDescriptor()
		icbDesc.SetCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch | metal.MTLIndirectCommandTypeConcurrentDispatchThreads)
		icbDesc.SetInheritBuffers(false)
		icbDesc.SetInheritPipelineState(false)
		icbDesc.SetMaxKernelBufferBindCount(4)
		icb := device.NewIndirectCommandBufferWithDescriptorMaxCommandCountOptions(icbDesc, 1, metal.MTLResourceStorageModeShared)

		c0 := indirectComputeCommandAtIndexFast(icb, 0)
		emitUnary(fastICBSink{c0}, pso, inBuf, outBuf, n)

		resident := []metal.MTLResource{inBuf, outBuf, sizeBuf}
		residentIDs := resourceIDsForFastUse(nil, resident)
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		useResourcesIDsFast(enc, resident, residentIDs, metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
		executeCommandsInBufferWithRangeFast(enc, icb, foundation.NSRange{Location: 0, Length: 1})
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		copy(out, unsafe.Slice((*float32)(outBuf.Contents()), n))
	})
	return out, nil
}

// gemvICB records the gemv kernel once into an ICB and replays it — isolates
// gemv-in-ICB (threadgroups dispatch, 10 buffer binds incl. scalars) from the
// multi-op path. Returns mat @ vec if correct.
func gemvICB(mat, vec []float32, outDim, inDim int) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	bm, bn, sm, sn, tm, tn := gemvTiles(inDim, outDim)
	pso, err := pipelineForICB(gemvKernelName("float32", bm, bn, sm, sn, tm, tn))
	if err != nil {
		return nil, err
	}
	out := make([]float32, outDim)
	withAutoreleasePool(func() {
		matBuf, vecBuf := shared(mat), shared(vec)
		outBuf := scratch(outDim)
		inB, outB, ldB := scalarI32(int32(inDim)), scalarI32(int32(outDim)), scalarI32(int32(inDim))
		bndB, bshB, vsB, msB := scalarI32(1), scalarI32(1), scalarI64(0), scalarI64(0)

		icbDesc := metal.NewMTLIndirectCommandBufferDescriptor()
		icbDesc.SetCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch | metal.MTLIndirectCommandTypeConcurrentDispatchThreads)
		icbDesc.SetInheritBuffers(false)
		icbDesc.SetInheritPipelineState(false)
		icbDesc.SetMaxKernelBufferBindCount(16)
		icb := device.NewIndirectCommandBufferWithDescriptorMaxCommandCountOptions(icbDesc, 1, metal.MTLResourceStorageModeShared)

		c0 := indirectComputeCommandAtIndexFast(icb, 0)
		emitGemv(fastICBSink{c0}, pso, matBuf, 0, vecBuf, outBuf, 0, inDim, outDim, bm, bn, sm, tm)

		resident := []metal.MTLResource{matBuf, vecBuf, outBuf, inB, outB, ldB, bndB, bshB, vsB, msB}
		residentIDs := resourceIDsForFastUse(nil, resident)
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		useResourcesIDsFast(enc, resident, residentIDs, metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
		executeCommandsInBufferWithRangeFast(enc, icb, foundation.NSRange{Location: 0, Length: 1})
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		copy(out, unsafe.Slice((*float32)(outBuf.Contents()), outDim))
	})
	return out, nil
}

// rebindProbeICB records ONE gemv command into an ICB, then replays it nRows
// times — re-setting only the output buffer's OFFSET (SetKernelBufferOffsetAtIndex
// at index 3) between replays so replay r writes row r of a tall output buffer.
// It is the smallest test of the cache-grow lever: an ICB's command bindings are
// recorded once, but re-setting one buffer offset per replay is far cheaper than
// re-encoding, and IS the mechanism the growing KV cache needs (the per-token
// write row advances while the rest of the command stays recorded). Returns the
// nRows×outDim output; every row must equal mat@vec if the rebind takes effect.
func rebindProbeICB(mat, vec []float32, outDim, inDim, nRows int) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	bm, bn, sm, sn, tm, tn := gemvTiles(inDim, outDim)
	pso, err := pipelineForICB(gemvKernelName("float32", bm, bn, sm, sn, tm, tn))
	if err != nil {
		return nil, err
	}
	out := make([]float32, nRows*outDim)
	withAutoreleasePool(func() {
		matBuf, vecBuf := shared(mat), shared(vec)
		outBuf := scratch(nRows * outDim)
		inB, outDimB, ldB := scalarI32(int32(inDim)), scalarI32(int32(outDim)), scalarI32(int32(inDim))
		bndB, bshB, vsB, msB := scalarI32(1), scalarI32(1), scalarI64(0), scalarI64(0)

		icbDesc := metal.NewMTLIndirectCommandBufferDescriptor()
		icbDesc.SetCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch | metal.MTLIndirectCommandTypeConcurrentDispatchThreads)
		icbDesc.SetInheritBuffers(false)
		icbDesc.SetInheritPipelineState(false)
		icbDesc.SetMaxKernelBufferBindCount(16)
		icb := device.NewIndirectCommandBufferWithDescriptorMaxCommandCountOptions(icbDesc, 1, metal.MTLResourceStorageModeShared)

		c0 := indirectComputeCommandAtIndexFast(icb, 0)
		emitGemv(fastICBSink{c0}, pso, matBuf, 0, vecBuf, outBuf, 0, inDim, outDim, bm, bn, sm, tm)

		resident := []metal.MTLResource{matBuf, vecBuf, outBuf, inB, outDimB, ldB, bndB, bshB, vsB, msB}
		residentIDs := resourceIDsForFastUse(nil, resident)
		for r := range nRows {
			// the only per-replay change: advance the output row (4 bytes/f32)
			setICBKernelBuffer(c0, outBuf, uint(r*outDim*4), 3)
			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			useResourcesIDsFast(enc, resident, residentIDs, metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
			executeCommandsInBufferWithRangeFast(enc, icb, foundation.NSRange{Location: 0, Length: 1})
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
		}
		copy(out, unsafe.Slice((*float32)(outBuf.Contents()), nRows*outDim))
	})
	return out, nil
}

// qmvICB records the bf16-activation 4-bit qmv ONCE into an ICB and replays it —
// the smallest proof that affine_qmv_bfloat16_t works as an INDIRECT command. It's
// a plain named kernel (no function constants, unlike rope/sdpa), so pipelineForICB
// should build it ICB-capable directly; this confirms that plus the w0 s1 b2 x3
// out4 K5 N6 binding as an ICB command. The qmv projection swap in the cache-grow
// ICB rests on this. Returns out = x @ Wᵀ; must equal QMVBF16 on the same bytes.
func qmvICB(x, wq, scales, biases []byte, outDim, inDim, groupSize, bits int) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	pso, err := pipelineForICB(qmvBF16KernelName(outDim, inDim, groupSize, bits))
	if err != nil {
		return nil, err
	}
	out := make([]byte, outDim*bf16Size)
	withAutoreleasePool(func() {
		wBuf, sBuf, bBuf := residentBytes(wq), residentBytes(scales), residentBytes(biases)
		xBuf := sharedBytes(x)
		outBuf := scratchBF16(outDim)
		kB, nB := scalarI32(int32(inDim)), scalarI32(int32(outDim))

		icbDesc := metal.NewMTLIndirectCommandBufferDescriptor()
		icbDesc.SetCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch | metal.MTLIndirectCommandTypeConcurrentDispatchThreads)
		icbDesc.SetInheritBuffers(false)
		icbDesc.SetInheritPipelineState(false)
		icbDesc.SetMaxKernelBufferBindCount(8)
		icb := device.NewIndirectCommandBufferWithDescriptorMaxCommandCountOptions(icbDesc, 1, metal.MTLResourceStorageModeShared)

		c0 := indirectComputeCommandAtIndexFast(icb, 0)
		emitQMV(fastICBSink{c0}, pso, wBuf, 0, sBuf, 0, bBuf, 0, xBuf, outBuf, 0, inDim, outDim)

		resident := []metal.MTLResource{wBuf, sBuf, bBuf, xBuf, outBuf, kB, nB}
		residentIDs := resourceIDsForFastUse(nil, resident)
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		useResourcesIDsFast(enc, resident, residentIDs, metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
		executeCommandsInBufferWithRangeFast(enc, icb, foundation.NSRange{Location: 0, Length: 1})
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), outDim*bf16Size))
	})
	return out, nil
}

// ropePipelineICB / sdpaVectorPipelineICB are the ICB-capable, function-constant
// pipelines — the new wrinkle for the attention ICB: combine the specialised
// function (func consts) with the ICB descriptor (supportIndirectCommandBuffers).
const (
	ropeICBKey                 = "rope_single_bfloat16|icb|trad=false"
	ropeICBTraditionalKey      = "rope_single_bfloat16|icb|trad=true"
	ropeFreqsICBKey            = "rope_single_freqs_bfloat16|icb|trad=false"
	ropeFreqsICBTraditionalKey = "rope_single_freqs_bfloat16|icb|trad=true"
)

func ropePipelineICBKey(traditional bool) string {
	if traditional {
		return ropeICBTraditionalKey
	}
	return ropeICBKey
}

func ropeFreqsPipelineICBKey(traditional bool) string {
	if traditional {
		return ropeFreqsICBTraditionalKey
	}
	return ropeFreqsICBKey
}

func ropePipelineICB(traditional bool) (metal.MTLComputePipelineState, error) {
	key := ropePipelineICBKey(traditional)
	icbPSOMu.Lock()
	defer icbPSOMu.Unlock()
	if pso, ok := icbPSOCache[key]; ok {
		return pso, nil
	}
	if library == nil || library.GetID() == 0 {
		return nil, core.NewError("native.ropePipelineICB: library unavailable")
	}
	fc := metal.NewMTLFunctionConstantValues()
	fwd, trad, transpose := uint8(1), uint8(0), uint8(0)
	if traditional {
		trad = 1
	}
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&fwd), metal.MTLDataTypeBool, 1)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&trad), metal.MTLDataTypeBool, 2)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&transpose), metal.MTLDataTypeBool, 3)
	fn, err := library.NewFunctionWithNameConstantValuesError("rope_single_bfloat16", fc)
	if err != nil {
		return nil, core.E("native.ropePipelineICB", "function", err)
	}
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.ropePipelineICB: kernel rope_single_bfloat16 not found")
	}
	desc := metal.NewMTLComputePipelineDescriptor()
	desc.SetComputeFunction(fn)
	desc.SetSupportIndirectCommandBuffers(true)
	pso, err := device.NewComputePipelineStateWithDescriptorOptionsReflectionError(desc, 0, nil)
	if err != nil {
		return nil, core.E("native.ropePipelineICB", "pipeline", err)
	}
	icbPSOCache[key] = pso
	return pso, nil
}

// ropeFreqsPipelineICB is ropePipelineICB for the explicit-periods rope (rope_single_freqs_bfloat16)
// — the kernel the host's encRopeDecode uses when a layer carries a periods spectrum (gemma4
// proportional-global or YaRN). Same fwd/trad/transpose constants as the base rope; ICB-replayable.
func ropeFreqsPipelineICB(traditional bool) (metal.MTLComputePipelineState, error) {
	key := ropeFreqsPipelineICBKey(traditional)
	icbPSOMu.Lock()
	defer icbPSOMu.Unlock()
	if pso, ok := icbPSOCache[key]; ok {
		return pso, nil
	}
	if library == nil || library.GetID() == 0 {
		return nil, core.NewError("native.ropeFreqsPipelineICB: library unavailable")
	}
	fc := metal.NewMTLFunctionConstantValues()
	fwd, trad, transpose := uint8(1), uint8(0), uint8(0)
	if traditional {
		trad = 1
	}
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&fwd), metal.MTLDataTypeBool, 1)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&trad), metal.MTLDataTypeBool, 2)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&transpose), metal.MTLDataTypeBool, 3)
	fn, err := library.NewFunctionWithNameConstantValuesError("rope_single_freqs_bfloat16", fc)
	if err != nil {
		return nil, core.E("native.ropeFreqsPipelineICB", "function", err)
	}
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.ropeFreqsPipelineICB: kernel rope_single_freqs_bfloat16 not found")
	}
	desc := metal.NewMTLComputePipelineDescriptor()
	desc.SetComputeFunction(fn)
	desc.SetSupportIndirectCommandBuffers(true)
	pso, err := device.NewComputePipelineStateWithDescriptorOptionsReflectionError(desc, 0, nil)
	if err != nil {
		return nil, core.E("native.ropeFreqsPipelineICB", "pipeline", err)
	}
	icbPSOCache[key] = pso
	return pso, nil
}

func sdpaVectorPipelineICB(name string) (metal.MTLComputePipelineState, error) {
	key := name + "|icb"
	icbPSOMu.Lock()
	defer icbPSOMu.Unlock()
	if pso, ok := icbPSOCache[key]; ok {
		return pso, nil
	}
	if library == nil || library.GetID() == 0 {
		return nil, core.NewError("native.sdpaVectorPipelineICB: library unavailable for " + name)
	}
	fc := metal.NewMTLFunctionConstantValues()
	off := uint8(0)
	for _, idx := range []uint{20, 21, 22, 23, 24, 25} {
		fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&off), metal.MTLDataTypeBool, idx)
	}
	fn, err := library.NewFunctionWithNameConstantValuesError(name, fc)
	if err != nil {
		return nil, core.E("native.sdpaVectorPipelineICB", name, err)
	}
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.sdpaVectorPipelineICB: kernel " + name + " not found")
	}
	desc := metal.NewMTLComputePipelineDescriptor()
	desc.SetComputeFunction(fn)
	desc.SetSupportIndirectCommandBuffers(true)
	pso, err := device.NewComputePipelineStateWithDescriptorOptionsReflectionError(desc, 0, nil)
	if err != nil {
		return nil, core.E("native.sdpaVectorPipelineICB", "pipeline "+name, err)
	}
	icbPSOCache[key] = pso
	return pso, nil
}

// sdpaVector2Pass1PipelineICB builds the ICB-recordable sdpa_vector_2pass_1 — the same
// function constants as the live builder (20-25 false: decode-time no mask/transpose/
// causal/sinks; 26 = blocks, baked because the kernel strides its key walk and indexes
// the partials by it) with supportIndirectCommandBuffers set. The recorded arch ICB
// replays this pair for GLOBAL layers so deep decode keeps the fanned-out reduction the
// single-pass kernel cannot give (one threadgroup per head serialising the whole cache
// — the deep-decode collapse).
func sdpaVector2Pass1PipelineICB(headDim int, blocks int32) (metal.MTLComputePipelineState, error) {
	key := core.Sprintf("sdpa_vector_2pass_1_bfloat16_t_%d_%d|b%d|icb", headDim, headDim, blocks)
	icbPSOMu.Lock()
	defer icbPSOMu.Unlock()
	if pso, ok := icbPSOCache[key]; ok {
		return pso, nil
	}
	if library == nil || library.GetID() == 0 {
		return nil, core.NewError("native.sdpaVector2Pass1PipelineICB: library unavailable")
	}
	fc := metal.NewMTLFunctionConstantValues()
	off := uint8(0)
	for _, idx := range []uint{20, 21, 22, 23, 24, 25} {
		fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&off), metal.MTLDataTypeBool, idx)
	}
	blk := blocks
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&blk), metal.MTLDataTypeInt, 26)
	name := core.Sprintf("sdpa_vector_2pass_1_bfloat16_t_%d_%d", headDim, headDim)
	fn, err := library.NewFunctionWithNameConstantValuesError(name, fc)
	if err != nil {
		return nil, core.E("native.sdpaVector2Pass1PipelineICB", name, err)
	}
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.sdpaVector2Pass1PipelineICB: kernel " + name + " not found")
	}
	desc := metal.NewMTLComputePipelineDescriptor()
	desc.SetComputeFunction(fn)
	desc.SetSupportIndirectCommandBuffers(true)
	pso, err := device.NewComputePipelineStateWithDescriptorOptionsReflectionError(desc, 0, nil)
	if err != nil {
		return nil, core.E("native.sdpaVector2Pass1PipelineICB", "pipeline "+name, err)
	}
	icbPSOCache[key] = pso
	return pso, nil
}

// sdpaVector2Pass2PipelineICB builds the ICB-recordable sdpa_vector_2pass_2 combine
// kernel (plain function, no constants — blocks arrives inlined at bind 4) with
// supportIndirectCommandBuffers set.
func sdpaVector2Pass2PipelineICB(headDim int) (metal.MTLComputePipelineState, error) {
	name := core.Sprintf("sdpa_vector_2pass_2_bfloat16_t_%d", headDim)
	key := name + "|icb"
	icbPSOMu.Lock()
	defer icbPSOMu.Unlock()
	if pso, ok := icbPSOCache[key]; ok {
		return pso, nil
	}
	if library == nil || library.GetID() == 0 {
		return nil, core.NewError("native.sdpaVector2Pass2PipelineICB: library unavailable")
	}
	fn := library.NewFunctionWithName(name)
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.sdpaVector2Pass2PipelineICB: kernel " + name + " not found")
	}
	desc := metal.NewMTLComputePipelineDescriptor()
	desc.SetComputeFunction(fn)
	desc.SetSupportIndirectCommandBuffers(true)
	pso, err := device.NewComputePipelineStateWithDescriptorOptionsReflectionError(desc, 0, nil)
	if err != nil {
		return nil, core.E("native.sdpaVector2Pass2PipelineICB", "pipeline "+name, err)
	}
	icbPSOCache[key] = pso
	return pso, nil
}

func sdpaVectorPipelineICBForHeadDim(headDim int) (metal.MTLComputePipelineState, error) {
	icbPSOMu.Lock()
	if pso, ok := sdpaVectorICBHeadDimPSOCache[headDim]; ok {
		icbPSOMu.Unlock()
		return pso, nil
	}
	icbPSOMu.Unlock()

	pso, err := sdpaVectorPipelineICB(core.Sprintf("sdpa_vector_bfloat16_t_%d_%d", headDim, headDim))
	if err != nil {
		return nil, err
	}

	icbPSOMu.Lock()
	if existing, ok := sdpaVectorICBHeadDimPSOCache[headDim]; ok {
		icbPSOMu.Unlock()
		return existing, nil
	}
	sdpaVectorICBHeadDimPSOCache[headDim] = pso
	icbPSOMu.Unlock()
	return pso, nil
}

type attentionBlockICBScratch struct {
	dModel, qDim, nHeads, nKVHeads, headDim, kvLen                         int
	x, k, v, out                                                           *pinnedNoCopyBytes
	xView, kView, vView                                                    cachedNoCopyBytesView
	normed, q, qr, attn, attnOut                                           metal.MTLBuffer
	offBuf                                                                 metal.MTLBuffer
	epsBuf, ropeScaleBuf, ropeBaseBuf, sdpaScaleBuf                        metal.MTLBuffer
	axisBuf, wsBuf, qInBuf, qOutBuf, qLdBuf, oInBuf, oOutBuf, oLdBuf       metal.MTLBuffer
	bndBuf, bshBuf, vsBuf, msBuf, ropeMatBuf, gqaBuf, nBuf, khsBuf, kssBuf metal.MTLBuffer
	vhsBuf, vssBuf, addCntBuf                                              metal.MTLBuffer
	offPtr                                                                 *int32
	epsPtr, ropeScalePtr, ropeBasePtr, sdpaScalePtr                        *float32
	icbDesc                                                                metal.MTLIndirectCommandBufferDescriptor
	icb                                                                    metal.MTLIndirectCommandBuffer
	rng                                                                    foundation.NSRange
	residentRes                                                            []metal.MTLResource
	residentIDs                                                            []objc.ID
	normID, wqID, woID                                                     uintptr
	xID, kID, vID                                                          uintptr
}

type attentionBlockICBScratchKey struct {
	dModel, qDim, nHeads, nKVHeads, headDim, kvLen int
}

type attentionBlockICBScratchPool struct {
	core.Pool[*attentionBlockICBScratch]
}

var attentionBlockICBScratchPools sync.Map

func newICBI32Storage(v int32) (metal.MTLBuffer, *int32, error) {
	buf := device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared)
	if buf == nil || buf.GetID() == 0 {
		return nil, nil, core.NewError("native.newICBI32Storage: failed to create scalar buffer")
	}
	ptr := (*int32)(buf.Contents())
	*ptr = v
	return buf, ptr, nil
}

func newICBF32Storage(v float32) (metal.MTLBuffer, *float32, error) {
	buf := device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared)
	if buf == nil || buf.GetID() == 0 {
		return nil, nil, core.NewError("native.newICBF32Storage: failed to create scalar buffer")
	}
	ptr := (*float32)(buf.Contents())
	*ptr = v
	return buf, ptr, nil
}

func newAttentionBlockICBScratch(dModel, qDim, nHeads, nKVHeads, headDim, kvLen int, base, scale float32, offset int, eps float32) (*attentionBlockICBScratch, error) {
	x, err := newPinnedNoCopyBytes(dModel * bf16Size)
	if err != nil {
		return nil, err
	}
	k, err := newPinnedNoCopyBytes(nKVHeads * kvLen * headDim * bf16Size)
	if err != nil {
		x.Close()
		return nil, err
	}
	v, err := newPinnedNoCopyBytes(nKVHeads * kvLen * headDim * bf16Size)
	if err != nil {
		x.Close()
		k.Close()
		return nil, err
	}
	out, err := newPinnedNoCopyBytes(dModel * bf16Size)
	if err != nil {
		x.Close()
		k.Close()
		v.Close()
		return nil, err
	}
	offBuf, offPtr, err := newICBI32Storage(int32(offset))
	if err != nil {
		x.Close()
		k.Close()
		v.Close()
		out.Close()
		return nil, err
	}
	epsBuf, epsPtr, err := newICBF32Storage(eps)
	if err != nil {
		x.Close()
		k.Close()
		v.Close()
		out.Close()
		return nil, err
	}
	ropeScaleBuf, ropeScalePtr, err := newICBF32Storage(scale)
	if err != nil {
		x.Close()
		k.Close()
		v.Close()
		out.Close()
		return nil, err
	}
	ropeBaseBuf, ropeBasePtr, err := newICBF32Storage(float32(math.Log2(float64(base))))
	if err != nil {
		x.Close()
		k.Close()
		v.Close()
		out.Close()
		return nil, err
	}
	sdpaScaleBuf, sdpaScalePtr, err := newICBF32Storage(scale)
	if err != nil {
		x.Close()
		k.Close()
		v.Close()
		out.Close()
		return nil, err
	}
	icbDesc := metal.NewMTLIndirectCommandBufferDescriptor()
	icbDesc.SetCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch | metal.MTLIndirectCommandTypeConcurrentDispatchThreads)
	icbDesc.SetInheritBuffers(false)
	icbDesc.SetInheritPipelineState(false)
	icbDesc.SetMaxKernelBufferBindCount(16)
	return &attentionBlockICBScratch{
		dModel: dModel, qDim: qDim, nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim, kvLen: kvLen,
		x: x, k: k, v: v, out: out,
		normed: scratchBF16(dModel), q: scratchBF16(qDim), qr: scratchBF16(qDim), attn: scratchBF16(qDim), attnOut: scratchBF16(dModel),
		offBuf: offBuf, epsBuf: epsBuf, ropeScaleBuf: ropeScaleBuf, ropeBaseBuf: ropeBaseBuf, sdpaScaleBuf: sdpaScaleBuf,
		axisBuf: scalarI32(int32(dModel)), wsBuf: scalarI32(1),
		qInBuf: scalarI32(int32(dModel)), qOutBuf: scalarI32(int32(qDim)), qLdBuf: scalarI32(int32(dModel)),
		oInBuf: scalarI32(int32(qDim)), oOutBuf: scalarI32(int32(dModel)), oLdBuf: scalarI32(int32(qDim)),
		bndBuf: scalarI32(1), bshBuf: scalarI32(1), vsBuf: scalarI64(0), msBuf: scalarI64(0),
		ropeMatBuf: scalarI64(int64(headDim)), gqaBuf: scalarI32(int32(nHeads / nKVHeads)), nBuf: scalarI32(int32(kvLen)),
		khsBuf: scalarI64(int64(kvLen * headDim)), kssBuf: scalarI64(int64(headDim)),
		vhsBuf: scalarI64(int64(kvLen * headDim)), vssBuf: scalarI64(int64(headDim)), addCntBuf: scalarI32(int32(dModel)),
		offPtr: offPtr, epsPtr: epsPtr, ropeScalePtr: ropeScalePtr, ropeBasePtr: ropeBasePtr, sdpaScalePtr: sdpaScalePtr,
		icbDesc:     icbDesc,
		rng:         foundation.NSRange{Location: 0, Length: 6},
		residentRes: make([]metal.MTLResource, 0, 37),
	}, nil
}

func (s *attentionBlockICBScratch) matches(dModel, qDim, nHeads, nKVHeads, headDim, kvLen int) bool {
	return s != nil &&
		s.dModel == dModel && s.qDim == qDim && s.nHeads == nHeads && s.nKVHeads == nKVHeads && s.headDim == headDim && s.kvLen == kvLen &&
		s.x != nil && s.k != nil && s.v != nil && s.out != nil &&
		s.normed != nil && s.q != nil && s.qr != nil && s.attn != nil && s.attnOut != nil &&
		s.offBuf != nil && s.epsBuf != nil && s.ropeScaleBuf != nil && s.ropeBaseBuf != nil && s.sdpaScaleBuf != nil &&
		s.offPtr != nil && s.epsPtr != nil && s.ropeScalePtr != nil && s.ropeBasePtr != nil && s.sdpaScalePtr != nil
}

func attentionBlockICBScratchPoolFor(dModel, qDim, nHeads, nKVHeads, headDim, kvLen int) *attentionBlockICBScratchPool {
	key := attentionBlockICBScratchKey{dModel: dModel, qDim: qDim, nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim, kvLen: kvLen}
	if v, ok := attentionBlockICBScratchPools.Load(key); ok {
		return v.(*attentionBlockICBScratchPool)
	}
	pool := &attentionBlockICBScratchPool{}
	actual, _ := attentionBlockICBScratchPools.LoadOrStore(key, pool)
	return actual.(*attentionBlockICBScratchPool)
}

func getAttentionBlockICBScratch(dModel, qDim, nHeads, nKVHeads, headDim, kvLen int, base, scale float32, offset int, eps float32) (*attentionBlockICBScratch, error) {
	if s := attentionBlockICBScratchPoolFor(dModel, qDim, nHeads, nKVHeads, headDim, kvLen).Get(); s != nil {
		if s.matches(dModel, qDim, nHeads, nKVHeads, headDim, kvLen) {
			s.updateScalars(base, scale, offset, eps)
			return s, nil
		}
		s.Close()
	}
	return newAttentionBlockICBScratch(dModel, qDim, nHeads, nKVHeads, headDim, kvLen, base, scale, offset, eps)
}

func putAttentionBlockICBScratch(s *attentionBlockICBScratch) {
	if s != nil {
		attentionBlockICBScratchPoolFor(s.dModel, s.qDim, s.nHeads, s.nKVHeads, s.headDim, s.kvLen).Put(s)
	}
}

func (s *attentionBlockICBScratch) Close() {
	if s == nil {
		return
	}
	if s.x != nil {
		s.x.Close()
		s.x = nil
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
	s.xView.Close()
	s.kView.Close()
	s.vView.Close()
	s.normed, s.q, s.qr, s.attn, s.attnOut = nil, nil, nil, nil, nil
	s.icb = nil
	s.residentRes = nil
}

func (s *attentionBlockICBScratch) updateScalars(base, scale float32, offset int, eps float32) {
	*s.offPtr = int32(offset)
	*s.epsPtr = eps
	*s.ropeScalePtr = scale
	*s.ropeBasePtr = float32(math.Log2(float64(base)))
	*s.sdpaScalePtr = scale
}

func (s *attentionBlockICBScratch) buffers(x, kCache, vCache []byte) (metal.MTLBuffer, metal.MTLBuffer, metal.MTLBuffer, metal.MTLBuffer, error) {
	xBuf, ok := s.xView.buffer(x)
	if !ok {
		var err error
		xBuf, err = s.x.copyBuffer(x)
		if err != nil {
			return nil, nil, nil, nil, err
		}
	}
	kBuf, ok := s.kView.buffer(kCache)
	if !ok {
		var err error
		kBuf, err = s.k.copyBuffer(kCache)
		if err != nil {
			return nil, nil, nil, nil, err
		}
	}
	vBuf, ok := s.vView.buffer(vCache)
	if !ok {
		var err error
		vBuf, err = s.v.copyBuffer(vCache)
		if err != nil {
			return nil, nil, nil, nil, err
		}
	}
	return xBuf, kBuf, vBuf, s.out.buf, nil
}

func (s *attentionBlockICBScratch) record(
	rmsPSO, gemvQPSO, gemvOPSO, ropePSO, sdpaPSO, addPSO metal.MTLComputePipelineState,
	xBuf, kBuf, vBuf metal.MTLBuffer,
	nwBuf, wqBuf, woBuf metal.MTLBuffer,
	bmQ, bnQ, smQ, tmQ, bmO, bnO, smO, tmO int,
) {
	normID, wqID, woID := uintptr(nwBuf.GetID()), uintptr(wqBuf.GetID()), uintptr(woBuf.GetID())
	xID, kID, vID := uintptr(xBuf.GetID()), uintptr(kBuf.GetID()), uintptr(vBuf.GetID())
	if s.icb != nil && s.normID == normID && s.wqID == wqID && s.woID == woID && s.xID == xID && s.kID == kID && s.vID == vID {
		return
	}
	qDim := s.qDim
	if s.icb == nil {
		s.icb = device.NewIndirectCommandBufferWithDescriptorMaxCommandCountOptions(s.icbDesc, 6, metal.MTLResourceStorageModeShared)
	}

	epsBuf, axisBuf, wsBuf := s.epsBuf, s.axisBuf, s.wsBuf
	qInB, qOutB, qLdB := s.qInBuf, s.qOutBuf, s.qLdBuf
	oInB, oOutB, oLdB := s.oInBuf, s.oOutBuf, s.oLdBuf
	bndB, bshB, vsB, msB := s.bndBuf, s.bshBuf, s.vsBuf, s.msBuf
	ropeMatB := s.ropeMatBuf
	gqaB, nB := s.gqaBuf, s.nBuf
	khsB, kssB := s.khsBuf, s.kssBuf
	vhsB, vssB := s.vhsBuf, s.vssBuf
	addCntB := s.addCntBuf

	resident := s.residentRes[:0]
	resident = append(resident,
		xBuf, nwBuf, wqBuf, woBuf, kBuf, vBuf, s.normed, s.q, s.qr, s.attn, s.attnOut, s.out.buf,
		s.offBuf, epsBuf, axisBuf, wsBuf, qInB, qOutB, qLdB, oInB, oOutB, oLdB, bndB, bshB, vsB, msB,
		s.ropeScaleBuf, ropeMatB, s.ropeBaseBuf, gqaB, nB, khsB, kssB, vhsB, vssB, s.sdpaScaleBuf, addCntB,
	)
	s.residentRes = resident

	rmsTG := rmsThreadgroup(s.dModel, rmsPSO)
	gemvGrid := func(outDim, bm, sm, tm int) uint { return uint((outDim + bm*sm*tm - 1) / (bm * sm * tm)) }

	c := indirectComputeCommandAtIndexFast(s.icb, 0)
	setICBPSO(c, rmsPSO)
	setICBKernelBuffer(c, xBuf, 0, 0)
	setICBKernelBuffer(c, nwBuf, 0, 1)
	setICBKernelBuffer(c, s.normed, 0, 2)
	setICBKernelBuffer(c, epsBuf, 0, 3)
	setICBKernelBuffer(c, axisBuf, 0, 4)
	setICBKernelBuffer(c, wsBuf, 0, 5)
	concurrentDispatchThreads(c, metal.MTLSize{Width: rmsTG, Height: 1, Depth: 1}, metal.MTLSize{Width: rmsTG, Height: 1, Depth: 1})

	c = indirectComputeCommandAtIndexFast(s.icb, 1)
	setICBBarrier(c)
	setICBPSO(c, gemvQPSO)
	setICBKernelBuffer(c, wqBuf, 0, 0)
	setICBKernelBuffer(c, s.normed, 0, 1)
	setICBKernelBuffer(c, s.q, 0, 3)
	setICBKernelBuffer(c, qInB, 0, 4)
	setICBKernelBuffer(c, qOutB, 0, 5)
	setICBKernelBuffer(c, qLdB, 0, 6)
	setICBKernelBuffer(c, bndB, 0, 9)
	setICBKernelBuffer(c, bshB, 0, 10)
	setICBKernelBuffer(c, vsB, 0, 11)
	setICBKernelBuffer(c, msB, 0, 12)
	concurrentDispatchThreadgroups(c, metal.MTLSize{Width: gemvGrid(qDim, bmQ, smQ, tmQ), Height: 1, Depth: 1}, metal.MTLSize{Width: 32, Height: uint(bnQ), Depth: uint(bmQ)})

	c = indirectComputeCommandAtIndexFast(s.icb, 2)
	setICBBarrier(c)
	setICBPSO(c, ropePSO)
	setICBKernelBuffer(c, s.q, 0, 0)
	setICBKernelBuffer(c, s.qr, 0, 1)
	setICBKernelBuffer(c, s.offBuf, 0, 2)
	setICBKernelBuffer(c, s.ropeScaleBuf, 0, 3)
	setICBKernelBuffer(c, ropeMatB, 0, 4)
	setICBKernelBuffer(c, s.ropeBaseBuf, 0, 10)
	ropeDim0 := uint(s.headDim / 2)
	concurrentDispatchThreads(c, metal.MTLSize{Width: ropeDim0, Height: uint(s.nHeads), Depth: 1}, metal.MTLSize{Width: ropeDim0, Height: 1, Depth: 1})

	c = indirectComputeCommandAtIndexFast(s.icb, 3)
	setICBBarrier(c)
	setICBPSO(c, sdpaPSO)
	setICBKernelBuffer(c, s.qr, 0, 0)
	setICBKernelBuffer(c, kBuf, 0, 1)
	setICBKernelBuffer(c, vBuf, 0, 2)
	setICBKernelBuffer(c, s.attn, 0, 3)
	setICBKernelBuffer(c, gqaB, 0, 4)
	setICBKernelBuffer(c, nB, 0, 5)
	setICBKernelBuffer(c, khsB, 0, 6)
	setICBKernelBuffer(c, kssB, 0, 7)
	setICBKernelBuffer(c, vhsB, 0, 8)
	setICBKernelBuffer(c, vssB, 0, 9)
	setICBKernelBuffer(c, s.sdpaScaleBuf, 0, 10)
	concurrentDispatchThreadgroups(c, metal.MTLSize{Width: uint(s.nHeads), Height: 1, Depth: 1}, metal.MTLSize{Width: 1024, Height: 1, Depth: 1})

	c = indirectComputeCommandAtIndexFast(s.icb, 4)
	setICBBarrier(c)
	setICBPSO(c, gemvOPSO)
	setICBKernelBuffer(c, woBuf, 0, 0)
	setICBKernelBuffer(c, s.attn, 0, 1)
	setICBKernelBuffer(c, s.attnOut, 0, 3)
	setICBKernelBuffer(c, oInB, 0, 4)
	setICBKernelBuffer(c, oOutB, 0, 5)
	setICBKernelBuffer(c, oLdB, 0, 6)
	setICBKernelBuffer(c, bndB, 0, 9)
	setICBKernelBuffer(c, bshB, 0, 10)
	setICBKernelBuffer(c, vsB, 0, 11)
	setICBKernelBuffer(c, msB, 0, 12)
	concurrentDispatchThreadgroups(c, metal.MTLSize{Width: gemvGrid(s.dModel, bmO, smO, tmO), Height: 1, Depth: 1}, metal.MTLSize{Width: 32, Height: uint(bnO), Depth: uint(bmO)})

	c = indirectComputeCommandAtIndexFast(s.icb, 5)
	setICBBarrier(c)
	setICBPSO(c, addPSO)
	setICBKernelBuffer(c, xBuf, 0, 0)
	setICBKernelBuffer(c, s.attnOut, 0, 1)
	setICBKernelBuffer(c, s.out.buf, 0, 2)
	setICBKernelBuffer(c, addCntB, 0, 3)
	addGroup := min(uint(s.dModel), uint(256))
	concurrentDispatchThreads(c, metal.MTLSize{Width: uint(s.dModel), Height: 1, Depth: 1}, metal.MTLSize{Width: addGroup, Height: 1, Depth: 1})

	s.normID, s.wqID, s.woID = normID, wqID, woID
	s.xID, s.kID, s.vID = xID, kID, vID
}

// AttentionBlockICB records the bf16 attention block once into an ICB and replays
// it `replays` times — proving ICB replay across a real func-const multi-op chain
// (rms→gemv→rope→sdpa→gemv→add), every scalar a buffer, a barrier on each
// consumer, residency on every buffer. With replays=1 it must equal AttentionBlock
// byte-for-byte. Inputs/outputs raw bf16 bytes; same shapes as AttentionBlock.
func AttentionBlockICB(x, normWeight, wQ, wO, kCache, vCache []byte, dModel, nHeads, nKVHeads, headDim, kvLen int, base, scale float32, offset int, eps float32, replays int) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if replays < 1 {
		replays = 1
	}
	qDim := nHeads * headDim

	// axis-selected rms: the single-row kernel over-caps its threadgroup past 4096 dims
	// and Metal drops the dispatch silently (the #348 class); the looped kernel covers any.
	rmsPSO, err := pipelineForICB(rmsKernelBF16(dModel))
	if err != nil {
		return nil, err
	}
	bmQ, bnQ, smQ, snQ, tmQ, tnQ := gemvTiles(dModel, qDim)
	gemvQPSO, err := pipelineForICB(gemvKernelName("bfloat16", bmQ, bnQ, smQ, snQ, tmQ, tnQ))
	if err != nil {
		return nil, err
	}
	bmO, bnO, smO, snO, tmO, tnO := gemvTiles(qDim, dModel)
	gemvOPSO, err := pipelineForICB(gemvKernelName("bfloat16", bmO, bnO, smO, snO, tmO, tnO))
	if err != nil {
		return nil, err
	}
	ropePSO, err := ropePipelineICB(false)
	if err != nil {
		return nil, err
	}
	sdpaPSO, err := sdpaVectorPipelineICBForHeadDim(headDim)
	if err != nil {
		return nil, err
	}
	addPSO, err := pipelineForICB("vv_Addbfloat16")
	if err != nil {
		return nil, err
	}

	out := make([]byte, dModel*bf16Size)
	var encErr error
	withAutoreleasePool(func() {
		sc, err := getAttentionBlockICBScratch(dModel, qDim, nHeads, nKVHeads, headDim, kvLen, base, scale, offset, eps)
		if err != nil {
			encErr = err
			return
		}
		defer putAttentionBlockICBScratch(sc)
		xBuf, kBuf, vBuf, _, err := sc.buffers(x, kCache, vCache)
		if err != nil {
			encErr = err
			return
		}
		nwBuf := residentBytes(normWeight)
		wqBuf, woBuf := residentBytes(wQ), residentBytes(wO)
		sc.record(rmsPSO, gemvQPSO, gemvOPSO, ropePSO, sdpaPSO, addPSO, xBuf, kBuf, vBuf, nwBuf, wqBuf, woBuf, bmQ, bnQ, smQ, tmQ, bmO, bnO, smO, tmO)
		sc.residentIDs = resourceIDsForFastUse(sc.residentIDs, sc.residentRes)
		residentIDs := sc.residentIDs
		for r := 0; r < replays; r++ {
			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			useResourcesIDsFast(enc, sc.residentRes, residentIDs, metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
			executeCommandsInBufferWithRangeFast(enc, sc.icb, sc.rng)
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
		}
		copy(out, sc.out.bytes[:len(out)])
	})
	if encErr != nil {
		return nil, encErr
	}
	return out, nil
}

// scalar{I32,I64,F32} return a shared device buffer holding one immutable constant, MEMOISED by value
// for the process: a buffer holding "5" (a count, an axis size, an eps) is valid for any model, so the
// ICB recorder reuses one across every op + every re-record instead of minting a fresh buffer each time
// (the recorder binds these read-only; the per-token-VARYING buffers — N, sliding offset — use their own
// rebindable buffers, never these). This is also what lets the dispatchSink's icbSink bind a scalar as a
// buffer with zero per-record allocation. A few dozen tiny buffers leak for the process lifetime — nil.
var (
	scalarBufMu  sync.Mutex
	scalarI32Buf = map[int32]metal.MTLBuffer{}
	scalarI64Buf = map[int64]metal.MTLBuffer{}
	scalarF32Buf = map[float32]metal.MTLBuffer{}
)

func scalarI32(v int32) metal.MTLBuffer {
	scalarBufMu.Lock()
	if b, ok := scalarI32Buf[v]; ok {
		scalarBufMu.Unlock()
		return b
	}
	b := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&v), 4, metal.MTLResourceStorageModeShared)
	scalarI32Buf[v] = b
	scalarBufMu.Unlock()
	return b
}

func scalarI64(v int64) metal.MTLBuffer {
	scalarBufMu.Lock()
	if b, ok := scalarI64Buf[v]; ok {
		scalarBufMu.Unlock()
		return b
	}
	b := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&v), 8, metal.MTLResourceStorageModeShared)
	scalarI64Buf[v] = b
	scalarBufMu.Unlock()
	return b
}

func scalarF32(v float32) metal.MTLBuffer {
	scalarBufMu.Lock()
	if b, ok := scalarF32Buf[v]; ok {
		scalarBufMu.Unlock()
		return b
	}
	b := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&v), 4, metal.MTLResourceStorageModeShared)
	scalarF32Buf[v] = b
	scalarBufMu.Unlock()
	return b
}

// NormProjectICB computes the same rms→projection as NormProject, but records the
// two ops once into an ICB and replays it `replays` times (the decode-loop
// pattern). Returns the output of the final replay. With replays=1 it must equal
// NormProject byte-for-byte — same kernels, same data, only the submission path
// differs. Proves ICB replay on a real dependent sequence (scalar-as-buffer +
// SetBarrier + residency). float32.
func NormProjectICB(x, normWeight, projWeight []float32, dIn, dOut int, eps float32, replays int) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != dIn || len(normWeight) != dIn || len(projWeight) != dOut*dIn {
		return nil, core.NewError("native.NormProjectICB: size mismatch")
	}
	if replays < 1 {
		replays = 1
	}

	rmsName := "rmsfloat32"
	if dIn > rmsLoopedLimit {
		rmsName = "rms_loopedfloat32" // single-row over-caps its threadgroup past 4096 dims
	}
	rmsPSO, err := pipelineForICB(rmsName)
	if err != nil {
		return nil, err
	}
	bm, bn, sm, sn, tm, tn := gemvTiles(dIn, dOut)
	gemvPSO, err := pipelineForICB(gemvKernelName("float32", bm, bn, sm, sn, tm, tn))
	if err != nil {
		return nil, err
	}

	out := make([]float32, dOut)
	withAutoreleasePool(func() {
		// persistent data buffers
		xBuf := shared(x)
		nwBuf := residentFloat32(normWeight)
		pwBuf := residentFloat32(projWeight)
		tmpBuf, outBuf := scratch(dIn), scratch(dOut)
		// scalar params as buffers (ICB can't setBytes inline)
		epsBuf, axisBuf, wsBuf := scalarF32(eps), scalarI32(int32(dIn)), scalarI32(1)
		inBuf, outdimBuf, ldBuf := scalarI32(int32(dIn)), scalarI32(int32(dOut)), scalarI32(int32(dIn))
		bndBuf, bshBuf, vsBuf, msBuf := scalarI32(1), scalarI32(1), scalarI64(0), scalarI64(0)
		resident := []metal.MTLResource{xBuf, nwBuf, pwBuf, tmpBuf, outBuf, epsBuf, axisBuf, wsBuf, inBuf, outdimBuf, ldBuf, bndBuf, bshBuf, vsBuf, msBuf}
		var residentIDBuf [15]objc.ID
		residentIDs := resourceIDsForFastUse(residentIDBuf[:0], resident)

		// record the 2-op sequence once
		icbDesc := metal.NewMTLIndirectCommandBufferDescriptor()
		icbDesc.SetCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch | metal.MTLIndirectCommandTypeConcurrentDispatchThreads)
		icbDesc.SetInheritBuffers(false)
		icbDesc.SetInheritPipelineState(false)
		icbDesc.SetMaxKernelBufferBindCount(16) // gemv binds up to index 12
		icb := device.NewIndirectCommandBufferWithDescriptorMaxCommandCountOptions(icbDesc, 2, metal.MTLResourceStorageModeShared)

		// cmd 0: rmsnorm  x -> tmp
		c0 := indirectComputeCommandAtIndexFast(icb, 0)
		emitRMSNorm(fastICBSink{c0}, rmsPSO, xBuf, nwBuf, tmpBuf, 0, dIn, eps, rmsThreadgroup(dIn, rmsPSO))

		// cmd 1: gemv  projW @ tmp -> out
		c1 := indirectComputeCommandAtIndexFast(icb, 1)
		setICBBarrier(c1) // wait for c0's tmp write to be visible before reading it
		emitGemv(fastICBSink{c1}, gemvPSO, pwBuf, 0, tmpBuf, outBuf, 0, dIn, dOut, bm, bn, sm, tm)

		// replay the recorded sequence
		rng := foundation.NSRange{Location: 0, Length: 2}
		for r := 0; r < replays; r++ {
			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			useResourcesIDsFast(enc, resident, residentIDs, metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
			executeCommandsInBufferWithRangeFast(enc, icb, rng)
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
		}
		copy(out, unsafe.Slice((*float32)(outBuf.Contents()), dOut))
	})
	return out, nil
}
