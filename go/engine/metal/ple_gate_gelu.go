// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	"github.com/tmc/apple/metal"
)

// ple_gate_gelu.go — the fused PLE gate+gelu·slab dispatch and the K-row
// rms-residual tail: together they take the batched epilogue's PLE chain from
// five dispatches per layer to three (gate+gelu → proj qmm_t → rms+add). At
// MTP-verify K the chain is launch-bound (#372: the resid+epilogue gpu-trace
// bucket held a fixed 3.8ms at K=5 and K=33), so the win is launch count.
// Both pieces are ordinary hazard-tracked dispatches — the grid-barrier
// megakernel variant measured broken threadgroup co-residency and is banked
// as a negative result on #372.

var (
	lthnPLEGateGeluPSOMu       sync.Mutex
	lthnPLEGateGeluPSOCache    = map[int]metal.MTLComputePipelineState{} // bits -> PSO
	lthnPLEGateGeluMissing     = map[int]bool{}
	lthnPLEGateGeluICBPSOCache = map[int]metal.MTLComputePipelineState{} // bits -> ICB PSO (nil = miss)
	lthnPLEGateGeluICBProbed   = map[int]bool{}
)

// lthnPLEGateGeluPipeline resolves (and caches, including failures) the fused
// gate+gelu kernel with the quant width baked as function constant 0.
func lthnPLEGateGeluPipeline(bits int) (metal.MTLComputePipelineState, bool) {
	lthnPLEGateGeluPSOMu.Lock()
	defer lthnPLEGateGeluPSOMu.Unlock()
	if pso, ok := lthnPLEGateGeluPSOCache[bits]; ok {
		return pso, true
	}
	if lthnPLEGateGeluMissing[bits] {
		return nil, false
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		lthnPLEGateGeluMissing[bits] = true
		return nil, false
	}
	fc := metal.NewMTLFunctionConstantValues()
	b := uint32(bits)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&b), metal.MTLDataTypeUInt, 0)
	fn, err := customLibrary.NewFunctionWithNameConstantValuesError("lthn_ple_gate_gelu_rows", fc)
	if err != nil || fn == nil || fn.GetID() == 0 {
		lthnPLEGateGeluMissing[bits] = true
		return nil, false
	}
	pso, perr := device.NewComputePipelineStateWithFunctionError(fn)
	if perr != nil {
		lthnPLEGateGeluMissing[bits] = true
		return nil, false
	}
	lthnPLEGateGeluPSOCache[bits] = pso
	return pso, true
}

// lthnPLEGateGeluPipelineICB is lthnPLEGateGeluPipeline with
// supportIndirectCommandBuffers set — the variant the MTP verify-tail ICB
// records. Same bits function constant, same failure caching.
func lthnPLEGateGeluPipelineICB(bits int) (metal.MTLComputePipelineState, bool) {
	lthnPLEGateGeluPSOMu.Lock()
	defer lthnPLEGateGeluPSOMu.Unlock()
	if lthnPLEGateGeluICBProbed[bits] {
		pso := lthnPLEGateGeluICBPSOCache[bits]
		return pso, pso != nil
	}
	lthnPLEGateGeluICBProbed[bits] = true
	if customLibrary == nil || customLibrary.GetID() == 0 {
		return nil, false
	}
	fc := metal.NewMTLFunctionConstantValues()
	b := uint32(bits)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&b), metal.MTLDataTypeUInt, 0)
	fn, err := customLibrary.NewFunctionWithNameConstantValuesError("lthn_ple_gate_gelu_rows", fc)
	if err != nil || fn == nil || fn.GetID() == 0 {
		return nil, false
	}
	desc := metal.NewMTLComputePipelineDescriptor()
	desc.SetComputeFunction(fn)
	desc.SetSupportIndirectCommandBuffers(true)
	pso, perr := device.NewComputePipelineStateWithDescriptorOptionsReflectionError(desc, 0, nil)
	if perr != nil {
		return nil, false
	}
	lthnPLEGateGeluICBPSOCache[bits] = pso
	return pso, true
}

// pleGateGeluRowsGeometryOK is the shared geometry gate for the fused rows
// kernel — the encode wrapper and the verify-tail recorder must agree on it.
func pleGateGeluRowsGeometryOK(rows, dModel, pliDim, gs, bits int) bool {
	return rows > 0 && dModel > 0 && pliDim > 0 && gs > 0 &&
		dModel%gs == 0 && (dModel*bits)%8 == 0
}

// emitPLEGateGeluRows records the fused rows dispatch through any sink —
// binding ABI w=0, scales=1, biases=2, x=3, ple=4, gated=5, dModel=6, pliDim=7,
// rows=8, gs=9 on the qmv-fast grid. One body behind encPLEGateGeluRows (live)
// and the verify-tail ICB recorder; pso caller-provided (ICB variant differs).
func emitPLEGateGeluRows[S dispatchSink](sink S, pso metal.MTLComputePipelineState,
	gatePacked, gateScales, gateBiases bufView,
	x metal.MTLBuffer, xOff uint,
	ple metal.MTLBuffer, pleOff uint,
	gated metal.MTLBuffer, gatedOff uint,
	rows, dModel, pliDim, gs int) {
	sink.setPSO(pso)
	sink.setBuf(gatePacked.buf, gatePacked.off, 0)
	sink.setBuf(gateScales.buf, gateScales.off, 1)
	sink.setBuf(gateBiases.buf, gateBiases.off, 2)
	sink.setBuf(x, xOff, 3)
	sink.setBuf(ple, pleOff, 4)
	sink.setBuf(gated, gatedOff, 5)
	sink.setI32(int32(dModel), 6)
	sink.setI32(int32(pliDim), 7)
	sink.setI32(int32(rows), 8)
	sink.setI32(int32(gs), 9)
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: 1, Height: uint((rows*pliDim + 1) / 2), Depth: 1},
		metal.MTLSize{Width: 32, Height: 2, Depth: 1},
	)
}

// encPLEGateGeluRows encodes gated[r,o] = bf16(gelu(bf16(gate(x[r]))_o) ·
// ple[r,o]) for all K rows in one dispatch, reporting handled=false (nothing
// encoded) when the kernel or geometry is unavailable so the caller keeps the
// composed gate-qmm_t + gelu pair. x rows are contiguous [rows × dModel] bf16
// at xOff; ple and gated are contiguous [rows × pliDim] bf16 at their offsets.
func encPLEGateGeluRows(enc metal.MTLComputeCommandEncoder,
	gatePacked, gateScales, gateBiases bufView,
	x metal.MTLBuffer, xOff uint,
	ple metal.MTLBuffer, pleOff uint,
	gated metal.MTLBuffer, gatedOff uint,
	rows, dModel, pliDim, gs, bits int) (bool, error) {
	if !pleGateGeluRowsGeometryOK(rows, dModel, pliDim, gs, bits) {
		return false, nil
	}
	pso, ok := lthnPLEGateGeluPipeline(bits)
	if !ok {
		return false, nil
	}
	emitPLEGateGeluRows(encSink{enc}, pso, gatePacked, gateScales, gateBiases, x, xOff, ple, pleOff, gated, gatedOff, rows, dModel, pliDim, gs)
	return true, nil
}
