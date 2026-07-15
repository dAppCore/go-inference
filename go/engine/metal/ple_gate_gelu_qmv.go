// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// ple_gate_gelu_qmv.go — the fused PLE gate+gelu dispatch (#373 stage-count
// lever): one op where the decode recorded two barriered stages (gate qmv →
// gelu·pli). The kernel is qmv_fast_impl verbatim with the gelu·pli product
// applied at the store, so its output bytes equal the composed pair's exactly
// — the ICB and the re-encode path stay byte-equal without a re-encode twin.

var (
	pleGateGeluPSOMu       sync.Mutex
	pleGateGeluPSOCache    = map[pleGateGeluKey]metal.MTLComputePipelineState{}
	pleGateGeluICBPSOMu    sync.Mutex
	pleGateGeluICBPSOCache = map[pleGateGeluKey]metal.MTLComputePipelineState{}
)

type pleGateGeluKey struct {
	groupSize, bits int
}

func pleGateGeluKernelName(groupSize, bits int) string {
	return core.Sprintf("lthn_ple_gate_gelu_qmv_bfloat16_t_gs_%d_b_%d", groupSize, bits)
}

// pleGateGeluPipeline builds (and caches) the fused PLE gate+gelu pipeline —
// the encoder-path variant (the re-encode twin and tests drive it).
func pleGateGeluPipeline(groupSize, bits int) (metal.MTLComputePipelineState, error) {
	key := pleGateGeluKey{groupSize: groupSize, bits: bits}
	pleGateGeluPSOMu.Lock()
	defer pleGateGeluPSOMu.Unlock()
	if pso, ok := pleGateGeluPSOCache[key]; ok {
		return pso, nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		return nil, core.NewError("native.pleGateGeluPipeline: custom library unavailable")
	}
	name := pleGateGeluKernelName(groupSize, bits)
	fn := customLibrary.NewFunctionWithName(name)
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.pleGateGeluPipeline: kernel " + name + " not found")
	}
	pso, err := device.NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		return nil, core.E("native.pleGateGeluPipeline", name, err)
	}
	pleGateGeluPSOCache[key] = pso
	return pso, nil
}

// pleGateGeluPipelineICB is pleGateGeluPipeline with indirect-command-buffer
// support — the variant the decode ICB records and replays per token.
func pleGateGeluPipelineICB(groupSize, bits int) (metal.MTLComputePipelineState, error) {
	key := pleGateGeluKey{groupSize: groupSize, bits: bits}
	pleGateGeluICBPSOMu.Lock()
	defer pleGateGeluICBPSOMu.Unlock()
	if pso, ok := pleGateGeluICBPSOCache[key]; ok {
		return pso, nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		return nil, core.NewError("native.pleGateGeluPipelineICB: custom library unavailable")
	}
	name := pleGateGeluKernelName(groupSize, bits)
	fn := customLibrary.NewFunctionWithName(name)
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.pleGateGeluPipelineICB: kernel " + name + " not found")
	}
	desc := metal.NewMTLComputePipelineDescriptor()
	desc.SetComputeFunction(fn)
	desc.SetSupportIndirectCommandBuffers(true)
	pso, err := device.NewComputePipelineStateWithDescriptorOptionsReflectionError(desc, 0, nil)
	if err != nil {
		return nil, core.E("native.pleGateGeluPipelineICB", name, err)
	}
	pleGateGeluICBPSOCache[key] = pso
	return pso, nil
}

// emitPLEGateGelu records the fused PLE gate+gelu op through any sink: w=0,
// scales=1, biases=2, x=3, pli=4 (bound at the layer's slab offset), y=5,
// K=6, N=7 — the qmv_fast grid ((outDim+7)/8 threadgroups of 32×2).
func emitPLEGateGelu[S dispatchSink](sink S, pso metal.MTLComputePipelineState, w qmvWeight, x metal.MTLBuffer, pli metal.MTLBuffer, pliOff uint, y metal.MTLBuffer, inDim, outDim int) {
	sink.setPSO(pso)
	sink.setBuf(w.wq.buf, w.wq.off, 0)
	sink.setBuf(w.scales.buf, w.scales.off, 1)
	sink.setBuf(w.biases.buf, w.biases.off, 2)
	sink.setBuf(x, 0, 3)
	sink.setBuf(pli, pliOff, 4)
	sink.setBuf(y, 0, 5)
	sink.setI32(int32(inDim), 6)
	sink.setI32(int32(outDim), 7)
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: 1, Height: uint((outDim + 7) / 8), Depth: 1},
		metal.MTLSize{Width: 32, Height: 2, Depth: 1},
	)
}

// encPLEGateGeluQMV is the encoder-path form — the re-encode twin and the
// byte-parity gate drive it.
func encPLEGateGeluQMV(enc metal.MTLComputeCommandEncoder, w qmvWeight, x metal.MTLBuffer, pli metal.MTLBuffer, pliOff uint, y metal.MTLBuffer, inDim, outDim, gs, bits int) error {
	pso, err := pleGateGeluPipeline(gs, bits)
	if err != nil {
		return err
	}
	emitPLEGateGelu(encSink{enc}, pso, w, x, pli, pliOff, y, inDim, outDim)
	return nil
}
