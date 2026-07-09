// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// lthn_sdpa_multiq_bf16_<D> — the batched pass's multi-query causal SDPA (kernels/
// lthn_sdpa_multiq.metal): K causal query rows in ONE dispatch, grid (nHeads, K), each
// (head, query) threadgroup running MLX's sdpa_vector loop unchanged with the per-query
// length cap computed in-kernel (key i used iff i <= N-K+s). Byte-identical per row to K
// single-query dispatches; valid only when the batch's K/V rows landed in the same cache
// buffer with no ring eviction inside the batch (the fold's direct case).

// sdpaMultiQDisabledForTest forces the fold back onto per-row SDPA dispatches — the A/B
// lever for the multi-query kernel's parity/engagement tests. Production never sets it.
var sdpaMultiQDisabledForTest bool

var (
	sdpaMultiQMu       sync.Mutex
	sdpaMultiQPSOCache = map[int]metal.MTLComputePipelineState{}
	sdpaMultiQMissing  = map[int]bool{}
)

// sdpaMultiQPipelineForHeadDim resolves (and caches) the multi-query causal SDPA pipeline
// for a head dim. The kernel is instantiated for the gemma head geometries {64,128,256,512};
// anything else (or a missing custom metallib) reports unavailable and the caller keeps the
// per-row SDPA path.
func sdpaMultiQPipelineForHeadDim(headDim int) (metal.MTLComputePipelineState, bool) {
	sdpaMultiQMu.Lock()
	defer sdpaMultiQMu.Unlock()
	if pso, ok := sdpaMultiQPSOCache[headDim]; ok {
		return pso, true
	}
	if sdpaMultiQMissing[headDim] {
		return nil, false
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		sdpaMultiQMissing[headDim] = true
		return nil, false
	}
	fn := customLibrary.NewFunctionWithName(core.Sprintf("lthn_sdpa_multiq_bf16_%d", headDim))
	if fn == nil || fn.GetID() == 0 {
		sdpaMultiQMissing[headDim] = true
		return nil, false
	}
	pso, err := device.NewComputePipelineStateWithFunctionError(fn)
	if err != nil || pso == nil || pso.GetID() == 0 {
		sdpaMultiQMissing[headDim] = true
		return nil, false
	}
	sdpaMultiQPSOCache[headDim] = pso
	return pso, true
}

// gpuHasSDPAMultiQ reports whether the multi-query causal SDPA kernel is loadable for a
// head dim — the fold's gate alongside the no-evict and 2-pass-knee conditions.
func gpuHasSDPAMultiQ(headDim int) bool {
	_, ok := sdpaMultiQPipelineForHeadDim(headDim)
	return ok
}

// encSDPAMultiQCausal encodes the K-query causal SDPA: query row s (query-major slab, rows of
// nHeads·headDim) attends keys [0 .. nTotal-K+s] of the cache, out row s lands query-major in
// the attention slab. nTotal is the live length INCLUDING this batch (nBase+K). The caller
// guarantees no ring eviction inside the batch and nTotal below the 2-pass knee, so every row
// matches the single-query kernel byte for byte.
func encSDPAMultiQCausal(enc metal.MTLComputeCommandEncoder, q, k, v, out metal.MTLBuffer, nHeads, nKVHeads, headDim, kRows, nTotal int, kHeadStride, kSeqStride, vHeadStride, vSeqStride int64, scale float32) error {
	pso, ok := sdpaMultiQPipelineForHeadDim(headDim)
	if !ok {
		return core.NewError("native.encSDPAMultiQCausal: kernel unavailable for headDim")
	}
	sink := encSink{enc}
	sink.setPSO(pso)
	sink.setBuf(q, 0, 0)
	sink.setBuf(k, 0, 1)
	sink.setBuf(v, 0, 2)
	sink.setBuf(out, 0, 3)
	sink.setI32(int32(nHeads/nKVHeads), 4) // gqa_factor
	sink.setI32(int32(nTotal), 5)
	sink.setI64(kHeadStride, 6)
	sink.setI64(kSeqStride, 7)
	sink.setI64(vHeadStride, 8)
	sink.setI64(vSeqStride, 9)
	sink.setF32(scale, 10)
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: uint(nHeads), Height: uint(kRows), Depth: 1},
		metal.MTLSize{Width: 1024, Height: 1, Depth: 1},
	)
	return nil
}
