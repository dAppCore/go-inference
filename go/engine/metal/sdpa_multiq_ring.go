// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// The staged sliding tail's batched lane (kernels/lthn_sdpa_multiq_ring.metal): when a big batch
// evicts through a FULL ring, the per-row landing+SDPA interleave is replaced by — per layer —
// staged K/V (roped/normed in the stage), ONE two-segment multi-query SDPA reading the pre-batch
// ring (minus each query's evicted run) plus the staged causal rows, and a deferred bulk landing
// (two contiguous-run copies) after every layer has read the pre-batch state. Shared-KV layers
// read the owner's TRUE pre-batch window this way — sequential semantics the per-row tail could
// not give them once the owner had landed. Token-identity lane (fp accumulation order differs),
// engaged only at steelGEMMMinRows and a full ring; below that the byte-identical per-row
// interleave stays.

// stagedRingDisabledForTest forces the staged tail back onto the per-row landing+SDPA interleave
// — the A/B lever for the ring lane's closeness/engagement tests.
var stagedRingDisabledForTest bool

// stagedRingDispatchesForTest counts ring-SDPA dispatches while pieceTimingOn — the engagement
// receipt for the staged batched lane.
var stagedRingDispatchesForTest int64

var (
	sdpaRingMu       sync.Mutex
	sdpaRingPSOCache = map[int]metal.MTLComputePipelineState{}
	sdpaRingMissing  = map[int]bool{}

	copyPSOOnce sync.Once
	copyPSO     metal.MTLComputePipelineState
	copyPSOErr  error
)

func sdpaMultiQRingPipelineForHeadDim(headDim int) (metal.MTLComputePipelineState, bool) {
	sdpaRingMu.Lock()
	defer sdpaRingMu.Unlock()
	if pso, ok := sdpaRingPSOCache[headDim]; ok {
		return pso, true
	}
	if sdpaRingMissing[headDim] {
		return nil, false
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		sdpaRingMissing[headDim] = true
		return nil, false
	}
	fn := customLibrary.NewFunctionWithName(core.Sprintf("lthn_sdpa_multiq_ring_bf16_%d", headDim))
	if fn == nil || fn.GetID() == 0 {
		sdpaRingMissing[headDim] = true
		return nil, false
	}
	pso, err := device.NewComputePipelineStateWithFunctionError(fn)
	if err != nil || pso == nil || pso.GetID() == 0 {
		sdpaRingMissing[headDim] = true
		return nil, false
	}
	sdpaRingPSOCache[headDim] = pso
	return pso, true
}

// gpuHasSDPAMultiQRing reports whether the two-segment ring SDPA kernel is loadable for a head
// dim — one of the staged batched lane's gates.
func gpuHasSDPAMultiQRing(headDim int) bool {
	_, ok := sdpaMultiQRingPipelineForHeadDim(headDim)
	return ok
}

func copyPipeline() (metal.MTLComputePipelineState, error) {
	copyPSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			copyPSOErr = core.NewError("native.copyPipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_copy_bf16")
		if fn == nil || fn.GetID() == 0 {
			copyPSOErr = core.NewError("native.copyPipeline: kernel lthn_copy_bf16 not found")
			return
		}
		copyPSO, copyPSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return copyPSO, copyPSOErr
}

// gpuHasCopyKernel reports whether the contiguous bf16 copy kernel is loadable — the deferred
// landing's transport.
func gpuHasCopyKernel() bool {
	pso, err := copyPipeline()
	return err == nil && pso != nil && pso.GetID() != 0
}

// encCopyBF16Contig encodes out[0..n) = in[0..n) (bf16 elements) at the given byte offsets — the
// deferred landing's contiguous-run copy. Per-element identity: landed bytes equal staged bytes.
func encCopyBF16Contig(enc metal.MTLComputeCommandEncoder, in, out metal.MTLBuffer, inOff, outOff uint, n int) error {
	pso, err := copyPipeline()
	if err != nil {
		return err
	}
	sink := encSink{enc}
	sink.setPSO(pso)
	sink.setBuf(in, inOff, 0)
	sink.setBuf(out, outOff, 1)
	sink.setI32(int32(n), 2)
	sink.dispatchThreads(
		metal.MTLSize{Width: uint(n), Height: 1, Depth: 1},
		metal.MTLSize{Width: uint(elemGroupTG(n)), Height: 1, Depth: 1},
	)
	return nil
}

// encSDPAMultiQRing encodes the two-segment multi-query SDPA: query row s (query-major slab)
// attends the live pre-batch ring rows minus its evicted run plus the staged window rows
// [max(0, s-slideW+1) .. s]; out rows land query-major. ringLive = min(basePos, slideW) — the
// kernel handles a partial or fresh ring AND a batch wider than the window, so a chunk may CROSS
// the ring wrap. The ring buffers must still hold the pre-batch state (the landing is deferred).
func encSDPAMultiQRing(enc metal.MTLComputeCommandEncoder, q, ringK, ringV, stageK, stageV, out metal.MTLBuffer, nHeads, nKVHeads, headDim, kRows, slideW, slotBase, ringLive int, kHeadStride, kSeqStride, vHeadStride, vSeqStride int64, scale float32) error {
	pso, ok := sdpaMultiQRingPipelineForHeadDim(headDim)
	if !ok {
		return core.NewError("native.encSDPAMultiQRing: kernel unavailable for headDim")
	}
	if pieceTimingOn {
		stagedRingDispatchesForTest++
	}
	sink := encSink{enc}
	sink.setPSO(pso)
	sink.setBuf(q, 0, 0)
	sink.setBuf(ringK, 0, 1)
	sink.setBuf(ringV, 0, 2)
	sink.setBuf(out, 0, 3)
	sink.setI32(int32(nHeads/nKVHeads), 4) // gqa_factor
	sink.setI32(int32(slideW), 5)
	sink.setI64(kHeadStride, 6)
	sink.setI64(kSeqStride, 7)
	sink.setI64(vHeadStride, 8)
	sink.setI64(vSeqStride, 9)
	sink.setF32(scale, 10)
	sink.setBuf(stageK, 0, 11)
	sink.setBuf(stageV, 0, 12)
	sink.setI32(int32(slotBase), 13)
	sink.setI32(int32(ringLive), 14)
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: uint(nHeads), Height: uint(kRows), Depth: 1},
		metal.MTLSize{Width: 1024, Height: 1, Depth: 1},
	)
	return nil
}
