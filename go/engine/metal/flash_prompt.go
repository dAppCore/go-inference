// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"sync"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// flash_prompt.go — the streaming-softmax prompt SDPA (#375): one dispatch per
// chunk-layer replaces the GEMM composition's three, and S never exists. K/V
// stream through threadgroup tiles shared by 16-query blocks (lthn_flash_prompt
// — the multiQ vector idioms with the device K/V re-reads lifted to shared
// tiles), so the composition's S round-trip — whose cache eviction taxed every
// neighbouring GEMM ~6× its own bandwidth (the #367 conviction) — is gone,
// and K traffic drops by the query-block width on top.

// flashPromptBQ is the query-tile width both instantiations run (4 simdgroups
// × 4 queries); the kernel derives it from its template constants — this
// mirror sizes the dispatch grid.
const flashPromptBQ = 16

// flashPromptEnabled routes the batched pass's deep-prompt global-layer SDPA
// onto the flash kernel instead of the GEMM composition (LTHN_FLASH_PROMPT=1
// — the A/B lever; both lanes are token-identity against the multiQ oracle).
// OPT-IN for now: the v1 kernel is output-correct (token-identical to the
// composition at 8K) but its scalar dot loop runs 1366 vs the composition's
// 3012 tok/s prefill — a full simd reduction per key per query cannot race
// steel GEMMs on QKᵀ. v2 is the simdgroup-MMA score/PV upgrade; the flip to
// default rides its receipt.
var flashPromptEnabled = os.Getenv("LTHN_FLASH_PROMPT") == "1"

var (
	flashPromptPSOMu    sync.Mutex
	flashPromptPSOCache = map[int]metal.MTLComputePipelineState{}
)

// flashPromptPipeline resolves (and caches, including failures) the flash
// prompt kernel for a head dim — instantiated at 256 and 512 (the gemma4
// global geometries; mlx's fused steel_attention caps at 128, which is why
// this kernel exists in the house library at all).
func flashPromptPipeline(hd int) (metal.MTLComputePipelineState, bool) {
	flashPromptPSOMu.Lock()
	defer flashPromptPSOMu.Unlock()
	if pso, ok := flashPromptPSOCache[hd]; ok {
		return pso, pso != nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		flashPromptPSOCache[hd] = nil
		return nil, false
	}
	fn := customLibrary.NewFunctionWithName(core.Sprintf("lthn_flash_prompt_bf16_%d", hd))
	if fn == nil || fn.GetID() == 0 {
		flashPromptPSOCache[hd] = nil
		return nil, false
	}
	pso, err := device.NewComputePipelineStateWithFunctionError(fn)
	if err != nil || pso == nil || pso.GetID() == 0 {
		flashPromptPSOCache[hd] = nil
		return nil, false
	}
	flashPromptPSOCache[hd] = pso
	return pso, true
}

// gpuHasFlashPrompt reports whether the flash prompt kernel exists for a head
// dim (256/512 shipped; anything else keeps the GEMM composition).
func gpuHasFlashPrompt(hd int) bool {
	_, ok := flashPromptPipeline(hd)
	return ok
}

// encFlashPromptSDPA encodes the whole chunk-layer prompt attention as ONE
// dispatch: kRows query rows (query-major slab, row stride qDim) against the
// first nTotal rows of the layer's K/V (row stride kvDim, kv head at kvh·hd),
// output query-major into the attention slab. Same causal rule as the multiQ
// kernel and the composition: query s attends keys [0 .. nTotal-kRows+s].
func encFlashPromptSDPA(enc metal.MTLComputeCommandEncoder, q, k, v, out metal.MTLBuffer,
	nHeads, nKVHeads, hd, kRows, nTotal, qDim, kvDim int, scale float32) error {
	pso, ok := flashPromptPipeline(hd)
	if !ok {
		return core.NewError("native.encFlashPromptSDPA: no flash prompt kernel for head dim")
	}
	if nKVHeads <= 0 || nHeads%nKVHeads != 0 {
		return core.NewError("native.encFlashPromptSDPA: nHeads must be a multiple of nKVHeads")
	}
	// the engine's slabs are element-strided per ROW; the kernel takes the
	// multiQ stride ABI (head stride + seq stride in ELEMENTS)
	sink := encSink{enc}
	sink.setPSO(pso)
	sink.setBuf(q, 0, 0)
	sink.setBuf(k, 0, 1)
	sink.setBuf(v, 0, 2)
	sink.setBuf(out, 0, 3)
	sink.setI32(int32(nHeads/nKVHeads), 4)
	sink.setI32(int32(nTotal), 5)
	sink.setI64(int64(hd), 6)    // k head stride: heads packed in the row
	sink.setI64(int64(kvDim), 7) // k seq stride: one cache row
	sink.setI64(int64(hd), 8)
	sink.setI64(int64(kvDim), 9)
	sink.setF32(scale, 10)
	sink.setI32(int32(kRows), 11)
	sink.setI32(int32(nHeads), 12)
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: uint(nHeads), Height: uint((kRows + flashPromptBQ - 1) / flashPromptBQ), Depth: 1},
		metal.MTLSize{Width: 32, Height: 4, Depth: 1},
	)
	return nil
}
