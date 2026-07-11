// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"sync"
	"unsafe"

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
// onto the flash lane instead of the GEMM composition (LTHN_FLASH_PROMPT=0
// restores the composition — the A/B lever; both lanes are token-identity
// against the multiQ oracle). Default ON: the steel BD-256 lane (v2) is
// token-identical to the composition with rate parity at 8-62K and +4% at
// 118K (e2b receipts, 2026-07-13), runs ONE dispatch instead of three, and
// never allocates the up-to-2GB sdpaPromptS scratch pair. The v1 vector
// kernel stays in-tree as the parity oracle; 512-dim (31B) keeps the
// composition until the split-D instantiation (phase 2b).
var flashPromptEnabled = os.Getenv("LTHN_FLASH_PROMPT") != "0"

var (
	flashPromptPSOMu    sync.Mutex
	flashPromptPSOCache = map[int]metal.MTLComputePipelineState{}
)

// --- v2: MLX's steel flash-attention template at the head dim nobody ships ---
//
// steel_attention is BD-parameterised; upstream instantiates 64/80/128 only.
// lthn_steel_attn_256.metal instantiates bq32/bk16/bd256/wm4/wn1 (TG memory
// 29.2KB — fits), and this host side resolves it with the attention function
// constants (do_causal=true, no mask/sinks) and feeds the AttnParams ABI.

const (
	steelAttnBQ      = 32
	steelAttnBK      = 16
	steelAttnThreads = 128 // wm4 × wn1 × 32
)

// steelAttnParams mirrors mlx::steel::AttnParams byte-for-byte: fourteen
// 4-byte fields (56 bytes, 8-aligned) then four int64[3] stride blocks.
type steelAttnParams struct {
	b, h, d              int32
	qL, kL               int32
	gqaFactor            int32
	scale                float32
	nq, nk               int32
	nqAligned, nkAligned int32
	qLRem, kLRem, qLOff  int32
	qStrides             [3]int64
	kStrides             [3]int64
	vStrides             [3]int64
	oStrides             [3]int64
}

type steelAttnKey struct {
	hd             int
	alignQ, alignK bool
}

var (
	steelAttnPSOMu    sync.Mutex
	steelAttnPSOCache = map[steelAttnKey]metal.MTLComputePipelineState{}
)

// steelAttnPipeline resolves the BD-instantiated steel attention with the
// alignment function constants (200/201) and the fixed lane shape: causal,
// no mask, no sinks (301 true, 300/302 false).
func steelAttnPipeline(hd int, alignQ, alignK bool) (metal.MTLComputePipelineState, bool) {
	steelAttnPSOMu.Lock()
	defer steelAttnPSOMu.Unlock()
	key := steelAttnKey{hd: hd, alignQ: alignQ, alignK: alignK}
	if pso, ok := steelAttnPSOCache[key]; ok {
		return pso, pso != nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		steelAttnPSOCache[key] = nil
		return nil, false
	}
	name := core.Sprintf("steel_attention_bfloat16_bq%d_bk%d_bd%d_wm4_wn1_maskbool_", steelAttnBQ, steelAttnBK, hd)
	fc := metal.NewMTLFunctionConstantValues()
	aQ, aK := alignQ, alignK
	hasMask, doCausal, hasSinks := false, true, false
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&aQ), metal.MTLDataTypeBool, 200)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&aK), metal.MTLDataTypeBool, 201)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&hasMask), metal.MTLDataTypeBool, 300)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&doCausal), metal.MTLDataTypeBool, 301)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&hasSinks), metal.MTLDataTypeBool, 302)
	fn, err := customLibrary.NewFunctionWithNameConstantValuesError(name, fc)
	if err != nil || fn == nil || fn.GetID() == 0 {
		steelAttnPSOCache[key] = nil
		return nil, false
	}
	pso, perr := device.NewComputePipelineStateWithFunctionError(fn)
	if perr != nil || pso == nil || pso.GetID() == 0 {
		steelAttnPSOCache[key] = nil
		return nil, false
	}
	steelAttnPSOCache[key] = pso
	return pso, true
}

// encSteelAttnPrompt encodes the chunk-layer prompt attention as ONE steel
// flash dispatch: query-major slab q (row stride qDim, head h at h·hd),
// K/V caches (row stride kvDim, kv head at kvh·hd), output query-major.
// Causality via qL_off = basePos: query s attends keys [0 .. basePos+s],
// identical to the multiQ rule.
func encSteelAttnPrompt(enc metal.MTLComputeCommandEncoderObject, q, k, v, out metal.MTLBuffer,
	nHeads, nKVHeads, hd, kRows, nTotal, qDim, kvDim int, scale float32) error {
	alignQ := kRows%steelAttnBQ == 0
	alignK := nTotal%steelAttnBK == 0
	pso, ok := steelAttnPipeline(hd, alignQ, alignK)
	if !ok {
		return core.NewError("native.encSteelAttnPrompt: steel attention pipeline unavailable")
	}
	nq := (kRows + steelAttnBQ - 1) / steelAttnBQ
	nk := (nTotal + steelAttnBK - 1) / steelAttnBK
	p := steelAttnParams{
		b: 1, h: int32(nHeads), d: int32(hd),
		qL: int32(kRows), kL: int32(nTotal),
		gqaFactor: int32(nHeads / nKVHeads), scale: scale,
		nq: int32(nq), nk: int32(nk),
		nqAligned: int32(kRows / steelAttnBQ), nkAligned: int32(nTotal / steelAttnBK),
		qLRem: int32(kRows % steelAttnBQ), kLRem: int32(nTotal % steelAttnBK),
		qLOff:    int32(nTotal - kRows),
		qStrides: [3]int64{0, int64(hd), int64(qDim)},
		kStrides: [3]int64{0, int64(hd), int64(kvDim)},
		vStrides: [3]int64{0, int64(hd), int64(kvDim)},
		oStrides: [3]int64{0, int64(hd), int64(qDim)},
	}
	sink := encSink{enc}
	sink.setPSO(pso)
	sink.setBuf(q, 0, 0)
	sink.setBuf(k, 0, 1)
	sink.setBuf(v, 0, 2)
	sink.setBuf(out, 0, 3)
	pb := unsafe.Slice((*byte)(unsafe.Pointer(&p)), unsafe.Sizeof(p))
	enc.SetBytesLengthAtIndex(pb, uint(len(pb)), 4)
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: uint(nq), Height: uint(nHeads), Depth: 1},
		metal.MTLSize{Width: steelAttnThreads, Height: 1, Depth: 1},
	)
	return nil
}

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

// gpuHasFlashPrompt reports whether a flash prompt lane exists for a head dim:
// the steel BD-256 instantiation (v2 — the production lane), with the v1
// vector-flash kernel kept in-tree as its parity oracle. 512 stays on the
// GEMM composition until the split-D treatment (phase 2b).
func gpuHasFlashPrompt(hd int) bool {
	if hd == 256 {
		_, ok := steelAttnPipeline(hd, true, true)
		return ok
	}
	return false
}

// encFlashPromptSDPA encodes the whole chunk-layer prompt attention as ONE
// dispatch: kRows query rows (query-major slab, row stride qDim) against the
// first nTotal rows of the layer's K/V (row stride kvDim, kv head at kvh·hd),
// output query-major into the attention slab. Same causal rule as the multiQ
// kernel and the composition: query s attends keys [0 .. nTotal-kRows+s].
func encFlashPromptSDPA(enc metal.MTLComputeCommandEncoderObject, q, k, v, out metal.MTLBuffer,
	nHeads, nKVHeads, hd, kRows, nTotal, qDim, kvDim int, scale float32) error {
	if hd == 256 { // v2: the steel BD-256 instantiation (one MMA flash dispatch)
		return encSteelAttnPrompt(enc, q, k, v, out, nHeads, nKVHeads, hd, kRows, nTotal, qDim, kvDim, scale)
	}
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
