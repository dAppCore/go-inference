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

// --- 2b: head dim 512 via split-D (lthn_attn_splitd) ---

const (
	splitDAttnBQ      = 16
	splitDAttnBK      = 16
	splitDAttnThreads = 64 // wm2 × wn1 × 32
)

// splitDAttnMinKV depth-gates the 512 lane's LOWER bound: split-D recomputes
// QK per half, costing ~8% at 8K on 31B, crossing the composition at 32K
// (207 vs 204). BUT the win is a BAND, not a ramp: the 235K all-defaults
// ingest ran ≥42% SLOWER with split-D engaged (50min+ vs the 35.6min
// baseline, 2026-07-13) — at extreme depth the kernel's serial per-tile kv
// loop (64 threads, two barrier-bracketed half-fills per block) compounds
// past the budget-capped composition. So the 512 lane is OPT-IN
// (LTHN_FLASH_512=1) until the upper bound is receipted and band-gated;
// the composition remains the 512 default.
const splitDAttnMinKV = 32768

// flash512Enabled opts the 512 split-D lane in (see splitDAttnMinKV — the
// win is a band and the upper bound is unreceipted).
var flash512Enabled = os.Getenv("LTHN_FLASH_512") == "1"

var (
	splitDAttnPSOMu    sync.Mutex
	splitDAttnPSOCache = map[steelAttnKey]metal.MTLComputePipelineState{}
)

// splitDAttnPipeline resolves the split-D flash kernel (hd 512 = two 256
// halves; grid.z picks the half) with the same alignment/causal constants.
func splitDAttnPipeline(alignQ, alignK bool) (metal.MTLComputePipelineState, bool) {
	splitDAttnPSOMu.Lock()
	defer splitDAttnPSOMu.Unlock()
	key := steelAttnKey{hd: 512, alignQ: alignQ, alignK: alignK}
	if pso, ok := splitDAttnPSOCache[key]; ok {
		return pso, pso != nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		splitDAttnPSOCache[key] = nil
		return nil, false
	}
	name := core.Sprintf("lthn_attn_splitd_bfloat16_bq%d_bk%d_bdh256_wm2_wn1", splitDAttnBQ, splitDAttnBK)
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
		splitDAttnPSOCache[key] = nil
		return nil, false
	}
	pso, perr := device.NewComputePipelineStateWithFunctionError(fn)
	if perr != nil || pso == nil || pso.GetID() == 0 {
		splitDAttnPSOCache[key] = nil
		return nil, false
	}
	splitDAttnPSOCache[key] = pso
	return pso, true
}

// encSteelAttnSplitD encodes the hd-512 chunk-layer prompt attention as ONE
// split-D flash dispatch pair: grid (NQ, H, 2), each z-half recomputing the
// full S from both Q·K halves and writing its own 256-wide O half. Same
// causal rule, same params ABI (D = the full 512).
func encSteelAttnSplitD(enc metal.MTLComputeCommandEncoderObject, q, k, v, out metal.MTLBuffer,
	nHeads, nKVHeads, hd, kRows, nTotal, qDim, kvDim int, scale float32) error {
	alignQ := kRows%splitDAttnBQ == 0
	alignK := nTotal%splitDAttnBK == 0
	pso, ok := splitDAttnPipeline(alignQ, alignK)
	if !ok {
		return core.NewError("native.encSteelAttnSplitD: split-D pipeline unavailable")
	}
	nq := (kRows + splitDAttnBQ - 1) / splitDAttnBQ
	nk := (nTotal + splitDAttnBK - 1) / splitDAttnBK
	p := steelAttnParams{
		b: 1, h: int32(nHeads), d: int32(hd),
		qL: int32(kRows), kL: int32(nTotal),
		gqaFactor: int32(nHeads / nKVHeads), scale: scale,
		nq: int32(nq), nk: int32(nk),
		nqAligned: int32(kRows / splitDAttnBQ), nkAligned: int32(nTotal / splitDAttnBK),
		qLRem: int32(kRows % splitDAttnBQ), kLRem: int32(nTotal % splitDAttnBK),
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
		metal.MTLSize{Width: uint(nq), Height: uint(nHeads), Depth: 2},
		metal.MTLSize{Width: splitDAttnThreads, Height: 1, Depth: 1},
	)
	return nil
}

// --- phase 3: the q8-READING flash (lthn_attn_q8) ---
//
// K/V arrive as the engine's int8 codes + f32 group scales and dequantise
// inside the tile load — no mirrors, no dequant dispatches, half the K/V
// device bytes. Two instantiations share one body: nh1 (whole 256 head) and
// nh2 (split-D 512).

// flashQ8Enabled opts the q8-reading flash lane in (LTHN_FLASH_Q8=1).
// FALSIFIED as a default (2026-07-13): the in-loader dequant runs per
// QUERY-TILE, so a q8 owner's prefix dequantises O(N²/BQ) times where the
// mirror lane dequantises once per chunk — parity-correct (the gate passes)
// but 551 vs 2171 tok/s at 62K on e2b. The mirror + bf16-flash economics
// win at depth; the kernel stays in-tree for a future once-per-chunk
// staging shape.
var flashQ8Enabled = os.Getenv("LTHN_FLASH_Q8") == "1"

// flashQ8OffForTest pins the mirror-dequant lane in-process — the parity
// tests A/B the q8-reading flash against it on fresh sessions.
var flashQ8OffForTest bool

// q8StageEnabled routes a q8 owner's per-chunk prefix dequant into the shared
// ping-pong staging pair instead of per-layer full-cacheRows mirror planes —
// the once-per-chunk staging shape the FALSIFIED note above anticipated
// (#375): same dequant kernel, same flash/GEMM consumers, one transient plane
// pair per parity instead of one persistent plane per global owner (the
// 31B@256K ~19GB ingest-peak cut). Default ON; LTHN_Q8_STAGE=0 restores the
// mirror planes.
var q8StageEnabled = os.Getenv("LTHN_Q8_STAGE") != "0"

// q8StageOffForTest pins the legacy mirror-plane lane in-process — the
// byte-identity A/B (TestQ8StagePromptMatchesMirrorLane) flips it.
var q8StageOffForTest bool

// prefillSkipSharedEnabled skips the trailing KV-shared (non-cache-owning)
// layers on NON-FINAL prefill chunks: those layers land no cache rows and
// their outputs feed only the chunk's unread boundary hidden, so the work is
// dead — later positions reach the prompt through the OWNER layers' KV. This
// is the mechanism behind mlx-lm's prefill lead (#381: its lazy DCE prunes
// the same 20-of-35 gemma4 layers per chunk). Token-identical by
// construction; the final chunk and every decode pass run the full stack.
// Default ON; LTHN_PREFILL_SKIP_SHARED=0 restores full-stack chunks.
var prefillSkipSharedEnabled = os.Getenv("LTHN_PREFILL_SKIP_SHARED") != "0"

// prefillSkipSharedOffForTest pins the full-stack chunk lane in-process — the
// byte-identity A/B (TestArchSessionPrefillChunksSkipSharedSuffix) flips it.
var prefillSkipSharedOffForTest bool

type flashQ8Key struct {
	nHalves        int
	alignQ, alignK bool
}

var (
	flashQ8PSOMu    sync.Mutex
	flashQ8PSOCache = map[flashQ8Key]metal.MTLComputePipelineState{}
)

func flashQ8Pipeline(nHalves int, alignQ, alignK bool) (metal.MTLComputePipelineState, bool) {
	flashQ8PSOMu.Lock()
	defer flashQ8PSOMu.Unlock()
	key := flashQ8Key{nHalves: nHalves, alignQ: alignQ, alignK: alignK}
	if pso, ok := flashQ8PSOCache[key]; ok {
		return pso, pso != nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		flashQ8PSOCache[key] = nil
		return nil, false
	}
	name := "lthn_attn_q8_bfloat16_bq32_bk16_bdh256_nh1_wm4_wn1"
	if nHalves == 2 {
		name = "lthn_attn_q8_bfloat16_bq16_bk16_bdh256_nh2_wm2_wn1"
	}
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
		flashQ8PSOCache[key] = nil
		return nil, false
	}
	pso, perr := device.NewComputePipelineStateWithFunctionError(fn)
	if perr != nil || pso == nil || pso.GetID() == 0 {
		flashQ8PSOCache[key] = nil
		return nil, false
	}
	flashQ8PSOCache[key] = pso
	return pso, true
}

// flashQ8Usable is the seam's routing check for a q8 OWNER's chunk: the lane
// exists for the head dim and the depth clears the split-D crossover on 512
// (the q8 reads halve the recompute's traffic cost, so the true crossover
// may sit lower — receipt pending; the bf16 gate is reused conservatively).
func flashQ8Usable(hd, nTotal int) bool {
	switch hd {
	case 256:
		_, ok := flashQ8Pipeline(1, true, true)
		return ok
	case 512:
		if nTotal < splitDAttnMinKV {
			return false
		}
		_, ok := flashQ8Pipeline(2, true, true)
		return ok
	}
	return false
}

// encFlashPromptQ8 encodes the chunk-layer prompt attention for a q8 owner as
// flash dispatches reading the int8 codes + f32 scales directly. Same causal
// rule and params ABI as the bf16 lanes; K_strides[2] carries kvDim (the q8
// code row width in elements).
func encFlashPromptQ8(enc metal.MTLComputeCommandEncoderObject, q, kCodes, vCodes, kScales, vScales, out metal.MTLBuffer,
	nHeads, nKVHeads, hd, kRows, nTotal, qDim, kvDim int, scale float32) error {
	nHalves := 1
	bq := steelAttnBQ
	threads := steelAttnThreads
	if hd == 512 {
		nHalves, bq, threads = 2, splitDAttnBQ, splitDAttnThreads
	} else if hd != 256 {
		return core.NewError("native.encFlashPromptQ8: unsupported head dim")
	}
	alignQ := kRows%bq == 0
	alignK := nTotal%steelAttnBK == 0
	pso, ok := flashQ8Pipeline(nHalves, alignQ, alignK)
	if !ok {
		return core.NewError("native.encFlashPromptQ8: q8 flash pipeline unavailable")
	}
	nq := (kRows + bq - 1) / bq
	nk := (nTotal + steelAttnBK - 1) / steelAttnBK
	p := steelAttnParams{
		b: 1, h: int32(nHeads), d: int32(hd),
		qL: int32(kRows), kL: int32(nTotal),
		gqaFactor: int32(nHeads / nKVHeads), scale: scale,
		nq: int32(nq), nk: int32(nk),
		nqAligned: int32(kRows / bq), nkAligned: int32(nTotal / steelAttnBK),
		qLRem: int32(kRows % bq), kLRem: int32(nTotal % steelAttnBK),
		qLOff:    int32(nTotal - kRows),
		qStrides: [3]int64{0, int64(hd), int64(qDim)},
		kStrides: [3]int64{0, int64(hd), int64(kvDim)},
		vStrides: [3]int64{0, int64(hd), int64(kvDim)},
		oStrides: [3]int64{0, int64(hd), int64(qDim)},
	}
	sink := encSink{enc}
	sink.setPSO(pso)
	sink.setBuf(q, 0, 0)
	sink.setBuf(kCodes, 0, 1)
	sink.setBuf(vCodes, 0, 2)
	sink.setBuf(out, 0, 3)
	pb := unsafe.Slice((*byte)(unsafe.Pointer(&p)), unsafe.Sizeof(p))
	enc.SetBytesLengthAtIndex(pb, uint(len(pb)), 4)
	sink.setBuf(kScales, 0, 5)
	sink.setBuf(vScales, 0, 6)
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: uint(nq), Height: uint(nHeads), Depth: uint(nHalves)},
		metal.MTLSize{Width: uint(threads), Height: 1, Depth: 1},
	)
	return nil
}

// --- phase 4: the sliding-window flash (lthn_attn_win) ---
//
// gemma4's MAJORITY layers window at W; the deferred-ring lane served them
// with the multiQ vector kernel — every query row re-reading its whole
// window from device. The window flash streams each query TILE's fixed
// ≤ W+BQ key span once through threadgroup tiles (two-source: the wrapped
// pre-batch ring + the chunk's stage slabs), with the window floor added to
// the causal mask. Depth-flat by construction, traffic ÷BQ.

// flashWinEnabled gates the sliding-window flash (LTHN_FLASH_WIN=0 restores
// the multiQ ring kernel — the A/B lever).
var flashWinEnabled = os.Getenv("LTHN_FLASH_WIN") != "0"

// flashWinOffForTest pins the multiQ ring lane in-process for parity A/Bs.
var flashWinOffForTest bool

// attnWinParams mirrors the kernel's AttnWinParams.
type attnWinParams struct {
	winW     int32
	ringLive int32
}

var (
	flashWinPSOMu    sync.Mutex
	flashWinPSOCache = map[steelAttnKey]metal.MTLComputePipelineState{}
)

func flashWinPipeline(alignQ bool) (metal.MTLComputePipelineState, bool) {
	flashWinPSOMu.Lock()
	defer flashWinPSOMu.Unlock()
	key := steelAttnKey{hd: 256, alignQ: alignQ, alignK: true}
	if pso, ok := flashWinPSOCache[key]; ok {
		return pso, pso != nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		flashWinPSOCache[key] = nil
		return nil, false
	}
	fc := metal.NewMTLFunctionConstantValues()
	aQ, aK := alignQ, true // the window loader zero-fills; K alignment is moot
	hasMask, doCausal, hasSinks := false, true, false
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&aQ), metal.MTLDataTypeBool, 200)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&aK), metal.MTLDataTypeBool, 201)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&hasMask), metal.MTLDataTypeBool, 300)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&doCausal), metal.MTLDataTypeBool, 301)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&hasSinks), metal.MTLDataTypeBool, 302)
	fn, err := customLibrary.NewFunctionWithNameConstantValuesError("lthn_attn_win_bfloat16_bq32_bk16_bd256_wm4_wn1", fc)
	if err != nil || fn == nil || fn.GetID() == 0 {
		flashWinPSOCache[key] = nil
		return nil, false
	}
	pso, perr := device.NewComputePipelineStateWithFunctionError(fn)
	if perr != nil || pso == nil || pso.GetID() == 0 {
		flashWinPSOCache[key] = nil
		return nil, false
	}
	flashWinPSOCache[key] = pso
	return pso, true
}

// flashWinUsable reports the sliding-window flash lane for a head dim (256
// only — both gemma4 families' sliding geometry).
func flashWinUsable(hd int) bool {
	if hd != 256 {
		return false
	}
	_, ok := flashWinPipeline(true)
	return ok
}

// encFlashWindowSDPA encodes a sliding layer's chunk attention as ONE window
// flash dispatch: ring K/V (wrapped at winW rows) + the chunk's stage slabs,
// query tile streams its own window span. ringLive = min(basePos, winW).
func encFlashWindowSDPA(enc metal.MTLComputeCommandEncoderObject, q, kRing, vRing, kStage, vStage, out metal.MTLBuffer,
	nHeads, nKVHeads, hd, kRows, winW, basePos, ringLive, qDim, kvDim int, scale float32) error {
	alignQ := kRows%steelAttnBQ == 0
	pso, ok := flashWinPipeline(alignQ)
	if !ok {
		return core.NewError("native.encFlashWindowSDPA: window flash pipeline unavailable")
	}
	nq := (kRows + steelAttnBQ - 1) / steelAttnBQ
	p := steelAttnParams{
		b: 1, h: int32(nHeads), d: int32(hd),
		qL: int32(kRows), kL: int32(basePos + kRows),
		gqaFactor: int32(nHeads / nKVHeads), scale: scale,
		nq: int32(nq), nk: 0,
		nqAligned: int32(kRows / steelAttnBQ), nkAligned: 0,
		qLRem: int32(kRows % steelAttnBQ), kLRem: 0,
		qLOff:    int32(basePos),
		qStrides: [3]int64{0, int64(hd), int64(qDim)},
		kStrides: [3]int64{0, int64(hd), int64(kvDim)},
		vStrides: [3]int64{0, int64(hd), int64(kvDim)},
		oStrides: [3]int64{0, int64(hd), int64(qDim)},
	}
	w := attnWinParams{winW: int32(winW), ringLive: int32(ringLive)}
	sink := encSink{enc}
	sink.setPSO(pso)
	sink.setBuf(q, 0, 0)
	sink.setBuf(kRing, 0, 1)
	sink.setBuf(vRing, 0, 2)
	sink.setBuf(out, 0, 3)
	pb := unsafe.Slice((*byte)(unsafe.Pointer(&p)), unsafe.Sizeof(p))
	enc.SetBytesLengthAtIndex(pb, uint(len(pb)), 4)
	sink.setBuf(kStage, 0, 5)
	sink.setBuf(vStage, 0, 6)
	wb := unsafe.Slice((*byte)(unsafe.Pointer(&w)), unsafe.Sizeof(w))
	enc.SetBytesLengthAtIndex(wb, uint(len(wb)), 7)
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: uint(nq), Height: uint(nHeads), Depth: 1},
		metal.MTLSize{Width: steelAttnThreads, Height: 1, Depth: 1},
	)
	return nil
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
	switch hd {
	case 256:
		_, ok := steelAttnPipeline(hd, true, true)
		return ok
	case 512:
		_, ok := splitDAttnPipeline(true, true)
		return ok
	}
	return false
}

// flashPromptUsable is the seam's routing check: the lane exists for the head
// dim AND the depth is on flash's side of its receipted crossover. 256 wins
// or ties everywhere; 512 (split-D) pays its QK recompute below ~32K and
// keeps the composition there.
func flashPromptUsable(hd, nTotal int) bool {
	if !gpuHasFlashPrompt(hd) {
		return false
	}
	if hd == 512 && (!flash512Enabled || nTotal < splitDAttnMinKV) {
		return false
	}
	return true
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
	if hd == 512 { // 2b: split-D — the seam's flashPromptUsable already depth-gated it
		return encSteelAttnSplitD(enc, q, k, v, out, nHeads, nKVHeads, hd, kRows, nTotal, qDim, kvDim, scale)
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
