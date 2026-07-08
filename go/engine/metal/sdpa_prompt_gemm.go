// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"sync"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// Prompt-scale SDPA as a steel GEMM composition — the deep-prefill attention lane.
//
// The multiQ vector kernel (one threadgroup per query row, the whole K/V stream re-read
// per row) is the right shape at decode and short-prompt scale, but its DRAM traffic is
// rows × kv per chunk: at 97K tokens the per-key cost measurably degrades to 2× its floor
// as concurrent threadgroups outgrow the SLC (the #345 ramp), and even the floor runs the
// n² at a small fraction of the machine's matmul rate. mlx-lm's gemma4 prompt attention is
// the same composition below (headDim 256 exceeds MLX's fused steel_attention, which caps
// at 128) and prefills 63K tokens ~11× faster than the vector kernel.
//
// Per (query head h, kv head kvh = h/gqa) the composition is three dispatches:
//
//	S[K × N] = Q_h[K × hd] @ K_kvh[N × hd]ᵀ      (steel nt GEMM, strided head views)
//	P        = softmaxCausal(S × scale)            (lthn_softmax_causal_rows_bf16, in place)
//	O_h      = P[K × N] @ V_kvh[N × hd]           (steel nn GEMM)
//
// K is read once per head instead of once per query row — the traffic drops by the row
// count. S double-buffers across heads so head h+1's wide GEMM1 overlaps head h's skinny
// GEMM2 (32 output tiles) instead of draining the GPU behind it.
//
// Numerics: steel accumulates f32 inside simdgroup MMAs and the softmax runs f32 math, but
// S stores bf16 between stages — a different rounding than the vector kernel's all-f32
// per-row stream. Large-row prefill already trades byte- for token-identity at exactly this
// boundary (the fold's qmm and the ≥32-row steel projections), so this lane rides the same
// tier, pinned by the closeness test in sdpa_prompt_gemm_test.go.

// sdpaPromptGEMMMinKV is the attended length at which the batched dense pass's global-layer
// SDPA switches from the multiQ vector kernel to the GEMM composition. Below it the vector
// kernel's single dispatch wins (no S round-trip, byte-identical to the per-row oracle);
// past it the GEMM's traffic advantage dominates and grows with depth.
const sdpaPromptGEMMMinKV = 4096

// sdpaPromptGEMMMaxRows bounds the chunk row count the composition accepts — the S scratch
// is sized headBatch × rows × maxLen, so an unchunked prompt (a no-sliding-window arch feeds
// the whole prompt as ONE chunk) must stay on the vector kernel rather than ask for an N × N
// slab. Covers the widest window-multiple chunk the prefill policy can emit (8 windows + the
// absorbed tail on a 1024-window arch).
const sdpaPromptGEMMMaxRows = 8192

// sdpaPromptSBudgetBytes caps ONE S scratch buffer for head-batch sizing. The head batch
// adapts under it: at 512-row chunks the whole GQA group batches as today; at wide chunks
// the batch drops toward 1 — where the M dimension itself already fills the machine —
// instead of asking for a gqa × rows × maxLen slab measured in gigabytes.
const sdpaPromptSBudgetBytes = 1 << 30

// sdpaPromptSMaxBytes is the hard ceiling: when even a SINGLE head's S slab (kRows × nCap
// bf16) would exceed it, the composition declines and the chunk keeps the multiQ vector
// kernel — wide chunks on a 256K-context arch don't buy multi-gigabyte scratch.
const sdpaPromptSMaxBytes = 2 << 30

// sdpaPromptGEMMFeasible reports whether the S scratch for this chunk geometry fits the
// hard ceiling at any head batch.
func sdpaPromptGEMMFeasible(kRows, nCap int) bool {
	return int64(kRows)*int64(nCap)*bf16Size <= sdpaPromptSMaxBytes
}

// sdpaPromptHeadBatch picks how many of a GQA group's query heads share one batched GEMM
// dispatch, the largest count whose S slab (heads × kRows × nCap bf16) fits the budget,
// always at least 1.
func sdpaPromptHeadBatch(gqa, kRows, nCap int) int {
	hb := gqa
	for hb > 1 && int64(hb)*int64(kRows)*int64(nCap)*bf16Size > sdpaPromptSBudgetBytes {
		hb--
	}
	if hb < 1 {
		return 1
	}
	return hb
}

// sdpaPromptGEMMDisabledForTest forces the batched pass's global-layer SDPA back onto the
// multiQ vector kernel at any depth — the A/B lever for the closeness and engagement tests.
var sdpaPromptGEMMDisabledForTest bool

// sdpaPromptGEMMEnvDisabled reports the field kill-switch: LTHN_SDPA_GEMM=0 pins deep-prompt
// attention back onto the multiQ vector kernel without a rebuild — the live A/B lever for
// separating a generation-trajectory change (token-identity tier) from a real quality fault.
func sdpaPromptGEMMEnvDisabled() bool { return os.Getenv("LTHN_SDPA_GEMM") == "0" }

var (
	softmaxCausalPSOMu sync.Mutex
	softmaxCausalPSO   metal.MTLComputePipelineState
	softmaxCausalErr   error
	softmaxCausalOnce  sync.Once
)

// softmaxCausalPipeline loads lthn_softmax_causal_rows_bf16 from the sibling custom
// metallib. Cached forever; an absent kernel reports an error and the caller falls back.
func softmaxCausalPipeline() (metal.MTLComputePipelineState, error) {
	softmaxCausalOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			softmaxCausalErr = core.NewError("native.softmaxCausalPipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_softmax_causal_rows_bf16")
		if fn == nil || fn.GetID() == 0 {
			softmaxCausalErr = core.NewError("native.softmaxCausalPipeline: kernel lthn_softmax_causal_rows_bf16 not found")
			return
		}
		softmaxCausalPSO, softmaxCausalErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	softmaxCausalPSOMu.Lock()
	defer softmaxCausalPSOMu.Unlock()
	return softmaxCausalPSO, softmaxCausalErr
}

// gpuHasPromptSDPAGEMM reports whether every pipeline the GEMM composition needs is
// buildable: both steel GEMM variants and the causal softmax kernel.
func gpuHasPromptSDPAGEMM() bool {
	if _, err := softmaxCausalPipeline(); err != nil {
		return false
	}
	if _, ok := steelGEMMPipelineTrans(true, false, false, true); !ok {
		return false
	}
	if _, ok := steelGEMMPipelineTrans(false, false, false, false); !ok {
		return false
	}
	return true
}

// encSoftmaxCausalRows encodes the in-place causal row softmax over S[totalRows × n] bf16.
// totalRows may stack several heads' kRows-row blocks ([head][kRows][n], the batched-GEMM
// layout); the kernel derives each row's query index as row % kRows, keeps keys
// [0 .. n-kRows+s] at scaled scores, and writes the masked tail zero.
func encSoftmaxCausalRows(enc metal.MTLComputeCommandEncoder, s metal.MTLBuffer, kRows, totalRows, n int, scale float32) error {
	pso, err := softmaxCausalPipeline()
	if err != nil {
		return err
	}
	sink := encSink{enc}
	sink.setPSO(pso)
	sink.setBuf(s, 0, 0)
	sink.setI32(int32(n), 1)
	sink.setI32(int32(kRows), 2)
	sink.setF32(scale, 3)
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: uint(totalRows), Height: 1, Depth: 1},
		metal.MTLSize{Width: 1024, Height: 1, Depth: 1},
	)
	return nil
}

// encSDPAPromptGEMM encodes the full prompt-attention GEMM composition for one global layer's
// chunk: kRows query rows (query-major slab q, row stride qDim) against the first nTotal rows
// of the layer's K/V caches (row stride kvDim, head offset kvh*hd), output into the attention
// slab (query-major, row stride qDim). headBatch adjacent q-heads of a GQA group run as ONE
// batched GEMM pair — grid depth carries them off the scalar batch strides (B stride 0
// broadcasts the shared K/V), so GEMM2's skinny output (hd columns) fills headBatch× more of
// the machine at narrow chunks; wide chunks carry the occupancy in M and run smaller batches
// under the S budget. s0/s1 are the two S scratch buffers, each at least headBatch × kRows ×
// nTotal bf16, alternated across batches so the next wide QKᵀ overlaps the previous skinny
// P@V. Reports an error only for a missing pipeline — the caller falls back to the multiQ
// vector kernel.
func encSDPAPromptGEMM(enc metal.MTLComputeCommandEncoder, q, k, v, out, s0, s1 metal.MTLBuffer,
	nHeads, nKVHeads, hd, kRows, nTotal, qDim, kvDim, headBatch int, scale float32) error {
	if nKVHeads <= 0 || nHeads%nKVHeads != 0 {
		return core.NewError("native.encSDPAPromptGEMM: nHeads must be a multiple of nKVHeads")
	}
	gqa := nHeads / nKVHeads
	if headBatch < 1 {
		headBatch = 1
	}
	sBufs := [2]metal.MTLBuffer{s0, s1}
	headBlock := int64(kRows) * int64(nTotal)
	dispatch := 0
	for kvh := 0; kvh < nKVHeads; kvh++ {
		kvhOff := uint(kvh * hd * bf16Size)
		for j := 0; j < gqa; j += headBatch {
			hb := min(headBatch, gqa-j)
			sBuf := sBufs[dispatch&1]
			dispatch++
			groupOff := uint((kvh*gqa + j) * hd * bf16Size)
			// S[b][kRows × nTotal] = Q_(kvh·gqa+j+b) @ K_kvhᵀ for hb adjacent heads at once
			if !encGemmBF16StridedBatch(enc, true, q, k, sBuf, groupOff, kvhOff, 0,
				kRows, nTotal, hd, qDim, kvDim, nTotal,
				hb, int64(hd), 0, headBlock) {
				return core.NewError("native.encSDPAPromptGEMM: steel nt pipeline unavailable")
			}
			// P = softmaxCausal(S × scale) over the hb·kRows stacked rows, masked tails zeroed
			if err := encSoftmaxCausalRows(enc, sBuf, kRows, hb*kRows, nTotal, scale); err != nil {
				return err
			}
			// O_(kvh·gqa+j+b) = P[b] @ V_kvh
			if !encGemmBF16StridedBatch(enc, false, sBuf, v, out, 0, kvhOff, groupOff,
				kRows, hd, nTotal, nTotal, kvDim, qDim,
				hb, headBlock, 0, int64(hd)) {
				return core.NewError("native.encSDPAPromptGEMM: steel nn pipeline unavailable")
			}
		}
	}
	return nil
}
