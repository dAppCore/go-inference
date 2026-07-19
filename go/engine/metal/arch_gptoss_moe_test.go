// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"math"
	"testing"

	"dappco.re/go/inference/model"
)

// TestGptOssMoE_GptOssRouterTopK_Good proves the router-bias placement: the bias lands BEFORE
// top-k (mlx-lm gpt_oss.py: router = nn.Linear(..., bias=True), so the bias is inside the logits
// the top-k reads). The fixture's bias FLIPS the selection — raw logits pick expert 0, biased
// logits pick expert 2 — and the combine weights are softmax over the biased top-k, renormalised.
func TestGptOssMoE_GptOssRouterTopK_Good(t *testing.T) {
	logits := []float32{1.0, 0.0, 0.5, -2.0}
	bias := f32ToBf16Slice([]float32{0, 0, 1.0, 0}) // lifts expert 2's logit 0.5 → 1.5, past expert 0's 1.0

	idx, wts, err := gptOssRouterTopK(logits, bias, 2)
	if err != nil {
		t.Fatalf("gptOssRouterTopK: %v", err)
	}
	if len(idx) != 2 || idx[0] != 2 || idx[1] != 0 {
		t.Fatalf("selection = %v, want [2 0] — the bias must apply BEFORE top-k", idx)
	}
	// weights = softmax over the biased selection {1.5, 1.0} renormalised: e^1.5/(e^1.5+e^1.0), ...
	d := math.Exp(1.5) + math.Exp(1.0)
	want0, want1 := math.Exp(1.5)/d, math.Exp(1.0)/d
	got := bf16ToF32Slice(wts)
	if math.Abs(float64(got[0])-want0) > 0.01 || math.Abs(float64(got[1])-want1) > 0.01 {
		t.Fatalf("combine weights = %v, want ~[%.4f %.4f] (softmax over the biased top-k, renormalised)", got, want0, want1)
	}
	if s := float64(got[0]) + float64(got[1]); math.Abs(s-1) > 0.01 {
		t.Fatalf("combine weights sum %v, want 1 (NormaliseMoETopK)", s)
	}
}

// TestGptOssMoE_GptOssRouterTopK_Bad proves the guards: topK out of range and a wrong-length bias
// both refuse.
func TestGptOssMoE_GptOssRouterTopK_Bad(t *testing.T) {
	if _, _, err := gptOssRouterTopK([]float32{1, 2}, nil, 3); err == nil {
		t.Fatal("expected topK > len(logits) to refuse")
	}
	if _, _, err := gptOssRouterTopK([]float32{1, 2}, nil, 0); err == nil {
		t.Fatal("expected topK <= 0 to refuse")
	}
	if _, _, err := gptOssRouterTopK([]float32{1, 2}, f32ToBf16Slice([]float32{0}), 1); err == nil {
		t.Fatal("expected a bias shorter than the logits to refuse")
	}
}

// TestGptOssMoE_GptOssRouterTopK_Ugly proves nil bias ≡ zero bias — the no-bias path is the
// identical decision and identical weights (the regression contract for a checkpoint without a
// router bias).
func TestGptOssMoE_GptOssRouterTopK_Ugly(t *testing.T) {
	logits := []float32{0.3, -1.1, 2.0, 0.9}
	idxNil, wtsNil, err := gptOssRouterTopK(logits, nil, 2)
	if err != nil {
		t.Fatalf("gptOssRouterTopK(nil bias): %v", err)
	}
	idxZero, wtsZero, err := gptOssRouterTopK(logits, f32ToBf16Slice(make([]float32, 4)), 2)
	if err != nil {
		t.Fatalf("gptOssRouterTopK(zero bias): %v", err)
	}
	if len(idxNil) != len(idxZero) || idxNil[0] != idxZero[0] || idxNil[1] != idxZero[1] {
		t.Fatalf("nil-bias selection %v != zero-bias selection %v", idxNil, idxZero)
	}
	if !bytes.Equal(wtsNil, wtsZero) {
		t.Fatal("nil-bias weights differ from zero-bias weights — the no-bias path is not neutral")
	}
}

// TestGptOssMoE_MoeToQuant_Good proves the gpt_oss mapping: a clamped-SwiGLU arch carries the
// marker, the limit, and every additive bias off the Linears' own .Bias captures.
func TestGptOssMoE_MoeToQuant_Good(t *testing.T) {
	e := &model.LoadedMoE{
		Router:  &model.Linear{Weight: make([]byte, 8), Bias: make([]byte, 4*bf16Size)},
		ExpGate: &model.Linear{Weight: make([]byte, 8), Bias: make([]byte, 4*8*bf16Size)},
		ExpUp:   &model.Linear{Weight: make([]byte, 8), Bias: make([]byte, 4*8*bf16Size)},
		ExpDown: &model.Linear{Weight: make([]byte, 8), Bias: make([]byte, 4*16*bf16Size)},
	}
	arch := model.Arch{
		Experts: 4, TopK: 2, ExpertFF: 8, Hidden: 16,
		Activation: "gpt_oss_clamped_swiglu", SwigluLimit: 7,
	}
	q := moeToQuant(e, arch)
	if !q.ClampedSwiGLU || q.SwigluLimit != 7 {
		t.Fatalf("ClampedSwiGLU/SwigluLimit = %v/%v, want true/7", q.ClampedSwiGLU, q.SwigluLimit)
	}
	if len(q.RouterBias) != 4*bf16Size {
		t.Fatalf("RouterBias = %d bytes, want %d", len(q.RouterBias), 4*bf16Size)
	}
	if len(q.ExpGateBias) != 4*8*bf16Size || len(q.ExpUpBias) != 4*8*bf16Size || len(q.ExpDownBias) != 4*16*bf16Size {
		t.Fatalf("expert biases = %d/%d/%d bytes, want the batched [experts×outDim] shapes", len(q.ExpGateBias), len(q.ExpUpBias), len(q.ExpDownBias))
	}
}

// TestGptOssMoE_MoeToQuant_Ugly is the OTHER-ARCH regression: a non-clamped activation (gemma /
// qwen) maps zero gpt_oss fields even when the checkpoint's Linears happen to carry .bias tensors
// — the marker gates the whole block, so the existing lanes' weights are byte-identical to before.
func TestGptOssMoE_MoeToQuant_Ugly(t *testing.T) {
	e := &model.LoadedMoE{
		Router:  &model.Linear{Weight: make([]byte, 8), Bias: make([]byte, 4*bf16Size)},
		ExpGate: &model.Linear{Weight: make([]byte, 8), Bias: make([]byte, 4*8*bf16Size)},
	}
	arch := model.Arch{Experts: 4, TopK: 2, ExpertFF: 8, Hidden: 16, Activation: "silu"}
	q := moeToQuant(e, arch)
	if q.ClampedSwiGLU || q.SwigluLimit != 0 {
		t.Fatalf("non-gpt_oss arch mapped ClampedSwiGLU=%v/limit=%v, want false/0", q.ClampedSwiGLU, q.SwigluLimit)
	}
	if q.RouterBias != nil || q.ExpGateBias != nil || q.ExpUpBias != nil || q.ExpDownBias != nil {
		t.Fatal("non-gpt_oss arch mapped additive MoE biases — the marker must gate the whole block")
	}
}
