// SPDX-Licence-Identifier: EUPL-1.2

package gptoss

import (
	"testing"

	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// TestWeightNames pins the GPT-OSS tensor layout: the load-bearing norm mapping (the FFN pre-norm IS
// post_attention_layernorm, no gemma-style post-attention sandwich norm), the MoE block (batched
// experts, router, no shared expert), and the standard attention names kept from StandardWeightNames.
func TestWeightNames(t *testing.T) {
	w := WeightNames()

	if w.MLPNorm != ".post_attention_layernorm.weight" {
		t.Errorf("MLPNorm = %q, want .post_attention_layernorm.weight (llama/qwen 2-norm layout)", w.MLPNorm)
	}
	if w.PostAttnNorm != "" {
		t.Errorf("PostAttnNorm = %q, want \"\" — gpt_oss has no gemma-style post-attention sandwich norm", w.PostAttnNorm)
	}
	if w.NormBiasOne {
		t.Error("NormBiasOne = true, want false (gpt_oss is plain RMSNorm, not gemma's +1 fold)")
	}

	if w.MoE.Router != ".mlp.router" {
		t.Errorf("MoE.Router = %q, want .mlp.router", w.MoE.Router)
	}
	if w.MoE.ExpGate != ".mlp.experts.gate_proj" || w.MoE.ExpUp != ".mlp.experts.up_proj" || w.MoE.ExpDown != ".mlp.experts.down_proj" {
		t.Errorf("MoE routed experts = %q/%q/%q, want .mlp.experts.{gate,up,down}_proj", w.MoE.ExpGate, w.MoE.ExpUp, w.MoE.ExpDown)
	}
	if w.MoE.PreFFNorm != ".post_attention_layernorm.weight" {
		t.Errorf("MoE.PreFFNorm = %q, want .post_attention_layernorm.weight", w.MoE.PreFFNorm)
	}
	if w.MoE.SharedGate != "" || w.MoE.SharedUp != "" || w.MoE.SharedDown != "" || w.MoE.SharedSigmoid != "" {
		t.Errorf("MoE shared-expert names = %+v, want all empty — gpt_oss has no always-on shared expert", w.MoE)
	}

	// Standard names kept: gpt_oss is llama-shaped for embed/attention, no override needed.
	if w.Embed != "model.embed_tokens" || w.LMHead != "lm_head" || w.FinalNorm != "model.norm.weight" {
		t.Errorf("model-level names = %+v, want the canonical HF ids", w)
	}
	if w.AttnNorm != ".input_layernorm.weight" {
		t.Errorf("AttnNorm = %q, want .input_layernorm.weight", w.AttnNorm)
	}
	if w.Q != ".self_attn.q_proj" || w.K != ".self_attn.k_proj" || w.V != ".self_attn.v_proj" || w.O != ".self_attn.o_proj" {
		t.Errorf("attention projections = %+v, want the standard self_attn.{q,k,v,o}_proj suffixes", w)
	}
}

// gptossSyntheticTensors builds a hermetic single-layer, 2-expert tensor set using GPT-OSS's REAL tensor
// names (not simplified/custom names — this proves WeightNames() resolves against the checkpoint's ACTUAL
// naming convention through model.Assemble, the synthetic-weight pattern qwen35's tests use). hidden=4,
// 2 local experts, expertFF=8 — small enough to hand-construct, shaped enough to exercise the batched
// [numExperts*outDim, packedInDim] expert tensor layout real checkpoints ship (see weights.go doc).
func gptossSyntheticTensors() map[string]safetensors.Tensor {
	mat := func(rows, cols int) safetensors.Tensor {
		return safetensors.Tensor{Shape: []int{rows, cols}, Data: make([]byte, rows*cols*2), Dtype: "BF16"}
	}
	vec := func(n int) safetensors.Tensor {
		return safetensors.Tensor{Shape: []int{n}, Data: make([]byte, n*2), Dtype: "BF16"}
	}
	const hidden, expertFF, numExperts = 4, 8, 2
	return map[string]safetensors.Tensor{
		"model.embed_tokens.weight":                      mat(8, hidden),
		"model.norm.weight":                              vec(hidden),
		"model.layers.0.input_layernorm.weight":          vec(hidden),
		"model.layers.0.self_attn.q_proj.weight":         mat(hidden, hidden),
		"model.layers.0.self_attn.k_proj.weight":         mat(hidden, hidden),
		"model.layers.0.self_attn.v_proj.weight":         mat(hidden, hidden),
		"model.layers.0.self_attn.o_proj.weight":         mat(hidden, hidden),
		"model.layers.0.post_attention_layernorm.weight": vec(hidden),
		"model.layers.0.mlp.router.weight":               mat(numExperts, hidden),
		"model.layers.0.mlp.experts.gate_proj.weight":    mat(numExperts*expertFF, hidden),
		"model.layers.0.mlp.experts.up_proj.weight":      mat(numExperts*expertFF, hidden),
		"model.layers.0.mlp.experts.down_proj.weight":    mat(numExperts*hidden, expertFF),
	}
}

func gptossSyntheticArch() model.Arch {
	return model.Arch{
		Hidden: 4, Heads: 2, KVHeads: 2, HeadDim: 2, FF: 8,
		Experts: 2, TopK: 1, ExpertFF: 8,
		MoEGating: model.MoEGatingSoftmax, NormaliseMoETopK: true,
		Layer: []model.LayerSpec{{
			Attention: model.GlobalAttention, CacheIndex: 0, KVShareFrom: 0,
			HeadDim: 2, KVHeads: 2, MoE: true,
		}},
	}
}

// TestWeightNames_Assemble_Good proves WeightNames() resolves a gpt_oss-shaped MoE layer end-to-end
// through model.Assemble: the MoE branch is taken (not the dense Gate/Up/Down branch), and every routed-
// expert weight the router needs is present.
func TestWeightNames_Assemble_Good(t *testing.T) {
	m, err := model.Assemble(gptossSyntheticTensors(), gptossSyntheticArch(), WeightNames())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if len(m.Layers) != 1 {
		t.Fatalf("len(Layers) = %d, want 1", len(m.Layers))
	}
	L := m.Layers[0]
	if L.Gate != nil || L.Up != nil || L.Down != nil {
		t.Fatalf("layer 0 took the dense branch (Gate/Up/Down set), want MoE: %+v", L)
	}
	if L.MoE == nil {
		t.Fatal("layer 0 MoE is nil — WeightNames().MoE did not resolve against gpt_oss-shaped tensor names")
	}
	if L.MoE.Router == nil || L.MoE.ExpGate == nil || L.MoE.ExpUp == nil || L.MoE.ExpDown == nil {
		t.Fatalf("layer 0 MoE missing a routed-expert weight: %+v", L.MoE)
	}
	if len(L.MoE.PreFFNorm) == 0 {
		t.Fatal("layer 0 MoE.PreFFNorm did not resolve (post_attention_layernorm mapping broken)")
	}
	if L.MoE.SharedGate != nil {
		t.Fatalf("layer 0 MoE.SharedGate resolved to non-nil, want nil — gpt_oss has no shared expert: %+v", L.MoE.SharedGate)
	}
	if L.Q == nil || L.K == nil || L.O == nil {
		t.Fatalf("layer 0 missing a required attention weight: %+v", L)
	}
}

// TestWeightNames_Assemble_Bad proves a checkpoint missing the router tensor entirely still assembles
// (Assemble does not hard-require every MoE sub-weight — see model.LoadedModel.ValidateRequired) but the
// resulting MoE.Router is nil, so a caller consuming L.MoE.Router without a nil check would panic —
// pinning that Assemble does NOT fabricate a router, it faithfully reports absence.
func TestWeightNames_Assemble_Bad(t *testing.T) {
	tensors := gptossSyntheticTensors()
	delete(tensors, "model.layers.0.mlp.router.weight")
	m, err := model.Assemble(tensors, gptossSyntheticArch(), WeightNames())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if m.Layers[0].MoE == nil {
		t.Fatal("layer 0 MoE is nil — a missing router tensor should not remove the whole MoE block")
	}
	if m.Layers[0].MoE.Router != nil {
		t.Fatal("MoE.Router resolved to non-nil despite the router tensor being absent from the checkpoint")
	}
}

// TestWeightNames_Assemble_Ugly proves the batched expert tensor's declared SHAPE (2-D
// [numExperts*expertFF, hidden] here, 3-D [numExperts, expertFF, hidden] in some real conversions) does
// not matter to Assemble — only the RAW BYTE LENGTH does (engine/metal's moe_batch.go re-derives the
// per-expert stride from arch.ExpertFF/NumExperts, not the tensor's own shape/rank — see weights.go doc).
// A 3-D-shaped tensor with byte-identical Data to the 2-D fixture must resolve identically.
func TestWeightNames_Assemble_Ugly(t *testing.T) {
	tensors := gptossSyntheticTensors()
	t2d := tensors["model.layers.0.mlp.experts.gate_proj.weight"]
	tensors["model.layers.0.mlp.experts.gate_proj.weight"] = safetensors.Tensor{
		Shape: []int{2, 8, 4}, Data: t2d.Data, Dtype: t2d.Dtype, // same bytes, 3-D shape instead of [16,4]
	}
	m, err := model.Assemble(tensors, gptossSyntheticArch(), WeightNames())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if m.Layers[0].MoE.ExpGate == nil || len(m.Layers[0].MoE.ExpGate.Weight) != len(t2d.Data) {
		t.Fatalf("a 3-D-shaped expert tensor did not resolve to the same byte length as its 2-D sibling")
	}
}
