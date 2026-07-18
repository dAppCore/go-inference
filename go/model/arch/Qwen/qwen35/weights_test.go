// SPDX-Licence-Identifier: EUPL-1.2

package qwen35

import "testing"

// TestWeightNames pins the Qwen 3.6 hybrid tensor layout: the load-bearing norm mapping (the FFN pre-norm
// IS post_attention_layernorm, and PostAttnNorm is empty so the gated-delta mixer applies no post-mixer
// norm), the switch_mlp routed-expert + shared-expert names, and the standard attention/QK-norm names kept
// from StandardWeightNames.
func TestWeightNames(t *testing.T) {
	w := WeightNames()

	// The norm coupling: MLP/FFN pre-norm is post_attention_layernorm, no post-attention (mixer-output) norm.
	if w.MLPNorm != ".post_attention_layernorm.weight" {
		t.Errorf("MLPNorm = %q, want .post_attention_layernorm.weight (qwen 2-norm layout)", w.MLPNorm)
	}
	if w.PostAttnNorm != "" {
		t.Errorf("PostAttnNorm = %q, want \"\" — a non-empty value would make the gated-delta mixer post-norm its output", w.PostAttnNorm)
	}
	if w.NormBiasOne {
		t.Error("NormBiasOne = true, want false (qwen is plain RMSNorm, not gemma's +1 fold)")
	}

	// MoE: batched switch_mlp routed experts + mlp.gate router + shared expert with its σ gate.
	if w.MoE.Router != ".mlp.gate" {
		t.Errorf("MoE.Router = %q, want .mlp.gate", w.MoE.Router)
	}
	if w.MoE.ExpGate != ".mlp.switch_mlp.gate_proj" || w.MoE.ExpUp != ".mlp.switch_mlp.up_proj" || w.MoE.ExpDown != ".mlp.switch_mlp.down_proj" {
		t.Errorf("MoE routed experts = %q/%q/%q, want .mlp.switch_mlp.{gate,up,down}_proj", w.MoE.ExpGate, w.MoE.ExpUp, w.MoE.ExpDown)
	}
	if w.MoE.SharedGate != ".mlp.shared_expert.gate_proj" || w.MoE.SharedUp != ".mlp.shared_expert.up_proj" || w.MoE.SharedDown != ".mlp.shared_expert.down_proj" {
		t.Errorf("MoE shared expert = %q/%q/%q, want .mlp.shared_expert.{gate,up,down}_proj", w.MoE.SharedGate, w.MoE.SharedUp, w.MoE.SharedDown)
	}
	if w.MoE.SharedSigmoid != ".mlp.shared_expert_gate" {
		t.Errorf("MoE.SharedSigmoid = %q, want .mlp.shared_expert_gate", w.MoE.SharedSigmoid)
	}
	if w.MoE.PreFFNorm != ".post_attention_layernorm.weight" {
		t.Errorf("MoE.PreFFNorm = %q, want .post_attention_layernorm.weight", w.MoE.PreFFNorm)
	}

	// Standard names kept: attention input norm, QK-norm, the attention + dense-FFN projections.
	if w.AttnNorm != ".input_layernorm.weight" {
		t.Errorf("AttnNorm = %q, want .input_layernorm.weight", w.AttnNorm)
	}
	if w.QNorm != ".self_attn.q_norm.weight" || w.KNorm != ".self_attn.k_norm.weight" {
		t.Errorf("QK-norm = %q/%q, want .self_attn.{q,k}_norm.weight", w.QNorm, w.KNorm)
	}
	if w.Q != ".self_attn.q_proj" || w.O != ".self_attn.o_proj" {
		t.Errorf("attention projections = %q/%q, want .self_attn.{q,o}_proj", w.Q, w.O)
	}
	if w.Gate != ".mlp.gate_proj" || w.Up != ".mlp.up_proj" || w.Down != ".mlp.down_proj" {
		t.Errorf("dense FFN = %q/%q/%q, want .mlp.{gate,up,down}_proj", w.Gate, w.Up, w.Down)
	}
}
