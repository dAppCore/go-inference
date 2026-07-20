// SPDX-Licence-Identifier: EUPL-1.2

package granitemoe

import "testing"

// TestFactoryWeightNames_Good pins the GraniteMoE factory tensor layout: the llama/mistral 2-norm
// coupling (post_attention_layernorm is the pre-MoE norm, no gemma-style sandwich norm), the router and
// fused-expert names pointed straight at the checkpoint's own already-packed tensors, and that the
// standard attention names are kept from StandardWeightNames.
func TestFactoryWeightNames_Good(t *testing.T) {
	w := FactoryWeightNames()

	if w.MLPNorm != ".post_attention_layernorm.weight" {
		t.Errorf("MLPNorm = %q, want .post_attention_layernorm.weight (llama/mistral 2-norm layout)", w.MLPNorm)
	}
	if w.PostAttnNorm != "" {
		t.Errorf("PostAttnNorm = %q, want \"\" — a non-empty value would apply a second (gemma-style) norm GraniteMoE does not carry", w.PostAttnNorm)
	}
	if w.NormBiasOne {
		t.Error("NormBiasOne = true, want false (GraniteMoE is plain RMSNorm, not gemma's +1 fold)")
	}

	if w.MoE.Router != ".block_sparse_moe.router.layer" {
		t.Errorf("MoE.Router = %q, want .block_sparse_moe.router.layer", w.MoE.Router)
	}
	if w.MoE.ExpGateUp != ".block_sparse_moe.input_linear" {
		t.Errorf("MoE.ExpGateUp = %q, want .block_sparse_moe.input_linear (checkpoint ships gate‖up already packed)", w.MoE.ExpGateUp)
	}
	if w.MoE.ExpDown != ".block_sparse_moe.output_linear" {
		t.Errorf("MoE.ExpDown = %q, want .block_sparse_moe.output_linear", w.MoE.ExpDown)
	}
	if w.MoE.ExpGate != "" || w.MoE.ExpUp != "" {
		t.Errorf("MoE.ExpGate/ExpUp = %q/%q, want both \"\" — GraniteMoE has no separate gate/up tensors, only the fused input_linear", w.MoE.ExpGate, w.MoE.ExpUp)
	}
	if w.MoE.PreFFNorm != ".post_attention_layernorm.weight" {
		t.Errorf("MoE.PreFFNorm = %q, want .post_attention_layernorm.weight", w.MoE.PreFFNorm)
	}

	if w.AttnNorm != ".input_layernorm.weight" {
		t.Errorf("AttnNorm = %q, want .input_layernorm.weight", w.AttnNorm)
	}
	if w.Q != ".self_attn.q_proj" || w.K != ".self_attn.k_proj" || w.V != ".self_attn.v_proj" || w.O != ".self_attn.o_proj" {
		t.Errorf("attention projections = %q/%q/%q/%q, want .self_attn.{q,k,v,o}_proj", w.Q, w.K, w.V, w.O)
	}
}

// TestFactoryWeightNames_Bad guards against GraniteMoE acquiring a shared expert by accident — real
// GraniteMoE checkpoints (as opposed to GraniteMoeShared, a DIFFERENT model_type this package does not
// register) carry no shared_expert.* tensors, so a stray non-empty name here would make assembleMoE
// probe a tensor that never exists (harmless — LoadLinear is nil-safe on a missing name — but a silent
// declaration drift worth pinning), and would silently mismatch Config.Arch's unconditional
// SharedExperts: 0.
func TestFactoryWeightNames_Bad(t *testing.T) {
	w := FactoryWeightNames()
	if w.MoE.SharedGate != "" || w.MoE.SharedUp != "" || w.MoE.SharedDown != "" || w.MoE.SharedSigmoid != "" {
		t.Fatalf("GraniteMoE declared a shared expert it does not have: %+v", w.MoE)
	}
}

// TestFactoryWeightNames_Ugly proves the router and fused-expert names never collide with each other — a
// collision would silently alias two roles onto one tensor. MLPNorm deliberately EQUALS MoE.PreFFNorm
// (both name post_attention_layernorm.weight) — not a collision: Assemble's spec.MoE branch consults only
// PreFFNorm (GraniteMoE has no dense layers), so MLPNorm's identical value is inert, not double-loaded.
func TestFactoryWeightNames_Ugly(t *testing.T) {
	w := FactoryWeightNames()
	names := map[string]string{"router": w.MoE.Router, "expGateUp": w.MoE.ExpGateUp, "expDown": w.MoE.ExpDown}
	seen := make(map[string]string, len(names))
	for role, name := range names {
		if other, ok := seen[name]; ok {
			t.Fatalf("role %q and %q share the same tensor name %q", role, other, name)
		}
		seen[name] = role
	}
	if w.MLPNorm != w.MoE.PreFFNorm {
		t.Errorf("MLPNorm = %q, MoE.PreFFNorm = %q — expected them equal (both name the pre-FFN norm; GraniteMoE has no dense-layer fallback to make MLPNorm's value matter)", w.MLPNorm, w.MoE.PreFFNorm)
	}
}
