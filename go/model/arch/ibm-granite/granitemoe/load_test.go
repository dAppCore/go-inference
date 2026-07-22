// SPDX-Licence-Identifier: EUPL-1.2

package granitemoe

import (
	"bytes"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// tinyGraniteMoEConfig is the same synthetic checkpoint shape TestTinyGraniteMoEForward_Good (the
// Composed route, integration_test.go) exercises: 1 layer, 4 experts, top-2. Reused here so the factory
// route below is proven against the identical fixture the composed route already passes — same tensors,
// same config, two different load paths.
const tinyGraniteMoEConfig = `{"model_type":"granitemoe","hidden_size":8,"intermediate_size":4,"num_hidden_layers":1,` +
	`"num_attention_heads":2,"num_key_value_heads":1,"num_local_experts":4,"num_experts_per_tok":2,` +
	`"vocab_size":32,"rms_norm_eps":0.00001,"rope_theta":10000,"tie_word_embeddings":true,"hidden_act":"silu",` +
	`"logits_scaling":6,"residual_multiplier":0.22,"embedding_multiplier":12,"attention_multiplier":0.125}`

// writeTinyGraniteMoEDir materialises tinyWeights() + tinyGraniteMoEConfig as a checkpoint directory.
func writeTinyGraniteMoEDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), tinyGraniteMoEConfig); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	blob, err := safetensors.Encode(tinyWeights())
	if err != nil {
		t.Fatalf("encode weights: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(blob)); err != nil {
		t.Fatalf("write model.safetensors: %v", err)
	}
	return dir
}

// TestTinyGraniteMoEFactoryLoad_Good is the #50 bar for this arch: model.Load (the factory route —
// model.Assemble + arch_session) now succeeds for GraniteMoE, where it used
// to fail (Weights was the zero-value model.WeightNames{}, so Assemble rejected the checkpoint as missing
// model.embed_tokens rather than routing it correctly). It also proves the "same tensor maps" parity
// method: the loaded MoE weights are byte-identical to the checkpoint's own input_linear/output_linear/
// router.layer tensors — GraniteMoE ships them already packed, so unlike Mixtral there is no concatenation
// step, just naming.
func TestTinyGraniteMoEFactoryLoad_Good(t *testing.T) {
	dir := writeTinyGraniteMoEDir(t)

	loaded, mapping, err := model.Load(dir)
	if err != nil {
		t.Fatalf("model.Load: %v (GraniteMoE must load through the factory route alone)", err)
	}
	defer func() { _ = mapping.Close() }()

	const experts, topK, ff = 4, 2, 4
	if loaded.Arch.Experts != experts || loaded.Arch.TopK != topK || loaded.Arch.ExpertFF != ff {
		t.Fatalf("Arch MoE geometry = experts %d topK %d expertFF %d, want %d/%d/%d", loaded.Arch.Experts, loaded.Arch.TopK, loaded.Arch.ExpertFF, experts, topK, ff)
	}
	if loaded.Arch.SharedExperts != 0 {
		t.Fatalf("Arch.SharedExperts = %d, want 0 (GraniteMoE has no shared expert)", loaded.Arch.SharedExperts)
	}
	if len(loaded.Layers) != 1 {
		t.Fatalf("layers = %d, want 1", len(loaded.Layers))
	}
	L := loaded.Layers[0]
	if L.MoE == nil {
		t.Fatal("layer 0 MoE nil — Assemble did not route through the MoE branch")
	}
	if L.MoE.Router == nil || L.MoE.ExpGateUp == nil || L.MoE.ExpDown == nil {
		t.Fatalf("MoE weights not loaded: router=%v expGateUp=%v expDown=%v", L.MoE.Router, L.MoE.ExpGateUp, L.MoE.ExpDown)
	}
	if L.MoE.ExpGate != nil || L.MoE.ExpUp != nil {
		t.Error("GraniteMoE has no separate gate/up tensors — ExpGate/ExpUp should stay nil (only the fused ExpGateUp loads)")
	}
	if L.MoE.SharedGate != nil || L.MoE.SharedUp != nil || L.MoE.SharedDown != nil {
		t.Error("GraniteMoE has no shared expert — Shared* fields should stay nil")
	}
	// Dense-MLP fields stay unset on a MoE layer (Assemble's spec.MoE branch never touches them).
	if L.Gate != nil || L.Up != nil || L.Down != nil {
		t.Error("a MoE layer should leave the dense Gate/Up/Down fields nil")
	}

	// "same tensor maps": GraniteMoE ships its packed tensors NATIVELY, so the factory route's bytes must
	// be byte-identical to the checkpoint's OWN input_linear/output_linear/router.layer.weight tensors —
	// no concatenation, no transformation, just naming.
	tensors := tinyWeights()
	wantGateUp := tensors["model.layers.0.block_sparse_moe.input_linear.weight"].Data
	wantDown := tensors["model.layers.0.block_sparse_moe.output_linear.weight"].Data
	wantRouter := tensors["model.layers.0.block_sparse_moe.router.layer.weight"].Data
	if !bytes.Equal(L.MoE.ExpGateUp.Weight, wantGateUp) {
		t.Error("ExpGateUp.Weight bytes != source input_linear.weight tensor")
	}
	if !bytes.Equal(L.MoE.ExpDown.Weight, wantDown) {
		t.Error("ExpDown.Weight bytes != source output_linear.weight tensor")
	}
	if !bytes.Equal(L.MoE.Router.Weight, wantRouter) {
		t.Error("Router.Weight bytes != source router.layer.weight tensor")
	}
}

// --- synthetic checkpoint fixtures (moved from the deleted composed-route integration_test.go, #50) ---

func tinyWeights() map[string]safetensors.Tensor {
	const hidden, vocab, ff, heads, kvHeads, headDim, experts = 8, 32, 4, 2, 1, 4, 4
	seed := seededValues{state: 0x6772616e}
	norm := func() safetensors.Tensor {
		values := seed.next(hidden)
		for i := range values {
			values[i] += 1
		}
		return tensor(values, hidden)
	}
	return map[string]safetensors.Tensor{
		"model.embed_tokens.weight":                            tensor(seed.next(vocab*hidden), vocab, hidden),
		"model.norm.weight":                                    norm(),
		"model.layers.0.input_layernorm.weight":                norm(),
		"model.layers.0.post_attention_layernorm.weight":       norm(),
		"model.layers.0.self_attn.q_proj.weight":               tensor(seed.next(heads*headDim*hidden), heads*headDim, hidden),
		"model.layers.0.self_attn.k_proj.weight":               tensor(seed.next(kvHeads*headDim*hidden), kvHeads*headDim, hidden),
		"model.layers.0.self_attn.v_proj.weight":               tensor(seed.next(kvHeads*headDim*hidden), kvHeads*headDim, hidden),
		"model.layers.0.self_attn.o_proj.weight":               tensor(seed.next(hidden*heads*headDim), hidden, heads*headDim),
		"model.layers.0.block_sparse_moe.input_linear.weight":  tensor(seed.next(experts*2*ff*hidden), experts, 2*ff, hidden),
		"model.layers.0.block_sparse_moe.output_linear.weight": tensor(seed.next(experts*hidden*ff), experts, hidden, ff),
		"model.layers.0.block_sparse_moe.router.layer.weight":  tensor(seed.next(experts*hidden), experts, hidden),
	}
}

type seededValues struct{ state uint32 }

func (s *seededValues) next(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		s.state = 1664525*s.state + 1013904223
		out[i] = float32(int32(s.state>>24)-128) / 256
	}
	return out
}
