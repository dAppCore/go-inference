// SPDX-Licence-Identifier: EUPL-1.2

package llama4_test

import (
	"bytes"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/arch/meta-llama/llama4"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// tinyLlama4Config mirrors TestTinyLlama4TextForwardAndGenerate_Good's inline config
// (integration_test.go): 2 layers (0 dense, 1 MoE — moe_layers:[1], the interleaved wrinkle none of
// dbrx/qwenmoe/mixtral's all-MoE checkpoints have), 2 experts, and — via tinyWeights() — a shared
// expert on the MoE layer. Reused here so the factory route below is proven against the identical
// fixture the composed route already passes (TestTinyLlama4TextForwardAndGenerate_Good) — same
// tensors, same config, two different load paths.
const tinyLlama4Config = `{"model_type":"llama4","text_config":{"model_type":"llama4_text","hidden_size":8,"intermediate_size":4,"intermediate_size_mlp":12,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"num_local_experts":2,"num_experts_per_tok":1,"moe_layers":[1],"no_rope_layers":[1,0],"vocab_size":32,"rms_norm_eps":1e-5,"rope_theta":500000,"use_qk_norm":true},"tie_word_embeddings":false}`

// writeTinyLlama4Dir materialises tensors + config as an on-disk checkpoint directory — the RAW
// checkpoint layout (language_model.model.*, feed_forward.*, packed 3-D expert tensors) model.Load
// reads straight off disk; model.Load runs NormalizeConfig (NormalizeWeights + packExperts) itself,
// so this fixture must NOT be pre-normalised.
func writeTinyLlama4Dir(t *testing.T, tensors map[string]safetensors.Tensor, config string) string {
	t.Helper()
	dir := t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), config); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	blob, err := safetensors.Encode(tensors)
	if err != nil {
		t.Fatalf("encode weights: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(blob)); err != nil {
		t.Fatalf("write model.safetensors: %v", err)
	}
	return dir
}

// TestTinyLlama4FactoryLoad_Good is the #50 bar for this arch: model.Load (the factory route —
// model.Assemble) now succeeds for Llama 4, where previously spec.Weights
// was the WeightNames zero value (every lookup missed) and model.Assemble rejected the checkpoint
// outright. It also proves the first half of the #18 parity method — "same tensor maps": the packed
// MoE expert weights model.Assemble loads are byte-identical to NormalizeWeights' own
// (separately-tested) decode of the packed checkpoint tensors, concatenated in expert-index order —
// AND that the interleaved dense/MoE layer pattern (moe_layers:[1]) routes each layer correctly.
func TestTinyLlama4FactoryLoad_Good(t *testing.T) {
	tensors := tinyWeights()
	dir := writeTinyLlama4Dir(t, tensors, tinyLlama4Config)

	loaded, mapping, err := model.Load(dir)
	if err != nil {
		t.Fatalf("model.Load: %v (Llama 4 must load through the factory route alone)", err)
	}
	defer func() { _ = mapping.Close() }()

	const hidden, expertFF, experts = 8, 4, 2
	if loaded.Arch.Experts != experts || loaded.Arch.TopK != 1 || loaded.Arch.ExpertFF != expertFF {
		t.Fatalf("Arch MoE geometry = experts %d topK %d expertFF %d, want %d/1/%d", loaded.Arch.Experts, loaded.Arch.TopK, loaded.Arch.ExpertFF, experts, expertFF)
	}
	if loaded.Arch.SharedExperts != 1 {
		t.Fatalf("Arch.SharedExperts = %d, want 1 (Llama 4 always carries a shared expert)", loaded.Arch.SharedExperts)
	}
	if len(loaded.Layers) != 2 {
		t.Fatalf("layers = %d, want 2", len(loaded.Layers))
	}

	// Layer 0 is dense (moe_layers:[1]).
	dense := loaded.Layers[0]
	if dense.MoE != nil {
		t.Fatal("layer 0 MoE non-nil — it is declared dense (moe_layers:[1])")
	}
	if dense.Gate == nil || dense.Up == nil || dense.Down == nil {
		t.Fatal("layer 0 dense MLP weights not loaded")
	}

	// Layer 1 is MoE, with a shared expert.
	moe := loaded.Layers[1]
	if moe.MoE == nil {
		t.Fatal("layer 1 MoE nil — Assemble did not route through the MoE branch")
	}
	if moe.MoE.Router == nil || moe.MoE.ExpGate == nil || moe.MoE.ExpUp == nil || moe.MoE.ExpDown == nil {
		t.Fatalf("routed MoE weights not loaded: router=%v gate=%v up=%v down=%v", moe.MoE.Router, moe.MoE.ExpGate, moe.MoE.ExpUp, moe.MoE.ExpDown)
	}
	if moe.MoE.SharedGate == nil || moe.MoE.SharedUp == nil || moe.MoE.SharedDown == nil {
		t.Fatalf("shared-expert weights not loaded: gate=%v up=%v down=%v", moe.MoE.SharedGate, moe.MoE.SharedUp, moe.MoE.SharedDown)
	}
	if moe.MoE.SharedSigmoid != nil {
		t.Error("SharedSigmoid should stay nil — Llama 4's shared expert has no sigmoid gate tensor")
	}
	if moe.MoE.ExpGate.OutDim != experts*expertFF || moe.MoE.ExpGate.InDim != hidden {
		t.Errorf("ExpGate dims = %d->%d, want %d->%d (packed across %d experts)", moe.MoE.ExpGate.InDim, moe.MoE.ExpGate.OutDim, hidden, experts*expertFF, experts)
	}
	// Dense-MLP fields stay unset on a MoE layer (Assemble's spec.MoE branch never touches them).
	if moe.Gate != nil || moe.Up != nil || moe.Down != nil {
		t.Error("a MoE layer should leave the dense Gate/Up/Down fields nil")
	}

	// "same tensor maps": the packed bytes equal NormalizeWeights' own per-expert decode of the
	// packed checkpoint tensors, concatenated in index order — proving packExperts didn't silently
	// re-derive the packed gate_up_proj/down_proj split differently from the tested Composed-route
	// decode.
	normalised, err := llama4.NormalizeWeights(tensors)
	if err != nil {
		t.Fatalf("NormalizeWeights: %v", err)
	}
	var wantGate, wantUp, wantDown []byte
	for e := 0; e < experts; e++ {
		wantGate = append(wantGate, normalised[core.Sprintf("language_model.model.layers.1.mlp.experts.%d.gate_proj.weight", e)].Data...)
		wantUp = append(wantUp, normalised[core.Sprintf("language_model.model.layers.1.mlp.experts.%d.up_proj.weight", e)].Data...)
		wantDown = append(wantDown, normalised[core.Sprintf("language_model.model.layers.1.mlp.experts.%d.down_proj.weight", e)].Data...)
	}
	if !bytes.Equal(moe.MoE.ExpGate.Weight, wantGate) {
		t.Error("ExpGate.Weight bytes != NormalizeWeights' per-expert gate_proj tensors concatenated in order")
	}
	if !bytes.Equal(moe.MoE.ExpUp.Weight, wantUp) {
		t.Error("ExpUp.Weight bytes != NormalizeWeights' per-expert up_proj tensors concatenated in order")
	}
	if !bytes.Equal(moe.MoE.ExpDown.Weight, wantDown) {
		t.Error("ExpDown.Weight bytes != NormalizeWeights' per-expert down_proj tensors concatenated in order")
	}
	if !bytes.Equal(moe.MoE.SharedGate.Weight, normalised["language_model.model.layers.1.mlp.shared_expert.gate_proj.weight"].Data) {
		t.Error("SharedGate.Weight bytes != source mlp.shared_expert.gate_proj.weight tensor")
	}
}

// --- synthetic checkpoint fixtures (moved from the deleted composed-route integration_test.go, #50) ---

func tinyWeights() map[string]safetensors.Tensor {
	const d, vocab, expertFF, denseFF, heads, kvHeads, headDim, experts = 8, 32, 4, 12, 2, 1, 4, 2
	norm := func() safetensors.Tensor { return tensor(d, d) }
	w := map[string]safetensors.Tensor{
		"language_model.model.embed_tokens.weight": tensor(vocab*d, vocab, d),
		"language_model.model.norm.weight":         norm(), "language_model.lm_head.weight": tensor(vocab*d, vocab, d),
	}
	for layer := range 2 {
		p := core.Sprintf("language_model.model.layers.%d.", layer)
		w[p+"input_layernorm.weight"], w[p+"post_attention_layernorm.weight"] = norm(), norm()
		w[p+"self_attn.q_proj.weight"] = tensor(heads*headDim*d, heads*headDim, d)
		w[p+"self_attn.k_proj.weight"] = tensor(kvHeads*headDim*d, kvHeads*headDim, d)
		w[p+"self_attn.v_proj.weight"] = tensor(kvHeads*headDim*d, kvHeads*headDim, d)
		w[p+"self_attn.o_proj.weight"] = tensor(d*heads*headDim, d, heads*headDim)
		if layer == 0 {
			w[p+"feed_forward.gate_proj.weight"] = tensor(denseFF*d, denseFF, d)
			w[p+"feed_forward.up_proj.weight"] = tensor(denseFF*d, denseFF, d)
			w[p+"feed_forward.down_proj.weight"] = tensor(d*denseFF, d, denseFF)
		} else {
			w[p+"feed_forward.router.weight"] = tensor(experts*d, experts, d)
			w[p+"feed_forward.experts.gate_up_proj"] = tensor(experts*d*2*expertFF, experts, d, 2*expertFF)
			w[p+"feed_forward.experts.down_proj"] = tensor(experts*expertFF*d, experts, expertFF, d)
			w[p+"feed_forward.shared_expert.gate_proj.weight"] = tensor(expertFF*d, expertFF, d)
			w[p+"feed_forward.shared_expert.up_proj.weight"] = tensor(expertFF*d, expertFF, d)
			w[p+"feed_forward.shared_expert.down_proj.weight"] = tensor(d*expertFF, d, expertFF)
		}
	}
	return w
}

func tensor(n int, shape ...int) safetensors.Tensor {
	data := make([]byte, n*2)
	state := uint32(n + 17)
	for i := range n {
		state = 1664525*state + 1013904223
		value := uint16(0x3c00 + (state>>29)&3)
		data[2*i], data[2*i+1] = byte(value), byte(value>>8)
	}
	return safetensors.Tensor{Dtype: "BF16", Shape: shape, Data: data}
}
