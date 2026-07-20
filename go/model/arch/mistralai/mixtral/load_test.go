// SPDX-Licence-Identifier: EUPL-1.2

package mixtral_test

import (
	"bytes"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// tinyMixtralConfig is the same synthetic checkpoint shape TestTinyMixtralForwardAndGenerate_Good (the
// Composed route) exercises: 1 layer, 2 experts, top-1. Reused here so the factory route below is proven
// against the identical fixture the composed route already passes — same tensors, same config, two
// different load paths.
const tinyMixtralConfig = `{"model_type":"mixtral","hidden_size":8,"intermediate_size":12,"num_hidden_layers":1,` +
	`"num_attention_heads":2,"num_key_value_heads":1,"num_local_experts":2,"num_experts_per_tok":1,` +
	`"vocab_size":32,"rms_norm_eps":1e-5,"rope_theta":10000,"tie_word_embeddings":false}`

// writeTinyMixtralDir materialises tinyMixtralWeights() + tinyMixtralConfig as a checkpoint directory.
func writeTinyMixtralDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), tinyMixtralConfig); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	blob, err := safetensors.Encode(tinyMixtralWeights())
	if err != nil {
		t.Fatalf("encode weights: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(blob)); err != nil {
		t.Fatalf("write model.safetensors: %v", err)
	}
	return dir
}

// TestTinyMixtralFactoryLoad_Good is the #50 bar for this arch: model.Load (the factory route —
// model.Assemble + arch_session) now succeeds for Mixtral, where it used to
// reject it as composed-only (see model.Load's "is a composed/hybrid arch" check). It also proves the
// first half of the #18 parity method — "same tensor maps": the packed MoE expert weights model.Assemble
// loads are byte-identical to the checkpoint's own per-expert tensors, concatenated in expert-index order,
// not re-derived or approximated.
func TestTinyMixtralFactoryLoad_Good(t *testing.T) {
	dir := writeTinyMixtralDir(t)

	loaded, mapping, err := model.Load(dir)
	if err != nil {
		t.Fatalf("model.Load: %v (Mixtral must load through the factory route alone)", err)
	}
	defer func() { _ = mapping.Close() }()

	const hidden, ff, experts = 8, 12, 2
	if loaded.Arch.Experts != experts || loaded.Arch.TopK != 1 || loaded.Arch.ExpertFF != ff {
		t.Fatalf("Arch MoE geometry = experts %d topK %d expertFF %d, want %d/1/%d", loaded.Arch.Experts, loaded.Arch.TopK, loaded.Arch.ExpertFF, experts, ff)
	}
	if len(loaded.Layers) != 1 {
		t.Fatalf("layers = %d, want 1", len(loaded.Layers))
	}
	L := loaded.Layers[0]
	if L.MoE == nil {
		t.Fatal("layer 0 MoE nil — Assemble did not route through the MoE branch")
	}
	if L.MoE.Router == nil || L.MoE.ExpGate == nil || L.MoE.ExpUp == nil || L.MoE.ExpDown == nil {
		t.Fatalf("MoE weights not loaded: router=%v gate=%v up=%v down=%v", L.MoE.Router, L.MoE.ExpGate, L.MoE.ExpUp, L.MoE.ExpDown)
	}
	if L.MoE.ExpGate.OutDim != experts*ff || L.MoE.ExpGate.InDim != hidden {
		t.Errorf("ExpGate dims = %d->%d, want %d->%d (packed across %d experts)", L.MoE.ExpGate.InDim, L.MoE.ExpGate.OutDim, hidden, experts*ff, experts)
	}
	if L.MoE.ExpDown.OutDim != experts*hidden || L.MoE.ExpDown.InDim != ff {
		t.Errorf("ExpDown dims = %d->%d, want %d->%d", L.MoE.ExpDown.InDim, L.MoE.ExpDown.OutDim, ff, experts*hidden)
	}
	// Dense-MLP fields stay unset on a MoE layer (Assemble's spec.MoE branch never touches them).
	if L.Gate != nil || L.Up != nil || L.Down != nil {
		t.Error("a MoE layer should leave the dense Gate/Up/Down fields nil")
	}

	// "same tensor maps": packed bytes == the source per-expert tensors, concatenated in index order.
	tensors := tinyMixtralWeights()
	var wantGate, wantUp, wantDown []byte
	for e := 0; e < experts; e++ {
		wantGate = append(wantGate, tensors[core.Sprintf("model.layers.0.block_sparse_moe.experts.%d.w1.weight", e)].Data...)
		wantUp = append(wantUp, tensors[core.Sprintf("model.layers.0.block_sparse_moe.experts.%d.w3.weight", e)].Data...)
		wantDown = append(wantDown, tensors[core.Sprintf("model.layers.0.block_sparse_moe.experts.%d.w2.weight", e)].Data...)
	}
	if !bytes.Equal(L.MoE.ExpGate.Weight, wantGate) {
		t.Error("ExpGate.Weight bytes != source per-expert w1 tensors concatenated in order")
	}
	if !bytes.Equal(L.MoE.ExpUp.Weight, wantUp) {
		t.Error("ExpUp.Weight bytes != source per-expert w3 tensors concatenated in order")
	}
	if !bytes.Equal(L.MoE.ExpDown.Weight, wantDown) {
		t.Error("ExpDown.Weight bytes != source per-expert w2 tensors concatenated in order")
	}
}

// --- synthetic checkpoint fixtures (moved from the deleted composed-route integration_test.go, #50) ---

type seededWeights struct{ state uint32 }

func (s *seededWeights) values(n int) []uint16 {
	out := make([]uint16, n)
	for i := range out {
		s.state = 1664525*s.state + 1013904223
		out[i] = 0x3c00 + uint16((s.state>>28)&7)
	}
	return out
}

func tinyMixtralWeights() map[string]safetensors.Tensor {
	const hidden, vocab, expertFF, heads, kvHeads, headDim, experts = 8, 32, 12, 2, 1, 4, 2
	s := seededWeights{state: 0x5eed1234}
	norm := func() safetensors.Tensor {
		values := make([]uint16, hidden)
		for i := range values {
			values[i] = 0x3f80
		}
		return bf16Tensor(values, hidden)
	}
	tensors := map[string]safetensors.Tensor{
		"model.embed_tokens.weight":                      bf16Tensor(s.values(vocab*hidden), vocab, hidden),
		"model.norm.weight":                              norm(),
		"lm_head.weight":                                 bf16Tensor(s.values(vocab*hidden), vocab, hidden),
		"model.layers.0.input_layernorm.weight":          norm(),
		"model.layers.0.post_attention_layernorm.weight": norm(),
		"model.layers.0.self_attn.q_proj.weight":         bf16Tensor(s.values(heads*headDim*hidden), heads*headDim, hidden),
		"model.layers.0.self_attn.k_proj.weight":         bf16Tensor(s.values(kvHeads*headDim*hidden), kvHeads*headDim, hidden),
		"model.layers.0.self_attn.v_proj.weight":         bf16Tensor(s.values(kvHeads*headDim*hidden), kvHeads*headDim, hidden),
		"model.layers.0.self_attn.o_proj.weight":         bf16Tensor(s.values(hidden*heads*headDim), hidden, heads*headDim),
		"model.layers.0.block_sparse_moe.gate.weight":    bf16Tensor(s.values(experts*hidden), experts, hidden),
	}
	for expert := range experts {
		prefix := core.Sprintf("model.layers.0.block_sparse_moe.experts.%d", expert)
		tensors[prefix+".w1.weight"] = bf16Tensor(s.values(expertFF*hidden), expertFF, hidden)
		tensors[prefix+".w2.weight"] = bf16Tensor(s.values(hidden*expertFF), hidden, expertFF)
		tensors[prefix+".w3.weight"] = bf16Tensor(s.values(expertFF*hidden), expertFF, hidden)
	}
	return tensors
}

func bf16Tensor(values []uint16, shape ...int) safetensors.Tensor {
	data := make([]byte, len(values)*2)
	for i, value := range values {
		data[2*i], data[2*i+1] = byte(value), byte(value>>8)
	}
	return safetensors.Tensor{Dtype: "BF16", Shape: shape, Data: data}
}
