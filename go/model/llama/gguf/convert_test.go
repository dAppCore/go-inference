// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"math/rand"
	"testing"

	core "dappco.re/go"
	basegguf "dappco.re/go/inference/model/gguf"
)

func TestConvert_isLlamaConfig_Good(t *testing.T) {
	if !isLlamaConfig([]byte(`{"model_type":"llama"}`)) {
		t.Fatal("llama config not detected")
	}
}

func TestConvert_isLlamaConfig_Bad(t *testing.T) {
	if isLlamaConfig([]byte(`{"model_type":"mistral"}`)) {
		t.Fatal("mistral config detected as llama")
	}
}

func TestConvert_isLlamaConfig_Ugly(t *testing.T) {
	if isLlamaConfig([]byte(`{`)) {
		t.Fatal("malformed config detected as llama")
	}
}

func TestConvert_isLlamaSupportedQuantizeFormat_Good(t *testing.T) {
	if !isLlamaSupportedQuantizeFormat(basegguf.QuantizeQ4_K_M) || !isLlamaSupportedQuantizeFormat(basegguf.QuantizeQ8_0) {
		t.Fatal("q4_k_m and q8_0 must be supported")
	}
}

func TestConvert_isLlamaSupportedQuantizeFormat_Bad(t *testing.T) {
	if isLlamaSupportedQuantizeFormat(basegguf.QuantizeQ6_K) {
		t.Fatal("q6_k must remain outside the first lane")
	}
}

// TestConvert_quantizeLlamaModelPack_Good uses varied seeded values rather
// than constant fills, so every encoder sees positive, negative, and changing
// magnitudes while the receipt stays deterministic.
func TestConvert_quantizeLlamaModelPack_Good(t *testing.T) {
	root := t.TempDir()
	tokenizer := []byte(`{"model":{"vocab":{"a":0,"b":1,"c":2,"d":3},"merges":[["a","b"]]},"added_tokens":[]}`)
	if result := core.WriteFile(core.PathJoin(root, "tokenizer.json"), tokenizer, 0o600); !result.OK {
		t.Fatalf("write tokenizer.json: %v", result.Err())
	}
	config := []byte(`{"model_type":"llama","max_position_embeddings":128,"hidden_size":16,"intermediate_size":32,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"vocab_size":4,"rms_norm_eps":1e-5,"rope_theta":10000,"bos_token_id":0,"eos_token_id":1}`)
	random := rand.New(rand.NewSource(32))
	values := func(count int) []float32 {
		out := make([]float32, count)
		for i := range out {
			out[i] = random.Float32()*4 - 2
		}
		return out
	}
	tensors := []basegguf.DenseSafetensor{
		{Name: "model.embed_tokens.weight", Shape: []uint64{16, 16}, Data: values(256)},
		{Name: "model.layers.0.self_attn.q_proj.weight", Shape: []uint64{16, 16}, Data: values(256)},
		{Name: "model.layers.0.input_layernorm.weight", Shape: []uint64{16}, Data: values(16)},
	}
	got, metadata, err := quantizeLlamaModelPack(basegguf.Source{Root: root}, config, tensors, basegguf.QuantizeQ4_K_M)
	if err != nil {
		t.Fatalf("quantizeLlamaModelPack: %v", err)
	}
	if len(got) != 3 || got[0].Name != "token_embd.weight" || got[1].Name != "blk.0.attn_q.weight" {
		t.Fatalf("canonical tensors = %#v", got)
	}
	if got[0].Type != basegguf.TensorTypeQ4K || got[1].Type != basegguf.TensorTypeQ4K || got[2].Type != basegguf.TensorTypeF32 {
		t.Fatalf("tensor types = [%d %d %d], want [Q4_K Q4_K F32]", got[0].Type, got[1].Type, got[2].Type)
	}
	foundArchitecture := false
	for _, entry := range metadata {
		foundArchitecture = foundArchitecture || entry.Key == "general.architecture" && entry.Value == "llama"
	}
	if !foundArchitecture {
		t.Fatal("metadata lacks general.architecture=llama")
	}
}
