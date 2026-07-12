// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	core "dappco.re/go"
	basegguf "dappco.re/go/inference/model/gguf"
	"math/rand"
	"testing"
)

func TestRegister_Qwen3QuantizeLane_Good(t *testing.T) {
	root := t.TempDir()
	if r := core.WriteFile(core.PathJoin(root, "tokenizer.json"), []byte(`{"model":{"vocab":{"a":0},"merges":[]}}`), 0o600); !r.OK {
		t.Fatal(r.Err())
	}
	cfg := []byte(`{"model_type":"qwen3","max_position_embeddings":128,"hidden_size":16,"intermediate_size":32,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"vocab_size":1}`)
	rng := rand.New(rand.NewSource(43))
	values := make([]float32, 32)
	for i := range values {
		values[i] = rng.Float32()*4 - 2
	}
	got, meta, err := basegguf.NewTransformerQuantizeLane(qwen3Spec).Quantize(basegguf.Source{Root: root}, cfg, []basegguf.DenseSafetensor{{Name: "model.layers.0.self_attn.q_norm.weight", Shape: []uint64{32}, Data: values}}, basegguf.QuantizeQ8_0)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Name != "blk.0.attn_q_norm.weight" || got[0].Type != basegguf.TensorTypeF32 {
		t.Fatalf("tensor = %#v", got)
	}
	if meta[0].Value != "qwen3" {
		t.Fatalf("architecture = %v", meta[0].Value)
	}
}
func TestRegister_Qwen3QuantizeLane_Bad(t *testing.T) {
	if basegguf.NewTransformerQuantizeLane(qwen3Spec).Detect([]byte(`{"model_type":"qwen3_moe"}`)) {
		t.Fatal("MoE detected as dense qwen3")
	}
}
func TestRegister_Qwen3QuantizeLane_Ugly(t *testing.T) {
	if basegguf.NewTransformerQuantizeLane(qwen3Spec).SupportsFormat(basegguf.QuantizeQ4_0) {
		t.Fatal("q4_0 unexpectedly supported")
	}
}
