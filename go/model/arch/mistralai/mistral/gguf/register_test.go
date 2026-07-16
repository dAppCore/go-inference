// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"math/rand"
	"testing"

	core "dappco.re/go"
	basegguf "dappco.re/go/inference/model/gguf"
)

func TestRegister_TransformerQuantizeLane_Good(t *testing.T) {
	if !basegguf.NewTransformerQuantizeLane(mistralSpec).Detect([]byte(`{"model_type":"mistral3"}`)) {
		t.Fatal("mistral3 config not detected")
	}
	testMistralLane(t, basegguf.QuantizeQ4_K_M, basegguf.TensorTypeQ4K)
	testMistralLane(t, basegguf.QuantizeQ8_0, basegguf.TensorTypeQ8_0)
}

func TestRegister_TransformerQuantizeLane_Bad(t *testing.T) {
	if basegguf.NewTransformerQuantizeLane(mistralSpec).SupportsFormat(basegguf.QuantizeQ6_K) {
		t.Fatal("q6_k unexpectedly supported")
	}
}

func TestRegister_TransformerQuantizeLane_Ugly(t *testing.T) {
	if basegguf.NewTransformerQuantizeLane(mistralSpec).Detect([]byte(`{`)) {
		t.Fatal("malformed config detected")
	}
}

func testMistralLane(t *testing.T, format basegguf.QuantizeFormat, want uint32) {
	t.Helper()
	root := t.TempDir()
	if r := core.WriteFile(core.PathJoin(root, "tokenizer.json"), []byte(`{"model":{"vocab":{"a":0,"b":1},"merges":[["a","b"]]}}`), 0o600); !r.OK {
		t.Fatal(r.Err())
	}
	config := []byte(`{"model_type":"mistral3","max_position_embeddings":128,"hidden_size":16,"intermediate_size":32,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":2,"rms_norm_eps":0.00001,"rope_theta":10000}`)
	rng := rand.New(rand.NewSource(41))
	values := make([]float32, 256)
	for i := range values {
		values[i] = rng.Float32()*4 - 2
	}
	got, metadata, err := basegguf.NewTransformerQuantizeLane(mistralSpec).Quantize(basegguf.Source{Root: root}, config, []basegguf.DenseSafetensor{{Name: "model.layers.0.self_attn.q_proj.weight", Shape: []uint64{16, 16}, Data: values}}, format)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Name != "blk.0.attn_q.weight" || got[0].Type != want {
		t.Fatalf("tensor = %#v", got)
	}
	if metadata[0].Key != "general.architecture" || metadata[0].Value != "mistral3" {
		t.Fatalf("metadata = %#v", metadata[0])
	}
}
