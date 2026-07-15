// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	core "dappco.re/go"
	basegguf "dappco.re/go/inference/model/gguf"
	"math/rand"
	"testing"
)

func TestRegister_Phi3QuantizeLane_Good(t *testing.T) {
	root := t.TempDir()
	if r := core.WriteFile(core.PathJoin(root, "tokenizer.json"), []byte(`{"model":{"vocab":{"a":0},"merges":[]}}`), 0o600); !r.OK {
		t.Fatal(r.Err())
	}
	cfg := []byte(`{"model_type":"phi3","max_position_embeddings":128,"hidden_size":16,"intermediate_size":32,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":2,"vocab_size":1,"rms_norm_eps":0.00001}`)
	rng := rand.New(rand.NewSource(47))
	values := make([]float32, 256)
	for i := range values {
		values[i] = rng.Float32()*4 - 2
	}
	got, meta, err := basegguf.NewTransformerQuantizeLane(phiSpec).Quantize(basegguf.Source{Root: root}, cfg, []basegguf.DenseSafetensor{{Name: "model.layers.0.mlp.down_proj.weight", Shape: []uint64{16, 16}, Data: values}}, basegguf.QuantizeQ4_K_M)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Name != "blk.0.ffn_down.weight" || got[0].Type != basegguf.TensorTypeQ6K {
		t.Fatalf("tensor = %#v", got)
	}
	if meta[0].Value != "phi3" {
		t.Fatalf("architecture = %v", meta[0].Value)
	}
}
func TestRegister_Phi3QuantizeLane_Bad(t *testing.T) {
	if basegguf.NewTransformerQuantizeLane(phiSpec).Detect([]byte(`{"model_type":"phi"}`)) {
		t.Fatal("Phi-2 detected as Phi-3")
	}
}
func TestRegister_Phi3QuantizeLane_Ugly(t *testing.T) {
	if basegguf.NewTransformerQuantizeLane(phiSpec).SupportsFormat("") {
		t.Fatal("empty format supported")
	}
}
