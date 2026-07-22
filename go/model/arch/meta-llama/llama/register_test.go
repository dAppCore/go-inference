// SPDX-Licence-Identifier: EUPL-1.2

package llama

import (
	"testing"

	"dappco.re/go/inference/model"
)

func TestLlamaRegistered_Good(t *testing.T) {
	spec, ok := model.LookupArch("llama")
	if !ok {
		t.Fatal("model_type llama is not registered")
	}
	if spec.Weights.Embed != "model.embed_tokens" || spec.Weights.LMHead != "lm_head" || spec.Weights.FinalNorm != "model.norm.weight" {
		t.Fatalf("model weight names = %+v", spec.Weights)
	}
	if spec.Weights.LayerPrefix != "model.layers.%d" ||
		spec.Weights.AttnNorm != ".input_layernorm.weight" ||
		spec.Weights.MLPNorm != ".post_attention_layernorm.weight" {
		t.Fatalf("layer/norm weight names = %+v", spec.Weights)
	}
	if spec.Weights.Q != ".self_attn.q_proj" || spec.Weights.K != ".self_attn.k_proj" ||
		spec.Weights.V != ".self_attn.v_proj" || spec.Weights.O != ".self_attn.o_proj" {
		t.Fatalf("attention weight names = %+v", spec.Weights)
	}
	if spec.Weights.Gate != ".mlp.gate_proj" || spec.Weights.Up != ".mlp.up_proj" || spec.Weights.Down != ".mlp.down_proj" {
		t.Fatalf("MLP weight names = %+v", spec.Weights)
	}
	if spec.Weights.PostAttnNorm != "" || spec.Weights.PostFFNorm != "" || spec.Weights.NormBiasOne {
		t.Fatalf("non-Llama norm roles enabled: %+v", spec.Weights)
	}
}

func TestLlamaRegistered_Bad(t *testing.T) {
	spec, ok := model.LookupArch("llama")
	if !ok {
		t.Fatal("model_type llama is not registered")
	}
	if _, err := spec.Parse([]byte("not json")); err == nil {
		t.Fatal("malformed config accepted")
	}
}

func TestLlamaRegistered_Ugly(t *testing.T) {
	spec, ok := model.LookupArch("llama")
	if !ok {
		t.Fatal("model_type llama is not registered")
	}
	cfg, err := spec.Parse([]byte(`{"model_type":"llama","hidden_size":64,"intermediate_size":128,` +
		`"num_hidden_layers":1,"num_attention_heads":8,"num_key_value_heads":2,"vocab_size":32}`))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	arch, err := cfg.Arch()
	if err != nil || arch.HeadDim != 8 || arch.KVHeads != 2 {
		t.Fatalf("Arch = headDim %d kvHeads %d, err %v", arch.HeadDim, arch.KVHeads, err)
	}
}
