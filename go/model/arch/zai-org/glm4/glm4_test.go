// SPDX-Licence-Identifier: EUPL-1.2
package glm4

import (
	"dappco.re/go/inference/model"
	"testing"
)

const realConfig = `{"model_type":"glm4","hidden_size":4096,"intermediate_size":13696,"num_hidden_layers":40,"num_attention_heads":32,"num_key_value_heads":2,"head_dim":128,"vocab_size":151552,"rms_norm_eps":0.00001,"rope_theta":10000,"partial_rotary_factor":0.5,"tie_word_embeddings":false}` // huggingface.co/zai-org/GLM-4-9B-0414
func TestGlm4_Config_Arch_Good(t *testing.T) {
	s, _ := model.LookupArch("glm4")
	c, e := s.Parse([]byte(realConfig))
	if e != nil {
		t.Fatal(e)
	}
	a, e := c.Arch()
	if e != nil || a.RotaryDim != 64 || a.KVHeads != 2 {
		t.Fatalf("arch=%+v err=%v", a, e)
	}
	if s.Normalize == nil || s.Weights.PostAttnNorm != ".post_self_attn_layernorm.weight" {
		t.Fatal("GLM-4 index differences not declared")
	}
}
func TestGlm4_Config_Arch_Bad(t *testing.T) {
	if _, e := (&Config{ModelType: "glm4"}).Arch(); e == nil {
		t.Fatal("empty dimensions accepted")
	}
}

// TestGlm4_Config_Arch_Ugly pins the GQA head-divisibility guard: a syntactically
// valid, otherwise-complete config whose head count does not divide evenly by
// its kv-head count must be rejected by Arch() itself (distinct from _Bad's
// all-zero-fields rejection).
func TestGlm4_Config_Arch_Ugly(t *testing.T) {
	s, _ := model.LookupArch("glm4")
	c, e := s.Parse([]byte(`{"model_type":"glm4","hidden_size":16,"intermediate_size":32,"num_hidden_layers":1,"num_attention_heads":7,"num_key_value_heads":2,"head_dim":16,"vocab_size":8,"rms_norm_eps":0.00001,"rope_theta":10000}`))
	if e != nil {
		t.Fatal(e)
	}
	if _, e := c.Arch(); e == nil {
		t.Fatal("heads (7) not divisible by kv-heads (2) accepted")
	}
}

func TestGlm4_Config_InferFromWeights_Good(t *testing.T) {
	c := Config{HiddenSize: 4096}
	c.InferFromWeights(nil)
	if c.HiddenSize != 4096 {
		t.Fatalf("InferFromWeights changed config: %+v", c)
	}
}

func TestGlm4_Config_InferFromWeights_Bad(t *testing.T) {
	c := Config{}
	c.InferFromWeights(nil)
	if _, e := c.Arch(); e == nil {
		t.Fatal("empty config became valid after InferFromWeights")
	}
}

// TestGlm4_Config_InferFromWeights_Ugly proves the no-op does not paper over
// the GQA head-divisibility guard — distinct from _Bad's all-zero case.
func TestGlm4_Config_InferFromWeights_Ugly(t *testing.T) {
	c := Config{ModelType: "glm4", HiddenSize: 16, IntermediateSize: 32, NumHiddenLayers: 1, NumAttentionHeads: 7, NumKeyValueHeads: 2, HeadDim: 16, VocabSize: 8, RMSNormEps: 0.00001, RopeTheta: 10000}
	c.InferFromWeights(nil)
	if _, e := c.Arch(); e == nil {
		t.Fatal("heads not divisible by kv-heads became valid after InferFromWeights")
	}
}
