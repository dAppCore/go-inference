// SPDX-Licence-Identifier: EUPL-1.2
package glm4

import (
	"dappco.re/go/inference/model"
	"testing"
)

const realConfig = `{"model_type":"glm4","hidden_size":4096,"intermediate_size":13696,"num_hidden_layers":40,"num_attention_heads":32,"num_key_value_heads":2,"head_dim":128,"vocab_size":151552,"rms_norm_eps":0.00001,"rope_theta":10000,"partial_rotary_factor":0.5,"tie_word_embeddings":false}` // huggingface.co/zai-org/GLM-4-9B-0414
func TestConfig_Arch_Good(t *testing.T) {
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
func TestConfig_Arch_Bad(t *testing.T) {
	if _, e := (&Config{ModelType: "glm4"}).Arch(); e == nil {
		t.Fatal("empty dimensions accepted")
	}
}
func TestConfig_Arch_Ugly(t *testing.T) {
	s, _ := model.LookupArch("glm4")
	if _, e := s.Parse([]byte(`{`)); e == nil {
		t.Fatal("malformed JSON accepted")
	}
}
