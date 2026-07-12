// SPDX-Licence-Identifier: EUPL-1.2
package exaone4

import (
	"dappco.re/go/inference/model"
	"testing"
)

const realConfig = `{"model_type":"exaone4","hidden_size":5120,"intermediate_size":27392,"num_hidden_layers":64,"num_attention_heads":40,"num_key_value_heads":8,"head_dim":128,"vocab_size":102400,"rms_norm_eps":0.00001,"rope_theta":1000000,"sliding_window":4096,"sliding_window_pattern":"LLLG","rope_scaling":{"rope_type":"llama3","factor":16,"low_freq_factor":1,"high_freq_factor":4,"original_max_position_embeddings":8192},"tie_word_embeddings":false}` // huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B
func TestConfig_Arch_Good(t *testing.T) {
	s, _ := model.LookupArch("exaone4")
	c, e := s.Parse([]byte(realConfig))
	if e != nil {
		t.Fatal(e)
	}
	a, e := c.Arch()
	if e != nil || a.Layer[0].Attention != model.SlidingAttention || a.Layer[3].Attention != model.GlobalAttention || a.NormPlacement != model.NormPlacementPost {
		t.Fatalf("arch=%+v err=%v", a, e)
	}
	if s.Weights.AttnNorm != "" || s.Weights.PostFFNorm == "" {
		t.Fatal("EXAONE index norm placement not declared")
	}
}
func TestConfig_Arch_Bad(t *testing.T) {
	if _, e := (&Config{ModelType: "exaone4"}).Arch(); e == nil {
		t.Fatal("empty dimensions accepted")
	}
}
func TestConfig_Arch_Ugly(t *testing.T) {
	s, _ := model.LookupArch("exaone4")
	if _, e := s.Parse([]byte(`{`)); e == nil {
		t.Fatal("malformed JSON accepted")
	}
}
