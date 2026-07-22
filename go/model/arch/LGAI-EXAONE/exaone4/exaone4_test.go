// SPDX-Licence-Identifier: EUPL-1.2
package exaone4

import (
	"dappco.re/go/inference/model"
	"testing"
)

const realConfig = `{"model_type":"exaone4","hidden_size":5120,"intermediate_size":27392,"num_hidden_layers":64,"num_attention_heads":40,"num_key_value_heads":8,"head_dim":128,"vocab_size":102400,"rms_norm_eps":0.00001,"rope_theta":1000000,"sliding_window":4096,"sliding_window_pattern":"LLLG","rope_scaling":{"rope_type":"llama3","factor":16,"low_freq_factor":1,"high_freq_factor":4,"original_max_position_embeddings":8192},"tie_word_embeddings":false}` // huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B
func TestExaone4_Config_Arch_Good(t *testing.T) {
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
func TestExaone4_Config_Arch_Bad(t *testing.T) {
	if _, e := (&Config{ModelType: "exaone4"}).Arch(); e == nil {
		t.Fatal("empty dimensions accepted")
	}
}

// TestExaone4_Config_Arch_Ugly pins the GQA head-divisibility guard: a syntactically
// valid, otherwise-complete config whose head count does not divide evenly by
// its kv-head count must be rejected by Arch() itself (distinct from _Bad's
// all-zero-fields rejection).
func TestExaone4_Config_Arch_Ugly(t *testing.T) {
	s, _ := model.LookupArch("exaone4")
	c, e := s.Parse([]byte(`{"model_type":"exaone4","hidden_size":16,"intermediate_size":32,"num_hidden_layers":1,"num_attention_heads":7,"num_key_value_heads":2,"head_dim":16,"vocab_size":8,"rms_norm_eps":0.00001,"rope_theta":10000}`))
	if e != nil {
		t.Fatal(e)
	}
	if _, e := c.Arch(); e == nil {
		t.Fatal("heads (7) not divisible by kv-heads (2) accepted")
	}
}

func TestExaone4_Config_InferFromWeights_Good(t *testing.T) {
	c := Config{HiddenSize: 5120}
	c.InferFromWeights(nil)
	if c.HiddenSize != 5120 {
		t.Fatalf("InferFromWeights changed config: %+v", c)
	}
}

func TestExaone4_Config_InferFromWeights_Bad(t *testing.T) {
	c := Config{}
	c.InferFromWeights(nil)
	if _, e := c.Arch(); e == nil {
		t.Fatal("empty config became valid after InferFromWeights")
	}
}

// TestExaone4_Config_InferFromWeights_Ugly proves the no-op does not paper
// over the GQA head-divisibility guard — distinct from _Bad's all-zero case.
func TestExaone4_Config_InferFromWeights_Ugly(t *testing.T) {
	c := Config{ModelType: "exaone4", HiddenSize: 16, IntermediateSize: 32, NumHiddenLayers: 1, NumAttentionHeads: 7, NumKeyValueHeads: 2, HeadDim: 16, VocabSize: 8, RMSNormEps: 0.00001, RopeTheta: 10000}
	c.InferFromWeights(nil)
	if _, e := c.Arch(); e == nil {
		t.Fatal("heads not divisible by kv-heads became valid after InferFromWeights")
	}
}
