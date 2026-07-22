// SPDX-Licence-Identifier: EUPL-1.2
package ernie45

import (
	"dappco.re/go/inference/model"
	"testing"
)

const realConfig = `{"model_type":"ernie4_5","hidden_size":1024,"intermediate_size":3072,"num_hidden_layers":18,"num_attention_heads":16,"num_key_value_heads":2,"head_dim":128,"vocab_size":103424,"rms_norm_eps":0.00001,"rope_theta":500000,"tie_word_embeddings":true}` // huggingface.co/baidu/ERNIE-4.5-0.3B-PT
func TestErnie45_Config_Arch_Good(t *testing.T) {
	s, _ := model.LookupArch("ernie4_5")
	c, e := s.Parse([]byte(realConfig))
	if e != nil {
		t.Fatal(e)
	}
	a, e := c.Arch()
	if e != nil || a.HeadDim != 128 || a.Hidden/a.Heads == a.HeadDim {
		t.Fatalf("arch=%+v err=%v", a, e)
	}
}
func TestErnie45_Config_Arch_Bad(t *testing.T) {
	if _, e := (&Config{ModelType: "ernie4_5"}).Arch(); e == nil {
		t.Fatal("empty dimensions accepted")
	}
}

// TestErnie45_Config_Arch_Ugly pins the GQA head-divisibility guard: a syntactically
// valid, otherwise-complete config whose head count does not divide evenly by
// its kv-head count must be rejected by Arch() itself (distinct from _Bad's
// all-zero-fields rejection).
func TestErnie45_Config_Arch_Ugly(t *testing.T) {
	s, _ := model.LookupArch("ernie4_5")
	c, e := s.Parse([]byte(`{"model_type":"ernie4_5","hidden_size":16,"intermediate_size":32,"num_hidden_layers":1,"num_attention_heads":7,"num_key_value_heads":2,"head_dim":16,"vocab_size":8,"rms_norm_eps":0.00001,"rope_theta":10000}`))
	if e != nil {
		t.Fatal(e)
	}
	if _, e := c.Arch(); e == nil {
		t.Fatal("heads (7) not divisible by kv-heads (2) accepted")
	}
}

func TestErnie45_Config_InferFromWeights_Good(t *testing.T) {
	c := Config{HiddenSize: 1024}
	c.InferFromWeights(nil)
	if c.HiddenSize != 1024 {
		t.Fatalf("InferFromWeights changed config: %+v", c)
	}
}

func TestErnie45_Config_InferFromWeights_Bad(t *testing.T) {
	c := Config{}
	c.InferFromWeights(nil)
	if _, e := c.Arch(); e == nil {
		t.Fatal("empty config became valid after InferFromWeights")
	}
}

// TestErnie45_Config_InferFromWeights_Ugly proves the no-op does not paper
// over the GQA head-divisibility guard — distinct from _Bad's all-zero case.
func TestErnie45_Config_InferFromWeights_Ugly(t *testing.T) {
	c := Config{ModelType: "ernie4_5", HiddenSize: 16, IntermediateSize: 32, NumHiddenLayers: 1, NumAttentionHeads: 7, NumKeyValueHeads: 2, HeadDim: 16, VocabSize: 8, RMSNormEps: 0.00001, RopeTheta: 10000}
	c.InferFromWeights(nil)
	if _, e := c.Arch(); e == nil {
		t.Fatal("heads not divisible by kv-heads became valid after InferFromWeights")
	}
}
