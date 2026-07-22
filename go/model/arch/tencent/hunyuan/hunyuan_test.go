// SPDX-Licence-Identifier: EUPL-1.2
package hunyuan

import (
	"dappco.re/go/inference/model"
	"testing"
)

const realConfig = `{"model_type":"hunyuan_v1_dense","hidden_size":4096,"intermediate_size":14336,"num_hidden_layers":32,"num_attention_heads":32,"num_key_value_heads":8,"head_dim":128,"vocab_size":128167,"rms_norm_eps":0.00001,"rope_theta":10000,"use_qk_norm":true,"use_cla":false,"cla_share_factor":2,"tie_word_embeddings":true}` // huggingface.co/tencent/Hunyuan-7B-Instruct
func TestHunyuan_Config_Arch_Good(t *testing.T) {
	s, _ := model.LookupArch("hunyuan_v1_dense")
	c, e := s.Parse([]byte(realConfig))
	if e != nil {
		t.Fatal(e)
	}
	a, e := c.Arch()
	if e != nil || a.QKNormalization != model.QKRMSNorm || s.Weights.QNorm != ".self_attn.query_layernorm.weight" {
		t.Fatalf("arch=%+v err=%v", a, e)
	}
}
func TestHunyuan_Config_Arch_Bad(t *testing.T) {
	if _, e := (&Config{ModelType: "hunyuan_v1_dense"}).Arch(); e == nil {
		t.Fatal("empty dimensions accepted")
	}
}

// TestHunyuan_Config_Arch_Ugly pins the CLA share-factor guard: a syntactically valid,
// otherwise-complete config with use_cla=true but a non-positive
// cla_share_factor must be rejected by Arch() itself (distinct from _Bad's
// all-zero-fields rejection).
func TestHunyuan_Config_Arch_Ugly(t *testing.T) {
	s, _ := model.LookupArch("hunyuan_v1_dense")
	c, e := s.Parse([]byte(`{"model_type":"hunyuan_v1_dense","hidden_size":16,"intermediate_size":32,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":16,"vocab_size":8,"rms_norm_eps":0.00001,"rope_theta":10000,"use_cla":true,"cla_share_factor":0}`))
	if e != nil {
		t.Fatal(e)
	}
	if _, e := c.Arch(); e == nil {
		t.Fatal("use_cla with cla_share_factor<=0 accepted")
	}
}

func TestHunyuan_Config_InferFromWeights_Good(t *testing.T) {
	c := Config{HiddenSize: 4096}
	c.InferFromWeights(nil)
	if c.HiddenSize != 4096 {
		t.Fatalf("InferFromWeights changed config: %+v", c)
	}
}

func TestHunyuan_Config_InferFromWeights_Bad(t *testing.T) {
	c := Config{}
	c.InferFromWeights(nil)
	if _, e := c.Arch(); e == nil {
		t.Fatal("empty config became valid after InferFromWeights")
	}
}

// TestHunyuan_Config_InferFromWeights_Ugly proves the no-op does not paper
// over the CLA share-factor guard — distinct from _Bad's all-zero case.
func TestHunyuan_Config_InferFromWeights_Ugly(t *testing.T) {
	c := Config{ModelType: "hunyuan_v1_dense", HiddenSize: 16, IntermediateSize: 32, NumHiddenLayers: 1, NumAttentionHeads: 2, NumKeyValueHeads: 2, HeadDim: 16, VocabSize: 8, RMSNormEps: 0.00001, RopeTheta: 10000, UseCLA: true, CLAshareFactor: 0}
	c.InferFromWeights(nil)
	if _, e := c.Arch(); e == nil {
		t.Fatal("use_cla with cla_share_factor<=0 became valid after InferFromWeights")
	}
}
