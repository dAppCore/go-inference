// SPDX-Licence-Identifier: EUPL-1.2
package hunyuan

import (
	"dappco.re/go/inference/model"
	"testing"
)

const realConfig = `{"model_type":"hunyuan_v1_dense","hidden_size":4096,"intermediate_size":14336,"num_hidden_layers":32,"num_attention_heads":32,"num_key_value_heads":8,"head_dim":128,"vocab_size":128167,"rms_norm_eps":0.00001,"rope_theta":10000,"use_qk_norm":true,"use_cla":false,"cla_share_factor":2,"tie_word_embeddings":true}` // huggingface.co/tencent/Hunyuan-7B-Instruct
func TestConfig_Arch_Good(t *testing.T) {
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
func TestConfig_Arch_Bad(t *testing.T) {
	if _, e := (&Config{ModelType: "hunyuan_v1_dense"}).Arch(); e == nil {
		t.Fatal("empty dimensions accepted")
	}
}

// TestConfig_Arch_Ugly pins the CLA share-factor guard: a syntactically valid,
// otherwise-complete config with use_cla=true but a non-positive
// cla_share_factor must be rejected by Arch() itself (distinct from _Bad's
// all-zero-fields rejection).
func TestConfig_Arch_Ugly(t *testing.T) {
	s, _ := model.LookupArch("hunyuan_v1_dense")
	c, e := s.Parse([]byte(`{"model_type":"hunyuan_v1_dense","hidden_size":16,"intermediate_size":32,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":16,"vocab_size":8,"rms_norm_eps":0.00001,"rope_theta":10000,"use_cla":true,"cla_share_factor":0}`))
	if e != nil {
		t.Fatal(e)
	}
	if _, e := c.Arch(); e == nil {
		t.Fatal("use_cla with cla_share_factor<=0 accepted")
	}
}
