// SPDX-Licence-Identifier: EUPL-1.2
package ernie45

import (
	"dappco.re/go/inference/model"
	"testing"
)

const realConfig = `{"model_type":"ernie4_5","hidden_size":1024,"intermediate_size":3072,"num_hidden_layers":18,"num_attention_heads":16,"num_key_value_heads":2,"head_dim":128,"vocab_size":103424,"rms_norm_eps":0.00001,"rope_theta":500000,"tie_word_embeddings":true}` // huggingface.co/baidu/ERNIE-4.5-0.3B-PT
func TestConfig_Arch_Good(t *testing.T) {
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
func TestConfig_Arch_Bad(t *testing.T) {
	if _, e := (&Config{ModelType: "ernie4_5"}).Arch(); e == nil {
		t.Fatal("empty dimensions accepted")
	}
}
func TestConfig_Arch_Ugly(t *testing.T) {
	s, _ := model.LookupArch("ernie4_5")
	if _, e := s.Parse([]byte(`{`)); e == nil {
		t.Fatal("malformed JSON accepted")
	}
}
