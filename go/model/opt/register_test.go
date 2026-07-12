// SPDX-Licence-Identifier: EUPL-1.2

package opt

import (
	"dappco.re/go/inference/model"
	"testing"
)

func TestRegister_OPT_Good(t *testing.T) {
	spec, ok := model.LookupArch("opt")
	if !ok || spec.Parse == nil {
		t.Fatal("OPT architecture is not registered")
	}
	config, err := spec.Parse([]byte(`{"model_type":"opt","hidden_size":8,"word_embed_proj_dim":4,"num_attention_heads":2,"num_hidden_layers":1,"ffn_dim":16,"max_position_embeddings":8,"vocab_size":12}`))
	if err != nil {
		t.Fatal(err)
	}
	arch, err := config.Arch()
	if err != nil || arch.PositionOffset != 2 {
		t.Fatalf("arch=%+v err=%v", arch, err)
	}
}

func TestRegister_OPT_Bad(t *testing.T) {
	spec, ok := model.LookupArch("opt")
	if !ok {
		t.Fatal("OPT architecture is not registered")
	}
	if _, err := spec.Parse([]byte(`{"model_type":`)); err == nil {
		t.Fatal("malformed config accepted")
	}
}

func TestRegister_OPT_Ugly(t *testing.T) {
	spec, ok := model.LookupArch("opt")
	if !ok {
		t.Fatal("OPT architecture is not registered")
	}
	config, err := spec.Parse([]byte(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := config.Arch(); err == nil {
		t.Fatal("empty config accepted")
	}
}
