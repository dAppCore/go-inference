// SPDX-Licence-Identifier: EUPL-1.2

package gptneox

import (
	"testing"

	"dappco.re/go/inference/model"
)

// Tensor-key receipts use the canonical keys exposed by each checkpoint. These
// three checkpoints are unsharded and therefore publish no safetensors index.
func TestRegister_WeightNames_Good(t *testing.T) {
	cases := []struct{ kind, prefix, q string }{
		{"gpt_neox", "gpt_neox.layers.%d", ".attention.query_key_value"},
		{"gptj", "transformer.h.%d", ".attn.q_proj"},
		{"gpt_neo", "transformer.h.%d", ".attn.attention.q_proj"},
	}
	for _, tc := range cases {
		w := WeightNames(tc.kind)
		if w.LayerPrefix != tc.prefix || w.Q != tc.q {
			t.Errorf("%s names = %#v", tc.kind, w)
		}
	}
}

func TestRegister_WeightNames_Bad(t *testing.T) {
	w := WeightNames("unknown")
	if w.LayerPrefix != "model.layers.%d" {
		t.Fatalf("unknown mapping mutated: %#v", w)
	}
}

func TestRegister_WeightNames_Ugly(t *testing.T) {
	w := WeightNames("")
	if w.Embed == "" {
		t.Fatal("empty model type lost neutral defaults")
	}
}

func TestRegistered_Good(t *testing.T) {
	for _, name := range []string{"gpt_neox", "gptj", "gpt_neo"} {
		if _, ok := model.LookupArch(name); !ok {
			t.Errorf("%s not registered", name)
		}
	}
}
