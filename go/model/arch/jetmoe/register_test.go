// SPDX-Licence-Identifier: EUPL-1.2

package jetmoe

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

func TestRegister_LookupArch_Good(t *testing.T) {
	spec, ok := model.LookupArch("jetmoe")
	if !ok {
		t.Fatal("jetmoe architecture not registered")
	}
	parsed, err := spec.Parse([]byte(`{"model_type":"jetmoe","hidden_size":8}`))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := parsed.(*Config); !ok {
		t.Fatalf("parsed config = %T, want *jetmoe.Config", parsed)
	}
}

func TestRegister_LookupArch_Bad(t *testing.T) {
	spec, _ := model.LookupArch("jetmoe")
	if _, err := spec.Parse([]byte(`{"model_type":`)); err == nil {
		t.Fatal("malformed config accepted")
	}
}

// TestJetMoERegistered_Good proves the FACTORY route — LookupArch then the registered spec's
// Parse and the parsed config's Arch, the exact sequence model.Load drives — reaches the named
// MoA refusal for a well-formed jetmoe checkpoint, not the generic missing-weight error
// model.Assemble would raise if Arch ever resolved a geometry for it (#59 item 6).
func TestJetMoERegistered_Good(t *testing.T) {
	spec, ok := model.LookupArch("jetmoe")
	if !ok {
		t.Fatal("jetmoe architecture not registered")
	}
	config := []byte(`{"model_type":"jetmoe","hidden_size":8,"ffn_hidden_size":4,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"kv_channels":4,"moe_num_experts":2,"moe_top_k":1,"vocab_size":16}`)
	ac, err := spec.Parse(config)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if _, err := ac.Arch(); err == nil {
		t.Fatal("jetmoe Arch() accepted a checkpoint without a MoA attention implementation")
	} else if !core.Contains(err.Error(), "Mixture-of-Attention") {
		t.Fatalf("factory-route refusal %q must name the MoA gap", err.Error())
	}
}
