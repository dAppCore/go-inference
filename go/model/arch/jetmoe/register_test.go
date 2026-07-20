// SPDX-Licence-Identifier: EUPL-1.2

package jetmoe

import (
	"testing"

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
