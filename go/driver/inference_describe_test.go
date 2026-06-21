// SPDX-License-Identifier: EUPL-1.2

package driver

import (
	"net/http"
	"testing"

	coreprovider "dappco.re/go/api/pkg/provider"
)

// TestInferenceProvider_Describable_Good verifies the gated inference route
// group is OpenAPI-describable and surfaces every route it registers, so the
// core/api engine can mount it into the generated spec.
func TestInferenceProvider_Describable_Good(t *testing.T) {
	var _ coreprovider.Describable = (*InferenceProvider)(nil)

	p := NewInferenceProvider(nil)
	want := map[string]bool{
		http.MethodPost + " /chat/completions": false,
		http.MethodPost + " /completions":      false,
		http.MethodPost + " /messages":         false,
		http.MethodGet + " /models":            false,
	}
	descriptions := p.Describe()
	if len(descriptions) == 0 {
		t.Fatal("Describe returned no route descriptions")
	}
	for _, desc := range descriptions {
		key := desc.Method + " " + desc.Path
		if _, ok := want[key]; ok {
			want[key] = true
		}
	}
	for key, seen := range want {
		if !seen {
			t.Fatalf("expected route description for %s", key)
		}
	}
}
