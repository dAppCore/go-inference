// SPDX-License-Identifier: EUPL-1.2

package api

import (
	"net/http"
	"testing"

	coreprovider "dappco.re/go/api/pkg/provider"
)

// TestRoutesDescribableGoodScenario verifies the ML route group is
// OpenAPI-describable and surfaces every route it registers, so the core/api
// engine mounts it into the generated spec (and the SDK generators emit a typed
// client for it).
func TestRoutesDescribableGoodScenario(t *testing.T) {
	var _ coreprovider.Describable = (*Routes)(nil)

	r := NewRoutes(nil)
	want := map[string]bool{
		http.MethodGet + " /backends":  false,
		http.MethodGet + " /status":    false,
		http.MethodPost + " /generate": false,
	}
	descriptions := r.Describe()
	if len(descriptions) == 0 {
		t.Fatal("Describe returned no route descriptions")
	}
	for _, desc := range descriptions {
		if _, ok := want[desc.Method+" "+desc.Path]; ok {
			want[desc.Method+" "+desc.Path] = true
		}
	}
	for key, seen := range want {
		if !seen {
			t.Fatalf("expected route description for %s", key)
		}
	}
}
