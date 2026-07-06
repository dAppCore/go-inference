// SPDX-License-Identifier: EUPL-1.2

package driver

import (
	"net/http"
	"testing"

	coreprovider "dappco.re/go/api/pkg/provider"
)

// TestProvider_Describable_Good verifies the driver-orchestration route group is
// OpenAPI-describable and surfaces every route it registers, so the core/api
// engine mounts it into the generated spec (and the SDK generators emit a typed
// client for it).
func TestProvider_Describable_Good(t *testing.T) {
	var _ coreprovider.Describable = (*Provider)(nil)

	p := NewProvider(nil)
	want := map[string]bool{
		http.MethodGet + " /models": false,
		http.MethodPost + " /serve": false,
		http.MethodGet + " /status": false,
		http.MethodPost + " /stop":  false,
	}
	descriptions := p.Describe()
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
