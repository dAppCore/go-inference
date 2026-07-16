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

// TestRoutes_Routes_Describe_Good verifies Describe() surfaces every route
// registered in RegisterRoutes with a documented summary and the "ml" tag —
// the detail the core/api engine needs to mount the OpenAPI spec (and the
// SDK generators need to emit a typed client).
func TestRoutes_Routes_Describe_Good(t *testing.T) {
	r := NewRoutes(nil)
	descriptions := r.Describe()
	if len(descriptions) != 3 {
		t.Fatalf("Describe() returned %d descriptions, want 3", len(descriptions))
	}
	for _, desc := range descriptions {
		if desc.Summary == "" {
			t.Fatalf("route %s %s has no Summary", desc.Method, desc.Path)
		}
		if len(desc.Tags) == 0 || desc.Tags[0] != "ml" {
			t.Fatalf("route %s %s missing the ml tag: %v", desc.Method, desc.Path, desc.Tags)
		}
	}
}

// TestRoutes_Routes_Describe_Bad covers a nil receiver — Describe() builds
// its descriptions from static data only, never touching r.service, so it
// must still answer the full route list rather than panic.
func TestRoutes_Routes_Describe_Bad(t *testing.T) {
	var r *Routes
	descriptions := r.Describe()
	if len(descriptions) != 3 {
		t.Fatalf("Describe() on a nil *Routes returned %d descriptions, want 3", len(descriptions))
	}
}

// TestRoutes_Routes_Describe_Ugly checks the /generate route's request-body
// schema carries the full generateRequest field set — the detail the SDK
// generators rely on to emit a typed client, not just the route list.
func TestRoutes_Routes_Describe_Ugly(t *testing.T) {
	r := NewRoutes(nil)
	for _, desc := range r.Describe() {
		if desc.Method != http.MethodPost || desc.Path != "/generate" {
			continue
		}
		body := desc.RequestBody
		if body == nil {
			t.Fatalf("/generate RequestBody is nil, want a JSON Schema map")
		}
		props, ok := body["properties"].(map[string]any)
		if !ok {
			t.Fatalf("/generate RequestBody has no properties map")
		}
		for _, field := range []string{"prompt", "backend", "temperature", "max_tokens"} {
			if _, ok := props[field]; !ok {
				t.Fatalf("/generate RequestBody properties missing %q", field)
			}
		}
		return
	}
	t.Fatal("Describe() has no POST /generate route")
}
