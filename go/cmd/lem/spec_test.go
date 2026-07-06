// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"testing"

	core "dappco.re/go"
)

// TestRunSpecCommand_WritesFullSurface runs the spec exporter and asserts it
// writes an OpenAPI document carrying one representative route from every
// describable group — the proof that the whole HTTP surface reaches the SDK
// generators. A provider dropping out of the registration (or losing its
// Describe) fails here.
func TestRunSpecCommand_WritesFullSurface(t *testing.T) {
	out := filepath.Join(t.TempDir(), "openapi.json")
	if code := runSpecCommand(context.Background(), []string{"-o", out}, io.Discard, io.Discard); code != 0 {
		t.Fatalf("runSpecCommand exit = %d, want 0", code)
	}
	data, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("read spec: %v", err)
	}
	spec := string(data)
	for _, want := range []string{
		`"openapi"`,            // it is an OpenAPI document
		"/v1/score/content",    // serving/api AIProvider
		"/v1/ml/generate",      // serving/api ml Routes
		"/v1/driver/serve",     // engine/driver Provider
		"/v1/chat/completions", // engine/driver InferenceProvider
		"/v1/state/status",     // kv/sessionkv
	} {
		if !core.Contains(spec, want) {
			t.Fatalf("spec missing %q — a route group did not reach the definition", want)
		}
	}
}
