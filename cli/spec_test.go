// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"bytes"
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

// TestRunSpecCommand_BadFormat asserts exporter failures are reported as a
// command error and do not leave a successful artifact behind.
func TestRunSpecCommand_BadFormat(t *testing.T) {
	out := filepath.Join(t.TempDir(), "openapi.invalid")
	var stdout, stderr bytes.Buffer
	if code := runSpecCommand(context.Background(), []string{"-format", "invalid", "-o", out}, &stdout, &stderr); code != 1 {
		t.Fatalf("runSpecCommand invalid format exit = %d, want 1; stderr=%s", code, stderr.String())
	}
	if _, err := os.Stat(out); !os.IsNotExist(err) {
		t.Errorf("invalid format created %s, stat error=%v", out, err)
	}
}
