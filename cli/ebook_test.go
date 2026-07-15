// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"

	core "dappco.re/go"
)

// TestRunEbookCommand_Good proves the verb renders a model directory into a
// non-empty .epub both with the weight plates (default) and without them
// (-weights=false, the readable manifesto). No model is loaded — the render is
// pure file I/O over the toy safetensors fixture.
func TestRunEbookCommand_Good(t *testing.T) {
	for _, tc := range []struct {
		name string
		args func(model, out string) []string
	}{
		{"with weights", func(model, out string) []string { return []string{"--model", model, "--out", out} }},
		{"no weights", func(model, out string) []string { return []string{"--model", model, "--out", out, "--weights=false"} }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := writeToyModel(t)
			out := filepath.Join(t.TempDir(), "book.epub")
			var stdout, stderr bytes.Buffer
			if code := runEbookCommand(context.Background(), tc.args(model, out), &stdout, &stderr); code != 0 {
				t.Fatalf("exit %d; stderr=%s", code, stderr.String())
			}
			info, err := os.Stat(out)
			if err != nil {
				t.Fatalf("epub not written: %v", err)
			}
			if info.Size() == 0 {
				t.Error("epub is empty")
			}
			if !core.Contains(stdout.String(), "wrote") {
				t.Errorf("no wrote-summary on stdout; got %q", stdout.String())
			}
		})
	}
}

// TestRunEbookCommand_Bad covers the argument guards that return before the
// render: a missing -model and an unknown flag both exit 2.
func TestRunEbookCommand_Bad(t *testing.T) {
	for _, tc := range []struct {
		name     string
		args     []string
		wantCode int
	}{
		{"missing model", nil, 2},
		{"unknown flag", []string{"--nonsense"}, 2},
		{"help", []string{"--help"}, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if code := runEbookCommand(context.Background(), tc.args, &stdout, &stderr); code != tc.wantCode {
				t.Fatalf("exit %d, want %d; stderr=%s", code, tc.wantCode, stderr.String())
			}
		})
	}
}

// TestRunEbookCommand_Ugly proves a model directory with no safetensors shards
// fails the build (exit 1) rather than writing an empty book.
func TestRunEbookCommand_Ugly(t *testing.T) {
	empty := t.TempDir()
	out := filepath.Join(t.TempDir(), "book.epub")
	var stdout, stderr bytes.Buffer
	if code := runEbookCommand(context.Background(), []string{"--model", empty, "--out", out}, &stdout, &stderr); code != 1 {
		t.Fatalf("exit %d, want 1; stderr=%s", code, stderr.String())
	}
	if _, err := os.Stat(out); err == nil {
		t.Error("epub written despite build failure")
	}
}
