// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/pack"
)

// packFixtureModel writes a toy model dir, packs it into a .model container via
// the create verb itself, and returns (srcDir, modelPath). It is the shared
// round-trip fixture for the read verbs (inspect/list/extract/hash) — the same
// pattern lthn-model-pack's test uses, so each read verb is exercised against a
// container the create verb actually produced.
func packFixtureModel(t *testing.T) (srcDir, modelPath string) {
	t.Helper()
	srcDir = writeToyModel(t)
	modelPath = filepath.Join(t.TempDir(), "toy.model")
	var stdout, stderr bytes.Buffer
	if code := runPackCommand(context.Background(),
		[]string{"create", "-arch", "gemma4", "-quant", "4", srcDir, modelPath},
		&stdout, &stderr); code != 0 {
		t.Fatalf("fixture create exit %d; stderr=%s", code, stderr.String())
	}
	return srcDir, modelPath
}

// TestRunPackCommand_Dispatch covers the verb router: an empty argument list and
// an unknown subcommand both exit 2 with the usage banner, while -h exits 0.
func TestRunPackCommand_Dispatch(t *testing.T) {
	for _, tc := range []struct {
		name     string
		args     []string
		wantCode int
		wantHelp bool // usage banner expected somewhere in the output
	}{
		{"no args", nil, 2, true},
		{"unknown subcommand", []string{"frobnicate"}, 2, true},
		{"help flag", []string{"-h"}, 0, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			code := runPackCommand(context.Background(), tc.args, &stdout, &stderr)
			if code != tc.wantCode {
				t.Fatalf("exit %d, want %d", code, tc.wantCode)
			}
			if tc.wantHelp && !core.Contains(stdout.String()+stderr.String(), "Build and read .model containers") {
				t.Errorf("usage banner missing; stdout=%q stderr=%q", stdout.String(), stderr.String())
			}
		})
	}
}

// TestRunPackCreate proves the create verb tars a model dir into a readable
// .model container (Good), rejects the wrong positional count (Bad), and fails
// on a missing source directory (Ugly).
func TestRunPackCreate(t *testing.T) {
	t.Run("Good", func(t *testing.T) {
		src := writeToyModel(t)
		out := filepath.Join(t.TempDir(), "created.model")
		var stdout, stderr bytes.Buffer
		if code := runPackCreate([]string{"-arch", "gemma4", "-quant", "4", src, out}, &stdout, &stderr); code != 0 {
			t.Fatalf("exit %d; stderr=%s", code, stderr.String())
		}
		if _, err := os.Stat(out); err != nil {
			t.Fatalf("container not written: %v", err)
		}
		// The manifest the verb embedded must read back with the flag values.
		manifest, _, r := pack.Inspect(out)
		if !r.OK {
			t.Fatalf("inspect created container: %s", r.Error())
		}
		if manifest.Model.Architecture != "gemma4" {
			t.Errorf("architecture = %q, want gemma4", manifest.Model.Architecture)
		}
		if manifest.Model.QuantBits != 4 {
			t.Errorf("quant bits = %d, want 4", manifest.Model.QuantBits)
		}
	})
	t.Run("Bad", func(t *testing.T) {
		src := writeToyModel(t)
		var stdout, stderr bytes.Buffer
		// One positional where two are required.
		if code := runPackCreate([]string{src}, &stdout, &stderr); code != 2 {
			t.Fatalf("exit %d, want 2; stderr=%s", code, stderr.String())
		}
	})
	t.Run("Ugly", func(t *testing.T) {
		missing := filepath.Join(t.TempDir(), "does-not-exist")
		out := filepath.Join(t.TempDir(), "out.model")
		var stdout, stderr bytes.Buffer
		if code := runPackCreate([]string{missing, out}, &stdout, &stderr); code != 1 {
			t.Fatalf("exit %d, want 1; stderr=%s", code, stderr.String())
		}
	})
}

// TestRunPackInspect proves inspect prints the manifest in both text and JSON
// (Good), rejects a missing positional (Bad), and fails on a non-container file
// (Ugly).
func TestRunPackInspect(t *testing.T) {
	_, modelPath := packFixtureModel(t)
	t.Run("Good/text", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		if code := runPackInspect([]string{modelPath}, &stdout, &stderr); code != 0 {
			t.Fatalf("exit %d; stderr=%s", code, stderr.String())
		}
		if !core.Contains(stdout.String(), "gemma4") {
			t.Errorf("text inspect missing architecture; got %q", stdout.String())
		}
	})
	t.Run("Good/json", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		if code := runPackInspect([]string{"-json", modelPath}, &stdout, &stderr); code != 0 {
			t.Fatalf("exit %d; stderr=%s", code, stderr.String())
		}
		out := stdout.String()
		if !core.HasPrefix(core.Trim(out), "{") || !core.Contains(out, "gemma4") {
			t.Errorf("json inspect not a JSON manifest; got %q", out)
		}
	})
	t.Run("Bad", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		if code := runPackInspect(nil, &stdout, &stderr); code != 2 {
			t.Fatalf("exit %d, want 2; stderr=%s", code, stderr.String())
		}
	})
	t.Run("Ugly", func(t *testing.T) {
		notContainer := writeToyModel(t) + "/config.json"
		var stdout, stderr bytes.Buffer
		if code := runPackInspect([]string{notContainer}, &stdout, &stderr); code != 1 {
			t.Fatalf("exit %d, want 1; stderr=%s", code, stderr.String())
		}
	})
}

// TestRunPackList proves list enumerates the payload entries in text and JSON
// (Good), rejects a missing positional (Bad), and fails on a missing file (Ugly).
func TestRunPackList(t *testing.T) {
	_, modelPath := packFixtureModel(t)
	t.Run("Good/text", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		if code := runPackList([]string{modelPath}, &stdout, &stderr); code != 0 {
			t.Fatalf("exit %d; stderr=%s", code, stderr.String())
		}
		if !core.Contains(stdout.String(), "config.json") {
			t.Errorf("list missing config.json entry; got %q", stdout.String())
		}
	})
	t.Run("Good/json", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		if code := runPackList([]string{"-json", modelPath}, &stdout, &stderr); code != 0 {
			t.Fatalf("exit %d; stderr=%s", code, stderr.String())
		}
		if !core.Contains(stdout.String(), "config.json") {
			t.Errorf("json list missing config.json entry; got %q", stdout.String())
		}
	})
	t.Run("Bad", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		if code := runPackList(nil, &stdout, &stderr); code != 2 {
			t.Fatalf("exit %d, want 2; stderr=%s", code, stderr.String())
		}
	})
	t.Run("Ugly", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		missing := filepath.Join(t.TempDir(), "nope.model")
		if code := runPackList([]string{missing}, &stdout, &stderr); code != 1 {
			t.Fatalf("exit %d, want 1; stderr=%s", code, stderr.String())
		}
	})
}

// TestRunPackExtract proves extract unpacks a container to a directory (Good),
// refuses a non-empty destination without -overwrite but proceeds with it
// (Ugly), and rejects the wrong positional count (Bad).
func TestRunPackExtract(t *testing.T) {
	t.Run("Good", func(t *testing.T) {
		src, modelPath := packFixtureModel(t)
		dest := filepath.Join(t.TempDir(), "extracted")
		var stdout, stderr bytes.Buffer
		if code := runPackExtract([]string{modelPath, dest}, &stdout, &stderr); code != 0 {
			t.Fatalf("exit %d; stderr=%s", code, stderr.String())
		}
		want, err := os.ReadFile(filepath.Join(src, "config.json"))
		if err != nil {
			t.Fatalf("read source config: %v", err)
		}
		got, err := os.ReadFile(filepath.Join(dest, "config.json"))
		if err != nil {
			t.Fatalf("read extracted config: %v", err)
		}
		if string(got) != string(want) {
			t.Errorf("extracted config.json diverged from source")
		}
	})
	t.Run("Bad", func(t *testing.T) {
		_, modelPath := packFixtureModel(t)
		var stdout, stderr bytes.Buffer
		if code := runPackExtract([]string{modelPath}, &stdout, &stderr); code != 2 {
			t.Fatalf("exit %d, want 2; stderr=%s", code, stderr.String())
		}
	})
	t.Run("Ugly", func(t *testing.T) {
		_, modelPath := packFixtureModel(t)
		dest := filepath.Join(t.TempDir(), "occupied")
		if r := core.MkdirAll(dest, 0o755); !r.OK {
			t.Fatalf("mkdir dest: %v", r.Err())
		}
		if r := core.WriteFile(filepath.Join(dest, "squatter"), []byte("in the way"), 0o644); !r.OK {
			t.Fatalf("write squatter: %v", r.Err())
		}
		var stdout, stderr bytes.Buffer
		if code := runPackExtract([]string{modelPath, dest}, &stdout, &stderr); code != 1 {
			t.Fatalf("blocked extract exit %d, want 1; stderr=%s", code, stderr.String())
		}
		var out2, err2 bytes.Buffer
		if code := runPackExtract([]string{"-overwrite", modelPath, dest}, &out2, &err2); code != 0 {
			t.Fatalf("forced extract exit %d, want 0; stderr=%s", code, err2.String())
		}
	})
}

// TestRunPackHash proves hash prints the canonical model-pack hash of a directory
// (Good), rejects a missing positional (Bad), and fails on a missing directory
// (Ugly).
func TestRunPackHash(t *testing.T) {
	t.Run("Good", func(t *testing.T) {
		src := writeToyModel(t)
		var stdout, stderr bytes.Buffer
		if code := runPackHash([]string{src}, &stdout, &stderr); code != 0 {
			t.Fatalf("exit %d; stderr=%s", code, stderr.String())
		}
		if core.Trim(stdout.String()) == "" {
			t.Error("hash output empty")
		}
		// The printed hash must equal the library's own hash of the same dir.
		want, r := pack.Hash(src)
		if !r.OK {
			t.Fatalf("library hash: %s", r.Error())
		}
		if core.Trim(stdout.String()) != want {
			t.Errorf("printed hash %q != library hash %q", core.Trim(stdout.String()), want)
		}
	})
	t.Run("Bad", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		if code := runPackHash(nil, &stdout, &stderr); code != 2 {
			t.Fatalf("exit %d, want 2; stderr=%s", code, stderr.String())
		}
	})
	t.Run("Ugly", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		missing := filepath.Join(t.TempDir(), "gone")
		if code := runPackHash([]string{missing}, &stdout, &stderr); code != 1 {
			t.Fatalf("exit %d, want 1; stderr=%s", code, stderr.String())
		}
	})
}

// TestNonEmpty covers the inspect summary's fallback helper: a present value
// passes through, an empty value yields the fallback.
func TestNonEmpty(t *testing.T) {
	for _, tc := range []struct {
		name, value, fallback, want string
	}{
		{"present", "gemma4", "(unknown)", "gemma4"},
		{"empty", "", "(unknown)", "(unknown)"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := nonEmpty(tc.value, tc.fallback); got != tc.want {
				t.Errorf("nonEmpty(%q, %q) = %q, want %q", tc.value, tc.fallback, got, tc.want)
			}
		})
	}
}
