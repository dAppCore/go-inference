// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"bytes"
	"context"
	"path/filepath"
	"testing"

	core "dappco.re/go"
)

// TestStringListFlag_Set proves the repeatable flag appends each occurrence in
// order, so -image a.png -image b.png collects both paths.
func TestStringListFlag_Set(t *testing.T) {
	var list stringListFlag
	for _, v := range []string{"a.png", "b.png", "c.png"} {
		if err := list.Set(v); err != nil {
			t.Fatalf("Set(%q): %v", v, err)
		}
	}
	if got := []string(list); len(got) != 3 || got[0] != "a.png" || got[2] != "c.png" {
		t.Errorf("collected = %v, want [a.png b.png c.png]", got)
	}
}

// TestStringListFlag_String proves the flag renders as a comma-joined list, that
// an empty list renders empty, and that a nil *stringListFlag receiver returns
// "" via the defensive guard rather than dereferencing nil.
func TestStringListFlag_String(t *testing.T) {
	for _, tc := range []struct {
		name string
		list stringListFlag
		want string
	}{
		{"empty", stringListFlag{}, ""},
		{"multi", stringListFlag{"a.png", "b.png"}, "a.png,b.png"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.list.String(); got != tc.want {
				t.Errorf("String() = %q, want %q", got, tc.want)
			}
		})
	}
	t.Run("nil receiver", func(t *testing.T) {
		var p *stringListFlag
		if got := p.String(); got != "" {
			t.Errorf("nil receiver String() = %q, want empty", got)
		}
	})
}

// TestRunGenerateCommand_Rejects covers the argument + prompt-file guards that
// return before any model is loaded: the wrong positional count, an unreadable
// -prompt-file, and an empty -prompt-file. A bogus model path is safe here
// because every case returns before generate.RunGenerate is reached.
func TestRunGenerateCommand_Rejects(t *testing.T) {
	emptyFile := filepath.Join(t.TempDir(), "empty.txt")
	if r := core.WriteFile(emptyFile, nil, 0o644); !r.OK {
		t.Fatalf("write empty prompt file: %v", r.Err())
	}
	for _, tc := range []struct {
		name     string
		args     []string
		wantCode int
	}{
		{"no positional", nil, 2},
		{"two positionals", []string{"model-a", "model-b"}, 2},
		{"prompt-file missing", []string{"-prompt-file", filepath.Join(t.TempDir(), "nope.txt"), "model"}, 1},
		{"prompt-file empty", []string{"-prompt-file", emptyFile, "model"}, 1},
		{"help", []string{"-h"}, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if code := runGenerateCommand(context.Background(), tc.args, &stdout, &stderr); code != tc.wantCode {
				t.Fatalf("exit %d, want %d; stderr=%s", code, tc.wantCode, stderr.String())
			}
		})
	}
}
