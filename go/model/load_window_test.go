// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"os"
	"path/filepath"
	"testing"
)

// TestProbeDirContextWindow gates the checkpoint-window probe the context
// default rides on: prefer-the-larger across top-level and text_config, 0 for
// unreadable/absent so callers keep their floor.
func TestProbeDirContextWindow(t *testing.T) {
	write := func(t *testing.T, body string) string {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		return dir
	}
	if got := ProbeDirContextWindow(write(t, `{"max_position_embeddings":131072}`)); got != 131072 {
		t.Fatalf("top-level window = %d, want 131072", got)
	}
	if got := ProbeDirContextWindow(write(t, `{"max_position_embeddings":4096,"text_config":{"max_position_embeddings":262144}}`)); got != 262144 {
		t.Fatalf("nested-larger window = %d, want 262144", got)
	}
	if got := ProbeDirContextWindow(write(t, `{"model_type":"gemma4"}`)); got != 0 {
		t.Fatalf("absent window = %d, want 0", got)
	}
	if got := ProbeDirContextWindow(t.TempDir()); got != 0 {
		t.Fatalf("missing config window = %d, want 0", got)
	}
}
