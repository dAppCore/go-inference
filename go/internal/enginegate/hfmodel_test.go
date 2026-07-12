// SPDX-Licence-Identifier: EUPL-1.2

package enginegate

import (
	"path/filepath"
	"testing"

	core "dappco.re/go"
)

// hubSnapshot builds a fake Hugging Face hub entry
// (<home>/.cache/huggingface/hub/models--<repo>/snapshots/<hash>) and returns the
// snapshot directory HFModelPath should resolve to.
func hubSnapshot(t *testing.T, home, dirName, hash string) string {
	t.Helper()
	snap := filepath.Join(home, ".cache", "huggingface", "hub", dirName, "snapshots", hash)
	if r := core.MkdirAll(snap, 0o755); !r.OK {
		t.Fatalf("mkdir %s: %v", snap, r.Err())
	}
	return snap
}

// TestHFModelPath proves the resolver maps a repo id to its cached snapshot
// directory: an exact org/name match, and a trailing-"*" prefix match for
// families whose exact pack name varies. Both are success paths, so the passed
// testing.TB is never Skipped or Failed. (The skip branches — no home, no cache,
// model absent — call t.Skip on the caller's TB; exercising them needs a fake
// testing.TB, which the sealed interface forbids, so they are left uncovered.)
func TestHFModelPath(t *testing.T) {
	t.Run("exact match", func(t *testing.T) {
		home := t.TempDir()
		t.Setenv("HOME", home)
		want := hubSnapshot(t, home, "models--mlx-community--gemma-4-e2b-it-6bit", "deadbeef")

		got := HFModelPath(t, "mlx-community/gemma-4-e2b-it-6bit")
		if got != want {
			t.Errorf("HFModelPath = %q, want %q", got, want)
		}
	})

	t.Run("prefix match", func(t *testing.T) {
		home := t.TempDir()
		t.Setenv("HOME", home)
		want := hubSnapshot(t, home, "models--mlx-community--Qwen3-Next-80B-A3B", "cafef00d")

		got := HFModelPath(t, "mlx-community/Qwen3-Next*")
		if got != want {
			t.Errorf("HFModelPath (prefix) = %q, want %q", got, want)
		}
	})
}
