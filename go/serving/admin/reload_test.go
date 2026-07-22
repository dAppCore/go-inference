// SPDX-Licence-Identifier: EUPL-1.2

package admin

import (
	"testing"

	core "dappco.re/go"
)

// TestReload_ListKnownModels_Good proves a models root with several verified
// entries returns every one of them, not just the first found.
func TestReload_ListKnownModels_Good(t *testing.T) {
	seedModel(t, "alpha")
	dir := standardModelDir()
	if r := core.MkdirAll(core.PathJoin(dir, "bravo"), 0o755); !r.OK {
		t.Fatalf("mkdir bravo: %v", r.Value)
	}
	if r := core.WriteFile(core.PathJoin(dir, "bravo", shaManifestFilename), []byte("x"), 0o600); !r.OK {
		t.Fatalf("seed bravo sha: %v", r.Value)
	}

	got := ListKnownModels()
	want := map[string]bool{"alpha": true, "bravo": true}
	if len(got) != 2 {
		t.Fatalf("ListKnownModels() = %v, want 2 entries", got)
	}
	for _, name := range got {
		if !want[name] {
			t.Fatalf("ListKnownModels() included unexpected %q", name)
		}
	}
}

// TestReload_ListKnownModels_Bad proves a models root that exists but is a
// plain FILE (not a directory) is treated as no known models rather than
// panicking — a different failure shape than a missing root entirely.
func TestReload_ListKnownModels_Bad(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	root := standardModelDir()
	if r := core.MkdirAll(core.PathDir(root), 0o755); !r.OK {
		t.Fatalf("mkdir models parent: %v", r.Value)
	}
	if r := core.WriteFile(root, []byte("not a directory"), 0o644); !r.OK {
		t.Fatalf("seed file-not-dir root: %v", r.Value)
	}

	if got := ListKnownModels(); len(got) != 0 {
		t.Fatalf("ListKnownModels() with a file where the root dir should be = %v, want empty", got)
	}
}

// TestReload_ListKnownModels_Ugly proves an existing-but-empty models root
// (the fresh-install state, before any model is ever downloaded) returns an
// empty, non-nil slice rather than nil — callers range over it directly.
func TestReload_ListKnownModels_Ugly(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if r := core.MkdirAll(standardModelDir(), 0o755); !r.OK {
		t.Fatalf("mkdir empty models root: %v", r.Value)
	}

	got := ListKnownModels()
	if got == nil {
		t.Fatal("ListKnownModels() on an empty root = nil, want a non-nil empty slice")
	}
	if len(got) != 0 {
		t.Fatalf("ListKnownModels() on an empty root = %v, want empty", got)
	}
}
