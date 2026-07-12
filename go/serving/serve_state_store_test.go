// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/state"
	"dappco.re/go/inference/model/state/filestore"
)

// statSize returns path's byte size via core.Stat (os.FileInfo behind Result).
func statSize(t *testing.T, path string) int64 {
	t.Helper()
	r := core.Stat(path)
	if !r.OK {
		t.Fatalf("stat %s: %v", path, r.Value)
	}
	return r.Value.(interface{ Size() int64 }).Size()
}

// TestOpenContinuityStore_RAMDefault_Good pins the inverted default: an unset
// -state-store holds conversations in a pure-RAM store — no path, no file
// touched, and a cleanup that is safe to call. This is the MacBook-class
// default (no per-turn disk round-trip for a cache discarded at shutdown).
func TestOpenContinuityStore_RAMDefault_Good(t *testing.T) {
	store, cleanup, where, err := openContinuityStore(context.Background(), "")
	if err != nil {
		t.Fatalf("openContinuityStore(unset): %v", err)
	}
	if _, ok := store.(*state.InMemoryStore); !ok {
		t.Fatalf("unset -state-store gave %T, want a *state.InMemoryStore (RAM tier)", store)
	}
	if !core.Contains(where, "RAM") {
		t.Fatalf("boot notice = %q, want it to mention RAM", where)
	}
	if cleanup == nil {
		t.Fatal("cleanup must never be nil")
	}
	cleanup() // a RAM store's cleanup is a no-op; it must not panic.
}

// TestOpenContinuityStore_Unopenable_Bad pins the degrade: an explicit durable
// path that cannot be opened (its parent is a regular file, so the store dir
// can't be made) returns an error and no store — serve then falls back to
// stateless rather than crashing.
func TestOpenContinuityStore_Unopenable_Bad(t *testing.T) {
	blocker := t.TempDir() + "/not-a-dir"
	if r := core.WriteFile(blocker, []byte("x"), 0o644); !r.OK {
		t.Fatalf("seed blocker file: %v", r.Value)
	}
	// A store path UNDER the regular file: mkdir of the parent must fail.
	_, _, _, err := openContinuityStore(context.Background(), blocker+"/store.kv")
	if err == nil {
		t.Fatal("a durable store under a regular file must error, not succeed")
	}
}

// TestOpenContinuityStore_DurableFile_Ugly pins the opt-in durable lane: an
// explicit path is the per-project file store, so the file is created on disk
// and the boot notice names it — chats that asked for durability keep .kv.
func TestOpenContinuityStore_DurableFile_Ugly(t *testing.T) {
	path := t.TempDir() + "/project.kv"
	store, cleanup, where, err := openContinuityStore(context.Background(), path)
	if err != nil {
		t.Fatalf("openContinuityStore(%q): %v", path, err)
	}
	defer cleanup()
	if _, ok := store.(*filestore.Store); !ok {
		t.Fatalf("explicit -state-store gave %T, want a *filestore.Store (durable tier)", store)
	}
	if !core.Stat(path).OK {
		t.Fatalf("durable store did not create the file at %s", path)
	}
	if !core.Contains(where, path) {
		t.Fatalf("boot notice = %q, want it to name the store path", where)
	}
}

// TestOpenConversationStore_Create_Good proves the durable opener creates a
// usable store at a fresh path.
func TestOpenConversationStore_Create_Good(t *testing.T) {
	ctx := context.Background()
	path := t.TempDir() + "/fresh.kv"
	store, err := openConversationStore(ctx, path)
	if err != nil {
		t.Fatalf("open fresh: %v", err)
	}
	defer store.Close()
	if _, err := store.PutBytes(ctx, []byte("turn-block"), state.PutOptions{}); err != nil {
		t.Fatalf("put: %v", err)
	}
	if !core.Stat(path).OK {
		t.Fatalf("store file not created at %s", path)
	}
}

// TestOpenConversationStore_DurableSurvives_Bad proves the durable lane keeps
// its bytes across reopens — per-project state is never wiped.
func TestOpenConversationStore_DurableSurvives_Bad(t *testing.T) {
	ctx := context.Background()
	path := t.TempDir() + "/project.kv"

	first, err := openConversationStore(ctx, path)
	if err != nil {
		t.Fatalf("first open: %v", err)
	}
	if _, err := first.PutBytes(ctx, []byte("durable-project-block"), state.PutOptions{}); err != nil {
		t.Fatalf("put: %v", err)
	}
	if err := first.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	grown := statSize(t, path)

	second, err := openConversationStore(ctx, path)
	if err != nil {
		t.Fatalf("second open: %v", err)
	}
	defer second.Close()
	kept := statSize(t, path)
	if kept != grown {
		t.Fatalf("durable reopen changed the file: %d -> %d, want untouched", grown, kept)
	}
}

// TestOpenConversationStore_MakesParent_Ugly proves a store path in a
// not-yet-existing directory is created (the parent is mkdir'd on first use).
func TestOpenConversationStore_MakesParent_Ugly(t *testing.T) {
	ctx := context.Background()
	path := t.TempDir() + "/nested/dir/project.kv"
	store, err := openConversationStore(ctx, path)
	if err != nil {
		t.Fatalf("open with missing parent: %v", err)
	}
	defer store.Close()
	if !core.Stat(path).OK {
		t.Fatalf("store file not created at %s (parent dir not made)", path)
	}
}

// TestWireContinuity_NilEnabler_Ugly pins the cleanup contract: even the
// degraded (no-enabler) path returns a callable cleanup, so RunServe's
// unconditional defer never panics.
func TestWireContinuity_NilEnabler_Ugly(t *testing.T) {
	cleanup := wireContinuity(context.Background(), ServeConfig{}, newHotSwapResolver("", "", 0, nil), nil)
	if cleanup == nil {
		t.Fatal("wireContinuity returned a nil cleanup")
	}
	cleanup()
}
