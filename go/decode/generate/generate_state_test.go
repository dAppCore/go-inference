// SPDX-Licence-Identifier: EUPL-1.2

package generate

import (
	"context"
	"testing"

	core "dappco.re/go"
)

// generate_state_test.go covers the pure, no-model arms of generate_state.go:
// openStateStore's create / reopen / mkdir-failure paths. The model-bound turn
// loop (runStateTurn → runStateSession) is exercised by the supervised real-model
// test, which skips without the checkpoint + metallib.

// TestOpenStateStore_FreshCreatesDirAndFile_Good proves openStateStore creates
// the store file and its missing parent directory on first use, returning a
// usable store.
func TestOpenStateStore_FreshCreatesDirAndFile_Good(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "state", "agent.kv") // the "state" parent dir does not exist yet
	store, err := openStateStore(context.Background(), path)
	if err != nil {
		t.Fatalf("openStateStore(fresh): %v", err)
	}
	defer store.Close()
	if store == nil {
		t.Fatal("openStateStore returned a nil store with no error")
	}
	if !core.Stat(path).OK {
		t.Fatalf("openStateStore did not create the store file at %s", path)
	}
}

// TestOpenStateStore_ReopenExisting_Good proves an already-present store file
// takes the Open (not Create) path and reopens cleanly.
func TestOpenStateStore_ReopenExisting_Good(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "agent.kv")
	first, err := openStateStore(context.Background(), path)
	if err != nil {
		t.Fatalf("openStateStore(create): %v", err)
	}
	first.Close()

	second, err := openStateStore(context.Background(), path)
	if err != nil {
		t.Fatalf("openStateStore(reopen): %v", err)
	}
	defer second.Close()
	if second == nil {
		t.Fatal("reopened store is nil")
	}
}

// TestOpenStateStore_ParentIsFile_Bad proves openStateStore surfaces the mkdir
// failure when the store's parent path is an existing file (not a directory)
// rather than silently proceeding to a Create that would fault deeper.
func TestOpenStateStore_ParentIsFile_Bad(t *testing.T) {
	blocker := core.PathJoin(t.TempDir(), "blocker")
	if r := core.WriteFile(blocker, []byte("x"), 0o644); !r.OK {
		t.Fatalf("seed blocker file: %v", r.Value)
	}
	// The store path treats the file as its parent directory — MkdirAll must fail.
	path := core.PathJoin(blocker, "agent.kv")
	if _, err := openStateStore(context.Background(), path); err == nil {
		t.Fatal("openStateStore under a file-as-directory: want mkdir error, got nil")
	}
}
