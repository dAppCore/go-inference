// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/state"
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

// TestResolveStateStorePath_Ephemeral_Good pins the defaulted lane: an empty
// -state-store resolves to the home-relative conversations.kv AND marks it
// ephemeral — serve made the store, so serve wipes it per run.
func TestResolveStateStorePath_Ephemeral_Good(t *testing.T) {
	path, ephemeral := resolveStateStorePath("")
	if path == "" || !core.Contains(path, "conversations.kv") {
		t.Fatalf("default path = %q, want the home conversations.kv", path)
	}
	if !ephemeral {
		t.Fatal("defaulted store must be ephemeral")
	}
}

// TestResolveStateStorePath_Explicit_Bad pins the durable lane: any explicit
// path — even one equal to the default's literal location — is per-project
// state and is never marked ephemeral.
func TestResolveStateStorePath_Explicit_Bad(t *testing.T) {
	path, ephemeral := resolveStateStorePath("/tmp/project-a.kv")
	if path != "/tmp/project-a.kv" || ephemeral {
		t.Fatalf("explicit path = (%q, ephemeral=%v), want it kept verbatim and durable", path, ephemeral)
	}
}

// TestResolveStateStorePath_Whitespace_Ugly pins trimming: a whitespace-only
// flag is the unset flag, not a store named " ".
func TestResolveStateStorePath_Whitespace_Ugly(t *testing.T) {
	path, ephemeral := resolveStateStorePath("   ")
	if !ephemeral || !core.Contains(path, "conversations.kv") {
		t.Fatalf("whitespace flag resolved to (%q, ephemeral=%v), want the ephemeral default", path, ephemeral)
	}
}

// TestOpenConversationStore_EphemeralWipes_Good proves the launch wipe: a
// store re-opened as ephemeral comes back EMPTY — whatever a previous run (or
// crash) accumulated is gone.
func TestOpenConversationStore_EphemeralWipes_Good(t *testing.T) {
	ctx := context.Background()
	path := t.TempDir() + "/conversations.kv"

	first, err := openConversationStore(ctx, path, true)
	if err != nil {
		t.Fatalf("first open: %v", err)
	}
	if _, err := first.PutBytes(ctx, []byte("stale-turn-block"), state.PutOptions{}); err != nil {
		t.Fatalf("put: %v", err)
	}
	if err := first.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	grown := statSize(t, path)

	second, err := openConversationStore(ctx, path, true)
	if err != nil {
		t.Fatalf("second open: %v", err)
	}
	defer second.Close()
	wiped := statSize(t, path)
	if wiped >= grown {
		t.Fatalf("ephemeral reopen kept old bytes: %d -> %d, want a fresh (smaller) store", grown, wiped)
	}
}

// TestOpenConversationStore_DurableSurvives_Bad proves the explicit lane keeps
// its bytes across reopens — per-project state is never wiped.
func TestOpenConversationStore_DurableSurvives_Bad(t *testing.T) {
	ctx := context.Background()
	path := t.TempDir() + "/project.kv"

	first, err := openConversationStore(ctx, path, false)
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

	second, err := openConversationStore(ctx, path, false)
	if err != nil {
		t.Fatalf("second open: %v", err)
	}
	defer second.Close()
	kept := statSize(t, path)
	if kept != grown {
		t.Fatalf("durable reopen changed the file: %d -> %d, want untouched", grown, kept)
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
