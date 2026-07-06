// SPDX-Licence-Identifier: EUPL-1.2

package lora

import (
	"context"
	"sync"
	"testing"
)

// fakeLoader records every Load/Unload the Pool drives — it stands in for the
// real go-mlx apply/unload that this package never performs itself. Set loadErr
// / unloadErr to exercise the failure paths.
type fakeLoader struct {
	mu        sync.Mutex
	loaded    []string // ids in load order
	unloaded  []string // ids in unload order
	loads     int
	unloads   int
	loadErr   error
	unloadErr error
}

func (f *fakeLoader) Load(_ context.Context, ref AdapterRef) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.loadErr != nil {
		return f.loadErr
	}
	f.loads++
	f.loaded = append(f.loaded, ref.ID())
	return nil
}

func (f *fakeLoader) Unload(_ context.Context, id string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.unloadErr != nil {
		return f.unloadErr
	}
	f.unloads++
	f.unloaded = append(f.unloaded, id)
	return nil
}

// ref is a tiny helper so the tests read against adapter names, not paths.
func ref(name string) AdapterRef {
	return AdapterRef{Name: name, Path: "/models/" + name, BaseModel: "gemma-e4b"}
}

// errBoom is a sentinel Loader failure for the load-error path.
var errBoom = context.DeadlineExceeded

// TestLora_AdapterRef_ID_Good covers the deterministic id contract: the same
// Name+Path always yields the same id, regardless of which AdapterRef value
// constructed it, and a differing Path changes the id.
func TestLora_AdapterRef_ID_Good(t *testing.T) {
	a := ref("alpha")
	if a.ID() != ref("alpha").ID() {
		t.Fatalf("AdapterRef.ID() not deterministic for the same name+path")
	}
	if a.ID() == (AdapterRef{Name: "alpha", Path: "/other"}).ID() {
		t.Fatalf("AdapterRef.ID() must change when Path changes")
	}
}

// TestLora_AdapterRef_ID_Bad covers the doc-comment's stated collision guard:
// a naive Name+Path concatenation would make ("ab","c") and ("a","bc")
// collide, but the NUL-separated seed must keep their ids distinct.
func TestLora_AdapterRef_ID_Bad(t *testing.T) {
	x := AdapterRef{Name: "ab", Path: "c"}
	y := AdapterRef{Name: "a", Path: "bc"}
	if x.ID() == y.ID() {
		t.Fatalf("AdapterRef.ID() collided for (%+v) and (%+v): NUL separator not preventing concatenation collision", x, y)
	}
}

// TestLora_AdapterRef_ID_Ugly covers the degenerate zero-value ref: an empty
// Name and Path still produce a stable, well-formed SHA-256 hex id rather
// than panicking or returning an empty string.
func TestLora_AdapterRef_ID_Ugly(t *testing.T) {
	var zero AdapterRef
	id := zero.ID()
	if len(id) != 64 {
		t.Fatalf("AdapterRef{}.ID() = %q (len %d), want 64-char SHA-256 hex", id, len(id))
	}
	if id != zero.ID() {
		t.Fatalf("AdapterRef{}.ID() not stable across calls")
	}
}
