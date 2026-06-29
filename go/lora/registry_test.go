// SPDX-Licence-Identifier: EUPL-1.2

// Tests for the adapter Registry.

package lora

import "testing"

// TestLoRA_Registry_Good covers the adapter book-keeping: register then look up,
// list is sorted, deterministic ids are stable for the same name+path, and
// acquire/release ref-counting tracks in-flight use.
func TestLoRA_Registry_Good(t *testing.T) {
	r := NewRegistry()

	a := ref("alpha")
	b := ref("beta")
	if err := r.Register(a); err != nil {
		t.Fatalf("register alpha: %v", err)
	}
	if err := r.Register(b); err != nil {
		t.Fatalf("register beta: %v", err)
	}

	got, err := r.Get("alpha")
	if err != nil {
		t.Fatalf("get alpha: %v", err)
	}
	if got.Name != "alpha" || got.ID() == "" {
		t.Fatalf("get alpha: unexpected %+v", got)
	}

	// Deterministic id: same name+path → same id, regardless of construction.
	if ref("alpha").ID() != a.ID() {
		t.Fatalf("deterministic id mismatch for alpha")
	}
	// Different path → different id.
	if (AdapterRef{Name: "alpha", Path: "/other"}).ID() == a.ID() {
		t.Fatalf("differing path must change the id")
	}

	// List is sorted by name for deterministic output.
	list := r.List()
	if len(list) != 2 || list[0].Name != "alpha" || list[1].Name != "beta" {
		t.Fatalf("list: want [alpha beta], got %+v", list)
	}

	// Acquire bumps the ref-count and returns the resolved id.
	id, err := r.Acquire("alpha")
	if err != nil {
		t.Fatalf("acquire alpha: %v", err)
	}
	if id != a.ID() {
		t.Fatalf("acquire: want id %q, got %q", a.ID(), id)
	}
	if !r.InUse(id) {
		t.Fatalf("alpha should be in use after acquire")
	}
	if got := r.RefCount(id); got != 1 {
		t.Fatalf("refcount after one acquire: want 1, got %d", got)
	}

	// A second acquire stacks the count; one release leaves it still in use.
	if _, err := r.Acquire("alpha"); err != nil {
		t.Fatalf("second acquire: %v", err)
	}
	if got := r.RefCount(id); got != 2 {
		t.Fatalf("refcount after two acquires: want 2, got %d", got)
	}
	r.Release(id)
	if !r.InUse(id) {
		t.Fatalf("alpha still in use after one of two releases")
	}
	r.Release(id)
	if r.InUse(id) {
		t.Fatalf("alpha should be free after balanced releases")
	}
	if got := r.RefCount(id); got != 0 {
		t.Fatalf("refcount after balanced releases: want 0, got %d", got)
	}
}

// TestLoRA_Registry_Bad covers the error paths: re-registering a name, looking up
// / acquiring an unknown name, and unregistering an unknown name all return a
// typed error rather than corrupting the registry.
func TestLoRA_Registry_Bad(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(ref("alpha")); err != nil {
		t.Fatalf("register alpha: %v", err)
	}

	if err := r.Register(ref("alpha")); err == nil {
		t.Fatalf("duplicate register: want error")
	}

	if _, err := r.Get("ghost"); err == nil {
		t.Fatalf("get unknown: want error")
	}
	if _, err := r.Acquire("ghost"); err == nil {
		t.Fatalf("acquire unknown: want error")
	}
	if err := r.Unregister("ghost"); err == nil {
		t.Fatalf("unregister unknown: want error")
	}

	// Registering an unnamed adapter is rejected — the name is the lookup key.
	if err := r.Register(AdapterRef{Path: "/x"}); err == nil {
		t.Fatalf("nameless register: want error")
	}
}

// TestLoRA_Registry_Ugly covers boundary book-keeping: releasing an id with no
// outstanding refs never drops below zero, unregister removes a free adapter, and
// unregistering an in-use adapter is refused so the Pool can't lose an in-flight
// adapter from under a request.
func TestLoRA_Registry_Ugly(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(ref("alpha")); err != nil {
		t.Fatalf("register alpha: %v", err)
	}
	id := ref("alpha").ID()

	// Release with no outstanding ref is a harmless no-op (clamped at zero).
	r.Release(id)
	if got := r.RefCount(id); got != 0 {
		t.Fatalf("over-release must clamp at 0, got %d", got)
	}
	// Releasing an utterly unknown id is also a no-op (no panic).
	r.Release("never-seen")

	// In-use adapters cannot be unregistered (would orphan an in-flight ref).
	if _, err := r.Acquire("alpha"); err != nil {
		t.Fatalf("acquire alpha: %v", err)
	}
	if err := r.Unregister("alpha"); err == nil {
		t.Fatalf("unregister of in-use adapter: want error")
	}
	r.Release(id)

	// Once free, unregister succeeds and the name is gone.
	if err := r.Unregister("alpha"); err != nil {
		t.Fatalf("unregister free alpha: %v", err)
	}
	if _, err := r.Get("alpha"); err == nil {
		t.Fatalf("alpha should be gone after unregister")
	}
	if got := len(r.List()); got != 0 {
		t.Fatalf("empty registry: want 0 listed, got %d", got)
	}

	// RefCount / InUse of an unknown id are defined: zero and false.
	if r.RefCount("ghost") != 0 || r.InUse("ghost") {
		t.Fatalf("unknown id: want refcount 0, not in use")
	}
}
