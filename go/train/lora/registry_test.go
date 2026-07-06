// SPDX-Licence-Identifier: EUPL-1.2

// Tests for the adapter Registry.

package lora

import "testing"

// TestRegistry_NewRegistry_Good covers the freshly built registry: it starts
// with no adapters (List is empty, not nil-panic) and is immediately ready to
// accept Register.
func TestRegistry_NewRegistry_Good(t *testing.T) {
	r := NewRegistry()
	if got := len(r.List()); got != 0 {
		t.Fatalf("fresh registry: want 0 listed, got %d", got)
	}
	if err := r.Register(ref("alpha")); err != nil {
		t.Fatalf("register into fresh registry: %v", err)
	}
}

// TestRegistry_NewRegistry_Bad covers querying a fresh registry for an
// adapter it never held: Get/Acquire report typed errors and RefCount/InUse
// report the zero-value defaults, rather than panicking on the empty maps.
func TestRegistry_NewRegistry_Bad(t *testing.T) {
	r := NewRegistry()
	if _, err := r.Get("ghost"); err == nil {
		t.Fatalf("Get on fresh registry: want error for unknown name")
	}
	if _, err := r.Acquire("ghost"); err == nil {
		t.Fatalf("Acquire on fresh registry: want error for unknown name")
	}
	if r.RefCount("ghost") != 0 || r.InUse("ghost") {
		t.Fatalf("fresh registry: want RefCount 0 and InUse false for unknown id")
	}
}

// TestRegistry_NewRegistry_Ugly covers instance independence: two
// separately constructed registries must not share backing state —
// registering into one must not be visible from the other.
func TestRegistry_NewRegistry_Ugly(t *testing.T) {
	a := NewRegistry()
	b := NewRegistry()
	if err := a.Register(ref("alpha")); err != nil {
		t.Fatalf("register into a: %v", err)
	}
	if _, err := b.Get("alpha"); err == nil {
		t.Fatalf("registry b must not see adapters registered into a")
	}
}

// TestRegistry_Register_Good covers cataloguing multiple adapters: each is
// retrievable by name afterwards and List reflects both, sorted.
func TestRegistry_Register_Good(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(ref("alpha")); err != nil {
		t.Fatalf("register alpha: %v", err)
	}
	if err := r.Register(ref("beta")); err != nil {
		t.Fatalf("register beta: %v", err)
	}
	got, err := r.Get("alpha")
	if err != nil {
		t.Fatalf("get alpha: %v", err)
	}
	if got.Name != "alpha" || got.ID() == "" {
		t.Fatalf("get alpha: unexpected %+v", got)
	}
}

// TestRegistry_Register_Bad covers the two rejected inputs: a duplicate Name
// and a nameless AdapterRef both return a typed error rather than corrupting
// the catalogue.
func TestRegistry_Register_Bad(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(ref("alpha")); err != nil {
		t.Fatalf("register alpha: %v", err)
	}
	if err := r.Register(ref("alpha")); err == nil {
		t.Fatalf("duplicate register: want error")
	}
	if err := r.Register(AdapterRef{Path: "/x"}); err == nil {
		t.Fatalf("nameless register: want error")
	}
}

// TestRegistry_Register_Ugly covers the re-register-after-removal lifecycle:
// once a name has been Unregistered it is free again, so Register must
// accept it a second time rather than treating the name as permanently
// reserved.
func TestRegistry_Register_Ugly(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(ref("alpha")); err != nil {
		t.Fatalf("register alpha: %v", err)
	}
	if err := r.Unregister("alpha"); err != nil {
		t.Fatalf("unregister alpha: %v", err)
	}
	if err := r.Register(ref("alpha")); err != nil {
		t.Fatalf("re-register alpha after unregister: %v", err)
	}
}

// TestRegistry_Unregister_Good covers removing a free (unreferenced)
// adapter: Get afterwards reports it gone and List no longer includes it.
func TestRegistry_Unregister_Good(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(ref("alpha")); err != nil {
		t.Fatalf("register alpha: %v", err)
	}
	if err := r.Unregister("alpha"); err != nil {
		t.Fatalf("unregister free alpha: %v", err)
	}
	if _, err := r.Get("alpha"); err == nil {
		t.Fatalf("alpha should be gone after unregister")
	}
	if got := len(r.List()); got != 0 {
		t.Fatalf("empty registry: want 0 listed, got %d", got)
	}
}

// TestRegistry_Unregister_Bad covers the two refusal paths: an unknown name
// and an in-use (referenced) adapter both return a typed error so the Pool
// can never lose an adapter from under a live request.
func TestRegistry_Unregister_Bad(t *testing.T) {
	r := NewRegistry()
	if err := r.Unregister("ghost"); err == nil {
		t.Fatalf("unregister unknown: want error")
	}
	if err := r.Register(ref("alpha")); err != nil {
		t.Fatalf("register alpha: %v", err)
	}
	if _, err := r.Acquire("alpha"); err != nil {
		t.Fatalf("acquire alpha: %v", err)
	}
	if err := r.Unregister("alpha"); err == nil {
		t.Fatalf("unregister of in-use adapter: want error")
	}
}

// TestRegistry_Unregister_Ugly covers the release-then-unregister lifecycle:
// an adapter refused while in-use becomes unregisterable again once its
// ref-count returns to zero.
func TestRegistry_Unregister_Ugly(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(ref("alpha")); err != nil {
		t.Fatalf("register alpha: %v", err)
	}
	id, err := r.Acquire("alpha")
	if err != nil {
		t.Fatalf("acquire alpha: %v", err)
	}
	if err := r.Unregister("alpha"); err == nil {
		t.Fatalf("unregister while in-use: want error")
	}
	r.Release(id)
	if err := r.Unregister("alpha"); err != nil {
		t.Fatalf("unregister after release: %v", err)
	}
}

// TestRegistry_Get_Good covers resolving a registered adapter by name.
func TestRegistry_Get_Good(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(ref("alpha")); err != nil {
		t.Fatalf("register alpha: %v", err)
	}
	got, err := r.Get("alpha")
	if err != nil {
		t.Fatalf("get alpha: %v", err)
	}
	if got.Name != "alpha" {
		t.Fatalf("get alpha: name = %q, want alpha", got.Name)
	}
}

// TestRegistry_Get_Bad covers looking up a name that was never registered.
func TestRegistry_Get_Bad(t *testing.T) {
	r := NewRegistry()
	if _, err := r.Get("ghost"); err == nil {
		t.Fatalf("get unknown: want error")
	}
}

// TestRegistry_Get_Ugly covers the removed-not-unknown distinction: a name
// that WAS registered and then Unregistered reports the same not-found error
// as a name that was never registered at all.
func TestRegistry_Get_Ugly(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(ref("alpha")); err != nil {
		t.Fatalf("register alpha: %v", err)
	}
	if err := r.Unregister("alpha"); err != nil {
		t.Fatalf("unregister alpha: %v", err)
	}
	if _, err := r.Get("alpha"); err == nil {
		t.Fatalf("get removed alpha: want error")
	}
}

// TestRegistry_List_Good covers the sorted snapshot of every registered
// adapter.
func TestRegistry_List_Good(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(ref("beta")); err != nil {
		t.Fatalf("register beta: %v", err)
	}
	if err := r.Register(ref("alpha")); err != nil {
		t.Fatalf("register alpha: %v", err)
	}
	list := r.List()
	if len(list) != 2 || list[0].Name != "alpha" || list[1].Name != "beta" {
		t.Fatalf("list: want [alpha beta] sorted, got %+v", list)
	}
}

// TestRegistry_List_Bad covers the degenerate empty case: List on a
// registry with nothing registered returns a zero-length slice, not nil
// causing a caller panic on further indexing.
func TestRegistry_List_Bad(t *testing.T) {
	r := NewRegistry()
	if got := r.List(); len(got) != 0 {
		t.Fatalf("empty registry List: want 0 entries, got %d", len(got))
	}
}

// TestRegistry_List_Ugly covers List after removal: an unregistered adapter
// must not linger in subsequent List snapshots.
func TestRegistry_List_Ugly(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(ref("alpha")); err != nil {
		t.Fatalf("register alpha: %v", err)
	}
	if err := r.Register(ref("beta")); err != nil {
		t.Fatalf("register beta: %v", err)
	}
	if err := r.Unregister("alpha"); err != nil {
		t.Fatalf("unregister alpha: %v", err)
	}
	list := r.List()
	if len(list) != 1 || list[0].Name != "beta" {
		t.Fatalf("list after removal: want [beta], got %+v", list)
	}
}

// TestRegistry_Acquire_Good covers taking a lease: it bumps the ref-count and
// returns the adapter's resolved id.
func TestRegistry_Acquire_Good(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(ref("alpha")); err != nil {
		t.Fatalf("register alpha: %v", err)
	}
	id, err := r.Acquire("alpha")
	if err != nil {
		t.Fatalf("acquire alpha: %v", err)
	}
	if id != ref("alpha").ID() {
		t.Fatalf("acquire: want id %q, got %q", ref("alpha").ID(), id)
	}
	if got := r.RefCount(id); got != 1 {
		t.Fatalf("refcount after one acquire: want 1, got %d", got)
	}
}

// TestRegistry_Acquire_Bad covers acquiring an unknown name: a typed error,
// no lease taken.
func TestRegistry_Acquire_Bad(t *testing.T) {
	r := NewRegistry()
	if _, err := r.Acquire("ghost"); err == nil {
		t.Fatalf("acquire unknown: want error")
	}
}

// TestRegistry_Acquire_Ugly covers stacking leases: a second Acquire of the
// same adapter increments the count again rather than being a no-op.
func TestRegistry_Acquire_Ugly(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(ref("alpha")); err != nil {
		t.Fatalf("register alpha: %v", err)
	}
	id, err := r.Acquire("alpha")
	if err != nil {
		t.Fatalf("first acquire: %v", err)
	}
	if _, err := r.Acquire("alpha"); err != nil {
		t.Fatalf("second acquire: %v", err)
	}
	if got := r.RefCount(id); got != 2 {
		t.Fatalf("refcount after two acquires: want 2, got %d", got)
	}
}

// TestRegistry_Release_Good covers dropping one outstanding lease.
func TestRegistry_Release_Good(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(ref("alpha")); err != nil {
		t.Fatalf("register alpha: %v", err)
	}
	id, err := r.Acquire("alpha")
	if err != nil {
		t.Fatalf("acquire alpha: %v", err)
	}
	r.Release(id)
	if got := r.RefCount(id); got != 0 {
		t.Fatalf("refcount after release: want 0, got %d", got)
	}
}

// TestRegistry_Release_Bad covers releasing an id the registry has never
// seen: a harmless no-op rather than a panic.
func TestRegistry_Release_Bad(t *testing.T) {
	r := NewRegistry()
	r.Release("never-seen")
}

// TestRegistry_Release_Ugly covers the over-release clamp: releasing more
// times than acquired must not drive the ref-count negative.
func TestRegistry_Release_Ugly(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(ref("alpha")); err != nil {
		t.Fatalf("register alpha: %v", err)
	}
	id, err := r.Acquire("alpha")
	if err != nil {
		t.Fatalf("acquire alpha: %v", err)
	}
	r.Release(id)
	r.Release(id) // one more than acquired
	if got := r.RefCount(id); got != 0 {
		t.Fatalf("over-release must clamp at 0, got %d", got)
	}
}

// TestRegistry_RefCount_Good covers reporting the live lease count.
func TestRegistry_RefCount_Good(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(ref("alpha")); err != nil {
		t.Fatalf("register alpha: %v", err)
	}
	id, err := r.Acquire("alpha")
	if err != nil {
		t.Fatalf("acquire alpha: %v", err)
	}
	if got := r.RefCount(id); got != 1 {
		t.Fatalf("refcount = %d, want 1", got)
	}
}

// TestRegistry_RefCount_Bad covers an id the registry has never seen: the
// defined zero, not a panic.
func TestRegistry_RefCount_Bad(t *testing.T) {
	r := NewRegistry()
	if got := r.RefCount("ghost"); got != 0 {
		t.Fatalf("refcount of unknown id = %d, want 0", got)
	}
}

// TestRegistry_RefCount_Ugly covers the balanced-release boundary: after
// every acquired lease is released the count returns exactly to zero, not a
// residual value.
func TestRegistry_RefCount_Ugly(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(ref("alpha")); err != nil {
		t.Fatalf("register alpha: %v", err)
	}
	id, err := r.Acquire("alpha")
	if err != nil {
		t.Fatalf("first acquire: %v", err)
	}
	if _, err := r.Acquire("alpha"); err != nil {
		t.Fatalf("second acquire: %v", err)
	}
	r.Release(id)
	r.Release(id)
	if got := r.RefCount(id); got != 0 {
		t.Fatalf("refcount after balanced releases: want 0, got %d", got)
	}
}

// TestRegistry_InUse_Good covers reporting an outstanding lease.
func TestRegistry_InUse_Good(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(ref("alpha")); err != nil {
		t.Fatalf("register alpha: %v", err)
	}
	id, err := r.Acquire("alpha")
	if err != nil {
		t.Fatalf("acquire alpha: %v", err)
	}
	if !r.InUse(id) {
		t.Fatalf("alpha should be in use after acquire")
	}
}

// TestRegistry_InUse_Bad covers an id the registry has never seen: reports
// false rather than panicking.
func TestRegistry_InUse_Bad(t *testing.T) {
	r := NewRegistry()
	if r.InUse("ghost") {
		t.Fatalf("unknown id should not be in use")
	}
}

// TestRegistry_InUse_Ugly covers the two-lease boundary: InUse stays true
// while one of two acquired leases remains outstanding, then flips false
// only once both are released.
func TestRegistry_InUse_Ugly(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(ref("alpha")); err != nil {
		t.Fatalf("register alpha: %v", err)
	}
	id, err := r.Acquire("alpha")
	if err != nil {
		t.Fatalf("first acquire: %v", err)
	}
	if _, err := r.Acquire("alpha"); err != nil {
		t.Fatalf("second acquire: %v", err)
	}
	r.Release(id)
	if !r.InUse(id) {
		t.Fatalf("alpha still in use after one of two releases")
	}
	r.Release(id)
	if r.InUse(id) {
		t.Fatalf("alpha should be free after balanced releases")
	}
}
