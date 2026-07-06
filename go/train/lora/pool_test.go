// SPDX-Licence-Identifier: EUPL-1.2

// Tests for the adapter serving Pool.

package lora

import (
	"context"
	"testing"
)

// TestPool_NewPool_Good covers building a pool from a Config: it is
// immediately ready to Register and serve adapters.
func TestPool_NewPool_Good(t *testing.T) {
	p := NewPool(Config{Loader: &fakeLoader{}, Policy: NewLRUEvictionPolicy(), Capacity: 2})
	if err := p.Register(ref("alpha")); err != nil {
		t.Fatalf("register into freshly built pool: %v", err)
	}
	if got := len(p.Resident()); got != 0 {
		t.Fatalf("freshly built pool: want 0 resident, got %d", got)
	}
}

// TestPool_NewPool_Bad covers the malformed-Capacity guard: a negative
// Capacity clamps to zero rather than propagating a nonsensical negative
// bound (which would make `len(p.resident) >= p.capacity` always true).
func TestPool_NewPool_Bad(t *testing.T) {
	p := NewPool(Config{Loader: &fakeLoader{}, Policy: NewLRUEvictionPolicy(), Capacity: -1})
	if err := p.Register(ref("a")); err != nil {
		t.Fatalf("register a: %v", err)
	}
	if _, _, err := p.Use(context.Background(), "a"); !IsCannotFit(err) {
		t.Fatalf("negative-capacity pool: want CannotFit like a zero-capacity pool, got %v", err)
	}
}

// TestPool_NewPool_Ugly covers the boundary Capacity of exactly zero (as
// opposed to a negative value clamped to it): an explicitly empty pool can
// never admit any adapter.
func TestPool_NewPool_Ugly(t *testing.T) {
	p := NewPool(Config{Loader: &fakeLoader{}, Policy: NewLRUEvictionPolicy(), Capacity: 0})
	if err := p.Register(ref("a")); err != nil {
		t.Fatalf("register a: %v", err)
	}
	if _, _, err := p.Use(context.Background(), "a"); !IsCannotFit(err) {
		t.Fatalf("zero-capacity pool: want CannotFit, got %v", err)
	}
}

// TestPool_Register_Good covers cataloguing an adapter through the Pool: it
// becomes reachable by name for Use.
func TestPool_Register_Good(t *testing.T) {
	fl := &fakeLoader{}
	p := NewPool(Config{Loader: fl, Policy: NewLRUEvictionPolicy(), Capacity: 2})
	if err := p.Register(ref("alpha")); err != nil {
		t.Fatalf("register alpha: %v", err)
	}
	if _, _, err := p.Use(context.Background(), "alpha"); err != nil {
		t.Fatalf("use registered alpha: %v", err)
	}
}

// TestPool_Register_Bad covers the duplicate-name refusal delegated to the
// Registry.
func TestPool_Register_Bad(t *testing.T) {
	p := NewPool(Config{Loader: &fakeLoader{}, Policy: NewLRUEvictionPolicy(), Capacity: 2})
	if err := p.Register(ref("alpha")); err != nil {
		t.Fatalf("register alpha: %v", err)
	}
	if err := p.Register(ref("alpha")); err == nil {
		t.Fatalf("duplicate register: want error")
	}
}

// TestPool_Register_Ugly covers the nameless-adapter refusal: the Name is
// the Registry's lookup key, so an empty one must be rejected rather than
// silently accepted.
func TestPool_Register_Ugly(t *testing.T) {
	p := NewPool(Config{Loader: &fakeLoader{}, Policy: NewLRUEvictionPolicy(), Capacity: 2})
	if err := p.Register(AdapterRef{Path: "/x"}); err == nil {
		t.Fatalf("nameless register: want error")
	}
}

// TestPool_Unregister_Good covers removing a resident, unreferenced
// adapter: it is unloaded and dropped from the working set.
func TestPool_Unregister_Good(t *testing.T) {
	fl := &fakeLoader{}
	p := NewPool(Config{Loader: fl, Policy: NewLRUEvictionPolicy(), Capacity: 2})
	if err := p.Register(ref("a")); err != nil {
		t.Fatalf("register a: %v", err)
	}
	if _, release, err := p.Use(context.Background(), "a"); err != nil {
		t.Fatalf("use a: %v", err)
	} else {
		release()
	}
	if err := p.Unregister("a"); err != nil {
		t.Fatalf("unregister free a: %v", err)
	}
	if p.IsResident("a") {
		t.Fatalf("unregistered a must not be resident")
	}
	if fl.unloads != 1 {
		t.Fatalf("unregister should unload the resident adapter, unloads=%d", fl.unloads)
	}
}

// TestPool_Unregister_Bad covers the two refusal paths: an unknown adapter
// and one still in flight (referenced) both return a typed error.
func TestPool_Unregister_Bad(t *testing.T) {
	fl := &fakeLoader{}
	p := NewPool(Config{Loader: fl, Policy: NewLRUEvictionPolicy(), Capacity: 2})
	if err := p.Unregister("ghost"); err == nil {
		t.Fatalf("unregister unknown: want error")
	}
	if err := p.Register(ref("a")); err != nil {
		t.Fatalf("register a: %v", err)
	}
	if _, _, err := p.Use(context.Background(), "a"); err != nil {
		t.Fatalf("use a: %v", err)
	}
	if err := p.Unregister("a"); err == nil {
		t.Fatalf("unregister in-use a: want error")
	}
}

// TestPool_Unregister_Ugly covers unregistering an adapter that was
// registered but never Used: the never-resident branch must skip the
// Loader.Unload call entirely (nothing was ever loaded).
func TestPool_Unregister_Ugly(t *testing.T) {
	fl := &fakeLoader{}
	p := NewPool(Config{Loader: fl, Policy: NewLRUEvictionPolicy(), Capacity: 2})
	if err := p.Register(ref("a")); err != nil {
		t.Fatalf("register a: %v", err)
	}
	if err := p.Unregister("a"); err != nil {
		t.Fatalf("unregister never-used a: %v", err)
	}
	if fl.unloads != 0 {
		t.Fatalf("unregistering a never-resident adapter must not unload, unloads=%d", fl.unloads)
	}
}

// TestPool_Use_Good covers the serving manager happy path: first Use loads
// the adapter, a second Use of the same adapter is a resident hit (no
// reload), and a second adapter co-resides under capacity.
func TestPool_Use_Good(t *testing.T) {
	fl := &fakeLoader{}
	p := NewPool(Config{Loader: fl, Policy: NewLRUEvictionPolicy(), Capacity: 2})
	if err := p.Register(ref("alpha")); err != nil {
		t.Fatalf("register alpha: %v", err)
	}

	ctx := context.Background()
	id, release, err := p.Use(ctx, "alpha")
	if err != nil {
		t.Fatalf("first use alpha: %v", err)
	}
	if id != ref("alpha").ID() {
		t.Fatalf("use: want alpha id, got %q", id)
	}
	if fl.loads != 1 {
		t.Fatalf("first use: want 1 load, got %d", fl.loads)
	}
	if !p.IsResident("alpha") {
		t.Fatalf("alpha should be resident after use")
	}
	release()

	// Second use of a resident adapter does NOT reload.
	_, release2, err := p.Use(ctx, "alpha")
	if err != nil {
		t.Fatalf("second use alpha: %v", err)
	}
	if fl.loads != 1 {
		t.Fatalf("resident hit must not reload, loads=%d", fl.loads)
	}
	release2()

	// A second adapter co-resides under capacity 2.
	if err := p.Register(ref("beta")); err != nil {
		t.Fatalf("register beta: %v", err)
	}
	_, release3, err := p.Use(ctx, "beta")
	if err != nil {
		t.Fatalf("use beta: %v", err)
	}
	release3()
	if fl.loads != 2 {
		t.Fatalf("want 2 distinct loads, got %d", fl.loads)
	}
	res := p.Resident()
	if len(res) != 2 || res[0] != "alpha" || res[1] != "beta" {
		t.Fatalf("resident: want [alpha beta], got %v", res)
	}
}

// TestPool_Use_Bad covers the admission-failure and unknown-name paths: an
// unknown adapter, a zero-capacity pool (can't fit even when empty), and a
// full pool where every resident adapter is referenced or pinned (can't
// admit).
func TestPool_Use_Bad(t *testing.T) {
	ctx := context.Background()
	fl := &fakeLoader{}

	// Unknown adapter → typed error, nothing loaded.
	p := NewPool(Config{Loader: fl, Policy: NewLRUEvictionPolicy(), Capacity: 2})
	if _, _, err := p.Use(ctx, "ghost"); err == nil {
		t.Fatalf("use unknown: want error")
	}

	// Zero capacity: an adapter can never fit even on an empty pool.
	zp := NewPool(Config{Loader: fl, Policy: NewLRUEvictionPolicy(), Capacity: 0})
	if err := zp.Register(ref("a")); err != nil {
		t.Fatalf("register a: %v", err)
	}
	if _, _, err := zp.Use(ctx, "a"); !IsCannotFit(err) {
		t.Fatalf("zero-capacity use: want CannotFit, got %v", err)
	}

	// Admission that can't evict enough: capacity 1, hold the sole resident,
	// then demand a different adapter → nothing evictable → typed error.
	bp := NewPool(Config{Loader: fl, Policy: NewLRUEvictionPolicy(), Capacity: 1})
	for _, n := range []string{"a", "b"} {
		if err := bp.Register(ref(n)); err != nil {
			t.Fatalf("register %s: %v", n, err)
		}
	}
	_, ra, err := bp.Use(ctx, "a") // a now resident AND referenced
	if err != nil {
		t.Fatalf("use a: %v", err)
	}
	if _, _, err := bp.Use(ctx, "b"); !IsCannotAdmit(err) {
		t.Fatalf("no-evictable use: want CannotAdmit, got %v", err)
	}
	ra()
}

// TestPool_Use_Ugly covers eviction and its interaction with pinning and
// Loader failure: at capacity, the LRU unreferenced resident is evicted (and
// unloaded); a referenced adapter is spared in favour of an unreferenced
// one; a pinned adapter is never evicted; and a Loader failure surfaces the
// error, leaves nothing resident, and frees the reserved slot.
func TestPool_Use_Ugly(t *testing.T) {
	ctx := context.Background()
	fl := &fakeLoader{}
	p := NewPool(Config{Loader: fl, Policy: NewLRUEvictionPolicy(), Capacity: 2})
	for _, n := range []string{"a", "b", "c"} {
		if err := p.Register(ref(n)); err != nil {
			t.Fatalf("register %s: %v", n, err)
		}
	}

	// Load a then b (a is now LRU), each released so neither is referenced.
	_, ra, _ := p.Use(ctx, "a")
	ra()
	_, rb, _ := p.Use(ctx, "b")
	rb()

	// c at capacity → evict LRU unreferenced (a), load c.
	_, rc, err := p.Use(ctx, "c")
	if err != nil {
		t.Fatalf("use c: %v", err)
	}
	rc()
	if p.IsResident("a") {
		t.Fatalf("a (LRU) should have been evicted for c")
	}
	if fl.unloads != 1 || fl.unloaded[0] != ref("a").ID() {
		t.Fatalf("want a unloaded, got unloads=%d %v", fl.unloads, fl.unloaded)
	}

	// Now resident: b, c. Hold a ref on b (the LRU), then use a again.
	// b is LRU but referenced → c must be evicted instead.
	_, rb2, _ := p.Use(ctx, "b") // b held in-flight
	_, ra2, err := p.Use(ctx, "a")
	if err != nil {
		t.Fatalf("reuse a: %v", err)
	}
	if !p.IsResident("b") {
		t.Fatalf("referenced b must not be evicted")
	}
	if p.IsResident("c") {
		t.Fatalf("unreferenced c should have been evicted, not b")
	}
	ra2()
	rb2()

	// Pinning: resident now a, b. Pin a (the LRU), use c → b evicted, a spared.
	p.Pin("a")
	_, rc2, err := p.Use(ctx, "c")
	if err != nil {
		t.Fatalf("use c with a pinned: %v", err)
	}
	rc2()
	if !p.IsResident("a") {
		t.Fatalf("pinned a must survive eviction")
	}
	if p.IsResident("b") {
		t.Fatalf("b should have been evicted (a pinned)")
	}

	// Load failure surfaces from the Loader, leaves nothing resident, and the
	// reserved capacity slot is released so a later good load still fits.
	ep := NewPool(Config{Loader: &fakeLoader{loadErr: errBoom}, Policy: NewLRUEvictionPolicy(), Capacity: 1})
	if err := ep.Register(ref("a")); err != nil {
		t.Fatalf("register a: %v", err)
	}
	if _, _, err := ep.Use(ctx, "a"); err == nil {
		t.Fatalf("load failure: want error")
	}
	if ep.IsResident("a") {
		t.Fatalf("failed load must not be resident")
	}
	if got := len(ep.Resident()); got != 0 {
		t.Fatalf("failed load must free its slot, resident=%v", ep.Resident())
	}
}

// TestPool_Pin_Good covers pinning a resident adapter: it survives an
// eviction that would otherwise have picked it as the LRU victim.
func TestPool_Pin_Good(t *testing.T) {
	ctx := context.Background()
	fl := &fakeLoader{}
	p := NewPool(Config{Loader: fl, Policy: NewLRUEvictionPolicy(), Capacity: 1})
	for _, n := range []string{"a", "b"} {
		if err := p.Register(ref(n)); err != nil {
			t.Fatalf("register %s: %v", n, err)
		}
	}
	_, ra, _ := p.Use(ctx, "a")
	ra()
	p.Pin("a")
	if _, _, err := p.Use(ctx, "b"); !IsCannotAdmit(err) {
		t.Fatalf("pinned-only blockage: want CannotAdmit, got %v", err)
	}
	if !p.IsResident("a") {
		t.Fatalf("pinned a must still be resident")
	}
}

// TestPool_Pin_Bad covers pinning a name the Pool has never registered: a
// no-op, no panic, and it never becomes resident as a side effect.
func TestPool_Pin_Bad(t *testing.T) {
	p := NewPool(Config{Loader: &fakeLoader{}, Policy: NewLRUEvictionPolicy(), Capacity: 2})
	p.Pin("nobody")
	if p.IsResident("nobody") {
		t.Fatalf("pinning an unregistered adapter must not make it resident")
	}
}

// TestPool_Pin_Ugly covers pinning a registered-but-not-resident adapter:
// pin only protects something already loaded, so this is also a no-op
// (mirrors residency.Pin) rather than pre-loading it.
func TestPool_Pin_Ugly(t *testing.T) {
	p := NewPool(Config{Loader: &fakeLoader{}, Policy: NewLRUEvictionPolicy(), Capacity: 2})
	if err := p.Register(ref("a")); err != nil {
		t.Fatalf("register a: %v", err)
	}
	p.Pin("a")
	if p.IsResident("a") {
		t.Fatalf("pinning a never-used adapter must not make it resident")
	}
}

// TestPool_Unpin_Good covers returning a pinned adapter to normal eviction
// eligibility: once unpinned it can be evicted like any other unreferenced
// resident.
func TestPool_Unpin_Good(t *testing.T) {
	ctx := context.Background()
	fl := &fakeLoader{}
	p := NewPool(Config{Loader: fl, Policy: NewLRUEvictionPolicy(), Capacity: 1})
	for _, n := range []string{"a", "b"} {
		if err := p.Register(ref(n)); err != nil {
			t.Fatalf("register %s: %v", n, err)
		}
	}
	_, ra, _ := p.Use(ctx, "a")
	ra()
	p.Pin("a")
	p.Unpin("a")
	if _, _, err := p.Use(ctx, "b"); err != nil {
		t.Fatalf("use b after unpinning a: %v", err)
	}
	if p.IsResident("a") {
		t.Fatalf("unpinned a should have been evicted for b")
	}
}

// TestPool_Unpin_Bad covers unpinning a name the Pool has never registered:
// a no-op, no panic.
func TestPool_Unpin_Bad(t *testing.T) {
	p := NewPool(Config{Loader: &fakeLoader{}, Policy: NewLRUEvictionPolicy(), Capacity: 2})
	p.Unpin("nobody")
}

// TestPool_Unpin_Ugly covers unpinning an adapter that was never pinned in
// the first place: a harmless no-op, no state corruption.
func TestPool_Unpin_Ugly(t *testing.T) {
	ctx := context.Background()
	fl := &fakeLoader{}
	p := NewPool(Config{Loader: fl, Policy: NewLRUEvictionPolicy(), Capacity: 2})
	if err := p.Register(ref("a")); err != nil {
		t.Fatalf("register a: %v", err)
	}
	if _, release, err := p.Use(ctx, "a"); err != nil {
		t.Fatalf("use a: %v", err)
	} else {
		release()
	}
	p.Unpin("a") // was never pinned
	if !p.IsResident("a") {
		t.Fatalf("unpinning a never-pinned adapter must not evict it")
	}
}

// TestPool_IsResident_Good covers reporting that an adapter is currently
// loaded on the base model.
func TestPool_IsResident_Good(t *testing.T) {
	ctx := context.Background()
	p := NewPool(Config{Loader: &fakeLoader{}, Policy: NewLRUEvictionPolicy(), Capacity: 2})
	if err := p.Register(ref("a")); err != nil {
		t.Fatalf("register a: %v", err)
	}
	if _, release, err := p.Use(ctx, "a"); err != nil {
		t.Fatalf("use a: %v", err)
	} else {
		release()
	}
	if !p.IsResident("a") {
		t.Fatalf("a should be resident after use")
	}
}

// TestPool_IsResident_Bad covers a name the Pool has never registered:
// reports false rather than panicking.
func TestPool_IsResident_Bad(t *testing.T) {
	p := NewPool(Config{Loader: &fakeLoader{}, Policy: NewLRUEvictionPolicy(), Capacity: 2})
	if p.IsResident("ghost") {
		t.Fatalf("unregistered adapter should not be resident")
	}
}

// TestPool_IsResident_Ugly covers a registered-but-never-used adapter:
// registration alone does not make an adapter resident.
func TestPool_IsResident_Ugly(t *testing.T) {
	p := NewPool(Config{Loader: &fakeLoader{}, Policy: NewLRUEvictionPolicy(), Capacity: 2})
	if err := p.Register(ref("a")); err != nil {
		t.Fatalf("register a: %v", err)
	}
	if p.IsResident("a") {
		t.Fatalf("registered-but-never-used adapter should not be resident")
	}
}

// TestPool_Resident_Good covers the sorted snapshot of every currently
// loaded adapter name.
func TestPool_Resident_Good(t *testing.T) {
	ctx := context.Background()
	p := NewPool(Config{Loader: &fakeLoader{}, Policy: NewLRUEvictionPolicy(), Capacity: 2})
	for _, n := range []string{"beta", "alpha"} {
		if err := p.Register(ref(n)); err != nil {
			t.Fatalf("register %s: %v", n, err)
		}
		if _, release, err := p.Use(ctx, n); err != nil {
			t.Fatalf("use %s: %v", n, err)
		} else {
			release()
		}
	}
	res := p.Resident()
	if len(res) != 2 || res[0] != "alpha" || res[1] != "beta" {
		t.Fatalf("resident: want [alpha beta] sorted, got %v", res)
	}
}

// TestPool_Resident_Bad covers the degenerate empty case: a pool with
// nothing ever used reports a zero-length slice, not nil causing a caller
// panic on further indexing.
func TestPool_Resident_Bad(t *testing.T) {
	p := NewPool(Config{Loader: &fakeLoader{}, Policy: NewLRUEvictionPolicy(), Capacity: 2})
	if got := p.Resident(); len(got) != 0 {
		t.Fatalf("empty pool Resident: want 0 entries, got %d", len(got))
	}
}

// TestPool_Resident_Ugly covers Resident after an eviction: the evicted
// adapter's name must no longer appear once a replacement has taken its
// slot.
func TestPool_Resident_Ugly(t *testing.T) {
	ctx := context.Background()
	p := NewPool(Config{Loader: &fakeLoader{}, Policy: NewLRUEvictionPolicy(), Capacity: 1})
	for _, n := range []string{"a", "b"} {
		if err := p.Register(ref(n)); err != nil {
			t.Fatalf("register %s: %v", n, err)
		}
	}
	_, ra, _ := p.Use(ctx, "a")
	ra()
	_, rb, err := p.Use(ctx, "b") // evicts a (capacity 1)
	if err != nil {
		t.Fatalf("use b: %v", err)
	}
	rb()
	res := p.Resident()
	if len(res) != 1 || res[0] != "b" {
		t.Fatalf("resident after eviction: want [b], got %v", res)
	}
}
