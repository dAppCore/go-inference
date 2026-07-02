// SPDX-Licence-Identifier: EUPL-1.2

// Tests for the adapter serving Pool.

package lora

import (
	"context"
	"testing"
)

// TestLoRA_Use_Good covers the serving manager happy path: first Use loads the
// adapter, a second Use of the same adapter is a resident hit (no reload), the
// release closure drops the ref, and Resident reflects the working set.
func TestLoRA_Use_Good(t *testing.T) {
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

// TestLoRA_Use_Bad covers admission and eviction at capacity: filling the pool
// then using a third adapter evicts the LRU resident (and unloads it), a
// referenced adapter is spared in favour of an unreferenced one, and a pinned
// adapter is never evicted.
func TestLoRA_Use_Bad(t *testing.T) {
	fl := &fakeLoader{}
	p := NewPool(Config{Loader: fl, Policy: NewLRUEvictionPolicy(), Capacity: 2})
	for _, n := range []string{"a", "b", "c"} {
		if err := p.Register(ref(n)); err != nil {
			t.Fatalf("register %s: %v", n, err)
		}
	}
	ctx := context.Background()

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
}

// TestLoRA_Pool_Ugly covers the typed-error and boundary paths: an unknown
// adapter, a zero-capacity pool (can't fit even when empty), an admission that
// can't evict enough because every resident is referenced or pinned, a load
// failure surfacing from the Loader, and Pin/Unpin of an absent adapter.
func TestLoRA_Pool_Ugly(t *testing.T) {
	ctx := context.Background()

	// Unknown adapter → typed error, nothing loaded.
	fl := &fakeLoader{}
	p := NewPool(Config{Loader: fl, Policy: NewLRUEvictionPolicy(), Capacity: 2})
	if _, _, err := p.Use(ctx, "ghost"); err == nil {
		t.Fatalf("use unknown: want error")
	}

	// Zero capacity: an adapter can never fit even on an empty pool.
	zp := NewPool(Config{Loader: fl, Policy: NewLRUEvictionPolicy(), Capacity: 0})
	if err := zp.Register(ref("a")); err != nil {
		t.Fatalf("register a: %v", err)
	}
	_, _, err := zp.Use(ctx, "a")
	if err == nil {
		t.Fatalf("zero-capacity use: want error")
	}
	if !IsCannotFit(err) {
		t.Fatalf("zero-capacity: want CannotFit error, got %v", err)
	}

	// Admission that can't evict enough: capacity 1, hold the sole resident, then
	// demand a different adapter → nothing evictable → typed error.
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
	_, _, err = bp.Use(ctx, "b") // capacity 1, a is pinned-by-ref → can't admit
	if err == nil {
		t.Fatalf("no-evictable use: want error")
	}
	if !IsCannotAdmit(err) {
		t.Fatalf("no-evictable: want CannotAdmit error, got %v", err)
	}
	ra()
	// After release, b admits by evicting the now-free a.
	_, rb, err := bp.Use(ctx, "b")
	if err != nil {
		t.Fatalf("use b after release: %v", err)
	}
	rb()
	if bp.IsResident("a") {
		t.Fatalf("freed a should have been evicted for b")
	}

	// Pinned-only blockage: capacity 1, pin the resident, demand another.
	pp := NewPool(Config{Loader: fl, Policy: NewLRUEvictionPolicy(), Capacity: 1})
	for _, n := range []string{"a", "b"} {
		if err := pp.Register(ref(n)); err != nil {
			t.Fatalf("register %s: %v", n, err)
		}
	}
	_, rpa, _ := pp.Use(ctx, "a")
	rpa()
	pp.Pin("a")
	if _, _, err := pp.Use(ctx, "b"); !IsCannotAdmit(err) {
		t.Fatalf("pinned-only blockage: want CannotAdmit, got %v", err)
	}
	pp.Unpin("a") // now a is evictable again
	if _, rpb, err := pp.Use(ctx, "b"); err != nil {
		t.Fatalf("use b after unpin a: %v", err)
	} else {
		rpb()
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

	// Pin/Unpin of an absent adapter is a no-op (no panic, no residency).
	p.Pin("nobody")
	p.Unpin("nobody")
	if p.IsResident("nobody") {
		t.Fatalf("pinning an absent adapter must not make it resident")
	}
}

// TestLoRA_Pool_Unregister covers cross-cutting registry+pool teardown: an
// unreferenced resident adapter can be unregistered (and is unloaded + dropped
// from the working set), while an in-flight one cannot.
func TestLoRA_Pool_Unregister(t *testing.T) {
	fl := &fakeLoader{}
	p := NewPool(Config{Loader: fl, Policy: NewLRUEvictionPolicy(), Capacity: 2})
	if err := p.Register(ref("a")); err != nil {
		t.Fatalf("register a: %v", err)
	}
	ctx := context.Background()

	_, ra, err := p.Use(ctx, "a")
	if err != nil {
		t.Fatalf("use a: %v", err)
	}

	// In-flight → unregister refused.
	if err := p.Unregister("a"); err == nil {
		t.Fatalf("unregister in-use a: want error")
	}
	ra()

	// Free → unregister unloads and removes it from the resident set.
	if err := p.Unregister("a"); err != nil {
		t.Fatalf("unregister free a: %v", err)
	}
	if p.IsResident("a") {
		t.Fatalf("unregistered a must not be resident")
	}
	if fl.unloads != 1 {
		t.Fatalf("unregister should unload the resident adapter, unloads=%d", fl.unloads)
	}
	// Unregistering an unknown adapter is an error.
	if err := p.Unregister("ghost"); err == nil {
		t.Fatalf("unregister unknown: want error")
	}
}

// TestLoRA_Pool_Config covers Config edge cases and the typed-error predicates in
// isolation: a negative Capacity clamps to zero (admits nothing), and the
// CannotFit / CannotAdmit predicates report false for a nil or unrelated error.
func TestLoRA_Pool_Config(t *testing.T) {
	// Negative capacity is clamped to zero — behaves like a zero-capacity pool.
	np := NewPool(Config{Loader: &fakeLoader{}, Policy: NewLRUEvictionPolicy(), Capacity: -1})
	if err := np.Register(ref("a")); err != nil {
		t.Fatalf("register a: %v", err)
	}
	_, _, err := np.Use(context.Background(), "a")
	if !IsCannotFit(err) {
		t.Fatalf("negative capacity: want CannotFit, got %v", err)
	}

	// The predicates are total: a nil or unrelated error is neither kind.
	if IsCannotFit(nil) || IsCannotAdmit(nil) {
		t.Fatalf("nil error must not match either fit predicate")
	}
	if IsCannotFit(errBoom) || IsCannotAdmit(errBoom) {
		t.Fatalf("unrelated error must not match either fit predicate")
	}
}
