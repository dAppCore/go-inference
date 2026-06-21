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

// TestLoRA_Eviction_Good covers the LRU policy in isolation: the least-recently
// marked id is the victim, re-marking moves an id to most-recent so a different
// id becomes LRU, and removing an id drops it from tracking.
func TestLoRA_Eviction_Good(t *testing.T) {
	p := NewLRUEvictionPolicy()

	p.MarkUsed("a")
	p.MarkUsed("b")
	p.MarkUsed("c")

	// All three are candidates → a is the LRU victim.
	id, ok := p.SelectVictim([]string{"a", "b", "c"})
	if !ok || id != "a" {
		t.Fatalf("want victim a, got %q ok=%v", id, ok)
	}

	// Re-mark a → it is now most-recent, so b is the LRU.
	p.MarkUsed("a")
	id, ok = p.SelectVictim([]string{"a", "b", "c"})
	if !ok || id != "b" {
		t.Fatalf("after re-mark a, want victim b, got %q ok=%v", id, ok)
	}

	// Restrict candidates: only c and a are eligible → c is older than a now.
	id, ok = p.SelectVictim([]string{"c", "a"})
	if !ok || id != "c" {
		t.Fatalf("want victim c from {c,a}, got %q ok=%v", id, ok)
	}

	// Remove b, then the candidate set {b} has no tracked member.
	p.Remove("b")
	if _, ok := p.SelectVictim([]string{"b"}); ok {
		t.Fatalf("removed id b should not be selectable")
	}
}

// TestLoRA_Eviction_Bad covers selection when nothing matches: an empty
// candidate set and a candidate set with no tracked ids both report ok=false
// rather than inventing a victim.
func TestLoRA_Eviction_Bad(t *testing.T) {
	p := NewLRUEvictionPolicy()
	p.MarkUsed("a")

	if _, ok := p.SelectVictim(nil); ok {
		t.Fatalf("nil candidates: want ok=false")
	}
	if _, ok := p.SelectVictim([]string{}); ok {
		t.Fatalf("empty candidates: want ok=false")
	}
	if _, ok := p.SelectVictim([]string{"z"}); ok {
		t.Fatalf("untracked candidate: want ok=false")
	}
}

// TestLoRA_Eviction_Ugly covers degenerate calls: marking/removing the empty id
// is a harmless no-op, and a candidate that was never marked but appears as the
// only option is still not a tracked victim.
func TestLoRA_Eviction_Ugly(t *testing.T) {
	p := NewLRUEvictionPolicy()

	// Empty id is ignored (mirrors SGLang's None handling) — no panic.
	p.MarkUsed("")
	p.Remove("")
	if _, ok := p.SelectVictim([]string{""}); ok {
		t.Fatalf("empty-id candidate must not be a victim")
	}

	// Removing an unknown id is a no-op.
	p.Remove("never-seen")

	// A single tracked id is trivially its own victim.
	p.MarkUsed("solo")
	id, ok := p.SelectVictim([]string{"solo"})
	if !ok || id != "solo" {
		t.Fatalf("want solo victim, got %q ok=%v", id, ok)
	}
}

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

// TestLoRA_Pool_Good covers the serving manager happy path: first Use loads the
// adapter, a second Use of the same adapter is a resident hit (no reload), the
// release closure drops the ref, and Resident reflects the working set.
func TestLoRA_Pool_Good(t *testing.T) {
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

// TestLoRA_Pool_Bad covers admission and eviction at capacity: filling the pool
// then using a third adapter evicts the LRU resident (and unloads it), a
// referenced adapter is spared in favour of an unreferenced one, and a pinned
// adapter is never evicted.
func TestLoRA_Pool_Bad(t *testing.T) {
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

// errBoom is a sentinel Loader failure for the load-error path.
var errBoom = context.DeadlineExceeded
