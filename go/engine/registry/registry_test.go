// SPDX-Licence-Identifier: EUPL-1.2

package registry

import (
	"testing"

	core "dappco.re/go"
)

// sampleEntry builds a populated catalogue entry for tests.
//
//	e := sampleEntry("lemma", 4_500_000_000)
func sampleEntry(id string, footprint uint64, aliases ...string) Entry {
	return Entry{
		ID:            id,
		Aliases:       aliases,
		Architecture:  "gemma4",
		Params:        4_500_000_000,
		ContextLength: 131072,
		Quantisation:  "Q4_K_M",
		Format:        FormatGGUF,
		MemoryBytes:   footprint,
		DeviceFit:     []string{"metal"},
		Capabilities: Capabilities{
			Tools:     true,
			Vision:    true,
			Grammar:   true,
			Streaming: true,
		},
		Source: Source{LocalPath: "/models/" + id},
		Status: StatusReady,
	}
}

func newSeededRegistry(t *testing.T) *Registry {
	t.Helper()
	r := New()
	if pr := r.Put(sampleEntry("gemma-4-31b-it", 24_000_000_000, "lemrd", "gemma4-31b")); !pr.OK {
		t.Fatalf("seed put lemrd: %v", pr.Error())
	}
	if pr := r.Put(sampleEntry("gemma-4-4b-it", 4_500_000_000, "lemma")); !pr.OK {
		t.Fatalf("seed put lemma: %v", pr.Error())
	}
	return r
}

// ids extracts the ids of a slice of entries for test diagnostics.
func ids(es []Entry) []string {
	out := make([]string, len(es))
	for i, e := range es {
		out[i] = e.ID
	}
	return out
}

// failStore is a Store that can be told to fail its Put or Delete — used to
// drive the Registry's store-rejection branches (re-index on failed Put,
// early return on failed Delete) that the in-memory store never exercises on
// its own.
//
//	s := &failStore{MemStore: NewMemStore(), failPut: true}
type failStore struct {
	*MemStore
	failPut    bool // Put returns a failed Result instead of storing
	failDelete bool // Delete returns a failed Result instead of removing
}

// Put fails when failPut is set, otherwise delegates to the embedded MemStore.
//
//	s.Put(registry.Entry{ID: "x"})
func (s *failStore) Put(e Entry) core.Result {
	if s.failPut {
		return core.Fail(core.E("failStore.Put", "forced failure", nil))
	}
	return s.MemStore.Put(e)
}

// Delete fails when failDelete is set, otherwise delegates to the embedded
// MemStore.
//
//	s.Delete("x")
func (s *failStore) Delete(id string) core.Result {
	if s.failDelete {
		return core.Fail(core.E("failStore.Delete", "forced failure", nil))
	}
	return s.MemStore.Delete(id)
}

func TestRegistry_New_Good(t *testing.T) {
	r := New()
	if all := r.List(); len(all) != 0 {
		t.Fatalf("fresh registry: got %d entries, want 0", len(all))
	}
	if pr := r.Put(sampleEntry("fresh", 1, "f")); !pr.OK {
		t.Fatalf("put on fresh registry: %v", pr.Error())
	}
	if res := r.Resolve("f"); !res.OK {
		t.Fatalf("resolve on fresh registry: %v", res.Error())
	}
}

func TestRegistry_New_Bad(t *testing.T) {
	// A fresh registry's internal maps are initialised, not nil — an unknown
	// lookup fails cleanly rather than panicking on a nil alias map.
	r := New()
	if res := r.Resolve("anything"); res.OK {
		t.Fatalf("resolve on empty registry should fail, got %+v", res.Value)
	}
	if res := r.Get("anything"); res.OK {
		t.Fatalf("get on empty registry should fail, got %+v", res.Value)
	}
}

func TestRegistry_New_Ugly(t *testing.T) {
	// Two independently constructed registries never share state.
	a := New()
	b := New()
	if pr := a.Put(sampleEntry("only-in-a", 1, "oia")); !pr.OK {
		t.Fatalf("put into a: %v", pr.Error())
	}
	if res := b.Resolve("oia"); res.OK {
		t.Fatalf("b should not see entries put into a, got %+v", res.Value)
	}
}

func TestRegistry_NewWithStore_Good(t *testing.T) {
	// A store that already holds entries must have its alias index rebuilt by
	// NewWithStore so resolution works without any Put on the new Registry.
	s := NewMemStore()
	if pr := s.Put(sampleEntry("gemma-4-4b-it", 4_500_000_000, "lemma", "lemma-e4b")); !pr.OK {
		t.Fatalf("seed store: %v", pr.Error())
	}
	if pr := s.Put(sampleEntry("gemma-4-31b-it", 24_000_000_000, "lemrd")); !pr.OK {
		t.Fatalf("seed store: %v", pr.Error())
	}

	r := NewWithStore(s)

	// Canonical id and every alias resolve straight away.
	for _, name := range []string{"gemma-4-4b-it", "lemma", "lemma-e4b", "lemrd"} {
		res := r.Resolve(name)
		if !res.OK {
			t.Fatalf("resolve %q after NewWithStore: %v", name, res.Error())
		}
	}
	if id := r.Resolve("lemma-e4b").Value.(Entry).ID; id != "gemma-4-4b-it" {
		t.Errorf("rebuilt alias points wrong: got %q, want gemma-4-4b-it", id)
	}
}

func TestRegistry_NewWithStore_Bad(t *testing.T) {
	// An empty store yields a registry that resolves nothing — NewWithStore
	// does not fabricate entries.
	r := NewWithStore(NewMemStore())
	if all := r.List(); len(all) != 0 {
		t.Fatalf("empty store: got %d entries, want 0", len(all))
	}
	if res := r.Resolve("anything"); res.OK {
		t.Fatalf("resolve against empty store should fail, got %+v", res.Value)
	}
}

func TestRegistry_NewWithStore_Ugly(t *testing.T) {
	// NewWithStore trusts the store's existing content: two entries whose
	// aliases already collide (a state Put itself would reject) are indexed
	// in store order (sorted by id), the later entry silently winning the
	// shared name.
	s := NewMemStore()
	if pr := s.Put(sampleEntry("first", 1, "shared")); !pr.OK {
		t.Fatalf("seed first: %v", pr.Error())
	}
	if pr := s.Put(sampleEntry("second", 1, "shared")); !pr.OK {
		t.Fatalf("seed second: %v", pr.Error())
	}

	r := NewWithStore(s)
	res := r.Resolve("shared")
	if !res.OK {
		t.Fatalf("resolve shared alias: %v", res.Error())
	}
	if id := res.Value.(Entry).ID; id != "second" {
		t.Errorf("last-indexed entry should own the shared alias: got %q, want second", id)
	}
	if len(r.List()) != 2 {
		t.Fatalf("both entries should still be listed: got %d", len(r.List()))
	}
}

func TestRegistry_Put_Good(t *testing.T) {
	r := New()
	e := sampleEntry("gemma-4-4b-it", 4_500_000_000, "lemma")
	if pr := r.Put(e); !pr.OK {
		t.Fatalf("put: %v", pr.Error())
	}
	got := r.Get("gemma-4-4b-it")
	if !got.OK {
		t.Fatalf("get after put: %v", got.Error())
	}
	if got.Value.(Entry).MemoryBytes != e.MemoryBytes {
		t.Errorf("stored entry mismatch: got %+v", got.Value)
	}

	// Re-putting the same id updates in place rather than erroring.
	e.MemoryBytes = 9_000_000_000
	if pr := r.Put(e); !pr.OK {
		t.Fatalf("update in place: %v", pr.Error())
	}
	if got := r.Get("gemma-4-4b-it").Value.(Entry).MemoryBytes; got != 9_000_000_000 {
		t.Errorf("updated footprint: got %d, want 9000000000", got)
	}
}

func TestRegistry_Put_Bad(t *testing.T) {
	// An empty id is rejected outright.
	if pr := (New()).Put(Entry{ID: ""}); pr.OK {
		t.Fatalf("put empty id should fail, got %+v", pr.Value)
	}

	// A name already owned by a different entry is rejected.
	r := New()
	if pr := r.Put(sampleEntry("gemma-4-4b-it", 1, "lemma")); !pr.OK {
		t.Fatalf("seed put: %v", pr.Error())
	}
	if pr := r.Put(sampleEntry("other-model", 1, "lemma")); pr.OK {
		t.Fatalf("duplicate alias should fail, got %+v", pr.Value)
	}
	if pr := r.Put(sampleEntry("brand-new", 1, "gemma-4-4b-it")); pr.OK {
		t.Fatalf("alias colliding with existing id should fail, got %+v", pr.Value)
	}

	// When the underlying store rejects a Put, Put surfaces the failure and
	// leaves the alias index unchanged (no half-applied entry).
	s := &failStore{MemStore: NewMemStore(), failPut: true}
	r2 := NewWithStore(s)
	if pr := r2.Put(sampleEntry("rejected", 1, "rej")); pr.OK {
		t.Fatalf("put should fail when the store rejects it, got %+v", pr.Value)
	}
	if res := r2.Resolve("rej"); res.OK {
		t.Fatalf("alias of a rejected entry should not resolve, got %+v", res.Value)
	}
}

func TestRegistry_Put_Ugly(t *testing.T) {
	// An alias that normalises to empty (pure whitespace) is skipped, not
	// indexed — the entry still goes in and resolves by id.
	r := New()
	if pr := r.Put(sampleEntry("blank-alias", 1, "   ", "real")); !pr.OK {
		t.Fatalf("put with a blank alias should succeed: %v", pr.Error())
	}
	if res := r.Resolve("blank-alias"); !res.OK {
		t.Fatalf("entry with blank alias should resolve by id: %v", res.Error())
	}
	if res := r.Resolve("real"); !res.OK {
		t.Fatalf("non-blank alias should still resolve: %v", res.Error())
	}
	if res := r.Resolve("   "); res.OK {
		t.Fatalf("blank alias should not resolve, got %+v", res.Value)
	}

	// An update-in-place that the store rejects must re-index the previous
	// entry, so the original aliases keep resolving (the rollback branch).
	s := &failStore{MemStore: NewMemStore()}
	r2 := NewWithStore(s)
	if pr := r2.Put(sampleEntry("model", 4_000_000_000, "orig")); !pr.OK {
		t.Fatalf("initial put: %v", pr.Error())
	}
	s.failPut = true
	if pr := r2.Put(sampleEntry("model", 9_000_000_000, "orig", "added")); pr.OK {
		t.Fatalf("update should fail when the store rejects it, got %+v", pr.Value)
	}
	res := r2.Resolve("orig")
	if !res.OK {
		t.Fatalf("original alias should still resolve after a rejected update: %v", res.Error())
	}
	if got := res.Value.(Entry).MemoryBytes; got != 4_000_000_000 {
		t.Errorf("footprint after rejected update: got %d, want 4000000000", got)
	}
	if res := r2.Resolve("added"); res.OK {
		t.Fatalf("new alias of a rejected update should not resolve, got %+v", res.Value)
	}
}

func TestRegistry_Get_Good(t *testing.T) {
	r := newSeededRegistry(t)

	// Get returns the entry stored under the exact canonical id.
	got := r.Get("gemma-4-4b-it")
	if !got.OK {
		t.Fatalf("get by id: %v", got.Error())
	}
	if id := got.Value.(Entry).ID; id != "gemma-4-4b-it" {
		t.Errorf("id: got %q, want gemma-4-4b-it", id)
	}
}

func TestRegistry_Get_Bad(t *testing.T) {
	r := newSeededRegistry(t)

	// Get does NOT resolve aliases — only the canonical id hits.
	if res := r.Get("lemma"); res.OK {
		t.Fatalf("get by alias should miss (Get is id-only), got %+v", res.Value)
	}

	// Get of an unknown id fails cleanly.
	if res := r.Get("does-not-exist"); res.OK {
		t.Fatalf("get of unknown id should fail, got %+v", res.Value)
	}
}

func TestRegistry_Get_Ugly(t *testing.T) {
	// Unlike Resolve, Get does not normalise case or whitespace — it is an
	// exact canonical-id lookup.
	r := New()
	if pr := r.Put(sampleEntry("Gemma-4-4B-IT", 1, "lemma")); !pr.OK {
		t.Fatalf("put: %v", pr.Error())
	}
	if res := r.Get("gemma-4-4b-it"); res.OK {
		t.Fatalf("Get should be case-sensitive, unexpectedly matched: %+v", res.Value)
	}
	if res := r.Get("Gemma-4-4B-IT"); !res.OK {
		t.Fatalf("exact-case get should hit: %v", res.Error())
	}
}

func TestRegistry_Resolve_Good(t *testing.T) {
	r := newSeededRegistry(t)

	// Resolve by canonical id.
	res := r.Resolve("gemma-4-4b-it")
	if !res.OK {
		t.Fatalf("resolve by id: %v", res.Error())
	}
	if got := res.Value.(Entry).ID; got != "gemma-4-4b-it" {
		t.Errorf("id: got %q, want gemma-4-4b-it", got)
	}

	// Resolve by alias.
	res = r.Resolve("lemma")
	if !res.OK {
		t.Fatalf("resolve by alias: %v", res.Error())
	}
	if got := res.Value.(Entry).ID; got != "gemma-4-4b-it" {
		t.Errorf("alias id: got %q, want gemma-4-4b-it", got)
	}

	// Resolve by a second alias on the other entry.
	res = r.Resolve("lemrd")
	if !res.OK {
		t.Fatalf("resolve second alias: %v", res.Error())
	}
	if got := res.Value.(Entry).ID; got != "gemma-4-31b-it" {
		t.Errorf("alias id: got %q, want gemma-4-31b-it", got)
	}
}

func TestRegistry_Resolve_Bad(t *testing.T) {
	r := newSeededRegistry(t)

	// Unknown id/alias fails.
	if res := r.Resolve("does-not-exist"); res.OK {
		t.Fatalf("unknown id should fail, got %+v", res.Value)
	}

	// Empty query fails.
	if res := r.Resolve(""); res.OK {
		t.Fatalf("empty query should fail, got %+v", res.Value)
	}
}

func TestRegistry_Resolve_Ugly(t *testing.T) {
	r := New()
	// Mixed-case and surrounding whitespace still resolve.
	if pr := r.Put(sampleEntry("Gemma-4-4B-IT", 4_500_000_000, "Lemma")); !pr.OK {
		t.Fatalf("put: %v", pr.Error())
	}
	if res := r.Resolve("  gEmMa-4-4b-it "); !res.OK {
		t.Fatalf("case/space-insensitive id resolve failed: %v", res.Error())
	}
	if res := r.Resolve("LEMMA"); !res.OK {
		t.Fatalf("case-insensitive alias resolve failed: %v", res.Error())
	}

	// Duplicate alias across two entries is rejected at Put time.
	if pr := r.Put(sampleEntry("other-model", 1, "lemma")); pr.OK {
		t.Fatalf("duplicate alias should fail, got %+v", pr.Value)
	}

	// An alias that collides with an existing id is rejected.
	if pr := r.Put(sampleEntry("brand-new", 1, "gemma-4-4b-it")); pr.OK {
		t.Fatalf("alias colliding with existing id should fail, got %+v", pr.Value)
	}

	// Re-Put of the same id (update in place) keeps resolution working and
	// does not trip the self-alias guard.
	if pr := r.Put(sampleEntry("Gemma-4-4B-IT", 9_000_000_000, "Lemma", "lemma-e4b")); !pr.OK {
		t.Fatalf("update in place should succeed: %v", pr.Error())
	}
	res := r.Resolve("lemma-e4b")
	if !res.OK {
		t.Fatalf("new alias after update should resolve: %v", res.Error())
	}
	if got := res.Value.(Entry).MemoryBytes; got != 9_000_000_000 {
		t.Errorf("updated footprint: got %d, want 9000000000", got)
	}
}

func TestRegistry_List_Good(t *testing.T) {
	r := newSeededRegistry(t)

	// List returns every entry, sorted by id for determinism.
	all := r.List()
	if len(all) != 2 {
		t.Fatalf("list: got %d, want 2", len(all))
	}
	if all[0].ID != "gemma-4-31b-it" || all[1].ID != "gemma-4-4b-it" {
		t.Errorf("list order: got %v", ids(all))
	}
}

func TestRegistry_List_Bad(t *testing.T) {
	// An empty registry lists nothing.
	r := New()
	if all := r.List(); len(all) != 0 {
		t.Fatalf("empty list: got %d, want 0", len(all))
	}
}

func TestRegistry_List_Ugly(t *testing.T) {
	// Sort order survives entries added out of order and one subsequently
	// deleted — List always reflects current store content, re-sorted.
	r := New()
	for _, id := range []string{"zeta", "alpha", "mu"} {
		if pr := r.Put(sampleEntry(id, 1)); !pr.OK {
			t.Fatalf("put %s: %v", id, pr.Error())
		}
	}
	if dr := r.Delete("mu"); !dr.OK {
		t.Fatalf("delete mu: %v", dr.Error())
	}
	all := r.List()
	if got := ids(all); len(got) != 2 || got[0] != "alpha" || got[1] != "zeta" {
		t.Errorf("list after delete: got %v, want [alpha zeta]", got)
	}
}

func TestRegistry_Filter_Good(t *testing.T) {
	r := newSeededRegistry(t)
	if pr := r.Put(func() Entry {
		e := sampleEntry("no-tools", 1_000_000_000, "nt")
		e.Capabilities.Tools = false
		return e
	}()); !pr.OK {
		t.Fatalf("put no-tools: %v", pr.Error())
	}
	tools := r.Filter(Filter{Tools: true})
	if len(tools) != 2 {
		t.Fatalf("tools filter: got %d, want 2", len(tools))
	}
	for _, e := range tools {
		if e.ID == "no-tools" {
			t.Errorf("tools filter let through a non-tools entry")
		}
	}
}

func TestRegistry_Filter_Bad(t *testing.T) {
	// Requiring a capability nothing in the catalogue has returns empty, not
	// an error and not nil-panicking.
	r := New()
	plain := sampleEntry("plain", 1, "p")
	plain.Capabilities = Capabilities{}
	if pr := r.Put(plain); !pr.OK {
		t.Fatalf("put: %v", pr.Error())
	}
	if got := r.Filter(Filter{Tools: true}); len(got) != 0 {
		t.Fatalf("filter with no matching entries: got %d, want 0", len(got))
	}
}

func TestRegistry_Filter_Ugly(t *testing.T) {
	// Each capability field must independently reject an entry lacking that
	// one capability — exercising the Grammar and Streaming guards in matches
	// (Tools is covered in Filter_Good, Vision in FitsDeviceWith_Good).
	r := New()
	noGrammar := sampleEntry("no-grammar", 1, "ng")
	noGrammar.Capabilities.Grammar = false
	noStreaming := sampleEntry("no-streaming", 1, "ns")
	noStreaming.Capabilities.Streaming = false
	full := sampleEntry("full", 1, "f")
	for _, e := range []Entry{noGrammar, noStreaming, full} {
		if pr := r.Put(e); !pr.OK {
			t.Fatalf("put %s: %v", e.ID, pr.Error())
		}
	}

	g := r.Filter(Filter{Grammar: true})
	if len(g) != 2 {
		t.Fatalf("grammar filter: got %d, want 2", len(g))
	}
	for _, e := range g {
		if e.ID == "no-grammar" {
			t.Errorf("grammar filter let through a non-grammar entry")
		}
	}

	s := r.Filter(Filter{Streaming: true})
	if len(s) != 2 {
		t.Fatalf("streaming filter: got %d, want 2", len(s))
	}
	for _, e := range s {
		if e.ID == "no-streaming" {
			t.Errorf("streaming filter let through a non-streaming entry")
		}
	}
}

func TestRegistry_Delete_Good(t *testing.T) {
	r := newSeededRegistry(t)
	if dr := r.Delete("gemma-4-4b-it"); !dr.OK {
		t.Fatalf("delete: %v", dr.Error())
	}
	// Gone by id and by its alias.
	if res := r.Resolve("gemma-4-4b-it"); res.OK {
		t.Fatalf("deleted id still resolves")
	}
	if res := r.Resolve("lemma"); res.OK {
		t.Fatalf("alias of deleted entry still resolves")
	}
	// The alias is now free for reuse on a new entry.
	if pr := r.Put(sampleEntry("reused", 1, "lemma")); !pr.OK {
		t.Fatalf("reusing freed alias should succeed: %v", pr.Error())
	}
}

func TestRegistry_Delete_Bad(t *testing.T) {
	// Deleting an id that was never there fails cleanly.
	r := New()
	if dr := r.Delete("ghost"); dr.OK {
		t.Fatalf("delete missing id should fail")
	}

	// When the store rejects a Delete after the entry was found, Delete
	// returns the store's failure and leaves the alias index intact.
	s := &failStore{MemStore: NewMemStore(), failDelete: true}
	r2 := NewWithStore(s)
	if pr := r2.Put(sampleEntry("keep", 1, "k")); !pr.OK {
		t.Fatalf("put: %v", pr.Error())
	}
	if dr := r2.Delete("keep"); dr.OK {
		t.Fatalf("delete should fail when the store rejects it, got %+v", dr.Value)
	}
	if res := r2.Resolve("k"); !res.OK {
		t.Fatalf("alias should survive a rejected delete: %v", res.Error())
	}
}

func TestRegistry_Delete_Ugly(t *testing.T) {
	// Deleting the same id twice: the second call finds nothing and fails —
	// Delete is not idempotent-success, it reports the miss honestly.
	r := New()
	if pr := r.Put(sampleEntry("once", 1, "o")); !pr.OK {
		t.Fatalf("put: %v", pr.Error())
	}
	if dr := r.Delete("once"); !dr.OK {
		t.Fatalf("first delete: %v", dr.Error())
	}
	if dr := r.Delete("once"); dr.OK {
		t.Fatalf("second delete of the same id should fail, got %+v", dr.Value)
	}
}

func TestRegistry_FitsDevice_Good(t *testing.T) {
	r := newSeededRegistry(t)

	// A 96 GB budget fits both models.
	fits := r.FitsDevice(96 << 30)
	if len(fits) != 2 {
		t.Fatalf("96GB budget: got %d entries, want 2", len(fits))
	}

	// A budget that fits only the 4B model.
	fits = r.FitsDevice(8 << 30)
	if len(fits) != 1 {
		t.Fatalf("8GB budget: got %d entries, want 1", len(fits))
	}
	if fits[0].ID != "gemma-4-4b-it" {
		t.Errorf("8GB budget entry: got %q, want gemma-4-4b-it", fits[0].ID)
	}

	// Results are ordered largest-footprint-first (best fit for a budget).
	fits = r.FitsDevice(96 << 30)
	if fits[0].MemoryBytes < fits[1].MemoryBytes {
		t.Errorf("expected descending footprint order, got %d then %d",
			fits[0].MemoryBytes, fits[1].MemoryBytes)
	}
}

func TestRegistry_FitsDevice_Bad(t *testing.T) {
	r := newSeededRegistry(t)

	// A budget smaller than every model yields nothing.
	if fits := r.FitsDevice(1 << 20); len(fits) != 0 {
		t.Fatalf("tiny budget: got %d entries, want 0", len(fits))
	}

	// Zero budget yields nothing (not "everything").
	if fits := r.FitsDevice(0); len(fits) != 0 {
		t.Fatalf("zero budget: got %d entries, want 0", len(fits))
	}

	// An entry with an unknown (zero) footprint never fits — it cannot be
	// placed without a known memory cost.
	r2 := New()
	unknown := sampleEntry("mystery", 0, "myst")
	if pr := r2.Put(unknown); !pr.OK {
		t.Fatalf("put unknown footprint: %v", pr.Error())
	}
	if fits := r2.FitsDevice(96 << 30); len(fits) != 0 {
		t.Fatalf("zero-footprint entry should not fit: got %d", len(fits))
	}
}

func TestRegistry_FitsDevice_Ugly(t *testing.T) {
	r := New()
	// Footprint exactly equal to the budget fits (inclusive bound).
	if pr := r.Put(sampleEntry("exact", 4_000_000_000, "ex")); !pr.OK {
		t.Fatalf("put: %v", pr.Error())
	}
	if fits := r.FitsDevice(4_000_000_000); len(fits) != 1 {
		t.Fatalf("exact-fit boundary: got %d, want 1", len(fits))
	}
	// One byte over the budget does not fit.
	if fits := r.FitsDevice(3_999_999_999); len(fits) != 0 {
		t.Fatalf("one byte over: got %d, want 0", len(fits))
	}

	// Combining a capability filter with device-fit: only ready + vision
	// entries that fit the budget come back.
	r3 := New()
	big := sampleEntry("big-vision", 30_000_000_000, "bigv")
	small := sampleEntry("small-text", 2_000_000_000, "smt")
	small.Capabilities.Vision = false
	drafting := sampleEntry("drafting", 1_000_000_000, "draft")
	drafting.Status = StatusDraft
	for _, e := range []Entry{big, small, drafting} {
		if pr := r3.Put(e); !pr.OK {
			t.Fatalf("put %s: %v", e.ID, pr.Error())
		}
	}
	fits := r3.FitsDeviceWith(8<<30, Filter{Vision: true, ReadyOnly: true})
	if len(fits) != 0 {
		t.Fatalf("big vision model exceeds 8GB, none should fit: got %d", len(fits))
	}
	fits = r3.FitsDeviceWith(96<<30, Filter{Vision: true, ReadyOnly: true})
	if len(fits) != 1 || fits[0].ID != "big-vision" {
		t.Fatalf("only big-vision is ready+vision+fits: got %v", ids(fits))
	}
}

func TestRegistry_FitsDeviceWith_Good(t *testing.T) {
	r := New()
	big := sampleEntry("big-vision", 30_000_000_000, "bigv")
	small := sampleEntry("small-text", 2_000_000_000, "smt")
	small.Capabilities.Vision = false
	for _, e := range []Entry{big, small} {
		if pr := r.Put(e); !pr.OK {
			t.Fatalf("put %s: %v", e.ID, pr.Error())
		}
	}
	fits := r.FitsDeviceWith(96<<30, Filter{Vision: true})
	if len(fits) != 1 || fits[0].ID != "big-vision" {
		t.Fatalf("vision filter + budget: got %v, want [big-vision]", ids(fits))
	}
}

func TestRegistry_FitsDeviceWith_Bad(t *testing.T) {
	// The filter matches, but the budget excludes it — the combined query
	// still returns nothing.
	r := New()
	if pr := r.Put(sampleEntry("big-vision", 30_000_000_000, "bigv")); !pr.OK {
		t.Fatalf("put: %v", pr.Error())
	}
	fits := r.FitsDeviceWith(8<<30, Filter{Vision: true})
	if len(fits) != 0 {
		t.Fatalf("budget too small for the only match: got %d, want 0", len(fits))
	}
}

func TestRegistry_FitsDeviceWith_Ugly(t *testing.T) {
	// Two entries with identical footprints must be ordered by id (the
	// comparator tie-break branch in FitsDeviceWith), not left in map order.
	r := New()
	if pr := r.Put(sampleEntry("zeta", 4_000_000_000, "z")); !pr.OK {
		t.Fatalf("put zeta: %v", pr.Error())
	}
	if pr := r.Put(sampleEntry("alpha", 4_000_000_000, "a")); !pr.OK {
		t.Fatalf("put alpha: %v", pr.Error())
	}

	fits := r.FitsDeviceWith(96<<30, Filter{ReadyOnly: true})
	if len(fits) != 2 {
		t.Fatalf("both should fit: got %d, want 2", len(fits))
	}
	// Equal footprints → ascending id order ("alpha" before "zeta").
	if fits[0].ID != "alpha" || fits[1].ID != "zeta" {
		t.Errorf("tie-break order: got %v, want [alpha zeta]", ids(fits))
	}
}
