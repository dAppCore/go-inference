// SPDX-Licence-Identifier: EUPL-1.2

package registry

import (
	"testing"

	core "dappco.re/go"
)

// failStore is a Store that can be told to fail its Put or Delete after a given
// number of successful calls — used to drive the Registry's store-rejection
// branches (re-index on failed Put, early return on failed Delete) that the
// in-memory store never exercises on its own.
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

func TestRegistry_Put_Ugly(t *testing.T) {
	// An alias that normalises to empty (pure whitespace) is skipped, not
	// indexed — the entry still goes in and resolves by id.
	r := New()
	e := sampleEntry("blank-alias", 1_000_000_000, "   ", "real")
	if pr := r.Put(e); !pr.OK {
		t.Fatalf("put with a blank alias should succeed: %v", pr.Error())
	}
	if res := r.Resolve("blank-alias"); !res.OK {
		t.Fatalf("entry with blank alias should resolve by id: %v", res.Error())
	}
	if res := r.Resolve("real"); !res.OK {
		t.Fatalf("non-blank alias should still resolve: %v", res.Error())
	}
	// The blank alias resolves to nothing (it was never indexed).
	if res := r.Resolve("   "); res.OK {
		t.Fatalf("blank alias should not resolve, got %+v", res.Value)
	}
}

func TestRegistry_Put_Bad(t *testing.T) {
	// When the underlying store rejects a Put, Put surfaces the failure and
	// leaves the alias index unchanged (no half-applied entry).
	s := &failStore{MemStore: NewMemStore(), failPut: true}
	r := NewWithStore(s)

	pr := r.Put(sampleEntry("rejected", 1_000_000_000, "rej"))
	if pr.OK {
		t.Fatalf("put should fail when the store rejects it, got %+v", pr.Value)
	}
	// The rejected entry's alias was never committed.
	if res := r.Resolve("rej"); res.OK {
		t.Fatalf("alias of a rejected entry should not resolve, got %+v", res.Value)
	}
	if res := r.Resolve("rejected"); res.OK {
		t.Fatalf("id of a rejected entry should not resolve, got %+v", res.Value)
	}
}

func TestRegistry_Put_StoreFailRestoresIndex(t *testing.T) {
	// An update-in-place that the store rejects must re-index the previous entry,
	// so the original aliases keep resolving (the rollback branch in Put).
	s := &failStore{MemStore: NewMemStore()}
	r := NewWithStore(s)

	// First Put succeeds and indexes the original aliases.
	if pr := r.Put(sampleEntry("model", 4_000_000_000, "orig")); !pr.OK {
		t.Fatalf("initial put: %v", pr.Error())
	}

	// Now force the store to reject the update.
	s.failPut = true
	if pr := r.Put(sampleEntry("model", 9_000_000_000, "orig", "added")); pr.OK {
		t.Fatalf("update should fail when the store rejects it, got %+v", pr.Value)
	}

	// The original entry still resolves by its original alias — the index was
	// restored after the rejected update.
	res := r.Resolve("orig")
	if !res.OK {
		t.Fatalf("original alias should still resolve after a rejected update: %v", res.Error())
	}
	if got := res.Value.(Entry).MemoryBytes; got != 4_000_000_000 {
		t.Errorf("footprint after rejected update: got %d, want 4000000000", got)
	}
	// The would-be new alias never took hold.
	if res := r.Resolve("added"); res.OK {
		t.Fatalf("new alias of a rejected update should not resolve, got %+v", res.Value)
	}
}

func TestRegistry_Delete_Bad(t *testing.T) {
	// When the store rejects a Delete after the entry was found, Delete returns
	// the store's failure and leaves the alias index intact.
	s := &failStore{MemStore: NewMemStore(), failDelete: true}
	r := NewWithStore(s)
	if pr := r.Put(sampleEntry("keep", 1_000_000_000, "k")); !pr.OK {
		t.Fatalf("put: %v", pr.Error())
	}

	dr := r.Delete("keep")
	if dr.OK {
		t.Fatalf("delete should fail when the store rejects it, got %+v", dr.Value)
	}
	// The entry and its alias survive a rejected delete.
	if res := r.Resolve("k"); !res.OK {
		t.Fatalf("alias should survive a rejected delete: %v", res.Error())
	}
}

func TestRegistry_Filter_Capabilities(t *testing.T) {
	// Each capability filter must independently reject an entry lacking that one
	// capability — exercising the Grammar and Streaming guards in matches.
	r := New()

	noGrammar := sampleEntry("no-grammar", 1_000_000_000, "ng")
	noGrammar.Capabilities.Grammar = false
	noStreaming := sampleEntry("no-streaming", 1_000_000_000, "ns")
	noStreaming.Capabilities.Streaming = false
	full := sampleEntry("full", 1_000_000_000, "f")

	for _, e := range []Entry{noGrammar, noStreaming, full} {
		if pr := r.Put(e); !pr.OK {
			t.Fatalf("put %s: %v", e.ID, pr.Error())
		}
	}

	// Grammar filter excludes the entry without grammar.
	g := r.Filter(Filter{Grammar: true})
	if len(g) != 2 {
		t.Fatalf("grammar filter: got %d, want 2", len(g))
	}
	for _, e := range g {
		if e.ID == "no-grammar" {
			t.Errorf("grammar filter let through a non-grammar entry")
		}
	}

	// Streaming filter excludes the entry without streaming.
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

func TestRegistry_FitsDeviceWith_TieBreak(t *testing.T) {
	// Two entries with identical footprints must be ordered by id (the comparator
	// tie-break branch in FitsDeviceWith), not left in map order.
	r := New()
	if pr := r.Put(sampleEntry("zeta", 4_000_000_000, "z")); !pr.OK {
		t.Fatalf("put zeta: %v", pr.Error())
	}
	if pr := r.Put(sampleEntry("alpha", 4_000_000_000, "a")); !pr.OK {
		t.Fatalf("put alpha: %v", pr.Error())
	}

	fits := r.FitsDevice(96 << 30)
	if len(fits) != 2 {
		t.Fatalf("both should fit: got %d, want 2", len(fits))
	}
	// Equal footprints → ascending id order ("alpha" before "zeta").
	if fits[0].ID != "alpha" || fits[1].ID != "zeta" {
		t.Errorf("tie-break order: got %v, want [alpha zeta]", ids(fits))
	}
}

func TestMemStore_Put_Bad(t *testing.T) {
	// The store's own empty-id guard (the Registry catches empty ids before the
	// store, so this is only reachable by calling the store directly).
	s := NewMemStore()
	if r := s.Put(Entry{ID: ""}); r.OK {
		t.Fatalf("MemStore.Put with empty id should fail, got %+v", r.Value)
	}
}

func TestMemStore_Delete_Bad(t *testing.T) {
	// The store's own missing-id guard, reachable only by a direct store call.
	s := NewMemStore()
	if r := s.Delete("ghost"); r.OK {
		t.Fatalf("MemStore.Delete of a missing id should fail, got %+v", r.Value)
	}
}
