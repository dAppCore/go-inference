// SPDX-Licence-Identifier: EUPL-1.2

package registry

import "testing"

func TestStore_NewMemStore_Good(t *testing.T) {
	s := NewMemStore()
	if all := s.List(); len(all) != 0 {
		t.Fatalf("fresh store: got %d entries, want 0", len(all))
	}
	if pr := s.Put(sampleEntry("fresh", 1)); !pr.OK {
		t.Fatalf("put on fresh store: %v", pr.Error())
	}
}

func TestStore_NewMemStore_Bad(t *testing.T) {
	// A fresh store's internal map is initialised, not nil — a lookup on an
	// empty store fails cleanly rather than panicking.
	s := NewMemStore()
	if r := s.Get("anything"); r.OK {
		t.Fatalf("get on empty store should fail, got %+v", r.Value)
	}
}

func TestStore_NewMemStore_Ugly(t *testing.T) {
	// Two independently constructed stores never share state.
	a := NewMemStore()
	b := NewMemStore()
	if pr := a.Put(sampleEntry("only-in-a", 1)); !pr.OK {
		t.Fatalf("put into a: %v", pr.Error())
	}
	if r := b.Get("only-in-a"); r.OK {
		t.Fatalf("b should not see entries put into a, got %+v", r.Value)
	}
}

func TestStore_MemStore_Put_Good(t *testing.T) {
	s := NewMemStore()
	e := sampleEntry("gemma-4-4b-it", 4_500_000_000)
	if pr := s.Put(e); !pr.OK {
		t.Fatalf("put: %v", pr.Error())
	}
	got := s.Get("gemma-4-4b-it")
	if !got.OK || got.Value.(Entry).MemoryBytes != e.MemoryBytes {
		t.Fatalf("stored entry mismatch: %+v", got)
	}
}

func TestStore_MemStore_Put_Bad(t *testing.T) {
	s := NewMemStore()
	if pr := s.Put(Entry{ID: ""}); pr.OK {
		t.Fatalf("put with empty id should fail, got %+v", pr.Value)
	}
}

func TestStore_MemStore_Put_Ugly(t *testing.T) {
	// The store has no concept of aliases — it is purely id-keyed, so two
	// entries sharing an alias both succeed (alias uniqueness is the
	// Registry's job, not the Store's).
	s := NewMemStore()
	if pr := s.Put(sampleEntry("a", 1, "shared")); !pr.OK {
		t.Fatalf("put a: %v", pr.Error())
	}
	if pr := s.Put(sampleEntry("b", 1, "shared")); !pr.OK {
		t.Fatalf("put b: %v", pr.Error())
	}
	if len(s.List()) != 2 {
		t.Fatalf("both entries should be stored despite the shared alias: got %d", len(s.List()))
	}

	// Re-Put of the same id updates in place rather than erroring.
	if pr := s.Put(sampleEntry("a", 9_000_000_000, "shared")); !pr.OK {
		t.Fatalf("update in place: %v", pr.Error())
	}
	if got := s.Get("a").Value.(Entry).MemoryBytes; got != 9_000_000_000 {
		t.Errorf("updated footprint: got %d, want 9000000000", got)
	}
}

func TestStore_MemStore_Get_Good(t *testing.T) {
	s := NewMemStore()
	if pr := s.Put(sampleEntry("gemma-4-4b-it", 4_500_000_000)); !pr.OK {
		t.Fatalf("put: %v", pr.Error())
	}
	got := s.Get("gemma-4-4b-it")
	if !got.OK {
		t.Fatalf("get: %v", got.Error())
	}
	if id := got.Value.(Entry).ID; id != "gemma-4-4b-it" {
		t.Errorf("id: got %q, want gemma-4-4b-it", id)
	}
}

func TestStore_MemStore_Get_Bad(t *testing.T) {
	s := NewMemStore()
	if r := s.Get("does-not-exist"); r.OK {
		t.Fatalf("get of unknown id should fail, got %+v", r.Value)
	}
}

func TestStore_MemStore_Get_Ugly(t *testing.T) {
	// Get is an exact, case-sensitive key lookup — the store itself does no
	// normalisation (that lives in the Registry's alias index).
	s := NewMemStore()
	if pr := s.Put(sampleEntry("Gemma-4-4B-IT", 1)); !pr.OK {
		t.Fatalf("put: %v", pr.Error())
	}
	if r := s.Get("gemma-4-4b-it"); r.OK {
		t.Fatalf("get should be case-sensitive, unexpectedly matched: %+v", r.Value)
	}
}

func TestStore_MemStore_List_Good(t *testing.T) {
	s := NewMemStore()
	for _, id := range []string{"zeta", "alpha", "mu"} {
		if pr := s.Put(sampleEntry(id, 1)); !pr.OK {
			t.Fatalf("put %s: %v", id, pr.Error())
		}
	}
	all := s.List()
	got := make([]string, len(all))
	for i, e := range all {
		got[i] = e.ID
	}
	want := []string{"alpha", "mu", "zeta"}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("list order: got %v, want %v", got, want)
		}
	}
}

func TestStore_MemStore_List_Bad(t *testing.T) {
	s := NewMemStore()
	if all := s.List(); len(all) != 0 {
		t.Fatalf("empty store list: got %d, want 0", len(all))
	}
}

func TestStore_MemStore_List_Ugly(t *testing.T) {
	// The store copies the Entry struct into List's slice, but slice-typed
	// fields (Aliases) still share the stored entry's backing array —
	// mutating a returned entry's Aliases element reaches into the store.
	s := NewMemStore()
	if pr := s.Put(sampleEntry("a", 1, "orig")); !pr.OK {
		t.Fatalf("put: %v", pr.Error())
	}
	all := s.List()
	all[0].Aliases[0] = "mutated"
	got := s.Get("a").Value.(Entry)
	if got.Aliases[0] != "mutated" {
		t.Fatalf("expected shared backing array to reflect the mutation, got %q", got.Aliases[0])
	}
}

func TestStore_MemStore_Delete_Good(t *testing.T) {
	s := NewMemStore()
	if pr := s.Put(sampleEntry("gemma-4-4b-it", 1)); !pr.OK {
		t.Fatalf("put: %v", pr.Error())
	}
	if dr := s.Delete("gemma-4-4b-it"); !dr.OK {
		t.Fatalf("delete: %v", dr.Error())
	}
	if r := s.Get("gemma-4-4b-it"); r.OK {
		t.Fatalf("deleted entry still gettable")
	}
}

func TestStore_MemStore_Delete_Bad(t *testing.T) {
	s := NewMemStore()
	if dr := s.Delete("ghost"); dr.OK {
		t.Fatalf("delete of a missing id should fail, got %+v", dr.Value)
	}
}

func TestStore_MemStore_Delete_Ugly(t *testing.T) {
	// Delete fully frees the id for reuse — a subsequent Put with the same id
	// is a fresh insert, not blocked by any leftover state.
	s := NewMemStore()
	if pr := s.Put(sampleEntry("reusable", 1)); !pr.OK {
		t.Fatalf("put: %v", pr.Error())
	}
	if dr := s.Delete("reusable"); !dr.OK {
		t.Fatalf("delete: %v", dr.Error())
	}
	if pr := s.Put(sampleEntry("reusable", 9_000_000_000)); !pr.OK {
		t.Fatalf("re-put after delete: %v", pr.Error())
	}
	if got := s.Get("reusable").Value.(Entry).MemoryBytes; got != 9_000_000_000 {
		t.Errorf("re-put after delete: got %d, want 9000000000", got)
	}
}
