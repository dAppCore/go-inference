// SPDX-Licence-Identifier: EUPL-1.2

package registry

import "testing"

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

	// Put with an empty id is rejected.
	if pr := (New()).Put(Entry{ID: ""}); pr.OK {
		t.Fatalf("put empty id should fail, got %+v", pr.Value)
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

	// Filter by capability.
	r.Put(func() Entry {
		e := sampleEntry("no-tools", 1_000_000_000, "nt")
		e.Capabilities.Tools = false
		return e
	}())
	tools := r.Filter(Filter{Tools: true})
	if len(tools) != 2 {
		t.Fatalf("tools filter: got %d, want 2", len(tools))
	}
}

func TestRegistry_List_Bad(t *testing.T) {
	// An empty registry lists nothing and resolves nothing.
	r := New()
	if all := r.List(); len(all) != 0 {
		t.Fatalf("empty list: got %d, want 0", len(all))
	}
	if res := r.Resolve("anything"); res.OK {
		t.Fatalf("empty registry resolve should fail")
	}
	// Delete of a missing id fails cleanly.
	if dr := r.Delete("ghost"); dr.OK {
		t.Fatalf("delete missing id should fail")
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

// ids extracts the ids of a slice of entries for test diagnostics.
func ids(es []Entry) []string {
	out := make([]string, len(es))
	for i, e := range es {
		out[i] = e.ID
	}
	return out
}
