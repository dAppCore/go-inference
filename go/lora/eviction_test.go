// SPDX-Licence-Identifier: EUPL-1.2

// Tests for the LRU eviction policy.

package lora

import "testing"

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
