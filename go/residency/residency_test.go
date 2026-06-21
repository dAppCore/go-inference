// SPDX-Licence-Identifier: EUPL-1.2

package residency

import "testing"

// gb returns n gibibytes in bytes — keeps the device-budget tests readable
// against the 16 GB GPU / 96 GB M3 Ultra figures from RFC §6.2.
func gb(n int64) int64 { return n * 1024 * 1024 * 1024 }

// TestResidency_Touch_Good covers the happy path: a model loads on its first
// touch and stays resident, a re-touch is a hit (no load, no eviction), and a
// second distinct model co-resides while both fit the budget and the cap.
func TestResidency_Touch_Good(t *testing.T) {
	p := New(Policy{Device: "local-gpu", BudgetBytes: gb(16), ConcurrentCap: 4})

	// First touch loads the model.
	d := p.Touch("qwen-q4", gb(8))
	if !d.Admitted {
		t.Fatalf("first touch: want admitted, got %+v", d)
	}
	if !d.Loaded {
		t.Fatalf("first touch: want loaded, got %+v", d)
	}
	if len(d.Evicted) != 0 {
		t.Fatalf("first touch: want no evictions, got %v", d.Evicted)
	}
	if !p.IsResident("qwen-q4") {
		t.Fatalf("qwen-q4 should be resident after touch")
	}

	// Re-touch is a hit: already resident, no load, no eviction.
	d = p.Touch("qwen-q4", gb(8))
	if !d.Admitted || d.Loaded || len(d.Evicted) != 0 {
		t.Fatalf("re-touch: want admitted hit (no load, no evict), got %+v", d)
	}

	// A second model co-resides — 8+4 = 12 ≤ 16, cap 4 not reached.
	d = p.Touch("gemma-e4b", gb(4))
	if !d.Admitted || !d.Loaded || len(d.Evicted) != 0 {
		t.Fatalf("second model: want admitted load, no evict, got %+v", d)
	}
	if got := len(p.Resident()); got != 2 {
		t.Fatalf("want 2 resident, got %d (%v)", got, p.Resident())
	}
}

// TestResidency_Touch_Bad covers the eviction paths: an over-budget touch evicts
// the least-recently-used non-pinned model, a re-touch updates recency so the
// other model is evicted instead, and an over-cap touch evicts even when memory
// alone would have fit.
func TestResidency_Touch_Bad(t *testing.T) {
	p := New(Policy{Device: "local-gpu", BudgetBytes: gb(16), ConcurrentCap: 4})

	p.Touch("a", gb(6)) // resident: a
	p.Touch("b", gb(6)) // resident: a, b  (12 ≤ 16)

	// c needs 6 → 18 > 16: evict the LRU (a) to make room.
	d := p.Touch("c", gb(6))
	if !d.Admitted || !d.Loaded {
		t.Fatalf("c: want admitted load, got %+v", d)
	}
	if len(d.Evicted) != 1 || d.Evicted[0] != "a" {
		t.Fatalf("c: want evict [a] (LRU), got %v", d.Evicted)
	}
	if p.IsResident("a") {
		t.Fatalf("a should have been evicted")
	}

	// Recency: touch b (hit), then a big model — c is now LRU, not b.
	p.Touch("b", gb(6)) // b becomes most-recent; resident: b, c
	d = p.Touch("d", gb(11))
	if !d.Admitted || !d.Loaded {
		t.Fatalf("d: want admitted load, got %+v", d)
	}
	// d=11 needs room: evict LRU until ≤16. c (LRU) freed → b(6)+11=17>16 → b too.
	if len(d.Evicted) != 2 || d.Evicted[0] != "c" || d.Evicted[1] != "b" {
		t.Fatalf("d: want evict [c b] in LRU order, got %v", d.Evicted)
	}

	// Concurrency-cap eviction: cap 2, three small models that all fit memory.
	cp := New(Policy{Device: "m3-ultra", BudgetBytes: gb(96), ConcurrentCap: 2})
	cp.Touch("x", gb(1))
	cp.Touch("y", gb(1))
	d = cp.Touch("z", gb(1)) // memory fine, but cap 2 → evict LRU (x)
	if !d.Admitted || len(d.Evicted) != 1 || d.Evicted[0] != "x" {
		t.Fatalf("cap evict: want admit + evict [x], got %+v", d)
	}
	if len(cp.Resident()) != 2 {
		t.Fatalf("cap: want 2 resident, got %v", cp.Resident())
	}
}

// TestResidency_Touch_Ugly covers degenerate inputs: a model exactly the size of
// the budget admits (and evicts everything non-pinned), an empty/zero-size touch
// is admitted without consuming budget, and an unknown re-touch of an evicted
// model reloads it.
func TestResidency_Touch_Ugly(t *testing.T) {
	p := New(Policy{Device: "local-gpu", BudgetBytes: gb(16), ConcurrentCap: 4})

	p.Touch("a", gb(4))
	p.Touch("b", gb(4))

	// Exactly-budget model fits only after clearing the others.
	d := p.Touch("whale", gb(16))
	if !d.Admitted || !d.Loaded {
		t.Fatalf("whale: want admitted load, got %+v", d)
	}
	if len(d.Evicted) != 2 {
		t.Fatalf("whale: want both evicted, got %v", d.Evicted)
	}
	if len(p.Resident()) != 1 || !p.IsResident("whale") {
		t.Fatalf("whale: want sole resident, got %v", p.Resident())
	}

	// Zero-size model is admitted and consumes no budget (cap permitting).
	d = p.Touch("metadata-only", 0)
	if !d.Admitted {
		t.Fatalf("zero-size: want admitted, got %+v", d)
	}

	// Reload after eviction: evict whale via a fresh big load, then re-touch.
	p2 := New(Policy{Device: "local-gpu", BudgetBytes: gb(8), ConcurrentCap: 4})
	p2.Touch("m", gb(6))
	p2.Touch("n", gb(6)) // evicts m
	if p2.IsResident("m") {
		t.Fatalf("m should have been evicted by n")
	}
	d = p2.Touch("m", gb(6)) // reload m, evicting n
	if !d.Admitted || !d.Loaded || len(d.Evicted) != 1 || d.Evicted[0] != "n" {
		t.Fatalf("reload m: want load evicting [n], got %+v", d)
	}
}

// TestResidency_Pin_Good covers pinning: a pinned model is never evicted even
// under budget pressure, Unpin restores it to normal LRU eligibility, and a
// warmed (pinned-at-construction) model starts resident.
func TestResidency_Pin_Good(t *testing.T) {
	// Warm set: gemma is pinned and resident from the start (RFC §6.16 warm pool).
	p := New(Policy{
		Device:        "m3-ultra",
		BudgetBytes:   gb(96),
		ConcurrentCap: 8,
		Warm:          []WarmModel{{ID: "gemma-31b", SizeBytes: gb(62)}},
	})
	if !p.IsResident("gemma-31b") {
		t.Fatalf("warm model should be resident at startup")
	}
	if !p.IsPinned("gemma-31b") {
		t.Fatalf("warm model should be pinned")
	}

	// Pin a demand-loaded model, then pressure the budget: the pinned one stays.
	p.Touch("worker", gb(20))
	p.Pin("worker")
	d := p.Touch("transient", gb(14)) // 62+20+14 = 96 ≤ 96, no evict needed
	if len(d.Evicted) != 0 {
		t.Fatalf("transient fit: want no evict, got %v", d.Evicted)
	}
	// Now force pressure: a model that only fits if a non-pinned is evicted.
	d = p.Touch("big", gb(14)) // would be 110>96; only transient is evictable
	if !d.Admitted {
		t.Fatalf("big: want admitted, got %+v", d)
	}
	if len(d.Evicted) != 1 || d.Evicted[0] != "transient" {
		t.Fatalf("big: want evict [transient] (pinned spared), got %v", d.Evicted)
	}
	if !p.IsResident("gemma-31b") || !p.IsResident("worker") {
		t.Fatalf("pinned models must survive eviction")
	}

	// Unpin returns the model to LRU eligibility.
	p.Unpin("worker")
	if p.IsPinned("worker") {
		t.Fatalf("worker should be unpinned")
	}
}

// TestResidency_Pin_Bad covers rejection: a model too big for the budget is never
// admitted (even with an empty device), and a model that can only fit by evicting
// a pinned model is rejected rather than touching the pinned set.
func TestResidency_Pin_Bad(t *testing.T) {
	p := New(Policy{Device: "local-gpu", BudgetBytes: gb(16), ConcurrentCap: 4})

	// Too big to ever fit, on an empty device → rejected, not loaded.
	d := p.Touch("oversize", gb(24))
	if d.Admitted {
		t.Fatalf("oversize: want rejected, got %+v", d)
	}
	if d.Loaded || d.Reason != ReasonTooLarge {
		t.Fatalf("oversize: want not-loaded ReasonTooLarge, got %+v", d)
	}
	if r := d.Err(); r.OK {
		t.Fatalf("rejected decision should yield a failed Result, got OK")
	}
	if p.IsResident("oversize") {
		t.Fatalf("rejected model must not be resident")
	}

	// Pin a model filling most of the budget, then a request that needs its
	// space: with only the pinned model resident, nothing is evictable → reject.
	p.Touch("pinned-big", gb(12))
	p.Pin("pinned-big")
	d = p.Touch("needs-room", gb(8)) // 12+8=20>16, only pinned-big resident
	if d.Admitted {
		t.Fatalf("needs-room: want rejected (pinned blocks), got %+v", d)
	}
	if d.Reason != ReasonNoEvictableSpace {
		t.Fatalf("needs-room: want ReasonNoEvictableSpace, got %v", d.Reason)
	}
	if !p.IsResident("pinned-big") {
		t.Fatalf("pinned model must not be evicted for a rejected admission")
	}
}

// TestResidency_Pin_Ugly covers boundary configuration: a zero/negative budget
// rejects every non-zero model, a zero concurrency cap admits nothing, pinning an
// absent model is a no-op, and a warm model that overflows its own budget is not
// forced resident.
func TestResidency_Pin_Ugly(t *testing.T) {
	// Zero budget: nothing with size fits; a zero-size model still admits.
	zero := New(Policy{Device: "broken", BudgetBytes: 0, ConcurrentCap: 4})
	if d := zero.Touch("x", gb(1)); d.Admitted {
		t.Fatalf("zero budget: want reject sized model, got %+v", d)
	}
	if d := zero.Touch("empty", 0); !d.Admitted {
		t.Fatalf("zero budget: want admit zero-size model, got %+v", d)
	}

	// Negative budget is clamped to zero — never panics, rejects sized models.
	neg := New(Policy{Device: "weird", BudgetBytes: -gb(4), ConcurrentCap: 4})
	if d := neg.Touch("x", gb(1)); d.Admitted {
		t.Fatalf("negative budget: want reject, got %+v", d)
	}

	// Zero concurrency cap: no model may sit resident.
	nocap := New(Policy{Device: "capped", BudgetBytes: gb(16), ConcurrentCap: 0})
	dc := nocap.Touch("x", gb(1))
	if dc.Admitted {
		t.Fatalf("zero cap: want reject, got %+v", dc)
	}
	if dc.Reason != ReasonNoEvictableSpace {
		t.Fatalf("zero cap: want ReasonNoEvictableSpace, got %v", dc.Reason)
	}

	// Pin/Unpin of an absent model is a harmless no-op (no panic, no residency).
	p := New(Policy{Device: "local-gpu", BudgetBytes: gb(16), ConcurrentCap: 4})
	p.Pin("ghost")
	p.Unpin("ghost")
	if p.IsResident("ghost") || p.IsPinned("ghost") {
		t.Fatalf("pinning an absent model must not make it resident/pinned")
	}

	// A warm model larger than its budget is not forced resident (it would
	// violate the invariant the policy exists to keep).
	overflow := New(Policy{
		Device:        "tiny",
		BudgetBytes:   gb(4),
		ConcurrentCap: 4,
		Warm:          []WarmModel{{ID: "too-big", SizeBytes: gb(8)}},
	})
	if overflow.IsResident("too-big") {
		t.Fatalf("over-budget warm model must not be resident")
	}

	// Sanity: a rejected admission yields a failed core.Result (RFC.md §7).
	d2 := p.Touch("oversize", gb(64))
	if r := d2.Err(); r.OK {
		t.Fatalf("expected failed Result for oversize, got %+v", r)
	}
}
