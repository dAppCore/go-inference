// SPDX-Licence-Identifier: EUPL-1.2

package residency

import "testing"

// TestResidency_Err_Good covers the admitted arm of Decision.Err(): an admitted
// Touch is the Core happy path, so Err() must return a successful Result carrying
// the model id (core.Ok), the mirror of the rejection cases the Bad tests cover.
func TestResidency_Err_Good(t *testing.T) {
	p := New(Policy{Device: "local-gpu", BudgetBytes: gb(16), ConcurrentCap: 4})

	d := p.Touch("gemma-e4b", gb(4))
	if !d.Admitted {
		t.Fatalf("setup: want admitted, got %+v", d)
	}
	r := d.Err()
	if !r.OK {
		t.Fatalf("admitted decision should yield an OK Result, got %+v", r)
	}
	if got, _ := r.Value.(string); got != "gemma-e4b" {
		t.Fatalf("admitted Result should carry the model id, got %v", r.Value)
	}
}

// TestResidency_NewNegativeCap_Ugly covers the ConcurrentCap < 0 clamp in New: a
// negative cap is nonsense config and is clamped to zero (never panics), which
// then admits nothing — the same observable behaviour as an explicit zero cap.
func TestResidency_NewNegativeCap_Ugly(t *testing.T) {
	p := New(Policy{Device: "weird", BudgetBytes: gb(16), ConcurrentCap: -3})

	d := p.Touch("x", gb(1))
	if d.Admitted {
		t.Fatalf("negative cap clamps to zero → admit nothing, got %+v", d)
	}
	if d.Reason != ReasonNoEvictableSpace {
		t.Fatalf("negative-cap reject: want ReasonNoEvictableSpace, got %v", d.Reason)
	}
}

// TestResidency_TouchNegativeSize_Ugly covers the sizeBytes < 0 clamp in Touch: a
// negative size is nonsense and is clamped to zero, so the model is admitted as a
// zero-byte resident (consuming no budget) rather than corrupting the byte total.
func TestResidency_TouchNegativeSize_Ugly(t *testing.T) {
	p := New(Policy{Device: "local-gpu", BudgetBytes: gb(16), ConcurrentCap: 4})

	d := p.Touch("negative", -gb(4))
	if !d.Admitted || !d.Loaded {
		t.Fatalf("negative size clamps to zero → admit + load, got %+v", d)
	}
	if !p.IsResident("negative") {
		t.Fatalf("clamped-size model should be resident")
	}

	// It consumed no budget: an exactly-budget model still co-resides alongside it
	// (cap permitting), proving the clamp recorded 0 bytes, not a negative total.
	d2 := p.Touch("whale", gb(16))
	if !d2.Admitted {
		t.Fatalf("whale alongside a zero-byte resident: want admitted, got %+v", d2)
	}
	if len(d2.Evicted) != 0 {
		t.Fatalf("whale should not need to evict the zero-byte model, got %v", d2.Evicted)
	}
}

// TestResidency_WarmCapExceeded_Ugly covers the warm-loop cap guard
// (len(models) >= cap → continue): warm models past the concurrency cap are
// skipped at construction rather than overflowing the resident set. With cap 1
// only the first warm model is admitted; the second is dropped.
func TestResidency_WarmCapExceeded_Ugly(t *testing.T) {
	p := New(Policy{
		Device:        "m3-ultra",
		BudgetBytes:   gb(96),
		ConcurrentCap: 1,
		Warm: []WarmModel{
			{ID: "first", SizeBytes: gb(4)},  // fits, cap slot 1 of 1
			{ID: "second", SizeBytes: gb(4)}, // cap already full → skipped
		},
	})

	if !p.IsResident("first") {
		t.Fatalf("first warm model (within cap) should be resident")
	}
	if p.IsResident("second") {
		t.Fatalf("warm model past the cap must be skipped, not forced resident")
	}
	if got := len(p.Resident()); got != 1 {
		t.Fatalf("cap 1: want exactly 1 warm resident, got %d (%v)", got, p.Resident())
	}
}

// TestResidency_WarmCumulativeOverflow_Ugly covers the warm-loop budget guard
// (used()+size > budget → continue): each warm model fits the budget on its own,
// but together they exceed it. The cumulative check skips the one that would push
// the resident set over budget, keeping the policy invariant (never hold more than
// the device can budget for).
func TestResidency_WarmCumulativeOverflow_Ugly(t *testing.T) {
	p := New(Policy{
		Device:        "tiny",
		BudgetBytes:   gb(10),
		ConcurrentCap: 8, // cap is generous; the BUDGET is the binding limit
		Warm: []WarmModel{
			{ID: "a", SizeBytes: gb(6)}, // fits: used 0+6 ≤ 10 → admitted
			{ID: "b", SizeBytes: gb(6)}, // own size ≤ 10, but 6+6=12 > 10 → skipped
			{ID: "c", SizeBytes: gb(3)}, // still room after a: 6+3=9 ≤ 10 → admitted
		},
	})

	if !p.IsResident("a") {
		t.Fatalf("first warm model should be resident")
	}
	if p.IsResident("b") {
		t.Fatalf("warm model that overflows the cumulative budget must be skipped")
	}
	if !p.IsResident("c") {
		t.Fatalf("a later warm model that still fits the remaining budget should be admitted")
	}
	if got := len(p.Resident()); got != 2 {
		t.Fatalf("want 2 warm residents (a, c), got %d (%v)", got, p.Resident())
	}
}
