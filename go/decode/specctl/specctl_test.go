// SPDX-Licence-Identifier: EUPL-1.2

package specctl_test

import (
	"math"
	"sync"
	"testing"

	"dappco.re/go/inference/decode/specctl"
)

// approx reports whether a and b are within a small epsilon — accept-rate maths
// is floating point, so exact equality would be brittle.
func approx(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

// --- Record -----------------------------------------------------------------

// Good: a run of all-accepted proposals drives the accept rate to 1.0, and a
// run of all-rejected drives it back toward 0.0 — the EMA tracks recent acceptance.
func TestSpecCtl_Record_Good(t *testing.T) {
	c := specctl.New(specctl.Controller{Min: 1, Max: 8, Window: 4})

	// Every proposed token accepted → rate climbs to 1.0.
	for i := 0; i < 50; i++ {
		c.Record(8, 8)
	}
	if r := c.AcceptRate(); !approx(r, 1.0) {
		t.Fatalf("all-accepted: AcceptRate = %v, want ~1.0", r)
	}

	// Now nothing accepted → rate decays toward 0.0.
	for i := 0; i < 200; i++ {
		c.Record(8, 0)
	}
	if r := c.AcceptRate(); r > 0.01 {
		t.Fatalf("all-rejected: AcceptRate = %v, want ~0.0", r)
	}

	// A partial sample sits strictly between the extremes.
	c.Reset()
	for i := 0; i < 200; i++ {
		c.Record(4, 2)
	}
	if r := c.AcceptRate(); r <= 0.4 || r >= 0.6 {
		t.Fatalf("half-accepted: AcceptRate = %v, want ~0.5", r)
	}
}

// Bad: proposed==0 is a no-op — it must not move the rate or divide by zero.
func TestSpecCtl_Record_Bad(t *testing.T) {
	c := specctl.New(specctl.Controller{Min: 1, Max: 8, Window: 4})
	for i := 0; i < 10; i++ {
		c.Record(4, 4) // establish rate 1.0
	}
	before := c.AcceptRate()

	c.Record(0, 0) // no-op
	c.Record(0, 5) // no-op even with a nonsense accepted count
	if after := c.AcceptRate(); !approx(before, after) {
		t.Fatalf("zero-proposed changed rate: before=%v after=%v", before, after)
	}
}

// Ugly: accepted > proposed is clamped to proposed (rate never exceeds 1.0);
// negative inputs are floored at zero rather than producing a negative rate.
func TestSpecCtl_Record_Ugly(t *testing.T) {
	c := specctl.New(specctl.Controller{Min: 1, Max: 8, Window: 4})

	// accepted far exceeds proposed → treated as a full-accept sample, rate ≤ 1.
	for i := 0; i < 50; i++ {
		c.Record(4, 999)
	}
	if r := c.AcceptRate(); r > 1.0 || !approx(r, 1.0) {
		t.Fatalf("accepted>proposed: AcceptRate = %v, want clamped ~1.0", r)
	}

	// Negative accepted is floored to zero → behaves as a full-reject sample.
	for i := 0; i < 200; i++ {
		c.Record(4, -7)
	}
	if r := c.AcceptRate(); r < 0 || r > 0.01 {
		t.Fatalf("negative accepted: AcceptRate = %v, want ~0.0", r)
	}

	// Negative proposed is non-positive → no-op (same guard as zero).
	c.Reset()
	for i := 0; i < 10; i++ {
		c.Record(4, 4)
	}
	before := c.AcceptRate()
	c.Record(-3, 2)
	if after := c.AcceptRate(); !approx(before, after) {
		t.Fatalf("negative proposed moved rate: before=%v after=%v", before, after)
	}
}

// --- NextLength -------------------------------------------------------------

// Good: high acceptance pushes the recommendation toward Max, low toward Min,
// and a mid rate lands somewhere strictly between.
func TestSpecCtl_NextLength_Good(t *testing.T) {
	hi := specctl.New(specctl.Controller{Min: 1, Max: 8, Window: 4})
	for i := 0; i < 100; i++ {
		hi.Record(8, 8)
	}
	if n := hi.NextLength(); n != 8 {
		t.Fatalf("high acceptance: NextLength = %d, want 8 (Max)", n)
	}

	lo := specctl.New(specctl.Controller{Min: 1, Max: 8, Window: 4})
	for i := 0; i < 300; i++ {
		lo.Record(8, 0)
	}
	if n := lo.NextLength(); n != 1 {
		t.Fatalf("low acceptance: NextLength = %d, want 1 (Min)", n)
	}

	mid := specctl.New(specctl.Controller{Min: 2, Max: 10, Window: 4})
	for i := 0; i < 300; i++ {
		mid.Record(10, 5) // ~0.5 accept rate
	}
	n := mid.NextLength()
	if n <= 2 || n >= 10 {
		t.Fatalf("mid acceptance: NextLength = %d, want strictly inside (2,10)", n)
	}
}

// Bad: a fresh controller (no Record yet) returns a usable cold-start default —
// the optimistic Max — so the drafter speculates until evidence says otherwise.
func TestSpecCtl_NextLength_Bad(t *testing.T) {
	c := specctl.New(specctl.Controller{Min: 2, Max: 6, Window: 4})
	if n := c.NextLength(); n != 6 {
		t.Fatalf("cold start: NextLength = %d, want 6 (optimistic Max)", n)
	}
	if r := c.AcceptRate(); !approx(r, 1.0) {
		t.Fatalf("cold start: AcceptRate = %v, want 1.0", r)
	}
}

// Ugly: the result is always inside [Min, Max] regardless of how the rate is
// driven, including a degenerate Min==Max controller where there is no range.
func TestSpecCtl_NextLength_Ugly(t *testing.T) {
	c := specctl.New(specctl.Controller{Min: 3, Max: 9, Window: 4})
	// Hammer it with mixed feedback and assert the bound holds at every step.
	for i := 0; i < 500; i++ {
		if i%3 == 0 {
			c.Record(9, 9)
		} else {
			c.Record(9, 0)
		}
		if n := c.NextLength(); n < 3 || n > 9 {
			t.Fatalf("bounds violated at step %d: NextLength = %d, want [3,9]", i, n)
		}
	}

	// Degenerate range: Min==Max → the only legal length is that value.
	flat := specctl.New(specctl.Controller{Min: 5, Max: 5, Window: 4})
	for i := 0; i < 20; i++ {
		flat.Record(5, 2)
		if n := flat.NextLength(); n != 5 {
			t.Fatalf("flat range: NextLength = %d, want 5", n)
		}
	}
}

// --- Config -----------------------------------------------------------------

// Good: a sensible config is used as given — Min/Max bounds drive the output range.
func TestSpecCtl_Config_Good(t *testing.T) {
	c := specctl.New(specctl.Controller{Min: 2, Max: 16, Window: 8})
	if n := c.NextLength(); n != 16 {
		t.Fatalf("config: cold-start NextLength = %d, want 16 (Max)", n)
	}
	for i := 0; i < 400; i++ {
		c.Record(16, 0)
	}
	if n := c.NextLength(); n != 2 {
		t.Fatalf("config: low-rate NextLength = %d, want 2 (Min)", n)
	}
}

// Bad: out-of-range config is clamped — Min<1 becomes 1, a tiny/zero Window
// still yields a working EMA, and the controller is never dead.
func TestSpecCtl_New_Bad(t *testing.T) {
	c := specctl.New(specctl.Controller{Min: 0, Max: 4, Window: 0})
	if n := c.NextLength(); n != 4 {
		t.Fatalf("clamped Min: cold-start NextLength = %d, want 4", n)
	}
	for i := 0; i < 200; i++ {
		c.Record(4, 0)
	}
	if n := c.NextLength(); n != 1 {
		t.Fatalf("clamped Min: low-rate NextLength = %d, want 1 (Min clamped up from 0)", n)
	}

	// Negative Window is clamped to a usable smoothing factor (single-sample EMA).
	w := specctl.New(specctl.Controller{Min: 1, Max: 4, Window: -5})
	w.Record(4, 4)
	if r := w.AcceptRate(); r < 0 || r > 1 {
		t.Fatalf("negative window: AcceptRate = %v out of [0,1]", r)
	}
}

// Ugly: Max < Min is repaired so Max >= Min (the range never inverts), and
// extreme negatives collapse to the Min==Max==1 degenerate but still-valid case.
func TestSpecCtl_Config_Ugly(t *testing.T) {
	c := specctl.New(specctl.Controller{Min: 8, Max: 2, Window: 4}) // inverted
	if n := c.NextLength(); n < 8 {
		t.Fatalf("inverted range: NextLength = %d, want >= Min(8)", n)
	}
	// With Max repaired to >= Min, the range is non-empty and the bound holds.
	for i := 0; i < 200; i++ {
		c.Record(8, 0)
	}
	n := c.NextLength()
	if n < 8 {
		t.Fatalf("inverted range low-rate: NextLength = %d, want >= 8", n)
	}

	all := specctl.New(specctl.Controller{Min: -10, Max: -20, Window: -1})
	if n := all.NextLength(); n != 1 {
		t.Fatalf("all-negative config: NextLength = %d, want 1", n)
	}
}

// --- Reset ------------------------------------------------------------------

// Reset returns the accept rate to the cold-start default so NextLength is Max again.
func TestSpecCtl_Reset(t *testing.T) {
	c := specctl.New(specctl.Controller{Min: 1, Max: 8, Window: 4})
	for i := 0; i < 100; i++ {
		c.Record(8, 0) // drive rate down
	}
	if n := c.NextLength(); n != 1 {
		t.Fatalf("pre-reset: NextLength = %d, want 1", n)
	}
	c.Reset()
	if r := c.AcceptRate(); !approx(r, 1.0) {
		t.Fatalf("post-reset: AcceptRate = %v, want 1.0", r)
	}
	if n := c.NextLength(); n != 8 {
		t.Fatalf("post-reset: NextLength = %d, want 8 (Max)", n)
	}
}

// --- Concurrency ------------------------------------------------------------

// The controller is documented safe to share; the race detector must stay quiet
// under concurrent Record / NextLength / AcceptRate.
func TestSpecCtl_Concurrent(t *testing.T) {
	c := specctl.New(specctl.Controller{Min: 1, Max: 8, Window: 8})
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 1000; i++ {
				c.Record(8, i%9)
				_ = c.NextLength()
				_ = c.AcceptRate()
			}
		}()
	}
	wg.Wait()
	if n := c.NextLength(); n < 1 || n > 8 {
		t.Fatalf("post-race NextLength = %d, want [1,8]", n)
	}
}
