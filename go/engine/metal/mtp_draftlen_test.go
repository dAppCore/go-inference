// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

// TestMTPDraftLenRaisesOnHotStreak pins the additive ladder: two full-accept
// cycles raise the cap one step, two more raise it again, capped at max.
func TestMTPDraftLenRaisesOnHotStreak(t *testing.T) {
	if mtpDraftLenDisabled {
		t.Skip("LTHN_MTP_DRAFTLEN=0")
	}
	d := newMTPDraftLen(4)
	if got := d.next(100); got != 4 {
		t.Fatalf("initial next = %d, want 4", got)
	}
	d.cycle(true, mtpDraftLenDeepPos)
	if d.cap != 4 {
		t.Fatalf("cap after 1 hot = %d, want 4 (bar is 2)", d.cap)
	}
	d.cycle(true, mtpDraftLenDeepPos)
	if d.cap != 6 {
		t.Fatalf("cap after 2 hot = %d, want 6", d.cap)
	}
	d.cycle(true, mtpDraftLenDeepPos)
	d.cycle(true, mtpDraftLenDeepPos)
	if d.cap != 8 {
		t.Fatalf("cap after 4 hot = %d, want 8", d.cap)
	}
	d.cycle(true, mtpDraftLenDeepPos)
	d.cycle(true, mtpDraftLenDeepPos)
	if d.cap != 8 {
		t.Fatalf("cap past max = %d, want 8", d.cap)
	}
}

// TestMTPDraftLenShallowNeverRaises pins the depth gate: full-accept streaks
// below mtpDraftLenDeepPos leave the cap at base (the shallow lane stays
// byte-identical to the fixed-K engine).
func TestMTPDraftLenShallowNeverRaises(t *testing.T) {
	if mtpDraftLenDisabled {
		t.Skip("LTHN_MTP_DRAFTLEN=0")
	}
	d := newMTPDraftLen(4)
	for range 6 {
		d.cycle(true, mtpDraftLenDeepPos-1)
	}
	if d.cap != 4 {
		t.Fatalf("shallow cap = %d, want 4", d.cap)
	}
}

// TestMTPDraftLenResetsOnPartialCycle pins the re-anchor: any non-full cycle
// drops the cap straight back to base and clears the streak.
func TestMTPDraftLenResetsOnPartialCycle(t *testing.T) {
	if mtpDraftLenDisabled {
		t.Skip("LTHN_MTP_DRAFTLEN=0")
	}
	d := newMTPDraftLen(4)
	d.cycle(true, mtpDraftLenDeepPos)
	d.cycle(true, mtpDraftLenDeepPos)
	if d.cap != 6 {
		t.Fatalf("setup cap = %d, want 6", d.cap)
	}
	d.cycle(false, mtpDraftLenDeepPos)
	if d.cap != 4 || d.hot != 0 {
		t.Fatalf("after partial: cap=%d hot=%d, want 4,0", d.cap, d.hot)
	}
	// a lone full accept after the reset must not raise (streak restarted)
	d.cycle(true, mtpDraftLenDeepPos)
	if d.cap != 4 {
		t.Fatalf("cap after reset+1 hot = %d, want 4", d.cap)
	}
}

// TestMTPDraftLenNextRespectsRemaining pins that the emitted block never
// exceeds the tokens still wanted, whatever the cap.
func TestMTPDraftLenNextRespectsRemaining(t *testing.T) {
	d := newMTPDraftLen(4)
	d.cap = 8
	if got := d.next(3); got != 3 {
		t.Fatalf("next(3) = %d, want 3", got)
	}
	if got := d.next(50); got != 8 {
		t.Fatalf("next(50) = %d, want 8", got)
	}
}
