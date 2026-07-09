// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "os"

// mtpDraftLen — #359's first live scheduling slice: an additive-only dynamic
// draft cap driven by acceptance history. The LTHN_MTP_CONF calibration runs
// showed 40% of 26B cycles (30% on 12B) accept the whole fixed K=4 block and
// end with cumprod ~0.99 — the drafter wants to keep going and is right, and
// the waste is worst exactly where drafting pays most (83% acceptance at 16K
// depth, where every avoided verify forward is the expensive deep kind). So:
// consecutive fully-accepted cycles raise the cap toward mtpDraftLenMax;
// any partial cycle resets it to the caller's base. Output is unchanged by
// construction — verification pins the emitted stream to the target's greedy
// continuation whatever the block length; only the cycle economics move.
// Confidence-cumprod truncation (the θ=0.40 early stop) is the next slice —
// it needs a per-token probability on the live path, which this deliberately
// avoids: acceptance history is free.
//
// LTHN_MTP_DRAFTLEN=0 restores the fixed base cap byte-exactly (the repro
// anchor, same convention as LTHN_MTP_REENGAGE).
var mtpDraftLenDisabled = os.Getenv("LTHN_MTP_DRAFTLEN") == "0"

const (
	// mtpDraftLenMax bounds the raised cap: DSpark ships block7 drafters and
	// our verify lane is sized for small widths (#354), so 8 stays inside
	// the measured-cheap regime (block + carried lead = 9 verify rows).
	mtpDraftLenMax = 8
	// mtpDraftLenHotBar — full-accept cycles in a row before a raise. One
	// full block is common noise (a hot streak is not); two in a row was the
	// calibration sample's reliable "still hot" signal.
	mtpDraftLenHotBar = 2
	// mtpDraftLenRaise — cap step per hot streak (4 → 6 → 8).
	mtpDraftLenRaise = 2
	// mtpDraftLenDeepPos gates raises to depth: the calibration curves put
	// dense full-accept streaks at depth (83% acceptance at 16K vs 48%
	// shallow), and the live A/B confirmed the trade — +8.6% at 16K, −2.7%
	// shallow when raises applied everywhere. Below this position the cap
	// stays at base and the loop is byte-identical to the fixed-K engine.
	mtpDraftLenDeepPos = 4096
)

type mtpDraftLen struct {
	base int // caller-resolved fixed draft tokens (flag or engine default)
	cap  int // live cap, [base, mtpDraftLenMax]
	hot  int // consecutive fully-accepted cycles at the current cap
}

func newMTPDraftLen(base int) mtpDraftLen {
	return mtpDraftLen{base: base, cap: base}
}

// next returns the block size for the coming draft call.
func (d *mtpDraftLen) next(remaining int) int {
	return min(d.cap, remaining)
}

// cycle feeds one verify outcome. fullAccept must be the cycle's AllAccepted
// verdict (carry-aware), not a raw count comparison; pos is the target
// position, gating raises to the deep regime where the streaks are dense.
func (d *mtpDraftLen) cycle(fullAccept bool, pos int) {
	if mtpDraftLenDisabled {
		return
	}
	if !fullAccept {
		d.cap = d.base
		d.hot = 0
		return
	}
	if pos < mtpDraftLenDeepPos {
		return
	}
	d.hot++
	if d.hot >= mtpDraftLenHotBar && d.cap < mtpDraftLenMax {
		d.cap = min(d.cap+mtpDraftLenRaise, mtpDraftLenMax)
		d.hot = 0
	}
}
