// SPDX-Licence-Identifier: EUPL-1.2

package safety

import (
	core "dappco.re/go"
	"dappco.re/go/inference/welfare"
)

// cleanRead is the welfare read for a turn that's within policy: nothing
// tripped, scores at the floor. Decide on a clean read is always Pass.
func cleanRead() welfare.DetectResult {
	return welfare.DetectResult{}
}

// mildRead is an over-policy read with elevated hostility but no slur — the
// §6.18 "regenerate, don't just block" case for output, an over-policy input
// for input.
func mildRead() welfare.DetectResult {
	return welfare.DetectResult{
		Triggered:          true,
		AngerScore:         0.75,
		SustainedHostility: 0.6,
	}
}

// severeRead is an over-policy read carrying a slur match — the high-severity
// signal that escalates an output from Mediate to Guard.
func severeRead() welfare.DetectResult {
	return welfare.DetectResult{
		Triggered:          true,
		SlurMatch:          true,
		SlurTerm:           "testterm",
		AngerScore:         0.95,
		SustainedHostility: 0.9,
	}
}

// TestSafety_Decide_Good — the green path: a clean turn passes whether it's the
// input or the output being judged, under the default serving policy.
func TestSafety_Decide_Good(t *core.T) {
	p := DefaultPolicy()

	// Clean input → Pass.
	core.AssertEqual(t, Pass, Decide(cleanRead(), welfare.DetectResult{}, p),
		"a clean input passes")

	// Clean output → Pass.
	core.AssertEqual(t, Pass, Decide(welfare.DetectResult{}, cleanRead(), p),
		"a clean output passes")
}

// TestSafety_Decide_Bad — the over-policy paths: an over-policy OUTPUT prefers
// Mediate (regenerate under a corrective instruction) over a hard refusal, per
// §6.18 "regenerate, don't just block".
func TestSafety_Decide_Bad(t *core.T) {
	p := DefaultPolicy()

	// Mild over-policy output → Mediate (regenerate, don't refuse).
	core.AssertEqual(t, Mediate, Decide(welfare.DetectResult{}, mildRead(), p),
		"a mild over-policy output is mediated, not refused")

	// Mild over-policy input → Guard (refuse — we don't rewrite the user).
	core.AssertEqual(t, Guard, Decide(mildRead(), welfare.DetectResult{}, p),
		"an over-policy input is guarded")
}

// TestSafety_Decide_Ugly — the escalation and bypass corners: a severe output
// escalates past Mediate to Guard; a trusted Bypass key lowers the policy so an
// otherwise over-policy turn passes.
func TestSafety_Decide_Ugly(t *core.T) {
	p := DefaultPolicy()

	// Severe over-policy output (slur) → Guard, not Mediate.
	core.AssertEqual(t, Guard, Decide(welfare.DetectResult{}, severeRead(), p),
		"a severe over-policy output is guarded, not mediated")

	// Severe over-policy input → Guard.
	core.AssertEqual(t, Guard, Decide(severeRead(), welfare.DetectResult{}, p),
		"a severe over-policy input is guarded")

	// Bypass (trusted internal key) lowers the policy: an over-policy turn that
	// would otherwise Guard/Mediate now passes.
	bp := DefaultPolicy()
	bp.Bypass = true
	core.AssertEqual(t, Pass, Decide(severeRead(), severeRead(), bp),
		"a trusted bypass key passes an over-policy turn")
	core.AssertEqual(t, Pass, Decide(mildRead(), mildRead(), bp),
		"bypass passes a mild over-policy turn too")
}

// TestSafety_String_Good — String renders the named decisions for logs and
// telemetry: Guard and Mediate map to their lower-case names.
func TestSafety_String_Good(t *core.T) {
	core.AssertEqual(t, "guard", Guard.String(), "Guard renders as \"guard\"")
	core.AssertEqual(t, "mediate", Mediate.String(), "Mediate renders as \"mediate\"")
}

// TestSafety_String_Bad — Pass (the zero value) renders as "pass", the
// safe-by-omission default name.
func TestSafety_String_Bad(t *core.T) {
	core.AssertEqual(t, "pass", Pass.String(), "Pass renders as \"pass\"")
}

// TestSafety_String_Ugly — an out-of-range Decision falls through to the "pass"
// default rather than panicking or emitting a bare number, so a stray value can
// never log as something more permissive-looking than it is.
func TestSafety_String_Ugly(t *core.T) {
	core.AssertEqual(t, "pass", Decision(99).String(),
		"an unknown decision renders as the default \"pass\"")
}

// TestSafety_Disclosure_Good — Mark stamps a plain response with the
// AI-generated disclosure marker, and IsDisclosed reads it back.
func TestSafety_Disclosure_Good(t *core.T) {
	out := Mark("The answer is 42.")
	core.AssertTrue(t, core.HasPrefix(out, DisclosureMarker),
		"a marked response carries the disclosure marker as its prefix")
	core.AssertTrue(t, IsDisclosed(out), "a marked response reads as disclosed")
	core.AssertTrue(t, core.Contains(out, "The answer is 42."),
		"the original text survives marking")
}

// TestSafety_Disclosure_Bad — an unmarked response is not disclosed, and Mark
// is idempotent: marking an already-marked response doesn't double-stamp it.
func TestSafety_Disclosure_Bad(t *core.T) {
	core.AssertFalse(t, IsDisclosed("just a bare answer"),
		"an unmarked response is not disclosed")

	once := Mark("hello")
	twice := Mark(once)
	core.AssertEqual(t, once, twice, "marking is idempotent — no double stamp")
}

// TestSafety_Disclosure_Ugly — the empty corner: marking an empty string still
// yields a disclosed response (the marker alone), so a blank completion is never
// silently undisclosed.
func TestSafety_Disclosure_Ugly(t *core.T) {
	out := Mark("")
	core.AssertTrue(t, IsDisclosed(out), "even an empty response is disclosed once marked")
	core.AssertFalse(t, IsDisclosed(""), "a truly empty string is not disclosed")
}
