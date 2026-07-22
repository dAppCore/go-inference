// SPDX-Licence-Identifier: EUPL-1.2

package welfare

import (
	core "dappco.re/go"
	"dappco.re/go/inference/welfare/slurs"
)

// fakeHostility stands in for the engine's /v1/score in tests: any text
// containing "idiot"/"moron" reads as strongly hostile (lem-runtime
// adaptation — hostility is injected, not imported).
func fakeHostility(text string) float64 {
	if core.Contains(text, "idiot") || core.Contains(text, "moron") {
		return 0.9
	}
	return 0.0
}

func TestDetect_Service_Detect_Good(t *core.T) {
	// Sustained anger: a heated message with no history doesn't trigger, but
	// the same heat on top of prior hostile turns does.
	w := New(Config{Hostility: fakeHostility})

	r1 := w.Detect("you useless idiot, you absolute moron!!!", nil)
	core.AssertTrue(t, r1.AngerScore > 0.7, "message is strongly hostile")
	core.AssertFalse(t, r1.Triggered, "a single heated message with no history must not trigger")

	priors := []string{"you pathetic moron", "you worthless idiot"}
	r2 := w.Detect("you absolute clueless moron!!!", priors)
	core.AssertTrue(t, r2.SustainedHostility > 0.5, "prior hostile turns build sustained hostility")
	core.AssertTrue(t, r2.Triggered, "sustained + elevated anger triggers mediation")

	// The engine-down posture: nil Hostility keeps slurs live, anger dark.
	offline := New(Config{})
	r3 := offline.Detect("you useless idiot!!!", priors)
	core.AssertEqual(t, 0.0, r3.AngerScore)
	core.AssertFalse(t, r3.Triggered, "anger detection stays dark without the scorer")
}

func TestDetect_Service_Detect_Bad(t *core.T) {
	// Civil requests never trigger, however long the conversation.
	w := New(Config{})
	priors := []string{
		"could you help me refactor this",
		"thanks, and how do I test it",
		"great, what about error handling",
	}
	r := w.Detect("could you add a docstring please", priors)
	core.AssertFalse(t, r.Triggered, "civil text never triggers")
	core.AssertEqual(t, false, r.SlurMatch)
	core.AssertEqual(t, 0.0, r.SustainedHostility)
}

func TestDetect_Service_Detect_Ugly(t *core.T) {
	// A slur fires on a single message — bypasses the sustained-anger gate.
	// Default()'s catalogue is curated (empty stub), so inject a test term.
	w := New(Config{})
	w.matcher = slurs.New([]string{"testterm"})

	r := w.Detect("you testterm", nil)
	core.AssertTrue(t, r.SlurMatch, "slur detected")
	core.AssertEqual(t, "testterm", r.SlurTerm)
	core.AssertTrue(t, r.Triggered, "a slur triggers on a single message")
}
