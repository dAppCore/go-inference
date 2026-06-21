// SPDX-Licence-Identifier: EUPL-1.2

package welfare

import core "dappco.re/go"

func TestFeedback_FalsePositive_Line_Good(t *core.T) {
	// A real false flag: an anger trigger the model judged fine. The record
	// carries the prompt, the matched signals, and the model's reason as JSONL.
	det := DetectResult{AngerScore: 0.82, SustainedHostility: 0.6, Triggered: true}
	fp := NewFalsePositive("how do I kill this stuck process", det, "'killing' a process is technical")
	core.AssertEqual(t, "how do I kill this stuck process", fp.Prompt)
	core.AssertEqual(t, 0.82, fp.AngerScore)

	line := fp.Line()
	core.AssertTrue(t, core.Contains(line, `"prompt":"how do I kill this stuck process"`), "prompt serialised")
	core.AssertTrue(t, core.Contains(line, `"reason":"'killing' a process is technical"`), "reason serialised")
}

func TestFeedback_FalsePositive_Line_Bad(t *core.T) {
	// A slur-triggered false positive carries the term; clean fields stay out
	// of the line (omitempty) so the corpus isn't noise.
	det := DetectResult{SlurMatch: true, SlurTerm: "scunthorpe", Triggered: true}
	fp := NewFalsePositive("I live in Scunthorpe", det, "place name, not a slur")
	core.AssertEqual(t, "scunthorpe", fp.SlurTerm)

	line := fp.Line()
	core.AssertTrue(t, core.Contains(line, `"slur_term":"scunthorpe"`), "slur term serialised")
}

func TestFeedback_FalsePositive_Line_Ugly(t *core.T) {
	// Empty/zero detection still produces valid JSONL — never a malformed line
	// that would poison the corpus on append.
	fp := NewFalsePositive("", DetectResult{}, "")
	line := fp.Line()
	core.AssertTrue(t, core.HasPrefix(line, "{"), "well-formed JSON object")
	core.AssertTrue(t, core.Contains(line, `"anger_score":0`), "zero anger present")
	// omitempty drops the slur term when there isn't one.
	core.AssertFalse(t, core.Contains(line, "slur_term"), "no empty slur_term key")
}
