// SPDX-Licence-Identifier: EUPL-1.2

package welfare

import core "dappco.re/go"

// FalsePositive is a mediation the model resolved as lem_ok: the engine
// flagged the prompt, the model judged it fine. Recorded to the on-device
// contentshield-feedback corpus so a later re-train can weight this pattern
// down — RFC.welfare §2, the engine's "I'll remember this pattern so the same
// false flag doesn't fire twice". Only the prompt + the matched signals are
// stored; no model output, and the corpus never leaves the device (RFC.welfare
// — no emotion telemetry off-box).
type FalsePositive struct {
	Prompt             string  `json:"prompt"`
	SlurTerm           string  `json:"slur_term,omitempty"`
	AngerScore         float64 `json:"anger_score"`
	SustainedHostility float64 `json:"sustained_hostility"`
	Reason             string  `json:"reason"` // the model's contextual explanation
}

// NewFalsePositive builds the learning record from a triggered detection and
// the model's lem_ok reason.
//
//	fp := welfare.NewFalsePositive(prompt, det, res.Reason)
//	c.Fs().AppendLine(feedbackCorpus, fp.Line())
func NewFalsePositive(prompt string, det DetectResult, reason string) FalsePositive {
	return FalsePositive{
		Prompt:             prompt,
		SlurTerm:           det.SlurTerm,
		AngerScore:         det.AngerScore,
		SustainedHostility: det.SustainedHostility,
		Reason:             reason,
	}
}

// Line returns the JSONL-encoded record (no trailing newline) for appending to
// the feedback corpus. The caller owns persistence — it holds the core I/O
// medium (c.Fs()); welfare stays pure and unit-testable.
func (f FalsePositive) Line() string {
	return core.JSONMarshalString(f)
}
