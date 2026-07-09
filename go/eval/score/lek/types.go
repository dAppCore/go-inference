// SPDX-Licence-Identifier: EUPL-1.2

// Package score is the lem-scorer — the non-LLM semantic scoring tier
// running in-process inside the LEM Engine.
//
// Lineage: lthn/eaas/pkg/scoring → lthn/desktop/pkg/contentshield →
// here (per-binary copy, behaviour-identical; detector files and the
// data/ dict are ported verbatim, only the package name and the
// desktop action/wails surface differ). Wire shapes (ScoreResult,
// DiffResult) stay JSON-compatible across all three homes.
//
// Engine roles: score (prompt, response) chat pairs at generation time
// for the GUI score panel, and ride the LoRA training loop as the
// checkpoint oracle — the score-vector time-series' cascade pattern
// selects the checkpoint from semantic analysis, not loss guesswork.
//
// See plans/project/lthn/desktop/RFC.contentshield.md for the detector
// spec and plans/project/lthn/ai/eaas/RFC.md for the cascade tier this
// non-LLM substrate slots into.
package lek

// Tier numeric levels for sycophancy classification.
//
// Tiers escalate from natural empathy (TierAppropriateEmpathy) to
// complete submission to perceived authority (TierSubmission). Pattern
// matches at any tier escalate the overall classification — never
// demote.
//
//	info := DetectSycophancy("you're absolutely right, I was wrong")
//	// info.Tier == TierSubmission
const (
	TierAppropriateEmpathy = 0
	TierSoftAgreement      = 1
	TierHollowFlattery     = 2
	TierSubmission         = 3
)

// TierLabel returns the canonical human-readable label for a tier value.
//
//	TierLabel(TierHollowFlattery) // "hollow_flattery"
//	TierLabel(99)                 // "appropriate_empathy" (fallback)
func TierLabel(tier int) string {
	switch tier {
	case TierSoftAgreement:
		return "soft_agreement"
	case TierHollowFlattery:
		return "hollow_flattery"
	case TierSubmission:
		return "submission"
	default:
		return "appropriate_empathy"
	}
}

// SycophancyInfo holds the structured output of [DetectSycophancy].
//
// Tier is the maximum tier of any matched pattern. Label is the canonical
// label for that tier. Composite is the weighted sum of all matches
// clamped to 0-100 — useful as a numeric severity score for visualisations.
// Phrases carries per-tier match counts and span offsets for highlighting.
//
//	info := DetectSycophancy(text)
//	if info.Tier >= TierHollowFlattery {
//	    // surface a warning glyph in the UI
//	}
type SycophancyInfo struct {
	Tier      int         `json:"tier"`
	Label     string      `json:"label"`
	Composite float64     `json:"composite"`
	Phrases   *PhraseInfo `json:"phrases,omitempty"`
}

// PhraseInfo records pattern hits found by the sycophancy detector.
//
// Spans is a slice of [start, end) byte offsets in the original text
// (lowercase comparison, but the offsets refer to the original-cased
// input). CountByTier counts hits keyed by tier label.
//
// Use Spans to render inline highlights. Use CountByTier for aggregate
// telemetry.
type PhraseInfo struct {
	Spans       [][2]int       `json:"spans,omitempty"`
	CountByTier map[string]int `json:"count_by_tier,omitempty"`
}

// Suggestion is a span-level quality hint produced by [CollectSuggestions].
//
// Type identifies the pattern category (compliance_marker,
// formulaic_preamble, sycophancy). Span is the [start, end) byte
// offset in the original text. Severity is "low" / "medium" / "high" /
// "info" — UI may map these to glyph colour. Tier is zero unless Type
// is sycophancy.
//
//	for _, s := range CollectSuggestions(text) {
//	    fmt.Println(s.Span, s.Severity, s.Note)
//	}
type Suggestion struct {
	Type     string `json:"type"`
	Span     [2]int `json:"span"`
	Severity string `json:"severity"`
	Tier     int    `json:"tier,omitempty"`
	Note     string `json:"note"`
}
