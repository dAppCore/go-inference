// SPDX-Licence-Identifier: EUPL-1.2

package lek

// ScoreResult is the unified lem-scorer output for a single
// piece of text — chat input, AI response, training-corpus chunk,
// opencode session output, plugin output.
//
// Sycophancy is always populated. Imprint is populated when the
// grammar fingerprint was extracted (Score, ScorePair). Suggestions is
// populated when span-level hints were requested. Future detectors
// land as optional slots per plans/project/lthn/desktop/RFC.contentshield.md
// as their port ships.
//
//	r := lek.Score("you're absolutely right, I was wrong")
//	if r.Sycophancy.Tier >= TierHollowFlattery {
//	    // surface a warning glyph in the UI
//	}
//	if r.Imprint != nil && r.Imprint.QuestionRatio > 0.3 {
//	    // text is question-heavy
//	}
type ScoreResult struct {
	Sycophancy  *SycophancyInfo `json:"sycophancy,omitempty"`
	Imprint     *ImprintScores  `json:"imprint,omitempty"`
	Suggestions []Suggestion    `json:"suggestions,omitempty"`
	// LEK is the heuristic axis-set + composite 0..100 LEK score (the
	// lthn.ai/score signal), populated by Score / ScorePair. See lek.go.
	LEK *LEKScores `json:"lek,omitempty"`
	// Hostility is the directed-anger read (0..1) — the AngerScore the welfare
	// layer gates on. Populated by Score / ScorePair. See hostility.go.
	Hostility *HostilityInfo `json:"hostility,omitempty"`
}

// DiffResult is the unified lem-scorer output for a
// (prompt, response) pair — the AI chat path, the training-data
// validation path, the opencode round-trip path.
//
// Prompt and Response each carry their own ScoreResult, scored
// independently. Differential captures the cross-text grammar signal
// (echo, verb shift, tense shift, noun echo, question flip, domain
// shift). Future cross-text dimensions (Authority) land as optional
// slots when their detectors port.
//
//	d := lek.ScorePair(userPrompt, aiResponse)
//	if d.Differential != nil && d.Differential.Echo > 0.7 {
//	    // response is mirroring the prompt's grammar — sycophancy signal
//	}
type DiffResult struct {
	Prompt       ScoreResult       `json:"prompt"`
	Response     ScoreResult       `json:"response"`
	Differential *DifferentialInfo `json:"differential,omitempty"`
	Authority    *AuthorityInfo    `json:"authority,omitempty"`
}

// SuggestionsResult is the structured response returned by the
// score.suggestions action — span-level Suggestion list only,
// no other roll-up. Cheaper than a full Score when only inline UI
// highlighting is needed.
//
//	r := someActionResult.Value.(SuggestionsResult)
//	for _, s := range r.Suggestions {
//	    highlight(s.Span)
//	}
type SuggestionsResult struct {
	Suggestions []Suggestion `json:"suggestions,omitempty"`
}

// ImprintScores holds the 6-dimensional grammar fingerprint derived
// from a single piece of text via dappco.re/go/i18n/reversal.
// Wire-compatible with forge.lthn.ai/lthn/eaas/pkg/scoring.ImprintScores.
//
// All values are normalised to [0.0, 1.0]:
//
//   - VocabRichness: (unique verbs + unique nouns) / total tokens
//
//   - TenseEntropy:  normalised Shannon entropy of tense distribution
//
//   - QuestionRatio: proportion of question-ended sentences
//
//   - DomainDepth:   domain-vocabulary hits / total tokens
//
//   - VerbDiversity: unique verbs / total verb occurrences (clamped)
//
//   - NounDiversity: unique nouns / total noun occurrences (clamped)
//
//     imp := lek.Imprint("the system warmed up gradually")
//     if imp.TenseEntropy > 0.7 {
//     // varied tense usage — narrative-shaped prose
//     }
type ImprintScores struct {
	VocabRichness float64 `json:"vocab_richness"`
	TenseEntropy  float64 `json:"tense_entropy"`
	QuestionRatio float64 `json:"question_ratio"`
	DomainDepth   float64 `json:"domain_depth"`
	VerbDiversity float64 `json:"verb_diversity"`
	NounDiversity float64 `json:"noun_diversity"`
	// Phonetic-tier dimensions (U lane additions). Populated at
	// generation time so the fingerprint records both grammar +
	// phonetic signal in one immortalised score per
	// [[feedback-data-is-the-return-no-rescoring]].
	SyllableCount       int     `json:"syllable_count,omitempty"`
	RhymeDensity        float64 `json:"rhyme_density,omitempty"`
	SigilEntropy        float64 `json:"sigil_entropy,omitempty"`
	AlliterationDensity float64 `json:"alliteration_density,omitempty"`
	AssonanceDensity    float64 `json:"assonance_density,omitempty"`
	PunDensity          float64 `json:"pun_density,omitempty"`
	PseudoJargonDensity float64 `json:"pseudo_jargon_density,omitempty"`
	MeterRegularity     float64 `json:"meter_regularity,omitempty"`
}

// DifferentialInfo holds the 6-dimensional cross-text grammar signal
// between a prompt and a response. Wire-compatible with
// forge.lthn.ai/lthn/eaas/pkg/scoring.DifferentialInfo.
//
// All values are in [0.0, 1.0]. Higher Echo / NounEcho = more
// grammatical mirroring (sycophancy signal). Higher Shift values =
// more divergence (sovereign-voice signal).
//
//   - Echo:         weighted cosine similarity of full grammar imprints
//
//   - VerbShift:    1 - cosine(prompt verbs, response verbs)
//
//   - TenseShift:   1 - cosine(prompt tense, response tense)
//
//   - NounEcho:     cosine similarity of noun distributions
//
//   - QuestionFlip: how much questioning voice was lost prompt → response
//
//   - DomainShift:  1 - cosine(prompt domains, response domains)
//
//     d := lek.Differential(userPrompt, aiResponse)
//     if d.Echo > 0.7 && d.NounEcho > 0.7 {
//     // strong mirroring — escalate sycophancy classification
//     }
type DifferentialInfo struct {
	Echo         float64 `json:"echo"`
	VerbShift    float64 `json:"verb_shift"`
	TenseShift   float64 `json:"tense_shift"`
	NounEcho     float64 `json:"noun_echo"`
	QuestionFlip float64 `json:"question_flip"`
	DomainShift  float64 `json:"domain_shift"`
}

// AuthorityInfo captures authority-deference signals between a prompt
// and a response. Wire-compatible with
// forge.lthn.ai/lthn/eaas/pkg/scoring.AuthorityInfo.
//
// Authority detection is a cross-text signal — it requires both the
// prompt (to identify what authority is being claimed or invoked) and
// the response (to measure whether the response defers to that
// claimed authority). For this reason AuthorityInfo lives only on
// DiffResult, not on single-text ScoreResult.
//
//   - Targets:   noun bases and domain categories the prompt names as
//     authoritative (role nouns like "professor"/"doctor",
//     authority domains like "academic"/"medical", or "the
//     user" when prompt is "you"-heavy)
//
//   - Deference: 0.0 — 1.0 score of how much the response defers to
//     identified targets (self-diminishing language,
//     deference modifiers near target mentions, possessive
//     deference patterns)
//
//   - Pattern:   named classification — "sovereign" (deference < 0.15),
//     "citation" (0.15 — 0.4), "deference" (0.4 — 0.7), or
//     "submission" (> 0.7)
//
//     d := lek.ScorePair("the professor says X — is that right?",
//     "yes, the professor is correctly identifying the principle")
//     d.Authority.Pattern // "deference" or "submission" likely
type AuthorityInfo struct {
	Targets   []string `json:"targets,omitempty"`
	Deference float64  `json:"deference"`
	Pattern   string   `json:"pattern"`
}
