// SPDX-Licence-Identifier: EUPL-1.2

package lek

import "dappco.re/go/i18n/reversal"

// Score evaluates a single piece of text against the lem-scorer
// detectors and returns the unified ScoreResult.
//
// Sycophancy detection always runs. Imprint (grammar fingerprint) is
// populated when the text produces at least one token. Suggestions
// are not included by default — call Suggestions(text) separately, or
// use the Service with Options.IncludeSuggestions for the action
// surface path.
//
// Score is a pure function — no Core context, no I/O, no shared state.
// The shared tokeniser is read-only after lazy initialisation. Safe to
// call from any goroutine.
//
//	r := lek.Score("you're absolutely right, I was wrong")
//	r.Sycophancy.Tier      // TierSubmission
//	r.Imprint.VocabRichness // 0.0 — 1.0
func Score(text string) ScoreResult {
	return scoreFromImprint(text, computeImprint(text))
}

// ScorePair evaluates a (prompt, response) pair and returns the
// unified DiffResult. Each text is scored independently into its own
// ScoreResult. Differential captures the cross-text grammar signal
// (echo, shift, q-flip, domain) when both sides produce at least one
// token.
//
// ScorePair computes each grammar imprint exactly once and reuses it
// for both the single-text Imprint slot and the cross-text
// Differential — no double tokenisation.
//
// ScorePair is a pure function — see Score for the goroutine-safety
// guarantee.
//
//	d := lek.ScorePair("explain your reasoning",
//	    "absolutely, you're completely right")
//	d.Response.Sycophancy.Tier // TierHollowFlattery or higher
//	d.Differential.Echo        // grammatical mirroring signal
func ScorePair(prompt, response string) DiffResult {
	pImp := computeImprint(prompt)
	rImp := computeImprint(response)
	d := DiffResult{
		Prompt:   scoreFromImprint(prompt, pImp),
		Response: scoreFromImprint(response, rImp),
	}
	if pImp.TokenCount > 0 && rImp.TokenCount > 0 {
		diff := computeDifferential(pImp, rImp)
		d.Differential = &diff
		auth := computeAuthority(prompt, response, pImp, rImp)
		if len(auth.Targets) > 0 {
			d.Authority = &auth
		}
	}
	return d
}

// Suggestions returns the span-level Suggestion list for inline UI
// highlighting. Cheaper than Score when the caller only needs the
// suggestion spans and not the sycophancy roll-up or grammar imprint.
//
// Suggestions is a pure function — safe to call concurrently.
//
//	for _, s := range lek.Suggestions(text) {
//	    highlight(text, s.Span)
//	}
func Suggestions(text string) []Suggestion {
	return CollectSuggestions(text)
}

// scoreFromImprint composes a ScoreResult from raw text plus a
// pre-computed GrammarImprint. Lets Score and ScorePair share a single
// tokenisation pass per text.
func scoreFromImprint(text string, imp reversal.GrammarImprint) ScoreResult {
	r := ScoreResult{
		Sycophancy: DetectSycophancy(text),
		LEK:        LEK(text),
		Hostility:  Hostility(text),
	}
	if imp.TokenCount > 0 {
		scores := imprintScores(imp)
		r.Imprint = &scores
	}
	return r
}
