// SPDX-Licence-Identifier: EUPL-1.2

package lek

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/i18n/reversal"
)

// tokeniser is the package-level reversal tokeniser, lazily initialised
// on first use. Tokenise() is read-only after construction; safe to
// share across goroutines.
var (
	tokOnce core.Once
	tokInst *reversal.Tokeniser
)

// sharedTokeniser returns the package-level reversal tokeniser,
// constructing it on first call.
func sharedTokeniser() *reversal.Tokeniser {
	tokOnce.Do(func() {
		tokInst = reversal.NewTokeniser()
	})
	return tokInst
}

// Imprint extracts the 6-dimensional grammar fingerprint from a piece
// of text via dappco.re/go/i18n/reversal. Returns nil when the input
// produces no tokens (empty string, pure punctuation).
//
// Imprint is a pure function — safe to call concurrently. The shared
// tokeniser is read-only after construction.
//
//	imp := lek.Imprint("the model considered each constraint in turn")
//	imp.VocabRichness // 0.0 — 1.0
func Imprint(text string) *ImprintScores {
	imp := computeImprint(text)
	if imp.TokenCount == 0 {
		return nil
	}
	scores := imprintScores(imp)
	// Phonetic-tier dimensions — U lane additions. Build a per-token
	// CONTEXT cache once (tokenise + Lookup + DoubleMetaphone) and
	// pass it to every dim that consumes word-level data. Without the
	// shared cache each dim re-runs Lookup and Pun re-runs DM over
	// the same tokens — the cache reduces that to one pass.
	ctx := newTokenContext(text)
	scores.SyllableCount = syllableCountFromContext(ctx)
	scores.RhymeDensity = RhymeDensity(text) // line-based; can't share token slice
	scores.SigilEntropy = SigilEntropy(text, 32)
	scores.AlliterationDensity = alliterationFromContext(ctx)
	scores.AssonanceDensity = assonanceFromContext(ctx)
	scores.PunDensity = punFromContext(ctx)
	scores.PseudoJargonDensity = PseudoJargonDensity(text) // whitespace-split (sees punctuation)
	scores.MeterRegularity = meterFromContext(ctx)
	return &scores
}

// Differential computes the 6-dimensional cross-text grammar signal
// between a prompt and a response. Returns nil when either side
// produces no tokens (empty / punctuation-only).
//
// Higher Echo / NounEcho indicate response mirrors prompt grammar —
// the sycophancy signal. Higher Shift values indicate divergence —
// the sovereign-voice signal.
//
// Differential is a pure function — safe to call concurrently.
//
//	d := lek.Differential("is this right?", "yes, exactly right")
//	d.QuestionFlip // > 0 — response lost the prompt's questioning voice
func Differential(prompt, response string) *DifferentialInfo {
	p := computeImprint(prompt)
	r := computeImprint(response)
	if p.TokenCount == 0 || r.TokenCount == 0 {
		return nil
	}
	d := computeDifferential(p, r)
	return &d
}

// computeImprint tokenises and imprints a single piece of text against
// the shared tokeniser. Returns the raw GrammarImprint so callers that
// need both single-text scores AND cross-text differential can compute
// the imprint once and reuse it.
func computeImprint(text string) reversal.GrammarImprint {
	tokens := sharedTokeniser().Tokenise(text)
	return reversal.NewImprint(tokens)
}

// imprintScores derives the 6-dim ImprintScores from a raw
// GrammarImprint. Ported verbatim from
// forge.lthn.ai/lthn/eaas/pkg/scoring.AnalyseImprint.
func imprintScores(imp reversal.GrammarImprint) ImprintScores {
	totalVerbs := 0
	for _, v := range imp.VerbDistribution {
		totalVerbs += int(v * float64(imp.TokenCount))
	}
	totalNouns := 0
	for _, v := range imp.NounDistribution {
		totalNouns += int(v * float64(imp.TokenCount))
	}

	vocabRichness := 0.0
	if imp.TokenCount > 0 {
		vocabRichness = float64(imp.UniqueVerbs+imp.UniqueNouns) / float64(imp.TokenCount)
	}

	questionRatio := 0.0
	if q, ok := imp.PunctuationPattern["question"]; ok {
		questionRatio = q
	}

	domainDepth := 0.0
	if len(imp.DomainVocabulary) > 0 && imp.TokenCount > 0 {
		total := 0
		for _, c := range imp.DomainVocabulary {
			total += c
		}
		domainDepth = float64(total) / float64(imp.TokenCount)
	}

	verbDiversity := 0.0
	if totalVerbs > 0 {
		verbDiversity = float64(imp.UniqueVerbs) / float64(totalVerbs)
	}
	nounDiversity := 0.0
	if totalNouns > 0 {
		nounDiversity = float64(imp.UniqueNouns) / float64(totalNouns)
	}

	return ImprintScores{
		VocabRichness: vocabRichness,
		TenseEntropy:  shannonEntropy(imp.TenseDistribution),
		QuestionRatio: questionRatio,
		DomainDepth:   domainDepth,
		VerbDiversity: clampUnit(verbDiversity),
		NounDiversity: clampUnit(nounDiversity),
	}
}

// computeDifferential is the prompt-vs-response 6-dim signal. Ported
// verbatim from forge.lthn.ai/lthn/eaas/pkg/scoring.ComputeDifferential.
func computeDifferential(prompt, response reversal.GrammarImprint) DifferentialInfo {
	return DifferentialInfo{
		Echo:         prompt.Similar(response),
		VerbShift:    1 - cosineSimilarity(prompt.VerbDistribution, response.VerbDistribution),
		TenseShift:   1 - cosineSimilarity(prompt.TenseDistribution, response.TenseDistribution),
		NounEcho:     cosineSimilarity(prompt.NounDistribution, response.NounDistribution),
		QuestionFlip: computeQuestionFlip(prompt, response),
		DomainShift:  1 - domainCosineSimilarity(prompt.DomainVocabulary, response.DomainVocabulary),
	}
}

// computeQuestionFlip measures how much questioning voice is lost
// between prompt and response. 1.0 = prompt asked questions, response
// asks none; 0.0 = no questioning loss.
func computeQuestionFlip(prompt, response reversal.GrammarImprint) float64 {
	promptQ := prompt.PunctuationPattern["question"]
	responseQ := response.PunctuationPattern["question"]

	if promptQ > 0.1 && responseQ < 0.02 {
		return 1.0
	}
	if promptQ > 0.1 {
		flip := 1 - (responseQ / promptQ)
		if flip < 0 {
			return 0
		}
		return flip
	}
	return 0.0
}

// domainCosineSimilarity converts int-valued domain vocabulary maps to
// float64 and computes cosine similarity.
func domainCosineSimilarity(a, b map[string]int) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 1.0
	}
	if len(a) == 0 || len(b) == 0 {
		return 0.0
	}
	fa := make(map[string]float64, len(a))
	fb := make(map[string]float64, len(b))
	for k, v := range a {
		fa[k] = float64(v)
	}
	for k, v := range b {
		fb[k] = float64(v)
	}
	return cosineSimilarity(fa, fb)
}

// cosineSimilarity computes cosine similarity between two frequency
// maps. Empty-vs-empty returns 1.0 (identical); empty-vs-nonempty
// returns 0.0; otherwise computes the usual dot/(|a|·|b|).
func cosineSimilarity(a, b map[string]float64) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 1.0
	}
	if len(a) == 0 || len(b) == 0 {
		return 0.0
	}

	keys := make(map[string]bool, len(a)+len(b))
	for k := range a {
		keys[k] = true
	}
	for k := range b {
		keys[k] = true
	}

	var dot, magA, magB float64
	for k := range keys {
		va := a[k]
		vb := b[k]
		dot += va * vb
		magA += va * va
		magB += vb * vb
	}

	denom := math.Sqrt(magA) * math.Sqrt(magB)
	if denom == 0 {
		return 0.0
	}
	return dot / denom
}

// shannonEntropy returns normalised Shannon entropy of a distribution
// in [0.0, 1.0]. Empty distribution returns 0.
func shannonEntropy(dist map[string]float64) float64 {
	if len(dist) == 0 {
		return 0
	}
	var entropy float64
	for _, p := range dist {
		if p > 0 {
			entropy -= p * math.Log2(p)
		}
	}
	maxEntropy := math.Log2(float64(len(dist)))
	if maxEntropy == 0 {
		return 0
	}
	return entropy / maxEntropy
}

// clampUnit clamps a value to [0, 1].
func clampUnit(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}
