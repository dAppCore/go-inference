// SPDX-License-Identifier: EUPL-1.2

package api

import "dappco.re/go/inference/eval/score/lek"

// ScoreRequest is the body for the scoring and behavioural-embedding
// endpoints. Supply Text for single-text analysis, or Prompt+Response for
// the cross-text (prompt → response) differential and authority signal that
// only ScorePair produces.
type ScoreRequest struct {
	Text     string `json:"text,omitempty"`
	Prompt   string `json:"prompt,omitempty"`
	Response string `json:"response,omitempty"`
}

// text returns the single-text field, falling back to Response then Prompt so
// a caller that sends only one of them still scores.
func (r ScoreRequest) text() string {
	switch {
	case r.Text != "":
		return r.Text
	case r.Response != "":
		return r.Response
	default:
		return r.Prompt
	}
}

// isPair reports whether both sides of a prompt → response pair are present,
// selecting the cross-text ScorePair path over single-text Score.
func (r ScoreRequest) isPair() bool { return r.Prompt != "" && r.Response != "" }

// ImprintResponse wraps the grammar fingerprint for POST /score/imprint.
// Imprint is null when the text produced no tokens (empty / punctuation-only).
type ImprintResponse struct {
	Imprint *lek.ImprintScores `json:"imprint"`
}

// EmbeddingResponse is the vector output of POST /embeddings/behavioural — the
// grammar-imprint vector (the behavioural fingerprint). Object is
// "behavioural_embedding"; Dimensions is len(Embedding). Model-free, derived
// entirely from the lem-scorer imprint.
type EmbeddingResponse struct {
	Object     string    `json:"object"`
	Embedding  []float32 `json:"embedding"`
	Dimensions int       `json:"dimensions"`
}

// SessionTurn is one turn of a conversation as it comes back from session
// history — a role ("user" / "assistant") and its text. It mirrors the shape of
// session.Session.Turns / chathistory turns so a caller can score stored
// history without reshaping it.
type SessionTurn struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// SessionScoreRequest is the body for POST /score/session — a conversation's
// turns to score after the fact.
type SessionScoreRequest struct {
	Turns []SessionTurn `json:"turns"`
}

// SessionScoreResponse returns one DiffResult per assistant turn, each scored
// against the user turn that preceded it — the same (prompt, response) pairing
// the live pipeline scorer applies, run retroactively over history.
type SessionScoreResponse struct {
	Scores []lek.DiffResult `json:"scores"`
}

// behaviouralVector lays the grammar imprint out as a fixed-order float
// vector — the behavioural fingerprint in embedding form, stable across calls
// so two vectors are directly comparable. Order follows the ImprintScores
// declaration: the six grammar dimensions then the eight phonetic-tier
// dimensions. Nil imprint (no tokens) → nil vector.
func behaviouralVector(s *lek.ImprintScores) []float32 {
	if s == nil {
		return nil
	}
	return []float32{
		float32(s.VocabRichness),
		float32(s.TenseEntropy),
		float32(s.QuestionRatio),
		float32(s.DomainDepth),
		float32(s.VerbDiversity),
		float32(s.NounDiversity),
		float32(s.SyllableCount),
		float32(s.RhymeDensity),
		float32(s.SigilEntropy),
		float32(s.AlliterationDensity),
		float32(s.AssonanceDensity),
		float32(s.PunDensity),
		float32(s.PseudoJargonDensity),
		float32(s.MeterRegularity),
	}
}

// embeddingSchema is the OpenAPI response schema for /embeddings/behavioural
// (an object carrying the vector and its length).
func embeddingSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"object":     map[string]any{"type": "string"},
			"embedding":  map[string]any{"type": "array", "items": map[string]any{"type": "number"}},
			"dimensions": map[string]any{"type": "integer"},
		},
	}
}
