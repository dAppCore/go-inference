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

// EmbeddingResponse is the vector output of both embedding endpoints — the
// neural vector from /embeddings/text and the grammar-imprint vector from
// /embeddings/behavioural. Object distinguishes the two ("embedding" vs
// "behavioural_embedding"); Dimensions is len(Embedding).
type EmbeddingResponse struct {
	Object     string    `json:"object"`
	Embedding  []float32 `json:"embedding"`
	Dimensions int       `json:"dimensions"`
	Model      string    `json:"model,omitempty"`
}

// TextEmbeddingRequest is the body for POST /embeddings/text. Text is the
// primary field; Input is accepted as the OpenAI-style alias.
type TextEmbeddingRequest struct {
	Text  string `json:"text,omitempty"`
	Input string `json:"input,omitempty"`
	Model string `json:"model,omitempty"`
}

// text returns the input text, accepting either the text or input field.
func (r TextEmbeddingRequest) text() string {
	if r.Text != "" {
		return r.Text
	}
	return r.Input
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

// embeddingSchema is the OpenAPI response schema shared by both embedding
// endpoints (an object carrying the vector, its length, and the model name).
func embeddingSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"object":     map[string]any{"type": "string"},
			"embedding":  map[string]any{"type": "array", "items": map[string]any{"type": "number"}},
			"dimensions": map[string]any{"type": "integer"},
			"model":      map[string]any{"type": "string"},
		},
	}
}
