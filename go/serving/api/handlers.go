// SPDX-License-Identifier: EUPL-1.2

package api

import (
	"net/http"

	"dappco.re/go/inference"
	"dappco.re/go/inference/eval/score/lek"
	"github.com/gin-gonic/gin"
)

// architectureDecisionTODO flags the one surface still awaiting a design call:
// where scored results are persisted (the getScore retrieval path).
const architectureDecisionTODO = "architectural-decision-needed: score-persistence backend (go-store KV) not yet selected"

// embedText returns a neural text-embedding vector from the injected
// embedding model. Reports 503 when the provider was built without one
// (WithEmbedder), keeping the model-free scoring endpoints usable regardless.
func (p *AIProvider) embedText(c *gin.Context) {
	if c == nil {
		return
	}
	var req TextEmbeddingRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		respondError(c, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	text := req.text()
	if text == "" {
		respondError(c, http.StatusBadRequest, "invalid_request", "text is required")
		return
	}
	if p == nil || p.embedder == nil {
		respondError(c, http.StatusServiceUnavailable, "no_embedding_model", "no embedding model is configured for this provider")
		return
	}
	result, err := p.embedder.Embed(c.Request.Context(), inference.EmbeddingRequest{
		Model:     req.Model,
		Input:     []string{text},
		Normalize: true,
	})
	if err != nil {
		respondError(c, http.StatusInternalServerError, "embedding_failed", err.Error())
		return
	}
	var vec []float32
	if result != nil && len(result.Vectors) > 0 {
		vec = result.Vectors[0]
	}
	c.JSON(http.StatusOK, EmbeddingResponse{
		Object:     "embedding",
		Embedding:  vec,
		Dimensions: len(vec),
		Model:      req.Model,
	})
}

// embedBehavioural returns the grammar imprint as a behavioural fingerprint
// vector. The imprint IS the behavioural embedding — a fixed-order projection
// of the lem-scorer grammar+phonetic dimensions. Needs no model.
func (p *AIProvider) embedBehavioural(c *gin.Context) {
	if c == nil {
		return
	}
	var req ScoreRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		respondError(c, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	text := req.text()
	if text == "" {
		respondError(c, http.StatusBadRequest, "invalid_request", "text is required")
		return
	}
	vec := behaviouralVector(lek.Imprint(text))
	c.JSON(http.StatusOK, EmbeddingResponse{
		Object:     "behavioural_embedding",
		Embedding:  vec,
		Dimensions: len(vec),
	})
}

// scoreContent runs the in-process lem-scorer. A single text yields a
// ScoreResult (sycophancy, LEK, hostility, imprint); a prompt+response pair
// yields a DiffResult that adds the cross-text differential and authority
// signal. Needs no model.
func (p *AIProvider) scoreContent(c *gin.Context) {
	if c == nil {
		return
	}
	var req ScoreRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		respondError(c, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	if req.isPair() {
		c.JSON(http.StatusOK, lek.ScorePair(req.Prompt, req.Response))
		return
	}
	text := req.text()
	if text == "" {
		respondError(c, http.StatusBadRequest, "invalid_request", "text or prompt+response is required")
		return
	}
	c.JSON(http.StatusOK, lek.Score(text))
}

// scoreImprint returns the grammar+phonetic imprint of the text (null imprint
// when the text produces no tokens). Needs no model.
func (p *AIProvider) scoreImprint(c *gin.Context) {
	if c == nil {
		return
	}
	var req ScoreRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		respondError(c, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	text := req.text()
	if text == "" {
		respondError(c, http.StatusBadRequest, "invalid_request", "text is required")
		return
	}
	c.JSON(http.StatusOK, ImprintResponse{Imprint: lek.Imprint(text)})
}

// scoreSession runs the scorer after the fact over a conversation's turns from
// session history: each assistant turn is scored against the user turn that
// preceded it — the same lek.ScorePair pairing the live pipeline applies, so a
// turn scored live and re-scored here is identical. Stateless: the caller
// supplies the turns it loaded from history, so this needs no store binding.
func (p *AIProvider) scoreSession(c *gin.Context) {
	if c == nil {
		return
	}
	var req SessionScoreRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		respondError(c, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	if len(req.Turns) == 0 {
		respondError(c, http.StatusBadRequest, "invalid_request", "turns is required")
		return
	}
	scores := make([]lek.DiffResult, 0, len(req.Turns))
	lastUser := ""
	for _, turn := range req.Turns {
		switch turn.Role {
		case "user":
			lastUser = turn.Content
		case "assistant":
			scores = append(scores, lek.ScorePair(lastUser, turn.Content))
		}
	}
	c.JSON(http.StatusOK, SessionScoreResponse{Scores: scores})
}

// getScore retrieves a stored score result. Still awaiting the persistence
// backend decision — see architectureDecisionTODO.
func (p *AIProvider) getScore(c *gin.Context) {
	if c == nil {
		return
	}
	respondNotImplemented(c, "score retrieval")
}

func (p *AIProvider) health(c *gin.Context) {
	if c == nil {
		return
	}
	c.JSON(http.StatusOK, gin.H{
		"ok":       true,
		"provider": "ai",
		"status":   "healthy",
	})
}

// respondError writes the provider's raw error envelope (matching the raw-body
// convention of health and respondNotImplemented).
func respondError(c *gin.Context, status int, code, message string) {
	c.JSON(status, gin.H{
		"error":   code,
		"message": message,
	})
}

func respondNotImplemented(c *gin.Context, surface string) {
	c.JSON(http.StatusNotImplemented, gin.H{
		"error":   "not_implemented",
		"message": surface + " is not implemented yet",
		"todo":    architectureDecisionTODO,
	})
}
