// SPDX-License-Identifier: EUPL-1.2

package api

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

const architectureDecisionTODO = "architectural-decision-needed: Snider-class follow-up to choose Ollama proxy, LiteLLM, in-process go-mlx, or hybrid execution"

func (p *AIProvider) embedText(c *gin.Context) {
	if c == nil {
		return
	}
	// TODO(#1015): Implement after the Snider-class architecture decision is made.
	respondNotImplemented(c, "text embedding generation")
}

func (p *AIProvider) embedBehavioural(c *gin.Context) {
	if c == nil {
		return
	}
	// TODO(#1015): Implement after the Snider-class architecture decision is made.
	respondNotImplemented(c, "behavioural embedding generation")
}

func (p *AIProvider) scoreContent(c *gin.Context) {
	if c == nil {
		return
	}
	// TODO(#1015): Implement after the Snider-class architecture decision is made.
	respondNotImplemented(c, "content scoring")
}

func (p *AIProvider) scoreImprint(c *gin.Context) {
	if c == nil {
		return
	}
	// TODO(#1015): Implement after the Snider-class architecture decision is made.
	respondNotImplemented(c, "imprint scoring")
}

func (p *AIProvider) getScore(c *gin.Context) {
	if c == nil {
		return
	}
	// TODO(#1015): Implement after the Snider-class architecture decision is made.
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

func respondNotImplemented(c *gin.Context, surface string) {
	c.JSON(http.StatusNotImplemented, gin.H{
		"error":   "not_implemented",
		"message": surface + " is not implemented yet",
		"todo":    architectureDecisionTODO,
	})
}
