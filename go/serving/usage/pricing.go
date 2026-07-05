// SPDX-Licence-Identifier: EUPL-1.2

package usage

// Pricing is the per-token price sheet for one model/endpoint, expressed
// **per 1,000 tokens** (the unit OpenAI/OpenRouter price sheets use). A rate of
// 0 means that token class is free for this endpoint — local on-device models
// price at all zeros and therefore cost nothing (§6.2, local-first).
//
//	p := usage.Pricing{PromptPer1K: 1.00, CompletionPer1K: 2.00, CacheReadPer1K: 0.10}
//	cost := usage.Cost(turnUsage, p)
type Pricing struct {
	PromptPer1K     float64 `json:"prompt_per_1k"`
	CompletionPer1K float64 `json:"completion_per_1k"`

	// CacheReadPer1K prices CachedTokens (prompt tokens served from cache);
	// CacheWritePer1K prices CacheWriteTokens (tokens written into the cache).
	CacheReadPer1K  float64 `json:"cache_read_per_1k"`
	CacheWritePer1K float64 `json:"cache_write_per_1k"`

	// BYOK marks a bring-your-own-key request: the caller paid the provider
	// directly with their own key, so the platform bills nothing (Cost == 0) and
	// UpstreamCost records what the caller's key was charged upstream (§6.17).
	BYOK         bool    `json:"is_byok,omitempty"`
	UpstreamCost float64 `json:"upstream_cost,omitempty"`
}

// perK applies a per-1K rate to a token count.
func perK(tokens int, ratePer1K float64) float64 {
	return float64(tokens) / 1000.0 * ratePer1K
}

// Cost computes the platform-billable cost of u under p. Cached tokens are a
// subset of the prompt billed at the cheaper cache-read rate, so the prompt line
// charges only the uncached remainder (clamped at zero — cached can't exceed the
// real prompt, but a mis-reported count must never bill negative). Reasoning
// tokens bill at the completion rate. A BYOK request costs the platform nothing
// — see AccountedCost for the figure that actually gets recorded.
//
//	cost := usage.Cost(response.Usage, modelPricing)
func Cost(u Usage, p Pricing) float64 {
	if p.BYOK {
		return 0
	}

	uncachedPrompt := u.PromptTokens - u.CachedTokens
	if uncachedPrompt < 0 {
		uncachedPrompt = 0
	}

	return perK(uncachedPrompt, p.PromptPer1K) +
		perK(u.CachedTokens, p.CacheReadPer1K) +
		perK(u.CacheWriteTokens, p.CacheWritePer1K) +
		perK(u.CompletionTokens, p.CompletionPer1K) +
		perK(u.ReasoningTokens, p.CompletionPer1K)
}

// AccountedCost is the figure the metrics log records (§3.2): the upstream cost
// the caller bore for a BYOK request, or the computed platform cost otherwise.
// This is the single number a generation lookup (§6.6) returns for a past
// request.
//
//	recorded := modelPricing.AccountedCost(response.Usage)
func (p Pricing) AccountedCost(u Usage) float64 {
	if p.BYOK {
		return p.UpstreamCost
	}
	return Cost(u, p)
}
