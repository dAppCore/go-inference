// SPDX-Licence-Identifier: EUPL-1.2

// Package usage is the usage & cost accounting (RFC §6.6). It turns the
// token counts a response carries into a billable cost, and aggregates usage
// across the turns of a request or the items of a batch (§6.3).
//
// It is pure and deterministic — no clock, no I/O, no provider calls — so the
// serving path can account a response and the metrics logger (§3.2) can record
// it without either depending on the other.
//
//	u := usage.Add(turn1, turn2)              // aggregate two turns
//	u.Normalise()                             // fill Total if a turn left it 0
//	cost := usage.Cost(u, modelPricing)       // billable platform cost
package usage

// Usage is the token accounting for one response, one request (summed across
// its turns), or one batch item. Counts are whole tokens. Cached tokens are a
// SUBSET of the prompt — the slice of the prompt served from cache at the
// cache-read rate — so a cost calc bills (prompt - cached) at the prompt rate
// and `cached` at the cheaper cache-read rate (§6.11).
//
//	u := usage.Usage{PromptTokens: 1200, CompletionTokens: 300, CachedTokens: 800}
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`

	// ReasoningTokens are completion-side tokens a reasoning model spent
	// thinking; billed at the completion rate.
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`

	// CachedTokens is the portion of PromptTokens served from a prompt/KV cache
	// (cache READ); CacheWriteTokens is the portion written INTO the cache.
	CachedTokens     int `json:"cached_tokens,omitempty"`
	CacheWriteTokens int `json:"cache_write_tokens,omitempty"`

	// Multimodal token counts, where a backend reports them separately (§6.12).
	AudioTokens int `json:"audio_tokens,omitempty"`
	ImageTokens int `json:"image_tokens,omitempty"`
	VideoTokens int `json:"video_tokens,omitempty"`
}

// Normalise fills TotalTokens from PromptTokens+CompletionTokens when a turn
// reported it as zero. A non-zero Total is trusted as-is — a provider may bill a
// total above prompt+completion (reasoning, tool overhead), and we don't
// second-guess it.
//
//	u := usage.Usage{PromptTokens: 100, CompletionTokens: 20}
//	u.Normalise() // u.TotalTokens == 120
func (u *Usage) Normalise() {
	if u.TotalTokens == 0 {
		u.TotalTokens = u.PromptTokens + u.CompletionTokens
	}
}

// Add aggregates two usage records field-by-field, normalising each operand
// first so a zero Total never drags the sum's total down. Add(a, zero) is a
// (normalised) — identity with Total filled.
//
//	combined := usage.Add(promptStage, completionStage)
func Add(a, b Usage) Usage {
	a.Normalise()
	b.Normalise()
	return Usage{
		PromptTokens:     a.PromptTokens + b.PromptTokens,
		CompletionTokens: a.CompletionTokens + b.CompletionTokens,
		TotalTokens:      a.TotalTokens + b.TotalTokens,
		ReasoningTokens:  a.ReasoningTokens + b.ReasoningTokens,
		CachedTokens:     a.CachedTokens + b.CachedTokens,
		CacheWriteTokens: a.CacheWriteTokens + b.CacheWriteTokens,
		AudioTokens:      a.AudioTokens + b.AudioTokens,
		ImageTokens:      a.ImageTokens + b.ImageTokens,
		VideoTokens:      a.VideoTokens + b.VideoTokens,
	}
}

// Sum aggregates a batch of usage records into one. Sum(nil) and Sum of an
// empty slice are the zero Usage. Each item is normalised as it folds in, so a
// batch of turns that each left Total unset still totals correctly.
//
//	batchUsage := usage.Sum(perItemUsage)
func Sum(usages []Usage) Usage {
	var total Usage
	for _, u := range usages {
		total = Add(total, u)
	}
	return total
}
