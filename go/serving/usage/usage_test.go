// SPDX-Licence-Identifier: EUPL-1.2

package usage

import core "dappco.re/go"

// TestUsage_Sum_Good aggregates several turns of a request — prompt, completion,
// reasoning, and cached tokens all add up, and Total is filled from the parts
// where a turn left it zero.
func TestUsage_Sum_Good(t *core.T) {
	turns := []Usage{
		{PromptTokens: 100, CompletionTokens: 20, ReasoningTokens: 5, CachedTokens: 40},
		{PromptTokens: 50, CompletionTokens: 10, CacheWriteTokens: 50},
		{PromptTokens: 30, CompletionTokens: 5, AudioTokens: 12, ImageTokens: 3, VideoTokens: 1},
	}

	got := Sum(turns)

	core.AssertEqual(t, 180, got.PromptTokens, "prompt tokens sum across turns")
	core.AssertEqual(t, 35, got.CompletionTokens, "completion tokens sum across turns")
	core.AssertEqual(t, 5, got.ReasoningTokens, "reasoning tokens carry through")
	core.AssertEqual(t, 40, got.CachedTokens, "cached (cache-read) tokens sum")
	core.AssertEqual(t, 50, got.CacheWriteTokens, "cache-write tokens sum")
	core.AssertEqual(t, 12, got.AudioTokens, "audio tokens sum")
	core.AssertEqual(t, 3, got.ImageTokens, "image tokens sum")
	core.AssertEqual(t, 1, got.VideoTokens, "video tokens sum")
	// Each turn carried a zero Total, so Sum normalises to prompt+completion.
	core.AssertEqual(t, 215, got.TotalTokens, "total is prompt+completion when unset")
}

// TestUsage_Sum_Bad covers the empty batch — Sum of nothing is the zero Usage,
// not a panic.
func TestUsage_Sum_Bad(t *core.T) {
	core.AssertEqual(t, Usage{}, Sum(nil), "Sum(nil) is the zero usage")
	core.AssertEqual(t, Usage{}, Sum([]Usage{}), "Sum of an empty slice is the zero usage")

	// Add with a zero operand is identity (and still normalises Total).
	a := Usage{PromptTokens: 7, CompletionTokens: 3}
	core.AssertEqual(t, 10, Add(a, Usage{}).TotalTokens, "Add(a, zero) keeps a and fills Total")
}

// TestUsage_Sum_Ugly pins the normalisation rule: a caller-supplied Total is
// trusted (a provider may bill a total that exceeds prompt+completion, e.g.
// reasoning), but a zero Total is reconstructed.
func TestUsage_Sum_Ugly(t *core.T) {
	// Provider reported its own total — Normalise leaves it alone.
	reported := Usage{PromptTokens: 100, CompletionTokens: 20, TotalTokens: 130}
	reported.Normalise()
	core.AssertEqual(t, 130, reported.TotalTokens, "a non-zero Total is trusted, not overwritten")

	// Zero total → reconstructed from prompt+completion.
	bare := Usage{PromptTokens: 100, CompletionTokens: 20}
	bare.Normalise()
	core.AssertEqual(t, 120, bare.TotalTokens, "a zero Total is filled from prompt+completion")

	// Sum normalises each operand before adding, so a mix of reported and bare
	// totals aggregates correctly: 130 + 120 = 250.
	mixed := Sum([]Usage{reported, {PromptTokens: 100, CompletionTokens: 20}})
	core.AssertEqual(t, 250, mixed.TotalTokens, "mixed reported/bare totals aggregate")
}

// TestUsage_Cost_Good prices a usage record. Pricing is per-1K tokens; cached
// tokens are billed at the cheaper cache-read rate, NOT the prompt rate, so the
// prompt line only charges the uncached remainder.
func TestUsage_Cost_Good(t *core.T) {
	u := Usage{
		PromptTokens:     1000, // includes the 400 cached below
		CompletionTokens: 500,
		CachedTokens:     400,
		CacheWriteTokens: 200,
		ReasoningTokens:  100,
	}
	p := Pricing{
		PromptPer1K:     1.00, // £1.00 / 1K prompt tokens
		CompletionPer1K: 2.00,
		CacheReadPer1K:  0.10,
		CacheWritePer1K: 1.25,
	}

	// prompt: (1000-400)/1000 * 1.00 = 0.60
	// cache-read: 400/1000 * 0.10   = 0.04
	// cache-write: 200/1000 * 1.25  = 0.25
	// completion: 500/1000 * 2.00   = 1.00
	// reasoning billed at completion rate: 100/1000 * 2.00 = 0.20
	// total = 2.09
	core.AssertInDelta(t, 2.09, Cost(u, p), 1e-9, "cost sums each token class at its own rate")
}

// TestUsage_Cost_Bad: zero pricing yields zero cost regardless of token counts,
// and zero usage costs nothing — zero-completion insurance (RFC §6.11) means an
// empty generation is free.
func TestUsage_Cost_Bad(t *core.T) {
	loaded := Usage{PromptTokens: 9999, CompletionTokens: 9999, CachedTokens: 5000}
	core.AssertInDelta(t, 0.0, Cost(loaded, Pricing{}), 1e-9, "zero pricing → zero cost")
	core.AssertInDelta(t, 0.0, Cost(Usage{}, Pricing{PromptPer1K: 99}), 1e-9, "zero usage → zero cost")
}

// TestUsage_Cost_Ugly covers BYOK: the platform charges nothing for a
// bring-your-own-key request (the caller paid the provider directly), but the
// upstream cost the caller bore is still reported. And cached tokens that
// exceed the prompt count must not drive the uncached prompt charge negative.
func TestUsage_Cost_Ugly(t *core.T) {
	p := Pricing{
		PromptPer1K:     1.00,
		CompletionPer1K: 2.00,
		CacheReadPer1K:  0.10,
		BYOK:            true,
		UpstreamCost:    0.42, // what the caller's own key was billed upstream
	}
	u := Usage{PromptTokens: 1000, CompletionTokens: 500}

	// BYOK → the platform's billable cost is zero; the upstream figure is the
	// accounted cost instead.
	core.AssertInDelta(t, 0.0, Cost(u, p), 1e-9, "BYOK platform cost is zero")
	core.AssertInDelta(t, 0.42, p.AccountedCost(u), 1e-9, "BYOK accounted cost is the upstream figure")

	// Non-BYOK AccountedCost is just the computed platform cost.
	pPlatform := Pricing{PromptPer1K: 1.00, CompletionPer1K: 2.00}
	core.AssertInDelta(t, 2.00, pPlatform.AccountedCost(u), 1e-9, "platform accounted cost equals Cost")

	// Cached tokens larger than the prompt count clamp the uncached prompt to
	// zero rather than charging a negative amount.
	odd := Usage{PromptTokens: 100, CachedTokens: 9999, CompletionTokens: 0}
	costClamped := Cost(odd, Pricing{PromptPer1K: 1.00, CacheReadPer1K: 0.10})
	// prompt uncached clamps to 0; cache-read still bills the reported cached
	// count: 9999/1000 * 0.10 = 0.9999
	core.AssertInDelta(t, 0.9999, costClamped, 1e-9, "uncached prompt clamps at zero, cache-read still bills")
}
