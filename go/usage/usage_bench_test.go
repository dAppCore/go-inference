// SPDX-Licence-Identifier: EUPL-1.2

package usage_test

import (
	"testing"

	"dappco.re/go/inference/usage"
)

// Package sinks defeat dead-code elimination so the benchmarked work survives.
var (
	sinkUsage usage.Usage
	sinkCost  float64
)

// benchTurns is a realistic multi-turn request as the serving path accumulates
// it: a long initial prompt with a cache hit + write, then several shorter
// tool-use turns, one reasoning, a multimodal one. Some turns leave TotalTokens
// unset so Sum's per-operand Normalise runs; one carries its own Total so the
// trusted-total branch runs too.
var benchTurns = []usage.Usage{
	{PromptTokens: 1800, CompletionTokens: 120, CachedTokens: 1200, CacheWriteTokens: 600},
	{PromptTokens: 320, CompletionTokens: 64, CachedTokens: 256},
	{PromptTokens: 410, CompletionTokens: 88, ReasoningTokens: 40},
	{PromptTokens: 290, CompletionTokens: 52, CachedTokens: 200},
	{PromptTokens: 360, CompletionTokens: 70, AudioTokens: 24},
	{PromptTokens: 275, CompletionTokens: 47, ImageTokens: 12, VideoTokens: 2},
	{PromptTokens: 305, CompletionTokens: 61, TotalTokens: 366},
	{PromptTokens: 240, CompletionTokens: 39},
}

// benchPricing is a representative remote model price sheet (per 1K tokens), so
// every line of Cost (prompt, cache-read, cache-write, completion, reasoning)
// charges a non-zero rate.
var benchPricing = usage.Pricing{
	PromptPer1K:     3.00,
	CompletionPer1K: 15.00,
	CacheReadPer1K:  0.30,
	CacheWritePer1K: 3.75,
}

// benchUsageOne is one request's aggregated usage (every token class populated)
// — the input Cost prices per request.
var benchUsageOne = usage.Usage{
	PromptTokens:     4000,
	CompletionTokens: 541,
	ReasoningTokens:  40,
	CachedTokens:     1856,
	CacheWriteTokens: 600,
	TotalTokens:      4581,
}

// BenchmarkAdd folds two turns field-by-field — a value return, alloc-free.
func BenchmarkAdd(b *testing.B) {
	a, c := benchTurns[0], benchTurns[1]
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkUsage = usage.Add(a, c)
	}
}

// BenchmarkSum aggregates a full multi-turn request (per request): fold each
// turn, normalising as it goes. A value accumulator over a slice — alloc-free.
func BenchmarkSum(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkUsage = usage.Sum(benchTurns)
	}
}

// BenchmarkNormalise is the pointer-receiver Total fill: a fresh struct each
// iteration so the reconstruction branch runs every time (not the no-op a
// pre-filled Total would give). Pure arithmetic on a stack value.
func BenchmarkNormalise(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		u := usage.Usage{PromptTokens: 1000, CompletionTokens: 200}
		u.Normalise()
		sinkUsage = u
	}
}

// BenchmarkCost prices one request's aggregated usage (per request): five
// per-1K rate lines summed — scalar float arithmetic, alloc-free.
func BenchmarkCost(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkCost = usage.Cost(benchUsageOne, benchPricing)
	}
}

// BenchmarkAccountedCost_Platform is the recorded figure for a normal request:
// the BYOK guard falls through to Cost.
func BenchmarkAccountedCost_Platform(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkCost = benchPricing.AccountedCost(benchUsageOne)
	}
}

// BenchmarkAccountedCost_BYOK is the BYOK branch: the platform bills nothing and
// the upstream figure is returned directly (no Cost call).
func BenchmarkAccountedCost_BYOK(b *testing.B) {
	p := usage.Pricing{BYOK: true, UpstreamCost: 0.42}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkCost = p.AccountedCost(benchUsageOne)
	}
}
