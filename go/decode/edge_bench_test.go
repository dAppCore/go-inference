// SPDX-Licence-Identifier: EUPL-1.2

// Deeper-edge benchmarks for the decode harness — covers acceptance
// branches the happy-path benches in decode_bench_test.go don't reach:
// all-reject, single-accept-then-reject, candidates-shorter-than-target,
// candidates-longer-than-target, and the NormaliseMaxTokens edges
// (negative, zero, max-int, every-arg-positive).
//
// Per AX-11 — buildAcceptanceResult is the inner loop both Speculative
// and PromptLookup share; its branch shape depends on whether the
// candidate stream agrees with target. The existing 25-pct-reject bench
// covers the typical mixed path; this file covers the extremes so the
// allocator profile under fully-rejected (worst-case cloneToken count)
// and fully-accepted (best-case) is visible alongside.
//
// normaliseMaxTokens is called twice per Speculative / once per
// PromptLookup; the existing benches cover "first positive" and "falls
// through". The edge variants (negative / int-max / mixed) catch the
// rare-but-real configurations callers can pass through GenerateConfig.
//
// Run:    go test -bench='BenchmarkDecode_Edge' -benchmem -run='^$' ./go/decode

package decode

import (
	"context"
	"math"
	"testing"
)

// buildDecodeTokensAllReject mints n Tokens where every token disagrees
// with the target via a flipped sign on ID — exercises the maximum
// reject path in buildAcceptanceResult (every iteration takes the
// fallback append). This is the worst-case for cloneToken volume since
// every emitted token is a target clone rather than a candidate clone.
func buildDecodeTokensAllReject(n int) []Token {
	tokens := make([]Token, n)
	for i := 0; i < n; i++ {
		tokens[i] = Token{ID: -int32(i + 1), Text: "tok"}
	}
	return tokens
}

// buildDecodeTokensFirstAcceptThenReject mints n Tokens where token 0
// matches the target and the remainder reject — the "single hit at
// start" shape some prompt-lookup callers see (first cache-hit then
// drift). Catches branch-predictor flips between accept and reject.
func buildDecodeTokensFirstAcceptThenReject(n int) []Token {
	tokens := make([]Token, n)
	tokens[0] = Token{ID: 1, Text: "tok"}
	for i := 1; i < n; i++ {
		tokens[i] = Token{ID: -int32(i + 1), Text: "tok"}
	}
	return tokens
}

// --- buildAcceptanceResult edges (256-token shape stress-tests
// branch density without dominating the bench in append growth) ---

func BenchmarkDecode_Edge_BuildAcceptance_AllAccept_256(b *testing.B) {
	target := buildDecodeTokens(256)
	candidates := buildDecodeTokens(256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkResult = buildAcceptanceResult(ModeSpeculative, "p", target, candidates, 256)
	}
}

func BenchmarkDecode_Edge_BuildAcceptance_AllReject_256(b *testing.B) {
	target := buildDecodeTokens(256)
	candidates := buildDecodeTokensAllReject(256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkResult = buildAcceptanceResult(ModeSpeculative, "p", target, candidates, 256)
	}
}

func BenchmarkDecode_Edge_BuildAcceptance_FirstAcceptThenReject_256(b *testing.B) {
	target := buildDecodeTokens(256)
	candidates := buildDecodeTokensFirstAcceptThenReject(256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkResult = buildAcceptanceResult(ModeSpeculative, "p", target, candidates, 256)
	}
}

// CandidatesShorterThanTarget — the typical prompt-lookup miss path
// where the lookup table runs out before the target stream is exhausted
// and the loop falls through to "no candidate, append target".
func BenchmarkDecode_Edge_BuildAcceptance_CandidatesShorterThanTarget_256(b *testing.B) {
	target := buildDecodeTokens(256)
	candidates := buildDecodeTokens(64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkResult = buildAcceptanceResult(ModeSpeculative, "p", target, candidates, 256)
	}
}

// CandidatesLongerThanTarget — speculative drafts that overshoot the
// target; extra candidates are silently discarded by the limit cap.
// Exercises the limit-clamp path that bounds 'out' to len(target).
func BenchmarkDecode_Edge_BuildAcceptance_CandidatesLongerThanTarget_256(b *testing.B) {
	target := buildDecodeTokens(256)
	candidates := buildDecodeTokens(512)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkResult = buildAcceptanceResult(ModeSpeculative, "p", target, candidates, 256)
	}
}

// MaxTokensClampsTarget — emulates the case where the caller's
// MaxTokens is tighter than the target stream; out is sized to
// maxTokens and the loop short-circuits early. Validates the limit
// branch above the 'limit = len(target)' default.
func BenchmarkDecode_Edge_BuildAcceptance_MaxTokensClampsTarget_256(b *testing.B) {
	target := buildDecodeTokens(2048)
	candidates := buildDecodeTokens(2048)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkResult = buildAcceptanceResult(ModeSpeculative, "p", target, candidates, 256)
	}
}

// --- normaliseMaxTokens edges (called twice per Speculative,
// once per PromptLookup) ---

func BenchmarkDecode_Edge_NormaliseMaxTokens_Negative(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkInt = normaliseMaxTokens(-1, 0, 0)
	}
}

func BenchmarkDecode_Edge_NormaliseMaxTokens_MaxInt(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkInt = normaliseMaxTokens(math.MaxInt32, 0, 0)
	}
}

// MixedNegativesThenPositive — first two args reject, third returns.
// Exercises the loop continuation path beyond the simple "first
// positive" benchmark.
func BenchmarkDecode_Edge_NormaliseMaxTokens_MixedNegativesThenPositive(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkInt = normaliseMaxTokens(-1, -1, 128)
	}
}

// --- Speculative end-to-end under the all-reject shape — the
// scheduler-adjacent dominant cost is target-clone count, not
// candidate-clone; this is the worst-case for that. ---

func BenchmarkDecode_Edge_Speculative_AllReject_256Tokens(b *testing.B) {
	target := scriptGen(buildDecodeTokens(256))
	draft := scriptGen(buildDecodeTokensAllReject(256))
	ctx := context.Background()
	cfg := SpeculativeConfig{Prompt: "p", MaxTokens: 256, DraftTokens: 256, TargetGenerate: target, DraftGenerate: draft}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkResult, decodeSinkErr = Speculative(ctx, cfg)
	}
}

// PromptLookup_EmptyCache — the cold-start lookup case the harness
// will see during the first few tokens of a long generation, before
// the lookup table has been populated by repeated context. Candidates
// is nil so every iteration falls through to the target append.
func BenchmarkDecode_Edge_PromptLookup_EmptyCache_256Tokens(b *testing.B) {
	target := scriptGen(buildDecodeTokens(256))
	ctx := context.Background()
	cfg := PromptLookupConfig{Prompt: "p", MaxTokens: 256, TargetGenerate: target, LookupTokens: nil}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkResult, decodeSinkErr = PromptLookup(ctx, cfg)
	}
}
