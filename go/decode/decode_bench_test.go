// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the driver-neutral decode-optimisation harness —
// Speculative + PromptLookup over synthetic generators, plus the
// per-token equality, render, and clone primitives.
//
// Per AX-11 — Speculative + PromptLookup fire once per decode bench
// run, but the inner buildAcceptanceResult loop calls TokenEqual +
// cloneToken per emitted token, and TokensText concatenates the whole
// stream. The longest streams the harness sees today are 2048 tokens.
//
// Run:    go test -bench='BenchmarkDecode' -benchmem -run='^$' ./go/decode

package decode

import (
	"context"
	"testing"
	"time"
)

// Sinks defeat compiler DCE.
var (
	decodeSinkResult Result
	decodeSinkErr    error
	decodeSinkText   string
	decodeSinkTokens []Token
	decodeSinkBool   bool
	decodeSinkInt    int
	decodeSinkDur    time.Duration
)

// buildDecodeTokens mints n Tokens with a representative ID + Text
// shape (no Value — drivers populate one or the other, not both,
// in the typical hot path).
func buildDecodeTokens(n int) []Token {
	tokens := make([]Token, n)
	for i := 0; i < n; i++ {
		tokens[i] = Token{ID: int32(i + 1), Text: "tok"}
	}
	return tokens
}

// buildDecodeTokensSkewed mints n Tokens where every 4th token
// disagrees with the target — exercises the reject branch in
// buildAcceptanceResult.
func buildDecodeTokensSkewed(n int) []Token {
	tokens := make([]Token, n)
	for i := 0; i < n; i++ {
		id := int32(i + 1)
		if i%4 == 3 {
			id = -id
		}
		tokens[i] = Token{ID: id, Text: "tok"}
	}
	return tokens
}

// scriptGen wraps a fixed token stream in a GenerateFunc.
func scriptGen(tokens []Token) GenerateFunc {
	return func(context.Context, string, GenerateConfig) (Generation, error) {
		return Generation{Tokens: tokens}, nil
	}
}

// --- Speculative + PromptLookup end-to-end ---

func BenchmarkDecode_Speculative_32Tokens(b *testing.B) {
	target := scriptGen(buildDecodeTokens(32))
	draft := scriptGen(buildDecodeTokens(32))
	ctx := context.Background()
	cfg := SpeculativeConfig{Prompt: "p", MaxTokens: 32, DraftTokens: 32, TargetGenerate: target, DraftGenerate: draft}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkResult, decodeSinkErr = Speculative(ctx, cfg)
	}
}

func BenchmarkDecode_Speculative_256Tokens(b *testing.B) {
	target := scriptGen(buildDecodeTokens(256))
	draft := scriptGen(buildDecodeTokens(256))
	ctx := context.Background()
	cfg := SpeculativeConfig{Prompt: "p", MaxTokens: 256, DraftTokens: 256, TargetGenerate: target, DraftGenerate: draft}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkResult, decodeSinkErr = Speculative(ctx, cfg)
	}
}

func BenchmarkDecode_Speculative_2048Tokens(b *testing.B) {
	target := scriptGen(buildDecodeTokens(2048))
	draft := scriptGen(buildDecodeTokens(2048))
	ctx := context.Background()
	cfg := SpeculativeConfig{Prompt: "p", MaxTokens: 2048, DraftTokens: 2048, TargetGenerate: target, DraftGenerate: draft}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkResult, decodeSinkErr = Speculative(ctx, cfg)
	}
}

// Skewed exercises the reject path inside buildAcceptanceResult — every
// 4th draft token mismatches, forcing a fallback append.
func BenchmarkDecode_Speculative_256Tokens_25PctReject(b *testing.B) {
	target := scriptGen(buildDecodeTokens(256))
	draft := scriptGen(buildDecodeTokensSkewed(256))
	ctx := context.Background()
	cfg := SpeculativeConfig{Prompt: "p", MaxTokens: 256, DraftTokens: 256, TargetGenerate: target, DraftGenerate: draft}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkResult, decodeSinkErr = Speculative(ctx, cfg)
	}
}

func BenchmarkDecode_PromptLookup_32Tokens(b *testing.B) {
	target := scriptGen(buildDecodeTokens(32))
	ctx := context.Background()
	cfg := PromptLookupConfig{Prompt: "p", MaxTokens: 32, TargetGenerate: target, LookupTokens: buildDecodeTokens(32)}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkResult, decodeSinkErr = PromptLookup(ctx, cfg)
	}
}

func BenchmarkDecode_PromptLookup_256Tokens(b *testing.B) {
	target := scriptGen(buildDecodeTokens(256))
	ctx := context.Background()
	cfg := PromptLookupConfig{Prompt: "p", MaxTokens: 256, TargetGenerate: target, LookupTokens: buildDecodeTokens(256)}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkResult, decodeSinkErr = PromptLookup(ctx, cfg)
	}
}

func BenchmarkDecode_PromptLookup_2048Tokens(b *testing.B) {
	target := scriptGen(buildDecodeTokens(2048))
	ctx := context.Background()
	cfg := PromptLookupConfig{Prompt: "p", MaxTokens: 2048, TargetGenerate: target, LookupTokens: buildDecodeTokens(2048)}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkResult, decodeSinkErr = PromptLookup(ctx, cfg)
	}
}

// --- buildAcceptanceResult in isolation (the inner loop both
// Speculative + PromptLookup share) ---

func BenchmarkDecode_BuildAcceptance_32Tokens(b *testing.B) {
	target := buildDecodeTokens(32)
	candidates := buildDecodeTokens(32)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkResult = buildAcceptanceResult(ModeSpeculative, "p", target, candidates, 32)
	}
}

func BenchmarkDecode_BuildAcceptance_256Tokens(b *testing.B) {
	target := buildDecodeTokens(256)
	candidates := buildDecodeTokens(256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkResult = buildAcceptanceResult(ModeSpeculative, "p", target, candidates, 256)
	}
}

func BenchmarkDecode_BuildAcceptance_2048Tokens(b *testing.B) {
	target := buildDecodeTokens(2048)
	candidates := buildDecodeTokens(2048)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkResult = buildAcceptanceResult(ModeSpeculative, "p", target, candidates, 2048)
	}
}

// --- TokensText (renders the emitted stream into the Result.Text) ---

func BenchmarkDecode_TokensText_32Tokens(b *testing.B) {
	tokens := buildDecodeTokens(32)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkText = TokensText(tokens)
	}
}

func BenchmarkDecode_TokensText_256Tokens(b *testing.B) {
	tokens := buildDecodeTokens(256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkText = TokensText(tokens)
	}
}

func BenchmarkDecode_TokensText_2048Tokens(b *testing.B) {
	tokens := buildDecodeTokens(2048)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkText = TokensText(tokens)
	}
}

// --- CloneTokens (fires per accepted token in buildAcceptanceResult,
// plus once per result handoff) ---

func BenchmarkDecode_CloneTokens_32Tokens(b *testing.B) {
	tokens := buildDecodeTokens(32)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkTokens = CloneTokens(tokens)
	}
}

func BenchmarkDecode_CloneTokens_256Tokens(b *testing.B) {
	tokens := buildDecodeTokens(256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkTokens = CloneTokens(tokens)
	}
}

func BenchmarkDecode_CloneTokens_2048Tokens(b *testing.B) {
	tokens := buildDecodeTokens(2048)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkTokens = CloneTokens(tokens)
	}
}

// --- TokenEqual (per-token branch — text-vs-value-vs-empty paths) ---

func BenchmarkDecode_TokenEqual_BothTextEqual(b *testing.B) {
	a := Token{ID: 1, Text: "abcdef"}
	c := Token{ID: 1, Text: "abcdef"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkBool = TokenEqual(a, c)
	}
}

func BenchmarkDecode_TokenEqual_IDMismatch(b *testing.B) {
	a := Token{ID: 1, Text: "abcdef"}
	c := Token{ID: 2, Text: "abcdef"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkBool = TokenEqual(a, c)
	}
}

func BenchmarkDecode_TokenEqual_EmptyTextSkipsCompare(b *testing.B) {
	a := Token{ID: 1}
	c := Token{ID: 1, Text: "abcdef"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkBool = TokenEqual(a, c)
	}
}

// --- normaliseMaxTokens (called twice per Speculative / once per
// PromptLookup) ---

func BenchmarkDecode_NormaliseMaxTokens_FirstPositive(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkInt = normaliseMaxTokens(64, 0, 0)
	}
}

func BenchmarkDecode_NormaliseMaxTokens_FallsThrough(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkInt = normaliseMaxTokens(0, 0, 0)
	}
}

// --- nonZeroDuration (fires three times per decode call) ---

func BenchmarkDecode_NonZeroDuration_Positive(b *testing.B) {
	d := 45 * time.Millisecond
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkDur = nonZeroDuration(d)
	}
}

func BenchmarkDecode_NonZeroDuration_Zero(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkDur = nonZeroDuration(0)
	}
}
