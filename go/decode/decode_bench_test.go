// SPDX-Licence-Identifier: EUPL-1.2

package decode

import (
	"context"
	"testing"
)

// Bench fixtures — generator closures that emit pre-built token slices
// without allocating per call (slice header copy only).

func makeTokens(n int) []Token {
	out := make([]Token, n)
	for i := range out {
		out[i] = Token{ID: int32(i + 1), Text: "t"}
	}
	return out
}

func makeMixedTokens(n int, divergeAt int) []Token {
	out := makeTokens(n)
	for i := divergeAt; i < n; i++ {
		out[i] = Token{ID: int32(-(i + 1)), Text: "x"}
	}
	return out
}

func staticGenerate(tokens []Token) GenerateFunc {
	return func(context.Context, string, GenerateConfig) (Generation, error) {
		return Generation{Tokens: tokens}, nil
	}
}

// BenchmarkSpeculative_AllAccepted measures the hot path where draft
// candidates match target one-for-one — exercises the accept branch +
// per-token clone + builder concatenation.
func BenchmarkSpeculative_AllAccepted(b *testing.B) {
	target := makeTokens(128)
	cfg := SpeculativeConfig{
		Prompt:         "bench",
		MaxTokens:      128,
		DraftTokens:    128,
		TargetGenerate: staticGenerate(target),
		DraftGenerate:  staticGenerate(target),
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.SetBytes(int64(len(target)))
	for b.Loop() {
		if _, err := Speculative(ctx, cfg); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSpeculative_HalfRejected exercises the reject branch where
// half the draft tokens diverge — measures fallback path cost.
func BenchmarkSpeculative_HalfRejected(b *testing.B) {
	target := makeTokens(128)
	draft := makeMixedTokens(128, 64)
	cfg := SpeculativeConfig{
		Prompt:         "bench",
		MaxTokens:      128,
		DraftTokens:    128,
		TargetGenerate: staticGenerate(target),
		DraftGenerate:  staticGenerate(draft),
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.SetBytes(int64(len(target)))
	for b.Loop() {
		if _, err := Speculative(ctx, cfg); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSpeculative_NoDraft (draft empty) exercises the path with
// zero candidates — all target tokens emitted without comparison.
func BenchmarkSpeculative_NoDraftMatch(b *testing.B) {
	target := makeTokens(128)
	cfg := SpeculativeConfig{
		Prompt:         "bench",
		MaxTokens:      128,
		DraftTokens:    1,
		TargetGenerate: staticGenerate(target),
		DraftGenerate:  staticGenerate(nil),
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.SetBytes(int64(len(target)))
	for b.Loop() {
		if _, err := Speculative(ctx, cfg); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkPromptLookup_AllAccepted exercises prompt-lookup with full
// candidate match. Single-target-call path.
func BenchmarkPromptLookup_AllAccepted(b *testing.B) {
	target := makeTokens(128)
	cfg := PromptLookupConfig{
		Prompt:         "bench",
		MaxTokens:      128,
		TargetGenerate: staticGenerate(target),
		LookupTokens:   target,
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.SetBytes(int64(len(target)))
	for b.Loop() {
		if _, err := PromptLookup(ctx, cfg); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkPromptLookup_HalfRejected exercises prompt-lookup with half
// candidates diverging from target.
func BenchmarkPromptLookup_HalfRejected(b *testing.B) {
	target := makeTokens(128)
	lookup := makeMixedTokens(128, 64)
	cfg := PromptLookupConfig{
		Prompt:         "bench",
		MaxTokens:      128,
		TargetGenerate: staticGenerate(target),
		LookupTokens:   lookup,
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.SetBytes(int64(len(target)))
	for b.Loop() {
		if _, err := PromptLookup(ctx, cfg); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSpeculative_Small measures latency-floor on tiny streams,
// where allocation overhead dominates over per-token work.
func BenchmarkSpeculative_Small(b *testing.B) {
	target := makeTokens(8)
	cfg := SpeculativeConfig{
		Prompt:         "bench",
		MaxTokens:      8,
		DraftTokens:    8,
		TargetGenerate: staticGenerate(target),
		DraftGenerate:  staticGenerate(target),
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.SetBytes(int64(len(target)))
	for b.Loop() {
		if _, err := Speculative(ctx, cfg); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSpeculative_Large measures throughput on long streams where
// per-token costs dominate.
func BenchmarkSpeculative_Large(b *testing.B) {
	target := makeTokens(1024)
	cfg := SpeculativeConfig{
		Prompt:         "bench",
		MaxTokens:      1024,
		DraftTokens:    1024,
		TargetGenerate: staticGenerate(target),
		DraftGenerate:  staticGenerate(target),
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.SetBytes(int64(len(target)))
	for b.Loop() {
		if _, err := Speculative(ctx, cfg); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkTokensText measures the per-token string concatenation path.
func BenchmarkTokensText(b *testing.B) {
	tokens := makeTokens(256)
	b.ReportAllocs()
	b.SetBytes(int64(len(tokens)))
	for b.Loop() {
		_ = TokensText(tokens)
	}
}

// BenchmarkCloneTokens measures the bulk clone path.
func BenchmarkCloneTokens(b *testing.B) {
	tokens := makeTokens(256)
	b.ReportAllocs()
	b.SetBytes(int64(len(tokens)))
	for b.Loop() {
		_ = CloneTokens(tokens)
	}
}

// BenchmarkTokenEqual_Match measures the accept hot-path token compare.
func BenchmarkTokenEqual_Match(b *testing.B) {
	a := Token{ID: 42, Text: "hello"}
	c := Token{ID: 42, Text: "hello"}
	b.ReportAllocs()
	for b.Loop() {
		if !TokenEqual(a, c) {
			b.Fatal("expected equal")
		}
	}
}

// BenchmarkTokenEqual_Mismatch measures the reject path.
func BenchmarkTokenEqual_Mismatch(b *testing.B) {
	a := Token{ID: 42, Text: "hello"}
	c := Token{ID: 42, Text: "world"}
	b.ReportAllocs()
	for b.Loop() {
		if TokenEqual(a, c) {
			b.Fatal("expected mismatch")
		}
	}
}
