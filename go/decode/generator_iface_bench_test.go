// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the Generator-interface migration (W11-L). The hot
// path question is: does an interface field cost more, less, or the
// same as the previous func-typed field for callers that build a
// fresh generator per call (the dominant go-mlx shape today)?
//
// Three shapes are bench'd against the same Speculative + PromptLookup
// inner loop:
//
//   - ClosurePerCall  — caller mints a fresh `func` per Speculative call
//     and assigns it to TargetGenerate / DraftGenerate. Wraps with
//     GeneratorFunc on assignment, but the closure itself escapes
//     because it captures the per-iteration tokens slice. This is the
//     shape every backend driver in go-cuda / go-rocm / go-mlx uses
//     today, and the one W11-L is designed to give them a cheaper
//     alternative to.
//
//   - PreboundFunc    — caller builds the GeneratorFunc once (outside
//     the timed loop) and reuses the same value across every call. No
//     per-call closure alloc — the closure was paid once. This is the
//     existing decode bench shape; included here for direct comparison.
//
//   - PooledStruct    — caller's Generator is a struct with a sync.Pool
//     for the per-call state and a Generate method on the pooled value.
//     Zero closure allocs because no closure exists; the interface
//     dispatch goes straight to the struct method. This is the shape
//     W11-L enables and the one go-mlx will adopt in the follow-up
//     `modelDecodeGenerate`-to-struct migration.
//
// Realistic goal: PooledStruct demonstrates a strict alloc-count
// reduction vs ClosurePerCall while staying within noise of PreboundFunc
// on wall time — i.e. the interface dispatch overhead is amortised
// away the moment the closure alloc disappears.
//
// Run:    go test -bench='BenchmarkDecode_GeneratorShape' -benchmem -run='^$' ./go/decode

package decode

import (
	"context"
	"sync"
	"testing"
)

// pooledScriptGenerator is the win-demonstrating shape: a struct that
// implements Generator on a value receiver, served by a sync.Pool.
// `tokens` is set per acquisition; Generate hands the slice back
// without re-allocating. The pool ensures the struct itself is
// recycled across calls — zero allocation in the steady state.
type pooledScriptGenerator struct {
	tokens []Token
}

// Generate satisfies decode.Generator. Value receiver: no per-call
// pointer alloc when the struct is held by value (or by *pool*).
func (g *pooledScriptGenerator) Generate(context.Context, string, GenerateConfig) (Generation, error) {
	return Generation{Tokens: g.tokens}, nil
}

// genPool recycles pooledScriptGenerator instances across the bench
// loop. In production this is the modelDecodeGenerator pool described
// in W11-L follow-up.
var genPool = sync.Pool{
	New: func() any { return &pooledScriptGenerator{} },
}

// acquirePooledGen rents a generator from the pool and parks the
// tokens slice on it. Caller is expected to call releasePooledGen
// directly — returning a release closure would heap-allocate the
// closure on every call and drown the whole win we're trying to
// measure. The straight pointer API is the production-realistic
// shape (go-mlx's modelDecodeGenerate follow-up will do the same).
func acquirePooledGen(tokens []Token) *pooledScriptGenerator {
	g := genPool.Get().(*pooledScriptGenerator)
	g.tokens = tokens
	return g
}

// releasePooledGen recycles a generator back to the pool. Caller is
// responsible for not touching the struct after the release call.
func releasePooledGen(g *pooledScriptGenerator) {
	g.tokens = nil
	genPool.Put(g)
}

// --- Speculative — three shapes side-by-side at 256 tokens ---

// ClosurePerCall — the shape every driver uses today. Closure captures
// `tokens` so it escapes; one alloc per Speculative call before decode
// even runs.
func BenchmarkDecode_GeneratorShape_Speculative_ClosurePerCall_256(b *testing.B) {
	targetTokens := buildDecodeTokens(256)
	draftTokens := buildDecodeTokens(256)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cfg := SpeculativeConfig{
			Prompt:      "p",
			MaxTokens:   256,
			DraftTokens: 256,
			TargetGenerate: GeneratorFunc(func(context.Context, string, GenerateConfig) (Generation, error) {
				return Generation{Tokens: targetTokens}, nil
			}),
			DraftGenerate: GeneratorFunc(func(context.Context, string, GenerateConfig) (Generation, error) {
				return Generation{Tokens: draftTokens}, nil
			}),
		}
		decodeSinkResult, decodeSinkErr = Speculative(ctx, cfg)
	}
}

// PreboundFunc — the existing decode bench shape. The closure was
// paid once outside the timed loop; only the inner-loop allocs show.
func BenchmarkDecode_GeneratorShape_Speculative_PreboundFunc_256(b *testing.B) {
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

// PooledStruct — the W11-L-enabled shape. Per call: pool Get (no
// alloc when the pool is warm), interface dispatch into Generate,
// pool Put. Zero closure allocs because there is no closure.
func BenchmarkDecode_GeneratorShape_Speculative_PooledStruct_256(b *testing.B) {
	targetTokens := buildDecodeTokens(256)
	draftTokens := buildDecodeTokens(256)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		target := acquirePooledGen(targetTokens)
		draft := acquirePooledGen(draftTokens)
		cfg := SpeculativeConfig{
			Prompt:         "p",
			MaxTokens:      256,
			DraftTokens:    256,
			TargetGenerate: target,
			DraftGenerate:  draft,
		}
		decodeSinkResult, decodeSinkErr = Speculative(ctx, cfg)
		releasePooledGen(draft)
		releasePooledGen(target)
	}
}

// --- PromptLookup — three shapes side-by-side at 256 tokens ---

func BenchmarkDecode_GeneratorShape_PromptLookup_ClosurePerCall_256(b *testing.B) {
	targetTokens := buildDecodeTokens(256)
	lookupTokens := buildDecodeTokens(256)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cfg := PromptLookupConfig{
			Prompt:    "p",
			MaxTokens: 256,
			TargetGenerate: GeneratorFunc(func(context.Context, string, GenerateConfig) (Generation, error) {
				return Generation{Tokens: targetTokens}, nil
			}),
			LookupTokens: lookupTokens,
		}
		decodeSinkResult, decodeSinkErr = PromptLookup(ctx, cfg)
	}
}

func BenchmarkDecode_GeneratorShape_PromptLookup_PreboundFunc_256(b *testing.B) {
	target := scriptGen(buildDecodeTokens(256))
	lookupTokens := buildDecodeTokens(256)
	ctx := context.Background()
	cfg := PromptLookupConfig{Prompt: "p", MaxTokens: 256, TargetGenerate: target, LookupTokens: lookupTokens}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkResult, decodeSinkErr = PromptLookup(ctx, cfg)
	}
}

func BenchmarkDecode_GeneratorShape_PromptLookup_PooledStruct_256(b *testing.B) {
	targetTokens := buildDecodeTokens(256)
	lookupTokens := buildDecodeTokens(256)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		target := acquirePooledGen(targetTokens)
		cfg := PromptLookupConfig{
			Prompt:         "p",
			MaxTokens:      256,
			TargetGenerate: target,
			LookupTokens:   lookupTokens,
		}
		decodeSinkResult, decodeSinkErr = PromptLookup(ctx, cfg)
		releasePooledGen(target)
	}
}
