// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the n-gram speculative drafter. The lookup core (shared by
// Draft and DraftNext) runs PER TOKEN in the decode loop, so the two paths that
// matter are the speculative HIT (a repeated suffix → an allocated draft) and
// the speculative MISS (no earlier occurrence → a full backwards scan and a nil
// draft). The supporting stateful methods (Update / Context / Reset / New) are
// covered too so every public entry point has an allocs/op number.
//
// Run:    go test -bench=. -benchmem -run='^$' ./ngram/

package ngram_test

import (
	"testing"

	"dappco.re/go/inference/decode/ngram"
)

// Sinks defeat dead-code elimination so the benchmarked work is not optimised
// away by the compiler.
var (
	ngramSinkDraft []int
	ngramSinkCtx   []int
	ngramSinkD     *ngram.Drafter
)

// repeatedContext builds an n-token context whose ids cycle with period `period`,
// so any suffix shorter than the period recurs every `period` tokens. This is the
// speculative-HIT fixture: the trailing n-gram always has an earlier occurrence
// one period back, so Draft returns a non-empty proposal — the match arm that
// allocates the output buffer.
func repeatedContext(n, period int) []int {
	ctx := make([]int, n)
	for i := range ctx {
		ctx[i] = i % period
	}
	return ctx
}

// uniqueContext builds an n-token context of strictly distinct ids, so no n-gram
// ever recurs. This is the speculative-MISS fixture: Draft scans the whole
// context at every n and returns nil — the worst-case scan and the zero-alloc
// path.
func uniqueContext(n int) []int {
	ctx := make([]int, n)
	for i := range ctx {
		ctx[i] = i
	}
	return ctx
}

// --- Draft: the per-token stateless lookup (the hot path) ---

// HIT: a repeated suffix matches an earlier occurrence, so the match arm runs and
// allocates the proposed draft. Scan cost is roughly constant in context length
// (the most-recent occurrence is one period back), so these isolate the output
// allocation rather than the scan.
func benchmarkDraftHit(b *testing.B, n int) {
	d := ngram.New(ngram.Config{MaxNgram: 3, MaxDraft: 8})
	ctx := repeatedContext(n, 64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ngramSinkDraft = d.Draft(ctx)
	}
}

func BenchmarkDraft_Hit_256(b *testing.B)  { benchmarkDraftHit(b, 256) }
func BenchmarkDraft_Hit_1024(b *testing.B) { benchmarkDraftHit(b, 1024) }
func BenchmarkDraft_Hit_4096(b *testing.B) { benchmarkDraftHit(b, 4096) }

// MISS: no suffix recurs, so Draft scans the whole context at every n and returns
// nil — no allocation, worst-case scan that scales with context length.
func benchmarkDraftMiss(b *testing.B, n int) {
	d := ngram.New(ngram.Config{MaxNgram: 3, MaxDraft: 8})
	ctx := uniqueContext(n)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ngramSinkDraft = d.Draft(ctx)
	}
}

func BenchmarkDraft_Miss_256(b *testing.B)  { benchmarkDraftMiss(b, 256) }
func BenchmarkDraft_Miss_1024(b *testing.B) { benchmarkDraftMiss(b, 1024) }
func BenchmarkDraft_Miss_4096(b *testing.B) { benchmarkDraftMiss(b, 4096) }

// --- DraftNext: the per-token stateful lookup (Draft over the running context) ---

func benchmarkDraftNextHit(b *testing.B, n int) {
	d := ngram.New(ngram.Config{MaxNgram: 3, MaxDraft: 8})
	d.Update(repeatedContext(n, 64))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ngramSinkDraft = d.DraftNext()
	}
}

func BenchmarkDraftNext_Hit_1024(b *testing.B) { benchmarkDraftNextHit(b, 1024) }

func benchmarkDraftNextMiss(b *testing.B, n int) {
	d := ngram.New(ngram.Config{MaxNgram: 3, MaxDraft: 8})
	d.Update(uniqueContext(n))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ngramSinkDraft = d.DraftNext()
	}
}

func BenchmarkDraftNext_Miss_1024(b *testing.B) { benchmarkDraftNextMiss(b, 1024) }

// --- Update: appends accepted tokens to the running context ---

// Steady-state append: the running buffer keeps its capacity across decode steps,
// so an append of a few accepted tokens reallocates only when it doubles. Growth
// is bounded by clearing the length (keeping capacity, via Reset) every 4096
// appends so the benchmark measures the realistic amortised per-step cost.
func BenchmarkUpdate(b *testing.B) {
	d := ngram.New(ngram.Config{MaxNgram: 3, MaxDraft: 8})
	tok := []int{1, 2, 3, 4}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		d.Update(tok)
		if i&0xFFF == 0xFFF {
			d.Reset()
		}
	}
}

// --- Context: returns a defensive copy of the running context ---

func BenchmarkContext(b *testing.B) {
	d := ngram.New(ngram.Config{MaxNgram: 3, MaxDraft: 8})
	d.Update(repeatedContext(1024, 64))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ngramSinkCtx = d.Context()
	}
}

// --- Reset: truncates the running context in place ---

func BenchmarkReset(b *testing.B) {
	d := ngram.New(ngram.Config{MaxNgram: 3, MaxDraft: 8})
	d.Update(repeatedContext(1024, 64))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		d.Reset()
	}
}

// --- New: constructs a Drafter ---

func BenchmarkNew(b *testing.B) {
	cfg := ngram.Config{MaxNgram: 3, MaxDraft: 8}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ngramSinkD = ngram.New(cfg)
	}
}
