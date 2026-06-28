// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the batch executor (batch.go) and rate limiter (limiter.go).
// Per AX-11 — Run / RunAsCompleted sit on every served batch and dispatch a
// worker pool per call; runOne and TokenBucket.Wait fire once PER REQUEST, so
// their per-call overhead is multiplied by the whole batch in the serving loop.
//
// The Call here is allocation-free (a pre-boxed result + a fixed Usage) so each
// benchmark measures the executor's own scheduling overhead, not the dispatch
// target's.
//
// Run:    go test -bench=. -benchmem -run='^$' ./batch/

package batch_test

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference/batch"
)

// benchCall is an allocation-free Call: it returns a pre-boxed result and a
// fixed Usage so a benchmark isolates the batch executor's overhead.
type benchCall struct {
	result any
	usage  batch.Usage
}

func (c benchCall) Do(ctx context.Context, index int, request any) (any, batch.Usage, error) {
	return c.result, c.usage, nil
}

// Pre-boxed once at init so neither the Call interface conversion nor the
// request elements add per-iteration boxing to the measured loop.
var (
	benchCallVal batch.Call = benchCall{result: "ok", usage: batch.Usage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2}}
	benchReqVal  any        = "req"
)

func makeReqs(n int) []any {
	r := make([]any, n)
	for i := range r {
		r[i] = benchReqVal
	}
	return r
}

// Sinks defeat compiler dead-code elimination.
var (
	benchResultSink batch.BatchResult
	benchItemSink   batch.ItemResult
	benchUsageSink  batch.Usage
	benchTBSink     *batch.TokenBucket
	benchErrSink    error
)

// --- Run (input-order) ---

func BenchmarkRun_Serial(b *core.B) {
	reqs := makeReqs(8)
	opts := batch.Options{Concurrency: 1, Call: benchCallVal}
	ctx := context.Background()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchResultSink = batch.Run(ctx, reqs, opts)
	}
}

func BenchmarkRun_Concurrent8(b *core.B) {
	reqs := makeReqs(8)
	opts := batch.Options{Concurrency: 8, Call: benchCallVal}
	ctx := context.Background()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchResultSink = batch.Run(ctx, reqs, opts)
	}
}

func BenchmarkRun_Concurrent64(b *core.B) {
	reqs := makeReqs(64)
	opts := batch.Options{Concurrency: 8, Call: benchCallVal}
	ctx := context.Background()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchResultSink = batch.Run(ctx, reqs, opts)
	}
}

// --- RunAsCompleted (streaming) ---

func BenchmarkRunAsCompleted_Concurrent64(b *core.B) {
	reqs := makeReqs(64)
	opts := batch.Options{Concurrency: 8, Call: benchCallVal}
	ctx := context.Background()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for it := range batch.RunAsCompleted(ctx, reqs, opts) {
			benchItemSink = it
		}
	}
}

// --- TokenBucket.Wait (per-request throttle) ---

// Unlimited: rate <= 0 → interval 0 → the fast path that never blocks.
func BenchmarkTokenBucket_Wait_Unlimited(b *core.B) {
	tb := batch.NewTokenBucket(0, 1)
	ctx := context.Background()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchErrSink = tb.Wait(ctx)
	}
}

// Available: a real interval but a burst so deep tokens are always present, so
// Wait takes the refill+consume path WITHOUT ever building a timer.
func BenchmarkTokenBucket_Wait_Available(b *core.B) {
	tb := batch.NewTokenBucket(1e6, 1_000_000_000_000)
	ctx := context.Background()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchErrSink = tb.Wait(ctx)
	}
}

// --- Constructor + value folding ---

func BenchmarkNewTokenBucket(b *core.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchTBSink = batch.NewTokenBucket(100, 10)
	}
}

func BenchmarkUsage_Add(b *core.B) {
	u := batch.Usage{PromptTokens: 1, CompletionTokens: 2, TotalTokens: 3}
	o := batch.Usage{PromptTokens: 4, CompletionTokens: 5, TotalTokens: 9}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchUsageSink = u.Add(o)
	}
}
