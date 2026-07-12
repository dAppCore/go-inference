// SPDX-Licence-Identifier: EUPL-1.2

// Allocation contract for batch mode's continuous-batching coordinator. Each
// bench iteration submits a fresh batch of requests against one long-lived
// engine and drains every scheduled token — this measures the single lockstep
// coordinator's own admission + delivery cost (one goroutine advancing the
// whole running set) against interleave mode's per-request goroutine model.
//
// Run: go test -bench=Batch -benchmem -run='^$' ./serving/scheduler/
package scheduler

import (
	"context"
	"strconv"
	"testing"

	"dappco.re/go/inference"
)

func BenchmarkBatch_Submit(b *testing.B) {
	const (
		n      = 32   // requests per iteration
		prompt = 64   // prompt tokens each (budget accounting)
		tokens = 16   // tokens generated each
		capN   = 8    // concurrency cap
		budget = 8192 // running token budget
	)
	ctx := context.Background()
	e := newBatchEngine(nil, Config{MaxConcurrent: capN, MaxQueue: n, MaxBatchTokens: budget, StreamBuffer: tokens})
	defer e.close()
	metrics := func() inference.GenerateMetrics { return inference.GenerateMetrics{} }

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chans := make([]<-chan inference.ScheduledToken, n)
		for r := range n {
			id := "req-" + strconv.Itoa(i) + "-" + strconv.Itoa(r)
			ch, err := e.submit(ctx, &batchReq{
				id:           id,
				promptTokens: prompt,
				src:          benchSource(id, tokens),
				metrics:      metrics,
			})
			if err != nil {
				b.Fatal(err)
			}
			chans[r] = ch
		}
		for _, ch := range chans {
			for range ch {
			}
		}
	}
}
