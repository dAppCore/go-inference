// SPDX-Licence-Identifier: EUPL-1.2

// Allocation contract for interleave mode's live admission scheduler. Each
// bench iteration submits a fresh batch of requests against one long-lived
// engine and drains every token — the per-request goroutine + channel overhead
// this mode deliberately pays for isolation (see interleave.go's doc) is what
// this measures against batch mode's single-goroutine lockstep coordinator.
//
// Run: go test -bench=Interleave -benchmem -run='^$' ./serving/scheduler/
package scheduler

import (
	"context"
	"strconv"
	"testing"

	"dappco.re/go/inference"
)

// benchSource yields n tokens with no synchronisation overhead — the bench
// isolates the engine's own scheduling cost, not a fake decode loop's.
func benchSource(id string, n int) source {
	return func(ctx context.Context) stream {
		return func(yield func(inference.Token) bool) {
			for i := range n {
				if !yield(inference.Token{Text: id + "-" + strconv.Itoa(i)}) {
					return
				}
			}
		}
	}
}

func BenchmarkInterleave_Submit(b *testing.B) {
	const (
		n         = 32 // requests per iteration
		tokens    = 16 // tokens generated each
		maxActive = 8  // concurrency cap
	)
	ctx := context.Background()
	e := newInterleaveEngine(Config{MaxConcurrent: maxActive, MaxQueue: n, StreamBuffer: tokens})
	defer e.close()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chans := make([]<-chan inference.Token, n)
		for r := range n {
			id := "req-" + strconv.Itoa(i) + "-" + strconv.Itoa(r)
			ch, err := e.submit(ctx, id, 0, benchSource(id, tokens))
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
