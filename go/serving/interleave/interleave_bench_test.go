// SPDX-Licence-Identifier: EUPL-1.2

// Allocation contract for the live admission scheduler. Each bench iteration
// submits a fresh batch of requests against one long-lived Engine and drains
// every token — the per-request goroutine + channel overhead this package
// deliberately pays for isolation (see interleave.go's package doc) is what
// this measures against schedule.Engine's single-goroutine Run.
//
// Run: go test -bench=. -benchmem -run='^$' ./serving/interleave/
package interleave

import (
	"context"
	"strconv"
	"testing"

	"dappco.re/go/inference"
)

// benchSource yields n tokens with no synchronisation overhead — the bench
// isolates the Engine's own scheduling cost, not a fake decode loop's.
func benchSource(id string, n int) Source {
	return func(ctx context.Context) Stream {
		return func(yield func(inference.Token) bool) {
			for i := range n {
				if !yield(inference.Token{Text: id + "-" + strconv.Itoa(i)}) {
					return
				}
			}
		}
	}
}

func BenchmarkEngine_Submit(b *testing.B) {
	const (
		n         = 32 // requests per iteration
		tokens    = 16 // tokens generated each
		maxActive = 8  // concurrency cap
	)
	ctx := context.Background()
	e := New(Config{MaxActive: maxActive, MaxQueue: n, StreamBuffer: tokens})
	defer e.Close()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chans := make([]<-chan inference.Token, n)
		for r := range n {
			id := "req-" + strconv.Itoa(i) + "-" + strconv.Itoa(r)
			ch, err := e.Submit(ctx, id, 0, benchSource(id, tokens))
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
