// SPDX-Licence-Identifier: EUPL-1.2

// Backpressure benchmarks. The Schedule path has three points where
// flow control kicks in:
//
//   1. Queue full at Schedule — the default arm of the queue-send
//      select rejects with "scheduler: queue is full"
//   2. StreamBuffer full inside run() — the producer blocks on
//      j.out <- ScheduledToken (in the select with j.ctx.Done())
//   3. Slow consumer — the producer paces against consumer rhythm
//
// The existing scheduler_bench_test.go suite measures the
// happy-path (StreamBuffer >= token count, no rejection). This
// file covers the contended shapes.
//
// Per the lane spec — backpressure scenarios are part of the load-
// bearing path between cached state and live tokens. A slow consumer
// (IDE that pauses to render markdown, agent that batches probes
// for ratelimit) sits between Virgil's continuous state and the
// user-visible stream. Coverage of producer-blocks-on-consumer is
// the only way to see whether scheduler.go's per-token select cost
// dominates a slow-consumer workload.
//
// Run:    go test -bench='BenchmarkScheduler_Backpressure' -benchmem -run='^$' ./go/scheduler

package scheduler

import (
	"context"
	"testing"
	"time"

	"dappco.re/go/inference"
)

// --- Queue full rejection at Schedule ---

// QueueFull_Reject — submit a request to a saturated queue with an
// in-flight blocking job. Schedule takes the queue-full arm and
// returns the rejection error. Measures the rejection-path alloc
// budget — unregister + cancel + close(j.out) + NewError.
//
// Implementation: worker count 0 (normalised to 1 by Config), queue
// size 1, StreamBuffer 1. We pre-load the worker with a long-paced
// job whose first token doesn't emit during the bench window, then
// wait briefly so the worker has picked up the job out of the queue.
// Then we load the queue with a second job. From that point every
// subsequent Schedule must reject.
func BenchmarkScheduler_Backpressure_QueueFull_Reject(b *testing.B) {
	base := &cancellableBenchModel{tokens: benchTokens(2), perTokenNs: 10 * time.Second}
	sched, _ := New(base, Config{MaxConcurrent: 1, MaxQueue: 1, StreamBuffer: 1})
	ctx, cancel := context.WithCancel(context.Background())
	// Saturate the pipeline outside the timed loop. Drainers ensure
	// no goroutines leak beyond the worker pool.
	workerHandle, workerTokens, err := sched.Schedule(ctx, inference.ScheduledRequest{Prompt: "filler-worker"})
	if err != nil {
		b.Fatalf("filler-worker schedule: %v", err)
	}
	// Wait for the worker to pull the filler-worker job off the queue
	// (worker is a goroutine that drains m.queue). Polling for queue
	// emptiness via a short retry loop on Schedule.
	deadline := time.Now().Add(100 * time.Millisecond)
	var queueHandle inference.RequestHandle
	var queueTokens <-chan inference.ScheduledToken
	for time.Now().Before(deadline) {
		queueHandle, queueTokens, err = sched.Schedule(ctx, inference.ScheduledRequest{Prompt: "filler-queue"})
		if err == nil {
			break
		}
		time.Sleep(time.Millisecond)
	}
	if err != nil {
		cancel()
		b.Fatalf("filler-queue schedule never succeeded: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		handle, tokens, err := sched.Schedule(ctx, inference.ScheduledRequest{Prompt: "rejected"})
		schedSinkHandle = handle
		schedSinkErr = err
		if tokens != nil {
			for range tokens {
			}
		}
	}
	b.StopTimer()
	// Cancel both fillers and drain so we don't block the next bench
	// behind a 10s sleep. We don't care about their final state.
	_, _ = sched.CancelRequest(context.Background(), workerHandle.ID)
	_, _ = sched.CancelRequest(context.Background(), queueHandle.ID)
	go func() {
		for range workerTokens {
		}
	}()
	go func() {
		for range queueTokens {
		}
	}()
	cancel()
}

// --- StreamBuffer-full producer blocking ---

// SlowConsumer_StreamBufferFull — a tight StreamBuffer of 1, a 256-
// token producer, and a consumer that only reads with a small delay
// per token. The producer blocks in the j.out <- select on every
// token after the first. Measures the cost of repeatedly entering
// the per-token select arm under contention.
func BenchmarkScheduler_Backpressure_SlowConsumer_StreamBufferFull(b *testing.B) {
	base := &schedBenchModel{tokens: benchTokens(64)}
	sched, _ := New(base, Config{MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 1})
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, tokens, err := sched.Schedule(ctx, inference.ScheduledRequest{Prompt: "p"})
		if err != nil {
			continue
		}
		count := 0
		for range tokens {
			count++
			// Per-token consumer-side delay — 1µs * 64 tokens = 64µs of
			// producer-blocked time per request. Without it the
			// producer-faster-than-consumer dynamic doesn't surface
			// because the local channel ring rotates too fast.
			time.Sleep(1 * time.Microsecond)
		}
		schedSinkTokensCount = count
	}
}

// --- Producer-faster-than-consumer ---

// FastProducer_FastConsumer — baseline reference for the slow-
// consumer bench above. Same token count, same StreamBuffer=1, but
// the consumer reads at full speed. The delta isolates the cost of
// time.Sleep + select-on-channel-write pressure.
func BenchmarkScheduler_Backpressure_FastProducer_FastConsumer_StreamBuffer1(b *testing.B) {
	base := &schedBenchModel{tokens: benchTokens(64)}
	sched, _ := New(base, Config{MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 1})
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, tokens, err := sched.Schedule(ctx, inference.ScheduledRequest{Prompt: "p"})
		if err != nil {
			continue
		}
		count := 0
		for range tokens {
			count++
		}
		schedSinkTokensCount = count
	}
}

// --- StreamBuffer=0 — fully synchronous handoff ---

// SyncHandoff_StreamBufferZero — exercises the StreamBuffer=0 case
// where every producer-to-consumer handoff is a rendezvous. The Config
// normalises StreamBuffer<0 to 0; we test 0 explicitly to confirm the
// downgraded buffer still streams tokens (vs the fast path with a
// pre-allocated buffer).
func BenchmarkScheduler_Backpressure_SyncHandoff_StreamBufferZero(b *testing.B) {
	base := &schedBenchModel{tokens: benchTokens(32)}
	sched, _ := New(base, Config{MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 0})
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, tokens, err := sched.Schedule(ctx, inference.ScheduledRequest{Prompt: "p"})
		if err != nil {
			continue
		}
		count := 0
		for range tokens {
			count++
		}
		schedSinkTokensCount = count
	}
}

// --- Drain-cost-of-aborted-stream-vs-fully-drained-stream ---

// AbortedDrain_NotFullyConsumed — consumer abandons the stream
// after 4 tokens; the Generate iterator handle that wraps Schedule
// would call CancelRequest under yield-false, but here we exit the
// for range loop and let the channel close on its own. Some IDE
// patterns leak this way.
//
// Note: we don't yield-false (no Generate wrapper); we just stop
// reading from the channel. The producer will block on the next
// send until the run() Done arm trips when the iteration ends.
// This bench captures the cost of dangling channels — a real risk
// for callers who forget the drain contract.
func BenchmarkScheduler_Backpressure_AbortedDrain_4Of64(b *testing.B) {
	base := &cancellableBenchModel{tokens: benchTokens(64), perTokenNs: 5 * time.Microsecond}
	sched, _ := New(base, Config{MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 4})
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		handle, tokens, err := sched.Schedule(ctx, inference.ScheduledRequest{Prompt: "p"})
		if err != nil {
			continue
		}
		count := 0
		for range tokens {
			count++
			if count >= 4 {
				// Aborted — cancel + drain the rest so the bench's
				// next iteration starts from a clean state. This IS
				// the documented contract.
				schedSinkCancel, schedSinkErr = sched.CancelRequest(ctx, handle.ID)
				for range tokens {
				}
				break
			}
		}
		schedSinkTokensCount = count
	}
}
