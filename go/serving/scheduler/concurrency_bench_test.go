// SPDX-Licence-Identifier: EUPL-1.2

// Concurrency-stress benchmarks for the scheduler. The existing
// scheduler_bench_test.go suite measures single-stream cost; this
// file measures Schedule + drain under parallel pressure across the
// MaxConcurrent knob (1 / 4 / 16 workers) at three request fan-outs
// (4 / 16 / 64 concurrent producers).
//
// Per [[project_kv_state_decode_loadbearing_for_portable_knowledge]] —
// decode + scheduler is the per-token consumer of continuous state.
// Real lthn.ai traffic is many-stream-at-once (IDE chat + agent
// dispatch + classification probes share a worker pool); single-
// stream benches miss the worker-queue + label-map contention that
// only appears under fan-out.
//
// The shared schedBenchModel from scheduler_bench_test.go is safe
// under parallel use — its iter.Seq closure has no shared state,
// just the immutable tokens slice. We reuse it.
//
// Per the lane spec: avoid the pre-existing race in
// TestModel_QueuesRequestsAndEmitsLatencyProbe_Good — the benches
// here use fresh schedulers per b.Run + RunParallel hands each PB
// its own goroutine; no shared state with that test.
//
// Sink discipline: under parallel/burst dispatch, multiple goroutines
// would race writing the package-level schedSink* variables. We use
// sync/atomic + a per-bench int64 counter instead, then add it into
// the package sink once at the bench end. That defeats DCE without
// creating a race.
//
// Run:    go test -bench='BenchmarkScheduler_Concurrent' -benchmem -run='^$' ./go/scheduler

package scheduler

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"

	"dappco.re/go/inference"
)

// drainSchedulerStream consumes a token channel to completion. Used
// inside parallel benches so producer back-pressure does not pile up.
func drainSchedulerStream(tokens <-chan inference.ScheduledToken) int {
	count := 0
	for range tokens {
		count++
	}
	return count
}

// --- Schedule + drain under RunParallel — the dominant concurrency
// stress for the queue + worker pool. Each pb iteration mints one
// request, drains it, recycles. ---

func BenchmarkScheduler_Schedule_Concurrent_4Workers_32Tokens(b *testing.B) {
	base := &schedBenchModel{tokens: benchTokens(32)}
	sched, _ := New(base, Config{MaxConcurrent: 4, MaxQueue: 64, StreamBuffer: 32})
	ctx := context.Background()
	var total atomic.Int64
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, tokens, err := sched.Schedule(ctx, inference.ScheduledRequest{Prompt: "p"})
			if err != nil {
				continue
			}
			total.Add(int64(drainSchedulerStream(tokens)))
		}
	})
	schedSinkTokensCount = int(total.Load())
}

func BenchmarkScheduler_Schedule_Concurrent_16Workers_32Tokens(b *testing.B) {
	base := &schedBenchModel{tokens: benchTokens(32)}
	sched, _ := New(base, Config{MaxConcurrent: 16, MaxQueue: 128, StreamBuffer: 32})
	ctx := context.Background()
	var total atomic.Int64
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, tokens, err := sched.Schedule(ctx, inference.ScheduledRequest{Prompt: "p"})
			if err != nil {
				continue
			}
			total.Add(int64(drainSchedulerStream(tokens)))
		}
	})
	schedSinkTokensCount = int(total.Load())
}

func BenchmarkScheduler_Schedule_Concurrent_1Worker_32Tokens(b *testing.B) {
	base := &schedBenchModel{tokens: benchTokens(32)}
	sched, _ := New(base, Config{MaxConcurrent: 1, MaxQueue: 256, StreamBuffer: 32})
	ctx := context.Background()
	var total atomic.Int64
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, tokens, err := sched.Schedule(ctx, inference.ScheduledRequest{Prompt: "p"})
			if err != nil {
				continue
			}
			total.Add(int64(drainSchedulerStream(tokens)))
		}
	})
	schedSinkTokensCount = int(total.Load())
}

// --- Burst dispatch — release N concurrent producers, wait for all to
// finish in turn. Captures the "spike of arrivals" shape rather than the
// steady-state RunParallel rhythm. ---

func benchScheduleBurst(b *testing.B, workers int, tokens int) {
	base := &schedBenchModel{tokens: benchTokens(tokens)}
	sched, _ := New(base, Config{
		MaxConcurrent: 4,
		MaxQueue:      workers * 2,
		StreamBuffer:  tokens,
	})
	ctx := context.Background()
	var total atomic.Int64
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var wg sync.WaitGroup
		wg.Add(workers)
		for range workers {
			go func() {
				defer wg.Done()
				_, stream, err := sched.Schedule(ctx, inference.ScheduledRequest{Prompt: "p"})
				if err != nil {
					return
				}
				total.Add(int64(drainSchedulerStream(stream)))
			}()
		}
		wg.Wait()
	}
	schedSinkTokensCount = int(total.Load())
}

func BenchmarkScheduler_Burst_4Producers_32Tokens(b *testing.B) {
	benchScheduleBurst(b, 4, 32)
}

func BenchmarkScheduler_Burst_16Producers_32Tokens(b *testing.B) {
	benchScheduleBurst(b, 16, 32)
}

func BenchmarkScheduler_Burst_64Producers_32Tokens(b *testing.B) {
	benchScheduleBurst(b, 64, 32)
}

// 256-token burst — measures whether the per-token label-write
// contention pattern compounds with stream length.
func BenchmarkScheduler_Burst_16Producers_256Tokens(b *testing.B) {
	benchScheduleBurst(b, 16, 256)
}

// --- Queue-saturation pressure — workers can't drain as fast as
// producers arrive; the queue depth oscillates near full. Captures
// the cost of the queue-full rejection path under steady pressure. ---

func BenchmarkScheduler_QueueSaturation_TinyQueue(b *testing.B) {
	base := &schedBenchModel{tokens: benchTokens(32)}
	sched, _ := New(base, Config{MaxConcurrent: 1, MaxQueue: 1, StreamBuffer: 4})
	ctx := context.Background()
	var total, errs atomic.Int64
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, tokens, err := sched.Schedule(ctx, inference.ScheduledRequest{Prompt: "p"})
			if err != nil {
				// Queue-full rejection — counted, drained, recycled.
				errs.Add(1)
				continue
			}
			total.Add(int64(drainSchedulerStream(tokens)))
		}
	})
	schedSinkTokensCount = int(total.Load() + errs.Load())
}

// --- CancelRequest hot-path under contention — when one goroutine
// is calling CancelRequest while another is calling Schedule, the
// shared mu.Lock around m.active is the synchronisation point. This
// bench measures the cost of contesting that lock at fan-out 4. ---

func BenchmarkScheduler_CancelRequest_NotFound_Parallel(b *testing.B) {
	base := &schedBenchModel{tokens: benchTokens(1)}
	sched, _ := New(base, Config{MaxConcurrent: 4, MaxQueue: 16, StreamBuffer: 4})
	ctx := context.Background()
	var total atomic.Int64
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			res, _ := sched.CancelRequest(ctx, "no-such-id")
			if res.Cancelled {
				total.Add(1)
			} else {
				total.Add(-1)
			}
		}
	})
	schedSinkTokensCount = int(total.Load())
}
