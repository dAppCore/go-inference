// SPDX-Licence-Identifier: EUPL-1.2

// Cancellation-path benchmarks. The existing scheduler suite covers
// CancelRequest_NotFound (the no-active-id fallback); this file adds
// the four scenarios that exercise the live-cancellation paths and
// the cost of cancel-propagation through emitProbe:
//
//   * Cancel BEFORE start — context cancelled while job sits in queue
//   * Cancel via parent context Done — Schedule short-circuits at
//     the ctx.Done() select arm
//   * Cancel DURING stream — j.cancel() inside the stream consumer
//   * Cancel via context.WithTimeout — emulates RPC deadline timeout
//
// Per [[project_kv_state_decode_loadbearing_for_portable_knowledge]] —
// when continuous-state runtime sits behind the scheduler, cancellation
// is the only way to release a stuck KV-restore. The cost of cancel
// propagation IS in the load-bearing path; coverage is mandatory.
//
// Pre-existing race in TestModel_QueuesRequestsAndEmitsLatencyProbe_Good
// noted in W7-D — this file uses fresh schedulers per bench so no
// shared state with that test path.
//
// Run:    go test -bench='BenchmarkScheduler_Cancel' -benchmem -run='^$' ./go/scheduler

package scheduler

import (
	"context"
	"iter"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// cancellableBenchModel emits its tokens slowly enough that mid-stream
// cancellation is observable in the bench window. We sleep briefly
// between tokens so the cancel arm of the run() select fires on the
// realistic 'producer in the middle of streaming' shape.
//
// Tokens slice is immutable; the closure has no shared state, so it's
// parallel-safe and reusable across b.N iterations.
type cancellableBenchModel struct {
	tokens     []inference.Token
	perTokenNs time.Duration
}

func (m *cancellableBenchModel) Generate(ctx context.Context, _ string, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.seq(ctx)
}

func (m *cancellableBenchModel) Chat(ctx context.Context, _ []inference.Message, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.seq(ctx)
}

func (m *cancellableBenchModel) Classify(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.ClassifyResult(nil))
}

func (m *cancellableBenchModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.BatchResult(nil))
}

func (m *cancellableBenchModel) ModelType() string                  { return "cancellable-bench" }
func (m *cancellableBenchModel) Info() inference.ModelInfo          { return inference.ModelInfo{} }
func (m *cancellableBenchModel) Metrics() inference.GenerateMetrics { return inference.GenerateMetrics{} }
func (m *cancellableBenchModel) Err() core.Result                  { return core.Ok(nil) }
func (m *cancellableBenchModel) Close() core.Result                { return core.Ok(nil) }

func (m *cancellableBenchModel) seq(ctx context.Context) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		for _, token := range m.tokens {
			if err := ctx.Err(); err != nil {
				return
			}
			if m.perTokenNs > 0 {
				timer := time.NewTimer(m.perTokenNs)
				select {
				case <-ctx.Done():
					timer.Stop()
					return
				case <-timer.C:
				}
			}
			if !yield(token) {
				return
			}
		}
	}
}

// --- CancelRequest mid-stream — start a stream that paces tokens
// over 100µs each, fire cancel after ~10µs, measure the cancel +
// drain cost. The j.cancel() must propagate via j.ctx.Done() into
// the run() select arm. ---

func BenchmarkScheduler_Cancel_MidStream(b *testing.B) {
	base := &cancellableBenchModel{tokens: benchTokens(64), perTokenNs: 100 * time.Microsecond}
	sched := New(base, Config{MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 4})
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		handle, tokens, err := sched.Schedule(ctx, inference.ScheduledRequest{Prompt: "p"})
		if err != nil {
			continue
		}
		// Let one token emit, then cancel — exercises the j.ctx.Done()
		// arm of the run loop inside the per-token select.
		first := true
		count := 0
		for range tokens {
			count++
			if first {
				schedSinkCancel, schedSinkErr = sched.CancelRequest(ctx, handle.ID)
				first = false
			}
		}
		schedSinkTokensCount = count
	}
}

// --- CancelRequest BEFORE start — queue the request behind a slow
// in-flight one so it's still in the queue when we cancel. The cancel
// path takes the same j.cancel() route but j.run() will hit the
// ctx.Err() check at the top of run() and emit a "cancelled" probe. ---

func BenchmarkScheduler_Cancel_BeforeStart_QueueWait(b *testing.B) {
	// Lead emits a small number of tokens — buffer accommodates them
	// so the lead's producer can run to completion in the background
	// while we cancel the queued one. StreamBuffer >= lead-tokens
	// avoids a producer-blocks-on-consumer deadlock with the queued
	// drain ordering below.
	base := &cancellableBenchModel{tokens: benchTokens(8), perTokenNs: 50 * time.Microsecond}
	sched := New(base, Config{MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 16})
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Lead with one in-flight job so the second sits in the queue.
		_, leadTokens, err := sched.Schedule(ctx, inference.ScheduledRequest{Prompt: "lead"})
		if err != nil {
			continue
		}
		queued, queuedTokens, err := sched.Schedule(ctx, inference.ScheduledRequest{Prompt: "queued"})
		if err != nil {
			for range leadTokens {
			}
			continue
		}
		// Cancel the queued one while the lead still runs.
		schedSinkCancel, schedSinkErr = sched.CancelRequest(ctx, queued.ID)
		// Drain lead first — its producer needs the buffered channel
		// drained even though it fits. Then drain queued (which the
		// worker will see-cancelled and emit nothing before closing).
		count := 0
		for range leadTokens {
			count++
		}
		for range queuedTokens {
			count++
		}
		schedSinkTokensCount = count
	}
}

// --- Schedule under cancelled parent context — fast-fail path; the
// context.Err() guard at Schedule entry should reject immediately. ---

func BenchmarkScheduler_Cancel_ParentContextAlreadyCancelled(b *testing.B) {
	base := &schedBenchModel{tokens: benchTokens(1)}
	sched := New(base, Config{MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 4})
	parent, cancel := context.WithCancel(context.Background())
	cancel()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		handle, tokens, err := sched.Schedule(parent, inference.ScheduledRequest{Prompt: "p"})
		schedSinkHandle = handle
		schedSinkErr = err
		if tokens != nil {
			for range tokens {
			}
		}
	}
}

// --- Schedule under context.WithTimeout that has already elapsed —
// same fast-fail path but via a timer-cancelled context. Validates
// the ctx.Err() check at entry returns immediately. ---

func BenchmarkScheduler_Cancel_TimeoutAlreadyElapsed(b *testing.B) {
	base := &schedBenchModel{tokens: benchTokens(1)}
	sched := New(base, Config{MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 4})
	parent := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ctx, cancel := context.WithTimeout(parent, 0)
		handle, tokens, err := sched.Schedule(ctx, inference.ScheduledRequest{Prompt: "p"})
		schedSinkHandle = handle
		schedSinkErr = err
		if tokens != nil {
			for range tokens {
			}
		}
		cancel()
	}
}

// --- Cancel via context.WithDeadline that elapses during stream —
// exercise the context-deadline path through the run() select. Three
// tokens emit before the deadline trips; remainder drained empty. ---

func BenchmarkScheduler_Cancel_DeadlineDuringStream(b *testing.B) {
	base := &cancellableBenchModel{tokens: benchTokens(32), perTokenNs: 100 * time.Microsecond}
	sched := New(base, Config{MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 4})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 500*time.Microsecond)
		_, tokens, err := sched.Schedule(ctx, inference.ScheduledRequest{Prompt: "p"})
		if err != nil {
			cancel()
			continue
		}
		count := 0
		for range tokens {
			count++
		}
		schedSinkTokensCount = count
		cancel()
	}
}

// --- Drain-after-cancel — the typical IDE pattern: cancel the
// request, then drain the channel to detect close. Captures the
// cost of the final j.out close + final probe emission. ---

func BenchmarkScheduler_Cancel_DrainAfterCancel_LongStream(b *testing.B) {
	base := &cancellableBenchModel{tokens: benchTokens(256), perTokenNs: 10 * time.Microsecond}
	sched := New(base, Config{MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 256})
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		handle, tokens, err := sched.Schedule(ctx, inference.ScheduledRequest{Prompt: "p"})
		if err != nil {
			continue
		}
		// Cancel immediately, then drain to close — no tokens may emit
		// before the cancel arm trips; this is the "fastest possible
		// rejection of an active stream" path.
		schedSinkCancel, schedSinkErr = sched.CancelRequest(ctx, handle.ID)
		count := 0
		for range tokens {
			count++
		}
		schedSinkTokensCount = count
	}
}
