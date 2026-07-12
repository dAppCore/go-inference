// SPDX-Licence-Identifier: EUPL-1.2

// Probe-sink throughput benchmarks. emitProbe fires on four event
// kinds per request (queued / start / first_token / complete) plus
// once on every CancelRequest. Each emit takes m.mu, reads queue
// depth + sink, releases, then calls sink.EmitProbe. The bench
// surface here:
//
//   * NoSink_Generate            - baseline: sink is nil, emitProbe
//                                  takes lock + checks nil, returns
//   * FastSink_Generate          - sink writes to a discard-counter,
//                                  no contention beyond emitProbe lock
//   * SlowSink_Generate          - sink acquires its own mutex per
//                                  event, simulates a serialising
//                                  metric exporter
//   * ManyProbeRequests_Cancel   - 64 Schedule+immediate-Cancel pairs
//                                  per b.N; cancel emits its own probe
//                                  in addition to the queued one
//   * NoSink_Generate_256Tokens  - sink-cost ablation against long
//                                  stream (4 probes spread across more
//                                  per-token work)
//
// Per the Wave 7 forward note: scheduler benches today run with
// ProbeSink: nil. This file makes the sink-cost dimension visible —
// nil vs fast vs slow — so future opt rounds can target the right
// thing (we know nil is cheap; how cheap is the cost gap?).
//
// Race-safe: every shared state is either atomic, owned by a single
// goroutine, or accessed only after b.StopTimer.
//
// Run:    go test -bench='BenchmarkScheduler_Probe' -benchmem -run='^$' ./go/scheduler

package scheduler

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"

	"dappco.re/go/inference"
)

// fastProbeSink is a counter-only sink — every EmitProbe is a single
// atomic increment. Captures the minimum work emitProbe can possibly
// hand off to under "no observability backend yet" conditions.
type fastProbeSink struct {
	count atomic.Int64
}

func (s *fastProbeSink) EmitProbe(_ inference.ProbeEvent) {
	s.count.Add(1)
}

// slowProbeSink holds a mutex across the body of EmitProbe, then
// does a trivial map insert + counter increment. Captures the cost
// when a serialising exporter (Prometheus pull, JSON-line log) is
// behind the sink. Real exporters are slower than this; this is a
// floor on the slow-sink cost.
type slowProbeSink struct {
	mu    sync.Mutex
	count int64
}

func (s *slowProbeSink) EmitProbe(event inference.ProbeEvent) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.count++
	// Touch a couple of event fields so the compiler can't DCE the
	// body. Reading the event is what a real exporter would do.
	if event.Scheduler != nil {
		s.count += int64(len(event.Labels))
	}
}

// --- Generate end-to-end under different sink shapes ---

func BenchmarkScheduler_Probe_NoSink_Generate_32Tokens(b *testing.B) {
	base := &schedBenchModel{tokens: benchTokens(32)}
	sched, _ := New(base, Config{MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 32, ProbeSink: nil})
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for range sched.Generate(ctx, "p") {
			count++
		}
		schedSinkTokensCount = count
	}
}

func BenchmarkScheduler_Probe_FastSink_Generate_32Tokens(b *testing.B) {
	base := &schedBenchModel{tokens: benchTokens(32)}
	sink := &fastProbeSink{}
	sched, _ := New(base, Config{MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 32, ProbeSink: sink})
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for range sched.Generate(ctx, "p") {
			count++
		}
		schedSinkTokensCount = count
	}
	b.StopTimer()
	_ = sink.count.Load()
}

func BenchmarkScheduler_Probe_SlowSink_Generate_32Tokens(b *testing.B) {
	base := &schedBenchModel{tokens: benchTokens(32)}
	sink := &slowProbeSink{}
	sched, _ := New(base, Config{MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 32, ProbeSink: sink})
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for range sched.Generate(ctx, "p") {
			count++
		}
		schedSinkTokensCount = count
	}
	b.StopTimer()
	sink.mu.Lock()
	_ = sink.count
	sink.mu.Unlock()
}

// --- 256-token variant — sink probes are constant per request (4 of
// them), but per-token cost grows with stream length. The ratio
// against 32-token measurements shows whether the sink dominates
// short streams or long streams. ---

func BenchmarkScheduler_Probe_NoSink_Generate_256Tokens(b *testing.B) {
	base := &schedBenchModel{tokens: benchTokens(256)}
	sched, _ := New(base, Config{MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 256, ProbeSink: nil})
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for range sched.Generate(ctx, "p") {
			count++
		}
		schedSinkTokensCount = count
	}
}

func BenchmarkScheduler_Probe_FastSink_Generate_256Tokens(b *testing.B) {
	base := &schedBenchModel{tokens: benchTokens(256)}
	sink := &fastProbeSink{}
	sched, _ := New(base, Config{MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 256, ProbeSink: sink})
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for range sched.Generate(ctx, "p") {
			count++
		}
		schedSinkTokensCount = count
	}
	b.StopTimer()
	_ = sink.count.Load()
}

// --- ManyProbeRequests via Schedule+Cancel — each pair emits at
// minimum a queued probe + a cancel probe; if the worker has picked
// the job up before the cancel arrives, also a start + cancelled
// probe. Captures the per-cancel emit cost at speed. ---

func BenchmarkScheduler_Probe_ManyProbeRequests_FastSink_ScheduleAndCancel(b *testing.B) {
	base := &cancellableBenchModel{tokens: benchTokens(32), perTokenNs: 50 * 1000} // 50µs per token
	sink := &fastProbeSink{}
	sched, _ := New(base, Config{MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 4, ProbeSink: sink})
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		handle, tokens, err := sched.Schedule(ctx, inference.ScheduledRequest{Prompt: "p"})
		if err != nil {
			continue
		}
		schedSinkCancel, schedSinkErr = sched.CancelRequest(ctx, handle.ID)
		for range tokens {
		}
	}
	b.StopTimer()
	_ = sink.count.Load()
}

// --- ProbeBus fan-out — wrap N sinks in a ProbeBus and measure the
// per-event fan-out cost. Real deployments often have a Prom sink +
// a JSON-log sink + a circuit-breaker sink behind one ProbeBus. ---

func BenchmarkScheduler_Probe_ProbeBusFanOut_3Sinks_Generate_32Tokens(b *testing.B) {
	base := &schedBenchModel{tokens: benchTokens(32)}
	sinkA := &fastProbeSink{}
	sinkB := &fastProbeSink{}
	sinkC := &fastProbeSink{}
	bus := inference.NewProbeBus(sinkA, sinkB, sinkC)
	sched, _ := New(base, Config{MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 32, ProbeSink: bus})
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for range sched.Generate(ctx, "p") {
			count++
		}
		schedSinkTokensCount = count
	}
	b.StopTimer()
	_ = sinkA.count.Load() + sinkB.count.Load() + sinkC.count.Load()
}

// --- SetProbeSink hot path — a deployment might swap the sink at
// runtime (rotating an exporter, switching from prod to debug). Each
// SetProbeSink takes m.mu. Measure the cost in isolation. ---

func BenchmarkScheduler_Probe_SetProbeSink(b *testing.B) {
	base := &schedBenchModel{tokens: benchTokens(1)}
	sched, _ := New(base, Config{MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 4})
	sink := &fastProbeSink{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sched.SetProbeSink(sink)
	}
}

func BenchmarkScheduler_Probe_SetProbeSink_Nil(b *testing.B) {
	base := &schedBenchModel{tokens: benchTokens(1)}
	sched, _ := New(base, Config{MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 4})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sched.SetProbeSink(nil)
	}
}
