// SPDX-Licence-Identifier: EUPL-1.2

// Error-propagation benchmarks. Three paths bubble errors through
// the scheduler:
//
//   1. Schedule fast-fail — nil model, nil context (post-cancel),
//      queue full. These return early without registering a job.
//   2. setErr / m.lastErr — Generate hits Schedule failure, calls
//      m.setErr(err); the next Err() reflects it.
//   3. m.base.Err() bubble — at end of run(), if the base model
//      reports an error, setErr captures it. Then Err() walks
//      lastErr first, base.Err() second.
//
// The existing CancelRequest_NotFound bench covers one happy-no-op
// path. This file covers the error-active paths so the rare-failure
// rhythm has measured cost.
//
// Run:    go test -bench='BenchmarkScheduler_ErrProp' -benchmem -run='^$' ./go/scheduler

package scheduler

import (
	"context"
	"iter"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// errBaseModel reports a persistent error via Err(). Used to bench
// the m.base.Err() bubble path through Generate's iter loop.
type errBaseModel struct {
	tokens []inference.Token
	err    error
}

func (m *errBaseModel) Generate(_ context.Context, _ string, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.seq()
}

func (m *errBaseModel) Chat(_ context.Context, _ []inference.Message, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.seq()
}

func (m *errBaseModel) Classify(context.Context, []string, ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	return nil, m.err
}

func (m *errBaseModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) ([]inference.BatchResult, error) {
	return nil, m.err
}

func (m *errBaseModel) ModelType() string                  { return "err-base" }
func (m *errBaseModel) Info() inference.ModelInfo          { return inference.ModelInfo{} }
func (m *errBaseModel) Metrics() inference.GenerateMetrics { return inference.GenerateMetrics{} }
func (m *errBaseModel) Err() error                         { return m.err }
func (m *errBaseModel) Close() error                       { return nil }

func (m *errBaseModel) seq() iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		for _, token := range m.tokens {
			if !yield(token) {
				return
			}
		}
	}
}

// --- Schedule on nil-model receiver — the m == nil || m.base == nil
// guard at Schedule entry. Single allocation for the core.NewError. ---

func BenchmarkScheduler_ErrProp_Schedule_NilModel(b *testing.B) {
	var sched *Model
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		handle, tokens, err := sched.Schedule(ctx, inference.ScheduledRequest{Prompt: "p"})
		schedSinkHandle = handle
		schedSinkErr = err
		_ = tokens
	}
}

// --- Schedule with a nil base.TextModel inside the scheduler — same
// guard but reaches it via New(nil, ...). Confirms the nil-receiver
// path doesn't hit a different cost shape. ---

func BenchmarkScheduler_ErrProp_Schedule_NilBaseInsideScheduler(b *testing.B) {
	sched := New(nil, Config{MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 4})
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		handle, tokens, err := sched.Schedule(ctx, inference.ScheduledRequest{Prompt: "p"})
		schedSinkHandle = handle
		schedSinkErr = err
		_ = tokens
	}
}

// --- Err() on a freshly-constructed scheduler — should return nil
// because lastErr is nil and base.Err() is nil. Walks m.mu + checks. ---

func BenchmarkScheduler_ErrProp_Err_Nil(b *testing.B) {
	base := &schedBenchModel{tokens: benchTokens(1)}
	sched := New(base, Config{MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 4})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		schedSinkErr = sched.Err()
	}
}

// --- Err() when m.lastErr is populated — setErr() path. We force
// lastErr by closing the base then calling setErr ourselves via
// Generate(failing).
//
// Actually the simplest way to set lastErr is to use a nil-model
// Generate loop, which calls m.setErr inside Generate. After that
// Err() returns the cached lastErr without walking to base.Err. ---

func BenchmarkScheduler_ErrProp_Err_LastErrCached(b *testing.B) {
	sched := New(nil, Config{MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 4})
	// Trigger setErr via Generate's nil-model failure path.
	for range sched.Generate(context.Background(), "p") {
		break
	}
	if sched.Err() == nil {
		b.Fatalf("expected lastErr to be set after nil-model Generate")
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		schedSinkErr = sched.Err()
	}
}

// --- Err() when only base.Err() returns an error — lastErr is nil,
// the m.base.Err() fallback path returns the persistent base error. ---

func BenchmarkScheduler_ErrProp_Err_BaseErrFallback(b *testing.B) {
	base := &errBaseModel{tokens: benchTokens(1), err: core.NewError("scheduler-bench: base failed")}
	sched := New(base, Config{MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 4})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		schedSinkErr = sched.Err()
	}
}

// --- Generate full loop into a base that reports Err() after the
// stream completes — the m.base.Err() bubble at end-of-run captures
// the error into setErr. Each iteration runs a fresh Generate so the
// timing per iter includes the full happy stream + the err catch. ---

func BenchmarkScheduler_ErrProp_Generate_BaseReportsErrAtEnd_32Tokens(b *testing.B) {
	base := &errBaseModel{tokens: benchTokens(32), err: core.NewError("scheduler-bench: base reported err")}
	sched := New(base, Config{MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 32})
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

// --- Schedule with an empty request ID — the nextRequestID() path is
// triggered. Existing benches cover the happy path where ID is empty
// but tokens are 1; this one's an explicit ID-gen-and-discard probe. ---

func BenchmarkScheduler_ErrProp_Schedule_EmptyIDGeneratesID(b *testing.B) {
	base := &schedBenchModel{tokens: benchTokens(1)}
	sched := New(base, Config{MaxConcurrent: 1, MaxQueue: 32, StreamBuffer: 4, RequestIDPrefix: "errprop"})
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		handle, tokens, err := sched.Schedule(ctx, inference.ScheduledRequest{Prompt: "p"})
		schedSinkHandle = handle
		schedSinkErr = err
		for range tokens {
		}
	}
}

// --- Schedule with a pre-populated ID — the core.Trim(req.ID) != ""
// arm short-circuits ID generation. The cost gap against EmptyID
// reflects the nextRequestID() hand-built path's contribution. ---

func BenchmarkScheduler_ErrProp_Schedule_PreSetID(b *testing.B) {
	base := &schedBenchModel{tokens: benchTokens(1)}
	sched := New(base, Config{MaxConcurrent: 1, MaxQueue: 32, StreamBuffer: 4})
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		handle, tokens, err := sched.Schedule(ctx, inference.ScheduledRequest{ID: "pre-set", Prompt: "p"})
		schedSinkHandle = handle
		schedSinkErr = err
		for range tokens {
		}
	}
}
