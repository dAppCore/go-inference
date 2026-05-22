// SPDX-Licence-Identifier: EUPL-1.2

package scheduler

import (
	"context"
	"iter"
	"testing"
	"time"

	"dappco.re/go/inference"
)

// benchModel is a minimal inference.TextModel that emits a pre-built
// token slice via iter.Seq[Token] with zero per-call allocations beyond
// the closure itself. Used by every scheduler bench so the bench
// measures scheduler overhead, not driver work.
type benchModel struct {
	tokens []inference.Token
}

func newBenchModel(n int) *benchModel {
	tokens := make([]inference.Token, n)
	for i := range tokens {
		tokens[i] = inference.Token{Text: "t"}
	}
	return &benchModel{tokens: tokens}
}

func (m *benchModel) seq() iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		for _, token := range m.tokens {
			if !yield(token) {
				return
			}
		}
	}
}

func (m *benchModel) Generate(context.Context, string, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.seq()
}

func (m *benchModel) Chat(context.Context, []inference.Message, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.seq()
}

func (m *benchModel) Classify(context.Context, []string, ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	return nil, nil
}

func (m *benchModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) ([]inference.BatchResult, error) {
	return nil, nil
}

func (m *benchModel) ModelType() string             { return "bench" }
func (m *benchModel) Info() inference.ModelInfo     { return inference.ModelInfo{Architecture: "bench"} }
func (m *benchModel) Metrics() inference.GenerateMetrics { return inference.GenerateMetrics{} }
func (m *benchModel) Err() error                    { return nil }
func (m *benchModel) Close() error                  { return nil }

// drainHandle consumes a scheduled token channel until close — the
// canonical scheduler client loop.
func drainHandle(tokens <-chan inference.ScheduledToken) int {
	count := 0
	for range tokens {
		count++
	}
	return count
}

// BenchmarkSchedule_Generate_Small measures the per-request scheduler
// overhead on a tiny stream — alloc-and-completion dominates over
// per-token cost. Latency floor.
func BenchmarkSchedule_Generate_Small(b *testing.B) {
	base := newBenchModel(8)
	sched := New(base, Config{MaxConcurrent: 1, MaxQueue: 1, StreamBuffer: 8})
	ctx := context.Background()
	b.ReportAllocs()
	b.SetBytes(int64(len(base.tokens)))
	for b.Loop() {
		_, tokens, err := sched.Schedule(ctx, inference.ScheduledRequest{Prompt: "p"})
		if err != nil {
			b.Fatal(err)
		}
		drainHandle(tokens)
	}
}

// BenchmarkSchedule_Generate_Large measures per-token costs on a long
// stream where stream-buffer + label-clone-per-token dominate.
func BenchmarkSchedule_Generate_Large(b *testing.B) {
	base := newBenchModel(512)
	sched := New(base, Config{MaxConcurrent: 1, MaxQueue: 1, StreamBuffer: 32})
	ctx := context.Background()
	b.ReportAllocs()
	b.SetBytes(int64(len(base.tokens)))
	for b.Loop() {
		_, tokens, err := sched.Schedule(ctx, inference.ScheduledRequest{Prompt: "p"})
		if err != nil {
			b.Fatal(err)
		}
		drainHandle(tokens)
	}
}

// BenchmarkSchedule_Chat measures the chat path which clones the
// messages slice on enqueue + on baseTokens.
func BenchmarkSchedule_Chat(b *testing.B) {
	base := newBenchModel(64)
	sched := New(base, Config{MaxConcurrent: 1, MaxQueue: 1, StreamBuffer: 16})
	msgs := []inference.Message{
		{Role: "system", Content: "you are a benchmark"},
		{Role: "user", Content: "go"},
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.SetBytes(int64(len(base.tokens)))
	for b.Loop() {
		_, tokens, err := sched.Schedule(ctx, inference.ScheduledRequest{Messages: msgs})
		if err != nil {
			b.Fatal(err)
		}
		drainHandle(tokens)
	}
}

// BenchmarkSchedule_WithLabels measures the cost of carrying labels
// through enqueue + per-token label clone path.
func BenchmarkSchedule_WithLabels(b *testing.B) {
	base := newBenchModel(64)
	sched := New(base, Config{MaxConcurrent: 1, MaxQueue: 1, StreamBuffer: 16})
	labels := map[string]string{
		"tenant":  "bench",
		"session": "abc-123",
		"trace":   "deadbeef",
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.SetBytes(int64(len(base.tokens)))
	for b.Loop() {
		_, tokens, err := sched.Schedule(ctx, inference.ScheduledRequest{Prompt: "p", Labels: labels})
		if err != nil {
			b.Fatal(err)
		}
		drainHandle(tokens)
	}
}

// BenchmarkSchedule_GenerateIter measures the Generate iter.Seq path
// which clients use most.
func BenchmarkSchedule_GenerateIter(b *testing.B) {
	base := newBenchModel(64)
	sched := New(base, Config{MaxConcurrent: 1, MaxQueue: 1, StreamBuffer: 16})
	ctx := context.Background()
	b.ReportAllocs()
	b.SetBytes(int64(len(base.tokens)))
	for b.Loop() {
		for range sched.Generate(ctx, "p") {
		}
	}
}

// BenchmarkSchedule_ChatIter measures the Chat iter.Seq path.
func BenchmarkSchedule_ChatIter(b *testing.B) {
	base := newBenchModel(64)
	sched := New(base, Config{MaxConcurrent: 1, MaxQueue: 1, StreamBuffer: 16})
	msgs := []inference.Message{{Role: "user", Content: "hi"}}
	ctx := context.Background()
	b.ReportAllocs()
	b.SetBytes(int64(len(base.tokens)))
	for b.Loop() {
		for range sched.Chat(ctx, msgs) {
		}
	}
}

// BenchmarkSchedule_Concurrent measures parallel client throughput
// where multiple goroutines enqueue + drain at once. Stresses queue
// admission + worker scheduling. Queue is sized to accept burst from
// every parallel goroutine (b.RunParallel × GOMAXPROCS) so the bench
// measures the happy path, not reject behaviour — that's covered by
// BenchmarkSchedule_QueueFullReject.
func BenchmarkSchedule_Concurrent(b *testing.B) {
	base := newBenchModel(32)
	sched := New(base, Config{MaxConcurrent: 4, MaxQueue: 4096, StreamBuffer: 8})
	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		ctx := context.Background()
		for pb.Next() {
			_, tokens, err := sched.Schedule(ctx, inference.ScheduledRequest{Prompt: "p"})
			if err != nil {
				b.Fatal(err)
			}
			drainHandle(tokens)
		}
	})
}

// BenchmarkSchedule_QueueFullReject measures the queue-overflow reject
// path — what happens when a misconfigured client floods a tiny queue.
// Uses a blocking model that parks until released, so the worker is
// busy + the queue slot is taken + further Schedules hit default-reject.
func BenchmarkSchedule_QueueFullReject(b *testing.B) {
	blocking := newBlockingModel()
	sched := New(blocking, Config{MaxConcurrent: 1, MaxQueue: 1, StreamBuffer: 0})
	ctx := context.Background()
	// Fill the in-flight worker slot.
	_, active, err := sched.Schedule(ctx, inference.ScheduledRequest{Prompt: "active"})
	if err != nil {
		b.Fatal(err)
	}
	if got := <-blocking.started; got != "active" {
		b.Fatalf("started = %q, want active", got)
	}
	// Fill the single queue slot.
	_, queued, err := sched.Schedule(ctx, inference.ScheduledRequest{Prompt: "queued"})
	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		blocking.release <- struct{}{}
		drainHandle(active)
		blocking.release <- struct{}{}
		drainHandle(queued)
	}()
	b.ReportAllocs()
	for b.Loop() {
		_, _, err := sched.Schedule(ctx, inference.ScheduledRequest{Prompt: "overflow"})
		if err == nil {
			b.Fatal("expected queue-full error")
		}
	}
}

// BenchmarkSchedule_Cancel measures the cancel hot path — register +
// queue + cancel + cleanup. Used heavily by IDE-style abort flows.
func BenchmarkSchedule_Cancel(b *testing.B) {
	base := newBenchModel(64)
	sched := New(base, Config{MaxConcurrent: 1, MaxQueue: 32, StreamBuffer: 8})
	ctx := context.Background()
	b.ReportAllocs()
	for b.Loop() {
		handle, tokens, err := sched.Schedule(ctx, inference.ScheduledRequest{Prompt: "p"})
		if err != nil {
			b.Fatal(err)
		}
		_, _ = sched.CancelRequest(ctx, handle.ID)
		drainHandle(tokens)
	}
}

// BenchmarkCloneLabels_Empty measures the per-emit baseline (empty
// labels — every token still allocates).
func BenchmarkCloneLabels_Empty(b *testing.B) {
	var in map[string]string
	b.ReportAllocs()
	for b.Loop() {
		_ = cloneLabels(in)
	}
}

// BenchmarkCloneLabels_Three measures the three-key common case (the
// labels set carried by tokens in BenchmarkSchedule_WithLabels).
func BenchmarkCloneLabels_Three(b *testing.B) {
	in := map[string]string{
		"tenant":  "bench",
		"session": "abc-123",
		"trace":   "deadbeef",
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = cloneLabels(in)
	}
}

// BenchmarkGenerateOptions_Full builds the GenerateOption slice for a
// fully-populated SamplerConfig — invoked once per worker enqueue,
// hot per request.
func BenchmarkGenerateOptions_Full(b *testing.B) {
	cfg := inference.SamplerConfig{
		MaxTokens:     128,
		Temperature:   0.7,
		TopK:          50,
		TopP:          0.95,
		RepeatPenalty: 1.05,
		StopTokens:    []int32{1, 2, 3},
		ReturnLogits:  true,
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = generateOptions(cfg)
	}
}

// BenchmarkGenerateOptions_Minimal measures the common-case lower
// bound — only temperature is set.
func BenchmarkGenerateOptions_Minimal(b *testing.B) {
	cfg := inference.SamplerConfig{Temperature: 0.7}
	b.ReportAllocs()
	for b.Loop() {
		_ = generateOptions(cfg)
	}
}

// BenchmarkMillisString measures the per-token label-format call. This
// runs twice per emitted token (queue + first-token latency).
func BenchmarkMillisString(b *testing.B) {
	d := 12345678 * time.Nanosecond
	b.ReportAllocs()
	for b.Loop() {
		_ = millisString(d)
	}
}
