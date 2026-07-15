// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the driver-neutral scheduler — Schedule/Generate
// roundtrip over an immediate-yielding base model, plus the pure
// helpers (generateOptions, cloneLabels, millis, millisString) that
// fire on every probe emission.
//
// Per AX-11 — Schedule + Generate run once per request, but
// emitProbe (and therefore cloneLabels + millisString) fires per
// scheduler event (queued / start / first_token / complete), and
// generateOptions is called once per dispatched job. With 20 in-flight
// requests on a 4-GPU box, each per-event helper compounds.
//
// Run:    go test -bench='BenchmarkScheduler' -benchmem -run='^$' ./go/scheduler

package scheduler

import (
	"context"
	"iter"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// Sinks defeat compiler DCE.
var (
	schedSinkOpts        []inference.GenerateOption
	schedSinkLabels      map[string]string
	schedSinkMillis      float64
	schedSinkMillisStr   string
	schedSinkHandle      inference.RequestHandle
	schedSinkCancel      inference.RequestCancelResult
	schedSinkErr         error
	schedSinkResult      core.Result
	schedSinkTokensCount int
	schedSinkModelType   string
	schedSinkInfo        inference.ModelInfo
	schedSinkMetrics     inference.GenerateMetrics
)

// schedBenchModel is a synchronous-iterator base model — yields the
// configured tokens immediately and returns. Safe to dispatch many
// Schedule calls against without leaking goroutines beyond the worker
// pool the bench creates once.
type schedBenchModel struct {
	tokens []inference.Token
}

func (m *schedBenchModel) Generate(_ context.Context, _ string, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.seq()
}

func (m *schedBenchModel) Chat(_ context.Context, _ []inference.Message, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.seq()
}

func (m *schedBenchModel) Classify(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.ClassifyResult(nil))
}

func (m *schedBenchModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.BatchResult(nil))
}

func (m *schedBenchModel) ModelType() string { return "sched-bench" }
func (m *schedBenchModel) Info() inference.ModelInfo {
	return inference.ModelInfo{Architecture: "qwen3"}
}
func (m *schedBenchModel) Metrics() inference.GenerateMetrics {
	return inference.GenerateMetrics{GeneratedTokens: len(m.tokens)}
}
func (m *schedBenchModel) Err() core.Result   { return core.Ok(nil) }
func (m *schedBenchModel) Close() core.Result { return core.Ok(nil) }

func (m *schedBenchModel) seq() iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		for _, token := range m.tokens {
			if !yield(token) {
				return
			}
		}
	}
}

func benchTokens(n int) []inference.Token {
	tokens := make([]inference.Token, n)
	for i := range n {
		tokens[i] = inference.Token{ID: int32(i + 1), Text: "tok"}
	}
	return tokens
}

// --- Generate end-to-end (Schedule + drain + close) ---

// 1 token — the dominant cost is queue+probe overhead, not token transfer.
func BenchmarkScheduler_Generate_1Token(b *testing.B) {
	base := &schedBenchModel{tokens: benchTokens(1)}
	sched, _ := New(base, Config{MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 4})
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for range sched.Generate(ctx, "prompt") {
			count++
		}
		schedSinkTokensCount = count
	}
}

// 32 tokens — closer to a realistic chat reply.
func BenchmarkScheduler_Generate_32Tokens(b *testing.B) {
	base := &schedBenchModel{tokens: benchTokens(32)}
	sched, _ := New(base, Config{MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 32})
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for range sched.Generate(ctx, "prompt") {
			count++
		}
		schedSinkTokensCount = count
	}
}

// 256 tokens — long reply; per-token label clone is the inner hot path.
func BenchmarkScheduler_Generate_256Tokens(b *testing.B) {
	base := &schedBenchModel{tokens: benchTokens(256)}
	sched, _ := New(base, Config{MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 256})
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for range sched.Generate(ctx, "prompt") {
			count++
		}
		schedSinkTokensCount = count
	}
}

// --- Schedule (just the handle return, no token drain) ---

func BenchmarkScheduler_Schedule_1Token(b *testing.B) {
	base := &schedBenchModel{tokens: benchTokens(1)}
	sched, _ := New(base, Config{MaxConcurrent: 1, MaxQueue: 32, StreamBuffer: 4})
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		handle, tokens, err := sched.Schedule(ctx, inference.ScheduledRequest{Prompt: "p"})
		schedSinkHandle = handle
		schedSinkErr = err
		// drain before next iteration so the queue doesn't fill.
		for range tokens {
		}
	}
}

// --- CancelRequest (no-active-id fallback) ---

func BenchmarkScheduler_CancelRequest_NotFound(b *testing.B) {
	base := &schedBenchModel{tokens: benchTokens(1)}
	sched, _ := New(base, Config{MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 4})
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		schedSinkCancel, schedSinkErr = sched.CancelRequest(ctx, "no-such-id")
	}
}

// --- Delegators (Classify / BatchGenerate / Info / Metrics / ModelType):
// the scheduler wraps the base model with only a nil guard, so these
// measure the scheduler-layer overhead on top of the base call. ---

func BenchmarkScheduler_Classify(b *testing.B) {
	base := &schedBenchModel{tokens: benchTokens(1)}
	sched, _ := New(base, Config{MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 4})
	ctx := context.Background()
	prompts := []string{"alpha", "beta", "gamma"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		schedSinkResult = sched.Classify(ctx, prompts)
	}
}

func BenchmarkScheduler_BatchGenerate(b *testing.B) {
	base := &schedBenchModel{tokens: benchTokens(1)}
	sched, _ := New(base, Config{MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 4})
	ctx := context.Background()
	prompts := []string{"alpha", "beta", "gamma"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		schedSinkResult = sched.BatchGenerate(ctx, prompts)
	}
}

// Info / Metrics / ModelType are nil-guarded value passthroughs — one
// bench proves the scheduler layer adds no allocation over the base read.
func BenchmarkScheduler_Accessors_InfoMetricsModelType(b *testing.B) {
	base := &schedBenchModel{tokens: benchTokens(1)}
	sched, _ := New(base, Config{MaxConcurrent: 1, MaxQueue: 4, StreamBuffer: 4})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		schedSinkInfo = sched.Info()
		schedSinkMetrics = sched.Metrics()
		schedSinkModelType = sched.ModelType()
	}
}

// --- generateOptions: capability matching — 1, 4, 16 sampler-fields
// populated (covers the spec's "capability sets of 1, 4, 16 GPUs" lens
// for the option-set the scheduler emits per dispatched job). ---

func BenchmarkScheduler_GenerateOptions_1Field(b *testing.B) {
	cfg := inference.SamplerConfig{MaxTokens: 64}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		schedSinkOpts = generateOptions(cfg)
	}
}

func BenchmarkScheduler_GenerateOptions_4Fields(b *testing.B) {
	cfg := inference.SamplerConfig{
		MaxTokens:   64,
		Temperature: 0.7,
		TopK:        40,
		TopP:        0.9,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		schedSinkOpts = generateOptions(cfg)
	}
}

// Full — every field populated; 16 stop tokens stand in for the
// "capability set of 16" knob mentioned in the spec.
func BenchmarkScheduler_GenerateOptions_FullSamplerWith16StopTokens(b *testing.B) {
	cfg := inference.SamplerConfig{
		MaxTokens:     64,
		Temperature:   0.7,
		TopK:          40,
		TopP:          0.9,
		RepeatPenalty: 1.1,
		StopTokens:    []int32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
		ReturnLogits:  true,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		schedSinkOpts = generateOptions(cfg)
	}
}

// --- cloneLabels: fires per emitted token via the run loop ---

func BenchmarkScheduler_CloneLabels_Empty(b *testing.B) {
	labels := map[string]string{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		schedSinkLabels = cloneLabels(labels)
	}
}

func BenchmarkScheduler_CloneLabels_OneEntry(b *testing.B) {
	labels := map[string]string{"request_id": "req-42"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		schedSinkLabels = cloneLabels(labels)
	}
}

func BenchmarkScheduler_CloneLabels_FiveEntries(b *testing.B) {
	labels := map[string]string{
		"request_id": "req-42",
		"tenant":     "lab",
		"priority":   "high",
		"feature":    "ide-chat",
		"agent":      "cladius",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		schedSinkLabels = cloneLabels(labels)
	}
}

func BenchmarkScheduler_CloneLabels_TwentyEntries(b *testing.B) {
	labels := map[string]string{}
	for i := range 20 {
		labels[(string)(rune('a'+i))] = "v"
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		schedSinkLabels = cloneLabels(labels)
	}
}

// --- millis + millisString (per probe-event call) ---

func BenchmarkScheduler_Millis_Positive(b *testing.B) {
	d := 45 * time.Millisecond
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		schedSinkMillis = millis(d)
	}
}

func BenchmarkScheduler_Millis_Zero(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		schedSinkMillis = millis(0)
	}
}

func BenchmarkScheduler_MillisString_Positive(b *testing.B) {
	d := 45 * time.Millisecond
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		schedSinkMillisStr = millisString(d)
	}
}
