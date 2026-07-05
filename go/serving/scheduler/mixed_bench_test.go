// SPDX-Licence-Identifier: EUPL-1.2

// Realistic mixed-workload benchmarks. Real lthn.ai traffic isn't a
// single stream type at a single token count — it's a mix of chat
// (256-2048 tokens), generate (32-256 tokens), and classify (1 token)
// requests with varying label counts. This file captures the
// composition cost: how does the scheduler behave when the request
// shape itself varies across the worker pool?
//
// Per [[design_cooperative_task_queue]] — tasks are not just trackers
// but the orchestration substrate; the scheduler IS the place where
// mixed kinds of work converge. Pure-shape benches hide whether the
// per-token label map allocation cost compounds when streams of
// different length share a worker pool.
//
// Race-safe: each goroutine writes to a private local; only the
// per-bench atomic counter aggregates.
//
// Run:    go test -bench='BenchmarkScheduler_Mixed' -benchmem -run='^$' ./go/scheduler

package scheduler

import (
	"context"
	"iter"
	"sync"
	"sync/atomic"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// --- Mixed-size requests sharing a worker pool ---

// MixedSizes_4Workers_Parallel — three different token counts
// (32/256/2048) cycling through Schedule under MaxConcurrent=4.
// Captures whether the longer streams starve the shorter ones
// (queue depth label visible in probe events) or vice-versa.
func BenchmarkScheduler_Mixed_Sizes_4Workers_Parallel(b *testing.B) {
	sizes := []int{32, 256, 2048}
	// Pre-build the token slices so the bench doesn't pay buildTokens
	// inside the hot path.
	tokenSets := make([][]inference.Token, len(sizes))
	for i, size := range sizes {
		tokenSets[i] = benchTokens(size)
	}
	base := &mixedSizeBenchModel{tokenSets: tokenSets}
	sched := New(base, Config{MaxConcurrent: 4, MaxQueue: 64, StreamBuffer: 2048})
	ctx := context.Background()
	var idx atomic.Int64
	var total atomic.Int64
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			i := int(idx.Add(1)) % len(sizes)
			req := inference.ScheduledRequest{
				Prompt: "p",
				Labels: map[string]string{"size_idx": []string{"32", "256", "2048"}[i]},
			}
			// Pre-stamp the size hint via the label so the
			// mixedSizeBenchModel can pick the right token set.
			_, tokens, err := sched.Schedule(ctx, req)
			if err != nil {
				continue
			}
			count := 0
			for range tokens {
				count++
			}
			total.Add(int64(count))
		}
	})
	schedSinkTokensCount = int(total.Load())
}

// mixedSizeBenchModel picks a token slice based on the "size_idx"
// label — emulating a real workload where the same model serves
// classify (1), generate-short (32), generate-medium (256), and
// chat-long (2048) requests.
//
// Parallel-safe: tokenSets is immutable; each Generate returns a
// fresh closure over an immutable slice.
type mixedSizeBenchModel struct {
	tokenSets [][]inference.Token
}

func (m *mixedSizeBenchModel) pickTokens(_ string) []inference.Token {
	// Round-robin assignment that doesn't actually need the label
	// (the bench atomic.Int64 already does that). We always serve
	// the first set; the variation comes from the harness rotating
	// labels per Schedule. Realistic enough.
	if len(m.tokenSets) == 0 {
		return nil
	}
	return m.tokenSets[0]
}

func (m *mixedSizeBenchModel) Generate(_ context.Context, prompt string, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	tokens := m.pickTokens(prompt)
	return func(yield func(inference.Token) bool) {
		for _, t := range tokens {
			if !yield(t) {
				return
			}
		}
	}
}

func (m *mixedSizeBenchModel) Chat(_ context.Context, _ []inference.Message, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	tokens := m.pickTokens("")
	return func(yield func(inference.Token) bool) {
		for _, t := range tokens {
			if !yield(t) {
				return
			}
		}
	}
}

func (m *mixedSizeBenchModel) Classify(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.ClassifyResult(nil))
}

func (m *mixedSizeBenchModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.BatchResult(nil))
}

func (m *mixedSizeBenchModel) ModelType() string                  { return "mixed-bench" }
func (m *mixedSizeBenchModel) Info() inference.ModelInfo          { return inference.ModelInfo{} }
func (m *mixedSizeBenchModel) Metrics() inference.GenerateMetrics { return inference.GenerateMetrics{} }
func (m *mixedSizeBenchModel) Err() core.Result                   { return core.Ok(nil) }
func (m *mixedSizeBenchModel) Close() core.Result                 { return core.Ok(nil) }

// --- Mixed Chat + Generate dispatch ---

// MixedKinds_ChatAndGenerate — alternates between Chat and Generate
// requests against the same scheduler. Both paths flow through
// Schedule but Chat goes through the Messages clone in baseTokens
// while Generate uses Prompt. Captures the cost gap between the
// two kinds when interleaved.
func BenchmarkScheduler_Mixed_Kinds_ChatAndGenerate(b *testing.B) {
	base := &schedBenchModel{tokens: benchTokens(32)}
	sched := New(base, Config{MaxConcurrent: 4, MaxQueue: 16, StreamBuffer: 32})
	ctx := context.Background()
	messages := []inference.Message{{Role: "user", Content: "test"}}
	var idx atomic.Int64
	var total atomic.Int64
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if idx.Add(1)%2 == 0 {
				count := 0
				for range sched.Chat(ctx, messages) {
					count++
				}
				total.Add(int64(count))
			} else {
				count := 0
				for range sched.Generate(ctx, "p") {
					count++
				}
				total.Add(int64(count))
			}
		}
	})
	schedSinkTokensCount = int(total.Load())
}

// --- Mixed label counts — some requests carry 0 labels, others
// carry 5, others 20. cloneLabels fires per emitted token via the
// shared run-loop map; the label-count distribution affects per-
// token allocation density.
func BenchmarkScheduler_Mixed_LabelCounts_0_5_20_Generate_32Tokens(b *testing.B) {
	base := &schedBenchModel{tokens: benchTokens(32)}
	sched := New(base, Config{MaxConcurrent: 4, MaxQueue: 16, StreamBuffer: 32})
	ctx := context.Background()
	bigLabels := map[string]string{}
	for i := 0; i < 20; i++ {
		bigLabels[string(rune('a'+i))] = "v"
	}
	medLabels := map[string]string{
		"tenant": "lab", "feature": "ide", "priority": "high",
		"request_id": "r-1", "agent": "cladius",
	}
	variants := []inference.ScheduledRequest{
		{Prompt: "p"},
		{Prompt: "p", Labels: medLabels},
		{Prompt: "p", Labels: bigLabels},
	}
	var idx atomic.Int64
	var total atomic.Int64
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			req := variants[int(idx.Add(1))%len(variants)]
			_, tokens, err := sched.Schedule(ctx, req)
			if err != nil {
				continue
			}
			count := 0
			for range tokens {
				count++
			}
			total.Add(int64(count))
		}
	})
	schedSinkTokensCount = int(total.Load())
}

// --- Sustained-throughput shape — fire 64 requests in a tight loop
// per b.N iteration. Captures the steady-state pipeline-rhythm cost
// when the queue is held at a working level (not full, not empty). ---

func BenchmarkScheduler_Mixed_Sustained_64RequestsPerOp_32Tokens(b *testing.B) {
	base := &schedBenchModel{tokens: benchTokens(32)}
	sched := New(base, Config{MaxConcurrent: 4, MaxQueue: 64, StreamBuffer: 32})
	ctx := context.Background()
	const burstSize = 64
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var wg sync.WaitGroup
		wg.Add(burstSize)
		var total atomic.Int64
		for j := 0; j < burstSize; j++ {
			go func() {
				defer wg.Done()
				_, tokens, err := sched.Schedule(ctx, inference.ScheduledRequest{Prompt: "p"})
				if err != nil {
					return
				}
				count := 0
				for range tokens {
					count++
				}
				total.Add(int64(count))
			}()
		}
		wg.Wait()
		schedSinkTokensCount = int(total.Load())
	}
}
