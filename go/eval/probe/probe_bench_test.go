// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the probe package — Event clone, Bus fanout, Recorder
// emit, SinkFunc dispatch. Per AX-11 — these fire per probe emitted
// during generation/training. A modest decode loop with logits +
// cache + memory probes fires 4-5 events per generated token; a
// training run fires thousands per epoch. CloneEvent is the inner-
// loop deep-copy used by every Bus and Recorder emit.
//
// Run:    go test -bench='BenchmarkProbe' -benchmem -run='^$' ./go/probe

package probe

import (
	"testing"
)

// Sinks defeat compiler DCE.
var (
	probeBenchSinkEvent  Event
	probeBenchSinkEvents []Event
)

// benchProbeEvent builds a representative Event with the payloads a
// decode-step probe carries: logits + entropy + cache + memory + meta.
// Mirrors the fixture in TestCloneEvent_DefensiveCopiesAllPayloads_Good
// but in bench-fixture style.
func benchProbeEvent() Event {
	return Event{
		Kind:  KindLogits,
		Phase: PhaseDecode,
		Step:  42,
		Token: &Token{ID: 7, Text: "answer", PromptTokens: 256, GeneratedTokens: 12},
		Logits: &Logits{
			Shape:      []int32{1, 1, 151936},
			VocabSize:  151936,
			MaxTokenID: 7,
			MaxLogit:   4.5,
			MinTokenID: 11,
			MinLogit:   -3.2,
			MeanLogit:  0.05,
			Top: []Logit{
				{TokenID: 7, Logit: 4.5, Probability: 0.42},
				{TokenID: 9, Logit: 4.2, Probability: 0.31},
				{TokenID: 11, Logit: 3.9, Probability: 0.18},
				{TokenID: 13, Logit: 3.7, Probability: 0.05},
				{TokenID: 15, Logit: 3.5, Probability: 0.04},
			},
			Meta: map[string]string{"sampler": "topk"},
		},
		Entropy: &Entropy{Value: 1.2, Unit: "nats"},
		Cache: &CachePressure{
			PromptTokens:    256,
			GeneratedTokens: 12,
			LayerCount:      28,
			CacheTokens:     268,
			ProcessedTokens: 268,
			MaxCacheTokens:  40960,
			Utilization:     0.0065,
		},
		Memory: &MemoryPressure{ActiveBytes: 4 << 30, PeakBytes: 6 << 30, CacheBytes: 1 << 30},
		Meta:   map[string]string{"run_id": "0xabc", "step": "42", "lane": "decode"},
	}
}

// --- CloneEvent ---
// Minimal — only Kind+Step set; no payloads or meta. Measures the
// fast path through the per-field nil checks.

func BenchmarkProbe_CloneEvent_Minimal(b *testing.B) {
	event := Event{Kind: KindToken, Step: 1}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeBenchSinkEvent = CloneEvent(event)
	}
}

// Typical decode-step shape — token + logits + entropy + cache +
// memory + meta. Hits every payload-clone branch.
func BenchmarkProbe_CloneEvent_TypicalDecode(b *testing.B) {
	event := benchProbeEvent()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeBenchSinkEvent = CloneEvent(event)
	}
}

// Training event shape — much smaller, only Training + Meta.
func BenchmarkProbe_CloneEvent_Training(b *testing.B) {
	event := Event{
		Kind:  KindTraining,
		Phase: PhaseTraining,
		Step:  100,
		Training: &Training{
			Epoch:        2,
			Step:         100,
			Loss:         0.25,
			LearningRate: 3e-4,
			GradNorm:     0.42,
		},
		Meta: map[string]string{"run": "sft", "step": "100"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeBenchSinkEvent = CloneEvent(event)
	}
}

// Router-decision shape — MoE / expert-residency probes.
func BenchmarkProbe_CloneEvent_Router(b *testing.B) {
	event := Event{
		Kind:  KindRouterDecision,
		Phase: PhaseDecode,
		Step:  10,
		RouterDecision: &RouterDecision{
			Layer:       12,
			TokenID:     7,
			ExpertIDs:   []int{3, 17, 28, 41},
			Weights:     []float32{0.42, 0.31, 0.18, 0.09},
			Temperature: 1.0,
		},
		ExpertResidency: &ExpertResidency{
			Action:             ExpertResidencyActionPageIn,
			Layer:              12,
			ExpertIDs:          []int{3, 17},
			ResidentExperts:    16,
			MaxResidentExperts: 32,
			LoadedBytes:        128 << 20,
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeBenchSinkEvent = CloneEvent(event)
	}
}

// Heads-coherence shape — exercises HeadSelection +
// LayerCoherence + Residual clone branches.
func BenchmarkProbe_CloneEvent_HeadsAndResidual(b *testing.B) {
	heads := make([]int, 16)
	scores := make([]float64, 16)
	for i := range heads {
		heads[i] = i
		scores[i] = float64(i) / 16
	}
	event := Event{
		Kind:  KindSelectedHeads,
		Phase: PhaseDecode,
		Step:  5,
		SelectedHeads: &HeadSelection{
			Layer:  12,
			Heads:  heads,
			Scores: scores,
		},
		LayerCoherence: &LayerCoherence{
			Layer:          12,
			KeyCoherence:   0.5,
			ValueCoherence: 0.6,
			CrossAlignment: 0.55,
			KVCoupling:     0.7,
			HeadEntropy:    1.1,
			PhaseLock:      0.42,
		},
		Residual: &ResidualSummary{
			Layer:    12,
			Mean:     0.01,
			Variance: 0.02,
			RMS:      0.15,
			L2Norm:   12.3,
			MaxAbs:   1.8,
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeBenchSinkEvent = CloneEvent(event)
	}
}

// --- Recorder.EmitProbe ---
// One Recorder, many emits (per probe call). Each emit deep-copies
// through CloneEvent and appends under the recorder lock.

func BenchmarkProbe_Recorder_EmitProbe(b *testing.B) {
	rec := NewRecorder()
	event := benchProbeEvent()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rec.EmitProbe(event)
	}
}

// --- Recorder.Events ---
// Read-side — copies the recorder buffer out. Bench against a
// pre-populated recorder shaped like a single-prompt decode loop
// (one event per generated token, 128 tokens).

func BenchmarkProbe_Recorder_Events_128(b *testing.B) {
	rec := NewRecorder()
	event := benchProbeEvent()
	for range 128 {
		rec.EmitProbe(event)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeBenchSinkEvents = rec.Events()
	}
}

// --- Bus.EmitProbe ---
// Fanout to N sinks — each EmitProbe deep-clones once per sink.

func BenchmarkProbe_Bus_EmitProbe_OneSink(b *testing.B) {
	bus := NewBus(NewRecorder())
	event := benchProbeEvent()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bus.EmitProbe(event)
	}
}

func BenchmarkProbe_Bus_EmitProbe_FourSinks(b *testing.B) {
	bus := NewBus(NewRecorder(), NewRecorder(), NewRecorder(), NewRecorder())
	event := benchProbeEvent()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bus.EmitProbe(event)
	}
}

func BenchmarkProbe_Bus_EmitProbe_Empty(b *testing.B) {
	bus := NewBus()
	event := benchProbeEvent()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bus.EmitProbe(event)
	}
}

// --- SinkFunc.EmitProbe ---
// Wraps a plain function — direct dispatch with no clone.

func BenchmarkProbe_SinkFunc_EmitProbe(b *testing.B) {
	var got Event
	f := SinkFunc(func(e Event) { got = e })
	event := Event{Kind: KindToken, Step: 1, Token: &Token{ID: 7}}
	_ = got
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f.EmitProbe(event)
	}
}

func BenchmarkProbe_SinkFunc_EmitProbe_NilFunc(b *testing.B) {
	var f SinkFunc
	event := Event{Kind: KindToken, Step: 1}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f.EmitProbe(event)
	}
}

// --- Bus.Add ---
// Append under the bus lock — fires once per AttachSink call.

func BenchmarkProbe_Bus_Add(b *testing.B) {
	sink := NewRecorder()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bus := NewBus()
		bus.Add(sink)
	}
}
