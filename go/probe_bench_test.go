// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the probe-event surface.
// Per AX-11 — backends emit probe events at the rate of generation
// (one per emitted token when ProbeEventToken is wired, one per layer
// per step for richer probes). ProbeBus.EmitProbe fires once per emit,
// and ProbeSinkFunc adapters wrap every consumer callback. Even a few
// nanoseconds per emit dominates the picture under research telemetry
// loads (think every-layer attention probes on 28-layer Qwen3).
//
// Run:    go test -bench=BenchmarkProbe -benchmem -run='^$' .

package inference

import (
	"testing"
)

// Sinks defeat compiler DCE.
var (
	probeBenchSinkEvent  ProbeEvent
	probeBenchSinkKind   ProbeEventKind
	probeBenchSinkCount  int
	probeBenchSinkBus    *ProbeBus
	probeBenchSinkSinkFn ProbeSinkFunc
)

// benchTokenEvent — minimal per-token decode probe (the per-step floor).
func benchTokenEvent() ProbeEvent {
	return ProbeEvent{
		Kind:  ProbeEventToken,
		Phase: ProbePhaseDecode,
		Step:  42,
		Token: &ProbeToken{
			ID:              7,
			Text:            "the",
			PromptTokens:    128,
			GeneratedTokens: 42,
		},
	}
}

// benchTypicalDecodeEvent — richer per-step shape mid-decode — cache
// + entropy + a top-5 logits summary. Closer to what a probe sink
// actually sees when research telemetry is on.
func benchTypicalDecodeEvent() ProbeEvent {
	return ProbeEvent{
		Kind:  ProbeEventLogits,
		Phase: ProbePhaseDecode,
		Step:  42,
		Logits: &ProbeLogits{
			VocabularySize: 151936,
			Top: []ProbeLogit{
				{ID: 7, Text: "the", Value: 0.34},
				{ID: 11, Text: "a", Value: 0.21},
				{ID: 23, Text: "and", Value: 0.12},
				{ID: 41, Text: "is", Value: 0.08},
				{ID: 67, Text: "to", Value: 0.05},
			},
			Min:  -12.5,
			Max:  9.8,
			Mean: -3.1,
		},
		Entropy: &ProbeEntropy{
			Value: 2.34,
			Unit:  "nats",
		},
		Cache: &ProbeCachePressure{
			PromptTokens:    128,
			GeneratedTokens: 42,
			CachedTokens:    96,
			CacheMode:       "paged-q8",
			HitRate:         0.75,
		},
	}
}

// benchTrainingEvent — what a training probe sink sees per step.
func benchTrainingEvent() ProbeEvent {
	return ProbeEvent{
		Kind:  ProbeEventTraining,
		Phase: ProbePhaseTraining,
		Step:  1024,
		Training: &ProbeTraining{
			Epoch:        2,
			Step:         1024,
			Loss:         1.234,
			LearningRate: 5e-5,
		},
		Memory: &ProbeMemoryPressure{
			ActiveBytes: 1 << 32, // 4 GiB
			PeakBytes:   1 << 33, // 8 GiB
			LimitBytes:  1 << 34, // 16 GiB
		},
		Labels: map[string]string{"adapter": "lora-domain-v2"},
	}
}

// --- ProbeSinkFunc.EmitProbe (the per-emit closure cost) ---

func BenchmarkProbe_ProbeSinkFunc_EmitProbe_Token(b *testing.B) {
	var captured ProbeEvent
	sink := ProbeSinkFunc(func(event ProbeEvent) {
		captured = event
	})
	event := benchTokenEvent()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sink.EmitProbe(event)
	}
	probeBenchSinkKind = captured.Kind
}

func BenchmarkProbe_ProbeSinkFunc_EmitProbe_TypicalDecode(b *testing.B) {
	var captured ProbeEvent
	sink := ProbeSinkFunc(func(event ProbeEvent) {
		captured = event
	})
	event := benchTypicalDecodeEvent()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sink.EmitProbe(event)
	}
	probeBenchSinkKind = captured.Kind
}

func BenchmarkProbe_ProbeSinkFunc_EmitProbe_Training(b *testing.B) {
	var captured ProbeEvent
	sink := ProbeSinkFunc(func(event ProbeEvent) {
		captured = event
	})
	event := benchTrainingEvent()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sink.EmitProbe(event)
	}
	probeBenchSinkKind = captured.Kind
}

// Nil-sink (Cladius dev path — probe sink not wired) — must be cheap.
func BenchmarkProbe_ProbeSinkFunc_EmitProbe_Nil(b *testing.B) {
	var sink ProbeSinkFunc
	event := benchTokenEvent()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sink.EmitProbe(event)
	}
}

// --- ProbeBus.EmitProbe fan-out cost ---

func BenchmarkProbe_NewProbeBus_NoSinks(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeBenchSinkBus = NewProbeBus()
	}
}

func BenchmarkProbe_NewProbeBus_OneSink(b *testing.B) {
	sink := ProbeSinkFunc(func(ProbeEvent) {})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeBenchSinkBus = NewProbeBus(sink)
	}
}

func BenchmarkProbe_NewProbeBus_FourSinks(b *testing.B) {
	s1 := ProbeSinkFunc(func(ProbeEvent) {})
	s2 := ProbeSinkFunc(func(ProbeEvent) {})
	s3 := ProbeSinkFunc(func(ProbeEvent) {})
	s4 := ProbeSinkFunc(func(ProbeEvent) {})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeBenchSinkBus = NewProbeBus(s1, s2, s3, s4)
	}
}

func BenchmarkProbe_ProbeBus_Add(b *testing.B) {
	bus := NewProbeBus()
	sink := ProbeSinkFunc(func(ProbeEvent) {})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bus.Add(sink)
	}
}

func BenchmarkProbe_ProbeBus_EmitProbe_OneSink(b *testing.B) {
	count := 0
	bus := NewProbeBus(ProbeSinkFunc(func(ProbeEvent) { count++ }))
	event := benchTokenEvent()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bus.EmitProbe(event)
	}
	probeBenchSinkCount = count
}

func BenchmarkProbe_ProbeBus_EmitProbe_FourSinks(b *testing.B) {
	count := 0
	bus := NewProbeBus(
		ProbeSinkFunc(func(ProbeEvent) { count++ }),
		ProbeSinkFunc(func(ProbeEvent) { count++ }),
		ProbeSinkFunc(func(ProbeEvent) { count++ }),
		ProbeSinkFunc(func(ProbeEvent) { count++ }),
	)
	event := benchTokenEvent()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bus.EmitProbe(event)
	}
	probeBenchSinkCount = count
}

func BenchmarkProbe_ProbeBus_EmitProbe_OneSink_TypicalDecode(b *testing.B) {
	count := 0
	bus := NewProbeBus(ProbeSinkFunc(func(ProbeEvent) { count++ }))
	event := benchTypicalDecodeEvent()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bus.EmitProbe(event)
	}
	probeBenchSinkCount = count
}

// Nil bus pointer — dev path; must be cheap.
func BenchmarkProbe_ProbeBus_EmitProbe_Nil(b *testing.B) {
	var bus *ProbeBus
	event := benchTokenEvent()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bus.EmitProbe(event)
	}
}

// Bus with a nil sink mixed in — exercises the nil-skip branch.
func BenchmarkProbe_ProbeBus_EmitProbe_WithNilSink(b *testing.B) {
	count := 0
	bus := &ProbeBus{
		sinks: []ProbeSink{
			nil,
			ProbeSinkFunc(func(ProbeEvent) { count++ }),
			nil,
			ProbeSinkFunc(func(ProbeEvent) { count++ }),
		},
	}
	event := benchTokenEvent()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bus.EmitProbe(event)
	}
	probeBenchSinkCount = count
}

// --- ProbeEvent construction (the value-cost backends pay at emit site) ---
// Each new() of a sub-shape (ProbeToken/ProbeLogits/...) is a heap-alloc
// pointer — surface those construction floors.

func BenchmarkProbe_ProbeEvent_Token(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeBenchSinkEvent = benchTokenEvent()
	}
}

func BenchmarkProbe_ProbeEvent_TypicalDecode(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeBenchSinkEvent = benchTypicalDecodeEvent()
	}
}

func BenchmarkProbe_ProbeEvent_Training(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeBenchSinkEvent = benchTrainingEvent()
	}
}

// Bare layer-coherence event (one-shot mid-decode probe) — the cheapest
// payload-bearing event shape.
func BenchmarkProbe_ProbeEvent_LayerCoherence(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeBenchSinkEvent = ProbeEvent{
			Kind:  ProbeEventLayerCoherence,
			Phase: ProbePhaseDecode,
			Step:  3,
			LayerCoherence: &ProbeLayerCoherence{
				Layer:          12,
				KVCoupling:     0.7,
				MeanCoherence:  0.8,
				PhaseLock:      0.9,
				SpectralStable: 0.6,
			},
		}
	}
}

// Router-decision event — emitted per MoE layer during decode.
func BenchmarkProbe_ProbeEvent_RouterDecision_8Experts(b *testing.B) {
	expertIDs := []int{0, 1, 2, 3, 4, 5, 6, 7}
	expertProbs := []float32{0.2, 0.18, 0.15, 0.12, 0.10, 0.09, 0.08, 0.08}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeBenchSinkEvent = ProbeEvent{
			Kind:  ProbeEventRouterDecision,
			Phase: ProbePhaseDecode,
			Step:  3,
			RouterDecision: &ProbeRouterDecision{
				Layer:       12,
				ExpertIDs:   expertIDs,
				ExpertProbs: expertProbs,
			},
		}
	}
}

// Scheduler event — emitted at queue boundaries, not per token.
func BenchmarkProbe_ProbeEvent_Scheduler(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeBenchSinkEvent = ProbeEvent{
			Kind:  ProbeEventScheduler,
			Phase: ProbePhaseQueue,
			Scheduler: &ProbeScheduler{
				RequestID:               "req-7",
				Event:                   "first_token",
				QueueDepth:              4,
				QueueLatencyMillis:      12.3,
				FirstTokenLatencyMillis: 45.6,
			},
		}
	}
}

// --- ProbeSinkFunc cast cost ---
// Used when a closure is passed where a ProbeSink is needed.

func BenchmarkProbe_ProbeSinkFunc_Cast(b *testing.B) {
	fn := func(ProbeEvent) {}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		probeBenchSinkSinkFn = ProbeSinkFunc(fn)
	}
}
