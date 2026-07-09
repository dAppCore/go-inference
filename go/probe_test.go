// SPDX-Licence-Identifier: EUPL-1.2

package inference

import "testing"

func TestProbe_ProbeSinkFunc_Good(t *testing.T) {
	var got ProbeEvent
	sink := ProbeSinkFunc(func(event ProbeEvent) {
		got = event
	})

	sink.EmitProbe(ProbeEvent{
		Kind: ProbeEventToken,
		Token: &ProbeToken{
			ID:   7,
			Text: "ok",
		},
	})

	checkEqual(t, ProbeEventToken, got.Kind)
	checkEqual(t, "ok", got.Token.Text)
}

func TestProbe_ProbeSinkFunc_EmitProbe_Good(t *testing.T) {
	var got ProbeEvent
	sink := ProbeSinkFunc(func(event ProbeEvent) {
		got = event
	})

	sink.EmitProbe(ProbeEvent{Kind: ProbeEventToken, Token: &ProbeToken{Text: "ok"}})

	checkEqual(t, ProbeEventToken, got.Kind)
	checkEqual(t, "ok", got.Token.Text)
}

func TestProbe_ProbeSinkFunc_EmitProbe_Bad(t *testing.T) {
	var sink ProbeSinkFunc
	event := ProbeEvent{Kind: ProbeEventTraining}

	sink.EmitProbe(event)

	checkNil(t, sink)
	checkEqual(t, ProbeEventTraining, event.Kind)
}

func TestProbe_ProbeSinkFunc_EmitProbe_Ugly(t *testing.T) {
	count := 0
	sink := ProbeSinkFunc(func(event ProbeEvent) {
		if event.Kind == ProbeEventEntropy {
			count++
		}
	})

	sink.EmitProbe(ProbeEvent{Kind: ProbeEventEntropy})
	sink.EmitProbe(ProbeEvent{Kind: ProbeEventMemoryPressure})

	checkEqual(t, 1, count)
}

func TestProbe_NewProbeBus_Good(t *testing.T) {
	var count int
	bus := NewProbeBus(ProbeSinkFunc(func(ProbeEvent) { count++ }))
	bus.Add(ProbeSinkFunc(func(ProbeEvent) { count++ }))

	bus.EmitProbe(ProbeEvent{Kind: ProbeEventMemoryPressure})

	checkEqual(t, 2, count)
}

func TestProbe_NewProbeBus_Bad(t *testing.T) {
	bus := NewProbeBus(nil)

	bus.EmitProbe(ProbeEvent{Kind: ProbeEventCachePressure})

	checkNotNil(t, bus)
	checkLen(t, bus.sinks, 0)
}

func TestProbe_NewProbeBus_Ugly(t *testing.T) {
	var got []ProbeEventKind
	bus := NewProbeBus(
		ProbeSinkFunc(func(event ProbeEvent) { got = append(got, event.Kind) }),
		nil,
		ProbeSinkFunc(func(event ProbeEvent) { got = append(got, event.Kind) }),
	)

	bus.EmitProbe(ProbeEvent{Kind: ProbeEventResidual})

	checkEqual(t, []ProbeEventKind{ProbeEventResidual, ProbeEventResidual}, got)
}

func TestProbe_ProbeBus_Add_Good(t *testing.T) {
	bus := NewProbeBus()
	sink := ProbeSinkFunc(func(ProbeEvent) {})

	bus.Add(sink)

	checkLen(t, bus.sinks, 1)
}

func TestProbe_ProbeBus_Add_Bad(t *testing.T) {
	var bus *ProbeBus

	bus.Add(nil)

	checkNil(t, bus)
}

func TestProbe_ProbeBus_Add_Ugly(t *testing.T) {
	bus := NewProbeBus()

	bus.Add(nil)
	bus.Add(ProbeSinkFunc(func(ProbeEvent) {}))

	checkLen(t, bus.sinks, 1)
}

func TestProbe_ProbeBus_EmitProbe_Good(t *testing.T) {
	var count int
	bus := NewProbeBus(
		ProbeSinkFunc(func(ProbeEvent) { count++ }),
		ProbeSinkFunc(func(ProbeEvent) { count++ }),
	)

	bus.EmitProbe(ProbeEvent{Kind: ProbeEventMemoryPressure})

	checkEqual(t, 2, count)
}

func TestProbe_ProbeBus_EmitProbe_Bad(t *testing.T) {
	var bus *ProbeBus
	event := ProbeEvent{Kind: ProbeEventCachePressure}

	bus.EmitProbe(event)

	checkNil(t, bus)
	checkEqual(t, ProbeEventCachePressure, event.Kind)
}

func TestProbe_ProbeBus_EmitProbe_Ugly(t *testing.T) {
	var count int
	bus := &ProbeBus{
		sinks: []ProbeSink{
			nil,
			ProbeSinkFunc(func(ProbeEvent) { count++ }),
		},
	}

	bus.EmitProbe(ProbeEvent{Kind: ProbeEventCachePressure})

	checkEqual(t, 1, count)
}

func TestProbeEventRichPayload(t *testing.T) {
	event := ProbeEvent{
		Kind:  ProbeEventLayerCoherence,
		Phase: ProbePhaseDecode,
		Step:  3,
		LayerCoherence: &ProbeLayerCoherence{
			Layer:          2,
			KVCoupling:     0.7,
			MeanCoherence:  0.8,
			PhaseLock:      0.9,
			SpectralStable: 0.6,
		},
		Cache: &ProbeCachePressure{
			PromptTokens:    128,
			GeneratedTokens: 16,
			CachedTokens:    96,
			CacheMode:       "paged-q8",
			HitRate:         0.75,
		},
	}

	checkEqual(t, ProbeEventLayerCoherence, event.Kind)
	checkEqual(t, ProbePhaseDecode, event.Phase)
	checkEqual(t, 2, event.LayerCoherence.Layer)
	checkEqual(t, "paged-q8", event.Cache.CacheMode)
}
