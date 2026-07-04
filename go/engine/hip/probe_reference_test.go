// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestProbeReferenceLogits_Good_SummarisesAndEmitsProbe(t *testing.T) {
	var events []inference.ProbeEvent
	probe, err := rocmReferenceLogitProbe(
		[]float32{-1, 2, 0.5},
		2,
		[]string{"a", "b", "c"},
		inference.ProbeSinkFunc(func(event inference.ProbeEvent) { events = append(events, event) }),
	)

	core.RequireNoError(t, err)
	core.AssertEqual(t, 3, probe.VocabularySize)
	core.AssertEqual(t, int32(1), probe.Top[0].ID)
	core.AssertEqual(t, "b", probe.Top[0].Text)
	assertFloat32Near(t, 2, probe.Max)
	assertFloat32Near(t, -1, probe.Min)
	assertFloat32Near(t, 0.5, probe.Mean)
	core.AssertEqual(t, 1, len(events))
	core.AssertEqual(t, inference.ProbeEventLogits, events[0].Kind)
}

func TestProbeReferenceHeadSelection_Good_SelectsTopHeadsAndEmitsProbe(t *testing.T) {
	var events []inference.ProbeEvent
	probe, err := rocmReferenceHeadSelection(
		[]float32{0.5, 0.9, 0.9, -1},
		2,
		3,
		inference.ProbeSinkFunc(func(event inference.ProbeEvent) { events = append(events, event) }),
	)

	core.RequireNoError(t, err)
	core.AssertEqual(t, 3, probe.Layer)
	core.AssertEqual(t, []int{1, 2}, probe.Heads)
	core.AssertEqual(t, 1, len(events))
	core.AssertEqual(t, inference.ProbeEventSelectedHeads, events[0].Kind)
	core.AssertEqual(t, []int{1, 2}, events[0].SelectedHeads.Heads)
}

func TestProbeReferenceLayerCoherence_Good_SummarisesAndEmitsProbe(t *testing.T) {
	var events []inference.ProbeEvent
	probe, err := rocmReferenceLayerCoherenceProbe(
		5,
		[][]float32{{1, 0}, {0, 1}},
		[][]float32{{1, 0}, {0, -1}},
		inference.ProbeSinkFunc(func(event inference.ProbeEvent) { events = append(events, event) }),
	)

	core.RequireNoError(t, err)
	core.AssertEqual(t, 5, probe.Layer)
	assertFloat64Near(t, 0, probe.KVCoupling, 0.0001)
	assertFloat64Near(t, 0, probe.MeanCoherence, 0.0001)
	assertFloat64Near(t, 0.75, probe.PhaseLock, 0.0001)
	assertFloat64Near(t, 0.6666, probe.SpectralStable, 0.0001)
	core.AssertEqual(t, 1, len(events))
	core.AssertEqual(t, inference.ProbeEventLayerCoherence, events[0].Kind)
	core.AssertEqual(t, 5, events[0].LayerCoherence.Layer)
}

func TestProbeReferenceEntropy_Good_SummarisesAndEmitsProbe(t *testing.T) {
	var events []inference.ProbeEvent
	probe, err := rocmReferenceEntropyProbe([]float32{0, 0}, inference.ProbeSinkFunc(func(event inference.ProbeEvent) {
		events = append(events, event)
	}))

	core.RequireNoError(t, err)
	assertFloat64Near(t, 0.6931, probe.Value, 0.0001)
	core.AssertEqual(t, "nats", probe.Unit)
	core.AssertEqual(t, 1, len(events))
	core.AssertEqual(t, inference.ProbeEventEntropy, events[0].Kind)
}

func TestProbeReferenceEntropy_Good_StableLargeLogits(t *testing.T) {
	probe, err := rocmReferenceEntropyProbe([]float32{1000, 999}, nil)

	core.RequireNoError(t, err)
	assertFloat64Near(t, 0.5822, probe.Value, 0.0001)
	core.AssertEqual(t, "nats", probe.Unit)
}

func TestProbeReferenceLogits_Bad_RejectsEmptyLogits(t *testing.T) {
	_, err := rocmReferenceLogitProbe(nil, 1, nil, nil)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "logits")
}

func TestProbeReferenceLogits_Bad_RejectsZeroTopK(t *testing.T) {
	_, err := rocmReferenceLogitProbe([]float32{1}, 0, nil, nil)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "top-k")
}

func TestProbeReferenceLogits_Bad_RejectsTopKBeyondVocabulary(t *testing.T) {
	_, err := rocmReferenceLogitProbe([]float32{1}, 2, nil, nil)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "top-k")
}

func TestProbeReferenceEntropy_Bad_RejectsEmptyLogits(t *testing.T) {
	_, err := rocmReferenceEntropyProbe(nil, nil)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "logits")
}

func TestProbeReferenceHeadSelection_Bad_RejectsEmptyScores(t *testing.T) {
	_, err := rocmReferenceHeadSelection(nil, 1, 0, nil)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "scores")
}

func TestProbeReferenceHeadSelection_Bad_RejectsZeroTopK(t *testing.T) {
	_, err := rocmReferenceHeadSelection([]float32{1}, 0, 0, nil)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "top-k")
}

func TestProbeReferenceHeadSelection_Bad_RejectsTopKBeyondHeadCount(t *testing.T) {
	_, err := rocmReferenceHeadSelection([]float32{1}, 2, 0, nil)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "top-k")
}

func TestProbeReferenceLayerCoherence_Bad_RejectsEmptyMatrices(t *testing.T) {
	_, err := rocmReferenceLayerCoherenceProbe(0, nil, nil, nil)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "non-empty")
}

func TestProbeReferenceLayerCoherence_Bad_RejectsEmptyRows(t *testing.T) {
	_, err := rocmReferenceLayerCoherenceProbe(0, [][]float32{{}}, [][]float32{{}}, nil)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "width")
}

func TestProbeReferenceLayerCoherence_Bad_RejectsMismatchedRowWidths(t *testing.T) {
	_, err := rocmReferenceLayerCoherenceProbe(0, [][]float32{{1, 2}}, [][]float32{{1}}, nil)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "width")
}

func TestProbeReferenceLayerCoherence_Bad_RejectsZeroVectors(t *testing.T) {
	_, err := rocmReferenceLayerCoherenceProbe(0, [][]float32{{0, 0}}, [][]float32{{0, 0}}, nil)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "score KV coupling")
}
