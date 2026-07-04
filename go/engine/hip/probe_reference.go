// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func rocmReferenceHeadSelection(scores []float32, topK, layer int, sink inference.ProbeSink) (inference.ProbeHeadSelection, error) {
	if len(scores) == 0 {
		return inference.ProbeHeadSelection{}, core.E("rocm.Probe.ReferenceHeadSelection", "head scores are required", nil)
	}
	if topK <= 0 || topK > len(scores) {
		return inference.ProbeHeadSelection{}, core.E("rocm.Probe.ReferenceHeadSelection", "top-k must be within head count", nil)
	}
	candidates := make([]hipReferenceCandidate, len(scores))
	for i, score := range scores {
		candidates[i] = hipReferenceCandidate{index: i, value: score}
	}
	sortHIPReferenceCandidates(candidates)
	probe := inference.ProbeHeadSelection{Layer: layer, Heads: make([]int, topK)}
	for i := 0; i < topK; i++ {
		probe.Heads[i] = candidates[i].index
	}
	if sink != nil {
		sink.EmitProbe(inference.ProbeEvent{
			Kind:          inference.ProbeEventSelectedHeads,
			Phase:         inference.ProbePhasePrefill,
			Labels:        map[string]string{"backend": "rocm", "source": "cpu_reference"},
			SelectedHeads: &probe,
		})
	}
	return probe, nil
}

func rocmReferenceLogitProbe(logits []float32, topK int, tokenTexts []string, sink inference.ProbeSink) (inference.ProbeLogits, error) {
	if len(logits) == 0 {
		return inference.ProbeLogits{}, core.E("rocm.Probe.ReferenceLogits", "logits are required", nil)
	}
	if topK <= 0 || topK > len(logits) {
		return inference.ProbeLogits{}, core.E("rocm.Probe.ReferenceLogits", "top-k must be within vocabulary size", nil)
	}
	candidates := make([]hipReferenceCandidate, len(logits))
	minValue := logits[0]
	maxValue := logits[0]
	mean := float32(0)
	for i, value := range logits {
		candidates[i] = hipReferenceCandidate{index: i, value: value}
		if value < minValue {
			minValue = value
		}
		if value > maxValue {
			maxValue = value
		}
		mean += value
	}
	mean /= float32(len(logits))
	sortHIPReferenceCandidates(candidates)
	top := make([]inference.ProbeLogit, topK)
	for i := 0; i < topK; i++ {
		index := candidates[i].index
		top[i] = inference.ProbeLogit{ID: int32(index), Value: candidates[i].value}
		if index < len(tokenTexts) {
			top[i].Text = tokenTexts[index]
		}
	}
	probe := inference.ProbeLogits{
		VocabularySize: len(logits),
		Top:            top,
		Min:            minValue,
		Max:            maxValue,
		Mean:           mean,
	}
	if sink != nil {
		sink.EmitProbe(inference.ProbeEvent{
			Kind:   inference.ProbeEventLogits,
			Phase:  inference.ProbePhaseDecode,
			Labels: map[string]string{"backend": "rocm", "source": "cpu_reference"},
			Logits: &probe,
		})
	}
	return probe, nil
}

func rocmReferenceLayerCoherenceProbe(layer int, keys, values [][]float32, sink inference.ProbeSink) (inference.ProbeLayerCoherence, error) {
	flatKeys, flatValues, err := flattenMatchedProbeMatrices(keys, values)
	if err != nil {
		return inference.ProbeLayerCoherence{}, err
	}
	kvCoupling, err := rocmReferenceCosineSimilarity(flatKeys, flatValues)
	if err != nil {
		return inference.ProbeLayerCoherence{}, core.E("rocm.Probe.ReferenceLayerCoherence", "score KV coupling", err)
	}
	meanCoherence := float64(0)
	for i := range keys {
		score, err := rocmReferenceCosineSimilarity(keys[i], values[i])
		if err != nil {
			return inference.ProbeLayerCoherence{}, core.E("rocm.Probe.ReferenceLayerCoherence", core.Sprintf("score token %d coherence", i), err)
		}
		meanCoherence += score
	}
	meanCoherence /= float64(len(keys))
	phaseLocked := 0
	meanAbsDelta := float64(0)
	for i := range flatKeys {
		if flatKeys[i]*flatValues[i] >= 0 {
			phaseLocked++
		}
		meanAbsDelta += math.Abs(float64(flatKeys[i] - flatValues[i]))
	}
	meanAbsDelta /= float64(len(flatKeys))
	probe := inference.ProbeLayerCoherence{
		Layer:          layer,
		KVCoupling:     kvCoupling,
		MeanCoherence:  meanCoherence,
		PhaseLock:      float64(phaseLocked) / float64(len(flatKeys)),
		SpectralStable: 1 / (1 + meanAbsDelta),
	}
	if sink != nil {
		sink.EmitProbe(inference.ProbeEvent{
			Kind:           inference.ProbeEventLayerCoherence,
			Phase:          inference.ProbePhasePrefill,
			Labels:         map[string]string{"backend": "rocm", "source": "cpu_reference"},
			LayerCoherence: &probe,
		})
	}
	return probe, nil
}

func flattenMatchedProbeMatrices(keys, values [][]float32) ([]float32, []float32, error) {
	if len(keys) == 0 || len(keys) != len(values) {
		return nil, nil, core.E("rocm.Probe.ReferenceLayerCoherence", "key and value matrices must be non-empty and equal length", nil)
	}
	width := len(keys[0])
	if width == 0 {
		return nil, nil, core.E("rocm.Probe.ReferenceLayerCoherence", "matrix width must be positive", nil)
	}
	flatKeys := make([]float32, 0, len(keys)*width)
	flatValues := make([]float32, 0, len(values)*width)
	for i := range keys {
		if len(keys[i]) != width || len(values[i]) != width {
			return nil, nil, core.E("rocm.Probe.ReferenceLayerCoherence", core.Sprintf("matrix row %d width does not match %d", i, width), nil)
		}
		flatKeys = append(flatKeys, keys[i]...)
		flatValues = append(flatValues, values[i]...)
	}
	return flatKeys, flatValues, nil
}

func rocmReferenceEntropyProbe(logits []float32, sink inference.ProbeSink) (inference.ProbeEntropy, error) {
	if len(logits) == 0 {
		return inference.ProbeEntropy{}, core.E("rocm.Probe.ReferenceEntropy", "logits are required", nil)
	}
	probs := softmaxFloat32(logits)
	entropy := float64(0)
	for _, prob := range probs {
		if prob > 0 {
			entropy -= float64(prob) * math.Log(float64(prob))
		}
	}
	probe := inference.ProbeEntropy{Value: entropy, Unit: "nats"}
	if sink != nil {
		sink.EmitProbe(inference.ProbeEvent{
			Kind:    inference.ProbeEventEntropy,
			Phase:   inference.ProbePhaseDecode,
			Labels:  map[string]string{"backend": "rocm", "source": "cpu_reference"},
			Entropy: &probe,
		})
	}
	return probe, nil
}
