// SPDX-Licence-Identifier: EUPL-1.2

package bundle

import (
	"math"

	"dappco.re/go/inference/kv"
)

// SAMIResult is the SAMI BOResult-compatible model-state visualization
// schema. Bundles store SAMI summaries alongside KV state so downstream
// dashboards can render coherence + cross-alignment without reloading
// raw caches.
type SAMIResult struct {
	Model               string    `json:"model"`
	Prompt              string    `json:"prompt"`
	Architecture        string    `json:"architecture"`
	NumLayers           int       `json:"num_layers"`
	NumHeads            int       `json:"num_heads"`
	SeqLen              int       `json:"seq_len"`
	HeadDim             int       `json:"head_dim"`
	MeanCoherence       float64   `json:"mean_coherence"`
	MeanCrossAlignment  float64   `json:"mean_cross_alignment"`
	MeanHeadEntropy     float64   `json:"mean_head_entropy"`
	PhaseLockScore      float64   `json:"phase_lock_score"`
	JointCollapseCount  int       `json:"joint_collapse_count"`
	LayerCoherence      []float64 `json:"layer_coherence"`
	LayerCrossAlignment []float64 `json:"layer_cross_alignment"`
	Composite           float64   `json:"composite"`
}

// SAMIOptions labels a SAMI export with caller-owned provenance.
type SAMIOptions struct {
	Model  string
	Prompt string
}

// SAMIFromKV converts K/V analysis into SAMI's visualization schema.
//
//	sami := bundle.SAMIFromKV(snapshot, analysis, bundle.SAMIOptions{Model: name})
func SAMIFromKV(snapshot *kv.Snapshot, analysis *kv.Analysis, opts SAMIOptions) SAMIResult {
	if snapshot == nil {
		return SAMIResult{}
	}
	if analysis == nil {
		analysis = kv.Analyze(snapshot)
	}
	numLayers := snapshot.NumLayers
	if numLayers <= 0 {
		numLayers = len(snapshot.Layers)
	}
	meanCoherence := meanUnit(analysis.MeanKeyCoherence, analysis.MeanValueCoherence)
	meanCross := clampUnit(analysis.MeanCrossAlignment)
	// Hoist analysis-field slices + fallback scalars out of the per-layer
	// loop. Without this, each iteration re-dereferences analysis three
	// times and re-reads the same fallback floats. Pre-clamp the fallback
	// scalars so the per-layer fallback path skips clampUnit entirely.
	layerKey := analysis.LayerKeyCoherence
	layerValue := analysis.LayerValueCoherence
	layerAlign := analysis.LayerCrossAlignment
	clampedFallbackKey := clampUnit(analysis.MeanKeyCoherence)
	clampedFallbackValue := clampUnit(analysis.MeanValueCoherence)
	clampedFallbackAlign := clampUnit(analysis.MeanCrossAlignment)
	keyLen := len(layerKey)
	valueLen := len(layerValue)
	alignLen := len(layerAlign)
	// Single backing alloc for both layer arrays — typical dashboard tick
	// runs SAMIFromKV per visualisation frame with precomputed analysis,
	// so trimming 2 allocs → 1 + 1 reslice saves a malloc per frame.
	// 3-arg slice expression caps capacity so consumer-side append doesn't
	// reach across into the sibling slice.
	buf := make([]float64, 2*numLayers)
	layerCoherence := buf[:numLayers:numLayers]
	layerCross := buf[numLayers : 2*numLayers : 2*numLayers]
	// Split into hot in-bounds prefix and fallback tail. The common case
	// is keyLen == valueLen == alignLen == numLayers — in that case the
	// tail loop runs zero iterations and the prefix loop has no per-
	// iteration bounds-check branches against the analysis slices.
	inBounds := min(alignLen, min(valueLen, min(keyLen, numLayers)))
	for layer := range inBounds {
		k := clampUnit(layerKey[layer])
		v := clampUnit(layerValue[layer])
		a := clampUnit(layerAlign[layer])
		// (k + v) / 2 stays in [0,1] when both operands do — no outer clamp.
		layerCoherence[layer] = (k + v) / 2.0
		layerCross[layer] = a
	}
	for layer := inBounds; layer < numLayers; layer++ {
		var k, v, a float64
		if layer < keyLen {
			k = clampUnit(layerKey[layer])
		} else {
			k = clampedFallbackKey
		}
		if layer < valueLen {
			v = clampUnit(layerValue[layer])
		} else {
			v = clampedFallbackValue
		}
		if layer < alignLen {
			a = clampUnit(layerAlign[layer])
		} else {
			a = clampedFallbackAlign
		}
		layerCoherence[layer] = (k + v) / 2.0
		layerCross[layer] = a
	}
	jointCollapseCount := max(analysis.JointCollapseCount, 0)
	if numLayers > 0 && jointCollapseCount > numLayers {
		jointCollapseCount = numLayers
	}
	return SAMIResult{
		Model:               opts.Model,
		Prompt:              opts.Prompt,
		Architecture:        snapshot.Architecture,
		NumLayers:           numLayers,
		NumHeads:            snapshot.NumHeads,
		SeqLen:              snapshot.SeqLen,
		HeadDim:             snapshot.HeadDim,
		MeanCoherence:       meanCoherence,
		MeanCrossAlignment:  meanCross,
		MeanHeadEntropy:     clampUnit(analysis.MeanHeadEntropy),
		PhaseLockScore:      clampUnit(analysis.PhaseLockScore),
		JointCollapseCount:  jointCollapseCount,
		LayerCoherence:      layerCoherence,
		LayerCrossAlignment: layerCross,
		Composite:           clampRange(float64(analysis.Composite())/100.0, 0, 100),
	}
}

func layerMetric(values []float64, index int, fallback float64) float64 {
	if index >= 0 && index < len(values) {
		return clampUnit(values[index])
	}
	return clampUnit(fallback)
}

func meanUnit(a, b float64) float64 {
	return clampUnit((clampUnit(a) + clampUnit(b)) / 2.0)
}

func clampUnit(value float64) float64 {
	return clampRange(value, 0, 1)
}

func clampRange(value, minValue, maxValue float64) float64 {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return minValue
	}
	if value < minValue {
		return minValue
	}
	if value > maxValue {
		return maxValue
	}
	return value
}
