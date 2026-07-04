// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

// MoETextLayerParts describes one decoder layer in neutral sparse-MoE terms.
// DenseReady covers the normal decoder path. RouterReady and ExpertsReady are
// required only for sparse layers.
type MoETextLayerParts struct {
	DenseReady   bool
	IsMoE        bool
	RouterReady  bool
	ExpertsReady bool
	OK           bool
}

// MoETextRuntimeSummary records a readiness walk over a model's text layers.
type MoETextRuntimeSummary struct {
	Layers       int
	DenseLayers  int
	SparseLayers int
	Available    bool
}

// MoETextLayerRuntimeReady reports whether one decoder layer has the dense and,
// when sparse, MoE parts required for native text decode.
func MoETextLayerRuntimeReady(parts MoETextLayerParts) bool {
	if !parts.OK || !parts.DenseReady {
		return false
	}
	if !parts.IsMoE {
		return true
	}
	return parts.RouterReady && parts.ExpertsReady
}

// MoETextLayersRuntimeAvailable reports whether every layer exposes the dense
// and sparse-MoE parts required by native text decode.
func MoETextLayersRuntimeAvailable[T any](layers []T, parts func(T) MoETextLayerParts) bool {
	return SummarizeMoETextLayersRuntime(layers, parts).Available
}

// SummarizeMoETextLayersRuntime walks model-family layers and returns both the
// aggregate readiness bit and the dense/sparse layer counts.
func SummarizeMoETextLayersRuntime[T any](layers []T, parts func(T) MoETextLayerParts) MoETextRuntimeSummary {
	summary := MoETextRuntimeSummary{Layers: len(layers)}
	if len(layers) == 0 || parts == nil {
		return summary
	}
	for _, layer := range layers {
		layerParts := parts(layer)
		if !MoETextLayerRuntimeReady(layerParts) {
			return summary
		}
		if layerParts.IsMoE {
			summary.SparseLayers++
		} else {
			summary.DenseLayers++
		}
	}
	summary.Available = true
	return summary
}
