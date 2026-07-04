// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import core "dappco.re/go"

const (
	// HybridAttentionLinear identifies cacheless linear-attention layers.
	HybridAttentionLinear = "linear_attention"
	// HybridAttentionFull identifies full/global attention layers with K/V.
	HybridAttentionFull = "full_attention"
)

// HybridAttentionLayerPlan describes the cache behaviour of one decoder layer.
type HybridAttentionLayerPlan struct {
	Layer      int
	Kind       string
	RequiresKV bool
	Window     int
	CacheIndex int
}

// HybridAttentionCachePlan maps model layers onto the smaller physical KV
// cache used by hybrid-attention architectures.
type HybridAttentionCachePlan struct {
	Layers            []HybridAttentionLayerPlan
	CacheIndexByLayer []int
	CachelessLayers   int
	GlobalLayers      int
}

// BuildHybridAttentionCachePlan expands layerTypes across numLayers and
// returns the non-identity cache topology used by hybrid-attention models.
func BuildHybridAttentionCachePlan(numLayers int, layerTypes []string, localWindow int) (HybridAttentionCachePlan, error) {
	if numLayers <= 0 {
		return HybridAttentionCachePlan{}, core.NewError("hybrid attention requires positive layer count")
	}
	if len(layerTypes) == 0 {
		return HybridAttentionCachePlan{}, core.NewError("hybrid attention requires linear_attention layer metadata")
	}
	pattern := make([]string, 0, len(layerTypes))
	for _, value := range layerTypes {
		kind, ok := ParseHybridAttentionKind(value)
		if !ok {
			return HybridAttentionCachePlan{}, core.NewError("hybrid attention unsupported layer type: " + value)
		}
		pattern = append(pattern, kind)
	}
	plan := HybridAttentionCachePlan{
		Layers:            make([]HybridAttentionLayerPlan, numLayers),
		CacheIndexByLayer: make([]int, numLayers),
	}
	for i := range plan.CacheIndexByLayer {
		plan.CacheIndexByLayer[i] = -1
	}
	for i := range numLayers {
		kind := pattern[i%len(pattern)]
		layer := HybridAttentionLayerPlan{
			Layer:      i,
			Kind:       kind,
			CacheIndex: -1,
		}
		switch kind {
		case HybridAttentionLinear:
			plan.CachelessLayers++
		case HybridAttentionFull:
			layer.RequiresKV = true
			layer.Window = localWindow
			layer.CacheIndex = plan.GlobalLayers
			plan.CacheIndexByLayer[i] = layer.CacheIndex
			plan.GlobalLayers++
		}
		plan.Layers[i] = layer
	}
	if plan.CachelessLayers == 0 {
		return HybridAttentionCachePlan{}, core.NewError("hybrid attention requires linear_attention layer metadata")
	}
	if plan.GlobalLayers == 0 {
		return HybridAttentionCachePlan{}, core.NewError("hybrid attention requires full_attention layer metadata")
	}
	return plan, nil
}

// ParseHybridAttentionKind canonicalises hybrid attention layer identifiers.
func ParseHybridAttentionKind(value string) (string, bool) {
	switch NormalizeDenseLayerType(value) {
	case "linear_attention", "linear":
		return HybridAttentionLinear, true
	case "full_attention", "global_attention", "attention", "full":
		return HybridAttentionFull, true
	default:
		return "", false
	}
}

// ExpandedLayerTypes returns the canonical layer type for each model layer.
func (plan HybridAttentionCachePlan) ExpandedLayerTypes() []string {
	if len(plan.Layers) == 0 {
		return nil
	}
	layerTypes := make([]string, len(plan.Layers))
	for i, layer := range plan.Layers {
		layerTypes[i] = layer.Kind
	}
	return layerTypes
}

// CacheIndexCSV returns the physical KV-cache index for each model layer.
func (plan HybridAttentionCachePlan) CacheIndexCSV() string {
	if len(plan.CacheIndexByLayer) == 0 {
		return ""
	}
	indexes := make([]string, len(plan.CacheIndexByLayer))
	for i, index := range plan.CacheIndexByLayer {
		indexes[i] = core.Sprintf("%d", index)
	}
	return core.Join(",", indexes...)
}

func normalizeHybridAttentionLayerType(value string) string {
	return NormalizeDenseLayerType(value)
}
