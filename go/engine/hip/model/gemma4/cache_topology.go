// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"strconv"
	"strings"
)

const (
	LayerTypeSlidingAttention = "sliding_attention"
	LayerTypeFullAttention    = "full_attention"
)

// CacheTopology is Gemma-4's model-owned view of local/global/shared KV caches.
// It mirrors the runtime cache ownership plan without importing a backend.
type CacheTopology struct {
	NumLayers              int
	LayerTypes             []string
	PreviousKVByLayer      []int
	CacheIndexByLayer      []int
	LocalWindowTokens      int
	LocalCaches            int
	GlobalCaches           int
	SharedLayers           int
	OwnerCaches            int
	FixedSlidingPrefillCap int
}

// CacheTopologyOf maps Gemma-4 config into the same shared-KV owner plan that
// the native Q4 path uses for decoding.
func CacheTopologyOf(cfg TextConfig) CacheTopology {
	layerTypes := LayerTypesOf(cfg)
	topology := CacheTopology{
		NumLayers:         len(layerTypes),
		LayerTypes:        layerTypes,
		LocalWindowTokens: positiveInt(cfg.SlidingWindow),
	}
	if len(layerTypes) == 0 {
		return topology
	}
	topology.PreviousKVByLayer, topology.CacheIndexByLayer = BuildCacheLayout(layerTypes, cfg.KVSharedLayers)
	for layerIndex, cacheIndex := range topology.CacheIndexByLayer {
		if cacheIndex < 0 {
			topology.SharedLayers++
			continue
		}
		topology.OwnerCaches++
		switch layerTypes[layerIndex] {
		case LayerTypeFullAttention:
			topology.GlobalCaches++
		case LayerTypeSlidingAttention:
			topology.LocalCaches++
		}
	}
	topology.FixedSlidingPrefillCap = FixedSlidingPrefillChunkLimit(cfg)
	return topology
}

// LayerTypesOf returns the normalized Gemma-4 per-layer attention class. When a
// config omits layer_types but declares a sliding pattern, the Gemma-4 default
// pattern is expanded and the final layer is forced global.
func LayerTypesOf(cfg TextConfig) []string {
	numLayers := positiveInt(cfg.NumLayers)
	if len(cfg.LayerTypes) > 0 {
		layerTypes := normalizeLayerTypes(cfg.LayerTypes)
		if numLayers > 0 && len(layerTypes) > numLayers {
			layerTypes = layerTypes[:numLayers]
		}
		return layerTypes
	}
	if numLayers <= 0 || (cfg.SlidingWindow <= 0 && cfg.SlidingWindowPattern <= 0) {
		return nil
	}
	pattern := cfg.SlidingWindowPattern
	if pattern <= 0 {
		pattern = 6
	}
	layerTypes := make([]string, numLayers)
	for index := range layerTypes {
		if pattern > 1 && (index+1)%pattern != 0 {
			layerTypes[index] = LayerTypeSlidingAttention
		} else {
			layerTypes[index] = LayerTypeFullAttention
		}
	}
	layerTypes[len(layerTypes)-1] = LayerTypeFullAttention
	return layerTypes
}

// BuildCacheLayout returns PreviousKVByLayer and CacheIndexByLayer. A
// CacheIndexByLayer entry of -1 means that layer borrows its owner's KV cache.
func BuildCacheLayout(layerTypes []string, sharedLayers int) ([]int, []int) {
	layerTypes = normalizeLayerTypes(layerTypes)
	previous := make([]int, len(layerTypes))
	cacheIndexByLayer := make([]int, len(layerTypes))
	for index := range previous {
		previous[index] = index
		cacheIndexByLayer[index] = -1
	}
	if len(layerTypes) == 0 {
		return previous, cacheIndexByLayer
	}
	if sharedLayers < 0 {
		sharedLayers = 0
	}
	firstShared := min(max(len(layerTypes)-sharedLayers, 0), len(layerTypes))
	latestByType := map[string]int{}
	nextCacheIndex := 0
	for index, layerType := range layerTypes {
		ownsCache := index < firstShared
		if !ownsCache {
			if previousOwner, ok := latestByType[layerType]; ok {
				previous[index] = previousOwner
			} else {
				ownsCache = true
			}
		}
		if ownsCache {
			previous[index] = index
			latestByType[layerType] = index
			cacheIndexByLayer[index] = nextCacheIndex
			nextCacheIndex++
		}
	}
	return previous, cacheIndexByLayer
}

// AttentionCacheLayout maps every layer to the cache index it should read from,
// or -1 if the owner/cache sits outside the supplied cache count.
func AttentionCacheLayout(cfg TextConfig, numLayers, numCaches int) []int {
	if numLayers <= 0 {
		numLayers = cfg.NumLayers
	}
	layout := make([]int, positiveInt(numLayers))
	for index := range layout {
		layout[index] = -1
	}
	if len(layout) == 0 || numCaches <= 0 {
		return layout
	}
	topology := CacheTopologyOf(cfg)
	for layerIndex := 0; layerIndex < len(layout) && layerIndex < len(topology.PreviousKVByLayer); layerIndex++ {
		ownerIndex := topology.PreviousKVByLayer[layerIndex]
		if ownerIndex < 0 || ownerIndex >= len(topology.CacheIndexByLayer) {
			continue
		}
		cacheIndex := topology.CacheIndexByLayer[ownerIndex]
		if cacheIndex < 0 || cacheIndex >= numCaches {
			continue
		}
		layout[layerIndex] = cacheIndex
	}
	return layout
}

// FixedSlidingPrefillChunkLimit reports the largest safe fixed-sliding prefill
// chunk. Pass fixed cache sizes to further cap the model's sliding window.
func FixedSlidingPrefillChunkLimit(cfg TextConfig, fixedCacheSizes ...int) int {
	if cfg.SlidingWindow <= 0 {
		return 0
	}
	limit := cfg.SlidingWindow
	for _, size := range fixedCacheSizes {
		if size > 0 && size < limit {
			limit = size
		}
	}
	return limit
}

func ApplyCacheTopologyLabels(labels map[string]string, topology CacheTopology) map[string]string {
	if labels == nil {
		labels = map[string]string{}
	}
	if topology.NumLayers > 0 {
		value := strconv.Itoa(topology.NumLayers)
		labels["attention_layer_count"] = value
		labels["gemma4_attention_layer_count"] = value
	}
	if len(topology.LayerTypes) > 0 {
		value := strings.Join(topology.LayerTypes, ",")
		labels["attention_layer_types"] = value
		labels["gemma4_attention_layer_types"] = value
	}
	if len(topology.PreviousKVByLayer) > 0 {
		value := intCSV(topology.PreviousKVByLayer)
		labels["attention_cache_owner_by_layer"] = value
		labels["gemma4_attention_cache_owner_by_layer"] = value
	}
	if len(topology.CacheIndexByLayer) > 0 {
		value := intCSV(topology.CacheIndexByLayer)
		labels["attention_cache_index_by_layer"] = value
		labels["gemma4_attention_cache_index_by_layer"] = value
	}
	if topology.LocalWindowTokens > 0 {
		value := strconv.Itoa(topology.LocalWindowTokens)
		labels["attention_cache_local_window_tokens"] = value
		labels["gemma4_attention_cache_local_window_tokens"] = value
	}
	if topology.OwnerCaches > 0 {
		value := strconv.Itoa(topology.OwnerCaches)
		labels["attention_cache_owner_count"] = value
		labels["gemma4_attention_cache_owner_count"] = value
	}
	if topology.LocalCaches > 0 {
		value := strconv.Itoa(topology.LocalCaches)
		labels["attention_cache_local_count"] = value
		labels["gemma4_attention_cache_local_count"] = value
	}
	if topology.GlobalCaches > 0 {
		value := strconv.Itoa(topology.GlobalCaches)
		labels["attention_cache_global_count"] = value
		labels["gemma4_attention_cache_global_count"] = value
	}
	if topology.SharedLayers > 0 {
		value := strconv.Itoa(topology.SharedLayers)
		labels["attention_cache_shared_layers"] = value
		labels["gemma4_attention_cache_shared_layers"] = value
	}
	if topology.FixedSlidingPrefillCap > 0 {
		value := strconv.Itoa(topology.FixedSlidingPrefillCap)
		labels["fixed_sliding_prefill_chunk_limit"] = value
		labels["gemma4_fixed_sliding_prefill_chunk_limit"] = value
	}
	return labels
}

func normalizeLayerTypes(values []string) []string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		switch normalizeLayerType(value) {
		case LayerTypeSlidingAttention:
			out = append(out, LayerTypeSlidingAttention)
		case LayerTypeFullAttention:
			out = append(out, LayerTypeFullAttention)
		}
	}
	return out
}

func normalizeLayerType(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	value = strings.ReplaceAll(value, "-", "_")
	value = strings.ReplaceAll(value, " ", "_")
	switch value {
	case "sliding", "local", "local_attention", "sliding_window", "sliding_attention":
		return LayerTypeSlidingAttention
	case "full", "global", "global_attention", "full_attention":
		return LayerTypeFullAttention
	default:
		return ""
	}
}

func intCSV(values []int) string {
	if len(values) == 0 {
		return ""
	}
	parts := make([]string, len(values))
	for index, value := range values {
		parts[index] = strconv.Itoa(value)
	}
	return strings.Join(parts, ",")
}
