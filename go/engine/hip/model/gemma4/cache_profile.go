// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import "strconv"

// CacheObservation is the backend-neutral live shape of one KV cache.
type CacheObservation struct {
	Tokens   int
	Capacity int
	Bounded  bool
}

// CacheProfile records Gemma-4's live local/global/shared KV-cache topology.
// It is the ROCm-side analogue of go-mlx's Gemma4Model.RecordCacheTopology,
// without importing a concrete GPU cache type.
type CacheProfile struct {
	Topology           CacheTopology
	TotalCaches        int
	LocalWindowTokens  int
	LocalCaches        int
	GlobalCaches       int
	SharedLayers       int
	MaxLocalTokens     int
	MaxLocalCapacity   int
	MaxGlobalTokens    int
	MaxGlobalCapacity  int
	LocalWindowLeaked  bool
	ObservedLayerCount int
}

func CacheProfileOf(cfg TextConfig, caches []CacheObservation) CacheProfile {
	topology := CacheTopologyOf(cfg)
	profile := CacheProfile{
		Topology:          topology,
		TotalCaches:       len(caches),
		LocalWindowTokens: topology.LocalWindowTokens,
		SharedLayers:      topology.SharedLayers,
	}
	for layerIndex, cacheIndex := range topology.CacheIndexByLayer {
		if cacheIndex < 0 {
			continue
		}
		if layerIndex >= len(topology.LayerTypes) || cacheIndex >= len(caches) {
			continue
		}
		cache := caches[cacheIndex]
		profile.ObservedLayerCount++
		tokens := positiveInt(cache.Tokens)
		capacity := positiveInt(cache.Capacity)
		switch topology.LayerTypes[layerIndex] {
		case LayerTypeFullAttention:
			profile.GlobalCaches++
			profile.MaxGlobalTokens = maxInt(profile.MaxGlobalTokens, tokens)
			profile.MaxGlobalCapacity = maxInt(profile.MaxGlobalCapacity, capacity)
		case LayerTypeSlidingAttention:
			profile.LocalCaches++
			profile.MaxLocalTokens = maxInt(profile.MaxLocalTokens, tokens)
			profile.MaxLocalCapacity = maxInt(profile.MaxLocalCapacity, capacity)
			if profile.LocalWindowTokens > 0 && (tokens > profile.LocalWindowTokens || capacity > profile.LocalWindowTokens || !cache.Bounded) {
				profile.LocalWindowLeaked = true
			}
		}
	}
	return profile
}

func ApplyCacheProfileLabels(labels map[string]string, profile CacheProfile) map[string]string {
	if labels == nil {
		labels = map[string]string{}
	}
	if profile.TotalCaches > 0 {
		labels["attention_cache_profile_total"] = strconv.Itoa(profile.TotalCaches)
		labels["gemma4_attention_cache_profile_total"] = labels["attention_cache_profile_total"]
	}
	if profile.ObservedLayerCount > 0 {
		labels["attention_cache_profile_observed_layers"] = strconv.Itoa(profile.ObservedLayerCount)
		labels["gemma4_attention_cache_profile_observed_layers"] = labels["attention_cache_profile_observed_layers"]
	}
	if profile.LocalWindowTokens > 0 {
		labels["attention_cache_profile_local_window_tokens"] = strconv.Itoa(profile.LocalWindowTokens)
		labels["gemma4_attention_cache_profile_local_window_tokens"] = labels["attention_cache_profile_local_window_tokens"]
	}
	if profile.LocalCaches > 0 {
		labels["attention_cache_profile_local_count"] = strconv.Itoa(profile.LocalCaches)
		labels["gemma4_attention_cache_profile_local_count"] = labels["attention_cache_profile_local_count"]
	}
	if profile.GlobalCaches > 0 {
		labels["attention_cache_profile_global_count"] = strconv.Itoa(profile.GlobalCaches)
		labels["gemma4_attention_cache_profile_global_count"] = labels["attention_cache_profile_global_count"]
	}
	if profile.SharedLayers > 0 {
		labels["attention_cache_profile_shared_layers"] = strconv.Itoa(profile.SharedLayers)
		labels["gemma4_attention_cache_profile_shared_layers"] = labels["attention_cache_profile_shared_layers"]
	}
	if profile.MaxLocalTokens > 0 {
		labels["attention_cache_profile_max_local_tokens"] = strconv.Itoa(profile.MaxLocalTokens)
		labels["gemma4_attention_cache_profile_max_local_tokens"] = labels["attention_cache_profile_max_local_tokens"]
	}
	if profile.MaxLocalCapacity > 0 {
		labels["attention_cache_profile_max_local_capacity"] = strconv.Itoa(profile.MaxLocalCapacity)
		labels["gemma4_attention_cache_profile_max_local_capacity"] = labels["attention_cache_profile_max_local_capacity"]
	}
	if profile.MaxGlobalTokens > 0 {
		labels["attention_cache_profile_max_global_tokens"] = strconv.Itoa(profile.MaxGlobalTokens)
		labels["gemma4_attention_cache_profile_max_global_tokens"] = labels["attention_cache_profile_max_global_tokens"]
	}
	if profile.MaxGlobalCapacity > 0 {
		labels["attention_cache_profile_max_global_capacity"] = strconv.Itoa(profile.MaxGlobalCapacity)
		labels["gemma4_attention_cache_profile_max_global_capacity"] = labels["attention_cache_profile_max_global_capacity"]
	}
	labels["attention_cache_profile_local_window_leaked"] = strconv.FormatBool(profile.LocalWindowLeaked)
	labels["gemma4_attention_cache_profile_local_window_leaked"] = labels["attention_cache_profile_local_window_leaked"]
	return labels
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
