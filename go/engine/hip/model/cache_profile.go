// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"strconv"
	"strings"

	"dappco.re/go/inference/engine/hip/profile"
)

const (
	CacheProfileContract = "rocm-cache-profile-v1"

	CacheObservationKindFull      = "full"
	CacheObservationKindRotating  = "rotating"
	CacheObservationKindFixed     = "fixed"
	CacheObservationKindPaged     = "paged"
	CacheObservationKindQuantized = "quantized"
	CacheObservationKindUnknown   = "unknown"
)

// CacheObservation is the backend-neutral live shape of one KV cache.
// HIP, CUDA, and CPU runtimes can report concrete cache state through this
// contract without importing each other's runtime types.
type CacheObservation struct {
	Kind            string            `json:"kind,omitempty"`
	Mode            string            `json:"mode,omitempty"`
	Layer           int               `json:"layer,omitempty"`
	Tokens          int               `json:"tokens,omitempty"`
	Capacity        int               `json:"capacity,omitempty"`
	ProcessedTokens int               `json:"processed_tokens,omitempty"`
	Bounded         bool              `json:"bounded,omitempty"`
	Local           bool              `json:"local,omitempty"`
	Global          bool              `json:"global,omitempty"`
	Shared          bool              `json:"shared,omitempty"`
	Cacheless       bool              `json:"cacheless,omitempty"`
	Full            bool              `json:"full,omitempty"`
	Rotating        bool              `json:"rotating,omitempty"`
	Fixed           bool              `json:"fixed,omitempty"`
	Paged           bool              `json:"paged,omitempty"`
	Quantized       bool              `json:"quantized,omitempty"`
	Labels          map[string]string `json:"labels,omitempty"`
}

func (observation CacheObservation) Clone() CacheObservation {
	observation.Labels = cloneStringMap(observation.Labels)
	return observation
}

// CacheProfileOptions carries architecture topology that may not be visible
// from a generic cache object.
type CacheProfileOptions struct {
	Architecture      string            `json:"architecture,omitempty"`
	LocalWindowTokens int               `json:"local_window_tokens,omitempty"`
	SharedLayers      int               `json:"shared_layers,omitempty"`
	CachelessLayers   int               `json:"cacheless_layers,omitempty"`
	Labels            map[string]string `json:"labels,omitempty"`
}

// CacheProfile reports how live K/V caches are shaped after a generation turn.
// It mirrors the go-mlx metal profile as a model-owned ROCm contract.
type CacheProfile struct {
	Contract           string            `json:"contract,omitempty"`
	Architecture       string            `json:"architecture,omitempty"`
	TotalCaches        int               `json:"total_caches,omitempty"`
	LocalCaches        int               `json:"local_caches,omitempty"`
	GlobalCaches       int               `json:"global_caches,omitempty"`
	SharedLayers       int               `json:"shared_layers,omitempty"`
	CachelessLayers    int               `json:"cacheless_layers,omitempty"`
	LocalWindowTokens  int               `json:"local_window_tokens,omitempty"`
	MaxLocalTokens     int               `json:"max_local_tokens,omitempty"`
	MaxLocalCapacity   int               `json:"max_local_capacity,omitempty"`
	MaxGlobalTokens    int               `json:"max_global_tokens,omitempty"`
	MaxGlobalCapacity  int               `json:"max_global_capacity,omitempty"`
	MaxCacheTokens     int               `json:"max_cache_tokens,omitempty"`
	MaxCacheCapacity   int               `json:"max_cache_capacity,omitempty"`
	MaxProcessedTokens int               `json:"max_processed_tokens,omitempty"`
	FullCaches         int               `json:"full_caches,omitempty"`
	RotatingCaches     int               `json:"rotating_caches,omitempty"`
	FixedCaches        int               `json:"fixed_caches,omitempty"`
	PagedCaches        int               `json:"paged_caches,omitempty"`
	QuantizedCaches    int               `json:"quantized_caches,omitempty"`
	UnknownCaches      int               `json:"unknown_caches,omitempty"`
	UnboundedCaches    int               `json:"unbounded_caches,omitempty"`
	LocalWindowLeaked  bool              `json:"local_window_leaked,omitempty"`
	Labels             map[string]string `json:"labels,omitempty"`
}

func (cacheProfile CacheProfile) Matched() bool {
	return cacheProfile.Contract != "" &&
		(cacheProfile.Architecture != "" ||
			cacheProfile.TotalCaches > 0 ||
			cacheProfile.SharedLayers > 0 ||
			cacheProfile.CachelessLayers > 0)
}

func (cacheProfile CacheProfile) Clone() CacheProfile {
	cacheProfile.Labels = cloneStringMap(cacheProfile.Labels)
	return cacheProfile
}

// BuildCacheProfile summarizes live cache observations into the model cache
// profile contract used by reactive engine selection.
func BuildCacheProfile(options CacheProfileOptions, observations []CacheObservation) CacheProfile {
	cacheProfile := CacheProfile{
		Contract:          CacheProfileContract,
		Architecture:      profile.ArchitectureID(options.Architecture),
		LocalWindowTokens: positiveCacheProfileInt(options.LocalWindowTokens),
		SharedLayers:      positiveCacheProfileInt(options.SharedLayers),
		CachelessLayers:   positiveCacheProfileInt(options.CachelessLayers),
	}
	for _, observation := range observations {
		cacheProfile.recordObservation(observation)
	}
	cacheProfile.Labels = ApplyCacheProfileLabels(cloneStringMap(options.Labels), cacheProfile)
	return cacheProfile.Clone()
}

func CacheProfileLabels(cacheProfile CacheProfile) map[string]string {
	return ApplyCacheProfileLabels(nil, cacheProfile)
}

func ApplyCacheProfileLabels(labels map[string]string, cacheProfile CacheProfile) map[string]string {
	if labels == nil {
		labels = map[string]string{}
	}
	if !cacheProfile.Matched() {
		return labels
	}
	labels["engine_cache_profile_contract"] = firstNonEmpty(cacheProfile.Contract, CacheProfileContract)
	labels["engine_cache_profile_local_window_leaked"] = strconv.FormatBool(cacheProfile.LocalWindowLeaked)
	if cacheProfile.Architecture != "" {
		labels["engine_cache_profile_architecture"] = cacheProfile.Architecture
	}
	writePositiveCacheProfileLabel(labels, "engine_cache_profile_total", cacheProfile.TotalCaches)
	writePositiveCacheProfileLabel(labels, "engine_cache_profile_local_count", cacheProfile.LocalCaches)
	writePositiveCacheProfileLabel(labels, "engine_cache_profile_global_count", cacheProfile.GlobalCaches)
	writePositiveCacheProfileLabel(labels, "engine_cache_profile_shared_layers", cacheProfile.SharedLayers)
	writePositiveCacheProfileLabel(labels, "engine_cache_profile_cacheless_layers", cacheProfile.CachelessLayers)
	writePositiveCacheProfileLabel(labels, "engine_cache_profile_local_window_tokens", cacheProfile.LocalWindowTokens)
	writePositiveCacheProfileLabel(labels, "engine_cache_profile_max_local_tokens", cacheProfile.MaxLocalTokens)
	writePositiveCacheProfileLabel(labels, "engine_cache_profile_max_local_capacity", cacheProfile.MaxLocalCapacity)
	writePositiveCacheProfileLabel(labels, "engine_cache_profile_max_global_tokens", cacheProfile.MaxGlobalTokens)
	writePositiveCacheProfileLabel(labels, "engine_cache_profile_max_global_capacity", cacheProfile.MaxGlobalCapacity)
	writePositiveCacheProfileLabel(labels, "engine_cache_profile_max_cache_tokens", cacheProfile.MaxCacheTokens)
	writePositiveCacheProfileLabel(labels, "engine_cache_profile_max_cache_capacity", cacheProfile.MaxCacheCapacity)
	writePositiveCacheProfileLabel(labels, "engine_cache_profile_max_processed_tokens", cacheProfile.MaxProcessedTokens)
	writePositiveCacheProfileLabel(labels, "engine_cache_profile_full_count", cacheProfile.FullCaches)
	writePositiveCacheProfileLabel(labels, "engine_cache_profile_rotating_count", cacheProfile.RotatingCaches)
	writePositiveCacheProfileLabel(labels, "engine_cache_profile_fixed_count", cacheProfile.FixedCaches)
	writePositiveCacheProfileLabel(labels, "engine_cache_profile_paged_count", cacheProfile.PagedCaches)
	writePositiveCacheProfileLabel(labels, "engine_cache_profile_quantized_count", cacheProfile.QuantizedCaches)
	writePositiveCacheProfileLabel(labels, "engine_cache_profile_unknown_count", cacheProfile.UnknownCaches)
	writePositiveCacheProfileLabel(labels, "engine_cache_profile_unbounded_count", cacheProfile.UnboundedCaches)
	return labels
}

func (cacheProfile *CacheProfile) recordObservation(observation CacheObservation) {
	if cacheProfile == nil {
		return
	}
	if observation.Cacheless {
		cacheProfile.CachelessLayers++
		if observation.Shared {
			cacheProfile.SharedLayers++
		}
		return
	}

	kind := cacheObservationKind(observation)
	tokens := positiveCacheProfileInt(observation.Tokens)
	capacity := positiveCacheProfileInt(observation.Capacity)
	processedTokens := positiveCacheProfileInt(observation.ProcessedTokens)
	if processedTokens == 0 {
		processedTokens = tokens
	}

	cacheProfile.TotalCaches++
	cacheProfile.MaxCacheTokens = max(cacheProfile.MaxCacheTokens, tokens)
	cacheProfile.MaxCacheCapacity = max(cacheProfile.MaxCacheCapacity, capacity)
	cacheProfile.MaxProcessedTokens = max(cacheProfile.MaxProcessedTokens, processedTokens)
	if !observation.Bounded {
		cacheProfile.UnboundedCaches++
	}
	if observation.Shared {
		cacheProfile.SharedLayers++
	}

	local := observation.Local || kind == CacheObservationKindRotating || kind == CacheObservationKindFixed
	global := observation.Global || kind == CacheObservationKindFull
	if local {
		cacheProfile.LocalCaches++
		cacheProfile.MaxLocalTokens = max(cacheProfile.MaxLocalTokens, tokens)
		cacheProfile.MaxLocalCapacity = max(cacheProfile.MaxLocalCapacity, capacity)
		if cacheProfile.LocalWindowTokens > 0 && (tokens > cacheProfile.LocalWindowTokens || capacity > cacheProfile.LocalWindowTokens || !observation.Bounded) {
			cacheProfile.LocalWindowLeaked = true
		}
	}
	if global {
		cacheProfile.GlobalCaches++
		cacheProfile.MaxGlobalTokens = max(cacheProfile.MaxGlobalTokens, tokens)
		cacheProfile.MaxGlobalCapacity = max(cacheProfile.MaxGlobalCapacity, capacity)
	}

	switch kind {
	case CacheObservationKindFull:
		cacheProfile.FullCaches++
	case CacheObservationKindRotating:
		cacheProfile.RotatingCaches++
	case CacheObservationKindFixed:
		cacheProfile.FixedCaches++
	case CacheObservationKindPaged:
		cacheProfile.PagedCaches++
	case CacheObservationKindQuantized:
		cacheProfile.QuantizedCaches++
	default:
		cacheProfile.UnknownCaches++
	}
}

func cacheObservationKind(observation CacheObservation) string {
	kind := normalizeCacheObservationKind(observation.Kind)
	if kind != "" {
		return kind
	}
	mode := normalizeCacheMode(observation.Mode)
	switch {
	case observation.Quantized || mode == CacheModeQ8 || mode == CacheModeKQ8VQ4 || mode == CacheModeTurboQuant:
		return CacheObservationKindQuantized
	case observation.Paged || mode == CacheModePaged:
		return CacheObservationKindPaged
	case observation.Fixed || mode == CacheModeFixed:
		return CacheObservationKindFixed
	case observation.Rotating:
		return CacheObservationKindRotating
	case observation.Full || mode == SequenceMixerCacheModeDefault || mode == CacheModeFP16:
		return CacheObservationKindFull
	default:
		return CacheObservationKindUnknown
	}
}

func normalizeCacheObservationKind(kind string) string {
	kind = strings.ToLower(strings.TrimSpace(kind))
	kind = strings.ReplaceAll(kind, "_", "-")
	switch kind {
	case "", "cache":
		return ""
	case "kv", "kv-cache", "full-attention", "global":
		return CacheObservationKindFull
	case "rotating", "rotating-kv", "sliding", "sliding-window", "local":
		return CacheObservationKindRotating
	case "fixed", "fixed-kv":
		return CacheObservationKindFixed
	case "paged", "paged-kv":
		return CacheObservationKindPaged
	case "quant", "quantized", "quantized-kv", "q8", "k-q8-v-q4", "turboquant":
		return CacheObservationKindQuantized
	case "unknown":
		return CacheObservationKindUnknown
	default:
		return kind
	}
}

func writePositiveCacheProfileLabel(labels map[string]string, key string, value int) {
	if value > 0 {
		labels[key] = strconv.Itoa(value)
	}
}

func positiveCacheProfileInt(value int) int {
	if value < 0 {
		return 0
	}
	return value
}
