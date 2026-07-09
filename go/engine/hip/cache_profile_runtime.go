// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"strconv"

	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/inference/engine/hip/model"
)

func (cache *rocmKVCache) CacheProfile(architecture string) rocmmodel.CacheProfile {
	observation, ok := rocmKVCacheObservation(cache, nil)
	if !ok {
		return rocmmodel.CacheProfile{}
	}
	return rocmmodel.BuildCacheProfile(rocmmodel.CacheProfileOptions{Architecture: architecture}, []rocmmodel.CacheObservation{observation})
}

func (cache *rocmDeviceKVCache) CacheProfile(architecture string) rocmmodel.CacheProfile {
	observation, ok := rocmDeviceKVCacheObservation(cache, nil)
	if !ok {
		return rocmmodel.CacheProfile{}
	}
	return rocmmodel.BuildCacheProfile(rocmmodel.CacheProfileOptions{Architecture: architecture}, []rocmmodel.CacheObservation{observation})
}

func (service *BlockCacheService) CacheProfile(ctx context.Context, architecture string) (rocmmodel.CacheProfile, error) {
	if service == nil {
		return rocmmodel.CacheProfile{}, nil
	}
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return rocmmodel.CacheProfile{}, err
		}
	}
	service.mu.Lock()
	defer service.mu.Unlock()
	return service.cacheProfileLocked(architecture), nil
}

func (m *rocmModel) CacheProfile(ctx context.Context) (profile rocmmodel.CacheProfile, err error) {
	m.clearLastError()
	defer func() {
		if err != nil {
			m.setLastFailure(err)
		}
	}()
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return rocmmodel.CacheProfile{}, err
		}
	}
	architecture := ""
	if m != nil {
		architecture = m.ModelIdentity().Architecture
	}
	return m.blockCacheService().CacheProfile(ctx, architecture)
}

func (service *BlockCacheService) cacheProfileLocked(architecture string) rocmmodel.CacheProfile {
	if service == nil || len(service.blocks) == 0 {
		return rocmmodel.CacheProfile{}
	}
	observations := make([]rocmmodel.CacheObservation, 0, len(service.blocks))
	for _, block := range service.blocks {
		if observation, ok := cacheBlockObservation(block, service.cacheMode); ok {
			observations = append(observations, observation)
		}
	}
	if len(observations) == 0 {
		return rocmmodel.CacheProfile{}
	}
	return rocmmodel.BuildCacheProfile(rocmmodel.CacheProfileOptions{
		Architecture: architecture,
		Labels:       cloneStringMap(service.labels),
	}, observations)
}

func cacheBlockObservation(block cacheBlock, fallbackMode string) (rocmmodel.CacheObservation, bool) {
	tokens := block.ref.TokenCount
	mode := firstNonEmptyString(block.ref.Encoding, fallbackMode)
	labels := cloneStringMap(block.labels)
	if labels == nil {
		labels = cloneStringMap(block.ref.Labels)
	}
	capacity := rocmCacheObservationCapacity(tokens, cacheObservationLabelInt(labels, "kv_cache_block_size"), cacheObservationLabelInt(labels, "kv_pages"))
	if capacity == 0 {
		capacity = tokens
	}
	if tokens <= 0 && capacity <= 0 {
		return rocmmodel.CacheObservation{}, false
	}
	observation := rocmmodel.CacheObservation{
		Kind:     rocmCacheObservationKind(mode),
		Mode:     mode,
		Tokens:   tokens,
		Capacity: capacity,
		Bounded:  capacity > 0,
		Paged:    true,
		Labels:   labels,
	}
	if observation.Kind == rocmmodel.CacheObservationKindQuantized {
		observation.Quantized = true
	}
	return observation, true
}

func rocmKVCacheObservation(cache *rocmKVCache, labels map[string]string) (rocmmodel.CacheObservation, bool) {
	if cache == nil || cache.PageCount() == 0 {
		return rocmmodel.CacheObservation{}, false
	}
	labels = cloneStringMap(labels)
	if labels == nil {
		labels = rocmKVCacheObservationLabels(cache)
	}
	tokens := cache.TokenCount()
	capacity := rocmCacheObservationCapacity(tokens, cache.blockSize, cache.PageCount())
	observation := rocmmodel.CacheObservation{
		Kind:     rocmCacheObservationKind(cache.mode),
		Mode:     cache.mode,
		Tokens:   tokens,
		Capacity: capacity,
		Bounded:  capacity > 0,
		Paged:    cache.PageCount() > 0,
		Labels:   labels,
	}
	if observation.Kind == rocmmodel.CacheObservationKindQuantized {
		observation.Quantized = true
	}
	return observation, true
}

func rocmDeviceKVCacheObservation(cache *rocmDeviceKVCache, labels map[string]string) (rocmmodel.CacheObservation, bool) {
	if cache == nil || cache.PageCount() == 0 {
		return rocmmodel.CacheObservation{}, false
	}
	labels = cloneStringMap(labels)
	if labels == nil {
		labels = rocmDeviceKVCacheObservationLabels(cache)
	}
	tokens := cache.TokenCount()
	capacity := rocmCacheObservationCapacity(tokens, cache.blockSize, cache.PageCount())
	observation := rocmmodel.CacheObservation{
		Kind:     rocmCacheObservationKind(cache.mode),
		Mode:     cache.mode,
		Tokens:   tokens,
		Capacity: capacity,
		Bounded:  capacity > 0,
		Paged:    cache.PageCount() > 0,
		Labels:   labels,
	}
	if observation.Kind == rocmmodel.CacheObservationKindQuantized {
		observation.Quantized = true
	}
	return observation, true
}

func rocmKVCacheObservationLabels(cache *rocmKVCache) map[string]string {
	if cache == nil {
		return nil
	}
	labels := map[string]string{
		"kv_backing":          "package_local",
		"kv_block_size":       strconv.Itoa(cache.blockSize),
		"kv_cache_block_size": strconv.Itoa(cache.blockSize),
		"kv_device_backing":   "planned",
		"kv_pages":            strconv.Itoa(cache.PageCount()),
		"kv_tokens":           strconv.Itoa(cache.TokenCount()),
	}
	if keyWidth, valueWidth, ok := cache.LastVectorWidths(); ok {
		labels["kv_key_width"] = strconv.Itoa(keyWidth)
		labels["kv_value_width"] = strconv.Itoa(valueWidth)
	}
	return labels
}

func rocmDeviceKVCacheObservationLabels(cache *rocmDeviceKVCache) map[string]string {
	if cache == nil {
		return nil
	}
	labels := make(map[string]string, 8)
	cache.addStatsLabels(labels)
	return labels
}

func rocmCacheObservationKind(mode string) string {
	switch mode {
	case rocmKVCacheModeQ8, rocmKVCacheModeKQ8VQ4:
		return rocmmodel.CacheObservationKindQuantized
	case rocmKVCacheModeFP16:
		return rocmmodel.CacheObservationKindPaged
	case "block-prefix", "retained-state", "attached-drafter":
		return rocmmodel.CacheObservationKindPaged
	default:
		if isROCmKVCacheMode(mode) {
			return rocmmodel.CacheObservationKindQuantized
		}
		return rocmmodel.CacheObservationKindPaged
	}
}

func rocmCacheObservationCapacity(tokens, blockSize, pages int) int {
	switch {
	case blockSize > 0 && pages > 0:
		return blockSize * pages
	case tokens > 0:
		return tokens
	default:
		return 0
	}
}

func cacheObservationLabelInt(labels map[string]string, key string) int {
	value, err := positiveIntLabel(labels, key, 0)
	if err != nil {
		return 0
	}
	return value
}

func rocmApplyCacheProfileLabels(labels map[string]string, profile rocmmodel.CacheProfile) map[string]string {
	if !profile.Matched() {
		return labels
	}
	return rocmmodel.ApplyCacheProfileLabels(labels, profile)
}

func rocmCacheProfileFromStats(architecture string, stats inference.CacheStats) rocmmodel.CacheProfile {
	tokens := cacheStatsCachedTokens(stats)
	if tokens == 0 {
		tokens = cacheObservationLabelInt(stats.Labels, "kv_tokens")
	}
	capacity := rocmCacheObservationCapacity(tokens, cacheObservationLabelInt(stats.Labels, "kv_cache_block_size"), stats.Blocks)
	if tokens <= 0 && capacity <= 0 {
		return rocmmodel.CacheProfile{}
	}
	observation := rocmmodel.CacheObservation{
		Kind:     rocmCacheObservationKind(stats.CacheMode),
		Mode:     stats.CacheMode,
		Tokens:   tokens,
		Capacity: capacity,
		Bounded:  capacity > 0,
		Paged:    stats.Blocks > 0,
		Labels:   cloneStringMap(stats.Labels),
	}
	if observation.Kind == rocmmodel.CacheObservationKindQuantized {
		observation.Quantized = true
	}
	return rocmmodel.BuildCacheProfile(rocmmodel.CacheProfileOptions{Architecture: architecture}, []rocmmodel.CacheObservation{observation})
}
