// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"slices"
	"strconv"
	"sync"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/state"
)

const blockCacheRestoreMillisPerToken = 0.01

var metadataRuntimeOnlyCacheLabelKeys = []string{
	"kv_cache_constructible",
	"kv_cache_snapshot",
	"kv_device_backing",
	"kv_device_bytes",
	"kv_device_error",
	"kv_device_pages",
	"kv_device_restore",
	"kv_device_tokens",
}

var metadataShapeOnlyCacheLabelKeys = []string{
	"kv_cache_block_size",
	"kv_key_width",
	"kv_value_width",
}

var diskRuntimeOnlyCacheLabelKeys = []string{
	"disk_cache_restore",
	"disk_chunk_id",
	"disk_codec",
	"disk_encoding",
	"disk_kind",
}

// BlockCacheConfig describes compatibility identity for a metadata-first ROCm
// block-prefix cache. DiskStore writes portable cache refs only; native KV
// pages remain runtime-owned.
type BlockCacheConfig struct {
	ModelHash     string
	AdapterHash   string
	TokenizerHash string
	CacheMode     string
	DiskStore     state.BinaryWriter
	DiskURI       string
	Labels        map[string]string
	deviceDriver  nativeHIPDriver
}

// BlockCacheService is a metadata-first prompt/KV cache service.
type BlockCacheService struct {
	mu            sync.Mutex
	modelHash     string
	adapterHash   string
	tokenizerHash string
	cacheMode     string
	diskStore     state.BinaryWriter
	diskURI       string
	labels        map[string]string
	deviceDriver  nativeHIPDriver
	blocks        map[string]cacheBlock
	hits          uint64
	misses        uint64
	evictions     uint64
	restoreMillis float64
}

type cacheBlock struct {
	ref          inference.CacheBlockRef
	tokens       []int32
	labels       map[string]string
	diskPayload  []byte
	diskEncoding string
	diskKind     string
	diskBytes    uint64
	deviceKV     *rocmDeviceKVCache
}

type cacheBlockDiskPayload struct {
	ID            string            `json:"id"`
	Kind          string            `json:"kind"`
	ModelHash     string            `json:"model_hash,omitempty"`
	AdapterHash   string            `json:"adapter_hash,omitempty"`
	TokenizerHash string            `json:"tokenizer_hash,omitempty"`
	TokenStart    int               `json:"token_start"`
	TokenCount    int               `json:"token_count"`
	Encoding      string            `json:"encoding"`
	SizeBytes     uint64            `json:"size_bytes"`
	Labels        map[string]string `json:"labels,omitempty"`
}

// NewBlockCacheService creates a metadata-first cache service.
func NewBlockCacheService(cfg BlockCacheConfig) *BlockCacheService {
	mode := cfg.CacheMode
	if mode == "" {
		mode = "block-prefix"
	}
	return &BlockCacheService{
		modelHash:     cfg.ModelHash,
		adapterHash:   cfg.AdapterHash,
		tokenizerHash: cfg.TokenizerHash,
		cacheMode:     mode,
		diskStore:     cfg.DiskStore,
		diskURI:       cfg.DiskURI,
		labels:        cloneStringMap(cfg.Labels),
		deviceDriver:  cfg.deviceDriver,
		blocks:        map[string]cacheBlock{},
	}
}

func (service *BlockCacheService) CacheStats(ctx context.Context) (inference.CacheStats, error) {
	if service == nil {
		return inference.CacheStats{}, core.E("rocm.CacheStats", "cache service is nil", nil)
	}
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return inference.CacheStats{}, err
		}
	}
	service.mu.Lock()
	defer service.mu.Unlock()
	return service.statsLocked(), nil
}

func (service *BlockCacheService) WarmCache(ctx context.Context, req inference.CacheWarmRequest) (inference.CacheWarmResult, error) {
	if service == nil {
		return inference.CacheWarmResult{}, core.E("rocm.CacheWarm", "cache service is nil", nil)
	}
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return inference.CacheWarmResult{}, err
		}
	}
	tokens := append([]int32(nil), req.Tokens...)
	if len(tokens) == 0 && core.Trim(req.Prompt) != "" {
		tokens = approximateTokenIDs(req.Prompt)
	}
	if len(tokens) == 0 {
		return inference.CacheWarmResult{}, core.E("rocm.CacheWarm", "prompt or tokens are required", nil)
	}

	service.mu.Lock()
	defer service.mu.Unlock()
	if err := service.checkCompatibilityLocked(req); err != nil {
		return inference.CacheWarmResult{}, err
	}
	mode := firstNonEmptyString(req.Mode, service.cacheMode)
	labels := mergeStringMaps(service.labels, req.Labels)
	scrubDiskRuntimeLabels(labels)
	if service.diskStore == nil {
		delete(labels, "disk_uri")
	}
	sizeBytes, diskPayload, diskEncoding, diskKind, kvCache, err := service.cacheBlockPayload(tokens, mode, labels)
	if err != nil {
		return inference.CacheWarmResult{}, err
	}
	modelHash := firstNonEmptyString(req.Model.Hash, service.modelHash)
	adapterHash := firstNonEmptyString(req.Adapter.Hash, service.adapterHash)
	tokenizerHash := firstNonEmptyString(req.Labels["tokenizer_hash"], service.tokenizerHash)
	shape := cacheCompatibilityShape(labels)
	id := service.blockIDLocked(tokens, mode, modelHash, adapterHash, tokenizerHash, shape)
	block, ok := service.blocks[id]
	resultLabels := labels
	if ok {
		service.hits++
		service.restoreMillis += float64(block.ref.TokenCount) * blockCacheRestoreMillisPerToken
		resultLabels = block.labels
	} else {
		if restored, hit, err := service.restoreCacheBlockFromDiskLocked(ctx, id, tokens, mode, modelHash, adapterHash, tokenizerHash, labels); err != nil {
			return inference.CacheWarmResult{}, err
		} else if hit {
			service.hits++
			service.restoreMillis += float64(restored.ref.TokenCount) * blockCacheRestoreMillisPerToken
			service.blocks[id] = restored
			block = restored
			resultLabels = restored.labels
		} else {
			if prefixBlock, hit := service.prefixBlockLocked(tokens, mode, modelHash, adapterHash, tokenizerHash, shape); hit {
				service.hits++
				service.restoreMillis += float64(prefixBlock.ref.TokenCount) * blockCacheRestoreMillisPerToken
				labels["prefix_hit"] = "true"
			} else {
				service.misses++
			}
			block = cacheBlock{
				tokens:       tokens,
				labels:       labels,
				diskPayload:  diskPayload,
				diskEncoding: diskEncoding,
				diskKind:     diskKind,
				ref: inference.CacheBlockRef{
					ID:            id,
					Kind:          "prompt",
					ModelHash:     modelHash,
					AdapterHash:   adapterHash,
					TokenizerHash: tokenizerHash,
					TokenStart:    0,
					TokenCount:    len(tokens),
					SizeBytes:     sizeBytes,
					Encoding:      mode,
					Labels:        labels,
				},
			}
			service.attachDeviceKVCacheLocked(&block, kvCache)
			diskBytes, err := service.persistCacheBlockLocked(ctx, &block)
			if err != nil {
				return inference.CacheWarmResult{}, err
			}
			block.diskBytes = diskBytes
			service.blocks[id] = block
			resultLabels = block.labels
		}
	}
	stats := service.statsLocked()
	return inference.CacheWarmResult{Blocks: []inference.CacheBlockRef{cloneCacheBlockRef(block.ref)}, Stats: stats, Labels: cloneStringMap(resultLabels)}, nil
}

func (service *BlockCacheService) ClearCache(ctx context.Context, labels map[string]string) (inference.CacheStats, error) {
	if service == nil {
		return inference.CacheStats{}, core.E("rocm.CacheClear", "cache service is nil", nil)
	}
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return inference.CacheStats{}, err
		}
	}
	service.mu.Lock()
	defer service.mu.Unlock()
	var closeErr error
	for id, block := range service.blocks {
		if labelsMatch(block.labels, labels) {
			if err := block.closeDeviceKV(); err != nil && closeErr == nil {
				closeErr = err
			}
			delete(service.blocks, id)
			service.evictions++
		}
	}
	stats := service.statsLocked()
	if closeErr != nil {
		return stats, closeErr
	}
	return stats, nil
}

func (service *BlockCacheService) CacheEntries(ctx context.Context, labels map[string]string) ([]inference.CacheBlockRef, error) {
	if service == nil {
		return nil, core.E("rocm.CacheEntries", "cache service is nil", nil)
	}
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
	}
	service.mu.Lock()
	defer service.mu.Unlock()
	refs := make([]inference.CacheBlockRef, 0, len(service.blocks))
	for _, block := range service.blocks {
		if labelsMatch(block.labels, labels) {
			refs = append(refs, cloneCacheBlockRef(block.ref))
		}
	}
	slices.SortFunc(refs, func(a, b inference.CacheBlockRef) int {
		if a.ID < b.ID {
			return -1
		}
		if a.ID > b.ID {
			return 1
		}
		return 0
	})
	return refs, nil
}

func (service *BlockCacheService) Close() error {
	if service == nil {
		return nil
	}
	service.mu.Lock()
	defer service.mu.Unlock()
	var closeErr error
	for id, block := range service.blocks {
		if err := block.closeDeviceKV(); err != nil && closeErr == nil {
			closeErr = err
		}
		delete(service.blocks, id)
	}
	return closeErr
}

func (service *BlockCacheService) checkCompatibilityLocked(req inference.CacheWarmRequest) error {
	if service.modelHash != "" && req.Model.Hash != "" && service.modelHash != req.Model.Hash {
		return core.E("rocm.CacheWarm", "model hash mismatch", nil)
	}
	if service.adapterHash != "" && req.Adapter.Hash != "" && service.adapterHash != req.Adapter.Hash {
		return core.E("rocm.CacheWarm", "adapter hash mismatch", nil)
	}
	if service.tokenizerHash != "" && req.Labels["tokenizer_hash"] != "" && service.tokenizerHash != req.Labels["tokenizer_hash"] {
		return core.E("rocm.CacheWarm", "tokenizer hash mismatch", nil)
	}
	return nil
}

func (service *BlockCacheService) cacheBlockPayload(tokens []int32, mode string, labels map[string]string) (uint64, []byte, string, string, *rocmKVCache, error) {
	if isROCmKVCacheMode(mode) {
		blockSize, err := rocmKVCacheBlockSize(labels)
		if err != nil {
			return 0, nil, "", "", nil, core.E("rocm.CacheWarm", "parse KV cache block size", err)
		}
		cache, err := newROCmKVCache(mode, blockSize)
		if err != nil {
			return 0, nil, "", "", nil, core.E("rocm.CacheWarm", "construct KV cache page", err)
		}
		keyWidth, valueWidth, err := rocmKVVectorWidths(labels)
		if err != nil {
			return 0, nil, "", "", nil, core.E("rocm.CacheWarm", "parse KV vector widths", err)
		}
		keys, values := cacheWarmKVTensors(tokens, keyWidth, valueWidth)
		if err := cache.AppendVectors(0, keyWidth, valueWidth, keys, values); err != nil {
			return 0, nil, "", "", nil, core.E("rocm.CacheWarm", "encode KV cache page", err)
		}
		payload, err := cache.Snapshot()
		if err != nil {
			return 0, nil, "", "", nil, core.E("rocm.CacheWarm", "snapshot KV cache page", err)
		}
		labels["kv_backing"] = "package_local"
		labels["kv_cache_block_size"] = core.Sprintf("%d", blockSize)
		labels["kv_device_backing"] = "planned"
		labels["kv_pages"] = core.Sprintf("%d", cache.PageCount())
		labels["kv_tokens"] = core.Sprintf("%d", cache.TokenCount())
		labels["kv_cache_constructible"] = "true"
		labels["kv_cache_snapshot"] = "portable"
		labels["kv_key_width"] = core.Sprintf("%d", keyWidth)
		labels["kv_value_width"] = core.Sprintf("%d", valueWidth)
		return cache.MemoryBytes(), payload, rocmKVSnapshotEncoding, "rocm-cache-kv-state", cache, nil
	}
	if mode != "" && mode != "block-prefix" {
		return 0, nil, "", "", nil, core.E("rocm.CacheWarm", core.Sprintf("unsupported cache mode %q", mode), nil)
	}
	scrubMetadataShapeLabels(labels)
	scrubMetadataRuntimeLabels(labels)
	labels["kv_backing"] = "metadata"
	return uint64(len(tokens) * 4), nil, "rocm/cache-block+json", "rocm-cache-block", nil, nil
}

func rocmKVCacheBlockSize(labels map[string]string) (int, error) {
	return positiveIntLabel(labels, "kv_cache_block_size", defaultROCmKVBlockSize)
}

func rocmKVVectorWidths(labels map[string]string) (int, int, error) {
	keyWidth, err := positiveIntLabel(labels, "kv_key_width", 1)
	if err != nil {
		return 0, 0, err
	}
	valueWidth, err := positiveIntLabel(labels, "kv_value_width", keyWidth)
	if err != nil {
		return 0, 0, err
	}
	return keyWidth, valueWidth, nil
}

func positiveIntLabel(labels map[string]string, key string, fallback int) (int, error) {
	value := core.Trim(labels[key])
	if value == "" {
		return fallback, nil
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed <= 0 {
		return 0, core.E("rocm.CacheWarm", key+" must be a positive integer", err)
	}
	return parsed, nil
}

func cacheWarmKVTensors(tokens []int32, keyWidth, valueWidth int) ([]float32, []float32) {
	keys := make([]float32, len(tokens)*keyWidth)
	values := make([]float32, len(tokens)*valueWidth)
	cacheWarmKVTensorsInto(tokens, keyWidth, valueWidth, keys, values)
	return keys, values
}

func cacheWarmKVTensorsInto(tokens []int32, keyWidth, valueWidth int, keys, values []float32) {
	for i, token := range tokens {
		for j := 0; j < keyWidth; j++ {
			keys[i*keyWidth+j] = float32(token) + float32(j)/1000
		}
		for j := 0; j < valueWidth; j++ {
			values[i*valueWidth+j] = float32(token) - float32(j)/1000
		}
	}
}

func isROCmKVCacheMode(mode string) bool {
	switch mode {
	case rocmKVCacheModeFP16, rocmKVCacheModeQ8, rocmKVCacheModeKQ8VQ4:
		return true
	default:
		return false
	}
}

func (service *BlockCacheService) statsLocked() inference.CacheStats {
	var memoryBytes uint64
	var diskBytes uint64
	var cachedTokens int
	var largestBlock cacheBlock
	for _, block := range service.blocks {
		memoryBytes += block.ref.SizeBytes
		diskBytes += block.diskBytes
		cachedTokens += block.ref.TokenCount
		if block.ref.TokenCount > largestBlock.ref.TokenCount {
			largestBlock = block
		}
	}
	total := service.hits + service.misses
	var hitRate float64
	if total > 0 {
		hitRate = float64(service.hits) / float64(total)
	}
	labels := cloneStringMap(service.labels)
	if labels == nil {
		labels = map[string]string{}
	}
	delete(labels, "disk_uri")
	scrubDiskRuntimeLabels(labels)
	scrubMetadataShapeLabels(labels)
	scrubMetadataRuntimeLabels(labels)
	if cachedTokens > 0 {
		labels["cached_tokens"] = core.Sprintf("%d", cachedTokens)
	}
	cacheMode := service.cacheMode
	if largestBlock.ref.Encoding != "" {
		cacheMode = largestBlock.ref.Encoding
	}
	for _, key := range []string{"kv_backing", "kv_cache_block_size", "kv_cache_constructible", "kv_cache_snapshot", "kv_device_backing", "kv_device_bytes", "kv_device_error", "kv_device_pages", "kv_device_restore", "kv_device_tokens", "kv_key_width", "kv_value_width", "kv_pages", "kv_tokens", "disk_cache_restore", "disk_uri", "disk_codec", "disk_chunk_id", "disk_encoding", "disk_kind"} {
		if largestBlock.labels[key] != "" {
			labels[key] = largestBlock.labels[key]
		}
	}
	labels = rocmApplyCacheProfileLabels(labels, service.cacheProfileLocked(""))
	return inference.CacheStats{
		Blocks:        len(service.blocks),
		MemoryBytes:   memoryBytes,
		DiskBytes:     diskBytes,
		Hits:          service.hits,
		Misses:        service.misses,
		Evictions:     service.evictions,
		HitRate:       hitRate,
		RestoreMillis: service.restoreMillis,
		CacheMode:     cacheMode,
		Labels:        labels,
	}
}

func (service *BlockCacheService) persistCacheBlockLocked(ctx context.Context, block *cacheBlock) (uint64, error) {
	if service == nil || service.diskStore == nil || block == nil {
		return 0, nil
	}
	uri := firstNonEmptyString(block.labels["disk_uri"], service.diskURI)
	if uri == "" {
		uri = "rocm://cache/" + block.ref.ID
	}
	block.labels["disk_uri"] = uri
	payload := append([]byte(nil), block.diskPayload...)
	diskEncoding := firstNonEmptyString(block.diskEncoding, "rocm/cache-block+json")
	diskKind := firstNonEmptyString(block.diskKind, "rocm-cache-block")
	block.labels["disk_encoding"] = diskEncoding
	block.labels["disk_kind"] = diskKind
	if len(payload) == 0 {
		var err error
		payload, err = json.Marshal(cacheBlockDiskPayload{
			ID:            block.ref.ID,
			Kind:          block.ref.Kind,
			ModelHash:     block.ref.ModelHash,
			AdapterHash:   block.ref.AdapterHash,
			TokenizerHash: block.ref.TokenizerHash,
			TokenStart:    block.ref.TokenStart,
			TokenCount:    block.ref.TokenCount,
			Encoding:      block.ref.Encoding,
			SizeBytes:     block.ref.SizeBytes,
			Labels:        cloneStringMap(block.labels),
		})
		if err != nil {
			return 0, core.E("rocm.CacheWarm", "encode disk cache ref", err)
		}
	} else if block.diskEncoding == rocmKVSnapshotEncoding {
		annotated, err := annotateCacheKVSnapshot(payload, block)
		if err != nil {
			return 0, err
		}
		payload = annotated
	}
	ref, err := service.diskStore.PutBytes(ctx, payload, state.PutOptions{
		URI:   uri,
		Kind:  diskKind,
		Track: diskEncoding,
		Tags:  cloneStringMap(block.labels),
	})
	if err != nil {
		return 0, core.E("rocm.CacheWarm", "write disk cache ref", err)
	}
	block.labels["disk_codec"] = firstNonEmptyString(ref.Codec, state.CodecMemory)
	block.labels["disk_chunk_id"] = core.Sprintf("%d", ref.ChunkID)
	block.ref.Labels = block.labels
	return uint64(len(payload)), nil
}

func annotateCacheKVSnapshot(payload []byte, block *cacheBlock) ([]byte, error) {
	var snapshot rocmKVCacheSnapshot
	if err := json.Unmarshal(payload, &snapshot); err != nil {
		return nil, core.E("rocm.CacheWarm", "decode KV disk cache snapshot", err)
	}
	snapshot.CacheBlockID = block.ref.ID
	snapshot.ModelHash = block.ref.ModelHash
	snapshot.AdapterHash = block.ref.AdapterHash
	snapshot.TokenizerHash = block.ref.TokenizerHash
	snapshot.Labels = cloneStringMap(block.labels)
	annotated, err := json.Marshal(snapshot)
	if err != nil {
		return nil, core.E("rocm.CacheWarm", "encode KV disk cache snapshot", err)
	}
	return annotated, nil
}

func (service *BlockCacheService) restoreCacheBlockFromDiskLocked(ctx context.Context, id string, tokens []int32, mode, modelHash, adapterHash, tokenizerHash string, labels map[string]string) (cacheBlock, bool, error) {
	if service == nil || service.diskStore == nil {
		return cacheBlock{}, false, nil
	}
	store, ok := service.diskStore.(state.Store)
	if !ok || store == nil {
		return cacheBlock{}, false, nil
	}
	uri := service.cacheBlockDiskURI(id, labels)
	chunk, err := state.ResolveURI(ctx, store, uri)
	if err != nil {
		if rocmStateChunkNotFound(err) {
			return cacheBlock{}, false, nil
		}
		return cacheBlock{}, false, core.E("rocm.CacheWarm", "resolve disk cache ref", err)
	}
	if isROCmKVCacheMode(mode) {
		return service.restoreKVCacheBlockFromDisk(id, uri, chunk, tokens, mode, modelHash, adapterHash, tokenizerHash, labels)
	}
	return service.restoreMetadataCacheBlockFromDisk(id, uri, chunk, tokens, mode, modelHash, adapterHash, tokenizerHash, labels)
}

func (service *BlockCacheService) restoreKVCacheBlockFromDisk(id, uri string, chunk state.Chunk, tokens []int32, mode, modelHash, adapterHash, tokenizerHash string, labels map[string]string) (cacheBlock, bool, error) {
	var snapshot rocmKVCacheSnapshot
	if err := json.Unmarshal(chunk.Data, &snapshot); err != nil {
		return cacheBlock{}, false, core.E("rocm.CacheWarm", "decode disk KV cache ref", err)
	}
	if snapshot.CacheBlockID != "" && snapshot.CacheBlockID != id {
		return cacheBlock{}, false, core.E("rocm.CacheWarm", "disk KV cache ref does not match warm request", nil)
	}
	if snapshot.ModelHash != "" && modelHash != "" && snapshot.ModelHash != modelHash {
		return cacheBlock{}, false, core.E("rocm.CacheWarm", "disk KV cache ref does not match warm request", nil)
	}
	if snapshot.AdapterHash != "" && adapterHash != "" && snapshot.AdapterHash != adapterHash {
		return cacheBlock{}, false, core.E("rocm.CacheWarm", "disk KV cache ref does not match warm request", nil)
	}
	if snapshot.TokenizerHash != "" && tokenizerHash != "" && snapshot.TokenizerHash != tokenizerHash {
		return cacheBlock{}, false, core.E("rocm.CacheWarm", "disk KV cache ref does not match warm request", nil)
	}
	if cachePayloadIdentityLabelMismatch(snapshot.Labels, "model_hash", modelHash) ||
		cachePayloadIdentityLabelMismatch(snapshot.Labels, "adapter_hash", adapterHash) ||
		cachePayloadIdentityLabelMismatch(snapshot.Labels, "tokenizer_hash", tokenizerHash) ||
		cachePayloadShapeLabelMismatch(snapshot.Labels, labels) {
		return cacheBlock{}, false, core.E("rocm.CacheWarm", "disk KV cache ref does not match warm request", nil)
	}
	cache, err := newROCmKVCacheFromSnapshot(chunk.Data)
	if err != nil {
		return cacheBlock{}, false, core.E("rocm.CacheWarm", "restore disk KV cache ref", err)
	}
	if cache.mode != mode || cache.TokenCount() != len(tokens) {
		return cacheBlock{}, false, core.E("rocm.CacheWarm", "disk KV cache ref does not match warm request", nil)
	}
	if err := validateCacheKVSnapshotTokens(cache, tokens); err != nil {
		return cacheBlock{}, false, err
	}
	cacheLabels := mergeStringMaps(labels, cache.Stats().Labels)
	cacheLabels["disk_cache_restore"] = "hit"
	cacheLabels["disk_uri"] = uri
	cacheLabels["disk_chunk_id"] = core.Sprintf("%d", chunk.Ref.ChunkID)
	cacheLabels["disk_codec"] = firstNonEmptyString(chunk.Ref.Codec, state.CodecMemory)
	cacheLabels["disk_encoding"] = rocmKVSnapshotEncoding
	cacheLabels["disk_kind"] = "rocm-cache-kv-state"
	cacheLabels["kv_cache_snapshot"] = "portable"
	block := cacheBlock{
		tokens:       append([]int32(nil), tokens...),
		labels:       cacheLabels,
		diskEncoding: rocmKVSnapshotEncoding,
		diskKind:     "rocm-cache-kv-state",
		diskBytes:    uint64(len(chunk.Data)),
		ref: inference.CacheBlockRef{
			ID:            id,
			Kind:          "prompt",
			ModelHash:     modelHash,
			AdapterHash:   adapterHash,
			TokenizerHash: tokenizerHash,
			TokenStart:    0,
			TokenCount:    len(tokens),
			SizeBytes:     cache.MemoryBytes(),
			Encoding:      mode,
			Labels:        cacheLabels,
		},
	}
	service.attachDeviceKVCacheLocked(&block, cache)
	return block, true, nil
}

func (service *BlockCacheService) attachDeviceKVCacheLocked(block *cacheBlock, cache *rocmKVCache) {
	if service == nil || block == nil || cache == nil || service.deviceDriver == nil {
		return
	}
	if !service.deviceDriver.Available() {
		block.labels["kv_device_backing"] = "unavailable"
		block.ref.Labels = block.labels
		return
	}
	device, err := cache.MirrorToDevice(service.deviceDriver)
	if err != nil {
		block.labels["kv_device_backing"] = "failed"
		block.labels["kv_device_error"] = err.Error()
		block.ref.Labels = block.labels
		return
	}
	block.deviceKV = device
	block.labels["kv_device_backing"] = "mirrored"
	block.labels["kv_device_pages"] = core.Sprintf("%d", device.PageCount())
	block.labels["kv_device_tokens"] = core.Sprintf("%d", device.TokenCount())
	block.labels["kv_device_bytes"] = core.Sprintf("%d", device.MemoryBytes())
	if block.labels["disk_cache_restore"] == "hit" {
		block.labels["kv_device_restore"] = "mirrored"
	}
	block.ref.Labels = block.labels
}

func (block *cacheBlock) closeDeviceKV() error {
	if block == nil || block.deviceKV == nil {
		return nil
	}
	err := block.deviceKV.Close()
	block.deviceKV = nil
	return err
}

func validateCacheKVSnapshotTokens(cache *rocmKVCache, tokens []int32) error {
	keyWidth, valueWidth, ok := cache.LastVectorWidths()
	if !ok {
		return core.E("rocm.CacheWarm", "disk KV cache ref does not match warm request", nil)
	}
	expected, err := newROCmKVCache(cache.mode, cache.blockSize)
	if err != nil {
		return err
	}
	maxBlockTokens := cache.blockSize
	if maxBlockTokens <= 0 || maxBlockTokens > len(tokens) {
		maxBlockTokens = len(tokens)
	}
	keys := make([]float32, maxBlockTokens*keyWidth)
	values := make([]float32, maxBlockTokens*valueWidth)
	for tokenStart := 0; tokenStart < len(tokens); tokenStart += maxBlockTokens {
		tokenEnd := tokenStart + maxBlockTokens
		if tokenEnd > len(tokens) {
			tokenEnd = len(tokens)
		}
		blockTokens := tokens[tokenStart:tokenEnd]
		keyCount := len(blockTokens) * keyWidth
		valueCount := len(blockTokens) * valueWidth
		cacheWarmKVTensorsInto(blockTokens, keyWidth, valueWidth, keys[:keyCount], values[:valueCount])
		if err := expected.AppendVectors(tokenStart, keyWidth, valueWidth, keys[:keyCount], values[:valueCount]); err != nil {
			return err
		}
	}
	if !rocmKVCacheBlocksEqual(cache.blocks, expected.blocks) {
		return core.E("rocm.CacheWarm", "disk KV cache ref does not match warm request", nil)
	}
	return nil
}

func rocmKVCacheBlocksEqual(left, right []rocmKVCacheBlock) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index].tokenStart != right[index].tokenStart ||
			left[index].tokenCount != right[index].tokenCount ||
			left[index].keyWidth != right[index].keyWidth ||
			left[index].valueWidth != right[index].valueWidth ||
			!rocmKVEncodedTensorEqual(left[index].key, right[index].key) ||
			!rocmKVEncodedTensorEqual(left[index].value, right[index].value) {
			return false
		}
	}
	return true
}

func rocmKVEncodedTensorEqual(left, right rocmKVEncodedTensor) bool {
	return left.encoding == right.encoding &&
		left.length == right.length &&
		left.scale == right.scale &&
		left.sizeBytes == right.sizeBytes &&
		slices.Equal(left.scales, right.scales) &&
		slices.Equal(left.f16, right.f16) &&
		slices.Equal(left.q8, right.q8) &&
		slices.Equal(left.packedQ4, right.packedQ4)
}

func float32SlicesEqual(left, right []float32) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func (service *BlockCacheService) restoreMetadataCacheBlockFromDisk(id, uri string, chunk state.Chunk, tokens []int32, mode, modelHash, adapterHash, tokenizerHash string, labels map[string]string) (cacheBlock, bool, error) {
	var payload cacheBlockDiskPayload
	if err := json.Unmarshal(chunk.Data, &payload); err != nil {
		return cacheBlock{}, false, core.E("rocm.CacheWarm", "decode disk cache ref", err)
	}
	if payload.ID != id || payload.TokenCount != len(tokens) || payload.Encoding != mode {
		return cacheBlock{}, false, core.E("rocm.CacheWarm", "disk cache ref does not match warm request", nil)
	}
	if payload.Kind != "" && payload.Kind != "prompt" {
		return cacheBlock{}, false, core.E("rocm.CacheWarm", "disk cache ref does not match warm request", nil)
	}
	if payload.TokenStart != 0 {
		return cacheBlock{}, false, core.E("rocm.CacheWarm", "disk cache ref does not match warm request", nil)
	}
	if payload.SizeBytes != uint64(len(tokens)*4) {
		return cacheBlock{}, false, core.E("rocm.CacheWarm", "disk cache ref does not match warm request", nil)
	}
	if payload.ModelHash != "" && modelHash != "" && payload.ModelHash != modelHash {
		return cacheBlock{}, false, core.E("rocm.CacheWarm", "disk cache ref does not match warm request", nil)
	}
	if payload.AdapterHash != "" && adapterHash != "" && payload.AdapterHash != adapterHash {
		return cacheBlock{}, false, core.E("rocm.CacheWarm", "disk cache ref does not match warm request", nil)
	}
	if payload.TokenizerHash != "" && tokenizerHash != "" && payload.TokenizerHash != tokenizerHash {
		return cacheBlock{}, false, core.E("rocm.CacheWarm", "disk cache ref does not match warm request", nil)
	}
	if cachePayloadIdentityLabelMismatch(payload.Labels, "model_hash", modelHash) ||
		cachePayloadIdentityLabelMismatch(payload.Labels, "adapter_hash", adapterHash) ||
		cachePayloadIdentityLabelMismatch(payload.Labels, "tokenizer_hash", tokenizerHash) {
		return cacheBlock{}, false, core.E("rocm.CacheWarm", "disk cache ref does not match warm request", nil)
	}
	if cachePayloadHasShapeLabels(payload.Labels) && cacheCompatibilityShape(payload.Labels) != cacheCompatibilityShape(labels) {
		return cacheBlock{}, false, core.E("rocm.CacheWarm", "disk cache ref does not match warm request", nil)
	}
	if cachePayloadHasMetadataRuntimeLabels(payload.Labels) {
		return cacheBlock{}, false, core.E("rocm.CacheWarm", "disk cache ref does not match warm request", nil)
	}
	cacheLabels := mergeStringMaps(labels, payload.Labels)
	cacheLabels["disk_cache_restore"] = "hit"
	cacheLabels["disk_uri"] = uri
	cacheLabels["disk_chunk_id"] = core.Sprintf("%d", chunk.Ref.ChunkID)
	cacheLabels["disk_codec"] = firstNonEmptyString(chunk.Ref.Codec, state.CodecMemory)
	cacheLabels["disk_encoding"] = "rocm/cache-block+json"
	cacheLabels["disk_kind"] = "rocm-cache-block"
	return cacheBlock{
		tokens:       append([]int32(nil), tokens...),
		labels:       cacheLabels,
		diskEncoding: "rocm/cache-block+json",
		diskKind:     "rocm-cache-block",
		diskBytes:    uint64(len(chunk.Data)),
		ref: inference.CacheBlockRef{
			ID:            id,
			Kind:          firstNonEmptyString(payload.Kind, "prompt"),
			ModelHash:     firstNonEmptyString(payload.ModelHash, modelHash),
			AdapterHash:   firstNonEmptyString(payload.AdapterHash, adapterHash),
			TokenizerHash: firstNonEmptyString(payload.TokenizerHash, tokenizerHash),
			TokenStart:    payload.TokenStart,
			TokenCount:    payload.TokenCount,
			SizeBytes:     payload.SizeBytes,
			Encoding:      mode,
			Labels:        cacheLabels,
		},
	}, true, nil
}

func cachePayloadHasShapeLabels(labels map[string]string) bool {
	if labels["kv_backing"] != "" {
		return true
	}
	for _, key := range metadataShapeOnlyCacheLabelKeys {
		if labels[key] != "" {
			return true
		}
	}
	return false
}

func cachePayloadHasMetadataRuntimeLabels(labels map[string]string) bool {
	for _, key := range metadataRuntimeOnlyCacheLabelKeys {
		if labels[key] != "" {
			return true
		}
	}
	return false
}

func scrubMetadataRuntimeLabels(labels map[string]string) {
	for _, key := range metadataRuntimeOnlyCacheLabelKeys {
		delete(labels, key)
	}
}

func scrubMetadataShapeLabels(labels map[string]string) {
	for _, key := range metadataShapeOnlyCacheLabelKeys {
		delete(labels, key)
	}
}

func scrubDiskRuntimeLabels(labels map[string]string) {
	for _, key := range diskRuntimeOnlyCacheLabelKeys {
		delete(labels, key)
	}
}

func cachePayloadIdentityLabelMismatch(labels map[string]string, key, want string) bool {
	if labels == nil || want == "" {
		return false
	}
	got := labels[key]
	return got != "" && got != want
}

func cachePayloadShapeLabelMismatch(payloadLabels, requestLabels map[string]string) bool {
	if len(payloadLabels) == 0 {
		return false
	}
	if payloadLabels["kv_backing"] != "" && payloadLabels["kv_backing"] != requestLabels["kv_backing"] {
		return true
	}
	for _, key := range metadataShapeOnlyCacheLabelKeys {
		if payloadLabels[key] != "" && payloadLabels[key] != requestLabels[key] {
			return true
		}
	}
	return false
}

func (service *BlockCacheService) cacheBlockDiskURI(id string, labels map[string]string) string {
	uri := ""
	if labels != nil {
		uri = labels["disk_uri"]
	}
	uri = firstNonEmptyString(uri, service.diskURI)
	if uri != "" {
		return uri
	}
	return "rocm://cache/" + id
}

func rocmStateChunkNotFound(err error) bool {
	if err == nil {
		return false
	}
	return core.Contains(err.Error(), "not found")
}

func (service *BlockCacheService) blockIDLocked(tokens []int32, mode, modelHash, adapterHash, tokenizerHash, shape string) string {
	hasher := sha256.New()
	_, _ = hasher.Write([]byte(core.Concat(modelHash, "\x00", adapterHash, "\x00", tokenizerHash, "\x00", mode, "\x00", shape)))
	for _, token := range tokens {
		_, _ = hasher.Write([]byte(core.Sprintf("\x00%d", token)))
	}
	return "rocm-cache-" + hex.EncodeToString(hasher.Sum(nil))[:24]
}

func (service *BlockCacheService) prefixBlockLocked(tokens []int32, mode, modelHash, adapterHash, tokenizerHash, shape string) (cacheBlock, bool) {
	var best cacheBlock
	var bestLen int
	for _, block := range service.blocks {
		if block.ref.Encoding != mode || len(block.tokens) == 0 || len(block.tokens) > len(tokens) {
			continue
		}
		if block.ref.ModelHash != modelHash || block.ref.AdapterHash != adapterHash || block.ref.TokenizerHash != tokenizerHash {
			continue
		}
		if cacheCompatibilityShape(block.labels) != shape {
			continue
		}
		matches := true
		for i := range block.tokens {
			if block.tokens[i] != tokens[i] {
				matches = false
				break
			}
		}
		if matches && len(block.tokens) > bestLen {
			best = block
			bestLen = len(block.tokens)
		}
	}
	return best, bestLen > 0
}

func cacheCompatibilityShape(labels map[string]string) string {
	return core.Concat(
		labels["kv_backing"], "\x00",
		labels["kv_cache_block_size"], "\x00",
		labels["kv_key_width"], "\x00",
		labels["kv_value_width"],
	)
}

func cloneCacheBlockRef(ref inference.CacheBlockRef) inference.CacheBlockRef {
	ref.Labels = cloneStringMap(ref.Labels)
	return ref
}

func (m *rocmModel) CacheStats(ctx context.Context) (stats inference.CacheStats, err error) {
	m.clearLastError()
	defer func() {
		if err != nil {
			m.setLastFailure(err)
		}
	}()
	return m.blockCacheService().CacheStats(ctx)
}

func (m *rocmModel) WarmCache(ctx context.Context, req inference.CacheWarmRequest) (result inference.CacheWarmResult, err error) {
	m.clearLastError()
	defer func() {
		if err != nil {
			m.setLastFailure(err)
		}
	}()
	return m.blockCacheService().WarmCache(ctx, req)
}

func (m *rocmModel) ClearCache(ctx context.Context, labels map[string]string) (stats inference.CacheStats, err error) {
	m.clearLastError()
	defer func() {
		if err != nil {
			m.setLastFailure(err)
		}
	}()
	return m.blockCacheService().ClearCache(ctx, labels)
}

func (m *rocmModel) CacheEntries(ctx context.Context, labels map[string]string) (entries []inference.CacheBlockRef, err error) {
	m.clearLastError()
	defer func() {
		if err != nil {
			m.setLastFailure(err)
		}
	}()
	return m.blockCacheService().CacheEntries(ctx, labels)
}

func (m *rocmModel) blockCacheService() *BlockCacheService {
	if m == nil {
		return NewBlockCacheService(BlockCacheConfig{CacheMode: "block-prefix"})
	}
	m.stateMutex.Lock()
	defer m.stateMutex.Unlock()
	if m.cache == nil {
		m.cache = NewBlockCacheService(BlockCacheConfig{
			ModelHash:    m.modelIdentity().Hash,
			AdapterHash:  m.adapter.Hash,
			CacheMode:    "block-prefix",
			Labels:       map[string]string{"backend": "rocm"},
			deviceDriver: m.blockCacheDeviceDriver(),
		})
	}
	return m.cache
}

func (m *rocmModel) blockCacheDeviceDriver() nativeHIPDriver {
	if m == nil {
		return nil
	}
	loaded, ok := m.native.(*hipLoadedModel)
	if !ok || loaded == nil {
		return nil
	}
	return loaded.driver
}

func labelsMatch(labels, filter map[string]string) bool {
	if len(filter) == 0 {
		return true
	}
	for key, value := range filter {
		if labels[key] != value {
			return false
		}
	}
	return true
}
