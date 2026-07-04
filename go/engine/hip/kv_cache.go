// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"encoding/json"
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/state"
)

const (
	rocmKVCacheModeFP16       = "fp16"
	rocmKVCacheModeQ8         = "q8"
	rocmKVCacheModeKQ8VQ4     = "k-q8-v-q4"
	rocmKVEncodingFP16        = "fp16"
	rocmKVEncodingQ8          = "q8"
	rocmKVEncodingQ4          = "q4"
	rocmKVEncodingQ8Rows      = "q8-rows"
	rocmKVEncodingQ4Rows      = "q4-rows"
	rocmKVEncodingQ8RowsI     = "q8-rows-interleaved"
	rocmKVEncodingQ4RowsI     = "q4-rows-interleaved"
	rocmKVSnapshotEncoding    = "rocm/kv-cache+json"
	rocmKVBlockBundleEncoding = "rocm/kv-cache-block-bundle+json"
	rocmKVBlockRawEncoding    = "rocm/kv-cache-block+raw"
	rocmKVBlockBundleKind     = "rocm-kv-state-block-bundle"
	rocmKVBlockKind           = "rocm-kv-state-block"
	defaultROCmKVBlockSize    = 16
	rocmKVRestoreMillisUnit   = 0.01
)

type rocmKVCache struct {
	mode          string
	blockSize     int
	keyWidth      int
	valueWidth    int
	blocks        []rocmKVCacheBlock
	hits          uint64
	misses        uint64
	restoreMillis float64
}

type rocmKVCacheBlock struct {
	tokenStart int
	tokenCount int
	keyWidth   int
	valueWidth int
	key        rocmKVEncodedTensor
	value      rocmKVEncodedTensor
}

type rocmKVEncodedTensor struct {
	encoding  string
	length    int
	scale     float32
	scales    []float32
	f16       []uint16
	q8        []int8
	packedQ4  []byte
	sizeBytes uint64
}

type rocmKVCacheSnapshot struct {
	Version       int                        `json:"version"`
	Mode          string                     `json:"mode"`
	BlockSize     int                        `json:"block_size"`
	CacheBlockID  string                     `json:"cache_block_id,omitempty"`
	ModelHash     string                     `json:"model_hash,omitempty"`
	AdapterHash   string                     `json:"adapter_hash,omitempty"`
	TokenizerHash string                     `json:"tokenizer_hash,omitempty"`
	Labels        map[string]string          `json:"labels,omitempty"`
	Blocks        []rocmKVCacheBlockSnapshot `json:"blocks"`
}

type rocmKVCacheBlockSnapshot struct {
	TokenStart int                         `json:"token_start"`
	TokenCount int                         `json:"token_count"`
	KeyWidth   int                         `json:"key_width,omitempty"`
	ValueWidth int                         `json:"value_width,omitempty"`
	Key        rocmKVEncodedTensorSnapshot `json:"key"`
	Value      rocmKVEncodedTensorSnapshot `json:"value"`
}

type rocmKVBlockBundleSnapshot struct {
	Version     int                    `json:"version"`
	Kind        string                 `json:"kind"`
	Mode        string                 `json:"mode"`
	BlockSize   int                    `json:"block_size"`
	TokenCount  int                    `json:"token_count"`
	MemoryBytes uint64                 `json:"memory_bytes,omitempty"`
	Labels      map[string]string      `json:"labels,omitempty"`
	Blocks      []rocmKVBlockBundleRef `json:"blocks,omitempty"`
}

type rocmKVBlockBundleRef struct {
	Index      int               `json:"index"`
	URI        string            `json:"uri"`
	ChunkID    int               `json:"chunk_id,omitempty"`
	State      state.ChunkRef    `json:"state,omitempty"`
	TokenStart int               `json:"token_start"`
	TokenCount int               `json:"token_count"`
	KeyWidth   int               `json:"key_width,omitempty"`
	ValueWidth int               `json:"value_width,omitempty"`
	SizeBytes  uint64            `json:"size_bytes,omitempty"`
	Encoding   string            `json:"encoding,omitempty"`
	Labels     map[string]string `json:"labels,omitempty"`
}

type rocmKVBlockBundleWakeSnapshot struct {
	Kind       string                     `json:"kind"`
	Mode       string                     `json:"mode"`
	BlockSize  int                        `json:"block_size"`
	TokenCount int                        `json:"token_count"`
	Blocks     []rocmKVBlockBundleWakeRef `json:"blocks,omitempty"`
}

type rocmKVBlockBundleWakeRef struct {
	Index      int    `json:"index"`
	URI        string `json:"uri"`
	uriRaw     []byte
	ChunkID    int            `json:"chunk_id,omitempty"`
	State      state.ChunkRef `json:"state,omitempty"`
	TokenStart int            `json:"token_start"`
	TokenCount int            `json:"token_count"`
	KeyWidth   int            `json:"key_width,omitempty"`
	ValueWidth int            `json:"value_width,omitempty"`
	SizeBytes  uint64         `json:"size_bytes,omitempty"`
	Encoding   string         `json:"encoding,omitempty"`
}

func (ref rocmKVBlockBundleWakeRef) fullBundleRef() rocmKVBlockBundleRef {
	uri := ref.URI
	if uri == "" && len(ref.uriRaw) > 0 {
		uri = string(ref.uriRaw)
	}
	return rocmKVBlockBundleRef{
		Index:      ref.Index,
		URI:        uri,
		ChunkID:    ref.ChunkID,
		State:      ref.State,
		TokenStart: ref.TokenStart,
		TokenCount: ref.TokenCount,
		KeyWidth:   ref.KeyWidth,
		ValueWidth: ref.ValueWidth,
		SizeBytes:  ref.SizeBytes,
		Encoding:   ref.Encoding,
	}
}

type rocmKVEncodedTensorSnapshot struct {
	Encoding  string    `json:"encoding"`
	Length    int       `json:"length"`
	Scale     float32   `json:"scale,omitempty"`
	Scales    []float32 `json:"scales,omitempty"`
	F16       []uint16  `json:"f16,omitempty"`
	Q8        []int8    `json:"q8,omitempty"`
	PackedQ4  []byte    `json:"packed_q4,omitempty"`
	SizeBytes uint64    `json:"size_bytes,omitempty"`
}

func newROCmKVCache(mode string, blockSize int) (*rocmKVCache, error) {
	if mode == "" {
		mode = rocmKVCacheModeFP16
	}
	switch mode {
	case rocmKVCacheModeFP16, rocmKVCacheModeQ8, rocmKVCacheModeKQ8VQ4:
	default:
		return nil, core.E("rocm.KVCache", core.Sprintf("unsupported cache mode %q", mode), nil)
	}
	if blockSize <= 0 {
		blockSize = defaultROCmKVBlockSize
	}
	return &rocmKVCache{mode: mode, blockSize: blockSize}, nil
}

func newROCmKVCacheFromSnapshot(data []byte) (*rocmKVCache, error) {
	if len(data) == 0 {
		return nil, core.E("rocm.KVCache.Snapshot", "snapshot payload is empty", nil)
	}
	var snapshot rocmKVCacheSnapshot
	if err := json.Unmarshal(data, &snapshot); err != nil {
		return nil, core.E("rocm.KVCache.Snapshot", "decode snapshot", err)
	}
	if snapshot.Version != 1 {
		return nil, core.E("rocm.KVCache.Snapshot", core.Sprintf("unsupported snapshot version %d", snapshot.Version), nil)
	}
	cache, err := newROCmKVCache(snapshot.Mode, snapshot.BlockSize)
	if err != nil {
		return nil, err
	}
	for _, blockSnapshot := range snapshot.Blocks {
		block, err := blockSnapshot.toBlock()
		if err != nil {
			return nil, err
		}
		if err := cache.validateVectorShape(block.keyWidth, block.valueWidth); err != nil {
			return nil, err
		}
		cache.blocks, err = insertROCmKVCacheBlock(cache.blocks, block)
		if err != nil {
			return nil, err
		}
		cache.setVectorShape(block.keyWidth, block.valueWidth)
	}
	return cache, nil
}

func (cache *rocmKVCache) Append(tokenStart int, keys, values []float32) error {
	return cache.AppendVectors(tokenStart, 1, 1, keys, values)
}

func (cache *rocmKVCache) AppendToken(tokenStart int, key, value []float32) error {
	return cache.AppendVectors(tokenStart, len(key), len(value), key, value)
}

func (cache *rocmKVCache) AppendVectors(tokenStart, keyWidth, valueWidth int, keys, values []float32) error {
	if cache == nil {
		return core.E("rocm.KVCache.Append", "cache is nil", nil)
	}
	if tokenStart < 0 {
		return core.E("rocm.KVCache.Append", "token start must be non-negative", nil)
	}
	if keyWidth <= 0 || valueWidth <= 0 {
		return core.E("rocm.KVCache.Append", "key and value widths must be positive", nil)
	}
	if len(keys) == 0 || len(values) == 0 {
		return core.E("rocm.KVCache.Append", "key and value tensors must be non-empty", nil)
	}
	if len(keys)%keyWidth != 0 || len(values)%valueWidth != 0 {
		return core.E("rocm.KVCache.Append", "key and value tensor lengths must align with vector widths", nil)
	}
	tokenCount := len(keys) / keyWidth
	if tokenCount != len(values)/valueWidth {
		return core.E("rocm.KVCache.Append", "key and value tensors must describe the same token count", nil)
	}
	if err := cache.validateVectorShape(keyWidth, valueWidth); err != nil {
		return err
	}
	keyEncoding, valueEncoding := rocmKVEncodingsForMode(cache.mode)
	blocks := make([]rocmKVCacheBlock, 0, (tokenCount+cache.blockSize-1)/cache.blockSize)
	for tokenOffset := 0; tokenOffset < tokenCount; tokenOffset += cache.blockSize {
		tokenEnd := tokenOffset + cache.blockSize
		if tokenEnd > tokenCount {
			tokenEnd = tokenCount
		}
		keyStart := tokenOffset * keyWidth
		keyEnd := tokenEnd * keyWidth
		valueStart := tokenOffset * valueWidth
		valueEnd := tokenEnd * valueWidth
		key, err := encodeROCmKVTensor(keyEncoding, keys[keyStart:keyEnd])
		if err != nil {
			return err
		}
		value, err := encodeROCmKVTensor(valueEncoding, values[valueStart:valueEnd])
		if err != nil {
			return err
		}
		blocks = append(blocks, rocmKVCacheBlock{
			tokenStart: tokenStart + tokenOffset,
			tokenCount: tokenEnd - tokenOffset,
			keyWidth:   keyWidth,
			valueWidth: valueWidth,
			key:        key,
			value:      value,
		})
	}
	next := append([]rocmKVCacheBlock(nil), cache.blocks...)
	for _, block := range blocks {
		var err error
		next, err = insertROCmKVCacheBlock(next, block)
		if err != nil {
			return err
		}
	}
	cache.blocks = next
	cache.setVectorShape(keyWidth, valueWidth)
	return nil
}

func (cache *rocmKVCache) Snapshot() ([]byte, error) {
	if cache == nil {
		return nil, core.E("rocm.KVCache.Snapshot", "cache is nil", nil)
	}
	snapshot := rocmKVCacheSnapshot{
		Version:   1,
		Mode:      cache.mode,
		BlockSize: cache.blockSize,
		Blocks:    make([]rocmKVCacheBlockSnapshot, 0, len(cache.blocks)),
	}
	for _, block := range cache.blocks {
		snapshot.Blocks = append(snapshot.Blocks, block.snapshot())
	}
	payload, err := json.Marshal(snapshot)
	if err != nil {
		return nil, core.E("rocm.KVCache.Snapshot", "encode snapshot", err)
	}
	return payload, nil
}

func (cache *rocmKVCache) snapshotBlock(block rocmKVCacheBlock) ([]byte, error) {
	if cache == nil {
		return nil, core.E("rocm.KVCache.SnapshotBlock", "cache is nil", nil)
	}
	snapshot := rocmKVCacheSnapshot{
		Version:   1,
		Mode:      cache.mode,
		BlockSize: cache.blockSize,
		Blocks:    []rocmKVCacheBlockSnapshot{block.snapshot()},
	}
	payload, err := json.Marshal(snapshot)
	if err != nil {
		return nil, core.E("rocm.KVCache.SnapshotBlock", "encode snapshot block", err)
	}
	return payload, nil
}

func (cache *rocmKVCache) Clone() (*rocmKVCache, error) {
	if cache == nil {
		return nil, core.E("rocm.KVCache.Clone", "cache is nil", nil)
	}
	clone := &rocmKVCache{
		mode:          cache.mode,
		blockSize:     cache.blockSize,
		keyWidth:      cache.keyWidth,
		valueWidth:    cache.valueWidth,
		blocks:        make([]rocmKVCacheBlock, len(cache.blocks)),
		hits:          cache.hits,
		misses:        cache.misses,
		restoreMillis: cache.restoreMillis,
	}
	for i, block := range cache.blocks {
		clone.blocks[i] = block.clone()
	}
	return clone, nil
}

func (cache *rocmKVCache) Prefix(tokenCount int) (*rocmKVCache, error) {
	if cache == nil {
		return nil, core.E("rocm.KVCache.Prefix", "cache is nil", nil)
	}
	if tokenCount <= 0 {
		return nil, core.E("rocm.KVCache.Prefix", "token count must be positive", nil)
	}
	if tokenCount > cache.TokenCount() {
		return nil, core.E("rocm.KVCache.Prefix", "token count exceeds cache", nil)
	}
	if tokenCount == cache.TokenCount() {
		return cache.Clone()
	}
	keyWidth, valueWidth, ok := cache.restoreVectorWidths()
	if !ok {
		return nil, core.E("rocm.KVCache.Prefix", "cache vector shape is not available", nil)
	}
	prefix := &rocmKVCache{
		mode:       cache.mode,
		blockSize:  cache.blockSize,
		keyWidth:   keyWidth,
		valueWidth: valueWidth,
		blocks:     make([]rocmKVCacheBlock, 0, len(cache.blocks)),
	}
	cursor := 0
	for _, block := range cache.blocks {
		if block.tokenStart != cursor {
			return nil, core.E("rocm.KVCache.Prefix", "cache block range is not available", nil)
		}
		blockEnd := block.tokenStart + block.tokenCount
		if blockEnd <= tokenCount {
			prefix.blocks = append(prefix.blocks, block.clone())
			cursor = blockEnd
			if cursor == tokenCount {
				return prefix, nil
			}
			continue
		}
		partialTokens := tokenCount - block.tokenStart
		if partialTokens <= 0 {
			break
		}
		key, err := block.key.prefixRows(block.keyWidth, partialTokens)
		if err != nil {
			return nil, core.E("rocm.KVCache.Prefix", "prefix partial key block", err)
		}
		value, err := block.value.prefixRows(block.valueWidth, partialTokens)
		if err != nil {
			return nil, core.E("rocm.KVCache.Prefix", "prefix partial value block", err)
		}
		prefix.blocks = append(prefix.blocks, rocmKVCacheBlock{
			tokenStart: block.tokenStart,
			tokenCount: partialTokens,
			keyWidth:   block.keyWidth,
			valueWidth: block.valueWidth,
			key:        key,
			value:      value,
		})
		return prefix, nil
	}
	return nil, core.E("rocm.KVCache.Prefix", "cache block range is not available", nil)
}

func (cache *rocmKVCache) Restore(tokenStart, tokenCount int) ([]float32, []float32, error) {
	if cache == nil {
		return nil, nil, core.E("rocm.KVCache.Restore", "cache is nil", nil)
	}
	if tokenStart < 0 || tokenCount <= 0 {
		return nil, nil, core.E("rocm.KVCache.Restore", "token range must be positive", nil)
	}
	if len(cache.blocks) == 0 {
		cache.misses++
		return nil, nil, core.E("rocm.KVCache.Restore", "cache block range is not available", nil)
	}
	keyWidth, valueWidth, ok := cache.restoreVectorWidths()
	if !ok {
		return nil, nil, core.E("rocm.KVCache.Restore", "cache vector shape is not available", nil)
	}
	keys := make([]float32, tokenCount*keyWidth)
	values := make([]float32, tokenCount*valueWidth)
	return cache.RestoreInto(tokenStart, tokenCount, keys, values)
}

func (cache *rocmKVCache) RestoreInto(tokenStart, tokenCount int, keys, values []float32) ([]float32, []float32, error) {
	if cache == nil {
		return nil, nil, core.E("rocm.KVCache.Restore", "cache is nil", nil)
	}
	if tokenStart < 0 || tokenCount <= 0 {
		return nil, nil, core.E("rocm.KVCache.Restore", "token range must be positive", nil)
	}
	keyWidth, valueWidth, ok := cache.restoreVectorWidths()
	if !ok {
		return nil, nil, core.E("rocm.KVCache.Restore", "cache vector shape is not available", nil)
	}
	if len(keys) < tokenCount*keyWidth || len(values) < tokenCount*valueWidth {
		return nil, nil, core.E("rocm.KVCache.Restore", "restore output buffers are too small", nil)
	}
	keys = keys[:tokenCount*keyWidth]
	values = values[:tokenCount*valueWidth]
	end := tokenStart + tokenCount
	cursor := tokenStart
	for _, block := range cache.blocks {
		blockEnd := block.tokenStart + block.tokenCount
		if blockEnd <= cursor || block.tokenStart >= end {
			continue
		}
		if block.tokenStart > cursor {
			break
		}
		startOffset := cursor - block.tokenStart
		endOffset := block.tokenCount
		if blockEnd > end {
			endOffset = end - block.tokenStart
		}
		outputTokenOffset := cursor - tokenStart
		if err := block.key.decodeRowsRangeInto(keys[outputTokenOffset*block.keyWidth:], block.keyWidth, startOffset, endOffset); err != nil {
			return nil, nil, core.E("rocm.KVCache.Restore", "decode key block", err)
		}
		if err := block.value.decodeRowsRangeInto(values[outputTokenOffset*block.valueWidth:], block.valueWidth, startOffset, endOffset); err != nil {
			return nil, nil, core.E("rocm.KVCache.Restore", "decode value block", err)
		}
		cursor = block.tokenStart + endOffset
		if cursor == end {
			cache.hits++
			cache.restoreMillis += float64(tokenCount) * rocmKVRestoreMillisUnit
			return keys, values, nil
		}
	}
	cache.misses++
	return nil, nil, core.E("rocm.KVCache.Restore", "cache block range is not available", nil)
}

func (cache *rocmKVCache) restoreVectorWidths() (int, int, bool) {
	if cache == nil {
		return 0, 0, false
	}
	if cache.keyWidth > 0 && cache.valueWidth > 0 {
		return cache.keyWidth, cache.valueWidth, true
	}
	return cache.LastVectorWidths()
}

func (cache *rocmKVCache) Stats() inference.CacheStats {
	if cache == nil {
		return inference.CacheStats{}
	}
	total := cache.hits + cache.misses
	hitRate := float64(0)
	if total > 0 {
		hitRate = float64(cache.hits) / float64(total)
	}
	labels := map[string]string{
		"kv_backing":          "package_local",
		"kv_block_size":       core.Sprintf("%d", cache.blockSize),
		"kv_cache_block_size": core.Sprintf("%d", cache.blockSize),
		"kv_device_backing":   "planned",
		"kv_pages":            core.Sprintf("%d", cache.PageCount()),
		"kv_tokens":           core.Sprintf("%d", cache.TokenCount()),
	}
	if keyWidth, valueWidth, ok := cache.LastVectorWidths(); ok {
		labels["kv_key_width"] = core.Sprintf("%d", keyWidth)
		labels["kv_value_width"] = core.Sprintf("%d", valueWidth)
	}
	labels = rocmApplyCacheProfileLabels(labels, cache.CacheProfile(""))
	return inference.CacheStats{
		Blocks:        len(cache.blocks),
		MemoryBytes:   cache.MemoryBytes(),
		Hits:          cache.hits,
		Misses:        cache.misses,
		HitRate:       hitRate,
		RestoreMillis: cache.restoreMillis,
		CacheMode:     cache.mode,
		Labels:        labels,
	}
}

func (cache *rocmKVCache) MemoryBytes() uint64 {
	if cache == nil {
		return 0
	}
	var total uint64
	for _, block := range cache.blocks {
		total += block.key.sizeBytes + block.value.sizeBytes
	}
	return total
}

func (cache *rocmKVCache) PageCount() int {
	if cache == nil {
		return 0
	}
	return len(cache.blocks)
}

func (cache *rocmKVCache) TokenCount() int {
	if cache == nil {
		return 0
	}
	var maxEnd int
	for _, block := range cache.blocks {
		if end := block.tokenStart + block.tokenCount; end > maxEnd {
			maxEnd = end
		}
	}
	return maxEnd
}

func (cache *rocmKVCache) LastVectorWidths() (int, int, bool) {
	if cache == nil || len(cache.blocks) == 0 {
		return 0, 0, false
	}
	if cache.keyWidth > 0 && cache.valueWidth > 0 {
		return cache.keyWidth, cache.valueWidth, true
	}
	last := cache.blocks[len(cache.blocks)-1]
	return last.keyWidth, last.valueWidth, true
}

func (cache *rocmKVCache) validateVectorShape(keyWidth, valueWidth int) error {
	if cache == nil {
		return core.E("rocm.KVCache.Append", "cache is nil", nil)
	}
	if cache.keyWidth == 0 && cache.valueWidth == 0 {
		return nil
	}
	if cache.keyWidth != keyWidth || cache.valueWidth != valueWidth {
		return core.E("rocm.KVCache.Append", "KV vector widths must match existing cache shape", nil)
	}
	return nil
}

func (cache *rocmKVCache) setVectorShape(keyWidth, valueWidth int) {
	if cache == nil || cache.keyWidth != 0 || cache.valueWidth != 0 {
		return
	}
	cache.keyWidth = keyWidth
	cache.valueWidth = valueWidth
}

func (block rocmKVCacheBlock) snapshot() rocmKVCacheBlockSnapshot {
	return rocmKVCacheBlockSnapshot{
		TokenStart: block.tokenStart,
		TokenCount: block.tokenCount,
		KeyWidth:   block.keyWidth,
		ValueWidth: block.valueWidth,
		Key:        block.key.snapshot(),
		Value:      block.value.snapshot(),
	}
}

func (block rocmKVCacheBlock) clone() rocmKVCacheBlock {
	return rocmKVCacheBlock{
		tokenStart: block.tokenStart,
		tokenCount: block.tokenCount,
		keyWidth:   block.keyWidth,
		valueWidth: block.valueWidth,
		key:        block.key.clone(),
		value:      block.value.clone(),
	}
}

func insertROCmKVCacheBlock(blocks []rocmKVCacheBlock, block rocmKVCacheBlock) ([]rocmKVCacheBlock, error) {
	if block.tokenStart < 0 || block.tokenCount <= 0 {
		return nil, core.E("rocm.KVCache.Pages", "invalid block token range", nil)
	}
	blockEnd := block.tokenStart + block.tokenCount
	if blockEnd <= block.tokenStart {
		return nil, core.E("rocm.KVCache.Pages", "invalid block token range", nil)
	}
	index := 0
	for index < len(blocks) && blocks[index].tokenStart < block.tokenStart {
		index++
	}
	if index > 0 {
		previousEnd := blocks[index-1].tokenStart + blocks[index-1].tokenCount
		if previousEnd > block.tokenStart {
			return nil, core.E("rocm.KVCache.Pages", "cache block ranges must not overlap", nil)
		}
	}
	if index < len(blocks) && blockEnd > blocks[index].tokenStart {
		return nil, core.E("rocm.KVCache.Pages", "cache block ranges must not overlap", nil)
	}
	blocks = append(blocks, rocmKVCacheBlock{})
	copy(blocks[index+1:], blocks[index:])
	blocks[index] = block
	return blocks, nil
}

func (snapshot rocmKVCacheBlockSnapshot) toBlock() (rocmKVCacheBlock, error) {
	if snapshot.TokenStart < 0 || snapshot.TokenCount <= 0 {
		return rocmKVCacheBlock{}, core.E("rocm.KVCache.Snapshot", "invalid block token range", nil)
	}
	keyWidth := firstPositiveInt(snapshot.KeyWidth, 1)
	valueWidth := firstPositiveInt(snapshot.ValueWidth, 1)
	key, err := snapshot.Key.toTensor()
	if err != nil {
		return rocmKVCacheBlock{}, err
	}
	value, err := snapshot.Value.toTensor()
	if err != nil {
		return rocmKVCacheBlock{}, err
	}
	if key.length != snapshot.TokenCount*keyWidth || value.length != snapshot.TokenCount*valueWidth {
		return rocmKVCacheBlock{}, core.E("rocm.KVCache.Snapshot", "block tensor length mismatch", nil)
	}
	return rocmKVCacheBlock{
		tokenStart: snapshot.TokenStart,
		tokenCount: snapshot.TokenCount,
		keyWidth:   keyWidth,
		valueWidth: valueWidth,
		key:        key,
		value:      value,
	}, nil
}

func rocmKVEncodingsForMode(mode string) (string, string) {
	switch mode {
	case rocmKVCacheModeQ8:
		return rocmKVEncodingQ8, rocmKVEncodingQ8
	case rocmKVCacheModeKQ8VQ4:
		return rocmKVEncodingQ8, rocmKVEncodingQ4
	default:
		return rocmKVEncodingFP16, rocmKVEncodingFP16
	}
}

func (tensor rocmKVEncodedTensor) snapshot() rocmKVEncodedTensorSnapshot {
	return rocmKVEncodedTensorSnapshot{
		Encoding:  tensor.encoding,
		Length:    tensor.length,
		Scale:     tensor.scale,
		Scales:    append([]float32(nil), tensor.scales...),
		F16:       append([]uint16(nil), tensor.f16...),
		Q8:        append([]int8(nil), tensor.q8...),
		PackedQ4:  append([]byte(nil), tensor.packedQ4...),
		SizeBytes: tensor.sizeBytes,
	}
}

func (tensor rocmKVEncodedTensor) clone() rocmKVEncodedTensor {
	return rocmKVEncodedTensor{
		encoding:  tensor.encoding,
		length:    tensor.length,
		scale:     tensor.scale,
		scales:    append([]float32(nil), tensor.scales...),
		f16:       append([]uint16(nil), tensor.f16...),
		q8:        append([]int8(nil), tensor.q8...),
		packedQ4:  append([]byte(nil), tensor.packedQ4...),
		sizeBytes: tensor.sizeBytes,
	}
}

func (tensor rocmKVEncodedTensor) prefixRows(rowWidth, rows int) (rocmKVEncodedTensor, error) {
	if rowWidth <= 0 || rows <= 0 || tensor.length <= 0 || tensor.length%rowWidth != 0 {
		return rocmKVEncodedTensor{}, core.E("rocm.KVCache.Prefix", "tensor row shape mismatch", nil)
	}
	rowCount := tensor.length / rowWidth
	if rows > rowCount {
		return rocmKVEncodedTensor{}, core.E("rocm.KVCache.Prefix", "tensor prefix row count mismatch", nil)
	}
	if rows == rowCount {
		return tensor.clone(), nil
	}
	prefixLength := rows * rowWidth
	switch tensor.encoding {
	case rocmKVEncodingFP16:
		if len(tensor.f16) < prefixLength {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.Prefix", "fp16 tensor length mismatch", nil)
		}
		return rocmKVEncodedTensor{
			encoding:  tensor.encoding,
			length:    prefixLength,
			scale:     tensor.scale,
			f16:       tensor.f16[:prefixLength],
			sizeBytes: uint64(prefixLength * 2),
		}, nil
	case rocmKVEncodingQ8:
		if len(tensor.q8) < prefixLength {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.Prefix", "q8 tensor length mismatch", nil)
		}
		return rocmKVEncodedTensor{
			encoding:  tensor.encoding,
			length:    prefixLength,
			scale:     tensor.scale,
			q8:        tensor.q8[:prefixLength],
			sizeBytes: uint64(4 + prefixLength),
		}, nil
	case rocmKVEncodingQ8Rows, rocmKVEncodingQ8RowsI:
		if len(tensor.q8) < prefixLength || len(tensor.scales) < rows {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.Prefix", "q8 row tensor length mismatch", nil)
		}
		sizeBytes := uint64(rows*4 + prefixLength)
		if tensor.encoding == rocmKVEncodingQ8RowsI {
			sizeBytes = uint64(rows * (4 + rowWidth))
		}
		return rocmKVEncodedTensor{
			encoding:  tensor.encoding,
			length:    prefixLength,
			scales:    tensor.scales[:rows],
			q8:        tensor.q8[:prefixLength],
			sizeBytes: sizeBytes,
		}, nil
	case rocmKVEncodingQ4:
		packedLength := (prefixLength + 1) / 2
		if len(tensor.packedQ4) < packedLength {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.Prefix", "q4 tensor length mismatch", nil)
		}
		packed := tensor.packedQ4[:packedLength]
		if prefixLength%2 == 1 {
			packed = append([]byte(nil), packed...)
			packed[len(packed)-1] &= 0x0f
		}
		return rocmKVEncodedTensor{
			encoding:  tensor.encoding,
			length:    prefixLength,
			scale:     tensor.scale,
			packedQ4:  packed,
			sizeBytes: uint64(4 + packedLength),
		}, nil
	case rocmKVEncodingQ4Rows, rocmKVEncodingQ4RowsI:
		packedLength := (prefixLength + 1) / 2
		if len(tensor.packedQ4) < packedLength || len(tensor.scales) < rows {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.Prefix", "q4 row tensor length mismatch", nil)
		}
		packed := tensor.packedQ4[:packedLength]
		if prefixLength%2 == 1 {
			packed = append([]byte(nil), packed...)
			packed[len(packed)-1] &= 0x0f
		}
		sizeBytes := uint64(rows*4 + packedLength)
		if tensor.encoding == rocmKVEncodingQ4RowsI {
			sizeBytes = uint64(rows * (4 + (rowWidth+1)/2))
		}
		return rocmKVEncodedTensor{
			encoding:  tensor.encoding,
			length:    prefixLength,
			scales:    tensor.scales[:rows],
			packedQ4:  packed,
			sizeBytes: sizeBytes,
		}, nil
	default:
		return rocmKVEncodedTensor{}, core.E("rocm.KVCache.Prefix", core.Sprintf("unsupported tensor encoding %q", tensor.encoding), nil)
	}
}

func (snapshot rocmKVEncodedTensorSnapshot) toTensor() (rocmKVEncodedTensor, error) {
	if snapshot.Length <= 0 {
		return rocmKVEncodedTensor{}, core.E("rocm.KVCache.Snapshot", "tensor length must be positive", nil)
	}
	tensor := rocmKVEncodedTensor{
		encoding:  snapshot.Encoding,
		length:    snapshot.Length,
		scale:     snapshot.Scale,
		scales:    append([]float32(nil), snapshot.Scales...),
		f16:       append([]uint16(nil), snapshot.F16...),
		q8:        append([]int8(nil), snapshot.Q8...),
		packedQ4:  append([]byte(nil), snapshot.PackedQ4...),
		sizeBytes: snapshot.SizeBytes,
	}
	switch tensor.encoding {
	case rocmKVEncodingFP16:
		if len(tensor.f16) != tensor.length {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.Snapshot", "fp16 tensor length mismatch", nil)
		}
		if tensor.sizeBytes == 0 {
			tensor.sizeBytes = uint64(len(tensor.f16) * 2)
		}
	case rocmKVEncodingQ8:
		if len(tensor.q8) != tensor.length {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.Snapshot", "q8 tensor length mismatch", nil)
		}
		if tensor.scale <= 0 {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.Snapshot", "q8 scale must be positive", nil)
		}
		if tensor.sizeBytes == 0 {
			tensor.sizeBytes = uint64(len(tensor.q8) + 4)
		}
	case rocmKVEncodingQ8Rows, rocmKVEncodingQ8RowsI:
		if len(tensor.q8) != tensor.length {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.Snapshot", "q8 row tensor length mismatch", nil)
		}
		if len(tensor.scales) == 0 {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.Snapshot", "q8 row scales are required", nil)
		}
		for _, scale := range tensor.scales {
			if scale <= 0 {
				return rocmKVEncodedTensor{}, core.E("rocm.KVCache.Snapshot", "q8 row scale must be positive", nil)
			}
		}
		if tensor.sizeBytes == 0 {
			tensor.sizeBytes = uint64(len(tensor.q8) + len(tensor.scales)*4)
		}
	case rocmKVEncodingQ4:
		if len(tensor.packedQ4) != (tensor.length+1)/2 {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.Snapshot", "q4 tensor length mismatch", nil)
		}
		if tensor.scale <= 0 {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.Snapshot", "q4 scale must be positive", nil)
		}
		if tensor.sizeBytes == 0 {
			tensor.sizeBytes = uint64(len(tensor.packedQ4) + 4)
		}
	case rocmKVEncodingQ4Rows, rocmKVEncodingQ4RowsI:
		if len(tensor.packedQ4) != (tensor.length+1)/2 {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.Snapshot", "q4 row tensor length mismatch", nil)
		}
		if len(tensor.scales) == 0 {
			return rocmKVEncodedTensor{}, core.E("rocm.KVCache.Snapshot", "q4 row scales are required", nil)
		}
		for _, scale := range tensor.scales {
			if scale <= 0 {
				return rocmKVEncodedTensor{}, core.E("rocm.KVCache.Snapshot", "q4 row scale must be positive", nil)
			}
		}
		if tensor.sizeBytes == 0 {
			tensor.sizeBytes = uint64(len(tensor.packedQ4) + len(tensor.scales)*4)
		}
	default:
		return rocmKVEncodedTensor{}, core.E("rocm.KVCache.Snapshot", core.Sprintf("unsupported tensor encoding %q", tensor.encoding), nil)
	}
	return tensor, nil
}

func encodeROCmKVTensor(encoding string, values []float32) (rocmKVEncodedTensor, error) {
	return encodeROCmKVTensorRows(encoding, values, len(values), 1)
}

func encodeROCmKVTensorRows(encoding string, values []float32, rowWidth, rowCount int) (rocmKVEncodedTensor, error) {
	if rowWidth <= 0 || rowCount <= 0 || len(values) != rowWidth*rowCount {
		return rocmKVEncodedTensor{}, core.E("rocm.KVCache.Encode", "row-scaled tensor shape mismatch", nil)
	}
	switch encoding {
	case rocmKVEncodingFP16:
		out := rocmKVEncodedTensor{encoding: encoding, length: len(values), f16: make([]uint16, len(values))}
		for i, value := range values {
			out.f16[i] = rocmFloat32ToFloat16(value)
		}
		out.sizeBytes = uint64(len(out.f16) * 2)
		return out, nil
	case rocmKVEncodingQ8:
		scale := rocmQuantScale(values, 127)
		out := rocmKVEncodedTensor{encoding: encoding, length: len(values), scale: scale, q8: make([]int8, len(values))}
		for i, value := range values {
			out.q8[i] = int8(clampInt(int(math.Round(float64(value/scale))), -127, 127))
		}
		out.sizeBytes = uint64(len(out.q8) + 4)
		return out, nil
	case rocmKVEncodingQ8Rows, rocmKVEncodingQ8RowsI:
		out := rocmKVEncodedTensor{encoding: encoding, length: len(values), scales: make([]float32, rowCount), q8: make([]int8, len(values))}
		for row := 0; row < rowCount; row++ {
			start := row * rowWidth
			end := start + rowWidth
			scale := rocmQuantScale(values[start:end], 127)
			out.scales[row] = scale
			for i, value := range values[start:end] {
				out.q8[start+i] = int8(clampInt(int(math.Round(float64(value/scale))), -127, 127))
			}
		}
		out.sizeBytes = uint64(len(out.q8) + len(out.scales)*4)
		return out, nil
	case rocmKVEncodingQ4:
		scale := rocmQuantScale(values, 7)
		out := rocmKVEncodedTensor{encoding: encoding, length: len(values), scale: scale, packedQ4: make([]byte, (len(values)+1)/2)}
		for i, value := range values {
			quantized := int8(clampInt(int(math.Round(float64(value/scale))), -8, 7))
			packed := packSignedQ4(quantized)
			if i%2 == 0 {
				out.packedQ4[i/2] = packed
			} else {
				out.packedQ4[i/2] |= packed << 4
			}
		}
		out.sizeBytes = uint64(len(out.packedQ4) + 4)
		return out, nil
	case rocmKVEncodingQ4Rows, rocmKVEncodingQ4RowsI:
		out := rocmKVEncodedTensor{encoding: encoding, length: len(values), scales: make([]float32, rowCount), packedQ4: make([]byte, (len(values)+1)/2)}
		for row := 0; row < rowCount; row++ {
			start := row * rowWidth
			end := start + rowWidth
			scale := rocmQuantScale(values[start:end], 7)
			out.scales[row] = scale
			for i, value := range values[start:end] {
				index := start + i
				quantized := int8(clampInt(int(math.Round(float64(value/scale))), -8, 7))
				packed := packSignedQ4(quantized)
				if index%2 == 0 {
					out.packedQ4[index/2] = packed
				} else {
					out.packedQ4[index/2] |= packed << 4
				}
			}
		}
		out.sizeBytes = uint64(len(out.packedQ4) + len(out.scales)*4)
		return out, nil
	default:
		return rocmKVEncodedTensor{}, core.E("rocm.KVCache.Encode", core.Sprintf("unsupported tensor encoding %q", encoding), nil)
	}
}

func (tensor rocmKVEncodedTensor) decode() []float32 {
	return tensor.decodeRows(tensor.length)
}

func (tensor rocmKVEncodedTensor) decodeRows(rowWidth int) []float32 {
	if rowWidth <= 0 {
		rowWidth = tensor.length
	}
	out := make([]float32, tensor.length)
	_ = tensor.decodeRowsRangeInto(out, rowWidth, 0, tensor.length/rowWidth)
	return out
}

func (tensor rocmKVEncodedTensor) decodeRowsRangeInto(out []float32, rowWidth, startRow, endRow int) error {
	switch tensor.encoding {
	case rocmKVEncodingFP16:
		return tensor.decodeRowsRangeFP16Into(out, rowWidth, startRow, endRow)
	case rocmKVEncodingQ8:
		return tensor.decodeRowsRangeQ8Into(out, rowWidth, startRow, endRow)
	case rocmKVEncodingQ8Rows, rocmKVEncodingQ8RowsI:
		return tensor.decodeRowsRangeQ8RowsInto(out, rowWidth, startRow, endRow)
	case rocmKVEncodingQ4:
		return tensor.decodeRowsRangeQ4Into(out, rowWidth, startRow, endRow)
	case rocmKVEncodingQ4Rows, rocmKVEncodingQ4RowsI:
		return tensor.decodeRowsRangeQ4RowsInto(out, rowWidth, startRow, endRow)
	default:
		return core.E("rocm.KVCache.Decode", core.Sprintf("unsupported tensor encoding %q", tensor.encoding), nil)
	}
}

func (tensor rocmKVEncodedTensor) decodeRowsRangeShape(rowWidth, startRow, endRow int, out []float32) (int, int, error) {
	if rowWidth <= 0 || tensor.length <= 0 || tensor.length%rowWidth != 0 {
		return 0, 0, core.E("rocm.KVCache.Decode", "row shape mismatch", nil)
	}
	rowCount := tensor.length / rowWidth
	if startRow < 0 || endRow < startRow || endRow > rowCount {
		return 0, 0, core.E("rocm.KVCache.Decode", "row range mismatch", nil)
	}
	count := (endRow - startRow) * rowWidth
	if len(out) < count {
		return 0, 0, core.E("rocm.KVCache.Decode", "decode output buffer is too small", nil)
	}
	return startRow * rowWidth, count, nil
}

func (tensor rocmKVEncodedTensor) decodeRowsRangeFP16Into(out []float32, rowWidth, startRow, endRow int) error {
	start, count, err := tensor.decodeRowsRangeShape(rowWidth, startRow, endRow, out)
	if err != nil {
		return err
	}
	if len(tensor.f16) < start+count {
		return core.E("rocm.KVCache.Decode", "fp16 tensor length mismatch", nil)
	}
	for i, value := range tensor.f16[start : start+count] {
		out[i] = hipFloat16ToFloat32(value)
	}
	return nil
}

func (tensor rocmKVEncodedTensor) decodeRowsRangeQ8Into(out []float32, rowWidth, startRow, endRow int) error {
	start, count, err := tensor.decodeRowsRangeShape(rowWidth, startRow, endRow, out)
	if err != nil {
		return err
	}
	if len(tensor.q8) < start+count {
		return core.E("rocm.KVCache.Decode", "q8 tensor length mismatch", nil)
	}
	for i, value := range tensor.q8[start : start+count] {
		out[i] = float32(value) * tensor.scale
	}
	return nil
}

func (tensor rocmKVEncodedTensor) decodeRowsRangeQ8RowsInto(out []float32, rowWidth, startRow, endRow int) error {
	start, count, err := tensor.decodeRowsRangeShape(rowWidth, startRow, endRow, out)
	if err != nil {
		return err
	}
	if len(tensor.q8) < start+count || len(tensor.scales) < endRow {
		return core.E("rocm.KVCache.Decode", "q8 row tensor length mismatch", nil)
	}
	for i, value := range tensor.q8[start : start+count] {
		row := startRow + i/rowWidth
		out[i] = float32(value) * tensor.scales[row]
	}
	return nil
}

func (tensor rocmKVEncodedTensor) decodeRowsRangeQ4Into(out []float32, rowWidth, startRow, endRow int) error {
	start, count, err := tensor.decodeRowsRangeShape(rowWidth, startRow, endRow, out)
	if err != nil {
		return err
	}
	if len(tensor.packedQ4) < (start+count+1)/2 {
		return core.E("rocm.KVCache.Decode", "q4 tensor length mismatch", nil)
	}
	for i := 0; i < count; i++ {
		index := start + i
		packed := tensor.packedQ4[index/2]
		if index%2 == 1 {
			packed >>= 4
		}
		out[i] = float32(unpackSignedQ4(packed&0x0f)) * tensor.scale
	}
	return nil
}

func (tensor rocmKVEncodedTensor) decodeRowsRangeQ4RowsInto(out []float32, rowWidth, startRow, endRow int) error {
	start, count, err := tensor.decodeRowsRangeShape(rowWidth, startRow, endRow, out)
	if err != nil {
		return err
	}
	if len(tensor.packedQ4) < (start+count+1)/2 || len(tensor.scales) < endRow {
		return core.E("rocm.KVCache.Decode", "q4 row tensor length mismatch", nil)
	}
	for i := 0; i < count; i++ {
		index := start + i
		packed := tensor.packedQ4[index/2]
		if index%2 == 1 {
			packed >>= 4
		}
		row := startRow + i/rowWidth
		out[i] = float32(unpackSignedQ4(packed&0x0f)) * tensor.scales[row]
	}
	return nil
}

func rocmQuantScale(values []float32, maxQuant int) float32 {
	maxAbs := float32(0)
	for _, value := range values {
		if abs := float32(math.Abs(float64(value))); abs > maxAbs {
			maxAbs = abs
		}
	}
	if maxAbs == 0 {
		return 1
	}
	return maxAbs / float32(maxQuant)
}

func packSignedQ4(value int8) byte {
	if value < 0 {
		return byte(value+16) & 0x0f
	}
	return byte(value) & 0x0f
}

func unpackSignedQ4(value byte) int8 {
	value &= 0x0f
	if value >= 8 {
		return int8(value) - 16
	}
	return int8(value)
}

func rocmFloat32ToFloat16(value float32) uint16 {
	bits := math.Float32bits(value)
	sign := uint16((bits >> 16) & 0x8000)
	exponent := int((bits>>23)&0xff) - 127 + 15
	mantissa := bits & 0x7fffff
	if exponent <= 0 {
		return sign
	}
	if exponent >= 0x1f {
		return sign | 0x7c00
	}
	return sign | uint16(exponent<<10) | uint16(mantissa>>13)
}

func clampInt(value, min, max int) int {
	if value < min {
		return min
	}
	if value > max {
		return max
	}
	return value
}
