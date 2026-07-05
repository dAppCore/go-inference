// SPDX-Licence-Identifier: EUPL-1.2

// Package blockcache exposes a block-prefix cache metadata layer that fronts
// the native prompt cache with stable, portable block identities.
//
//	service := blockcache.New(blockcache.Config{BlockSize: 512, ...})
//	stats, _ := service.CacheStats(ctx)
package blockcache

import (
	"context"
	"crypto/sha256"
	"hash"
	"maps"
	"sync"

	core "dappco.re/go"
	"dappco.re/go/inference"
	state "dappco.re/go/inference/model/state"
)

const (
	// DefaultBlockSize is the token chunk size used for portable block
	// prefix identities when callers do not choose a size.
	DefaultBlockSize = 512

	mode        = "block-prefix"
	diskVersion = 1
)

// Config configures the block-prefix cache metadata layer.
type Config struct {
	BlockSize     int
	ModelHash     string
	AdapterHash   string
	TokenizerHash string
	Tokenize      func(prompt string) ([]int32, error)
	WarmPrompt    func(ctx context.Context, prompt string) error
	ClearRuntime  func()
	DiskPath      string
	StateStore    state.Writer
	// Deprecated: use StateStore.
	MemvidStore state.Writer
}

// Service exposes stable block-prefix refs through
// inference.CacheService. It records block identities in memory, optionally
// persists them on disk, and delegates actual KV warming to the native prompt
// cache when a prompt warmer is configured.
type Service struct {
	mu             sync.Mutex
	cfg            Config
	blockSizeLabel string
	// prefixTokenLabels caches the pre-rendered decimal string for the
	// "prefix_tokens" label value at offsets blockSize, 2*blockSize,
	// ... up to len(prefixTokenLabels). blockRefs reads this slice
	// directly when end aligns to a multiple of blockSize, skipping a
	// per-block core.Itoa heap allocation (Itoa(>99) allocates each
	// call). Index 0 unused — entry i holds the string for end ==
	// (i+1)*blockSize. Populated up-front in New so the slice is
	// immutable after construction — concurrent blockRefs callers
	// read it lock-free.
	prefixTokenLabels []string
	blocks            map[string]inference.CacheBlockRef
	memoryBytes       uint64
	hits              uint64
	misses            uint64
	cleared           uint64
	evictions         uint64
	diskCorrupt       uint64
	diskLoaded        bool
}

// prefixTokenLabelCacheSize bounds how many aligned-end labels New
// pre-renders. 32 covers prompts up to ~16384 tokens at BlockSize=512,
// which is the typical prefill window. Beyond the cap, blockRefs
// falls back to core.Itoa. Sized small so per-Service construction
// stays sub-microsecond — pre-rendering 32 strings is amortised by
// the first WarmCache that uses more than a single aligned block.
const prefixTokenLabelCacheSize = 32

type diskRecord struct {
	Version  int                     `json:"version"`
	Ref      inference.CacheBlockRef `json:"ref"`
	Tokens   []int32                 `json:"tokens,omitempty"`
	StateRef *state.ChunkRef         `json:"state_ref,omitempty"`
	// Deprecated: retained for older disk records.
	MemvidRef *state.ChunkRef `json:"memvid_ref,omitempty"`
}

type statePayload struct {
	Version       int                     `json:"version"`
	BlockID       string                  `json:"block_id"`
	Ref           inference.CacheBlockRef `json:"ref"`
	Tokens        []int32                 `json:"tokens,omitempty"`
	Encoding      string                  `json:"encoding,omitempty"`
	CacheMode     string                  `json:"cache_mode,omitempty"`
	PayloadFormat string                  `json:"payload_format,omitempty"`
}

// New returns a cache metadata service with stable prefix refs.
//
//	service := blockcache.New(blockcache.Config{BlockSize: 512})
func New(cfg Config) *Service {
	if cfg.BlockSize <= 0 {
		cfg.BlockSize = DefaultBlockSize
	}
	cfg.DiskPath = core.Trim(cfg.DiskPath)
	// Pre-render the aligned-end "prefix_tokens" label strings up-front
	// so subsequent blockRefs calls can return them by reference
	// without a per-block core.Itoa heap allocation. Real Services live
	// the duration of a model registration and amortise the
	// construction cost across many WarmCache calls.
	prefixLabels := make([]string, prefixTokenLabelCacheSize+1)
	for i := 1; i <= prefixTokenLabelCacheSize; i++ {
		prefixLabels[i] = core.Itoa(i * cfg.BlockSize)
	}
	return &Service{
		cfg:               cfg,
		blockSizeLabel:    core.Itoa(cfg.BlockSize),
		prefixTokenLabels: prefixLabels,
		blocks:            map[string]inference.CacheBlockRef{},
	}
}

// DiskPath persistence is opt-in via the typed blockcache.Config.DiskPath field
// (set by a caller that wants disk-backed block metadata) — there is no env
// reader. The metaladapter prod path leaves it unset (in-memory block cache).

// CacheStats reports in-memory block metadata and cumulative warm hit/miss
// counters.
func (service *Service) CacheStats(ctx context.Context) (inference.CacheStats, error) {
	if err := cacheContextError(ctx); err != nil {
		return inference.CacheStats{}, err
	}
	if service == nil {
		return inference.CacheStats{}, core.NewError("mlx: block cache service is nil")
	}
	service.mu.Lock()
	defer service.mu.Unlock()
	if err := service.ensureDiskLoadedLocked(); err != nil {
		return inference.CacheStats{}, err
	}
	return service.statsLocked(), nil
}

// CacheEntries returns stable cache block refs, optionally filtered by labels.
func (service *Service) CacheEntries(ctx context.Context, labels map[string]string) ([]inference.CacheBlockRef, error) {
	if err := cacheContextError(ctx); err != nil {
		return nil, err
	}
	if service == nil {
		return nil, core.NewError("mlx: block cache service is nil")
	}
	service.mu.Lock()
	defer service.mu.Unlock()
	if err := service.ensureDiskLoadedLocked(); err != nil {
		return nil, err
	}
	entries := make([]inference.CacheBlockRef, 0, len(service.blocks))
	for _, ref := range service.blocks {
		if len(labels) > 0 && !blockRefMatchesLabels(ref, labels) {
			continue
		}
		entries = append(entries, cloneCacheBlockRef(ref))
	}
	sortCacheBlockRefs(entries)
	return entries, nil
}

// WarmCache creates stable block refs for the request and optionally warms the
// native prompt cache when a prompt and warmer are present.
func (service *Service) WarmCache(ctx context.Context, req inference.CacheWarmRequest) (inference.CacheWarmResult, error) {
	if err := cacheContextError(ctx); err != nil {
		return inference.CacheWarmResult{}, err
	}
	if service == nil {
		return inference.CacheWarmResult{}, core.NewError("mlx: block cache service is nil")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	tokens, err := service.requestTokens(req)
	if err != nil {
		return inference.CacheWarmResult{}, err
	}
	if len(tokens) == 0 {
		return inference.CacheWarmResult{}, core.NewError("mlx: cache warm requires prompt or tokens")
	}
	if service.cfg.WarmPrompt != nil && core.Trim(req.Prompt) != "" {
		if err := service.cfg.WarmPrompt(ctx, req.Prompt); err != nil {
			return inference.CacheWarmResult{}, err
		}
	}

	labels := service.compatibilityLabels(req)
	refs := service.blockRefs(req, tokens, labels)
	service.mu.Lock()
	defer service.mu.Unlock()
	if err := service.ensureDiskLoadedLocked(); err != nil {
		return inference.CacheWarmResult{}, err
	}
	for i, ref := range refs {
		if _, ok := service.blocks[ref.ID]; ok {
			service.hits++
			continue
		}
		service.misses++
		storedRef, err := service.writeDiskBlockLocked(ctx, ref, tokens[:ref.TokenStart+ref.TokenCount])
		if err != nil {
			return inference.CacheWarmResult{}, err
		}
		refs[i] = storedRef
		service.blocks[ref.ID] = storedRef
		service.memoryBytes += storedRef.SizeBytes
	}
	return inference.CacheWarmResult{
		Blocks: refs,
		Stats:  service.statsLocked(),
		Labels: labels,
	}, nil
}

// ClearCache clears all refs, or only refs whose metadata matches labels.
func (service *Service) ClearCache(ctx context.Context, labels map[string]string) (inference.CacheStats, error) {
	if err := cacheContextError(ctx); err != nil {
		return inference.CacheStats{}, err
	}
	if service == nil {
		return inference.CacheStats{}, core.NewError("mlx: block cache service is nil")
	}
	service.mu.Lock()
	defer service.mu.Unlock()
	if err := service.ensureDiskLoadedLocked(); err != nil {
		return inference.CacheStats{}, err
	}
	if len(labels) == 0 {
		service.blocks = map[string]inference.CacheBlockRef{}
		service.memoryBytes = 0
		service.hits = 0
		service.misses = 0
		service.cleared++
		if err := service.clearDiskLocked(); err != nil {
			return inference.CacheStats{}, err
		}
		if service.cfg.ClearRuntime != nil {
			service.cfg.ClearRuntime()
		}
		return service.statsLocked(), nil
	}
	for id, ref := range service.blocks {
		if blockRefMatchesLabels(ref, labels) {
			if err := service.removeDiskBlockLocked(ref.ID); err != nil {
				return inference.CacheStats{}, err
			}
			delete(service.blocks, id)
			service.memoryBytes -= ref.SizeBytes
			service.cleared++
		}
	}
	return service.statsLocked(), nil
}

func (service *Service) requestTokens(req inference.CacheWarmRequest) ([]int32, error) {
	if len(req.Tokens) > 0 {
		return req.Tokens, nil
	}
	if core.Trim(req.Prompt) == "" {
		return nil, nil
	}
	if service.cfg.Tokenize == nil {
		return nil, core.NewError("mlx: cache warm prompt requires tokenizer")
	}
	tokens, err := service.cfg.Tokenize(req.Prompt)
	if err != nil {
		return nil, err
	}
	return core.SliceClone(tokens), nil
}

func (service *Service) blockRefs(req inference.CacheWarmRequest, tokens []int32, labels map[string]string) []inference.CacheBlockRef {
	blockSize := service.cfg.BlockSize
	if blockSize <= 0 {
		blockSize = DefaultBlockSize
	}
	modelHash := core.FirstNonBlank(service.cfg.ModelHash, req.Model.Hash, req.Model.ID)
	adapterHash := core.FirstNonBlank(service.cfg.AdapterHash, req.Adapter.Hash)
	tokenizerHash := core.FirstNonBlank(service.cfg.TokenizerHash, req.Labels["tokenizer_hash"])
	refs := make([]inference.CacheBlockRef, 0, (len(tokens)+blockSize-1)/blockSize)
	// Stream the SHA256 once across the cumulative prefix and emit a
	// block ID at every boundary. sha256.Sum does not alter the hash
	// state, so each Sum captures the digest of the prefix up to the
	// current write position — identical to the previous per-block
	// blockCacheID call but without re-hashing earlier tokens.
	//
	// The hash.Hash and its reusable encode buffer are borrowed from a
	// package-level pool. blockRefs runs lock-free (WarmCache calls it
	// before taking service.mu), so concurrent warms run concurrent
	// blockRefs — the scratch must be per-call, never a Service field.
	// Pooling reclaims the two heap allocations (sha256.New escaping
	// through the hash.Hash interface, plus the encode buffer) that
	// would otherwise be discarded on every call.
	scratch := acquireBlockCacheHasher(modelHash, adapterHash, tokenizerHash, req.Mode)
	defer releaseBlockCacheHasher(scratch)
	hash := scratch.h
	writeBlockCacheHeaderInto(hash, scratch.buf, modelHash, adapterHash, tokenizerHash, req.Mode)
	for start := 0; start < len(tokens); start += blockSize {
		end := min(start+blockSize, len(tokens))
		writeBlockCacheTokensInto(hash, scratch.buf, tokens[start:end])
		digest := hash.Sum(scratch.sum[:0])
		refLabels := cloneBlockCacheLabelsExtra(labels, 2)
		refLabels["block_index"] = core.Itoa(len(refs))
		refLabels["prefix_tokens"] = service.prefixTokenLabel(end, blockSize)
		ref := inference.CacheBlockRef{
			ID:            core.HexEncode(digest),
			Kind:          "prefix",
			ModelHash:     modelHash,
			AdapterHash:   adapterHash,
			TokenizerHash: tokenizerHash,
			TokenStart:    start,
			TokenCount:    end - start,
			SizeBytes:     uint64(end-start) * 4,
			Encoding:      "token-prefix/int32",
			Labels:        refLabels,
		}
		ref = service.withDiskLabels(ref)
		refs = append(refs, ref)
	}
	return refs
}

// prefixTokenLabel returns the decimal string form of end. When end
// aligns to a multiple of blockSize within the pre-rendered cache it
// returns the cached string with no allocation; otherwise it falls
// back to core.Itoa (the partial-final-block case, plus any end
// beyond the cache cap).
func (service *Service) prefixTokenLabel(end, blockSize int) string {
	if blockSize <= 0 || end <= 0 || end%blockSize != 0 {
		return core.Itoa(end)
	}
	index := end / blockSize
	if index < len(service.prefixTokenLabels) {
		return service.prefixTokenLabels[index]
	}
	return core.Itoa(end)
}

// blockCacheTokenBatch is the token count encoded per hash.Write — 64
// int32s is 256 bytes, enough to amortise the hash.Hash interface
// dispatch without an oversized per-call buffer.
const blockCacheTokenBatch = 64

// blockCacheHasher bundles the sha256 stream, its reusable encode
// buffer, and the digest scratch so all three can be recycled across
// calls as one pooled unit (replacing the former per-call
// makeBlockCacheEncodeBuffer allocation; sizing now lives in
// acquireBlockCacheHasher). sum holds the hash.Sum output: as a struct
// field it lives on the pooled heap object instead of escaping a fresh
// stack array through the hash.Hash interface on every call.
type blockCacheHasher struct {
	h   hash.Hash
	buf []byte
	sum [sha256.Size]byte
}

// blockCacheHasherPool recycles blockCacheHasher units across blockRefs
// and blockCacheID calls. It is package-level (not a Service field) on
// purpose: blockRefs runs before WarmCache takes service.mu, so two
// concurrent warms run blockRefs concurrently — a per-Service hasher
// would race. The pool hands each goroutine its own unit.
var blockCacheHasherPool = sync.Pool{
	New: func() any {
		return &blockCacheHasher{
			h:   sha256.New(),
			buf: make([]byte, 0, blockCacheTokenBatch*4),
		}
	},
}

// acquireBlockCacheHasher borrows a reset hasher whose encode buffer is
// sized for the larger of the header (four length-prefixed identity
// strings) and one full token batch (256 bytes), matching the original
// makeBlockCacheEncodeBuffer sizing. A pooled buffer that is too small
// for a long prod-scale header (sha256:+hex hashes) is grown once and
// kept, so the pool stays effective for those callers instead of
// silently realloc-and-discarding per call.
func acquireBlockCacheHasher(model, adapter, tokenizer, mode string) *blockCacheHasher {
	scratch := blockCacheHasherPool.Get().(*blockCacheHasher)
	scratch.h.Reset()
	headerLen := 16 + len(model) + len(adapter) + len(tokenizer) + len(mode)
	capacity := blockCacheTokenBatch * 4
	if headerLen > capacity {
		capacity = headerLen
	}
	if cap(scratch.buf) < capacity {
		scratch.buf = make([]byte, 0, capacity)
	}
	return scratch
}

// releaseBlockCacheHasher returns a hasher to the pool. The buffer is
// retained at whatever capacity it grew to; the hash is reset on the
// next acquire, never here, so a released unit carries no live state.
func releaseBlockCacheHasher(scratch *blockCacheHasher) {
	blockCacheHasherPool.Put(scratch)
}

// writeBlockCacheHeaderInto composes the four length-prefixed identity
// strings into buf and writes them to the hash once. buf is reused for
// subsequent token writes by the caller — it is reset to length zero
// before this returns so callers can reslice from the same backing
// array without a fresh allocation.
func writeBlockCacheHeaderInto(h hash.Hash, buf []byte, model, adapter, tokenizer, mode string) {
	buf = buf[:0]
	buf = appendBlockCacheLenPrefixed(buf, model)
	buf = appendBlockCacheLenPrefixed(buf, adapter)
	buf = appendBlockCacheLenPrefixed(buf, tokenizer)
	buf = appendBlockCacheLenPrefixed(buf, mode)
	h.Write(buf)
}

// appendBlockCacheLenPrefixed appends a uint32 LE length prefix
// followed by value to buf and returns the new buf.
func appendBlockCacheLenPrefixed(buf []byte, value string) []byte {
	n := uint32(len(value))
	buf = append(buf, byte(n), byte(n>>8), byte(n>>16), byte(n>>24))
	return append(buf, value...)
}

// writeBlockCacheTokensInto encodes tokens as little-endian int32 bytes
// into the reusable buf and writes them to the hash, batching up to
// blockCacheTokenBatch tokens per Write to amortise hash.Hash interface
// dispatch. buf must have cap >= blockCacheTokenBatch*4 (guaranteed by
// makeBlockCacheEncodeBuffer); it is resliced from length zero so no
// allocation occurs.
func writeBlockCacheTokensInto(h hash.Hash, buf []byte, tokens []int32) {
	for start := 0; start < len(tokens); start += blockCacheTokenBatch {
		end := min(start+blockCacheTokenBatch, len(tokens))
		scratch := buf[:0]
		for _, token := range tokens[start:end] {
			value := uint32(token)
			scratch = append(scratch, byte(value), byte(value>>8), byte(value>>16), byte(value>>24))
		}
		h.Write(scratch)
	}
}

func (service *Service) compatibilityLabels(req inference.CacheWarmRequest) map[string]string {
	labels := cloneBlockCacheLabelsExtra(req.Labels, 4)
	labels["cache_mode"] = mode
	labels["block_size"] = service.blockSizeLabel
	labels["model_match"] = boolLabel(cacheIdentityMatches(service.cfg.ModelHash, core.FirstNonBlank(req.Model.Hash, req.Model.ID)))
	labels["adapter_match"] = boolLabel(cacheIdentityMatches(service.cfg.AdapterHash, req.Adapter.Hash))
	labels["tokenizer_match"] = boolLabel(cacheIdentityMatches(service.cfg.TokenizerHash, req.Labels["tokenizer_hash"]))
	return labels
}

func (service *Service) statsLocked() inference.CacheStats {
	stats := inference.CacheStats{
		Blocks:    len(service.blocks),
		Hits:      service.hits,
		Misses:    service.misses,
		Evictions: service.evictions,
		CacheMode: mode,
		Labels: map[string]string{
			"block_size": service.blockSizeLabel,
			"cleared":    core.FormatUint(service.cleared, 10),
		},
	}
	if service.diskEnabled() {
		stats.DiskBytes = service.diskBytesLocked()
		stats.Labels["disk_path"] = service.cfg.DiskPath
		stats.Labels["disk_blocks"] = core.Itoa(len(core.PathGlob(core.PathJoin(service.cfg.DiskPath, "*.json"))))
		stats.Labels["disk_corrupt"] = core.FormatUint(service.diskCorrupt, 10)
	}
	if service.stateStoreEnabled() {
		stats.Labels["cold_store"] = "state"
	}
	stats.MemoryBytes = service.memoryBytes
	total := service.hits + service.misses
	if total > 0 {
		stats.HitRate = float64(service.hits) / float64(total)
	}
	return stats
}

func (service *Service) diskEnabled() bool {
	return service != nil && service.cfg.DiskPath != ""
}

func (service *Service) stateStoreEnabled() bool {
	return service != nil && service.stateStore() != nil
}

func (service *Service) stateStore() state.Writer {
	if service == nil {
		return nil
	}
	if service.cfg.StateStore != nil {
		return service.cfg.StateStore
	}
	return service.cfg.MemvidStore
}

func (service *Service) withDiskLabels(ref inference.CacheBlockRef) inference.CacheBlockRef {
	if !service.diskEnabled() || ref.ID == "" {
		return ref
	}
	labels := cloneBlockCacheLabelsExtra(ref.Labels, 2)
	labels["disk"] = "true"
	labels["disk_path"] = service.diskBlockPath(ref.ID)
	ref.Labels = labels
	return ref
}

func (service *Service) ensureDiskLoadedLocked() error {
	if !service.diskEnabled() || service.diskLoaded {
		return nil
	}
	if result := core.MkdirAll(service.cfg.DiskPath, 0o700); !result.OK {
		return core.E("Service.ensureDiskLoaded", "create disk cache directory", resultError(result))
	}
	for _, path := range core.PathGlob(core.PathJoin(service.cfg.DiskPath, "*.json")) {
		record, ok := service.readDiskRecord(path)
		if !ok {
			service.quarantineDiskBlock(path)
			continue
		}
		if !service.diskRecordCompatible(record) {
			continue
		}
		ref := service.withDiskLabels(record.Ref)
		chunkRef := record.StateRef
		if chunkRef == nil {
			chunkRef = record.MemvidRef
		}
		if chunkRef != nil {
			ref = withStateLabels(ref, *chunkRef)
		}
		service.blocks[record.Ref.ID] = ref
		service.memoryBytes += ref.SizeBytes
	}
	service.diskLoaded = true
	return nil
}

func (service *Service) readDiskRecord(path string) (diskRecord, bool) {
	read := core.ReadFile(path)
	if !read.OK {
		return diskRecord{}, false
	}
	data, ok := read.Value.([]byte)
	if !ok {
		return diskRecord{}, false
	}
	var record diskRecord
	result := core.JSONUnmarshal(data, &record)
	if !result.OK || record.Version != diskVersion || record.Ref.ID == "" {
		return diskRecord{}, false
	}
	return record, true
}

func (service *Service) diskRecordCompatible(record diskRecord) bool {
	if record.Ref.ID == "" {
		return false
	}
	if !cacheIdentityMatches(service.cfg.ModelHash, record.Ref.ModelHash) {
		return false
	}
	if !cacheIdentityMatches(service.cfg.AdapterHash, record.Ref.AdapterHash) {
		return false
	}
	return cacheIdentityMatches(service.cfg.TokenizerHash, record.Ref.TokenizerHash)
}

func (service *Service) writeDiskBlockLocked(ctx context.Context, ref inference.CacheBlockRef, tokens []int32) (inference.CacheBlockRef, error) {
	if !service.diskEnabled() {
		return ref, nil
	}
	if result := core.MkdirAll(service.cfg.DiskPath, 0o700); !result.OK {
		return inference.CacheBlockRef{}, core.E("Service.writeDiskBlock", "create disk cache directory", resultError(result))
	}
	var stateRef *state.ChunkRef
	if service.stateStoreEnabled() {
		written, err := service.writeStateBlock(ctx, ref, tokens)
		if err != nil {
			return inference.CacheBlockRef{}, err
		}
		stateRef = &written
		ref = withStateLabels(ref, written)
	}
	record := diskRecord{
		Version:  diskVersion,
		Ref:      service.withDiskLabels(ref),
		StateRef: stateRef,
	}
	if stateRef == nil {
		record.Tokens = core.SliceClone(tokens)
	}
	data := core.JSONMarshal(record)
	if !data.OK {
		return inference.CacheBlockRef{}, core.E("Service.writeDiskBlock", "marshal disk cache record", resultError(data))
	}
	write := core.WriteFile(service.diskBlockPath(ref.ID), data.Value.([]byte), 0o600)
	if !write.OK {
		return inference.CacheBlockRef{}, core.E("Service.writeDiskBlock", "write disk cache record", resultError(write))
	}
	return record.Ref, nil
}

func (service *Service) writeStateBlock(ctx context.Context, ref inference.CacheBlockRef, tokens []int32) (state.ChunkRef, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	store := service.stateStore()
	if store == nil {
		return state.ChunkRef{}, core.NewError("mlx: state store is nil")
	}
	payload := statePayload{
		Version:       diskVersion,
		BlockID:       ref.ID,
		Ref:           ref,
		Tokens:        core.SliceClone(tokens),
		Encoding:      ref.Encoding,
		CacheMode:     mode,
		PayloadFormat: "token-prefix/int32-json",
	}
	chunk, err := store.Put(ctx, core.JSONMarshalString(payload), state.PutOptions{
		URI:   "mlx://cache/block/" + ref.ID,
		Title: "go-mlx block cache " + ref.ID,
		Kind:  "kv-block-prefix",
		Track: mode,
		Tags: map[string]string{
			"block_id":       ref.ID,
			"model_hash":     ref.ModelHash,
			"adapter_hash":   ref.AdapterHash,
			"tokenizer_hash": ref.TokenizerHash,
			"encoding":       ref.Encoding,
		},
		Labels: []string{"go-mlx", "block-cache", mode},
	})
	if err != nil {
		return state.ChunkRef{}, core.E("Service.writeStateBlock", "write State payload", err)
	}
	return chunk, nil
}

func withStateLabels(ref inference.CacheBlockRef, chunk state.ChunkRef) inference.CacheBlockRef {
	labels := cloneBlockCacheLabelsExtra(ref.Labels, 4)
	labels["cold_store"] = "state"
	labels["state_chunk_id"] = core.Itoa(chunk.ChunkID)
	if chunk.Codec != "" {
		labels["state_codec"] = chunk.Codec
	}
	if chunk.Segment != "" {
		labels["state_segment"] = chunk.Segment
	}
	if chunk.HasFrameOffset {
		labels["state_frame_offset"] = core.FormatUint(chunk.FrameOffset, 10)
	}
	ref.Labels = labels
	return ref
}

func (service *Service) clearDiskLocked() error {
	if !service.diskEnabled() {
		return nil
	}
	if result := core.RemoveAll(service.cfg.DiskPath); !result.OK {
		return core.E("Service.clearDisk", "remove disk cache directory", resultError(result))
	}
	if result := core.MkdirAll(service.cfg.DiskPath, 0o700); !result.OK {
		return core.E("Service.clearDisk", "recreate disk cache directory", resultError(result))
	}
	return nil
}

func (service *Service) removeDiskBlockLocked(id string) error {
	if !service.diskEnabled() || id == "" {
		return nil
	}
	result := core.Remove(service.diskBlockPath(id))
	if result.OK {
		return nil
	}
	err := resultError(result)
	if err != nil && core.IsNotExist(err) {
		return nil
	}
	return core.E("Service.removeDiskBlock", "remove disk cache record", err)
}

func (service *Service) quarantineDiskBlock(path string) {
	service.evictions++
	service.diskCorrupt++
	// Best-effort removal of an already-condemned corrupt record; the block
	// is counted evicted regardless. The Result is consulted rather than
	// blind-discarded: a not-exist failure means the file already vanished
	// (nothing to do), and any other failure is non-fatal here because the
	// next disk-load pass re-quarantines the still-present record.
	if result := core.Remove(path); !result.OK {
		if err := resultError(result); err != nil && core.IsNotExist(err) {
			return
		}
	}
}

func (service *Service) diskBytesLocked() uint64 {
	if !service.diskEnabled() {
		return 0
	}
	var total uint64
	for _, path := range core.PathGlob(core.PathJoin(service.cfg.DiskPath, "*.json")) {
		stat := core.Stat(path)
		if stat.OK {
			if info, ok := stat.Value.(core.FsFileInfo); ok && info.Size() > 0 {
				total += uint64(info.Size())
				continue
			}
		}
		read := core.ReadFile(path)
		if read.OK {
			if data, ok := read.Value.([]byte); ok {
				total += uint64(len(data))
			}
		}
	}
	return total
}

func (service *Service) diskBlockPath(id string) string {
	return core.PathJoin(service.cfg.DiskPath, id+".json")
}

func blockCacheID(modelHash, adapterHash, tokenizerHash, mode string, prefix []int32) string {
	scratch := acquireBlockCacheHasher(modelHash, adapterHash, tokenizerHash, mode)
	defer releaseBlockCacheHasher(scratch)
	hash := scratch.h
	writeBlockCacheHeaderInto(hash, scratch.buf, modelHash, adapterHash, tokenizerHash, mode)
	writeBlockCacheTokensInto(hash, scratch.buf, prefix)
	return core.HexEncode(hash.Sum(scratch.sum[:0]))
}

// HashModelParts returns a stable SHA-256 hex hash of the supplied identity
// parts. Used by callers (Metal cache adapter) to derive stable model and
// tokenizer hashes for block-prefix cache identity.
//
//	hash := blockcache.HashModelParts(info.Architecture, info.VocabSize)
func HashModelParts(parts ...any) string {
	return core.SHA256HexString(core.JSONMarshalString(parts))
}

func blockRefMatchesLabels(ref inference.CacheBlockRef, labels map[string]string) bool {
	for key, want := range labels {
		switch key {
		case "model_hash":
			if ref.ModelHash != want {
				return false
			}
		case "adapter_hash":
			if ref.AdapterHash != want {
				return false
			}
		case "tokenizer_hash":
			if ref.TokenizerHash != want {
				return false
			}
		default:
			if ref.Labels[key] != want {
				return false
			}
		}
	}
	return true
}

func cacheIdentityMatches(actual, requested string) bool {
	if actual == "" || requested == "" {
		return true
	}
	return actual == requested
}

func boolLabel(value bool) string {
	if value {
		return "true"
	}
	return "false"
}

func cacheContextError(ctx context.Context) error {
	if ctx == nil {
		return nil
	}
	return ctx.Err()
}

func cloneBlockCacheLabels(input map[string]string) map[string]string {
	return core.MapClone(input)
}

func cloneBlockCacheLabelsExtra(input map[string]string, extra int) map[string]string {
	if extra < 0 {
		extra = 0
	}
	out := make(map[string]string, len(input)+extra)
	maps.Copy(out, input)
	return out
}

func cloneCacheBlockRef(ref inference.CacheBlockRef) inference.CacheBlockRef {
	ref.Labels = cloneBlockCacheLabels(ref.Labels)
	return ref
}

// sortCacheBlockRefsInsertionThreshold is the size below which the
// insertion sort beats the comparator-closure overhead of pdqsort.
const sortCacheBlockRefsInsertionThreshold = 32

func sortCacheBlockRefs(entries []inference.CacheBlockRef) {
	// Insertion sort wins for small N because the closure dispatch in
	// core.SliceSortFunc costs more than the extra compares. For larger
	// N, pdqsort's O(N log N) trounces insertion sort's O(N²) — the
	// 256-entry case drops from ~152us to ~6us.
	if len(entries) <= sortCacheBlockRefsInsertionThreshold {
		for i := 1; i < len(entries); i++ {
			current := entries[i]
			j := i - 1
			for j >= 0 && cacheBlockRefLess(current, entries[j]) {
				entries[j+1] = entries[j]
				j--
			}
			entries[j+1] = current
		}
		return
	}
	core.SliceSortFunc(entries, cacheBlockRefLess)
}

func cacheBlockRefLess(a, b inference.CacheBlockRef) bool {
	if a.TokenStart != b.TokenStart {
		return a.TokenStart < b.TokenStart
	}
	return a.ID < b.ID
}

func resultError(result core.Result) error {
	if err, ok := result.Value.(error); ok {
		return err
	}
	if result.OK {
		return nil
	}
	if message := result.Error(); message != "" {
		return core.NewError(message)
	}
	return core.NewError("unknown block cache result error")
}
