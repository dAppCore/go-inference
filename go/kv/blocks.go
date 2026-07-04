// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
)

const (
	// KVSnapshotStateBlockKind identifies one State chunk containing a KV block.
	KVSnapshotStateBlockKind = "go-mlx/kv-snapshot-block"
	// StateBlockBundleKind identifies a collection of State KV blocks.
	StateBlockBundleKind = "go-mlx/kv-snapshot-block-bundle"
	// StateBlockVersion is the block envelope schema version.
	StateBlockVersion = 1

	// KVSnapshotMemvidBlockKind identifies one old memvid-named chunk
	// containing a KV block.
	//
	// Deprecated: use KVSnapshotStateBlockKind.
	KVSnapshotMemvidBlockKind = KVSnapshotStateBlockKind
	// MemvidBlockBundleKind identifies a collection of old memvid-named KV
	// blocks.
	//
	// Deprecated: use StateBlockBundleKind.
	MemvidBlockBundleKind = StateBlockBundleKind
	// MemvidBlockVersion is the block envelope schema version.
	//
	// Deprecated: use StateBlockVersion.
	MemvidBlockVersion = StateBlockVersion

	kvSnapshotStatePayloadRaw        = "raw"
	kvSnapshotStatePayloadJSONBase64 = "json-base64"
)

// kvSnapshotStateBlockDefaultLabels is the per-block label pair used
// when the caller passes empty StateBlockOptions.Labels — shared
// across blocks so the per-block PutOptions skips a slice allocation.
// State stores treat PutOptions.Labels as read-only input.
var kvSnapshotStateBlockDefaultLabels = []string{"go-mlx", "kv-snapshot-block"}

// Constant validation errors hoisted to package vars — each previously
// allocated a fresh core.NewError on the (rare but hot under churn)
// failure path. Sharing instances also makes errors.Is comparable for
// callers distinguishing "store nil" from "block range invalid" without
// parsing message text.
var (
	errBlockRangeInvalid           = core.NewError("mlx: invalid KV snapshot block range")
	errLayerRawTensorRangeInvalid  = core.NewError("mlx: invalid KV snapshot layer raw tensor range")
	errRawTensorBlockRangeInvalid  = core.NewError("mlx: invalid KV snapshot raw tensor block range")
	errTensorBlockRangeInvalid     = core.NewError("mlx: invalid KV snapshot tensor block range")
	errBundleKindInvalid           = core.NewError("mlx: invalid State KV block bundle kind")
	errBlockKindInvalid            = core.NewError("mlx: invalid State KV block kind")
	errBlockArchMismatch           = core.NewError("mlx: KV snapshot block architecture mismatch")
	errBlockHeadCountMismatch      = core.NewError("mlx: KV snapshot block head count mismatch")
	errBlockNil                    = core.NewError("mlx: KV snapshot block is nil")
	errBlockLayerCountMismatch     = core.NewError("mlx: KV snapshot block layer count mismatch")
	errBlockMetadataMismatch       = core.NewError("mlx: KV snapshot block metadata mismatch")
	errBlockCompressedPayloadSplit = core.NewError("mlx: KV snapshot compressed payload block requires full range")
	errBlockShapeMismatch          = core.NewError("mlx: KV snapshot block shape mismatch")
	errBlockSizeTooSmall           = core.NewError("mlx: KV snapshot block size must be > 0")
	errBlockSplitNeedsHeadDim      = core.NewError("mlx: KV snapshot block split requires head dimension")
	errBlockSplitNeedsTokens       = core.NewError("mlx: KV snapshot block split requires tokens matching sequence length")
	errBlockTokenCountMismatch     = core.NewError("mlx: KV snapshot block token count mismatch")
	errBlockYieldNil               = core.NewError("mlx: KV snapshot block yield is nil")
	errBlocksEmpty                 = core.NewError("mlx: KV snapshot blocks are empty")
	errBlocksNotContiguous         = core.NewError("mlx: KV snapshot blocks are not contiguous")
	errBlocksOutOfOrder            = core.NewError("mlx: KV snapshot blocks are not ordered by index")
	errSnapshotNil                 = core.NewError("mlx: KV snapshot is nil")
	errLayerMixesWindowLens        = core.NewError("mlx: KV snapshot layer mixes cache window lengths")
	errLayerRawShapeMismatch       = core.NewError("mlx: KV snapshot layer raw shape does not match sequence dimensions")
	errLayerRawByteLenMismatch     = core.NewError("mlx: KV snapshot layer raw tensor byte length mismatch")
	errLayerRawDtypeMismatch       = core.NewError("mlx: KV snapshot layer raw tensor dtype mismatch")
	errLayerRawTensorShape         = core.NewError("mlx: KV snapshot layer raw tensor shape mismatch")
	errRawTensorByteLenInvalid     = core.NewError("mlx: KV snapshot raw tensor byte length is invalid")
	errRawTensorDtypeMismatch      = core.NewError("mlx: KV snapshot raw tensor dtype mismatch")
	errRawTensorShapeSeq           = core.NewError("mlx: KV snapshot raw tensor shape does not match sequence length")
	errTensorShapeSeqHead          = core.NewError("mlx: KV snapshot tensor shape does not match sequence/head dimensions")
	errBundleNoBlocks              = core.NewError("mlx: State KV block bundle has no blocks")
	errBundleNil                   = core.NewError("mlx: State KV block bundle is nil")
	errBundleTokenCountEmpty       = core.NewError("mlx: State KV block bundle token count is empty")
	errBundleURIRequired           = core.NewError("mlx: State KV block bundle URI is required")
	errBlockNonByteData            = core.NewError("mlx: State KV block decoded to non-byte data")
	errBlockHashMismatch           = core.NewError("mlx: State KV block hash mismatch")
	errBlockPayloadLenMismatch     = core.NewError("mlx: State KV block payload length mismatch")
	errBlockRefHashMismatch        = core.NewError("mlx: State KV block ref hash mismatch")
	errBlockStreamNil              = core.NewError("mlx: State KV block stream is nil")
	errBlockTokenOffsetMismatch    = core.NewError("mlx: State KV block token offset mismatch")
	errPrefixBlocksNoCover         = core.NewError("mlx: State KV prefix blocks do not cover requested tokens")
	errPrefixExceedsBundle         = core.NewError("mlx: State KV prefix exceeds bundle token count")
	errPrefixNoCoveringBlocks      = core.NewError("mlx: State KV prefix has no covering blocks")
	errRawBlockHashMismatch        = core.NewError("mlx: State raw KV block hash mismatch")
	errRawBlockPayloadLenMismatch  = core.NewError("mlx: State raw KV block payload length mismatch")
	errStateStoreNil               = core.NewError("mlx: state store is nil")
	errTokenBlockMetadata          = core.NewError("mlx: State token block metadata mismatch")
	errTokenBlockTokenCount        = core.NewError("mlx: State token block token count mismatch")
	errTokenBlocksNotContiguous    = core.NewError("mlx: State token blocks are not contiguous")
	errTokenPrefixNoCover          = core.NewError("mlx: State token prefix blocks do not cover requested tokens")
	errTokenPrefixExceeds          = core.NewError("mlx: State token prefix exceeds bundle token count")
	errTokenPrefixNoBlocks         = core.NewError("mlx: State token prefix has no covering blocks")
	errStreamedBlockNil            = core.NewError("mlx: streamed KV snapshot block is nil")
	errUnsupportedLayerRawTensor   = core.NewError("mlx: unsupported KV snapshot layer raw tensor")
	errUnsupportedRawTensorDtype   = core.NewError("mlx: unsupported KV snapshot raw tensor dtype")
	errUnsupportedBlockEncoding    = core.NewError("mlx: unsupported State KV block binary encoding")
	errUnsupportedBundleVersion    = core.NewError("mlx: unsupported State KV block bundle version")
	errUnsupportedBlockVersion     = core.NewError("mlx: unsupported State KV block version")
)

// Block is one contiguous token range from a KV snapshot.
type Block struct {
	Index      int
	TokenStart int
	TokenCount int
	Hash       string
	Snapshot   *Snapshot
}

// StateTokenBlock is the token-only view of one durable State KV block.
type StateTokenBlock struct {
	Index      int
	TokenStart int
	TokenCount int
	Hash       string
	Tokens     []int32
}

// StateBlockOptions controls durable State-backed KV block storage.
type StateBlockOptions struct {
	BlockSize         int
	KVEncoding        Encoding
	URI               string
	Title             string
	Kind              string
	Track             string
	Tags              map[string]string
	Labels            []string
	ReusePrefix       *StateBlockBundle
	ReusePrefixTokens int
	// ReusePrefixTrusted declares the parent prefix identical BY
	// CONSTRUCTION (an append-only session sleeping over its own prior
	// sleep — the conversation-continuity lane): whole parent blocks below
	// the trusted boundary are grafted by reference without re-capturing or
	// re-hashing them, so the per-turn sleep cost tracks the TURN, not the
	// whole conversation. Arbitrary parent reuse keeps the hash check.
	ReusePrefixTrusted bool
}

// MemvidBlockOptions controls old memvid-named KV block storage.
//
// Deprecated: use StateBlockOptions. The persisted format is now described as
// State; older memvid names remain as compatibility wrappers.
type MemvidBlockOptions = StateBlockOptions

// StateBlockBundle is a portable manifest for durable State KV blocks.
type StateBlockBundle struct {
	Version      int             `json:"version"`
	Kind         string          `json:"kind"`
	SnapshotHash string          `json:"snapshot_hash,omitempty"`
	KVEncoding   Encoding        `json:"kv_encoding,omitempty"`
	Architecture string          `json:"architecture,omitempty"`
	TokenCount   int             `json:"token_count,omitempty"`
	TokenOffset  int             `json:"token_offset,omitempty"`
	BlockSize    int             `json:"block_size,omitempty"`
	NumLayers    int             `json:"num_layers,omitempty"`
	NumHeads     int             `json:"num_heads,omitempty"`
	SeqLen       int             `json:"seq_len,omitempty"`
	HeadDim      int             `json:"head_dim,omitempty"`
	ReusedBlocks int             `json:"reused_blocks,omitempty"`
	Blocks       []StateBlockRef `json:"blocks,omitempty"`
}

// MemvidBlockBundle is a portable manifest for old memvid-named KV blocks.
//
// Deprecated: use StateBlockBundle. The persisted format is now described as
// State; older memvid names remain as compatibility wrappers.
type MemvidBlockBundle = StateBlockBundle

// StateBlockRef links one logical KV block to a durable State chunk.
type StateBlockRef struct {
	Index            int            `json:"index"`
	TokenStart       int            `json:"token_start"`
	TokenCount       int            `json:"token_count"`
	KVHash           string         `json:"kv_hash,omitempty"`
	PayloadEncoding  string         `json:"payload_encoding,omitempty"`
	PayloadByteCount int            `json:"payload_byte_count,omitempty"`
	State            state.ChunkRef `json:"state"`
	// Deprecated: retained only so older bundles using json:"memvid" can wake.
	Memvid state.ChunkRef `json:"memvid"`
}

// MemvidBlockRef links one logical KV block to an old memvid-named chunk.
//
// Deprecated: use StateBlockRef. The persisted format is now described as
// State; older memvid names remain as compatibility wrappers.
type MemvidBlockRef = StateBlockRef

type kvSnapshotStateBlockEnvelope struct {
	Version          int    `json:"version"`
	Kind             string `json:"kind"`
	BlockIndex       int    `json:"block_index"`
	TokenStart       int    `json:"token_start"`
	TokenCount       int    `json:"token_count"`
	KVHash           string `json:"kv_hash"`
	KVEncoding       string `json:"kv_encoding,omitempty"`
	BinaryEncoding   string `json:"binary_encoding"`
	PayloadByteCount int    `json:"payload_byte_count,omitempty"`
	Data             string `json:"data"`
}

// SplitBlocks splits a KV snapshot into contiguous token-range blocks.
func (s *Snapshot) SplitBlocks(blockSize int) ([]Block, error) {
	// walkBlocks emits one block per blockSize-aligned range; mirror the
	// SaveStateBlocks estimate so growth-loop reallocs vanish for typical
	// snapshots. A layer-window adjustment may add one extra boundary —
	// the +1 absorbs it without overshoot.
	expectedBlocks := 1
	if blockSize > 0 && s != nil && len(s.Tokens) > 0 {
		expectedBlocks = (len(s.Tokens)+blockSize-1)/blockSize + 1
	}
	blocks := make([]Block, 0, expectedBlocks)
	err := s.walkBlocks(blockSize, true, func(block Block) (bool, error) {
		blocks = append(blocks, block)
		return true, nil
	})
	if err != nil {
		return nil, err
	}
	return blocks, nil
}

// RangeBlocks streams contiguous token-range blocks to yield without retaining
// every sliced block at once. Returning false from yield stops iteration.
func (s *Snapshot) RangeBlocks(blockSize int, yield func(Block) bool) error {
	if yield == nil {
		return errBlockYieldNil
	}
	return s.walkBlocks(blockSize, true, func(block Block) (bool, error) {
		return yield(block), nil
	})
}

func (s *Snapshot) walkBlocks(blockSize int, includeHash bool, yield func(Block) (bool, error)) error {
	if s == nil {
		return errSnapshotNil
	}
	if blockSize <= 0 {
		return errBlockSizeTooSmall
	}
	seqLen := EffectiveSeqLen(s)
	if seqLen <= 0 || len(s.Tokens) != seqLen {
		return errBlockSplitNeedsTokens
	}
	if s.HeadDim <= 0 {
		return errBlockSplitNeedsHeadDim
	}
	baseOffset := max(EffectiveTokenOffset(s)-seqLen, 0)
	boundaries, err := s.blockBoundaries(blockSize, seqLen)
	if err != nil {
		return err
	}
	// includeHash signals an external observer of the block snapshots —
	// SplitBlocks / RangeBlocks return blocks to the caller, so each
	// snapshot needs cloned slices for independent ownership. The internal
	// SaveStateBlocks path passes includeHash=false; it encodes + hashes
	// each block within yield and discards the snapshot before the next
	// iteration, so non-cloning sub-views are safe.
	cloneSlices := includeHash
	for i := 0; i < len(boundaries)-1; i++ {
		start := boundaries[i]
		end := boundaries[i+1]
		blockSnapshot, err := s.sliceBlockInternal(start, end, baseOffset, end == seqLen, cloneSlices)
		if err != nil {
			return err
		}
		var hash string
		if includeHash {
			hash, err = HashSnapshot(blockSnapshot)
			if err != nil {
				return err
			}
		}
		ok, err := yield(Block{
			Index:      i,
			TokenStart: start,
			TokenCount: end - start,
			Hash:       hash,
			Snapshot:   blockSnapshot,
		})
		if err != nil {
			return err
		}
		if !ok {
			return nil
		}
	}
	return nil
}

func (s *Snapshot) blockBoundaries(blockSize, seqLen int) ([]int, error) {
	if snapshotHasLayerCompressedPayloads(s) {
		return []int{0, seqLen}, nil
	}
	// Build directly into a sorted, dedup'd slice — boundary count is
	// O(seqLen/blockSize) + O(layers), typically <10. Mapping was the
	// 4th-largest alloc source on SaveStateBlocks.
	expected := 2 + (seqLen / blockSize) + len(s.Layers)
	boundaries := make([]int, 0, expected)
	// Deterministic boundaries are pre-sorted: 0, blockSize, 2*blockSize, ..., seqLen.
	boundaries = append(boundaries, 0)
	for next := blockSize; next < seqLen; next += blockSize {
		boundaries = append(boundaries, next)
	}
	boundaries = append(boundaries, seqLen)
	for _, layer := range s.Layers {
		windowLen, err := kvSnapshotLayerWindowLen(layer, seqLen, s.HeadDim)
		if err != nil {
			return nil, core.E("Snapshot.SplitBlocks", "layer window", err)
		}
		if windowLen <= 0 || windowLen >= seqLen {
			continue
		}
		boundaries = kvBoundaryInsert(boundaries, seqLen-windowLen)
	}
	return boundaries, nil
}

// kvBoundaryInsert keeps boundaries sorted + deduped while inserting v.
// boundaries is small (≤ seqLen/blockSize + few layer-window slots)
// so linear scan beats map ops or a binary search + memmove.
func kvBoundaryInsert(boundaries []int, v int) []int {
	for i, b := range boundaries {
		if b == v {
			return boundaries
		}
		if b > v {
			boundaries = append(boundaries, 0)
			copy(boundaries[i+1:], boundaries[i:])
			boundaries[i] = v
			return boundaries
		}
	}
	return append(boundaries, v)
}

func kvBlockPayloadSlices(payloads [][]byte, clone bool) [][]byte {
	if len(payloads) == 0 {
		return nil
	}
	out := make([][]byte, len(payloads))
	for i := range payloads {
		if clone {
			out[i] = core.SliceClone(payloads[i])
			continue
		}
		out[i] = payloads[i]
	}
	return out
}

func (s *Snapshot) SliceBlock(start, end, baseOffset int, final bool) (*Snapshot, error) {
	return s.sliceBlockInternal(start, end, baseOffset, final, true)
}

// sliceBlockInternal is the implementation of SliceBlock. When cloneSlices
// is false, per-head Key/Value/KeyBytes/ValueBytes return as sub-views of
// the parent snapshot — used only by walkBlocks(includeHash=false), the
// SaveStateBlocks path that immediately encodes and discards each block.
func (s *Snapshot) sliceBlockInternal(start, end, baseOffset int, final bool, cloneSlices bool) (*Snapshot, error) {
	if start < 0 || end <= start || end > len(s.Tokens) {
		return nil, errBlockRangeInvalid
	}
	seqLen := EffectiveSeqLen(s)
	layers := make([]LayerSnapshot, len(s.Layers))
	// Heads-slab: one backing slice across all layers collapses N per-layer
	// make([]HeadSnapshot,...) into a single allocation. Hot during
	// SaveStateBlocks — fires per checkpoint block × number of layers.
	// Layers with no overlap (windowLen <= 0) skip head slicing entirely;
	// the slab still under-uses the backing buffer in that case but never
	// over-allocates because we size against NumHeads.
	var headSlab []HeadSnapshot
	var slabCursor int
	if s.NumHeads > 0 && len(s.Layers) > 0 {
		headSlab = make([]HeadSnapshot, len(s.Layers)*s.NumHeads)
	}
	for layerIndex, layer := range s.Layers {
		windowLen, err := kvSnapshotLayerWindowLen(layer, seqLen, s.HeadDim)
		if err != nil {
			return nil, core.E("Snapshot.SplitBlocks", "layer window", err)
		}
		windowStart := seqLen - windowLen
		overlapStart := max(start, windowStart)
		overlapEnd := min(end, seqLen)
		layers[layerIndex] = LayerSnapshot{
			Layer:      layer.Layer,
			CacheIndex: layer.CacheIndex,
			CacheMode:  layer.CacheMode,
			MaxSize:    layer.MaxSize,
		}
		if len(layer.TurboQuantPayloads) > 0 {
			if start != 0 || end != seqLen {
				return nil, errBlockCompressedPayloadSplit
			}
			layers[layerIndex].TurboQuantPayloads = kvBlockPayloadSlices(layer.TurboQuantPayloads, cloneSlices)
			continue
		}
		if windowLen <= 0 || overlapStart >= overlapEnd {
			continue
		}
		localStart := overlapStart - windowStart
		localEnd := overlapEnd - windowStart
		keyLayerBytes, keyLayerShape, err := sliceKVSnapshotLayerRawTensorOpt(layer.KeyBytes, layer.KeyDType, layer.KeyShape, localStart, localEnd, cloneSlices)
		if err != nil {
			return nil, core.E("Snapshot.SplitBlocks", "slice native layer key tensor", err)
		}
		valueLayerBytes, valueLayerShape, err := sliceKVSnapshotLayerRawTensorOpt(layer.ValueBytes, layer.ValueDType, layer.ValueShape, localStart, localEnd, cloneSlices)
		if err != nil {
			return nil, core.E("Snapshot.SplitBlocks", "slice native layer value tensor", err)
		}
		layers[layerIndex].KeyDType = layer.KeyDType
		layers[layerIndex].KeyBytes = keyLayerBytes
		layers[layerIndex].KeyShape = keyLayerShape
		layers[layerIndex].ValueDType = layer.ValueDType
		layers[layerIndex].ValueBytes = valueLayerBytes
		layers[layerIndex].ValueShape = valueLayerShape
		headCount := len(layer.Heads)
		if headSlab != nil && slabCursor+headCount <= len(headSlab) {
			layers[layerIndex].Heads = headSlab[slabCursor : slabCursor+headCount : slabCursor+headCount]
			slabCursor += headCount
		} else {
			layers[layerIndex].Heads = make([]HeadSnapshot, headCount)
		}
		for headIndex, head := range layer.Heads {
			key, err := sliceKVSnapshotTensorOpt(head.Key, localStart, localEnd, s.HeadDim, windowLen, cloneSlices)
			if err != nil {
				return nil, core.E("Snapshot.SplitBlocks", "slice key tensor", err)
			}
			value, err := sliceKVSnapshotTensorOpt(head.Value, localStart, localEnd, s.HeadDim, windowLen, cloneSlices)
			if err != nil {
				return nil, core.E("Snapshot.SplitBlocks", "slice value tensor", err)
			}
			keyBytes, err := sliceKVSnapshotRawTensorOpt(head.KeyBytes, head.KeyDType, localStart, localEnd, windowLen, len(head.Key), cloneSlices)
			if err != nil {
				return nil, core.E("Snapshot.SplitBlocks", "slice native key tensor", err)
			}
			valueBytes, err := sliceKVSnapshotRawTensorOpt(head.ValueBytes, head.ValueDType, localStart, localEnd, windowLen, len(head.Value), cloneSlices)
			if err != nil {
				return nil, core.E("Snapshot.SplitBlocks", "slice native value tensor", err)
			}
			layers[layerIndex].Heads[headIndex] = HeadSnapshot{
				Key:        key,
				KeyDType:   head.KeyDType,
				KeyBytes:   keyBytes,
				Value:      value,
				ValueDType: head.ValueDType,
				ValueBytes: valueBytes,
			}
		}
	}
	var tokens []int32
	if cloneSlices {
		tokens = core.SliceClone(s.Tokens[start:end])
	} else {
		tokens = s.Tokens[start:end]
	}
	block := &Snapshot{
		Version:       effectiveVersion(s, KVSnapshotEncodingFloat32),
		Architecture:  s.Architecture,
		Tokens:        tokens,
		TokenOffset:   baseOffset + end,
		NumLayers:     s.NumLayers,
		NumHeads:      s.NumHeads,
		SeqLen:        end - start,
		HeadDim:       s.HeadDim,
		NumQueryHeads: s.NumQueryHeads,
		Layers:        layers,
	}
	if final {
		if cloneSlices {
			block.Generated = core.SliceClone(s.Generated)
			block.LogitShape = core.SliceClone(s.LogitShape)
			block.Logits = core.SliceClone(s.Logits)
		} else {
			block.Generated = s.Generated
			block.LogitShape = s.LogitShape
			block.Logits = s.Logits
		}
	}
	return block, nil
}

func ValidateStateBlockBundle(bundle *StateBlockBundle) error {
	if bundle == nil {
		return errBundleNil
	}
	if bundle.Version <= 0 || bundle.Version > StateBlockVersion {
		return errUnsupportedBundleVersion
	}
	if bundle.Kind != StateBlockBundleKind {
		return errBundleKindInvalid
	}
	if bundle.TokenCount <= 0 {
		return errBundleTokenCountEmpty
	}
	if len(bundle.Blocks) == 0 {
		return errBundleNoBlocks
	}
	return nil
}

// ValidateMemvidBlockBundle checks an old memvid-named KV block bundle.
//
// Deprecated: use ValidateStateBlockBundle.
func ValidateMemvidBlockBundle(bundle *MemvidBlockBundle) error {
	return ValidateStateBlockBundle(bundle)
}

func ClearTerminalState(snapshot *Snapshot) {
	if snapshot == nil {
		return
	}
	snapshot.Generated = nil
	snapshot.LogitShape = nil
	snapshot.Logits = nil
}

func loadKVSnapshotStateBlock(ctx context.Context, store state.Store, ref StateBlockRef) (Block, error) {
	return LoadStateBlockWithOptions(ctx, store, ref, LoadOptions{})
}

// LoadStateBlockWithOptions loads one durable State KV block with explicit
// decode options.
func LoadStateBlockWithOptions(ctx context.Context, store state.Store, ref StateBlockRef, opts LoadOptions) (Block, error) {
	if ref.PayloadEncoding == kvSnapshotStatePayloadRaw {
		return loadRawKVSnapshotStateBlockWithOptions(ctx, store, ref, opts)
	}
	chunk, err := state.Resolve(ctx, store, stateBlockChunkRef(ref).ChunkID)
	if err != nil {
		return Block{}, core.E("LoadFromStateBlocks", "resolve State block", err)
	}
	var envelope kvSnapshotStateBlockEnvelope
	if result := core.JSONUnmarshalString(chunk.Text, &envelope); !result.OK {
		return Block{}, core.E("LoadFromStateBlocks", "parse block envelope", ResultError(result))
	}
	data, err := decodeKVSnapshotStateBlockEnvelope(envelope, ref.KVHash)
	if err != nil {
		return Block{}, err
	}
	snapshot, err := parseKVSnapshotWithOptions(data, opts)
	if err != nil {
		return Block{}, err
	}
	return Block{
		Index:      envelope.BlockIndex,
		TokenStart: envelope.TokenStart,
		TokenCount: envelope.TokenCount,
		Hash:       envelope.KVHash,
		Snapshot:   snapshot,
	}, nil
}

// LoadMemvidBlockWithOptions loads one memvid KV block with explicit decode
// options.
//
// Deprecated: use LoadStateBlockWithOptions.
func LoadMemvidBlockWithOptions(ctx context.Context, store state.Store, ref StateBlockRef, opts LoadOptions) (Block, error) {
	return LoadStateBlockWithOptions(ctx, store, ref, opts)
}

// LoadStateBlockTokens loads only token IDs from one durable State KV block.
func LoadStateBlockTokens(ctx context.Context, store state.Store, ref StateBlockRef) (StateTokenBlock, error) {
	return LoadStateBlockTokensWithOptions(ctx, store, ref, LoadOptions{})
}

// LoadStateBlockTokensWithOptions loads only token IDs from one durable State
// KV block. Decode options are accepted for symmetry with full block loading;
// tensor payloads are skipped rather than decoded.
func LoadStateBlockTokensWithOptions(ctx context.Context, store state.Store, ref StateBlockRef, _ LoadOptions) (StateTokenBlock, error) {
	if ref.PayloadEncoding == kvSnapshotStatePayloadRaw {
		data, err := loadRawStateBlockPayload(ctx, store, ref)
		if err != nil {
			return StateTokenBlock{}, err
		}
		tokens, err := parseKVSnapshotTokens(data)
		if err != nil {
			return StateTokenBlock{}, err
		}
		return StateTokenBlock{
			Index:      ref.Index,
			TokenStart: ref.TokenStart,
			TokenCount: ref.TokenCount,
			Hash:       ref.KVHash,
			Tokens:     tokens,
		}, nil
	}
	chunk, err := state.Resolve(ctx, store, stateBlockChunkRef(ref).ChunkID)
	if err != nil {
		return StateTokenBlock{}, core.E("LoadFromStateBlocks", "resolve State token block", err)
	}
	var envelope kvSnapshotStateBlockEnvelope
	if result := core.JSONUnmarshalString(chunk.Text, &envelope); !result.OK {
		return StateTokenBlock{}, core.E("LoadFromStateBlocks", "parse token block envelope", ResultError(result))
	}
	data, err := decodeKVSnapshotStateBlockEnvelope(envelope, ref.KVHash)
	if err != nil {
		return StateTokenBlock{}, err
	}
	tokens, err := parseKVSnapshotTokens(data)
	if err != nil {
		return StateTokenBlock{}, err
	}
	return StateTokenBlock{
		Index:      envelope.BlockIndex,
		TokenStart: envelope.TokenStart,
		TokenCount: envelope.TokenCount,
		Hash:       envelope.KVHash,
		Tokens:     tokens,
	}, nil
}

func loadRawKVSnapshotStateBlockWithOptions(ctx context.Context, store state.Store, ref StateBlockRef, opts LoadOptions) (Block, error) {
	data, err := loadRawStateBlockPayload(ctx, store, ref)
	if err != nil {
		return Block{}, err
	}
	snapshot, err := parseKVSnapshotWithOptions(data, opts)
	if err != nil {
		return Block{}, err
	}
	return Block{
		Index:      ref.Index,
		TokenStart: ref.TokenStart,
		TokenCount: ref.TokenCount,
		Hash:       ref.KVHash,
		Snapshot:   snapshot,
	}, nil
}

func loadRawStateBlockPayload(ctx context.Context, store state.Store, ref StateBlockRef) ([]byte, error) {
	chunk, err := state.BorrowRefBytes(ctx, store, stateBlockChunkRef(ref))
	if err != nil {
		return nil, core.E("LoadFromStateBlocks", "resolve raw State block", err)
	}
	data := chunk.Data
	if ref.PayloadByteCount > 0 && len(data) != ref.PayloadByteCount {
		return nil, errRawBlockPayloadLenMismatch
	}
	hash := core.SHA256Hex(data)
	if ref.KVHash != "" && hash != ref.KVHash {
		return nil, errRawBlockHashMismatch
	}
	return data, nil
}

// StateBlockChunkRef returns the current State chunk ref for a block,
// falling back to the deprecated json:"memvid" ref for older bundles.
func StateBlockChunkRef(ref StateBlockRef) state.ChunkRef {
	if ref.State.ChunkID != 0 || ref.State.Segment != "" || ref.State.Codec != "" || ref.State.HasFrameOffset {
		return ref.State
	}
	return ref.Memvid
}

func stateBlockChunkRef(ref StateBlockRef) state.ChunkRef {
	return StateBlockChunkRef(ref)
}

func decodeKVSnapshotStateBlockEnvelope(envelope kvSnapshotStateBlockEnvelope, expectedHash string) ([]byte, error) {
	if envelope.Version <= 0 || envelope.Version > StateBlockVersion {
		return nil, errUnsupportedBlockVersion
	}
	if envelope.Kind != KVSnapshotStateBlockKind {
		return nil, errBlockKindInvalid
	}
	if envelope.BinaryEncoding != "base64" {
		return nil, errUnsupportedBlockEncoding
	}
	decoded := core.Base64Decode(envelope.Data)
	if !decoded.OK {
		return nil, core.E("LoadFromStateBlocks", "decode block payload", ResultError(decoded))
	}
	data, ok := decoded.Value.([]byte)
	if !ok {
		return nil, errBlockNonByteData
	}
	if envelope.PayloadByteCount > 0 && len(data) != envelope.PayloadByteCount {
		return nil, errBlockPayloadLenMismatch
	}
	hash := core.SHA256Hex(data)
	if envelope.KVHash != "" && hash != envelope.KVHash {
		return nil, errBlockHashMismatch
	}
	if expectedHash != "" && hash != expectedHash {
		return nil, errBlockRefHashMismatch
	}
	return data, nil
}

func EffectiveSeqLen(snapshot *Snapshot) int {
	if snapshot == nil {
		return 0
	}
	if snapshot.SeqLen > 0 {
		return snapshot.SeqLen
	}
	return len(snapshot.Tokens)
}
