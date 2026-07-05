// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"

	core "dappco.re/go"
	state "dappco.re/go/inference/model/state"
)

// LoadFromStateBlocks restores a full KV snapshot from a State block manifest.
func LoadFromStateBlocks(ctx context.Context, store state.Store, bundle *StateBlockBundle) (*Snapshot, error) {
	return LoadFromStateBlocksWithOptions(ctx, store, bundle, LoadOptions{})
}

// LoadFromMemvidBlocks restores a full KV snapshot from a memvid block manifest.
//
// Deprecated: use LoadFromStateBlocks.
func LoadFromMemvidBlocks(ctx context.Context, store state.Store, bundle *StateBlockBundle) (*Snapshot, error) {
	return LoadFromStateBlocks(ctx, store, bundle)
}

// LoadStateBlockBundle restores a KV block manifest by URI from the
// same State store as its referenced blocks.
func LoadStateBlockBundle(ctx context.Context, store state.Store, uri string) (*StateBlockBundle, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return nil, errStateStoreNil
	}
	if core.Trim(uri) == "" {
		return nil, errBundleURIRequired
	}
	chunk, err := state.ResolveURI(ctx, store, uri)
	if err != nil {
		return nil, core.E("LoadStateBlockBundle", "resolve State bundle", err)
	}
	var bundle StateBlockBundle
	if result := core.JSONUnmarshalString(chunk.Text, &bundle); !result.OK {
		return nil, core.E("LoadStateBlockBundle", "parse bundle", result.Err())
	}
	if err := ValidateStateBlockBundle(&bundle); err != nil {
		return nil, err
	}
	return &bundle, nil
}

// LoadMemvidBlockBundle restores a KV block manifest by URI from an old
// memvid-named store.
//
// Deprecated: use LoadStateBlockBundle.
func LoadMemvidBlockBundle(ctx context.Context, store state.Store, uri string) (*MemvidBlockBundle, error) {
	return LoadStateBlockBundle(ctx, store, uri)
}

// LoadFromStateBlocksWithOptions restores a full KV snapshot from a
// State block manifest with explicit decode options.
func LoadFromStateBlocksWithOptions(ctx context.Context, store state.Store, bundle *StateBlockBundle, opts LoadOptions) (*Snapshot, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return nil, errStateStoreNil
	}
	if bundle == nil {
		return nil, errBundleNil
	}
	if bundle.Version <= 0 || bundle.Version > StateBlockVersion {
		return nil, errUnsupportedBundleVersion
	}
	if bundle.Kind != StateBlockBundleKind {
		return nil, errBundleKindInvalid
	}
	if len(bundle.Blocks) == 0 {
		return nil, errBlocksEmpty
	}
	// Stream-assemble: load each block, fold into the assembled snapshot,
	// then release the per-block snapshot pointer. Avoids holding every
	// per-block []float32 / []byte alive until AssembleBlocks runs.
	snapshot, err := loadAndAssembleStateBlocks(ctx, store, bundle, opts)
	if err != nil {
		return nil, err
	}
	if bundle.TokenOffset > 0 && snapshot.TokenOffset != bundle.TokenOffset {
		return nil, errBlockTokenOffsetMismatch
	}
	return snapshot, nil
}

// loadAndAssembleStateBlocks streams blocks from a State bundle into a
// single assembled snapshot without retaining the per-block Snapshot
// pointers between iterations. The first block defines the assembled
// shape (Architecture, Layer count, head dimensions, raw tensor dtypes
// + shapes) — subsequent blocks fold into the same skeleton.
func loadAndAssembleStateBlocks(ctx context.Context, store state.Store, bundle *StateBlockBundle, opts LoadOptions) (*Snapshot, error) {
	// Validate ordering up front against bundle.Blocks rather than after
	// loading every snapshot. The full block snapshots aren't required
	// for ordering checks.
	totalTokens := 0
	nextStart := 0
	for index, ref := range bundle.Blocks {
		if ref.Index != index {
			return nil, errBlocksOutOfOrder
		}
		if ref.TokenStart != nextStart || ref.TokenCount <= 0 {
			return nil, errBlocksNotContiguous
		}
		nextStart += ref.TokenCount
		totalTokens += ref.TokenCount
	}
	var assembled *Snapshot
	var lastBlock *Snapshot
	for index, ref := range bundle.Blocks {
		block, err := LoadStateBlockWithOptions(ctx, store, ref, opts)
		if err != nil {
			return nil, err
		}
		if block.Snapshot == nil {
			return nil, errBlockNil
		}
		if block.Index != index || block.TokenStart != ref.TokenStart || block.TokenCount != ref.TokenCount {
			return nil, errBlockMetadataMismatch
		}
		if len(block.Snapshot.Tokens) != ref.TokenCount {
			return nil, errBlockTokenCountMismatch
		}
		if assembled == nil {
			first := block.Snapshot
			assembled = &Snapshot{
				Version:       first.Version,
				Architecture:  first.Architecture,
				NumLayers:     first.NumLayers,
				NumHeads:      first.NumHeads,
				HeadDim:       first.HeadDim,
				NumQueryHeads: first.NumQueryHeads,
				Layers:        emptyKVSnapshotLayers(first.Layers),
				Tokens:        make([]int32, 0, totalTokens),
			}
			// Pre-size assembled per-head byte buffers from bundle metadata
			// rather than walking the full block list — the bundle's
			// PayloadByteCount sums the raw block payload sizes, which
			// approximates the head byte counts when payload encoding is
			// raw. Falls back to no pre-size when bytes counts aren't
			// available; appendKVSnapshotRawBlock then handles growth.
			preSizeAssembledRawBytesFromFirst(assembled, first, len(bundle.Blocks))
		}
		if err := appendKVSnapshotBlock(assembled, block.Snapshot); err != nil {
			return nil, err
		}
		lastBlock = block.Snapshot
	}
	if assembled == nil || lastBlock == nil {
		return nil, errBlocksEmpty
	}
	assembled.Generated = core.SliceClone(lastBlock.Generated)
	assembled.TokenOffset = lastBlock.TokenOffset
	assembled.LogitShape = core.SliceClone(lastBlock.LogitShape)
	assembled.Logits = core.SliceClone(lastBlock.Logits)
	if assembled.TokenOffset == 0 {
		assembled.TokenOffset = len(assembled.Tokens)
	}
	return assembled, nil
}

func loadAndAssembleStateBlockPrefix(ctx context.Context, store state.Store, bundle *StateBlockBundle, prefixTokens int, opts LoadOptions) (*Snapshot, error) {
	blockCount, err := stateBlockPrefixCoverage(bundle, prefixTokens)
	if err != nil {
		return nil, err
	}
	var assembled *Snapshot
	var lastBlock *Snapshot
	for index := range blockCount {
		ref := bundle.Blocks[index]
		block, err := LoadStateBlockWithOptions(ctx, store, ref, opts)
		if err != nil {
			return nil, err
		}
		if block.Snapshot == nil {
			return nil, errBlockNil
		}
		if block.Index != ref.Index || block.TokenStart != ref.TokenStart || block.TokenCount != ref.TokenCount {
			return nil, errBlockMetadataMismatch
		}
		if len(block.Snapshot.Tokens) != ref.TokenCount {
			return nil, errBlockTokenCountMismatch
		}
		blockSnapshot := block.Snapshot
		if ref.TokenStart+ref.TokenCount > prefixTokens {
			trimEnd := prefixTokens - ref.TokenStart
			if trimEnd <= 0 {
				break
			}
			baseOffset := EffectiveTokenOffset(blockSnapshot) - EffectiveSeqLen(blockSnapshot)
			if baseOffset < 0 {
				baseOffset = ref.TokenStart
			}
			blockSnapshot, err = blockSnapshot.SliceBlock(0, trimEnd, baseOffset, false)
			if err != nil {
				return nil, err
			}
		}
		if assembled == nil {
			first := blockSnapshot
			assembled = &Snapshot{
				Version:       first.Version,
				Architecture:  first.Architecture,
				NumLayers:     first.NumLayers,
				NumHeads:      first.NumHeads,
				HeadDim:       first.HeadDim,
				NumQueryHeads: first.NumQueryHeads,
				Layers:        emptyKVSnapshotLayers(first.Layers),
				Tokens:        make([]int32, 0, prefixTokens),
			}
			preSizeAssembledRawBytesFromFirst(assembled, first, blockCount)
		}
		if err := appendKVSnapshotBlock(assembled, blockSnapshot); err != nil {
			return nil, err
		}
		lastBlock = blockSnapshot
	}
	if assembled == nil || lastBlock == nil {
		return nil, errPrefixNoCoveringBlocks
	}
	assembled.Generated = core.SliceClone(lastBlock.Generated)
	assembled.TokenOffset = lastBlock.TokenOffset
	assembled.LogitShape = core.SliceClone(lastBlock.LogitShape)
	assembled.Logits = core.SliceClone(lastBlock.Logits)
	if assembled.TokenOffset == 0 {
		assembled.TokenOffset = len(assembled.Tokens)
	}
	return assembled, nil
}

func stateBlockPrefixCoverage(bundle *StateBlockBundle, prefixTokens int) (int, error) {
	if bundle == nil || len(bundle.Blocks) == 0 {
		return 0, errPrefixNoCoveringBlocks
	}
	nextStart := 0
	totalTokens := 0
	blockCount := 0
	for index, ref := range bundle.Blocks {
		if ref.TokenStart >= prefixTokens {
			break
		}
		if ref.Index != index {
			return 0, errBlocksOutOfOrder
		}
		if ref.TokenStart != nextStart || ref.TokenCount <= 0 {
			return 0, errBlocksNotContiguous
		}
		nextStart += ref.TokenCount
		totalTokens += ref.TokenCount
		blockCount++
		if totalTokens >= prefixTokens {
			break
		}
	}
	if blockCount == 0 {
		return 0, errPrefixNoCoveringBlocks
	}
	if totalTokens < prefixTokens {
		return 0, errPrefixBlocksNoCover
	}
	return blockCount, nil
}

// preSizeAssembledRawBytesFromFirst pre-allocates per-head KeyBytes /
// ValueBytes buffers in assembled by extrapolating from the first
// block's byte count × the block count — cheaper than the full-blocks
// pre-pass when blocks are uniformly sized.
func preSizeAssembledRawBytesFromFirst(assembled *Snapshot, first *Snapshot, blockCount int) {
	if assembled == nil || first == nil || blockCount <= 0 {
		return
	}
	for layerIndex := range assembled.Layers {
		if layerIndex >= len(first.Layers) {
			continue
		}
		firstLayer := first.Layers[layerIndex]
		dstLayer := &assembled.Layers[layerIndex]
		if keyCap := len(firstLayer.KeyBytes) * blockCount; keyCap > 0 {
			dstLayer.KeyBytes = make([]byte, 0, keyCap)
		}
		if valueCap := len(firstLayer.ValueBytes) * blockCount; valueCap > 0 {
			dstLayer.ValueBytes = make([]byte, 0, valueCap)
		}
		for headIndex := range assembled.Layers[layerIndex].Heads {
			if headIndex >= len(firstLayer.Heads) {
				continue
			}
			firstHead := firstLayer.Heads[headIndex]
			dstHead := &dstLayer.Heads[headIndex]
			if keyCap := len(firstHead.KeyBytes) * blockCount; keyCap > 0 {
				dstHead.KeyBytes = make([]byte, 0, keyCap)
			}
			if valueCap := len(firstHead.ValueBytes) * blockCount; valueCap > 0 {
				dstHead.ValueBytes = make([]byte, 0, valueCap)
			}
			// Pre-size the float32 Key/Value slices on the float32-encoded
			// path. appendKVSnapshotBlock appends head.Key/head.Value per
			// block; without this hint they ride Go's geometric grow (one or
			// two reallocs by block 3). The KeyBytes/ValueBytes pre-size above
			// only covers the native raw path.
			if keyCap := len(firstHead.Key) * blockCount; keyCap > 0 {
				dstHead.Key = make([]float32, 0, keyCap)
			}
			if valueCap := len(firstHead.Value) * blockCount; valueCap > 0 {
				dstHead.Value = make([]float32, 0, valueCap)
			}
		}
	}
}

// LoadFromMemvidBlocksWithOptions restores a full KV snapshot from a
// memvid block manifest with explicit decode options.
//
// Deprecated: use LoadFromStateBlocksWithOptions.
func LoadFromMemvidBlocksWithOptions(ctx context.Context, store state.Store, bundle *StateBlockBundle, opts LoadOptions) (*Snapshot, error) {
	return LoadFromStateBlocksWithOptions(ctx, store, bundle, opts)
}

// LoadPrefixFromStateBlocks restores only the State KV blocks needed
// to cover prefixTokens. The returned snapshot is suitable for prompt-cache
// warmup; non-final prefixes intentionally omit logits.
func LoadPrefixFromStateBlocks(ctx context.Context, store state.Store, bundle *StateBlockBundle, prefixTokens int) (*Snapshot, error) {
	return LoadPrefixFromStateBlocksWithOptions(ctx, store, bundle, prefixTokens, LoadOptions{})
}

// LoadPrefixFromMemvidBlocks restores only the memvid KV blocks needed
// to cover prefixTokens. The returned snapshot is suitable for prompt-cache
// warmup; non-final prefixes intentionally omit logits.
//
// Deprecated: use LoadPrefixFromStateBlocks.
func LoadPrefixFromMemvidBlocks(ctx context.Context, store state.Store, bundle *StateBlockBundle, prefixTokens int) (*Snapshot, error) {
	return LoadPrefixFromStateBlocks(ctx, store, bundle, prefixTokens)
}

// LoadPrefixFromStateBlocksWithOptions restores only the State KV
// blocks needed to cover prefixTokens with explicit decode options.
func LoadPrefixFromStateBlocksWithOptions(ctx context.Context, store state.Store, bundle *StateBlockBundle, prefixTokens int, opts LoadOptions) (*Snapshot, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return nil, errStateStoreNil
	}
	if err := ValidateStateBlockBundle(bundle); err != nil {
		return nil, err
	}
	if prefixTokens <= 0 || prefixTokens == bundle.TokenCount {
		return LoadFromStateBlocksWithOptions(ctx, store, bundle, opts)
	}
	if prefixTokens > bundle.TokenCount {
		return nil, errPrefixExceedsBundle
	}
	snapshot, err := loadAndAssembleStateBlockPrefix(ctx, store, bundle, prefixTokens, opts)
	if err != nil {
		return nil, err
	}
	if len(snapshot.Tokens) == prefixTokens {
		if prefixTokens < bundle.TokenCount {
			ClearTerminalState(snapshot)
		}
		return snapshot, nil
	}
	if len(snapshot.Tokens) < prefixTokens {
		return nil, errPrefixBlocksNoCover
	}
	baseOffset := max(EffectiveTokenOffset(snapshot)-EffectiveSeqLen(snapshot), 0)
	trimmed, err := snapshot.SliceBlock(0, prefixTokens, baseOffset, false)
	if err != nil {
		return nil, err
	}
	return trimmed, nil
}

// LoadPrefixFromMemvidBlocksWithOptions restores only the memvid KV
// blocks needed to cover prefixTokens with explicit decode options.
//
// Deprecated: use LoadPrefixFromStateBlocksWithOptions.
func LoadPrefixFromMemvidBlocksWithOptions(ctx context.Context, store state.Store, bundle *StateBlockBundle, prefixTokens int, opts LoadOptions) (*Snapshot, error) {
	return LoadPrefixFromStateBlocksWithOptions(ctx, store, bundle, prefixTokens, opts)
}

// LoadPrefixTokensFromStateBlocks restores only token IDs from a State block
// manifest. It intentionally avoids K/V assembly, which is the correct wake
// path for folded State because the compact prompt will be prefetched again.
func LoadPrefixTokensFromStateBlocks(ctx context.Context, store state.Store, bundle *StateBlockBundle, prefixTokens int) ([]int32, error) {
	return LoadPrefixTokensFromStateBlocksWithOptions(ctx, store, bundle, prefixTokens, LoadOptions{})
}

// LoadPrefixTokensFromStateBlocksWithOptions restores only token IDs from the
// blocks needed to cover prefixTokens with explicit decode options.
func LoadPrefixTokensFromStateBlocksWithOptions(ctx context.Context, store state.Store, bundle *StateBlockBundle, prefixTokens int, opts LoadOptions) ([]int32, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return nil, errStateStoreNil
	}
	if err := ValidateStateBlockBundle(bundle); err != nil {
		return nil, err
	}
	if prefixTokens <= 0 {
		prefixTokens = bundle.TokenCount
	}
	if prefixTokens > bundle.TokenCount {
		return nil, errTokenPrefixExceeds
	}
	// Inline iteration over bundle.Blocks skips the intermediate
	// stateBlockRefsForPrefix slice allocation — we already break when the
	// running token count covers prefixTokens, the same condition
	// stateBlockRefsForPrefix uses to truncate.
	if len(bundle.Blocks) == 0 {
		return nil, errTokenPrefixNoBlocks
	}
	tokens := make([]int32, 0, prefixTokens)
	nextStart := 0
	expectedIndex := 0
	covered := false
	for _, ref := range bundle.Blocks {
		if ref.TokenStart >= prefixTokens {
			break
		}
		if ref.Index != expectedIndex || ref.TokenStart != nextStart || ref.TokenCount <= 0 {
			return nil, errTokenBlocksNotContiguous
		}
		// Fast path: when the block is raw-payload-stored (the predominant
		// case after the SaveStateBlocks switch to BinaryWriter), parse
		// tokens directly into the result slice. Avoids the per-block
		// []int32 allocation that LoadStateBlockTokensWithOptions would
		// otherwise pay through parseKVSnapshotTokens.
		var blockTokenCount int
		var err error
		if ref.PayloadEncoding == kvSnapshotStatePayloadRaw {
			data, derr := loadRawStateBlockPayload(ctx, store, ref)
			if derr != nil {
				return nil, derr
			}
			before := len(tokens)
			tokens, err = parseKVSnapshotTokensInto(tokens, data)
			if err != nil {
				return nil, err
			}
			blockTokenCount = len(tokens) - before
		} else {
			block, lerr := LoadStateBlockTokensWithOptions(ctx, store, ref, opts)
			if lerr != nil {
				return nil, lerr
			}
			if block.Index != ref.Index || block.TokenStart != ref.TokenStart || block.TokenCount != ref.TokenCount {
				return nil, errTokenBlockMetadata
			}
			tokens = append(tokens, block.Tokens...)
			blockTokenCount = len(block.Tokens)
		}
		if blockTokenCount != ref.TokenCount {
			return nil, errTokenBlockTokenCount
		}
		nextStart += ref.TokenCount
		expectedIndex++
		covered = true
		if len(tokens) >= prefixTokens {
			break
		}
	}
	if !covered {
		return nil, errTokenPrefixNoBlocks
	}
	if len(tokens) < prefixTokens {
		return nil, errTokenPrefixNoCover
	}
	return tokens[:prefixTokens], nil
}
