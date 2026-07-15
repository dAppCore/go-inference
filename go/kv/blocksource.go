// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"

	core "dappco.re/go"
	state "dappco.re/go/inference/model/state"
)

// blocksource.go: streaming durable State KV blocks as portable [Block]s
// without assembling a full CPU snapshot first — the engine-neutral form of
// the per-block restore path. docs/engine-merge.md retires the metal-typed
// kvconv.MetalKVSnapshotBlockSource against this: the session KV-block
// restorer consumes a BlockSource in kv terms, each engine converts inward.

var (
	errBlockSourceStoreNil     = core.NewError("mlx: state store is nil")
	errBlockSourcePrefixExceed = core.NewError("mlx: State KV prefix exceeds bundle token count")
	errBlockSourceNoCovering   = core.NewError("mlx: State KV prefix has no covering blocks")
	errBlockSourceOutOfRange   = core.NewError("mlx: State KV block index is out of range")
	errBlockSourceMetaMismatch = core.NewError("mlx: State KV block metadata mismatch")
	errBlockSourceSnapshotNil  = core.NewError("mlx: State KV block snapshot is nil")
	errBlockSourceInvalidTrim  = core.NewError("mlx: State KV prefix has invalid trim range")
)

// BlockSource streams KV snapshot blocks lazily — the per-block restore path
// that avoids a full CPU-side assembled [Snapshot]. Load(ctx, index) yields the
// index-th covering block; BlockCount blocks cover PrefixTokens.
type BlockSource struct {
	TokenCount   int
	PrefixTokens int
	BlockCount   int
	Load         func(context.Context, int) (Block, error)
}

// StateBlockSource builds a streamed BlockSource that lazily loads and trims
// the durable State KV blocks covering prefixTokens (0 = the whole bundle).
//
//	src, err := kv.StateBlockSource(ctx, store, bundle, prefixTokens)
func StateBlockSource(ctx context.Context, store state.Store, bundle *StateBlockBundle, prefixTokens int) (BlockSource, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return BlockSource{}, errBlockSourceStoreNil
	}
	if err := ValidateStateBlockBundle(bundle); err != nil {
		return BlockSource{}, err
	}
	if prefixTokens <= 0 {
		prefixTokens = bundle.TokenCount
	}
	if prefixTokens > bundle.TokenCount {
		return BlockSource{}, errBlockSourcePrefixExceed
	}
	blocks := bundle.Blocks
	blockCount, err := stateBlockSourceCoverage(blocks, prefixTokens)
	if err != nil {
		return BlockSource{}, err
	}
	source := BlockSource{
		TokenCount:   bundle.TokenCount,
		PrefixTokens: prefixTokens,
		BlockCount:   blockCount,
	}
	// Hoist invariants out of the per-block closure. KVEncoding is bundle-
	// scoped — checking it once at construction lets each Load call use
	// the captured loadOpts directly without re-branching on every block.
	loadOpts := LoadOptions{}
	if bundle.KVEncoding == EncodingNative {
		loadOpts.RawKVOnly = true
	}
	source.Load = func(loadCtx context.Context, index int) (Block, error) {
		if loadCtx == nil {
			loadCtx = ctx
		}
		if index < 0 || index >= blockCount {
			return Block{}, errBlockSourceOutOfRange
		}
		ref := &blocks[index]
		block, err := LoadStateBlockWithOptions(loadCtx, store, *ref, loadOpts)
		if err != nil {
			return Block{}, err
		}
		if block.TokenStart != ref.TokenStart || block.TokenCount != ref.TokenCount {
			return Block{}, errBlockSourceMetaMismatch
		}
		snapshot := block.Snapshot
		if snapshot == nil {
			return Block{}, errBlockSourceSnapshotNil
		}
		if block.TokenStart+block.TokenCount > prefixTokens {
			trimTokens := prefixTokens - block.TokenStart
			if trimTokens <= 0 {
				return Block{}, errBlockSourceInvalidTrim
			}
			baseOffset := max(EffectiveTokenOffset(snapshot)-EffectiveSeqLen(snapshot), 0)
			trimmed, trimErr := snapshot.SliceBlock(0, trimTokens, baseOffset, false)
			if trimErr != nil {
				return Block{}, trimErr
			}
			snapshot = trimmed
			block.TokenCount = trimTokens
		}
		if block.TokenStart+block.TokenCount < bundle.TokenCount {
			ClearTerminalState(snapshot)
		}
		return Block{
			Index:      index,
			TokenStart: block.TokenStart,
			TokenCount: block.TokenCount,
			Snapshot:   snapshot,
		}, nil
	}
	return source, nil
}

func stateBlockSourceCoverage(blocks []StateBlockRef, prefixTokens int) (int, error) {
	if len(blocks) == 0 {
		return 0, errBlockSourceNoCovering
	}
	nextStart := 0
	blockCount := 0
	for i := range blocks {
		ref := &blocks[i]
		if ref.TokenStart >= prefixTokens {
			break
		}
		if ref.Index != i || ref.TokenStart != nextStart || ref.TokenCount <= 0 {
			return 0, errBlockSourceMetaMismatch
		}
		nextStart += ref.TokenCount
		blockCount++
		if nextStart >= prefixTokens {
			break
		}
	}
	if blockCount == 0 || nextStart < prefixTokens {
		return 0, errBlockSourceNoCovering
	}
	return blockCount, nil
}
