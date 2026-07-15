// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"context"
	"encoding/binary"
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/model/safetensors"
)

const nativeKVSnapshotDTypeBF16 = "bfloat16"

// KVBlockSource streams root KV snapshot blocks without requiring callers to
// assemble a full CPU-side kv.Snapshot first. RetainedLogits borrows the source
// session's retained boundary buffer; consume or copy it before mutating/closing
// the source session.
type KVBlockSource struct {
	TokenCount          int
	PrefixTokens        int
	TrustedPrefixTokens int
	FirstBlockIndex     int
	CachedIDs           []int32
	RetainedLogits      []byte
	BlockCount          int
	Load                func(int) (kv.Block, error)
	nativeStateSource   *SessionStateBlockSource
}

// CaptureKV captures the session's current native K/V cache as a root KV
// snapshot. Native stores cache rows token-major; root KV snapshots store raw
// layer slabs as [1, heads, seq, head_dim], so capture transposes once at the
// API boundary and keeps the resident cache layout unchanged.
func (s *ArchSession) CaptureKV() (*kv.Snapshot, error) {
	return s.CaptureKVWithOptions(kv.CaptureOptions{})
}

// CaptureKVWithOptions captures native K/V as root kv.Snapshot data without
// depending on pkg/metal. RawKVOnly preserves the fast native BF16 slab path;
// the default path also derives per-head float32 tensors for portable callers.
func (s *ArchSession) CaptureKVWithOptions(opts kv.CaptureOptions) (*kv.Snapshot, error) {
	if s == nil {
		return nil, core.NewError("native.CaptureKV: nil session")
	}
	if opts.BlockStartToken < 0 {
		return nil, core.NewError("native.CaptureKV: block start token must be >= 0")
	}
	if s.pos <= 0 {
		return nil, core.NewError("native.CaptureKV: empty cache")
	}
	if s.pos > s.maxLen {
		return nil, core.NewError("native.CaptureKV: position outside maxLen")
	}
	if len(s.cachedIDs) != s.pos {
		return nil, core.NewError("native.CaptureKV: cached ids do not match position")
	}
	views, err := s.stateLayerViews()
	if err != nil {
		return nil, err
	}
	layers := s.kvSnapshotLayerMetadata()
	for _, view := range views {
		start, tokenCount, err := nativeKVLayerCaptureWindow(view, s.pos)
		if err != nil {
			return nil, err
		}
		keyRows, valueRows, err := stateBlockLayerBytes(view, start, tokenCount, s.pos)
		if err != nil {
			return nil, err
		}
		if len(keyRows) != tokenCount*view.rowBytes || len(valueRows) != tokenCount*view.rowBytes {
			return nil, core.NewError("native.CaptureKV: layer payload size mismatch")
		}
		keySlab := make([]byte, len(keyRows))
		valueSlab := make([]byte, len(valueRows))
		nativeKVTokenRowsToLayerSlab(keySlab, keyRows, tokenCount, view.kvHeads, view.headDim)
		nativeKVTokenRowsToLayerSlab(valueSlab, valueRows, tokenCount, view.kvHeads, view.headDim)
		shape := []int32{1, int32(view.kvHeads), int32(tokenCount), int32(view.headDim)}
		layer := kv.LayerSnapshot{
			Layer:      view.layer,
			CacheIndex: view.cacheIndex,
			CacheMode:  view.cacheMode,
			MaxSize:    view.maxSize,
			KeyDType:   nativeKVSnapshotDTypeBF16,
			KeyBytes:   keySlab,
			KeyShape:   append([]int32(nil), shape...),
			ValueDType: nativeKVSnapshotDTypeBF16,
			ValueBytes: valueSlab,
			ValueShape: append([]int32(nil), shape...),
		}
		if !opts.RawKVOnly {
			layer.Heads = nativeKVLayerSlabHeads(keySlab, valueSlab, tokenCount, view.kvHeads, view.headDim)
		}
		// q8 layers carry the store's RAW int8 codes + f32 scales verbatim under
		// kv.KVNativeDTypeQ8 so a q8→q8 sleep/wake is bit-exact (the bf16 slab
		// above would double-quantise on restore, perturbing every prefix row —
		// #1846). The bf16-derived Heads stay populated for portable (non-q8)
		// consumers; only the layer-level K/V bytes+dtype switch to the raw block.
		if s.state.icb != nil && s.state.icb.kvQ8.on(view.layer) {
			kPacked, vPacked, err := s.state.icb.captureQ8LayerRaw(view.layer, start, tokenCount)
			if err != nil {
				return nil, err
			}
			layer.KeyDType = kv.KVNativeDTypeQ8
			layer.KeyBytes = kPacked
			layer.ValueDType = kv.KVNativeDTypeQ8
			layer.ValueBytes = vPacked
		}
		layers[view.layer] = layer
	}
	logits, logitShape, err := s.captureKVLogits()
	if err != nil {
		return nil, err
	}
	return &kv.Snapshot{
		Version:       kv.SnapshotVersion,
		Tokens:        append([]int32(nil), s.cachedIDs...),
		TokenOffset:   s.pos,
		NumLayers:     len(s.state.specs),
		NumHeads:      s.arch.MaxKVHeads(),
		SeqLen:        s.pos,
		HeadDim:       s.arch.MaxHeadDim(),
		NumQueryHeads: s.arch.Heads,
		LogitShape:    logitShape,
		Logits:        logits,
		Layers:        layers,
	}, nil
}

// KVBlockSource returns a loader over the current resident native K/V cache as
// root kv.Block snapshots. Blocks borrow resident state until each load returns;
// the returned kv.Block owns its byte payloads.
func (s *ArchSession) KVBlockSource(blockSize int, opts kv.CaptureOptions) (KVBlockSource, error) {
	if s == nil {
		return KVBlockSource{}, core.NewError("native.KVBlockSource: nil session")
	}
	stateSource, err := s.StateBlockSourceFrom(opts.BlockStartToken, blockSize)
	if err != nil {
		return KVBlockSource{}, err
	}
	source := KVBlockSource{
		TokenCount:          stateSource.Position,
		PrefixTokens:        stateSource.Position,
		TrustedPrefixTokens: stateSource.trustedPrefixTokens(),
		FirstBlockIndex:     stateSource.firstBlockIndex,
		CachedIDs:           append([]int32(nil), stateSource.CachedIDs...),
		RetainedLogits:      stateSource.RetainedLogits,
		BlockCount:          stateSource.BlockCount,
		nativeStateSource:   &stateSource,
	}
	source.Load = func(index int) (kv.Block, error) {
		block, err := stateSource.Load(index)
		if err != nil {
			return kv.Block{}, err
		}
		return s.kvBlockFromStateBlock(stateSource, block, opts)
	}
	return source, nil
}

// RangeKVBlocks streams root KV snapshot blocks from the resident native K/V
// cache. CaptureOptions.BlockStartToken skips whole blocks ending at or before
// the trusted boundary, mirroring the root State-block sleep lane.
func (s *ArchSession) RangeKVBlocks(blockSize int, opts kv.CaptureOptions, yield func(kv.Block) (bool, error)) error {
	if yield == nil {
		return core.NewError("native.RangeKVBlocks: nil yield")
	}
	source, err := s.KVBlockSource(blockSize, opts)
	if err != nil {
		return err
	}
	for i := 0; i < source.BlockCount; i++ {
		block, err := source.Load(i)
		if err != nil {
			return err
		}
		ok, err := yield(block)
		if err != nil || !ok {
			return err
		}
	}
	return nil
}

// RestoreKV restores a root KV snapshot into the resident native cache. It
// accepts native BF16 layer slabs directly and falls back to per-head float32
// tensors by converting them once into the native BF16 slab layout.
func (s *ArchSession) RestoreKV(snapshot *kv.Snapshot) error {
	if s == nil {
		return core.NewError("native.RestoreKV: nil session")
	}
	// A bf16 snapshot restored into a q8 session lands its rows through the
	// bf16 mirror and requantises them here (today's behaviour, honestly lossy).
	// A q8-NATIVE snapshot instead writes the int8 codes straight into the store
	// (bit-exact) and sets rawQ8Restore, so this flush is skipped — flushing
	// would clobber the just-written codes with a requantise of the stale mirror.
	var rawQ8Restore bool
	if s.state.icb != nil {
		defer func() {
			if !rawQ8Restore {
				s.state.icb.flushQ8Mirrors(s.pos) // requantise the restored rows (pos set by the restore)
			}
		}()
	}
	if snapshot == nil {
		return core.NewError("native.RestoreKV: nil snapshot")
	}
	if snapshot.Version > kv.SnapshotVersion {
		return core.NewError("native.RestoreKV: unsupported snapshot version")
	}
	position := kv.EffectiveTokenOffset(snapshot)
	if position <= 0 || position > s.maxLen {
		return core.NewError("native.RestoreKV: position outside maxLen")
	}
	if snapshot.NumLayers > 0 && snapshot.NumLayers != len(s.state.specs) {
		return core.NewError("native.RestoreKV: layer count mismatch")
	}
	targetViews, err := s.stateLayerViews()
	if err != nil {
		return err
	}
	if len(targetViews) > 0 && len(snapshot.Layers) == 0 {
		return core.NewError("native.RestoreKV: snapshot has no layers")
	}
	for _, view := range targetViews {
		layer, ok := nativeKVSnapshotLayer(snapshot, view.layer)
		if !ok {
			return core.NewError("native.RestoreKV: missing layer")
		}
		if err := nativeKVValidateLayerMetadata("native.RestoreKV", layer, view); err != nil {
			return err
		}
		wantTokens := position
		if view.maxSize > 0 && position > view.cacheRows {
			wantTokens = view.cacheRows
		}
		if nativeKVLayerIsQ8Native(layer) {
			tokenCount, kvd, err := nativeKVQ8NativeWindow(layer, view)
			if err != nil {
				return err
			}
			if tokenCount != wantTokens {
				return core.NewError("native.RestoreKV: q8-native layer window length mismatch")
			}
			start := position - tokenCount
			if s.state.icb != nil && s.state.icb.kvQ8.on(view.layer) {
				// q8 target: land the captured int8 codes + scales verbatim.
				if err := s.state.icb.restoreQ8LayerRaw(view.layer, start, tokenCount, layer.KeyBytes, layer.ValueBytes); err != nil {
					return err
				}
				rawQ8Restore = true
				continue
			}
			// Non-q8 target: a bf16 (or other) store cannot hold int8 codes, so
			// dequantise the q8 block into bf16 token rows. Documented lossy — the
			// bytes match a live q8 read, but the store keeps the bf16 view.
			keyRows, err := q8NativeToTokenRowsBF16(layer.KeyBytes, tokenCount, kvd)
			if err != nil {
				return err
			}
			valueRows, err := q8NativeToTokenRowsBF16(layer.ValueBytes, tokenCount, kvd)
			if err != nil {
				return err
			}
			block := SessionStateLayerBlock{
				Layer:      view.layer,
				CacheIndex: view.cacheIndex,
				CacheMode:  view.cacheMode,
				MaxSize:    view.maxSize,
				KVHeads:    view.kvHeads,
				HeadDim:    view.headDim,
				RowBytes:   view.rowBytes,
				KeyBytes:   keyRows,
				ValueBytes: valueRows,
			}
			if err := restoreStateBlockLayer(view, start, tokenCount, position, block); err != nil {
				return err
			}
			continue
		}
		if keySlab, valueSlab, tokenCount, ok, err := nativeKVLayerSnapshotDirectBF16Slabs("native.RestoreKV", layer, view); err != nil {
			return err
		} else if ok {
			if tokenCount != wantTokens {
				return core.NewError("native.RestoreKV: layer window length mismatch")
			}
			if err := restoreNativeKVLayerSlabs("native.RestoreKV", view, position-tokenCount, tokenCount, position, keySlab, valueSlab); err != nil {
				return err
			}
			continue
		}
		keySlab, valueSlab, tokenCount, err := nativeKVLayerSnapshotSlabs(layer, view)
		if err != nil {
			return err
		}
		if tokenCount != wantTokens {
			return core.NewError("native.RestoreKV: layer window length mismatch")
		}
		keyRows := make([]byte, tokenCount*view.rowBytes)
		valueRows := make([]byte, tokenCount*view.rowBytes)
		nativeKVLayerSlabToTokenRows(keyRows, keySlab, tokenCount, view.kvHeads, view.headDim)
		nativeKVLayerSlabToTokenRows(valueRows, valueSlab, tokenCount, view.kvHeads, view.headDim)
		block := SessionStateLayerBlock{
			Layer:      view.layer,
			CacheIndex: view.cacheIndex,
			CacheMode:  view.cacheMode,
			MaxSize:    view.maxSize,
			KVHeads:    view.kvHeads,
			HeadDim:    view.headDim,
			RowBytes:   view.rowBytes,
			KeyBytes:   keyRows,
			ValueBytes: valueRows,
		}
		if err := restoreStateBlockLayer(view, position-tokenCount, tokenCount, position, block); err != nil {
			return err
		}
	}
	if err := s.reloadPagedStateLayerViews(position, targetViews); err != nil {
		return err
	}
	if err := s.restoreKVSnapshotMetadata(snapshot, position); err != nil {
		return err
	}
	// #1846 wake landing: a woken q8 session lands its appended turns through the
	// position-invariant per-token lane, so a resident prefix + appended suffix is
	// byte-identical to a fresh whole prefill (the batched append is tile-position
	// sensitive and the q8 store amplifies that wobble into token flips). Auto-
	// armed here — the restore is the engine-owned seam — so a woken session is
	// answer-stable without the caller opting in. Appends are short turns, so the
	// per-token landing there is decode-speed and negligible.
	if s.state.icb != nil && s.state.icb.hasKVQ8() {
		s.reuseCanonicalLanding = true
	}
	return nil
}

func nativeKVLayerSnapshotDirectBF16Slabs(scope string, layer kv.LayerSnapshot, view sessionStateLayerView) ([]byte, []byte, int, bool, error) {
	if len(layer.TurboQuantPayloads) > 0 || len(layer.KeyBytes) == 0 || len(layer.ValueBytes) == 0 {
		return nil, nil, 0, false, nil
	}
	if !nativeKVIsBF16DType(layer.KeyDType) || !nativeKVIsBF16DType(layer.ValueDType) {
		return nil, nil, 0, false, nil
	}
	keySlab, keySeq, err := nativeKVLayerRawSlabBF16(layer.KeyBytes, layer.KeyDType, layer.KeyShape, view)
	if err != nil {
		return nil, nil, 0, true, core.E(scope, "native layer key", err)
	}
	valueSlab, valueSeq, err := nativeKVLayerRawSlabBF16(layer.ValueBytes, layer.ValueDType, layer.ValueShape, view)
	if err != nil {
		return nil, nil, 0, true, core.E(scope, "native layer value", err)
	}
	if keySeq != valueSeq {
		return nil, nil, 0, true, core.NewError(scope + ": layer key/value window mismatch")
	}
	return keySlab, valueSlab, keySeq, true, nil
}

func restoreNativeKVLayerSlabs(scope string, view sessionStateLayerView, start, tokenCount, position int, keySlab, valueSlab []byte) error {
	if view.rowBytes <= 0 || view.cacheRows <= 0 || view.kvHeads <= 0 || view.headDim <= 0 {
		return core.NewError(scope + ": invalid layer view geometry")
	}
	headBytes := view.headDim * bf16Size
	want := view.kvHeads * tokenCount * headBytes
	if tokenCount <= 0 || len(keySlab) != want || len(valueSlab) != want {
		return core.NewError(scope + ": layer slab size mismatch")
	}
	if view.maxSize > 0 && position > view.cacheRows {
		windowStart := position - view.cacheRows
		if start < windowStart {
			return core.NewError(scope + ": layer starts before sliding cache window")
		}
	} else {
		off := start * view.rowBytes
		n := tokenCount * view.rowBytes
		if off < 0 || off+n > len(view.keyBytes) || off+n > len(view.valueBytes) {
			return core.NewError(scope + ": layer exceeds cache rows")
		}
	}
	for token := range tokenCount {
		slot := start + token
		if view.maxSize > 0 && position > view.cacheRows {
			slot %= view.cacheRows
		}
		dstRow := slot * view.rowBytes
		for head := 0; head < view.kvHeads; head++ {
			src := (head*tokenCount + token) * headBytes
			dst := dstRow + head*headBytes
			if dst < 0 || dst+headBytes > len(view.keyBytes) || dst+headBytes > len(view.valueBytes) {
				return core.NewError(scope + ": layer exceeds cache rows")
			}
			copy(view.keyBytes[dst:dst+headBytes], keySlab[src:src+headBytes])
			copy(view.valueBytes[dst:dst+headBytes], valueSlab[src:src+headBytes])
		}
	}
	return nil
}

// RestoreKVBlocks restores root KV snapshot blocks directly into the resident
// native cache. It avoids assembling the blocks into a monolithic CPU snapshot
// before writing cache rows.
// RestoreKVBlockSource adapts the engine-neutral streamed block source
// (kv.BlockSource — the durable-store wake path) onto RestoreKVBlocks. The
// neutral source is the untrusted from-disk shape: no trusted prefix, no
// resident ids, no retained logits, blocks indexed from zero. This is the
// seam that lets a raw-q8 block bundle wake bit-exactly — the CPU-assembled
// snapshot fallback cannot carry kv.KVNativeDTypeQ8 payloads.
func (s *ArchSession) RestoreKVBlockSource(ctx context.Context, source kv.BlockSource) error {
	if ctx == nil {
		ctx = context.Background()
	}
	return s.RestoreKVBlocks(KVBlockSource{
		TokenCount:   source.TokenCount,
		PrefixTokens: source.PrefixTokens,
		BlockCount:   source.BlockCount,
		Load: func(index int) (kv.Block, error) {
			if err := ctx.Err(); err != nil {
				return kv.Block{}, err
			}
			return source.Load(ctx, index)
		},
	})
}

func (s *ArchSession) RestoreKVBlocks(source KVBlockSource) error {
	// A raw q8-native block landing writes int8 codes straight into the store; the
	// restore target's stateLayerViews already materialised the bf16 mirror, so
	// flushing (requantising the stale mirror into the codes) would clobber the
	// just-written codes — skip it for those layers (#1846 block lane). A bf16
	// block into a q8 store still lands through the mirror and needs the flush.
	var rawQ8Restore bool
	if s != nil && s.state.icb != nil {
		defer func() {
			if !rawQ8Restore {
				s.state.icb.flushQ8Mirrors(s.pos) // requantise the restored rows (pos set by the restore)
			}
		}()
	}
	if s == nil {
		return core.NewError("native.RestoreKVBlocks: nil session")
	}
	if source.TokenCount <= 0 || source.TokenCount > s.maxLen {
		return core.NewError("native.RestoreKVBlocks: token count outside maxLen")
	}
	if source.BlockCount < 0 {
		return core.NewError("native.RestoreKVBlocks: negative block count")
	}
	if source.BlockCount > 0 && source.Load == nil {
		return core.NewError("native.RestoreKVBlocks: nil block loader")
	}
	prefixTokens := source.PrefixTokens
	if prefixTokens <= 0 {
		prefixTokens = source.TokenCount
	}
	if prefixTokens <= 0 || prefixTokens > source.TokenCount || prefixTokens > s.maxLen {
		return core.NewError("native.RestoreKVBlocks: prefix tokens outside token count")
	}
	trustedPrefix := source.TrustedPrefixTokens
	if trustedPrefix < 0 || trustedPrefix > prefixTokens {
		return core.NewError("native.RestoreKVBlocks: trusted prefix outside token count")
	}
	if trustedPrefix > 0 {
		if err := s.validateKVBlockTrustedPrefix(source, trustedPrefix); err != nil {
			return err
		}
	}
	if source.BlockCount == 0 {
		if trustedPrefix != prefixTokens {
			return core.NewError("native.RestoreKVBlocks: empty block source")
		}
		return s.restoreTrustedKVBlockMetadata(source, prefixTokens)
	}
	if stateSource, ok, err := source.nativeRestoreStateSource(prefixTokens); err != nil {
		return err
	} else if ok {
		if err := s.RestoreStateBlocks(stateSource); err != nil {
			return core.E("native.RestoreKVBlocks", "native state source", err)
		}
		s.armCanonicalLandingForKVQ8() // woken q8 append lands position-invariant (#1846)
		return nil
	}
	targetViews, err := s.stateLayerViews()
	if err != nil {
		return err
	}
	cachedIDs := s.kvBlockCachedIDScratch(prefixTokens)
	if trustedPrefix > 0 {
		cachedIDs = append(cachedIDs, s.cachedIDs[:trustedPrefix]...)
	}
	expectedStart := trustedPrefix
	expectedIndex := source.FirstBlockIndex
	var finalSnapshot *kv.Snapshot
	for i := 0; i < source.BlockCount && expectedStart < prefixTokens; i++ {
		block, err := source.Load(i)
		if err != nil {
			return err
		}
		if block.Snapshot == nil {
			return core.NewError("native.RestoreKVBlocks: nil block snapshot")
		}
		if (source.FirstBlockIndex > 0 || trustedPrefix == 0) && block.Index != expectedIndex+i {
			return core.NewError("native.RestoreKVBlocks: block index mismatch")
		}
		if block.TokenStart != expectedStart {
			return core.NewError("native.RestoreKVBlocks: block token start mismatch")
		}
		if block.TokenCount <= 0 {
			return core.NewError("native.RestoreKVBlocks: invalid block token range")
		}
		if block.TokenStart+block.TokenCount > prefixTokens {
			trimCount := prefixTokens - block.TokenStart
			if trimCount <= 0 {
				return core.NewError("native.RestoreKVBlocks: invalid block token range")
			}
			if nativeKVSnapshotHasTurboQuantPayload(block.Snapshot) {
				if err := s.restoreKVSnapshotBlockLayersPrefix(block, trimCount, prefixTokens, targetViews); err != nil {
					return core.E("native.RestoreKVBlocks", "restore prefix block", err)
				}
				cachedIDs = append(cachedIDs, block.Snapshot.Tokens[:trimCount]...)
				expectedStart += trimCount
				continue
			}
			trimmed, err := nativeKVSliceBlockPrefix(block.Snapshot, trimCount, block.TokenStart)
			if err != nil {
				return core.E("native.RestoreKVBlocks", "slice prefix block", err)
			}
			block.TokenCount = trimCount
			block.Snapshot = trimmed
		}
		if block.TokenStart+block.TokenCount > prefixTokens {
			return core.NewError("native.RestoreKVBlocks: invalid block token range")
		}
		if block.Snapshot.SeqLen != 0 && block.Snapshot.SeqLen != block.TokenCount {
			return core.NewError("native.RestoreKVBlocks: block seq length mismatch")
		}
		if kv.EffectiveTokenOffset(block.Snapshot) != block.TokenStart+block.TokenCount {
			return core.NewError("native.RestoreKVBlocks: block token offset mismatch")
		}
		if len(block.Snapshot.Tokens) != block.TokenCount {
			return core.NewError("native.RestoreKVBlocks: block token count mismatch")
		}
		landedRawQ8, err := s.restoreKVSnapshotBlockLayers(block, prefixTokens, targetViews)
		if err != nil {
			return err
		}
		if landedRawQ8 {
			rawQ8Restore = true
		}
		cachedIDs = append(cachedIDs, block.Snapshot.Tokens...)
		expectedStart += block.TokenCount
		finalSnapshot = block.Snapshot
	}
	if expectedStart != prefixTokens {
		return core.NewError("native.RestoreKVBlocks: block coverage does not match token count")
	}
	var generated []int32
	var logitShape []int32
	var logits []float32
	if finalSnapshot != nil && prefixTokens == source.TokenCount {
		generated = finalSnapshot.Generated
		logitShape = finalSnapshot.LogitShape
		logits = finalSnapshot.Logits
	}
	if err := s.reloadPagedStateLayerViews(prefixTokens, targetViews); err != nil {
		return err
	}
	s.armCanonicalLandingForKVQ8() // woken q8 append lands position-invariant (#1846)
	if prefixTokens == source.TokenCount && len(source.RetainedLogits) > 0 {
		return s.restoreKVBlockMetadataRetainedLogits(cachedIDs, generated, source.RetainedLogits, prefixTokens)
	}
	return s.restoreKVBlockMetadata(cachedIDs, generated, logitShape, logits, prefixTokens)
}

func (s *ArchSession) kvBlockCachedIDScratch(n int) []int32 {
	if s == nil {
		return nil
	}
	if cap(s.kvBlockCachedIDs) < n {
		s.kvBlockCachedIDs = make([]int32, 0, n)
	}
	return s.kvBlockCachedIDs[:0]
}

func (source KVBlockSource) nativeRestoreStateSource(prefixTokens int) (SessionStateBlockSource, bool, error) {
	if source.nativeStateSource == nil {
		return SessionStateBlockSource{}, false, nil
	}
	state := *source.nativeStateSource
	if source.FirstBlockIndex != state.firstBlockIndex || source.BlockCount != state.BlockCount || source.TrustedPrefixTokens != state.trustedPrefixTokens() {
		return SessionStateBlockSource{}, false, nil
	}
	if prefixTokens < 0 || prefixTokens > state.Position {
		return SessionStateBlockSource{}, false, core.NewError("native.RestoreKVBlocks: prefix tokens outside native source")
	}
	if len(state.CachedIDs) < prefixTokens {
		return SessionStateBlockSource{}, false, core.NewError("native.RestoreKVBlocks: native source ids missing")
	}
	state.Position = prefixTokens
	state.CachedIDs = state.CachedIDs[:prefixTokens]
	state.CachedPromptIDs = nil
	state.CachedPromptHidden = nil
	state.CachedPromptLogits = nil
	state.RetainedHidden = nil
	if prefixTokens != source.TokenCount {
		state.RetainedLogits = nil
	}
	if err := nativeKVTrimStateRestoreBlocks(&state, prefixTokens); err != nil {
		return SessionStateBlockSource{}, false, err
	}
	return state, true, nil
}

func nativeKVTrimStateRestoreBlocks(source *SessionStateBlockSource, prefixTokens int) error {
	if source == nil {
		return core.NewError("native.RestoreKVBlocks: nil native state source")
	}
	if len(source.blockBoundaries) <= 1 {
		return nil
	}
	if source.firstBlockIndex < 0 || source.firstBlockIndex >= len(source.blockBoundaries) {
		return core.NewError("native.RestoreKVBlocks: native block index outside boundaries")
	}
	endBoundary := source.firstBlockIndex
	for endBoundary < len(source.blockBoundaries) && source.blockBoundaries[endBoundary] < prefixTokens {
		endBoundary++
	}
	if endBoundary >= len(source.blockBoundaries) {
		return core.NewError("native.RestoreKVBlocks: native prefix outside block boundaries")
	}
	if source.blockBoundaries[endBoundary] != prefixTokens {
		if endBoundary == source.firstBlockIndex {
			return core.NewError("native.RestoreKVBlocks: native prefix before first block")
		}
		boundaries := append(source.blockBoundaries[:endBoundary:endBoundary], prefixTokens)
		source.blockBoundaries = boundaries
	} else {
		source.blockBoundaries = source.blockBoundaries[:endBoundary+1]
	}
	source.BlockCount = endBoundary - source.firstBlockIndex
	source.totalBlockCount = len(source.blockBoundaries) - 1
	return nil
}

func nativeKVSliceBlockPrefix(snapshot *kv.Snapshot, tokenCount, baseOffset int) (*kv.Snapshot, error) {
	return snapshot.SliceBlock(0, tokenCount, baseOffset, false)
}

func nativeKVSnapshotHasTurboQuantPayload(snapshot *kv.Snapshot) bool {
	if snapshot == nil {
		return false
	}
	for _, layer := range snapshot.Layers {
		if len(layer.TurboQuantPayloads) > 0 {
			return true
		}
	}
	return false
}

func nativeKVLayerSnapshotPrefixSlabs(layer kv.LayerSnapshot, view sessionStateLayerView, tokenCount int) ([]byte, []byte, error) {
	if len(layer.TurboQuantPayloads) > 0 {
		keyPrefix, valuePrefix, seqLen, err := nativeTurboQuantKVLayerPrefixSlabs(layer.TurboQuantPayloads, view, tokenCount)
		if err != nil {
			return nil, nil, err
		}
		if seqLen != tokenCount {
			return nil, nil, core.NewError("native.RestoreKVBlocks: turboquant prefix length mismatch")
		}
		return keyPrefix, valuePrefix, nil
	}
	keySlab, valueSlab, seqLen, err := nativeKVLayerSnapshotSlabs(layer, view)
	if err != nil {
		return nil, nil, err
	}
	if tokenCount > seqLen {
		return nil, nil, core.NewError("native.RestoreKVBlocks: compressed prefix outside layer window")
	}
	keyPrefix, err := nativeKVLayerSlabPrefix(keySlab, seqLen, tokenCount, view.kvHeads, view.headDim)
	if err != nil {
		return nil, nil, core.E("native.RestoreKVBlocks", "slice compressed key prefix", err)
	}
	valuePrefix, err := nativeKVLayerSlabPrefix(valueSlab, seqLen, tokenCount, view.kvHeads, view.headDim)
	if err != nil {
		return nil, nil, core.E("native.RestoreKVBlocks", "slice compressed value prefix", err)
	}
	return keyPrefix, valuePrefix, nil
}

func nativeKVLayerSlabPrefix(src []byte, seqLen, tokenCount, heads, headDim int) ([]byte, error) {
	if tokenCount <= 0 || tokenCount > seqLen || heads <= 0 || headDim <= 0 {
		return nil, core.NewError("native.RestoreKVBlocks: invalid layer slab prefix geometry")
	}
	rowBytes := headDim * bf16Size
	if len(src) != heads*seqLen*rowBytes {
		return nil, core.NewError("native.RestoreKVBlocks: layer slab prefix size mismatch")
	}
	out := make([]byte, heads*tokenCount*rowBytes)
	for head := range heads {
		srcStart := head * seqLen * rowBytes
		srcEnd := srcStart + tokenCount*rowBytes
		dstStart := head * tokenCount * rowBytes
		copy(out[dstStart:dstStart+tokenCount*rowBytes], src[srcStart:srcEnd])
	}
	return out, nil
}

func (s *ArchSession) restoreTrustedKVBlockMetadata(source KVBlockSource, prefixTokens int) error {
	if len(source.RetainedLogits) > 0 && len(source.RetainedLogits) != s.arch.Vocab*bf16Size {
		return core.NewError("native.RestoreKVBlocks: retained logits size mismatch")
	}
	if len(source.RetainedLogits) > 0 && prefixTokens == source.TokenCount {
		return s.restoreKVBlockMetadataRetainedLogits(s.cachedIDs[:prefixTokens], nil, source.RetainedLogits, prefixTokens)
	}
	metadata := &kv.Snapshot{
		Tokens:      append([]int32(nil), s.cachedIDs[:prefixTokens]...),
		TokenOffset: prefixTokens,
	}
	return s.restoreKVSnapshotMetadata(metadata, prefixTokens)
}

func (s *ArchSession) validateKVBlockTrustedPrefix(source KVBlockSource, trustedPrefix int) error {
	if s.pos < trustedPrefix {
		return core.NewError("native.RestoreKVBlocks: trusted prefix not resident")
	}
	if len(s.cachedIDs) < trustedPrefix {
		return core.NewError("native.RestoreKVBlocks: trusted prefix resident ids missing")
	}
	if len(source.CachedIDs) < trustedPrefix {
		return core.NewError("native.RestoreKVBlocks: trusted prefix source ids missing")
	}
	for i := range trustedPrefix {
		if s.cachedIDs[i] != source.CachedIDs[i] {
			return core.NewError("native.RestoreKVBlocks: trusted prefix ids mismatch")
		}
	}
	return nil
}

func (s *ArchSession) kvSnapshotLayerMetadata() []kv.LayerSnapshot {
	layers := make([]kv.LayerSnapshot, len(s.state.specs))
	for li, spec := range s.state.specs {
		layers[li] = kv.LayerSnapshot{
			Layer:      li,
			CacheIndex: spec.CacheIndex,
			CacheMode:  nativeStateCacheModeFixed,
			MaxSize:    s.stateCacheMaxSize(spec),
		}
	}
	return layers
}

func (s *ArchSession) kvBlockFromStateBlock(source SessionStateBlockSource, block SessionStateBlock, opts kv.CaptureOptions) (kv.Block, error) {
	if block.TokenCount <= 0 {
		return kv.Block{}, core.NewError("native.KVBlockSource: empty block")
	}
	end := block.TokenStart + block.TokenCount
	if block.TokenStart < 0 || end > source.Position {
		return kv.Block{}, core.NewError("native.KVBlockSource: block outside position")
	}
	if len(source.CachedIDs) < end {
		return kv.Block{}, core.NewError("native.KVBlockSource: cached ids do not cover block")
	}
	layers := s.kvSnapshotLayerMetadata()
	for _, layerBlock := range block.Layers {
		if layerBlock.Layer < 0 || layerBlock.Layer >= len(layers) {
			return kv.Block{}, core.NewError("native.KVBlockSource: invalid block layer")
		}
		layer, err := nativeKVLayerBlockSnapshot(layerBlock, block.TokenCount, opts.RawKVOnly)
		if err != nil {
			return kv.Block{}, err
		}
		// q8-armed layers carry the store's RAW int8 codes + f32 scales verbatim
		// under kv.KVNativeDTypeQ8 so a q8→q8 block sleep/wake is bit-exact — the
		// bf16 slab above double-quantises on restore, perturbing every prefix row
		// (#1846 block lane, mirroring the snapshot lane). Gated on RawKVOnly:
		// that is the native (uncompressed) lane whose block is wire-encoded
		// as-is, so the raw dtype is carried. The non-native lane keeps its
		// dequantised per-head float32 (TurboQuant / kv.Analyze / blocks_save
		// re-quantise it) — the wire codec cannot carry a raw tensor alongside a
		// non-native encoding (errRawTensorNeedsNative), and RawKVOnly is only
		// set for EncodingNative (SaveKVBlocksToState), so the two never collide.
		if opts.RawKVOnly && s.state.icb != nil && s.state.icb.kvQ8.on(layerBlock.Layer) {
			kPacked, vPacked, err := s.state.icb.captureQ8LayerRaw(layerBlock.Layer, block.TokenStart, block.TokenCount)
			if err != nil {
				return kv.Block{}, err
			}
			layer.KeyDType = kv.KVNativeDTypeQ8
			layer.KeyBytes = kPacked
			layer.ValueDType = kv.KVNativeDTypeQ8
			layer.ValueBytes = vPacked
		}
		layers[layerBlock.Layer] = layer
	}
	snapshot := &kv.Snapshot{
		Version:       kv.SnapshotVersion,
		Tokens:        append([]int32(nil), source.CachedIDs[block.TokenStart:end]...),
		TokenOffset:   end,
		NumLayers:     len(s.state.specs),
		NumHeads:      s.arch.MaxKVHeads(),
		SeqLen:        block.TokenCount,
		HeadDim:       s.arch.MaxHeadDim(),
		NumQueryHeads: s.arch.Heads,
		Layers:        layers,
	}
	if end == source.Position && len(source.RetainedLogits) > 0 {
		if len(source.RetainedLogits) != s.arch.Vocab*bf16Size {
			return kv.Block{}, core.NewError("native.KVBlockSource: retained logits size mismatch")
		}
		snapshot.LogitShape = []int32{1, int32(s.arch.Vocab)}
		snapshot.Logits = bf16ToF32Slice(source.RetainedLogits)
	}
	return kv.Block{
		Index:      block.Index,
		TokenStart: block.TokenStart,
		TokenCount: block.TokenCount,
		Snapshot:   snapshot,
	}, nil
}

func nativeKVLayerBlockSnapshot(block SessionStateLayerBlock, tokenCount int, rawOnly bool) (kv.LayerSnapshot, error) {
	layer := kv.LayerSnapshot{
		Layer:      block.Layer,
		CacheIndex: block.CacheIndex,
		CacheMode:  block.CacheMode,
		MaxSize:    block.MaxSize,
	}
	if len(block.KeyBytes) == 0 && len(block.ValueBytes) == 0 {
		return layer, nil
	}
	if block.KVHeads <= 0 || block.HeadDim <= 0 || block.RowBytes <= 0 {
		return kv.LayerSnapshot{}, core.NewError("native.KVBlockSource: invalid layer geometry")
	}
	if len(block.KeyBytes) != tokenCount*block.RowBytes || len(block.ValueBytes) != tokenCount*block.RowBytes {
		return kv.LayerSnapshot{}, core.NewError("native.KVBlockSource: layer payload size mismatch")
	}
	keySlab := make([]byte, len(block.KeyBytes))
	valueSlab := make([]byte, len(block.ValueBytes))
	nativeKVTokenRowsToLayerSlab(keySlab, block.KeyBytes, tokenCount, block.KVHeads, block.HeadDim)
	nativeKVTokenRowsToLayerSlab(valueSlab, block.ValueBytes, tokenCount, block.KVHeads, block.HeadDim)
	if rawOnly {
		// Native (fast) path: keep the layer-level bf16 slab so restore reads it
		// directly (nativeKVLayerSnapshotSlabs' raw branch).
		shape := []int32{1, int32(block.KVHeads), int32(tokenCount), int32(block.HeadDim)}
		layer.KeyDType = nativeKVSnapshotDTypeBF16
		layer.KeyBytes = keySlab
		layer.KeyShape = append([]int32(nil), shape...)
		layer.ValueDType = nativeKVSnapshotDTypeBF16
		layer.ValueBytes = valueSlab
		layer.ValueShape = append([]int32(nil), shape...)
		return layer, nil
	}
	// Non-native encoding (q8/float32): emit ONLY the per-head float32 tensors.
	// The layer-level raw bf16 slab cannot encode as anything but native (the
	// snapshot encoder rejects a raw tensor under q8/float32 with
	// errRawTensorNeedsNative), while the per-head float32 quantises cleanly; and
	// dropping the redundant bf16 copy makes the q8 store actually smaller. Restore
	// falls to nativeKVHeadSnapshotSlabs when the layer-level slab is absent.
	layer.Heads = nativeKVLayerSlabHeads(keySlab, valueSlab, tokenCount, block.KVHeads, block.HeadDim)
	return layer, nil
}

// restoreKVSnapshotBlockLayers lands one block's layers into the resident cache
// and reports landedRawQ8 = at least one layer took the bit-exact q8-native path
// (int8 codes written straight into a q8 store), so the caller skips the mirror
// requantise flush that would otherwise clobber those codes (#1846 block lane).
func (s *ArchSession) restoreKVSnapshotBlockLayers(block kv.Block, position int, targetViews []sessionStateLayerView) (landedRawQ8 bool, err error) {
	for _, view := range targetViews {
		layer, ok := nativeKVSnapshotLayer(block.Snapshot, view.layer)
		if !ok {
			return landedRawQ8, core.NewError("native.RestoreKVBlocks: missing layer")
		}
		if err := nativeKVValidateLayerMetadata("native.RestoreKVBlocks", layer, view); err != nil {
			return landedRawQ8, err
		}
		if nativeKVLayerIsQ8Native(layer) {
			raw, err := s.restoreQ8NativeBlockLayer(layer, view, block.TokenStart, block.TokenCount, position)
			if err != nil {
				return landedRawQ8, err
			}
			if raw {
				landedRawQ8 = true
			}
			continue
		}
		if !nativeKVLayerHasPayload(layer) {
			empty := SessionStateLayerBlock{
				Layer:      view.layer,
				CacheIndex: view.cacheIndex,
				CacheMode:  view.cacheMode,
				MaxSize:    view.maxSize,
				KVHeads:    view.kvHeads,
				HeadDim:    view.headDim,
				RowBytes:   view.rowBytes,
			}
			if err := restoreStateBlockLayer(view, block.TokenStart, block.TokenCount, position, empty); err != nil {
				return landedRawQ8, err
			}
			continue
		}
		if keySlab, valueSlab, tokenCount, ok, err := nativeKVLayerSnapshotDirectBF16Slabs("native.RestoreKVBlocks", layer, view); err != nil {
			return landedRawQ8, err
		} else if ok {
			if tokenCount != block.TokenCount {
				return landedRawQ8, core.NewError("native.RestoreKVBlocks: layer window length mismatch")
			}
			if err := restoreNativeKVLayerSlabs("native.RestoreKVBlocks", view, block.TokenStart, tokenCount, position, keySlab, valueSlab); err != nil {
				return landedRawQ8, err
			}
			continue
		}
		var keyRows, valueRows []byte
		var tokenCount int
		if len(layer.TurboQuantPayloads) > 0 {
			if ok, err := s.restoreTurboQuantKVLayerRowsInto(view, block.TokenStart, block.TokenCount, position, layer.TurboQuantPayloads, 0); err != nil {
				return landedRawQ8, err
			} else if ok {
				continue
			}
			var err error
			keyRows, valueRows, tokenCount, err = nativeTurboQuantKVLayerRows(layer.TurboQuantPayloads, view)
			if err != nil {
				return landedRawQ8, err
			}
		} else {
			keySlab, valueSlab, seqLen, err := nativeKVLayerSnapshotSlabs(layer, view)
			if err != nil {
				return landedRawQ8, err
			}
			tokenCount = seqLen
			keyRows = make([]byte, tokenCount*view.rowBytes)
			valueRows = make([]byte, tokenCount*view.rowBytes)
			nativeKVLayerSlabToTokenRows(keyRows, keySlab, tokenCount, view.kvHeads, view.headDim)
			nativeKVLayerSlabToTokenRows(valueRows, valueSlab, tokenCount, view.kvHeads, view.headDim)
		}
		if tokenCount != block.TokenCount {
			return landedRawQ8, core.NewError("native.RestoreKVBlocks: layer window length mismatch")
		}
		layerBlock := SessionStateLayerBlock{
			Layer:      view.layer,
			CacheIndex: view.cacheIndex,
			CacheMode:  view.cacheMode,
			MaxSize:    view.maxSize,
			KVHeads:    view.kvHeads,
			HeadDim:    view.headDim,
			RowBytes:   view.rowBytes,
			KeyBytes:   keyRows,
			ValueBytes: valueRows,
		}
		if err := restoreStateBlockLayer(view, block.TokenStart, block.TokenCount, position, layerBlock); err != nil {
			return landedRawQ8, err
		}
	}
	return landedRawQ8, nil
}

func (s *ArchSession) restoreKVSnapshotBlockLayersPrefix(block kv.Block, tokenCount, position int, targetViews []sessionStateLayerView) error {
	if block.Snapshot == nil {
		return core.NewError("native.RestoreKVBlocks: nil block snapshot")
	}
	if tokenCount <= 0 || tokenCount > block.TokenCount || tokenCount > len(block.Snapshot.Tokens) {
		return core.NewError("native.RestoreKVBlocks: invalid compressed prefix range")
	}
	if block.Snapshot.SeqLen != 0 && block.Snapshot.SeqLen < tokenCount {
		return core.NewError("native.RestoreKVBlocks: block seq length mismatch")
	}
	for _, view := range targetViews {
		layer, ok := nativeKVSnapshotLayer(block.Snapshot, view.layer)
		if !ok {
			return core.NewError("native.RestoreKVBlocks: missing layer")
		}
		if err := nativeKVValidateLayerMetadata("native.RestoreKVBlocks", layer, view); err != nil {
			return err
		}
		if !nativeKVLayerHasPayload(layer) {
			empty := SessionStateLayerBlock{
				Layer:      view.layer,
				CacheIndex: view.cacheIndex,
				CacheMode:  view.cacheMode,
				MaxSize:    view.maxSize,
				KVHeads:    view.kvHeads,
				HeadDim:    view.headDim,
				RowBytes:   view.rowBytes,
			}
			if err := restoreStateBlockLayer(view, block.TokenStart, tokenCount, position, empty); err != nil {
				return err
			}
			continue
		}
		var keyRows, valueRows []byte
		if len(layer.TurboQuantPayloads) > 0 {
			if ok, err := s.restoreTurboQuantKVLayerRowsInto(view, block.TokenStart, tokenCount, position, layer.TurboQuantPayloads, tokenCount); err != nil {
				return err
			} else if ok {
				continue
			}
			var seqLen int
			var err error
			keyRows, valueRows, seqLen, err = nativeTurboQuantKVLayerPrefixRows(layer.TurboQuantPayloads, view, tokenCount)
			if err != nil {
				return err
			}
			if seqLen != tokenCount {
				return core.NewError("native.RestoreKVBlocks: turboquant prefix length mismatch")
			}
		} else {
			keySlab, valueSlab, err := nativeKVLayerSnapshotPrefixSlabs(layer, view, tokenCount)
			if err != nil {
				return err
			}
			keyRows = make([]byte, tokenCount*view.rowBytes)
			valueRows = make([]byte, tokenCount*view.rowBytes)
			nativeKVLayerSlabToTokenRows(keyRows, keySlab, tokenCount, view.kvHeads, view.headDim)
			nativeKVLayerSlabToTokenRows(valueRows, valueSlab, tokenCount, view.kvHeads, view.headDim)
		}
		layerBlock := SessionStateLayerBlock{
			Layer:      view.layer,
			CacheIndex: view.cacheIndex,
			CacheMode:  view.cacheMode,
			MaxSize:    view.maxSize,
			KVHeads:    view.kvHeads,
			HeadDim:    view.headDim,
			RowBytes:   view.rowBytes,
			KeyBytes:   keyRows,
			ValueBytes: valueRows,
		}
		if err := restoreStateBlockLayer(view, block.TokenStart, tokenCount, position, layerBlock); err != nil {
			return err
		}
	}
	return nil
}

func (s *ArchSession) restoreTurboQuantKVLayerRowsInto(view sessionStateLayerView, start, tokenCount, position int, payloads [][]byte, prefixTokens int) (bool, error) {
	keyRows, valueRows, ok, err := nativeKVResidentLayerRows(view, start, tokenCount, position)
	if err != nil || !ok {
		return ok, err
	}
	rotated, normalised := s.turboQuantKVDecodeScratch(view.headDim)
	parsed, err := s.turboQuantKVPayloads(payloads, view)
	if err != nil {
		return true, err
	}
	seqLen, err := nativeTurboQuantKVLayerPayloadsRowsIntoScratch(parsed, view, prefixTokens, keyRows, valueRows, rotated, normalised)
	if err != nil {
		return true, err
	}
	if seqLen != tokenCount {
		return true, core.NewError("native.RestoreKVBlocks: turboquant layer window length mismatch")
	}
	return true, nil
}

func (s *ArchSession) turboQuantKVDecodeScratch(headDim int) ([]float64, []float64) {
	if s == nil || headDim <= 0 {
		return nil, nil
	}
	if cap(s.turboQuantRotated) < headDim {
		s.turboQuantRotated = make([]float64, headDim)
	} else {
		s.turboQuantRotated = s.turboQuantRotated[:headDim]
	}
	if cap(s.turboQuantNormed) < headDim {
		s.turboQuantNormed = make([]float64, headDim)
	} else {
		s.turboQuantNormed = s.turboQuantNormed[:headDim]
	}
	return s.turboQuantRotated, s.turboQuantNormed
}

func nativeKVResidentLayerRows(view sessionStateLayerView, start, tokenCount, position int) ([]byte, []byte, bool, error) {
	if view.rowBytes <= 0 || view.cacheRows <= 0 {
		return nil, nil, false, core.NewError("native.RestoreKVBlocks: invalid layer view geometry")
	}
	if tokenCount <= 0 {
		return nil, nil, false, core.NewError("native.RestoreKVBlocks: invalid layer token count")
	}
	n := tokenCount * view.rowBytes
	if view.maxSize <= 0 || position <= view.cacheRows {
		off := start * view.rowBytes
		if off < 0 || off+n > len(view.keyBytes) || off+n > len(view.valueBytes) {
			return nil, nil, false, core.NewError("native.RestoreKVBlocks: layer exceeds cache rows")
		}
		return view.keyBytes[off : off+n], view.valueBytes[off : off+n], true, nil
	}
	windowStart := position - view.cacheRows
	if start+tokenCount <= windowStart || start < windowStart {
		return nil, nil, false, nil
	}
	slot := start % view.cacheRows
	if slot+tokenCount > view.cacheRows {
		return nil, nil, false, nil
	}
	off := slot * view.rowBytes
	if off < 0 || off+n > len(view.keyBytes) || off+n > len(view.valueBytes) {
		return nil, nil, false, core.NewError("native.RestoreKVBlocks: sliding layer exceeds cache rows")
	}
	return view.keyBytes[off : off+n], view.valueBytes[off : off+n], true, nil
}

func (s *ArchSession) captureKVLogits() ([]float32, []int32, error) {
	var logits []byte
	switch {
	case len(s.retainedLogits) == s.arch.Vocab*bf16Size:
		logits = s.retainedLogits
	case len(s.retainedHidden) == s.arch.Hidden*bf16Size:
		var err error
		logits, err = s.BoundaryLogits()
		if err != nil {
			return nil, nil, err
		}
	default:
		return nil, nil, nil
	}
	if len(logits) == 0 {
		return nil, nil, nil
	}
	if len(logits) != s.arch.Vocab*bf16Size {
		return nil, nil, core.NewError("native.CaptureKV: boundary logits size mismatch")
	}
	return bf16ToF32Slice(logits), []int32{1, int32(s.arch.Vocab)}, nil
}

func (s *ArchSession) restoreKVSnapshotMetadata(snapshot *kv.Snapshot, position int) error {
	cachedIDs := s.kvBlockCachedIDScratch(position)
	cachedIDs = append(cachedIDs, snapshot.Tokens...)
	return s.restoreKVBlockMetadata(cachedIDs, snapshot.Generated, snapshot.LogitShape, snapshot.Logits, position)
}

func (s *ArchSession) restoreKVBlockMetadata(cachedIDs, generated []int32, logitShape []int32, logits []float32, position int) error {
	if len(generated) > 0 && len(cachedIDs)+len(generated) <= position {
		cachedIDs = append(cachedIDs, generated...)
	}
	if len(cachedIDs) > position {
		return core.NewError("native.RestoreKV: cached ids exceed position")
	}
	s.pos = position
	if err := s.truncateSpeculativeKV(s.pos); err != nil {
		return err
	}
	s.cachedIDs = append(s.cachedIDs[:0], cachedIDs...)
	s.clearCachedPromptHidden()
	s.resetRetainedHidden()
	if len(logits) == 0 {
		return nil
	}
	if len(logitShape) > 0 {
		total := 1
		for _, dim := range logitShape {
			if dim <= 0 {
				return core.NewError("native.RestoreKV: invalid logit shape")
			}
			total *= int(dim)
		}
		if total != len(logits) {
			return core.NewError("native.RestoreKV: logit shape mismatch")
		}
	}
	if len(logits) != s.arch.Vocab {
		return core.NewError("native.RestoreKV: logits size mismatch")
	}
	s.rememberRetainedLogits(f32ToBf16Slice(logits))
	return nil
}

func (s *ArchSession) restoreKVBlockMetadataRetainedLogits(cachedIDs, generated []int32, retainedLogits []byte, position int) error {
	if len(retainedLogits) != s.arch.Vocab*bf16Size {
		return core.NewError("native.RestoreKVBlocks: retained logits size mismatch")
	}
	if err := s.restoreKVBlockMetadata(cachedIDs, generated, nil, nil, position); err != nil {
		return err
	}
	s.rememberRetainedLogits(retainedLogits)
	return nil
}

func nativeKVSnapshotLayer(snapshot *kv.Snapshot, layerIndex int) (kv.LayerSnapshot, bool) {
	if snapshot == nil || layerIndex < 0 {
		return kv.LayerSnapshot{}, false
	}
	if layerIndex < len(snapshot.Layers) {
		layer := snapshot.Layers[layerIndex]
		if layer.Layer == layerIndex {
			return layer, true
		}
	}
	for _, layer := range snapshot.Layers {
		if layer.Layer == layerIndex {
			return layer, true
		}
	}
	return kv.LayerSnapshot{}, false
}

func nativeKVLayerCaptureWindow(view sessionStateLayerView, position int) (int, int, error) {
	if view.rowBytes <= 0 || view.cacheRows <= 0 {
		return 0, 0, core.NewError("native.CaptureKV: invalid layer view geometry")
	}
	if position <= 0 {
		return 0, 0, core.NewError("native.CaptureKV: empty cache")
	}
	tokenCount := position
	if view.maxSize > 0 && position > view.cacheRows {
		tokenCount = view.cacheRows
	}
	return position - tokenCount, tokenCount, nil
}

func nativeKVLayerSnapshotSlabs(layer kv.LayerSnapshot, view sessionStateLayerView) ([]byte, []byte, int, error) {
	if len(layer.TurboQuantPayloads) > 0 {
		return nativeTurboQuantKVLayerSlabs(layer.TurboQuantPayloads, view)
	}
	if len(layer.KeyBytes) > 0 || len(layer.ValueBytes) > 0 {
		keySlab, keySeq, err := nativeKVLayerRawSlabBF16(layer.KeyBytes, layer.KeyDType, layer.KeyShape, view)
		if err != nil {
			return nil, nil, 0, core.E("native.RestoreKV", "native layer key", err)
		}
		valueSlab, valueSeq, err := nativeKVLayerRawSlabBF16(layer.ValueBytes, layer.ValueDType, layer.ValueShape, view)
		if err != nil {
			return nil, nil, 0, core.E("native.RestoreKV", "native layer value", err)
		}
		if keySeq != valueSeq {
			return nil, nil, 0, core.NewError("native.RestoreKV: layer key/value window mismatch")
		}
		return keySlab, valueSlab, keySeq, nil
	}
	return nativeKVHeadSnapshotSlabs(layer, view)
}

func nativeKVValidateLayerMetadata(scope string, layer kv.LayerSnapshot, view sessionStateLayerView) error {
	if layer.CacheIndex >= 0 && layer.CacheIndex != view.cacheIndex {
		return core.NewError(scope + ": cache-index mismatch")
	}
	if layer.CacheMode != "" && view.cacheMode != "" && layer.CacheMode != view.cacheMode && !nativeKVRestorableSourceCacheMode(layer.CacheMode) {
		return core.NewError(scope + ": cache-mode mismatch")
	}
	if layer.MaxSize > 0 && layer.MaxSize != view.maxSize && !nativeKVRestorableSourceMaxSize(layer) {
		return core.NewError(scope + ": cache max-size mismatch")
	}
	return nil
}

func nativeKVRestorableSourceMaxSize(layer kv.LayerSnapshot) bool {
	return layer.CacheMode != "" && nativeKVRestorableSourceCacheMode(layer.CacheMode)
}

func nativeKVRestorableSourceCacheMode(mode string) bool {
	switch mode {
	case "", "fp16", "q8", "k-q8-v-q4", "paged", nativeStateCacheModeFixed, "turboquant", "rotating", "sliding":
		return true
	default:
		return false
	}
}

func nativeKVLayerHasPayload(layer kv.LayerSnapshot) bool {
	if len(layer.TurboQuantPayloads) > 0 || len(layer.KeyBytes) > 0 || len(layer.ValueBytes) > 0 || len(layer.Heads) > 0 {
		return true
	}
	return false
}

func nativeKVLayerRawSlabBF16(raw []byte, dtype string, shape []int32, view sessionStateLayerView) ([]byte, int, error) {
	if len(raw) == 0 || len(shape) != 4 {
		return nil, 0, core.NewError("missing native slab")
	}
	_, bytesPerValue, ok := nativeKVRawDType(dtype)
	if !ok {
		return nil, 0, core.NewError("unsupported native dtype")
	}
	if shape[0] != 1 || int(shape[1]) != view.kvHeads || int(shape[3]) != view.headDim {
		return nil, 0, core.NewError("native slab shape mismatch")
	}
	tokenCount := int(shape[2])
	if tokenCount <= 0 {
		return nil, 0, core.NewError("native slab token count invalid")
	}
	elements := tokenCount * view.kvHeads * view.headDim
	if len(raw) != elements*bytesPerValue {
		return nil, 0, core.NewError("native slab byte length mismatch")
	}
	if nativeKVIsBF16DType(dtype) {
		return raw, tokenCount, nil
	}
	out := make([]byte, elements*bf16Size)
	if err := nativeKVRawToBF16(out, raw, dtype); err != nil {
		return nil, 0, err
	}
	return out, tokenCount, nil
}

func nativeKVHeadSnapshotSlabs(layer kv.LayerSnapshot, view sessionStateLayerView) ([]byte, []byte, int, error) {
	if len(layer.Heads) != view.kvHeads {
		return nil, nil, 0, core.NewError("native.RestoreKV: head count mismatch")
	}
	tokenCount := 0
	for _, head := range layer.Heads {
		keySeq, err := nativeKVHeadSnapshotSeqLen(head.Key, head.KeyBytes, head.KeyDType, view.headDim)
		if err != nil {
			return nil, nil, 0, core.E("native.RestoreKV", "head key", err)
		}
		valueSeq, err := nativeKVHeadSnapshotSeqLen(head.Value, head.ValueBytes, head.ValueDType, view.headDim)
		if err != nil {
			return nil, nil, 0, core.E("native.RestoreKV", "head value", err)
		}
		if keySeq != valueSeq {
			return nil, nil, 0, core.NewError("native.RestoreKV: head key/value window mismatch")
		}
		if tokenCount == 0 {
			tokenCount = keySeq
			continue
		}
		if keySeq != tokenCount {
			return nil, nil, 0, core.NewError("native.RestoreKV: head window length mismatch")
		}
	}
	if tokenCount <= 0 {
		return nil, nil, 0, core.NewError("native.RestoreKV: missing head payload")
	}
	keySlab := make([]byte, view.kvHeads*tokenCount*view.headDim*bf16Size)
	valueSlab := make([]byte, len(keySlab))
	for headIndex, head := range layer.Heads {
		headOff := headIndex * tokenCount * view.headDim * bf16Size
		if err := nativeKVFillHeadBF16(keySlab[headOff:headOff+tokenCount*view.headDim*bf16Size], head.Key, head.KeyBytes, head.KeyDType, tokenCount, view.headDim); err != nil {
			return nil, nil, 0, core.E("native.RestoreKV", "head key", err)
		}
		if err := nativeKVFillHeadBF16(valueSlab[headOff:headOff+tokenCount*view.headDim*bf16Size], head.Value, head.ValueBytes, head.ValueDType, tokenCount, view.headDim); err != nil {
			return nil, nil, 0, core.E("native.RestoreKV", "head value", err)
		}
	}
	return keySlab, valueSlab, tokenCount, nil
}

func nativeKVHeadSnapshotSeqLen(values []float32, raw []byte, dtype string, headDim int) (int, error) {
	if headDim <= 0 {
		return 0, core.NewError("invalid head dim")
	}
	if len(raw) > 0 {
		_, bytesPerValue, ok := nativeKVRawDType(dtype)
		if !ok {
			return 0, core.NewError("unsupported head raw dtype")
		}
		rowBytes := headDim * bytesPerValue
		if len(raw)%rowBytes != 0 {
			return 0, core.NewError("head raw byte length mismatch")
		}
		return len(raw) / rowBytes, nil
	}
	if len(values) == 0 {
		return 0, core.NewError("missing head tensor")
	}
	if len(values)%headDim != 0 {
		return 0, core.NewError("head tensor length mismatch")
	}
	return len(values) / headDim, nil
}

func nativeKVFillHeadBF16(dst []byte, values []float32, raw []byte, dtype string, tokenCount, headDim int) error {
	want := tokenCount * headDim * bf16Size
	if len(dst) != want {
		return core.NewError("native.RestoreKV: destination size mismatch")
	}
	if len(raw) > 0 {
		_, bytesPerValue, ok := nativeKVRawDType(dtype)
		if !ok || len(raw) != tokenCount*headDim*bytesPerValue {
			return core.NewError("native.RestoreKV: raw head payload mismatch")
		}
		return nativeKVRawToBF16(dst, raw, dtype)
	}
	if len(values) != tokenCount*headDim {
		return core.NewError("native.RestoreKV: float32 head payload mismatch")
	}
	for i, v := range values {
		h := f32ToBF16(v)
		dst[i*bf16Size], dst[i*bf16Size+1] = byte(h), byte(h>>8)
	}
	return nil
}

func nativeKVLayerSlabHeads(keySlab, valueSlab []byte, tokenCount, heads, headDim int) []kv.HeadSnapshot {
	if tokenCount <= 0 || heads <= 0 || headDim <= 0 {
		return nil
	}
	headBytes := tokenCount * headDim * bf16Size
	out := make([]kv.HeadSnapshot, heads)
	for head := range heads {
		off := head * headBytes
		out[head] = kv.HeadSnapshot{
			Key:   bf16ToF32Slice(keySlab[off : off+headBytes]),
			Value: bf16ToF32Slice(valueSlab[off : off+headBytes]),
		}
	}
	return out
}

func nativeKVTokenRowsToLayerSlab(dst, src []byte, tokenCount, heads, headDim int) {
	rowBytes := heads * headDim * bf16Size
	headBytes := headDim * bf16Size
	for head := range heads {
		for token := range tokenCount {
			srcOff := token*rowBytes + head*headBytes
			dstOff := (head*tokenCount + token) * headBytes
			copy(dst[dstOff:dstOff+headBytes], src[srcOff:srcOff+headBytes])
		}
	}
}

func nativeKVLayerSlabToTokenRows(dst, src []byte, tokenCount, heads, headDim int) {
	rowBytes := heads * headDim * bf16Size
	headBytes := headDim * bf16Size
	for token := range tokenCount {
		for head := range heads {
			srcOff := (head*tokenCount + token) * headBytes
			dstOff := token*rowBytes + head*headBytes
			copy(dst[dstOff:dstOff+headBytes], src[srcOff:srcOff+headBytes])
		}
	}
}

func nativeKVIsBF16DType(dtype string) bool {
	canonical, _, ok := nativeKVRawDType(dtype)
	return ok && canonical == nativeKVSnapshotDTypeBF16
}

func nativeKVRawDType(dtype string) (string, int, bool) {
	switch {
	case nativeKVASCIIEqualFold(dtype, "bfloat16") || nativeKVASCIIEqualFold(dtype, "bf16"):
		return nativeKVSnapshotDTypeBF16, bf16Size, true
	case nativeKVASCIIEqualFold(dtype, "float16") || nativeKVASCIIEqualFold(dtype, "f16"):
		return "float16", 2, true
	case nativeKVASCIIEqualFold(dtype, "float32") || nativeKVASCIIEqualFold(dtype, "f32"):
		return "float32", 4, true
	default:
		return "", 0, false
	}
}

func nativeKVASCIIEqualFold(got, want string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		g := got[i]
		w := want[i]
		if 'A' <= g && g <= 'Z' {
			g += 'a' - 'A'
		}
		if g != w {
			return false
		}
	}
	return true
}

func nativeKVRawToBF16(dst, raw []byte, dtype string) error {
	canonical, bytesPerValue, ok := nativeKVRawDType(dtype)
	if !ok || len(dst)%bf16Size != 0 || len(raw) != len(dst)/bf16Size*bytesPerValue {
		return core.NewError("native.RestoreKV: raw payload size mismatch")
	}
	switch canonical {
	case nativeKVSnapshotDTypeBF16:
		copy(dst, raw)
	case "float16":
		for i := 0; i < len(dst)/bf16Size; i++ {
			v := safetensors.Float16ToFloat32(binary.LittleEndian.Uint16(raw[i*2 : i*2+2]))
			h := f32ToBF16(v)
			dst[i*bf16Size], dst[i*bf16Size+1] = byte(h), byte(h>>8)
		}
	case "float32":
		for i := 0; i < len(dst)/bf16Size; i++ {
			v := math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4 : i*4+4]))
			h := f32ToBF16(v)
			dst[i*bf16Size], dst[i*bf16Size+1] = byte(h), byte(h>>8)
		}
	default:
		return core.NewError("native.RestoreKV: unsupported raw dtype")
	}
	return nil
}
