// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

const nativeStateCacheModeFixed = "fixed"

// SessionStateLayerBlock is one layer's K/V cache bytes for a contiguous token
// range. KeyBytes and ValueBytes are views into the session's resident Metal
// buffers when produced by StateBlockSource or RangeStateBlocks; callers must
// consume or copy them before mutating/closing the source session.
type SessionStateLayerBlock struct {
	Layer      int
	CacheIndex int
	CacheMode  string
	MaxSize    int
	KVHeads    int
	HeadDim    int
	RowBytes   int
	KeyBytes   []byte
	ValueBytes []byte
}

// SessionStateBlock is a contiguous token range from the native session state.
type SessionStateBlock struct {
	Index      int
	TokenStart int
	TokenCount int
	Layers     []SessionStateLayerBlock
}

// SessionStateBlockSource streams native session state blocks without first
// assembling a monolithic SerializeState blob. CachedPromptHidden,
// CachedPromptLogits, RetainedHidden, and RetainedLogits borrow the source
// session's boundary buffers; consume or copy them before mutating/closing the
// source session.
type SessionStateBlockSource struct {
	Position           int
	CachedIDs          []int32
	CachedPromptIDs    []int32
	CachedPromptHidden []byte
	CachedPromptLogits []byte
	RetainedHidden     []byte
	RetainedLogits     []byte
	BlockCount         int
	Load               func(int) (SessionStateBlock, error)
	blockSize          int
	firstBlockIndex    int
	totalBlockCount    int
	trustedPrefix      int
	blockBoundaries    []int
	views              []sessionStateLayerView
}

// TrustPrefixBlocks records that this source intentionally skips whole leading
// blocks already resident in the target session. RestoreStateBlocks validates
// the resident token IDs before grafting the suffix blocks.
func (source *SessionStateBlockSource) TrustPrefixBlocks(blockSize, firstBlockIndex int) error {
	if source == nil {
		return core.NewError("native.SessionStateBlockSource: nil source")
	}
	if blockSize <= 0 {
		return core.NewError("native.SessionStateBlockSource: block size must be > 0")
	}
	if firstBlockIndex < 0 {
		return core.NewError("native.SessionStateBlockSource: first block index must be >= 0")
	}
	if firstBlockIndex == 0 {
		source.blockSize = 0
		source.firstBlockIndex = 0
		source.totalBlockCount = source.BlockCount
		source.trustedPrefix = 0
		return nil
	}
	trustedPrefix := firstBlockIndex * blockSize
	if err := source.TrustPrefixTokens(trustedPrefix, firstBlockIndex); err != nil {
		return err
	}
	source.blockSize = blockSize
	return nil
}

// TrustPrefixTokens records an exact skipped token prefix for block streams
// whose absolute block indexes are not a uniform blockSize grid.
func (source *SessionStateBlockSource) TrustPrefixTokens(trustedPrefix, firstBlockIndex int) error {
	if source == nil {
		return core.NewError("native.SessionStateBlockSource: nil source")
	}
	if trustedPrefix < 0 {
		return core.NewError("native.SessionStateBlockSource: trusted prefix must be >= 0")
	}
	if trustedPrefix > source.Position {
		return core.NewError("native.SessionStateBlockSource: trusted prefix outside position")
	}
	if firstBlockIndex < 0 {
		return core.NewError("native.SessionStateBlockSource: first block index must be >= 0")
	}
	if trustedPrefix == 0 {
		source.blockSize = 0
		source.firstBlockIndex = 0
		source.totalBlockCount = source.BlockCount
		source.trustedPrefix = 0
		return nil
	}
	if firstBlockIndex == 0 {
		return core.NewError("native.SessionStateBlockSource: first block index must be > 0 for trusted prefix")
	}
	source.blockSize = 0
	source.firstBlockIndex = firstBlockIndex
	source.totalBlockCount = firstBlockIndex + source.BlockCount
	source.trustedPrefix = trustedPrefix
	return nil
}

type sessionStateLayerView struct {
	layer      int
	kvHeads    int
	headDim    int
	rowBytes   int
	cacheIndex int
	cacheMode  string
	maxSize    int
	cacheRows  int
	keyBytes   []byte
	valueBytes []byte
	paged      *devicePagedKVCache
}

// StateBlockSource returns a block loader over the current resident K/V cache.
// K/V payload slices returned by Load are zero-copy views into this session.
func (s *ArchSession) StateBlockSource(blockSize int) (SessionStateBlockSource, error) {
	return s.StateBlockSourceFrom(0, blockSize)
}

// StateBlockSourceFrom is StateBlockSource with metal-style trusted-prefix
// sleep: full blocks ending at or before startToken are skipped, but yielded
// block indexes remain absolute in the original block grid.
func (s *ArchSession) StateBlockSourceFrom(startToken, blockSize int) (SessionStateBlockSource, error) {
	blockCount, firstBlock, totalBlocks, boundaries, views, err := s.stateBlockPlan(startToken, blockSize)
	if err != nil {
		return SessionStateBlockSource{}, err
	}
	retainedLogits := s.retainedLogits
	if len(retainedLogits) == 0 && len(s.retainedHidden) == s.arch.Hidden*bf16Size {
		var err error
		retainedLogits, err = s.BoundaryLogits()
		if err != nil {
			return SessionStateBlockSource{}, err
		}
	}
	sourceBoundaries := append([]int(nil), boundaries...)
	source := SessionStateBlockSource{
		Position:           s.pos,
		CachedIDs:          append([]int32(nil), s.cachedIDs...),
		CachedPromptIDs:    append([]int32(nil), s.cachedPromptIDs...),
		CachedPromptHidden: s.cachedPromptHidden,
		CachedPromptLogits: s.cachedPromptLogits,
		RetainedHidden:     s.retainedHidden,
		RetainedLogits:     retainedLogits,
		BlockCount:         blockCount,
		blockSize:          blockSize,
		firstBlockIndex:    firstBlock,
		totalBlockCount:    totalBlocks,
		blockBoundaries:    sourceBoundaries,
		views:              views,
	}
	source.Load = func(index int) (SessionStateBlock, error) {
		return loadStateBlockFromBoundaries(firstBlock+index, sourceBoundaries, source.Position, views)
	}
	return source, nil
}

// RangeStateBlocks visits native session-state blocks in order. It is the
// native analogue of metal's ranged K/V capture, but it stays CGO-free and uses
// ArchSession's resident buffers directly. The yielded block and its layer
// descriptors are only valid until the callback returns.
func (s *ArchSession) RangeStateBlocks(blockSize int, yield func(SessionStateBlock) (bool, error)) error {
	return s.RangeStateBlocksFrom(0, blockSize, yield)
}

// RangeStateBlocksFrom visits native session-state blocks after startToken.
func (s *ArchSession) RangeStateBlocksFrom(startToken, blockSize int, yield func(SessionStateBlock) (bool, error)) error {
	if yield == nil {
		return core.NewError("native.RangeStateBlocks: nil yield")
	}
	blockCount, firstBlock, _, boundaries, views, err := s.stateBlockPlan(startToken, blockSize)
	if err != nil {
		return err
	}
	layers := s.stateBlockLayerScratch(len(views))
	for i := range blockCount {
		block, err := fillStateBlockFromBoundaries(firstBlock+i, boundaries, s.pos, views, layers)
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

// RestoreStateBlocks restores a session from streamed native state blocks. It
// copies only the current block's K/V range into resident buffers and restores
// the small prompt/retained metadata needed for GenerateFromCache and prefix
// reuse.
func (s *ArchSession) RestoreStateBlocks(source SessionStateBlockSource) error {
	if s == nil {
		return core.NewError("native.RestoreStateBlocks: nil session")
	}
	if source.Position < 0 || source.Position > s.maxLen {
		return core.NewError("native.RestoreStateBlocks: position outside maxLen")
	}
	if len(source.CachedIDs) > source.Position {
		return core.NewError("native.RestoreStateBlocks: cached ids exceed position")
	}
	if source.BlockCount < 0 {
		return core.NewError("native.RestoreStateBlocks: negative block count")
	}
	if source.BlockCount > 0 && source.Load == nil {
		return core.NewError("native.RestoreStateBlocks: nil block loader")
	}
	if source.Position == 0 && source.BlockCount != 0 {
		return core.NewError("native.RestoreStateBlocks: zero-position source has blocks")
	}
	trustedPrefix := source.trustedPrefixTokens()
	if source.Position > 0 && source.BlockCount == 0 && trustedPrefix != source.Position {
		return core.NewError("native.RestoreStateBlocks: non-empty source has no blocks")
	}
	if trustedPrefix > 0 {
		if err := s.validateStateBlockTrustedPrefix(source, trustedPrefix); err != nil {
			return err
		}
	}
	if source.BlockCount == 0 {
		return s.restoreStateBlockMetadata(source)
	}
	targetViews, err := s.stateLayerViews()
	if err != nil {
		return err
	}
	ownerCount := len(targetViews)
	sourceLayers := s.stateBlockLayerScratch(ownerCount)
	expectedStart := trustedPrefix
	expectedIndex := source.firstBlockIndex
	for i := 0; i < source.BlockCount; i++ {
		block, err := source.loadInto(i, sourceLayers)
		if err != nil {
			return err
		}
		if err := restoreStateBlock(expectedIndex+i, expectedStart, source.Position, ownerCount, targetViews, block); err != nil {
			return err
		}
		expectedStart += block.TokenCount
	}
	if expectedStart != source.Position {
		return core.NewError("native.RestoreStateBlocks: block coverage does not match position")
	}
	if err := s.restoreStateBlockMetadata(source); err != nil {
		return err
	}
	// ICB sessions: the block copies above already landed in the live ICB cache
	// buffers (the views wrap them directly); the paged caches are dormant, so
	// re-uploading the slabs into pages would only write a store decode never
	// reads. Paged sessions still need the upload — their views are a host
	// snapshot, not the live pages.
	if s.state.icb == nil {
		if err := s.reloadPagedStateLayerViews(source.Position, targetViews); err != nil {
			return err
		}
	}
	s.restoredKV = true // restored K/V: appends take the token path (decode-parity carve-out)
	return nil
}

func (source SessionStateBlockSource) trustedPrefixTokens() int {
	if source.trustedPrefix > 0 {
		if source.trustedPrefix > source.Position {
			return source.Position
		}
		return source.trustedPrefix
	}
	if source.blockSize <= 0 || source.firstBlockIndex <= 0 {
		return 0
	}
	if len(source.blockBoundaries) > source.firstBlockIndex {
		prefix := source.blockBoundaries[source.firstBlockIndex]
		if prefix > source.Position {
			return source.Position
		}
		return prefix
	}
	prefix := source.firstBlockIndex * source.blockSize
	if prefix > source.Position {
		return source.Position
	}
	return prefix
}

func (s *ArchSession) validateStateBlockTrustedPrefix(source SessionStateBlockSource, trustedPrefix int) error {
	if trustedPrefix < 0 || trustedPrefix > source.Position {
		return core.NewError("native.RestoreStateBlocks: trusted prefix outside position")
	}
	if s.pos < trustedPrefix {
		return core.NewError("native.RestoreStateBlocks: trusted prefix not resident")
	}
	if len(source.CachedIDs) < trustedPrefix {
		return core.NewError("native.RestoreStateBlocks: trusted prefix source ids missing")
	}
	if len(s.cachedIDs) < trustedPrefix {
		return core.NewError("native.RestoreStateBlocks: trusted prefix resident ids missing")
	}
	for i := range trustedPrefix {
		if s.cachedIDs[i] != source.CachedIDs[i] {
			return core.NewError("native.RestoreStateBlocks: trusted prefix ids mismatch")
		}
	}
	return nil
}

func (source SessionStateBlockSource) loadInto(index int, layers []SessionStateLayerBlock) (SessionStateBlock, error) {
	if len(source.views) > 0 && len(source.blockBoundaries) > 1 {
		return fillStateBlockFromBoundaries(source.firstBlockIndex+index, source.blockBoundaries, source.Position, source.views, layers)
	}
	if len(source.views) > 0 && source.blockSize > 0 {
		return fillStateBlock(source.firstBlockIndex+index, source.blockSize, source.totalBlockCount, source.Position, source.views, layers)
	}
	return source.Load(index)
}

func loadStateBlock(index, blockSize, blockCount, position int, views []sessionStateLayerView) (SessionStateBlock, error) {
	layers := make([]SessionStateLayerBlock, len(views))
	return fillStateBlock(index, blockSize, blockCount, position, views, layers)
}

func loadStateBlockFromBoundaries(index int, boundaries []int, position int, views []sessionStateLayerView) (SessionStateBlock, error) {
	layers := make([]SessionStateLayerBlock, len(views))
	return fillStateBlockFromBoundaries(index, boundaries, position, views, layers)
}

func fillStateBlock(index, blockSize, blockCount, position int, views []sessionStateLayerView, layers []SessionStateLayerBlock) (SessionStateBlock, error) {
	if index < 0 || index >= blockCount {
		return SessionStateBlock{}, core.NewError("native.StateBlockSource.Load: block index out of range")
	}
	start := index * blockSize
	if start >= position {
		return SessionStateBlock{}, core.NewError("native.StateBlockSource.Load: block start outside position")
	}
	end := min(start+blockSize, position)
	return fillStateBlockSpan(index, start, end, position, views, layers)
}

func fillStateBlockFromBoundaries(index int, boundaries []int, position int, views []sessionStateLayerView, layers []SessionStateLayerBlock) (SessionStateBlock, error) {
	if len(boundaries) < 2 {
		return SessionStateBlock{}, core.NewError("native.StateBlockSource.Load: invalid block boundaries")
	}
	if index < 0 || index >= len(boundaries)-1 {
		return SessionStateBlock{}, core.NewError("native.StateBlockSource.Load: block index out of range")
	}
	start := boundaries[index]
	end := boundaries[index+1]
	return fillStateBlockSpan(index, start, end, position, views, layers)
}

func fillStateBlockSpan(index, start, end, position int, views []sessionStateLayerView, layers []SessionStateLayerBlock) (SessionStateBlock, error) {
	if start < 0 || end <= start || end > position {
		return SessionStateBlock{}, core.NewError("native.StateBlockSource.Load: invalid block range")
	}
	tokenCount := end - start
	if len(layers) != len(views) {
		return SessionStateBlock{}, core.NewError("native.StateBlockSource.Load: layer descriptor size mismatch")
	}
	for i, view := range views {
		keyBytes, valueBytes, err := stateBlockLayerBytes(view, start, tokenCount, position)
		if err != nil {
			return SessionStateBlock{}, err
		}
		layers[i] = SessionStateLayerBlock{
			Layer:      view.layer,
			CacheIndex: view.cacheIndex,
			CacheMode:  view.cacheMode,
			MaxSize:    view.maxSize,
			KVHeads:    view.kvHeads,
			HeadDim:    view.headDim,
			RowBytes:   view.rowBytes,
			KeyBytes:   keyBytes,
			ValueBytes: valueBytes,
		}
	}
	return SessionStateBlock{Index: index, TokenStart: start, TokenCount: tokenCount, Layers: layers}, nil
}

func stateBlockLayerBytes(view sessionStateLayerView, start, tokenCount, position int) ([]byte, []byte, error) {
	if view.rowBytes <= 0 || view.cacheRows <= 0 {
		return nil, nil, core.NewError("native.StateBlockSource.Load: invalid layer view geometry")
	}
	n := tokenCount * view.rowBytes
	if view.maxSize <= 0 || position <= view.cacheRows {
		off := start * view.rowBytes
		if off < 0 || off+n > len(view.keyBytes) || off+n > len(view.valueBytes) {
			return nil, nil, core.NewError("native.StateBlockSource.Load: block exceeds cache rows")
		}
		return view.keyBytes[off : off+n], view.valueBytes[off : off+n], nil
	}
	windowStart := position - view.cacheRows
	if start+tokenCount <= windowStart {
		return nil, nil, nil
	}
	if start < windowStart {
		return nil, nil, core.NewError("native.StateBlockSource.Load: block starts before sliding cache window")
	}
	slot := start % view.cacheRows
	if slot+tokenCount <= view.cacheRows {
		off := slot * view.rowBytes
		if off < 0 || off+n > len(view.keyBytes) || off+n > len(view.valueBytes) {
			return nil, nil, core.NewError("native.StateBlockSource.Load: sliding block exceeds cache rows")
		}
		return view.keyBytes[off : off+n], view.valueBytes[off : off+n], nil
	}
	keyBytes := make([]byte, n)
	valueBytes := make([]byte, n)
	for t := range tokenCount {
		slot := (start + t) % view.cacheRows
		src := slot * view.rowBytes
		dst := t * view.rowBytes
		if src < 0 || src+view.rowBytes > len(view.keyBytes) || src+view.rowBytes > len(view.valueBytes) {
			return nil, nil, core.NewError("native.StateBlockSource.Load: sliding block exceeds cache rows")
		}
		copy(keyBytes[dst:dst+view.rowBytes], view.keyBytes[src:src+view.rowBytes])
		copy(valueBytes[dst:dst+view.rowBytes], view.valueBytes[src:src+view.rowBytes])
	}
	return keyBytes, valueBytes, nil
}

func restoreStateBlock(index, expectedStart, position, ownerCount int, targetViews []sessionStateLayerView, block SessionStateBlock) error {
	if block.Index != index {
		return core.NewError("native.RestoreStateBlocks: block index mismatch")
	}
	if block.TokenStart != expectedStart {
		return core.NewError("native.RestoreStateBlocks: block token start mismatch")
	}
	if block.TokenCount <= 0 {
		return core.NewError("native.RestoreStateBlocks: empty block")
	}
	if block.TokenStart+block.TokenCount > position {
		return core.NewError("native.RestoreStateBlocks: block exceeds position")
	}
	if len(block.Layers) != ownerCount {
		return core.NewError("native.RestoreStateBlocks: block layer count mismatch")
	}
	var seenStack [128]bool
	seen := seenStack[:]
	if len(targetViews) > len(seenStack) {
		seen = make([]bool, len(targetViews))
	} else {
		seen = seen[:len(targetViews)]
	}
	for _, layer := range block.Layers {
		viewIndex := -1
		for i, view := range targetViews {
			if view.layer == layer.Layer {
				viewIndex = i
				break
			}
		}
		if viewIndex < 0 {
			return core.NewError("native.RestoreStateBlocks: invalid block layer")
		}
		if seen[viewIndex] {
			return core.NewError("native.RestoreStateBlocks: duplicate block layer")
		}
		seen[viewIndex] = true
		view := targetViews[viewIndex]
		if layer.KVHeads > 0 && layer.KVHeads != view.kvHeads {
			return core.NewError("native.RestoreStateBlocks: kv-head count mismatch")
		}
		if layer.HeadDim > 0 && layer.HeadDim != view.headDim {
			return core.NewError("native.RestoreStateBlocks: head-dim mismatch")
		}
		if layer.CacheMode != "" && view.cacheMode != "" && layer.CacheMode != view.cacheMode && !nativeKVRestorableSourceCacheMode(layer.CacheMode) {
			return core.NewError("native.RestoreStateBlocks: cache-mode mismatch")
		}
		if layer.MaxSize > 0 && layer.MaxSize != view.maxSize && !nativeKVRestorableStateSourceMaxSize(layer) {
			return core.NewError("native.RestoreStateBlocks: cache max-size mismatch")
		}
		if layer.RowBytes != view.rowBytes {
			return core.NewError("native.RestoreStateBlocks: row-byte mismatch")
		}
		if err := restoreStateBlockLayer(view, block.TokenStart, block.TokenCount, position, layer); err != nil {
			return err
		}
	}
	return nil
}

func nativeKVRestorableStateSourceMaxSize(layer SessionStateLayerBlock) bool {
	return layer.CacheMode != "" && nativeKVRestorableSourceCacheMode(layer.CacheMode)
}

func restoreStateBlockLayer(view sessionStateLayerView, start, tokenCount, position int, layer SessionStateLayerBlock) error {
	if view.rowBytes <= 0 || view.cacheRows <= 0 {
		return core.NewError("native.RestoreStateBlocks: invalid layer view geometry")
	}
	if view.maxSize > 0 && position > view.cacheRows {
		windowStart := position - view.cacheRows
		if start+tokenCount <= windowStart {
			if len(layer.KeyBytes) == 0 && len(layer.ValueBytes) == 0 {
				return nil
			}
			return core.NewError("native.RestoreStateBlocks: expired sliding block has KV payload")
		}
	}
	n := tokenCount * view.rowBytes
	if len(layer.KeyBytes) != n || len(layer.ValueBytes) != n {
		return core.NewError("native.RestoreStateBlocks: block payload size mismatch")
	}
	if view.maxSize <= 0 || position <= view.cacheRows {
		off := start * view.rowBytes
		if off < 0 || off+n > len(view.keyBytes) || off+n > len(view.valueBytes) {
			return core.NewError("native.RestoreStateBlocks: block exceeds cache rows")
		}
		copy(view.keyBytes[off:off+n], layer.KeyBytes)
		copy(view.valueBytes[off:off+n], layer.ValueBytes)
		return nil
	}
	if start < position-view.cacheRows {
		return core.NewError("native.RestoreStateBlocks: block starts before sliding cache window")
	}
	for t := range tokenCount {
		slot := (start + t) % view.cacheRows
		dst := slot * view.rowBytes
		src := t * view.rowBytes
		if dst < 0 || dst+view.rowBytes > len(view.keyBytes) || dst+view.rowBytes > len(view.valueBytes) {
			return core.NewError("native.RestoreStateBlocks: sliding block exceeds cache rows")
		}
		if src+view.rowBytes > len(layer.KeyBytes) || src+view.rowBytes > len(layer.ValueBytes) {
			return core.NewError("native.RestoreStateBlocks: block payload size mismatch")
		}
		copy(view.keyBytes[dst:dst+view.rowBytes], layer.KeyBytes[src:src+view.rowBytes])
		copy(view.valueBytes[dst:dst+view.rowBytes], layer.ValueBytes[src:src+view.rowBytes])
	}
	return nil
}

func (s *ArchSession) restoreStateBlockMetadata(source SessionStateBlockSource) error {
	if len(source.CachedPromptHidden) > 0 && len(source.CachedPromptHidden) != s.arch.Hidden*bf16Size {
		return core.NewError("native.RestoreStateBlocks: prompt hidden size mismatch")
	}
	if len(source.CachedPromptLogits) > 0 && len(source.CachedPromptLogits) != s.arch.Vocab*bf16Size {
		return core.NewError("native.RestoreStateBlocks: prompt logits size mismatch")
	}
	if len(source.RetainedHidden) > 0 && len(source.RetainedHidden) != s.arch.Hidden*bf16Size {
		return core.NewError("native.RestoreStateBlocks: retained hidden size mismatch")
	}
	if len(source.RetainedLogits) > 0 && len(source.RetainedLogits) != s.arch.Vocab*bf16Size {
		return core.NewError("native.RestoreStateBlocks: retained logits size mismatch")
	}
	s.pos = source.Position
	s.cachedIDs = append(s.cachedIDs[:0], source.CachedIDs...)
	if len(source.CachedPromptHidden) > 0 {
		s.rememberCachedPromptEntry(source.CachedPromptIDs, source.CachedPromptHidden, source.CachedPromptLogits)
	} else {
		s.clearCachedPromptHidden()
	}
	if len(source.RetainedHidden) == 0 {
		s.resetRetainedHidden()
	} else {
		s.rememberRetainedHidden(source.RetainedHidden)
	}
	if len(source.RetainedLogits) == 0 {
		s.resetRetainedLogits()
	} else {
		s.rememberRetainedLogits(source.RetainedLogits)
	}
	return nil
}

func (s *ArchSession) stateLayerViews() ([]sessionStateLayerView, error) {
	return s.stateLayerViewsRefreshing(nil)
}

// stateLayerViewsRefreshing is stateLayerViews with a paged-refresh filter:
// needed non-nil re-materialises only the named cache indices' snapshots. The
// MTP drafter export reads 2 winner caches of the target's 30-48 owners, and
// refreshing the rest cost ~10ms per draft block on the paged 26B (#355).
// Every state save/restore path passes nil (refresh all); a filtered call can
// leave other views' snapshot bytes stale, which is safe because every reader
// refreshes through this entry first.
func (s *ArchSession) stateLayerViewsRefreshing(needed map[int]bool) ([]sessionStateLayerView, error) {
	// TurboQuant sessions decline every KV-view consumer (v1): snapshot capture
	// / restore, -state sleep-wake, and the drafter export all read bf16-shaped
	// cache bytes the packed code caches do not hold (unlike q8, which keeps
	// bf16 mirrors). This is THE choke point every such path funnels through —
	// and it gates BOTH carriers (recorded-ICB and state-lane, tq_kv_state.go):
	// a second carrier must never slip past a decline written for the first.
	if s.hasKVTQAny() {
		return nil, core.NewError("native.ArchSession: -kv-cache turboquant declines KV snapshot / conversation-state sleep (v1) — serve stateless or drop -kv-cache")
	}
	ownerCount := s.ownedStateCacheLayers()
	icb := s.state.icb != nil
	if len(s.stateBlockViews) == ownerCount && s.stateBlockViewsICB == icb {
		// Only a paged-KV session needs its materialised snapshot re-copied — that
		// snapshot goes stale as decode appends tokens to the pages. An ICB session
		// keeps its live K/V in the ICB's own cache buffers (snapshotCacheViews
		// returns those, with the ICB geometry override below), while its paged
		// caches are allocated-but-unused. Refreshing an ICB session's views from
		// that empty paged snapshot clears them to zeros (linearSnapshot clear()s
		// then copies only populated pages, of which there are none) — the MTP
		// drafter then cross-attends a zeroed target Key and drafts garbage, which
		// on a quant (ICB) target collapsed speculative acceptance to 0%.
		if !icb {
			if err := s.refreshPagedStateLayerViews(s.stateBlockViews, needed); err != nil {
				return nil, err
			}
		} else if err := s.state.icb.refreshQ8SnapshotMirrors(s.pos); err != nil {
			// cached views hold q8 mirror pointers — re-dequantise so a second
			// save sees the live cache, not the previous sleep's bytes.
			return nil, err
		}
		return s.stateBlockViews, nil
	}
	views := s.stateBlockViews
	if cap(views) < ownerCount {
		views = make([]sessionStateLayerView, 0, ownerCount)
	} else {
		views = views[:0]
	}
	for li, spec := range s.state.specs {
		if !spec.OwnsCache() {
			continue
		}
		paged := s.state.layerPagedKV(li)
		k, _, kPtr, vPtr, err := s.snapshotCacheViews(li)
		if err != nil {
			return nil, err
		}
		cacheBytes := 0
		if paged != nil {
			cacheBytes = paged.snapshotBytes
		} else {
			cacheBytes = int(bufferLengthFast(k))
		}
		cacheRows := s.stateCacheRows(spec)
		rowBytes, err := s.stateCacheRowBytes(cacheBytes, cacheRows)
		if s.state.icb != nil && li < len(s.state.icb.rowBytes) && li < len(s.state.icb.cacheRows) {
			if s.state.icb.rowBytes[li] > 0 && s.state.icb.cacheRows[li] > 0 {
				rowBytes = s.state.icb.rowBytes[li]
				cacheRows = s.state.icb.cacheRows[li]
				cacheBytes = rowBytes * cacheRows
				err = nil
			}
		}
		if err != nil {
			return nil, err
		}
		headDim := headDimOf(spec, s.arch.HeadDim)
		views = append(views, sessionStateLayerView{
			layer:      li,
			kvHeads:    stateLayerViewKVHeads(spec, s.arch.KVHeads, headDim, rowBytes),
			headDim:    headDim,
			rowBytes:   rowBytes,
			cacheIndex: spec.CacheIndex,
			cacheMode:  nativeStateCacheModeFixed,
			maxSize:    s.stateCacheMaxSize(spec),
			cacheRows:  cacheRows,
			keyBytes:   unsafe.Slice(kPtr, cacheBytes),
			valueBytes: unsafe.Slice(vPtr, cacheBytes),
			paged:      paged,
		})
	}
	s.stateBlockViews = views
	s.stateBlockViewsICB = icb
	return s.stateBlockViews, nil
}

func stateLayerViewKVHeads(spec model.LayerSpec, archKVHeads, headDim, rowBytes int) int {
	if rowBytes > 0 && headDim > 0 {
		rowUnit := headDim * bf16Size
		if rowUnit > 0 && rowBytes%rowUnit == 0 {
			if heads := rowBytes / rowUnit; heads > 0 {
				return heads
			}
		}
	}
	return kvHeadsOf(spec, archKVHeads)
}

// refreshPagedStateLayerViews re-materialises the paged caches' linear
// snapshots in place. It passes the view's FULL cacheRows as the snapshot
// extent deliberately: linearSnapshot's rows parameter sizes the destination
// (populated pages alone carry data, the remainder is cleared), and a constant
// extent keeps snapshotBytes stable so the shared snapshot buffers are reused
// rather than reallocated as position grows. Contrast reloadPagedStateLayerViews,
// whose tokens parameter is a valid-row COUNT and must stay position-bounded.
func (s *ArchSession) refreshPagedStateLayerViews(views []sessionStateLayerView, needed map[int]bool) error {
	for i := range views {
		cache := views[i].paged
		if cache == nil {
			continue
		}
		if needed != nil && !needed[views[i].cacheIndex] {
			continue
		}
		_, _, kPtr, vPtr, err := cache.linearSnapshot(views[i].cacheRows)
		if err != nil {
			return err
		}
		cacheBytes := cache.snapshotBytes
		views[i].keyBytes = unsafe.Slice(kPtr, cacheBytes)
		views[i].valueBytes = unsafe.Slice(vPtr, cacheBytes)
	}
	return nil
}

// reloadPagedStateLayerViews restores the paged caches from the views' linear
// snapshot bytes. tokens = min(position, cacheRows) is load-bearing: unlike
// linearSnapshot's extent parameter, loadLinearSnapshot's tokens is the count
// of rows it claims as populated (it becomes the cache length and, on q8, the
// quantisation span) — raw cacheRows here would resurrect garbage rows beyond
// the session position. The asymmetry with refreshPagedStateLayerViews is by
// design; TestDevicePagedKVCacheLinearSnapshotRoundTrip pins the pair
// (snapshot at full extent, reload at populated count).
func (s *ArchSession) reloadPagedStateLayerViews(position int, views []sessionStateLayerView) error {
	for i := range views {
		cache := views[i].paged
		if cache == nil {
			continue
		}
		tokens := min(position, views[i].cacheRows)
		if err := cache.loadLinearSnapshot(views[i].keyBytes, views[i].valueBytes, tokens); err != nil {
			return err
		}
	}
	return nil
}

func (s *ArchSession) stateBlockPlan(startToken, blockSize int) (int, int, int, []int, []sessionStateLayerView, error) {
	if s == nil {
		return 0, 0, 0, nil, nil, core.NewError("native.StateBlockSource: nil session")
	}
	if blockSize <= 0 {
		return 0, 0, 0, nil, nil, core.NewError("native.StateBlockSource: block size must be > 0")
	}
	if startToken < 0 {
		return 0, 0, 0, nil, nil, core.NewError("native.StateBlockSource: start token must be >= 0")
	}
	if s.pos < 0 || s.pos > s.maxLen {
		return 0, 0, 0, nil, nil, core.NewError("native.StateBlockSource: position outside maxLen")
	}
	views, err := s.stateLayerViews()
	if err != nil {
		return 0, 0, 0, nil, nil, err
	}
	boundaries := s.stateBlockBoundaries(blockSize, s.pos, views)
	totalBlocks := 0
	if len(boundaries) > 1 {
		totalBlocks = len(boundaries) - 1
	}
	firstBlock := 0
	for firstBlock < totalBlocks && boundaries[firstBlock+1] <= startToken {
		firstBlock++
	}
	return totalBlocks - firstBlock, firstBlock, totalBlocks, boundaries, views, nil
}

func (s *ArchSession) stateBlockBoundaries(blockSize, position int, views []sessionStateLayerView) []int {
	if position <= 0 {
		s.stateBlockBounds = s.stateBlockBounds[:0]
		return s.stateBlockBounds
	}
	expected := 2 + position/blockSize + 2*len(views)
	if cap(s.stateBlockBounds) < expected {
		s.stateBlockBounds = make([]int, 0, expected)
	} else {
		s.stateBlockBounds = s.stateBlockBounds[:0]
	}
	boundaries := s.stateBlockBounds
	boundaries = append(boundaries, 0)
	for next := blockSize; next < position; next += blockSize {
		boundaries = append(boundaries, next)
	}
	boundaries = append(boundaries, position)
	for _, view := range views {
		if view.maxSize <= 0 || view.cacheRows <= 0 || position <= view.cacheRows {
			continue
		}
		windowStart := position - view.cacheRows
		if windowStart <= 0 || windowStart >= position {
			continue
		}
		boundaries = stateBlockBoundaryInsert(boundaries, windowStart)
		for wrap := ((windowStart / view.cacheRows) + 1) * view.cacheRows; wrap < position; wrap += view.cacheRows {
			boundaries = stateBlockBoundaryInsert(boundaries, wrap)
		}
	}
	s.stateBlockBounds = boundaries
	return boundaries
}

func stateBlockBoundaryInsert(boundaries []int, boundary int) []int {
	for i, existing := range boundaries {
		if existing == boundary {
			return boundaries
		}
		if existing > boundary {
			boundaries = append(boundaries, 0)
			copy(boundaries[i+1:], boundaries[i:])
			boundaries[i] = boundary
			return boundaries
		}
	}
	return append(boundaries, boundary)
}

func (s *ArchSession) ownedStateCacheLayers() int {
	n := 0
	for _, spec := range s.state.specs {
		if spec.OwnsCache() {
			n++
		}
	}
	return n
}

func (s *ArchSession) stateBlockLayerScratch(n int) []SessionStateLayerBlock {
	if cap(s.stateBlockLayers) < n {
		s.stateBlockLayers = make([]SessionStateLayerBlock, n)
	} else {
		s.stateBlockLayers = s.stateBlockLayers[:n]
	}
	return s.stateBlockLayers
}

func (s *ArchSession) stateCacheRows(spec model.LayerSpec) int {
	if s.arch.SlidingWindow > 0 && s.arch.SlidingWindow < s.maxLen && spec.Attention != model.GlobalAttention {
		return s.arch.SlidingWindow
	}
	return s.maxLen
}

func (s *ArchSession) stateCacheMaxSize(spec model.LayerSpec) int {
	if s.arch.SlidingWindow > 0 && s.arch.SlidingWindow < s.maxLen && spec.Attention != model.GlobalAttention {
		return s.arch.SlidingWindow
	}
	return 0
}

func (s *ArchSession) stateCacheRowBytes(cacheBytes, cacheRows int) (int, error) {
	if cacheRows <= 0 {
		return 0, core.NewError("native.sessionStateBlocks: maxLen must be > 0")
	}
	if cacheBytes%cacheRows != 0 {
		return 0, core.NewError("native.sessionStateBlocks: cache length is not row-aligned")
	}
	return cacheBytes / cacheRows, nil
}
