// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference/kv"
)

// HIPSessionStateBlockSource is a self-contained stream of portable HIP
// session-state blocks. Unlike Metal's source, its snapshots own copied host
// data: HIP has no portable borrowed-buffer API for its native pages.
//
// StateBlockSourceFrom may omit a trusted resident prefix. RestoreStateBlocks
// validates that prefix's token IDs in the target before it restores the suffix.
type HIPSessionStateBlockSource struct {
	Position  int
	Tokens    []int32
	Generated []int32
	Blocks    []kv.Block

	blockSize       int
	firstBlockIndex int
	trustedPrefix   int
}

// TrustPrefixBlocks records that leading full blocks are already resident in a
// restore target. Callers normally receive this state from StateBlockSourceFrom.
func (source *HIPSessionStateBlockSource) TrustPrefixBlocks(blockSize, firstBlockIndex int) error {
	if source == nil {
		return core.NewError("hip.HIPSessionStateBlockSource: nil source")
	}
	if blockSize <= 0 {
		return core.NewError("hip.HIPSessionStateBlockSource: block size must be positive")
	}
	if firstBlockIndex < 0 {
		return core.NewError("hip.HIPSessionStateBlockSource: first block index must be non-negative")
	}
	return source.TrustPrefixTokens(firstBlockIndex*blockSize, firstBlockIndex)
}

// TrustPrefixTokens records an exact resident token prefix for a block stream.
func (source *HIPSessionStateBlockSource) TrustPrefixTokens(tokens, firstBlockIndex int) error {
	if source == nil {
		return core.NewError("hip.HIPSessionStateBlockSource: nil source")
	}
	if tokens < 0 || tokens > source.Position {
		return core.NewError("hip.HIPSessionStateBlockSource: trusted prefix outside position")
	}
	if firstBlockIndex < 0 || (tokens > 0 && firstBlockIndex == 0) {
		return core.NewError("hip.HIPSessionStateBlockSource: trusted prefix block index is invalid")
	}
	source.trustedPrefix = tokens
	source.firstBlockIndex = firstBlockIndex
	return nil
}

// StateBlockSource captures all retained session state as portable K/V blocks.
func (s *hipEngineSession) StateBlockSource(blockSize int) (HIPSessionStateBlockSource, error) {
	return s.StateBlockSourceFrom(0, blockSize)
}

// StateBlockSourceFrom captures the state blocks needed after startToken.
// Whole blocks before startToken are omitted and become a trusted target prefix.
func (s *hipEngineSession) StateBlockSourceFrom(startToken, blockSize int) (HIPSessionStateBlockSource, error) {
	if blockSize <= 0 {
		return HIPSessionStateBlockSource{}, core.NewError("hip.EngineSession.StateBlockSource: block size must be positive")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return HIPSessionStateBlockSource{}, core.NewError("hip.EngineSession.StateBlockSource: session is closed")
	}
	if startToken < 0 || startToken > len(s.tokens) {
		return HIPSessionStateBlockSource{}, core.NewError("hip.EngineSession.StateBlockSource: start token outside session")
	}
	if len(s.pendingEmbeddings) > 0 {
		return HIPSessionStateBlockSource{}, core.NewError("hip.EngineSession.StateBlockSource: custom embeddings must be forwarded before state capture")
	}
	firstBlockIndex := startToken / blockSize
	trustedPrefix := firstBlockIndex * blockSize
	blocks, err := s.stateBlocksLocked(blockSize, trustedPrefix)
	if err != nil {
		return HIPSessionStateBlockSource{}, err
	}
	return HIPSessionStateBlockSource{
		Position:        len(s.tokens),
		Tokens:          core.SliceClone(s.tokens),
		Generated:       core.SliceClone(s.generated),
		Blocks:          cloneHIPStateBlocks(blocks),
		blockSize:       blockSize,
		firstBlockIndex: firstBlockIndex,
		trustedPrefix:   trustedPrefix,
	}, nil
}

// RestoreStateBlocks grafts a portable state-block source into this session.
// A suffix source requires this session to hold exactly the trusted token
// prefix, with matching IDs. It restores the assembled snapshot through the
// existing raw-device-page restore path, preserving encoded KV bytes.
func (s *hipEngineSession) RestoreStateBlocks(source HIPSessionStateBlockSource) error {
	if err := source.validate(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return core.NewError("hip.EngineSession.RestoreStateBlocks: session is closed")
	}
	trustedPrefix := source.trustedPrefix
	if trustedPrefix > 0 {
		if len(s.tokens) != trustedPrefix {
			return core.NewError("hip.EngineSession.RestoreStateBlocks: trusted prefix is not the complete resident session")
		}
		for index := range trustedPrefix {
			if s.tokens[index] != source.Tokens[index] {
				return core.NewError("hip.EngineSession.RestoreStateBlocks: trusted prefix token IDs differ")
			}
		}
	}
	if source.Position == 0 {
		if err := s.closeDeviceLocked(); err != nil {
			return err
		}
		s.pending = nil
		s.pendingEmbeddings = nil
		s.tokens = nil
		s.generated = nil
		return nil
	}
	if len(source.Blocks) == 0 {
		if trustedPrefix != source.Position {
			return core.NewError("hip.EngineSession.RestoreStateBlocks: source does not cover session position")
		}
		s.generated = core.SliceClone(source.Generated)
		return nil
	}
	blocks := source.Blocks
	if trustedPrefix > 0 {
		prefix, err := s.stateBlocksLocked(source.blockSize, 0)
		if err != nil {
			return err
		}
		blocks = append(prefix, source.Blocks...)
	}
	snapshot, err := kv.AssembleBlocks(blocks)
	if err != nil {
		return core.E("hip.EngineSession.RestoreStateBlocks", "assemble K/V blocks", err)
	}
	snapshot.Tokens = core.SliceClone(source.Tokens)
	snapshot.Generated = core.SliceClone(source.Generated)
	snapshot.TokenOffset = source.Position
	return s.restoreFromKVLocked(context.Background(), snapshot)
}

func (source HIPSessionStateBlockSource) validate() error {
	if source.Position < 0 || len(source.Tokens) != source.Position {
		return core.NewError("hip.HIPSessionStateBlockSource: token metadata does not match position")
	}
	if source.blockSize <= 0 {
		return core.NewError("hip.HIPSessionStateBlockSource: block size must be positive")
	}
	if source.trustedPrefix < 0 || source.trustedPrefix > source.Position {
		return core.NewError("hip.HIPSessionStateBlockSource: trusted prefix outside position")
	}
	if source.firstBlockIndex < 0 {
		return core.NewError("hip.HIPSessionStateBlockSource: first block index must be non-negative")
	}
	if len(source.Blocks) == 0 {
		return nil
	}
	nextStart := source.trustedPrefix
	nextIndex := source.firstBlockIndex
	for _, block := range source.Blocks {
		if block.Index != nextIndex || block.TokenStart != nextStart || block.TokenCount <= 0 || block.Snapshot == nil || len(block.Snapshot.Tokens) != block.TokenCount {
			return core.NewError("hip.HIPSessionStateBlockSource: blocks are not contiguous")
		}
		nextStart += block.TokenCount
		nextIndex++
	}
	if nextStart != source.Position {
		return core.NewError("hip.HIPSessionStateBlockSource: blocks do not cover session position")
	}
	return nil
}

func (s *hipEngineSession) stateBlocksLocked(blockSize, startToken int) ([]kv.Block, error) {
	host, err := s.hostStateLocked()
	if err != nil {
		return nil, err
	}
	forwarded := host.tokenCountForConfig(s.cfg)
	if forwarded > len(s.tokens) {
		return nil, core.NewError("hip.EngineSession.StateBlockSource: retained KV exceeds token history")
	}
	firstIndex := startToken / blockSize
	blocks := make([]kv.Block, 0, (len(s.tokens)-startToken+blockSize-1)/blockSize)
	for index, start := firstIndex, firstIndex*blockSize; start < len(s.tokens); index, start = index+1, start+blockSize {
		count := blockSize
		if start+count > len(s.tokens) {
			count = len(s.tokens) - start
		}
		kvCount := count
		if start >= forwarded {
			kvCount = 0
		} else if start+kvCount > forwarded {
			kvCount = forwarded - start
		}
		snapshot, err := hipDecodeStateToSnapshot(hipSliceDecodeStateTokens(host, s.cfg, start, kvCount), s.cfg, hipTokenWindow(s.tokens, start, count), nil, kv.CaptureOptions{RawKVOnly: true})
		if err != nil {
			return nil, err
		}
		if kvCount > 0 {
			if err := hipAttachDeviceKVPayloadRange(snapshot, s.device, start, kvCount); err != nil {
				return nil, err
			}
		}
		snapshot.TokenOffset = start + count
		blocks = append(blocks, kv.Block{Index: index, TokenStart: start, TokenCount: count, Snapshot: snapshot})
	}
	return blocks, nil
}

func cloneHIPStateBlocks(blocks []kv.Block) []kv.Block {
	if len(blocks) == 0 {
		return nil
	}
	cloned := make([]kv.Block, len(blocks))
	for index, block := range blocks {
		cloned[index] = block
		cloned[index].Snapshot = block.Snapshot.Clone()
	}
	return cloned
}

// WarmPromptCache records ids as the next reusable HIP prompt boundary. HIP's
// native path has no zero-output prefill entry point, so this does not claim to
// materialise K/V immediately; the next combined decode forwards only the
// unmaterialised suffix, while a resident prefix remains native-owned.
func (s *hipEngineSession) WarmPromptCache(ids []int32) error {
	_, err := s.PrefillTokensCached(ids)
	return err
}
