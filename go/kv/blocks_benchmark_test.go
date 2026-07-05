// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"
	"testing"

	state "dappco.re/go/inference/model/state"
)

var (
	stateBlocksBenchmarkSnapshot *Snapshot
	stateBlocksBenchmarkTokens   []int32
)

func benchmarkStateBlocksFixture(tb testing.TB) (state.Store, *StateBlockBundle) {
	tb.Helper()
	store := state.NewInMemoryStore(nil)
	snapshot := benchmarkStateBlocksSnapshot(1536, 512)
	bundle, err := snapshot.SaveStateBlocks(context.Background(), store, StateBlockOptions{
		BlockSize:  512,
		KVEncoding: EncodingNative,
	})
	if err != nil {
		tb.Fatalf("SaveStateBlocks() error = %v", err)
	}
	if len(bundle.Blocks) != 3 {
		tb.Fatalf("blocks = %d, want 3", len(bundle.Blocks))
	}
	return store, bundle
}

func benchmarkNativeLayerSlabStateBlocksFixture(tb testing.TB) (state.Store, *StateBlockBundle) {
	tb.Helper()
	store := state.NewInMemoryStore(nil)
	snapshot := benchmarkNativeLayerSlabSnapshot(1536, 1, 64)
	bundle, err := snapshot.SaveStateBlocks(context.Background(), store, StateBlockOptions{
		BlockSize:  512,
		KVEncoding: EncodingNative,
	})
	if err != nil {
		tb.Fatalf("SaveStateBlocks(native layer slab) error = %v", err)
	}
	if len(bundle.Blocks) != 3 {
		tb.Fatalf("blocks = %d, want 3", len(bundle.Blocks))
	}
	return store, bundle
}

func benchmarkStateBlocksSnapshot(tokenCount, localWindow int) *Snapshot {
	tokens := make([]int32, tokenCount)
	fullKey := make([]float32, tokenCount)
	fullValue := make([]float32, tokenCount)
	localKey := make([]float32, localWindow)
	localValue := make([]float32, localWindow)
	for i := range tokenCount {
		tokens[i] = int32(i + 1)
		fullKey[i] = float32(i)
		fullValue[i] = float32(i + 1000)
	}
	for i := range localWindow {
		localKey[i] = float32(i + 2000)
		localValue[i] = float32(i + 3000)
	}
	return &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        tokens,
		TokenOffset:   tokenCount,
		NumLayers:     2,
		NumHeads:      1,
		SeqLen:        tokenCount,
		HeadDim:       1,
		NumQueryHeads: 1,
		Layers: []LayerSnapshot{
			{
				Layer:      0,
				CacheIndex: 0,
				Heads: []HeadSnapshot{{
					Key:   fullKey,
					Value: fullValue,
				}},
			},
			{
				Layer:      1,
				CacheIndex: 1,
				Heads: []HeadSnapshot{{
					Key:   localKey,
					Value: localValue,
				}},
			},
		},
	}
}

func benchmarkNativeLayerSlabSnapshot(tokenCount, heads, headDim int) *Snapshot {
	tokens := make([]int32, tokenCount)
	B, H, L, D := 1, heads, tokenCount, headDim
	bytesPerValue := 2
	slabBytes := B * H * L * D * bytesPerValue
	keyBytes := make([]byte, slabBytes)
	valueBytes := make([]byte, slabBytes)
	for i := range tokenCount {
		tokens[i] = int32(i + 1)
	}
	for i := range keyBytes {
		keyBytes[i] = byte(i)
		valueBytes[i] = byte(i + 17)
	}
	return &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        tokens,
		TokenOffset:   tokenCount,
		NumLayers:     1,
		NumHeads:      heads,
		SeqLen:        tokenCount,
		HeadDim:       headDim,
		NumQueryHeads: heads,
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			KeyDType:   "float16",
			KeyBytes:   keyBytes,
			KeyShape:   []int32{int32(B), int32(H), int32(L), int32(D)},
			ValueDType: "float16",
			ValueBytes: valueBytes,
			ValueShape: []int32{int32(B), int32(H), int32(L), int32(D)},
			Heads:      make([]HeadSnapshot, heads),
		}},
	}
}
