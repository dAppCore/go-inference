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

// benchmarkNativeLayerMultiHeadSlabStateBlocksFixture builds a durable bundle
// whose layer-raw tensors carry MULTIPLE KV heads ([1,H,L,D], B*H>1) — the
// shape engine/metal captures for every grouped-query model. The single-head
// fixture (B*H==1) exercises only the linear fast-append assembly path; this
// one drives the interleaved multi-head assembly that appendKVSnapshotLayerRawBlock
// must fold without an O(N^2) per-block rebuild.
func benchmarkNativeLayerMultiHeadSlabStateBlocksFixture(tb testing.TB) (state.Store, *StateBlockBundle) {
	tb.Helper()
	store := state.NewInMemoryStore(nil)
	snapshot := benchmarkNativeLayerSlabSnapshot(1536, 4, 64)
	bundle, err := snapshot.SaveStateBlocks(context.Background(), store, StateBlockOptions{
		BlockSize:  512,
		KVEncoding: EncodingNative,
	})
	if err != nil {
		tb.Fatalf("SaveStateBlocks(native multi-head layer slab) error = %v", err)
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

// benchmarkGlobalSlidingMixFixture builds a durable bundle whose native
// multi-head layer-raw tensors mix full-attention (global) layers with
// sliding-window layers — the global+sliding cache shape a 31B-class model
// captures. A sliding layer is empty in the leading blocks, so its block 0 is
// the wrong assembly shape donor; if the placement buffer is not seeded from a
// later block the wake regresses to the O(N^2) merged-rebuild path. This is the
// fixture the single-shape full-attention slab fixtures did not cover.
func benchmarkGlobalSlidingMixFixture(tb testing.TB) (state.Store, *StateBlockBundle) {
	tb.Helper()
	store := state.NewInMemoryStore(nil)
	snapshot := benchmarkGlobalSlidingMixSnapshot(2048, 4, 4, 1024, 4, 64)
	bundle, err := snapshot.SaveStateBlocks(context.Background(), store, StateBlockOptions{
		BlockSize:  128,
		KVEncoding: EncodingNative,
	})
	if err != nil {
		tb.Fatalf("SaveStateBlocks(global+sliding mix) error = %v", err)
	}
	if len(bundle.Blocks) != 16 {
		tb.Fatalf("blocks = %d, want 16", len(bundle.Blocks))
	}
	return store, bundle
}

// benchmarkGlobalSlidingMixSnapshot builds globalLayers full-attention native
// layers (L=tokenCount) followed by slidingLayers windowed layers (L=window),
// each a [1,heads,L,headDim] float16 slab.
func benchmarkGlobalSlidingMixSnapshot(tokenCount, globalLayers, slidingLayers, window, heads, headDim int) *Snapshot {
	tokens := make([]int32, tokenCount)
	for i := range tokenCount {
		tokens[i] = int32(i + 1)
	}
	slab := func(l int) ([]byte, []byte, []int32) {
		n := heads * l * headDim * 2
		kb := make([]byte, n)
		vb := make([]byte, n)
		for i := range kb {
			kb[i] = byte(i)
			vb[i] = byte(i + 17)
		}
		return kb, vb, []int32{1, int32(heads), int32(l), int32(headDim)}
	}
	layers := make([]LayerSnapshot, 0, globalLayers+slidingLayers)
	for l := range globalLayers {
		kb, vb, sh := slab(tokenCount)
		layers = append(layers, LayerSnapshot{
			Layer: l, CacheIndex: l,
			KeyDType: "float16", KeyBytes: kb, KeyShape: sh,
			ValueDType: "float16", ValueBytes: vb, ValueShape: append([]int32(nil), sh...),
			Heads: make([]HeadSnapshot, heads),
		})
	}
	slideL := min(tokenCount, window)
	for l := range slidingLayers {
		kb, vb, sh := slab(slideL)
		layers = append(layers, LayerSnapshot{
			Layer: globalLayers + l, CacheIndex: globalLayers + l, MaxSize: window,
			KeyDType: "float16", KeyBytes: kb, KeyShape: sh,
			ValueDType: "float16", ValueBytes: vb, ValueShape: append([]int32(nil), sh...),
			Heads: make([]HeadSnapshot, heads),
		})
	}
	return &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "qwen3",
		Tokens:        tokens,
		TokenOffset:   tokenCount,
		NumLayers:     globalLayers + slidingLayers,
		NumHeads:      heads,
		SeqLen:        tokenCount,
		HeadDim:       headDim,
		NumQueryHeads: heads,
		Layers:        layers,
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
