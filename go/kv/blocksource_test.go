// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"
	"testing"

	core "dappco.re/go"
	state "dappco.re/go/inference/model/state"
)

// stateBlockSourceFixture saves the canonical 4-token / 2-block test snapshot
// to an in-memory store and returns the store + bundle StateBlockSource reads.
func stateBlockSourceFixture(t *testing.T) (state.Store, *StateBlockBundle) {
	t.Helper()
	store := state.NewInMemoryStore(nil)
	snapshot := kvSnapshotBlocksTestSnapshot()
	bundle, err := snapshot.SaveStateBlocks(context.Background(), store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
		URI:        "mlx://blocksource/src",
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks() error = %v", err)
	}
	return store, bundle
}

func TestStateBlockSource_StateBlockSource_Good(t *testing.T) {
	ctx := context.Background()
	store, bundle := stateBlockSourceFixture(t)

	src, err := StateBlockSource(ctx, store, bundle, 0)
	if err != nil {
		t.Fatalf("StateBlockSource() error = %v", err)
	}
	if src.TokenCount != bundle.TokenCount || src.PrefixTokens != bundle.TokenCount {
		t.Fatalf("source token span = {count:%d prefix:%d}, want %d", src.TokenCount, src.PrefixTokens, bundle.TokenCount)
	}
	if src.BlockCount != len(bundle.Blocks) {
		t.Fatalf("source BlockCount = %d, want %d", src.BlockCount, len(bundle.Blocks))
	}

	seen := 0
	for i := 0; i < src.BlockCount; i++ {
		block, loadErr := src.Load(ctx, i)
		if loadErr != nil {
			t.Fatalf("Load(%d) error = %v", i, loadErr)
		}
		if block.Index != i {
			t.Fatalf("block[%d].Index = %d, want %d", i, block.Index, i)
		}
		if block.Snapshot == nil {
			t.Fatalf("block[%d].Snapshot is nil", i)
		}
		seen += block.TokenCount
	}
	if seen != bundle.TokenCount {
		t.Fatalf("streamed token total = %d, want %d", seen, bundle.TokenCount)
	}
}

func TestStateBlockSource_StateBlockSource_GoodPrefixTrimsMidBlock(t *testing.T) {
	ctx := context.Background()
	store, bundle := stateBlockSourceFixture(t)

	// Cover the first token only — the first block (2 tokens) trims to 1.
	src, err := StateBlockSource(ctx, store, bundle, 1)
	if err != nil {
		t.Fatalf("StateBlockSource(prefix=1) error = %v", err)
	}
	if src.PrefixTokens != 1 || src.BlockCount != 1 {
		t.Fatalf("source = {prefix:%d blocks:%d}, want {1 1}", src.PrefixTokens, src.BlockCount)
	}

	block, err := src.Load(ctx, 0)
	if err != nil {
		t.Fatalf("Load(0) error = %v", err)
	}
	if block.TokenCount != 1 {
		t.Fatalf("trimmed block TokenCount = %d, want 1", block.TokenCount)
	}
	if block.Snapshot == nil {
		t.Fatal("trimmed block Snapshot is nil")
	}
}

func TestStateBlockSource_StateBlockSource_BadNilStore(t *testing.T) {
	_, bundle := stateBlockSourceFixture(t)

	if _, err := StateBlockSource(context.Background(), nil, bundle, 0); err == nil {
		t.Fatal("StateBlockSource(nil store) error = nil")
	}
}

func TestStateBlockSource_StateBlockSource_BadPrefixExceedsBundle(t *testing.T) {
	store, bundle := stateBlockSourceFixture(t)

	if _, err := StateBlockSource(context.Background(), store, bundle, bundle.TokenCount+1); err == nil {
		t.Fatal("StateBlockSource(prefix > TokenCount) error = nil")
	}
}

func TestStateBlockSource_StateBlockSource_UglyLoadOutOfRange(t *testing.T) {
	ctx := context.Background()
	store, bundle := stateBlockSourceFixture(t)

	src, err := StateBlockSource(ctx, store, bundle, 0)
	if err != nil {
		t.Fatalf("StateBlockSource() error = %v", err)
	}
	if _, err := src.Load(ctx, -1); err == nil {
		t.Fatal("Load(-1) error = nil, want out-of-range")
	}
	if _, err := src.Load(ctx, src.BlockCount); err == nil {
		t.Fatal("Load(BlockCount) error = nil, want out-of-range")
	}
}

func ExampleStateBlockSource() {
	store := state.NewInMemoryStore(nil)
	snapshot := kvSnapshotBlocksTestSnapshot()
	bundle, _ := snapshot.SaveStateBlocks(context.Background(), store, StateBlockOptions{
		BlockSize: 2, KVEncoding: EncodingNative, URI: "mlx://example/src",
	})

	src, _ := StateBlockSource(context.Background(), store, bundle, 0)
	block, _ := src.Load(context.Background(), 0)
	core.Println("tokens:", block.TokenCount)
	// Output: tokens: 2
}
