// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"
	"testing"

	state "dappco.re/go/inference/state"
)

func BenchmarkSaveStateBlocks_NativeLayerSingleHeadSlabThreeBlocks(b *testing.B) {
	ctx := context.Background()
	snapshot := benchmarkNativeLayerSlabSnapshot(1536, 1, 64)
	opts := StateBlockOptions{
		BlockSize:  512,
		KVEncoding: EncodingNative,
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		store := state.NewInMemoryStore(nil)
		bundle, err := snapshot.SaveStateBlocks(ctx, store, opts)
		if err != nil {
			b.Fatal(err)
		}
		if len(bundle.Blocks) != 3 {
			b.Fatalf("blocks = %d, want 3", len(bundle.Blocks))
		}
	}
}
