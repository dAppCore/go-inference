// SPDX-Licence-Identifier: EUPL-1.2

package state

import (
	"context"
	"fmt"
)

// ExampleDedupStore shows two conversations sleeping the same prefix block: the
// second write is deduped, so the inner store holds one physical copy and both
// bundles reference the same chunk.
func ExampleDedupStore() {
	inner := NewInMemoryStore(nil)
	store := NewDedupStore(inner)
	ctx := context.Background()

	block := []byte("shared system-prompt KV block")
	a, _ := store.PutBytes(ctx, block, PutOptions{URI: "conv-a/block/0"})
	b, _ := store.PutBytes(ctx, block, PutOptions{URI: "conv-b/block/0"})

	fmt.Println("same chunk:", a.ChunkID == b.ChunkID)
	fmt.Println("physical chunks:", inner.ChunkCount())
	fmt.Println("writes avoided:", store.Stats().Dedups)
	// Output:
	// same chunk: true
	// physical chunks: 1
	// writes avoided: 1
}
