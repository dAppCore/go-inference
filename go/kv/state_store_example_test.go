// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
)

// ExampleSnapshot_SaveState writes a KV snapshot to a State cold store as a
// base64-wrapped envelope and reports the chunk it produced.
func ExampleSnapshot_SaveState() {
	store := state.NewInMemoryStore(nil)
	snapshot := testSnapshot()

	ref, err := snapshot.SaveState(context.Background(), store, StateOptions{
		KVEncoding: EncodingQ8,
		URI:        "mlx://session/example",
	})
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("written:", ref.ChunkID > 0)
	// Output: written: true
}

// ExampleLoadFromState resolves and decodes a KV snapshot from a State chunk
// ref written by SaveState.
func ExampleLoadFromState() {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	ref, err := testSnapshot().SaveState(ctx, store, StateOptions{KVEncoding: EncodingQ8})
	if err != nil {
		core.Println("error:", err)
		return
	}

	loaded, err := LoadFromState(ctx, store, ref)
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("architecture:", loaded.Architecture)
	// Output: architecture: gemma4_text
}

// ExampleSnapshot_SaveMemvid writes a KV snapshot through the deprecated
// SaveMemvid alias, which forwards transparently to SaveState.
func ExampleSnapshot_SaveMemvid() {
	store := state.NewInMemoryStore(nil)
	ref, err := testSnapshot().SaveMemvid(context.Background(), store, MemvidOptions{KVEncoding: EncodingQ8})
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("written:", ref.ChunkID > 0)
	// Output: written: true
}

// ExampleLoadFromStateWithOptions decodes a KV snapshot from State with explicit
// decode options; RawKVOnly keeps the raw key bytes on each head.
func ExampleLoadFromStateWithOptions() {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	ref, err := testSnapshot().SaveState(ctx, store, StateOptions{KVEncoding: EncodingNative})
	if err != nil {
		core.Println("error:", err)
		return
	}

	loaded, err := LoadFromStateWithOptions(ctx, store, ref, LoadOptions{RawKVOnly: true})
	if err != nil {
		core.Println("error:", err)
		return
	}
	head, _ := loaded.Head(0, 0)
	core.Println("raw bytes retained:", len(head.KeyBytes) > 0)
	// Output: raw bytes retained: true
}

// ExampleLoadFromMemvid decodes a chunk written by SaveState through the
// deprecated LoadFromMemvid alias.
func ExampleLoadFromMemvid() {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	ref, err := testSnapshot().SaveState(ctx, store, StateOptions{KVEncoding: EncodingQ8})
	if err != nil {
		core.Println("error:", err)
		return
	}

	loaded, err := LoadFromMemvid(ctx, store, ref)
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("architecture:", loaded.Architecture)
	// Output: architecture: gemma4_text
}

// ExampleLoadFromMemvidWithOptions decodes a chunk through the deprecated
// options-bearing alias, forwarding RawKVOnly to the canonical path.
func ExampleLoadFromMemvidWithOptions() {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	ref, err := testSnapshot().SaveState(ctx, store, StateOptions{KVEncoding: EncodingNative})
	if err != nil {
		core.Println("error:", err)
		return
	}

	loaded, err := LoadFromMemvidWithOptions(ctx, store, ref, LoadOptions{RawKVOnly: true})
	if err != nil {
		core.Println("error:", err)
		return
	}
	head, _ := loaded.Head(0, 0)
	core.Println("raw bytes retained:", len(head.KeyBytes) > 0)
	// Output: raw bytes retained: true
}

// ExampleEffectiveTokenOffset shows the explicit-offset path: when TokenOffset
// is set it is returned verbatim, independent of the token count.
func ExampleEffectiveTokenOffset() {
	offset := EffectiveTokenOffset(&Snapshot{TokenOffset: 42, Tokens: []int32{1, 2, 3}})
	core.Println("offset:", offset)
	// Output: offset: 42
}
