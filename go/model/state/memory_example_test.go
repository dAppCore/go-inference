// SPDX-Licence-Identifier: EUPL-1.2

package state

import (
	"context"
	"fmt"
)

// ExampleNewInMemoryStore seeds a store with a pre-built text map — handy
// for tests and fixtures that need Store content without a real backend.
func ExampleNewInMemoryStore() {
	store := NewInMemoryStore(map[int]string{1: "seeded chunk"})

	text, err := store.Get(context.Background(), 1)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(text)
	// Output:
	// seeded chunk
}

// ExampleNewInMemoryStoreWithManifest seeds both chunk text and an
// explicit ChunkRef manifest, letting a fixture control codec/segment
// metadata that NewInMemoryStore would otherwise derive automatically.
func ExampleNewInMemoryStoreWithManifest() {
	store := NewInMemoryStoreWithManifest(
		map[int]string{1: "seeded"},
		map[int]ChunkRef{1: {Codec: "custom/codec"}},
	)

	chunk, err := store.Resolve(context.Background(), 1)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(chunk.Ref.Codec)
	// Output:
	// custom/codec
}

// ExampleInMemoryStore_Get returns a chunk's text by ID.
func ExampleInMemoryStore_Get() {
	store := NewInMemoryStore(map[int]string{1: "hello"})

	text, err := store.Get(context.Background(), 1)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(text)
	// Output:
	// hello
}

// ExampleInMemoryStore_Resolve returns the full Chunk (text plus ref
// metadata) for an ID.
func ExampleInMemoryStore_Resolve() {
	store := NewInMemoryStore(map[int]string{1: "hello"})

	chunk, err := store.Resolve(context.Background(), 1)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(chunk.Text, chunk.Ref.Codec)
	// Output:
	// hello memory/plaintext
}

// ExampleInMemoryStore_ResolveBytes returns a chunk's binary payload,
// deriving it from Text when the chunk was written as plain text.
func ExampleInMemoryStore_ResolveBytes() {
	store := NewInMemoryStore(map[int]string{1: "hello"})

	chunk, err := store.ResolveBytes(context.Background(), 1)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(string(chunk.Data))
	// Output:
	// hello
}

// ExampleInMemoryStore_BorrowBytes returns a live view onto the store's
// own backing slice — no defensive copy.
func ExampleInMemoryStore_BorrowBytes() {
	store := NewInMemoryStore(nil)
	ref, err := store.PutBytes(context.Background(), []byte("borrowed"), PutOptions{})
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	borrowed, err := store.BorrowBytes(context.Background(), ref.ChunkID)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(string(borrowed.Data))
	// Output:
	// borrowed
}

// ExampleInMemoryStore_BorrowRefBytes overlays a caller-supplied Segment
// onto the borrowed view's Ref.
func ExampleInMemoryStore_BorrowRefBytes() {
	store := NewInMemoryStore(nil)
	ref, err := store.PutBytes(context.Background(), []byte("segmented"), PutOptions{})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	ref.Segment = "epoch-1"

	borrowed, err := store.BorrowRefBytes(context.Background(), ref)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(borrowed.Ref.Segment)
	// Output:
	// epoch-1
}

// ExampleInMemoryStore_ResolveURI resolves a chunk by its caller-assigned
// URI rather than its numeric ChunkID.
func ExampleInMemoryStore_ResolveURI() {
	store := NewInMemoryStore(nil)
	if _, err := store.Put(context.Background(), "uri-addressed", PutOptions{URI: "state://demo/1"}); err != nil {
		fmt.Println("error:", err)
		return
	}

	chunk, err := store.ResolveURI(context.Background(), "state://demo/1")
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(chunk.Text)
	// Output:
	// uri-addressed
}

// ExampleInMemoryStore_Put writes a text chunk and returns its assigned
// ChunkRef.
func ExampleInMemoryStore_Put() {
	store := NewInMemoryStore(nil)

	ref, err := store.Put(context.Background(), "hello", PutOptions{})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(ref.ChunkID, ref.Codec)
	// Output:
	// 1 memory/plaintext
}

// ExampleInMemoryStore_PutBytes writes a binary chunk and returns its
// assigned ChunkRef.
func ExampleInMemoryStore_PutBytes() {
	store := NewInMemoryStore(nil)

	ref, err := store.PutBytes(context.Background(), []byte{0, 1, 2}, PutOptions{})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(ref.ChunkID, ref.Codec)
	// Output:
	// 1 memory/plaintext
}
