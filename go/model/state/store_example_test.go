// SPDX-Licence-Identifier: EUPL-1.2

package state

import (
	"context"
	"fmt"
)

// ExampleResolve resolves a chunk by ID from any Store, backed here by the
// in-memory implementation.
func ExampleResolve() {
	store := NewInMemoryStore(map[int]string{1: "hello state"})

	chunk, err := Resolve(context.Background(), store, 1)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(chunk.Text)
	// Output:
	// hello state
}

// ExampleResolveBytes resolves a chunk's binary payload, backfilling Data
// from Text when the store only carries plain text.
func ExampleResolveBytes() {
	store := NewInMemoryStore(map[int]string{1: "hello state"})

	chunk, err := ResolveBytes(context.Background(), store, 1)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(string(chunk.Data))
	// Output:
	// hello state
}

// ExampleResolveRefBytes resolves a chunk's binary payload by full
// ChunkRef rather than bare ID.
func ExampleResolveRefBytes() {
	store := NewInMemoryStore(nil)
	ref, err := store.PutBytes(context.Background(), []byte("archived bytes"), PutOptions{})
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	chunk, err := ResolveRefBytes(context.Background(), store, ref)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(string(chunk.Data))
	// Output:
	// archived bytes
}

// ExampleBorrowBytes returns a zero-copy view onto a store's own backing
// slice when the store supports it.
func ExampleBorrowBytes() {
	store := NewInMemoryStore(nil)
	ref, err := store.PutBytes(context.Background(), []byte("borrowed"), PutOptions{})
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	borrowed, err := BorrowBytes(context.Background(), store, ref.ChunkID)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(string(borrowed.Data))
	// Output:
	// borrowed
}

// ExampleBorrowRefBytes overlays a caller-supplied Segment onto the
// borrowed view's Ref.
func ExampleBorrowRefBytes() {
	store := NewInMemoryStore(nil)
	ref, err := store.PutBytes(context.Background(), []byte("segmented"), PutOptions{})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	ref.Segment = "epoch-1"

	borrowed, err := BorrowRefBytes(context.Background(), store, ref)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(borrowed.Ref.Segment, string(borrowed.Data))
	// Output:
	// epoch-1 segmented
}

// ExampleResolveURI resolves a chunk by its caller-assigned URI rather
// than its numeric ChunkID.
func ExampleResolveURI() {
	store := NewInMemoryStore(nil)
	if _, err := store.Put(context.Background(), "uri-addressed", PutOptions{URI: "state://demo/1"}); err != nil {
		fmt.Println("error:", err)
		return
	}

	chunk, err := ResolveURI(context.Background(), store, "state://demo/1")
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(chunk.Text)
	// Output:
	// uri-addressed
}

// ExampleMergeRef overlays a partial ChunkRef onto a base ref — only the
// fields the overlay actually sets are replaced.
func ExampleMergeRef() {
	base := ChunkRef{ChunkID: 7, Codec: CodecMemory, Segment: "epoch-1"}
	overlay := ChunkRef{Codec: CodecStateVideo}

	merged := MergeRef(base, overlay)
	fmt.Println(merged.ChunkID, merged.Codec, merged.Segment)
	// Output:
	// 7 state/qr-video epoch-1
}

// ExampleChunkNotFoundError_Error formats a not-found error for a specific
// chunk ID.
func ExampleChunkNotFoundError_Error() {
	err := &ChunkNotFoundError{ID: 42}
	fmt.Println(err.Error())
	// Output:
	// state chunk 42 not found
}

// ExampleChunkNotFoundError_Unwrap exposes the package-level sentinel so
// callers can match with errors.Is/core.Is regardless of the concrete ID.
func ExampleChunkNotFoundError_Unwrap() {
	err := &ChunkNotFoundError{ID: 42}
	fmt.Println(err.Unwrap() == ErrChunkNotFound)
	// Output:
	// true
}

// ExampleURIChunkNotFoundError_Error formats a not-found error for a
// specific URI.
func ExampleURIChunkNotFoundError_Error() {
	err := &URIChunkNotFoundError{URI: "state://missing"}
	fmt.Println(err.Error())
	// Output:
	// state chunk URI "state://missing" not found
}

// ExampleURIChunkNotFoundError_Unwrap exposes the same package-level
// sentinel as ChunkNotFoundError, so callers can match either error shape
// with a single errors.Is/core.Is check.
func ExampleURIChunkNotFoundError_Unwrap() {
	err := &URIChunkNotFoundError{URI: "state://missing"}
	fmt.Println(err.Unwrap() == ErrChunkNotFound)
	// Output:
	// true
}
