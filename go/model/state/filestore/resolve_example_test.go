// SPDX-Licence-Identifier: EUPL-1.2

package filestore

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference/model/state"
)

func ExampleStore_Get() {
	ctx := context.Background()
	dir, cleanup, ok := exampleFilestoreTempDir()
	if !ok {
		return
	}
	defer cleanup()

	store, err := Create(ctx, core.PathJoin(dir, "get.mvlog"))
	if err != nil {
		core.Println(err)
		return
	}
	defer store.Close()
	ref, err := store.Put(ctx, "hello", state.PutOptions{})
	if err != nil {
		core.Println(err)
		return
	}

	text, err := store.Get(ctx, ref.ChunkID)
	core.Println(err == nil, text)
	// Output: true hello
}

func ExampleStore_Resolve() {
	ctx := context.Background()
	dir, cleanup, ok := exampleFilestoreTempDir()
	if !ok {
		return
	}
	defer cleanup()

	store, err := Create(ctx, core.PathJoin(dir, "resolve.mvlog"))
	if err != nil {
		core.Println(err)
		return
	}
	defer store.Close()
	ref, err := store.Put(ctx, "hello", state.PutOptions{})
	if err != nil {
		core.Println(err)
		return
	}

	chunk, err := store.Resolve(ctx, ref.ChunkID)
	core.Println(err == nil, chunk.Text)
	// Output: true hello
}

func ExampleStore_ResolveURI() {
	ctx := context.Background()
	dir, cleanup, ok := exampleFilestoreTempDir()
	if !ok {
		return
	}
	defer cleanup()

	store, err := Create(ctx, core.PathJoin(dir, "resolveuri.mvlog"))
	if err != nil {
		core.Println(err)
		return
	}
	defer store.Close()
	if _, err := store.Put(ctx, "hello", state.PutOptions{URI: "mlx://example/1"}); err != nil {
		core.Println(err)
		return
	}

	chunk, err := store.ResolveURI(ctx, "mlx://example/1")
	core.Println(err == nil, chunk.Text)
	// Output: true hello
}

func ExampleStore_ResolveBytes() {
	ctx := context.Background()
	dir, cleanup, ok := exampleFilestoreTempDir()
	if !ok {
		return
	}
	defer cleanup()

	store, err := Create(ctx, core.PathJoin(dir, "resolvebytes.mvlog"))
	if err != nil {
		core.Println(err)
		return
	}
	defer store.Close()
	ref, err := store.PutBytes(ctx, []byte{0, 1, 2, 255}, state.PutOptions{})
	if err != nil {
		core.Println(err)
		return
	}

	// ResolveBytes skips the []byte-to-string decode Resolve performs,
	// so binary payloads round-trip untouched.
	chunk, err := store.ResolveBytes(ctx, ref.ChunkID)
	core.Println(err == nil, chunk.Data)
	// Output: true [0 1 2 255]
}

func ExampleStore_BorrowBytes() {
	ctx := context.Background()
	dir, cleanup, ok := exampleFilestoreTempDir()
	if !ok {
		return
	}
	defer cleanup()

	store, err := Create(ctx, core.PathJoin(dir, "borrowbytes.mvlog"))
	if err != nil {
		core.Println(err)
		return
	}
	defer store.Close()
	ref, err := store.PutBytes(ctx, []byte("hello"), state.PutOptions{})
	if err != nil {
		core.Println(err)
		return
	}

	borrowed, err := store.BorrowBytes(ctx, ref.ChunkID)
	core.Println(err == nil, string(borrowed.Data))
	// Output: true hello
}

func ExampleStore_ResolveRefBytes() {
	ctx := context.Background()
	dir, cleanup, ok := exampleFilestoreTempDir()
	if !ok {
		return
	}
	defer cleanup()

	store, err := Create(ctx, core.PathJoin(dir, "resolverefbytes.mvlog"))
	if err != nil {
		core.Println(err)
		return
	}
	defer store.Close()
	ref, err := store.PutBytes(ctx, []byte("hello"), state.PutOptions{})
	if err != nil {
		core.Println(err)
		return
	}

	// A bare ChunkID ref (no frame offset) delegates straight to
	// ResolveBytes — the frame-offset fast path only engages when the
	// caller already knows the on-disk record position.
	chunk, err := store.ResolveRefBytes(ctx, state.ChunkRef{ChunkID: ref.ChunkID})
	core.Println(err == nil, string(chunk.Data))
	// Output: true hello
}

func ExampleStore_BorrowRefBytes() {
	ctx := context.Background()
	dir, cleanup, ok := exampleFilestoreTempDir()
	if !ok {
		return
	}
	defer cleanup()

	store, err := Create(ctx, core.PathJoin(dir, "borrowrefbytes.mvlog"))
	if err != nil {
		core.Println(err)
		return
	}
	defer store.Close()
	ref, err := store.PutBytes(ctx, []byte("hello"), state.PutOptions{})
	if err != nil {
		core.Println(err)
		return
	}

	borrowed, err := store.BorrowRefBytes(ctx, state.ChunkRef{ChunkID: ref.ChunkID})
	core.Println(err == nil, string(borrowed.Data))
	// Output: true hello
}
