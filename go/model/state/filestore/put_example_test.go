// SPDX-Licence-Identifier: EUPL-1.2

package filestore

import (
	"context"
	stdio "io"

	core "dappco.re/go"
	"dappco.re/go/inference/model/state"
)

func ExampleStore_Put() {
	ctx := context.Background()
	dir, cleanup, ok := exampleFilestoreTempDir()
	if !ok {
		return
	}
	defer cleanup()

	store, err := Create(ctx, core.PathJoin(dir, "put.mvlog"))
	if err != nil {
		core.Println(err)
		return
	}
	defer store.Close()

	ref, err := store.Put(ctx, "hello", state.PutOptions{URI: "mlx://put/1"})
	core.Println(err == nil, ref.ChunkID)
	// Output: true 1
}

func ExampleStore_PutBytes() {
	ctx := context.Background()
	dir, cleanup, ok := exampleFilestoreTempDir()
	if !ok {
		return
	}
	defer cleanup()

	store, err := Create(ctx, core.PathJoin(dir, "putbytes.mvlog"))
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
	chunk, err := store.ResolveBytes(ctx, ref.ChunkID)
	core.Println(err == nil, chunk.Data)
	// Output: true [0 1 2 255]
}

func ExampleStore_PutBytesStream() {
	ctx := context.Background()
	dir, cleanup, ok := exampleFilestoreTempDir()
	if !ok {
		return
	}
	defer cleanup()

	store, err := Create(ctx, core.PathJoin(dir, "putbytesstream.mvlog"))
	if err != nil {
		core.Println(err)
		return
	}
	defer store.Close()

	// PutBytesStream writes directly from the callback's writer without
	// staging the payload in memory first — useful when the caller
	// already has the data split across multiple writes.
	ref, err := store.PutBytesStream(ctx, 5, state.PutOptions{}, func(w stdio.Writer) error {
		if _, err := w.Write([]byte("he")); err != nil {
			return err
		}
		_, err := w.Write([]byte("llo"))
		return err
	})
	if err != nil {
		core.Println(err)
		return
	}
	chunk, err := store.ResolveBytes(ctx, ref.ChunkID)
	core.Println(err == nil, string(chunk.Data))
	// Output: true hello
}
