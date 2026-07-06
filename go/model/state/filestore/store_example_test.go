// SPDX-Licence-Identifier: EUPL-1.2

package filestore

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference/model/state"
)

func ExampleCreate() {
	ctx := context.Background()
	dir, cleanup, ok := exampleFilestoreTempDir()
	if !ok {
		return
	}
	defer cleanup()

	store, err := Create(ctx, core.PathJoin(dir, "create.mvlog"))
	if err != nil {
		core.Println(err)
		return
	}
	defer store.Close()

	core.Println(store.ChunkCount())
	// Output: 0
}

func ExampleOpen() {
	ctx := context.Background()
	dir, cleanup, ok := exampleFilestoreTempDir()
	if !ok {
		return
	}
	defer cleanup()
	path := core.PathJoin(dir, "open.mvlog")

	store, err := Create(ctx, path)
	if err != nil {
		core.Println(err)
		return
	}
	if _, err := store.Put(ctx, "hello", state.PutOptions{}); err != nil {
		core.Println(err)
		return
	}
	if err := store.Close(); err != nil {
		core.Println(err)
		return
	}

	reopened, err := Open(ctx, path)
	if err != nil {
		core.Println(err)
		return
	}
	defer reopened.Close()

	core.Println(reopened.ChunkCount())
	// Output: 1
}

func ExampleOpenWithSegmentAlias() {
	ctx := context.Background()
	dir, cleanup, ok := exampleFilestoreTempDir()
	if !ok {
		return
	}
	defer cleanup()
	sourcePath := core.PathJoin(dir, "source.mvlog")
	relocatedPath := core.PathJoin(dir, "relocated.mvlog")

	source, err := Create(ctx, sourcePath)
	if err != nil {
		core.Println(err)
		return
	}
	ref, err := source.PutBytes(ctx, []byte("relocated"), state.PutOptions{})
	if err != nil {
		core.Println(err)
		return
	}
	if err := source.Close(); err != nil {
		core.Println(err)
		return
	}
	// Copy the file to a new path — OpenWithSegmentAlias permits refs
	// whose Segment still names the original (pre-relocation) path.
	read := core.ReadFile(sourcePath)
	if !read.OK {
		core.Println(read.Error())
		return
	}
	if write := core.WriteFile(relocatedPath, read.Value.([]byte), 0o600); !write.OK {
		core.Println(write.Error())
		return
	}

	aliased, err := OpenWithSegmentAlias(ctx, relocatedPath, sourcePath)
	if err != nil {
		core.Println(err)
		return
	}
	defer aliased.Close()

	chunk, err := state.ResolveRefBytes(ctx, aliased, ref)
	core.Println(err == nil, string(chunk.Data))
	// Output: true relocated
}

func ExampleOpenRegionWithSegmentAlias() {
	ctx := context.Background()
	dir, cleanup, ok := exampleFilestoreTempDir()
	if !ok {
		return
	}
	defer cleanup()
	sourcePath := core.PathJoin(dir, "source.mvlog")
	containerPath := core.PathJoin(dir, "session.kv")

	source, err := Create(ctx, sourcePath)
	if err != nil {
		core.Println(err)
		return
	}
	ref, err := source.PutBytes(ctx, []byte("region payload"), state.PutOptions{})
	if err != nil {
		core.Println(err)
		return
	}
	if err := source.Close(); err != nil {
		core.Println(err)
		return
	}
	read := core.ReadFile(sourcePath)
	if !read.OK {
		core.Println(read.Error())
		return
	}
	// Embed the state log inside a larger container file, prefixed by
	// bytes that do not belong to the log itself — this is the shape a
	// KV-cache session file uses in production.
	prefix := []byte("KVST-header")
	sourceBytes := read.Value.([]byte)
	container := append(append([]byte(nil), prefix...), sourceBytes...)
	if write := core.WriteFile(containerPath, container, 0o600); !write.OK {
		core.Println(write.Error())
		return
	}

	store, err := OpenRegionWithSegmentAlias(ctx, containerPath, int64(len(prefix)), int64(len(sourceBytes)), sourcePath)
	if err != nil {
		core.Println(err)
		return
	}
	defer store.Close()

	chunk, err := state.ResolveRefBytes(ctx, store, ref)
	core.Println(err == nil, string(chunk.Data))
	// Output: true region payload
}

func ExampleStore_Path() {
	ctx := context.Background()
	dir, cleanup, ok := exampleFilestoreTempDir()
	if !ok {
		return
	}
	defer cleanup()
	path := core.PathJoin(dir, "path.mvlog")

	store, err := Create(ctx, path)
	if err != nil {
		core.Println(err)
		return
	}
	defer store.Close()

	core.Println(store.Path() == path)
	// Output: true
}

func ExampleStore_ChunkCount() {
	ctx := context.Background()
	dir, cleanup, ok := exampleFilestoreTempDir()
	if !ok {
		return
	}
	defer cleanup()

	store, err := Create(ctx, core.PathJoin(dir, "chunkcount.mvlog"))
	if err != nil {
		core.Println(err)
		return
	}
	defer store.Close()
	if _, err := store.Put(ctx, "hello", state.PutOptions{}); err != nil {
		core.Println(err)
		return
	}

	core.Println(store.ChunkCount())
	// Output: 1
}

func ExampleStore_Close() {
	ctx := context.Background()
	dir, cleanup, ok := exampleFilestoreTempDir()
	if !ok {
		return
	}
	defer cleanup()

	store, err := Create(ctx, core.PathJoin(dir, "close.mvlog"))
	if err != nil {
		core.Println(err)
		return
	}

	err = store.Close()
	core.Println(err == nil)
	// Output: true
}

// exampleFilestoreTempDir provisions a scratch directory shared by every
// Example* function in this package — Example tests receive no *testing.T,
// so there is no t.TempDir() to lean on. Shared across resolve/store/put
// example files, which live in the same package.
func exampleFilestoreTempDir() (string, func(), bool) {
	dirResult := core.MkdirTemp("", "filestore-example-*")
	if !dirResult.OK {
		return "", func() {}, false
	}
	dir := dirResult.Value.(string)
	return dir, func() { core.RemoveAll(dir) }, true
}
