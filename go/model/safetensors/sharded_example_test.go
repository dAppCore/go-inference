// SPDX-Licence-Identifier: EUPL-1.2

package safetensors_test

import (
	"fmt"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// ExampleLoadDir reads a checkpoint directory holding a single model.safetensors (no
// index) — the same shape a sharded checkpoint returns once merged, so callers assemble
// the model identically either way.
func ExampleLoadDir() {
	dir, cleanup := mkTempDir()
	defer cleanup()

	blob, err := safetensors.Encode(map[string]safetensors.Tensor{
		"bias": {Dtype: "F32", Shape: []int{1}, Data: []byte{0, 0, 0, 64}},
	})
	if err != nil {
		fmt.Println("encode:", err)
		return
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(blob)); err != nil {
		fmt.Println("write:", err)
		return
	}

	tensors, err := safetensors.LoadDir(dir)
	if err != nil {
		fmt.Println("load dir:", err)
		return
	}
	fmt.Printf("bias: %s %v\n", tensors["bias"].Dtype, tensors["bias"].Shape)
	// Output:
	// bias: F32 [1]
}

// ExampleLoadDirMmap is LoadDir's zero-copy sibling: it memory-maps the checkpoint
// directory instead of reading it into the heap. Same directory layouts, same merged
// Tensors map — Close (via DirMapping) unmaps every shard.
func ExampleLoadDirMmap() {
	dir, cleanup := mkTempDir()
	defer cleanup()

	blob, err := safetensors.Encode(map[string]safetensors.Tensor{
		"weight": {Dtype: "F32", Shape: []int{1}, Data: []byte{0, 0, 128, 63}},
	})
	if err != nil {
		fmt.Println("encode:", err)
		return
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(blob)); err != nil {
		fmt.Println("write:", err)
		return
	}

	dm, err := safetensors.LoadDirMmap(dir)
	if err != nil {
		fmt.Println("load dir mmap:", err)
		return
	}
	defer dm.Close()
	fmt.Printf("shards=%d weight=%s %v\n", len(dm.Shards), dm.Tensors["weight"].Dtype, dm.Tensors["weight"].Shape)
	// Output:
	// shards=1 weight=F32 [1]
}

// ExampleDirMapping_Close unmaps every shard a DirMapping holds. It is safe to call on a
// nil *DirMapping, so it is the natural deferred cleanup right after LoadDirMmap.
func ExampleDirMapping_Close() {
	dir, cleanup := mkTempDir()
	defer cleanup()

	blob, err := safetensors.Encode(map[string]safetensors.Tensor{
		"x": {Dtype: "U8", Shape: []int{1}, Data: []byte{1}},
	})
	if err != nil {
		fmt.Println("encode:", err)
		return
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(blob)); err != nil {
		fmt.Println("write:", err)
		return
	}

	dm, err := safetensors.LoadDirMmap(dir)
	if err != nil {
		fmt.Println("load dir mmap:", err)
		return
	}
	if err := dm.Close(); err != nil {
		fmt.Println("close:", err)
		return
	}
	// Safe on a nil *DirMapping too.
	var nilDM *safetensors.DirMapping
	if err := nilDM.Close(); err != nil {
		fmt.Println("nil close:", err)
		return
	}
	fmt.Println("closed")
	// Output:
	// closed
}
