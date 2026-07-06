// SPDX-Licence-Identifier: EUPL-1.2

package safetensors_test

import (
	"fmt"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// ExampleLoadMmap memory-maps a safetensors file and Parses it without copying the
// weights: every Tensor.Data views the mapped file directly (on unix; a full-read
// fallback on other platforms keeps the same shape). Close unmaps once every view is
// done.
func ExampleLoadMmap() {
	dir, cleanup := mkTempDir()
	defer cleanup()
	path := core.PathJoin(dir, "model.safetensors")

	blob, err := safetensors.Encode(map[string]safetensors.Tensor{
		"embed": {Dtype: "F32", Shape: []int{2}, Data: []byte{0, 0, 128, 63, 0, 0, 0, 64}},
	})
	if err != nil {
		fmt.Println("encode:", err)
		return
	}
	if err := coreio.Local.Write(path, string(blob)); err != nil {
		fmt.Println("write:", err)
		return
	}

	m, err := safetensors.LoadMmap(path)
	if err != nil {
		fmt.Println("load mmap:", err)
		return
	}
	defer m.Close()
	fmt.Printf("embed: %s %v\n", m.Tensors["embed"].Dtype, m.Tensors["embed"].Shape)
	// Output:
	// embed: F32 [2]
}

// ExampleMapping_Close releases the memory-mapped file. It is safe to call on a nil
// *Mapping, so it is the natural deferred cleanup right after LoadMmap.
func ExampleMapping_Close() {
	dir, cleanup := mkTempDir()
	defer cleanup()
	path := core.PathJoin(dir, "model.safetensors")

	blob, err := safetensors.Encode(map[string]safetensors.Tensor{
		"x": {Dtype: "U8", Shape: []int{1}, Data: []byte{1}},
	})
	if err != nil {
		fmt.Println("encode:", err)
		return
	}
	if err := coreio.Local.Write(path, string(blob)); err != nil {
		fmt.Println("write:", err)
		return
	}

	m, err := safetensors.LoadMmap(path)
	if err != nil {
		fmt.Println("load mmap:", err)
		return
	}
	if err := m.Close(); err != nil {
		fmt.Println("close:", err)
		return
	}
	// Safe on a nil *Mapping too.
	var nilM *safetensors.Mapping
	if err := nilM.Close(); err != nil {
		fmt.Println("nil close:", err)
		return
	}
	fmt.Println("closed")
	// Output:
	// closed
}
