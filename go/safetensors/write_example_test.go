// SPDX-Licence-Identifier: EUPL-1.2

package safetensors_test

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/safetensors"
)

// ExampleWriteSubset writes a chosen subset of tensors from an indexed
// source file into a fresh safetensors container without loading whole
// tensors into memory — payloads stream through bounded chunks. Here a
// two-tensor source is indexed and only "alpha" is written to the subset
// file, which is then read back to confirm the value survives bit-exact.
func ExampleWriteSubset() {
	dir, cleanup := mkTempDir()
	defer cleanup()
	src := core.PathJoin(dir, "src.safetensors")
	dst := core.PathJoin(dir, "subset.safetensors")

	// Source: two F32 tensors, header sorted by name (alpha < beta).
	alpha := []float32{1, 2, 3}
	beta := []float32{-4.5, 1024.25}
	payload := make([]byte, 0, (len(alpha)+len(beta))*4)
	for _, v := range alpha {
		payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(v))
	}
	for _, v := range beta {
		payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(v))
	}
	header := `{"alpha":{"dtype":"F32","shape":[3],"data_offsets":[0,12]},` +
		`"beta":{"dtype":"F32","shape":[2],"data_offsets":[12,20]}}`
	if err := buildSafetensors(src, header, payload); err != nil {
		fmt.Println("build:", err)
		return
	}

	index, err := safetensors.ReadIndex(src)
	if err != nil {
		fmt.Println("read index:", err)
		return
	}
	if err := safetensors.WriteSubset(context.Background(), dst,
		[]safetensors.TensorRef{index.Tensors["alpha"]}); err != nil {
		fmt.Println("write subset:", err)
		return
	}

	back, err := safetensors.ReadIndex(dst)
	if err != nil {
		fmt.Println("read back:", err)
		return
	}
	values, err := safetensors.ReadRefValues(back.Tensors["alpha"])
	if err != nil {
		fmt.Println("read values:", err)
		return
	}
	fmt.Printf("subset tensors: %v\n", back.Names)
	fmt.Printf("alpha values: %v\n", values)
	// Output:
	// subset tensors: [alpha]
	// alpha values: [1 2 3]
}
