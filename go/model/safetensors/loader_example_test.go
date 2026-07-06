// SPDX-Licence-Identifier: EUPL-1.2

package safetensors_test

import (
	"encoding/binary"
	"fmt"
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// ExampleParse decodes an in-memory safetensors blob (the 8-byte little-endian
// header length, the JSON header, then the tensor data) straight from bytes — no
// file involved. Encode below builds the matching blob.
func ExampleParse() {
	payload := make([]byte, 0, 2*4)
	for _, v := range []float32{1, 2} {
		payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(v))
	}
	header := `{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}`
	blob := make([]byte, 8+len(header)+len(payload))
	binary.LittleEndian.PutUint64(blob[:8], uint64(len(header)))
	copy(blob[8:], header)
	copy(blob[8+len(header):], payload)

	tensors, err := safetensors.Parse(blob)
	if err != nil {
		fmt.Println("parse:", err)
		return
	}
	fmt.Printf("weight: %s %v\n", tensors["weight"].Dtype, tensors["weight"].Shape)
	// Output:
	// weight: F32 [2]
}

// ExampleEncode is the inverse of Parse: tensors → blob, laid out in
// deterministic sorted-name order so the same tensor set always encodes to the
// same bytes.
func ExampleEncode() {
	blob, err := safetensors.Encode(map[string]safetensors.Tensor{
		"weight": {Dtype: "F32", Shape: []int{1}, Data: []byte{0, 0, 128, 63}},
	})
	if err != nil {
		fmt.Println("encode:", err)
		return
	}
	back, err := safetensors.Parse(blob)
	if err != nil {
		fmt.Println("parse:", err)
		return
	}
	fmt.Printf("round-trip dtype=%s shape=%v\n", back["weight"].Dtype, back["weight"].Shape)
	// Output:
	// round-trip dtype=F32 shape=[1]
}

// ExampleLoad reads a safetensors file straight off disk — the whole-file
// counterpart to LoadMmap, for callers that don't need the zero-copy mapping.
func ExampleLoad() {
	dir, cleanup := mkTempDir()
	defer cleanup()
	path := core.PathJoin(dir, "model.safetensors")

	blob, err := safetensors.Encode(map[string]safetensors.Tensor{
		"bias": {Dtype: "F32", Shape: []int{1}, Data: []byte{0, 0, 0, 64}},
	})
	if err != nil {
		fmt.Println("encode:", err)
		return
	}
	if result := core.WriteFile(path, blob, 0o644); !result.OK {
		fmt.Println("write:", result.Value)
		return
	}

	tensors, err := safetensors.Load(path)
	if err != nil {
		fmt.Println("load:", err)
		return
	}
	fmt.Printf("bias: %s %v\n", tensors["bias"].Dtype, tensors["bias"].Shape)
	// Output:
	// bias: F32 [1]
}
