// SPDX-Licence-Identifier: EUPL-1.2

package safetensors_test

import (
	"fmt"

	"dappco.re/go/inference/model/safetensors"
)

// ExampleFloat32ToBFloat16 rounds a float32 to bfloat16 bits with round-to-nearest-even —
// the scalar building block WidenF16ToBF16 applies to every element of an F16 tensor.
func ExampleFloat32ToBFloat16() {
	fmt.Printf("%#04x\n", safetensors.Float32ToBFloat16(1.0))
	// Output:
	// 0x3f80
}

// ExampleWidenF16ToBF16 converts a raw little-endian F16 tensor payload to the equivalent BF16
// payload, element by element, keeping the same byte length.
func ExampleWidenF16ToBF16() {
	f16 := []byte{0x00, 0x3c} // 1.0 as little-endian IEEE-754 half
	bf16 := safetensors.WidenF16ToBF16(f16)
	fmt.Printf("% x\n", bf16)
	// Output:
	// 80 3f
}

// ExampleDirMapping_WidenF16TensorsToBF16 widens every F16 tensor in a loaded mapping to BF16 in
// place — the pass a loader runs once, after Normalize and before the assembler reads tensors,
// so the byte-native engine's bfloat16_t kernels never misread an F16 pack.
func ExampleDirMapping_WidenF16TensorsToBF16() {
	dm := &safetensors.DirMapping{Tensors: map[string]safetensors.Tensor{
		"model.norm.weight": {Dtype: "F16", Shape: []int{1}, Data: []byte{0x00, 0x3c}},
	}}
	n := dm.WidenF16TensorsToBF16()
	fmt.Println(n, dm.Tensors["model.norm.weight"].Dtype)
	// Output:
	// 1 BF16
}

// ExampleDirMapping_IsWidened reports whether a tensor's Data is a widened (fresh heap BF16)
// buffer rather than a page-aligned shard mmap view — the zero-copy binder's discriminator for
// binding a widened companion tensor resident instead of failing the "not a mapped view" guard.
func ExampleDirMapping_IsWidened() {
	dm := &safetensors.DirMapping{Tensors: map[string]safetensors.Tensor{
		"model.norm.weight": {Dtype: "F16", Shape: []int{1}, Data: []byte{0x00, 0x3c}},
	}}
	dm.WidenF16TensorsToBF16()
	fmt.Println(dm.IsWidened(dm.Tensors["model.norm.weight"].Data))
	// Output:
	// true
}
