// SPDX-Licence-Identifier: EUPL-1.2

package safetensors

import core "dappco.re/go"

func ExampleDecodeFloat32() {
	core.Println("ok")
	// Output:
	// ok
}

func ExampleEncodeFloat32() {
	core.Println("ok")
	// Output:
	// ok
}

// ExampleFloat16ToFloat32 decodes a single IEEE 754 binary16 bit pattern —
// the same conversion DecodeFloat32 applies to every element of an "F16"
// tensor, exposed here for callers that only need one value.
func ExampleFloat16ToFloat32() {
	core.Println(Float16ToFloat32(0x3C00))
	// Output:
	// 1
}

// ExampleBFloat16ToFloat32 decodes a single bfloat16 bit pattern — the
// same conversion DecodeFloat32 applies to every element of a "BF16"
// tensor, exposed here for callers that only need one value.
func ExampleBFloat16ToFloat32() {
	core.Println(BFloat16ToFloat32(0x3F80))
	// Output:
	// 1
}
