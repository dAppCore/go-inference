// SPDX-Licence-Identifier: EUPL-1.2
package fp8_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/quant/fp8"
)

func ExampleQuantize() {
	tensor, _ := fp8.Quantize([]float32{-1, 0, 1})
	core.Println(len(tensor.Data), tensor.Scale > 0)
	// Output:
	// 3 true
}
func ExampleDequantize() {
	values, _ := fp8.Dequantize(fp8.Tensor{Data: []byte{0x38}, Scale: 1})
	core.Println(values[0])
	// Output:
	// 1
}
