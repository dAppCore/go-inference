// SPDX-Licence-Identifier: EUPL-1.2
package nf4_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/quant/nf4"
)

func ExampleQuantize() {
	tensor, _ := nf4.Quantize([]float32{-1, 0, 1, 0}, []int{2, 2})
	core.Println(len(tensor.Data), len(tensor.Absmax))
	// Output:
	// 2 1
}
func ExampleDequantize() {
	values, _ := nf4.Dequantize(nf4.Tensor{Data: []byte{0x0f}, Absmax: []float32{1}, Shape: []int{2}})
	core.Println(values)
	// Output:
	// [-1 1]
}
