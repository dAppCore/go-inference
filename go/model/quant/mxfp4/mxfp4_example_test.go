// SPDX-Licence-Identifier: EUPL-1.2
package mxfp4_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/quant/mxfp4"
)

func ExampleQuantize() {
	tensor, _ := mxfp4.Quantize([]float32{-1, 0, 1, 0}, []int{2, 2})
	core.Println(len(tensor.Data), len(tensor.Scale))
	// Output:
	// 2 1
}
func ExampleDequantize() {
	values, _ := mxfp4.Dequantize(mxfp4.Tensor{Data: []byte{0xC4}, Scale: []float32{1}, Shape: []int{2}})
	core.Println(values)
	// Output:
	// [-2 2]
}
