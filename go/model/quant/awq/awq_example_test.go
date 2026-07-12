// SPDX-Licence-Identifier: EUPL-1.2

package awq_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/quant/awq"
)

func ExampleQuantize() {
	values := make([]float32, 32*32)
	for i := range values {
		values[i] = float32(i%9-4) / 10
	}
	tensor, err := awq.Quantize(values, 32, 32, awq.Options{Bits: 4, GroupSize: 32, ZeroPoint: true})
	core.Println(err == nil, tensor.QWeightShape, tensor.QZerosShape, tensor.ScalesShape)
	// Output:
	// true [32 4] [1 4] [1 32]
}

func ExampleDequantize() {
	values := make([]float32, 32*32)
	tensor, _ := awq.Quantize(values, 32, 32, awq.Options{Bits: 4, GroupSize: 32, ZeroPoint: true})
	restored, err := awq.Dequantize(tensor)
	core.Println(err == nil, len(restored))
	// Output:
	// true 1024
}

func ExampleOptions() {
	options := awq.Options{Bits: 4, GroupSize: 128, ZeroPoint: true}
	core.Println(options.Bits, options.GroupSize, options.ZeroPoint)
	// Output:
	// 4 128 true
}

func ExampleTensor() {
	tensor := awq.Tensor{Shape: [2]int{32, 128}, Bits: 4, GroupSize: 128}
	core.Println(tensor.Shape, tensor.Bits)
	// Output:
	// [32 128] 4
}
