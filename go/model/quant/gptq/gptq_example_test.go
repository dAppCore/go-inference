// SPDX-Licence-Identifier: EUPL-1.2

package gptq_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/quant/gptq"
)

func ExampleQuantize() {
	values := make([]float32, 32*32)
	for i := range values {
		values[i] = float32(i%9-4) / 10
	}
	tensor, err := gptq.Quantize(values, 32, 32, gptq.Options{Bits: 4, GroupSize: 32, Symmetric: true})
	core.Println(err == nil, tensor.QWeightShape, tensor.QZerosShape, tensor.ScalesShape)
	// Output:
	// true [4 32] [1 4] [1 32]
}

func ExampleDequantize() {
	values := make([]float32, 32*32)
	tensor, _ := gptq.Quantize(values, 32, 32, gptq.Options{Bits: 4, GroupSize: 32, Symmetric: true})
	restored, err := gptq.Dequantize(tensor)
	core.Println(err == nil, len(restored))
	// Output:
	// true 1024
}

func ExampleOptions() {
	options := gptq.Options{Bits: 4, GroupSize: 128, Symmetric: true}
	core.Println(options.Bits, options.GroupSize, options.Symmetric)
	// Output:
	// 4 128 true
}

func ExampleTensor() {
	tensor := gptq.Tensor{Shape: [2]int{32, 128}, Bits: 4, GroupSize: 128}
	core.Println(tensor.Shape, tensor.Bits)
	// Output:
	// [32 128] 4
}
