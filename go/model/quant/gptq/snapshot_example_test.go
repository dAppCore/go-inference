// SPDX-Licence-Identifier: EUPL-1.2

package gptq_test

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference/model/quant/gptq"
)

func ExampleConvertSnapshot() {
	_, err := gptq.ConvertSnapshot(context.Background(), "", "/tmp/out", gptq.Options{}, nil)
	core.Println(err != nil)
	// Output:
	// true
}

func ExampleResult() {
	result := gptq.Result{TensorCount: 4, QuantizedWeights: 1}
	core.Println(result.TensorCount, result.QuantizedWeights)
	// Output:
	// 4 1
}
