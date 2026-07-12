// SPDX-Licence-Identifier: EUPL-1.2

package awq_test

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference/model/quant/awq"
)

func ExampleConvertSnapshot() {
	_, err := awq.ConvertSnapshot(context.Background(), "", "/tmp/out", awq.Options{}, nil)
	core.Println(err != nil)
	// Output:
	// true
}

func ExampleResult() {
	result := awq.Result{TensorCount: 3, QuantizedWeights: 1}
	core.Println(result.TensorCount, result.QuantizedWeights)
	// Output:
	// 3 1
}
