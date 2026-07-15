// SPDX-Licence-Identifier: EUPL-1.2
package fp8_test

import (
	"context"
	core "dappco.re/go"
	"dappco.re/go/inference/model/quant/fp8"
)

func ExampleConvertSnapshot() {
	_, err := fp8.ConvertSnapshot(context.Background(), "", "/tmp/out", nil)
	core.Println(err != nil)
	// Output:
	// true
}
