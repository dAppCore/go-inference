// SPDX-Licence-Identifier: EUPL-1.2
package nf4_test

import (
	"context"
	core "dappco.re/go"
	"dappco.re/go/inference/model/quant/nf4"
)

func ExampleConvertSnapshot() {
	_, err := nf4.ConvertSnapshot(context.Background(), "", "/tmp/out", nil)
	core.Println(err != nil)
	// Output:
	// true
}
