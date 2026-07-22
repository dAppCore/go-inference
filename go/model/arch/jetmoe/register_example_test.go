// SPDX-Licence-Identifier: EUPL-1.2

package jetmoe_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	_ "dappco.re/go/inference/model/arch/jetmoe"
)

func Example() {
	_, ok := model.LookupArch("jetmoe")
	core.Println(ok)
	// Output: true
}
