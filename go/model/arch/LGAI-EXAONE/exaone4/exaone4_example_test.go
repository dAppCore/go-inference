// SPDX-Licence-Identifier: EUPL-1.2
package exaone4_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	_ "dappco.re/go/inference/model/arch/LGAI-EXAONE/exaone4"
)

func ExampleConfig() {
	_, ok := model.LookupArch("exaone4")
	core.Println(ok) // Output: true
}
