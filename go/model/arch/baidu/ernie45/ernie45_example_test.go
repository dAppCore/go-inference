// SPDX-Licence-Identifier: EUPL-1.2
package ernie45_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	_ "dappco.re/go/inference/model/arch/baidu/ernie45"
)

func ExampleConfig() {
	_, ok := model.LookupArch("ernie4_5")
	core.Println(ok) // Output: true
}
