// SPDX-Licence-Identifier: EUPL-1.2
package glm4_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	_ "dappco.re/go/inference/model/glm4"
)

func ExampleConfig() {
	_, ok := model.LookupArch("glm4")
	core.Println(ok) // Output: true
}
