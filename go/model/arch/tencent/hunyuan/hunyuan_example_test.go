// SPDX-Licence-Identifier: EUPL-1.2
package hunyuan_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	_ "dappco.re/go/inference/model/arch/tencent/hunyuan"
)

func ExampleConfig() {
	_, ok := model.LookupArch("hunyuan_v1_dense")
	core.Println(ok) // Output: true
}
