// SPDX-Licence-Identifier: EUPL-1.2

package parser

import core "dappco.re/go"

func ExampleNormaliseKey() {
	core.Println(NormaliseKey("Qwen-3.5"))
	// Output: qwen_3_5
}

func ExampleFamily() {
	core.Println(Family(Hint{Architecture: "qwen3"}))
	// Output: qwen
}
