// SPDX-Licence-Identifier: EUPL-1.2

package spine_test

import (
	"fmt"

	"dappco.re/go/inference/lora"
	"dappco.re/go/inference/spine"
)

func ExampleModelInfoToBundle() {
	info := spine.ModelInfo{Architecture: "gemma3", VocabSize: 262144, NumLayers: 26}
	b := spine.ModelInfoToBundle(info)
	fmt.Println(b.Architecture, b.VocabSize, b.NumLayers)
	// Output: gemma3 262144 26
}

func ExampleModelInfoToMemory() {
	info := spine.ModelInfo{Architecture: "gemma3", HiddenSize: 1152, ContextLength: 8192}
	m := spine.ModelInfoToMemory(info)
	fmt.Println(m.Architecture, m.HiddenSize, m.ContextLength)
	// Output: gemma3 1152 8192
}

func ExampleParserHint() {
	info := spine.ModelInfo{Architecture: "qwen3", Adapter: lora.AdapterInfo{Name: "coder"}}
	hint := spine.ParserHint(info)
	fmt.Println(hint.Architecture, hint.AdapterName)
	// Output: qwen3 coder
}
