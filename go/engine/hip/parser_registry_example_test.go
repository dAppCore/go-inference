// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

func ExampleParserRegistry_ParseReasoning() {
	result, _ := NewParserRegistry("qwen3").ParseReasoning(nil, "<think>plan</think>answer")
	core.Println(result.VisibleText)
	// Output: answer
}

func ExampleParserRegistry_ParseTools() {
	result, _ := NewParserRegistry("mistral").ParseTools(nil, `<tool_call>{"name":"search","arguments":{"q":"rocm"}}</tool_call>`)
	core.Println(result.Calls[0].Name)
	// Output: search
}

func Example_rocmModel_ParseReasoning() {
	model := &rocmModel{modelInfo: inference.ModelInfo{Architecture: "gemma4_text"}}
	result, _ := model.ParseReasoning(nil, "<start_of_turn>analysis\nplan<end_of_turn>answer")
	core.Println(result.VisibleText)
	// Output: answer
}

func Example_rocmModel_ParseTools() {
	model := &rocmModel{modelInfo: inference.ModelInfo{Architecture: "mistral"}}
	result, _ := model.ParseTools(nil, `<tool_call>{"name":"search","arguments":{"q":"rocm"}}</tool_call>`)
	core.Println(result.Calls[0].Name)
	// Output: search
}
