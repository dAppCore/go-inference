// SPDX-Licence-Identifier: EUPL-1.2

package parser

import core "dappco.re/go"

func ExampleParseLlamaToolCalls() {
	calls, visible := ParseLlamaToolCalls(`<|python_tag|>{"type":"function","name":"weather","parameters":{"city":"London"}}<|eom_id|>`)
	core.Println(len(calls), calls[0].Name, visible)
	// Output: 1 weather
}

func ExampleRenderLlamaToolDeclarations() {
	text := RenderLlamaToolDeclarations([]ToolDecl{{Name: "weather"}})
	core.Println(core.Contains(text, `"name":"weather"`))
	// Output: true
}
