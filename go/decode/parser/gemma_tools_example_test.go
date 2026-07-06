// SPDX-Licence-Identifier: EUPL-1.2

package parser

import core "dappco.re/go"

func ExampleRenderGemmaToolDeclarations() {
	tools := []ToolDecl{{
		Name:        "get_weather",
		Description: "Gets the current weather.",
		Properties: map[string]ToolParam{
			"city": {Type: "string", Description: "the city name"},
		},
		Required: []string{"city"},
	}}
	core.Println(RenderGemmaToolDeclarations(tools))
	// Output: <|tool>declaration:get_weather{description:<|"|>Gets the current weather.<|"|>,parameters:{properties:{city:{description:<|"|>the city name<|"|>,type:<|"|>STRING<|"|>} },required:[<|"|>city<|"|>],type:<|"|>OBJECT<|"|>} }<tool|>
}

func ExampleRenderGemmaToolCall() {
	core.Println(RenderGemmaToolCall("get_weather", `{"city":"Paris"}`))
	// Output: <|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>
}

func ExampleParseGemmaToolCalls() {
	text := ToolCallOpenMarker + "call:get_weather{city:" + ToolArgQuoteMarker + "Paris" + ToolArgQuoteMarker + "}" + ToolCallCloseMarker
	calls, visible := ParseGemmaToolCalls(text)
	core.Println(calls[0].Name, calls[0].ArgumentsJSON, visible == "")
	// Output: get_weather {"city":"Paris"} true
}
