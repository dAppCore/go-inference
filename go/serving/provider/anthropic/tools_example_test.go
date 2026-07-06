// SPDX-Licence-Identifier: EUPL-1.2

package anthropic

import core "dappco.re/go"

func ExampleRenderToolDeclarations() {
	tools := []Tool{{
		Name:        "get_current_temperature",
		Description: "Gets the current temperature for a given location.",
		InputSchema: ToolInputSchema{
			Type: "object",
			Properties: map[string]ToolProperty{
				"location": {Type: "string", Description: "The city name"},
			},
			Required: []string{"location"},
		},
	}}
	decl := RenderToolDeclarations(tools)
	core.Println(core.Contains(decl, "get_current_temperature"))
	// Output:
	// true
}
