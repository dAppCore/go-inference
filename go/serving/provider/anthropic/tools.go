// SPDX-Licence-Identifier: EUPL-1.2

// tools.go bridges Anthropic tool declarations to Gemma 4's native
// function-calling prompt format. Claude Code sends tools in the Anthropic wire
// shape ({name, description, input_schema}); the Gemma 4 declaration syntax it
// renders into is shared with the OpenAI provider via
// decode/parser.RenderGemmaToolDeclarations, so the format has one home. The
// model then emits <|tool_call>call:NAME{arg:<|"|>val<|"|>}<tool_call|> in reply
// (see reference/gemma/capabilities-text-function-calling-gemma4.md).
package anthropic

import (
	"dappco.re/go/inference/decode/parser"
)

// gemmaToolQuote is Gemma 4's string-value delimiter inside tool syntax (one
// vocab token, not a literal quote) — the shared marker owned by the parser
// grammar, aliased here for the render tests.
const gemmaToolQuote = parser.ToolArgQuoteMarker

// Tool is one Anthropic tool declaration (a function the model may call).
type Tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema ToolInputSchema `json:"input_schema"`
}

// ToolInputSchema is the JSON-schema object describing a tool's parameters.
type ToolInputSchema struct {
	Type       string                  `json:"type"`
	Properties map[string]ToolProperty `json:"properties,omitempty"`
	Required   []string                `json:"required,omitempty"`
}

// ToolProperty is one parameter's schema (its type + description).
type ToolProperty struct {
	Type        string `json:"type"`
	Description string `json:"description,omitempty"`
}

// RenderToolDeclarations converts the Anthropic tools into the neutral ToolDecl
// shape and renders them through the shared Gemma 4 renderer.
func RenderToolDeclarations(tools []Tool) string {
	if len(tools) == 0 {
		return ""
	}
	decls := make([]parser.ToolDecl, len(tools))
	for i, t := range tools {
		props := make(map[string]parser.ToolParam, len(t.InputSchema.Properties))
		for name, p := range t.InputSchema.Properties {
			props[name] = parser.ToolParam{Type: p.Type, Description: p.Description}
		}
		decls[i] = parser.ToolDecl{
			Name:        t.Name,
			Description: t.Description,
			Properties:  props,
			Required:    t.InputSchema.Required,
		}
	}
	return parser.RenderGemmaToolDeclarations(decls)
}
