// SPDX-Licence-Identifier: EUPL-1.2

// tools.go bridges Anthropic tool declarations to Gemma 4's native
// function-calling prompt format. Claude Code sends tools in the Anthropic wire
// shape ({name, description, input_schema}); Gemma 4 was trained on its own
// declaration syntax — <|tool>declaration:NAME{description:<|"|>..<|"|>,
// parameters:{properties:{p:{..,type:<|"|>STRING<|"|>} },required:[<|"|>p<|"|>],
// type:<|"|>OBJECT<|"|>} }<tool|> — with <|tool>, <tool|> and the <|"|> string
// delimiter as real vocab tokens. RenderToolDeclarations produces exactly that so
// the model emits <|tool_call>call:NAME{arg:<|"|>val<|"|>}<tool_call|> in reply
// (see reference/gemma/capabilities-text-function-calling-gemma4.md).
package anthropic

import (
	"slices"

	core "dappco.re/go"
)

// gemmaToolQuote is Gemma 4's string-value delimiter inside tool syntax — one
// vocab token, not a literal double-quote, so string values carrying commas or
// braces don't break the enclosing structure.
const gemmaToolQuote = "<|\"|>"

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

// RenderToolDeclarations renders the tools into Gemma 4's native declaration
// syntax, one <|tool>…<tool|> block per tool, concatenated. Empty when no tools
// are supplied. Property order is sorted for a deterministic prompt (the model
// does not key on declaration order).
func RenderToolDeclarations(tools []Tool) string {
	if len(tools) == 0 {
		return ""
	}
	b := core.NewBuilder()
	for _, t := range tools {
		b.WriteString("<|tool>declaration:")
		b.WriteString(t.Name)
		b.WriteString("{description:")
		b.WriteString(gemmaToolQuote)
		b.WriteString(t.Description)
		b.WriteString(gemmaToolQuote)
		b.WriteString(",parameters:{properties:{")
		for i, name := range sortedPropertyNames(t.InputSchema.Properties) {
			if i > 0 {
				b.WriteString(",")
			}
			p := t.InputSchema.Properties[name]
			b.WriteString(name)
			b.WriteString(":{description:")
			b.WriteString(gemmaToolQuote)
			b.WriteString(p.Description)
			b.WriteString(gemmaToolQuote)
			b.WriteString(",type:")
			b.WriteString(gemmaToolQuote)
			b.WriteString(gemmaSchemaType(p.Type))
			b.WriteString(gemmaToolQuote)
			b.WriteString("} ")
		}
		b.WriteString("},required:[")
		for i, r := range t.InputSchema.Required {
			if i > 0 {
				b.WriteString(",")
			}
			b.WriteString(gemmaToolQuote)
			b.WriteString(r)
			b.WriteString(gemmaToolQuote)
		}
		b.WriteString("],type:")
		b.WriteString(gemmaToolQuote)
		b.WriteString("OBJECT")
		b.WriteString(gemmaToolQuote)
		b.WriteString("} }<tool|>")
	}
	return b.String()
}

// gemmaSchemaType maps a JSON-schema type name to Gemma 4's uppercase form
// (string -> STRING, integer -> INTEGER, …). An unknown type is upper-cased
// as-is so a novel schema still renders rather than dropping the field.
func gemmaSchemaType(t string) string {
	switch core.Lower(t) {
	case "string":
		return "STRING"
	case "integer":
		return "INTEGER"
	case "number":
		return "NUMBER"
	case "boolean":
		return "BOOLEAN"
	case "object":
		return "OBJECT"
	case "array":
		return "ARRAY"
	default:
		if t == "" {
			return "STRING"
		}
		return core.Upper(t)
	}
}

// sortedPropertyNames returns the property keys in a stable order — a JSON
// object is unordered, so sorting keeps the rendered prompt deterministic
// (test-pinnable, cache-friendly).
func sortedPropertyNames(props map[string]ToolProperty) []string {
	names := make([]string, 0, len(props))
	for name := range props {
		names = append(names, name)
	}
	slices.Sort(names)
	return names
}
