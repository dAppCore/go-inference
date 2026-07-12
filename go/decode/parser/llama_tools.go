// SPDX-Licence-Identifier: EUPL-1.2

// llama_tools.go implements Meta Llama 3.1's JSON tool-call prompt format.
// Reference: https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/prompt_format.md
package parser

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

const (
	LlamaToolCallOpenMarker  = "<|python_tag|>"
	LlamaToolCallCloseMarker = "<|eom_id|>"
)

type llamaToolDefinition struct {
	Type     string            `json:"type"`
	Function llamaToolFunction `json:"function"`
}

type llamaToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  llamaToolSchema `json:"parameters"`
}

type llamaToolSchema struct {
	Type       string                       `json:"type"`
	Properties map[string]llamaToolProperty `json:"properties,omitempty"`
	Required   []string                     `json:"required,omitempty"`
}

type llamaToolProperty struct {
	Type        string `json:"type"`
	Description string `json:"description,omitempty"`
}

type llamaToolCall struct {
	Type       string         `json:"type"`
	Name       string         `json:"name"`
	Parameters map[string]any `json:"parameters"`
}

func RenderLlamaToolDeclarations(tools []ToolDecl) string {
	if len(tools) == 0 {
		return ""
	}
	definitions := make([]llamaToolDefinition, len(tools))
	for i, tool := range tools {
		properties := make(map[string]llamaToolProperty, len(tool.Properties))
		for name, property := range tool.Properties {
			properties[name] = llamaToolProperty{Type: property.Type, Description: property.Description}
		}
		definitions[i] = llamaToolDefinition{Type: "function", Function: llamaToolFunction{
			Name: tool.Name, Description: tool.Description,
			Parameters: llamaToolSchema{Type: "object", Properties: properties, Required: tool.Required},
		}}
	}
	return "Environment: ipython\n\nAnswer the user's question by making use of the following functions if needed.\n" +
		"If none of the functions can be used, please say so.\n" +
		"Here is a list of functions in JSON format:\n" + core.JSONMarshalString(definitions) +
		"\nReturn function calls in JSON format."
}

func ParseLlamaToolCalls(text string) ([]inference.ToolCall, string) {
	if !core.Contains(text, LlamaToolCallOpenMarker) {
		return nil, text
	}
	var calls []inference.ToolCall
	visible := core.NewBuilder()
	rest := text
	for {
		open := core.Index(rest, LlamaToolCallOpenMarker)
		if open < 0 {
			visible.WriteString(rest)
			break
		}
		visible.WriteString(rest[:open])
		after := rest[open+len(LlamaToolCallOpenMarker):]
		close := core.Index(after, LlamaToolCallCloseMarker)
		if close < 0 {
			visible.WriteString(rest[open:])
			break
		}
		var wire llamaToolCall
		span := core.Trim(after[:close])
		result := core.JSONUnmarshal([]byte(span), &wire)
		if result.OK && wire.Type == "function" && wire.Name != "" {
			calls = append(calls, inference.ToolCall{Type: "function", Name: wire.Name, ArgumentsJSON: core.JSONMarshalString(wire.Parameters)})
		} else {
			visible.WriteString(rest[open : open+len(LlamaToolCallOpenMarker)+close+len(LlamaToolCallCloseMarker)])
		}
		rest = after[close+len(LlamaToolCallCloseMarker):]
	}
	return calls, core.Trim(visible.String())
}

func ParseToolCalls(architecture, text string) ([]inference.ToolCall, string) {
	if isLlamaToolArchitecture(architecture) {
		return ParseLlamaToolCalls(text)
	}
	return ParseGemmaToolCalls(text)
}

func RenderToolDeclarations(architecture string, tools []ToolDecl) string {
	if isLlamaToolArchitecture(architecture) {
		return RenderLlamaToolDeclarations(tools)
	}
	return RenderGemmaToolDeclarations(tools)
}

func ToolCallOpenMarkerFor(architecture string) string {
	if isLlamaToolArchitecture(architecture) {
		return LlamaToolCallOpenMarker
	}
	return ToolCallOpenMarker
}

func isLlamaToolArchitecture(architecture string) bool {
	return core.Contains(core.Lower(architecture), "llama")
}
