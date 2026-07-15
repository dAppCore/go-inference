// SPDX-Licence-Identifier: EUPL-1.2

package parser

import (
	"testing"

	"dappco.re/go/inference"
)

var llamaBenchCalls []inference.ToolCall
var llamaBenchText string

func BenchmarkParseLlamaToolCalls(b *testing.B) {
	text := `<|python_tag|>{"type":"function","name":"weather","parameters":{"city":"London"}}<|eom_id|>`
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		llamaBenchCalls, llamaBenchText = ParseLlamaToolCalls(text)
	}
}

func BenchmarkRenderLlamaToolDeclarations(b *testing.B) {
	tools := []ToolDecl{{Name: "weather", Properties: map[string]ToolParam{"city": {Type: "string"}}, Required: []string{"city"}}}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		llamaBenchText = RenderLlamaToolDeclarations(tools)
	}
}
