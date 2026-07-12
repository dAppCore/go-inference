// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for Anthropic tool-declaration rendering. Per AX-11 —
// RenderToolDeclarations runs once per Messages request that carries tools
// (Claude Code sends its full tool set on every turn), converting the Anthropic
// wire shape into the neutral ToolDecl and rendering it through the shared
// Gemma renderer. The bench records the per-request conversion + render cost
// across a realistic multi-tool set.
//
// Run:    go test -bench=RenderToolDeclarations -benchmem -run='^$' .
package anthropic

import "testing"

var toolsBenchSinkString string

// benchTools is a realistic Claude Code tool set: a few functions, each with a
// handful of typed parameters and a required list.
func benchTools() []Tool {
	return []Tool{
		{
			Name:        "read_file",
			Description: "Read a file from the workspace",
			InputSchema: ToolInputSchema{
				Type: "object",
				Properties: map[string]ToolProperty{
					"path":   {Type: "string", Description: "Absolute path to the file"},
					"offset": {Type: "integer", Description: "Line to start from"},
					"limit":  {Type: "integer", Description: "Number of lines to read"},
				},
				Required: []string{"path"},
			},
		},
		{
			Name:        "search_code",
			Description: "Search the codebase for a pattern",
			InputSchema: ToolInputSchema{
				Type: "object",
				Properties: map[string]ToolProperty{
					"pattern": {Type: "string", Description: "Regex to search for"},
					"glob":    {Type: "string", Description: "File glob to restrict the search"},
				},
				Required: []string{"pattern"},
			},
		},
		{
			Name:        "run_command",
			Description: "Run a shell command",
			InputSchema: ToolInputSchema{
				Type: "object",
				Properties: map[string]ToolProperty{
					"command": {Type: "string", Description: "The command to run"},
					"timeout": {Type: "integer", Description: "Timeout in milliseconds"},
				},
				Required: []string{"command"},
			},
		},
	}
}

func BenchmarkAnthropic_RenderToolDeclarations(b *testing.B) {
	tools := benchTools()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toolsBenchSinkString = RenderToolDeclarations(tools)
	}
}
