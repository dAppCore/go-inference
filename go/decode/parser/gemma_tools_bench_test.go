// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the shared Gemma 4 tool renderers + parser — the
// declaration render fires once per request that carries a toolset (every
// Claude-Code-style agentic turn re-sends its full tool list), RenderGemmaToolCall
// fires once per prior assistant tool call a stateless client replays, and
// ParseGemmaToolCalls fires on every completion that may carry a call. Per
// AX-11 these are the per-request / per-response hot paths for the two
// providers (anthropic + openai) that both funnel through the neutral ToolDecl
// form, so a fold here lands once for both.
//
// Run:    go test -bench='Benchmark_GemmaTools' -benchmem -run='^$' ./decode/parser

package parser

import "testing"

// Sinks defeat compiler DCE.
var (
	gemmaBenchString  string
	gemmaBenchVisible string
)

// benchGemmaToolset mirrors a realistic agentic toolset: several tools, each
// with a handful of typed+described parameters and a required list — the shape
// a coding agent re-sends on every turn.
func benchGemmaToolset() []ToolDecl {
	return []ToolDecl{
		{
			Name:        "read_file",
			Description: "Reads a file from the local filesystem and returns its contents.",
			Properties: map[string]ToolParam{
				"path":   {Type: "string", Description: "The absolute path to the file to read."},
				"offset": {Type: "integer", Description: "The line number to start reading from."},
				"limit":  {Type: "integer", Description: "The number of lines to read."},
			},
			Required: []string{"path"},
		},
		{
			Name:        "write_file",
			Description: "Writes content to a file, overwriting any existing file.",
			Properties: map[string]ToolParam{
				"path":    {Type: "string", Description: "The absolute path to the file to write."},
				"content": {Type: "string", Description: "The content to write to the file."},
			},
			Required: []string{"path", "content"},
		},
		{
			Name:        "run_command",
			Description: "Executes a shell command and returns its combined output.",
			Properties: map[string]ToolParam{
				"command": {Type: "string", Description: "The command line to execute."},
				"timeout": {Type: "number", Description: "Optional timeout in milliseconds."},
				"detach":  {Type: "boolean", Description: "Run the command in the background."},
			},
			Required: []string{"command"},
		},
		{
			Name:        "search_code",
			Description: "Searches the codebase for a pattern and returns matching lines.",
			Properties: map[string]ToolParam{
				"pattern": {Type: "string", Description: "The regular expression to search for."},
				"glob":    {Type: "string", Description: "An optional glob to filter the files searched."},
			},
			Required: []string{"pattern"},
		},
	}
}

// --- RenderGemmaToolDeclarations -----------------------------------------

func Benchmark_GemmaTools_RenderDeclarations(b *testing.B) {
	tools := benchGemmaToolset()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gemmaBenchString = RenderGemmaToolDeclarations(tools)
	}
}

// --- RenderGemmaToolCall -------------------------------------------------

func Benchmark_GemmaTools_RenderToolCall(b *testing.B) {
	const args = `{"path":"/etc/hosts","offset":0,"limit":200,"flags":["read","utf8"]}`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gemmaBenchString = RenderGemmaToolCall("read_file", args)
	}
}

// --- ParseGemmaToolCalls -------------------------------------------------

func Benchmark_GemmaTools_ParseToolCalls(b *testing.B) {
	text := "Let me read that file for you. " + ToolCallOpenMarker +
		`call:read_file{path:<|"|>/etc/hosts<|"|>,limit:200}` + ToolCallCloseMarker +
		" and I will summarise the result."
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, gemmaBenchVisible = ParseGemmaToolCalls(text)
	}
}
