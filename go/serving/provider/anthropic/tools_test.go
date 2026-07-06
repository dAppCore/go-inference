// SPDX-Licence-Identifier: EUPL-1.2

package anthropic

import "testing"

// TestRenderToolDeclarations_MatchesGemmaDoc pins the renderer against the exact
// declaration in reference/gemma/capabilities-text-function-calling-gemma4.md —
// the format Gemma 4 was trained on. Divergence here means the model may not
// recognise the tool, so this string is the contract.
func TestRenderToolDeclarations_MatchesGemmaDoc(t *testing.T) {
	tools := []Tool{{
		Name:        "get_current_temperature",
		Description: "Gets the current temperature for a given location.",
		InputSchema: ToolInputSchema{
			Type: "object",
			Properties: map[string]ToolProperty{
				"location": {Type: "string", Description: "The city name, e.g. San Francisco"},
			},
			Required: []string{"location"},
		},
	}}
	q := gemmaToolQuote
	want := "<|tool>declaration:get_current_temperature{description:" + q +
		"Gets the current temperature for a given location." + q +
		",parameters:{properties:{location:{description:" + q +
		"The city name, e.g. San Francisco" + q + ",type:" + q + "STRING" + q + "} }," +
		"required:[" + q + "location" + q + "],type:" + q + "OBJECT" + q + "} }<tool|>"
	if got := RenderToolDeclarations(tools); got != want {
		t.Fatalf("RenderToolDeclarations mismatch:\n got: %s\nwant: %s", got, want)
	}
}

// TestRenderToolDeclarations_Empty pins that no tools render to no prompt text.
func TestRenderToolDeclarations_Empty(t *testing.T) {
	if got := RenderToolDeclarations(nil); got != "" {
		t.Fatalf("empty tools = %q, want empty", got)
	}
}

// TestRenderToolDeclarations_MultiTool pins that multiple tools each get their
// own <|tool>…<tool|> block.
func TestRenderToolDeclarations_MultiTool(t *testing.T) {
	tools := []Tool{
		{Name: "a", InputSchema: ToolInputSchema{Type: "object"}},
		{Name: "b", InputSchema: ToolInputSchema{Type: "object"}},
	}
	got := RenderToolDeclarations(tools)
	if n := countSubstr(got, "<|tool>declaration:"); n != 2 {
		t.Fatalf("multi-tool render had %d declarations, want 2: %s", n, got)
	}
}

// TestGemmaSchemaType pins the JSON-schema -> Gemma uppercase type mapping,
// including the empty default and the unknown-type passthrough.
func TestGemmaSchemaType(t *testing.T) {
	cases := map[string]string{
		"string": "STRING", "integer": "INTEGER", "number": "NUMBER",
		"boolean": "BOOLEAN", "object": "OBJECT", "array": "ARRAY",
		"": "STRING", "geo": "GEO",
	}
	for in, want := range cases {
		if got := gemmaSchemaType(in); got != want {
			t.Fatalf("gemmaSchemaType(%q) = %q, want %q", in, got, want)
		}
	}
}

func countSubstr(s, sub string) int {
	n := 0
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			n++
		}
	}
	return n
}
