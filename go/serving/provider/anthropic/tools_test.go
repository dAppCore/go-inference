// SPDX-Licence-Identifier: EUPL-1.2

package anthropic

import "testing"

// TestTools_RenderToolDeclarations_Good pins the renderer against the exact
// declaration in reference/gemma/capabilities-text-function-calling-gemma4.md —
// the format Gemma 4 was trained on. Divergence here means the model may not
// recognise the tool, so this string is the contract.
func TestTools_RenderToolDeclarations_Good(t *testing.T) {
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

// TestTools_RenderToolDeclarations_Bad pins that no tools render to no prompt
// text (the edge/invalid-ish input this API accepts rather than rejecting).
func TestTools_RenderToolDeclarations_Bad(t *testing.T) {
	if got := RenderToolDeclarations(nil); got != "" {
		t.Fatalf("empty tools = %q, want empty", got)
	}
}

// TestTools_RenderToolDeclarations_Ugly pins that multiple tools each get
// their own <|tool>…<tool|> block.
func TestTools_RenderToolDeclarations_Ugly(t *testing.T) {
	tools := []Tool{
		{Name: "a", InputSchema: ToolInputSchema{Type: "object"}},
		{Name: "b", InputSchema: ToolInputSchema{Type: "object"}},
	}
	got := RenderToolDeclarations(tools)
	if n := countSubstr(got, "<|tool>declaration:"); n != 2 {
		t.Fatalf("multi-tool render had %d declarations, want 2: %s", n, got)
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
