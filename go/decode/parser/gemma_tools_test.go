// SPDX-Licence-Identifier: EUPL-1.2

package parser

import "testing"

// TestParseGemmaToolCalls_DocExample pins the reference call from the gemma4
// function-calling doc: one string argument, no leftover visible text.
func TestParseGemmaToolCalls_DocExample(t *testing.T) {
	text := ToolCallOpenMarker + "call:get_current_temperature{location:" +
		ToolArgQuoteMarker + "London" + ToolArgQuoteMarker + "}" + ToolCallCloseMarker
	calls, visible := ParseGemmaToolCalls(text)
	if len(calls) != 1 {
		t.Fatalf("calls = %d, want 1", len(calls))
	}
	if calls[0].Name != "get_current_temperature" {
		t.Fatalf("name = %q, want get_current_temperature", calls[0].Name)
	}
	if calls[0].ArgumentsJSON != `{"location":"London"}` {
		t.Fatalf("args = %q, want {\"location\":\"London\"}", calls[0].ArgumentsJSON)
	}
	if visible != "" {
		t.Fatalf("visible = %q, want empty", visible)
	}
}

// TestParseGemmaToolCalls_MixedArgs pins the value typing: a <|"|>-quoted value
// stays a string (and may carry a comma the bare split would have broken), while
// bare values become JSON number / boolean literals. Leading prose survives as
// the trimmed visible text.
func TestParseGemmaToolCalls_MixedArgs(t *testing.T) {
	text := "Sure. " + ToolCallOpenMarker + "call:f{a:" + ToolArgQuoteMarker + "x,y" +
		ToolArgQuoteMarker + ",n:5,b:true}" + ToolCallCloseMarker
	calls, visible := ParseGemmaToolCalls(text)
	if len(calls) != 1 {
		t.Fatalf("calls = %d, want 1", len(calls))
	}
	if calls[0].ArgumentsJSON != `{"a":"x,y","n":5,"b":true}` {
		t.Fatalf("args = %q, want the string/number/bool mix with the comma preserved", calls[0].ArgumentsJSON)
	}
	if visible != "Sure." {
		t.Fatalf("visible = %q, want \"Sure.\"", visible)
	}
}

// TestParseGemmaToolCalls_NoCall pins the fast path: plain prose returns
// unchanged with no calls.
func TestParseGemmaToolCalls_NoCall(t *testing.T) {
	calls, visible := ParseGemmaToolCalls("just a normal answer")
	if calls != nil {
		t.Fatalf("calls = %v, want nil", calls)
	}
	if visible != "just a normal answer" {
		t.Fatalf("visible = %q, want the input unchanged", visible)
	}
}

// TestParseGemmaToolCalls_NestedArgs pins the recursive arg parse: nested
// {objects}, [arrays], arrays-of-objects, and a <|"|>-quoted value carrying a
// comma inside a nested object all round-trip to structured JSON (not a
// stringified blob with leaked markers).
func TestParseGemmaToolCalls_NestedArgs(t *testing.T) {
	q := ToolArgQuoteMarker
	cases := []struct{ inner, want string }{
		{"filter:{status:" + q + "open" + q + "}", `{"filter":{"status":"open"}}`},
		{"tags:[" + q + "a" + q + "," + q + "b" + q + "]", `{"tags":["a","b"]}`},
		{"items:[{id:1},{id:2}]", `{"items":[{"id":1},{"id":2}]}`},
		{"loc:{lat:1.5,label:" + q + "x,y" + q + "}", `{"loc":{"lat":1.5,"label":"x,y"}}`},
	}
	for _, c := range cases {
		calls, _ := ParseGemmaToolCalls(ToolCallOpenMarker + "call:f{" + c.inner + "}" + ToolCallCloseMarker)
		if len(calls) != 1 {
			t.Fatalf("%q: no call parsed", c.inner)
		}
		if calls[0].ArgumentsJSON != c.want {
			t.Fatalf("nested args for %q =\n got %q\nwant %q", c.inner, calls[0].ArgumentsJSON, c.want)
		}
	}
}

// TestRenderGemmaToolDeclarations pins the shared renderer against the exact
// declaration in the gemma4 function-calling reference — the format the model
// was trained on, so both providers must produce it byte-for-byte.
func TestRenderGemmaToolDeclarations(t *testing.T) {
	tools := []ToolDecl{{
		Name:        "get_current_temperature",
		Description: "Gets the current temperature for a given location.",
		Properties: map[string]ToolParam{
			"location": {Type: "string", Description: "The city name, e.g. San Francisco"},
		},
		Required: []string{"location"},
	}}
	q := ToolArgQuoteMarker
	want := "<|tool>declaration:get_current_temperature{description:" + q +
		"Gets the current temperature for a given location." + q +
		",parameters:{properties:{location:{description:" + q +
		"The city name, e.g. San Francisco" + q + ",type:" + q + "STRING" + q + "} }," +
		"required:[" + q + "location" + q + "],type:" + q + "OBJECT" + q + "} }<tool|>"
	if got := RenderGemmaToolDeclarations(tools); got != want {
		t.Fatalf("RenderGemmaToolDeclarations mismatch:\n got: %s\nwant: %s", got, want)
	}
	if RenderGemmaToolDeclarations(nil) != "" {
		t.Fatal("no tools should render empty")
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
