// SPDX-Licence-Identifier: EUPL-1.2

package parser

import "testing"

// TestGemmaTools_ParseGemmaToolCalls_Good pins the reference call from the
// gemma4 function-calling doc: one string argument, no leftover visible text.
func TestGemmaTools_ParseGemmaToolCalls_Good(t *testing.T) {
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

// TestGemmaTools_ParseGemmaToolCalls_Bad pins the fast path: plain prose
// returns unchanged with no calls.
func TestGemmaTools_ParseGemmaToolCalls_Bad(t *testing.T) {
	calls, visible := ParseGemmaToolCalls("just a normal answer")
	if calls != nil {
		t.Fatalf("calls = %v, want nil", calls)
	}
	if visible != "just a normal answer" {
		t.Fatalf("visible = %q, want the input unchanged", visible)
	}
}

// TestGemmaTools_ParseGemmaToolCalls_Ugly pins the recursive arg parse: nested
// {objects}, [arrays], arrays-of-objects, and a <|"|>-quoted value carrying a
// comma inside a nested object all round-trip to structured JSON (not a
// stringified blob with leaked markers).
func TestGemmaTools_ParseGemmaToolCalls_Ugly(t *testing.T) {
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

// TestGemmaTools_RenderGemmaToolDeclarations_Good pins the shared renderer
// against the exact declaration in the gemma4 function-calling reference —
// the format the model was trained on, so both providers must produce it
// byte-for-byte.
func TestGemmaTools_RenderGemmaToolDeclarations_Good(t *testing.T) {
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
}

// TestGemmaTools_RenderGemmaToolDeclarations_Bad pins the empty-input guard:
// no tools renders an empty string, not an empty declaration block.
func TestGemmaTools_RenderGemmaToolDeclarations_Bad(t *testing.T) {
	if RenderGemmaToolDeclarations(nil) != "" {
		t.Fatal("no tools should render empty")
	}
	if RenderGemmaToolDeclarations([]ToolDecl{}) != "" {
		t.Fatal("an empty (non-nil) tool slice should also render empty")
	}
}

// TestGemmaTools_RenderGemmaToolDeclarations_Ugly pins the multi-tool,
// multi-property, no-required-fields edge: property order is sorted
// (deterministic prompt) and multiple tools concatenate one block per tool.
func TestGemmaTools_RenderGemmaToolDeclarations_Ugly(t *testing.T) {
	tools := []ToolDecl{
		{
			Name:        "list_files",
			Description: "Lists files in a directory.",
			Properties: map[string]ToolParam{
				"zpath":    {Type: "string", Description: "directory path"},
				"apattern": {Type: "string", Description: "glob filter"},
			},
			// no Required — the [] must render empty, not omitted.
		},
		{
			Name:        "now",
			Description: "Returns the current time.",
			Properties:  map[string]ToolParam{},
		},
	}
	got := RenderGemmaToolDeclarations(tools)
	q := ToolArgQuoteMarker
	firstBlock := "<|tool>declaration:list_files{description:" + q + "Lists files in a directory." + q +
		",parameters:{properties:{apattern:{description:" + q + "glob filter" + q + ",type:" + q + "STRING" + q + "} ," +
		"zpath:{description:" + q + "directory path" + q + ",type:" + q + "STRING" + q + "} }," +
		"required:[],type:" + q + "OBJECT" + q + "} }<tool|>"
	secondBlock := "<|tool>declaration:now{description:" + q + "Returns the current time." + q +
		",parameters:{properties:{},required:[],type:" + q + "OBJECT" + q + "} }<tool|>"
	if want := firstBlock + secondBlock; got != want {
		t.Fatalf("multi-tool render mismatch:\n got: %s\nwant: %s", got, want)
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

// TestGemmaTools_RenderGemmaToolCall_Good pins that RenderGemmaToolCall is
// the inverse of ParseGemmaToolCalls — mixed string/number/bool args render
// in gemma4's wire form (string wrapped in the quote marker, numbers/bools
// bare, keys sorted) and the render re-parses to the original call (#300).
func TestGemmaTools_RenderGemmaToolCall_Good(t *testing.T) {
	got := RenderGemmaToolCall("get_weather", `{"city":"Paris","days":5,"metric":true}`)
	want := `<|tool_call>call:get_weather{city:<|"|>Paris<|"|>,days:5,metric:true}<tool_call|>`
	if got != want {
		t.Fatalf("render = %q, want %q", got, want)
	}
	if calls, _ := ParseGemmaToolCalls(got); len(calls) != 1 || calls[0].Name != "get_weather" {
		t.Fatalf("re-parse = %+v, want one get_weather call", calls)
	}
}

// TestGemmaTools_RenderGemmaToolCall_Bad pins the malformed/empty-arguments
// guard: an empty (or non-JSON) arguments string renders an empty {} body,
// not a dropped call.
func TestGemmaTools_RenderGemmaToolCall_Bad(t *testing.T) {
	if got := RenderGemmaToolCall("now", ""); got != "<|tool_call>call:now{}<tool_call|>" {
		t.Fatalf("empty-args render = %q, want the {} call span", got)
	}
	if got := RenderGemmaToolCall("now", "not json"); got != "<|tool_call>call:now{}<tool_call|>" {
		t.Fatalf("malformed-args render = %q, want the {} call span", got)
	}
}

// TestGemmaTools_RenderGemmaToolCall_Ugly pins the recursive edge: nested
// object + array args recurse rather than stringifying the whole payload.
func TestGemmaTools_RenderGemmaToolCall_Ugly(t *testing.T) {
	got := RenderGemmaToolCall("search", `{"filter":{"lang":"go"},"tags":["a","b"]}`)
	want := `<|tool_call>call:search{filter:{lang:<|"|>go<|"|>},tags:[<|"|>a<|"|>,<|"|>b<|"|>]}<tool_call|>`
	if got != want {
		t.Fatalf("nested render = %q, want %q", got, want)
	}
}

// TestGemmaTools_SupportsToolSyntax_Good pins the accepted Gemma 4 spellings —
// the exact architecture strings engine/hip's own gemma4 detection matches
// against (draft_detect.go's Contains(value, "gemma4")).
func TestGemmaTools_SupportsToolSyntax_Good(t *testing.T) {
	for _, arch := range []string{"gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text", "GEMMA4_TEXT"} {
		if !SupportsToolSyntax(arch) {
			t.Fatalf("SupportsToolSyntax(%q) = false, want true", arch)
		}
	}
}

// TestGemmaTools_SupportsToolSyntax_Bad pins the rejected architectures — a
// non-Gemma-4 family (including the coarser "gemma3" bucket, which shares
// gemma_tools.go's reasoning markers but was never trained on this tool
// syntax) and the empty string.
func TestGemmaTools_SupportsToolSyntax_Bad(t *testing.T) {
	for _, arch := range []string{"qwen3", "gemma3", "gemma3_text", "gpt_oss", "mistral", ""} {
		if SupportsToolSyntax(arch) {
			t.Fatalf("SupportsToolSyntax(%q) = true, want false", arch)
		}
	}
}

// TestGemmaTools_SupportsToolSyntax_Ugly pins that the check is a substring
// match on the raw architecture string — no NormaliseKey hyphen/dot folding —
// so a hyphenated form like "gemma-4-e4b" (which would fold to "gemma_4_e4b",
// splitting the digit from the name) correctly does NOT match; only a
// contiguous "gemma4" run does, wherever it sits in a longer identifier.
func TestGemmaTools_SupportsToolSyntax_Ugly(t *testing.T) {
	if SupportsToolSyntax("gemma-4-e4b") {
		t.Fatal(`SupportsToolSyntax("gemma-4-e4b") = true, want false (hyphenated form, not a "gemma4" run)`)
	}
	if !SupportsToolSyntax("prefix_gemma4_suffix") {
		t.Fatal(`SupportsToolSyntax("prefix_gemma4_suffix") = false, want true (contiguous "gemma4" run anywhere in the string)`)
	}
}
