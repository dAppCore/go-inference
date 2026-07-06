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
