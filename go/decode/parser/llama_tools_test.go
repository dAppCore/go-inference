// SPDX-Licence-Identifier: EUPL-1.2

package parser

import (
	"testing"

	core "dappco.re/go"
)

func TestLlamaTools_SupportsToolSyntax_Good(t *testing.T) {
	for _, arch := range []string{"llama", "llama3", "llama3_1", "LlamaForCausalLM"} {
		if !SupportsToolSyntax(arch) {
			t.Fatalf("SupportsToolSyntax(%q) = false, want true", arch)
		}
	}
}

func TestLlamaTools_ParseLlamaToolCalls_Good(t *testing.T) {
	in := `Before <|python_tag|>{"type":"function","name":"get_weather","parameters":{"city":"London"}}<|eom_id|>`
	calls, visible := ParseLlamaToolCalls(in)
	if len(calls) != 1 || calls[0].Name != "get_weather" || calls[0].ArgumentsJSON != `{"city":"London"}` {
		t.Fatalf("ParseLlamaToolCalls() calls = %+v, want get_weather with city", calls)
	}
	if visible != "Before" {
		t.Fatalf("ParseLlamaToolCalls() visible = %q, want Before", visible)
	}
}

func TestLlamaTools_ParseLlamaToolCalls_Bad(t *testing.T) {
	calls, visible := ParseLlamaToolCalls("ordinary answer")
	if len(calls) != 0 || visible != "ordinary answer" {
		t.Fatalf("ParseLlamaToolCalls() = %+v, %q, want no calls and unchanged prose", calls, visible)
	}
}

func TestLlamaTools_ParseLlamaToolCalls_Ugly(t *testing.T) {
	in := `<|python_tag|>{"type":"function","name":"broken","parameters":` + LlamaToolCallCloseMarker
	calls, visible := ParseLlamaToolCalls(in)
	if len(calls) != 0 || visible != in {
		t.Fatalf("malformed ParseLlamaToolCalls() = %+v, %q, want raw input visible", calls, visible)
	}
}

func TestLlamaTools_RenderLlamaToolDeclarations_Good(t *testing.T) {
	got := RenderLlamaToolDeclarations([]ToolDecl{{
		Name: "get_weather", Description: "Get weather",
		Properties: map[string]ToolParam{"city": {Type: "string", Description: "City name"}},
		Required:   []string{"city"},
	}})
	for _, want := range []string{"Environment: ipython", "Here is a list of functions in JSON format:", `"name":"get_weather"`, `"required":["city"]`, "Return function calls in JSON format."} {
		if !core.Contains(got, want) {
			t.Fatalf("RenderLlamaToolDeclarations() = %q, want substring %q", got, want)
		}
	}
}

func TestLlamaTools_RenderLlamaToolDeclarations_Bad(t *testing.T) {
	if got := RenderLlamaToolDeclarations(nil); got != "" {
		t.Fatalf("RenderLlamaToolDeclarations(nil) = %q, want empty", got)
	}
}

func TestLlamaTools_RenderLlamaToolDeclarations_Ugly(t *testing.T) {
	tools := []ToolDecl{{Name: "z"}, {Name: "a"}}
	got := RenderLlamaToolDeclarations(tools)
	if core.Index(got, `"name":"z"`) >= core.Index(got, `"name":"a"`) {
		t.Fatalf("RenderLlamaToolDeclarations() = %q, want declaration order preserved", got)
	}
}
