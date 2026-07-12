// SPDX-Licence-Identifier: EUPL-1.2

package anthropic

import (
	"testing"

	core "dappco.re/go"
)

func weatherTool(name string) Tool {
	return Tool{Name: name, Description: "d", InputSchema: ToolInputSchema{Type: "object"}}
}

// --- ToolChoice.resolve ------------------------------------------------------

// TestToolChoice_Resolve_Good pins the plain mappings: nil (tool_choice
// omitted) and {"type":"auto"} both resolve to tools.ChoiceAuto(), and
// {"type":"any"} to tools.ChoiceRequired() (Anthropic's "at least one call"
// mode, the same obligation OpenAI spells "required").
func TestToolChoice_Resolve_Good(t *testing.T) {
	var nilChoice *ToolChoice
	if got := nilChoice.resolve().Mode; got != "auto" {
		t.Fatalf("nil ToolChoice.resolve().Mode = %q, want auto", got)
	}
	if got := (&ToolChoice{Type: "auto"}).resolve().Mode; got != "auto" {
		t.Fatalf("auto ToolChoice.resolve().Mode = %q, want auto", got)
	}
	if got := (&ToolChoice{Type: "any"}).resolve().Mode; got != "required" {
		t.Fatalf("any ToolChoice.resolve().Mode = %q, want required", got)
	}
}

// TestToolChoice_Resolve_Bad pins "none" and the named-tool form.
func TestToolChoice_Resolve_Bad(t *testing.T) {
	if got := (&ToolChoice{Type: "none"}).resolve().Mode; got != "none" {
		t.Fatalf("none ToolChoice.resolve().Mode = %q, want none", got)
	}
	resolved := (&ToolChoice{Type: "tool", Name: "get_weather"}).resolve()
	if resolved.Mode != "tool" || resolved.Name != "get_weather" {
		t.Fatalf("tool ToolChoice.resolve() = %+v, want tool/get_weather", resolved)
	}
}

// TestToolChoice_Resolve_Ugly pins case/whitespace tolerance and that an
// unrecognised Type falls back to auto rather than erroring.
func TestToolChoice_Resolve_Ugly(t *testing.T) {
	if got := (&ToolChoice{Type: " ANY "}).resolve().Mode; got != "required" {
		t.Fatalf("padded/upper Type resolve = %q, want required", got)
	}
	if got := (&ToolChoice{Type: "whatever"}).resolve().Mode; got != "auto" {
		t.Fatalf("unrecognised Type resolve = %q, want auto fallback", got)
	}
}

// --- ResolveOfferedTools -----------------------------------------------------

// TestResolveOfferedTools_Good pins the common paths: no tools declared is a
// no-op regardless of choice, and a nil/auto choice offers everything declared.
func TestResolveOfferedTools_Good(t *testing.T) {
	if offered, err := ResolveOfferedTools(nil, nil); err != nil || len(offered) != 0 {
		t.Fatalf("ResolveOfferedTools(nil, nil) = %+v, %v, want empty/nil, no error", offered, err)
	}
	declared := []Tool{weatherTool("get_weather"), weatherTool("search")}
	offered, err := ResolveOfferedTools(declared, nil)
	if err != nil || len(offered) != 2 {
		t.Fatalf("ResolveOfferedTools(declared, nil) = %+v, %v, want both tools, no error", offered, err)
	}
	offered, err = ResolveOfferedTools(declared, &ToolChoice{Type: "auto"})
	if err != nil || len(offered) != 2 {
		t.Fatalf("ResolveOfferedTools(declared, auto) = %+v, %v, want both tools, no error", offered, err)
	}
}

// TestResolveOfferedTools_Bad pins tool_choice type "none" — every declared
// tool is suppressed.
func TestResolveOfferedTools_Bad(t *testing.T) {
	declared := []Tool{weatherTool("get_weather"), weatherTool("search")}
	offered, err := ResolveOfferedTools(declared, &ToolChoice{Type: "none"})
	if err != nil {
		t.Fatalf("ResolveOfferedTools(declared, none): unexpected error: %v", err)
	}
	if len(offered) != 0 {
		t.Fatalf("ResolveOfferedTools(declared, none) = %+v, want none offered", offered)
	}
}

// TestResolveOfferedTools_Ugly pins the named-tool narrowing and the two
// caller-error cases agent/tools.Resolve reports: an undeclared name, and
// "any" with nothing declared.
func TestResolveOfferedTools_Ugly(t *testing.T) {
	declared := []Tool{weatherTool("get_weather"), weatherTool("search")}
	offered, err := ResolveOfferedTools(declared, &ToolChoice{Type: "tool", Name: "search"})
	if err != nil {
		t.Fatalf("ResolveOfferedTools named choice: unexpected error: %v", err)
	}
	if len(offered) != 1 || offered[0].Name != "search" {
		t.Fatalf("ResolveOfferedTools named choice = %+v, want only search", offered)
	}

	if _, err := ResolveOfferedTools(declared, &ToolChoice{Type: "tool", Name: "not_declared"}); err == nil {
		t.Fatal("ResolveOfferedTools naming an undeclared tool: expected error, got nil")
	}
	if _, err := ResolveOfferedTools(nil, &ToolChoice{Type: "any"}); err == nil {
		t.Fatal("ResolveOfferedTools any with nothing declared: expected error, got nil")
	}
}

// --- InferenceMessages / tool_choice integration ----------------------------

// TestInferenceMessages_ToolChoiceNone_Good pins that tool_choice:"none" keeps
// the rendered declarations out of the system turn entirely.
func TestInferenceMessages_ToolChoiceNone_Good(t *testing.T) {
	req := MessageRequest{
		Messages:   []Message{{Role: "user", Content: []ContentBlock{{Type: "text", Text: "hi"}}}},
		Tools:      []Tool{weatherTool("get_weather")},
		ToolChoice: &ToolChoice{Type: "none"},
	}
	msgs := InferenceMessages(req)
	if len(msgs) != 1 || msgs[0].Role != "user" {
		t.Fatalf("InferenceMessages with tool_choice:none = %+v, want just the user turn (no system turn)", msgs)
	}
}

// TestInferenceMessages_ToolChoiceContradiction_Bad pins the safe degrade:
// InferenceMessages cannot report the ResolveOfferedTools error (no error
// return — see its doc comment), so a contradictory tool_choice falls back to
// declaring every tool rather than silently rendering none.
func TestInferenceMessages_ToolChoiceContradiction_Bad(t *testing.T) {
	req := MessageRequest{
		Messages:   []Message{{Role: "user", Content: []ContentBlock{{Type: "text", Text: "hi"}}}},
		Tools:      []Tool{weatherTool("get_weather")},
		ToolChoice: &ToolChoice{Type: "tool", Name: "not_declared"},
	}
	msgs := InferenceMessages(req)
	if len(msgs) != 2 || msgs[0].Role != "system" || !core.Contains(msgs[0].Content, "declaration:get_weather") {
		t.Fatalf("InferenceMessages with a contradictory tool_choice = %+v, want the safe fallback (declare everything)", msgs)
	}
}

// TestInferenceMessages_ToolChoiceNamed_Ugly pins the narrowing end to end:
// only the chosen tool's declaration reaches the system turn.
func TestInferenceMessages_ToolChoiceNamed_Ugly(t *testing.T) {
	req := MessageRequest{
		Messages:   []Message{{Role: "user", Content: []ContentBlock{{Type: "text", Text: "hi"}}}},
		Tools:      []Tool{weatherTool("get_weather"), weatherTool("search")},
		ToolChoice: &ToolChoice{Type: "tool", Name: "get_weather"},
	}
	msgs := InferenceMessages(req)
	if len(msgs) != 2 {
		t.Fatalf("InferenceMessages with a named tool_choice = %+v, want a system + user turn", msgs)
	}
	if !core.Contains(msgs[0].Content, "declaration:get_weather") || core.Contains(msgs[0].Content, "declaration:search") {
		t.Fatalf("system turn = %q, want only get_weather declared", msgs[0].Content)
	}
}
