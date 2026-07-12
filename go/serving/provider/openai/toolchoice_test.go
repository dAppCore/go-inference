// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"context"
	"iter"
	"testing"

	"dappco.re/go/inference"
)

// recordingModel is a stubModel that captures the messages each Chat call
// receives and can report a caller-chosen architecture — used across the
// tool_choice / capability-gate / response_format tests to assert what
// actually reached the model's prompt, and to drive the Gemma-4-only
// capability gate both ways. sequenced, when non-nil, serves a different
// token slice per successive Chat call (index clamped to the last entry),
// for the response_format repair-loop tests where a reprompt must return
// different output than the original call.
type recordingModel struct {
	stubModel
	arch      string
	received  []inference.Message
	calls     int
	sequenced [][]inference.Token
}

func (m *recordingModel) Info() inference.ModelInfo {
	if m.arch == "" {
		return inference.ModelInfo{Architecture: "qwen3"}
	}
	return inference.ModelInfo{Architecture: m.arch}
}

func (m *recordingModel) Chat(_ context.Context, messages []inference.Message, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	m.received = messages
	m.calls++
	if m.sequenced == nil {
		return m.seq()
	}
	idx := m.calls - 1
	if idx >= len(m.sequenced) {
		idx = len(m.sequenced) - 1
	}
	tokens := m.sequenced[idx]
	return func(yield func(inference.Token) bool) {
		for _, token := range tokens {
			if !yield(token) {
				return
			}
		}
	}
}

// --- ToolChoice.resolve ----------------------------------------------------

// TestToolChoice_Resolve_Good pins the plain mode mappings: nil (tool_choice
// omitted) and "auto" both resolve to tools.ChoiceAuto(), "required" to
// tools.ChoiceRequired().
func TestToolChoice_Resolve_Good(t *testing.T) {
	var nilChoice *ToolChoice
	if got := nilChoice.resolve().Mode; got != "auto" {
		t.Fatalf("nil ToolChoice.resolve().Mode = %q, want auto", got)
	}
	if got := (&ToolChoice{Mode: "auto"}).resolve().Mode; got != "auto" {
		t.Fatalf("auto ToolChoice.resolve().Mode = %q, want auto", got)
	}
	if got := (&ToolChoice{Mode: "required"}).resolve().Mode; got != "required" {
		t.Fatalf("required ToolChoice.resolve().Mode = %q, want required", got)
	}
}

// TestToolChoice_Resolve_Bad pins "none" and a named function choice.
func TestToolChoice_Resolve_Bad(t *testing.T) {
	if got := (&ToolChoice{Mode: "none"}).resolve().Mode; got != "none" {
		t.Fatalf("none ToolChoice.resolve().Mode = %q, want none", got)
	}
	resolved := (&ToolChoice{Mode: "function", Name: "get_weather"}).resolve()
	if resolved.Mode != "tool" || resolved.Name != "get_weather" {
		t.Fatalf("function ToolChoice.resolve() = %+v, want tool/get_weather", resolved)
	}
}

// TestToolChoice_Resolve_Ugly pins case/whitespace tolerance and that an
// unrecognised Mode falls back to auto rather than erroring.
func TestToolChoice_Resolve_Ugly(t *testing.T) {
	if got := (&ToolChoice{Mode: " REQUIRED "}).resolve().Mode; got != "required" {
		t.Fatalf("padded/upper Mode resolve = %q, want required", got)
	}
	if got := (&ToolChoice{Mode: "whatever"}).resolve().Mode; got != "auto" {
		t.Fatalf("unrecognised Mode resolve = %q, want auto fallback", got)
	}
}

// --- resolveOfferedTools -----------------------------------------------------

func weatherTool(name string) Tool {
	return Tool{Type: "function", Function: ToolFunction{Name: name, Description: "d"}}
}

// TestResolveOfferedTools_Good pins the common paths: no tools declared is a
// no-op regardless of choice, and a nil/auto choice offers everything declared
// unchanged (same slice, not a defensive copy — the pre-tool_choice behaviour).
func TestResolveOfferedTools_Good(t *testing.T) {
	if offered, err := resolveOfferedTools(nil, nil); err != nil || len(offered) != 0 {
		t.Fatalf("resolveOfferedTools(nil, nil) = %+v, %v, want empty/nil, no error", offered, err)
	}
	declared := []Tool{weatherTool("get_weather"), weatherTool("search")}
	offered, err := resolveOfferedTools(declared, nil)
	if err != nil || len(offered) != 2 {
		t.Fatalf("resolveOfferedTools(declared, nil) = %+v, %v, want both tools, no error", offered, err)
	}
	offered, err = resolveOfferedTools(declared, &ToolChoice{Mode: "auto"})
	if err != nil || len(offered) != 2 {
		t.Fatalf("resolveOfferedTools(declared, auto) = %+v, %v, want both tools, no error", offered, err)
	}
}

// TestResolveOfferedTools_Bad pins tool_choice:"none" — every declared tool is
// suppressed, so the caller never renders a declaration the model can't act on
// this turn.
func TestResolveOfferedTools_Bad(t *testing.T) {
	declared := []Tool{weatherTool("get_weather"), weatherTool("search")}
	offered, err := resolveOfferedTools(declared, &ToolChoice{Mode: "none"})
	if err != nil {
		t.Fatalf("resolveOfferedTools(declared, none): unexpected error: %v", err)
	}
	if len(offered) != 0 {
		t.Fatalf("resolveOfferedTools(declared, none) = %+v, want none offered", offered)
	}
}

// TestResolveOfferedTools_Ugly pins the named-tool narrowing (only the chosen
// tool is offered, order-independent of its position in declared) and the two
// caller-error cases agent/tools.Resolve reports: an undeclared name, and
// "required" with nothing declared.
func TestResolveOfferedTools_Ugly(t *testing.T) {
	declared := []Tool{weatherTool("get_weather"), weatherTool("search")}
	offered, err := resolveOfferedTools(declared, &ToolChoice{Mode: "function", Name: "search"})
	if err != nil {
		t.Fatalf("resolveOfferedTools named choice: unexpected error: %v", err)
	}
	if len(offered) != 1 || offered[0].Function.Name != "search" {
		t.Fatalf("resolveOfferedTools named choice = %+v, want only search", offered)
	}

	if _, err := resolveOfferedTools(declared, &ToolChoice{Mode: "function", Name: "not_declared"}); err == nil {
		t.Fatal("resolveOfferedTools naming an undeclared tool: expected error, got nil")
	}

	if _, err := resolveOfferedTools(nil, &ToolChoice{Mode: "required"}); err == nil {
		t.Fatal("resolveOfferedTools required with nothing declared: expected error, got nil")
	}
}
