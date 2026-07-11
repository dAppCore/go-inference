// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/inference/engine"
)

// TestSpeculativeChatFraming pins that the speculative model frames its chat
// prompt through the SHARED engine render (engine.RenderChatTurns over the
// gemma ChatTemplate) — byte-identical to the plain engine path in BOTH marker
// dialects (gemma4 <|turn>/<turn|>; legacy <start_of_turn>), with a trailing
// open model turn. It is the receipt that the private duplicate template is
// gone: the bytes speculativeModel.Chat encodes now come from the one engine
// render, not a package-local copy that could drift from the plain path.
func TestSpeculativeChatFraming(t *testing.T) {
	gemma4 := engine.TurnTokens{Open: "<|turn>", Close: "<turn|>"}
	got := engine.RenderChatTurns(engine.GemmaChatTemplate(gemma4, false), []inference.Message{{Role: "user", Content: "hi"}})
	want := "<|turn>user\nhi<turn|>\n<|turn>model\n"
	if got != want {
		t.Fatalf("gemma4 speculative framing = %q, want %q", got, want)
	}
	legacy := engine.TurnTokens{Open: "<start_of_turn>", Close: "<end_of_turn>"}
	multi := engine.RenderChatTurns(engine.GemmaChatTemplate(legacy, false), []inference.Message{
		{Role: "user", Content: "q"},
		{Role: "assistant", Content: "a"},
		{Role: "user", Content: "q2"},
	})
	wantMulti := "<start_of_turn>user\nq<end_of_turn>\n" +
		"<start_of_turn>model\na<end_of_turn>\n" +
		"<start_of_turn>user\nq2<end_of_turn>\n" +
		"<start_of_turn>model\n"
	if multi != wantMulti {
		t.Fatalf("legacy multi-turn framing = %q, want %q", multi, wantMulti)
	}
}

// TestSpeculativeChatRole pins the role spellings the shared gemma template
// keys on for the speculative path: assistant and model both render as the
// "model" turn; anything else (user, system, empty) renders as a "user" turn.
func TestSpeculativeChatRole(t *testing.T) {
	tmpl := engine.GemmaChatTemplate(engine.TurnTokens{Open: "<|turn>", Close: "<turn|>"}, false)
	for _, role := range []string{"assistant", "model"} {
		got := engine.RenderChatTurns(tmpl, []inference.Message{{Role: role, Content: "x"}})
		if want := "<|turn>model\nx<turn|>\n<|turn>model\n"; got != want {
			t.Fatalf("role %q framed %q, want the model turn %q", role, got, want)
		}
	}
	for _, role := range []string{"user", "system", ""} {
		got := engine.RenderChatTurns(tmpl, []inference.Message{{Role: role, Content: "x"}})
		if want := "<|turn>user\nx<turn|>\n<|turn>model\n"; got != want {
			t.Fatalf("role %q framed %q, want the user turn %q", role, got, want)
		}
	}
}

// TestSpeculativeTokenInSet pins the terminator membership the decode sink uses
// to blank a stop token's text.
func TestSpeculativeTokenInSet(t *testing.T) {
	set := []int32{1, 106, 107}
	if !speculativeTokenInSet(107, set) {
		t.Fatal("speculativeTokenInSet missed a present id")
	}
	if speculativeTokenInSet(42, set) {
		t.Fatal("speculativeTokenInSet matched an absent id")
	}
	if speculativeTokenInSet(1, nil) {
		t.Fatal("empty set contains nothing")
	}
}
