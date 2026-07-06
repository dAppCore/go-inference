// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference"
)

// TestFormatSpeculativeChatTurns pins that the speculative model's chat framing
// is byte-identical to the plain engine path's gemma turn template: each turn as
// "<start_of_turn>ROLE\nCONTENT<end_of_turn>\n", then a trailing open model turn.
// Divergence here would give the drafter a different prompt shape than the plain
// path, so keep it locked.
func TestFormatSpeculativeChatTurns(t *testing.T) {
	got := formatSpeculativeChatTurns([]inference.Message{{Role: "user", Content: "hi"}})
	want := "<start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\n"
	if got != want {
		t.Fatalf("formatSpeculativeChatTurns = %q, want %q", got, want)
	}
	multi := formatSpeculativeChatTurns([]inference.Message{
		{Role: "user", Content: "q"},
		{Role: "assistant", Content: "a"},
		{Role: "user", Content: "q2"},
	})
	wantMulti := "<start_of_turn>user\nq<end_of_turn>\n" +
		"<start_of_turn>model\na<end_of_turn>\n" +
		"<start_of_turn>user\nq2<end_of_turn>\n" +
		"<start_of_turn>model\n"
	if multi != wantMulti {
		t.Fatalf("multi-turn framing = %q, want %q", multi, wantMulti)
	}
}

// TestSpeculativeChatRole pins the role mapping the template keys on: assistant
// and model both render as "model"; anything else is "user".
func TestSpeculativeChatRole(t *testing.T) {
	for _, role := range []string{"assistant", "model"} {
		if got := speculativeChatRole(role); got != "model" {
			t.Fatalf("speculativeChatRole(%q) = %q, want model", role, got)
		}
	}
	for _, role := range []string{"user", "system", ""} {
		if got := speculativeChatRole(role); got != "user" {
			t.Fatalf("speculativeChatRole(%q) = %q, want user", role, got)
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
