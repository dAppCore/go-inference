// SPDX-Licence-Identifier: EUPL-1.2

package dotsocr

import "testing"

// TestBuildPrompt_Good replays prompt_golden.json's expanded_text — captured from the REAL
// chat_template.json rendered through transformers' apply_chat_template for a list-content
// (image+text) user message, the checkpoint's own README-documented usage — proving buildPrompt's
// hand-derived shape (no <|user|>/<|endofuser|> wrapper, one leading space; see buildPrompt's doc
// comment for how that was discovered) is byte-identical to the real processor's output.
func TestBuildPrompt_Good(t *testing.T) {
	g := readPromptGolden(t)
	got := buildPrompt(g.Prompt, g.NMergedImageTokens)
	if got != g.ExpandedText {
		t.Fatalf("buildPrompt mismatch:\n got: %q\nwant: %q", got, g.ExpandedText)
	}
}

// TestBuildPrompt_Ugly proves zero image tokens (a surprising but valid call shape — e.g. a
// caller probing prompt structure without an image) still produces a well-formed, non-panicking
// string: the imgpad run is simply empty, not omitted or malformed.
func TestBuildPrompt_Ugly(t *testing.T) {
	got := buildPrompt("hello", 0)
	want := " <|img|><|endofimg|>hello<|assistant|>"
	if got != want {
		t.Fatalf("buildPrompt(zero image tokens) = %q, want %q", got, want)
	}
}
