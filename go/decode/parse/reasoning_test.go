// SPDX-Licence-Identifier: EUPL-1.2

// Tests for the reasoning splitter (ReasoningParser, Gemma4Reasoning).

package parse

import "testing"

func TestParse_Reasoning_Good(t *testing.T) {
	// A think block is split out; everything after </think> is the content.
	p := Gemma4Reasoning()
	reasoning, content := p.Parse("<think>step one\nstep two</think>The answer is 42.")

	if reasoning != "step one\nstep two" {
		t.Fatalf("reasoning = %q", reasoning)
	}
	if content != "The answer is 42." {
		t.Fatalf("content = %q", content)
	}
}

func TestParse_Reasoning_Good_NoStartTokenButHasEnd(t *testing.T) {
	// force_reasoning: the leading text up to </think> is reasoning even with no
	// explicit <think> opener (DeepSeek-R1 style).
	p := ReasoningParser{ThinkStart: "<think>", ThinkEnd: "</think>", ForceReasoning: true}
	reasoning, content := p.Parse("thinking out loud</think>final")

	if reasoning != "thinking out loud" {
		t.Fatalf("reasoning = %q", reasoning)
	}
	if content != "final" {
		t.Fatalf("content = %q", content)
	}
}

func TestParse_Reasoning_Bad_NoThinkBlock(t *testing.T) {
	// No think tokens and not forced: it's all content, no reasoning.
	p := Gemma4Reasoning()
	reasoning, content := p.Parse("Just an answer.")

	if reasoning != "" {
		t.Fatalf("reasoning = %q, want empty", reasoning)
	}
	if content != "Just an answer." {
		t.Fatalf("content = %q", content)
	}
}

func TestParse_Reasoning_Ugly_Unterminated(t *testing.T) {
	// A think block that opens but never closes: everything after the opener is
	// reasoning, content is empty (matches the truncated-reasoning branch).
	p := Gemma4Reasoning()
	reasoning, content := p.Parse("<think>cut off mid thought")

	if reasoning != "cut off mid thought" {
		t.Fatalf("reasoning = %q", reasoning)
	}
	if content != "" {
		t.Fatalf("content = %q, want empty", content)
	}
}

func TestParse_Reasoning_Ugly_ForceUnterminated(t *testing.T) {
	// force_reasoning with no end token at all: the whole text is reasoning.
	p := ReasoningParser{ThinkStart: "<think>", ThinkEnd: "</think>", ForceReasoning: true}
	reasoning, content := p.Parse("everything is a thought")

	if reasoning != "everything is a thought" {
		t.Fatalf("reasoning = %q", reasoning)
	}
	if content != "" {
		t.Fatalf("content = %q, want empty", content)
	}
}

func TestParse_Reasoning_Ugly_RepeatedStartTokens(t *testing.T) {
	// Several leading <think> openers are all stripped before the block (matches
	// the `while startswith` loop), then split at </think>.
	p := Gemma4Reasoning()
	reasoning, content := p.Parse("<think><think>doubled</think>done")

	if reasoning != "doubled" {
		t.Fatalf("reasoning = %q", reasoning)
	}
	if content != "done" {
		t.Fatalf("content = %q", content)
	}
}
func TestParse_Reasoning_Ugly_StartTokenMidString(t *testing.T) {
	// Start token present mid-string (not at the very start). SGLang sets
	// in_reasoning=True because the start token appears anywhere, but only strips
	// it when the text *begins* with it. So everything up to </think> — including
	// the literal "intro <think>" prefix — is reasoning, and " outro" is content.
	p := Gemma4Reasoning()
	reasoning, content := p.Parse("intro <think>mid</think> outro")

	if reasoning != "intro <think>mid" {
		t.Fatalf("reasoning = %q, want the whole pre-end prefix", reasoning)
	}
	if content != " outro" {
		t.Fatalf("content = %q, want the post-end remainder", content)
	}
}
