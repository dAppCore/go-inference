// SPDX-Licence-Identifier: EUPL-1.2

package parser

import (
	"testing"
)

func TestReasoning_BuiltinParsers_Good(t *testing.T) {
	cases := []struct {
		name      string
		arch      string
		text      string
		visible   string
		reasoning string
		kind      string
	}{
		{
			name:      "qwen think tags",
			arch:      "qwen3",
			text:      "pre<think>plan</think>answer",
			visible:   "preanswer",
			reasoning: "plan",
			kind:      "thinking",
		},
		{
			name:      "gemma turn markers",
			arch:      "gemma4_text",
			text:      "<start_of_turn>thinking\nplan<end_of_turn>done",
			visible:   "done",
			reasoning: "plan",
			kind:      "thinking",
		},
		{
			name:      "gpt oss channel markers",
			arch:      "gpt_oss",
			text:      "<|channel>analysis\nplan<|channel>final\nanswer",
			visible:   "answer",
			reasoning: "plan",
			kind:      "analysis",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ForHint(Hint{Architecture: tc.arch}).ParseReasoning(nil, tc.text)
			if err != nil {
				t.Fatalf("ParseReasoning() error = %v", err)
			}
			if got.VisibleText != tc.visible {
				t.Fatalf("VisibleText = %q, want %q", got.VisibleText, tc.visible)
			}
			if len(got.Reasoning) != 1 {
				t.Fatalf("Reasoning len = %d, want 1: %+v", len(got.Reasoning), got.Reasoning)
			}
			if got.Reasoning[0].Text != tc.reasoning || got.Reasoning[0].Kind != tc.kind {
				t.Fatalf("Reasoning[0] = %+v, want %q/%q", got.Reasoning[0], tc.kind, tc.reasoning)
			}
		})
	}
}
