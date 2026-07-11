// SPDX-Licence-Identifier: EUPL-1.2

package parser

import (
	"testing"

	core "dappco.re/go"
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

// TestReasoning_FindReasoningStart_AnchorMatchesNaive_Good pins the lead-byte
// anchor scan byte-for-byte to the naive min-index / longest-tie reference
// across every builtin marker family and 240k adversarial fragment streams
// (stray '<', '|', partial markers). The guard for the O(markers×text)→O(text)
// rewrite: a change that breaks equivalence fails here.
func TestReasoning_FindReasoningStart_AnchorMatchesNaive_Good(t *testing.T) {
	naive := func(text string, markers []reasoningMarker) (int, reasoningMarker, bool) {
		best := -1
		var marker reasoningMarker
		for _, c := range markers {
			idx := indexString(text, c.start)
			if idx < 0 {
				continue
			}
			if best < 0 || idx < best || idx == best && len(c.start) > len(marker.start) {
				best = idx
				marker = c
			}
		}
		return best, marker, best >= 0
	}
	sets := [][]reasoningMarker{qwenMarkers(), gemmaMarkers(), gptOSSMarkers(), genericMarkers()}
	frags := []string{
		"word ", "the ", "<", ">", "|", "\n", "<think>", "</think>", "<thinking>",
		"<|channel>", "analysis\n", "final\n", "<start_of_turn>", "thinking\n",
		"thought\n", "<end_of_turn>", "reasoning\n", "<reason", "chan", "x", "  ",
		"<|channel>analysis\n", "<start_of_turn>thought\n",
	}
	st := uint64(0xd1b54a32d192ed03)
	rnd := func() uint64 { st = st*6364136223846793005 + 1442695040888963407; return st >> 1 }
	for _, markers := range sets {
		leads := reasoningMarkerLeadBytes(markers)
		for iter := 0; iter < 60000; iter++ {
			b := core.NewBuilder()
			for k := int(rnd()%10) + 1; k > 0; k-- {
				b.WriteString(frags[rnd()%uint64(len(frags))])
			}
			text := b.String()
			ai, am, aok := findReasoningStart(text, markers, leads)
			ni, nm, nok := naive(text, markers)
			if ai != ni || aok != nok || am.start != nm.start {
				t.Fatalf("text=%q anchor=(%d,%q,%v) naive=(%d,%q,%v)", text, ai, am.start, aok, ni, nm.start, nok)
			}
		}
	}
}
