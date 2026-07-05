// SPDX-Licence-Identifier: EUPL-1.2

package parser

import "testing"

// TestPairedReasoningMarkers_Good pins the authoritative span table: every
// entry has a start, an end and a kind, and the qwen <think> spelling is
// present (the derived generic set excludes it; the extractor consumes all).
func TestPairedReasoningMarkers_Good(t *testing.T) {
	markers := PairedReasoningMarkers()
	if len(markers) == 0 {
		t.Fatal("empty marker table")
	}
	sawThink := false
	for _, m := range markers {
		if m.Start == "" || m.End == "" || m.Kind == "" {
			t.Fatalf("incomplete marker %+v", m)
		}
		if m.Start == "<think>" {
			sawThink = true
		}
	}
	if !sawThink {
		t.Fatal("<think> missing from the paired table")
	}
}

// TestPairedReasoningMarkers_Bad pins the derived-view consistency: every
// generic Processor marker start exists in the authoritative table (a drift
// here means the two engines' grammars split again — the bug this file kills).
func TestPairedReasoningMarkers_Bad(t *testing.T) {
	table := map[string]string{}
	for _, m := range PairedReasoningMarkers() {
		table[m.Start] = m.End
	}
	for _, gm := range genericMarkers() {
		end, ok := table[gm.start]
		if !ok {
			t.Fatalf("generic marker %q not in the authoritative table", gm.start)
		}
		if len(gm.ends) != 1 || gm.ends[0] != end {
			t.Fatalf("generic marker %q ends %v drifted from table end %q", gm.start, gm.ends, end)
		}
	}
}

// TestIsReasoningChannel_Good pins the reasoning channel names, including
// gpt-oss harmony's analysis channel.
func TestIsReasoningChannel_Good(t *testing.T) {
	for _, name := range []string{"thought", "thinking", "reasoning", "analysis"} {
		if !IsReasoningChannel(name) {
			t.Fatalf("%q should be a reasoning channel", name)
		}
	}
}

// TestIsReasoningChannel_Bad pins the content channels: final/assistant (and
// anything unrecognised) stay visible.
func TestIsReasoningChannel_Bad(t *testing.T) {
	for _, name := range []string{"final", "assistant", "commentary", ""} {
		if IsReasoningChannel(name) {
			t.Fatalf("%q must not be a reasoning channel", name)
		}
	}
}

// TestIsReasoningChannel_Ugly pins case sensitivity: the extractor lowercases
// channel names before classification, so the grammar matches lowercase only.
func TestIsReasoningChannel_Ugly(t *testing.T) {
	if IsReasoningChannel("Thought") {
		t.Fatal("classification is lowercase-only; callers lowercase first")
	}
}

// TestGrammarConstants pins the literal marker strings — three consumers
// (Processor, openai ThinkingExtractor, tokenizer DecodeToken) share them, so
// a change here is a protocol change, not a refactor.
func TestGrammarConstants(t *testing.T) {
	if ChannelOpenMarker != "<|channel>" || ChannelCloseMarker != "<channel|>" {
		t.Fatalf("channel markers drifted: %q %q", ChannelOpenMarker, ChannelCloseMarker)
	}
	if GemmaTurnTerminator != "<end_of_turn>" {
		t.Fatalf("turn terminator drifted: %q", GemmaTurnTerminator)
	}
}
