// SPDX-Licence-Identifier: EUPL-1.2

package parser

import "testing"

// TestGrammar_PairedReasoningMarkers_Good pins the authoritative span table: every
// entry has a start, an end and a kind, and the qwen <think> spelling is
// present (the derived generic set excludes it; the extractor consumes all).
func TestGrammar_PairedReasoningMarkers_Good(t *testing.T) {
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

// TestGrammar_PairedReasoningMarkers_Bad pins the derived-view consistency: every
// generic Processor marker start exists in the authoritative table (a drift
// here means the two engines' grammars split again — the bug this file kills).
func TestGrammar_PairedReasoningMarkers_Bad(t *testing.T) {
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

// TestGrammar_PairedReasoningMarkers_Ugly pins the shared-view contract: the
// doc comment on PairedReasoningMarkers promises a package-owned view, not a
// defensive copy — two calls must return the same backing array, so a caller
// that (wrongly) mutates one sees it reflected on the other.
func TestGrammar_PairedReasoningMarkers_Ugly(t *testing.T) {
	first := PairedReasoningMarkers()
	second := PairedReasoningMarkers()
	if len(first) == 0 || len(second) == 0 {
		t.Fatal("empty marker table")
	}
	original := first[0].Kind
	first[0].Kind = "mutated-by-test"
	if second[0].Kind != "mutated-by-test" {
		t.Fatal("PairedReasoningMarkers must return the shared backing array, not a copy")
	}
	first[0].Kind = original // restore — the slice is package-global state
}

// TestGrammar_IsReasoningChannel_Good pins the reasoning channel names, including
// gpt-oss harmony's analysis channel.
func TestGrammar_IsReasoningChannel_Good(t *testing.T) {
	for _, name := range []string{"thought", "thinking", "reasoning", "analysis"} {
		if !IsReasoningChannel(name) {
			t.Fatalf("%q should be a reasoning channel", name)
		}
	}
}

// TestGrammar_IsReasoningChannel_Bad pins the content channels: final/assistant (and
// anything unrecognised) stay visible.
func TestGrammar_IsReasoningChannel_Bad(t *testing.T) {
	for _, name := range []string{"final", "assistant", "commentary", ""} {
		if IsReasoningChannel(name) {
			t.Fatalf("%q must not be a reasoning channel", name)
		}
	}
}

// TestGrammar_IsReasoningChannel_Ugly pins case sensitivity: the extractor lowercases
// channel names before classification, so the grammar matches lowercase only.
func TestGrammar_IsReasoningChannel_Ugly(t *testing.T) {
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
