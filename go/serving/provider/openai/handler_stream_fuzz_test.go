// SPDX-Licence-Identifier: EUPL-1.2

// Differential fuzz pinning the bounded-window contentStreamer to the
// pre-optimisation detection path. oracleStreamer replays the exact old
// serveStreaming loop body — full `emittedContent + contentDelta` concat and
// whole-candidate scans — so any divergence in a streamed token's emitted
// slice, tool/stop boundary, or final content surfaces as a fuzz failure before
// the fix lands. Adversarial streams split the tool marker and stop sequences at
// every byte boundary (including mid-rune), overlap marker prefixes, and pack
// marker-dense text through both paths.
package openai

import (
	"math/rand"
	"sort"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/decode/parser"
)

// oracleStreamer is the frozen reference: serveStreaming's per-token detection
// exactly as it stood before the bounded-window rewrite (a fresh full-candidate
// concat + whole-string scan every token).
type oracleStreamer struct {
	emittedContent string
	stops          []string
	inTool         bool
	stopped        bool
	stoppedAt      int
}

func (o *oracleStreamer) step(contentDelta string) streamOutcome {
	candidate := o.emittedContent + contentDelta
	if o.inTool {
		o.emittedContent = candidate
		return streamOutcome{swallow: true}
	}
	if idx := core.Index(candidate, parser.ToolCallOpenMarker); idx >= 0 {
		o.inTool = true
		out := streamOutcome{tool: true}
		if idx > len(o.emittedContent) {
			out.visible = candidate[len(o.emittedContent):idx]
		}
		o.emittedContent = candidate
		return out
	}
	stopCut, stopHit := firstStopSequenceCut(candidate, o.stops)
	if !stopHit {
		o.emittedContent = candidate
		return streamOutcome{visible: contentDelta}
	}
	out := streamOutcome{stop: true}
	if stopCut > len(o.emittedContent) {
		out.visible = candidate[len(o.emittedContent):stopCut]
	}
	o.stopped = true
	o.stoppedAt = stopCut
	o.emittedContent = candidate[:stopCut]
	return out
}

func (o *oracleStreamer) emitted() string { return o.emittedContent }

// streamCase feeds identical deltas + stops through the oracle and the windowed
// contentStreamer, asserting byte-identical per-token outcomes, final emitted
// content, and inTool state. It mirrors serveStreaming's break on a stop cut.
func streamCase(t *testing.T, deltas, stops []string) {
	t.Helper()
	cs := newContentStreamer(stops)
	o := &oracleStreamer{stops: stops}
	for i, d := range deltas {
		got := cs.step(d)
		want := o.step(d)
		if got != want {
			t.Fatalf("token %d %q stops=%q: windowed=%+v oracle=%+v\ndeltas=%q", i, d, stops, got, want, deltas)
		}
		if cs.inTool != o.inTool {
			t.Fatalf("token %d: inTool windowed=%v oracle=%v deltas=%q", i, cs.inTool, o.inTool, deltas)
		}
		if got.stop {
			break // serveStreaming breaks the token loop on a stop cut
		}
	}
	if ce, oe := cs.emitted(), o.emitted(); ce != oe {
		t.Fatalf("final emitted mismatch: windowed=%q oracle=%q\ndeltas=%q stops=%q", ce, oe, deltas, stops)
	}
	if cs.inTool != o.inTool {
		t.Fatalf("final inTool: windowed=%v oracle=%v", cs.inTool, o.inTool)
	}
}

// TestContentStreamer_DifferentialFuzz_EveryBoundary splits each fixture at
// every possible 2-way and 3-way byte boundary — the systematic proof that the
// tool marker and stop sequences are detected identically no matter where a
// token break falls (including mid-marker and mid-rune).
func TestContentStreamer_DifferentialFuzz_EveryBoundary(t *testing.T) {
	fixtures := []struct {
		text  string
		stops []string
	}{
		{"abc<|tool_call>def", nil},
		{"<|tool_call>call:get_weather{}<tool_call|>", nil},
		{"plain text no markers at all", []string{"STOP"}},
		{"hello STOP world", []string{"STOP"}},
		{"answerSTOP", []string{"STOP"}},
		{"<|tool_call>", []string{"STOP"}},
		{"pre<|tool_call>post", []string{"tool"}},   // stop is a substring of the marker
		{"aSTOPb<|tool_call>c", []string{"STOP"}},    // stop earlier, but tool is checked first
		{"<|tool_ca<|tool_call>x", nil},              // overlapping marker prefix
		{"end<|im_end|>tail", []string{"<|im_end|>"}}, // stop longer than the tool marker
		{"café<|tool_call>世界", nil},                  // unicode either side of the marker
		{"世界STOP世界", []string{"STOP"}},
		{"<|tool_call", []string{"call>"}},   // marker never completes; stop is its tail
		{"x<|>y<|tool_call>z", []string{"<|>"}},
		{"multi STOP and END here", []string{"STOP", "END"}},
	}
	cases := 0
	for _, f := range fixtures {
		b := []byte(f.text)
		for p := 0; p <= len(b); p++ {
			streamCase(t, []string{string(b[:p]), string(b[p:])}, f.stops)
			cases++
		}
		for p := 0; p <= len(b); p++ {
			for q := p; q <= len(b); q++ {
				streamCase(t, []string{string(b[:p]), string(b[p:q]), string(b[q:])}, f.stops)
				cases++
			}
		}
	}
	t.Logf("every-boundary differential: %d split cases across %d fixtures", cases, len(fixtures))
}

// streamFuzzAtoms are the adversarial building blocks: the full tool marker and
// its every-prefix/suffix, stop sequences and their fragments, marker-dense
// punctuation, reasoning markers, and unicode / invalid-UTF8 runs.
var streamFuzzAtoms = []string{
	"<|tool_call>", "<tool_call|>", "call:get_weather{}", `<|"|>`,
	"<|tool_", "tool_call>", "|tool_call>", "<|too", "l_call>", "<|tool_call",
	"STOP", "END", "<|im_end|>", "\n\n", "ST", "OP", "<|im_", "_end|>",
	"hello ", "world", "the quick brown fox ", "a", "b", " ",
	"café", "世界", "🙂", "naïve", "\xff\xfe", "<", "|", ">",
	"<think>", "</think>", "<|channel>analysis", "<channel|>",
}

var streamFuzzStops = [][]string{
	nil, {}, {"STOP"}, {"END"}, {"STOP", "END"}, {"<|im_end|>"},
	{"\n\n"}, {"世界"}, {"tool_call"}, {"X"}, {"call>", "STOP"}, {"<|"},
}

// randomByteSplit cuts b into parts at random byte positions — deliberately not
// rune-aligned, so mid-rune (byte-straddling) token boundaries are exercised.
func randomByteSplit(rng *rand.Rand, b []byte, parts int) []string {
	if parts <= 1 || len(b) == 0 {
		return []string{string(b)}
	}
	cuts := make([]int, 0, parts-1)
	for range parts - 1 {
		cuts = append(cuts, rng.Intn(len(b)+1))
	}
	sort.Ints(cuts)
	out := make([]string, 0, parts)
	prev := 0
	for _, c := range cuts {
		out = append(out, string(b[prev:c]))
		prev = c
	}
	return append(out, string(b[prev:]))
}

// TestContentStreamer_DifferentialFuzz_Random drives tens of thousands of random
// adversarial streams (deterministic seed) through both paths.
func TestContentStreamer_DifferentialFuzz_Random(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5EED))
	const trials = 20000
	for range trials {
		var content []byte
		for range 1 + rng.Intn(8) {
			content = append(content, streamFuzzAtoms[rng.Intn(len(streamFuzzAtoms))]...)
		}
		deltas := randomByteSplit(rng, content, 1+rng.Intn(6))
		stops := streamFuzzStops[rng.Intn(len(streamFuzzStops))]
		streamCase(t, deltas, stops)
	}
	t.Logf("randomised differential: %d trials (seed 0x5EED)", trials)
}

// FuzzContentStreamer is the native go-fuzz entry point (run: go test -fuzz
// FuzzContentStreamer). The seed corpus carries the boundary-spanning shapes;
// the engine explores from there.
func FuzzContentStreamer(f *testing.F) {
	f.Add("abc<|tool_call>def", uint8(3), "STOP")
	f.Add("hello STOP world", uint8(5), "STOP")
	f.Add("<|tool_call>call:f{}<tool_call|>", uint8(2), "")
	f.Add("café<|tool_call>世界", uint8(4), "END")
	f.Add("<|tool_ca<|tool_call>x", uint8(6), "<|")
	f.Fuzz(func(t *testing.T, content string, parts uint8, stopA string) {
		rng := rand.New(rand.NewSource(int64(len(content))*131 + int64(parts)))
		deltas := randomByteSplit(rng, []byte(content), int(parts%8)+1)
		var stops []string
		if stopA != "" {
			stops = []string{stopA}
		}
		streamCase(t, deltas, stops)
	})
}
