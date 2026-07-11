// SPDX-Licence-Identifier: EUPL-1.2

// Differential fuzz pinning the bounded-window streamStopWindow to the
// pre-optimisation adapter stop path. oracleStreamStop replays the exact old
// GenerateStream/ChatStream loop — applyStopSequences over the whole builder
// every token — so any divergence in the callback token sequence, the early-stop
// point, or source-token consumption surfaces before the fix lands. Adversarial
// streams split stop sequences at every byte boundary (including mid-rune).
package serving

import (
	"math/rand"
	"sort"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

var errStopCB = core.NewError("cb stop")

// oracleStreamStop is the frozen reference: the adapter's stop-truncating stream
// loop exactly as it stood before the bounded-window rewrite. It records how many
// source tokens were consumed before a stop cut (or an early callback error).
func oracleStreamStop(toks []inference.Token, stops []string, cb func(string) error) (consumed int, err error) {
	full := core.NewBuilder()
	emitted := 0
	for _, tok := range toks {
		consumed++
		full.WriteString(tok.Text)
		truncated := applyStopSequences(full.String(), stops)
		if len(truncated) > emitted {
			if e := cb(truncated[emitted:]); e != nil {
				return consumed, e
			}
			emitted = len(truncated)
		}
		if len(truncated) < full.Len() {
			return consumed, nil // stop reached
		}
	}
	return consumed, nil
}

// adapterStopCase runs identical tokens + stops through the oracle and the
// windowed streamStopWindow, asserting identical callback sequences, identical
// early-stop callback errors, and identical source-token consumption (the stream
// must stop pulling tokens at the same point).
func adapterStopCase(t *testing.T, toks []inference.Token, stops []string, stopAt string) {
	t.Helper()

	var oracleOut []string
	oConsumed, oErr := oracleStreamStop(toks, stops, func(s string) error {
		oracleOut = append(oracleOut, s)
		if stopAt != "" && s == stopAt {
			return errStopCB
		}
		return nil
	})

	var winOut []string
	winConsumed := 0
	seq := func(yield func(inference.Token) bool) {
		for _, tk := range toks {
			winConsumed++
			if !yield(tk) {
				return
			}
		}
	}
	wErr := streamStopWindow(seq, stops, func(s string) error {
		winOut = append(winOut, s)
		if stopAt != "" && s == stopAt {
			return errStopCB
		}
		return nil
	})

	if !sameStrings(oracleOut, winOut) {
		t.Fatalf("callback seq mismatch\noracle=%q\nwindow=%q\ntoks=%v stops=%q", oracleOut, winOut, toks, stops)
	}
	if (oErr == nil) != (wErr == nil) {
		t.Fatalf("error mismatch oracle=%v window=%v\ntoks=%v stops=%q", oErr, wErr, toks, stops)
	}
	if oConsumed != winConsumed {
		t.Fatalf("consumed mismatch oracle=%d window=%d\ntoks=%v stops=%q", oConsumed, winConsumed, toks, stops)
	}
}

func sameStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestStreamStopWindow_DifferentialFuzz_EveryBoundary splits each fixture at
// every 2-way and 3-way byte boundary — the systematic proof that a stop
// sequence straddling a token break is cut identically to the whole-reply scan.
func TestStreamStopWindow_DifferentialFuzz_EveryBoundary(t *testing.T) {
	fixtures := []struct {
		text  string
		stops []string
	}{
		{"hello STOP world", []string{"STOP"}},
		{"answerSTOP", []string{"STOP"}},
		{"no stop here at all", []string{"STOP"}},
		{"end<|im_end|>tail", []string{"<|im_end|>"}},
		{"double\n\nnewline stop", []string{"\n\n"}},
		{"世界STOP世界", []string{"STOP"}},
		{"multi END and STOP", []string{"STOP", "END"}},
		{"STOPatstart", []string{"STOP"}},
		{"prefixSTO", []string{"STOP"}}, // partial, never completes
	}
	cases := 0
	for _, f := range fixtures {
		b := []byte(f.text)
		for p := 0; p <= len(b); p++ {
			adapterStopCase(t, []inference.Token{{Text: string(b[:p])}, {Text: string(b[p:])}}, f.stops, "")
			cases++
		}
		for p := 0; p <= len(b); p++ {
			for q := p; q <= len(b); q++ {
				adapterStopCase(t, []inference.Token{{Text: string(b[:p])}, {Text: string(b[p:q])}, {Text: string(b[q:])}}, f.stops, "")
				cases++
			}
		}
	}
	t.Logf("adapter every-boundary differential: %d split cases across %d fixtures", cases, len(fixtures))
}

var adapterFuzzAtoms = []string{
	"STOP", "END", "<|im_end|>", "\n\n", "ST", "OP", "<|im_", "_end|>",
	"hello ", "world", "the quick brown fox ", "a", "b", " ",
	"café", "世界", "🙂", "\xff\xfe", "<", "|", ">", "answer",
}

var adapterFuzzStops = [][]string{
	{"STOP"}, {"END"}, {"STOP", "END"}, {"<|im_end|>"}, {"\n\n"}, {"世界"}, {"X"}, {""},
}

func splitBytesRandom(rng *rand.Rand, b []byte, parts int) []inference.Token {
	if parts <= 1 || len(b) == 0 {
		return []inference.Token{{Text: string(b)}}
	}
	cuts := make([]int, 0, parts-1)
	for range parts - 1 {
		cuts = append(cuts, rng.Intn(len(b)+1))
	}
	sort.Ints(cuts)
	out := make([]inference.Token, 0, parts)
	prev := 0
	for _, c := range cuts {
		out = append(out, inference.Token{Text: string(b[prev:c])})
		prev = c
	}
	return append(out, inference.Token{Text: string(b[prev:])})
}

// TestStreamStopWindow_DifferentialFuzz_Random drives tens of thousands of random
// adversarial streams (deterministic seed), including runs whose callback errors
// out early, through both paths.
func TestStreamStopWindow_DifferentialFuzz_Random(t *testing.T) {
	rng := rand.New(rand.NewSource(0xA11CE))
	const trials = 20000
	for range trials {
		var content []byte
		for range 1 + rng.Intn(8) {
			content = append(content, adapterFuzzAtoms[rng.Intn(len(adapterFuzzAtoms))]...)
		}
		toks := splitBytesRandom(rng, content, 1+rng.Intn(6))
		stops := adapterFuzzStops[rng.Intn(len(adapterFuzzStops))]
		stopAt := ""
		if rng.Intn(4) == 0 && len(toks) > 0 {
			stopAt = toks[rng.Intn(len(toks))].Text // provoke an early callback error
		}
		adapterStopCase(t, toks, stops, stopAt)
	}
	t.Logf("adapter randomised differential: %d trials (seed 0xA11CE)", trials)
}

// FuzzStreamStopWindow is the native go-fuzz entry point (run: go test -fuzz
// FuzzStreamStopWindow).
func FuzzStreamStopWindow(f *testing.F) {
	f.Add("hello STOP world", uint8(5), "STOP")
	f.Add("answerSTOPtail", uint8(3), "STOP")
	f.Add("end<|im_end|>x", uint8(4), "<|im_end|>")
	f.Add("世界STOP世界", uint8(6), "STOP")
	f.Fuzz(func(t *testing.T, content string, parts uint8, stopA string) {
		rng := rand.New(rand.NewSource(int64(len(content))*131 + int64(parts)))
		toks := splitBytesRandom(rng, []byte(content), int(parts%8)+1)
		var stops []string
		if stopA != "" {
			stops = []string{stopA}
		}
		adapterStopCase(t, toks, stops, "")
	})
}
