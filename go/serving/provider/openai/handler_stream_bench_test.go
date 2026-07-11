// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the streamed content-detection path — the per-token stop /
// tool-marker scan that fires on every SSE chunk of a chat-completion. The
// oracle variant is the pre-optimisation full-candidate concat (O(N²) bytes);
// the windowed variant is the bounded-window contentStreamer. Realistic 2K and
// 8K-token replies show the B/op collapse.
//
// Run: go test -bench='BenchmarkContentStreamer' -benchtime=20x -benchmem -run='^$' .

package openai

import "testing"

// buildStreamDeltas splits a realistic assistant reply into n short tokens with
// a stop sequence that never triggers — the worst case where the whole reply is
// scanned and accumulated to the end.
func buildStreamDeltas(n int) []string {
	words := []string{"the ", "model ", "emits ", "one ", "token ", "at ", "a ",
		"time ", "and ", "we ", "must ", "scan ", "for ", "stop ", "or ", "tool ",
		"markers ", "across ", "every ", "chunk ", "boundary. "}
	out := make([]string, n)
	for i := range out {
		out[i] = words[i%len(words)]
	}
	return out
}

var (
	benchStreamOutcome streamOutcome
	benchStreamEmitted string
)

func benchContentStreamer(b *testing.B, n int, windowed bool) {
	deltas := buildStreamDeltas(n)
	stops := []string{"<|im_end|>"}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if windowed {
			cs := newContentStreamer(stops)
			for _, d := range deltas {
				benchStreamOutcome = cs.step(d)
			}
			benchStreamEmitted = cs.emitted()
		} else {
			o := &oracleStreamer{stops: stops}
			for _, d := range deltas {
				benchStreamOutcome = o.step(d)
			}
			benchStreamEmitted = o.emitted()
		}
	}
}

func BenchmarkContentStreamer_Windowed_2K(b *testing.B) { benchContentStreamer(b, 2048, true) }
func BenchmarkContentStreamer_Oracle_2K(b *testing.B)   { benchContentStreamer(b, 2048, false) }
func BenchmarkContentStreamer_Windowed_8K(b *testing.B) { benchContentStreamer(b, 8192, true) }
func BenchmarkContentStreamer_Oracle_8K(b *testing.B)   { benchContentStreamer(b, 8192, false) }
