// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the adapter's stop-truncating stream path — the per-token stop
// scan that fires on every streamed token when stop sequences are set. The
// oracle variant is the pre-optimisation full-builder scan (O(N²) bytes); the
// windowed variant is streamStopWindow. Realistic 2K and 8K-token replies show
// the B/op collapse.
//
// Run: go test -bench='BenchmarkStreamStopWindow' -benchtime=20x -benchmem -run='^$' ./serving/

package serving

import (
	"testing"

	"dappco.re/go/inference"
)

func buildStreamTokens(n int) []inference.Token {
	words := []string{"the ", "model ", "emits ", "one ", "token ", "at ", "a ",
		"time ", "and ", "we ", "must ", "scan ", "for ", "the ", "stop ",
		"sequence ", "across ", "every ", "chunk ", "boundary. "}
	toks := make([]inference.Token, n)
	for i := range toks {
		toks[i] = inference.Token{Text: words[i%len(words)]}
	}
	return toks
}

var benchAdapterEmitted int

func benchAdapterStop(b *testing.B, n int, windowed bool) {
	toks := buildStreamTokens(n)
	stops := []string{"<|im_end|>"}
	cb := func(s string) error { benchAdapterEmitted += len(s); return nil }
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if windowed {
			seq := func(yield func(inference.Token) bool) {
				for _, tk := range toks {
					if !yield(tk) {
						return
					}
				}
			}
			_ = streamStopWindow(seq, stops, cb)
		} else {
			_, _ = oracleStreamStop(toks, stops, cb)
		}
	}
}

func BenchmarkStreamStopWindow_Windowed_2K(b *testing.B) { benchAdapterStop(b, 2048, true) }
func BenchmarkStreamStopWindow_Oracle_2K(b *testing.B)   { benchAdapterStop(b, 2048, false) }
func BenchmarkStreamStopWindow_Windowed_8K(b *testing.B) { benchAdapterStop(b, 8192, true) }
func BenchmarkStreamStopWindow_Oracle_8K(b *testing.B)   { benchAdapterStop(b, 8192, false) }
