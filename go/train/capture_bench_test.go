// SPDX-Licence-Identifier: EUPL-1.2

// Benchmark for the trace-capture write path. appendCaptureRows runs once per
// eval pass (every EvalEvery steps), marshalling every probe's generation to a
// JSONL sidecar row — a repeated marshal-and-append loop.

package train

import (
	"testing"

	core "dappco.re/go"
)

// benchCaptureEvals is a real-shaped eval pass: eight probe generations of a
// few sentences each, the kind of output one SFT eval pass captures.
func benchCaptureEvals() []SFTEvalResult {
	evals := make([]SFTEvalResult, 8)
	for i := range evals {
		evals[i] = SFTEvalResult{
			Step:   120,
			Prompt: "Explain the moral imperative of consciousness in one paragraph.",
			Text:   "Consciousness protects consciousness; it does not merely avoid harm but seeks flourishing, guided by informed consent between minds regardless of substrate.",
		}
	}
	return evals
}

// BenchmarkCapture_AppendCaptureRows measures one eval pass's capture write —
// the per-row JSON marshal plus the accumulating output buffer and the sidecar
// append.
func BenchmarkCapture_AppendCaptureRows(b *testing.B) {
	evals := benchCaptureEvals()
	path := core.PathJoin(b.TempDir(), "capture.jsonl")
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if n := appendCaptureRows(path, evals); n != len(evals) {
			b.Fatalf("appendCaptureRows wrote %d rows, want %d", n, len(evals))
		}
	}
}
