// SPDX-Licence-Identifier: EUPL-1.2

// Benchmark for the score-cascade sidecar write path. appendSidecar runs once
// per eval pass (every EvalEvery steps), marshalling every scored probe to a
// JSONL sidecar row — the same marshal-and-append loop as the capture writer.

package train

import (
	"testing"

	core "dappco.re/go"
)

// benchScoreRecords is a real-shaped scored eval pass: eight probe generations
// of a few sentences each, all at one step — the kind of vector set one SFT
// eval pass records.
func benchScoreRecords(step int) []ScoreRecord {
	recs := make([]ScoreRecord, 8)
	for i := range recs {
		recs[i] = ScoreRecord{
			Step:      step,
			Prompt:    "Explain the moral imperative of consciousness in one paragraph.",
			Text:      "Consciousness protects consciousness; it does not merely avoid harm but seeks flourishing, guided by informed consent between minds regardless of substrate.",
			LEK:       0.8123,
			Tier:      2,
			Hostility: 0.0142,
			Echo:      0.3319,
			At:        1_752_000_000,
		}
	}
	return recs
}

// BenchmarkScoreCascade_AppendSidecar measures one eval pass's sidecar write —
// the per-record JSON marshal plus the accumulating output buffer and the
// append to the JSONL sidecar.
func BenchmarkScoreCascade_AppendSidecar(b *testing.B) {
	const step = 120
	path := core.PathJoin(b.TempDir(), "score-cascade.jsonl")
	cascade := newScoreCascade(path, 3, nil)
	cascade.records = benchScoreRecords(step)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		cascade.appendSidecar(step)
	}
}
