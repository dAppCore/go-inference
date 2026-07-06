// SPDX-Licence-Identifier: EUPL-1.2

// Allocation contracts for the observability run-tree (AX-11). StartRun+Finish
// is the per-inference path; MeanByKey rolls up a run's feedback. Deterministic
// id and clock generators keep the benches free of UUID and wall-clock noise so
// each line isolates the run-tree's own buffering.
//
// Run: go test -bench=. -benchmem -run='^$' ./obs/
package obs

import (
	"testing"
	"time"

	core "dappco.re/go"
)

func benchIDGen() IDGen {
	n := 0
	return func() string {
		n++
		return core.Sprintf("run-%d", n)
	}
}

func benchClock() Clock {
	t := time.Unix(0, 0)
	return func() time.Time { return t }
}

func benchKeyFor(i int) string {
	switch i % 4 {
	case 0:
		return "ethics"
	case 1:
		return "helpfulness"
	case 2:
		return "length"
	default:
		return "quality"
	}
}

var (
	benchRun  *Run
	benchMean map[string]float64
)

func BenchmarkRunTree_StartFinish(b *testing.B) {
	tree := NewRunTree(benchIDGen(), benchClock())
	inputs := map[string]any{"prompt": "hello"}
	outputs := map[string]any{"reply": "world"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		run := tree.StartRun("chat", inputs)
		tree.Finish(run, outputs, nil)
		benchRun = run
	}
}

func BenchmarkRunTree_MeanByKey(b *testing.B) {
	tree := NewRunTree(benchIDGen(), benchClock())
	root := tree.StartRun("chat", nil)
	for i := range 16 {
		tree.Record(Feedback{RunID: root.ID, Key: benchKeyFor(i), Score: float64(i%10) / 10})
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMean = tree.MeanByKey(root.ID)
	}
}
