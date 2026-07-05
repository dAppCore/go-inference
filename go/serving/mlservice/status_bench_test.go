package mlservice

import (
	"testing"

	core "dappco.re/go"
)

var (
	benchTraining []trainingRow
	benchGen      []genRow
)

// benchStatusRows mimics InfluxDB training_status output: several models, each
// appearing multiple times (latest-first), so the dedup "seen" path is exercised.
func benchStatusRows() []map[string]any {
	rows := make([]map[string]any, 0, 10)
	for _, m := range []string{"gemma4", "ministral", "qwen35", "gemma4", "ministral"} {
		rows = append(rows, map[string]any{
			"model": m, "status": "running", "iteration": float64(120),
			"total_iters": float64(500), "pct": float64(24.0),
		})
	}
	return rows
}

func benchLossRows() []map[string]any {
	rows := make([]map[string]any, 0, 10)
	for _, m := range []string{"gemma4", "ministral", "qwen35", "gemma4"} {
		rows = append(rows, map[string]any{"model": m, "loss": float64(1.234)})
	}
	return rows
}

func benchGenRows() []map[string]any {
	rows := make([]map[string]any, 0, 8)
	for _, w := range []string{"worker-a", "worker-b", "worker-c", "worker-a"} {
		rows = append(rows, map[string]any{
			"worker": w, "completed": float64(42), "target": float64(100), "pct": float64(42.0),
		})
	}
	return rows
}

func BenchmarkStatus_dedupeTraining(b *testing.B) {
	statusRows := benchStatusRows()
	lossRows := benchLossRows()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchTraining = dedupeTraining(statusRows, lossRows)
	}
}

func BenchmarkStatus_dedupeGeneration(b *testing.B) {
	rows := benchGenRows()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchGen = dedupeGeneration(rows)
	}
}

func BenchmarkStatus_PrintStatus(b *testing.B) {
	queries := map[string][]map[string]any{
		"training_status":     benchStatusRows(),
		"training_loss":       benchLossRows(),
		"golden_gen_progress": benchGenRows(),
		"expansion_progress":  benchGenRows(),
	}
	influx, _ := newFakeInflux(b, queries, 0)
	buf := core.NewBuffer(nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf.Reset()
		benchResult = PrintStatus(influx, buf)
	}
}
