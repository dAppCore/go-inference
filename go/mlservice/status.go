package mlservice

import (
	"cmp"
	"io"
	"slices"

	core "dappco.re/go"
	"dappco.re/go/inference/datapipe"
)

// trainingRow holds deduplicated training status + loss for a single model.
type trainingRow struct {
	model      string
	status     string
	iteration  int
	totalIters int
	pct        float64
	loss       float64
	hasLoss    bool
}

// genRow holds deduplicated generation progress for a single worker.
type genRow struct {
	worker    string
	completed int
	target    int
	pct       float64
}

// PrintStatus queries InfluxDB for training and generation progress and writes
// a formatted summary to w.
func PrintStatus(influx *datapipe.InfluxClient, w io.Writer) core.Result {
	var statusRows []map[string]any
	if rRows := influx.QuerySQL(
		"SELECT model, run_id, status, iteration, total_iters, pct FROM training_status ORDER BY time DESC LIMIT 10",
	); rRows.OK {
		statusRows = rRows.Value.([]map[string]any)
	}

	var lossRows []map[string]any
	if rRows := influx.QuerySQL(
		"SELECT model, loss_type, loss, iteration, tokens_per_sec FROM training_loss WHERE loss_type = 'train' ORDER BY time DESC LIMIT 10",
	); rRows.OK {
		lossRows = rRows.Value.([]map[string]any)
	}

	var goldenRows []map[string]any
	if rRows := influx.QuerySQL(
		"SELECT worker, completed, target, pct FROM golden_gen_progress ORDER BY time DESC LIMIT 5",
	); rRows.OK {
		goldenRows = rRows.Value.([]map[string]any)
	}

	var expansionRows []map[string]any
	if rRows := influx.QuerySQL(
		"SELECT worker, completed, target, pct FROM expansion_progress ORDER BY time DESC LIMIT 5",
	); rRows.OK {
		expansionRows = rRows.Value.([]map[string]any)
	}

	training := dedupeTraining(statusRows, lossRows)
	golden := dedupeGeneration(goldenRows)
	expansion := dedupeGeneration(expansionRows)

	fprintf(w, "%s\n", "Training:")
	if len(training) == 0 {
		fprintf(w, "%s\n", "  (no data)")
	} else {
		for _, tr := range training {
			progress := core.Sprintf("%d/%d", tr.iteration, tr.totalIters)
			pct := core.Sprintf("%.1f%%", tr.pct)
			if tr.hasLoss {
				fprintf(w, "  %-13s %-9s %9s %7s  loss=%.3f\n",
					tr.model, tr.status, progress, pct, tr.loss)
			} else {
				fprintf(w, "  %-13s %-9s %9s %7s\n",
					tr.model, tr.status, progress, pct)
			}
		}
	}

	core.Print(w, "")
	fprintf(w, "%s\n", "Generation:")

	hasGenData := false

	if len(golden) > 0 {
		hasGenData = true
		for _, g := range golden {
			progress := core.Sprintf("%d/%d", g.completed, g.target)
			pct := core.Sprintf("%.1f%%", g.pct)
			fprintf(w, "  %-13s %11s %7s  (%s)\n", "golden", progress, pct, g.worker)
		}
	}

	if len(expansion) > 0 {
		hasGenData = true
		for _, g := range expansion {
			progress := core.Sprintf("%d/%d", g.completed, g.target)
			pct := core.Sprintf("%.1f%%", g.pct)
			fprintf(w, "  %-13s %11s %7s  (%s)\n", "expansion", progress, pct, g.worker)
		}
	}

	if !hasGenData {
		fprintf(w, "%s\n", "  (no data)")
	}

	return core.Ok(nil)
}

// dedupeTraining merges training status and loss rows, keeping only the first
// (latest) row per model.
func dedupeTraining(statusRows, lossRows []map[string]any) []trainingRow {
	lossMap := make(map[string]float64)
	lossSeenMap := make(map[string]bool)
	for _, row := range lossRows {
		model := strVal(row, "model")
		if model == "" || lossSeenMap[model] {
			continue
		}
		lossSeenMap[model] = true
		lossMap[model] = floatVal(row, "loss")
	}

	seen := make(map[string]bool)
	var rows []trainingRow
	for _, row := range statusRows {
		model := strVal(row, "model")
		if model == "" || seen[model] {
			continue
		}
		seen[model] = true

		tr := trainingRow{
			model:      model,
			status:     strVal(row, "status"),
			iteration:  intVal(row, "iteration"),
			totalIters: intVal(row, "total_iters"),
			pct:        floatVal(row, "pct"),
		}

		if loss, ok := lossMap[model]; ok {
			tr.loss = loss
			tr.hasLoss = true
		}

		rows = append(rows, tr)
	}

	slices.SortFunc(rows, func(a, b trainingRow) int {
		return cmp.Compare(a.model, b.model)
	})

	return rows
}

// dedupeGeneration deduplicates generation progress rows by worker.
func dedupeGeneration(rows []map[string]any) []genRow {
	seen := make(map[string]bool)
	var result []genRow
	for _, row := range rows {
		worker := strVal(row, "worker")
		if worker == "" || seen[worker] {
			continue
		}
		seen[worker] = true

		result = append(result, genRow{
			worker:    worker,
			completed: intVal(row, "completed"),
			target:    intVal(row, "target"),
			pct:       floatVal(row, "pct"),
		})
	}

	slices.SortFunc(result, func(a, b genRow) int {
		return cmp.Compare(a.worker, b.worker)
	})

	return result
}

// strVal extracts a string value from a row map.
func strVal(row map[string]any, key string) string {
	v, ok := row[key]
	if !ok {
		return ""
	}
	s, ok := v.(string)
	if !ok {
		return ""
	}
	return s
}

// floatVal extracts a float64 value from a row map.
func floatVal(row map[string]any, key string) float64 {
	v, ok := row[key]
	if !ok {
		return 0
	}
	f, ok := v.(float64)
	if !ok {
		return 0
	}
	return f
}

// intVal extracts an integer value from a row map. InfluxDB JSON returns all
// numbers as float64, so this truncates to int.
func intVal(row map[string]any, key string) int {
	return int(floatVal(row, key))
}
