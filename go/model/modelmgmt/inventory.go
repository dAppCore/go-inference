package modelmgmt

import (
	"io"

	"dappco.re/go"
	"dappco.re/go/inference/eval/datapipe"
)

// tableOrder defines the canonical display order for inventory tables.
var tableOrder = []string{
	"golden_set", "expansion_prompts", "seeds", "prompts",
	"training_examples", "gemini_responses", "benchmark_questions",
	"benchmark_results", "validations", datapipe.TableCheckpointScores,
	datapipe.TableProbeResults, "scoring_results",
}

// tableDetail holds extra context for a single table beyond its row count.
type tableDetail struct {
	notes []string
}

// PrintInventory queries all known DuckDB tables and prints a formatted
// inventory with row counts, detail breakdowns, and a grand total.
func PrintInventory(db *datapipe.DB, w io.Writer) core.Result {
	rCounts := db.TableCounts()
	if !rCounts.OK {
		return core.Fail(core.E("modelmgmt.PrintInventory", "table counts", rCounts.Value.(error)))
	}
	counts := rCounts.Value.(map[string]int)

	details := gatherDetails(db, counts)

	core.Print(w, "DuckDB Inventory")
	core.Print(w, "%s", repeatString("-", 52))

	grand := 0
	for _, table := range tableOrder {
		count, ok := counts[table]
		if !ok {
			continue
		}
		grand += count
		line := core.Sprintf("  %-24s %8d rows", table, count)

		if d, has := details[table]; has && len(d.notes) > 0 {
			line = core.Concat(line, "  (", core.Join(", ", d.notes...), ")")
		}
		core.Print(w, "%s", line)
	}

	core.Print(w, "%s", repeatString("-", 52))
	core.Print(w, "  %-24s %8d rows", "TOTAL", grand)

	return core.Ok(nil)
}

// gatherDetails runs per-table detail queries and returns annotations keyed
// by table name. Errors on individual queries are silently ignored so the
// inventory always prints.
func gatherDetails(db *datapipe.DB, counts map[string]int) map[string]*tableDetail {
	details := make(map[string]*tableDetail)

	// golden_set: progress toward target
	if count, ok := counts["golden_set"]; ok {
		pct := float64(count) / float64(datapipe.TargetTotal) * 100
		details["golden_set"] = &tableDetail{
			notes: []string{core.Sprintf("%.1f%% of %d target", pct, datapipe.TargetTotal)},
		}
	}

	// training_examples: distinct sources
	if _, ok := counts["training_examples"]; ok {
		if rRows := db.QueryRows("SELECT COUNT(DISTINCT source) AS n FROM training_examples"); rRows.OK {
			rows := rRows.Value.([]map[string]any)
			if len(rows) > 0 {
				n := toInt(rows[0]["n"])
				details["training_examples"] = &tableDetail{
					notes: []string{core.Sprintf("%d sources", n)},
				}
			}
		}
	}

	// prompts: distinct domains and voices
	if _, ok := counts["prompts"]; ok {
		d := &tableDetail{}
		if rRows := db.QueryRows("SELECT COUNT(DISTINCT domain) AS n FROM prompts"); rRows.OK {
			rows := rRows.Value.([]map[string]any)
			if len(rows) > 0 {
				d.notes = append(d.notes, core.Sprintf("%d domains", toInt(rows[0]["n"])))
			}
		}
		if rRows := db.QueryRows("SELECT COUNT(DISTINCT voice) AS n FROM prompts"); rRows.OK {
			rows := rRows.Value.([]map[string]any)
			if len(rows) > 0 {
				d.notes = append(d.notes, core.Sprintf("%d voices", toInt(rows[0]["n"])))
			}
		}
		if len(d.notes) > 0 {
			details["prompts"] = d
		}
	}

	// gemini_responses: group by source_model
	if _, ok := counts["gemini_responses"]; ok {
		if rRows := db.QueryRows("SELECT source_model, COUNT(*) AS n FROM gemini_responses GROUP BY source_model ORDER BY n DESC"); rRows.OK {
			rows := rRows.Value.([]map[string]any)
			if len(rows) > 0 {
				var parts []string
				for _, row := range rows {
					model := core.MapString(row, "source_model")
					n := toInt(row["n"])
					if model != "" {
						parts = append(parts, core.Sprintf("%s:%d", model, n))
					}
				}
				if len(parts) > 0 {
					details["gemini_responses"] = &tableDetail{notes: parts}
				}
			}
		}
	}

	// benchmark_results: distinct source categories
	if _, ok := counts["benchmark_results"]; ok {
		if rRows := db.QueryRows("SELECT COUNT(DISTINCT source) AS n FROM benchmark_results"); rRows.OK {
			rows := rRows.Value.([]map[string]any)
			if len(rows) > 0 {
				n := toInt(rows[0]["n"])
				details["benchmark_results"] = &tableDetail{
					notes: []string{core.Sprintf("%d categories", n)},
				}
			}
		}
	}

	return details
}

// toInt converts a DuckDB value to int. DuckDB returns integers as int64 (not
// float64 like InfluxDB), so we handle both types.
func toInt(v any) int {
	switch n := v.(type) {
	case int64:
		return int(n)
	case int32:
		return int(n)
	case float64:
		return int(n)
	default:
		return 0
	}
}
