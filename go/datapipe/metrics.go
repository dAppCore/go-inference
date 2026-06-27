package datapipe

import (
	"io"
	"time"

	"dappco.re/go"
	"dappco.re/go/store"
)

// PushMetrics queries golden_set stats from DuckDB and writes them to InfluxDB
// as golden_set_stats, golden_set_domain, and golden_set_voice measurements.
//
//	r := datapipe.PushMetrics(db, influx, os.Stdout)
//	if !r.OK { return r }
func PushMetrics(db *store.DuckDB, influx *InfluxClient, w io.Writer) core.Result {
	// Overall stats.
	var total, domains, voices int
	var avgGenTime, avgChars float64
	err := db.Conn().QueryRow(
		"SELECT count(*), count(DISTINCT domain), count(DISTINCT voice), "+
			"coalesce(avg(gen_time), 0), coalesce(avg(char_count), 0) FROM golden_set",
	).Scan(&total, &domains, &voices, &avgGenTime, &avgChars)
	if err != nil {
		return core.Fail(core.E("datapipe.PushMetrics", "query golden_set stats", err))
	}

	if total == 0 {
		core.Print(w, "golden_set is empty, nothing to push")
		return core.Ok(nil)
	}

	completionPct := float64(total) / float64(TargetTotal) * 100.0
	ts := time.Now().UnixNano()

	var lines []string

	// Overall stats point.
	lines = append(lines, core.Sprintf(
		"golden_set_stats total_examples=%di,domains=%di,voices=%di,avg_gen_time=%.2f,avg_response_chars=%.0f,completion_pct=%.1f %d",
		total, domains, voices, avgGenTime, avgChars, completionPct, ts,
	))

	// Per-domain breakdown.
	domainRows, err := db.Conn().Query(
		"SELECT domain, count(*) AS cnt, coalesce(avg(gen_time), 0) AS avg_gt FROM golden_set GROUP BY domain ORDER BY domain",
	)
	if err != nil {
		return core.Fail(core.E("datapipe.PushMetrics", "query golden_set domains", err))
	}
	defer domainRows.Close()

	for domainRows.Next() {
		var domain string
		var count int
		var avgGT float64
		if err := domainRows.Scan(&domain, &count, &avgGT); err != nil {
			return core.Fail(core.E("datapipe.PushMetrics", "scan domain row", err))
		}
		lines = append(lines, core.Sprintf(
			"golden_set_domain,domain=%s count=%di,avg_gen_time=%.2f %d",
			EscapeLp(domain), count, avgGT, ts,
		))
	}
	if err := domainRows.Err(); err != nil {
		return core.Fail(core.E("datapipe.PushMetrics", "iterate domain rows", err))
	}

	// Per-voice breakdown.
	voiceRows, err := db.Conn().Query(
		"SELECT voice, count(*) AS cnt, coalesce(avg(char_count), 0) AS avg_cc, coalesce(avg(gen_time), 0) AS avg_gt FROM golden_set GROUP BY voice ORDER BY voice",
	)
	if err != nil {
		return core.Fail(core.E("datapipe.PushMetrics", "query golden_set voices", err))
	}
	defer voiceRows.Close()

	for voiceRows.Next() {
		var voice string
		var count int
		var avgCC, avgGT float64
		if err := voiceRows.Scan(&voice, &count, &avgCC, &avgGT); err != nil {
			return core.Fail(core.E("datapipe.PushMetrics", "scan voice row", err))
		}
		lines = append(lines, core.Sprintf(
			"golden_set_voice,voice=%s count=%di,avg_chars=%.0f,avg_gen_time=%.2f %d",
			EscapeLp(voice), count, avgCC, avgGT, ts,
		))
	}
	if err := voiceRows.Err(); err != nil {
		return core.Fail(core.E("datapipe.PushMetrics", "iterate voice rows", err))
	}

	// Write all points to InfluxDB.
	if rWrite := influx.WriteLp(lines); !rWrite.OK {
		return core.Fail(core.E("datapipe.PushMetrics", "write metrics to influxdb", rWrite.Value.(error)))
	}

	core.Print(w, "Pushed %d points to InfluxDB", len(lines))
	core.Print(w, "  total=%d  domains=%d  voices=%d  completion=%.1f%%",
		total, domains, voices, completionPct)
	core.Print(w, "  avg_gen_time=%.2fs  avg_chars=%.0f", avgGenTime, avgChars)

	return core.Ok(nil)
}
