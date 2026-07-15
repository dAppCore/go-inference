package datapipe

import (
	"io" // Note: AX-6 intrinsic - io.Writer is the public output surface; core exposes no Writer primitive.

	"dappco.re/go"
	"dappco.re/go/store"
)

// SeedInfluxConfig holds options for the SeedInflux migration.
type SeedInfluxConfig struct {
	Force     bool
	BatchSize int
}

// SeedInflux migrates golden_set rows from DuckDB into InfluxDB as
// gold_gen measurement points. This is a one-time migration tool;
// it skips the write when InfluxDB already contains all records
// unless Force is set.
//
//	r := datapipe.SeedInflux(db, influx, cfg, os.Stdout)
//	if !r.OK { return r }
func SeedInflux(db *store.DuckDB, influx *InfluxClient, cfg SeedInfluxConfig, w io.Writer) core.Result {
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 500
	}

	// Count source rows in DuckDB.
	var total int
	if err := db.Conn().QueryRow("SELECT count(*) FROM golden_set").Scan(&total); err != nil {
		return core.Fail(core.E("datapipe.SeedInflux", "no golden_set table", err))
	}

	// Check how many distinct records InfluxDB already has.
	existing := 0
	rQuery := influx.QuerySQL("SELECT count(DISTINCT i) AS n FROM gold_gen")
	if rQuery.OK {
		rows := rQuery.Value.([]map[string]any)
		if len(rows) > 0 {
			if n, ok := rows[0]["n"].(float64); ok {
				existing = int(n)
			}
		}
	}

	core.Print(w, "DuckDB has %d records, InfluxDB golden_gen has %d", total, existing)

	if existing >= total && !cfg.Force {
		core.Print(w, "InfluxDB already has all records. Use --force to re-seed.")
		return core.Ok(nil)
	}

	// Query all golden_set rows from DuckDB.
	dbRows, err := db.Conn().Query(
		"SELECT idx, seed_id, domain, voice, gen_time, char_count FROM golden_set ORDER BY idx",
	)
	if err != nil {
		return core.Fail(core.E("datapipe.SeedInflux", "query golden_set", err))
	}
	defer dbRows.Close()

	var batch []string
	written := 0

	for dbRows.Next() {
		var idx int
		var seedID, domain, voice string
		var genTime float64
		var charCount int

		if err := dbRows.Scan(&idx, &seedID, &domain, &voice, &genTime, &charCount); err != nil {
			return core.Fail(core.E("datapipe.SeedInflux", core.Sprintf("scan row %d", written), err))
		}

		// Build line protocol point.
		// Tags: i (idx), w (worker), d (domain), v (voice)
		// Fields: seed_id (string), gen_time (float), chars (integer)
		escapedSeedID := core.Replace(seedID, `"`, `\"`)

		line := core.Sprintf(
			"gold_gen,i=%s,w=migration,d=%s,v=%s seed_id=\"%s\",gen_time=%v,chars=%di",
			EscapeLp(core.Itoa(idx)),
			EscapeLp(domain),
			EscapeLp(voice),
			escapedSeedID,
			genTime,
			charCount,
		)
		batch = append(batch, line)

		if len(batch) >= cfg.BatchSize {
			if rWrite := influx.WriteLp(batch); !rWrite.OK {
				return core.Fail(core.E("datapipe.SeedInflux", core.Sprintf("write batch at row %d", written), rWrite.Value.(error)))
			}
			written += len(batch)
			batch = batch[:0]

			if written%2000 == 0 {
				core.Print(w, "  wrote %d / %d", written, total)
			}
		}
	}

	if err := dbRows.Err(); err != nil {
		return core.Fail(core.E("datapipe.SeedInflux", "iterate golden_set rows", err))
	}

	// Flush remaining batch.
	if len(batch) > 0 {
		if rWrite := influx.WriteLp(batch); !rWrite.OK {
			return core.Fail(core.E("datapipe.SeedInflux", "write final batch", rWrite.Value.(error)))
		}
		written += len(batch)
	}

	core.Print(w, "Seeded %d records into InfluxDB golden_gen", written)
	return core.Ok(nil)
}
