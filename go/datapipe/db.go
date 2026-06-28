package datapipe

import (
	"database/sql"

	"dappco.re/go"
	_ "github.com/marcboeker/go-duckdb/v2"
)

// DB wraps a DuckDB connection.
type DB struct {
	conn *sql.DB
	path string
}

// OpenDB opens a DuckDB database file in read-only mode to avoid locking
// issues with the Python pipeline.
//
//	r := datapipe.OpenDB("/data/training.duckdb")
//	if !r.OK { return r }
//	db := r.Value.(*datapipe.DB)
func OpenDB(path string) core.Result {
	conn, err := sql.Open("duckdb", path+"?access_mode=READ_ONLY")
	if err != nil {
		return core.Fail(core.E("datapipe.OpenDB", core.Sprintf("open duckdb %s", path), err))
	}
	if err := conn.Ping(); err != nil {
		conn.Close()
		return core.Fail(core.E("datapipe.OpenDB", core.Sprintf("ping duckdb %s", path), err))
	}
	return core.Ok(&DB{conn: conn, path: path})
}

// OpenDBReadWrite opens a DuckDB database in read-write mode.
//
//	r := datapipe.OpenDBReadWrite("/data/training.duckdb")
//	if !r.OK { return r }
//	db := r.Value.(*datapipe.DB)
func OpenDBReadWrite(path string) core.Result {
	conn, err := sql.Open("duckdb", path)
	if err != nil {
		return core.Fail(core.E("datapipe.OpenDBReadWrite", core.Sprintf("open duckdb %s", path), err))
	}
	if err := conn.Ping(); err != nil {
		conn.Close()
		return core.Fail(core.E("datapipe.OpenDBReadWrite", core.Sprintf("ping duckdb %s", path), err))
	}
	return core.Ok(&DB{conn: conn, path: path})
}

// Close closes the database connection.
//
//	r := db.Close()
//	if !r.OK { return r }
func (db *DB) Close() core.Result {
	return core.ResultOf(nil, db.conn.Close())
}

// Path returns the database file path.
func (db *DB) Path() string {
	return db.path
}

// Exec executes a query without returning rows.
//
//	r := db.Exec("DELETE FROM training_examples WHERE idx = ?", idx)
//	if !r.OK { return r }
func (db *DB) Exec(query string, args ...any) core.Result {
	_, err := db.conn.Exec(query, args...)
	return core.ResultOf(nil, err)
}

// QueryRowScan executes a query expected to return at most one row and scans
// the result into dest. It is a convenience wrapper around sql.DB.QueryRow.
//
//	r := db.QueryRowScan("SELECT COUNT(*) FROM golden_set", &count)
//	if !r.OK { return r }
func (db *DB) QueryRowScan(query string, dest any, args ...any) core.Result {
	return core.ResultOf(nil, db.conn.QueryRow(query, args...).Scan(dest))
}

// GoldenSetRow represents one row from the golden_set table.
type GoldenSetRow struct {
	Idx       int
	SeedID    string
	Domain    string
	Voice     string
	Prompt    string
	Response  string
	GenTime   float64
	CharCount int
}

// ExpansionPromptRow represents one row from the expansion_prompts table.
type ExpansionPromptRow struct {
	Idx      int64
	SeedID   string
	Region   string
	Domain   string
	Language string
	Prompt   string
	PromptEn string
	Priority int
	Status   string
}

// QueryGoldenSet returns all golden set rows with responses >= minChars.
//
//	r := db.QueryGoldenSet(100)
//	if !r.OK { return r }
//	rows := r.Value.([]datapipe.GoldenSetRow)
func (db *DB) QueryGoldenSet(minChars int) core.Result {
	rows, err := db.conn.Query(
		"SELECT idx, seed_id, domain, voice, prompt, response, gen_time, char_count "+
			"FROM golden_set WHERE char_count >= ? ORDER BY idx",
		minChars,
	)
	if err != nil {
		return core.Fail(core.E("datapipe.DB.QueryGoldenSet", "query golden_set", err))
	}
	defer rows.Close()

	var result []GoldenSetRow
	for rows.Next() {
		var r GoldenSetRow
		if err := rows.Scan(&r.Idx, &r.SeedID, &r.Domain, &r.Voice,
			&r.Prompt, &r.Response, &r.GenTime, &r.CharCount); err != nil {
			return core.Fail(core.E("datapipe.DB.QueryGoldenSet", "scan golden_set row", err))
		}
		result = append(result, r)
	}
	return core.ResultOf(result, rows.Err())
}

// CountGoldenSet returns the total count of golden set rows.
//
//	r := db.CountGoldenSet()
//	if !r.OK { return r }
//	count := r.Value.(int)
func (db *DB) CountGoldenSet() core.Result {
	var count int
	err := db.conn.QueryRow("SELECT COUNT(*) FROM golden_set").Scan(&count)
	if err != nil {
		return core.Fail(core.E("datapipe.DB.CountGoldenSet", "count golden_set", err))
	}
	return core.Ok(count)
}

// QueryExpansionPrompts returns expansion prompts filtered by status.
//
//	r := db.QueryExpansionPrompts("pending", 50)
//	if !r.OK { return r }
//	rows := r.Value.([]datapipe.ExpansionPromptRow)
func (db *DB) QueryExpansionPrompts(status string, limit int) core.Result {
	query := "SELECT idx, seed_id, region, domain, language, prompt, prompt_en, priority, status " +
		"FROM expansion_prompts"
	var args []any

	if status != "" {
		query += " WHERE status = ?"
		args = append(args, status)
	}
	query += " ORDER BY priority, idx"

	if limit > 0 {
		query += core.Sprintf(" LIMIT %d", limit)
	}

	rows, err := db.conn.Query(query, args...)
	if err != nil {
		return core.Fail(core.E("datapipe.DB.QueryExpansionPrompts", "query expansion_prompts", err))
	}
	defer rows.Close()

	var result []ExpansionPromptRow
	for rows.Next() {
		var r ExpansionPromptRow
		if err := rows.Scan(&r.Idx, &r.SeedID, &r.Region, &r.Domain,
			&r.Language, &r.Prompt, &r.PromptEn, &r.Priority, &r.Status); err != nil {
			return core.Fail(core.E("datapipe.DB.QueryExpansionPrompts", "scan expansion_prompt row", err))
		}
		result = append(result, r)
	}
	return core.ResultOf(result, rows.Err())
}

// CountExpansionPrompts returns total and pending counts as a [2]int.
//
//	r := db.CountExpansionPrompts()
//	if !r.OK { return r }
//	counts := r.Value.([2]int)  // counts[0]=total, counts[1]=pending
func (db *DB) CountExpansionPrompts() core.Result {
	var total, pending int
	if err := db.conn.QueryRow("SELECT COUNT(*) FROM expansion_prompts").Scan(&total); err != nil {
		return core.Fail(core.E("datapipe.DB.CountExpansionPrompts", "count expansion_prompts", err))
	}
	if err := db.conn.QueryRow("SELECT COUNT(*) FROM expansion_prompts WHERE status = 'pending'").Scan(&pending); err != nil {
		return core.Fail(core.E("datapipe.DB.CountExpansionPrompts", "count pending expansion_prompts", err))
	}
	return core.Ok([2]int{total, pending})
}

// UpdateExpansionStatus updates the status of an expansion prompt by idx.
//
//	r := db.UpdateExpansionStatus(42, "done")
//	if !r.OK { return r }
func (db *DB) UpdateExpansionStatus(idx int64, status string) core.Result {
	_, err := db.conn.Exec("UPDATE expansion_prompts SET status = ? WHERE idx = ?", status, idx)
	if err != nil {
		return core.Fail(core.E("datapipe.DB.UpdateExpansionStatus", core.Sprintf("update expansion_prompt %d", idx), err))
	}
	return core.Ok(nil)
}

// QueryRows executes an arbitrary SQL query and returns results as maps.
//
//	r := db.QueryRows("SELECT * FROM golden_set LIMIT 10")
//	if !r.OK { return r }
//	rows := r.Value.([]map[string]any)
func (db *DB) QueryRows(query string, args ...any) core.Result {
	rows, err := db.conn.Query(query, args...)
	if err != nil {
		return core.Fail(core.E("datapipe.DB.QueryRows", "query", err))
	}
	defer rows.Close()

	cols, err := rows.Columns()
	if err != nil {
		return core.Fail(core.E("datapipe.DB.QueryRows", "columns", err))
	}

	// Scan into one reusable backing pair: ptrs[i] always points at values[i],
	// so the slices are identical every row. The per-row map below snapshots
	// each value out before the next Scan overwrites it, so reuse is safe and
	// keeps the result byte-identical while dropping two allocations per row.
	values := make([]any, len(cols))
	ptrs := make([]any, len(cols))
	for i := range values {
		ptrs[i] = &values[i]
	}

	var result []map[string]any
	for rows.Next() {
		if err := rows.Scan(ptrs...); err != nil {
			return core.Fail(core.E("datapipe.DB.QueryRows", "scan", err))
		}
		row := make(map[string]any, len(cols))
		for i, col := range cols {
			row[col] = values[i]
		}
		result = append(result, row)
	}
	return core.ResultOf(result, rows.Err())
}

// EnsureScoringTables creates the scoring tables if they don't exist.
func (db *DB) EnsureScoringTables() {
	db.conn.Exec(core.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
		model TEXT, run_id TEXT, label TEXT, iteration INTEGER,
		correct INTEGER, total INTEGER, accuracy DOUBLE,
		scored_at TIMESTAMP DEFAULT current_timestamp,
		PRIMARY KEY (run_id, label)
	)`, TableCheckpointScores))
	db.conn.Exec(core.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
		model TEXT, run_id TEXT, label TEXT, probe_id TEXT,
		passed BOOLEAN, response TEXT, iteration INTEGER,
		scored_at TIMESTAMP DEFAULT current_timestamp,
		PRIMARY KEY (run_id, label, probe_id)
	)`, TableProbeResults))
	db.conn.Exec(`CREATE TABLE IF NOT EXISTS scoring_results (
		model TEXT, prompt_id TEXT, suite TEXT,
		dimension TEXT, score DOUBLE,
		scored_at TIMESTAMP DEFAULT current_timestamp
	)`)
}

// WriteScoringResult writes a single scoring dimension result to DuckDB.
//
//	r := db.WriteScoringResult("gemma-3-1b", "p001", "capability", "reasoning", 8.5)
//	if !r.OK { return r }
func (db *DB) WriteScoringResult(model, promptID, suite, dimension string, score float64) core.Result {
	_, err := db.conn.Exec(
		`INSERT INTO scoring_results (model, prompt_id, suite, dimension, score) VALUES (?, ?, ?, ?, ?)`,
		model, promptID, suite, dimension, score,
	)
	return core.ResultOf(nil, err)
}

// TableCounts returns row counts for all known tables.
//
//	r := db.TableCounts()
//	if !r.OK { return r }
//	counts := r.Value.(map[string]int)
func (db *DB) TableCounts() core.Result {
	tables := []string{"golden_set", "expansion_prompts", "seeds", "prompts",
		"training_examples", "gemini_responses", "benchmark_questions", "benchmark_results", "validations",
		TableCheckpointScores, TableProbeResults, "scoring_results"}

	counts := make(map[string]int)
	for _, t := range tables {
		var count int
		err := db.conn.QueryRow(core.Sprintf("SELECT COUNT(*) FROM %s", t)).Scan(&count)
		if err != nil {
			continue
		}
		counts[t] = count
	}
	return core.Ok(counts)
}
