package datapipe

import "dappco.re/go"

func seedMLDB(t *core.T) *DB {
	t.Helper()
	db := newTestDB(t)
	requireResultOK(t, db.Exec(`CREATE TABLE golden_set (
		idx INTEGER, seed_id VARCHAR, domain VARCHAR, voice VARCHAR,
		prompt VARCHAR, response VARCHAR, gen_time DOUBLE, char_count INTEGER
	)`))
	requireResultOK(t, db.Exec(`INSERT INTO golden_set VALUES (1,'s1','ethics','calm','p','long response',1.5,13)`))
	requireResultOK(t, db.Exec(`CREATE TABLE expansion_prompts (
		idx BIGINT, seed_id VARCHAR, region VARCHAR, domain VARCHAR, language VARCHAR,
		prompt VARCHAR, prompt_en VARCHAR, priority INTEGER, status VARCHAR
	)`))
	requireResultOK(t, db.Exec(`INSERT INTO expansion_prompts VALUES (1,'s1','en','ethics','en','p','',2,'pending')`))
	return db
}

func TestDb_OpenDB_Good(t *core.T) {
	db := seedMLDB(t)
	path := db.Path()
	// Close the writer before opening read-only: duckdb (go-duckdb/v2)
	// refuses a second handle to the same file under a different access_mode
	// while another connection is live. In production the Python pipeline
	// writes and exits before Go opens the file read-only.
	requireResultOK(t, db.Close())
	r := OpenDB(path)
	requireResultOK(t, r)
	ro := r.Value.(*DB)
	defer ro.Close()
	core.AssertEqual(t, path, ro.Path())
}

func TestDb_OpenDB_Bad(t *core.T) {
	r := OpenDB(core.JoinPath(t.TempDir(), "missing.duckdb"))
	assertResultError(t, r)
	core.AssertFalse(t, r.OK)
	core.AssertError(t, r.Value.(error))
}

func TestDb_OpenDB_Ugly(t *core.T) {
	db := seedMLDB(t)
	path := db.Path()
	requireResultOK(t, db.Close()) // see TestDb_OpenDB_Good: writer must close before a read-only handle
	r := OpenDB(path)
	requireResultOK(t, r)
	ro := r.Value.(*DB)
	assertResultError(t, ro.Exec("CREATE TABLE blocked(x INTEGER)"))
	_ = ro.Close()
}

func TestDb_OpenDBReadWrite_Good(t *core.T) {
	r := OpenDBReadWrite(core.JoinPath(t.TempDir(), "rw.duckdb"))
	requireResultOK(t, r)
	db := r.Value.(*DB)
	defer db.Close()
	assertResultOK(t, db.Exec("CREATE TABLE ok(x INTEGER)"))
}

func TestDb_OpenDBReadWrite_Bad(t *core.T) {
	r := OpenDBReadWrite(core.JoinPath(t.TempDir(), "missing", "rw.duckdb"))
	assertResultError(t, r)
	core.AssertFalse(t, r.OK)
	core.AssertError(t, r.Value.(error))
}

func TestDb_OpenDBReadWrite_Ugly(t *core.T) {
	r := OpenDBReadWrite("")
	assertResultOK(t, r)
	db := r.Value.(*DB)
	core.AssertNotNil(t, db)
	_ = db.Close()
}

func TestDb_DB_Close_Good(t *core.T) {
	db := newTestDB(t)
	err := db.Close()
	assertResultOK(t, err)
	assertResultError(t, db.Exec("SELECT 1"))
}

func TestDb_DB_Close_Bad(t *core.T) {
	db := newTestDB(t)
	first := db.Close()
	second := db.Close()
	assertResultOK(t, first)
	assertResultOK(t, second)
}

func TestDb_DB_Close_Ugly(t *core.T) {
	db := newTestDB(t)
	assertResultOK(t, db.Exec("CREATE TABLE before_close(x INTEGER)"))
	assertResultOK(t, db.Close())
	assertResultError(t, db.Exec("SELECT * FROM before_close"))
}

func TestDb_DB_Path_Good(t *core.T) {
	db := newTestDB(t)
	got := db.Path()
	core.AssertContains(t, got, "test.duckdb")
}

func TestDb_DB_Path_Bad(t *core.T) {
	db := &DB{path: ""}
	got := db.Path()
	core.AssertEqual(t, "", got)
}

func TestDb_DB_Path_Ugly(t *core.T) {
	db := seedMLDB(t)
	got := db.Path()
	core.AssertTrue(t, len(got) > 0)
}

func TestDb_DB_Exec_Good(t *core.T) {
	db := newTestDB(t)
	err := db.Exec("CREATE TABLE exec_good(x INTEGER)")
	assertResultOK(t, err)
}

func TestDb_DB_Exec_Bad(t *core.T) {
	db := newTestDB(t)
	err := db.Exec("SELECT * FROM missing_table")
	assertResultError(t, err)
}

func TestDb_DB_Exec_Ugly(t *core.T) {
	db := newTestDB(t)
	requireResultOK(t, db.Exec("CREATE TABLE exec_arg(x INTEGER)"))
	err := db.Exec("INSERT INTO exec_arg VALUES (?)", 7)
	assertResultOK(t, err)
}

func TestDb_DB_QueryRowScan_Good(t *core.T) {
	db := newTestDB(t)
	var got int
	err := db.QueryRowScan("SELECT ?", &got, 7)
	requireResultOK(t, err)
	core.AssertEqual(t, 7, got)
}

func TestDb_DB_QueryRowScan_Bad(t *core.T) {
	db := newTestDB(t)
	var got int
	err := db.QueryRowScan("SELECT missing FROM nowhere", &got)
	assertResultError(t, err)
}

func TestDb_DB_QueryRowScan_Ugly(t *core.T) {
	db := newTestDB(t)
	var got string
	err := db.QueryRowScan("SELECT 'value'", &got)
	requireResultOK(t, err)
	core.AssertEqual(t, "value", got)
}

func TestDb_DB_QueryGoldenSet_Good(t *core.T) {
	db := seedMLDB(t)
	r := db.QueryGoldenSet(1)
	requireResultOK(t, r)
	rows := r.Value.([]GoldenSetRow)
	core.AssertLen(t, rows, 1)
}

func TestDb_DB_QueryGoldenSet_Bad(t *core.T) {
	db := newTestDB(t)
	r := db.QueryGoldenSet(1)
	assertResultError(t, r)
}

func TestDb_DB_QueryGoldenSet_Ugly(t *core.T) {
	db := seedMLDB(t)
	r := db.QueryGoldenSet(999)
	requireResultOK(t, r)
	rows := r.Value.([]GoldenSetRow)
	core.AssertEmpty(t, rows)
}

func TestDb_DB_CountGoldenSet_Good(t *core.T) {
	db := seedMLDB(t)
	r := db.CountGoldenSet()
	requireResultOK(t, r)
	count := r.Value.(int)
	core.AssertEqual(t, 1, count)
}

func TestDb_DB_CountGoldenSet_Bad(t *core.T) {
	db := newTestDB(t)
	r := db.CountGoldenSet()
	assertResultError(t, r)
}

func TestDb_DB_CountGoldenSet_Ugly(t *core.T) {
	db := seedMLDB(t)
	requireResultOK(t, db.Exec("DELETE FROM golden_set"))
	r := db.CountGoldenSet()
	requireResultOK(t, r)
	count := r.Value.(int)
	core.AssertEqual(t, 0, count)
}

func TestDb_DB_QueryExpansionPrompts_Good(t *core.T) {
	db := seedMLDB(t)
	r := db.QueryExpansionPrompts("pending", 1)
	requireResultOK(t, r)
	rows := r.Value.([]ExpansionPromptRow)
	core.AssertLen(t, rows, 1)
}

func TestDb_DB_QueryExpansionPrompts_Bad(t *core.T) {
	db := newTestDB(t)
	r := db.QueryExpansionPrompts("pending", 1)
	assertResultError(t, r)
}

func TestDb_DB_QueryExpansionPrompts_Ugly(t *core.T) {
	db := seedMLDB(t)
	r := db.QueryExpansionPrompts("done", 0)
	requireResultOK(t, r)
	rows := r.Value.([]ExpansionPromptRow)
	core.AssertEmpty(t, rows)
}

func TestDb_DB_CountExpansionPrompts_Good(t *core.T) {
	db := seedMLDB(t)
	r := db.CountExpansionPrompts()
	requireResultOK(t, r)
	counts := r.Value.([2]int)
	core.AssertEqual(t, 1, counts[0])
	core.AssertEqual(t, 1, counts[1])
}

func TestDb_DB_CountExpansionPrompts_Bad(t *core.T) {
	db := newTestDB(t)
	r := db.CountExpansionPrompts()
	assertResultError(t, r)
}

func TestDb_DB_CountExpansionPrompts_Ugly(t *core.T) {
	db := seedMLDB(t)
	requireResultOK(t, db.Exec("UPDATE expansion_prompts SET status = 'done'"))
	r := db.CountExpansionPrompts()
	requireResultOK(t, r)
	counts := r.Value.([2]int)
	core.AssertEqual(t, 1, counts[0])
	core.AssertEqual(t, 0, counts[1])
}

func TestDb_DB_UpdateExpansionStatus_Good(t *core.T) {
	db := seedMLDB(t)
	err := db.UpdateExpansionStatus(1, "done")
	requireResultOK(t, err)
	r := db.QueryExpansionPrompts("done", 1)
	requireResultOK(t, r)
	rows := r.Value.([]ExpansionPromptRow)
	core.AssertLen(t, rows, 1)
}

func TestDb_DB_UpdateExpansionStatus_Bad(t *core.T) {
	db := newTestDB(t)
	err := db.UpdateExpansionStatus(1, "done")
	assertResultError(t, err)
}

func TestDb_DB_UpdateExpansionStatus_Ugly(t *core.T) {
	db := seedMLDB(t)
	err := db.UpdateExpansionStatus(99, "done")
	requireResultOK(t, err)
	r := db.QueryExpansionPrompts("pending", 0)
	requireResultOK(t, r)
	rows := r.Value.([]ExpansionPromptRow)
	core.AssertLen(t, rows, 1)
}

func TestDb_DB_QueryRows_Good(t *core.T) {
	db := newTestDB(t)
	r := db.QueryRows("SELECT 7 AS n")
	requireResultOK(t, r)
	rows := r.Value.([]map[string]any)
	core.AssertEqual(t, 7, toInt(rows[0]["n"]))
}

func TestDb_DB_QueryRows_Bad(t *core.T) {
	db := newTestDB(t)
	r := db.QueryRows("SELECT * FROM missing_table")
	assertResultError(t, r)
}

func TestDb_DB_QueryRows_Ugly(t *core.T) {
	db := newTestDB(t)
	r := db.QueryRows("SELECT ? AS value", "x")
	requireResultOK(t, r)
	rows := r.Value.([]map[string]any)
	core.AssertEqual(t, "x", rows[0]["value"])
}

func TestDb_DB_EnsureScoringTables_Good(t *core.T) {
	db := newTestDB(t)
	db.EnsureScoringTables()
	r := db.TableCounts()
	requireResultOK(t, r)
	counts := r.Value.(map[string]int)
	core.AssertContains(t, counts, TableCheckpointScores)
}

func TestDb_DB_EnsureScoringTables_Bad(t *core.T) {
	db := newTestDB(t)
	requireResultOK(t, db.Close())
	core.AssertNotPanics(t, func() { db.EnsureScoringTables() })
}

func TestDb_DB_EnsureScoringTables_Ugly(t *core.T) {
	db := newTestDB(t)
	db.EnsureScoringTables()
	db.EnsureScoringTables()
	assertResultOK(t, db.WriteScoringResult("m", "p", "suite", "dim", 1))
}

func TestDb_DB_WriteScoringResult_Good(t *core.T) {
	db := newTestDB(t)
	db.EnsureScoringTables()
	err := db.WriteScoringResult("m", "p", "suite", "dim", 1.5)
	assertResultOK(t, err)
}

func TestDb_DB_WriteScoringResult_Bad(t *core.T) {
	db := newTestDB(t)
	err := db.WriteScoringResult("m", "p", "suite", "dim", 1.5)
	assertResultError(t, err)
}

func TestDb_DB_WriteScoringResult_Ugly(t *core.T) {
	db := newTestDB(t)
	db.EnsureScoringTables()
	err := db.WriteScoringResult("", "", "", "", 0)
	assertResultOK(t, err)
}

func TestDb_DB_TableCounts_Good(t *core.T) {
	db := seedMLDB(t)
	r := db.TableCounts()
	requireResultOK(t, r)
	counts := r.Value.(map[string]int)
	core.AssertEqual(t, 1, counts["golden_set"])
}

func TestDb_DB_TableCounts_Bad(t *core.T) {
	db := newTestDB(t)
	r := db.TableCounts()
	requireResultOK(t, r)
	counts := r.Value.(map[string]int)
	core.AssertEmpty(t, counts)
}

func TestDb_DB_TableCounts_Ugly(t *core.T) {
	db := newTestDB(t)
	db.EnsureScoringTables()
	r := db.TableCounts()
	requireResultOK(t, r)
	counts := r.Value.(map[string]int)
	core.AssertContains(t, counts, "scoring_results")
}
