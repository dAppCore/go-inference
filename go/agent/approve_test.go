package agent

import (
	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/store"
)

func seedApproveDB(t *core.T) *store.DuckDB {
	t.Helper()
	db := newStoreDuckDB(t)
	requireResultOK(t, db.Exec(`CREATE TABLE expansion_raw (
		idx INTEGER, seed_id VARCHAR, region VARCHAR, domain VARCHAR,
		prompt VARCHAR, response VARCHAR, gen_time DOUBLE, model VARCHAR
	)`))
	requireResultOK(t, db.Exec(`CREATE TABLE expansion_scores (
		idx INTEGER, heuristic_score DOUBLE, heuristic_pass BOOLEAN, judge_pass BOOLEAN
	)`))
	requireResultOK(t, db.Exec("INSERT INTO expansion_raw VALUES (1,'s1','en','ethics','prompt','response',1.0,'m')"))
	requireResultOK(t, db.Exec("INSERT INTO expansion_scores VALUES (1,0.9,true,true)"))
	return db
}

func TestApprove_ApproveExpansions_Good(t *core.T) {
	db := seedApproveDB(t)
	out := core.JoinPath(t.TempDir(), "approved.jsonl")
	err := ApproveExpansions(db, ApproveConfig{Output: out}, core.NewBuffer(nil))
	requireResultOK(t, err)
	data, readErr := coreio.Local.Read(out)
	core.RequireNoError(t, readErr)
	core.AssertContains(t, data, "response")
}

func TestApprove_ApproveExpansions_Bad(t *core.T) {
	db := newStoreDuckDB(t)
	err := ApproveExpansions(db, ApproveConfig{Output: core.JoinPath(t.TempDir(), "out.jsonl")}, core.NewBuffer(nil))
	assertResultError(t, err)

	// The query succeeds, but the output path is itself an existing
	// directory — a distinct failure point (create output file) from the
	// query failure above. coreio.Local.Create auto-creates any missing
	// parent directories, so a merely-absent path would not fail here.
	seeded := seedApproveDB(t)
	badOut := core.JoinPath(t.TempDir(), "already-a-dir")
	core.RequireNoError(t, coreio.Local.EnsureDir(badOut))
	err2 := ApproveExpansions(seeded, ApproveConfig{Output: badOut}, core.NewBuffer(nil))
	assertResultError(t, err2, "create output")
}

func TestApprove_ApproveExpansions_Ugly(t *core.T) {
	db := seedApproveDB(t)
	requireResultOK(t, db.Exec("UPDATE expansion_scores SET heuristic_pass = false"))
	out := core.JoinPath(t.TempDir(), "empty.jsonl")
	err := ApproveExpansions(db, ApproveConfig{Output: out}, core.NewBuffer(nil))
	requireResultOK(t, err)
	data, readErr := coreio.Local.Read(out)
	core.RequireNoError(t, readErr)
	core.AssertEqual(t, "", data)

	// A NULL in a non-nullable scanned column (region) fails the row scan
	// itself rather than the query or the file write.
	nullDB := newStoreDuckDB(t)
	requireResultOK(t, nullDB.Exec(`CREATE TABLE expansion_raw (
		idx INTEGER, seed_id VARCHAR, region VARCHAR, domain VARCHAR,
		prompt VARCHAR, response VARCHAR, gen_time DOUBLE, model VARCHAR
	)`))
	requireResultOK(t, nullDB.Exec(`CREATE TABLE expansion_scores (
		idx INTEGER, heuristic_score DOUBLE, heuristic_pass BOOLEAN, judge_pass BOOLEAN
	)`))
	requireResultOK(t, nullDB.Exec("INSERT INTO expansion_raw VALUES (1,'s1',NULL,'ethics','prompt','response',1.0,'m')"))
	requireResultOK(t, nullDB.Exec("INSERT INTO expansion_scores VALUES (1,0.9,true,true)"))
	scanOut := core.JoinPath(t.TempDir(), "scan-fail.jsonl")
	scanErr := ApproveExpansions(nullDB, ApproveConfig{Output: scanOut}, core.NewBuffer(nil))
	assertResultError(t, scanErr, "scan approved row")
}
