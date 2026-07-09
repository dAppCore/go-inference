package score

import (
	"dappco.re/go"
	"dappco.re/go/store"
)

func seedCoverageDB(t *core.T) *store.DuckDB {
	t.Helper()
	db := newStoreDuckDB(t)
	requireResultOK(t, db.Exec(`CREATE TABLE seeds (
		source_file VARCHAR, region VARCHAR, seed_id VARCHAR, domain VARCHAR, prompt VARCHAR
	)`))
	return db
}

func TestCoverage_PrintCoverage_Good(t *core.T) {
	db := seedCoverageDB(t)
	requireResultOK(t, db.Exec("INSERT INTO seeds VALUES ('f','en-us','s1','ethics','prompt')"))
	buf := core.NewBuffer(nil)
	requireResultOK(t, PrintCoverage(db, buf))
	core.AssertContains(t, buf.String(), "Total seeds: 1")
}

func TestCoverage_PrintCoverage_Bad(t *core.T) {
	db := newStoreDuckDB(t)
	assertResultError(t, PrintCoverage(db, core.NewBuffer(nil)))
}

func TestCoverage_PrintCoverage_Ugly(t *core.T) {
	db := seedCoverageDB(t)
	buf := core.NewBuffer(nil)
	requireResultOK(t, PrintCoverage(db, buf))
	core.AssertContains(t, buf.String(), "Total seeds: 0")
}
