package score

import (
	"dappco.re/go"
	"dappco.re/go/store"
)

func seedNormalizeDB(t *core.T) *store.DuckDB {
	t.Helper()
	db := seedCoverageDB(t)
	requireResultOK(t, db.Exec("CREATE TABLE prompts(prompt VARCHAR)"))
	requireResultOK(t, db.Exec("CREATE TABLE golden_set(prompt VARCHAR)"))
	return db
}

func TestNormalize_NormalizeSeeds_Good(t *core.T) {
	db := seedNormalizeDB(t)
	requireResultOK(t, db.Exec("INSERT INTO seeds VALUES ('f','en-us','s1','ethics','a long enough prompt')"))
	buf := core.NewBuffer(nil)
	err := NormalizeSeeds(db, NormalizeConfig{MinLength: 3}, buf)
	requireResultOK(t, err)
	core.AssertContains(t, buf.String(), "Expansion prompts created: 1")
}

func TestNormalize_NormalizeSeeds_Bad(t *core.T) {
	db := newStoreDuckDB(t)
	err := NormalizeSeeds(db, NormalizeConfig{MinLength: 3}, core.NewBuffer(nil))
	assertResultError(t, err)
}

func TestNormalize_NormalizeSeeds_Ugly(t *core.T) {
	db := seedNormalizeDB(t)
	buf := core.NewBuffer(nil)
	err := NormalizeSeeds(db, NormalizeConfig{MinLength: 3}, buf)
	assertResultError(t, err)
	core.AssertContains(t, err.Error(), "empty")
}
