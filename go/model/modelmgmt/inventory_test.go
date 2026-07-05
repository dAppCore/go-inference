package modelmgmt

import "dappco.re/go"

func TestInventory_PrintInventory_Good(t *core.T) {
	db := seedMLDB(t)
	buf := core.NewBuffer(nil)
	requireResultOK(t, PrintInventory(db, buf))
	core.AssertContains(t, buf.String(), "DuckDB Inventory")
}

func TestInventory_PrintInventory_Bad(t *core.T) {
	db := newTestDB(t)
	buf := core.NewBuffer(nil)
	requireResultOK(t, PrintInventory(db, buf))
	core.AssertContains(t, buf.String(), "TOTAL")
}

func TestInventory_PrintInventory_Ugly(t *core.T) {
	db := newTestDB(t)
	db.EnsureScoringTables()
	buf := core.NewBuffer(nil)
	requireResultOK(t, PrintInventory(db, buf))
	core.AssertContains(t, buf.String(), "scoring_results")
}
