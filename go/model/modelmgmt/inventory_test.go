package modelmgmt

import (
	"dappco.re/go"
	"dappco.re/go/inference/eval/datapipe"
)

// seedInventoryDetailsDB creates and populates the tables gatherDetails
// annotates, so the per-table detail queries all have data to summarise.
func seedInventoryDetailsDB(t *core.T) *datapipe.DB {
	t.Helper()
	db := newTestDB(t)
	requireResultOK(t, db.Exec(`CREATE TABLE golden_set (response VARCHAR)`))
	requireResultOK(t, db.Exec(`INSERT INTO golden_set VALUES ('r')`))
	requireResultOK(t, db.Exec(`CREATE TABLE training_examples (source VARCHAR)`))
	requireResultOK(t, db.Exec(`INSERT INTO training_examples VALUES ('a'),('b')`))
	requireResultOK(t, db.Exec(`CREATE TABLE prompts (domain VARCHAR, voice VARCHAR)`))
	requireResultOK(t, db.Exec(`INSERT INTO prompts VALUES ('ethics','calm'),('law','stern')`))
	requireResultOK(t, db.Exec(`CREATE TABLE gemini_responses (source_model VARCHAR)`))
	requireResultOK(t, db.Exec(`INSERT INTO gemini_responses VALUES ('gemini-1.5'),('gemini-1.5'),('gemini-2')`))
	requireResultOK(t, db.Exec(`CREATE TABLE benchmark_results (source VARCHAR)`))
	requireResultOK(t, db.Exec(`INSERT INTO benchmark_results VALUES ('results'),('results'),('scale')`))
	return db
}

func detailNotes(t *core.T, details map[string]*tableDetail, table string) string {
	t.Helper()
	d, ok := details[table]
	core.AssertTrue(t, ok)
	return core.Join(", ", d.notes...)
}

func TestInventory_toInt_Good(t *core.T) {
	core.AssertEqual(t, 42, toInt(int64(42)))
}

func TestInventory_toInt_Bad(t *core.T) {
	core.AssertEqual(t, 0, toInt("not a number"))
	core.AssertEqual(t, 0, toInt(nil))
}

func TestInventory_toInt_Ugly(t *core.T) {
	// DuckDB and InfluxDB hand back different numeric widths.
	core.AssertEqual(t, 7, toInt(int32(7)))
	core.AssertEqual(t, 3, toInt(float64(3.9)))
}

func TestInventory_gatherDetails_Good(t *core.T) {
	db := seedInventoryDetailsDB(t)
	rCounts := db.TableCounts()
	requireResultOK(t, rCounts)
	details := gatherDetails(db, rCounts.Value.(map[string]int))
	core.AssertContains(t, detailNotes(t, details, "golden_set"), "target")
	core.AssertContains(t, detailNotes(t, details, "training_examples"), "2 sources")
}

func TestInventory_gatherDetails_Bad(t *core.T) {
	// No known tables present -> no annotations.
	details := gatherDetails(newTestDB(t), map[string]int{})
	core.AssertEmpty(t, details)
}

func TestInventory_gatherDetails_Ugly(t *core.T) {
	db := seedInventoryDetailsDB(t)
	rCounts := db.TableCounts()
	requireResultOK(t, rCounts)
	details := gatherDetails(db, rCounts.Value.(map[string]int))
	prompts := detailNotes(t, details, "prompts")
	core.AssertContains(t, prompts, "2 domains")
	core.AssertContains(t, prompts, "2 voices")
	core.AssertContains(t, detailNotes(t, details, "gemini_responses"), "gemini-1.5:2")
	core.AssertContains(t, detailNotes(t, details, "benchmark_results"), "2 categories")
}

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
