package datapipe

import (
	"dappco.re/go"
	"dappco.re/go/store"
)

func seedGoldenStoreDB(t *core.T) *store.DuckDB {
	t.Helper()
	db := newStoreDuckDB(t)
	requireResultOK(t, db.Exec(`CREATE TABLE golden_set (
		idx INTEGER, seed_id VARCHAR, domain VARCHAR, voice VARCHAR,
		gen_time DOUBLE, char_count INTEGER
	)`))
	return db
}

func TestMetrics_PushMetrics_Good(t *core.T) {
	db := seedGoldenStoreDB(t)
	requireResultOK(t, db.Exec("INSERT INTO golden_set VALUES (1,'s1','ethics','calm',1.0,80)"))
	influx, rec := newFakeInflux(t, nil, 0)
	err := PushMetrics(db, influx, core.NewBuffer(nil))
	requireResultOK(t, err)
	core.AssertEqual(t, 1, rec.writeCount())
}

func TestMetrics_PushMetrics_Bad(t *core.T) {
	db := newStoreDuckDB(t)
	influx, _ := newFakeInflux(t, nil, 0)
	err := PushMetrics(db, influx, core.NewBuffer(nil))
	assertResultError(t, err)
}

func TestMetrics_PushMetrics_Ugly(t *core.T) {
	db := seedGoldenStoreDB(t)
	influx, rec := newFakeInflux(t, nil, 0)
	err := PushMetrics(db, influx, core.NewBuffer(nil))
	requireResultOK(t, err)
	core.AssertEqual(t, 0, rec.writeCount())
}
