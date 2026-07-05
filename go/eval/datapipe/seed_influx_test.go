package datapipe

import "dappco.re/go"

func TestSeedInflux_SeedInflux_Good(t *core.T) {
	db := seedGoldenStoreDB(t)
	requireResultOK(t, db.Exec("INSERT INTO golden_set VALUES (1,'s1','ethics','calm',1.0,80)"))
	influx, rec := newFakeInflux(t, map[string][]map[string]any{"gold_gen": {{"n": float64(0)}}}, 0)
	err := SeedInflux(db, influx, SeedInfluxConfig{BatchSize: 1}, core.NewBuffer(nil))
	requireResultOK(t, err)
	core.AssertEqual(t, 1, rec.writeCount())
}

func TestSeedInflux_SeedInflux_Bad(t *core.T) {
	db := newStoreDuckDB(t)
	influx, _ := newFakeInflux(t, nil, 0)
	err := SeedInflux(db, influx, SeedInfluxConfig{}, core.NewBuffer(nil))
	assertResultError(t, err)
}

func TestSeedInflux_SeedInflux_Ugly(t *core.T) {
	db := seedGoldenStoreDB(t)
	requireResultOK(t, db.Exec("INSERT INTO golden_set VALUES (1,'s1','ethics','calm',1.0,80)"))
	influx, rec := newFakeInflux(t, map[string][]map[string]any{"gold_gen": {{"n": float64(1)}}}, 0)
	err := SeedInflux(db, influx, SeedInfluxConfig{}, core.NewBuffer(nil))
	requireResultOK(t, err)
	core.AssertEqual(t, 0, rec.writeCount())
}
