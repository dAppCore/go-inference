package datapipe

import (
	"dappco.re/go"
	coreio "dappco.re/go/io"
)

func TestIngest_Ingest_Good(t *core.T) {
	logFile := core.JoinPath(t.TempDir(), "train.log")
	core.RequireNoError(t, coreio.Local.Write(logFile, "Iter 1: Train loss 0.5, Learning Rate 1e-5, It/sec 2.0, Tokens/sec 30.0\n"))
	influx, rec := newFakeInflux(t, nil, 0)
	requireResultOK(t, Ingest(influx, IngestConfig{TrainingLog: logFile, Model: "m", BatchSize: 1}, core.NewBuffer(nil)))
	core.AssertEqual(t, 1, rec.writeCount())
}

func TestIngest_Ingest_Bad(t *core.T) {
	influx, _ := newFakeInflux(t, nil, 0)
	assertResultError(t, Ingest(influx, IngestConfig{}, core.NewBuffer(nil)))
}

func TestIngest_Ingest_Ugly(t *core.T) {
	contentFile := core.JoinPath(t.TempDir(), "content.out")
	core.RequireNoError(t, coreio.Local.Write(contentFile, "not object\n"))
	influx, _ := newFakeInflux(t, nil, 0)
	assertResultError(t, Ingest(influx, IngestConfig{ContentFile: contentFile, Model: "m"}, core.NewBuffer(nil)))
}

func TestIngest_extractIteration_Good(t *core.T) {
	got := extractIteration("model@200")
	core.AssertEqual(t, 200, got)
}

func TestIngest_extractIteration_Bad(t *core.T) {
	got := extractIteration("model")
	core.AssertEqual(t, 0, got)
	got = extractIteration("model@")
	core.AssertEqual(t, 0, got)
	got = extractIteration("model@notanumber")
	core.AssertEqual(t, 0, got)
}

func TestIngest_extractIteration_Ugly(t *core.T) {
	got := extractIteration("")
	core.AssertEqual(t, 0, got)
	got = extractIteration("@100")
	core.AssertEqual(t, 100, got)
}

func TestIngest_toFloat64_Good(t *core.T) {
	v, ok := toFloat64(3.14)
	core.AssertTrue(t, ok)
	core.AssertEqual(t, 3.14, v)
}

func TestIngest_toFloat64_Bad(t *core.T) {
	v, ok := toFloat64(nil)
	core.AssertFalse(t, ok)
	core.AssertEqual(t, 0.0, v)
}

func TestIngest_toFloat64_Ugly(t *core.T) {
	v, ok := toFloat64(int64(42))
	core.AssertTrue(t, ok)
	core.AssertEqual(t, 42.0, v)
	v, ok = toFloat64(42)
	core.AssertTrue(t, ok)
	core.AssertEqual(t, 42.0, v)
	v, ok = toFloat64("3.14")
	core.AssertTrue(t, ok)
	core.AssertEqual(t, 3.14, v)
	v, ok = toFloat64("not-a-number")
	core.AssertFalse(t, ok)
}
