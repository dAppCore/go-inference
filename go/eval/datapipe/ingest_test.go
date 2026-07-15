package datapipe

import (
	"dappco.re/go"
	coreio "dappco.re/go/io"
)

func TestIngest_Ingest_Good(t *core.T) {
	dir := t.TempDir()
	contentFile := core.JoinPath(dir, "content.jsonl")
	core.RequireNoError(t, coreio.Local.Write(contentFile, `{"label":"m@10","aggregates":{"clarity":8.0}}`+"\n"))
	capFile := core.JoinPath(dir, "capability.jsonl")
	core.RequireNoError(t, coreio.Local.Write(capFile, `{"label":"m@10","accuracy":0.9,"correct":9,"total":10}`+"\n"))
	logFile := core.JoinPath(dir, "train.log")
	core.RequireNoError(t, coreio.Local.Write(logFile, "Iter 1: Train loss 0.5, Learning Rate 1e-5, It/sec 2.0, Tokens/sec 30.0\n"))

	influx, rec := newFakeInflux(t, nil, 0)
	// All three sources, one final flush each => three write batches.
	requireResultOK(t, Ingest(influx, IngestConfig{
		ContentFile: contentFile, CapabilityFile: capFile, TrainingLog: logFile,
		Model: "m", BatchSize: 100,
	}, core.NewBuffer(nil)))
	core.AssertEqual(t, 3, rec.writeCount())
}

func TestIngest_Ingest_Bad(t *core.T) {
	// No source file configured.
	influx, _ := newFakeInflux(t, nil, 0)
	assertResultError(t, Ingest(influx, IngestConfig{}, core.NewBuffer(nil)))

	// Source set but model missing.
	logFile := core.JoinPath(t.TempDir(), "train.log")
	core.RequireNoError(t, coreio.Local.Write(logFile, "Iter 1: Train loss 0.5, Learning Rate 1e-5, It/sec 2.0, Tokens/sec 30.0\n"))
	assertResultError(t, Ingest(influx, IngestConfig{TrainingLog: logFile}, core.NewBuffer(nil)), "--model is required")

	// Capability sub-ingest failure is wrapped by Ingest.
	capFile := core.JoinPath(t.TempDir(), "capability.jsonl")
	core.RequireNoError(t, coreio.Local.Write(capFile, `{"label":"m","accuracy":0.9,"correct":9,"total":10}`+"\n"))
	failInflux, _ := newFakeInflux(t, nil, 500)
	assertResultError(t, Ingest(failInflux, IngestConfig{CapabilityFile: capFile, Model: "m", BatchSize: 1}, core.NewBuffer(nil)), "ingest capability scores")

	// Training-log sub-ingest failure is wrapped by Ingest (RunID defaults to Model).
	assertResultError(t, Ingest(failInflux, IngestConfig{TrainingLog: logFile, Model: "m", BatchSize: 1}, core.NewBuffer(nil)), "ingest training log")
}

func TestIngest_Ingest_Ugly(t *core.T) {
	contentFile := core.JoinPath(t.TempDir(), "content.out")
	core.RequireNoError(t, coreio.Local.Write(contentFile, "not object\n"))
	influx, _ := newFakeInflux(t, nil, 0)
	assertResultError(t, Ingest(influx, IngestConfig{ContentFile: contentFile, Model: "m"}, core.NewBuffer(nil)))
}

func TestIngest_ingestContentScores_Good(t *core.T) {
	path := core.JoinPath(t.TempDir(), "content.jsonl")
	core.RequireNoError(t, coreio.Local.Write(path,
		`{"label":"gemma@200","aggregates":{"clarity":8.5,"depth":7.25},"probes":{"p001":{"scores":{"clarity":8.0,"depth":7.5}}}}`+"\n"))
	influx, rec := newFakeInflux(t, nil, 0)
	// BatchSize 1 forces the in-loop flush + slice reset after a successful write.
	r := ingestContentScores(influx, IngestConfig{ContentFile: path, Model: "gemma", RunID: "run1", BatchSize: 1}, core.NewBuffer(nil))
	requireResultOK(t, r)
	// 2 aggregate dimensions + 2 probe dimensions = 4 points, flushed as one batch.
	core.AssertEqual(t, 4, r.Value.(int))
	core.AssertEqual(t, 1, rec.writeCount())
}

func TestIngest_ingestContentScores_Bad(t *core.T) {
	influx, _ := newFakeInflux(t, nil, 0)
	// Missing file.
	assertResultError(t, ingestContentScores(influx, IngestConfig{ContentFile: core.JoinPath(t.TempDir(), "nope.jsonl"), Model: "m", BatchSize: 100}, core.NewBuffer(nil)), "open")

	// Malformed JSON line.
	badPath := core.JoinPath(t.TempDir(), "bad.jsonl")
	core.RequireNoError(t, coreio.Local.Write(badPath, "not-json\n"))
	assertResultError(t, ingestContentScores(influx, IngestConfig{ContentFile: badPath, Model: "m", BatchSize: 100}, core.NewBuffer(nil)), "parse json")

	// Write failures: mid-batch (small batch flushes in-loop) and final flush.
	path := core.JoinPath(t.TempDir(), "content.jsonl")
	core.RequireNoError(t, coreio.Local.Write(path, `{"label":"m","aggregates":{"clarity":8.5}}`+"\n"))
	failInflux, _ := newFakeInflux(t, nil, 500)
	assertResultError(t, ingestContentScores(failInflux, IngestConfig{ContentFile: path, Model: "m", BatchSize: 1}, core.NewBuffer(nil)), "write batch")
	assertResultError(t, ingestContentScores(failInflux, IngestConfig{ContentFile: path, Model: "m", BatchSize: 1000}, core.NewBuffer(nil)), "write final batch")
}

func TestIngest_ingestContentScores_Ugly(t *core.T) {
	// Blank lines skipped; non-numeric aggregate/probe values dropped; the "LEK"
	// label sets has_kernel=true.
	path := core.JoinPath(t.TempDir(), "content.jsonl")
	core.RequireNoError(t, coreio.Local.Write(path,
		"\n"+`{"label":"LEK-kernel@50","aggregates":{"clarity":9.0,"skip":null},"probes":{"p1":{"scores":{"depth":6.0,"skip":true}}}}`+"\n"))
	influx, rec := newFakeInflux(t, nil, 0)
	r := ingestContentScores(influx, IngestConfig{ContentFile: path, Model: "m", BatchSize: 100}, core.NewBuffer(nil))
	requireResultOK(t, r)
	core.AssertEqual(t, 2, r.Value.(int)) // clarity + depth; null/bool dropped
	core.AssertEqual(t, 1, rec.writeCount())

	// A single line larger than the scanner's 1 MiB cap trips scanner.Err().
	bigPath := core.JoinPath(t.TempDir(), "big.jsonl")
	core.RequireNoError(t, coreio.Local.Write(bigPath, `{"label":"`+core.Repeat("x", 1024*1024+64)+`"}`))
	assertResultError(t, ingestContentScores(influx, IngestConfig{ContentFile: bigPath, Model: "m", BatchSize: 100}, core.NewBuffer(nil)), "scan")
}

func TestIngest_ingestCapabilityScores_Good(t *core.T) {
	path := core.JoinPath(t.TempDir(), "capability.jsonl")
	core.RequireNoError(t, coreio.Local.Write(path,
		`{"label":"gemma@200","accuracy":0.85,"correct":85,"total":100,"by_category":{"math":{"correct":40,"total":50}}}`+"\n"))
	influx, rec := newFakeInflux(t, nil, 0)
	// BatchSize 1 forces the in-loop flush + slice reset after a successful write.
	r := ingestCapabilityScores(influx, IngestConfig{CapabilityFile: path, Model: "gemma", RunID: "run1", BatchSize: 1}, core.NewBuffer(nil))
	requireResultOK(t, r)
	// 1 overall + 1 category = 2 points, flushed as one batch.
	core.AssertEqual(t, 2, r.Value.(int))
	core.AssertEqual(t, 1, rec.writeCount())
}

func TestIngest_ingestCapabilityScores_Bad(t *core.T) {
	influx, _ := newFakeInflux(t, nil, 0)
	// Missing file.
	assertResultError(t, ingestCapabilityScores(influx, IngestConfig{CapabilityFile: core.JoinPath(t.TempDir(), "nope.jsonl"), Model: "m", BatchSize: 100}, core.NewBuffer(nil)), "open")

	// Malformed JSON line.
	badPath := core.JoinPath(t.TempDir(), "bad.jsonl")
	core.RequireNoError(t, coreio.Local.Write(badPath, "not-json\n"))
	assertResultError(t, ingestCapabilityScores(influx, IngestConfig{CapabilityFile: badPath, Model: "m", BatchSize: 100}, core.NewBuffer(nil)), "parse json")

	// Write failures: mid-batch and final flush.
	path := core.JoinPath(t.TempDir(), "capability.jsonl")
	core.RequireNoError(t, coreio.Local.Write(path, `{"label":"m","accuracy":0.9,"correct":9,"total":10}`+"\n"))
	failInflux, _ := newFakeInflux(t, nil, 500)
	assertResultError(t, ingestCapabilityScores(failInflux, IngestConfig{CapabilityFile: path, Model: "m", BatchSize: 1}, core.NewBuffer(nil)), "write batch")
	assertResultError(t, ingestCapabilityScores(failInflux, IngestConfig{CapabilityFile: path, Model: "m", BatchSize: 1000}, core.NewBuffer(nil)), "write final batch")
}

func TestIngest_ingestCapabilityScores_Ugly(t *core.T) {
	// A zero-total category leaves accuracy at 0 (division guarded); blanks skipped.
	path := core.JoinPath(t.TempDir(), "capability.jsonl")
	core.RequireNoError(t, coreio.Local.Write(path,
		"\n"+`{"label":"m@5","accuracy":0.5,"correct":5,"total":10,"by_category":{"empty":{"correct":0,"total":0},"logic":{"correct":3,"total":4}}}`+"\n"))
	influx, rec := newFakeInflux(t, nil, 0)
	r := ingestCapabilityScores(influx, IngestConfig{CapabilityFile: path, Model: "m", BatchSize: 100}, core.NewBuffer(nil))
	requireResultOK(t, r)
	core.AssertEqual(t, 3, r.Value.(int)) // overall + empty + logic
	core.AssertEqual(t, 1, rec.writeCount())

	// Oversized line trips scanner.Err().
	bigPath := core.JoinPath(t.TempDir(), "big.jsonl")
	core.RequireNoError(t, coreio.Local.Write(bigPath, `{"label":"`+core.Repeat("x", 1024*1024+64)+`"}`))
	assertResultError(t, ingestCapabilityScores(influx, IngestConfig{CapabilityFile: bigPath, Model: "m", BatchSize: 100}, core.NewBuffer(nil)), "scan")
}

func TestIngest_ingestTrainingLog_Good(t *core.T) {
	// Both a validation-loss and a training-loss line.
	path := core.JoinPath(t.TempDir(), "train.log")
	core.RequireNoError(t, coreio.Local.Write(path,
		"Iter 10: Val loss 0.601\nIter 10: Train loss 0.523, Learning Rate 1.0e-05, It/sec 2.15, Tokens/sec 30.42\n"))
	influx, rec := newFakeInflux(t, nil, 0)
	// BatchSize 1 forces the in-loop flush + slice reset after each successful write.
	r := ingestTrainingLog(influx, IngestConfig{TrainingLog: path, Model: "gemma", RunID: "run1", BatchSize: 1}, core.NewBuffer(nil))
	requireResultOK(t, r)
	core.AssertEqual(t, 2, r.Value.(int)) // val + train, one point per line
	core.AssertEqual(t, 2, rec.writeCount())
}

func TestIngest_ingestTrainingLog_Bad(t *core.T) {
	influx, _ := newFakeInflux(t, nil, 0)
	// Missing file.
	assertResultError(t, ingestTrainingLog(influx, IngestConfig{TrainingLog: core.JoinPath(t.TempDir(), "nope.log"), Model: "m", BatchSize: 100}, core.NewBuffer(nil)), "open")

	// Write failures: mid-batch and final flush.
	path := core.JoinPath(t.TempDir(), "train.log")
	core.RequireNoError(t, coreio.Local.Write(path, "Iter 1: Train loss 0.5, Learning Rate 1e-5, It/sec 2.0, Tokens/sec 30.0\n"))
	failInflux, _ := newFakeInflux(t, nil, 500)
	assertResultError(t, ingestTrainingLog(failInflux, IngestConfig{TrainingLog: path, Model: "m", BatchSize: 1}, core.NewBuffer(nil)), "write batch")
	assertResultError(t, ingestTrainingLog(failInflux, IngestConfig{TrainingLog: path, Model: "m", BatchSize: 1000}, core.NewBuffer(nil)), "write final batch")
}

func TestIngest_ingestTrainingLog_Ugly(t *core.T) {
	// Non-matching lines produce no points; result is OK with zero writes.
	path := core.JoinPath(t.TempDir(), "train.log")
	core.RequireNoError(t, coreio.Local.Write(path, "starting run\nsome unrelated log line\n"))
	influx, rec := newFakeInflux(t, nil, 0)
	r := ingestTrainingLog(influx, IngestConfig{TrainingLog: path, Model: "m", BatchSize: 100}, core.NewBuffer(nil))
	requireResultOK(t, r)
	core.AssertEqual(t, 0, r.Value.(int))
	core.AssertEqual(t, 0, rec.writeCount())

	// Oversized line trips scanner.Err().
	bigPath := core.JoinPath(t.TempDir(), "big.log")
	core.RequireNoError(t, coreio.Local.Write(bigPath, core.Repeat("x", 1024*1024+64)))
	assertResultError(t, ingestTrainingLog(influx, IngestConfig{TrainingLog: bigPath, Model: "m", BatchSize: 100}, core.NewBuffer(nil)), "scan")
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
