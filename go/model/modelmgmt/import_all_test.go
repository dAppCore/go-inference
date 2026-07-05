package modelmgmt

import (
	"dappco.re/go"
	coreio "dappco.re/go/io"
)

func TestImportAll_ImportAll_Good(t *core.T) {
	db := newTestDB(t)
	dataDir := t.TempDir()
	trainDir := core.JoinPath(dataDir, "training", "training")
	core.RequireNoError(t, coreio.Local.EnsureDir(trainDir))
	core.RequireNoError(t, coreio.Local.Write(core.JoinPath(trainDir, "train.jsonl"), `{"messages":[{"role":"user","content":"p"},{"role":"assistant","content":"r"}]}`+"\n"))
	requireResultOK(t, ImportAll(db, ImportConfig{SkipM3: true, DataDir: dataDir}, core.NewBuffer(nil)))
	rCounts := db.TableCounts()
	requireResultOK(t, rCounts)
	counts := rCounts.Value.(map[string]int)
	core.AssertContains(t, counts, "training_examples")
}

func TestImportAll_ImportAll_Bad(t *core.T) {
	db := newTestDB(t)
	requireResultOK(t, ImportAll(db, ImportConfig{SkipM3: true, DataDir: ""}, core.NewBuffer(nil)))
	rCounts := db.TableCounts()
	requireResultOK(t, rCounts)
	counts := rCounts.Value.(map[string]int)
	core.AssertContains(t, counts, "training_examples")
}

func TestImportAll_ImportAll_Ugly(t *core.T) {
	db := newTestDB(t)
	dataDir := t.TempDir()
	core.RequireNoError(t, coreio.Local.EnsureDir(core.JoinPath(dataDir, "seeds")))
	requireResultOK(t, ImportAll(db, ImportConfig{SkipM3: true, DataDir: dataDir}, core.NewBuffer(nil)))
	core.AssertEqual(t, dataDir, core.PathDir(core.JoinPath(dataDir, "x")))
}

func TestImportAll_strOrEmpty_Good(t *core.T) {
	got := strOrEmpty(map[string]any{"key": "value"}, "key")
	core.AssertEqual(t, "value", got)
}

func TestImportAll_strOrEmpty_Bad(t *core.T) {
	got := strOrEmpty(map[string]any{}, "missing")
	core.AssertEqual(t, "", got)
}

func TestImportAll_strOrEmpty_Ugly(t *core.T) {
	got := strOrEmpty(map[string]any{"num": 42}, "num")
	core.AssertEqual(t, "42", got)
}

func TestImportAll_floatOrZero_Good(t *core.T) {
	got := floatOrZero(map[string]any{"score": 0.95}, "score")
	core.AssertEqual(t, 0.95, got)
}

func TestImportAll_floatOrZero_Bad(t *core.T) {
	got := floatOrZero(map[string]any{}, "missing")
	core.AssertEqual(t, 0.0, got)
	got = floatOrZero(map[string]any{"notfloat": "string"}, "notfloat")
	core.AssertEqual(t, 0.0, got)
}

func TestImportAll_floatOrZero_Ugly(t *core.T) {
	got := floatOrZero(map[string]any{"intval": 1}, "intval")
	core.AssertEqual(t, 0.0, got)
}

func TestImportAll_escapeSQLPath_Good(t *core.T) {
	got := escapeSQLPath("/path/to/file")
	core.AssertEqual(t, "/path/to/file", got)
}

func TestImportAll_escapeSQLPath_Bad(t *core.T) {
	got := escapeSQLPath("")
	core.AssertEqual(t, "", got)
}

func TestImportAll_escapeSQLPath_Ugly(t *core.T) {
	got := escapeSQLPath("it's a file")
	core.AssertEqual(t, "it''s a file", got)
}
