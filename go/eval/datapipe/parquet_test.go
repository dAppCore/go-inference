package datapipe

import (
	"dappco.re/go"
	coreio "dappco.re/go/io"
)

func writeParquetJSONL(t *core.T, file string) {
	t.Helper()
	line := `{"messages":[{"role":"system","content":"sys"},{"role":"user","content":"prompt"},{"role":"assistant","content":"response"}]}` + "\n"
	core.RequireNoError(t, coreio.Local.Write(file, line))
}

func TestParquet_ExportParquet_Good(t *core.T) {
	dir := t.TempDir()
	writeParquetJSONL(t, core.JoinPath(dir, "train.jsonl"))
	r := ExportParquet(dir, "")
	requireResultOK(t, r)
	core.AssertEqual(t, 1, r.Value.(int))
	core.AssertTrue(t, coreio.Local.IsFile(core.JoinPath(dir, "parquet", "train.parquet")))
}

func TestParquet_ExportParquet_Bad(t *core.T) {
	r := ExportParquet(core.JoinPath(t.TempDir(), "missing"), "")
	assertResultOK(t, r)
	core.AssertEqual(t, 0, r.Value.(int))
}

func TestParquet_ExportParquet_Ugly(t *core.T) {
	dir := t.TempDir()
	core.RequireNoError(t, coreio.Local.Write(core.JoinPath(dir, "train.jsonl"), "not structured\n"))
	r := ExportParquet(dir, core.JoinPath(dir, "out"))
	requireResultOK(t, r)
	core.AssertEqual(t, 0, r.Value.(int))
}

func TestParquet_ExportSplitParquet_Good(t *core.T) {
	dir := t.TempDir()
	jsonl := core.JoinPath(dir, "split.jsonl")
	writeParquetJSONL(t, jsonl)
	r := ExportSplitParquet(jsonl, dir, "train")
	requireResultOK(t, r)
	core.AssertEqual(t, 1, r.Value.(int))
	core.AssertTrue(t, coreio.Local.IsFile(core.JoinPath(dir, "train.parquet")))
}

func TestParquet_ExportSplitParquet_Bad(t *core.T) {
	assertResultError(t, ExportSplitParquet(core.JoinPath(t.TempDir(), "missing.jsonl"), t.TempDir(), "bad"))
}

func TestParquet_ExportSplitParquet_Ugly(t *core.T) {
	file := core.JoinPath(t.TempDir(), "empty.jsonl")
	core.RequireNoError(t, coreio.Local.Write(file, "\n"))
	r := ExportSplitParquet(file, t.TempDir(), "empty")
	requireResultOK(t, r)
	core.AssertEqual(t, 0, r.Value.(int))
}
