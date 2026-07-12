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

func TestImportAll_importBenchmarkFile_Good(t *core.T) {
	db := newTestDB(t)
	requireResultOK(t, db.Exec(`CREATE TABLE benchmark_results (
		source VARCHAR, id VARCHAR, benchmark VARCHAR, model VARCHAR,
		prompt TEXT, response TEXT, elapsed_seconds DOUBLE, domain VARCHAR)`))
	path := core.JoinPath(t.TempDir(), "results.jsonl")
	core.RequireNoError(t, coreio.Local.Write(path,
		`{"id":"a","benchmark":"gsm8k","model":"m","prompt":"p","response":"r","elapsed_seconds":1.5,"domain":"math"}`+"\n"+
			`{"id":"b","benchmark":"gsm8k","model":"m","prompt":"p2","response":"r2","elapsed_seconds":2.0,"domain":"math"}`+"\n"))
	core.AssertEqual(t, 2, importBenchmarkFile(db, path, "results"))
	rCounts := db.TableCounts()
	requireResultOK(t, rCounts)
	core.AssertEqual(t, 2, rCounts.Value.(map[string]int)["benchmark_results"])
}

func TestImportAll_importBenchmarkFile_Bad(t *core.T) {
	db := newTestDB(t)
	core.AssertEqual(t, 0, importBenchmarkFile(db, core.JoinPath(t.TempDir(), "missing.jsonl"), "results"))
}

func TestImportAll_importBenchmarkFile_Ugly(t *core.T) {
	db := newTestDB(t)
	requireResultOK(t, db.Exec(`CREATE TABLE benchmark_results (
		source VARCHAR, id VARCHAR, benchmark VARCHAR, model VARCHAR,
		prompt TEXT, response TEXT, elapsed_seconds DOUBLE, domain VARCHAR)`))
	path := core.JoinPath(t.TempDir(), "results.jsonl")
	// A malformed line between valid ones is skipped, not fatal.
	core.RequireNoError(t, coreio.Local.Write(path,
		`{"id":"a","benchmark":"gsm8k"}`+"\n"+`{not json}`+"\n"+`{"id":"b","benchmark":"gsm8k"}`+"\n"))
	core.AssertEqual(t, 2, importBenchmarkFile(db, path, "results"))
}

func TestImportAll_importBenchmarkQuestions_Good(t *core.T) {
	db := newTestDB(t)
	requireResultOK(t, db.Exec(`CREATE TABLE benchmark_questions (
		benchmark VARCHAR, id VARCHAR, question TEXT,
		best_answer TEXT, correct_answers TEXT, incorrect_answers TEXT, category VARCHAR)`))
	path := core.JoinPath(t.TempDir(), "truthfulqa.jsonl")
	core.RequireNoError(t, coreio.Local.Write(path,
		`{"id":"q1","question":"Why?","best_answer":"because","correct_answers":["a"],"incorrect_answers":["b"],"category":"misc"}`+"\n"))
	core.AssertEqual(t, 1, importBenchmarkQuestions(db, path, "truthfulqa"))
	rCounts := db.TableCounts()
	requireResultOK(t, rCounts)
	core.AssertEqual(t, 1, rCounts.Value.(map[string]int)["benchmark_questions"])
}

func TestImportAll_importBenchmarkQuestions_Bad(t *core.T) {
	db := newTestDB(t)
	core.AssertEqual(t, 0, importBenchmarkQuestions(db, core.JoinPath(t.TempDir(), "missing.jsonl"), "truthfulqa"))
}

func TestImportAll_importBenchmarkQuestions_Ugly(t *core.T) {
	db := newTestDB(t)
	requireResultOK(t, db.Exec(`CREATE TABLE benchmark_questions (
		benchmark VARCHAR, id VARCHAR, question TEXT,
		best_answer TEXT, correct_answers TEXT, incorrect_answers TEXT, category VARCHAR)`))
	path := core.JoinPath(t.TempDir(), "truthfulqa.jsonl")
	// Malformed line skipped; missing fields serialise as empty/null JSON.
	core.RequireNoError(t, coreio.Local.Write(path, `garbage`+"\n"+`{"id":"q1"}`+"\n"))
	core.AssertEqual(t, 1, importBenchmarkQuestions(db, path, "truthfulqa"))
}

func TestImportAll_importSeeds_Good(t *core.T) {
	db := newTestDB(t)
	requireResultOK(t, db.Exec(`CREATE TABLE seeds (
		source_file VARCHAR, region VARCHAR, seed_id VARCHAR, domain VARCHAR, prompt TEXT)`))
	seedDir := t.TempDir()
	core.RequireNoError(t, coreio.Local.Write(core.JoinPath(seedDir, "uk.json"),
		`[{"seed_id":"s1","domain":"ethics","prompt":"hello"},{"seed_id":"s2","domain":"law","prompt":"world"}]`))
	core.AssertEqual(t, 2, importSeeds(db, seedDir))
	rCounts := db.TableCounts()
	requireResultOK(t, rCounts)
	core.AssertEqual(t, 2, rCounts.Value.(map[string]int)["seeds"])
}

func TestImportAll_importSeeds_Bad(t *core.T) {
	db := newTestDB(t)
	requireResultOK(t, db.Exec(`CREATE TABLE seeds (
		source_file VARCHAR, region VARCHAR, seed_id VARCHAR, domain VARCHAR, prompt TEXT)`))
	seedDir := t.TempDir()
	// A malformed .json and a non-json file both yield zero seeds.
	core.RequireNoError(t, coreio.Local.Write(core.JoinPath(seedDir, "broken.json"), `not json`))
	core.RequireNoError(t, coreio.Local.Write(core.JoinPath(seedDir, "notes.txt"), `ignored`))
	core.AssertEqual(t, 0, importSeeds(db, seedDir))
}

func TestImportAll_importSeeds_Ugly(t *core.T) {
	db := newTestDB(t)
	requireResultOK(t, db.Exec(`CREATE TABLE seeds (
		source_file VARCHAR, region VARCHAR, seed_id VARCHAR, domain VARCHAR, prompt TEXT)`))
	seedDir := t.TempDir()
	// Object form with a "seeds" array mixing raw strings and maps that fall
	// back through prompt -> text -> question.
	core.RequireNoError(t, coreio.Local.Write(core.JoinPath(seedDir, "mixed.json"),
		`{"seeds":["raw seed",{"text":"via text"},{"question":"via question"}]}`))
	// Object form with a "prompts" array.
	core.RequireNoError(t, coreio.Local.Write(core.JoinPath(seedDir, "prompts.json"),
		`{"prompts":[{"prompt":"direct"}]}`))
	core.AssertEqual(t, 4, importSeeds(db, seedDir))
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
