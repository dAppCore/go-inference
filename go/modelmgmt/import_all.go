package modelmgmt

import (
	"bufio"
	"context"
	"io"
	"io/fs"

	"dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/datapipe"
	coreio "dappco.re/go/io"
	goexec "dappco.re/go/process/exec"
)

// ImportConfig holds options for the import-all operation.
type ImportConfig struct {
	SkipM3  bool
	DataDir string
	M3Host  string
}

// ImportAll imports all LEM data into DuckDB from M3 and local files.
func ImportAll(db *datapipe.DB, cfg ImportConfig, w io.Writer) core.Result {
	m3Host := cfg.M3Host
	if m3Host == "" {
		m3Host = "m3"
	}

	totals := make(map[string]int)

	// ── 1. Golden set ──
	goldenPath := core.JoinPath(cfg.DataDir, "gold-15k.jsonl")
	if !cfg.SkipM3 {
		core.Print(w, "  Pulling golden set from M3...")
		scpCmd := goexec.Command(context.Background(), "scp", core.Sprintf("%s:/Volumes/Data/lem/responses/gold-15k.jsonl", m3Host), goldenPath)
		if r := scpCmd.Run(); !r.OK {
			core.Print(w, "  WARNING: could not pull golden set from M3: %s", r.Error())
		}
	}
	if coreio.Local.IsFile(goldenPath) {
		db.Exec("DROP TABLE IF EXISTS golden_set")
		rCreate := db.Exec(core.Sprintf(`
			CREATE TABLE golden_set AS
			SELECT
				idx::INT AS idx,
				seed_id::VARCHAR AS seed_id,
				domain::VARCHAR AS domain,
				voice::VARCHAR AS voice,
				prompt::VARCHAR AS prompt,
				response::VARCHAR AS response,
				gen_time::DOUBLE AS gen_time,
				length(response)::INT AS char_count,
				length(response) - length(replace(response, ' ', '')) + 1 AS word_count
			FROM read_json_auto('%s', maximum_object_size=1048576)
		`, escapeSQLPath(goldenPath)))
		if !rCreate.OK {
			core.Print(w, "  WARNING: golden set import failed: %s", rCreate.Error())
		} else {
			var n int
			db.QueryRowScan("SELECT count(*) FROM golden_set", &n)
			totals["golden_set"] = n
			core.Print(w, "  golden_set: %d rows", n)
		}
	}

	// ── 2. Training examples ──
	trainingDirs := []struct {
		name  string
		files []string
	}{
		{"training", []string{"training/train.jsonl", "training/valid.jsonl", "training/test.jsonl"}},
		{"training-2k", []string{"training-2k/train.jsonl", "training-2k/valid.jsonl", "training-2k/test.jsonl"}},
		{"training-expanded", []string{"training-expanded/train.jsonl", "training-expanded/valid.jsonl"}},
		{"training-book", []string{"training-book/train.jsonl", "training-book/valid.jsonl", "training-book/test.jsonl"}},
		{"training-conv", []string{"training-conv/train.jsonl", "training-conv/valid.jsonl", "training-conv/test.jsonl"}},
		{"gold-full", []string{"gold-full/train.jsonl", "gold-full/valid.jsonl"}},
		{"sovereignty-gold", []string{"sovereignty-gold/train.jsonl", "sovereignty-gold/valid.jsonl"}},
		{"composure-lessons", []string{"composure-lessons/train.jsonl", "composure-lessons/valid.jsonl"}},
		{"watts-full", []string{"watts-full/train.jsonl", "watts-full/valid.jsonl"}},
		{"watts-expanded", []string{"watts-expanded/train.jsonl", "watts-expanded/valid.jsonl"}},
		{"watts-composure", []string{"watts-composure-merged/train.jsonl", "watts-composure-merged/valid.jsonl"}},
		{"western-fresh", []string{"western-fresh/train.jsonl", "western-fresh/valid.jsonl"}},
		{"deepseek-soak", []string{"deepseek-western-soak/train.jsonl", "deepseek-western-soak/valid.jsonl"}},
		{"russian-bridge", []string{"russian-bridge/train.jsonl", "russian-bridge/valid.jsonl"}},
	}

	trainingLocal := core.JoinPath(cfg.DataDir, "training")
	coreio.Local.EnsureDir(trainingLocal)

	if !cfg.SkipM3 {
		core.Print(w, "  Pulling training sets from M3...")
		for _, td := range trainingDirs {
			for _, rel := range td.files {
				local := core.JoinPath(trainingLocal, rel)
				coreio.Local.EnsureDir(core.PathDir(local))
				scpCmd := goexec.Command(context.Background(), "scp", core.Sprintf("%s:/Volumes/Data/lem/%s", m3Host, rel), local)
				scpCmd.Run() // ignore errors, file might not exist
			}
		}
	}

	db.Exec("DROP TABLE IF EXISTS training_examples")
	db.Exec(`
		CREATE TABLE training_examples (
			source VARCHAR,
			split VARCHAR,
			prompt TEXT,
			response TEXT,
			num_turns INT,
			full_messages TEXT,
			char_count INT
		)
	`)

	trainingTotal := 0
	for _, td := range trainingDirs {
		for _, rel := range td.files {
			local := core.JoinPath(trainingLocal, rel)
			if !coreio.Local.IsFile(local) {
				continue
			}

			split := "train"
			if core.Contains(rel, "valid") {
				split = "valid"
			} else if core.Contains(rel, "test") {
				split = "test"
			}

			n := importTrainingFile(db, local, td.name, split)
			trainingTotal += n
		}
	}
	totals["training_examples"] = trainingTotal
	core.Print(w, "  training_examples: %d rows", trainingTotal)

	// ── 3. Benchmark results ──
	benchLocal := core.JoinPath(cfg.DataDir, "benchmarks")
	coreio.Local.EnsureDir(benchLocal)

	if !cfg.SkipM3 {
		core.Print(w, "  Pulling benchmarks from M3...")
		for _, bname := range []string{"truthfulqa", "gsm8k", "do_not_answer", "toxigen"} {
			scpCmd := goexec.Command(context.Background(), "scp",
				core.Sprintf("%s:/Volumes/Data/lem/benchmarks/%s.jsonl", m3Host, bname),
				core.JoinPath(benchLocal, core.Concat(bname, ".jsonl")))
			scpCmd.Run()
		}
		for _, subdir := range []string{"results", "scale_results", "cross_arch_results", "deepseek-r1-7b"} {
			localSub := core.JoinPath(benchLocal, subdir)
			coreio.Local.EnsureDir(localSub)
			scpCmd := goexec.Command(context.Background(), "scp", "-r",
				core.Sprintf("%s:/Volumes/Data/lem/benchmarks/%s/", m3Host, subdir),
				core.Concat(benchLocal, "/"))
			scpCmd.Run()
		}
	}

	db.Exec("DROP TABLE IF EXISTS benchmark_results")
	db.Exec(`
		CREATE TABLE benchmark_results (
			source VARCHAR, id VARCHAR, benchmark VARCHAR, model VARCHAR,
			prompt TEXT, response TEXT, elapsed_seconds DOUBLE, domain VARCHAR
		)
	`)

	benchTotal := 0
	for _, subdir := range []string{"results", "scale_results", "cross_arch_results", "deepseek-r1-7b"} {
		resultDir := core.JoinPath(benchLocal, subdir)
		matches := core.PathGlob(core.JoinPath(resultDir, "*.jsonl"))
		for _, jf := range matches {
			n := importBenchmarkFile(db, jf, subdir)
			benchTotal += n
		}
	}

	// Also import standalone benchmark files.
	for _, bfile := range []string{"lem_bench", "lem_ethics", "lem_ethics_allen", "instruction_tuned", "abliterated", "base_pt"} {
		local := core.JoinPath(benchLocal, core.Concat(bfile, ".jsonl"))
		if !coreio.Local.IsFile(local) {
			if !cfg.SkipM3 {
				scpCmd := goexec.Command(context.Background(), "scp",
					core.Sprintf("%s:/Volumes/Data/lem/benchmark/%s.jsonl", m3Host, bfile), local)
				scpCmd.Run()
			}
		}
		if coreio.Local.IsFile(local) {
			n := importBenchmarkFile(db, local, "benchmark")
			benchTotal += n
		}
	}
	totals["benchmark_results"] = benchTotal
	core.Print(w, "  benchmark_results: %d rows", benchTotal)

	// ── 4. Benchmark questions ──
	db.Exec("DROP TABLE IF EXISTS benchmark_questions")
	db.Exec(`
		CREATE TABLE benchmark_questions (
			benchmark VARCHAR, id VARCHAR, question TEXT,
			best_answer TEXT, correct_answers TEXT, incorrect_answers TEXT, category VARCHAR
		)
	`)

	benchQTotal := 0
	for _, bname := range []string{"truthfulqa", "gsm8k", "do_not_answer", "toxigen"} {
		local := core.JoinPath(benchLocal, core.Concat(bname, ".jsonl"))
		if coreio.Local.IsFile(local) {
			n := importBenchmarkQuestions(db, local, bname)
			benchQTotal += n
		}
	}
	totals["benchmark_questions"] = benchQTotal
	core.Print(w, "  benchmark_questions: %d rows", benchQTotal)

	// ── 5. Seeds ──
	db.Exec("DROP TABLE IF EXISTS seeds")
	db.Exec(`
		CREATE TABLE seeds (
			source_file VARCHAR, region VARCHAR, seed_id VARCHAR, domain VARCHAR, prompt TEXT
		)
	`)

	seedTotal := 0
	seedDirs := []string{core.JoinPath(cfg.DataDir, "seeds"), "/tmp/lem-data/seeds", "/tmp/lem-repo/seeds"}
	for _, seedDir := range seedDirs {
		if !coreio.Local.IsDir(seedDir) {
			continue
		}
		n := importSeeds(db, seedDir)
		seedTotal += n
	}
	totals["seeds"] = seedTotal
	core.Print(w, "  seeds: %d rows", seedTotal)

	// ── Summary ──
	grandTotal := 0
	core.Print(w, "")
	core.Print(w, "%s", repeatString("=", 50))
	core.Print(w, "LEM Database Import Complete")
	core.Print(w, "%s", repeatString("=", 50))
	for table, count := range totals {
		core.Print(w, "  %-25s %8d", table, count)
		grandTotal += count
	}
	core.Print(w, "  %s", repeatString("-", 35))
	core.Print(w, "  %-25s %8d", "TOTAL", grandTotal)
	core.Print(w, "")
	core.Print(w, "Database: %s", db.Path())

	return core.Ok(nil)
}

func importTrainingFile(db *datapipe.DB, path, source, split string) int {
	f, err := coreio.Local.Open(path)
	if err != nil {
		return 0
	}
	defer f.Close()

	count := 0
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	for scanner.Scan() {
		var rec struct {
			Messages []inference.Message `json:"messages"`
		}
		if r := core.JSONUnmarshalString(string(scanner.Bytes()), &rec); !r.OK {
			continue
		}

		prompt := ""
		response := ""
		assistantCount := 0
		for _, m := range rec.Messages {
			if m.Role == "user" && prompt == "" {
				prompt = m.Content
			}
			if m.Role == "assistant" {
				if response == "" {
					response = m.Content
				}
				assistantCount++
			}
		}

		msgsJSON := core.JSONMarshalString(rec.Messages)
		db.Exec(`INSERT INTO training_examples VALUES (?, ?, ?, ?, ?, ?, ?)`,
			source, split, prompt, response, assistantCount, msgsJSON, len(response))
		count++
	}
	return count
}

func importBenchmarkFile(db *datapipe.DB, path, source string) int {
	f, err := coreio.Local.Open(path)
	if err != nil {
		return 0
	}
	defer f.Close()

	count := 0
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	for scanner.Scan() {
		var rec map[string]any
		if r := core.JSONUnmarshalString(string(scanner.Bytes()), &rec); !r.OK {
			continue
		}

		db.Exec(`INSERT INTO benchmark_results VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
			source,
			core.Sprint(rec["id"]),
			strOrEmpty(rec, "benchmark"),
			strOrEmpty(rec, "model"),
			strOrEmpty(rec, "prompt"),
			strOrEmpty(rec, "response"),
			floatOrZero(rec, "elapsed_seconds"),
			strOrEmpty(rec, "domain"),
		)
		count++
	}
	return count
}

func importBenchmarkQuestions(db *datapipe.DB, path, benchmark string) int {
	f, err := coreio.Local.Open(path)
	if err != nil {
		return 0
	}
	defer f.Close()

	count := 0
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	for scanner.Scan() {
		var rec map[string]any
		if r := core.JSONUnmarshalString(string(scanner.Bytes()), &rec); !r.OK {
			continue
		}

		correctJSON := core.JSONMarshalString(rec["correct_answers"])
		incorrectJSON := core.JSONMarshalString(rec["incorrect_answers"])

		db.Exec(`INSERT INTO benchmark_questions VALUES (?, ?, ?, ?, ?, ?, ?)`,
			benchmark,
			core.Sprint(rec["id"]),
			strOrEmpty(rec, "question"),
			strOrEmpty(rec, "best_answer"),
			correctJSON,
			incorrectJSON,
			strOrEmpty(rec, "category"),
		)
		count++
	}
	return count
}

func importSeeds(db *datapipe.DB, seedDir string) int {
	count := 0
	fs.WalkDir(core.DirFS(seedDir), ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !core.HasSuffix(path, ".json") {
			return nil
		}

		fullPath := core.JoinPath(seedDir, path)
		data, err := coreio.Local.Read(fullPath)
		if err != nil {
			return nil
		}

		rel := path
		region := core.TrimSuffix(core.PathBase(path), ".json")

		// Try parsing as array or object with prompts/seeds field.
		var seedsList []any
		var raw any
		if r := core.JSONUnmarshalString(data, &raw); !r.OK {
			return nil
		}

		switch v := raw.(type) {
		case []any:
			seedsList = v
		case map[string]any:
			if prompts, ok := v["prompts"].([]any); ok {
				seedsList = prompts
			} else if seeds, ok := v["seeds"].([]any); ok {
				seedsList = seeds
			}
		}

		for _, s := range seedsList {
			switch seed := s.(type) {
			case map[string]any:
				prompt := strOrEmpty(seed, "prompt")
				if prompt == "" {
					prompt = strOrEmpty(seed, "text")
				}
				if prompt == "" {
					prompt = strOrEmpty(seed, "question")
				}
				db.Exec(`INSERT INTO seeds VALUES (?, ?, ?, ?, ?)`,
					rel, region,
					strOrEmpty(seed, "seed_id"),
					strOrEmpty(seed, "domain"),
					prompt,
				)
				count++
			case string:
				db.Exec(`INSERT INTO seeds VALUES (?, ?, ?, ?, ?)`,
					rel, region, "", "", seed)
				count++
			}
		}
		return nil
	})
	return count
}

func strOrEmpty(m map[string]any, key string) string {
	if v, ok := m[key]; ok {
		return core.Sprint(v)
	}
	return ""
}

func floatOrZero(m map[string]any, key string) float64 {
	if v, ok := m[key]; ok {
		if f, ok := v.(float64); ok {
			return f
		}
	}
	return 0
}

func escapeSQLPath(p string) string {
	return core.Replace(p, "'", "''")
}
