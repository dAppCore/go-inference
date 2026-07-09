// SPDX-Licence-Identifier: EUPL-1.2

package train

import (
	"context"
	"errors"
	"math"
	"slices"
	"strings"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func boolToCodeBenchmarkPassedTests(pass bool, total int) int {
	if pass {
		return total
	}
	return 0
}

// --- RunSSDCodeBenchmark ---

// TestRunSSDCodeBenchmark_Good runs the benchmark end-to-end: every
// sample is sampled NRepeat times with per-repeat seeds, candidates are
// post-processed and executed, the report aggregates pass@k, and the JSON report
// is written to OutputPath. The runner is pure hooks — no model, no Metal.
func TestSsdEval_RunSSDCodeBenchmark_Good(t *testing.T) {
	outputPath := core.PathJoin(t.TempDir(), "reports", "lcb.json")
	var prompts []string
	var configs []inference.GenerateConfig
	var executed []string

	report, err := RunSSDCodeBenchmark(context.Background(), SSDCodeBenchmarkRunner{
		Generate: func(_ context.Context, prompt string, cfg inference.GenerateConfig) (string, error) {
			prompts = append(prompts, prompt)
			configs = append(configs, cfg)
			if strings.Contains(prompt, "add") {
				return "```python\ndef add(a, b): return a + b\n```", nil
			}
			return "```python\ndef sub(a, b): return a - b\n```", nil
		},
		RunTests: func(_ context.Context, sample SSDCodeBenchmarkSample, candidate SSDCodeCandidate) (SSDCodeExecution, error) {
			executed = append(executed, sample.ID+"/"+candidate.Solution)
			return SSDCodeExecution{
				Passed:      strings.Contains(candidate.Solution, "+"),
				TotalTests:  len(sample.Tests),
				PassedTests: boolToCodeBenchmarkPassedTests(strings.Contains(candidate.Solution, "+"), len(sample.Tests)),
				DurationMS:  12,
			}, nil
		},
	}, []SSDCodeBenchmarkSample{
		{ID: "add", Prompt: "write add", Tests: []string{"assert add(1, 2) == 3"}},
		{ID: "sub", Prompt: "write sub", Tests: []string{"assert sub(3, 1) == 2"}},
	}, SSDCodeBenchmarkConfig{
		Benchmark:  "LiveCodeBench-v6",
		NRepeat:    2,
		Seeds:      []uint64{7, 1234},
		OutputPath: outputPath,
		Generate: inference.GenerateConfig{
			MaxTokens:     128,
			Temperature:   0.8,
			TopP:          0.95,
			TopK:          64,
			RepeatPenalty: 1.1,
		},
	})
	if err != nil {
		t.Fatalf("RunSSDCodeBenchmark() error = %v", err)
	}
	if len(prompts) != 4 || len(executed) != 4 {
		t.Fatalf("generated=%d executed=%d, want n_repeat per sample", len(prompts), len(executed))
	}
	if configs[0].MaxTokens != 128 || configs[0].Temperature != 0.8 || configs[0].TopP != 0.95 || configs[0].TopK != 64 || configs[0].RepeatPenalty != 1.1 {
		t.Fatalf("generate config = %+v, want caller sampling config", configs[0])
	}
	if len(configs) != 4 || !configs[0].SeedSet || configs[0].Seed != 7 || configs[1].Seed != 8 || configs[2].Seed != 7 || configs[3].Seed != 8 {
		t.Fatalf("generate seeds = %+v, want seed base plus repeat per sample", configs)
	}
	if report.Benchmark != "LiveCodeBench-v6" || report.Config.NRepeat != 2 || report.Config.OutputPath != outputPath {
		t.Fatalf("report config = %+v benchmark=%q", report.Config, report.Benchmark)
	}
	if report.Metrics.Samples != 2 || report.Metrics.Candidates != 4 || report.Metrics.Passed != 2 || report.Metrics.PassRate != 0.5 {
		t.Fatalf("metrics = %+v, want 2/4 candidates passing", report.Metrics)
	}
	if report.Metrics.PassAtK["pass@1"] != 0.5 {
		t.Fatalf("pass@k = %+v, want pass@1=0.5", report.Metrics.PassAtK)
	}
	if len(report.Results) != 2 || len(report.Results[0].Candidates) != 2 {
		t.Fatalf("results = %+v, want per-sample candidate results", report.Results)
	}
	if !report.Results[0].Candidates[0].Candidate.HasCode || !strings.Contains(report.Results[0].Candidates[0].Candidate.RawSolution, "```python") {
		t.Fatalf("candidate = %+v, want raw fenced output and extracted code marker", report.Results[0].Candidates[0].Candidate)
	}
	data := core.ReadFile(outputPath)
	if !data.OK {
		t.Fatalf("ReadFile(%s) error = %v", outputPath, data.Value)
	}
	if !strings.Contains(string(data.Value.([]byte)), `"benchmark": "LiveCodeBench-v6"`) {
		t.Fatalf("report file = %s, want benchmark JSON", string(data.Value.([]byte)))
	}
}

// TestRunSSDCodeBenchmark_Bad asserts the missing-hook rejections (no
// Generate, no RunTests) and the empty-samples rejection — the benchmark needs
// both hooks and at least one task.
func TestSsdEval_RunSSDCodeBenchmark_Bad(t *testing.T) {
	if _, err := RunSSDCodeBenchmark(context.Background(), SSDCodeBenchmarkRunner{}, nil, SSDCodeBenchmarkConfig{}); err == nil {
		t.Fatal("RunSSDCodeBenchmark() error = nil, want missing Generate")
	}
	if _, err := RunSSDCodeBenchmark(context.Background(), SSDCodeBenchmarkRunner{
		Generate: func(context.Context, string, inference.GenerateConfig) (string, error) { return "", nil },
	}, nil, SSDCodeBenchmarkConfig{}); err == nil {
		t.Fatal("RunSSDCodeBenchmark() error = nil, want missing RunTests")
	}
	// Both hooks present but no samples → rejected.
	if _, err := RunSSDCodeBenchmark(context.Background(), SSDCodeBenchmarkRunner{
		Generate: func(context.Context, string, inference.GenerateConfig) (string, error) { return "x", nil },
		RunTests: func(context.Context, SSDCodeBenchmarkSample, SSDCodeCandidate) (SSDCodeExecution, error) {
			return SSDCodeExecution{}, nil
		},
	}, nil, SSDCodeBenchmarkConfig{}); err == nil {
		t.Fatal("RunSSDCodeBenchmark(no samples) error = nil, want empty-samples rejection")
	}
}

// TestRunSSDCodeBenchmark_Ugly covers the defaulting and pass@k edges:
// an unset config floors NRepeat to 1 and the generate budget to the SSD
// default, and a 10-repeat run estimates pass@5 (and per-difficulty) while
// omitting pass@10.
func TestSsdEval_RunSSDCodeBenchmark_Ugly(t *testing.T) {
	t.Run("Defaults", func(t *testing.T) {
		report, err := RunSSDCodeBenchmark(context.Background(), SSDCodeBenchmarkRunner{
			Generate: func(context.Context, string, inference.GenerateConfig) (string, error) { return "solution", nil },
			RunTests: func(context.Context, SSDCodeBenchmarkSample, SSDCodeCandidate) (SSDCodeExecution, error) {
				return SSDCodeExecution{Passed: true, TotalTests: 1, PassedTests: 1}, nil
			},
		}, []SSDCodeBenchmarkSample{{Prompt: "p"}}, SSDCodeBenchmarkConfig{})
		if err != nil {
			t.Fatalf("RunSSDCodeBenchmark(defaults) error = %v", err)
		}
		if report.Config.NRepeat != 1 || report.Config.Generate.MaxTokens != defaultSSDMaxTokens {
			t.Fatalf("default config = %+v", report.Config)
		}
	})

	t.Run("PassAtKEstimation", func(t *testing.T) {
		calls := map[string]int{}
		report, err := RunSSDCodeBenchmark(context.Background(), SSDCodeBenchmarkRunner{
			Generate: func(_ context.Context, prompt string, _ inference.GenerateConfig) (string, error) {
				call := calls[prompt]
				calls[prompt] = call + 1
				return core.Sprintf("```python\n%s/%d\n```", prompt, call), nil
			},
			RunTests: func(_ context.Context, _ SSDCodeBenchmarkSample, candidate SSDCodeCandidate) (SSDCodeExecution, error) {
				solution := core.Trim(candidate.Solution)
				return SSDCodeExecution{
					Passed:      strings.HasSuffix(solution, "/0") || strings.HasSuffix(solution, "/1"),
					TotalTests:  1,
					PassedTests: boolToCodeBenchmarkPassedTests(strings.HasSuffix(solution, "/0") || strings.HasSuffix(solution, "/1"), 1),
				}, nil
			},
		}, []SSDCodeBenchmarkSample{
			{ID: "a", Prompt: "a", Tests: []string{"test"}, Meta: map[string]string{"difficulty": "easy"}},
			{ID: "b", Prompt: "b", Tests: []string{"test"}, Meta: map[string]string{"difficulty": "hard"}},
		}, SSDCodeBenchmarkConfig{NRepeat: 10})
		if err != nil {
			t.Fatalf("RunSSDCodeBenchmark() error = %v", err)
		}
		if math.Abs(report.Metrics.PassAtK["pass@1"]-0.2) > 0.000001 {
			t.Fatalf("pass@1 = %f, want 0.2", report.Metrics.PassAtK["pass@1"])
		}
		if math.Abs(report.Metrics.PassAtK["pass@5"]-0.777777) > 0.000001 {
			t.Fatalf("pass@5 = %f, want estimated 0.777777", report.Metrics.PassAtK["pass@5"])
		}
		if _, ok := report.Metrics.PassAtK["pass@10"]; ok {
			t.Fatalf("pass@k = %+v, did not want pass@10 for n_repeat=10", report.Metrics.PassAtK)
		}
		if math.Abs(report.Metrics.Difficulty["pass@5_easy"]-0.777777) > 0.000001 || math.Abs(report.Metrics.Difficulty["pass@5_hard"]-0.777777) > 0.000001 {
			t.Fatalf("difficulty metrics = %+v, want pass@5 per difficulty", report.Metrics.Difficulty)
		}
	})
}

// TestRunSSDCodeBenchmark_ContextAndRunnerErrors covers the orchestration
// error arms: a nil context defaults to Background, an already cancelled
// context aborts at the sample boundary, a failing Generate and a failing
// RunTests each surface mid-run, and a bad OutputPath fails the report
// write after the run completes.
func TestRunSSDCodeBenchmark_ContextAndRunnerErrors(t *testing.T) {
	okGen := func(context.Context, string, inference.GenerateConfig) (string, error) {
		return "```python\nx=1\n```", nil
	}
	okTests := func(context.Context, SSDCodeBenchmarkSample, SSDCodeCandidate) (SSDCodeExecution, error) {
		return SSDCodeExecution{Passed: true, TotalTests: 1, PassedTests: 1}, nil
	}
	samples := []SSDCodeBenchmarkSample{{ID: "a", Prompt: "p"}}

	// nil context → Background; run completes.
	if _, err := RunSSDCodeBenchmark(nil, SSDCodeBenchmarkRunner{Generate: okGen, RunTests: okTests}, samples, SSDCodeBenchmarkConfig{}); err != nil { //nolint:staticcheck // nil ctx is the branch under test
		t.Fatalf("RunSSDCodeBenchmark(nil ctx) error = %v, want nil", err)
	}

	// Cancelled context aborts at the first sample's ctx.Err() guard.
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := RunSSDCodeBenchmark(cancelled, SSDCodeBenchmarkRunner{Generate: okGen, RunTests: okTests}, samples, SSDCodeBenchmarkConfig{}); err == nil {
		t.Fatal("RunSSDCodeBenchmark(cancelled) error = nil, want context error")
	}

	// Generate error surfaces.
	genErr := errors.New("generate boom")
	if _, err := RunSSDCodeBenchmark(context.Background(), SSDCodeBenchmarkRunner{
		Generate: func(context.Context, string, inference.GenerateConfig) (string, error) { return "", genErr },
		RunTests: okTests,
	}, samples, SSDCodeBenchmarkConfig{}); !errors.Is(err, genErr) {
		t.Fatalf("RunSSDCodeBenchmark generate-error = %v, want %v", err, genErr)
	}

	// RunTests error surfaces.
	testErr := errors.New("runtests boom")
	if _, err := RunSSDCodeBenchmark(context.Background(), SSDCodeBenchmarkRunner{
		Generate: okGen,
		RunTests: func(context.Context, SSDCodeBenchmarkSample, SSDCodeCandidate) (SSDCodeExecution, error) {
			return SSDCodeExecution{}, testErr
		},
	}, samples, SSDCodeBenchmarkConfig{}); !errors.Is(err, testErr) {
		t.Fatalf("RunSSDCodeBenchmark runtests-error = %v, want %v", err, testErr)
	}

	// Bad OutputPath: a file where the parent directory must be → write fails.
	blocker := core.PathJoin(t.TempDir(), "blocker")
	if w := core.WriteFile(blocker, []byte("x"), 0o600); !w.OK {
		t.Fatalf("setup blocker: %v", w.Value)
	}
	badOut := core.PathJoin(blocker, "report.json")
	if _, err := RunSSDCodeBenchmark(context.Background(), SSDCodeBenchmarkRunner{Generate: okGen, RunTests: okTests}, samples, SSDCodeBenchmarkConfig{OutputPath: badOut}); err == nil {
		t.Fatal("RunSSDCodeBenchmark(bad OutputPath) error = nil, want report-write failure")
	}
}

// TestRunSSDCodeBenchmark_RepeatLevelCancellation covers the per-repeat
// ctx.Err() guard inside RunSSDCodeBenchmark: the context is live at
// sample start but a Generate call cancels it, so the next repeat's guard
// aborts the run.
func TestRunSSDCodeBenchmark_RepeatLevelCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	gen := func(context.Context, string, inference.GenerateConfig) (string, error) {
		cancel() // cancel after the first generation; the next repeat guard trips
		return "```python\nx=1\n```", nil
	}
	tests := func(context.Context, SSDCodeBenchmarkSample, SSDCodeCandidate) (SSDCodeExecution, error) {
		return SSDCodeExecution{}, nil
	}
	_, err := RunSSDCodeBenchmark(ctx, SSDCodeBenchmarkRunner{Generate: gen, RunTests: tests},
		[]SSDCodeBenchmarkSample{{ID: "a", Prompt: "p"}}, SSDCodeBenchmarkConfig{NRepeat: 4})
	if err == nil {
		t.Fatal("RunSSDCodeBenchmark with mid-repeat cancel = nil, want context error")
	}
}

// --- LoadSSDCodeBenchmarkJSONL ---

// TestLoadSSDCodeBenchmarkJSONL_Good parses LiveCodeBench-style rows into
// native samples, merging the prompt/starter-code, the public+private tests, and
// the metadata. Two distinct row shapes (full record + minimal id/prompt/test)
// both load.
func TestSsdEval_LoadSSDCodeBenchmarkJSONL_Good(t *testing.T) {
	raw := `{"question_id":"q1","question_content":"Write add.","starter_code":"def add(a,b):\n    pass","entry_point":"def add(a,b):\n    pass","is_stdin":false,"contest_date":"2025-03-01","public_test_cases":["assert add(1, 2) == 3"],"private_test_cases":["assert add(-1, 1) == 0"],"difficulty":"easy","platform":"leetcode"}`
	raw += "\n"
	raw += `{"id":"q2","prompt":"Write sub.","test":"assert sub(3, 1) == 2"}`
	samples, err := LoadSSDCodeBenchmarkJSONL(raw)
	if err != nil {
		t.Fatalf("LoadSSDCodeBenchmarkJSONL() error = %v", err)
	}
	if len(samples) != 2 {
		t.Fatalf("samples = %d, want 2", len(samples))
	}
	if samples[0].ID != "q1" || !strings.Contains(samples[0].Prompt, "Write add.") || !strings.Contains(samples[0].Prompt, "starter code") {
		t.Fatalf("sample[0] = %+v, want id and starter-code prompt", samples[0])
	}
	if len(samples[0].Tests) != 2 || samples[0].Tests[0] != "assert add(1, 2) == 3" || samples[0].Meta["difficulty"] != "easy" || samples[0].Meta["platform"] != "leetcode" ||
		samples[0].Meta["entry_point"] == "" || samples[0].Meta["is_stdin"] != "false" || samples[0].Meta["contest_date"] != "2025-03-01" {
		t.Fatalf("sample[0] tests/meta = %+v/%+v", samples[0].Tests, samples[0].Meta)
	}
	if samples[1].ID != "q2" || samples[1].Tests[0] != "assert sub(3, 1) == 2" {
		t.Fatalf("sample[1] = %+v", samples[1])
	}
}

// TestLoadSSDCodeBenchmarkJSONL_Bad asserts the loud failures: malformed
// JSON on a line errors with the line number, and an all-blank / no-prompt input
// produces the no-samples error rather than an empty success.
func TestSsdEval_LoadSSDCodeBenchmarkJSONL_Bad(t *testing.T) {
	if _, err := LoadSSDCodeBenchmarkJSONL(`{not valid json`); err == nil {
		t.Fatal("LoadSSDCodeBenchmarkJSONL(malformed) error = nil, want parse failure")
	}
	// A row with no prompt-bearing field is skipped; an all-skipped input yields
	// the no-samples error.
	if _, err := LoadSSDCodeBenchmarkJSONL(`{"id":"x","test":"assert true"}`); err == nil {
		t.Fatal("LoadSSDCodeBenchmarkJSONL(no prompt) error = nil, want no-samples rejection")
	}
}

// TestLoadSSDCodeBenchmarkJSONL_Ugly asserts blank and whitespace-only
// lines are skipped, so a file padded with empty lines still loads its real rows.
func TestSsdEval_LoadSSDCodeBenchmarkJSONL_Ugly(t *testing.T) {
	raw := "\n   \n" + `{"id":"q","prompt":"Write identity.","tests":["assert f(1) == 1"]}` + "\n\n"
	samples, err := LoadSSDCodeBenchmarkJSONL(raw)
	if err != nil {
		t.Fatalf("LoadSSDCodeBenchmarkJSONL(padded) error = %v", err)
	}
	if len(samples) != 1 || samples[0].ID != "q" {
		t.Fatalf("samples = %+v, want the one real row (blank lines skipped)", samples)
	}
}

// --- LoadSSDCodeBenchmarkJSONLFile ---

// TestLoadSSDCodeBenchmarkJSONLFile_Good loads tasks from a JSONL file
// path written to a temp dir.
func TestSsdEval_LoadSSDCodeBenchmarkJSONLFile_Good(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "lcb.jsonl")
	write := core.WriteFile(path, []byte(`{"id":"q","prompt":"Write identity.","tests":["assert f(1) == 1"]}`+"\n"), 0o644)
	if !write.OK {
		t.Fatalf("WriteFile() error = %v", write.Value)
	}
	samples, err := LoadSSDCodeBenchmarkJSONLFile(path)
	if err != nil {
		t.Fatalf("LoadSSDCodeBenchmarkJSONLFile() error = %v", err)
	}
	if len(samples) != 1 || samples[0].Tests[0] != "assert f(1) == 1" {
		t.Fatalf("samples = %+v", samples)
	}
}

// TestLoadSSDCodeBenchmarkJSONLFile_Bad asserts a missing file path
// surfaces the read error rather than returning empty samples.
func TestSsdEval_LoadSSDCodeBenchmarkJSONLFile_Bad(t *testing.T) {
	if _, err := LoadSSDCodeBenchmarkJSONLFile(core.PathJoin(t.TempDir(), "does-not-exist.jsonl")); err == nil {
		t.Fatal("LoadSSDCodeBenchmarkJSONLFile(missing) error = nil, want read failure")
	}
}

// TestLoadSSDCodeBenchmarkJSONLFile_Ugly asserts a file that exists but
// holds only blank lines surfaces the no-samples error (the file read succeeds,
// but parsing yields nothing).
func TestSsdEval_LoadSSDCodeBenchmarkJSONLFile_Ugly(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "blank.jsonl")
	if w := core.WriteFile(path, []byte("\n   \n\n"), 0o644); !w.OK {
		t.Fatalf("WriteFile() error = %v", w.Value)
	}
	if _, err := LoadSSDCodeBenchmarkJSONLFile(path); err == nil {
		t.Fatal("LoadSSDCodeBenchmarkJSONLFile(blank) error = nil, want no-samples rejection")
	}
}

// --- LoadSSDLiveCodeBenchV6JSONL ---

// TestLoadSSDLiveCodeBenchV6JSONL_Good parses then filters to the v6
// contest-date window (Feb–May 2025), keeping only the in-window rows.
func TestSsdEval_LoadSSDLiveCodeBenchV6JSONL_Good(t *testing.T) {
	raw := `{"id":"jan","prompt":"old","contest_date":"2025-01-31"}`
	raw += "\n"
	raw += `{"id":"feb","prompt":"first v6","contest_date":"2025-02-01","difficulty":"easy"}`
	raw += "\n"
	raw += `{"id":"may","prompt":"last v6","contest_date":"2025-05-31","difficulty":"hard"}`
	raw += "\n"
	raw += `{"id":"jun","prompt":"new","contest_date":"2025-06-01"}`

	samples, err := LoadSSDLiveCodeBenchV6JSONL(raw)
	if err != nil {
		t.Fatalf("LoadSSDLiveCodeBenchV6JSONL() error = %v", err)
	}
	if len(samples) != 2 || samples[0].ID != "feb" || samples[1].ID != "may" {
		t.Fatalf("samples = %+v, want Feb-May 2025 subset", samples)
	}
	if samples[0].Meta["difficulty"] != "easy" || samples[1].Meta["difficulty"] != "hard" {
		t.Fatalf("sample metadata = %+v/%+v", samples[0].Meta, samples[1].Meta)
	}
}

// TestLoadSSDLiveCodeBenchV6JSONL_Bad asserts an input whose rows all
// fall outside the v6 window yields the empty-subset error — a v6 load with no
// v6 tasks is a failure, not a silent empty set.
func TestSsdEval_LoadSSDLiveCodeBenchV6JSONL_Bad(t *testing.T) {
	if _, err := LoadSSDLiveCodeBenchV6JSONL(`{"id":"old","prompt":"old","contest_date":"2025-01-01"}`); err == nil {
		t.Fatal("LoadSSDLiveCodeBenchV6JSONL(out-of-window) error = nil, want empty v6 subset")
	}
	// Malformed JSON propagates from the underlying parse.
	if _, err := LoadSSDLiveCodeBenchV6JSONL(`{not valid`); err == nil {
		t.Fatal("LoadSSDLiveCodeBenchV6JSONL(malformed) error = nil, want parse failure")
	}
}

// TestLoadSSDLiveCodeBenchV6JSONL_Ugly asserts the window boundaries are
// half-open [2025-02-01, 2025-06-01): the first instant is in, the upper bound
// instant is out. A row with no contest_date is treated as out-of-window.
func TestSsdEval_LoadSSDLiveCodeBenchV6JSONL_Ugly(t *testing.T) {
	// Lower bound included, upper bound excluded, missing date excluded.
	raw := `{"id":"lower","prompt":"in","contest_date":"2025-02-01"}` + "\n" +
		`{"id":"upper","prompt":"out","contest_date":"2025-06-01"}` + "\n" +
		`{"id":"nodate","prompt":"out"}`
	samples, err := LoadSSDLiveCodeBenchV6JSONL(raw)
	if err != nil {
		t.Fatalf("LoadSSDLiveCodeBenchV6JSONL() error = %v", err)
	}
	if len(samples) != 1 || samples[0].ID != "lower" {
		t.Fatalf("samples = %+v, want only the lower-bound row (half-open window)", samples)
	}
}

// --- LoadSSDLiveCodeBenchV6JSONLFile ---

// TestLoadSSDLiveCodeBenchV6JSONLFile_Good loads the v6 subset from a
// file path.
func TestSsdEval_LoadSSDLiveCodeBenchV6JSONLFile_Good(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "lcb-v6.jsonl")
	write := core.WriteFile(path, []byte(`{"id":"q","prompt":"Write identity.","contest_date":"2025-03-15","tests":["assert f(1) == 1"]}`+"\n"), 0o644)
	if !write.OK {
		t.Fatalf("WriteFile() error = %v", write.Value)
	}
	samples, err := LoadSSDLiveCodeBenchV6JSONLFile(path)
	if err != nil {
		t.Fatalf("LoadSSDLiveCodeBenchV6JSONLFile() error = %v", err)
	}
	if len(samples) != 1 || samples[0].ID != "q" || samples[0].Meta["contest_date"] != "2025-03-15" {
		t.Fatalf("samples = %+v", samples)
	}
}

// TestLoadSSDLiveCodeBenchV6JSONLFile_Bad asserts a missing file path
// surfaces the read error.
func TestSsdEval_LoadSSDLiveCodeBenchV6JSONLFile_Bad(t *testing.T) {
	if _, err := LoadSSDLiveCodeBenchV6JSONLFile(core.PathJoin(t.TempDir(), "missing-v6.jsonl")); err == nil {
		t.Fatal("LoadSSDLiveCodeBenchV6JSONLFile(missing) error = nil, want read failure")
	}
}

// TestLoadSSDLiveCodeBenchV6JSONLFile_Ugly asserts a file whose rows are
// all outside the v6 window surfaces the empty-subset error after a successful
// read.
func TestSsdEval_LoadSSDLiveCodeBenchV6JSONLFile_Ugly(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "out-of-window.jsonl")
	if w := core.WriteFile(path, []byte(`{"id":"old","prompt":"old","contest_date":"2024-12-31"}`+"\n"), 0o644); !w.OK {
		t.Fatalf("WriteFile() error = %v", w.Value)
	}
	if _, err := LoadSSDLiveCodeBenchV6JSONLFile(path); err == nil {
		t.Fatal("LoadSSDLiveCodeBenchV6JSONLFile(out-of-window) error = nil, want empty-subset rejection")
	}
}

// --- FilterSSDLiveCodeBenchV6Samples ---

// TestFilterSSDLiveCodeBenchV6Samples_Good keeps only the samples whose
// contest_date meta falls in the v6 window, dropping the rest and cloning the
// survivors.
func TestSsdEval_FilterSSDLiveCodeBenchV6Samples_Good(t *testing.T) {
	in := []SSDCodeBenchmarkSample{
		{ID: "jan", Meta: map[string]string{"contest_date": "2025-01-15"}},
		{ID: "mar", Meta: map[string]string{"contest_date": "2025-03-15"}},
		{ID: "apr", Meta: map[string]string{"contest_date": "2025-04-30"}},
		{ID: "jun", Meta: map[string]string{"contest_date": "2025-06-15"}},
	}
	kept := FilterSSDLiveCodeBenchV6Samples(in)
	if len(kept) != 2 || kept[0].ID != "mar" || kept[1].ID != "apr" {
		t.Fatalf("filtered = %+v, want the Mar/Apr in-window rows", kept)
	}
	// The survivors are clones — mutating the result must not touch the input.
	kept[0].Meta["contest_date"] = "mutated"
	if in[1].Meta["contest_date"] != "2025-03-15" {
		t.Fatal("FilterSSDLiveCodeBenchV6Samples returned an aliased meta map, want a clone")
	}
}

// TestFilterSSDLiveCodeBenchV6Samples_Bad asserts samples with no
// contest_date (or an out-of-window one) are all dropped, yielding an empty
// (non-nil) slice rather than passing them through.
func TestSsdEval_FilterSSDLiveCodeBenchV6Samples_Bad(t *testing.T) {
	in := []SSDCodeBenchmarkSample{
		{ID: "nodate"},
		{ID: "old", Meta: map[string]string{"contest_date": "2024-01-01"}},
		{ID: "future", Meta: map[string]string{"contest_date": "2026-01-01"}},
	}
	kept := FilterSSDLiveCodeBenchV6Samples(in)
	if len(kept) != 0 {
		t.Fatalf("filtered = %+v, want empty (no in-window rows)", kept)
	}
}

// TestFilterSSDLiveCodeBenchV6Samples_Ugly asserts the empty input case
// returns an empty result without panicking, and that a whitespace-padded
// contest_date is trimmed before the window comparison.
func TestSsdEval_FilterSSDLiveCodeBenchV6Samples_Ugly(t *testing.T) {
	if kept := FilterSSDLiveCodeBenchV6Samples(nil); len(kept) != 0 {
		t.Fatalf("filtered nil = %+v, want empty", kept)
	}
	in := []SSDCodeBenchmarkSample{{ID: "padded", Meta: map[string]string{"contest_date": "  2025-03-15  "}}}
	kept := FilterSSDLiveCodeBenchV6Samples(in)
	if len(kept) != 1 || kept[0].ID != "padded" {
		t.Fatalf("filtered = %+v, want the padded-date row kept after trim", kept)
	}
}

// --- FormatSSDLiveCodeBenchPrompt ---

// TestFormatSSDLiveCodeBenchPrompt_Good asserts both prompt shapes: a
// stdin task gets the stdin/stdout framing, and a function task with an entry
// point gets the starter-code framing.
func TestSsdEval_FormatSSDLiveCodeBenchPrompt_Good(t *testing.T) {
	stdinPrompt := FormatSSDLiveCodeBenchPrompt(SSDCodeBenchmarkSample{
		Prompt: "Add two numbers.",
		Meta:   map[string]string{"is_stdin": "true"},
	})
	if !strings.Contains(stdinPrompt, "stdin") || !strings.Contains(stdinPrompt, "Add two numbers.") {
		t.Fatalf("stdin prompt = %q", stdinPrompt)
	}
	functionPrompt := FormatSSDLiveCodeBenchPrompt(SSDCodeBenchmarkSample{
		Prompt: "Implement add.",
		Meta:   map[string]string{"is_stdin": "false", "entry_point": "def add(a, b):\n    pass"},
	})
	if !strings.Contains(functionPrompt, "Starter code") || !strings.Contains(functionPrompt, "def add") {
		t.Fatalf("function prompt = %q", functionPrompt)
	}
}

// TestFormatSSDLiveCodeBenchPrompt_Bad asserts an empty prompt yields an
// empty string — there is nothing to frame.
func TestSsdEval_FormatSSDLiveCodeBenchPrompt_Bad(t *testing.T) {
	if got := FormatSSDLiveCodeBenchPrompt(SSDCodeBenchmarkSample{Meta: map[string]string{"is_stdin": "true"}}); got != "" {
		t.Fatalf("empty-prompt format = %q, want empty string", got)
	}
	if got := FormatSSDLiveCodeBenchPrompt(SSDCodeBenchmarkSample{Prompt: "   "}); got != "" {
		t.Fatalf("whitespace-prompt format = %q, want empty string", got)
	}
}

// TestFormatSSDLiveCodeBenchPrompt_Ugly asserts the fallbacks: a nil
// meta and an is_stdin=false sample with no entry point both fall back to the
// default stdin/stdout framing rather than emitting a broken starter block.
func TestSsdEval_FormatSSDLiveCodeBenchPrompt_Ugly(t *testing.T) {
	noMeta := FormatSSDLiveCodeBenchPrompt(SSDCodeBenchmarkSample{Prompt: "Solve it."})
	if !strings.Contains(noMeta, "stdin") || !strings.Contains(noMeta, "Solve it.") {
		t.Fatalf("nil-meta prompt = %q, want default stdin framing", noMeta)
	}
	noEntry := FormatSSDLiveCodeBenchPrompt(SSDCodeBenchmarkSample{
		Prompt: "Solve it.",
		Meta:   map[string]string{"is_stdin": "false"},
	})
	if !strings.Contains(noEntry, "stdin") {
		t.Fatalf("no-entry-point prompt = %q, want default stdin framing fallback", noEntry)
	}
}

// --- SSDPostProcessCode ---

// TestSSDPostProcessCode_Good extracts the LAST fenced code block and
// applies the LiveCodeBench cleanup (strips <code> tags and the python fence
// marker), ignoring earlier non-final fences.
func TestSsdEval_SSDPostProcessCode_Good(t *testing.T) {
	response := "analysis\n```go\nnot this\n```\nfinal\n```python\n<code>def add(a, b):\n    return a + b</code>\n```\n"
	code, ok := SSDPostProcessCode(response)
	if !ok {
		t.Fatal("SSDPostProcessCode() ok = false")
	}
	if core.Trim(code) != "def add(a, b):\n    return a + b" {
		t.Fatalf("code = %q", code)
	}
}

// TestSSDPostProcessCode_Bad asserts a response with no fenced code
// returns ("", false) — there is no candidate to extract.
func TestSsdEval_SSDPostProcessCode_Bad(t *testing.T) {
	if code, ok := SSDPostProcessCode("no fenced code"); ok || code != "" {
		t.Fatalf("missing fence = %q/%t, want empty false", code, ok)
	}
	// An unterminated fence (no closing ```) is also not a valid block.
	if code, ok := SSDPostProcessCode("```python\ndef f(): pass"); ok || code != "" {
		t.Fatalf("unterminated fence = %q/%t, want empty false", code, ok)
	}
}

// TestSSDPostProcessCode_Ugly asserts an empty fenced block extracts as
// an empty-but-present body (ok=true): the fence existed, even if its content is
// blank after cleanup.
func TestSsdEval_SSDPostProcessCode_Ugly(t *testing.T) {
	code, ok := SSDPostProcessCode("```python\n```")
	if !ok {
		t.Fatal("SSDPostProcessCode(empty fence) ok = false, want true (the fence was present)")
	}
	if core.Trim(code) != "" {
		t.Fatalf("empty-fence code = %q, want empty body", code)
	}
}

// --- ssdCodeBenchmarkGeneratePrompt (private) ---

// TestSsdCodeBenchmarkGeneratePrompt_Good covers the arms: no Meta returns
// the bare prompt, Meta without is_stdin returns the bare prompt, and an
// is_stdin sample with a usable prompt returns the formatted variant.
func TestSsdCodeBenchmarkGeneratePrompt_Good(t *testing.T) {
	if got := ssdCodeBenchmarkGeneratePrompt(SSDCodeBenchmarkSample{Prompt: "bare"}); got != "bare" {
		t.Fatalf("no-Meta prompt = %q, want bare", got)
	}
	if got := ssdCodeBenchmarkGeneratePrompt(SSDCodeBenchmarkSample{Prompt: "bare", Meta: map[string]string{"x": "y"}}); got != "bare" {
		t.Fatalf("no-is_stdin prompt = %q, want bare", got)
	}
	formatted := ssdCodeBenchmarkGeneratePrompt(SSDCodeBenchmarkSample{Prompt: "solve me", Meta: map[string]string{"is_stdin": "true"}})
	if formatted == "solve me" || formatted == "" {
		t.Fatalf("is_stdin prompt = %q, want formatted variant", formatted)
	}
}

// TestSsdCodeBenchmarkGeneratePrompt_Bad asserts an is_stdin sample with a
// blank prompt falls back to the (blank) bare prompt rather than a broken
// formatted string.
func TestSsdCodeBenchmarkGeneratePrompt_Bad(t *testing.T) {
	if got := ssdCodeBenchmarkGeneratePrompt(SSDCodeBenchmarkSample{Prompt: "   ", Meta: map[string]string{"is_stdin": "true"}}); got != "   " {
		t.Fatalf("blank is_stdin prompt = %q, want the blank sample prompt fallback", got)
	}
}

// TestSsdCodeBenchmarkGeneratePrompt_Ugly asserts a nil Meta map and an
// empty Meta map both resolve identically to the bare-prompt path.
func TestSsdCodeBenchmarkGeneratePrompt_Ugly(t *testing.T) {
	nilMeta := ssdCodeBenchmarkGeneratePrompt(SSDCodeBenchmarkSample{Prompt: "bare"})
	emptyMeta := ssdCodeBenchmarkGeneratePrompt(SSDCodeBenchmarkSample{Prompt: "bare", Meta: map[string]string{}})
	if nilMeta != "bare" || emptyMeta != "bare" {
		t.Fatalf("nil/empty meta = %q/%q, want bare/bare", nilMeta, emptyMeta)
	}
}

// --- computeSSDCodeBenchmarkPassAtK (private) ---

// TestComputeSSDCodeBenchmarkPassAtK_Good computes pass@1 for a fully
// scored single-candidate-per-sample result set.
func TestComputeSSDCodeBenchmarkPassAtK_Good(t *testing.T) {
	one := []SSDCodeBenchmarkSampleResult{{
		Sample:     SSDCodeBenchmarkSample{ID: "a"},
		Candidates: []SSDCodeBenchmarkCandidateResult{{Execution: SSDCodeExecution{Passed: true}}},
	}}
	got := computeSSDCodeBenchmarkPassAtK(one, 1)
	if got["pass@1"] != 1 {
		t.Fatalf("pass@1 = %v, want 1", got["pass@1"])
	}
}

// TestComputeSSDCodeBenchmarkPassAtK_Bad asserts an empty result set
// returns nil, and a candidate-less sample is skipped rather than dividing
// by zero.
func TestComputeSSDCodeBenchmarkPassAtK_Bad(t *testing.T) {
	if got := computeSSDCodeBenchmarkPassAtK(nil, 1); got != nil {
		t.Fatalf("pass@k over no results = %v, want nil", got)
	}
	skipped := []SSDCodeBenchmarkSampleResult{{Sample: SSDCodeBenchmarkSample{ID: "a"}}}
	if got := computeSSDCodeBenchmarkPassAtK(skipped, 1); got != nil {
		t.Fatalf("pass@k over candidate-less sample = %v, want nil", got)
	}
}

// TestComputeSSDCodeBenchmarkPassAtK_Ugly asserts nRepeat 10 asks for
// pass@1 and pass@5, but a sample with only 1 candidate skips pass@5
// (total < k) while pass@1 is still computed.
func TestComputeSSDCodeBenchmarkPassAtK_Ugly(t *testing.T) {
	one := []SSDCodeBenchmarkSampleResult{{
		Sample:     SSDCodeBenchmarkSample{ID: "a"},
		Candidates: []SSDCodeBenchmarkCandidateResult{{Execution: SSDCodeExecution{Passed: true}}},
	}}
	got := computeSSDCodeBenchmarkPassAtK(one, 10)
	if _, ok := got["pass@1"]; !ok {
		t.Fatalf("pass@k = %v, want pass@1 present", got)
	}
	if _, ok := got["pass@5"]; ok {
		t.Fatalf("pass@k = %v, did not want pass@5 (one candidate < 5)", got)
	}
}

// --- computeSSDCodeBenchmarkDifficultyMetrics (private) ---

// TestComputeSSDCodeBenchmarkDifficultyMetrics_Good aggregates pass@1 for
// a single difficulty bucket.
func TestComputeSSDCodeBenchmarkDifficultyMetrics_Good(t *testing.T) {
	few := []SSDCodeBenchmarkSampleResult{{
		Sample: SSDCodeBenchmarkSample{ID: "a", Meta: map[string]string{"difficulty": "easy"}},
		Candidates: []SSDCodeBenchmarkCandidateResult{
			{Execution: SSDCodeExecution{Passed: true}},
		},
	}}
	got := computeSSDCodeBenchmarkDifficultyMetrics(few, 1)
	if got["pass@1_easy"] != 1 {
		t.Fatalf("difficulty = %v, want pass@1_easy=1", got)
	}
}

// TestComputeSSDCodeBenchmarkDifficultyMetrics_Bad asserts an empty
// k-list (nRepeat 0) returns nil, and samples without difficulty meta,
// without meta at all, or without candidates contribute nothing.
func TestComputeSSDCodeBenchmarkDifficultyMetrics_Bad(t *testing.T) {
	if got := computeSSDCodeBenchmarkDifficultyMetrics(nil, 0); got != nil {
		t.Fatalf("difficulty over empty k-list = %v, want nil", got)
	}
	results := []SSDCodeBenchmarkSampleResult{
		{Sample: SSDCodeBenchmarkSample{ID: "nometa"}},                                                // Meta nil → skip
		{Sample: SSDCodeBenchmarkSample{ID: "blank", Meta: map[string]string{"difficulty": " "}}},     // blank difficulty → skip
		{Sample: SSDCodeBenchmarkSample{ID: "nocand", Meta: map[string]string{"difficulty": "easy"}}}, // no candidates → skip
	}
	if got := computeSSDCodeBenchmarkDifficultyMetrics(results, 1); got != nil {
		t.Fatalf("difficulty over all-skipped results = %v, want nil", got)
	}
}

// TestComputeSSDCodeBenchmarkDifficultyMetrics_Ugly asserts nRepeat 10
// asks for pass@1 and pass@5, but a difficulty sample with only 3
// candidates skips pass@5 (total < k) while still emitting pass@1_easy.
func TestComputeSSDCodeBenchmarkDifficultyMetrics_Ugly(t *testing.T) {
	few := []SSDCodeBenchmarkSampleResult{{
		Sample: SSDCodeBenchmarkSample{ID: "a", Meta: map[string]string{"difficulty": "easy"}},
		Candidates: []SSDCodeBenchmarkCandidateResult{
			{Execution: SSDCodeExecution{Passed: true}},
			{Execution: SSDCodeExecution{}},
			{Execution: SSDCodeExecution{}},
		},
	}}
	got := computeSSDCodeBenchmarkDifficultyMetrics(few, 10)
	if _, ok := got["pass@1_easy"]; !ok {
		t.Fatalf("difficulty = %v, want pass@1_easy present", got)
	}
	if _, ok := got["pass@5_easy"]; ok {
		t.Fatalf("difficulty = %v, did not want pass@5_easy (3 candidates < 5)", got)
	}
}

// --- ssdCodeBenchmarkKList (private) ---

// TestSsdCodeBenchmarkKList pins the n-repeat thresholds that widen the k-list.
func TestSsdCodeBenchmarkKList(t *testing.T) {
	cases := []struct {
		nRepeat int
		want    []int
	}{
		{1, []int{1}},
		{10, []int{1, 5}},
		{20, []int{1, 5, 10}},
		{32, []int{1, 5, 10, 16}},
		{40, []int{1, 5, 10, 16, 20}},
		{64, []int{1, 5, 10, 16, 20, 32}},
	}
	for _, tc := range cases {
		got := ssdCodeBenchmarkKList(tc.nRepeat)
		if !slices.Equal(got, tc.want) {
			t.Fatalf("ssdCodeBenchmarkKList(%d) = %v, want %v", tc.nRepeat, got, tc.want)
		}
	}
}

// --- lastSSDCodeFence (private) ---

// TestLastSSDCodeFence_Good extracts a complete fenced block.
func TestLastSSDCodeFence_Good(t *testing.T) {
	if body, ok := lastSSDCodeFence("```python\nx=1\n```"); !ok || core.Trim(body) != "x=1" {
		t.Fatalf("closed fence = (%q, %v), want (x=1, true)", body, ok)
	}
}

// TestLastSSDCodeFence_Bad covers the break arms: a fence opener with no
// following newline, and an opened-but-never-closed fence both yield no
// extracted body.
func TestLastSSDCodeFence_Bad(t *testing.T) {
	// "```" with no following newline → break before a body is found.
	if body, ok := lastSSDCodeFence("text ``` still text"); ok || body != "" {
		t.Fatalf("unclosed-no-newline fence = (%q, %v), want (\"\", false)", body, ok)
	}
	// Opener + language line but no closing fence → break.
	if body, ok := lastSSDCodeFence("```python\ncode without a close"); ok || body != "" {
		t.Fatalf("never-closed fence = (%q, %v), want (\"\", false)", body, ok)
	}
}

// TestLastSSDCodeFence_Ugly asserts a response with no fence markers at
// all returns no extracted body.
func TestLastSSDCodeFence_Ugly(t *testing.T) {
	if body, ok := lastSSDCodeFence("no fences here"); ok || body != "" {
		t.Fatalf("no-fence input = (%q, %v), want (\"\", false)", body, ok)
	}
}

// --- writeSSDCodeBenchmarkReport (private) ---

// TestWriteSSDCodeBenchmarkReport_Good creates a fresh nested directory
// and lands the report file inside it.
func TestWriteSSDCodeBenchmarkReport_Good(t *testing.T) {
	report := &SSDCodeBenchmarkReport{Version: 1}
	ok := core.PathJoin(t.TempDir(), "nested", "report.json")
	if err := writeSSDCodeBenchmarkReport(ok, report); err != nil {
		t.Fatalf("writeReport nested success error = %v", err)
	}
	if !core.ReadFile(ok).OK {
		t.Fatal("writeReport success but file unreadable")
	}
}

// TestWriteSSDCodeBenchmarkReport_Bad covers the two failure arms: a
// directory that cannot be created (parent is a file) and a path whose
// final component is itself a directory (write fails).
func TestWriteSSDCodeBenchmarkReport_Bad(t *testing.T) {
	report := &SSDCodeBenchmarkReport{Version: 1}

	// MkdirAll fails: parent is a regular file.
	blocker := core.PathJoin(t.TempDir(), "afile")
	if w := core.WriteFile(blocker, []byte("x"), 0o600); !w.OK {
		t.Fatalf("setup blocker: %v", w.Value)
	}
	if err := writeSSDCodeBenchmarkReport(core.PathJoin(blocker, "sub", "r.json"), report); err == nil {
		t.Fatal("writeReport under a file-parent = nil, want MkdirAll failure")
	}

	// WriteFile fails: the target path is itself an existing directory.
	dirAsFile := core.PathJoin(t.TempDir(), "report.json")
	if r := core.MkdirAll(dirAsFile, 0o755); !r.OK {
		t.Fatalf("setup dir-as-file: %v", r.Value)
	}
	if err := writeSSDCodeBenchmarkReport(dirAsFile, report); err == nil {
		t.Fatal("writeReport onto a directory path = nil, want WriteFile failure")
	}
}

// TestWriteSSDCodeBenchmarkReport_Ugly asserts a plain top-level path (no
// nested directory to create) still succeeds.
func TestWriteSSDCodeBenchmarkReport_Ugly(t *testing.T) {
	report := &SSDCodeBenchmarkReport{Version: 1}
	path := core.PathJoin(t.TempDir(), "report.json")
	if err := writeSSDCodeBenchmarkReport(path, report); err != nil {
		t.Fatalf("writeReport top-level error = %v", err)
	}
}

// --- nonZeroSSDCodeBenchmarkDuration (private) ---

// TestNonZeroSSDCodeBenchmarkDuration_Good passes a positive duration
// through unchanged.
func TestNonZeroSSDCodeBenchmarkDuration_Good(t *testing.T) {
	if got := nonZeroSSDCodeBenchmarkDuration(42 * time.Millisecond); got != 42*time.Millisecond {
		t.Fatalf("nonZeroDuration(42ms) = %v, want 42ms", got)
	}
}

// TestNonZeroSSDCodeBenchmarkDuration_Bad asserts a zero measured duration
// is floored to a single nanosecond rather than reporting a false zero.
func TestNonZeroSSDCodeBenchmarkDuration_Bad(t *testing.T) {
	if got := nonZeroSSDCodeBenchmarkDuration(0); got != time.Nanosecond {
		t.Fatalf("nonZeroDuration(0) = %v, want 1ns", got)
	}
}

// TestNonZeroSSDCodeBenchmarkDuration_Ugly asserts a negative duration
// (should never occur from time.Since, but defensively handled) also
// floors to one nanosecond.
func TestNonZeroSSDCodeBenchmarkDuration_Ugly(t *testing.T) {
	if got := nonZeroSSDCodeBenchmarkDuration(-5); got != time.Nanosecond {
		t.Fatalf("nonZeroDuration(-5) = %v, want 1ns", got)
	}
}
