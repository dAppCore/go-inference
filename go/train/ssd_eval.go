// SPDX-Licence-Identifier: EUPL-1.2

package train

import (
	"context"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// ssd_eval.go: the native SSD code-benchmark harness (sample candidate
// solutions from an injected Generate hook, execute them against an
// injected RunTests hook, aggregate pass@k) — hooks-based and model-free,
// same shape as RunSSDCodeBenchmark's go-mlx counterpart. Ported verbatim
// from go-mlx/go/train/ssd_eval.go: aside from these three sampling
// defaults (mirrored from go-mlx's ssd.go, which this package does not
// otherwise port — see the package doc) and the spine.GenerateConfig ->
// inference.GenerateConfig substitution, every symbol below is unchanged.
const (
	defaultSSDMaxTokens = 256
	defaultSSDTopK      = 64
	defaultSSDTopP      = 0.95
)

// SSDCodeBenchmarkConfig configures native code-generation
// benchmark runs such as LiveCodeBench-v6.
type SSDCodeBenchmarkConfig struct {
	Benchmark  string                   `json:"benchmark,omitempty"`
	NRepeat    int                      `json:"n_repeat,omitempty"`
	Generate   inference.GenerateConfig `json:"generate"`
	Seeds      []uint64                 `json:"seeds,omitempty"`
	OutputPath string                   `json:"output_path,omitempty"`
}

// SSDCodeBenchmarkRunner supplies generation and native
// code-execution test evaluation for each candidate.
type SSDCodeBenchmarkRunner struct {
	Generate func(context.Context, string, inference.GenerateConfig) (string, error)
	RunTests func(context.Context, SSDCodeBenchmarkSample, SSDCodeCandidate) (SSDCodeExecution, error)
}

// SSDCodeBenchmarkSample is one code benchmark task.
type SSDCodeBenchmarkSample struct {
	ID     string            `json:"id,omitempty"`
	Prompt string            `json:"prompt"`
	Tests  []string          `json:"tests,omitempty"`
	Meta   map[string]string `json:"meta,omitempty"`
}

// SSDCodeCandidate records one generated solution.
type SSDCodeCandidate struct {
	Repeat      int                      `json:"repeat"`
	Solution    string                   `json:"solution"`
	RawSolution string                   `json:"raw_solution,omitempty"`
	HasCode     bool                     `json:"has_code,omitempty"`
	Config      inference.GenerateConfig `json:"config"`
}

// SSDCodeExecution records the code-test outcome for one
// generated solution.
type SSDCodeExecution struct {
	Passed      bool          `json:"passed"`
	PassedTests int           `json:"passed_tests,omitempty"`
	TotalTests  int           `json:"total_tests,omitempty"`
	Duration    time.Duration `json:"duration,omitempty"`
	DurationMS  int64         `json:"duration_ms,omitempty"`
	Stdout      string        `json:"stdout,omitempty"`
	Stderr      string        `json:"stderr,omitempty"`
	Error       string        `json:"error,omitempty"`
}

// SSDCodeBenchmarkCandidateResult joins a candidate with
// its native code-test execution result.
type SSDCodeBenchmarkCandidateResult struct {
	Candidate SSDCodeCandidate `json:"candidate"`
	Execution SSDCodeExecution `json:"execution"`
}

// SSDCodeBenchmarkSampleResult records all candidates for
// one benchmark task.
type SSDCodeBenchmarkSampleResult struct {
	Sample     SSDCodeBenchmarkSample            `json:"sample"`
	Candidates []SSDCodeBenchmarkCandidateResult `json:"candidates"`
}

// SSDCodeBenchmarkMetrics aggregates benchmark pass rates.
type SSDCodeBenchmarkMetrics struct {
	Samples    int                `json:"samples,omitempty"`
	Candidates int                `json:"candidates,omitempty"`
	Passed     int                `json:"passed,omitempty"`
	Failed     int                `json:"failed,omitempty"`
	PassRate   float64            `json:"pass_rate,omitempty"`
	PassAtK    map[string]float64 `json:"pass_at_k,omitempty"`
	Difficulty map[string]float64 `json:"difficulty,omitempty"`
}

// SSDCodeBenchmarkReport is the JSON-serialisable output of
// a native SSD code benchmark run.
type SSDCodeBenchmarkReport struct {
	Version   int                            `json:"version"`
	Benchmark string                         `json:"benchmark,omitempty"`
	Config    SSDCodeBenchmarkConfig         `json:"config"`
	Metrics   SSDCodeBenchmarkMetrics        `json:"metrics"`
	Results   []SSDCodeBenchmarkSampleResult `json:"results"`
	Duration  time.Duration                  `json:"duration,omitempty"`
}

type ssdCodeBenchmarkJSONLRecord struct {
	ID               string            `json:"id"`
	QuestionID       string            `json:"question_id"`
	TaskID           string            `json:"task_id"`
	Prompt           string            `json:"prompt"`
	Question         string            `json:"question"`
	QuestionContent  string            `json:"question_content"`
	Problem          string            `json:"problem"`
	StarterCode      string            `json:"starter_code"`
	EntryPoint       string            `json:"entry_point"`
	IsStdin          *bool             `json:"is_stdin"`
	ContestDate      string            `json:"contest_date"`
	Test             string            `json:"test"`
	Tests            []string          `json:"tests"`
	PublicTestCases  []string          `json:"public_test_cases"`
	PrivateTestCases []string          `json:"private_test_cases"`
	Metadata         map[string]string `json:"metadata"`
	Difficulty       string            `json:"difficulty"`
	Platform         string            `json:"platform"`
}

// LoadSSDCodeBenchmarkJSONLFile loads benchmark tasks from
// a JSONL file path.
func LoadSSDCodeBenchmarkJSONLFile(path string) ([]SSDCodeBenchmarkSample, error) {
	read := core.ReadFile(path)
	if !read.OK {
		return nil, read.Value.(error)
	}
	return LoadSSDCodeBenchmarkJSONL(core.AsString(read.Value.([]byte)))
}

// LoadSSDLiveCodeBenchV6JSONLFile loads the LiveCodeBench-v6
// task subset from a JSONL file path.
func LoadSSDLiveCodeBenchV6JSONLFile(path string) ([]SSDCodeBenchmarkSample, error) {
	read := core.ReadFile(path)
	if !read.OK {
		return nil, read.Value.(error)
	}
	return LoadSSDLiveCodeBenchV6JSONL(core.AsString(read.Value.([]byte)))
}

// LoadSSDCodeBenchmarkJSONL loads LiveCodeBench-style JSONL
// task rows into native SSD code benchmark samples.
func LoadSSDCodeBenchmarkJSONL(raw string) ([]SSDCodeBenchmarkSample, error) {
	lines := core.Split(raw, "\n")
	samples := make([]SSDCodeBenchmarkSample, 0, len(lines))
	for index, line := range lines {
		line = core.Trim(line)
		if line == "" {
			continue
		}
		var record ssdCodeBenchmarkJSONLRecord
		if result := core.JSONUnmarshalString(line, &record); !result.OK {
			return nil, core.Errorf("mlx: parse SSD code benchmark JSONL record %d: %w", index+1, result.Value.(error))
		}
		sample, ok := record.ssdCodeBenchmarkSample()
		if !ok {
			continue
		}
		samples = append(samples, sample)
	}
	if len(samples) == 0 {
		return nil, core.NewError("mlx: SSD code benchmark JSONL produced no samples")
	}
	return samples, nil
}

// LoadSSDLiveCodeBenchV6JSONL loads LiveCodeBench-style
// JSONL and filters it to the v6 contest-date window.
func LoadSSDLiveCodeBenchV6JSONL(raw string) ([]SSDCodeBenchmarkSample, error) {
	samples, err := LoadSSDCodeBenchmarkJSONL(raw)
	if err != nil {
		return nil, err
	}
	samples = FilterSSDLiveCodeBenchV6Samples(samples)
	if len(samples) == 0 {
		return nil, core.NewError("mlx: LiveCodeBench-v6 JSONL produced no samples")
	}
	return samples, nil
}

// FilterSSDLiveCodeBenchV6Samples keeps samples from the
// LiveCodeBench-v6 contest-date window.
func FilterSSDLiveCodeBenchV6Samples(samples []SSDCodeBenchmarkSample) []SSDCodeBenchmarkSample {
	filtered := make([]SSDCodeBenchmarkSample, 0, len(samples))
	for _, sample := range samples {
		if ssdLiveCodeBenchV6ContestDate(sample.Meta["contest_date"]) {
			filtered = append(filtered, cloneSSDCodeBenchmarkSample(sample))
		}
	}
	return filtered
}

func ssdLiveCodeBenchV6ContestDate(date string) bool {
	date = core.Trim(date)
	return date >= "2025-02-01" && date < "2025-06-01"
}

func (r ssdCodeBenchmarkJSONLRecord) ssdCodeBenchmarkSample() (SSDCodeBenchmarkSample, bool) {
	prompt := firstSSDCodeBenchmarkString(r.Prompt, r.QuestionContent, r.Question, r.Problem)
	if prompt == "" {
		return SSDCodeBenchmarkSample{}, false
	}
	if starterCode := core.Trim(r.StarterCode); starterCode != "" {
		prompt = core.Concat(prompt, "\n\nstarter code:\n", starterCode)
	}
	tests := appendSSDCodeBenchmarkTests(nil, r.Tests...)
	tests = appendSSDCodeBenchmarkTests(tests, r.Test)
	tests = appendSSDCodeBenchmarkTests(tests, r.PublicTestCases...)
	tests = appendSSDCodeBenchmarkTests(tests, r.PrivateTestCases...)
	meta := core.MapClone(r.Metadata)
	if meta == nil {
		meta = make(map[string]string, 2)
	}
	if difficulty := core.Trim(r.Difficulty); difficulty != "" {
		meta["difficulty"] = difficulty
	}
	if platform := core.Trim(r.Platform); platform != "" {
		meta["platform"] = platform
	}
	if entryPoint := core.Trim(r.EntryPoint); entryPoint != "" {
		meta["entry_point"] = entryPoint
	}
	if contestDate := core.Trim(r.ContestDate); contestDate != "" {
		meta["contest_date"] = contestDate
	}
	if r.IsStdin != nil {
		meta["is_stdin"] = core.Sprintf("%t", *r.IsStdin)
	}
	if len(meta) == 0 {
		meta = nil
	}
	return SSDCodeBenchmarkSample{
		ID:     firstSSDCodeBenchmarkString(r.ID, r.QuestionID, r.TaskID),
		Prompt: prompt,
		Tests:  tests,
		Meta:   meta,
	}, true
}

// RunSSDCodeBenchmark samples candidate code solutions and
// delegates native execution of each candidate against the sample tests.
func RunSSDCodeBenchmark(ctx context.Context, runner SSDCodeBenchmarkRunner, samples []SSDCodeBenchmarkSample, cfg SSDCodeBenchmarkConfig) (*SSDCodeBenchmarkReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if runner.Generate == nil {
		return nil, core.NewError("mlx: SSD code benchmark generate function is nil")
	}
	if runner.RunTests == nil {
		return nil, core.NewError("mlx: SSD code benchmark RunTests function is nil")
	}
	cfg = normalizeSSDCodeBenchmarkConfig(cfg)
	if len(samples) == 0 {
		return nil, core.NewError("mlx: SSD code benchmark samples are empty")
	}

	start := time.Now()
	report := &SSDCodeBenchmarkReport{
		Version:   1,
		Benchmark: cfg.Benchmark,
		Config:    cfg,
		Results:   make([]SSDCodeBenchmarkSampleResult, 0, len(samples)),
	}
	for _, sample := range samples {
		if err := ctx.Err(); err != nil {
			return report, err
		}
		sampleResult := SSDCodeBenchmarkSampleResult{
			Sample:     cloneSSDCodeBenchmarkSample(sample),
			Candidates: make([]SSDCodeBenchmarkCandidateResult, 0, cfg.NRepeat),
		}
		for repeat := 0; repeat < cfg.NRepeat; repeat++ {
			if err := ctx.Err(); err != nil {
				return report, err
			}
			prompt := ssdCodeBenchmarkGeneratePrompt(sample)
			generateCfg := ssdCodeBenchmarkRepeatGenerateConfig(cfg, repeat)
			rawSolution, err := runner.Generate(ctx, prompt, generateCfg)
			if err != nil {
				return report, err
			}
			solution, hasCode := SSDPostProcessCode(rawSolution)
			candidate := SSDCodeCandidate{
				Repeat:      repeat,
				Solution:    solution,
				RawSolution: rawSolution,
				HasCode:     hasCode,
				Config:      generateCfg,
			}
			execution, err := runner.RunTests(ctx, sample, candidate)
			if err != nil {
				return report, err
			}
			sampleResult.Candidates = append(sampleResult.Candidates, SSDCodeBenchmarkCandidateResult{
				Candidate: candidate,
				Execution: execution,
			})
			report.Metrics.Candidates++
			if execution.Passed {
				report.Metrics.Passed++
			}
		}
		report.Results = append(report.Results, sampleResult)
	}
	report.Metrics.Samples = len(samples)
	report.Metrics.Failed = report.Metrics.Candidates - report.Metrics.Passed
	if report.Metrics.Candidates > 0 {
		report.Metrics.PassRate = float64(report.Metrics.Passed) / float64(report.Metrics.Candidates)
	}
	report.Metrics.PassAtK = computeSSDCodeBenchmarkPassAtK(report.Results, cfg.NRepeat)
	report.Metrics.Difficulty = computeSSDCodeBenchmarkDifficultyMetrics(report.Results, cfg.NRepeat)
	report.Duration = nonZeroSSDCodeBenchmarkDuration(time.Since(start))
	if cfg.OutputPath != "" {
		if err := writeSSDCodeBenchmarkReport(cfg.OutputPath, report); err != nil {
			return report, err
		}
	}
	return report, nil
}

// SSDPostProcessCode extracts the final fenced code block
// from a model response and applies the LiveCodeBench code cleanup.
func SSDPostProcessCode(response string) (string, bool) {
	code, ok := lastSSDCodeFence(response)
	if !ok {
		return "", false
	}
	return ssdPostProcessCode(code), true
}

// FormatSSDLiveCodeBenchPrompt returns the native prompt
// shape used for LiveCodeBench-v6-style code-generation tasks.
func FormatSSDLiveCodeBenchPrompt(sample SSDCodeBenchmarkSample) string {
	prompt := core.Trim(sample.Prompt)
	if prompt == "" {
		return ""
	}
	if sample.Meta != nil && sample.Meta["is_stdin"] == "false" {
		if entryPoint := core.Trim(sample.Meta["entry_point"]); entryPoint != "" {
			return core.Concat(
				"Write a Python solution for the problem. Return only the program inside one python code block.\n\nProblem:\n",
				prompt,
				"\n\nStarter code:\n```python\n",
				entryPoint,
				"\n```",
			)
		}
	}
	return core.Concat(
		"Write a Python program for the problem. Read from stdin, write to stdout, and return only the program inside one python code block.\n\nProblem:\n",
		prompt,
	)
}

func ssdCodeBenchmarkGeneratePrompt(sample SSDCodeBenchmarkSample) string {
	if sample.Meta == nil {
		return sample.Prompt
	}
	if _, ok := sample.Meta["is_stdin"]; !ok {
		return sample.Prompt
	}
	if prompt := FormatSSDLiveCodeBenchPrompt(sample); prompt != "" {
		return prompt
	}
	return sample.Prompt
}

func ssdCodeBenchmarkRepeatGenerateConfig(cfg SSDCodeBenchmarkConfig, repeat int) inference.GenerateConfig {
	generate := cfg.Generate
	if len(cfg.Seeds) > 0 {
		generate.Seed = cfg.Seeds[0] + uint64(repeat)
		generate.SeedSet = true
	}
	return generate
}

func normalizeSSDCodeBenchmarkConfig(cfg SSDCodeBenchmarkConfig) SSDCodeBenchmarkConfig {
	if cfg.NRepeat <= 0 {
		cfg.NRepeat = 1
	}
	if cfg.Generate.MaxTokens <= 0 {
		cfg.Generate.MaxTokens = defaultSSDMaxTokens
	}
	if cfg.Generate.TopK == 0 {
		cfg.Generate.TopK = defaultSSDTopK
	}
	if cfg.Generate.TopP == 0 {
		cfg.Generate.TopP = defaultSSDTopP
	}
	return cfg
}

func computeSSDCodeBenchmarkPassAtK(results []SSDCodeBenchmarkSampleResult, nRepeat int) map[string]float64 {
	kList := ssdCodeBenchmarkKList(nRepeat)
	if len(kList) == 0 || len(results) == 0 {
		return nil
	}
	sums := make(map[string]float64, len(kList))
	counts := make(map[string]int, len(kList))
	for _, result := range results {
		total := len(result.Candidates)
		if total == 0 {
			continue
		}
		correct := 0
		for _, candidate := range result.Candidates {
			if candidate.Execution.Passed {
				correct++
			}
		}
		for _, k := range kList {
			if total < k {
				continue
			}
			key := core.Sprintf("pass@%d", k)
			sums[key] += estimateSSDCodeBenchmarkPassAtK(total, correct, k)
			counts[key]++
		}
	}
	out := make(map[string]float64, len(sums))
	for key, sum := range sums {
		if counts[key] > 0 {
			out[key] = sum / float64(counts[key])
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func computeSSDCodeBenchmarkDifficultyMetrics(results []SSDCodeBenchmarkSampleResult, nRepeat int) map[string]float64 {
	kList := ssdCodeBenchmarkKList(nRepeat)
	if len(kList) == 0 {
		return nil
	}
	type bucket struct {
		sum   float64
		count int
	}
	buckets := make(map[string]bucket)
	for _, result := range results {
		if result.Sample.Meta == nil {
			continue
		}
		difficulty := core.Trim(result.Sample.Meta["difficulty"])
		if difficulty == "" {
			continue
		}
		total := len(result.Candidates)
		if total == 0 {
			continue
		}
		correct := 0
		for _, candidate := range result.Candidates {
			if candidate.Execution.Passed {
				correct++
			}
		}
		for _, k := range kList {
			if total < k {
				continue
			}
			key := core.Sprintf("pass@%d_%s", k, difficulty)
			value := buckets[key]
			value.sum += estimateSSDCodeBenchmarkPassAtK(total, correct, k)
			value.count++
			buckets[key] = value
		}
	}
	out := make(map[string]float64, len(buckets))
	for key, bucket := range buckets {
		if bucket.count > 0 {
			out[key] = bucket.sum / float64(bucket.count)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func ssdCodeBenchmarkKList(nRepeat int) []int {
	kList := []int{1}
	if nRepeat >= 10 {
		kList = append(kList, 5)
	}
	if nRepeat >= 20 {
		kList = append(kList, 10)
	}
	if nRepeat >= 32 {
		kList = append(kList, 16)
	}
	if nRepeat >= 40 {
		kList = append(kList, 20)
	}
	if nRepeat >= 64 {
		kList = append(kList, 32)
	}
	return kList
}

func estimateSSDCodeBenchmarkPassAtK(total, correct, k int) float64 {
	if total <= 0 || correct <= 0 || k <= 0 {
		return 0
	}
	if total-correct < k {
		return 1
	}
	fail := 1.0
	for i := range k {
		fail *= float64(total-correct-i) / float64(total-i)
	}
	return 1 - fail
}

func cloneSSDCodeBenchmarkSample(sample SSDCodeBenchmarkSample) SSDCodeBenchmarkSample {
	return SSDCodeBenchmarkSample{
		ID:     sample.ID,
		Prompt: sample.Prompt,
		Tests:  core.SliceClone(sample.Tests),
		Meta:   core.MapClone(sample.Meta),
	}
}

func firstSSDCodeBenchmarkString(values ...string) string {
	for _, value := range values {
		if trimmed := core.Trim(value); trimmed != "" {
			return trimmed
		}
	}
	return ""
}

func appendSSDCodeBenchmarkTests(target []string, values ...string) []string {
	for _, value := range values {
		if trimmed := core.Trim(value); trimmed != "" {
			target = append(target, trimmed)
		}
	}
	return target
}

func lastSSDCodeFence(response string) (string, bool) {
	var last string
	found := false
	remaining := response
	for {
		start := core.Index(remaining, "```")
		if start < 0 {
			break
		}
		afterStart := remaining[start+3:]
		newline := core.Index(afterStart, "\n")
		if newline < 0 {
			break
		}
		bodyStart := newline + 1
		afterLanguage := afterStart[bodyStart:]
		end := core.Index(afterLanguage, "```")
		if end < 0 {
			break
		}
		last = afterLanguage[:end]
		found = true
		remaining = afterLanguage[end+3:]
	}
	return last, found
}

func ssdPostProcessCode(code string) string {
	code = firstSSDSegment(code, "</code>")
	code = core.Replace(code, "```python", "")
	code = firstSSDSegment(code, "```")
	code = core.Replace(code, "<code>", "")
	return code
}

func firstSSDSegment(value, delimiter string) string {
	if index := core.Index(value, delimiter); index >= 0 {
		return value[:index]
	}
	return value
}

func writeSSDCodeBenchmarkReport(path string, report *SSDCodeBenchmarkReport) error {
	data := core.JSONMarshalIndent(report, "", "  ")
	if !data.OK {
		return data.Value.(error)
	}
	dir := core.PathDir(path)
	if dir != "" && dir != "." {
		if result := core.MkdirAll(dir, 0o755); !result.OK {
			return result.Value.(error)
		}
	}
	if result := core.WriteFile(path, data.Value.([]byte), 0o644); !result.OK {
		return result.Value.(error)
	}
	return nil
}

func nonZeroSSDCodeBenchmarkDuration(value time.Duration) time.Duration {
	if value <= 0 {
		return time.Nanosecond
	}
	return value
}
