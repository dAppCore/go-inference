// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"bufio"
	"bytes"
	"encoding/json"
	"os"
	"strings"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

const (
	SimpleSelfDistillationRecipe4BInstruct     = "SimpleSD-4B-instruct"
	SimpleSelfDistillationRecipe4BThinking     = "SimpleSD-4B-thinking"
	SimpleSelfDistillationRecipe30BA3BInstruct = "SimpleSD-30b-a3b-instruct"
)

type SimpleSelfDistillationRecipe struct {
	Name          string                                    `json:"name"`
	Model         string                                    `json:"model"`
	Dataset       string                                    `json:"dataset,omitempty"`
	DatasetConfig string                                    `json:"dataset_config,omitempty"`
	DatasetSplit  string                                    `json:"dataset_split,omitempty"`
	Train         SimpleSelfDistillationConfig              `json:"train"`
	Eval          SimpleSelfDistillationCodeBenchmarkConfig `json:"eval"`
	Notes         []string                                  `json:"notes,omitempty"`
}

type SimpleSelfDistillationCodeBenchmarkConfig struct {
	Benchmark  string                   `json:"benchmark,omitempty"`
	NRepeat    int                      `json:"n_repeat,omitempty"`
	Generate   inference.GenerateConfig `json:"generate"`
	Seeds      []uint64                 `json:"seeds,omitempty"`
	OutputPath string                   `json:"output_path,omitempty"`
}

type SimpleSelfDistillationCodeBenchmarkSample struct {
	ID     string            `json:"id,omitempty"`
	Prompt string            `json:"prompt"`
	Tests  []string          `json:"tests,omitempty"`
	Meta   map[string]string `json:"meta,omitempty"`
}

type simpleSelfDistillationCodeBenchmarkJSONLRecord struct {
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

func DefaultSimpleSelfDistillationConfig() SimpleSelfDistillationConfig {
	return SimpleSelfDistillationConfig{
		SampleMaxTokens:   defaultSimpleSelfDistillationMaxTokens,
		SampleTemperature: defaultSimpleSelfDistillationTemperature,
		SampleTopK:        defaultSimpleSelfDistillationTopK,
		SampleTopP:        defaultSimpleSelfDistillationTopP,
		RepetitionPenalty: defaultSimpleSelfDistillationRepetition,
		FilterShortestPct: defaultSimpleSelfDistillationFilterShortest,
	}
}

func DefaultSimpleSelfDistillationCodeBenchmarkConfig() SimpleSelfDistillationCodeBenchmarkConfig {
	return SimpleSelfDistillationCodeBenchmarkConfig{
		Benchmark: "LiveCodeBench-v6",
		NRepeat:   20,
		Seeds:     []uint64{0, 1234, 1234, 1234},
		Generate: inference.GenerateConfig{
			MaxTokens:   defaultSimpleSelfDistillationEvalMaxTokens,
			Temperature: defaultSimpleSelfDistillationEvalTemperature,
			TopP:        defaultSimpleSelfDistillationEvalTopP,
			TopK:        defaultSimpleSelfDistillationTopK,
		},
	}
}

func SimpleSelfDistillationRecipes() []SimpleSelfDistillationRecipe {
	train := DefaultSimpleSelfDistillationConfig()
	eval := DefaultSimpleSelfDistillationCodeBenchmarkConfig()
	return []SimpleSelfDistillationRecipe{
		newSimpleSelfDistillationRecipe(SimpleSelfDistillationRecipe4BInstruct, "apple/SimpleSD-4B-instruct", train, eval),
		newSimpleSelfDistillationRecipe(SimpleSelfDistillationRecipe4BThinking, "apple/SimpleSD-4B-thinking", train, eval),
		newSimpleSelfDistillationRecipe(SimpleSelfDistillationRecipe30BA3BInstruct, "apple/SimpleSD-30b-a3b-instruct", train, eval),
	}
}

func LookupSimpleSelfDistillationRecipe(name string) (SimpleSelfDistillationRecipe, bool) {
	for _, recipe := range SimpleSelfDistillationRecipes() {
		if recipe.Name == name || recipe.Model == name {
			return recipe, true
		}
	}
	return SimpleSelfDistillationRecipe{}, false
}

func LoadSimpleSelfDistillationCodeBenchmarkJSONLFile(path string) ([]SimpleSelfDistillationCodeBenchmarkSample, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return LoadSimpleSelfDistillationCodeBenchmarkJSONL(data)
}

func LoadSimpleSelfDistillationLiveCodeBenchV6JSONLFile(path string) ([]SimpleSelfDistillationCodeBenchmarkSample, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return LoadSimpleSelfDistillationLiveCodeBenchV6JSONL(data)
}

func LoadSimpleSelfDistillationCodeBenchmarkJSONL(raw []byte) ([]SimpleSelfDistillationCodeBenchmarkSample, error) {
	scanner := bufio.NewScanner(bytes.NewReader(raw))
	scanner.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	samples := make([]SimpleSelfDistillationCodeBenchmarkSample, 0, bytes.Count(raw, []byte{'\n'})+1)
	for index := 1; scanner.Scan(); index++ {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var record simpleSelfDistillationCodeBenchmarkJSONLRecord
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			return nil, core.Errorf("rocm: parse SSD code benchmark JSONL record %d: %w", index, err)
		}
		sample, ok := record.simpleSelfDistillationCodeBenchmarkSample()
		if !ok {
			continue
		}
		samples = append(samples, sample)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(samples) == 0 {
		return nil, core.NewError("rocm: SSD code benchmark JSONL produced no samples")
	}
	return samples, nil
}

func LoadSimpleSelfDistillationLiveCodeBenchV6JSONL(raw []byte) ([]SimpleSelfDistillationCodeBenchmarkSample, error) {
	samples, err := LoadSimpleSelfDistillationCodeBenchmarkJSONL(raw)
	if err != nil {
		return nil, err
	}
	samples = FilterSimpleSelfDistillationLiveCodeBenchV6Samples(samples)
	if len(samples) == 0 {
		return nil, core.NewError("rocm: LiveCodeBench-v6 JSONL produced no samples")
	}
	return samples, nil
}

func FilterSimpleSelfDistillationLiveCodeBenchV6Samples(samples []SimpleSelfDistillationCodeBenchmarkSample) []SimpleSelfDistillationCodeBenchmarkSample {
	filtered := make([]SimpleSelfDistillationCodeBenchmarkSample, 0, len(samples))
	for _, sample := range samples {
		if simpleSelfDistillationLiveCodeBenchV6ContestDate(sample.Meta["contest_date"]) {
			filtered = append(filtered, cloneSimpleSelfDistillationCodeBenchmarkSample(sample))
		}
	}
	return filtered
}

func newSimpleSelfDistillationRecipe(name, model string, train SimpleSelfDistillationConfig, eval SimpleSelfDistillationCodeBenchmarkConfig) SimpleSelfDistillationRecipe {
	return SimpleSelfDistillationRecipe{
		Name:          name,
		Model:         model,
		Dataset:       "microsoft/rStar-Coder",
		DatasetConfig: "seed_sft",
		DatasetSplit:  "train",
		Train:         train,
		Eval:          eval,
		Notes: []string{
			"Use the released model card for model-specific decode sampling when it differs from the upstream eval example.",
			"Store runtime artifacts under docs/runtime/ when reproducing this recipe locally.",
		},
	}
}

func simpleSelfDistillationLiveCodeBenchV6ContestDate(date string) bool {
	date = strings.TrimSpace(date)
	return date >= "2025-02-01" && date < "2025-06-01"
}

func (record simpleSelfDistillationCodeBenchmarkJSONLRecord) simpleSelfDistillationCodeBenchmarkSample() (SimpleSelfDistillationCodeBenchmarkSample, bool) {
	prompt := firstSimpleSelfDistillationCodeBenchmarkString(record.Prompt, record.QuestionContent, record.Question, record.Problem)
	if prompt == "" {
		return SimpleSelfDistillationCodeBenchmarkSample{}, false
	}
	if starterCode := strings.TrimSpace(record.StarterCode); starterCode != "" {
		prompt += "\n\nstarter code:\n" + starterCode
	}
	tests := appendSimpleSelfDistillationCodeBenchmarkTests(nil, record.Tests...)
	tests = appendSimpleSelfDistillationCodeBenchmarkTests(tests, record.Test)
	tests = appendSimpleSelfDistillationCodeBenchmarkTests(tests, record.PublicTestCases...)
	tests = appendSimpleSelfDistillationCodeBenchmarkTests(tests, record.PrivateTestCases...)
	meta := rocmCloneLabels(record.Metadata)
	if meta == nil {
		meta = make(map[string]string, 2)
	}
	if difficulty := strings.TrimSpace(record.Difficulty); difficulty != "" {
		meta["difficulty"] = difficulty
	}
	if platform := strings.TrimSpace(record.Platform); platform != "" {
		meta["platform"] = platform
	}
	if entryPoint := strings.TrimSpace(record.EntryPoint); entryPoint != "" {
		meta["entry_point"] = entryPoint
	}
	if contestDate := strings.TrimSpace(record.ContestDate); contestDate != "" {
		meta["contest_date"] = contestDate
	}
	if record.IsStdin != nil {
		meta["is_stdin"] = core.Sprintf("%t", *record.IsStdin)
	}
	if len(meta) == 0 {
		meta = nil
	}
	return SimpleSelfDistillationCodeBenchmarkSample{
		ID:     firstSimpleSelfDistillationCodeBenchmarkString(record.ID, record.QuestionID, record.TaskID),
		Prompt: prompt,
		Tests:  tests,
		Meta:   meta,
	}, true
}

func cloneSimpleSelfDistillationCodeBenchmarkSample(sample SimpleSelfDistillationCodeBenchmarkSample) SimpleSelfDistillationCodeBenchmarkSample {
	return SimpleSelfDistillationCodeBenchmarkSample{
		ID:     sample.ID,
		Prompt: sample.Prompt,
		Tests:  append([]string(nil), sample.Tests...),
		Meta:   rocmCloneLabels(sample.Meta),
	}
}

func firstSimpleSelfDistillationCodeBenchmarkString(values ...string) string {
	for _, value := range values {
		if trimmed := strings.TrimSpace(value); trimmed != "" {
			return trimmed
		}
	}
	return ""
}

func appendSimpleSelfDistillationCodeBenchmarkTests(target []string, values ...string) []string {
	for _, value := range values {
		if trimmed := strings.TrimSpace(value); trimmed != "" {
			target = append(target, trimmed)
		}
	}
	return target
}
