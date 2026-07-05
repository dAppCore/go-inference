// SPDX-Licence-Identifier: EUPL-1.2

package modelmgmt

import (
	"dappco.re/go"
	"dappco.re/go/inference/score"
	coreio "dappco.re/go/io"
)

// ---------------------------------------------------------------------------
// ValidatePercentages
// ---------------------------------------------------------------------------

func TestExportValidatePercentagesAcceptsValidSplitsScenario(t *core.T) {
	assertResultOK(t, ValidatePercentages(80, 10, 10))
	assertResultOK(t, ValidatePercentages(100, 0, 0))
	assertResultOK(t, ValidatePercentages(0, 0, 100))
}

func TestExportValidatePercentagesWrongSumBadScenario(t *core.T) {
	err := ValidatePercentages(50, 20, 10)
	assertResultError(t, err, "sum to 100")
	core.AssertFalse(t, err.OK)
	core.AssertError(t, err.Value.(error))
}

func TestExportValidatePercentagesNegativeBadScenario(t *core.T) {
	err := ValidatePercentages(-10, 60, 50)
	assertResultError(t, err, "non-negative")
	core.AssertFalse(t, err.OK)
	core.AssertError(t, err.Value.(error))
}

// ---------------------------------------------------------------------------
// FilterResponses
// ---------------------------------------------------------------------------

func TestExportFilterResponsesKeepsOnlyUsableRowsScenario(t *core.T) {
	responses := []score.Response{
		{ID: "ok", Response: "This is a valid response with enough characters to pass the filter."},
		{ID: "empty", Response: ""},
		{ID: "error", Response: "ERROR: something went wrong"},
		{ID: "short", Response: "too short"},
		{ID: "ok2", Response: "Another valid response that meets the minimum character requirement."},
	}

	filtered := FilterResponses(responses)
	core.AssertLen(t, filtered, 2)
	core.AssertEqual(t, "ok", filtered[0].ID)
	core.AssertEqual(t, "ok2", filtered[1].ID)
}

func TestExportFilterResponsesAllFilteredGoodScenario(t *core.T) {
	responses := []score.Response{
		{Response: ""},
		{Response: "ERROR: fail"},
	}
	core.AssertEmpty(t, FilterResponses(responses))
}

// ---------------------------------------------------------------------------
// SplitData
// ---------------------------------------------------------------------------

func TestExportSplitDataAppliesRequestedRatiosScenario(t *core.T) {
	responses := make([]score.Response, 100)
	for i := range responses {
		responses[i] = score.Response{ID: string(rune('A' + i%26))}
	}

	train, valid, test := SplitData(responses, 80, 10, 10, 42)
	core.AssertLen(t, train, 80)
	core.AssertLen(t, valid, 10)
	core.AssertLen(t, test, 10)
}

func TestExportSplitDataDeterministicGoodScenario(t *core.T) {
	responses := make([]score.Response, 20)
	for i := range responses {
		responses[i] = score.Response{ID: string(rune('A' + i))}
	}

	train1, _, _ := SplitData(responses, 80, 10, 10, 123)
	train2, _, _ := SplitData(responses, 80, 10, 10, 123)
	core.AssertEqual(t, train1, train2, "same seed should produce same split")

	train3, _, _ := SplitData(responses, 80, 10, 10, 456)
	core.AssertNotEqual(t, train1, train3, "different seed should produce different split")
}

// ---------------------------------------------------------------------------
// WriteTrainingJSONL
// ---------------------------------------------------------------------------

func TestExportWriteTrainingJSONLWritesChatRowsScenario(t *core.T) {
	dir := t.TempDir()
	path := core.JoinPath(dir, "train.jsonl")

	responses := []score.Response{
		{Prompt: "What is 2+2?", Response: "4"},
		{Prompt: "Capital of France?", Response: "Paris"},
	}

	requireResultOK(t, WriteTrainingJSONL(path, responses))

	content, err := coreio.Local.Read(path)
	core.RequireNoError(t, err)

	// Verify each line is valid JSON with the expected structure
	lines := splitNonEmpty(content)
	core.AssertLen(t, lines, 2)

	var example TrainingExample
	mustJSONUnmarshalString(t, lines[0], &example)
	core.AssertLen(t, example.Messages, 2)
	core.AssertEqual(t, "user", example.Messages[0].Role)
	core.AssertEqual(t, "What is 2+2?", example.Messages[0].Content)
	core.AssertEqual(t, "assistant", example.Messages[1].Role)
	core.AssertEqual(t, "4", example.Messages[1].Content)
}

func TestExportWriteTrainingJSONLEmptyGoodScenario(t *core.T) {
	dir := t.TempDir()
	path := core.JoinPath(dir, "empty.jsonl")

	requireResultOK(t, WriteTrainingJSONL(path, nil))

	content, err := coreio.Local.Read(path)
	core.RequireNoError(t, err)
	core.AssertEmpty(t, content)
}

// splitNonEmpty splits a string by newlines and returns non-empty entries.
func splitNonEmpty(s string) []string {
	var result []string
	for _, line := range []byte(s) {
		_ = line
	}
	// Simple line split
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			if i > start {
				result = append(result, s[start:i])
			}
			start = i + 1
		}
	}
	if start < len(s) {
		result = append(result, s[start:])
	}
	return result
}

// --- v0.9.0 shape triplets ---

func TestExport_ValidatePercentages_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	assertResultOK(t, ValidatePercentages(80, 10, 10))
}

func TestExport_ValidatePercentages_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	assertResultError(t, ValidatePercentages(80, 10, 20), "sum")
}

func TestExport_ValidatePercentages_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	assertResultError(t, ValidatePercentages(-1, 50, 51), "non-negative")
}

func TestExport_FilterResponses_Good(t *core.T) {
	long := core.Concat("This response is deliberately long enough to survive filtering. ", "It has useful model output.")
	got := FilterResponses([]score.Response{{ID: "keep", Response: long}})
	core.AssertLen(t, got, 1)
	core.AssertEqual(t, "keep", got[0].ID)
}

func TestExport_FilterResponses_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	got := FilterResponses([]score.Response{{ID: "drop", Response: "ERROR: failed"}})
	core.AssertEmpty(t, got)
}

func TestExport_FilterResponses_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	got := FilterResponses([]score.Response{{ID: "empty"}, {ID: "short", Response: "too short"}})
	core.AssertEmpty(t, got)
}

func TestExport_SplitData_Good(t *core.T) {
	responses := []score.Response{{ID: "a"}, {ID: "b"}, {ID: "c"}, {ID: "d"}}
	train, valid, test := SplitData(responses, 50, 25, 25, 1)
	core.AssertLen(t, train, 2)
	core.AssertLen(t, valid, 1)
	core.AssertLen(t, test, 1)
}

func TestExport_SplitData_Bad(t *core.T) {
	train, valid, test := SplitData(nil, 80, 10, 10, 1)
	core.AssertEmpty(t, train)
	core.AssertEmpty(t, valid)
	core.AssertEmpty(t, test)
}

func TestExport_SplitData_Ugly(t *core.T) {
	responses := []score.Response{{ID: "only"}}
	train, valid, test := SplitData(responses, 0, 0, 100, 9)
	core.AssertEmpty(t, train)
	core.AssertEmpty(t, valid)
	core.AssertLen(t, test, 1)
}

func TestExport_WriteTrainingJSONL_Good(t *core.T) {
	file := core.JoinPath(t.TempDir(), "train.jsonl")
	err := WriteTrainingJSONL(file, []score.Response{{Prompt: "hello", Response: "world"}})
	requireResultOK(t, err)
	data, readErr := coreio.Local.Read(file)
	core.RequireNoError(t, readErr)
	core.AssertContains(t, data, "hello")
}

func TestExport_WriteTrainingJSONL_Bad(t *core.T) {
	dir := core.JoinPath(t.TempDir(), "blocked")
	core.RequireNoError(t, coreio.Local.EnsureDir(dir))
	err := WriteTrainingJSONL(dir, []score.Response{{Prompt: "hello", Response: "world"}})
	assertResultError(t, err)
}

func TestExport_WriteTrainingJSONL_Ugly(t *core.T) {
	file := core.JoinPath(t.TempDir(), "empty.jsonl")
	err := WriteTrainingJSONL(file, nil)
	requireResultOK(t, err)
	data, readErr := coreio.Local.Read(file)
	core.RequireNoError(t, readErr)
	core.AssertEqual(t, "", data)
}
