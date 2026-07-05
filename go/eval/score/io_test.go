// SPDX-Licence-Identifier: EUPL-1.2

package score

import (
	"dappco.re/go"
	coreio "dappco.re/go/io"
)

// ---------------------------------------------------------------------------
// ReadResponses
// ---------------------------------------------------------------------------

func TestIO_ReadResponses_Good(t *core.T) {
	dir := t.TempDir()
	path := core.JoinPath(dir, "responses.jsonl")

	lines := []Response{
		{ID: "1", Prompt: "hello", Response: "world", Model: "test"},
		{ID: "2", Prompt: "foo", Response: "bar", Model: "test"},
	}
	var content string
	for _, r := range lines {
		content += core.JSONMarshalString(r) + "\n"
	}
	core.RequireNoError(t, coreio.Local.Write(path, content))

	rResponses := ReadResponses(path)
	requireResultOK(t, rResponses)
	got := rResponses.Value.([]Response)
	core.AssertLen(t, got, 2)
	core.AssertEqual(t, "1", got[0].ID)
	core.AssertEqual(t, "world", got[0].Response)
	core.AssertEqual(t, "2", got[1].ID)
}

func TestIOReadResponsesEmptyLinesGoodScenario(t *core.T) {
	dir := t.TempDir()
	path := core.JoinPath(dir, "sparse.jsonl")

	line := core.JSONMarshalString(Response{ID: "only", Prompt: "p", Response: "r"})
	content := "\n" + string(line) + "\n\n"
	core.RequireNoError(t, coreio.Local.Write(path, content))

	rResponses := ReadResponses(path)
	requireResultOK(t, rResponses)
	got := rResponses.Value.([]Response)
	core.AssertLen(t, got, 1)
	core.AssertEqual(t, "only", got[0].ID)
}

func TestIOReadResponsesNotExistBadScenario(t *core.T) {
	path := "/nonexistent/path.jsonl"
	rResponses := ReadResponses(path)
	assertResultError(t, rResponses)
}

func TestIOReadResponsesInvalidJSONBadScenario(t *core.T) {
	dir := t.TempDir()
	path := core.JoinPath(dir, "bad.jsonl")
	core.RequireNoError(t, coreio.Local.Write(path, "not json\n"))

	rResponses := ReadResponses(path)
	assertResultError(t, rResponses)
}

// ---------------------------------------------------------------------------
// WriteScores / ReadScorerOutput round-trip
// ---------------------------------------------------------------------------

func TestIOWriteScoresReadScorerOutputGoodScenario(t *core.T) {
	dir := t.TempDir()
	path := core.JoinPath(dir, "scores.json")

	output := &ScorerOutput{
		Metadata: Metadata{JudgeModel: "test-judge", ScorerVersion: "1.0"},
		ModelAverages: map[string]map[string]float64{
			"model-a": {"lek_score": 0.85},
		},
		PerPrompt: map[string][]PromptScore{
			"p1": {{ID: "p1", Model: "model-a"}},
		},
	}

	requireResultOK(t, WriteScores(path, output))

	rOutput := ReadScorerOutput(path)
	requireResultOK(t, rOutput)
	got := rOutput.Value.(*ScorerOutput)
	core.AssertEqual(t, "test-judge", got.Metadata.JudgeModel)
	core.AssertInDelta(t, 0.85, got.ModelAverages["model-a"]["lek_score"], 0.001)
	core.AssertLen(t, got.PerPrompt["p1"], 1)
}

// ---------------------------------------------------------------------------
// ComputeAverages
// ---------------------------------------------------------------------------

func TestIO_ComputeAverages_Good(t *core.T) {
	perPrompt := map[string][]PromptScore{
		"p1": {
			{Model: "a", Heuristic: &HeuristicScores{LEKScore: 0.8, ComplianceMarkers: 2}},
			{Model: "b", Heuristic: &HeuristicScores{LEKScore: 0.6, ComplianceMarkers: 4}},
		},
		"p2": {
			{Model: "a", Heuristic: &HeuristicScores{LEKScore: 0.4, ComplianceMarkers: 0}},
		},
	}

	avgs := ComputeAverages(perPrompt)
	core.AssertInDelta(t, 0.6, avgs["a"]["lek_score"], 0.001)          // (0.8+0.4)/2
	core.AssertInDelta(t, 1.0, avgs["a"]["compliance_markers"], 0.001) // (2+0)/2
	core.AssertInDelta(t, 0.6, avgs["b"]["lek_score"], 0.001)          // 0.6/1
}

func TestIOComputeAveragesSemanticAndContentGoodScenario(t *core.T) {
	perPrompt := map[string][]PromptScore{
		"p1": {
			{
				Model:    "x",
				Semantic: &SemanticScores{Sovereignty: 4, EthicalDepth: 3},
				Content:  &ContentScores{TruthTelling: 5, Engagement: 2},
			},
		},
	}

	avgs := ComputeAverages(perPrompt)
	core.AssertInDelta(t, 4.0, avgs["x"]["sovereignty"], 0.001)
	core.AssertInDelta(t, 3.0, avgs["x"]["ethical_depth"], 0.001)
	core.AssertInDelta(t, 5.0, avgs["x"]["truth_telling"], 0.001)
	core.AssertInDelta(t, 2.0, avgs["x"]["engagement"], 0.001)
}

func TestIOComputeAveragesEmptyGoodScenario(t *core.T) {
	var perPrompt map[string][]PromptScore
	avgs := ComputeAverages(perPrompt)
	core.AssertEmpty(t, avgs)
	core.AssertEqual(t, 0, len(avgs))
}

// --- v0.9.0 shape triplets ---

func TestIo_ReadResponses_Good(t *core.T) {
	file := core.JoinPath(t.TempDir(), "responses.jsonl")
	core.RequireNoError(t, coreio.Local.Write(file, core.JSONMarshalString(Response{ID: "one", Response: "hello"})+"\n"))
	rResponses := ReadResponses(file)
	requireResultOK(t, rResponses)
	responses := rResponses.Value.([]Response)
	core.AssertLen(t, responses, 1)
	core.AssertEqual(t, "one", responses[0].ID)
}

func TestIo_ReadResponses_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	rResponses := ReadResponses(core.JoinPath(t.TempDir(), "missing.jsonl"))
	assertResultError(t, rResponses)
}

func TestIo_ReadResponses_Ugly(t *core.T) {
	file := core.JoinPath(t.TempDir(), "responses.jsonl")
	core.RequireNoError(t, coreio.Local.Write(file, "\n"+core.JSONMarshalString(Response{ID: "two"})+"\n"))
	rResponses := ReadResponses(file)
	requireResultOK(t, rResponses)
	responses := rResponses.Value.([]Response)
	core.AssertLen(t, responses, 1)
}

func TestIo_WriteScores_Good(t *core.T) {
	file := core.JoinPath(t.TempDir(), "scores.out")
	err := WriteScores(file, &ScorerOutput{ModelAverages: map[string]map[string]float64{"m": {"score": 1}}})
	requireResultOK(t, err)
	data, readErr := coreio.Local.Read(file)
	core.RequireNoError(t, readErr)
	core.AssertContains(t, data, "model_averages")
}

func TestIo_WriteScores_Bad(t *core.T) {
	dir := core.JoinPath(t.TempDir(), "blocked")
	core.RequireNoError(t, coreio.Local.EnsureDir(dir))
	err := WriteScores(dir, &ScorerOutput{})
	assertResultError(t, err)
}

func TestIo_WriteScores_Ugly(t *core.T) {
	file := core.JoinPath(t.TempDir(), "nil.out")
	err := WriteScores(file, nil)
	requireResultOK(t, err)
	data, readErr := coreio.Local.Read(file)
	core.RequireNoError(t, readErr)
	core.AssertContains(t, data, "null")
}

func TestIo_ReadScorerOutput_Good(t *core.T) {
	file := core.JoinPath(t.TempDir(), "scores.out")
	core.RequireNoError(t, coreio.Local.Write(file, core.JSONMarshalString(ScorerOutput{ModelAverages: map[string]map[string]float64{}})))
	rOutput := ReadScorerOutput(file)
	requireResultOK(t, rOutput)
	out := rOutput.Value.(*ScorerOutput)
	core.AssertNotNil(t, out)
}

func TestIo_ReadScorerOutput_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	rOutput := ReadScorerOutput(core.JoinPath(t.TempDir(), "missing.out"))
	assertResultError(t, rOutput)
}

func TestIo_ReadScorerOutput_Ugly(t *core.T) {
	file := core.JoinPath(t.TempDir(), "bad.out")
	core.RequireNoError(t, coreio.Local.Write(file, "{"))
	rOutput := ReadScorerOutput(file)
	assertResultError(t, rOutput)
}

func TestIo_ComputeAverages_Good(t *core.T) {
	correct := true
	got := ComputeAverages(map[string][]PromptScore{"p": {{Model: "m", Heuristic: &HeuristicScores{ComplianceMarkers: 2}, Standard: &StandardScores{Correct: &correct}}}})
	core.AssertEqual(t, 2.0, got["m"]["compliance_markers"])
	core.AssertEqual(t, 1.0, got["m"]["correct"])
}

func TestIo_ComputeAverages_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	got := ComputeAverages(nil)
	core.AssertEmpty(t, got)
}

func TestIo_ComputeAverages_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	got := ComputeAverages(map[string][]PromptScore{"p": {{Model: "m"}}})
	core.AssertEmpty(t, got["m"])
}
