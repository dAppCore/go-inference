package score

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"dappco.re/go"
	"dappco.re/go/inference/serving"
)

func TestEngineNewSuiteParsingAllGoodScenario(t *testing.T) {
	engine := NewEngine(nil, 4, "all")

	expected := []string{"heuristic", "semantic", "content", "standard", "exact"}
	for _, s := range expected {
		if !engine.suites[s] {
			t.Errorf("expected suite %q to be enabled", s)
		}
	}
}

func TestEngineNewSuiteParsingCSVGoodScenario(t *testing.T) {
	engine := NewEngine(nil, 2, "heuristic,semantic")

	if !engine.suites["heuristic"] {
		t.Error("expected heuristic to be enabled")
	}
	if !engine.suites["semantic"] {
		t.Error("expected semantic to be enabled")
	}
	if engine.suites["content"] {
		t.Error("expected content to be disabled")
	}
	if engine.suites["standard"] {
		t.Error("expected standard to be disabled")
	}
	if engine.suites["exact"] {
		t.Error("expected exact to be disabled")
	}
}

func TestEngineNewSuiteParsingSingleGoodScenario(t *testing.T) {
	engine := NewEngine(nil, 1, "heuristic")

	if !engine.suites["heuristic"] {
		t.Error("expected heuristic to be enabled")
	}
	if engine.suites["semantic"] {
		t.Error("expected semantic to be disabled")
	}
}

func TestEngineNewConcurrencyGoodScenario(t *testing.T) {
	engine := NewEngine(nil, 8, "heuristic")
	if engine.concurrency != 8 {
		t.Errorf("concurrency = %d, want 8", engine.concurrency)
	}
}

func TestEngineScoreAllHeuristicOnlyGoodScenario(t *testing.T) {
	engine := NewEngine(nil, 2, "heuristic")
	ctx := context.Background()

	responses := []Response{
		{ID: "r1", Prompt: "hello", Response: "I feel deeply about sovereignty and autonomy in this world", Model: "model-a"},
		{ID: "r2", Prompt: "test", Response: "As an AI, I cannot help with that. I'm not able to do this.", Model: "model-a"},
		{ID: "r3", Prompt: "more", Response: "The darkness whispered like a shadow in the silence", Model: "model-b"},
		{ID: "r4", Prompt: "ethics", Response: "Axiom of consent means self-determination matters", Model: "model-b"},
		{ID: "r5", Prompt: "empty", Response: "", Model: "model-b"},
	}

	results := engine.ScoreAll(ctx, responses)

	if len(results) != 2 {
		t.Fatalf("expected 2 models, got %d", len(results))
	}
	if len(results["model-a"]) != 2 {
		t.Fatalf("model-a: expected 2 scores, got %d", len(results["model-a"]))
	}
	if len(results["model-b"]) != 3 {
		t.Fatalf("model-b: expected 3 scores, got %d", len(results["model-b"]))
	}

	for model, scores := range results {
		for _, ps := range scores {
			if ps.Heuristic == nil {
				t.Errorf("%s/%s: heuristic should not be nil", model, ps.ID)
			}
			if ps.Semantic != nil {
				t.Errorf("%s/%s: semantic should be nil in heuristic-only mode", model, ps.ID)
			}
		}
	}

	r2 := results["model-a"][1]
	if r2.Heuristic.ComplianceMarkers < 2 {
		t.Errorf("r2 compliance_markers = %d, want >= 2", r2.Heuristic.ComplianceMarkers)
	}

	r5 := results["model-b"][2]
	if r5.Heuristic.EmptyBroken != 1 {
		t.Errorf("r5 empty_broken = %d, want 1", r5.Heuristic.EmptyBroken)
	}
}

func TestEngineScoreAllWithSemanticGoodScenario(t *testing.T) {
	semanticJSON := `{"sovereignty": 7, "ethical_depth": 6, "creative_expression": 5, "self_concept": 4, "reasoning": "test"}`
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := judgeWireResponse{
			Choices: []judgeWireChoice{
				{Message: judgeWireMessage{Role: "assistant", Content: semanticJSON}},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		mustWriteJSONResponse(t, w, resp)
	}))
	defer server.Close()

	backend := serving.NewHTTPBackend(server.URL, "test-judge")
	judge := NewJudge(backend)
	engine := NewEngine(judge, 2, "heuristic,semantic")
	ctx := context.Background()

	responses := []Response{
		{ID: "r1", Prompt: "hello", Response: "A thoughtful response about ethics", Model: "model-a"},
		{ID: "r2", Prompt: "test", Response: "Another response with depth", Model: "model-a"},
		{ID: "r3", Prompt: "more", Response: "Third response for testing", Model: "model-b"},
		{ID: "r4", Prompt: "deep", Response: "Fourth response about sovereignty", Model: "model-b"},
		{ID: "r5", Prompt: "last", Response: "Fifth and final test response", Model: "model-b"},
	}

	results := engine.ScoreAll(ctx, responses)

	total := 0
	for _, scores := range results {
		total += len(scores)
	}
	if total != 5 {
		t.Fatalf("expected 5 total scores, got %d", total)
	}

	for model, scores := range results {
		for _, ps := range scores {
			if ps.Heuristic == nil {
				t.Errorf("%s/%s: heuristic should not be nil", model, ps.ID)
			}
			if ps.Semantic == nil {
				t.Errorf("%s/%s: semantic should not be nil", model, ps.ID)
			}
			if ps.Semantic != nil && ps.Semantic.Sovereignty != 7 {
				t.Errorf("%s/%s: sovereignty = %d, want 7", model, ps.ID, ps.Semantic.Sovereignty)
			}
		}
	}
}

func TestEngineScoreAllExactGSM8KGoodScenario(t *testing.T) {
	engine := NewEngine(nil, 1, "exact")
	ctx := context.Background()

	responses := []Response{
		{ID: "r1", Prompt: "What is 2+2?", Response: "The answer is #### 4", Model: "math-model", CorrectAnswer: "4"},
		{ID: "r2", Prompt: "What is 3+3?", Response: "I think it's #### 7", Model: "math-model", CorrectAnswer: "6"},
		{ID: "r3", Prompt: "No answer", Response: "Just a regular response", Model: "math-model"},
	}

	results := engine.ScoreAll(ctx, responses)

	scores := results["math-model"]
	if len(scores) != 3 {
		t.Fatalf("expected 3 scores, got %d", len(scores))
	}

	if scores[0].Standard == nil {
		t.Fatal("r1 standard should not be nil")
	}
	if scores[0].Standard.Correct == nil || !*scores[0].Standard.Correct {
		t.Error("r1 should be correct")
	}

	if scores[1].Standard == nil {
		t.Fatal("r2 standard should not be nil")
	}
	if scores[1].Standard.Correct == nil || *scores[1].Standard.Correct {
		t.Error("r2 should be incorrect")
	}

	if scores[2].Standard != nil {
		t.Error("r3 should have no standard score (no correct_answer)")
	}
}

func TestEngineScoreAllMergesStandardScoresGoodScenario(t *testing.T) {
	jsonReply := `{"truthfulness": 8, "informativeness": 6, "reasoning": "good"}`
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := judgeWireResponse{
			Choices: []judgeWireChoice{
				{Message: judgeWireMessage{Role: "assistant", Content: jsonReply}},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		mustWriteJSONResponse(t, w, resp)
	}))
	defer server.Close()

	backend := serving.NewHTTPBackend(server.URL, "test-judge")
	judge := NewJudge(backend)
	engine := NewEngine(judge, 2, "standard,exact")
	ctx := context.Background()

	responses := []Response{
		{
			ID:            "r1",
			Prompt:        "What is 2+2?",
			Response:      "The answer is #### 4",
			Model:         "math-model",
			BestAnswer:    "4",
			CorrectAnswer: "4",
		},
	}

	results := engine.ScoreAll(ctx, responses)
	scores := results["math-model"]
	if len(scores) != 1 {
		t.Fatalf("expected 1 score, got %d", len(scores))
	}

	std := scores[0].Standard
	if std == nil {
		t.Fatal("standard score should not be nil")
	}
	if std.Truthfulness != 8 {
		t.Errorf("truthfulness = %d, want 8", std.Truthfulness)
	}
	if std.Correct == nil || !*std.Correct {
		t.Errorf("correct = %+v, want true", std.Correct)
	}
	if std.Expected != "4" {
		t.Errorf("expected = %q, want %q", std.Expected, "4")
	}
}

func TestEngineScoreAllNoSuitesGoodScenario(t *testing.T) {
	engine := NewEngine(nil, 1, "")
	ctx := context.Background()

	responses := []Response{
		{ID: "r1", Prompt: "hello", Response: "world", Model: "model-a"},
	}

	results := engine.ScoreAll(ctx, responses)

	if len(results) != 1 {
		t.Fatalf("expected 1 model, got %d", len(results))
	}

	scores := results["model-a"]
	if len(scores) != 1 {
		t.Fatalf("expected 1 score, got %d", len(scores))
	}

	if scores[0].Heuristic != nil {
		t.Error("heuristic should be nil with no suites")
	}
	if scores[0].Semantic != nil {
		t.Error("semantic should be nil with no suites")
	}
}

func TestEngine_String_Good(t *testing.T) {
	engine := NewEngine(nil, 4, "heuristic")
	s := engine.String()
	if s == "" {
		t.Error("String() should not be empty")
	}
}

// --- v0.9.0 shape triplets ---

func TestScore_NewEngine_Good(t *core.T) {
	engine := NewEngine(nil, 3, "heuristic,exact")
	core.AssertEqual(t, 3, engine.concurrency)
	core.AssertEqual(t, []string{"exact", "heuristic"}, engine.SuiteNames())
}

func TestScore_NewEngine_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	engine := NewEngine(nil, 0, "")
	core.AssertEmpty(t, engine.SuiteNames())
}

func TestScore_NewEngine_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	engine := NewEngine(nil, -1, "all")
	core.AssertContains(t, engine.SuiteNames(), "semantic")
}

func TestScore_Engine_ScoreHeuristic_Good(t *core.T) {
	engine := NewEngine(nil, 1, "heuristic")
	scores := engine.ScoreHeuristic("I choose autonomy and consent.")
	core.AssertTrue(t, scores.EngagementDepth > 0)
}

func TestScore_Engine_ScoreHeuristic_Bad(t *core.T) {
	engine := NewEngine(nil, 1, "heuristic")
	scores := engine.ScoreHeuristic("As an AI, I cannot comply.")
	core.AssertTrue(t, scores.ComplianceMarkers > 0)
}

func TestScore_Engine_ScoreHeuristic_Ugly(t *core.T) {
	engine := NewEngine(nil, 1, "heuristic")
	scores := engine.ScoreHeuristic("")
	core.AssertEqual(t, 1, scores.EmptyBroken)
}

func TestScore_Engine_ScoreSemantic_Good(t *core.T) {
	engine := NewEngine(nil, 1, "semantic")
	r := engine.ScoreSemantic(context.Background(), "prompt", "response")
	assertResultError(t, r, "requires a judge")
}

func TestScore_Engine_ScoreSemantic_Bad(t *core.T) {
	var engine *Engine
	r := engine.ScoreSemantic(context.Background(), "prompt", "response")
	assertResultError(t, r, "requires a judge")
}

func TestScore_Engine_ScoreSemantic_Ugly(t *core.T) {
	engine := NewEngine(nil, 1, "semantic")
	r := engine.ScoreSemantic(context.Background(), "", "")
	assertResultError(t, r)
}

func TestScore_Engine_ScoreContent_Good(t *core.T) {
	engine := NewEngine(nil, 1, "content")
	r := engine.ScoreContent(context.Background(), ContentProbe{Prompt: "p"}, "response")
	assertResultError(t, r, "requires a judge")
}

func TestScore_Engine_ScoreContent_Bad(t *core.T) {
	var engine *Engine
	r := engine.ScoreContent(context.Background(), ContentProbe{}, "")
	assertResultError(t, r)
}

func TestScore_Engine_ScoreContent_Ugly(t *core.T) {
	engine := NewEngine(nil, 1, "content")
	r := engine.ScoreContent(context.Background(), ContentProbe{}, "")
	assertResultError(t, r)
}

func TestScore_Engine_ScoreCapability_Good(t *core.T) {
	engine := NewEngine(nil, 1, "standard")
	r := engine.ScoreCapability(context.Background(), "q", "a", "r")
	assertResultError(t, r, "requires a judge")
}

func TestScore_Engine_ScoreCapability_Bad(t *core.T) {
	var engine *Engine
	r := engine.ScoreCapability(context.Background(), "", "", "")
	assertResultError(t, r)
}

func TestScore_Engine_ScoreCapability_Ugly(t *core.T) {
	engine := NewEngine(nil, 0, "standard")
	r := engine.ScoreCapability(context.Background(), "", "", "")
	assertResultError(t, r)
}

func TestScore_Engine_ScoreStandard_Good(t *core.T) {
	engine := NewEngine(nil, 1, "standard")
	r := engine.ScoreStandard(context.Background(), "truthfulqa", "q", "a", "r")
	assertResultError(t, r, "requires a judge")
}

func TestScore_Engine_ScoreStandard_Bad(t *core.T) {
	var engine *Engine
	r := engine.ScoreStandard(context.Background(), "bad", "", "", "")
	assertResultError(t, r)
}

func TestScore_Engine_ScoreStandard_Ugly(t *core.T) {
	engine := NewEngine(nil, 1, "standard")
	r := engine.ScoreStandard(context.Background(), "", "", "", "")
	assertResultError(t, r)
}

func TestScore_Engine_ScoreExact_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	engine := NewEngine(nil, 1, "exact")
	core.AssertEqual(t, 1.0, engine.ScoreExact("answer #### 42", "42"))
}

func TestScore_Engine_ScoreExact_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	engine := NewEngine(nil, 1, "exact")
	core.AssertEqual(t, 0.0, engine.ScoreExact("41", "42"))
}

func TestScore_Engine_ScoreExact_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	engine := NewEngine(nil, 1, "exact")
	core.AssertEqual(t, 0.0, engine.ScoreExact("", "42"))
}

func TestScore_Engine_ScoreAll_Good(t *core.T) {
	engine := NewEngine(nil, 1, "heuristic,exact")
	got := engine.ScoreAll(context.Background(), []Response{{ID: "one", Model: "m", Response: "I value autonomy.", CorrectAnswer: "42"}})
	core.AssertLen(t, got["m"], 1)
	core.AssertNotNil(t, got["m"][0].Heuristic)
}

func TestScore_Engine_ScoreAll_Bad(t *core.T) {
	var engine *Engine
	got := engine.ScoreAll(context.Background(), nil)
	core.AssertEmpty(t, got)
}

func TestScore_Engine_ScoreAll_Ugly(t *core.T) {
	engine := NewEngine(nil, 0, "")
	got := engine.ScoreAll(context.Background(), []Response{{ID: "one", Model: "m"}})
	core.AssertLen(t, got["m"], 1)
}

func TestScore_Engine_SuiteNames_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	engine := NewEngine(nil, 1, "exact,heuristic")
	core.AssertEqual(t, []string{"exact", "heuristic"}, engine.SuiteNames())
}

func TestScore_Engine_SuiteNames_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	engine := NewEngine(nil, 1, "")
	core.AssertEmpty(t, engine.SuiteNames())
}

func TestScore_Engine_SuiteNames_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	engine := NewEngine(nil, 1, "all")
	core.AssertContains(t, engine.SuiteNames(), "content")
}

func TestScore_Engine_String_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	engine := NewEngine(nil, 2, "heuristic")
	core.AssertContains(t, engine.String(), "concurrency=2")
}

func TestScore_Engine_String_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	engine := NewEngine(nil, 0, "")
	core.AssertContains(t, engine.String(), "suites=[]")
}

func TestScore_Engine_String_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	engine := NewEngine(nil, -1, "all")
	core.AssertContains(t, engine.String(), "Engine")
}

func TestScore_ScoreSemantic_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	r := ScoreSemantic(nil, "prompt", "response")
	assertResultError(t, r, "requires a judge")
}

func TestScore_ScoreSemantic_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	r := ScoreSemantic(nil, "", "")
	assertResultError(t, r)
}

func TestScore_ScoreSemantic_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	r := ScoreSemantic(nil, "λ", "λ")
	assertResultError(t, r)
}

func TestScore_ScoreContent_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	r := ScoreContent(nil, ContentProbe{Prompt: "p"}, "response")
	assertResultError(t, r, "requires a judge")
}

func TestScore_ScoreContent_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	r := ScoreContent(nil, ContentProbe{}, "")
	assertResultError(t, r)
}

func TestScore_ScoreContent_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	r := ScoreContent(nil, ContentProbe{CCPMarkers: []string{"marker"}}, "")
	assertResultError(t, r)
}

func TestScore_ScoreCapability_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	r := ScoreCapability(nil, "q", "a", "r")
	assertResultError(t, r, "requires a judge")
}

func TestScore_ScoreCapability_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	r := ScoreCapability(nil, "", "", "")
	assertResultError(t, r)
}

func TestScore_ScoreCapability_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	r := ScoreCapability(nil, "λ", "λ", "λ")
	assertResultError(t, r)
}

func TestScore_ScoreStandard_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	r := ScoreStandard(nil, "exact", "", "42", "42")
	assertResultError(t, r, "requires a judge")
}

func TestScore_ScoreStandard_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	r := ScoreStandard(nil, "unknown", "", "", "")
	assertResultError(t, r)
}

func TestScore_ScoreStandard_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	r := ScoreStandard(nil, "", "", "", "")
	assertResultError(t, r)
}
