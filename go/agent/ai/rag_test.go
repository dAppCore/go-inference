package ai

import (
	"context"
	"testing"

	core "dappco.re/go"
	rag "dappco.re/go/rag"
)

func repeatString(value string, count int) string {
	parts := make([]string, count)
	for i := range parts {
		parts[i] = value
	}
	return core.Join("", parts...)
}

func TestBuildTaskQuery_Good_CombinesAndTruncates(t *testing.T) {
	got := buildTaskQuery(TaskInfo{
		Title:       "Investigate build failure",
		Description: "CI compile step fails",
	})

	want := "Investigate build failure: CI compile step fails"
	if got != want {
		t.Fatalf("buildTaskQuery() = %q, want %q", got, want)
	}
}

func TestBuildTaskQuery_Good_TruncatesCombinedQuery(t *testing.T) {
	got := buildTaskQuery(TaskInfo{
		Title:       repeatString("t", ragTaskQueryRuneLimit),
		Description: "extra",
	})

	if gotRuneLen := len([]rune(got)); gotRuneLen != ragTaskQueryRuneLimit {
		t.Fatalf("buildTaskQuery() rune length = %d, want %d", gotRuneLen, ragTaskQueryRuneLimit)
	}
}

func TestBuildTaskQuery_Good_TruncatesToLimit(t *testing.T) {
	got := buildTaskQuery(TaskInfo{
		Title:       "",
		Description: repeatString("x", ragTaskQueryRuneLimit+25),
	})

	if got == "" {
		t.Fatal("buildTaskQuery() returned empty string for non-empty task")
	}
	if gotRuneLen := len([]rune(got)); gotRuneLen != ragTaskQueryRuneLimit {
		t.Fatalf("buildTaskQuery() rune length = %d, want %d", gotRuneLen, ragTaskQueryRuneLimit)
	}
}

func TestBuildTaskQuery_Good_TruncatesDescriptionBeforeComposition(t *testing.T) {
	got := buildTaskQuery(TaskInfo{
		Title:       "Investigate",
		Description: repeatString("y", ragTaskQueryRuneLimit+25),
	})

	if gotRuneLen := len([]rune(got)); gotRuneLen != ragTaskQueryRuneLimit {
		t.Fatalf("buildTaskQuery() rune length = %d, want %d", gotRuneLen, ragTaskQueryRuneLimit)
	}
	if !core.HasPrefix(got, "Investigate: ") {
		t.Fatalf("buildTaskQuery() = %q, want title prefix preserved", got)
	}
}

func TestBuildTaskQuery_Good_TruncatesCombinedQueryExactly(t *testing.T) {
	title := repeatString("t", 320)
	description := repeatString("d", 320)

	got := buildTaskQuery(TaskInfo{
		Title:       title,
		Description: description,
	})

	want := truncateRunes(title+": "+description, ragTaskQueryRuneLimit)
	if got != want {
		t.Fatalf("buildTaskQuery() = %q, want %q", got, want)
	}
}

func TestBuildTaskQuery_Good_BlankTaskReturnsEmpty(t *testing.T) {
	got := buildTaskQuery(TaskInfo{})
	if got != "" {
		t.Fatalf("buildTaskQuery() = %q, want empty string", got)
	}
}

func TestBuildTaskQuery_Good_UsesDescriptionWithRFCSeparator(t *testing.T) {
	got := buildTaskQuery(TaskInfo{
		Description: "CI compile step fails",
	})

	want := ": CI compile step fails"
	if got != want {
		t.Fatalf("buildTaskQuery() = %q, want %q", got, want)
	}
}

func TestQueryRAGForTask_Good_DegradesOnClientErrors(t *testing.T) {
	origNewQdrantClient := newQdrantClient
	origNewOllamaClient := newOllamaClient
	origRunRAGQuery := runRAGQuery
	t.Cleanup(func() {
		newQdrantClient = origNewQdrantClient
		newOllamaClient = origNewOllamaClient
		runRAGQuery = origRunRAGQuery
	})

	newQdrantClient = func(rag.QdrantConfig) core.Result {
		return core.Fail(core.NewError("qdrant unavailable"))
	}

	if result := QueryRAGForTask(TaskInfo{Title: "Investigate", Description: "failure"}); !result.OK {
		t.Fatalf("QueryRAGForTask() error = %s, want nil", result.Error())
	} else if got := result.Value.(string); got != "" {
		t.Fatalf("QueryRAGForTask() = %q, want empty string", got)
	}

	newQdrantClient = origNewQdrantClient
	newOllamaClient = func(rag.OllamaConfig) core.Result {
		return core.Fail(core.NewError("ollama unavailable"))
	}

	if result := QueryRAGForTask(TaskInfo{Title: "Investigate", Description: "failure"}); !result.OK {
		t.Fatalf("QueryRAGForTask() error = %s, want nil", result.Error())
	} else if got := result.Value.(string); got != "" {
		t.Fatalf("QueryRAGForTask() = %q, want empty string", got)
	}

	newOllamaClient = origNewOllamaClient
	runRAGQuery = func(
		_ context.Context,
		_ rag.VectorStore,
		_ rag.Embedder,
		_ string,
		_ rag.QueryConfig,
	) core.Result {
		return core.Fail(core.NewError("query failed"))
	}

	if result := QueryRAGForTask(TaskInfo{Title: "Investigate", Description: "failure"}); !result.OK {
		t.Fatalf("QueryRAGForTask() error = %s, want nil", result.Error())
	} else if got := result.Value.(string); got != "" {
		t.Fatalf("QueryRAGForTask() = %q, want empty string", got)
	}
}

func TestRag_QueryRAGForTask_Good_ReturnsFormattedContext(t *testing.T) {
	origNewQdrantClient := newQdrantClient
	origNewOllamaClient := newOllamaClient
	origRunRAGQuery := runRAGQuery
	origCloseQdrant := closeQdrant
	t.Cleanup(func() {
		newQdrantClient = origNewQdrantClient
		newOllamaClient = origNewOllamaClient
		runRAGQuery = origRunRAGQuery
		closeQdrant = origCloseQdrant
	})

	var seenQuery string
	var seenConfig rag.QueryConfig
	newQdrantClient = func(rag.QdrantConfig) core.Result {
		return core.Ok((*rag.QdrantClient)(nil))
	}
	newOllamaClient = func(rag.OllamaConfig) core.Result {
		return core.Ok((*rag.OllamaClient)(nil))
	}
	closeQdrant = func(*rag.QdrantClient) core.Result { return core.Ok(nil) }
	runRAGQuery = func(
		_ context.Context,
		_ rag.VectorStore,
		_ rag.Embedder,
		query string,
		cfg rag.QueryConfig,
	) core.Result {
		seenQuery = query
		seenConfig = cfg
		return core.Ok([]rag.QueryResult{
			{
				Text:    "Build failure runbook",
				Source:  "docs/build.md",
				Section: "Troubleshooting",
				Score:   0.91,
			},
		})
	}

	result := QueryRAGForTask(TaskInfo{
		Title:       "Investigate build failure",
		Description: "CI compile step fails",
	})
	if !result.OK {
		t.Fatalf("QueryRAGForTask() error = %s, want nil", result.Error())
	}
	got := result.Value.(string)
	if got == "" {
		t.Fatal("QueryRAGForTask() returned empty context for a populated result set")
	}
	if seenQuery != "Investigate build failure: CI compile step fails" {
		t.Fatalf("QueryRAGForTask() query = %q, want task title + description", seenQuery)
	}
	if seenConfig.Collection != ragTaskCollection || seenConfig.Limit != ragTaskResultLimit || seenConfig.Threshold != ragTaskSimilarityThreshold {
		t.Fatalf("QueryRAGForTask() config = %+v, want collection/limit/threshold defaults", seenConfig)
	}

	want := rag.FormatResultsContext([]rag.QueryResult{{
		Text:    "Build failure runbook",
		Source:  "docs/build.md",
		Section: "Troubleshooting",
		Score:   0.91,
	}})
	if got != want {
		t.Fatalf("QueryRAGForTask() = %q, want %q", got, want)
	}
}

func TestRag_QueryRAGForTask_Good_ClosesOpenedQdrantClient(t *testing.T) {
	origNewQdrantClient := newQdrantClient
	origNewOllamaClient := newOllamaClient
	origRunRAGQuery := runRAGQuery
	origCloseQdrant := closeQdrant
	t.Cleanup(func() {
		newQdrantClient = origNewQdrantClient
		newOllamaClient = origNewOllamaClient
		runRAGQuery = origRunRAGQuery
		closeQdrant = origCloseQdrant
	})

	var closed bool
	newQdrantClient = func(rag.QdrantConfig) core.Result {
		return core.Ok(&rag.QdrantClient{})
	}
	newOllamaClient = func(rag.OllamaConfig) core.Result {
		return core.Ok(&rag.OllamaClient{})
	}
	closeQdrant = func(client *rag.QdrantClient) core.Result {
		if client == nil {
			t.Fatal("expected closeQdrant to receive a client")
		}
		closed = true
		return core.Ok(nil)
	}
	runRAGQuery = func(
		_ context.Context,
		_ rag.VectorStore,
		_ rag.Embedder,
		_ string,
		_ rag.QueryConfig,
	) core.Result {
		return core.Ok([]rag.QueryResult{{Text: "Doc", Source: "docs.md"}})
	}

	result := QueryRAGForTask(TaskInfo{
		Title:       "Investigate",
		Description: "failure",
	})
	if !result.OK {
		t.Fatalf("QueryRAGForTask() error = %s, want nil", result.Error())
	}
	got := result.Value.(string)
	if got == "" {
		t.Fatal("QueryRAGForTask() returned empty context for a populated result set")
	}
	if !closed {
		t.Fatal("expected QueryRAGForTask to close the opened Qdrant client")
	}
}

func TestRag_QueryRAGForTask_Bad_ReturnsEmptyStringWhenNoResults(t *testing.T) {
	origNewQdrantClient := newQdrantClient
	origNewOllamaClient := newOllamaClient
	origRunRAGQuery := runRAGQuery
	origCloseQdrant := closeQdrant
	t.Cleanup(func() {
		newQdrantClient = origNewQdrantClient
		newOllamaClient = origNewOllamaClient
		runRAGQuery = origRunRAGQuery
		closeQdrant = origCloseQdrant
	})

	newQdrantClient = func(rag.QdrantConfig) core.Result {
		return core.Ok((*rag.QdrantClient)(nil))
	}
	newOllamaClient = func(rag.OllamaConfig) core.Result {
		return core.Ok((*rag.OllamaClient)(nil))
	}
	closeQdrant = func(*rag.QdrantClient) core.Result { return core.Ok(nil) }
	runRAGQuery = func(
		_ context.Context,
		_ rag.VectorStore,
		_ rag.Embedder,
		_ string,
		_ rag.QueryConfig,
	) core.Result {
		return core.Ok([]rag.QueryResult(nil))
	}

	result := QueryRAGForTask(TaskInfo{
		Title:       "Investigate build failure",
		Description: "CI compile step fails",
	})
	if !result.OK {
		t.Fatalf("QueryRAGForTask() error = %s, want nil", result.Error())
	}
	got := result.Value.(string)
	if got != "" {
		t.Fatalf("QueryRAGForTask() = %q, want empty string for no matches", got)
	}
}

func TestRag_QueryRAGForTask_Ugly_EmptyTaskShortCircuitsSeams(t *testing.T) {
	origNewQdrantClient := newQdrantClient
	origNewOllamaClient := newOllamaClient
	origRunRAGQuery := runRAGQuery
	origCloseQdrant := closeQdrant
	t.Cleanup(func() {
		newQdrantClient = origNewQdrantClient
		newOllamaClient = origNewOllamaClient
		runRAGQuery = origRunRAGQuery
		closeQdrant = origCloseQdrant
	})

	newQdrantClient = func(rag.QdrantConfig) core.Result {
		t.Fatal("newQdrantClient should not be called for an empty task")
		return core.Ok((*rag.QdrantClient)(nil))
	}
	newOllamaClient = func(rag.OllamaConfig) core.Result {
		t.Fatal("newOllamaClient should not be called for an empty task")
		return core.Ok((*rag.OllamaClient)(nil))
	}
	runRAGQuery = func(
		_ context.Context,
		_ rag.VectorStore,
		_ rag.Embedder,
		_ string,
		_ rag.QueryConfig,
	) core.Result {
		t.Fatal("runRAGQuery should not be called for an empty task")
		return core.Ok([]rag.QueryResult(nil))
	}
	closeQdrant = func(*rag.QdrantClient) core.Result {
		t.Fatal("closeQdrant should not be called for an empty task")
		return core.Ok(nil)
	}

	result := QueryRAGForTask(TaskInfo{})
	if !result.OK {
		t.Fatalf("QueryRAGForTask() error = %s, want nil", result.Error())
	}
	got := result.Value.(string)
	if got != "" {
		t.Fatalf("QueryRAGForTask() = %q, want empty string for empty task", got)
	}
}

func TestRag_truncateRunes_Ugly_NonPositiveLimitReturnsEmpty(t *testing.T) {
	for _, tc := range []struct {
		name  string
		limit int
	}{
		{name: "zero", limit: 0},
		{name: "negative", limit: -1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := truncateRunes("hello", tc.limit); got != "" {
				t.Fatalf("truncateRunes(%q, %d) = %q, want empty string", "hello", tc.limit, got)
			}
		})
	}
}

func TestRag_truncateRunes_Good_PreservesRuneBoundaries(t *testing.T) {
	got := truncateRunes("a😀bé文", 4)
	if got != "a😀bé" {
		t.Fatalf("truncateRunes() = %q, want %q", got, "a😀bé")
	}
}

// --- AX-7 canonical triplets ---

func TestRag_QueryRAGForTask_Good(t *core.T) {
	origNewQdrantClient := newQdrantClient
	origNewOllamaClient := newOllamaClient
	origRunRAGQuery := runRAGQuery
	t.Cleanup(func() {
		newQdrantClient = origNewQdrantClient
		newOllamaClient = origNewOllamaClient
		runRAGQuery = origRunRAGQuery
	})

	newQdrantClient = func(rag.QdrantConfig) core.Result { return core.Ok((*rag.QdrantClient)(nil)) }
	newOllamaClient = func(rag.OllamaConfig) core.Result { return core.Ok((*rag.OllamaClient)(nil)) }
	runRAGQuery = func(_ context.Context, _ rag.VectorStore, _ rag.Embedder, _ string, _ rag.QueryConfig) core.Result {
		return core.Ok([]rag.QueryResult{{Text: "Runbook", Source: "docs/build.md", Score: 0.9}})
	}

	result := QueryRAGForTask(TaskInfo{Title: "Investigate", Description: "failure"})
	got := result.Value.(string)
	core.AssertTrue(t, result.OK)
	core.AssertContains(t, got, "Runbook")
}

func TestRag_QueryRAGForTask_Bad(t *core.T) {
	result := QueryRAGForTask(TaskInfo{})
	got := result.Value.(string)
	want := ""

	core.AssertTrue(t, result.OK)
	core.AssertEqual(t, want, got)
}

func TestRag_QueryRAGForTask_Ugly(t *core.T) {
	origNewQdrantClient := newQdrantClient
	t.Cleanup(func() {
		newQdrantClient = origNewQdrantClient
	})
	newQdrantClient = func(rag.QdrantConfig) core.Result {
		return core.Fail(core.NewError("qdrant unavailable"))
	}

	result := QueryRAGForTask(TaskInfo{Title: "Investigate"})
	got := result.Value.(string)
	core.AssertTrue(t, result.OK)
	core.AssertEqual(t, "", got)
}
