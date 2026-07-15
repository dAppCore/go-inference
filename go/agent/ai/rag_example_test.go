package ai

import (
	"context"

	core "dappco.re/go"
	rag "dappco.re/go/rag"
)

func ExampleQueryRAGForTask() {
	origNewQdrantClient := newQdrantClient
	origNewOllamaClient := newOllamaClient
	origRunRAGQuery := runRAGQuery
	origCloseQdrant := closeQdrant
	defer func() {
		newQdrantClient = origNewQdrantClient
		newOllamaClient = origNewOllamaClient
		runRAGQuery = origRunRAGQuery
		closeQdrant = origCloseQdrant
	}()

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
		return core.Ok([]rag.QueryResult{{Text: "Use the build runbook", Source: "docs/build.md", Section: "Checks", Score: 0.9}})
	}

	result := QueryRAGForTask(TaskInfo{Title: "Investigate build failure", Description: "CI failed"})
	contextText := result.Value.(string)

	core.Println(result.OK)
	core.Println(core.Contains(contextText, "Use the build runbook"))
	// Output:
	// true
	// true
}
