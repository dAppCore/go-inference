// RAG helpers for task-scoped documentation lookup.
package ai

import (
	"context"
	"time"
	"unicode/utf8"

	"dappco.re/go"
	rag "dappco.re/go/rag"
)

const (
	ragTaskCollection          = "hostuk-docs"
	ragTaskResultLimit         = 3
	ragTaskSimilarityThreshold = 0.5
	ragTaskQueryRuneLimit      = 500
)

var (
	newQdrantClient = func(cfg rag.QdrantConfig) core.Result {
		result := rag.NewQdrantClient(cfg)
		if !result.OK {
			return result
		}
		client, _ := result.Value.(*rag.QdrantClient)
		return core.Ok(client)
	}
	newOllamaClient = func(cfg rag.OllamaConfig) core.Result {
		result := rag.NewOllamaClient(cfg)
		if !result.OK {
			return result
		}
		client, _ := result.Value.(*rag.OllamaClient)
		return core.Ok(client)
	}
	runRAGQuery = func(ctx context.Context, store rag.VectorStore, embedder rag.Embedder, query string, cfg rag.QueryConfig) core.Result {
		result := rag.Query(ctx, store, embedder, query, cfg)
		if !result.OK {
			return result
		}
		results, _ := result.Value.([]rag.QueryResult)
		return core.Ok(results)
	}
	closeQdrant = func(client *rag.QdrantClient) core.Result { return client.Close() }
)

// ai.TaskInfo{Title: "Investigate build failure", Description: "CI compile step fails"} carries the minimal task data needed for RAG queries.
type TaskInfo struct {
	Title       string
	Description string
}

//	contextResult := ai.QueryRAGForTask(ai.TaskInfo{
//		Title:       "Investigate build failure",
//		Description: "CI compile step fails",
//	})
func QueryRAGForTask(task TaskInfo) core.Result {
	queryText := buildTaskQuery(task)
	if queryText == "" {
		return core.Ok("")
	}

	qdrantConfiguration := rag.DefaultQdrantConfig()
	qdrantResult := newQdrantClient(qdrantConfiguration)
	if !qdrantResult.OK {
		return core.Ok("")
	}
	qdrantClient, _ := qdrantResult.Value.(*rag.QdrantClient)
	if qdrantClient != nil {
		defer func() { closeQdrant(qdrantClient) }()
	}

	ollamaConfiguration := rag.DefaultOllamaConfig()
	ollamaResult := newOllamaClient(ollamaConfiguration)
	if !ollamaResult.OK {
		return core.Ok("")
	}
	ollamaClient, _ := ollamaResult.Value.(*rag.OllamaClient)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	queryConfiguration := rag.QueryConfig{
		Collection: ragTaskCollection,
		Limit:      ragTaskResultLimit,
		Threshold:  ragTaskSimilarityThreshold,
	}

	resultsResult := runRAGQuery(ctx, qdrantClient, ollamaClient, queryText, queryConfiguration)
	if !resultsResult.OK {
		return core.Ok("")
	}
	results, _ := resultsResult.Value.([]rag.QueryResult)
	if len(results) == 0 {
		return core.Ok("")
	}

	return core.Ok(rag.FormatResultsContext(results))
}

func buildTaskQuery(task TaskInfo) string {
	if core.Trim(task.Title) == "" && core.Trim(task.Description) == "" {
		return ""
	}

	return truncateRunes(task.Title+": "+task.Description, ragTaskQueryRuneLimit)
}

func truncateRunes(value string, limit int) string {
	if limit <= 0 {
		return ""
	}
	// Byte-length fast path: each rune uses ≥1 byte, so len(value) ≤ limit
	// implies RuneCount(value) ≤ limit. Skips utf8.RuneCountInString
	// entirely for ASCII-fits-budget inputs (the common case).
	if len(value) <= limit {
		return value
	}
	// Under-limit fast path: count runes without materialising a
	// []rune slice so the no-truncate branch stays zero-alloc.
	if core.RuneCount(value) <= limit {
		return value
	}
	// Clipping: walk runes via utf8.DecodeRuneInString and slice the
	// underlying bytes once. Avoids materialising a []rune (~4×len(value)
	// bytes) and the second string allocation.
	off, n := 0, 0
	for off < len(value) && n < limit {
		_, sz := utf8.DecodeRuneInString(value[off:])
		off += sz
		n++
	}
	return value[:off]
}
