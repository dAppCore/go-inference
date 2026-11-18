// Package ai provides the canonical AI facade for the core CLI.
//
// The RAG helpers (QueryRAGForTask, RAGContextAssembler) are disabled: see rag.go.
//
//	if err := ai.Record(ai.Event{Type: "security.scan", Repo: "wailsapp/wails"}); err != nil {
//		return err
//	}
package ai
