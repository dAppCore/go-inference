// SPDX-License-Identifier: EUPL-1.2

package ai

import (
	"context"
	"slices"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// RAGContextAssembler adapts the package RAG helper to provider context
// injection.
type RAGContextAssembler struct {
	Task  TaskInfo
	Query func(TaskInfo) core.Result
}

// AssembleContext returns formatted retrieval context for the current chat.
func (a RAGContextAssembler) AssembleContext(_ context.Context, messages []inference.Message) core.Result {
	task := a.Task
	if core.Trim(task.Title) == "" && core.Trim(task.Description) == "" {
		task.Title = lastUserMessage(messages)
	}
	if core.Trim(task.Title) == "" && core.Trim(task.Description) == "" {
		return core.Ok("")
	}
	query := a.Query
	if query == nil {
		query = QueryRAGForTask
	}
	result := query(task)
	if !result.OK {
		return result
	}
	contextText, _ := result.Value.(string)
	return core.Ok(contextText)
}

func lastUserMessage(messages []inference.Message) string {
	for _, message := range slices.Backward(messages) {
		if core.Lower(core.Trim(message.Role)) == "user" {
			return core.Trim(message.Content)
		}
	}
	return ""
}
