// SPDX-License-Identifier: EUPL-1.2

// RAG disabled (2026-11-18): go-rag is retired and temporal Markov graphs replace
// it. Kept for reference. Delete this constraint to re-enable.
//go:build ignore

package ai

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func ExampleRAGContextAssembler() {
	assembler := RAGContextAssembler{
		Query: func(task TaskInfo) core.Result {
			return core.Ok(core.Concat("context for ", task.Title))
		},
	}

	contextResult := assembler.AssembleContext(context.Background(), []inference.Message{
		{Role: "user", Content: "build failure"},
	})
	contextText := contextResult.Value.(string)
	core.Println(contextText)

	// Output:
	// context for build failure
}

func ExampleRAGContextAssembler_AssembleContext() {
	assembler := RAGContextAssembler{
		Query: func(task TaskInfo) core.Result {
			return core.Ok(core.Concat("context for ", task.Title))
		},
	}
	result := assembler.AssembleContext(context.Background(), []inference.Message{{Role: "user", Content: "incident"}})

	core.Println(result.Value.(string))
	// Output:
	// context for incident
}
