// SPDX-License-Identifier: EUPL-1.2

package ai

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestContext_RAGContextAssembler_Good_UsesLastUserMessage(t *testing.T) {
	assembler := RAGContextAssembler{
		Query: func(task TaskInfo) core.Result {
			if task.Title != "How do I fix this build?" {
				t.Fatalf("task title = %q, want last user message", task.Title)
			}
			return core.Ok("build runbook context")
		},
	}

	result := assembler.AssembleContext(context.Background(), []inference.Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "How do I fix this build?"},
	})
	if !result.OK {
		t.Fatalf("AssembleContext() error = %s", result.Error())
	}
	got, _ := result.Value.(string)
	if got != "build runbook context" {
		t.Fatalf("AssembleContext() = %q, want build runbook context", got)
	}
}

func TestContext_RAGContextAssembler_Bad_BlankMessagesSkipQuery(t *testing.T) {
	called := false
	assembler := RAGContextAssembler{
		Query: func(TaskInfo) core.Result {
			called = true
			return core.Ok("unexpected")
		},
	}

	result := assembler.AssembleContext(context.Background(), []inference.Message{{Role: "user", Content: "   "}})
	if !result.OK {
		t.Fatalf("AssembleContext() error = %s", result.Error())
	}
	got, _ := result.Value.(string)
	if got != "" {
		t.Fatalf("AssembleContext() = %q, want empty context", got)
	}
	if called {
		t.Fatal("AssembleContext() called query for blank messages")
	}
}

func TestContext_RAGContextAssembler_AssembleContext_Good(t *testing.T) {
	assembler := RAGContextAssembler{Query: func(TaskInfo) core.Result {
		return core.Ok("context")
	}}
	result := assembler.AssembleContext(context.Background(), []inference.Message{{Role: "user", Content: "question"}})

	if !result.OK || result.Value.(string) != "context" {
		t.Fatalf("RAGContextAssembler.AssembleContext() = %#v, want context", result)
	}
}

func TestContext_RAGContextAssembler_AssembleContext_Bad(t *testing.T) {
	assembler := RAGContextAssembler{Query: func(TaskInfo) core.Result {
		return core.Fail(core.E("test.rag", "query failed", nil))
	}}
	result := assembler.AssembleContext(context.Background(), []inference.Message{{Role: "user", Content: "question"}})

	if result.OK || !core.Contains(result.Error(), "query failed") {
		t.Fatalf("RAGContextAssembler.AssembleContext() = %#v, want query failure", result)
	}
}

func TestContext_RAGContextAssembler_AssembleContext_Ugly(t *testing.T) {
	called := false
	assembler := RAGContextAssembler{Query: func(TaskInfo) core.Result {
		called = true
		return core.Ok("unexpected")
	}}
	result := assembler.AssembleContext(context.Background(), []inference.Message{{Role: "user", Content: "   "}})

	if !result.OK || result.Value.(string) != "" || called {
		t.Fatalf("RAGContextAssembler.AssembleContext() = %#v called=%v, want blank short-circuit", result, called)
	}
}
