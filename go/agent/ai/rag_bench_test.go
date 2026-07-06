// SPDX-Licence-Identifier: EUPL-1.2

package ai

import (
	"context"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// AX-11 baseline benchmarks for the ai/rag + ai/context helpers.
//
// buildTaskQuery / truncateRunes / lastUserMessage all fire on the
// per-request context-assembly path — every chat that goes through
// RAGContextAssembler.AssembleContext pays this. The dominant cost
// of QueryRAGForTask itself is the qdrant + ollama RTT, but these
// pure helpers govern the alloc floor in the request-prep phase.
//
// Run:
//   go test -bench=. -benchmem -benchtime=300ms ./ai/...

// Sinks.
var (
	ragBenchSinkString string
	ragBenchSinkResult core.Result
)

// --- fixtures ---

func benchTaskInfo() TaskInfo {
	return TaskInfo{
		Title:       "Investigate CI build failure on macOS",
		Description: "The cgo build step fails with linker errors on the M3 Ultra runner after the Wails upgrade.",
	}
}

func benchTaskInfoLong() TaskInfo {
	long := strings.Repeat("paragraph of meaningful text that will exceed the rune limit by a comfortable margin. ", 20)
	return TaskInfo{Title: "long form research task", Description: long}
}

func benchUserMessages(n int) []inference.Message {
	out := make([]inference.Message, 0, n)
	for range n {
		out = append(out, inference.Message{Role: "system", Content: "context"})
		out = append(out, inference.Message{Role: "assistant", Content: "assistant response"})
	}
	out = append(out, inference.Message{Role: "user", Content: "the last user message we want to find"})
	return out
}

// --- buildTaskQuery — per-RAG-call task→query string ---

func BenchmarkRAG_buildTaskQuery_Typical(b *testing.B) {
	task := benchTaskInfo()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ragBenchSinkString = buildTaskQuery(task)
	}
}

func BenchmarkRAG_buildTaskQuery_Long(b *testing.B) {
	task := benchTaskInfoLong()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ragBenchSinkString = buildTaskQuery(task)
	}
}

func BenchmarkRAG_buildTaskQuery_Empty(b *testing.B) {
	task := TaskInfo{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ragBenchSinkString = buildTaskQuery(task)
	}
}

// --- truncateRunes — pure rune-clipping helper ---

func BenchmarkRAG_truncateRunes_NoTruncate(b *testing.B) {
	s := "short string well under the limit"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ragBenchSinkString = truncateRunes(s, 500)
	}
}

func BenchmarkRAG_truncateRunes_Clipped(b *testing.B) {
	s := strings.Repeat("a long body that needs clipping ", 50)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ragBenchSinkString = truncateRunes(s, 500)
	}
}

// --- lastUserMessage — per-AssembleContext linear scan ---

func BenchmarkRAG_lastUserMessage_LastIsUser(b *testing.B) {
	messages := benchUserMessages(5)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ragBenchSinkString = lastUserMessage(messages)
	}
}

func BenchmarkRAG_lastUserMessage_NoUser(b *testing.B) {
	messages := []inference.Message{
		{Role: "system", Content: "policy"},
		{Role: "assistant", Content: "response"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ragBenchSinkString = lastUserMessage(messages)
	}
}

// --- AssembleContext — per-Chat context assembly entry point ---

func BenchmarkRAG_AssembleContext_NoQueryHit(b *testing.B) {
	// Query stub that returns empty (simulates no matching docs).
	assembler := RAGContextAssembler{
		Task: benchTaskInfo(),
		Query: func(TaskInfo) core.Result {
			return core.Ok("")
		},
	}
	messages := []inference.Message{
		{Role: "user", Content: "user prompt"},
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ragBenchSinkResult = assembler.AssembleContext(ctx, messages)
	}
}

// --- AX-11 alloc-budget gates ---

// TestAllocBudget_RAG_buildTaskQuery locks the per-call task→query
// string build. Fires once per QueryRAGForTask / AssembleContext call.
func TestAllocBudget_RAG_buildTaskQuery(t *testing.T) {
	task := benchTaskInfo()

	// Behavioural lock — typical query is "Title: Description" form.
	out := buildTaskQuery(task)
	if out == "" {
		t.Fatalf("buildTaskQuery returned empty for non-empty task")
	}
	if !strings.Contains(out, "Investigate") || !strings.Contains(out, "cgo") {
		t.Fatalf("buildTaskQuery dropped content: %q", out)
	}

	avg := testing.AllocsPerRun(5, func() {
		ragBenchSinkString = buildTaskQuery(task)
	})
	// Ceiling: 1 — string concat allocates the joined backing.
	// truncateRunes under-limit fast path is zero-alloc (uses
	// core.RuneCount), so the only alloc is the Title+": "+Description
	// concat itself. Locks the per-chat-request floor.
	const budget = 1.0
	if avg > budget {
		t.Fatalf("buildTaskQuery alloc budget exceeded: %.1f allocs/call (budget=%.0f)\n"+
			"Fires once per RAG context assembly — per-chat-request floor.",
			avg, budget)
	}
}

// TestAllocBudget_RAG_truncateRunes_NoTruncate locks the under-limit
// fast path. When input fits, function returns the input string
// directly — should be zero allocs.
func TestAllocBudget_RAG_truncateRunes_NoTruncate(t *testing.T) {
	s := "short string well under the limit"

	// Behavioural lock — under-limit returns input verbatim.
	out := truncateRunes(s, 500)
	if out != s {
		t.Fatalf("truncateRunes mutated under-limit input: %q vs %q", out, s)
	}

	avg := testing.AllocsPerRun(5, func() {
		ragBenchSinkString = truncateRunes(s, 500)
	})
	// Ceiling: 0 — under-limit fast path uses core.RuneCount
	// (utf8.RuneCountInString) so the count check itself does
	// not allocate. Locks the contract: under-limit MUST stay
	// zero-alloc; any caller that hot-path-truncates pays only
	// for the explicit clipping branch.
	const budget = 0.0
	if avg > budget {
		t.Fatalf("truncateRunes(no truncate) alloc budget exceeded: %.1f allocs/call (budget=%.0f)",
			avg, budget)
	}
}

// TestAllocBudget_RAG_lastUserMessage locks the linear scan. Per-call
// alloc should be zero — function returns substrings from the input.
func TestAllocBudget_RAG_lastUserMessage(t *testing.T) {
	messages := benchUserMessages(5)

	// Behavioural lock — finds the last user-role message.
	out := lastUserMessage(messages)
	if out != "the last user message we want to find" {
		t.Fatalf("lastUserMessage wrong result: %q", out)
	}

	avg := testing.AllocsPerRun(5, func() {
		ragBenchSinkString = lastUserMessage(messages)
	})
	// Ceiling: 0 — pure read + return. core.Lower may allocate when
	// case conversion is needed, but role is already lowercase in
	// the fixture so the fast path applies.
	const budget = 0.0
	if avg > budget {
		t.Fatalf("lastUserMessage alloc budget exceeded: %.1f allocs/call (budget=%.0f)",
			avg, budget)
	}
}
