// SPDX-Licence-Identifier: EUPL-1.2

// Allocation benchmarks for the budget package (RFC §6.13, §6.2, §6.11). A
// placement decision is made once per request on the routing hot path: Decide
// counts the prompt and grades it against an Endpoint, FitsWindow / FitsMemory
// are the pure predicates underneath it, and Decision.String keys logs and
// metrics (§3.2). One benchmark per public symbol; a realistic multi-turn
// prompt fixture and a zero-alloc Counter, so the numbers measure budgeting
// itself rather than the injected tokeniser.
//
// Run: go test -bench=. -benchmem -run='^$' ./budget/
package budget_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/serving/chat"
	"dappco.re/go/inference/kv/budget"
)

// Sinks defeat compiler dead-code elimination — every benchmarked call writes
// its result to a package-level sink of the matching type.
var (
	sinkBool   bool
	sinkString string
	sinkResult budget.Result
	sinkBudget *budget.Budget
)

// benchCounter sizes a prompt by walking each message's text blocks and summing
// their byte lengths — the shape a real tokeniser's pre-pass takes, but
// allocation-free, so a Decide benchmark measures the budgeting logic rather
// than the injected counter.
type benchCounter struct{}

func (benchCounter) Count(messages []chat.Message, _ string) int {
	total := 0
	for _, m := range messages {
		for _, blk := range m.Content {
			if blk.Kind == chat.KindText {
				total += len(blk.Text)
			}
		}
	}
	return total
}

// benchMessages — a realistic multi-turn transcript (system + developer + user
// + assistant + tool), the prompt shape Decide counts per request.
func benchMessages() []chat.Message {
	return []chat.Message{
		{Role: chat.System, Content: []chat.ContentBlock{chat.Text("You are a helpful assistant. Use UK English.")}},
		{Role: chat.Developer, Content: []chat.ContentBlock{chat.Text("Prefer concise answers.")}},
		{Role: chat.User, Content: []chat.ContentBlock{chat.Text("What's the weather in London today, and should I take an umbrella?")}},
		{Role: chat.Assistant, Content: []chat.ContentBlock{chat.Text("Let me check the forecast for you.")}},
		{Role: chat.Tool, ToolCallID: "call_weather_1", Content: []chat.ContentBlock{chat.Text("18C, light rain expected this afternoon")}},
	}
}

// benchEndpoint — a roomy local device the realistic prompt fits.
var benchEndpoint = budget.Endpoint{ContextLen: 8192, MemoryBudget: 96 << 30, BytesPerToken: 4}

// --- pure predicates (§6.11, §6.2) ---

func BenchmarkBudget_FitsWindow(b *core.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkBool = budget.FitsWindow(1000, 512, 8192)
	}
}

func BenchmarkBudget_FitsMemory(b *core.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkBool = budget.FitsMemory(1712, 4, 96<<30)
	}
}

// --- Decision.String (§3.2) ---

func BenchmarkBudget_Decision_String(b *core.B) {
	d := budget.DecisionNeedsLargerEndpoint
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkString = d.String()
	}
}

// --- constructor ---

func BenchmarkBudget_New(b *core.B) {
	c := benchCounter{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkBudget = budget.New(c)
	}
}

// --- Decide: the per-request hot path (§6.2/§6.16) ---

func BenchmarkBudget_Decide(b *core.B) {
	bud := budget.New(benchCounter{})
	msgs := benchMessages()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkResult = bud.Decide(msgs, "gemma-4-31b", 512, benchEndpoint)
	}
}

func BenchmarkBudget_Decide_NilCounter(b *core.B) {
	bud := budget.New(nil)
	msgs := benchMessages()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkResult = bud.Decide(msgs, "gemma-4-31b", 512, benchEndpoint)
	}
}
