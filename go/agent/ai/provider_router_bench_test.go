// SPDX-Licence-Identifier: EUPL-1.2

package ai

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// AX-11 baseline benchmarks for the ai/provider_router hot path.
//
// Every routed Chat call shells through Chat() which calls
// normalisedMessages, generateOptions, contextMessages, and chatProvider
// in sequence. The router IS the per-request floor — a regression here
// scales 1× per inbound chat request across every consumer of the inference stack.
//
// Hot table:
//   - Chat (whole-call cost; bench against a synchronous fake model)
//   - normalisedMessages (per-call message slice clone)
//   - generateOptions (per-call options slice build)
//   - contextMessages (per-call context assembly)
//   - cloneProviderRoute (per-call when listing providers)
//
// Run:
//   go test -bench=. -benchmem -benchtime=300ms ./ai/...

// Sinks.
var (
	routerBenchSinkResult   core.Result
	routerBenchSinkMessages []inference.Message
	routerBenchSinkOptions  []inference.GenerateOption
	routerBenchSinkRoute    ProviderRoute
)

// --- fixtures ---

func benchProviderRequest() ProviderChatRequest {
	return ProviderChatRequest{
		Messages: []inference.Message{
			{Role: "system", Content: "You are helpful."},
			{Role: "user", Content: "What is the capital of France?"},
		},
		MaxTokens:   128,
		Temperature: 0.7,
		TopP:        0.9,
	}
}

func benchRouter(b *testing.B) *ProviderRouter {
	b.Helper()
	model := &routerFakeModel{
		modelType: "bench-model",
		output:    "Paris",
	}
	result := NewProviderRouter(ProviderRoute{
		Name:    "primary",
		ModelID: "bench-model",
		Model:   model,
	})
	if !result.OK {
		b.Fatalf("NewProviderRouter: %v", result.Error())
	}
	return result.Value.(*ProviderRouter)
}

// --- Chat — whole-call per-request cost ---

func BenchmarkProviderRouter_Chat_Typical(b *testing.B) {
	router := benchRouter(b)
	req := benchProviderRequest()
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		routerBenchSinkResult = router.Chat(ctx, req)
	}
}

// BenchmarkProviderRouter_Chat_Stream_50Tokens fires a streaming
// chat that yields 50 separate tokens — captures the per-token
// text-aggregation alloc shape in chatProvider. A 50-token reply
// is short for a real chat (typical responses are 200-1000+ tokens),
// but enough to surface O(N) vs O(N^2) growth differences.
func BenchmarkProviderRouter_Chat_Stream_50Tokens(b *testing.B) {
	tokens := make([]string, 50)
	for i := range tokens {
		tokens[i] = "tok "
	}
	model := &routerFakeModel{modelType: "bench-stream", tokens: tokens}
	result := NewProviderRouter(ProviderRoute{
		Name:    "primary",
		ModelID: "bench-stream",
		Model:   model,
	})
	if !result.OK {
		b.Fatalf("NewProviderRouter: %v", result.Error())
	}
	router := result.Value.(*ProviderRouter)
	req := benchProviderRequest()
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		routerBenchSinkResult = router.Chat(ctx, req)
	}
}

// --- normalisedMessages — per-call message clone ---

func BenchmarkProviderRouter_normalisedMessages_Typical(b *testing.B) {
	req := benchProviderRequest()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		routerBenchSinkMessages = req.normalisedMessages()
	}
}

// --- generateOptions — per-call options slice ---

func BenchmarkProviderRouter_generateOptions_Typical(b *testing.B) {
	req := benchProviderRequest()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		routerBenchSinkOptions = req.generateOptions()
	}
}

func BenchmarkProviderRouter_generateOptions_Empty(b *testing.B) {
	req := ProviderChatRequest{
		Messages: []inference.Message{{Role: "user", Content: "hi"}},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		routerBenchSinkOptions = req.generateOptions()
	}
}

// --- cloneProviderRoute — per-Providers-call route copy ---

func BenchmarkProviderRouter_cloneProviderRoute_NoLabels(b *testing.B) {
	route := ProviderRoute{
		Name:    "primary",
		ModelID: "bench-model",
		Model:   &routerFakeModel{modelType: "bench"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		routerBenchSinkRoute = cloneProviderRoute(route)
	}
}

func BenchmarkProviderRouter_cloneProviderRoute_WithLabels(b *testing.B) {
	route := ProviderRoute{
		Name:    "primary",
		ModelID: "bench-model",
		Model:   &routerFakeModel{modelType: "bench"},
		Labels:  map[string]string{"tier": "free", "region": "eu", "tenant": "default"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		routerBenchSinkRoute = cloneProviderRoute(route)
	}
}

// --- AX-11 alloc-budget gates ---

// TestAllocBudget_Router_normalisedMessages locks the per-call message-clone
// alloc count. This runs once per Chat() invocation; a regression that
// adds an alloc here scales 1× per inbound request.
func TestAllocBudget_Router_normalisedMessages(t *testing.T) {
	req := benchProviderRequest()

	// Behavioural lock — output is a fresh slice (mutating the result
	// doesn't affect req.Messages).
	out := req.normalisedMessages()
	if len(out) != len(req.Messages) {
		t.Fatalf("normalisedMessages dropped messages: got %d, want %d", len(out), len(req.Messages))
	}
	out[0].Content = "mutate"
	if req.Messages[0].Content == "mutate" {
		t.Fatalf("normalisedMessages did not clone — mutation leaked")
	}

	avg := testing.AllocsPerRun(5, func() {
		routerBenchSinkMessages = req.normalisedMessages()
	})
	// Ceiling: 2 — current measured 1 (Apple M3 Ultra: slice
	// backing array). The append([]inference.Message(nil), …) builds
	// a fresh slice; that's one alloc, the floor for this shape.
	const budget = 2.0
	if avg > budget {
		t.Fatalf("normalisedMessages alloc budget exceeded: %.1f allocs/call (budget=%.0f)\n"+
			"Fires once per Chat() — scales per inbound chat request.",
			avg, budget)
	}
}

// TestAllocBudget_Router_generateOptions locks the per-call options
// slice build. With 4 of 4 non-zero scalar opts set, expect ≤ 2 allocs
// (slice backing + per-option closures from inference.With*).
func TestAllocBudget_Router_generateOptions(t *testing.T) {
	req := benchProviderRequest()

	// Behavioural lock — len reflects which fields are non-zero.
	out := req.generateOptions()
	if len(out) != 3 {
		t.Fatalf("generateOptions: got %d opts, want 3 (MaxTokens + Temperature + TopP)", len(out))
	}

	avg := testing.AllocsPerRun(5, func() {
		routerBenchSinkOptions = req.generateOptions()
	})
	// Ceiling: 6 — current measured 4 (Apple M3 Ultra: slice + 3
	// closure boxes from inference.With* wrappers). The slice is
	// pre-sized via len(r.Options)+4 so no append-grow allocs.
	const budget = 6.0
	if avg > budget {
		t.Fatalf("generateOptions alloc budget exceeded: %.1f allocs/call (budget=%.0f)\n"+
			"Fires once per Chat() — per-request floor.",
			avg, budget)
	}
}

// TestAllocBudget_Router_cloneProviderRoute_NoLabels locks the route
// clone when there are no labels. Should be zero allocs — the struct
// is a value type and Labels is a nil map (no clone needed).
func TestAllocBudget_Router_cloneProviderRoute_NoLabels(t *testing.T) {
	route := ProviderRoute{
		Name:    "primary",
		ModelID: "bench-model",
		Model:   &routerFakeModel{modelType: "bench"},
	}

	// Behavioural lock — cloning preserves the route shape.
	cloned := cloneProviderRoute(route)
	if cloned.Name != route.Name || cloned.ModelID != route.ModelID {
		t.Fatalf("cloneProviderRoute dropped scalar fields")
	}
	if cloned.Labels != nil {
		t.Fatalf("cloneProviderRoute should leave nil Labels nil, got %v", cloned.Labels)
	}

	avg := testing.AllocsPerRun(5, func() {
		routerBenchSinkRoute = cloneProviderRoute(route)
	})
	// Ceiling: 0 — current measured 0. core.MapClone on a nil map
	// must return nil without allocation; if it doesn't, fix the
	// upstream helper.
	const budget = 0.0
	if avg > budget {
		t.Fatalf("cloneProviderRoute(no labels) alloc budget exceeded: %.1f allocs/call (budget=%.0f)\n"+
			"core.MapClone(nil) must be zero-alloc.",
			avg, budget)
	}
}
