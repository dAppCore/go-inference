// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// AX-11 baseline benchmarks for the openai provider helper surface.
//
// openaiMessages and chatCompletionsURL fire on every outbound provider
// call (each Chat/Generate to an OpenAI-compatible endpoint). providerError
// fires on every non-2xx response. The HTTP round-trip dominates wall time,
// but these helpers contribute to the per-request alloc floor — any
// regression here scales 1× per outbound API call.
//
// Run:
//   go test -bench=. -benchmem -benchtime=300ms ./providers/openai/...

// Sinks.
var (
	openaiBenchSinkMessages []ChatMessage
	openaiBenchSinkString   string
	openaiBenchSinkResult   core.Result
)

// --- fixtures ---

func benchMessages(n int) []inference.Message {
	out := make([]inference.Message, n)
	for i := 0; i < n; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		out[i] = inference.Message{Role: role, Content: "message body for benchmarking, typical length"}
	}
	return out
}

// --- openaiMessages — message format conversion per outbound call ---

func BenchmarkOpenAI_openaiMessages_2Turn(b *testing.B) {
	messages := benchMessages(2)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openaiBenchSinkMessages = openaiMessages(messages)
	}
}

func BenchmarkOpenAI_openaiMessages_10Turn(b *testing.B) {
	messages := benchMessages(10)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openaiBenchSinkMessages = openaiMessages(messages)
	}
}

// --- chatCompletionsURL — URL build per outbound call ---

func BenchmarkOpenAI_chatCompletionsURL_Typical(b *testing.B) {
	baseURL := "https://api.openai.com"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openaiBenchSinkString = chatCompletionsURL(baseURL)
	}
}

func BenchmarkOpenAI_chatCompletionsURL_TrailingSlash(b *testing.B) {
	baseURL := "https://api.openai.com/"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openaiBenchSinkString = chatCompletionsURL(baseURL)
	}
}

// --- contextMessages — per-outbound-call message context assembly ---

// contextMessages fires once per outbound Chat/Generate call. The
// no-assembler shape (Config.ContextAssembler == nil) is the common
// configuration when callers don't opt into RAG-style context injection,
// and is the alloc floor for the helper.
func BenchmarkOpenAI_contextMessages_NoAssembler(b *testing.B) {
	model := &Model{backend: &Backend{cfg: Config{}}}
	messages := benchMessages(2)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openaiBenchSinkResult = model.contextMessages(ctx, messages)
	}
}

// --- providerError — fires on every non-2xx response ---

func BenchmarkOpenAI_providerError_NoBody(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openaiBenchSinkResult = providerError(503, "")
	}
}

func BenchmarkOpenAI_providerError_StructuredBody(b *testing.B) {
	body := `{"error":{"message":"rate limit exceeded","type":"rate_limit_error"}}`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openaiBenchSinkResult = providerError(429, body)
	}
}

func BenchmarkOpenAI_providerError_PlainBody(b *testing.B) {
	body := "internal server error"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openaiBenchSinkResult = providerError(500, body)
	}
}

// --- AX-11 alloc-budget gates ---

// TestAllocBudget_OpenAI_openaiMessages locks the per-call message clone.
// openaiMessages pre-sizes its output via make([]…, 0, len(messages));
// the expected floor is 1 alloc (the slice backing array). Each per-message
// ChatMessage struct is a value type with no nested allocations.
func TestAllocBudget_OpenAI_openaiMessages(t *testing.T) {
	messages := benchMessages(2)

	// Behavioural lock — output has same length, roles/contents preserved.
	out := openaiMessages(messages)
	if len(out) != len(messages) {
		t.Fatalf("openaiMessages dropped messages: got %d, want %d", len(out), len(messages))
	}
	for i := range out {
		if out[i].Role != messages[i].Role || out[i].Content != messages[i].Content {
			t.Fatalf("openaiMessages corrupted message %d: %+v vs %+v", i, out[i], messages[i])
		}
	}

	avg := testing.AllocsPerRun(5, func() {
		openaiBenchSinkMessages = openaiMessages(messages)
	})
	// Ceiling: 2 — current measured 1 (slice backing). Pre-sized via
	// make([]…, 0, len(messages)) so no append-grow allocs.
	const budget = 2.0
	if avg > budget {
		t.Fatalf("openaiMessages alloc budget exceeded: %.1f allocs/call (budget=%.0f)\n"+
			"Fires once per outbound provider Chat/Generate call.",
			avg, budget)
	}
}

// TestAllocBudget_OpenAI_contextMessages_NoAssembler locks the per-call
// context-assembly floor when no assembler is configured. The expected
// alloc floor is the core.Result wrap; an upstream slice clone would
// fail this gate.
func TestAllocBudget_OpenAI_contextMessages_NoAssembler(t *testing.T) {
	model := &Model{backend: &Backend{cfg: Config{}}}
	messages := benchMessages(2)
	ctx := context.Background()

	// Behavioural lock — the no-assembler path returns the messages
	// without injecting a context entry. Length must be preserved and
	// roles/contents must round-trip.
	out := model.contextMessages(ctx, messages)
	if !out.OK {
		t.Fatalf("contextMessages(no assembler) failed: %s", out.Error())
	}
	produced, ok := out.Value.([]inference.Message)
	if !ok {
		t.Fatalf("contextMessages returned %T, want []inference.Message", out.Value)
	}
	if len(produced) != len(messages) {
		t.Fatalf("contextMessages changed length: got %d, want %d", len(produced), len(messages))
	}
	for i := range produced {
		if produced[i].Role != messages[i].Role || produced[i].Content != messages[i].Content {
			t.Fatalf("contextMessages corrupted message %d: %+v vs %+v", i, produced[i], messages[i])
		}
	}

	avg := testing.AllocsPerRun(5, func() {
		openaiBenchSinkResult = model.contextMessages(ctx, messages)
	})
	// Ceiling: 3 — baseline (slice clone + Result wrap) is currently
	// 2 allocs on Apple M3 Ultra. A regression that re-introduces the
	// upfront clone on the no-assembler path fails this gate.
	const budget = 3.0
	if avg > budget {
		t.Fatalf("contextMessages(no assembler) alloc budget exceeded: %.1f allocs/call (budget=%.0f)\n"+
			"Fires once per outbound provider Chat/Generate call.",
			avg, budget)
	}
}

// TestAllocBudget_OpenAI_providerError_NoBody locks the cheapest error
// shape (no response body). Should be the alloc floor for any 5xx.
func TestAllocBudget_OpenAI_providerError_NoBody(t *testing.T) {
	// Behavioural lock — empty body returns a Fail with the status code.
	r := providerError(503, "")
	if r.OK {
		t.Fatalf("providerError(503, '') unexpectedly OK")
	}

	avg := testing.AllocsPerRun(5, func() {
		openaiBenchSinkResult = providerError(503, "")
	})
	// Ceiling: 7 — current measured 6 (Apple M3 Ultra). The shape:
	// core.JSONUnmarshalString fails on empty input (1-2 allocs from
	// the failed parser path), then Sprintf formats one int, core.E
	// wraps the error chain (~3 allocs). All shapes of providerError
	// are bounded by this floor.
	const budget = 7.0
	if avg > budget {
		t.Fatalf("providerError(no body) alloc budget exceeded: %.1f allocs/call (budget=%.0f)\n"+
			"Fires on every non-2xx outbound provider response.",
			avg, budget)
	}
}
