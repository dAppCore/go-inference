// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the Anthropic Messages wire primitives.
// Per AX-11 — Marshal/Unmarshal of MessageRequest/MessageResponse fires
// once per Messages call, and InferenceMessages / GenerateOptions run
// at request-entry on every served chat turn. blockText is the
// per-content-block inner loop that runs over every message in the
// request transcript on every call.
//
// Run:    go test -bench='BenchmarkAnthropic' -benchtime=100ms -benchmem -run='^$' .

package anthropic

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// Sinks defeat compiler DCE.
var (
	anthropicSinkRequest  MessageRequest
	anthropicSinkResponse MessageResponse
	anthropicSinkMessages []inference.Message
	anthropicSinkOptions  []inference.GenerateOption
	anthropicSinkResult   core.Result
	anthropicSinkString   string
	anthropicSinkText     string
	anthropicSinkBytes    []byte
)

// --- Fixture builders ---

// buildAnthropicRequest produces a representative system+user+assistant
// transcript with the requested number of message turns. Each user
// message carries the typical short query shape; assistant turns carry
// longer multi-paragraph completions.
func buildAnthropicRequest(turns int) MessageRequest {
	temp := float32(0.7)
	topP := float32(0.95)
	topK := 64
	req := MessageRequest{
		Model:         "claude-3-5-sonnet",
		System:        "You are a helpful assistant. Be concise.",
		MaxTokens:     1024,
		Temperature:   &temp,
		TopP:          &topP,
		TopK:          &topK,
		StopSequences: []string{"</response>", "<|eot_id|>"},
	}
	user := "Please summarise the following short paragraph for me in one sentence."
	assistant := "The summary is concise and faithful to the original text. " +
		"It preserves the principal claim and the supporting detail without padding."
	for i := 0; i < turns; i++ {
		role := "user"
		text := user
		if i%2 == 1 {
			role = "assistant"
			text = assistant
		}
		req.Messages = append(req.Messages, Message{
			Role:    role,
			Content: []ContentBlock{{Type: "text", Text: text}},
		})
	}
	return req
}

// buildAnthropicResponse mirrors a real completion — multi-block text
// content with a trailing usage block.
func buildAnthropicResponse() MessageResponse {
	return NewTextResponse(
		"msg_bench",
		"claude-3-5-sonnet",
		"The summary is concise and faithful to the original text.",
		inference.GenerateMetrics{PromptTokens: 320, GeneratedTokens: 48},
	)
}

// --- JSON Marshal — fires at response emission ---

func BenchmarkAnthropic_MarshalMessageRequest_SingleTurn(b *testing.B) {
	req := buildAnthropicRequest(1)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		anthropicSinkString = core.JSONMarshalString(req)
	}
}

func BenchmarkAnthropic_MarshalMessageRequest_FiveTurn(b *testing.B) {
	req := buildAnthropicRequest(5)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		anthropicSinkString = core.JSONMarshalString(req)
	}
}

func BenchmarkAnthropic_MarshalMessageRequest_TwentyTurn(b *testing.B) {
	req := buildAnthropicRequest(20)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		anthropicSinkString = core.JSONMarshalString(req)
	}
}

func BenchmarkAnthropic_MarshalMessageResponse_Typical(b *testing.B) {
	resp := buildAnthropicResponse()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		anthropicSinkString = core.JSONMarshalString(resp)
	}
}

// --- Hand-rolled AppendMessageResponse — bypasses json.Marshal
// reflect path. Wins are visible when consumers reach for the helper
// directly (HTTP-response-emit), not when measured via JSONMarshalString.
// Per-W9-D pattern: caller pre-sizes the buffer once via the
// MessageResponseSize estimator so encoding lands at 1 alloc.

func BenchmarkAnthropic_AppendMessageResponse_Typical(b *testing.B) {
	resp := buildAnthropicResponse()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		anthropicSinkBytes = AppendMessageResponse(make([]byte, 0, MessageResponseSize(resp)), resp)
	}
}

func BenchmarkAnthropic_AppendMessageResponse_WithStopReason(b *testing.B) {
	resp := buildAnthropicResponse()
	resp.StopReason = "stop_sequence"
	resp.StopSequence = "</response>"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		anthropicSinkBytes = AppendMessageResponse(make([]byte, 0, MessageResponseSize(resp)), resp)
	}
}

// --- JSON Unmarshal — fires at request entry ---

func BenchmarkAnthropic_UnmarshalMessageRequest_SingleTurn(b *testing.B) {
	body := core.JSONMarshalString(buildAnthropicRequest(1))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var req MessageRequest
		anthropicSinkResult = core.JSONUnmarshalString(body, &req)
		anthropicSinkRequest = req
	}
}

func BenchmarkAnthropic_UnmarshalMessageRequest_FiveTurn(b *testing.B) {
	body := core.JSONMarshalString(buildAnthropicRequest(5))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var req MessageRequest
		anthropicSinkResult = core.JSONUnmarshalString(body, &req)
		anthropicSinkRequest = req
	}
}

func BenchmarkAnthropic_UnmarshalMessageRequest_TwentyTurn(b *testing.B) {
	body := core.JSONMarshalString(buildAnthropicRequest(20))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var req MessageRequest
		anthropicSinkResult = core.JSONUnmarshalString(body, &req)
		anthropicSinkRequest = req
	}
}

func BenchmarkAnthropic_UnmarshalMessageResponse_Typical(b *testing.B) {
	body := core.JSONMarshalString(buildAnthropicResponse())
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var resp MessageResponse
		anthropicSinkResult = core.JSONUnmarshalString(body, &resp)
		anthropicSinkResponse = resp
	}
}

// --- InferenceMessages — wire→internal conversion fired per request ---

func BenchmarkAnthropic_InferenceMessages_SingleTurn(b *testing.B) {
	req := buildAnthropicRequest(1)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		anthropicSinkMessages = InferenceMessages(req)
	}
}

func BenchmarkAnthropic_InferenceMessages_FiveTurn(b *testing.B) {
	req := buildAnthropicRequest(5)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		anthropicSinkMessages = InferenceMessages(req)
	}
}

func BenchmarkAnthropic_InferenceMessages_TwentyTurn(b *testing.B) {
	req := buildAnthropicRequest(20)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		anthropicSinkMessages = InferenceMessages(req)
	}
}

// --- GenerateOptions — sampling-field projection fired per request ---

func BenchmarkAnthropic_GenerateOptions_AllFieldsSet(b *testing.B) {
	req := buildAnthropicRequest(1)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		anthropicSinkOptions = GenerateOptions(req)
	}
}

func BenchmarkAnthropic_GenerateOptions_MinimalFields(b *testing.B) {
	req := MessageRequest{Model: "claude-3-5-sonnet", MaxTokens: 256}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		anthropicSinkOptions = GenerateOptions(req)
	}
}

// --- NewTextResponse — fires once per non-streaming completion ---

func BenchmarkAnthropic_NewTextResponse(b *testing.B) {
	metrics := inference.GenerateMetrics{PromptTokens: 320, GeneratedTokens: 48}
	text := "The summary is concise and faithful to the original text."
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		anthropicSinkResponse = NewTextResponse("msg_bench", "claude-3-5-sonnet", text, metrics)
	}
}

// --- blockText — per-content-block inner loop (unexported; reached via
// InferenceMessages but worth a direct bench at the boundary shape). ---
// Single text block — the dominant production shape.

func BenchmarkAnthropic_BlockText_SingleTextBlock(b *testing.B) {
	blocks := []ContentBlock{{Type: "text", Text: "hello world"}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		anthropicSinkText = blockText(blocks)
	}
}

// Multi-block — the streamed-back shape with prompt caching headers
// splitting an instruction prefix from the user payload.
func BenchmarkAnthropic_BlockText_FiveBlocks(b *testing.B) {
	blocks := []ContentBlock{
		{Type: "text", Text: "You are a helpful assistant. "},
		{Type: "text", Text: "Always respond in UK English. "},
		{Type: "text", Text: "Be concise. "},
		{Type: "text", Text: "Summarise the following paragraph: "},
		{Type: "text", Text: "The quick brown fox jumps over the lazy dog."},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		anthropicSinkText = blockText(blocks)
	}
}
