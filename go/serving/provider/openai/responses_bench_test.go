// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the OpenAI-compatible Responses wire primitives.
// Per AX-11 — the Responses endpoint is the OpenAI v1/responses path
// served by both the local runtime and proxy clients. These fixtures
// exercise the JSON ingress/egress, the wire→inference message
// projection, and the per-event stream marshal that fires per token in
// the response stream.
//
// Run:    go test -bench='BenchmarkResponses' -benchtime=100ms -benchmem -run='^$' .

package openai

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// Sinks defeat compiler DCE.
var (
	responsesSinkRequest  ResponseRequest
	responsesSinkResponse Response
	responsesSinkEvent    ResponseStreamEvent
	responsesSinkMessages []inference.Message
	responsesSinkOptions  []inference.GenerateOption
	responsesSinkErr      error
	responsesSinkString   string
	responsesSinkBytes    []byte
	responsesSinkResult   core.Result
)

// --- Fixture builders ---

// buildResponseRequest produces a representative Responses payload with
// the requested turn count. Mirrors what the v1/responses handler
// decodes at request entry.
func buildResponseRequest(turns int) ResponseRequest {
	temperature := float32(0.7)
	topP := float32(0.95)
	topK := 64
	maxOutputTokens := 256
	req := ResponseRequest{
		Model:           "qwen3",
		Instructions:    "You are a helpful assistant. Be concise.",
		Temperature:     &temperature,
		TopP:            &topP,
		TopK:            &topK,
		MaxOutputTokens: &maxOutputTokens,
		Stream:          true,
		Stop:            StopList{"<|im_end|>"},
	}
	for i := range turns {
		if i%2 == 0 {
			req.Input = append(req.Input, ResponseInputMessage{Role: "user", Content: "Summarise the paragraph in one sentence."})
		} else {
			req.Input = append(req.Input, ResponseInputMessage{Role: "assistant", Content: "The summary captures the key claim."})
		}
	}
	return req
}

// buildResponse mirrors a completed Responses body.
func buildResponse() Response {
	return NewTextResponse(
		"resp_bench",
		"qwen3",
		"The summary is concise and faithful to the original text.",
		inference.GenerateMetrics{PromptTokens: 200, GeneratedTokens: 32},
	)
}

// --- JSON Marshal ---

func BenchmarkResponses_MarshalRequest_SingleTurn(b *testing.B) {
	req := buildResponseRequest(1)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		responsesSinkString = core.JSONMarshalString(req)
	}
}

func BenchmarkResponses_MarshalRequest_FiveTurn(b *testing.B) {
	req := buildResponseRequest(5)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		responsesSinkString = core.JSONMarshalString(req)
	}
}

func BenchmarkResponses_MarshalRequest_TwentyTurn(b *testing.B) {
	req := buildResponseRequest(20)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		responsesSinkString = core.JSONMarshalString(req)
	}
}

func BenchmarkResponses_MarshalResponse_Typical(b *testing.B) {
	resp := buildResponse()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		responsesSinkString = core.JSONMarshalString(resp)
	}
}

// --- JSON Unmarshal ---

func BenchmarkResponses_UnmarshalRequest_SingleTurn(b *testing.B) {
	body := core.JSONMarshalString(buildResponseRequest(1))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var req ResponseRequest
		responsesSinkResult = core.JSONUnmarshalString(body, &req)
		responsesSinkRequest = req
	}
}

func BenchmarkResponses_UnmarshalRequest_FiveTurn(b *testing.B) {
	body := core.JSONMarshalString(buildResponseRequest(5))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var req ResponseRequest
		responsesSinkResult = core.JSONUnmarshalString(body, &req)
		responsesSinkRequest = req
	}
}

func BenchmarkResponses_UnmarshalRequest_TwentyTurn(b *testing.B) {
	body := core.JSONMarshalString(buildResponseRequest(20))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var req ResponseRequest
		responsesSinkResult = core.JSONUnmarshalString(body, &req)
		responsesSinkRequest = req
	}
}

func BenchmarkResponses_UnmarshalResponse_Typical(b *testing.B) {
	body := core.JSONMarshalString(buildResponse())
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var resp Response
		responsesSinkResult = core.JSONUnmarshalString(body, &resp)
		responsesSinkResponse = resp
	}
}

// --- ResponseMessages — wire→internal conversion per request ---

func BenchmarkResponses_ResponseMessages_SingleTurn(b *testing.B) {
	req := buildResponseRequest(1)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		responsesSinkMessages = ResponseMessages(req)
	}
}

func BenchmarkResponses_ResponseMessages_FiveTurn(b *testing.B) {
	req := buildResponseRequest(5)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		responsesSinkMessages = ResponseMessages(req)
	}
}

func BenchmarkResponses_ResponseMessages_TwentyTurn(b *testing.B) {
	req := buildResponseRequest(20)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		responsesSinkMessages = ResponseMessages(req)
	}
}

func BenchmarkResponses_ResponseMessages_InstructionsOnly(b *testing.B) {
	req := ResponseRequest{Model: "qwen3", Instructions: "Be concise."}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		responsesSinkMessages = ResponseMessages(req)
	}
}

// --- ResponseGenerateOptions — request-time sampling projection ---

func BenchmarkResponses_GenerateOptions_AllFieldsSet(b *testing.B) {
	req := buildResponseRequest(1)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		responsesSinkOptions, responsesSinkErr = ResponseGenerateOptions(req)
	}
}

// Instructions-only path — exercises the empty-input fallback branch
// that synthesises a ChatMessage from req.Instructions.
func BenchmarkResponses_GenerateOptions_InstructionsOnly(b *testing.B) {
	req := ResponseRequest{Model: "qwen3", Instructions: "Be concise."}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		responsesSinkOptions, responsesSinkErr = ResponseGenerateOptions(req)
	}
}

// --- NewTextResponse — fired once per non-streaming completion ---

func BenchmarkResponses_NewTextResponse(b *testing.B) {
	metrics := inference.GenerateMetrics{PromptTokens: 200, GeneratedTokens: 32}
	text := "The summary is concise and faithful to the original text."
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		responsesSinkResponse = NewTextResponse("resp_bench", "qwen3", text, metrics)
	}
}

// --- ResponseStreamEvent marshal — fired per streamed delta + final ---

func BenchmarkResponses_MarshalStreamEvent_Delta_ShortToken(b *testing.B) {
	event := ResponseStreamEvent{
		Type:  "response.output_text.delta",
		Delta: "Answer",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		responsesSinkString = core.JSONMarshalString(event)
	}
}

func BenchmarkResponses_MarshalStreamEvent_Delta_LongToken(b *testing.B) {
	delta := ""
	for range 64 {
		delta += "fragment "
	}
	event := ResponseStreamEvent{
		Type:  "response.output_text.delta",
		Delta: delta,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		responsesSinkString = core.JSONMarshalString(event)
	}
}

func BenchmarkResponses_MarshalStreamEvent_Completed(b *testing.B) {
	resp := buildResponse()
	event := ResponseStreamEvent{Type: "response.completed", Response: &resp}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		responsesSinkString = core.JSONMarshalString(event)
	}
}

func BenchmarkResponses_MarshalStreamEvent_ThoughtDelta(b *testing.B) {
	thought := "Let me think through this step by step."
	event := ResponseStreamEvent{
		Type:    "response.thought.delta",
		Delta:   "thinking",
		Thought: &thought,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		responsesSinkString = core.JSONMarshalString(event)
	}
}

// --- Hand-rolled encoders — wired into writeJSON fast-path + ---
// available as direct call sites for downstream Responses producers.

func BenchmarkResponses_AppendResponse_Typical(b *testing.B) {
	resp := buildResponse()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		responsesSinkBytes = appendResponse(make([]byte, 0, responseSize(resp)), resp)
	}
}

func BenchmarkResponses_AppendStreamEvent_Delta_ShortToken(b *testing.B) {
	event := ResponseStreamEvent{
		Type:  "response.output_text.delta",
		Delta: "Answer",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		responsesSinkBytes = appendResponseStreamEvent(make([]byte, 0, responseStreamEventSize(event)), event)
	}
}

func BenchmarkResponses_AppendStreamEvent_Delta_LongToken(b *testing.B) {
	delta := ""
	for range 64 {
		delta += "fragment "
	}
	event := ResponseStreamEvent{
		Type:  "response.output_text.delta",
		Delta: delta,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		responsesSinkBytes = appendResponseStreamEvent(make([]byte, 0, responseStreamEventSize(event)), event)
	}
}

func BenchmarkResponses_AppendStreamEvent_Completed(b *testing.B) {
	resp := buildResponse()
	event := ResponseStreamEvent{Type: "response.completed", Response: &resp}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		responsesSinkBytes = appendResponseStreamEvent(make([]byte, 0, responseStreamEventSize(event)), event)
	}
}

func BenchmarkResponses_AppendStreamEvent_ThoughtDelta(b *testing.B) {
	thought := "Let me think through this step by step."
	event := ResponseStreamEvent{
		Type:    "response.thought.delta",
		Delta:   "thinking",
		Thought: &thought,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		responsesSinkBytes = appendResponseStreamEvent(make([]byte, 0, responseStreamEventSize(event)), event)
	}
}

// --- Stream-event unmarshal — proxy clients pay this on every SSE frame ---

func BenchmarkResponses_UnmarshalStreamEvent_Delta(b *testing.B) {
	body := core.JSONMarshalString(ResponseStreamEvent{
		Type:  "response.output_text.delta",
		Delta: "Answer",
	})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var event ResponseStreamEvent
		responsesSinkResult = core.JSONUnmarshalString(body, &event)
		responsesSinkEvent = event
	}
}

func BenchmarkResponses_UnmarshalStreamEvent_Completed(b *testing.B) {
	resp := buildResponse()
	body := core.JSONMarshalString(ResponseStreamEvent{Type: "response.completed", Response: &resp})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var event ResponseStreamEvent
		responsesSinkResult = core.JSONUnmarshalString(body, &event)
		responsesSinkEvent = event
	}
}
