// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the Ollama-compatible wire primitives. Per AX-11 —
// every request handled by the /api/chat or /api/generate path runs
// JSON ingress/egress; InferenceMessages and GenerateOptions project
// the wire shape onto inference contracts on every served request, and
// the response constructors fire on every completion.
//
// Run:    go test -bench='BenchmarkOllama' -benchtime=100ms -benchmem -run='^$' .

package ollama

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// Sinks defeat compiler DCE.
var (
	ollamaSinkChatRequest      ChatRequest
	ollamaSinkChatResponse     ChatResponse
	ollamaSinkGenerateRequest  GenerateRequest
	ollamaSinkGenerateResponse GenerateResponse
	ollamaSinkTagsResponse     TagsResponse
	ollamaSinkShowRequest      ShowRequest
	ollamaSinkShowResponse     ShowResponse
	ollamaSinkMessages         []inference.Message
	ollamaSinkOptions          []inference.GenerateOption
	ollamaSinkString           string
	ollamaSinkResult           core.Result
)

// --- Fixture builders ---

// buildOllamaMessages builds a representative chat transcript of the
// requested turn count. Single-turn = user, multi-turn = alternating
// user/assistant.
func buildOllamaMessages(turns int) []Message {
	out := make([]Message, 0, turns)
	for i := range turns {
		if i%2 == 0 {
			out = append(out, Message{Role: "user", Content: "Summarise the paragraph in one sentence."})
		} else {
			out = append(out, Message{Role: "assistant", Content: "The summary is concise and faithful to the original text."})
		}
	}
	return out
}

func buildOllamaChatRequest(turns int) ChatRequest {
	return ChatRequest{
		Model:    "qwen3",
		Messages: buildOllamaMessages(turns),
		Stream:   true,
		Options:  Options{Temperature: 0.7, TopK: 64, TopP: 0.95, NumPredict: 256},
	}
}

func buildOllamaGenerateRequest() GenerateRequest {
	return GenerateRequest{
		Model:   "qwen3",
		Prompt:  "Summarise the paragraph in one sentence.",
		Stream:  true,
		Options: Options{Temperature: 0.7, TopK: 64, TopP: 0.95, NumPredict: 256},
	}
}

// --- JSON Marshal — request emission (client-side) ---

func BenchmarkOllama_MarshalChatRequest_SingleTurn(b *testing.B) {
	req := buildOllamaChatRequest(1)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ollamaSinkString = core.JSONMarshalString(req)
	}
}

func BenchmarkOllama_MarshalChatRequest_FiveTurn(b *testing.B) {
	req := buildOllamaChatRequest(5)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ollamaSinkString = core.JSONMarshalString(req)
	}
}

func BenchmarkOllama_MarshalChatRequest_TwentyTurn(b *testing.B) {
	req := buildOllamaChatRequest(20)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ollamaSinkString = core.JSONMarshalString(req)
	}
}

func BenchmarkOllama_MarshalGenerateRequest(b *testing.B) {
	req := buildOllamaGenerateRequest()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ollamaSinkString = core.JSONMarshalString(req)
	}
}

// --- JSON Marshal — response emission (server-side) ---

func BenchmarkOllama_MarshalChatResponse(b *testing.B) {
	resp := NewChatResponse("qwen3", "The summary is concise.", inference.GenerateMetrics{PromptTokens: 200, GeneratedTokens: 32})
	resp.TotalDuration = 1_500_000_000
	resp.LoadDuration = 100_000_000
	resp.PromptEvalDuration = 200_000_000
	resp.EvalDuration = 1_200_000_000
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ollamaSinkString = core.JSONMarshalString(resp)
	}
}

func BenchmarkOllama_MarshalGenerateResponse(b *testing.B) {
	resp := NewGenerateResponse("qwen3", "The summary is concise.", inference.GenerateMetrics{PromptTokens: 200, GeneratedTokens: 32})
	resp.TotalDuration = 1_500_000_000
	resp.LoadDuration = 100_000_000
	resp.PromptEvalDuration = 200_000_000
	resp.EvalDuration = 1_200_000_000
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ollamaSinkString = core.JSONMarshalString(resp)
	}
}

// /api/tags listing — fired by ollama clients on every model-list
// discovery (e.g. open-webui startup). Three sizes — 1, 5, 20 models.

func BenchmarkOllama_MarshalTagsResponse_OneModel(b *testing.B) {
	resp := TagsResponse{Models: []ModelTag{
		{Name: "qwen3:latest", Model: "qwen3", ModifiedAt: "2026-05-21T10:00:00Z", Size: 4_500_000_000},
	}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ollamaSinkString = core.JSONMarshalString(resp)
	}
}

func BenchmarkOllama_MarshalTagsResponse_FiveModels(b *testing.B) {
	resp := TagsResponse{Models: []ModelTag{
		{Name: "qwen3:latest", Model: "qwen3", Size: 4_500_000_000},
		{Name: "gemma3:4b", Model: "gemma3", Size: 2_300_000_000},
		{Name: "llama3:8b", Model: "llama3", Size: 4_700_000_000},
		{Name: "qwen2.5:14b", Model: "qwen2.5", Size: 8_900_000_000},
		{Name: "deepseek:7b", Model: "deepseek", Size: 4_100_000_000},
	}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ollamaSinkString = core.JSONMarshalString(resp)
	}
}

func BenchmarkOllama_MarshalTagsResponse_TwentyModels(b *testing.B) {
	models := make([]ModelTag, 20)
	for i := range models {
		models[i] = ModelTag{
			Name:       "model-bench:tag",
			Model:      "model-bench",
			ModifiedAt: "2026-05-21T10:00:00Z",
			Size:       int64(4_000_000_000 + i*100_000_000),
		}
	}
	resp := TagsResponse{Models: models}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ollamaSinkString = core.JSONMarshalString(resp)
	}
}

// --- JSON Unmarshal — request ingress (server-side) ---

func BenchmarkOllama_UnmarshalChatRequest_SingleTurn(b *testing.B) {
	body := core.JSONMarshalString(buildOllamaChatRequest(1))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var req ChatRequest
		ollamaSinkResult = core.JSONUnmarshalString(body, &req)
		ollamaSinkChatRequest = req
	}
}

func BenchmarkOllama_UnmarshalChatRequest_FiveTurn(b *testing.B) {
	body := core.JSONMarshalString(buildOllamaChatRequest(5))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var req ChatRequest
		ollamaSinkResult = core.JSONUnmarshalString(body, &req)
		ollamaSinkChatRequest = req
	}
}

func BenchmarkOllama_UnmarshalChatRequest_TwentyTurn(b *testing.B) {
	body := core.JSONMarshalString(buildOllamaChatRequest(20))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var req ChatRequest
		ollamaSinkResult = core.JSONUnmarshalString(body, &req)
		ollamaSinkChatRequest = req
	}
}

func BenchmarkOllama_UnmarshalGenerateRequest(b *testing.B) {
	body := core.JSONMarshalString(buildOllamaGenerateRequest())
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var req GenerateRequest
		ollamaSinkResult = core.JSONUnmarshalString(body, &req)
		ollamaSinkGenerateRequest = req
	}
}

// --- JSON Unmarshal — response ingestion (client-side) ---

func BenchmarkOllama_UnmarshalChatResponse(b *testing.B) {
	resp := NewChatResponse("qwen3", "The summary is concise.", inference.GenerateMetrics{PromptTokens: 200, GeneratedTokens: 32})
	body := core.JSONMarshalString(resp)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var r ChatResponse
		ollamaSinkResult = core.JSONUnmarshalString(body, &r)
		ollamaSinkChatResponse = r
	}
}

func BenchmarkOllama_UnmarshalGenerateResponse(b *testing.B) {
	resp := NewGenerateResponse("qwen3", "The summary is concise.", inference.GenerateMetrics{PromptTokens: 200, GeneratedTokens: 32})
	body := core.JSONMarshalString(resp)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var r GenerateResponse
		ollamaSinkResult = core.JSONUnmarshalString(body, &r)
		ollamaSinkGenerateResponse = r
	}
}

func BenchmarkOllama_UnmarshalTagsResponse_FiveModels(b *testing.B) {
	body := core.JSONMarshalString(TagsResponse{Models: []ModelTag{
		{Name: "qwen3:latest", Model: "qwen3", Size: 4_500_000_000},
		{Name: "gemma3:4b", Model: "gemma3", Size: 2_300_000_000},
		{Name: "llama3:8b", Model: "llama3", Size: 4_700_000_000},
		{Name: "qwen2.5:14b", Model: "qwen2.5", Size: 8_900_000_000},
		{Name: "deepseek:7b", Model: "deepseek", Size: 4_100_000_000},
	}})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var r TagsResponse
		ollamaSinkResult = core.JSONUnmarshalString(body, &r)
		ollamaSinkTagsResponse = r
	}
}

func BenchmarkOllama_UnmarshalShowRequest(b *testing.B) {
	body := `{"model":"qwen3:latest"}`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var req ShowRequest
		ollamaSinkResult = core.JSONUnmarshalString(body, &req)
		ollamaSinkShowRequest = req
	}
}

// --- InferenceMessages — wire→internal conversion fired per request ---

func BenchmarkOllama_InferenceMessages_SingleTurn(b *testing.B) {
	messages := buildOllamaMessages(1)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ollamaSinkMessages = InferenceMessages(messages)
	}
}

func BenchmarkOllama_InferenceMessages_FiveTurn(b *testing.B) {
	messages := buildOllamaMessages(5)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ollamaSinkMessages = InferenceMessages(messages)
	}
}

func BenchmarkOllama_InferenceMessages_TwentyTurn(b *testing.B) {
	messages := buildOllamaMessages(20)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ollamaSinkMessages = InferenceMessages(messages)
	}
}

// --- GenerateOptions — sampling-field projection per request ---

func BenchmarkOllama_GenerateOptions_AllFieldsSet(b *testing.B) {
	options := Options{Temperature: 0.7, TopK: 64, TopP: 0.95, NumPredict: 256}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ollamaSinkOptions = GenerateOptions(options)
	}
}

func BenchmarkOllama_GenerateOptions_NoFieldsSet(b *testing.B) {
	options := Options{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ollamaSinkOptions = GenerateOptions(options)
	}
}

// --- Response constructors — fire once per non-streaming completion ---

func BenchmarkOllama_NewChatResponse(b *testing.B) {
	metrics := inference.GenerateMetrics{PromptTokens: 200, GeneratedTokens: 32}
	text := "The summary is concise and faithful to the original text."
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ollamaSinkChatResponse = NewChatResponse("qwen3", text, metrics)
	}
}

func BenchmarkOllama_NewGenerateResponse(b *testing.B) {
	metrics := inference.GenerateMetrics{PromptTokens: 200, GeneratedTokens: 32}
	text := "The summary is concise and faithful to the original text."
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ollamaSinkGenerateResponse = NewGenerateResponse("qwen3", text, metrics)
	}
}

// --- Append* fast-path encoders ---
//
// These bench the direct-entry hand-rolled encoders consumers on the
// HTTP hot path should call (an in-tree serve handler reaching for
// AppendChatResponse rather than core.JSONMarshalString). Each
// bench is the consumer-facing measurement — the "real" win once
// the proxy/serve handler lifts off encoding/json.
//
// The pre-sized-buffer benches reuse a backing scratch buffer
// per-iteration to model the steady-state hot-loop case where the
// caller keeps a per-connection emission buffer. The make-each-call
// benches model the cold-path (one-shot non-streaming response).

var ollamaSinkBuf []byte

func BenchmarkOllama_AppendChatResponse_Streaming(b *testing.B) {
	resp := NewChatResponse("qwen3", "tok", inference.GenerateMetrics{})
	resp.Message.Role = ""
	resp.Done = false
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ollamaSinkBuf = AppendChatResponse(make([]byte, 0, chatResponseSize(resp)), resp)
	}
}

func BenchmarkOllama_AppendChatResponse_Final(b *testing.B) {
	resp := NewChatResponse("qwen3", "The summary is concise.", inference.GenerateMetrics{PromptTokens: 200, GeneratedTokens: 32})
	resp.TotalDuration = 1_500_000_000
	resp.LoadDuration = 100_000_000
	resp.PromptEvalDuration = 200_000_000
	resp.EvalDuration = 1_200_000_000
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ollamaSinkBuf = AppendChatResponse(make([]byte, 0, chatResponseSize(resp)), resp)
	}
}

func BenchmarkOllama_AppendGenerateResponse_Streaming(b *testing.B) {
	resp := NewGenerateResponse("qwen3", "tok", inference.GenerateMetrics{})
	resp.Done = false
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ollamaSinkBuf = AppendGenerateResponse(make([]byte, 0, generateResponseSize(resp)), resp)
	}
}

func BenchmarkOllama_AppendGenerateResponse_Final(b *testing.B) {
	resp := NewGenerateResponse("qwen3", "The summary is concise.", inference.GenerateMetrics{PromptTokens: 200, GeneratedTokens: 32})
	resp.TotalDuration = 1_500_000_000
	resp.LoadDuration = 100_000_000
	resp.PromptEvalDuration = 200_000_000
	resp.EvalDuration = 1_200_000_000
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ollamaSinkBuf = AppendGenerateResponse(make([]byte, 0, generateResponseSize(resp)), resp)
	}
}

func BenchmarkOllama_AppendTagsResponse_OneModel(b *testing.B) {
	resp := TagsResponse{Models: []ModelTag{
		{Name: "qwen3:latest", Model: "qwen3", ModifiedAt: "2026-05-21T10:00:00Z", Size: 4_500_000_000},
	}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ollamaSinkBuf = AppendTagsResponse(make([]byte, 0, tagsResponseSize(resp)), resp)
	}
}

func BenchmarkOllama_AppendTagsResponse_FiveModels(b *testing.B) {
	resp := TagsResponse{Models: []ModelTag{
		{Name: "qwen3:latest", Model: "qwen3", Size: 4_500_000_000},
		{Name: "gemma3:4b", Model: "gemma3", Size: 2_300_000_000},
		{Name: "llama3:8b", Model: "llama3", Size: 4_700_000_000},
		{Name: "qwen2.5:14b", Model: "qwen2.5", Size: 8_900_000_000},
		{Name: "deepseek:7b", Model: "deepseek", Size: 4_100_000_000},
	}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ollamaSinkBuf = AppendTagsResponse(make([]byte, 0, tagsResponseSize(resp)), resp)
	}
}

func BenchmarkOllama_AppendTagsResponse_TwentyModels(b *testing.B) {
	models := make([]ModelTag, 20)
	for i := range models {
		models[i] = ModelTag{
			Name:       "model-bench:tag",
			Model:      "model-bench",
			ModifiedAt: "2026-05-21T10:00:00Z",
			Size:       int64(4_000_000_000 + i*100_000_000),
		}
	}
	resp := TagsResponse{Models: models}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ollamaSinkBuf = AppendTagsResponse(make([]byte, 0, tagsResponseSize(resp)), resp)
	}
}
