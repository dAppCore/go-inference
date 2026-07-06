// SPDX-Licence-Identifier: EUPL-1.2

package compat

import (
	"bytes"
	"context"
	"iter"
	"net/http"
	"net/http/httptest"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	openaicompat "dappco.re/go/inference/serving/provider/openai"
)

// fakeTextModel is a deterministic inference.TextModel stand-in for driving the
// compat handlers without loading an engine: Chat/Generate stream a fixed token
// run so a handler's routing, decode, and response framing are exercised end to
// end over httptest. Info/ModelType back the /api/show path.
type fakeTextModel struct {
	tokens []inference.Token
	info   inference.ModelInfo
	mtype  string
}

func newFakeTextModel() *fakeTextModel {
	return &fakeTextModel{
		tokens: []inference.Token{{ID: 1, Text: "Hello"}, {ID: 2, Text: " world"}},
		info:   inference.ModelInfo{Architecture: "gemma4", QuantBits: 4},
		mtype:  "gemma4",
	}
}

func (m *fakeTextModel) stream() iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		for _, t := range m.tokens {
			if !yield(t) {
				return
			}
		}
	}
}

func (m *fakeTextModel) Generate(context.Context, string, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.stream()
}

func (m *fakeTextModel) Chat(context.Context, []inference.Message, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.stream()
}

func (m *fakeTextModel) Classify(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok(nil)
}

func (m *fakeTextModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok(nil)
}

func (m *fakeTextModel) ModelType() string                  { return m.mtype }
func (m *fakeTextModel) Info() inference.ModelInfo          { return m.info }
func (m *fakeTextModel) Metrics() inference.GenerateMetrics { return inference.GenerateMetrics{} }
func (m *fakeTextModel) Err() core.Result                   { return core.Ok(nil) }
func (m *fakeTextModel) Close() core.Result                 { return core.Ok(nil) }

// testMux builds the compat mux backed by a static resolver holding one fake
// model under "test-model".
func testMux() http.Handler {
	return NewMux(openaicompat.NewStaticResolver(map[string]inference.TextModel{
		"test-model": newFakeTextModel(),
	}))
}

// do issues one request against the compat mux and returns the recorder.
func do(t *testing.T, method, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(method, path, bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	testMux().ServeHTTP(rec, req)
	return rec
}

// TestOllamaChatHandler_MethodRejection_Bad pins that a non-POST /api/chat is
// rejected (405) before any model work.
func TestOllamaChatHandler_MethodRejection_Bad(t *testing.T) {
	rec := do(t, http.MethodGet, "/api/chat", "")
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("GET /api/chat = %d, want 405", rec.Code)
	}
}

// TestOllamaChatHandler_BadBody_Ugly pins that a malformed JSON body is a 400,
// not a panic or a 500.
func TestOllamaChatHandler_BadBody_Ugly(t *testing.T) {
	rec := do(t, http.MethodPost, "/api/chat", "{not json")
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("POST /api/chat malformed = %d, want 400", rec.Code)
	}
}

// TestOllamaChatHandler_HappyPath_Good pins the non-streaming /api/chat path: a
// valid request resolves the model, runs Chat, and returns 200 with the model's
// visible text in the response body.
func TestOllamaChatHandler_HappyPath_Good(t *testing.T) {
	rec := do(t, http.MethodPost, "/api/chat", `{"model":"test-model","messages":[{"role":"user","content":"hi"}]}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("POST /api/chat = %d, want 200 (body: %s)", rec.Code, rec.Body.String())
	}
	if !core.Contains(rec.Body.String(), "Hello world") {
		t.Fatalf("/api/chat body = %s, want the model's visible text", rec.Body.String())
	}
}

// TestOllamaGenerateHandler_HappyPath_Good pins the non-streaming /api/generate
// path: a prompt request returns 200 with the model's completion.
func TestOllamaGenerateHandler_HappyPath_Good(t *testing.T) {
	rec := do(t, http.MethodPost, "/api/generate", `{"model":"test-model","prompt":"hi"}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("POST /api/generate = %d, want 200 (body: %s)", rec.Code, rec.Body.String())
	}
	if !core.Contains(rec.Body.String(), "Hello world") {
		t.Fatalf("/api/generate body = %s, want the completion", rec.Body.String())
	}
}

// TestOllamaGenerateHandler_UnknownModel_Bad pins that an unresolved model name
// is surfaced (not a 200), so a client naming a missing model gets an error.
func TestOllamaGenerateHandler_UnknownModel_Bad(t *testing.T) {
	rec := do(t, http.MethodPost, "/api/generate", `{"model":"absent","prompt":"hi"}`)
	if rec.Code == http.StatusOK {
		t.Fatalf("POST /api/generate unknown model = 200, want an error status")
	}
}

// TestOllamaShowHandler_Good pins /api/show: it reports the resolved model's
// architecture + model_type from Info without running a decode.
func TestOllamaShowHandler_Good(t *testing.T) {
	rec := do(t, http.MethodPost, "/api/show", `{"model":"test-model"}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("POST /api/show = %d, want 200 (body: %s)", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	if !core.Contains(body, "gemma4") || !core.Contains(body, "q4") {
		t.Fatalf("/api/show body = %s, want architecture + quantization", body)
	}
}

// TestAnthropicMessagesHandler_MethodRejection_Bad pins the /v1/messages method
// gate (POST-only) — a GET is 405 before any model work.
func TestAnthropicMessagesHandler_MethodRejection_Bad(t *testing.T) {
	rec := do(t, http.MethodGet, "/v1/messages", "")
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("GET /v1/messages = %d, want 405", rec.Code)
	}
}

// TestOpenAIResponsesHandler_MethodRejection_Bad pins the /v1/responses method
// gate (POST-only).
func TestOpenAIResponsesHandler_MethodRejection_Bad(t *testing.T) {
	rec := do(t, http.MethodGet, "/v1/responses", "")
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("GET /v1/responses = %d, want 405", rec.Code)
	}
}

// TestOllamaChatHandler_Streaming_Good pins the streaming /api/chat path: with
// stream:true the handler emits per-token NDJSON frames carrying the content and
// a terminating done frame (the wire format locked in mux_test.go, driven end to
// end through the handler).
func TestOllamaChatHandler_Streaming_Good(t *testing.T) {
	rec := do(t, http.MethodPost, "/api/chat", `{"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":true}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("stream /api/chat = %d, want 200 (body: %s)", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	if !core.Contains(body, "Hello") || !core.Contains(body, `"done"`) {
		t.Fatalf("stream /api/chat body = %s, want NDJSON content frames + a done frame", body)
	}
}

// TestOllamaGenerateHandler_Streaming_Good pins the streaming /api/generate path:
// stream:true emits per-token NDJSON response frames.
func TestOllamaGenerateHandler_Streaming_Good(t *testing.T) {
	rec := do(t, http.MethodPost, "/api/generate", `{"model":"test-model","prompt":"hi","stream":true}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("stream /api/generate = %d, want 200 (body: %s)", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	if !core.Contains(body, "Hello") || !core.Contains(body, `"response"`) {
		t.Fatalf("stream /api/generate body = %s, want NDJSON response frames", body)
	}
}

// TestAnthropicMessagesHandler_HappyPath_Good pins the non-streaming /v1/messages
// path: a valid Anthropic request resolves the model, runs Chat, and returns 200
// with the model's visible text.
func TestAnthropicMessagesHandler_HappyPath_Good(t *testing.T) {
	rec := do(t, http.MethodPost, "/v1/messages", `{"model":"test-model","max_tokens":128,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("POST /v1/messages = %d, want 200 (body: %s)", rec.Code, rec.Body.String())
	}
	if !core.Contains(rec.Body.String(), "Hello world") {
		t.Fatalf("/v1/messages body = %s, want the model's visible text", rec.Body.String())
	}
}

// TestAnthropicMessagesHandler_MissingModel_Bad pins the model-required guard
// (400 before any resolve/decode-of-model).
func TestAnthropicMessagesHandler_MissingModel_Bad(t *testing.T) {
	rec := do(t, http.MethodPost, "/v1/messages", `{"max_tokens":128,"messages":[]}`)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("POST /v1/messages no model = %d, want 400", rec.Code)
	}
}

// TestAnthropicMessagesHandler_Streaming_Good pins the streaming /v1/messages
// path: stream:true emits SSE event frames carrying the content.
func TestAnthropicMessagesHandler_Streaming_Good(t *testing.T) {
	rec := do(t, http.MethodPost, "/v1/messages", `{"model":"test-model","max_tokens":128,"stream":true,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("stream /v1/messages = %d, want 200 (body: %s)", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	if !core.Contains(body, "event:") || !core.Contains(body, "Hello") {
		t.Fatalf("stream /v1/messages body = %s, want SSE event frames with content", body)
	}
}

// TestOpenAIResponsesHandler_HappyPath_Good pins the non-streaming /v1/responses
// path: a valid request returns 200 with the model's visible text in the
// response output.
func TestOpenAIResponsesHandler_HappyPath_Good(t *testing.T) {
	rec := do(t, http.MethodPost, "/v1/responses", `{"model":"test-model","input":[{"role":"user","content":"hi"}]}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("POST /v1/responses = %d, want 200 (body: %s)", rec.Code, rec.Body.String())
	}
	if !core.Contains(rec.Body.String(), "Hello world") {
		t.Fatalf("/v1/responses body = %s, want the model's visible text", rec.Body.String())
	}
}

// TestOpenAIResponsesHandler_MissingModel_Bad pins the model-required guard (400).
func TestOpenAIResponsesHandler_MissingModel_Bad(t *testing.T) {
	rec := do(t, http.MethodPost, "/v1/responses", `{"input":[{"role":"user","content":"hi"}]}`)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("POST /v1/responses no model = %d, want 400", rec.Code)
	}
}

// TestOpenAIResponsesHandler_Streaming_Good pins the streaming /v1/responses
// path: stream:true emits SSE data frames carrying the response delta.
func TestOpenAIResponsesHandler_Streaming_Good(t *testing.T) {
	rec := do(t, http.MethodPost, "/v1/responses", `{"model":"test-model","stream":true,"input":[{"role":"user","content":"hi"}]}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("stream /v1/responses = %d, want 200 (body: %s)", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	if !core.Contains(body, "data:") || !core.Contains(body, "Hello") {
		t.Fatalf("stream /v1/responses body = %s, want SSE data frames with content", body)
	}
}
