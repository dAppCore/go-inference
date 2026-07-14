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
	ollamacompat "dappco.re/go/inference/serving/provider/ollama"
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
	// received captures the last Chat call's messages — nil unless a test
	// reads it, and never inspected by the existing happy-path tests, so this
	// is purely additive (#37: tool_choice / capability-gate coverage).
	received []inference.Message
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

func (m *fakeTextModel) Chat(_ context.Context, messages []inference.Message, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	m.received = messages
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

// TestOllamaGenerateHandler_MethodRejection_Bad pins the /api/generate method
// gate (POST-only) — a GET is 405 before any body work.
func TestOllamaGenerateHandler_MethodRejection_Bad(t *testing.T) {
	rec := do(t, http.MethodGet, "/api/generate", "")
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("GET /api/generate = %d, want 405", rec.Code)
	}
}

// TestOllamaGenerateHandler_BadBody_Ugly pins that a malformed JSON body is a
// 400, not a panic or a 500.
func TestOllamaGenerateHandler_BadBody_Ugly(t *testing.T) {
	rec := do(t, http.MethodPost, "/api/generate", "{not json")
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("POST /api/generate malformed = %d, want 400", rec.Code)
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

// TestOllamaShowHandler_MethodRejection_Bad pins the /api/show method gate
// (POST-only) — a GET is 405 before any model work.
func TestOllamaShowHandler_MethodRejection_Bad(t *testing.T) {
	rec := do(t, http.MethodGet, "/api/show", "")
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("GET /api/show = %d, want 405", rec.Code)
	}
}

// TestOllamaShowHandler_BadBody_Ugly pins that a malformed JSON body is a 400.
func TestOllamaShowHandler_BadBody_Ugly(t *testing.T) {
	rec := do(t, http.MethodPost, "/api/show", "{not json")
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("POST /api/show malformed = %d, want 400", rec.Code)
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

// TestAnthropicMessagesHandler_BadBody_Ugly pins that a malformed JSON body
// is a 400, not a panic or a 500.
func TestAnthropicMessagesHandler_BadBody_Ugly(t *testing.T) {
	rec := do(t, http.MethodPost, "/v1/messages", "{not json")
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("POST /v1/messages malformed = %d, want 400", rec.Code)
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

// TestOpenAIResponsesHandler_BadBody_Ugly pins that a malformed JSON body is
// a 400, not a panic or a 500.
func TestOpenAIResponsesHandler_BadBody_Ugly(t *testing.T) {
	rec := do(t, http.MethodPost, "/v1/responses", "{not json")
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("POST /v1/responses malformed = %d, want 400", rec.Code)
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

// TestAnthropicMessagesHandler_Good_ToolsRenderedForGemma4 pins the declare
// half of the tool round trip through serving/compat's own handler (as
// opposed to the openai package's unit-level coverage): a tools request
// against the default gemma4 fakeTextModel renders the declaration into the
// model's prompt (asserted on what the fake actually received).
func TestAnthropicMessagesHandler_Good_ToolsRenderedForGemma4(t *testing.T) {
	model := newFakeTextModel() // gemma4 by default
	mux := NewMux(openaicompat.NewStaticResolver(map[string]inference.TextModel{"test-model": model}))
	rec := httptest.NewRecorder()
	body := `{"model":"test-model","max_tokens":128,"messages":[{"role":"user","content":[{"type":"text","text":"weather?"}]}],` +
		`"tools":[{"name":"get_weather","input_schema":{"type":"object"}}]}`
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader([]byte(body))))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s, want 200", rec.Code, rec.Body.String())
	}
	if len(model.received) == 0 || !core.Contains(model.received[0].Content, "<|tool>declaration:get_weather") {
		t.Fatalf("model received %+v, want the get_weather declaration in the system turn", model.received)
	}
}

// TestAnthropicMessagesHandler_Bad_ToolsRejectedForUnsupportedArchitecture
// pins the capability-honesty gate (#37) on the Anthropic Messages surface: a
// tools request against a non-Gemma-4 architecture is a clean 400, not a
// silent no-op.
func TestAnthropicMessagesHandler_Bad_ToolsRejectedForUnsupportedArchitecture(t *testing.T) {
	model := &fakeTextModel{tokens: []inference.Token{{Text: "ok"}}, info: inference.ModelInfo{Architecture: "qwen3"}}
	mux := NewMux(openaicompat.NewStaticResolver(map[string]inference.TextModel{"test-model": model}))
	rec := httptest.NewRecorder()
	body := `{"model":"test-model","max_tokens":128,"messages":[{"role":"user","content":[{"type":"text","text":"weather?"}]}],` +
		`"tools":[{"name":"get_weather","input_schema":{"type":"object"}}]}`
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader([]byte(body))))

	if rec.Code != http.StatusBadRequest || !core.Contains(rec.Body.String(), `"param":"tools"`) {
		t.Fatalf("status = %d body=%s, want 400 param=tools", rec.Code, rec.Body.String())
	}
	if model.received != nil {
		t.Fatal("model was invoked despite the capability gate rejecting the request")
	}
}

// TestAnthropicMessagesHandler_Good_ToolChoiceNoneSuppressesDeclarations pins
// that tool_choice:{"type":"none"} keeps the model from ever seeing the tool
// menu this turn — and, since nothing is offered, the capability gate does
// not fire even against a non-gemma4 architecture.
func TestAnthropicMessagesHandler_Good_ToolChoiceNoneSuppressesDeclarations(t *testing.T) {
	model := &fakeTextModel{tokens: []inference.Token{{Text: "plain answer"}}, info: inference.ModelInfo{Architecture: "qwen3"}}
	mux := NewMux(openaicompat.NewStaticResolver(map[string]inference.TextModel{"test-model": model}))
	rec := httptest.NewRecorder()
	body := `{"model":"test-model","max_tokens":128,"tool_choice":{"type":"none"},` +
		`"messages":[{"role":"user","content":[{"type":"text","text":"weather?"}]}],` +
		`"tools":[{"name":"get_weather","input_schema":{"type":"object"}}]}`
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader([]byte(body))))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s, want 200 (tool_choice:none offers nothing, so the gate never fires)", rec.Code, rec.Body.String())
	}
	if len(model.received) != 1 || core.Contains(model.received[0].Content, "declaration") {
		t.Fatalf("model received %+v, want the plain user turn with no tool declaration", model.received)
	}
}

// TestAnthropicMessagesHandler_Bad_ToolChoiceUndeclaredTool pins that naming a
// tool that was never declared is a 400, not a silently-ignored choice.
func TestAnthropicMessagesHandler_Bad_ToolChoiceUndeclaredTool(t *testing.T) {
	model := newFakeTextModel()
	mux := NewMux(openaicompat.NewStaticResolver(map[string]inference.TextModel{"test-model": model}))
	rec := httptest.NewRecorder()
	body := `{"model":"test-model","max_tokens":128,"tool_choice":{"type":"tool","name":"not_declared"},` +
		`"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],` +
		`"tools":[{"name":"get_weather","input_schema":{"type":"object"}}]}`
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader([]byte(body))))

	if rec.Code != http.StatusBadRequest || !core.Contains(rec.Body.String(), `"param":"tool_choice"`) {
		t.Fatalf("status = %d body=%s, want 400 param=tool_choice", rec.Code, rec.Body.String())
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

// TestOllamaTagsHandler_MethodRejection_Bad pins the /api/tags method gate
// (GET-only) — a POST is 405 before any model listing.
func TestOllamaTagsHandler_MethodRejection_Bad(t *testing.T) {
	rec := do(t, http.MethodPost, ollamacompat.DefaultTagsPath, "")
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("POST /api/tags = %d, want 405", rec.Code)
	}
}

// TestOllamaTagsHandler_NoModelNames_Good pins /api/tags against the plain
// StaticResolver testMux() uses: it doesn't advertise a model list (no
// ModelNames() method), so tags comes back empty rather than erroring.
func TestOllamaTagsHandler_NoModelNames_Good(t *testing.T) {
	rec := do(t, http.MethodGet, ollamacompat.DefaultTagsPath, "")
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /api/tags = %d, want 200 (body: %s)", rec.Code, rec.Body.String())
	}
	if got, want := rec.Body.String(), `{"models":[]}`; got != want {
		t.Fatalf("/api/tags body = %s, want %s", got, want)
	}
}

// namedStaticResolver pairs a StaticResolver with a ModelNames() listing —
// /api/tags only reports models for a resolver that advertises its own names.
type namedStaticResolver struct {
	*openaicompat.StaticResolver
	names []string
}

func (r *namedStaticResolver) ModelNames() []string { return r.names }

// TestOllamaTagsHandler_HappyPath_Good pins /api/tags: it lists a
// name-advertising resolver's known models as Ollama tag entries, without
// touching the model itself (no Generate/Chat call).
func TestOllamaTagsHandler_HappyPath_Good(t *testing.T) {
	resolver := &namedStaticResolver{
		StaticResolver: openaicompat.NewStaticResolver(map[string]inference.TextModel{"test-model": newFakeTextModel()}),
		names:          []string{"test-model"},
	}
	mux := NewMux(resolver)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, ollamacompat.DefaultTagsPath, nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /api/tags = %d, want 200 (body: %s)", rec.Code, rec.Body.String())
	}
	if !core.Contains(rec.Body.String(), "test-model") {
		t.Fatalf("/api/tags body = %s, want the resolver's known model name", rec.Body.String())
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

// thinkingFakeModel streams a gemma4 thought channel ahead of the answer —
// the fixture for the typed thinking-block shape on /v1/messages.
func thinkingFakeModel() *fakeTextModel {
	m := newFakeTextModel()
	m.tokens = []inference.Token{
		{ID: 1, Text: "<|channel>thought\nponder deeply<channel|>"},
		{ID: 2, Text: "The answer."},
	}
	return m
}

// doWith issues one request against a mux holding the given model.
func doWith(t *testing.T, model inference.TextModel, method, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(method, path, bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	NewMux(openaicompat.NewStaticResolver(map[string]inference.TextModel{"test-model": model})).ServeHTTP(rec, req)
	return rec
}

// TestAnthropicMessagesHandler_Good_ThinkingBlock pins the typed reasoning
// shape on non-streaming /v1/messages: the thought channel arrives as a
// leading {"type":"thinking"} content block and the text block carries ONLY
// the visible answer — it used to leak the reasoning into the text block
// (handover 2026-07-18b follow-up).
func TestAnthropicMessagesHandler_Good_ThinkingBlock(t *testing.T) {
	rec := doWith(t, thinkingFakeModel(), http.MethodPost, "/v1/messages",
		`{"model":"test-model","max_tokens":128,"messages":[{"role":"user","content":"hi"}]}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("POST /v1/messages = %d (body: %s)", rec.Code, rec.Body.String())
	}
	var resp struct {
		Content []struct {
			Type     string `json:"type"`
			Text     string `json:"text"`
			Thinking string `json:"thinking"`
		} `json:"content"`
	}
	if res := core.JSONUnmarshal(rec.Body.Bytes(), &resp); !res.OK {
		t.Fatalf("decode: %v (body: %s)", res.Error(), rec.Body.String())
	}
	if len(resp.Content) != 2 {
		t.Fatalf("content blocks = %d, want thinking + text (body: %s)", len(resp.Content), rec.Body.String())
	}
	if resp.Content[0].Type != "thinking" || !core.Contains(resp.Content[0].Thinking, "ponder deeply") {
		t.Fatalf("block 0 = %+v, want the thinking block", resp.Content[0])
	}
	if resp.Content[1].Type != "text" || resp.Content[1].Text != "The answer." {
		t.Fatalf("block 1 = %+v, want the clean text answer", resp.Content[1])
	}
	if core.Contains(resp.Content[1].Text, "ponder") {
		t.Fatal("reasoning leaked into the text block")
	}
}

// TestAnthropicMessagesHandler_Good_ThinkingStream pins the streamed shape:
// a thinking block opens at index 0 (thinking_delta events), closes when the
// answer begins, and the text block takes index 1 — the real Anthropic
// extended-thinking event sequence.
func TestAnthropicMessagesHandler_Good_ThinkingStream(t *testing.T) {
	rec := doWith(t, thinkingFakeModel(), http.MethodPost, "/v1/messages",
		`{"model":"test-model","max_tokens":128,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("stream /v1/messages = %d (body: %s)", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	for _, want := range []string{
		`"content_block":{"type":"thinking","thinking":""}`,
		`"type":"thinking_delta"`,
		`"index":1,"content_block":{"type":"text","text":""}`,
		`"type":"text_delta"`,
	} {
		if !core.Contains(body, want) {
			t.Fatalf("stream missing %q:\n%s", want, body)
		}
	}
	if core.Index(body, "thinking_delta") > core.Index(body, "text_delta") {
		t.Fatal("thinking must stream before the text block")
	}
	if core.Contains(body, `"type":"text_delta","text":"`+"\\n"+`ponder`) || core.Contains(body, `text_delta","text":"ponder`) {
		t.Fatal("reasoning leaked into text deltas")
	}
}
