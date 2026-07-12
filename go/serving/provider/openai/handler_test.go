// SPDX-Licence-Identifier: EUPL-1.2

// Tests for the chat-completions net/http route and its JSON
// response/error helpers. The richer thought/tool-call/streaming
// integration scenarios live in openai_test.go (TestOpenAI_Handler_Good_*);
// this file covers NewHandler, ServeHTTP's canonical success/reject
// shapes and early-return branches, serveStreaming's stop/thought/
// error/length-cap branches, and the small error/result helpers at the
// bottom of handler.go.
package openai

import (
	"context"
	"iter"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// TestHandler_NewHandler_Good covers the plain construction path — the
// returned handler is wired to the given resolver and serves through it.
func TestHandler_NewHandler_Good(t *testing.T) {
	resolver := NewStaticResolver(map[string]inference.TextModel{"qwen": &stubModel{}})
	handler := NewHandler(resolver)

	if handler == nil || handler.resolver == nil {
		t.Fatalf("NewHandler() = %#v, want a handler wired to the given resolver", handler)
	}
}

// TestHandler_NewHandler_Bad covers a nil resolver — NewHandler must
// still return a non-nil *Handler (ServeHTTP's own nil-resolver guard
// is what turns that into a 503, not a construction-time panic).
func TestHandler_NewHandler_Bad(t *testing.T) {
	handler := NewHandler(nil)

	if handler == nil || handler.resolver != nil {
		t.Fatalf("NewHandler(nil) = %#v, want a handler with a nil resolver, not a nil handler", handler)
	}
}

// TestHandler_NewHandler_Ugly covers that NewHandler accepts any
// Resolver implementation — not just *StaticResolver — by wiring
// through a functional ResolverFunc adapter.
func TestHandler_NewHandler_Ugly(t *testing.T) {
	called := false
	handler := NewHandler(ResolverFunc(func(context.Context, string) (inference.TextModel, error) {
		called = true
		return &stubModel{}, nil
	}))

	if _, err := handler.resolver.ResolveModel(context.Background(), "anything"); err != nil || !called {
		t.Fatalf("NewHandler() with a functional Resolver did not wire it through: err=%v called=%v", err, called)
	}
}

// TestHandler_ServeHTTP_Good drives the canonical non-streaming success
// path end-to-end — the richer thought/tool-call/streaming shapes are
// covered in openai_test.go (TestOpenAI_Handler_Good_*).
func TestHandler_ServeHTTP_Good(t *testing.T) {
	model := &stubModel{
		tokens:  []inference.Token{{Text: "hello"}},
		metrics: inference.GenerateMetrics{PromptTokens: 1, GeneratedTokens: 1},
	}
	handler := NewHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	rec := httptest.NewRecorder()
	body := `{"model":"qwen","messages":[{"role":"user","content":"hi"}]}`

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultChatCompletionsPath, strings.NewReader(body)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s, want 200", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"content":"hello"`) || !strings.Contains(rec.Body.String(), `"finish_reason":"stop"`) {
		t.Fatalf("body = %s, want hello content with stop finish reason", rec.Body.String())
	}
}

// TestHandler_ServeHTTP_Ugly covers the non-streaming tool_calls path —
// a model whose output carries a <|tool_call> span must flip
// finish_reason to "tool_calls" and populate ChatMessage.ToolCalls,
// rather than surfacing the raw gemma tool-call markup as content.
func TestHandler_ServeHTTP_Ugly(t *testing.T) {
	model := &stubModel{tokens: []inference.Token{{Text: "<|tool_call>call:get_weather{}<tool_call|>"}}}
	handler := NewHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	rec := httptest.NewRecorder()
	body := `{"model":"qwen","messages":[{"role":"user","content":"weather?"}]}`

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultChatCompletionsPath, strings.NewReader(body)))

	bodyText := rec.Body.String()
	if rec.Code != http.StatusOK || !strings.Contains(bodyText, `"finish_reason":"tool_calls"`) {
		t.Fatalf("status = %d body=%s, want 200 finish_reason=tool_calls", rec.Code, bodyText)
	}
	if !strings.Contains(bodyText, `"name":"get_weather"`) {
		t.Fatalf("body=%s, want the parsed tool call", bodyText)
	}
}

// TestHandler_ServeHTTP_Bad drives every early-return branch in
// Handler.ServeHTTP ahead of model resolution.
func TestHandler_ServeHTTP_Bad(t *testing.T) {
	model := &stubModel{tokens: []inference.Token{{Text: "hi"}}}
	okResolver := NewStaticResolver(map[string]inference.TextModel{"qwen": model})

	t.Run("nil-handler", func(t *testing.T) {
		var h *Handler
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultChatCompletionsPath, strings.NewReader(`{}`)))
		if rec.Code != http.StatusServiceUnavailable {
			t.Fatalf("status = %d, want 503", rec.Code)
		}
	})

	t.Run("nil-resolver", func(t *testing.T) {
		handler := NewHandler(nil)
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultChatCompletionsPath, strings.NewReader(`{}`)))
		if rec.Code != http.StatusServiceUnavailable {
			t.Fatalf("status = %d, want 503", rec.Code)
		}
	})

	t.Run("nil-request", func(t *testing.T) {
		handler := NewHandler(okResolver)
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, nil)
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("status = %d, want 400", rec.Code)
		}
	})

	t.Run("wrong-method", func(t *testing.T) {
		handler := NewHandler(okResolver)
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultChatCompletionsPath, nil))
		if rec.Code != http.StatusMethodNotAllowed {
			t.Fatalf("status = %d, want 405", rec.Code)
		}
		if got := rec.Header().Get("Allow"); got != http.MethodPost {
			t.Fatalf("Allow = %q, want POST", got)
		}
	})

	t.Run("malformed-body", func(t *testing.T) {
		handler := NewHandler(okResolver)
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultChatCompletionsPath, strings.NewReader(`{`)))
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("status = %d, want 400", rec.Code)
		}
	})

	t.Run("validate-error-missing-model", func(t *testing.T) {
		handler := NewHandler(okResolver)
		rec := httptest.NewRecorder()
		body := `{"messages":[{"role":"user","content":"hi"}]}`
		handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultChatCompletionsPath, strings.NewReader(body)))
		if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), `"param":"model"`) {
			t.Fatalf("status = %d body=%s, want 400 param=model", rec.Code, rec.Body.String())
		}
	})

	t.Run("stop-normalize-error", func(t *testing.T) {
		// A whitespace-only stop entry passes ValidateRequest (which
		// never inspects Stop) but fails NormalizeStopSequences.
		handler := NewHandler(okResolver)
		rec := httptest.NewRecorder()
		body := `{"model":"qwen","messages":[{"role":"user","content":"hi"}],"stop":["   "]}`
		handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultChatCompletionsPath, strings.NewReader(body)))
		if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), `"param":"stop"`) {
			t.Fatalf("status = %d body=%s, want 400 param=stop", rec.Code, rec.Body.String())
		}
	})

	t.Run("model-not-found", func(t *testing.T) {
		handler := NewHandler(okResolver)
		rec := httptest.NewRecorder()
		body := `{"model":"missing","messages":[{"role":"user","content":"hi"}]}`
		handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultChatCompletionsPath, strings.NewReader(body)))
		if rec.Code != http.StatusNotFound {
			t.Fatalf("status = %d body=%s, want 404", rec.Code, rec.Body.String())
		}
	})
}

// TestHandler_ServeNonStreaming_Bad_ModelError covers the
// model.Err()-not-OK branch — the model fails mid-generation and the
// handler must surface a 500 rather than a partial 200 response.
func TestHandler_ServeNonStreaming_Bad_ModelError(t *testing.T) {
	model := &stubModel{tokens: []inference.Token{{Text: "partial"}}, err: core.E("test", "generation failed", nil)}
	handler := NewHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	rec := httptest.NewRecorder()
	body := `{"model":"qwen","messages":[{"role":"user","content":"hi"}]}`

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultChatCompletionsPath, strings.NewReader(body)))

	if rec.Code != http.StatusInternalServerError || !strings.Contains(rec.Body.String(), "generation failed") {
		t.Fatalf("status = %d body=%s, want 500 generation failed", rec.Code, rec.Body.String())
	}
}

// TestHandler_ServeNonStreaming_Ugly_LengthCapReached covers the
// finish_reason="length" branch — MaxTokens set and reached.
func TestHandler_ServeNonStreaming_Ugly_LengthCapReached(t *testing.T) {
	model := &stubModel{
		tokens:  []inference.Token{{Text: "clipped"}},
		metrics: inference.GenerateMetrics{PromptTokens: 1, GeneratedTokens: 4},
	}
	handler := NewHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	rec := httptest.NewRecorder()
	body := `{"model":"qwen","messages":[{"role":"user","content":"hi"}],"max_tokens":4}`

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultChatCompletionsPath, strings.NewReader(body)))

	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"finish_reason":"length"`) {
		t.Fatalf("status = %d body=%s, want 200 finish_reason=length", rec.Code, rec.Body.String())
	}
}

// stopModel is a stubModel whose Chat replays a fixed token sequence
// through the streaming path — used to drive serveStreaming's
// stop-sequence mid-stream cut, which the plain stubModel.Chat (a
// single-shot replay with no per-token control) cannot exercise
// deterministically across a stop boundary that lands mid-token.
type stopModel struct {
	stubModel
}

func TestHandler_ServeStreaming_Ugly_StopSequenceCutsMidToken(t *testing.T) {
	// Stop sequence "TOP" lands inside the second token ("STOP more") —
	// stopCut falls strictly after the already-emitted prefix, exercising
	// the candidate[len(emittedContent):stopCut] branch (as opposed to
	// stopCut<=len(emittedContent), covered by the same-token case below).
	model := &stubModel{tokens: []inference.Token{{Text: "before "}, {Text: "STOP more"}}}
	handler := NewHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	rec := httptest.NewRecorder()
	body := `{"model":"qwen","messages":[{"role":"user","content":"hi"}],"stream":true,"stop":["TOP"]}`

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultChatCompletionsPath, strings.NewReader(body)))

	bodyText := rec.Body.String()
	if strings.Contains(bodyText, "more") {
		t.Fatalf("stream emitted content past the stop sequence: %s", bodyText)
	}
	if !strings.Contains(bodyText, `"content":"before "`) {
		t.Fatalf("stream missing pre-stop content: %s", bodyText)
	}
	if !strings.Contains(bodyText, "data: [DONE]") {
		t.Fatalf("stream missing DONE: %s", bodyText)
	}
}

// TestHandler_ServeStreaming_Ugly_StopSequenceAtTokenStart covers the
// stopCut<=len(emittedContent) branch — the stop sequence appears at
// the very start of the delta, so nothing new is emitted this chunk.
func TestHandler_ServeStreaming_Ugly_StopSequenceAtTokenStart(t *testing.T) {
	model := &stubModel{tokens: []inference.Token{{Text: "answer"}, {Text: "STOP"}}}
	handler := NewHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	rec := httptest.NewRecorder()
	body := `{"model":"qwen","messages":[{"role":"user","content":"hi"}],"stream":true,"stop":["STOP"]}`

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultChatCompletionsPath, strings.NewReader(body)))

	bodyText := rec.Body.String()
	if strings.Contains(bodyText, "STOP") {
		t.Fatalf("stream emitted the stop sequence itself: %s", bodyText)
	}
	if !strings.Contains(bodyText, `"content":"answer"`) {
		t.Fatalf("stream missing pre-stop content: %s", bodyText)
	}
}

// TestHandler_ServeStreaming_Bad_ModelError covers the streaming
// finish_reason="error" branch.
func TestHandler_ServeStreaming_Bad_ModelError(t *testing.T) {
	model := &stubModel{tokens: []inference.Token{{Text: "partial"}}, err: core.E("test", "stream failed", nil)}
	handler := NewHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	rec := httptest.NewRecorder()
	body := `{"model":"qwen","messages":[{"role":"user","content":"hi"}],"stream":true}`

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultChatCompletionsPath, strings.NewReader(body)))

	if !strings.Contains(rec.Body.String(), `"finish_reason":"error"`) {
		t.Fatalf("body=%s, want finish_reason=error", rec.Body.String())
	}
}

// TestHandler_ServeStreaming_Ugly_LengthCapReached covers the
// streaming finish_reason="length" branch.
func TestHandler_ServeStreaming_Ugly_LengthCapReached(t *testing.T) {
	model := &stubModel{
		tokens:  []inference.Token{{Text: "clipped"}},
		metrics: inference.GenerateMetrics{PromptTokens: 1, GeneratedTokens: 2},
	}
	handler := NewHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	rec := httptest.NewRecorder()
	body := `{"model":"qwen","messages":[{"role":"user","content":"hi"}],"stream":true,"max_tokens":2}`

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultChatCompletionsPath, strings.NewReader(body)))

	if !strings.Contains(rec.Body.String(), `"finish_reason":"length"`) {
		t.Fatalf("body=%s, want finish_reason=length", rec.Body.String())
	}
}

// thinkingTailStubModel yields a token that leaves a non-empty
// thought still buffered in the extractor (an opened-but-unclosed
// <think> span) so Flush's tail carries a thought — exercising
// serveStreaming's trailing "thoughtTail != ”" chunk.
type thinkingTailStubModel struct{ stubModel }

func (m *thinkingTailStubModel) Chat(context.Context, []inference.Message, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		yield(inference.Token{Text: "<think>unfinished thought"})
	}
}

func TestHandler_ServeStreaming_Ugly_ThoughtTailFlushed(t *testing.T) {
	handler := NewHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": &thinkingTailStubModel{}}))
	rec := httptest.NewRecorder()
	body := `{"model":"qwen","messages":[{"role":"user","content":"hi"}],"stream":true}`

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultChatCompletionsPath, strings.NewReader(body)))

	if !strings.Contains(rec.Body.String(), `"thought":"unfinished thought"`) {
		t.Fatalf("body=%s, want flushed thought tail", rec.Body.String())
	}
}

// straddlingMarkerStubModel yields a single token ending in a partial
// "</think>" close marker ("</thi") — unlike thinkingTailStubModel's
// "<think>unfinished thought" (which shares no suffix with "</think>"'s
// prefix set and so drains entirely inside ThinkingExtractor.Process,
// never reaching serveStreaming's post-loop Flush branch at all), the
// partial-marker suffix forces ThinkingExtractor.drain's non-final
// safe-suffix hold-back: "hidden" streams as a mid-loop thought chunk
// and "</thi" stays buffered in the extractor until serveStreaming
// calls Flush after the token loop ends.
type straddlingMarkerStubModel struct{ stubModel }

func (m *straddlingMarkerStubModel) Chat(context.Context, []inference.Message, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		yield(inference.Token{Text: "<think>hidden</thi"})
	}
}

// TestHandler_ServeStreaming_Ugly_PartialMarkerFlushedAsTailChunk
// covers serveStreaming's post-loop "visibleTail, thoughtTail :=
// extractor.Flush(); ... != ”" branch for real — see the
// straddlingMarkerStubModel doc comment for why
// TestHandler_ServeStreaming_Ugly_ThoughtTailFlushed's model does not
// actually reach this branch despite its name.
func TestHandler_ServeStreaming_Ugly_PartialMarkerFlushedAsTailChunk(t *testing.T) {
	handler := NewHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": &straddlingMarkerStubModel{}}))
	rec := httptest.NewRecorder()
	body := `{"model":"qwen","messages":[{"role":"user","content":"hi"}],"stream":true}`

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultChatCompletionsPath, strings.NewReader(body)))

	bodyText := rec.Body.String()
	if !strings.Contains(bodyText, `"thought":"hidden"`) {
		t.Fatalf("body=%s, want the mid-loop thought chunk", bodyText)
	}
	if !strings.Contains(bodyText, `"thought":"</thi"`) {
		t.Fatalf("body=%s, want the flushed partial-marker tail chunk", bodyText)
	}
}

// TestWriteJSON_Good drives writeJSON's type-switch fast paths for
// EmbeddingResponse and Response, plus the generic core.JSONMarshal
// fallback for a type the fast paths don't special-case.
func TestWriteJSON_Good(t *testing.T) {
	rec := httptest.NewRecorder()
	writeJSON(rec, http.StatusOK, EmbeddingResponse{Object: "list", Model: "m"})
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"object":"list"`) {
		t.Fatalf("EmbeddingResponse: status=%d body=%s", rec.Code, rec.Body.String())
	}

	rec = httptest.NewRecorder()
	writeJSON(rec, http.StatusOK, NewTextResponse("r1", "m", "hi", inference.GenerateMetrics{}))
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"object":"response"`) {
		t.Fatalf("Response: status=%d body=%s", rec.Code, rec.Body.String())
	}

	rec = httptest.NewRecorder()
	writeJSON(rec, http.StatusOK, map[string]string{"k": "v"})
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"k":"v"`) {
		t.Fatalf("generic fallback: status=%d body=%s", rec.Code, rec.Body.String())
	}
}

// TestWriteJSON_Bad_UnmarshalablePayload covers core.JSONMarshal's
// failure branch — writeJSON must fall back to "{}" rather than
// write a truncated/invalid body.
func TestWriteJSON_Bad_UnmarshalablePayload(t *testing.T) {
	rec := httptest.NewRecorder()
	writeJSON(rec, http.StatusOK, func() {})

	if rec.Body.String() != "{}" {
		t.Fatalf("body = %q, want {} fallback for an unmarshalable payload", rec.Body.String())
	}
}

// TestRequestValidationError_Error_Bad covers the nil-receiver guard.
func TestRequestValidationError_Error_Bad(t *testing.T) {
	var e *requestValidationError
	if got := e.Error(); got != "" {
		t.Fatalf("Error() = %q, want empty for nil receiver", got)
	}
}

// TestErrorParam_Good_Bad covers both branches: a *requestValidationError
// yields its param, any other error yields "".
func TestErrorParam_Good_Bad(t *testing.T) {
	if got := errorParam(requestError("bad", "model")); got != "model" {
		t.Fatalf("errorParam(requestValidationError) = %q, want model", got)
	}
	if got := errorParam(core.E("test", "generic", nil)); got != "" {
		t.Fatalf("errorParam(generic error) = %q, want empty", got)
	}
}

// --- #37: tool_choice / capability gate / response_format, HTTP round-trip -

// TestHandler_ServeHTTP_Good_ToolsRenderedForGemma4 pins the declare half of
// the round trip: a tools request against a Gemma 4 architecture renders the
// declarations into the model's prompt (a system turn) in Gemma 4's native
// syntax — asserted on what the fake model actually received, not just the
// HTTP response.
func TestHandler_ServeHTTP_Good_ToolsRenderedForGemma4(t *testing.T) {
	model := &recordingModel{arch: "gemma4_text", stubModel: stubModel{tokens: []inference.Token{{Text: "sunny"}}}}
	handler := NewHandler(NewStaticResolver(map[string]inference.TextModel{"gemma": model}))
	rec := httptest.NewRecorder()
	body := `{"model":"gemma","messages":[{"role":"user","content":"weather?"}],` +
		`"tools":[{"type":"function","function":{"name":"get_weather","description":"d","parameters":{"type":"object"}}}]}`

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultChatCompletionsPath, strings.NewReader(body)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s, want 200", rec.Code, rec.Body.String())
	}
	if len(model.received) == 0 || !strings.Contains(model.received[0].Content, "<|tool>declaration:get_weather") {
		t.Fatalf("model received %+v, want the get_weather declaration in the system turn", model.received)
	}
}

// TestHandler_ServeHTTP_Bad_ToolsRejectedForUnsupportedArchitecture pins the
// capability-honesty gate (#37): a tools request against an architecture with
// no Gemma 4 tool syntax is rejected with a clear 4xx instead of silently
// rendering a declaration menu the model could never read.
func TestHandler_ServeHTTP_Bad_ToolsRejectedForUnsupportedArchitecture(t *testing.T) {
	model := &recordingModel{arch: "qwen3", stubModel: stubModel{tokens: []inference.Token{{Text: "ok"}}}}
	handler := NewHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	rec := httptest.NewRecorder()
	body := `{"model":"qwen","messages":[{"role":"user","content":"weather?"}],` +
		`"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}]}`

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultChatCompletionsPath, strings.NewReader(body)))

	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), `"param":"tools"`) {
		t.Fatalf("status = %d body=%s, want 400 param=tools", rec.Code, rec.Body.String())
	}
	if model.calls != 0 {
		t.Fatal("model was invoked despite the capability gate rejecting the request")
	}
}

// TestHandler_ServeHTTP_Good_ToolChoiceNoneSuppressesDeclarations pins that
// tool_choice:"none" keeps the model from ever seeing the tool menu this turn
// — and, since nothing is offered, the capability gate does not fire even
// though the model's architecture could not have served them anyway.
func TestHandler_ServeHTTP_Good_ToolChoiceNoneSuppressesDeclarations(t *testing.T) {
	model := &recordingModel{arch: "qwen3", stubModel: stubModel{tokens: []inference.Token{{Text: "plain answer"}}}}
	handler := NewHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	rec := httptest.NewRecorder()
	body := `{"model":"qwen","messages":[{"role":"user","content":"weather?"}],"tool_choice":"none",` +
		`"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}]}`

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultChatCompletionsPath, strings.NewReader(body)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s, want 200 (tool_choice:none offers nothing, so the gate never fires)", rec.Code, rec.Body.String())
	}
	if len(model.received) != 1 || strings.Contains(model.received[0].Content, "declaration") {
		t.Fatalf("model received %+v, want the plain user turn with no tool declaration", model.received)
	}
}

// TestHandler_ServeHTTP_Bad_ToolChoiceUndeclaredTool pins that naming a tool
// that was never declared is a 400 (agent/tools.Resolve's caller-error case),
// not a silently-ignored choice.
func TestHandler_ServeHTTP_Bad_ToolChoiceUndeclaredTool(t *testing.T) {
	model := &recordingModel{arch: "gemma4_text", stubModel: stubModel{tokens: []inference.Token{{Text: "x"}}}}
	handler := NewHandler(NewStaticResolver(map[string]inference.TextModel{"gemma": model}))
	rec := httptest.NewRecorder()
	body := `{"model":"gemma","messages":[{"role":"user","content":"hi"}],` +
		`"tool_choice":{"type":"function","function":{"name":"not_declared"}},` +
		`"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}]}`

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultChatCompletionsPath, strings.NewReader(body)))

	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), `"param":"tool_choice"`) {
		t.Fatalf("status = %d body=%s, want 400 param=tool_choice", rec.Code, rec.Body.String())
	}
}

// TestHandler_ServeHTTP_Good_ToolChoiceNamedNarrowsDeclarations pins that a
// named function tool_choice narrows the rendered menu to just that tool, even
// though two were declared.
func TestHandler_ServeHTTP_Good_ToolChoiceNamedNarrowsDeclarations(t *testing.T) {
	model := &recordingModel{arch: "gemma4_text", stubModel: stubModel{tokens: []inference.Token{{Text: "x"}}}}
	handler := NewHandler(NewStaticResolver(map[string]inference.TextModel{"gemma": model}))
	rec := httptest.NewRecorder()
	body := `{"model":"gemma","messages":[{"role":"user","content":"hi"}],` +
		`"tool_choice":{"type":"function","function":{"name":"get_weather"}},` +
		`"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}},` +
		`{"type":"function","function":{"name":"search","parameters":{"type":"object"}}}]}`

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultChatCompletionsPath, strings.NewReader(body)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s, want 200", rec.Code, rec.Body.String())
	}
	prompt := model.received[0].Content
	if !strings.Contains(prompt, "declaration:get_weather") || strings.Contains(prompt, "declaration:search") {
		t.Fatalf("prompt = %q, want only get_weather declared", prompt)
	}
}

// TestHandler_ServeStreaming_Good_ToolChoiceNoneSuppressesDeclarations pins
// that tool_choice filtering applies identically on the streaming dispatch
// path — the offered-tools resolution happens once in ServeHTTP ahead of the
// stream/non-stream branch, not duplicated per path.
func TestHandler_ServeStreaming_Good_ToolChoiceNoneSuppressesDeclarations(t *testing.T) {
	model := &recordingModel{arch: "gemma4_text", stubModel: stubModel{tokens: []inference.Token{{Text: "plain"}}}}
	handler := NewHandler(NewStaticResolver(map[string]inference.TextModel{"gemma": model}))
	rec := httptest.NewRecorder()
	body := `{"model":"gemma","messages":[{"role":"user","content":"hi"}],"stream":true,"tool_choice":"none",` +
		`"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}]}`

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultChatCompletionsPath, strings.NewReader(body)))

	if len(model.received) != 1 || strings.Contains(model.received[0].Content, "declaration") {
		t.Fatalf("model received %+v, want the plain user turn with no tool declaration", model.received)
	}
	if !strings.Contains(rec.Body.String(), "data: [DONE]") {
		t.Fatalf("body=%s, want a well-formed stream close", rec.Body.String())
	}
}

// TestHandler_ServeHTTP_Good_ResponseFormatJSONObjectPassthrough pins the
// no-repair-needed path: already-valid JSON passes straight through response_format
// validation with no repair re-call.
func TestHandler_ServeHTTP_Good_ResponseFormatJSONObjectPassthrough(t *testing.T) {
	model := &recordingModel{stubModel: stubModel{tokens: []inference.Token{{Text: `{"ok":true}`}}}}
	handler := NewHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	rec := httptest.NewRecorder()
	body := `{"model":"qwen","messages":[{"role":"user","content":"give me json"}],"response_format":{"type":"json_object"}}`

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultChatCompletionsPath, strings.NewReader(body)))

	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"content":"{\"ok\":true}"`) {
		t.Fatalf("status = %d body=%s, want 200 with the JSON content unchanged", rec.Code, rec.Body.String())
	}
	if model.calls != 1 {
		t.Fatalf("model calls = %d, want exactly 1 (no repair re-call for already-valid JSON)", model.calls)
	}
}

// TestHandler_ServeHTTP_Good_ResponseFormatJSONSchemaRepairs pins the repair
// path end-to-end: the first generation misses a required field,
// validateStructuredOutput re-prompts the model, and the repaired second
// attempt is what the client receives.
func TestHandler_ServeHTTP_Good_ResponseFormatJSONSchemaRepairs(t *testing.T) {
	model := &recordingModel{sequenced: [][]inference.Token{
		{{Text: `{"name":"Ada"}`}},          // first generation — missing "age"
		{{Text: `{"name":"Ada","age":36}`}}, // repair re-call — matches the schema
	}}
	handler := NewHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	rec := httptest.NewRecorder()
	body := `{"model":"qwen","messages":[{"role":"user","content":"describe Ada"}],"response_format":` +
		`{"type":"json_schema","json_schema":{"name":"person","schema":{"required":["name","age"],` +
		`"properties":{"name":{"type":"string"},"age":{"type":"integer"}}}}}}`

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultChatCompletionsPath, strings.NewReader(body)))

	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `{\"name\":\"Ada\",\"age\":36}`) {
		t.Fatalf("status = %d body=%s, want 200 with the repaired payload", rec.Code, rec.Body.String())
	}
	if model.calls != 2 {
		t.Fatalf("model calls = %d, want exactly 2 (the original generation plus one repair)", model.calls)
	}
}

// TestHandler_ServeHTTP_Bad_ResponseFormatExhaustedRepair pins that a model
// which never produces a matching shape surfaces a 422, not a 200 carrying
// invalid content or a silently-dropped constraint.
func TestHandler_ServeHTTP_Bad_ResponseFormatExhaustedRepair(t *testing.T) {
	model := &recordingModel{sequenced: [][]inference.Token{
		{{Text: `not json`}},
		{{Text: `still not json`}},
		{{Text: `also not json`}},
	}}
	handler := NewHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	rec := httptest.NewRecorder()
	body := `{"model":"qwen","messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_object"}}`

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultChatCompletionsPath, strings.NewReader(body)))

	if rec.Code != http.StatusUnprocessableEntity {
		t.Fatalf("status = %d body=%s, want 422", rec.Code, rec.Body.String())
	}
}

// TestHandler_ServeHTTP_Bad_ResponseFormatStreamingRejected pins the
// documented streaming gap (#37): response_format that requires validation
// combined with stream=true is a clean 400 rather than a half-built feature
// (skipped validation, or a fake single-chunk "stream").
func TestHandler_ServeHTTP_Bad_ResponseFormatStreamingRejected(t *testing.T) {
	model := &stubModel{tokens: []inference.Token{{Text: `{"ok":true}`}}}
	handler := NewHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	rec := httptest.NewRecorder()
	body := `{"model":"qwen","messages":[{"role":"user","content":"hi"}],"stream":true,"response_format":{"type":"json_object"}}`

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultChatCompletionsPath, strings.NewReader(body)))

	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), `"param":"response_format"`) {
		t.Fatalf("status = %d body=%s, want 400 param=response_format", rec.Code, rec.Body.String())
	}
}

// TestHandler_ServeHTTP_Bad_ResponseFormatInvalidType covers ValidateRequest's
// response_format.type allow-list.
func TestHandler_ServeHTTP_Bad_ResponseFormatInvalidType(t *testing.T) {
	model := &stubModel{tokens: []inference.Token{{Text: "x"}}}
	handler := NewHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	rec := httptest.NewRecorder()
	body := `{"model":"qwen","messages":[{"role":"user","content":"hi"}],"response_format":{"type":"yaml"}}`

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultChatCompletionsPath, strings.NewReader(body)))

	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), `"param":"response_format"`) {
		t.Fatalf("status = %d body=%s, want 400 param=response_format", rec.Code, rec.Body.String())
	}
}

// TestHandler_ServeHTTP_Bad_ResponseFormatJSONSchemaMissingSchema covers
// ValidateRequest rejecting a json_schema format with no schema to validate
// against.
func TestHandler_ServeHTTP_Bad_ResponseFormatJSONSchemaMissingSchema(t *testing.T) {
	model := &stubModel{tokens: []inference.Token{{Text: "x"}}}
	handler := NewHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	rec := httptest.NewRecorder()
	body := `{"model":"qwen","messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_schema"}}`

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultChatCompletionsPath, strings.NewReader(body)))

	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), `"param":"response_format"`) {
		t.Fatalf("status = %d body=%s, want 400 param=response_format", rec.Code, rec.Body.String())
	}
}
