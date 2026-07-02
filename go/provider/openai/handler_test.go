// SPDX-Licence-Identifier: EUPL-1.2

// Tests for the chat-completions net/http route and its JSON
// response/error helpers. The happy-path non-streaming and streaming
// cases live in openai_test.go (TestOpenAI_Handler_Good_*); this file
// covers Handler.ServeHTTP's early-return branches, serveStreaming's
// stop/thought/error/length-cap branches, and the small error/result
// helpers at the bottom of handler.go.
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

// TestResultError_Good_Bad_Ugly covers all three shapes: OK (nil),
// failure carrying a genuine error, and failure carrying a
// non-error Value (hand-constructed — production call sites never
// build a Result this way, but the function must not panic on it).
func TestResultError_Good_Bad_Ugly(t *testing.T) {
	if err := resultError(core.Result{OK: true}); err != nil {
		t.Fatalf("resultError(OK) = %v, want nil", err)
	}

	wrapped := core.E("test", "boom", nil)
	if err := resultError(core.Result{OK: false, Value: wrapped}); err != wrapped {
		t.Fatalf("resultError(error Value) = %v, want %v", err, wrapped)
	}

	if err := resultError(core.Result{OK: false, Value: "not an error"}); err == nil {
		t.Fatal("resultError(non-error Value) = nil, want the unexpected-value fallback error")
	}
}
