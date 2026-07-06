// SPDX-Licence-Identifier: EUPL-1.2

package compat

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	openaicompat "dappco.re/go/inference/serving/provider/openai"
)

// TestNewResolver_Good pins the metal-backend BackendResolver construction:
// backend name fixed to "metal", the model path plumbed through unchanged,
// and every LoadOption preserved for the lazy load.
func TestNewResolver_Good(t *testing.T) {
	r := NewResolver("/models/x", inference.WithContextLen(4096))
	if r.BackendName != "metal" {
		t.Fatalf("BackendName = %q, want metal", r.BackendName)
	}
	if r.ModelPath != "/models/x" {
		t.Fatalf("ModelPath = %q, want /models/x", r.ModelPath)
	}
	if len(r.LoadOptions) != 1 {
		t.Fatalf("LoadOptions = %d, want the one WithContextLen passed in", len(r.LoadOptions))
	}
}

// TestNewHandler_NoBackend_Bad proves the handler NewHandler builds fails
// closed (an error status, not a panic or a 200) when the "metal" backend
// isn't registered in the test binary — the portable, no-GPU build shape.
func TestNewHandler_NoBackend_Bad(t *testing.T) {
	h := NewHandler("/models/does-not-exist")
	rec := httptest.NewRecorder()
	body := `{"model":"whatever","messages":[{"role":"user","content":"hi"}]}`
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, openaicompat.DefaultChatCompletionsPath, strings.NewReader(body)))
	if rec.Code == http.StatusOK {
		t.Fatal("chat completions with no metal backend registered = 200, want an error status")
	}
}

// TestNewModelMux_NoBackend_Bad proves NewModelMux composes the same
// lazily-loading resolver into the full route set, failing closed the same
// way on the chat-completions route.
func TestNewModelMux_NoBackend_Bad(t *testing.T) {
	mux := NewModelMux("/models/does-not-exist")
	rec := httptest.NewRecorder()
	body := `{"model":"whatever","messages":[{"role":"user","content":"hi"}]}`
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, openaicompat.DefaultChatCompletionsPath, strings.NewReader(body)))
	if rec.Code == http.StatusOK {
		t.Fatal("chat completions with no metal backend registered = 200, want an error status")
	}
}

// namedModelNamesResolver implements resolverModelNameLister directly (no
// StaticResolver wrapping) to pin resolverModelNames' lister branch in
// isolation.
type namedModelNamesResolver struct{ names []string }

func (namedModelNamesResolver) ResolveModel(context.Context, string) (inference.TextModel, error) {
	return nil, core.NewError("not implemented")
}
func (r namedModelNamesResolver) ModelNames() []string { return r.names }

// TestResolverModelNames_Lister_Good proves a resolver that implements
// ModelNames() directly is asked first, ahead of the BackendResolver
// fallback.
func TestResolverModelNames_Lister_Good(t *testing.T) {
	fake := namedModelNamesResolver{names: []string{"a", "b"}}
	got := resolverModelNames(fake)
	if len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("resolverModelNames(lister) = %v, want [a b]", got)
	}
}

// TestResolverModelNames_BackendResolver_Good proves a *BackendResolver with
// a model path reports its basename as the single known model.
func TestResolverModelNames_BackendResolver_Good(t *testing.T) {
	r := openaicompat.NewBackendResolver("metal", "/models/gemma4/model.gguf")
	got := resolverModelNames(r)
	if len(got) != 1 || got[0] != "model.gguf" {
		t.Fatalf("resolverModelNames(BackendResolver) = %v, want [model.gguf]", got)
	}
}

// TestResolverModelNames_BackendResolverNoPath_Good proves a *BackendResolver
// with no model path yet (constructed but not configured) reports no names.
func TestResolverModelNames_BackendResolverNoPath_Good(t *testing.T) {
	r := openaicompat.NewBackendResolver("metal", "")
	if got := resolverModelNames(r); got != nil {
		t.Fatalf("resolverModelNames(pathless BackendResolver) = %v, want nil", got)
	}
}

// TestResolverModelNames_Neither_Good proves a resolver that is neither a
// names-lister nor a *BackendResolver (a StaticResolver, or a ResolverFunc)
// reports no names rather than panicking on the failed type assertions.
func TestResolverModelNames_Neither_Good(t *testing.T) {
	resolver := openaicompat.NewStaticResolver(nil)
	if got := resolverModelNames(resolver); got != nil {
		t.Fatalf("resolverModelNames(neither) = %v, want nil", got)
	}
}

// TestFirstStopSequenceCut_NoContentOrStops_Good proves an empty content or
// an empty stop list is a clean no-cut rather than an out-of-range index.
func TestFirstStopSequenceCut_NoContentOrStops_Good(t *testing.T) {
	if idx, cut := firstStopSequenceCut("", []string{"STOP"}); cut || idx != 0 {
		t.Fatalf("firstStopSequenceCut(empty content) = (%d,%v), want (0,false)", idx, cut)
	}
	if idx, cut := firstStopSequenceCut("hello", nil); cut || idx != 0 {
		t.Fatalf("firstStopSequenceCut(no stops) = (%d,%v), want (0,false)", idx, cut)
	}
}

// TestFirstStopSequenceCut_NoMatch_Good proves content with no matching stop
// reports no cut.
func TestFirstStopSequenceCut_NoMatch_Good(t *testing.T) {
	if idx, cut := firstStopSequenceCut("hello world", []string{"STOP", ""}); cut || idx != 0 {
		t.Fatalf("firstStopSequenceCut(no match) = (%d,%v), want (0,false); an empty stop must be skipped, not matched", idx, cut)
	}
}

// TestFirstStopSequenceCut_EarliestWins_Good proves that when multiple stop
// sequences match, the earliest index in the content wins.
func TestFirstStopSequenceCut_EarliestWins_Good(t *testing.T) {
	idx, cut := firstStopSequenceCut("aaa STOP1 bbb STOP2", []string{"STOP2", "STOP1"})
	if !cut || idx != 4 {
		t.Fatalf("firstStopSequenceCut(earliest) = (%d,%v), want (4,true) — STOP1 at index 4 precedes STOP2", idx, cut)
	}
}

// TestNormalizeAnthropicStopSequences_Empty_Good proves a nil/empty input
// passes through as (nil,nil) rather than an empty-but-non-nil slice.
func TestNormalizeAnthropicStopSequences_Empty_Good(t *testing.T) {
	got, err := normalizeAnthropicStopSequences(nil)
	if err != nil || got != nil {
		t.Fatalf("normalizeAnthropicStopSequences(nil) = (%v,%v), want (nil,nil)", got, err)
	}
}

// TestNormalizeAnthropicStopSequences_Valid_Good proves non-empty stop
// strings pass through unchanged.
func TestNormalizeAnthropicStopSequences_Valid_Good(t *testing.T) {
	got, err := normalizeAnthropicStopSequences([]string{"STOP", "END"})
	if err != nil {
		t.Fatalf("normalizeAnthropicStopSequences: %v", err)
	}
	if len(got) != 2 || got[0] != "STOP" || got[1] != "END" {
		t.Fatalf("normalizeAnthropicStopSequences = %v, want [STOP END]", got)
	}
}

// TestNormalizeAnthropicStopSequences_EmptyEntry_Bad proves an empty string
// among the stop sequences is refused (Anthropic's contract: no blank stops).
func TestNormalizeAnthropicStopSequences_EmptyEntry_Bad(t *testing.T) {
	if _, err := normalizeAnthropicStopSequences([]string{"STOP", ""}); err == nil {
		t.Fatal("normalizeAnthropicStopSequences should reject an empty stop_sequences entry")
	}
}

// TestIndexString_Good pins indexString's substring search, including the
// empty-substring and longer-than-haystack edge cases indexOf/strings.Index
// callers rely on.
func TestIndexString_Good(t *testing.T) {
	cases := []struct {
		s, sub string
		want   int
	}{
		{"hello world", "world", 6},
		{"hello world", "xyz", -1},
		{"hello", "", 0},
		{"hi", "hello", -1},
		{"", "", 0},
	}
	for _, c := range cases {
		if got := indexString(c.s, c.sub); got != c.want {
			t.Fatalf("indexString(%q, %q) = %d, want %d", c.s, c.sub, got, c.want)
		}
	}
}

// TestDecodeWireJSON_Good proves the unsized wrapper (contentLength unknown)
// decodes a well-formed body into the target.
func TestDecodeWireJSON_Good(t *testing.T) {
	var into map[string]string
	if err := decodeWireJSON(strings.NewReader(`{"k":"v"}`), &into, "test.scope"); err != nil {
		t.Fatalf("decodeWireJSON: %v", err)
	}
	if into["k"] != "v" {
		t.Fatalf("decoded = %v, want k=v", into)
	}
}

// TestDecodeWireJSON_Malformed_Bad proves malformed JSON surfaces as an
// error through the unsized wrapper too.
func TestDecodeWireJSON_Malformed_Bad(t *testing.T) {
	var into map[string]string
	if err := decodeWireJSON(strings.NewReader(`{not json`), &into, "test.scope"); err == nil {
		t.Fatal("decodeWireJSON should error on malformed JSON")
	}
}

// TestReasoningText_Empty_Good proves no reasoning segments yields an empty
// string rather than allocating an empty builder's output.
func TestReasoningText_Empty_Good(t *testing.T) {
	if got := reasoningText(nil); got != "" {
		t.Fatalf("reasoningText(nil) = %q, want empty", got)
	}
}

// TestReasoningText_Concat_Good proves multiple reasoning segments are
// concatenated in order.
func TestReasoningText_Concat_Good(t *testing.T) {
	segments := []inference.ReasoningSegment{{Text: "first "}, {Text: "second"}}
	if got, want := reasoningText(segments), "first second"; got != want {
		t.Fatalf("reasoningText = %q, want %q", got, want)
	}
}

// TestRequireCompatMethod_NilRequest_Bad proves a nil *http.Request is
// refused with 400 rather than a nil-pointer panic on r.Method.
func TestRequireCompatMethod_NilRequest_Bad(t *testing.T) {
	rec := httptest.NewRecorder()
	if requireCompatMethod(rec, nil, http.MethodGet) {
		t.Fatal("requireCompatMethod(nil request) should report false")
	}
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", rec.Code)
	}
}

// TestResolveCompatModel_NilResolver_Bad proves an unconfigured (nil)
// resolver is refused with 503 rather than a nil-pointer panic on
// resolver.ResolveModel.
func TestResolveCompatModel_NilResolver_Bad(t *testing.T) {
	rec := httptest.NewRecorder()
	model, ok := resolveCompatModel(rec, context.Background(), nil, "any-model")
	if ok || model != nil {
		t.Fatalf("resolveCompatModel(nil resolver) = (%v,%v), want (nil,false)", model, ok)
	}
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503", rec.Code)
	}
}

// TestJSONHTMLSafe pins the pass-through predicate encoding/json's HTML-safe
// default keys on: printable ASCII passes, but the control range, the JSON
// structural bytes (" \), and the HTML-meta bytes (< > &) must be escaped.
func TestJSONHTMLSafe(t *testing.T) {
	for _, b := range []byte{'a', ' ', '~', '{', '}', '@'} {
		if !jsonHTMLSafe(b) {
			t.Fatalf("jsonHTMLSafe(%q) = false, want true (ordinary byte)", b)
		}
	}
	for _, b := range []byte{0x00, 0x1f, '"', '\\', '<', '>', '&'} {
		if jsonHTMLSafe(b) {
			t.Fatalf("jsonHTMLSafe(%q) = true, want false (must be escaped)", b)
		}
	}
}

// TestHexNibble pins the lowercase-hex digit mapping the \u00XX / \u202X escape
// branches emit, including that only the low nibble is read.
func TestHexNibble(t *testing.T) {
	cases := map[byte]byte{0x0: '0', 0x9: '9', 0xa: 'a', 0xf: 'f', 0x1f: 'f'}
	for in, want := range cases {
		if got := hexNibble(in); got != want {
			t.Fatalf("hexNibble(%#x) = %q, want %q", in, got, want)
		}
	}
}

// TestAppendJSONStringHTML pins the hand-rolled encoder against the contract it
// claims: byte-identity with encoding/json's HTML-safe Marshal (core
// .JSONMarshalString of a string). The HTML-meta escaping is the load-bearing
// case — a streamed code/markup delta routinely carries < > &, and a naive
// encoder that skips them corrupts the wire.
func TestAppendJSONStringHTML(t *testing.T) {
	cases := []string{
		"",
		"hello world",
		`a"b\c`,
		"tab\tnewline\ncarriage\r",
		"<div>&nbsp;</div>",
		"null\x00byte\x01",
		"café — naïve",     // multibyte pass-through
		" line par",        // separators encoding/json escapes
		"\xff\xfe invalid", // invalid UTF-8 -> �
	}
	for _, s := range cases {
		got := string(appendJSONStringHTML(nil, s))
		want := core.JSONMarshalString(s)
		if got != want {
			t.Fatalf("appendJSONStringHTML(%q) = %q, want %q (encoding/json parity)", s, got, want)
		}
	}
	// Explicitly document the HTML-meta contract: < > & become the escaped
	// < > & sequences, never raw.
	if got, want := string(appendJSONStringHTML(nil, "<>&")), "\"\\u003c\\u003e\\u0026\""; got != want {
		t.Fatalf("HTML-meta escaping = %q, want %q", got, want)
	}
}

// FuzzAppendJSONStringHTML is the equivalence lock the appendJSONStringHTML doc
// references: for ANY input the hand-rolled fast-path encoder must produce the
// exact bytes encoding/json would (core.JSONMarshalString), so the streaming
// wire encoders can stay off the reflect path without drifting from it.
func FuzzAppendJSONStringHTML(f *testing.F) {
	for _, s := range []string{"", "plain", `q"o\o`, "<>&", "\x00\x1f", " ", "\xff", "café"} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, s string) {
		got := string(appendJSONStringHTML(nil, s))
		if want := core.JSONMarshalString(s); got != want {
			t.Fatalf("appendJSONStringHTML(%q) = %q, want %q", s, got, want)
		}
	})
}

// TestWriteSSEData pins the OpenAI streaming frame shape: "data: <payload>\n\n".
func TestWriteSSEData(t *testing.T) {
	buf := core.NewBuffer()
	writeSSEData(buf, `{"x":1}`)
	if got, want := buf.String(), "data: {\"x\":1}\n\n"; got != want {
		t.Fatalf("writeSSEData = %q, want %q", got, want)
	}
}

// TestWriteSSEEvent pins the Anthropic streaming frame shape:
// "event: <name>\ndata: <payload>\n\n".
func TestWriteSSEEvent(t *testing.T) {
	buf := core.NewBuffer()
	writeSSEEvent(buf, "message_start", `{"y":2}`)
	if got, want := buf.String(), "event: message_start\ndata: {\"y\":2}\n\n"; got != want {
		t.Fatalf("writeSSEEvent = %q, want %q", got, want)
	}
}

// TestWriteNDJSONLine pins the Ollama streaming frame shape: "<payload>\n".
func TestWriteNDJSONLine(t *testing.T) {
	buf := core.NewBuffer()
	writeNDJSONLine(buf, `{"z":3}`)
	if got, want := buf.String(), "{\"z\":3}\n"; got != want {
		t.Fatalf("writeNDJSONLine = %q, want %q", got, want)
	}
}

// TestWriteResponseDeltaFrame pins the /v1/responses per-token delta frame and
// proves the delta is JSON-escaped (HTML-safe) rather than spliced raw.
func TestWriteResponseDeltaFrame(t *testing.T) {
	buf := core.NewBuffer()
	writeResponseDeltaFrame(buf, nil, "a<b")
	want := `data: {"type":"response.output_text.delta","delta":` + core.JSONMarshalString("a<b") + "}\n\n"
	if got := buf.String(); got != want {
		t.Fatalf("writeResponseDeltaFrame = %q, want %q", got, want)
	}
}

// TestWriteOllamaChatFrame pins the /api/chat per-token frame, model and content
// both JSON-escaped into the fixed punctuation.
func TestWriteOllamaChatFrame(t *testing.T) {
	buf := core.NewBuffer()
	writeOllamaChatFrame(buf, nil, "gemma", "hi&bye")
	want := `{"model":` + core.JSONMarshalString("gemma") +
		`,"message":{"role":"assistant","content":` + core.JSONMarshalString("hi&bye") +
		"},\"done\":false}\n"
	if got := buf.String(); got != want {
		t.Fatalf("writeOllamaChatFrame = %q, want %q", got, want)
	}
}

// TestWriteOllamaGenerateFrame pins the /api/generate per-token frame shape.
func TestWriteOllamaGenerateFrame(t *testing.T) {
	buf := core.NewBuffer()
	writeOllamaGenerateFrame(buf, nil, "gemma", "42>0")
	want := `{"model":` + core.JSONMarshalString("gemma") +
		`,"response":` + core.JSONMarshalString("42>0") +
		",\"done\":false}\n"
	if got := buf.String(); got != want {
		t.Fatalf("writeOllamaGenerateFrame = %q, want %q", got, want)
	}
}
