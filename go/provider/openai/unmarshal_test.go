// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"encoding/json"
	"reflect"
	"testing"

	"dappco.re/go/inference/jsonenc"
)

// TestUnmarshalChatCompletionRequest_ThinkingControls pins the hand-rolled
// decoder for the reasoning toggle: reasoning_effort (top-level string) and
// chat_template_kwargs.enable_thinking (nested object, vLLM/SGLang convention).
func TestUnmarshalChatCompletionRequest_ThinkingControls(t *testing.T) {
	in := `{"model":"m","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"none","chat_template_kwargs":{"foo":"bar","enable_thinking":false}}`
	var req ChatCompletionRequest
	if err := json.Unmarshal([]byte(in), &req); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if req.ReasoningEffort != "none" {
		t.Fatalf("ReasoningEffort = %q, want %q", req.ReasoningEffort, "none")
	}
	if req.ChatTemplateKwargs == nil || req.ChatTemplateKwargs.EnableThinking == nil || *req.ChatTemplateKwargs.EnableThinking {
		t.Fatalf("ChatTemplateKwargs.EnableThinking = %+v, want &false", req.ChatTemplateKwargs)
	}
}

// TestUnmarshalChatCompletionRequest_DirectShapes pins the hand-rolled
// decoder against direct JSON literals. Locks the per-field dispatch
// — present / absent / null variants of every pointer field, the
// StopList variant shape (string vs array), escape-heavy strings,
// multi-turn arrays.
func TestUnmarshalChatCompletionRequest_DirectShapes(t *testing.T) {
	temp := float32(0.7)
	topP := float32(0.95)
	minP := float32(0.05)
	topK := 64
	maxTok := 1024
	cases := []struct {
		name string
		in   string
		want ChatCompletionRequest
	}{
		{
			name: "minimal",
			in:   `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`,
			want: ChatCompletionRequest{
				Model:    "gpt-4",
				Messages: []ChatMessage{{Role: "user", Content: "hi"}},
			},
		},
		{
			name: "all-optional-fields-set",
			in:   `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"temperature":0.7,"top_p":0.95,"min_p":0.05,"top_k":64,"max_tokens":1024,"stream":true,"stop":["</s>"],"user":"u123"}`,
			want: ChatCompletionRequest{
				Model:       "gpt-4",
				Messages:    []ChatMessage{{Role: "user", Content: "hi"}},
				Temperature: &temp,
				TopP:        &topP,
				MinP:        &minP,
				TopK:        &topK,
				MaxTokens:   &maxTok,
				Stream:      true,
				Stop:        StopList{"</s>"},
				User:        "u123",
			},
		},
		{
			name: "stop-as-string",
			in:   `{"model":"gpt-4","messages":[],"stop":"END"}`,
			want: ChatCompletionRequest{
				Model:    "gpt-4",
				Messages: nil,
				Stop:     StopList{"END"},
			},
		},
		{
			name: "pointer-fields-null-keeps-zero",
			in:   `{"model":"gpt-4","messages":[],"temperature":null,"top_p":null,"min_p":null,"top_k":null,"max_tokens":null,"stream":null}`,
			want: ChatCompletionRequest{
				Model: "gpt-4",
			},
		},
		{
			name: "unknown-fields-ignored",
			in:   `{"model":"gpt-4","messages":[],"future":42,"extra":"x"}`,
			want: ChatCompletionRequest{
				Model: "gpt-4",
			},
		},
		{
			name: "whitespace-friendly",
			in: `{
				"model": "gpt-4",
				"messages": [
					{ "role": "user", "content": "hi" }
				]
			}`,
			want: ChatCompletionRequest{
				Model:    "gpt-4",
				Messages: []ChatMessage{{Role: "user", Content: "hi"}},
			},
		},
		{
			name: "escape-heavy",
			in:   `{"model":"gpt-4","messages":[{"role":"user","content":"a\nb \"c\" \\d"}]}`,
			want: ChatCompletionRequest{
				Model:    "gpt-4",
				Messages: []ChatMessage{{Role: "user", Content: "a\nb \"c\" \\d"}},
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var got ChatCompletionRequest
			if err := json.Unmarshal([]byte(tc.in), &got); err != nil {
				t.Fatalf("Unmarshal error = %v", err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("Unmarshal mismatch\ngot:  %+v\nwant: %+v", got, tc.want)
			}
		})
	}
}

func TestUnmarshalResponseRequest_DirectShapes(t *testing.T) {
	temp := float32(0.7)
	minP := float32(0.04)
	maxOut := 256
	cases := []struct {
		name string
		in   string
		want ResponseRequest
	}{
		{
			name: "minimal",
			in:   `{"model":"gpt-4","input":[{"role":"user","content":"hi"}]}`,
			want: ResponseRequest{
				Model: "gpt-4",
				Input: []ResponseInputMessage{{Role: "user", Content: "hi"}},
			},
		},
		{
			name: "with-instructions-and-options",
			in:   `{"model":"gpt-4","input":[{"role":"user","content":"hi"}],"instructions":"sys","temperature":0.7,"min_p":0.04,"max_output_tokens":256,"stream":true}`,
			want: ResponseRequest{
				Model:           "gpt-4",
				Input:           []ResponseInputMessage{{Role: "user", Content: "hi"}},
				Instructions:    "sys",
				Temperature:     &temp,
				MinP:            &minP,
				MaxOutputTokens: &maxOut,
				Stream:          true,
			},
		},
		{
			name: "stop-as-array",
			in:   `{"model":"gpt-4","input":[],"stop":["</s>","x"]}`,
			want: ResponseRequest{
				Model: "gpt-4",
				Stop:  StopList{"</s>", "x"},
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var got ResponseRequest
			if err := json.Unmarshal([]byte(tc.in), &got); err != nil {
				t.Fatalf("Unmarshal error = %v", err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("Unmarshal mismatch\ngot:  %+v\nwant: %+v", got, tc.want)
			}
		})
	}
}

// TestUnmarshalChatCompletionRequest_InvalidShapes asserts cleanly
// rejected error shapes — no panics, just errors.
func TestUnmarshalChatCompletionRequest_InvalidShapes(t *testing.T) {
	cases := []string{
		``,
		`{`,
		`}`,
		`{"messages":not-an-array}`,
		`{"temperature":"hot"}`,
	}
	for _, in := range cases {
		t.Run(in, func(t *testing.T) {
			var req ChatCompletionRequest
			if err := json.Unmarshal([]byte(in), &req); err == nil {
				t.Fatalf("Unmarshal(%q) returned nil error", in)
			}
		})
	}
}

// The tests below call UnmarshalJSON (and the unexported parse*
// walkers) directly rather than through encoding/json.Unmarshal.
// encoding/json.Unmarshal runs a whole-document checkValid syntax
// scan before ever invoking a custom UnmarshalJSON method — most
// byte-level malformations (bare keys, missing colons, unterminated
// objects) are generic JSON syntax errors checkValid rejects before
// our hand-rolled walker gets a chance to run. Calling UnmarshalJSON
// directly reaches those branches exactly as a streaming decoder
// would. Every error in this file's hand-rolled walker resolves to
// the single jsonenc.ErrInvalidJSON sentinel, so failures assert
// identity against it.

// TestUnmarshal_ChatCompletionRequest_Bad drives every malformed-
// shape branch in ChatCompletionRequest.UnmarshalJSON/unmarshalField
// not already reached by DirectShapes/InvalidShapes/ThinkingControls:
// a non-object top level, a malformed key, structural EOF/trailing-
// garbage at every loop boundary, and a wrong-typed value for each
// field.
func TestUnmarshal_ChatCompletionRequest_Bad(t *testing.T) {
	cases := []struct{ name, in string }{
		{"not-an-object", `42`},
		{"non-string-key", `{1:"x"}`},
		{"unterminated-key", `{"model`},
		{"missing-colon", `{"model" "x"}`},
		{"eof-after-value", `{"model":"x"`},
		{"trailing-garbage", `{"model":"x"]`},
		{"model-wrong-type", `{"model":42}`},
		{"messages-wrong-type", `{"messages":42}`},
		{"messages-element-wrong-type", `{"messages":[{"role":123}]}`},
		{"temperature-wrong-type", `{"temperature":"hot"}`},
		{"top_p-wrong-type", `{"top_p":"hot"}`},
		{"min_p-wrong-type", `{"min_p":"hot"}`},
		{"top_k-wrong-type", `{"top_k":"hot"}`},
		{"max_tokens-wrong-type", `{"max_tokens":"hot"}`},
		{"reasoning_effort-wrong-type", `{"reasoning_effort":42}`},
		{"chat_template_kwargs-wrong-type", `{"chat_template_kwargs":42}`},
		{"chat_template_kwargs-unterminated", `{"chat_template_kwargs":{`},
		{"stream-wrong-type", `{"stream":"nope"}`},
		{"stop-unskippable", `{"stop":bogus}`},
		{"stop-wrong-shape", `{"stop":42}`},
		{"user-wrong-type", `{"user":42}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var req ChatCompletionRequest
			if err := req.UnmarshalJSON([]byte(tc.in)); err != jsonenc.ErrInvalidJSON {
				t.Fatalf("UnmarshalJSON(%q) error = %v, want jsonenc.ErrInvalidJSON", tc.in, err)
			}
		})
	}
}

// TestUnmarshal_ChatCompletionRequest_Good covers the empty-object
// fast path and "chat_template_kwargs":null — neither exercised by
// DirectShapes (which nulls every other pointer field but not this
// one).
func TestUnmarshal_ChatCompletionRequest_Good(t *testing.T) {
	var empty ChatCompletionRequest
	if err := empty.UnmarshalJSON([]byte(`{}`)); err != nil || !reflect.DeepEqual(empty, ChatCompletionRequest{}) {
		t.Fatalf("UnmarshalJSON(%q) = %+v, err = %v", `{}`, empty, err)
	}

	var kwargsNull ChatCompletionRequest
	in := `{"model":"m","chat_template_kwargs":null}`
	if err := kwargsNull.UnmarshalJSON([]byte(in)); err != nil {
		t.Fatalf("UnmarshalJSON(%q) error = %v", in, err)
	}
	if kwargsNull.ChatTemplateKwargs != nil {
		t.Fatalf("UnmarshalJSON(%q) ChatTemplateKwargs = %+v, want nil", in, kwargsNull.ChatTemplateKwargs)
	}
}

// TestUnmarshal_ParseChatMessageArray_Bad drives parseChatMessageArray's
// own malformed-shape branches directly.
func TestUnmarshal_ParseChatMessageArray_Bad(t *testing.T) {
	cases := []struct{ name, in string }{
		{"not-an-array", `{}`},
		{"element-error", `[{"role":42}]`},
		{"eof-after-element", `[{"role":"user"}`},
		{"trailing-garbage", `[{"role":"user"}oops]`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, err := parseChatMessageArray([]byte(tc.in), 0); err != jsonenc.ErrInvalidJSON {
				t.Fatalf("parseChatMessageArray(%q) error = %v, want jsonenc.ErrInvalidJSON", tc.in, err)
			}
		})
	}
}

// TestUnmarshal_ParseChatMessageArray_Good covers the null literal.
func TestUnmarshal_ParseChatMessageArray_Good(t *testing.T) {
	msgs, next, err := parseChatMessageArray([]byte(`null`), 0)
	if err != nil || msgs != nil || next != 4 {
		t.Fatalf("parseChatMessageArray(null) = %v, %d, %v", msgs, next, err)
	}
}

// TestUnmarshal_ParseChatMessage_Bad drives parseChatMessage's own
// malformed-shape branches directly, including the multimodal
// content-array cold path (#98).
func TestUnmarshal_ParseChatMessage_Bad(t *testing.T) {
	cases := []struct{ name, in string }{
		{"not-an-object", `"str"`},
		{"non-string-key", `{1:"x"}`},
		{"unterminated-key", `{"role`},
		{"missing-colon", `{"role" "user"}`},
		{"role-wrong-type", `{"role":42}`},
		{"content-string-wrong-type", `{"role":"user","content":42}`},
		{"content-array-unskippable", `{"role":"user","content":[bogus]}`},
		{"unknown-field-malformed-value", `{"extra":bogus}`},
		{"eof-after-value", `{"role":"user"`},
		{"trailing-garbage", `{"role":"user"]`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, err := parseChatMessage([]byte(tc.in), 0); err != jsonenc.ErrInvalidJSON {
				t.Fatalf("parseChatMessage(%q) error = %v, want jsonenc.ErrInvalidJSON", tc.in, err)
			}
		})
	}
}

// TestUnmarshal_ParseChatMessage_Bad_ContentArrayElementWrongType
// covers the content-part-array element decode failure separately —
// that path delegates to core.JSONUnmarshal (stdlib reflect, not the
// jsonenc walker), so the error is a stdlib json.UnmarshalTypeError,
// not the jsonenc.ErrInvalidJSON sentinel every other branch in this
// file resolves to.
func TestUnmarshal_ParseChatMessage_Bad_ContentArrayElementWrongType(t *testing.T) {
	in := `{"role":"user","content":[{"type":42}]}`
	if _, _, err := parseChatMessage([]byte(in), 0); err == nil {
		t.Fatalf("parseChatMessage(%q) returned nil error", in)
	}
}

// TestUnmarshal_ParseChatMessage_Good covers the empty-object fast
// path, the content:null case, and an unknown field skipped ahead of
// role/content — none exercised by the ChatCompletionRequest-level
// DirectShapes table (which always sends fully-formed messages).
func TestUnmarshal_ParseChatMessage_Good(t *testing.T) {
	msg, next, err := parseChatMessage([]byte(`{}`), 0)
	if err != nil || !reflect.DeepEqual(msg, ChatMessage{}) || next != 2 {
		t.Fatalf("parseChatMessage(%q) = %+v, %d, %v", `{}`, msg, next, err)
	}

	contentNull, _, err := parseChatMessage([]byte(`{"role":"user","content":null}`), 0)
	if err != nil || contentNull.Content != "" {
		t.Fatalf("parseChatMessage(content:null) = %+v, err = %v", contentNull, err)
	}

	in := `{"extra":"ignored","role":"user","content":"hi"}`
	msg, _, err = parseChatMessage([]byte(in), 0)
	if err != nil {
		t.Fatalf("parseChatMessage(%q) error = %v", in, err)
	}
	if want := (ChatMessage{Role: "user", Content: "hi"}); !reflect.DeepEqual(msg, want) {
		t.Fatalf("parseChatMessage(%q) = %+v, want %+v", in, msg, want)
	}

	in = `{"role":"user","content":[{"type":"text","text":"What is in"},{"type":"image_url","image_url":{"url":"data:image/png;base64,UE5H"}},{"type":"text","text":"this?"}]}`
	msg, _, err = parseChatMessage([]byte(in), 0)
	if err != nil {
		t.Fatalf("parseChatMessage(content-array) error = %v", err)
	}
	if msg.Content != "What is in\nthis?" || len(msg.Images) != 1 {
		t.Fatalf("parseChatMessage(content-array) = %+v", msg)
	}
}

// TestUnmarshal_ResponseRequest_Bad mirrors TestUnmarshal_ChatCompletionRequest_Bad
// for the Responses API request shape.
func TestUnmarshal_ResponseRequest_Bad(t *testing.T) {
	cases := []struct{ name, in string }{
		{"not-an-object", `42`},
		{"non-string-key", `{1:"x"}`},
		{"unterminated-key", `{"model`},
		{"missing-colon", `{"model" "x"}`},
		{"eof-after-value", `{"model":"x"`},
		{"trailing-garbage", `{"model":"x"]`},
		{"model-wrong-type", `{"model":42}`},
		{"input-wrong-type", `{"input":42}`},
		{"input-element-wrong-type", `{"input":[{"role":42}]}`},
		{"instructions-wrong-type", `{"instructions":42}`},
		{"temperature-wrong-type", `{"temperature":"hot"}`},
		{"top_p-wrong-type", `{"top_p":"hot"}`},
		{"min_p-wrong-type", `{"min_p":"hot"}`},
		{"top_k-wrong-type", `{"top_k":"hot"}`},
		{"max_output_tokens-wrong-type", `{"max_output_tokens":"hot"}`},
		{"stream-wrong-type", `{"stream":"nope"}`},
		{"stop-unskippable", `{"stop":bogus}`},
		{"stop-wrong-shape", `{"stop":42}`},
		{"user-wrong-type", `{"user":42}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var req ResponseRequest
			if err := req.UnmarshalJSON([]byte(tc.in)); err != jsonenc.ErrInvalidJSON {
				t.Fatalf("UnmarshalJSON(%q) error = %v, want jsonenc.ErrInvalidJSON", tc.in, err)
			}
		})
	}
}

// TestUnmarshal_ResponseRequest_Good covers the empty-object fast
// path and every null-pointer-field variant (temperature/top_p/
// min_p/top_k/max_output_tokens/stream) in one pass — the existing
// DirectShapes table never nulls any of them.
func TestUnmarshal_ResponseRequest_Good(t *testing.T) {
	var empty ResponseRequest
	if err := empty.UnmarshalJSON([]byte(`{}`)); err != nil || !reflect.DeepEqual(empty, ResponseRequest{}) {
		t.Fatalf("UnmarshalJSON(%q) = %+v, err = %v", `{}`, empty, err)
	}

	var nulled ResponseRequest
	in := `{"model":"m","temperature":null,"top_p":null,"min_p":null,"top_k":null,"max_output_tokens":null,"stream":null}`
	if err := nulled.UnmarshalJSON([]byte(in)); err != nil {
		t.Fatalf("UnmarshalJSON(%q) error = %v", in, err)
	}
	if want := (ResponseRequest{Model: "m"}); !reflect.DeepEqual(nulled, want) {
		t.Fatalf("UnmarshalJSON(%q) = %+v, want %+v", in, nulled, want)
	}

	// top_p / top_k / user are set on ChatCompletionRequest's
	// DirectShapes table but never on ResponseRequest's — cover their
	// value-assigned success path plus an unknown field skipped ahead
	// of a known one.
	topP := float32(0.9)
	topK := 40
	var withOptions ResponseRequest
	in = `{"future":42,"model":"m","top_p":0.9,"top_k":40,"user":"u1"}`
	if err := withOptions.UnmarshalJSON([]byte(in)); err != nil {
		t.Fatalf("UnmarshalJSON(%q) error = %v", in, err)
	}
	want := ResponseRequest{Model: "m", TopP: &topP, TopK: &topK, User: "u1"}
	if !reflect.DeepEqual(withOptions, want) {
		t.Fatalf("UnmarshalJSON(%q) = %+v, want %+v", in, withOptions, want)
	}
}

// TestUnmarshal_ParseChatTemplateKwargs_Bad drives parseChatTemplateKwargs's
// own malformed-shape branches directly.
func TestUnmarshal_ParseChatTemplateKwargs_Bad(t *testing.T) {
	cases := []struct{ name, in string }{
		{"not-an-object", `"str"`},
		{"non-string-key", `{1:"x"}`},
		{"unterminated-key", `{"enable_thinking`},
		{"missing-colon", `{"enable_thinking" true}`},
		{"enable_thinking-wrong-type", `{"enable_thinking":"nope"}`},
		{"thinking_budget-wrong-type", `{"thinking_budget":"nope"}`},
		{"unknown-field-malformed-value", `{"extra":bogus}`},
		{"eof-after-value", `{"enable_thinking":true`},
		{"trailing-garbage", `{"enable_thinking":true]`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, err := parseChatTemplateKwargs([]byte(tc.in), 0); err != jsonenc.ErrInvalidJSON {
				t.Fatalf("parseChatTemplateKwargs(%q) error = %v, want jsonenc.ErrInvalidJSON", tc.in, err)
			}
		})
	}
}

// TestUnmarshal_ParseChatTemplateKwargs_Good covers the empty-object
// fast path, both fields' null handling, and an unknown-field skip.
func TestUnmarshal_ParseChatTemplateKwargs_Good(t *testing.T) {
	kw, next, err := parseChatTemplateKwargs([]byte(`{}`), 0)
	if err != nil || kw == nil || *kw != (ChatTemplateKwargs{}) || next != 2 {
		t.Fatalf("parseChatTemplateKwargs(%q) = %+v, %d, %v", `{}`, kw, next, err)
	}

	in := `{"extra":"ignored","enable_thinking":null,"thinking_budget":null}`
	kw, _, err = parseChatTemplateKwargs([]byte(in), 0)
	if err != nil {
		t.Fatalf("parseChatTemplateKwargs(%q) error = %v", in, err)
	}
	if kw.EnableThinking != nil || kw.ThinkingBudget != nil {
		t.Fatalf("parseChatTemplateKwargs(%q) = %+v, want both nil", in, kw)
	}
}

// TestUnmarshal_ParseResponseInputMessageArray_Bad drives the array
// walker's own malformed-shape branches directly.
func TestUnmarshal_ParseResponseInputMessageArray_Bad(t *testing.T) {
	cases := []struct{ name, in string }{
		{"not-an-array", `{}`},
		{"element-error", `[{"role":42}]`},
		{"eof-after-element", `[{"role":"user"}`},
		{"trailing-garbage", `[{"role":"user"}oops]`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, err := parseResponseInputMessageArray([]byte(tc.in), 0); err != jsonenc.ErrInvalidJSON {
				t.Fatalf("parseResponseInputMessageArray(%q) error = %v, want jsonenc.ErrInvalidJSON", tc.in, err)
			}
		})
	}
}

// TestUnmarshal_ParseResponseInputMessageArray_Good covers the null
// literal and a 2-element array (comma-continuation).
func TestUnmarshal_ParseResponseInputMessageArray_Good(t *testing.T) {
	msgs, next, err := parseResponseInputMessageArray([]byte(`null`), 0)
	if err != nil || msgs != nil || next != 4 {
		t.Fatalf("parseResponseInputMessageArray(null) = %v, %d, %v", msgs, next, err)
	}

	in := `[{"role":"user","content":"a"},{"role":"assistant","content":"b"}]`
	msgs, next, err = parseResponseInputMessageArray([]byte(in), 0)
	if err != nil {
		t.Fatalf("parseResponseInputMessageArray(%q) error = %v", in, err)
	}
	want := []ResponseInputMessage{{Role: "user", Content: "a"}, {Role: "assistant", Content: "b"}}
	if !reflect.DeepEqual(msgs, want) || next != len(in) {
		t.Fatalf("parseResponseInputMessageArray(%q) = %+v, %d; want %+v, %d", in, msgs, next, want, len(in))
	}
}

// TestUnmarshal_ParseResponseInputMessage_Bad drives
// parseResponseInputMessage's own malformed-shape branches directly.
func TestUnmarshal_ParseResponseInputMessage_Bad(t *testing.T) {
	cases := []struct{ name, in string }{
		{"not-an-object", `"str"`},
		{"non-string-key", `{1:"x"}`},
		{"unterminated-key", `{"role`},
		{"missing-colon", `{"role" "user"}`},
		{"role-wrong-type", `{"role":42}`},
		{"content-wrong-type", `{"role":"user","content":42}`},
		{"unknown-field-malformed-value", `{"extra":bogus}`},
		{"eof-after-value", `{"role":"user"`},
		{"trailing-garbage", `{"role":"user"]`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, err := parseResponseInputMessage([]byte(tc.in), 0); err != jsonenc.ErrInvalidJSON {
				t.Fatalf("parseResponseInputMessage(%q) error = %v, want jsonenc.ErrInvalidJSON", tc.in, err)
			}
		})
	}
}

// TestUnmarshal_ParseResponseInputMessage_Good covers the empty-
// object fast path and an unknown field skipped ahead of role/content.
func TestUnmarshal_ParseResponseInputMessage_Good(t *testing.T) {
	msg, next, err := parseResponseInputMessage([]byte(`{}`), 0)
	if err != nil || msg != (ResponseInputMessage{}) || next != 2 {
		t.Fatalf("parseResponseInputMessage(%q) = %+v, %d, %v", `{}`, msg, next, err)
	}

	in := `{"extra":"ignored","role":"user","content":"hi"}`
	msg, _, err = parseResponseInputMessage([]byte(in), 0)
	if err != nil {
		t.Fatalf("parseResponseInputMessage(%q) error = %v", in, err)
	}
	if want := (ResponseInputMessage{Role: "user", Content: "hi"}); msg != want {
		t.Fatalf("parseResponseInputMessage(%q) = %+v, want %+v", in, msg, want)
	}
}
