// SPDX-Licence-Identifier: EUPL-1.2

package ollama

import (
	"encoding/json"
	"reflect"
	"testing"

	"dappco.re/go/inference/jsonenc"
)

func TestUnmarshalChatRequest_DirectShapes(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want ChatRequest
	}{
		{
			name: "minimal",
			in:   `{"model":"qwen3","messages":[{"role":"user","content":"hi"}]}`,
			want: ChatRequest{
				Model:    "qwen3",
				Messages: []Message{{Role: "user", Content: "hi"}},
			},
		},
		{
			name: "with-stream-and-options",
			in:   `{"model":"qwen3","messages":[],"stream":true,"options":{"temperature":0.7,"top_k":64,"top_p":0.95,"min_p":0.05,"num_predict":256}}`,
			want: ChatRequest{
				Model:   "qwen3",
				Stream:  true,
				Options: Options{Temperature: 0.7, TopK: 64, TopP: 0.95, MinP: 0.05, NumPredict: 256},
			},
		},
		{
			name: "unknown-fields-ignored",
			in:   `{"model":"qwen3","messages":[],"future":42,"options":{"unknown":"x","temperature":0.5}}`,
			want: ChatRequest{
				Model:   "qwen3",
				Options: Options{Temperature: 0.5},
			},
		},
		{
			name: "options-null",
			in:   `{"model":"qwen3","messages":[],"options":null}`,
			want: ChatRequest{
				Model: "qwen3",
			},
		},
		{
			name: "escape-heavy",
			in:   `{"model":"qwen3","messages":[{"role":"user","content":"a\nb"}]}`,
			want: ChatRequest{
				Model:    "qwen3",
				Messages: []Message{{Role: "user", Content: "a\nb"}},
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var got ChatRequest
			if err := json.Unmarshal([]byte(tc.in), &got); err != nil {
				t.Fatalf("Unmarshal error = %v", err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got:  %+v\nwant: %+v", got, tc.want)
			}
		})
	}
}

func TestUnmarshalGenerateRequest_DirectShapes(t *testing.T) {
	in := `{"model":"qwen3","prompt":"hi","stream":true,"options":{"temperature":0.7,"top_p":0.9,"min_p":0.04,"num_predict":128}}`
	want := GenerateRequest{
		Model:   "qwen3",
		Prompt:  "hi",
		Stream:  true,
		Options: Options{Temperature: 0.7, TopP: 0.9, MinP: 0.04, NumPredict: 128},
	}
	var got GenerateRequest
	if err := json.Unmarshal([]byte(in), &got); err != nil {
		t.Fatalf("Unmarshal error = %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got:  %+v\nwant: %+v", got, want)
	}
}

func TestUnmarshalChatResponse_DirectShapes(t *testing.T) {
	in := `{"model":"qwen3","message":{"role":"assistant","content":"answer"},"done":true,"prompt_eval_count":10,"eval_count":5,"total_duration":1500000000}`
	want := ChatResponse{
		Model:           "qwen3",
		Message:         Message{Role: "assistant", Content: "answer"},
		Done:            true,
		PromptEvalCount: 10,
		EvalCount:       5,
		TotalDuration:   1500000000,
	}
	var got ChatResponse
	if err := json.Unmarshal([]byte(in), &got); err != nil {
		t.Fatalf("Unmarshal error = %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got:  %+v\nwant: %+v", got, want)
	}
}

func TestUnmarshalGenerateResponse_DirectShapes(t *testing.T) {
	in := `{"model":"qwen3","response":"hi","done":true,"prompt_eval_count":4,"eval_count":2}`
	want := GenerateResponse{
		Model:           "qwen3",
		Response:        "hi",
		Done:            true,
		PromptEvalCount: 4,
		EvalCount:       2,
	}
	var got GenerateResponse
	if err := json.Unmarshal([]byte(in), &got); err != nil {
		t.Fatalf("Unmarshal error = %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got:  %+v\nwant: %+v", got, want)
	}
}

func TestUnmarshalTagsResponse_DirectShapes(t *testing.T) {
	in := `{"models":[{"name":"qwen3:latest","model":"qwen3","modified_at":"2026-05-21T10:00:00Z","size":4000000000},{"name":"llama3:8b","model":"llama3","size":5000000000}]}`
	want := TagsResponse{
		Models: []ModelTag{
			{Name: "qwen3:latest", Model: "qwen3", ModifiedAt: "2026-05-21T10:00:00Z", Size: 4000000000},
			{Name: "llama3:8b", Model: "llama3", Size: 5000000000},
		},
	}
	var got TagsResponse
	if err := json.Unmarshal([]byte(in), &got); err != nil {
		t.Fatalf("Unmarshal error = %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got:  %+v\nwant: %+v", got, want)
	}
}

func TestUnmarshalChatRequest_InvalidShapes(t *testing.T) {
	cases := []string{
		``,
		`{`,
		`{"options":{`,
		`{"messages":not-array}`,
		`{"options":{"temperature":"hot"}}`,
	}
	for _, in := range cases {
		t.Run(in, func(t *testing.T) {
			var req ChatRequest
			if err := json.Unmarshal([]byte(in), &req); err == nil {
				t.Fatalf("Unmarshal(%q) returned nil error", in)
			}
		})
	}
}

// The tests below call UnmarshalJSON (and, where unexported, the
// underlying parse* walkers) directly rather than through
// encoding/json.Unmarshal. encoding/json.Unmarshal runs a whole-
// document checkValid syntax scan before ever invoking a custom
// UnmarshalJSON method — most single-byte-level malformations (bare
// keys, missing colons, unterminated objects) are generic JSON syntax
// errors that checkValid rejects before our hand-rolled walker gets a
// chance to run, so routing them through json.Unmarshal would never
// exercise the walker's own defensive branches. Calling UnmarshalJSON
// directly (a normal, exported method) reaches those branches exactly
// as a streaming decoder (json.Decoder, or a hand-rolled HTTP body
// reader) would, without checkValid's whole-document pre-validation
// in the way. Every error path in this file's hand-rolled walker
// resolves to the single jsonenc.ErrInvalidJSON sentinel, so failure
// cases assert identity against it rather than a substring.

// TestUnmarshal_ChatRequest_Bad drives every malformed-shape branch
// in ChatRequest.UnmarshalJSON and its unmarshalField dispatch that
// direct DirectShapes/InvalidShapes coverage does not reach: a
// non-object top level, a malformed key, structural EOF/trailing-
// garbage at every loop boundary, and a wrong-typed value for each
// field.
func TestUnmarshal_ChatRequest_Bad(t *testing.T) {
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
		{"stream-wrong-type", `{"stream":"nope"}`},
		{"options-wrong-type", `{"options":"nope"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var req ChatRequest
			if err := req.UnmarshalJSON([]byte(tc.in)); err != jsonenc.ErrInvalidJSON {
				t.Fatalf("UnmarshalJSON(%q) error = %v, want jsonenc.ErrInvalidJSON", tc.in, err)
			}
		})
	}
}

// TestUnmarshal_ChatRequest_Good covers the empty-object fast path
// and the null-keeps-zero-value handling on the "stream" field —
// both success paths the existing DirectShapes table never exercised.
func TestUnmarshal_ChatRequest_Good(t *testing.T) {
	var empty ChatRequest
	if err := empty.UnmarshalJSON([]byte(`{}`)); err != nil || !reflect.DeepEqual(empty, ChatRequest{}) {
		t.Fatalf("UnmarshalJSON(%q) = %+v, err = %v", `{}`, empty, err)
	}

	var streamed ChatRequest
	in := `{"model":"m","stream":null}`
	if err := streamed.UnmarshalJSON([]byte(in)); err != nil {
		t.Fatalf("UnmarshalJSON(%q) error = %v", in, err)
	}
	if streamed.Stream {
		t.Fatalf("UnmarshalJSON(%q) left Stream = true, want false (null keeps zero value)", in)
	}
}

// TestUnmarshal_GenerateRequest_Bad mirrors TestUnmarshal_ChatRequest_Bad
// for the /api/generate request shape.
func TestUnmarshal_GenerateRequest_Bad(t *testing.T) {
	cases := []struct{ name, in string }{
		{"not-an-object", `[]`},
		{"non-string-key", `{1:"x"}`},
		{"unterminated-key", `{"model`},
		{"missing-colon", `{"model" "x"}`},
		{"eof-after-value", `{"model":"x"`},
		{"trailing-garbage", `{"model":"x"]`},
		{"model-wrong-type", `{"model":42}`},
		{"prompt-wrong-type", `{"model":"m","prompt":42}`},
		{"stream-wrong-type", `{"model":"m","stream":"nope"}`},
		{"options-wrong-type", `{"model":"m","options":"nope"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var req GenerateRequest
			if err := req.UnmarshalJSON([]byte(tc.in)); err != jsonenc.ErrInvalidJSON {
				t.Fatalf("UnmarshalJSON(%q) error = %v, want jsonenc.ErrInvalidJSON", tc.in, err)
			}
		})
	}
}

// TestUnmarshal_GenerateRequest_Good covers the empty-object fast
// path, "stream":null, and the unknown-field skip — GenerateRequest's
// DirectShapes table only ever exercised the all-fields-present path.
func TestUnmarshal_GenerateRequest_Good(t *testing.T) {
	var empty GenerateRequest
	if err := empty.UnmarshalJSON([]byte(`{}`)); err != nil || empty != (GenerateRequest{}) {
		t.Fatalf("UnmarshalJSON(%q) = %+v, err = %v", `{}`, empty, err)
	}

	var streamed GenerateRequest
	in := `{"model":"m","stream":null}`
	if err := streamed.UnmarshalJSON([]byte(in)); err != nil || streamed.Stream {
		t.Fatalf("UnmarshalJSON(%q) = %+v, err = %v, want Stream=false", in, streamed, err)
	}

	var withUnknown GenerateRequest
	in = `{"model":"m","prompt":"p","future":42}`
	if err := withUnknown.UnmarshalJSON([]byte(in)); err != nil {
		t.Fatalf("UnmarshalJSON(%q) error = %v", in, err)
	}
	if withUnknown.Model != "m" || withUnknown.Prompt != "p" {
		t.Fatalf("UnmarshalJSON(%q) = %+v, lost known fields", in, withUnknown)
	}
}

// TestUnmarshal_ChatResponse_Bad covers ChatResponse.UnmarshalJSON —
// previously untested for any malformed input (its only prior
// coverage was TestUnmarshalChatResponse_DirectShapes' single
// all-fields-valid case).
func TestUnmarshal_ChatResponse_Bad(t *testing.T) {
	cases := []struct{ name, in string }{
		{"not-an-object", `42`},
		{"non-string-key", `{1:"x"}`},
		{"unterminated-key", `{"model`},
		{"missing-colon", `{"model" "x"}`},
		{"eof-after-value", `{"model":"x"`},
		{"trailing-garbage", `{"model":"x"]`},
		{"model-wrong-type", `{"model":42}`},
		{"message-wrong-type", `{"message":42}`},
		{"done-wrong-type", `{"done":"nope"}`},
		{"prompt_eval_count-wrong-type", `{"prompt_eval_count":"x"}`},
		{"eval_count-wrong-type", `{"eval_count":"x"}`},
		{"total_duration-wrong-type", `{"total_duration":"x"}`},
		{"load_duration-wrong-type", `{"load_duration":"x"}`},
		{"prompt_eval_duration-wrong-type", `{"prompt_eval_duration":"x"}`},
		{"eval_duration-wrong-type", `{"eval_duration":"x"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var resp ChatResponse
			if err := resp.UnmarshalJSON([]byte(tc.in)); err != jsonenc.ErrInvalidJSON {
				t.Fatalf("UnmarshalJSON(%q) error = %v, want jsonenc.ErrInvalidJSON", tc.in, err)
			}
		})
	}
}

// TestUnmarshal_ChatResponse_Good covers the empty-object fast path,
// "done":null, and the unknown-field skip.
func TestUnmarshal_ChatResponse_Good(t *testing.T) {
	var empty ChatResponse
	if err := empty.UnmarshalJSON([]byte(`{}`)); err != nil || empty != (ChatResponse{}) {
		t.Fatalf("UnmarshalJSON(%q) = %+v, err = %v", `{}`, empty, err)
	}

	var doneNull ChatResponse
	in := `{"model":"m","done":null}`
	if err := doneNull.UnmarshalJSON([]byte(in)); err != nil || doneNull.Done {
		t.Fatalf("UnmarshalJSON(%q) = %+v, err = %v, want Done=false", in, doneNull, err)
	}

	var withUnknown ChatResponse
	in = `{"model":"m","extra":"ignored","done":true}`
	if err := withUnknown.UnmarshalJSON([]byte(in)); err != nil {
		t.Fatalf("UnmarshalJSON(%q) error = %v", in, err)
	}
	if withUnknown.Model != "m" || !withUnknown.Done {
		t.Fatalf("UnmarshalJSON(%q) = %+v, lost known fields", in, withUnknown)
	}
}

// TestUnmarshal_GenerateResponse_Bad mirrors TestUnmarshal_ChatResponse_Bad
// for the /api/generate response shape.
func TestUnmarshal_GenerateResponse_Bad(t *testing.T) {
	cases := []struct{ name, in string }{
		{"not-an-object", `42`},
		{"non-string-key", `{1:"x"}`},
		{"unterminated-key", `{"model`},
		{"missing-colon", `{"model" "x"}`},
		{"eof-after-value", `{"model":"x"`},
		{"trailing-garbage", `{"model":"x"]`},
		{"model-wrong-type", `{"model":42}`},
		{"response-wrong-type", `{"response":42}`},
		{"done-wrong-type", `{"done":"nope"}`},
		{"prompt_eval_count-wrong-type", `{"prompt_eval_count":"x"}`},
		{"eval_count-wrong-type", `{"eval_count":"x"}`},
		{"total_duration-wrong-type", `{"total_duration":"x"}`},
		{"load_duration-wrong-type", `{"load_duration":"x"}`},
		{"prompt_eval_duration-wrong-type", `{"prompt_eval_duration":"x"}`},
		{"eval_duration-wrong-type", `{"eval_duration":"x"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var resp GenerateResponse
			if err := resp.UnmarshalJSON([]byte(tc.in)); err != jsonenc.ErrInvalidJSON {
				t.Fatalf("UnmarshalJSON(%q) error = %v, want jsonenc.ErrInvalidJSON", tc.in, err)
			}
		})
	}
}

// TestUnmarshal_GenerateResponse_Good covers the empty-object fast
// path, "done":null, and the unknown-field skip.
func TestUnmarshal_GenerateResponse_Good(t *testing.T) {
	var empty GenerateResponse
	if err := empty.UnmarshalJSON([]byte(`{}`)); err != nil || empty != (GenerateResponse{}) {
		t.Fatalf("UnmarshalJSON(%q) = %+v, err = %v", `{}`, empty, err)
	}

	var doneNull GenerateResponse
	in := `{"model":"m","done":null}`
	if err := doneNull.UnmarshalJSON([]byte(in)); err != nil || doneNull.Done {
		t.Fatalf("UnmarshalJSON(%q) = %+v, err = %v, want Done=false", in, doneNull, err)
	}

	var withUnknown GenerateResponse
	in = `{"model":"m","extra":true,"done":true}`
	if err := withUnknown.UnmarshalJSON([]byte(in)); err != nil {
		t.Fatalf("UnmarshalJSON(%q) error = %v", in, err)
	}
	if withUnknown.Model != "m" || !withUnknown.Done {
		t.Fatalf("UnmarshalJSON(%q) = %+v, lost known fields", in, withUnknown)
	}
}

// TestUnmarshal_TagsResponse_Bad covers TagsResponse.UnmarshalJSON's
// structural branches plus the "models" field's error propagation
// from parseModelTagArray and the default-arm's own malformed-value
// path (distinct from the field dispatch above because TagsResponse
// inlines its single-field switch rather than using a separate
// unmarshalField method).
func TestUnmarshal_TagsResponse_Bad(t *testing.T) {
	cases := []struct{ name, in string }{
		{"not-an-object", `42`},
		{"non-string-key", `{1:"x"}`},
		{"unterminated-key", `{"models`},
		{"missing-colon", `{"models" []}`},
		{"eof-after-value", `{"models":[]`},
		{"trailing-garbage", `{"models":[]oops}`},
		{"models-wrong-type", `{"models":42}`},
		{"unknown-field-malformed-value", `{"extra":bogus}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var resp TagsResponse
			if err := resp.UnmarshalJSON([]byte(tc.in)); err != jsonenc.ErrInvalidJSON {
				t.Fatalf("UnmarshalJSON(%q) error = %v, want jsonenc.ErrInvalidJSON", tc.in, err)
			}
		})
	}
}

// TestUnmarshal_TagsResponse_Good covers the empty-object fast path
// and an unknown field ahead of "models" — the leading unknown field
// forces the comma-continuation branch (every existing case has a
// single top-level field, so the loop never previously looped).
func TestUnmarshal_TagsResponse_Good(t *testing.T) {
	var empty TagsResponse
	if err := empty.UnmarshalJSON([]byte(`{}`)); err != nil || !reflect.DeepEqual(empty, TagsResponse{}) {
		t.Fatalf("UnmarshalJSON(%q) = %+v, err = %v", `{}`, empty, err)
	}

	var withUnknown TagsResponse
	in := `{"extra":"ignored","models":[{"name":"a"}]}`
	if err := withUnknown.UnmarshalJSON([]byte(in)); err != nil {
		t.Fatalf("UnmarshalJSON(%q) error = %v", in, err)
	}
	want := TagsResponse{Models: []ModelTag{{Name: "a"}}}
	if !reflect.DeepEqual(withUnknown, want) {
		t.Fatalf("UnmarshalJSON(%q) = %+v, want %+v", in, withUnknown, want)
	}
}

// TestUnmarshal_ParseMessageArray_Bad drives parseMessageArray's own
// malformed-shape branches directly — it is an unexported helper
// shared by ChatRequest and ChatResponse, so its statements need their
// own targeted coverage independent of either caller.
func TestUnmarshal_ParseMessageArray_Bad(t *testing.T) {
	cases := []struct{ name, in string }{
		{"not-an-array", `{}`},
		{"element-error", `[{"role":42}]`},
		{"eof-after-element", `[{"role":"user"}`},
		{"trailing-garbage", `[{"role":"user"}oops]`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, err := parseMessageArray([]byte(tc.in), 0); err != jsonenc.ErrInvalidJSON {
				t.Fatalf("parseMessageArray(%q) error = %v, want jsonenc.ErrInvalidJSON", tc.in, err)
			}
		})
	}
}

// TestUnmarshal_ParseMessageArray_Good covers the null literal
// (nil slice, no error) and a 2-element array — the only way to
// exercise the comma-continuation branch, since every prior
// DirectShapes case used a 0- or 1-element messages array.
func TestUnmarshal_ParseMessageArray_Good(t *testing.T) {
	msgs, next, err := parseMessageArray([]byte(`null`), 0)
	if err != nil || msgs != nil || next != 4 {
		t.Fatalf("parseMessageArray(null) = %v, %d, %v", msgs, next, err)
	}

	in := `[{"role":"user","content":"a"},{"role":"assistant","content":"b"}]`
	msgs, next, err = parseMessageArray([]byte(in), 0)
	if err != nil {
		t.Fatalf("parseMessageArray(%q) error = %v", in, err)
	}
	want := []Message{{Role: "user", Content: "a"}, {Role: "assistant", Content: "b"}}
	if !reflect.DeepEqual(msgs, want) || next != len(in) {
		t.Fatalf("parseMessageArray(%q) = %+v, %d; want %+v, %d", in, msgs, next, want, len(in))
	}
}

// TestUnmarshal_ParseMessage_Bad drives parseMessage's own malformed-
// shape branches directly.
func TestUnmarshal_ParseMessage_Bad(t *testing.T) {
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
			if _, _, err := parseMessage([]byte(tc.in), 0); err != jsonenc.ErrInvalidJSON {
				t.Fatalf("parseMessage(%q) error = %v, want jsonenc.ErrInvalidJSON", tc.in, err)
			}
		})
	}
}

// TestUnmarshal_ParseMessage_Good covers the empty-object fast path
// and an unknown field skipped ahead of the known role/content pair.
func TestUnmarshal_ParseMessage_Good(t *testing.T) {
	msg, next, err := parseMessage([]byte(`{}`), 0)
	if err != nil || msg != (Message{}) || next != 2 {
		t.Fatalf("parseMessage(%q) = %+v, %d, %v", `{}`, msg, next, err)
	}

	in := `{"extra":"ignored","role":"user","content":"hi"}`
	msg, _, err = parseMessage([]byte(in), 0)
	if err != nil {
		t.Fatalf("parseMessage(%q) error = %v", in, err)
	}
	if want := (Message{Role: "user", Content: "hi"}); msg != want {
		t.Fatalf("parseMessage(%q) = %+v, want %+v", in, msg, want)
	}
}

// TestUnmarshal_ParseOptions_Bad drives parseOptions's own malformed-
// shape branches directly — top_k/top_p/min_p/num_predict each had no
// error-path coverage prior to this (only "temperature" was pinned,
// via TestUnmarshalChatRequest_InvalidShapes).
func TestUnmarshal_ParseOptions_Bad(t *testing.T) {
	cases := []struct{ name, in string }{
		{"not-an-object", `"str"`},
		{"non-string-key", `{1:"x"}`},
		{"unterminated-key", `{"top_k`},
		{"missing-colon", `{"top_k" 1}`},
		{"top_k-wrong-type", `{"top_k":"x"}`},
		{"top_p-wrong-type", `{"top_p":"x"}`},
		{"min_p-wrong-type", `{"min_p":"x"}`},
		{"num_predict-wrong-type", `{"num_predict":"x"}`},
		{"unknown-field-malformed-value", `{"extra":bogus}`},
		{"eof-after-value", `{"top_k":1`},
		{"trailing-garbage", `{"top_k":1]`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, err := parseOptions([]byte(tc.in), 0); err != jsonenc.ErrInvalidJSON {
				t.Fatalf("parseOptions(%q) error = %v, want jsonenc.ErrInvalidJSON", tc.in, err)
			}
		})
	}
}

// TestUnmarshal_ParseOptions_Good covers the empty-object fast path.
func TestUnmarshal_ParseOptions_Good(t *testing.T) {
	opts, next, err := parseOptions([]byte(`{}`), 0)
	if err != nil || opts != (Options{}) || next != 2 {
		t.Fatalf("parseOptions(%q) = %+v, %d, %v", `{}`, opts, next, err)
	}
}

// TestUnmarshal_ParseModelTagArray_Bad drives parseModelTagArray's
// own malformed-shape branches directly.
func TestUnmarshal_ParseModelTagArray_Bad(t *testing.T) {
	cases := []struct{ name, in string }{
		{"not-an-array", `{}`},
		{"element-error", `[{"size":"x"}]`},
		{"eof-after-element", `[{"name":"a"}`},
		{"trailing-garbage", `[{"name":"a"}oops]`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, err := parseModelTagArray([]byte(tc.in), 0); err != jsonenc.ErrInvalidJSON {
				t.Fatalf("parseModelTagArray(%q) error = %v, want jsonenc.ErrInvalidJSON", tc.in, err)
			}
		})
	}
}

// TestUnmarshal_ParseModelTagArray_Good covers the null literal (nil
// slice, no error) — the /api/tags "no models installed" wire shape
// some ollama-compatible servers emit instead of an empty array.
func TestUnmarshal_ParseModelTagArray_Good(t *testing.T) {
	tags, next, err := parseModelTagArray([]byte(`null`), 0)
	if err != nil || tags != nil || next != 4 {
		t.Fatalf("parseModelTagArray(null) = %v, %d, %v", tags, next, err)
	}
}

// TestUnmarshal_ParseModelTag_Bad drives parseModelTag's own
// malformed-shape branches directly.
func TestUnmarshal_ParseModelTag_Bad(t *testing.T) {
	cases := []struct{ name, in string }{
		{"not-an-object", `"str"`},
		{"non-string-key", `{1:"x"}`},
		{"unterminated-key", `{"name`},
		{"missing-colon", `{"name" "x"}`},
		{"name-wrong-type", `{"name":42}`},
		{"model-wrong-type", `{"model":42}`},
		{"modified_at-wrong-type", `{"modified_at":42}`},
		{"size-wrong-type", `{"size":"x"}`},
		{"unknown-field-malformed-value", `{"extra":bogus}`},
		{"eof-after-value", `{"name":"a"`},
		{"trailing-garbage", `{"name":"a"]`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, err := parseModelTag([]byte(tc.in), 0); err != jsonenc.ErrInvalidJSON {
				t.Fatalf("parseModelTag(%q) error = %v, want jsonenc.ErrInvalidJSON", tc.in, err)
			}
		})
	}
}

// TestUnmarshal_ParseModelTag_Good covers the empty-object fast path
// and an unknown field skipped ahead of the known name field.
func TestUnmarshal_ParseModelTag_Good(t *testing.T) {
	tag, next, err := parseModelTag([]byte(`{}`), 0)
	if err != nil || tag != (ModelTag{}) || next != 2 {
		t.Fatalf("parseModelTag(%q) = %+v, %d, %v", `{}`, tag, next, err)
	}

	in := `{"extra":"ignored","name":"qwen3:latest"}`
	tag, _, err = parseModelTag([]byte(in), 0)
	if err != nil {
		t.Fatalf("parseModelTag(%q) error = %v", in, err)
	}
	if want := (ModelTag{Name: "qwen3:latest"}); tag != want {
		t.Fatalf("parseModelTag(%q) = %+v, want %+v", in, tag, want)
	}
}
