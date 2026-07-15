// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"encoding/json"
	"reflect"
	"testing"

	"dappco.re/go/inference/jsonenc"
)

// EmbeddingInput.UnmarshalJSON is exercised directly in
// services_test.go (TestServices_EmbeddingInput_UnmarshalJSON_*) —
// EmbeddingInput is declared in services.go, so its dedicated tests
// live in that file's test scope rather than here.

func TestUnmarshalEmbeddingRequest_DirectShapes(t *testing.T) {
	dim := 1024
	cases := []struct {
		name string
		in   string
		want EmbeddingRequest
	}{
		{
			name: "single-string-input",
			in:   `{"model":"text-embedding","input":"hello"}`,
			want: EmbeddingRequest{
				Model: "text-embedding",
				Input: EmbeddingInput{"hello"},
			},
		},
		{
			name: "array-input-and-options",
			in:   `{"model":"text-embedding","input":["a","b"],"encoding_format":"float","dimensions":1024,"normalize":true,"user":"u1"}`,
			want: EmbeddingRequest{
				Model:          "text-embedding",
				Input:          EmbeddingInput{"a", "b"},
				EncodingFormat: "float",
				Dimensions:     &dim,
				Normalize:      true,
				User:           "u1",
			},
		},
		{
			name: "dimensions-null",
			in:   `{"model":"text-embedding","input":"hello","dimensions":null}`,
			want: EmbeddingRequest{
				Model: "text-embedding",
				Input: EmbeddingInput{"hello"},
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var got EmbeddingRequest
			if err := json.Unmarshal([]byte(tc.in), &got); err != nil {
				t.Fatalf("Unmarshal error = %v", err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got:  %+v\nwant: %+v", got, tc.want)
			}
		})
	}
}

func TestUnmarshalRerankRequest_DirectShapes(t *testing.T) {
	in := `{"model":"rerank","query":"q","documents":["a","b","c"],"top_n":2}`
	want := RerankRequest{
		Model:     "rerank",
		Query:     "q",
		Documents: []string{"a", "b", "c"},
		TopN:      2,
	}
	var got RerankRequest
	if err := json.Unmarshal([]byte(in), &got); err != nil {
		t.Fatalf("Unmarshal error = %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got:  %+v\nwant: %+v", got, want)
	}
}

func TestUnmarshalCacheWarmRequest_DirectShapes(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want CacheWarmRequest
	}{
		{
			name: "prompt-mode",
			in:   `{"model":"m","prompt":"hi","mode":"warm","labels":{"k":"v"}}`,
			want: CacheWarmRequest{
				Model:  "m",
				Prompt: "hi",
				Mode:   "warm",
				Labels: map[string]string{"k": "v"},
			},
		},
		{
			name: "tokens-mode",
			in:   `{"model":"m","tokens":[1,2,3,4,5]}`,
			want: CacheWarmRequest{
				Model:  "m",
				Tokens: []int32{1, 2, 3, 4, 5},
			},
		},
		{
			name: "labels-null",
			in:   `{"model":"m","prompt":"hi","labels":null}`,
			want: CacheWarmRequest{
				Model:  "m",
				Prompt: "hi",
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var got CacheWarmRequest
			if err := json.Unmarshal([]byte(tc.in), &got); err != nil {
				t.Fatalf("Unmarshal error = %v", err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got:  %+v\nwant: %+v", got, tc.want)
			}
		})
	}
}

func TestUnmarshalCacheClearRequest_DirectShapes(t *testing.T) {
	in := `{"model":"m","labels":{"env":"prod","tier":"hot"}}`
	want := CacheClearRequest{
		Model:  "m",
		Labels: map[string]string{"env": "prod", "tier": "hot"},
	}
	var got CacheClearRequest
	if err := json.Unmarshal([]byte(in), &got); err != nil {
		t.Fatalf("Unmarshal error = %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got:  %+v\nwant: %+v", got, want)
	}
}

func TestUnmarshalCancelRequest_DirectShapes(t *testing.T) {
	in := `{"model":"m","id":"req_123"}`
	want := CancelRequest{Model: "m", ID: "req_123"}
	var got CancelRequest
	if err := json.Unmarshal([]byte(in), &got); err != nil {
		t.Fatalf("Unmarshal error = %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got:  %+v\nwant: %+v", got, want)
	}
}

// The tests below call UnmarshalJSON (and the unexported parse*
// helpers) directly rather than through encoding/json.Unmarshal, for
// the same reason as unmarshal_test.go: encoding/json's whole-
// document checkValid pre-scan rejects most byte-level malformations
// before our hand-rolled walker ever runs. Every error resolves to
// jsonenc.ErrInvalidJSON, so failures assert identity against it.

// TestServicesUnmarshal_EmbeddingRequest_UnmarshalJSON_Bad drives every
// malformed-shape branch in EmbeddingRequest.UnmarshalJSON/unmarshalField.
func TestServicesUnmarshal_EmbeddingRequest_UnmarshalJSON_Bad(t *testing.T) {
	cases := []struct{ name, in string }{
		{"not-an-object", `42`},
		{"non-string-key", `{1:"x"}`},
		{"unterminated-key", `{"model`},
		{"missing-colon", `{"model" "x"}`},
		{"eof-after-value", `{"model":"x"`},
		{"trailing-garbage", `{"model":"x"]`},
		{"model-wrong-type", `{"model":42}`},
		{"input-unskippable", `{"input":bogus}`},
		{"input-wrong-shape", `{"input":42}`},
		{"encoding_format-wrong-type", `{"encoding_format":42}`},
		{"dimensions-wrong-type", `{"dimensions":"x"}`},
		{"user-wrong-type", `{"user":42}`},
		{"normalize-wrong-type", `{"normalize":"nope"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var req EmbeddingRequest
			if err := req.UnmarshalJSON([]byte(tc.in)); err != jsonenc.ErrInvalidJSON {
				t.Fatalf("UnmarshalJSON(%q) error = %v, want jsonenc.ErrInvalidJSON", tc.in, err)
			}
		})
	}
}

// TestServicesUnmarshal_EmbeddingRequest_UnmarshalJSON_Good covers the
// empty-object fast path, dimensions/normalize null handling, and an
// unknown field skipped ahead of a known one.
func TestServicesUnmarshal_EmbeddingRequest_UnmarshalJSON_Good(t *testing.T) {
	var empty EmbeddingRequest
	if err := empty.UnmarshalJSON([]byte(`{}`)); err != nil || !reflect.DeepEqual(empty, EmbeddingRequest{}) {
		t.Fatalf("UnmarshalJSON(%q) = %+v, err = %v", `{}`, empty, err)
	}

	var nulled EmbeddingRequest
	in := `{"model":"m","input":"x","dimensions":null,"normalize":null}`
	if err := nulled.UnmarshalJSON([]byte(in)); err != nil {
		t.Fatalf("UnmarshalJSON(%q) error = %v", in, err)
	}
	want := EmbeddingRequest{Model: "m", Input: EmbeddingInput{"x"}}
	if !reflect.DeepEqual(nulled, want) {
		t.Fatalf("UnmarshalJSON(%q) = %+v, want %+v", in, nulled, want)
	}

	var withUnknown EmbeddingRequest
	in = `{"future":42,"model":"m","input":"x"}`
	if err := withUnknown.UnmarshalJSON([]byte(in)); err != nil {
		t.Fatalf("UnmarshalJSON(%q) error = %v", in, err)
	}
	if !reflect.DeepEqual(withUnknown, want) {
		t.Fatalf("UnmarshalJSON(%q) = %+v, want %+v", in, withUnknown, want)
	}
}

// TestServicesUnmarshal_EmbeddingRequest_UnmarshalJSON_Ugly covers a
// duplicate key — the hand-rolled walker does not de-duplicate, so
// the last occurrence wins.
func TestServicesUnmarshal_EmbeddingRequest_UnmarshalJSON_Ugly(t *testing.T) {
	var got EmbeddingRequest
	in := `{"model":"first","model":"second"}`
	if err := got.UnmarshalJSON([]byte(in)); err != nil {
		t.Fatalf("UnmarshalJSON(%q) error = %v", in, err)
	}
	if got.Model != "second" {
		t.Fatalf("UnmarshalJSON(duplicate key) Model = %q, want the last occurrence to win", got.Model)
	}
}

// TestServicesUnmarshal_RerankRequest_UnmarshalJSON_Bad drives every
// malformed-shape branch in RerankRequest.UnmarshalJSON/unmarshalField.
func TestServicesUnmarshal_RerankRequest_UnmarshalJSON_Bad(t *testing.T) {
	cases := []struct{ name, in string }{
		{"not-an-object", `42`},
		{"non-string-key", `{1:"x"}`},
		{"unterminated-key", `{"model`},
		{"missing-colon", `{"model" "x"}`},
		{"eof-after-value", `{"model":"x"`},
		{"trailing-garbage", `{"model":"x"]`},
		{"model-wrong-type", `{"model":42}`},
		{"query-wrong-type", `{"query":42}`},
		{"documents-unskippable", `{"documents":bogus}`},
		{"documents-wrong-shape", `{"documents":42}`},
		{"top_n-wrong-type", `{"top_n":"x"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var req RerankRequest
			if err := req.UnmarshalJSON([]byte(tc.in)); err != jsonenc.ErrInvalidJSON {
				t.Fatalf("UnmarshalJSON(%q) error = %v, want jsonenc.ErrInvalidJSON", tc.in, err)
			}
		})
	}
}

// TestServicesUnmarshal_RerankRequest_UnmarshalJSON_Good covers the
// empty-object fast path and an unknown field skipped ahead of a
// known one.
func TestServicesUnmarshal_RerankRequest_UnmarshalJSON_Good(t *testing.T) {
	var empty RerankRequest
	if err := empty.UnmarshalJSON([]byte(`{}`)); err != nil || !reflect.DeepEqual(empty, RerankRequest{}) {
		t.Fatalf("UnmarshalJSON(%q) = %+v, err = %v", `{}`, empty, err)
	}

	var withUnknown RerankRequest
	in := `{"future":42,"model":"m","query":"q","documents":["a"]}`
	if err := withUnknown.UnmarshalJSON([]byte(in)); err != nil {
		t.Fatalf("UnmarshalJSON(%q) error = %v", in, err)
	}
	want := RerankRequest{Model: "m", Query: "q", Documents: []string{"a"}}
	if !reflect.DeepEqual(withUnknown, want) {
		t.Fatalf("UnmarshalJSON(%q) = %+v, want %+v", in, withUnknown, want)
	}
}

// TestServicesUnmarshal_RerankRequest_UnmarshalJSON_Ugly covers a
// duplicate key — the last occurrence wins.
func TestServicesUnmarshal_RerankRequest_UnmarshalJSON_Ugly(t *testing.T) {
	var got RerankRequest
	in := `{"model":"first","model":"second"}`
	if err := got.UnmarshalJSON([]byte(in)); err != nil {
		t.Fatalf("UnmarshalJSON(%q) error = %v", in, err)
	}
	if got.Model != "second" {
		t.Fatalf("UnmarshalJSON(duplicate key) Model = %q, want the last occurrence to win", got.Model)
	}
}

// TestServicesUnmarshal_CancelRequest_UnmarshalJSON_Bad drives every
// malformed-shape branch in CancelRequest.UnmarshalJSON, including its
// own default-arm malformed-value path (CancelRequest inlines its
// field switch rather than using a separate unmarshalField method).
func TestServicesUnmarshal_CancelRequest_UnmarshalJSON_Bad(t *testing.T) {
	cases := []struct{ name, in string }{
		{"not-an-object", `42`},
		{"non-string-key", `{1:"x"}`},
		{"unterminated-key", `{"model`},
		{"missing-colon", `{"model" "x"}`},
		{"eof-after-value", `{"model":"x"`},
		{"trailing-garbage", `{"model":"x"]`},
		{"model-wrong-type", `{"model":42}`},
		{"id-wrong-type", `{"id":42}`},
		{"unknown-field-malformed-value", `{"extra":bogus}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var req CancelRequest
			if err := req.UnmarshalJSON([]byte(tc.in)); err != jsonenc.ErrInvalidJSON {
				t.Fatalf("UnmarshalJSON(%q) error = %v, want jsonenc.ErrInvalidJSON", tc.in, err)
			}
		})
	}
}

// TestServicesUnmarshal_CancelRequest_UnmarshalJSON_Good covers the
// empty-object fast path and an unknown field skipped ahead of
// model/id.
func TestServicesUnmarshal_CancelRequest_UnmarshalJSON_Good(t *testing.T) {
	var empty CancelRequest
	if err := empty.UnmarshalJSON([]byte(`{}`)); err != nil || empty != (CancelRequest{}) {
		t.Fatalf("UnmarshalJSON(%q) = %+v, err = %v", `{}`, empty, err)
	}

	var withUnknown CancelRequest
	in := `{"extra":"ignored","model":"m","id":"req_1"}`
	if err := withUnknown.UnmarshalJSON([]byte(in)); err != nil {
		t.Fatalf("UnmarshalJSON(%q) error = %v", in, err)
	}
	if want := (CancelRequest{Model: "m", ID: "req_1"}); withUnknown != want {
		t.Fatalf("UnmarshalJSON(%q) = %+v, want %+v", in, withUnknown, want)
	}
}

// TestServicesUnmarshal_CancelRequest_UnmarshalJSON_Ugly covers a
// duplicate key — the last occurrence wins.
func TestServicesUnmarshal_CancelRequest_UnmarshalJSON_Ugly(t *testing.T) {
	var got CancelRequest
	in := `{"id":"first","id":"second"}`
	if err := got.UnmarshalJSON([]byte(in)); err != nil {
		t.Fatalf("UnmarshalJSON(%q) error = %v", in, err)
	}
	if got.ID != "second" {
		t.Fatalf("UnmarshalJSON(duplicate key) ID = %q, want the last occurrence to win", got.ID)
	}
}

// TestServicesUnmarshal_CacheClearRequest_UnmarshalJSON_Bad drives
// every malformed-shape branch in CacheClearRequest.UnmarshalJSON.
func TestServicesUnmarshal_CacheClearRequest_UnmarshalJSON_Bad(t *testing.T) {
	cases := []struct{ name, in string }{
		{"not-an-object", `42`},
		{"non-string-key", `{1:"x"}`},
		{"unterminated-key", `{"model`},
		{"missing-colon", `{"model" "x"}`},
		{"eof-after-value", `{"model":"x"`},
		{"trailing-garbage", `{"model":"x"]`},
		{"model-wrong-type", `{"model":42}`},
		{"labels-wrong-type", `{"labels":42}`},
		{"unknown-field-malformed-value", `{"extra":bogus}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var req CacheClearRequest
			if err := req.UnmarshalJSON([]byte(tc.in)); err != jsonenc.ErrInvalidJSON {
				t.Fatalf("UnmarshalJSON(%q) error = %v, want jsonenc.ErrInvalidJSON", tc.in, err)
			}
		})
	}
}

// TestServicesUnmarshal_CacheClearRequest_UnmarshalJSON_Good covers
// the empty-object fast path and an unknown field skipped ahead of
// model/labels.
func TestServicesUnmarshal_CacheClearRequest_UnmarshalJSON_Good(t *testing.T) {
	var empty CacheClearRequest
	if err := empty.UnmarshalJSON([]byte(`{}`)); err != nil || !reflect.DeepEqual(empty, CacheClearRequest{}) {
		t.Fatalf("UnmarshalJSON(%q) = %+v, err = %v", `{}`, empty, err)
	}

	var withUnknown CacheClearRequest
	in := `{"extra":"ignored","model":"m","labels":{"k":"v"}}`
	if err := withUnknown.UnmarshalJSON([]byte(in)); err != nil {
		t.Fatalf("UnmarshalJSON(%q) error = %v", in, err)
	}
	want := CacheClearRequest{Model: "m", Labels: map[string]string{"k": "v"}}
	if !reflect.DeepEqual(withUnknown, want) {
		t.Fatalf("UnmarshalJSON(%q) = %+v, want %+v", in, withUnknown, want)
	}
}

// TestServicesUnmarshal_CacheClearRequest_UnmarshalJSON_Ugly covers a
// duplicate key — the last occurrence wins.
func TestServicesUnmarshal_CacheClearRequest_UnmarshalJSON_Ugly(t *testing.T) {
	var got CacheClearRequest
	in := `{"model":"first","model":"second"}`
	if err := got.UnmarshalJSON([]byte(in)); err != nil {
		t.Fatalf("UnmarshalJSON(%q) error = %v", in, err)
	}
	if got.Model != "second" {
		t.Fatalf("UnmarshalJSON(duplicate key) Model = %q, want the last occurrence to win", got.Model)
	}
}

// TestServicesUnmarshal_CacheWarmRequest_UnmarshalJSON_Bad drives
// every malformed-shape branch in CacheWarmRequest.UnmarshalJSON.
func TestServicesUnmarshal_CacheWarmRequest_UnmarshalJSON_Bad(t *testing.T) {
	cases := []struct{ name, in string }{
		{"not-an-object", `42`},
		{"non-string-key", `{1:"x"}`},
		{"unterminated-key", `{"model`},
		{"missing-colon", `{"model" "x"}`},
		{"eof-after-value", `{"model":"x"`},
		{"trailing-garbage", `{"model":"x"]`},
		{"model-wrong-type", `{"model":42}`},
		{"prompt-wrong-type", `{"prompt":42}`},
		{"tokens-wrong-type", `{"tokens":42}`},
		{"tokens-element-wrong-type", `{"tokens":["x"]}`},
		{"mode-wrong-type", `{"mode":42}`},
		{"labels-wrong-type", `{"labels":42}`},
		{"unknown-field-malformed-value", `{"extra":bogus}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var req CacheWarmRequest
			if err := req.UnmarshalJSON([]byte(tc.in)); err != jsonenc.ErrInvalidJSON {
				t.Fatalf("UnmarshalJSON(%q) error = %v, want jsonenc.ErrInvalidJSON", tc.in, err)
			}
		})
	}
}

// TestServicesUnmarshal_CacheWarmRequest_UnmarshalJSON_Good covers the
// empty-object fast path and an unknown field skipped ahead of the
// known ones.
func TestServicesUnmarshal_CacheWarmRequest_UnmarshalJSON_Good(t *testing.T) {
	var empty CacheWarmRequest
	if err := empty.UnmarshalJSON([]byte(`{}`)); err != nil || !reflect.DeepEqual(empty, CacheWarmRequest{}) {
		t.Fatalf("UnmarshalJSON(%q) = %+v, err = %v", `{}`, empty, err)
	}

	var withUnknown CacheWarmRequest
	in := `{"extra":"ignored","model":"m","prompt":"hi"}`
	if err := withUnknown.UnmarshalJSON([]byte(in)); err != nil {
		t.Fatalf("UnmarshalJSON(%q) error = %v", in, err)
	}
	want := CacheWarmRequest{Model: "m", Prompt: "hi"}
	if !reflect.DeepEqual(withUnknown, want) {
		t.Fatalf("UnmarshalJSON(%q) = %+v, want %+v", in, withUnknown, want)
	}
}

// TestServicesUnmarshal_CacheWarmRequest_UnmarshalJSON_Ugly covers a
// duplicate key — the last occurrence wins.
func TestServicesUnmarshal_CacheWarmRequest_UnmarshalJSON_Ugly(t *testing.T) {
	var got CacheWarmRequest
	in := `{"model":"first","model":"second"}`
	if err := got.UnmarshalJSON([]byte(in)); err != nil {
		t.Fatalf("UnmarshalJSON(%q) error = %v", in, err)
	}
	if got.Model != "second" {
		t.Fatalf("UnmarshalJSON(duplicate key) Model = %q, want the last occurrence to win", got.Model)
	}
}

// TestUnmarshal_ParseStringMap_Bad drives parseStringMap's own
// malformed-shape branches directly.
func TestUnmarshal_ParseStringMap_Bad(t *testing.T) {
	cases := []struct{ name, in string }{
		{"not-an-object", `42`},
		{"non-string-key", `{1:"x"}`},
		{"unterminated-key", `{"k`},
		{"missing-colon", `{"k" "v"}`},
		{"value-wrong-type", `{"k":42}`},
		{"eof-after-value", `{"k":"v"`},
		{"trailing-garbage", `{"k":"v"]`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, err := parseStringMap([]byte(tc.in), 0); err != jsonenc.ErrInvalidJSON {
				t.Fatalf("parseStringMap(%q) error = %v, want jsonenc.ErrInvalidJSON", tc.in, err)
			}
		})
	}
}

// TestUnmarshal_ParseStringMap_Good covers the null literal, the
// empty-object fast path, and a 2-key object (comma-continuation) —
// none exercised by the CacheWarm/CacheClear DirectShapes tables,
// which only ever set a single label.
func TestUnmarshal_ParseStringMap_Good(t *testing.T) {
	m, next, err := parseStringMap([]byte(`null`), 0)
	if err != nil || m != nil || next != 4 {
		t.Fatalf("parseStringMap(null) = %v, %d, %v", m, next, err)
	}

	m, next, err = parseStringMap([]byte(`{}`), 0)
	if err != nil || m != nil || next != 2 {
		t.Fatalf("parseStringMap(%q) = %v, %d, %v", `{}`, m, next, err)
	}

	in := `{"a":"1","b":"2"}`
	m, next, err = parseStringMap([]byte(in), 0)
	if err != nil {
		t.Fatalf("parseStringMap(%q) error = %v", in, err)
	}
	want := map[string]string{"a": "1", "b": "2"}
	if !reflect.DeepEqual(m, want) || next != len(in) {
		t.Fatalf("parseStringMap(%q) = %v, %d; want %v, %d", in, m, next, want, len(in))
	}
}

// TestUnmarshal_ParseInt32Array_Bad drives parseInt32Array's own
// malformed-shape branches directly.
func TestUnmarshal_ParseInt32Array_Bad(t *testing.T) {
	cases := []struct{ name, in string }{
		{"not-an-array", `{}`},
		{"element-wrong-type", `["x"]`},
		{"eof-after-element", `[1`},
		{"trailing-garbage", `[1oops]`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, err := parseInt32Array([]byte(tc.in), 0); err != jsonenc.ErrInvalidJSON {
				t.Fatalf("parseInt32Array(%q) error = %v, want jsonenc.ErrInvalidJSON", tc.in, err)
			}
		})
	}
}

// TestUnmarshal_ParseInt32Array_Good covers the null literal and the
// empty-array fast path.
func TestUnmarshal_ParseInt32Array_Good(t *testing.T) {
	toks, next, err := parseInt32Array([]byte(`null`), 0)
	if err != nil || toks != nil || next != 4 {
		t.Fatalf("parseInt32Array(null) = %v, %d, %v", toks, next, err)
	}

	toks, next, err = parseInt32Array([]byte(`[]`), 0)
	if err != nil || toks != nil || next != 2 {
		t.Fatalf("parseInt32Array(%q) = %v, %d, %v", `[]`, toks, next, err)
	}
}
