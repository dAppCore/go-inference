// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"encoding/json"
	"reflect"
	"testing"
)

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
