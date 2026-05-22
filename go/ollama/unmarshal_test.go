// SPDX-Licence-Identifier: EUPL-1.2

package ollama

import (
	"encoding/json"
	"reflect"
	"testing"
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
			in:   `{"model":"qwen3","messages":[],"stream":true,"options":{"temperature":0.7,"top_k":64,"top_p":0.95,"num_predict":256}}`,
			want: ChatRequest{
				Model:   "qwen3",
				Stream:  true,
				Options: Options{Temperature: 0.7, TopK: 64, TopP: 0.95, NumPredict: 256},
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
	in := `{"model":"qwen3","prompt":"hi","stream":true,"options":{"temperature":0.7,"top_p":0.9,"num_predict":128}}`
	want := GenerateRequest{
		Model:   "qwen3",
		Prompt:  "hi",
		Stream:  true,
		Options: Options{Temperature: 0.7, TopP: 0.9, NumPredict: 128},
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
