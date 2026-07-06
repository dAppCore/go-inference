// SPDX-Licence-Identifier: EUPL-1.2

package ollama

import (
	"encoding/json"
	"strings"
	"testing"
)

// noReallocAppend runs append against a buffer pre-sized to size and
// fails the test if the encoded output overflowed that capacity —
// the property the *Size estimators exist to guarantee (single
// allocation per encode, matching the package doc's stated contract).
// Returns the encoded bytes for further assertions.
func noReallocAppend(t *testing.T, size int, append func(buf []byte) []byte) []byte {
	t.Helper()
	buf := make([]byte, 0, size)
	got := append(buf)
	if len(got) > cap(buf) {
		t.Fatalf("size estimate %d undercounts encoded length %d — buffer would reallocate", size, len(got))
	}
	return got
}

// TestChunkenc_chatResponseSize_Good exercises the fully-populated
// ChatResponse shape — every optional count/duration field set, the
// canonical final-chunk. Asserts the documented single-allocation
// contract: the estimate must never fall short of the actual encoded
// length (over-estimating is an accepted perf trade-off per the
// package doc; under-estimating would force AppendChatResponse to
// regrow the buffer on the hottest path in the package).
func TestChunkenc_chatResponseSize_Good(t *testing.T) {
	resp := ChatResponse{
		Model:              "qwen3",
		Message:            Message{Role: "assistant", Content: "concise answer"},
		Done:               true,
		PromptEvalCount:    200,
		EvalCount:          32,
		TotalDuration:      1_500_000_000,
		LoadDuration:       100_000_000,
		PromptEvalDuration: 200_000_000,
		EvalDuration:       1_200_000_000,
	}
	noReallocAppend(t, chatResponseSize(resp), func(buf []byte) []byte { return AppendChatResponse(buf, resp) })
}

// TestChunkenc_chatResponseSize_Bad exercises the zero-value floor —
// every optional field omitted, only the always-emitted model/
// message/done trio contribute bytes — and the streaming-intermediate
// shape (Done false, no metrics yet), the hottest call in production:
// one estimate per generated token.
func TestChunkenc_chatResponseSize_Bad(t *testing.T) {
	cases := []ChatResponse{
		{},
		{Model: "qwen3", Message: Message{Content: "tok"}},
	}
	for _, resp := range cases {
		noReallocAppend(t, chatResponseSize(resp), func(buf []byte) []byte { return AppendChatResponse(buf, resp) })
	}
}

// TestChunkenc_chatResponseSize_Ugly feeds escape-heavy content
// (quotes, newlines, tabs) that expands under JSON escaping beyond
// the raw len() the estimator counts — the one shape where the
// buffer legitimately may need to regrow. append handles that safely;
// what must hold regardless is that the wire output stays correct and
// round-trips through encoding/json.
func TestChunkenc_chatResponseSize_Ugly(t *testing.T) {
	resp := ChatResponse{
		Model:   "qwen3",
		Message: Message{Role: "assistant", Content: strings.Repeat("\"quoted\"\n\t", 64)},
		Done:    true,
	}
	buf := AppendChatResponse(make([]byte, 0, chatResponseSize(resp)), resp)
	var back ChatResponse
	if err := json.Unmarshal(buf, &back); err != nil {
		t.Fatalf("Unmarshal(%s): %v", buf, err)
	}
	if back != resp {
		t.Fatalf("round-trip mismatch:\n got = %+v\nwant = %+v", back, resp)
	}
}

// TestChunkenc_generateResponseSize_Good mirrors the ChatResponse pin
// for the /api/generate shape.
func TestChunkenc_generateResponseSize_Good(t *testing.T) {
	resp := GenerateResponse{
		Model: "qwen3", Response: "The summary is concise.", Done: true,
		PromptEvalCount:    200,
		EvalCount:          32,
		TotalDuration:      1_500_000_000,
		LoadDuration:       100_000_000,
		PromptEvalDuration: 200_000_000,
		EvalDuration:       1_200_000_000,
	}
	noReallocAppend(t, generateResponseSize(resp), func(buf []byte) []byte { return AppendGenerateResponse(buf, resp) })
}

// TestChunkenc_generateResponseSize_Bad covers the zero-value floor
// and the streaming-token shape (Done false, no metrics).
func TestChunkenc_generateResponseSize_Bad(t *testing.T) {
	cases := []GenerateResponse{
		{},
		{Model: "qwen3", Response: "tok"},
	}
	for _, resp := range cases {
		noReallocAppend(t, generateResponseSize(resp), func(buf []byte) []byte { return AppendGenerateResponse(buf, resp) })
	}
}

// TestChunkenc_generateResponseSize_Ugly mirrors the ChatResponse
// escape-heavy round-trip check.
func TestChunkenc_generateResponseSize_Ugly(t *testing.T) {
	resp := GenerateResponse{
		Model:    "qwen3",
		Response: strings.Repeat("\"quoted\"\n\t", 64),
		Done:     true,
	}
	buf := AppendGenerateResponse(make([]byte, 0, generateResponseSize(resp)), resp)
	var back GenerateResponse
	if err := json.Unmarshal(buf, &back); err != nil {
		t.Fatalf("Unmarshal(%s): %v", buf, err)
	}
	if back != resp {
		t.Fatalf("round-trip mismatch:\n got = %+v\nwant = %+v", back, resp)
	}
}

// TestChunkenc_tagsResponseSize_Good pins the multi-tag shape —
// first entry minimal (Name only), second entry with every optional
// field set — which exercises the i>0 comma-separator accounting
// alongside every per-tag omitempty branch in one pass.
func TestChunkenc_tagsResponseSize_Good(t *testing.T) {
	resp := TagsResponse{Models: []ModelTag{
		{Name: "qwen3:latest"},
		{Name: "gemma3:4b", Model: "gemma3", ModifiedAt: "2026-05-21T10:00:00Z", Size: 2_300_000_000},
	}}
	noReallocAppend(t, tagsResponseSize(resp), func(buf []byte) []byte { return AppendTagsResponse(buf, resp) })
}

// TestChunkenc_tagsResponseSize_Bad covers the nil-Models ("null")
// floor and the empty-but-non-nil slice ("[]") shape — two distinct
// zero-tag encodings that must not be conflated.
func TestChunkenc_tagsResponseSize_Bad(t *testing.T) {
	cases := []TagsResponse{
		{},
		{Models: []ModelTag{}},
	}
	for _, resp := range cases {
		noReallocAppend(t, tagsResponseSize(resp), func(buf []byte) []byte { return AppendTagsResponse(buf, resp) })
	}
}

// TestChunkenc_tagsResponseSize_Ugly drives a larger, unicode-bearing
// batch (open-webui's /api/tags refresh can return dozens of models)
// through the same round-trip contract as the other Ugly cases.
func TestChunkenc_tagsResponseSize_Ugly(t *testing.T) {
	var resp TagsResponse
	for range 50 {
		resp.Models = append(resp.Models, ModelTag{
			Name:       "モデル:latest",
			Model:      "モデル",
			ModifiedAt: "2026-05-21T10:00:00Z",
			Size:       4_500_000_000,
		})
	}
	buf := AppendTagsResponse(make([]byte, 0, tagsResponseSize(resp)), resp)
	var back TagsResponse
	if err := json.Unmarshal(buf, &back); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if len(back.Models) != len(resp.Models) || back.Models[0] != resp.Models[0] {
		t.Fatalf("round-trip mismatch: got %d models, want %d", len(back.Models), len(resp.Models))
	}
}
