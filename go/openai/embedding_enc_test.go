// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"encoding/json"
	"math"
	"testing"

	"dappco.re/go/inference"
)

// TestEmbeddingResponse_AppendRoundTrip locks the hand-rolled
// embedding-response encoder against encoding/json's deserialiser.
// The wire shape is consumed by every OpenAI-compatible embedding
// client; round-trip on every embedding-model output preserves the
// per-element float32 values within the standard 'g' precision the
// stdlib emits.
func TestEmbeddingResponse_AppendRoundTrip(t *testing.T) {
	cases := []struct {
		name string
		in   EmbeddingResponse
	}{
		{"single-vector", EmbeddingResponse{
			Object: "list",
			Data: []EmbeddingResponseDatum{{
				Object:    "embedding",
				Index:     0,
				Embedding: []float32{0.1, -0.2, 0.75, 1.0},
			}},
			Model: "qwen3-embed",
			Usage: inference.EmbeddingUsage{PromptTokens: 4, TotalTokens: 4},
		}},
		{"multi-vector", EmbeddingResponse{
			Object: "list",
			Data: []EmbeddingResponseDatum{
				{Object: "embedding", Index: 0, Embedding: []float32{0.0, 0.5}},
				{Object: "embedding", Index: 1, Embedding: []float32{-1.0, 1.0}},
				{Object: "embedding", Index: 2, Embedding: []float32{1e-5, 1e5}},
			},
			Model: "qwen3-embed",
			Usage: inference.EmbeddingUsage{PromptTokens: 12, TotalTokens: 12},
		}},
		{"empty-vectors", EmbeddingResponse{
			Object: "list",
			Data:   []EmbeddingResponseDatum{{Object: "embedding", Index: 0, Embedding: []float32{}}},
			Model:  "qwen3-embed",
			Usage:  inference.EmbeddingUsage{PromptTokens: 0, TotalTokens: 0},
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			encoded := appendEmbeddingResponse(nil, tc.in)
			var back EmbeddingResponse
			if err := json.Unmarshal(encoded, &back); err != nil {
				t.Fatalf("json.Unmarshal(%s) error = %v", encoded, err)
			}
			if back.Object != tc.in.Object || back.Model != tc.in.Model {
				t.Fatalf("identity: got %+v, want %+v", back, tc.in)
			}
			if back.Usage != tc.in.Usage {
				t.Fatalf("usage: got %+v, want %+v", back.Usage, tc.in.Usage)
			}
			if len(back.Data) != len(tc.in.Data) {
				t.Fatalf("data len = %d, want %d", len(back.Data), len(tc.in.Data))
			}
			for i := range tc.in.Data {
				if back.Data[i].Object != tc.in.Data[i].Object || back.Data[i].Index != tc.in.Data[i].Index {
					t.Fatalf("data[%d] header: got %+v want %+v", i, back.Data[i], tc.in.Data[i])
				}
				if len(back.Data[i].Embedding) != len(tc.in.Data[i].Embedding) {
					t.Fatalf("data[%d].embedding len = %d, want %d", i, len(back.Data[i].Embedding), len(tc.in.Data[i].Embedding))
				}
				for j, v := range tc.in.Data[i].Embedding {
					if math.IsNaN(float64(v)) {
						continue
					}
					if back.Data[i].Embedding[j] != v {
						t.Fatalf("data[%d].embedding[%d] = %v, want %v", i, j, back.Data[i].Embedding[j], v)
					}
				}
			}
		})
	}
}
