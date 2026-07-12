// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"dappco.re/go/inference"
)

// wrappedServiceModel is a decorator that embeds a TextModel and re-exposes the
// base model via Unwrap — the shape of the welfare/policy guards. It deliberately
// does NOT forward EmbeddingModel/RerankModel, so a naive type assertion against
// it strips those capabilities; BaseTextModel must see through it.
type wrappedServiceModel struct {
	inference.TextModel
}

func (m wrappedServiceModel) Unwrap() inference.TextModel { return m.TextModel }

// TestEmbeddingsHandler_Good_Base64 encodes each vector as a base64 string of
// the raw little-endian float32 bytes when encoding_format=base64, and the
// bytes decode back to the model's vector.
func TestEmbeddingsHandler_Good_Base64(t *testing.T) {
	resolver := NewStaticResolver(map[string]inference.TextModel{"emb": &serviceModel{stubModel: &stubModel{}}})
	handler := NewEmbeddingsHandler(resolver)
	req := httptest.NewRequest(http.MethodPost, DefaultEmbeddingsPath,
		strings.NewReader(`{"model":"emb","input":"hello","encoding_format":"base64"}`))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s, want 200", rec.Code, rec.Body.String())
	}
	var resp struct {
		Data []struct {
			Embedding string `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v (body=%s)", err, rec.Body.String())
	}
	if len(resp.Data) != 1 {
		t.Fatalf("data len = %d, want 1", len(resp.Data))
	}
	raw, err := base64.StdEncoding.DecodeString(resp.Data[0].Embedding)
	if err != nil {
		t.Fatalf("embedding is not valid base64: %v", err)
	}
	got := decodeFloat32LE(t, raw)
	// serviceModel returns [len(input), 0.5]; one input string -> [1, 0.5].
	want := []float32{1, 0.5}
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("base64 decoded to %v, want %v", got, want)
	}
}

// TestEmbeddingsHandler_Bad_EncodingFormat rejects an unknown encoding_format
// with a clean 400 rather than silently defaulting.
func TestEmbeddingsHandler_Bad_EncodingFormat(t *testing.T) {
	resolver := NewStaticResolver(map[string]inference.TextModel{"emb": &serviceModel{stubModel: &stubModel{}}})
	handler := NewEmbeddingsHandler(resolver)
	req := httptest.NewRequest(http.MethodPost, DefaultEmbeddingsPath,
		strings.NewReader(`{"model":"emb","input":"hello","encoding_format":"hex"}`))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400 for unknown encoding_format", rec.Code)
	}
}

// TestEmbeddingsHandler_Good_UnwrapsGuard proves a wrapped embedder (welfare /
// policy guard shape) still serves /v1/embeddings — the capability gate unwraps
// past the decorator instead of stripping EmbeddingModel.
func TestEmbeddingsHandler_Good_UnwrapsGuard(t *testing.T) {
	base := &serviceModel{stubModel: &stubModel{}}
	wrapped := wrappedServiceModel{TextModel: base}
	// The decorator itself must NOT satisfy EmbeddingModel — otherwise this
	// test would pass without the unwrap and prove nothing.
	if _, ok := inference.TextModel(wrapped).(inference.EmbeddingModel); ok {
		t.Fatal("test decorator unexpectedly satisfies EmbeddingModel; unwrap would be untested")
	}
	resolver := NewStaticResolver(map[string]inference.TextModel{"emb": wrapped})
	handler := NewEmbeddingsHandler(resolver)
	req := httptest.NewRequest(http.MethodPost, DefaultEmbeddingsPath,
		strings.NewReader(`{"model":"emb","input":"hello"}`))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s, want 200 (wrapped embedder must serve)", rec.Code, rec.Body.String())
	}
}

// TestRerankHandler_Good_UnwrapsGuard is the rerank twin of the embeddings
// unwrap test.
func TestRerankHandler_Good_UnwrapsGuard(t *testing.T) {
	base := &serviceModel{stubModel: &stubModel{}}
	wrapped := wrappedServiceModel{TextModel: base}
	resolver := NewStaticResolver(map[string]inference.TextModel{"rr": wrapped})
	handler := NewRerankHandler(resolver)
	req := httptest.NewRequest(http.MethodPost, DefaultRerankPath,
		strings.NewReader(`{"model":"rr","query":"q","documents":["a","b"]}`))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s, want 200 (wrapped reranker must serve)", rec.Code, rec.Body.String())
	}
}

func decodeFloat32LE(t *testing.T, raw []byte) []float32 {
	t.Helper()
	if len(raw)%4 != 0 {
		t.Fatalf("raw byte length %d is not a multiple of 4", len(raw))
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}
