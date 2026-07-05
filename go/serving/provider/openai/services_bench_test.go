// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the OpenAI-compatible service-endpoint wire shapes:
// embeddings, rerank, cache stats/warm/clear, cancel. Per AX-11 — every
// embedding ingestion serialises an EmbeddingResponse with one
// EmbeddingResponseDatum per vector, and every rerank call serialises
// a RerankResult payload. EmbeddingInput.UnmarshalJSON variant parse is
// hit on every embeddings request.
//
// Run:    go test -bench='BenchmarkServices' -benchtime=100ms -benchmem -run='^$' .

package openai

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// Sinks defeat compiler DCE.
var (
	servicesSinkEmbedRequest   EmbeddingRequest
	servicesSinkEmbedResponse  EmbeddingResponse
	servicesSinkEmbeddingInput EmbeddingInput
	servicesSinkRerankRequest  RerankRequest
	servicesSinkRerankResponse RerankResponse
	servicesSinkCacheWarmReq   CacheWarmRequest
	servicesSinkCacheClearReq  CacheClearRequest
	servicesSinkCancelReq      CancelRequest
	servicesSinkCacheStats     inference.CacheStats
	servicesSinkErr            error
	servicesSinkString         string
	servicesSinkBytes          []byte
	servicesSinkResult         core.Result
)

// --- Fixture builders ---

// buildEmbeddingVectors generates synthetic vectors of the requested
// dimension and count — matches the production response shape where
// each input string maps to one vector.
func buildEmbeddingVectors(count, dim int) [][]float32 {
	out := make([][]float32, count)
	for i := range out {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = float32(i*dim+j) * 0.001
		}
		out[i] = vec
	}
	return out
}

func buildEmbeddingResponse(count, dim int) EmbeddingResponse {
	vectors := buildEmbeddingVectors(count, dim)
	data := make([]EmbeddingResponseDatum, 0, count)
	for i, vec := range vectors {
		data = append(data, EmbeddingResponseDatum{Object: "embedding", Index: i, Embedding: vec})
	}
	return EmbeddingResponse{
		Object: "list",
		Data:   data,
		Model:  "qwen3-embed",
		Usage:  inference.EmbeddingUsage{PromptTokens: count * 16, TotalTokens: count * 16},
	}
}

// --- EmbeddingInput.UnmarshalJSON — variant parse on every embeddings request ---

func BenchmarkServices_EmbeddingInput_UnmarshalJSON_SingleString(b *testing.B) {
	data := []byte(`"hello world"`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var input EmbeddingInput
		servicesSinkErr = input.UnmarshalJSON(data)
		servicesSinkEmbeddingInput = input
	}
}

func BenchmarkServices_EmbeddingInput_UnmarshalJSON_SmallArray(b *testing.B) {
	data := []byte(`["one","two","three"]`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var input EmbeddingInput
		servicesSinkErr = input.UnmarshalJSON(data)
		servicesSinkEmbeddingInput = input
	}
}

func BenchmarkServices_EmbeddingInput_UnmarshalJSON_TwentyArray(b *testing.B) {
	body := `["alpha","beta","gamma","delta","epsilon","zeta","eta","theta","iota","kappa","lambda","mu","nu","xi","omicron","pi","rho","sigma","tau","upsilon"]`
	data := []byte(body)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var input EmbeddingInput
		servicesSinkErr = input.UnmarshalJSON(data)
		servicesSinkEmbeddingInput = input
	}
}

// --- EmbeddingRequest — full request unmarshal at handler entry ---

func BenchmarkServices_UnmarshalEmbeddingRequest_SingleInput(b *testing.B) {
	body := `{"model":"qwen3-embed","input":"hello world","normalize":true}`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var req EmbeddingRequest
		servicesSinkResult = core.JSONUnmarshalString(body, &req)
		servicesSinkEmbedRequest = req
	}
}

func BenchmarkServices_UnmarshalEmbeddingRequest_ArrayInput(b *testing.B) {
	body := `{"model":"qwen3-embed","input":["one","two","three","four","five"],"normalize":true,"dimensions":768}`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var req EmbeddingRequest
		servicesSinkResult = core.JSONUnmarshalString(body, &req)
		servicesSinkEmbedRequest = req
	}
}

// --- EmbeddingResponse marshal — response emission ---
// Three dim/count shapes — small (1×384), medium (5×768), large (20×1024).

func BenchmarkServices_MarshalEmbeddingResponse_1x384(b *testing.B) {
	resp := buildEmbeddingResponse(1, 384)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		servicesSinkString = core.JSONMarshalString(resp)
	}
}

func BenchmarkServices_MarshalEmbeddingResponse_5x768(b *testing.B) {
	resp := buildEmbeddingResponse(5, 768)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		servicesSinkString = core.JSONMarshalString(resp)
	}
}

func BenchmarkServices_MarshalEmbeddingResponse_20x1024(b *testing.B) {
	resp := buildEmbeddingResponse(20, 1024)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		servicesSinkString = core.JSONMarshalString(resp)
	}
}

// --- Hand-rolled embedding-response encoder — writeJSON fast path ---
// Compares directly against the encoding/json reflect-walk path
// above. Per-element float32 emission scales with vector dim.

func BenchmarkServices_AppendEmbeddingResponse_1x384(b *testing.B) {
	resp := buildEmbeddingResponse(1, 384)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		servicesSinkBytes = appendEmbeddingResponse(make([]byte, 0, embeddingResponseSize(resp)), resp)
	}
}

func BenchmarkServices_AppendEmbeddingResponse_5x768(b *testing.B) {
	resp := buildEmbeddingResponse(5, 768)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		servicesSinkBytes = appendEmbeddingResponse(make([]byte, 0, embeddingResponseSize(resp)), resp)
	}
}

func BenchmarkServices_AppendEmbeddingResponse_20x1024(b *testing.B) {
	resp := buildEmbeddingResponse(20, 1024)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		servicesSinkBytes = appendEmbeddingResponse(make([]byte, 0, embeddingResponseSize(resp)), resp)
	}
}

// --- RerankRequest unmarshal ---

func BenchmarkServices_UnmarshalRerankRequest_FewDocs(b *testing.B) {
	body := `{"model":"qwen3-rerank","query":"core primitives","documents":["a","b","c"],"top_n":2}`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var req RerankRequest
		servicesSinkResult = core.JSONUnmarshalString(body, &req)
		servicesSinkRerankRequest = req
	}
}

func BenchmarkServices_UnmarshalRerankRequest_TwentyDocs(b *testing.B) {
	body := `{"model":"qwen3-rerank","query":"core primitives","documents":["alpha","beta","gamma","delta","epsilon","zeta","eta","theta","iota","kappa","lambda","mu","nu","xi","omicron","pi","rho","sigma","tau","upsilon"],"top_n":5}`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var req RerankRequest
		servicesSinkResult = core.JSONUnmarshalString(body, &req)
		servicesSinkRerankRequest = req
	}
}

// --- RerankResponse marshal ---

func BenchmarkServices_MarshalRerankResponse_FewResults(b *testing.B) {
	resp := RerankResponse{
		Object: "list",
		Model:  "qwen3-rerank",
		Results: []inference.RerankScore{
			{Index: 0, Score: 0.91, Text: "alpha"},
			{Index: 1, Score: 0.82, Text: "beta"},
			{Index: 2, Score: 0.74, Text: "gamma"},
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		servicesSinkString = core.JSONMarshalString(resp)
	}
}

func BenchmarkServices_MarshalRerankResponse_TwentyResults(b *testing.B) {
	results := make([]inference.RerankScore, 20)
	for i := range results {
		results[i] = inference.RerankScore{Index: i, Score: 0.95 - float64(i)*0.04, Text: "document text fragment"}
	}
	resp := RerankResponse{Object: "list", Model: "qwen3-rerank", Results: results}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		servicesSinkString = core.JSONMarshalString(resp)
	}
}

// --- Hand-rolled rerank-response encoder — writeJSON fast path ---

func BenchmarkServices_AppendRerankResponse_FewResults(b *testing.B) {
	resp := RerankResponse{
		Object: "list",
		Model:  "qwen3-rerank",
		Results: []inference.RerankScore{
			{Index: 0, Score: 0.91, Text: "alpha"},
			{Index: 1, Score: 0.82, Text: "beta"},
			{Index: 2, Score: 0.74, Text: "gamma"},
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		servicesSinkBytes = appendRerankResponse(make([]byte, 0, rerankResponseSize(resp)), resp)
	}
}

func BenchmarkServices_AppendRerankResponse_TwentyResults(b *testing.B) {
	results := make([]inference.RerankScore, 20)
	for i := range results {
		results[i] = inference.RerankScore{Index: i, Score: 0.95 - float64(i)*0.04, Text: "document text fragment"}
	}
	resp := RerankResponse{Object: "list", Model: "qwen3-rerank", Results: results}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		servicesSinkBytes = appendRerankResponse(make([]byte, 0, rerankResponseSize(resp)), resp)
	}
}

// --- CacheWarmRequest — KV cache prep request ingress ---

func BenchmarkServices_UnmarshalCacheWarmRequest_Prompt(b *testing.B) {
	body := `{"model":"qwen3","prompt":"You are a helpful assistant. Summarise this paragraph.","mode":"block-q8","labels":{"adapter":"none"}}`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var req CacheWarmRequest
		servicesSinkResult = core.JSONUnmarshalString(body, &req)
		servicesSinkCacheWarmReq = req
	}
}

func BenchmarkServices_UnmarshalCacheWarmRequest_Tokens(b *testing.B) {
	body := `{"model":"qwen3","tokens":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],"mode":"block-q8"}`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var req CacheWarmRequest
		servicesSinkResult = core.JSONUnmarshalString(body, &req)
		servicesSinkCacheWarmReq = req
	}
}

// --- CacheClearRequest ---

func BenchmarkServices_UnmarshalCacheClearRequest(b *testing.B) {
	body := `{"model":"qwen3","labels":{"adapter":"none","scope":"all"}}`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var req CacheClearRequest
		servicesSinkResult = core.JSONUnmarshalString(body, &req)
		servicesSinkCacheClearReq = req
	}
}

// --- CancelRequest ---

func BenchmarkServices_UnmarshalCancelRequest(b *testing.B) {
	body := `{"model":"qwen3","id":"req_1700000000_42"}`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var req CancelRequest
		servicesSinkResult = core.JSONUnmarshalString(body, &req)
		servicesSinkCancelReq = req
	}
}

// --- CacheStats marshal — what /v1/cache/stats returns per call ---

func BenchmarkServices_MarshalCacheStats(b *testing.B) {
	stats := inference.CacheStats{
		Blocks:    128,
		Hits:      9000,
		Misses:    1000,
		HitRate:   0.9,
		CacheMode: "block-q8",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		servicesSinkString = core.JSONMarshalString(stats)
	}
}
