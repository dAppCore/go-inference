// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the wire-contract shapes — the value-types that flow
// over scheduler queues, between the cache subsystem and consumers,
// and through the embed / rerank / tool-parse paths.
// Per AX-11 — these shapes are constructed at the rate of generation
// (one ScheduledToken per emitted token; one CacheStats per request;
// CacheBlockRef cloned per warm-cache call), so structural allocation
// pressure here adds to every served request.
//
// Run:    go test -bench=BenchmarkContracts -benchmem -run='^$' .

package inference

import (
	"context"
	"testing"
)

// Sinks defeat compiler DCE.
var (
	contractsBenchSinkRequestHandle    RequestHandle
	contractsBenchSinkCancelResult     RequestCancelResult
	contractsBenchSinkScheduledRequest ScheduledRequest
	contractsBenchSinkScheduledToken   ScheduledToken
	contractsBenchSinkCacheBlockRef    CacheBlockRef
	contractsBenchSinkCacheStats       CacheStats
	contractsBenchSinkCacheWarmReq     CacheWarmRequest
	contractsBenchSinkCacheWarmRes     CacheWarmResult
	contractsBenchSinkEmbedReq         EmbeddingRequest
	contractsBenchSinkEmbedRes         *EmbeddingResult
	contractsBenchSinkRerankReq        RerankRequest
	contractsBenchSinkRerankRes        *RerankResult
	contractsBenchSinkReasoningRes     ReasoningParseResult
	contractsBenchSinkToolRes          ToolParseResult
	contractsBenchSinkInspection       *ModelPackInspection
	contractsBenchSinkErr              error
	contractsBenchSinkChan             <-chan ScheduledToken
)

// benchScheduledRequestSmall — single short prompt, no labels.
// Tests the minimal allocation floor of the scheduler-input shape.
func benchScheduledRequestSmall() ScheduledRequest {
	return ScheduledRequest{
		ID:     "req-1",
		Model:  "qwen3",
		Prompt: "hello",
		Sampler: SamplerConfig{
			MaxTokens: 64,
		},
	}
}

// benchScheduledRequestTypical — typical chat input — 4 messages,
// realistic sampler config, request-side labels. Closer to what the
// scheduler enqueues per chat turn.
func benchScheduledRequestTypical() ScheduledRequest {
	return ScheduledRequest{
		ID:    "req-typical",
		Model: "qwen3",
		Messages: []Message{
			{Role: "system", Content: "You are a helpful assistant."},
			{Role: "user", Content: "What is 2+2?"},
			{Role: "assistant", Content: "4"},
			{Role: "user", Content: "Are you sure?"},
		},
		Sampler: SamplerConfig{
			MaxTokens:     256,
			Temperature:   0.7,
			TopK:          40,
			TopP:          0.9,
			RepeatPenalty: 1.1,
			StopTokens:    []int32{2},
		},
		Labels: map[string]string{"user_id": "u-42", "session": "s-7"},
	}
}

// benchCacheStats — typical request-time cache reading.
func benchCacheStats() CacheStats {
	return CacheStats{
		Blocks:        16,
		MemoryBytes:   1 << 28, // 256 MiB
		DiskBytes:     1 << 30, // 1 GiB
		Hits:          1024,
		Misses:        128,
		Evictions:     12,
		HitRate:       0.88,
		RestoreMillis: 4.2,
		CacheMode:     "paged-q8",
		Labels:        map[string]string{"profile": "qwen3-paged-q8"},
	}
}

// benchCacheBlockRef — single block descriptor (one of many in a
// CacheWarmResult). Allocated per warmed block.
func benchCacheBlockRef() CacheBlockRef {
	return CacheBlockRef{
		ID:            "block-7",
		Kind:          "kv",
		ModelHash:     "sha256:model",
		AdapterHash:   "sha256:adapter",
		TokenizerHash: "sha256:tok",
		TokenStart:    128,
		TokenCount:    256,
		SizeBytes:     1 << 22, // 4 MiB
		Encoding:      "paged-q8",
		Labels:        map[string]string{"layer": "12"},
	}
}

// benchReasoningParseResult — typical decode-event with 32 visible
// tokens + 1 thinking segment (Qwen3 / Gemma thinking-tokens shape).
func benchReasoningParseResult32Tokens() ReasoningParseResult {
	return ReasoningParseResult{
		VisibleText: "The answer is 4 — addition is commutative.",
		Reasoning: []ReasoningSegment{
			{
				Kind:       "think",
				Text:       "Confirm: 2+2 = 4. Already given as answer; reaffirm with brief justification.",
				StartToken: 0,
				EndToken:   32,
				Labels:     map[string]string{"channel": "thinking"},
			},
		},
	}
}

// benchReasoningParseResult256Tokens — long-form thinking channel.
func benchReasoningParseResult256Tokens() ReasoningParseResult {
	return ReasoningParseResult{
		VisibleText: "After step-by-step reasoning, the answer is 4.",
		Reasoning: []ReasoningSegment{
			{
				Kind:       "think",
				Text:       "Step 1: Identify the operation as addition. Step 2: Recall 2+2. Step 3: Apply the additive identity for natural numbers. Step 4: Cross-check by counting. Step 5: Confirm 4. Step 6: Make sure no edge cases (negative, decimal). Step 7: Final answer is 4.",
				StartToken: 0,
				EndToken:   256,
				Labels:     map[string]string{"channel": "thinking"},
			},
		},
	}
}

// --- ScheduledRequest / ScheduledToken construction ---
// One ScheduledToken per emitted token — the wire shape callers
// destructure per yield.

func BenchmarkContracts_ScheduledRequest_Small(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contractsBenchSinkScheduledRequest = benchScheduledRequestSmall()
	}
}

func BenchmarkContracts_ScheduledRequest_Typical(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contractsBenchSinkScheduledRequest = benchScheduledRequestTypical()
	}
}

func BenchmarkContracts_ScheduledToken(b *testing.B) {
	metrics := GenerateMetrics{PromptTokens: 128, GeneratedTokens: 1}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contractsBenchSinkScheduledToken = ScheduledToken{
			RequestID: "req-7",
			Token:     Token{ID: 42, Text: "hello"},
			Metrics:   metrics,
		}
	}
}

func BenchmarkContracts_RequestHandle(b *testing.B) {
	identity := ModelIdentity{Architecture: "qwen3"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contractsBenchSinkRequestHandle = RequestHandle{
			ID:    "req-1",
			Model: identity,
		}
	}
}

func BenchmarkContracts_RequestCancelResult(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contractsBenchSinkCancelResult = RequestCancelResult{
			ID:        "req-1",
			Cancelled: true,
			Reason:    "client closed connection",
		}
	}
}

// --- CacheStats / CacheBlockRef (per-request cache reading) ---

func BenchmarkContracts_CacheStats_Construct(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contractsBenchSinkCacheStats = benchCacheStats()
	}
}

func BenchmarkContracts_CacheBlockRef_Construct(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contractsBenchSinkCacheBlockRef = benchCacheBlockRef()
	}
}

// --- CacheWarmRequest / CacheWarmResult ---
// Per warm-cache call: 1 request shape + 1 result shape carrying N blocks.

func BenchmarkContracts_CacheWarmRequest_64Tokens(b *testing.B) {
	tokens := make([]int32, 64)
	for i := range tokens {
		tokens[i] = int32(i + 1)
	}
	model := ModelIdentity{Architecture: "qwen3"}
	adapter := AdapterIdentity{Format: "lora"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contractsBenchSinkCacheWarmReq = CacheWarmRequest{
			Model:   model,
			Adapter: adapter,
			Prompt:  "hello",
			Tokens:  tokens,
			Mode:    "paged-q8",
		}
	}
}

func BenchmarkContracts_CacheWarmResult_8Blocks(b *testing.B) {
	blocks := []CacheBlockRef{
		benchCacheBlockRef(), benchCacheBlockRef(), benchCacheBlockRef(), benchCacheBlockRef(),
		benchCacheBlockRef(), benchCacheBlockRef(), benchCacheBlockRef(), benchCacheBlockRef(),
	}
	stats := benchCacheStats()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contractsBenchSinkCacheWarmRes = CacheWarmResult{
			Blocks: blocks,
			Stats:  stats,
		}
	}
}

// --- Embedding wire-shape (per-request constructor cost) ---

func BenchmarkContracts_EmbeddingRequest_8Inputs(b *testing.B) {
	inputs := []string{"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contractsBenchSinkEmbedReq = EmbeddingRequest{
			Model:     "qwen3-embed",
			Input:     inputs,
			Normalize: true,
		}
	}
}

func BenchmarkContracts_EmbeddingResult_8Vectors(b *testing.B) {
	model := ModelIdentity{Architecture: "qwen3-embed"}
	model.Hash = "sha256:embed-1"
	vectors := make([][]float32, 8)
	for i := range vectors {
		vec := make([]float32, 64)
		for j := range vec {
			vec[j] = float32(i + j)
		}
		vectors[i] = vec
	}
	model.Path = "/models/embed"
	model.VocabSize = 32000
	model.NumLayers = 12
	model.HiddenSize = 768
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contractsBenchSinkEmbedRes = &EmbeddingResult{
			Model:   model,
			Vectors: vectors,
			Usage:   EmbeddingUsage{PromptTokens: 32, TotalTokens: 32},
		}
	}
}

// --- Rerank wire-shape ---

func BenchmarkContracts_RerankRequest_16Docs(b *testing.B) {
	docs := []string{
		"doc-a", "doc-b", "doc-c", "doc-d",
		"doc-e", "doc-f", "doc-g", "doc-h",
		"doc-i", "doc-j", "doc-k", "doc-l",
		"doc-m", "doc-n", "doc-o", "doc-p",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contractsBenchSinkRerankReq = RerankRequest{
			Model:     "qwen3-rerank",
			Query:     "what is the meaning",
			Documents: docs,
			TopN:      4,
		}
	}
}

func BenchmarkContracts_RerankResult_4Scores(b *testing.B) {
	model := ModelIdentity{Architecture: "qwen3-rerank"}
	results := []RerankScore{
		{Index: 0, Score: 0.91, Text: "doc-a"},
		{Index: 3, Score: 0.84, Text: "doc-d"},
		{Index: 7, Score: 0.71, Text: "doc-h"},
		{Index: 9, Score: 0.60, Text: "doc-j"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contractsBenchSinkRerankRes = &RerankResult{
			Model:   model,
			Results: results,
		}
	}
}

// --- ReasoningParseResult / ToolParseResult ---
// Constructed per-decode-event when models emit thinking/tool channels.

func BenchmarkContracts_ReasoningParseResult_32Tokens(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contractsBenchSinkReasoningRes = benchReasoningParseResult32Tokens()
	}
}

func BenchmarkContracts_ReasoningParseResult_256Tokens(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contractsBenchSinkReasoningRes = benchReasoningParseResult256Tokens()
	}
}

func BenchmarkContracts_ToolParseResult_OneCall(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contractsBenchSinkToolRes = ToolParseResult{
			VisibleText: "I'll search for that.",
			Calls: []ToolCall{
				{
					ID:            "call-1",
					Name:          "search",
					Type:          "function",
					ArgumentsJSON: `{"q":"core","limit":10}`,
				},
			},
		}
	}
}

func BenchmarkContracts_ToolParseResult_ThreeCalls(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contractsBenchSinkToolRes = ToolParseResult{
			VisibleText: "Running three tools in parallel.",
			Calls: []ToolCall{
				{ID: "call-1", Name: "search", Type: "function", ArgumentsJSON: `{"q":"alpha"}`},
				{ID: "call-2", Name: "fetch", Type: "function", ArgumentsJSON: `{"url":"https://x"}`},
				{ID: "call-3", Name: "write", Type: "function", ArgumentsJSON: `{"path":"/tmp/out"}`},
			},
		}
	}
}

// --- ModelPackInspection (one per model-pack scan) ---

func BenchmarkContracts_ModelPackInspection_Construct(b *testing.B) {
	model := ModelIdentity{Architecture: "qwen3", NumLayers: 28, QuantBits: 4}
	tokenizer := TokenizerIdentity{Kind: "sentencepiece", EOSID: 2}
	caps := []Capability{
		SupportedCapability(CapabilityGenerate, CapabilityGroupModel),
		SupportedCapability(CapabilityChat, CapabilityGroupModel),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contractsBenchSinkInspection = &ModelPackInspection{
			Path:         "/models/qwen3-1b",
			Format:       "safetensors",
			Model:        model,
			Tokenizer:    tokenizer,
			Supported:    true,
			Capabilities: caps,
		}
	}
}

// --- Through a model — exercises the full call shape under the
// optional-interface scheduler / cache / embed / rerank / parsers. ---

func BenchmarkContracts_SchedulerModel_Schedule(b *testing.B) {
	model := &contractModel{}
	req := benchScheduledRequestTypical()
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contractsBenchSinkRequestHandle, contractsBenchSinkChan, contractsBenchSinkErr = model.Schedule(ctx, req)
		// Drain the one-element channel so the test cleanup paths
		// match production usage and the GC can reclaim the buffer.
		for range contractsBenchSinkChan {
		}
	}
}

func BenchmarkContracts_CancellableModel_CancelRequest(b *testing.B) {
	model := &contractModel{}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contractsBenchSinkCancelResult, contractsBenchSinkErr = model.CancelRequest(ctx, "req-1")
	}
}

func BenchmarkContracts_CacheService_CacheStats(b *testing.B) {
	model := &contractModel{}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contractsBenchSinkCacheStats, contractsBenchSinkErr = model.CacheStats(ctx)
	}
}

func BenchmarkContracts_CacheService_WarmCache(b *testing.B) {
	model := &contractModel{}
	tokens := make([]int32, 64)
	for i := range tokens {
		tokens[i] = int32(i + 1)
	}
	req := CacheWarmRequest{Tokens: tokens}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contractsBenchSinkCacheWarmRes, contractsBenchSinkErr = model.WarmCache(ctx, req)
	}
}

func BenchmarkContracts_EmbeddingModel_Embed(b *testing.B) {
	model := &contractModel{}
	req := EmbeddingRequest{Input: []string{"hello"}}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contractsBenchSinkEmbedRes, contractsBenchSinkErr = model.Embed(ctx, req)
	}
}

func BenchmarkContracts_RerankModel_Rerank(b *testing.B) {
	model := &contractModel{}
	req := RerankRequest{Query: "core", Documents: []string{"doc"}}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contractsBenchSinkRerankRes, contractsBenchSinkErr = model.Rerank(ctx, req)
	}
}

func BenchmarkContracts_ReasoningParser_ParseReasoning(b *testing.B) {
	model := &contractModel{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contractsBenchSinkReasoningRes, contractsBenchSinkErr = model.ParseReasoning(nil, "answer")
	}
}

func BenchmarkContracts_ToolParser_ParseTools(b *testing.B) {
	model := &contractModel{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contractsBenchSinkToolRes, contractsBenchSinkErr = model.ParseTools(nil, "call")
	}
}

func BenchmarkContracts_ModelPackInspector_InspectModelPack(b *testing.B) {
	model := &contractModel{}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contractsBenchSinkInspection, contractsBenchSinkErr = model.InspectModelPack(ctx, "/models/qwen")
	}
}
