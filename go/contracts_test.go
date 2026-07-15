// SPDX-Licence-Identifier: EUPL-1.2

package inference

import (
	"context"
	"testing"
)

type contractModel struct {
	*stubTextModel
}

func (m *contractModel) Schedule(_ context.Context, req ScheduledRequest) (RequestHandle, <-chan ScheduledToken, error) {
	ch := make(chan ScheduledToken, 1)
	ch <- ScheduledToken{RequestID: req.ID, Token: Token{Text: "ok"}}
	close(ch)
	return RequestHandle{ID: req.ID}, ch, nil
}

func (m *contractModel) CancelRequest(_ context.Context, id string) (RequestCancelResult, error) {
	return RequestCancelResult{ID: id, Cancelled: id != ""}, nil
}

func (m *contractModel) CacheStats(context.Context) (CacheStats, error) {
	return CacheStats{Blocks: 2, Hits: 3, Misses: 1, HitRate: 0.75, CacheMode: "paged-q8"}, nil
}

func (m *contractModel) WarmCache(_ context.Context, req CacheWarmRequest) (CacheWarmResult, error) {
	return CacheWarmResult{Blocks: []CacheBlockRef{{ID: "block-1", TokenCount: len(req.Tokens)}}}, nil
}

func (m *contractModel) ClearCache(context.Context, map[string]string) (CacheStats, error) {
	return CacheStats{}, nil
}

func (m *contractModel) Embed(_ context.Context, req EmbeddingRequest) (*EmbeddingResult, error) {
	return &EmbeddingResult{Vectors: [][]float32{{1, 0}}, Usage: EmbeddingUsage{PromptTokens: len(req.Input), TotalTokens: len(req.Input)}}, nil
}

func (m *contractModel) Rerank(_ context.Context, req RerankRequest) (*RerankResult, error) {
	return &RerankResult{Results: []RerankScore{{Index: 0, Score: 0.9, Text: req.Documents[0]}}}, nil
}

func (m *contractModel) ParseReasoning(_ []Token, text string) (ReasoningParseResult, error) {
	return ReasoningParseResult{VisibleText: text, Reasoning: []ReasoningSegment{{Kind: "think", Text: "plan"}}}, nil
}

func (m *contractModel) ParseTools(_ []Token, text string) (ToolParseResult, error) {
	return ToolParseResult{VisibleText: text, Calls: []ToolCall{{ID: "call-1", Name: "search", Type: "function", ArgumentsJSON: `{"q":"core"}`}}}, nil
}

func (m *contractModel) InspectModelPack(_ context.Context, path string) (*ModelPackInspection, error) {
	return &ModelPackInspection{Path: path, Format: "safetensors", Supported: true, Model: ModelIdentity{Architecture: "qwen3"}}, nil
}

func (m *contractModel) WakeState(_ context.Context, req AgentMemoryWakeRequest) (*AgentMemoryWakeResult, error) {
	return &AgentMemoryWakeResult{
		Entry:        AgentMemoryRef{URI: req.EntryURI, TokenCount: 8},
		PrefixTokens: 8,
		BlocksRead:   2,
	}, nil
}

func (m *contractModel) SleepState(_ context.Context, req AgentMemorySleepRequest) (*AgentMemorySleepResult, error) {
	return &AgentMemorySleepResult{
		Entry:         AgentMemoryRef{URI: req.EntryURI, Title: req.Title, TokenCount: 9},
		TokenCount:    9,
		BlocksWritten: 3,
	}, nil
}

func (m *contractModel) ForkState(_ context.Context, req AgentMemoryWakeRequest) (AgentMemorySession, *AgentMemoryWakeResult, error) {
	return m, &AgentMemoryWakeResult{Entry: AgentMemoryRef{URI: req.EntryURI}, PrefixTokens: 8}, nil
}

func TestContracts_CapabilityID_Good(t *testing.T) {
	ids := []CapabilityID{
		CapabilityResponsesAPI,
		CapabilityAnthropicMessages,
		CapabilityOllamaCompat,
		CapabilityEmbeddings,
		CapabilityRerank,
		CapabilityScheduler,
		CapabilityRequestCancel,
		CapabilityCacheBlocks,
		CapabilityCacheDisk,
		CapabilityCacheWarm,
		CapabilityToolParse,
		CapabilityReasoningParse,
		CapabilitySpeculativeDecode,
		CapabilityPromptLookupDecode,
		CapabilityMoERouting,
		CapabilityMoELazyExperts,
		CapabilityJANGTQ,
		CapabilityCodebookVQ,
		CapabilityAgentMemory,
		CapabilityStateWake,
		CapabilityStateSleep,
		CapabilityStateFork,
	}

	seen := map[CapabilityID]bool{}
	for _, id := range ids {
		if id == "" {
			t.Fatal("capability ID must not be blank")
		}
		if seen[id] {
			t.Fatalf("duplicate capability ID %q", id)
		}
		seen[id] = true
	}
}

func TestContracts_SchedulerModel_Good(t *testing.T) {
	model := &contractModel{stubTextModel: &stubTextModel{}}

	_, ok := any(model).(SchedulerModel)
	checkTrue(t, ok)
	_, ok = any(model).(CancellableModel)
	checkTrue(t, ok)
	_, ok = any(model).(CacheService)
	checkTrue(t, ok)
	_, ok = any(model).(EmbeddingModel)
	checkTrue(t, ok)
	_, ok = any(model).(RerankModel)
	checkTrue(t, ok)
	_, ok = any(model).(ReasoningParser)
	checkTrue(t, ok)
	_, ok = any(model).(ToolParser)
	checkTrue(t, ok)
	_, ok = any(model).(ModelPackInspector)
	checkTrue(t, ok)
	_, ok = any(model).(AgentMemorySession)
	checkTrue(t, ok)
	_, ok = any(model).(AgentMemoryForker)
	checkTrue(t, ok)
}

func TestContracts_TextModelCapabilities_Good_InferNewOptionalInterfaces(t *testing.T) {
	report := TextModelCapabilities(RuntimeIdentity{Backend: "test"}, &contractModel{stubTextModel: &stubTextModel{}})

	checkTrue(t, report.Supports(CapabilityScheduler))
	checkTrue(t, report.Supports(CapabilityRequestCancel))
	checkTrue(t, report.Supports(CapabilityCacheBlocks))
	checkTrue(t, report.Supports(CapabilityCacheWarm))
	checkTrue(t, report.Supports(CapabilityEmbeddings))
	checkTrue(t, report.Supports(CapabilityRerank))
	checkTrue(t, report.Supports(CapabilityReasoningParse))
	checkTrue(t, report.Supports(CapabilityToolParse))
	checkTrue(t, report.Supports(CapabilityAgentMemory))
	checkTrue(t, report.Supports(CapabilityStateWake))
	checkTrue(t, report.Supports(CapabilityStateSleep))
	checkTrue(t, report.Supports(CapabilityStateFork))
}

func TestContracts_CacheService_Good(t *testing.T) {
	model := &contractModel{}
	service := any(model).(CacheService)

	stats, err := service.CacheStats(context.Background())
	checkNoError(t, err)
	checkEqual(t, "paged-q8", stats.CacheMode)

	warmed, err := service.WarmCache(context.Background(), CacheWarmRequest{Tokens: []int32{1, 2, 3}})
	checkNoError(t, err)
	checkLen(t, warmed.Blocks, 1)
	checkEqual(t, 3, warmed.Blocks[0].TokenCount)
}

func TestContracts_EmbeddingModel_Good(t *testing.T) {
	model := &contractModel{}

	embeddings, err := any(model).(EmbeddingModel).Embed(context.Background(), EmbeddingRequest{Input: []string{"hello"}})
	checkNoError(t, err)
	checkLen(t, embeddings.Vectors, 1)
	checkEqual(t, 1, embeddings.Usage.TotalTokens)

	reranked, err := any(model).(RerankModel).Rerank(context.Background(), RerankRequest{Query: "core", Documents: []string{"doc"}})
	checkNoError(t, err)
	checkLen(t, reranked.Results, 1)
	checkEqual(t, "doc", reranked.Results[0].Text)
}

func TestContracts_ReasoningParser_Good(t *testing.T) {
	model := &contractModel{}

	reasoning, err := any(model).(ReasoningParser).ParseReasoning(nil, "answer")
	checkNoError(t, err)
	checkEqual(t, "answer", reasoning.VisibleText)
	checkLen(t, reasoning.Reasoning, 1)

	tools, err := any(model).(ToolParser).ParseTools(nil, "call")
	checkNoError(t, err)
	checkLen(t, tools.Calls, 1)
	checkEqual(t, "search", tools.Calls[0].Name)
}

func TestContracts_ModelPackInspector_Good(t *testing.T) {
	inspection, err := any(&contractModel{}).(ModelPackInspector).InspectModelPack(context.Background(), "/models/qwen")

	checkNoError(t, err)
	checkTrue(t, inspection.Supported)
	checkEqual(t, "qwen3", inspection.Model.Architecture)
}

func TestContracts_AgentMemorySession_Good(t *testing.T) {
	model := &contractModel{}
	session := any(model).(AgentMemorySession)

	wake, err := session.WakeState(context.Background(), AgentMemoryWakeRequest{EntryURI: "mlx://memory/chapter-1"})
	checkNoError(t, err)
	checkEqual(t, 8, wake.PrefixTokens)
	checkEqual(t, "mlx://memory/chapter-1", wake.Entry.URI)

	sleep, err := session.SleepState(context.Background(), AgentMemorySleepRequest{EntryURI: "mlx://memory/chapter-1/after", Title: "after"})
	checkNoError(t, err)
	checkEqual(t, 9, sleep.TokenCount)
	checkEqual(t, "after", sleep.Entry.Title)

	forked, forkWake, err := any(model).(AgentMemoryForker).ForkState(context.Background(), AgentMemoryWakeRequest{EntryURI: "mlx://memory/chapter-1"})
	checkNoError(t, err)
	checkNotNil(t, forked)
	checkEqual(t, 8, forkWake.PrefixTokens)
}
