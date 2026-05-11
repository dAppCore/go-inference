// SPDX-Licence-Identifier: EUPL-1.2

package inference

import (
	"context"

	"dappco.re/go/inference/state"
)

// RequestHandle identifies an in-flight generation request without requiring
// a concrete scheduler implementation.
type RequestHandle struct {
	ID     string            `json:"id,omitempty"`
	Model  ModelIdentity     `json:"model,omitempty"`
	Labels map[string]string `json:"labels,omitempty"`
}

// RequestCancelResult records the outcome of a cancellation request.
type RequestCancelResult struct {
	ID        string            `json:"id,omitempty"`
	Cancelled bool              `json:"cancelled,omitempty"`
	Reason    string            `json:"reason,omitempty"`
	Labels    map[string]string `json:"labels,omitempty"`
}

// ScheduledRequest is the backend-neutral input to an optional request
// scheduler. Exactly one of Prompt or Messages is normally populated.
type ScheduledRequest struct {
	ID       string            `json:"id,omitempty"`
	Model    string            `json:"model,omitempty"`
	Prompt   string            `json:"prompt,omitempty"`
	Messages []Message         `json:"messages,omitempty"`
	Sampler  SamplerConfig     `json:"sampler,omitempty"`
	Labels   map[string]string `json:"labels,omitempty"`
}

// ScheduledToken carries a streamed token plus request-local telemetry.
type ScheduledToken struct {
	RequestID string            `json:"request_id,omitempty"`
	Token     Token             `json:"token,omitempty"`
	Metrics   GenerateMetrics   `json:"metrics,omitempty"`
	Labels    map[string]string `json:"labels,omitempty"`
}

// SchedulerModel exposes queue-aware generation without forcing every backend
// to implement server policy.
type SchedulerModel interface {
	Schedule(ctx context.Context, req ScheduledRequest) (RequestHandle, <-chan ScheduledToken, error)
}

// CancellableModel exposes request cancellation by stable request ID.
type CancellableModel interface {
	CancelRequest(ctx context.Context, id string) (RequestCancelResult, error)
}

// CacheBlockRef is a portable reference to a prompt/KV cache block.
type CacheBlockRef struct {
	ID            string            `json:"id,omitempty"`
	Kind          string            `json:"kind,omitempty"`
	ModelHash     string            `json:"model_hash,omitempty"`
	AdapterHash   string            `json:"adapter_hash,omitempty"`
	TokenizerHash string            `json:"tokenizer_hash,omitempty"`
	TokenStart    int               `json:"token_start,omitempty"`
	TokenCount    int               `json:"token_count,omitempty"`
	SizeBytes     uint64            `json:"size_bytes,omitempty"`
	Encoding      string            `json:"encoding,omitempty"`
	Labels        map[string]string `json:"labels,omitempty"`
}

// CacheStats records request-time cache health.
type CacheStats struct {
	Blocks        int               `json:"blocks,omitempty"`
	MemoryBytes   uint64            `json:"memory_bytes,omitempty"`
	DiskBytes     uint64            `json:"disk_bytes,omitempty"`
	Hits          uint64            `json:"hits,omitempty"`
	Misses        uint64            `json:"misses,omitempty"`
	Evictions     uint64            `json:"evictions,omitempty"`
	HitRate       float64           `json:"hit_rate,omitempty"`
	RestoreMillis float64           `json:"restore_millis,omitempty"`
	CacheMode     string            `json:"cache_mode,omitempty"`
	Labels        map[string]string `json:"labels,omitempty"`
}

// CacheWarmRequest asks a runtime to prepare cache blocks for a prompt.
type CacheWarmRequest struct {
	Model   ModelIdentity     `json:"model,omitempty"`
	Adapter AdapterIdentity   `json:"adapter,omitempty"`
	Prompt  string            `json:"prompt,omitempty"`
	Tokens  []int32           `json:"tokens,omitempty"`
	Mode    string            `json:"mode,omitempty"`
	Labels  map[string]string `json:"labels,omitempty"`
}

// CacheWarmResult reports which cache blocks are available after warming.
type CacheWarmResult struct {
	Blocks []CacheBlockRef   `json:"blocks,omitempty"`
	Stats  CacheStats        `json:"stats,omitempty"`
	Labels map[string]string `json:"labels,omitempty"`
}

// CacheService exposes cache inspection and warm/clear controls.
type CacheService interface {
	CacheStats(ctx context.Context) (CacheStats, error)
	WarmCache(ctx context.Context, req CacheWarmRequest) (CacheWarmResult, error)
	ClearCache(ctx context.Context, labels map[string]string) (CacheStats, error)
}

// EmbeddingRequest is a backend-neutral embedding request.
type EmbeddingRequest struct {
	Model     string            `json:"model,omitempty"`
	Input     []string          `json:"input,omitempty"`
	Normalize bool              `json:"normalize,omitempty"`
	Labels    map[string]string `json:"labels,omitempty"`
}

// EmbeddingUsage records token accounting for embedding calls.
type EmbeddingUsage struct {
	PromptTokens int `json:"prompt_tokens,omitempty"`
	TotalTokens  int `json:"total_tokens,omitempty"`
}

// EmbeddingResult is the portable output of an embedding model.
type EmbeddingResult struct {
	Model   ModelIdentity     `json:"model,omitempty"`
	Vectors [][]float32       `json:"vectors,omitempty"`
	Usage   EmbeddingUsage    `json:"usage,omitempty"`
	Labels  map[string]string `json:"labels,omitempty"`
}

// EmbeddingModel marks models that can produce vector embeddings.
type EmbeddingModel interface {
	Embed(ctx context.Context, req EmbeddingRequest) (*EmbeddingResult, error)
}

// RerankRequest asks a model to score documents against a query.
type RerankRequest struct {
	Model     string            `json:"model,omitempty"`
	Query     string            `json:"query,omitempty"`
	Documents []string          `json:"documents,omitempty"`
	TopN      int               `json:"top_n,omitempty"`
	Labels    map[string]string `json:"labels,omitempty"`
}

// RerankScore records one scored document.
type RerankScore struct {
	Index  int               `json:"index,omitempty"`
	Score  float64           `json:"score,omitempty"`
	Text   string            `json:"text,omitempty"`
	Labels map[string]string `json:"labels,omitempty"`
}

// RerankResult is the portable output of a rerank request.
type RerankResult struct {
	Model   ModelIdentity     `json:"model,omitempty"`
	Results []RerankScore     `json:"results,omitempty"`
	Labels  map[string]string `json:"labels,omitempty"`
}

// RerankModel marks models that can score candidate documents.
type RerankModel interface {
	Rerank(ctx context.Context, req RerankRequest) (*RerankResult, error)
}

// ReasoningSegment is a captured reasoning/thinking span.
type ReasoningSegment struct {
	Kind       string            `json:"kind,omitempty"`
	Text       string            `json:"text,omitempty"`
	StartToken int               `json:"start_token,omitempty"`
	EndToken   int               `json:"end_token,omitempty"`
	Labels     map[string]string `json:"labels,omitempty"`
}

// ReasoningParseResult separates visible model output from reasoning text.
type ReasoningParseResult struct {
	VisibleText string             `json:"visible_text,omitempty"`
	Reasoning   []ReasoningSegment `json:"reasoning,omitempty"`
	Labels      map[string]string  `json:"labels,omitempty"`
}

// ReasoningParser parses model-family-specific thinking channels.
type ReasoningParser interface {
	ParseReasoning(tokens []Token, text string) (ReasoningParseResult, error)
}

// ToolCall records a parsed model-emitted tool call.
type ToolCall struct {
	ID            string            `json:"id,omitempty"`
	Name          string            `json:"name,omitempty"`
	Type          string            `json:"type,omitempty"`
	ArgumentsJSON string            `json:"arguments_json,omitempty"`
	Labels        map[string]string `json:"labels,omitempty"`
}

// ToolParseResult separates user-visible text from tool calls.
type ToolParseResult struct {
	VisibleText string            `json:"visible_text,omitempty"`
	Calls       []ToolCall        `json:"calls,omitempty"`
	Labels      map[string]string `json:"labels,omitempty"`
}

// ToolParser parses model-family-specific tool-call formats.
type ToolParser interface {
	ParseTools(tokens []Token, text string) (ToolParseResult, error)
}

// ModelPackInspection records portable model-pack validation output.
type ModelPackInspection struct {
	Path         string            `json:"path,omitempty"`
	Format       string            `json:"format,omitempty"`
	Model        ModelIdentity     `json:"model,omitempty"`
	Tokenizer    TokenizerIdentity `json:"tokenizer,omitempty"`
	Supported    bool              `json:"supported,omitempty"`
	Capabilities []Capability      `json:"capabilities,omitempty"`
	Notes        []string          `json:"notes,omitempty"`
	Labels       map[string]string `json:"labels,omitempty"`
}

// ModelPackInspector inspects local model packs without loading tensors.
type ModelPackInspector interface {
	InspectModelPack(ctx context.Context, path string) (*ModelPackInspection, error)
}

type AgentMemoryRef = state.Ref
type AgentMemoryWakeRequest = state.WakeRequest
type AgentMemoryWakeResult = state.WakeResult
type AgentMemorySleepRequest = state.SleepRequest
type AgentMemorySleepResult = state.SleepResult
type AgentMemorySession = state.Session
type AgentMemoryForker = state.Forker
