---
title: Types
description: Token, Message, config structs, functional options, and all supporting types.
---

# Types

The types below are defined in the root `inference` package (`dappco.re/go/inference`) — the shared contract. The repository has many other packages (`engine/metal`, `engine/hip`, `serving/`, `model/`, `kv/`, `decode/`, `train/`, `eval/`, `model/state`, `cmd/lem`), but the contract types every engine and consumer share live at the root.

Fallible operations in this package return `core.Result` (from `dappco.re/go`), not `(T, error)` — see [Interfaces](interfaces.md).

## Core value types

### Token

```go
type Token struct {
    ID   int32
    Text string
}
```

The atomic unit of streaming output. `ID` is the vocabulary index; `Text` is the decoded string. Backends yield these through `iter.Seq[Token]` from `Generate` and `Chat`.

### Message

```go
type Message struct {
    Role    string   `json:"role"`             // "system", "user", "assistant"
    Content string   `json:"content"`
    Images  [][]byte `json:"images,omitempty"` // encoded image bytes (PNG/JPEG) for vision turns
}
```

A single turn in a multi-turn conversation. JSON tags are present for serialisation through MCP tool payloads and API responses. Pass a slice of these to `TextModel.Chat()`. `Images` is populated by the compat handlers from multimodal content parts; only engines implementing `VisionModel` serve image turns.

---

## Result types

### ClassifyResult

```go
type ClassifyResult struct {
    Token  Token     // Sampled/greedy token at last prompt position
    Logits []float32 // Raw vocab-sized logits (only when WithLogits is set)
}
```

Output from a single prefill-only forward pass via `Classify`. The `Logits` slice is only populated when `WithLogits()` is passed — it is `nil` by default to avoid allocating vocab-sized float arrays (e.g. 256,128 floats for Gemma3) on every classification call.

### BatchResult

```go
type BatchResult struct {
    Tokens []Token // All generated tokens for this prompt
    Err    error   // Per-prompt error (context cancel, OOM, etc.)
}
```

Per-prompt result from `BatchGenerate`. `Err` captures per-prompt failures rather than aborting the entire batch, so successful prompts still return their tokens.

---

## Metrics and metadata

### GenerateMetrics

```go
type GenerateMetrics struct {
    PromptTokens        int           // Input tokens (sum across batch for batch ops)
    GeneratedTokens     int           // Output tokens generated
    PrefillDuration     time.Duration // Time to process the prompt(s)
    DecodeDuration      time.Duration // Time for autoregressive decoding
    TotalDuration       time.Duration // Wall-clock time for the full operation
    PrefillTokensPerSec float64       // PromptTokens / PrefillDuration
    DecodeTokensPerSec  float64       // GeneratedTokens / DecodeDuration
    PeakMemoryBytes     uint64        // Peak GPU memory during this operation
    ActiveMemoryBytes   uint64        // Active GPU memory after operation
    ThinkingBudgetForced bool         // ThinkingBudget forced the thought-channel close
}
```

Performance data for the most recent inference operation. Retrieved via `TextModel.Metrics()` after an iterator is exhausted or a batch call returns.

`PeakMemoryBytes` and `ActiveMemoryBytes` are GPU-specific. CPU-only backends or HTTP backends may leave them at zero.

### ModelInfo

```go
type ModelInfo struct {
    Architecture string // e.g. "gemma3", "qwen3", "llama"
    VocabSize    int    // Vocabulary size
    NumLayers    int    // Number of transformer layers
    HiddenSize   int    // Hidden dimension
    QuantBits    int    // Quantisation bits (0 = unquantised, 4 = 4-bit, 8 = 8-bit)
    QuantGroup   int    // Quantisation group size (0 if unquantised)
}
```

Static metadata about a loaded model. `QuantBits` is zero for unquantised (FP16/BF16) models.

---

## Attention types

### AttentionSnapshot

```go
type AttentionSnapshot struct {
    NumLayers     int           `json:"num_layers"`
    NumHeads      int           `json:"num_heads"`       // num_kv_heads (may differ from query heads in GQA)
    SeqLen        int           `json:"seq_len"`         // number of tokens in the prompt
    HeadDim       int           `json:"head_dim"`
    NumQueryHeads int           `json:"num_query_heads"` // num_attention_heads (0 = Q not available)
    Keys          [][][]float32 `json:"keys"`            // [layer][head] -> flat float32 of len seq_len*head_dim
    Queries       [][][]float32 `json:"queries"`         // [layer][head] -> flat float32 (nil if K-only)
    Architecture  string        `json:"architecture"`
}
```

Q and/or K vectors extracted from the KV cache after a prefill pass. The `Keys` tensor is indexed `[layer][head][position*head_dim]` — each head's K vectors are flattened into a single slice of length `SeqLen * HeadDim`.

For GQA models (e.g. Gemma3 where `num_kv_heads < num_query_heads`), `NumHeads` reflects the KV head count. `NumQueryHeads` is non-zero only when query vectors are available.

**Method:**

```go
func (s *AttentionSnapshot) HasQueries() bool
```

Reports whether this snapshot contains query vectors (i.e. `Queries` is non-nil and non-empty).

---

## Training types

### LoRAConfig

```go
type LoRAConfig struct {
    Rank       int      // Decomposition rank (default 8)
    Alpha      float32  // Scaling factor (default 16)
    TargetKeys []string // Projection layer suffixes to target (default: q_proj, v_proj)
    BFloat16   bool     // Use BFloat16 for adapter weights (mixed precision)
}
```

Specifies LoRA adapter parameters for fine-tuning. Use `DefaultLoRAConfig()` to get standard defaults:

```go
func DefaultLoRAConfig() LoRAConfig {
    return LoRAConfig{
        Rank:       8,
        Alpha:      16,
        TargetKeys: []string{"q_proj", "v_proj"},
    }
}
```

`TargetKeys` lists the projection layer suffixes that receive LoRA adapters. The default targets query and value projections. Adding `"k_proj"` or `"o_proj"` increases trainable parameter count but may improve fine-tuning quality for some tasks.

---

## Generation options

Generation is configured via functional options applied to `GenerateConfig`.

### GenerateConfig

```go
type GenerateConfig struct {
    MaxTokens      int
    Temperature    float32
    TopK           int
    TopP           float32
    MinP           float32
    Seed           uint64
    SeedSet        bool
    StopTokens     []int32
    SuppressTokens []int32
    MinTokensBeforeStop int
    RepeatPenalty  float32
    ReturnLogits   bool
    EnableThinking *bool          // nil = model default; &true = on; &false = off
    ThinkingBudget int            // cap tokens in the thought channel; 0 = unlimited
    Thinking       ThinkingConfig // resolved thought-channel processing policy
    // Engine-neutral trace, cache-hygiene, and probe knobs — an engine
    // without the facility ignores them:
    TraceTokenPhases             bool     // per-token coarse phase timing to the trace log
    TraceTokenText               bool     // include decoded token text in the trace (debug)
    GenerationClearCache         bool     // drop device caches between generations
    GenerationClearCacheInterval int      // clear every N tokens; 0 = never
    ProbeSink                    probe.Sink // research-telemetry sink; nil = probing off
}
```

Defaults (from `DefaultGenerateConfig()`):

| Field | Default | Notes |
|-------|---------|-------|
| `Temperature` | 0.0 | Greedy decoding |
| `RepeatPenalty` | 1.0 | No repetition penalty |
| `MaxTokens` | 0 (**not defaulted**) | Absent, the engine resolves it to the model's context at generation time — a fixed default would truncate every generation at a guess |
| All others | zero value | Disabled |

### GenerateOption functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `WithMaxTokens` | `WithMaxTokens(n int) GenerateOption` | Cap output length |
| `WithTemperature` | `WithTemperature(t float32) GenerateOption` | Sampling temperature (0 = greedy) |
| `WithTopK` | `WithTopK(k int) GenerateOption` | Top-k sampling (0 = disabled) |
| `WithTopP` | `WithTopP(p float32) GenerateOption` | Nucleus sampling threshold (0 = disabled) |
| `WithMinP` | `WithMinP(p float32) GenerateOption` | Minimum-probability sampling relative to the top token |
| `WithSeed` | `WithSeed(seed uint64) GenerateOption` | Reproducible stochastic sampling for this request |
| `WithStopTokens` | `WithStopTokens(ids ...int32) GenerateOption` | Token IDs that stop generation |
| `WithSuppressTokens` | `WithSuppressTokens(ids ...int32) GenerateOption` | Token IDs masked out of the distribution |
| `WithMinTokensBeforeStop` | `WithMinTokensBeforeStop(n int) GenerateOption` | Suppress stop tokens until n tokens emitted |
| `WithRepeatPenalty` | `WithRepeatPenalty(p float32) GenerateOption` | Repetition penalty (1.0 = none) |
| `WithLogits` | `WithLogits() GenerateOption` | Return raw logits in `ClassifyResult` |
| `WithEnableThinking` | `WithEnableThinking(v *bool) GenerateOption` | Reasoning toggle for thinking-capable models |
| `WithThinkingBudget` | `WithThinkingBudget(tokens int) GenerateOption` | Cap thought-channel tokens; 0 = unlimited |
| `WithThinking` | `WithThinking(cfg ThinkingConfig) GenerateOption` | Resolved thought-channel processing policy |

### ApplyGenerateOpts

```go
func ApplyGenerateOpts(opts []GenerateOption) GenerateConfig
```

Builds a `GenerateConfig` from options, starting from `DefaultGenerateConfig()`. Called by backends at the start of each inference operation. Options are applied in order; the last write wins for scalar fields.

---

## Load options

Model loading is configured via functional options applied to `LoadConfig`.

### LoadConfig

```go
type LoadConfig struct {
    Backend       string // "metal", "rocm", "llama_cpp" (empty = auto-detect)
    ContextLen    int    // Context window size (0 = model default)
    GPULayers     int    // Layers to offload to GPU (-1 = all, 0 = none)
    ParallelSlots int    // Concurrent inference slots (0 = server default)
    AdapterPath   string // Path to LoRA adapter directory (empty = no adapter)
}
```

Default `GPULayers` is `-1` (full GPU offload). Positive values specify a layer count for partial offload (relevant to ROCm and llama.cpp; Metal always does full offload).

### LoadOption functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `WithBackend` | `WithBackend(name string) LoadOption` | Select specific backend by name |
| `WithContextLen` | `WithContextLen(n int) LoadOption` | Context window size |
| `WithGPULayers` | `WithGPULayers(n int) LoadOption` | GPU layer offload count (-1 = all) |
| `WithParallelSlots` | `WithParallelSlots(n int) LoadOption` | Concurrent inference slots |
| `WithAdapterPath` | `WithAdapterPath(path string) LoadOption` | LoRA adapter directory |

### ApplyLoadOpts

```go
func ApplyLoadOpts(opts []LoadOption) LoadConfig
```

Builds a `LoadConfig` from options. Default `GPULayers` is `-1`. Called by `LoadModel()` and by backends in their `LoadModel` implementations.

---

## Model discovery

### DiscoveredModel

```go
type DiscoveredModel struct {
    Path        string // Absolute path to the model directory or GGUF file
    ModelType   string // Architecture from config.json / GGUF metadata (e.g. "gemma3", "qwen3", "llama")
    QuantBits   int    // Quantisation bits (0 if unquantised or unknown)
    QuantGroup  int    // Quantisation group size
    QuantType   string // Quantisation type, when known (e.g. q4_k_m, q8_0)
    QuantFamily string // Quantisation family, when known (e.g. q4, q8)
    NumFiles    int    // Number of weight files
    Format      string // "safetensors" or "gguf" when known
}
```

Returned by `Discover()`. `Path` is always absolute. `ModelType` is read from `config.json`'s `model_type` field (or GGUF metadata). `Discover` walks the tree **recursively**, so a nested models directory yields every model beneath it.

---

## Device information

### DeviceInfo

```go
type DeviceInfo struct {
    Name                         string // e.g. "Apple M3 Ultra"
    Architecture                 string // e.g. "applegpu_g15d"
    MemorySize                   uint64 // total device memory in bytes
    MaxBufferLength              uint64 // largest single allocation the device allows
    MaxRecommendedWorkingSetSize uint64 // device-recommended working-set ceiling
}
```

Describes the accelerator a backend runs on — engine-neutral (Metal reports the Apple GPU; hip/cuda report theirs). Zero-valued fields mean the backend could not determine them. Retrieved via `inference.BackendDeviceInfo(name)`, which returns `false` when the backend is unregistered or does not implement `DeviceInfoProvider`.
