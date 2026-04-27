# Architecture — go-inference

## Purpose

`go-inference` is the shared interface contract for text generation backends in the Core Go ecosystem. It defines the types that GPU-specific backends implement and consumers depend on, without itself importing any backend or consumer code.

Module path: `dappco.re/go/inference`

## Design Philosophy

### Zero Dependencies

The package imports only the Go standard library (`context`, `fmt`, `iter`, `sync`, `time`, `encoding/json`, `os`, `path/filepath`). The sole exception is `testify` in the test tree.

This is a deliberate constraint. The package sits at the base of a dependency graph where:

- `go-mlx` pulls in CGO bindings against Apple's Metal framework
- `go-rocm` spawns a `llama-server` subprocess with AMD ROCm libraries
- `go-ml` links DuckDB and Parquet

None of those concerns belong in the interface layer. A backend can import `go-inference`; `go-inference` cannot import a backend. A consumer can import `go-inference`; `go-inference` cannot import a consumer.

### Minimal Interface Surface

New methods are only added when two or more existing consumers need them. The interfaces are deliberately narrow. Broader capability is achieved through additional interfaces (`BatchModel`, `StatsModel`) that embed `TextModel`, not through extending `TextModel` itself.

### Platform Agnostic

No build tags, no `//go:build` constraints, no `CGO_ENABLED` requirements appear in this package. It compiles cleanly on macOS, Linux, and Windows regardless of GPU availability.

## Ecosystem Position

```
go-inference (this package)  ← defines TextModel, Backend, Token, Message
      |
      |──────── implemented by ──────────────────────────────
      |                                                      |
 go-mlx                                                 go-rocm
 (darwin/arm64, Metal GPU)                     (linux/amd64, AMD ROCm)
      |                                                      |
      └───────────────── consumed by ────────────────────────┘
                              |
                           go-ml
                    (scoring engine, llama.cpp HTTP)
                              |
                           go-ai
                     (MCP hub, 30+ tools)
                              |
                          go-i18n
                   (domain classification via Gemma3-1B)
```

`go-ml` also provides a reverse adapter (`backend_http_textmodel.go`) that wraps an HTTP llama.cpp server as a `TextModel`, giving a third backend path without Metal or ROCm.

## Core Types

### Token

```go
type Token struct {
    ID   int32
    Text string
}
```

The atomic unit of streaming output. `ID` is the vocabulary index; `Text` is the decoded string. Backends yield these through `iter.Seq[Token]`.

### Message

```go
type Message struct {
    Role    string `json:"role"`    // "system", "user", "assistant"
    Content string `json:"content"`
}
```

A single turn in a multi-turn conversation. JSON tags are present for serialisation through MCP tool payloads and API responses.

### ClassifyResult

```go
type ClassifyResult struct {
    Token  Token
    Logits []float32
}
```

Output from a single prefill-only forward pass. `Logits` is populated only when `WithLogits()` is set; it is empty by default to avoid allocating vocab-sized float arrays for every classification call.

### BatchResult

```go
type BatchResult struct {
    Tokens []Token
    Err    error
}
```

Per-prompt result from `BatchGenerate`. `Err` carries per-prompt failures (context cancellation, OOM) rather than aborting the entire batch.

### GenerateMetrics

```go
type GenerateMetrics struct {
    PromptTokens        int
    GeneratedTokens     int
    PrefillDuration     time.Duration
    DecodeDuration      time.Duration
    TotalDuration       time.Duration
    PrefillTokensPerSec float64
    DecodeTokensPerSec  float64
    PeakMemoryBytes     uint64
    ActiveMemoryBytes   uint64
}
```

Performance data for the most recent inference operation. Retrieved via `TextModel.Metrics()` after an iterator is exhausted or a batch call returns. `PeakMemoryBytes` and `ActiveMemoryBytes` are GPU-specific; CPU-only backends may leave them at zero.

### ModelInfo

```go
type ModelInfo struct {
    Architecture string
    VocabSize    int
    NumLayers    int
    HiddenSize   int
    QuantBits    int
    QuantGroup   int
}
```

Static metadata about a loaded model. `QuantBits` is zero for unquantised (FP16/BF16) models.

### AttentionSnapshot

```go
type AttentionSnapshot struct {
    NumLayers    int
    NumHeads     int           // num_kv_heads (may differ from query heads in GQA)
    SeqLen       int           // number of tokens in the prompt
    HeadDim      int
    Keys         [][][]float32 // [layer][head] → flat float32 of len seq_len*head_dim
    Architecture string
}
```

Post-RoPE K vectors extracted from the KV cache after a single prefill pass. The `Keys` tensor is indexed `[layer][head][position*head_dim]` — each head's K vectors are flattened into a single slice of length `SeqLen * HeadDim`.

This type is consumed by LEM's Q/K Bone Orientation analysis engine, which computes coherence, cross-layer alignment, head entropy, phase-lock, and joint collapse metrics from the raw K tensors. The analysis is pure Go CPU math — no GPU dependencies.

For GQA models (e.g. Gemma3 where `num_kv_heads < num_query_heads`), `NumHeads` reflects the KV head count. Single-head layers use position-wise differentiation rather than pairwise head comparison.

## Optional Interfaces

### AttentionInspector

```go
type AttentionInspector interface {
    InspectAttention(ctx context.Context, prompt string, opts ...GenerateOption) (*AttentionSnapshot, error)
}
```

Backends may implement `AttentionInspector` to expose attention-level data for Q/K Bone Orientation analysis. This is an optional interface — consumers discover it via type assertion:

```go
if inspector, ok := model.(AttentionInspector); ok {
    snap, err := inspector.InspectAttention(ctx, prompt)
    // analyse snap.Keys
}
```

Following rule 3 of the stability contract: new capability is expressed as separate interfaces, not by extending `TextModel`. Backends that don't support attention inspection (HTTP, llama.cpp subprocess) are unaffected.

**Implementations:**
- `go-mlx` — Extracts post-RoPE K vectors from Metal KV cache after prefill (native GPU memory read)
- `go-ml` — `InferenceAdapter.InspectAttention()` delegates via type assertion to the underlying `TextModel`

## TextModel Interface

```go
type TextModel interface {
    Generate(ctx context.Context, prompt string, opts ...GenerateOption) iter.Seq[Token]
    Chat(ctx context.Context, messages []Message, opts ...GenerateOption) iter.Seq[Token]
    Classify(ctx context.Context, prompts []string, opts ...GenerateOption) ([]ClassifyResult, error)
    BatchGenerate(ctx context.Context, prompts []string, opts ...GenerateOption) ([]BatchResult, error)
    ModelType() string
    Info() ModelInfo
    Metrics() GenerateMetrics
    Err() error
    Close() error
}
```

Key design decisions:

**`context.Context` on streaming methods** — Required for HTTP handler cancellation, request timeouts, and graceful shutdown. The context is checked by backends at token boundaries.

**`iter.Seq[Token]` return type** — Go 1.23+ range-over-function iterators. The caller ranges over the sequence; the backend controls token production. The iterator pattern avoids channel overhead and lets the backend use direct memory access to GPU buffers.

**`Err() error`** — `iter.Seq` cannot carry errors alongside values. Following the `database/sql` `Row.Err()` pattern, the error from the most recent `Generate` or `Chat` call is stored internally and retrieved with `Err()` after the iterator finishes. End-of-sequence (EOS token) sets no error; context cancellation and OOM both set one.

**`Chat()` on the model** — Chat templates differ across architectures (Gemma3, Qwen3, Llama3 all use distinct formats). Placing template application in the backend means consumers receive already-formatted input regardless of model family. If templates lived in consumers, every consumer would need to duplicate model-specific formatting logic.

**`Classify()` and `BatchGenerate()`** — Two distinct batch operations with different performance characteristics. `Classify` is prefill-only (single forward pass, no autoregressive loop); it is the fast path for domain labelling in `go-i18n`. `BatchGenerate` runs full autoregressive decoding across multiple prompts in parallel.

**`Info()` and `Metrics()`** — Separated from `Generate`/`Chat` because they serve different call sites. `Info()` is called once after load; `Metrics()` is called after each inference operation for performance monitoring.

## Backend Interface

```go
type Backend interface {
    Name() string
    LoadModel(path string, opts ...LoadOption) (TextModel, error)
    Available() bool
}
```

**`Name()`** — Returns the registry key: `"metal"`, `"rocm"`, or `"llama_cpp"`. This is the string passed to `WithBackend()` by consumers.

**`LoadModel()`** — Accepts a filesystem path to a model directory (containing `config.json` and `.safetensors` weight files) and returns a ready-to-use `TextModel`. The model directory format follows the HuggingFace safetensors layout.

**`Available()`** — Reports whether the backend can run on the current hardware. This allows a backend to be registered unconditionally (e.g. in a shared binary) while still reporting false on platforms where its GPU runtime is absent. `Default()` skips unavailable backends.

## Backend Registry

The registry is a package-level `map[string]Backend` protected by a `sync.RWMutex`. It supports concurrent reads and exclusive writes.

```go
var (
    backendsMu sync.RWMutex
    backends   = map[string]Backend{}
)
```

**Registration** — Backends call `inference.Register(b Backend)` from their `init()` function. The `init()` is guarded by a build tag so it only compiles on the target platform:

```go
// In go-mlx: register_metal.go
//go:build darwin && arm64

func init() { inference.Register(metalBackend{}) }
```

```go
// In go-rocm: register_rocm.go
//go:build linux && amd64

func init() { inference.Register(&rocmBackend{}) }
```

Registering a name that already exists silently overwrites the previous entry. This allows test code to replace backends without a separate de-registration step.

**Discovery** — `Get(name)` performs a direct map lookup. `List()` returns all registered names (order undefined). `Default()` walks a priority list:

```go
for _, name := range []string{"metal", "rocm", "llama_cpp"} {
    if b, ok := backends[name]; ok && b.Available() {
        return b, nil
    }
}
// Fall back to any registered available backend.
```

The priority order encodes hardware preference: Metal (Apple Silicon) delivers the highest throughput for on-device inference on macOS; ROCm is preferred over llama.cpp's HTTP server on Linux because it provides direct GPU memory access without HTTP overhead.

**`LoadModel()` routing** — The top-level `LoadModel()` function is the primary consumer entry point:

```go
func LoadModel(path string, opts ...LoadOption) (TextModel, error) {
    cfg := ApplyLoadOpts(opts)
    if cfg.Backend != "" {
        b, ok := Get(cfg.Backend)
        // ... validate and use explicit backend
    }
    b, err := Default()
    // ... use auto-selected backend
}
```

Passing `WithBackend("rocm")` bypasses `Default()` entirely. This is the mechanism used in cross-platform binaries or tests that need to pin a specific backend.

## Functional Options

Generation and loading are configured through two independent option types, both following the standard Go functional options pattern.

### GenerateConfig and GenerateOption

```go
type GenerateConfig struct {
    MaxTokens     int
    Temperature   float32
    TopK          int
    TopP          float32
    StopTokens    []int32
    RepeatPenalty float32
    ReturnLogits  bool
}
```

Defaults (from `DefaultGenerateConfig()`): `MaxTokens=256`, `Temperature=0.0` (greedy), `RepeatPenalty=1.0` (no penalty), all others zero/disabled.

`ApplyGenerateOpts(opts []GenerateOption) GenerateConfig` is called by backends at the start of each inference operation. Options are applied in order; the last write wins for scalar fields.

`WithLogits()` is a flag rather than a value option because logit arrays are vocab-sized (256,128 floats for Gemma3) and should only be allocated when explicitly requested.

### LoadConfig and LoadOption

```go
type LoadConfig struct {
    Backend       string
    ContextLen    int
    GPULayers     int
    ParallelSlots int
}
```

Default `GPULayers` is `-1`, meaning full GPU offload. `0` forces CPU-only inference. Positive values specify a layer count for partial offload (relevant to ROCm and llama.cpp; Metal always does full offload).

`ParallelSlots` controls the number of concurrent inference slots the backend allocates. Higher values allow parallel `Generate`/`Chat` calls at the cost of increased VRAM usage. `0` defers to the backend's own default.

## Model Discovery

`Discover(baseDir string) ([]DiscoveredModel, error)` scans one level of a directory tree for model directories. A valid model directory must contain both `config.json` and at least one `.safetensors` file.

```go
type DiscoveredModel struct {
    Path      string
    ModelType string
    QuantBits int
    QuantGroup int
    NumFiles   int
}
```

`Path` is always an absolute filesystem path. `ModelType` is read from `config.json`'s `model_type` field. Invalid JSON in `config.json` is silently tolerated — the directory is included with an empty `ModelType`.

`Discover` also checks whether `baseDir` itself is a model directory and, if so, prepends it to the result so that direct-path usage (`Discover("/models/gemma3-1b")`) works without nesting.

## Stability Contract

This package is the shared contract. Every method signature change here requires coordinated updates to go-mlx, go-rocm, and go-ml. The following rules govern interface evolution:

1. Existing method signatures are never changed. Rename or remove nothing from `TextModel` or `Backend`.
2. New methods are only added when two or more consumers have a concrete need.
3. New capability is expressed as separate interfaces (`BatchModel`, `StatsModel`) that embed `TextModel`, allowing consumers to opt in with a type assertion.
4. `GenerateConfig` and `LoadConfig` may gain new fields with zero-value defaults; this is backwards compatible.
