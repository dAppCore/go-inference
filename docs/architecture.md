# Architecture — go-inference

## Purpose

`go-inference` is the sovereign local-inference repository for the Core Go ecosystem. It is the single home for everything needed to run a local model: the shared contract (`TextModel`, `Backend`, and supporting types), the GPU compute engines that implement it, the serving layer that exposes them over HTTP, and the `lem` binary that ties it together.

Historically this was a contract-only package that GPU backends in separate repositories (`go-mlx`, `go-rocm`) implemented. Those repositories are retired: their engines have been migrated in-tree as `engine/metal` and `engine/hip`, and the `lem` binary now compiles from `go-inference` alone.

Module path: `dappco.re/go/inference` · Go 1.26 · Licence EUPL-1.2.

## Dependencies

The package is **not** stdlib-only. It consumes the Core externals and a handful of third-party libraries:

- `dappco.re/go` (core) — the `core.Result`, `core.E`, `core.Fs`, and process primitives used throughout.
- `dappco.re/go/api`, `dappco.re/go/cli`, `dappco.re/go/log`, `dappco.re/go/process` — Core service surface.
- `github.com/gin-gonic/gin`, `github.com/google/uuid`, `github.com/modelcontextprotocol/go-sdk` — serving + MCP.
- `github.com/marcboeker/go-duckdb/v2`, `github.com/parquet-go/parquet-go` — dataset/eval storage.

Errors are constructed with `core.E(...)` (never `fmt.Errorf`); fallible calls return `core.Result` rather than `(T, error)` (see below). Externals are wired through the `go.work` workspace and `external/<dep>` submodules — there are no `replace` directives.

## The core.Result contract

Fallible operations across this package return `core.Result`, not the Go `(T, error)` tuple. A `Result` carries `OK bool` and `Value any`; on failure `Value` holds the error.

```go
r := inference.LoadModel("/models/gemma-4-e2b-it-4bit")
if !r.OK {
    log.Fatal(r.Error())
}
m := r.Value.(inference.TextModel)
defer m.Close()
```

`Generate` and `Chat` still return `iter.Seq[Token]` (a range-over-function iterator cannot carry a Result inline); the trailing error is retrieved with `m.Err()`, which itself returns a `core.Result` that is OK on clean end-of-sequence.

## Repository layout

```
go/                         module root — package inference (the contract)
├── inference.go            TextModel, Backend, registry, LoadModel()
├── options.go              GenerateConfig, LoadConfig, functional options
├── training.go             TrainableModel, Adapter, LoRAConfig, LoadTrainable()
├── discover.go             Discover() filesystem/GGUF scan
├── device.go               DeviceInfo, DeviceInfoProvider, BackendDeviceInfo()
├── capability.go           CapabilityReport + algorithm profiles
├── identity.go             re-export aliases from model/state
├── engine/
│   ├── metal/              Apple-GPU engine (package native, darwin/arm64, NO cgo)
│   └── hip/                AMD ROCm engine (package hip, linux/amd64)
├── serving/                OpenAI/Anthropic/Ollama HTTP servers over the engine
├── model/                  arch definitions + model/state (identity, agent memory)
└── kv/ decode/ train/ eval/ agent/ safety/ welfare/   supporting libraries
cli/                    the lem binary (serve/generate/ssd/sft/tune/pack/ebook/quant/spec)
gui/                        desktop GUI (repo root, separate module surface)
external/<dep>/             Core external dependencies as workspace submodules
```

## Engines

Two GPU engines live in-tree and register themselves against the contract via `init()`. See [Backends](backends.md) for the full detail.

### engine/metal — Apple GPU (darwin/arm64)

Package clause `native`, path `engine/metal`. "Metal" names the Apple Metal API this engine drives; it is **not** go-mlx's cgo `pkg/metal` (deleted, never ported). Key facts, verified in `engine/metal/device.go`:

- **No cgo.** It dispatches the compiled MLX Metal kernels directly from Go through the `github.com/tmc/apple` objc bridge (purego `objc_msgSend`), gated by `//go:build darwin && arm64`.
- It loads the **same** compiled `mlx.metallib` the reference MLX build ships, located via `MLX_METALLIB_PATH`, plus an optional sibling `lthn_kernels.metallib` of go-inference's own fused kernels (absent ⇒ those ops fall back to composed primitives).
- The kernels are shared with MLX; the **innovation is the encode path.** Because decode and diffusion are fixed per-step command sequences, the engine records the sequence once into an **Indirect Command Buffer (ICB)** and replays it per token, bypassing the host-side re-encode that dominates MLX's decode. A MoE arch falls back to the re-encode path (the ICB cannot host the router's host-side top-k).

Registers as backend `"metal"` when imported: `_ "dappco.re/go/inference/engine/metal"`.

### engine/hip — AMD ROCm (linux/amd64)

Package `hip`, path `engine/hip`. Native-first ROCm/HIP engine (the old `llama-server` subprocess bridge survives only behind the `rocm_legacy_server` build tag and is not built by default). Three build-tag variants of the backend exist: the native runtime (`linux && amd64 && !rocm_legacy_server`), a portable stub that reports `Available() == false` (`!linux || !amd64`), and the legacy server path. GGUF loading works; safetensors model-pack loading is not yet available in the current quarantine landing. Registers as backend `"rocm"` when imported: `_ "dappco.re/go/inference/engine/hip"`.

## Serving and the lem binary

`serving/` exposes a loaded engine over OpenAI-, Anthropic-, and Ollama-compatible HTTP (the multiplexer is `serving/compat/mux.go`). `serving.NewMLXBackend` loads a model through the Metal backend (`inference.LoadModel(..., WithBackend("metal"))`) and wraps it as a `serving.Backend`. Note the serving layer also carries `HTTPBackend` (name `"http"`) and `LlamaBackend` (name `"llama"`) adapters that wrap an external llama.cpp HTTP server as a `TextModel` — these are serving-level adapters, not registered `inference.Backend`s.

`cli/` (its own module — `go/` is a pure library) is Lethean's sovereign inference binary. Its subcommands are thin flag-parsing wrappers over the `serving` and training libraries: `serve`, `generate`, `ssd`, `sft`, `tune`, `pack`, `ebook`, `quant`, `spec`. `main.go` blank-imports `engine/metal` and `model/builtin` to register the Apple backend and the built-in arches. Built with `-tags embed_metallib`, `lem` bakes both gzipped metallibs into the binary and extracts them to a content-addressed cache at start, setting `MLX_METALLIB_PATH` — so the shipped binary runs from any path with nothing external to resolve.

## Core types

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
    Role    string   `json:"role"`    // "system", "user", "assistant"
    Content string   `json:"content"`
    Images  [][]byte `json:"images,omitempty"` // encoded image bytes for vision turns
}
```

A single turn in a multi-turn conversation. `Images` carries PNG/JPEG bytes attached by the compat handlers from multimodal content parts; only engines implementing `VisionModel` serve image turns.

### ClassifyResult

```go
type ClassifyResult struct {
    Token  Token
    Logits []float32
}
```

Output from a single prefill-only forward pass. `Logits` is populated only when `WithLogits()` is set; it is `nil` by default to avoid allocating vocab-sized float arrays for every classification call.

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
    PromptTokens         int
    GeneratedTokens      int
    PrefillDuration      time.Duration
    DecodeDuration       time.Duration
    TotalDuration        time.Duration
    PrefillTokensPerSec  float64
    DecodeTokensPerSec   float64
    PeakMemoryBytes      uint64
    ActiveMemoryBytes    uint64
    ThinkingBudgetForced bool
}
```

Performance data for the most recent inference operation, retrieved via `TextModel.Metrics()`. `PeakMemoryBytes`/`ActiveMemoryBytes` are GPU-specific. `ThinkingBudgetForced` reports whether a reasoning model's thought channel was force-closed by `ThinkingBudget`.

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
    NumLayers     int           `json:"num_layers"`
    NumHeads      int           `json:"num_heads"`       // num_kv_heads (may differ from query heads in GQA)
    SeqLen        int           `json:"seq_len"`
    HeadDim       int           `json:"head_dim"`
    NumQueryHeads int           `json:"num_query_heads"` // 0 = Q not available
    Keys          [][][]float32 `json:"keys"`            // [layer][head] → flat float32 of len seq_len*head_dim
    Queries       [][][]float32 `json:"queries"`         // [layer][head] → flat float32 (nil if K-only)
    Architecture  string        `json:"architecture"`
}

func (s *AttentionSnapshot) HasQueries() bool
```

Post-RoPE Q and/or K vectors extracted from the KV cache after a single prefill pass. `Keys` is indexed `[layer][head][position*head_dim]`. For GQA models (`num_kv_heads < num_query_heads`), `NumHeads` reflects the KV head count; `NumQueryHeads` is non-zero only when query vectors are captured. Consumed by LEM's Q/K Bone Orientation analysis — pure Go CPU math, no GPU dependency.

## TextModel interface

```go
type TextModel interface {
    Generate(ctx context.Context, prompt string, opts ...GenerateOption) iter.Seq[Token]
    Chat(ctx context.Context, messages []Message, opts ...GenerateOption) iter.Seq[Token]
    Classify(ctx context.Context, prompts []string, opts ...GenerateOption) core.Result
    BatchGenerate(ctx context.Context, prompts []string, opts ...GenerateOption) core.Result
    ModelType() string
    Info() ModelInfo
    Metrics() GenerateMetrics
    Err() core.Result
    Close() core.Result
}
```

Key design decisions:

**`context.Context` on streaming methods** — required for HTTP handler cancellation, request timeouts, and graceful shutdown. Checked by engines at token boundaries.

**`iter.Seq[Token]` return type** — Go 1.23+ range-over-function iterators. The caller ranges over the sequence; the engine controls token production, using direct GPU-buffer access without channel overhead.

**`Err() core.Result`** — `iter.Seq` cannot carry errors alongside values. Following the `database/sql` `Row.Err()` pattern, the error from the most recent `Generate`/`Chat` is stored internally and returned here. Clean end-of-sequence returns an OK Result; cancellation and OOM return a failure.

**`Classify` and `BatchGenerate` return `core.Result`** — the payload (`[]ClassifyResult` / `[]BatchResult`) is carried in `Value` when OK. `Classify` is prefill-only (single forward pass, no autoregressive loop) — the fast path for domain labelling. `BatchGenerate` runs full autoregressive decoding across prompts.

**`Chat()` on the model** — chat templates differ across architectures (Gemma, Qwen3, Llama all use distinct formats). Applying the template in the engine means consumers receive already-formatted input regardless of family.

### Optional capabilities

Extra capability is expressed through separate interfaces, discovered by type assertion — never by widening `TextModel`:

- `VisionModel { AcceptsImages() bool }` — a live probe of whether the loaded checkpoint accepts image turns (a vision-capable family may ship a snapshot without the tower). Implemented by the metal engine.
- `AttentionInspector { InspectAttention(...) (*AttentionSnapshot, error) }` — Q/K extraction for Bone Orientation analysis. Defined and forwarded by `serving.InferenceAdapter`; not implemented by an in-tree engine yet.
- Training uses the `engine.TrainerModel` / `engine.Trainer` seam (`OpenTrainer`), not the root `TrainableModel.ApplyLoRA` interface — see [Interfaces](interfaces.md).

## Backend interface

```go
type Backend interface {
    Name() string
    LoadModel(path string, opts ...LoadOption) core.Result
    Available() bool
}
```

**`Name()`** — the registry key: `"metal"` or `"rocm"` today. This is the string passed to `WithBackend()`.

**`LoadModel()`** — reads a model directory (safetensors: `config.json` + `.safetensors`; or a GGUF file for ROCm) and returns a ready `TextModel` in the Result's `Value` when OK.

**`Available()`** — reports whether the engine can run on the current hardware. A backend registers unconditionally (its build tag governs whether it compiles in at all) while still reporting `false` when the GPU runtime is absent; `Default()` skips unavailable backends.

A backend that can describe its accelerator without loading a model also implements `DeviceInfoProvider { DeviceInfo() DeviceInfo }`, reachable via `inference.BackendDeviceInfo("metal")`.

## Backend registry

The registry is a package-level `map[string]Backend` guarded by a Core mutex (`core.New().Lock("inference.backends").Mutex`).

**Registration** — engines call `inference.Register(b)` from an `init()` gated by the engine's build tags:

```go
// engine/metal (darwin && arm64)
func init() { inference.Register(metalBackend{}) }

// engine/hip (linux && amd64)
func init() { inference.Register(&rocmBackend{}) }
```

Registering a name that already exists overwrites the previous entry — test code can swap backends without a de-registration step.

**Discovery** — `Get(name) (Backend, bool)` is a direct lookup. `List() []string` returns registered names sorted alphabetically. `All() iter.Seq2[string, Backend]` iterates them. `Default() core.Result` walks the preference order `metal → rocm → llama_cpp`, returning the first available backend; if none of those are available it falls back to any registered available backend, and fails with `no backends registered` / `no backends available` otherwise. (`llama_cpp` remains a preference slot; no package in this repo registers it.)

**`LoadModel()` routing** — the top-level entry point:

```go
func LoadModel(path string, opts ...LoadOption) core.Result {
    cfg := ApplyLoadOpts(opts)
    if cfg.Backend != "" {
        // Get(cfg.Backend) → validate registered + Available() → b.LoadModel(...)
    }
    // else Default() → b.LoadModel(...)
}
```

`WithBackend("rocm")` pins a specific backend and bypasses `Default()`.

## Functional options

Two independent option types, both the standard Go functional-options pattern.

### GenerateConfig / GenerateOption

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
    EnableThinking *bool          // nil = model default
    ThinkingBudget int            // 0 = unlimited
    Thinking       ThinkingConfig // resolved thought-channel policy
    // trace + cache-hygiene + probe knobs (engine-neutral):
    TraceTokenPhases, TraceTokenText           bool
    GenerationClearCache                        bool
    GenerationClearCacheInterval                int
    ProbeSink                                   probe.Sink
}
```

`DefaultGenerateConfig()` sets `Temperature=0.0` (greedy) and `RepeatPenalty=1.0` (no penalty); everything else is the zero value. **`MaxTokens` is deliberately not defaulted** — absent (0) the engine resolves it to the model's context at generation time; a fixed default would truncate every generation at a guess.

`ApplyGenerateOpts(opts) GenerateConfig` starts from the defaults and applies options in order (last write wins for scalars). See [Types](types.md) for the full `With*` list.

### LoadConfig / LoadOption

```go
type LoadConfig struct {
    Backend       string
    ContextLen    int
    GPULayers     int
    ParallelSlots int
    AdapterPath   string
}
```

`ApplyLoadOpts` defaults `GPULayers` to `-1` (full GPU offload); `0` forces CPU-only; positive values request partial offload (ROCm/llama.cpp; Metal always does full offload). `AdapterPath` injects a LoRA adapter at load time without fusing it into the base weights.

## Model discovery

```go
func Discover(baseDir string) iter.Seq[DiscoveredModel]
```

`Discover` walks the directory tree under `baseDir` **recursively** (not one level), yielding every directory that contains `config.json` plus at least one `.safetensors` file. It also probes `baseDir` itself, so a direct model path works. The walk is lazy — a caller can `break` out of the range early.

```go
type DiscoveredModel struct {
    Path        string // always absolute
    ModelType   string // model_type from config.json / GGUF metadata
    QuantBits   int
    QuantGroup  int
    QuantType   string // e.g. q4_k_m, q8_0 (when known)
    QuantFamily string // e.g. q4, q8 (when known)
    NumFiles    int
    Format      string // "safetensors" or "gguf" (when known)
}
```

`ModelType` is read from `config.json`'s `model_type` field (or GGUF metadata). Invalid JSON is tolerated — the directory is still yielded with an empty `ModelType`.

## Stability contract

The root `inference` package is the shared contract every engine, the serving layer, and consumers depend on. Rules governing its evolution:

1. Existing method signatures on `TextModel` and `Backend` are not changed.
2. New methods are added only when two or more call sites have a concrete need.
3. New capability is expressed as **separate optional interfaces** (`VisionModel`, `AttentionInspector`, `TrainableModel`, `DeviceInfoProvider`) discovered by type assertion — never by widening `TextModel`.
4. `GenerateConfig` and `LoadConfig` may gain new fields with zero-value defaults; this is backwards compatible.
</content>
</invoke>
