<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# inference.go — TextModel + Backend + registry

**Package**: `dappco.re/go/inference`
**File**: `go/inference.go`

## What this is

The load-bearing file of the whole tetrad. Five concepts:

1. **`TextModel`** — the runtime-facing model interface (Generate, Chat, Classify, BatchGenerate, ModelType, Info, Metrics, Err, Close).
2. **`Backend`** — the platform-facing factory interface (Name, LoadModel, Available).
3. **The registry** — package-global map of name → Backend, written at `init()` time by each native driver.
4. **`Default()`** — preference resolver: metal → rocm → llama_cpp → any.
5. **`LoadModel(path, opts...)`** — top-level convenience that picks a backend and returns a ready model as a `core.Result`.

Plus support DTOs: `Token`, `Message`, `ClassifyResult`, `BatchResult`, `GenerateMetrics`, `ModelInfo`, `AttentionSnapshot`, `AttentionInspector`.

## TextModel

```go
type TextModel interface {
    Generate(ctx, prompt, ...GenerateOption) iter.Seq[Token]
    Chat(ctx, []Message, ...GenerateOption)  iter.Seq[Token]
    Classify(ctx, []string, ...GenerateOption) ([]ClassifyResult, error)
    BatchGenerate(ctx, []string, ...GenerateOption) ([]BatchResult, error)
    ModelType() string
    Info() ModelInfo
    Metrics() GenerateMetrics
    Err() error
    Close() error
}
```

Generate and Chat return Go 1.23+ range-over-func iterators. Errors are
retrieved post-iteration via `Err()` — same pattern as `database/sql`
`Row.Err()`. Don't ignore it; an iterator that stops early on an error
yields the same "iterator exhausted" signal as natural EOS.

Classify and BatchGenerate are batch calls returning slices — Classify
runs prefill-only (one forward pass per prompt, sample at the final
position) and is the fast path for classification scoring.

## Backend

```go
type Backend interface {
    Name() string
    LoadModel(path string, opts ...LoadOption) (TextModel, error)
    Available() bool
}
```

`Available()` returns false on hardware that can't run the backend —
`metal.Available()` is false on Linux, `rocm.Available()` is false on
darwin, etc. Used by `Default()` to skip registered-but-unusable
backends.

## Registry

Backends register at `init()`:

```go
// in go-mlx/register_metal.go (build-tagged darwin/arm64)
func init() { inference.Register(&metalbackend{}) }
```

Five operations on the global registry:

| Function | Returns | Notes |
|----------|---------|-------|
| `Register(b Backend)` | nothing | overwrites by name |
| `Get(name)` | `(Backend, bool)` | name lookup |
| `List()` | `[]string` | sorted names |
| `All()` | `iter.Seq2[string, Backend]` | sorted iteration |
| `Default()` | `core.Result` | preference resolver |

Preference order is hard-coded: `metal → rocm → llama_cpp → any`. The
"any" fallback iterates sorted names so behaviour is deterministic
across runs.

## LoadModel

```go
r := inference.LoadModel("/models/gemma3-1b")                     // auto
r := inference.LoadModel(path, inference.WithBackend("metal"))    // explicit
r := inference.LoadModel(path, inference.WithContextLen(8192))    // tuned

if !r.OK { return r }
model := r.Value.(TextModel)
defer model.Close()
```

Returns `core.Result`; the value is `TextModel`. Errors are wrapped
through the backend's name so the trace tells you which backend
refused.

## Token / Message / ClassifyResult / BatchResult

```go
type Token         struct { ID int32; Text string }
type Message       struct { Role, Content string }
type ClassifyResult struct { Token Token; Logits []float32 }
type BatchResult   struct { Tokens []Token; Err error }
```

`Logits` is nil unless the caller passed `inference.WithLogits()` —
populating logits doubles memory pressure and is off by default.

## GenerateMetrics + ModelInfo

`GenerateMetrics` is the post-operation telemetry snapshot:
- Token counts (prompt, generated)
- Timings (prefill duration, decode duration, total wall-clock)
- Throughput (prefill tok/s, decode tok/s — derived)
- Memory (peak / active GPU bytes)

`ModelInfo` is static metadata from the loaded model:
- Architecture (gemma3, qwen3, llama, …)
- VocabSize, NumLayers, HiddenSize
- QuantBits, QuantGroup

## AttentionSnapshot / AttentionInspector

Optional inspection interface — discovered by type assertion:

```go
if inspector, ok := model.(inference.AttentionInspector); ok {
    snap, err := inspector.InspectAttention(ctx, prompt)
}
```

Returns per-layer per-head K/Q tensors as flat float32 slices. Used by
go-ml capability probes and the agent-experience attention inspector
in core/ide.

## Why a global registry

Each backend lives in its own module behind build tags — Metal CGO
won't compile on Linux, ROCm bindings won't compile on darwin. A
caller importing `_ "dappco.re/go/mlx"` triggers its `init()` and the
backend appears in the registry; the caller's own code references no
darwin-specific symbols.

That's the trick. The contract package compiles everywhere; backends
plug themselves in via the side-channel of init time + build tags;
consumers ask `LoadModel("...")` and get whatever's actually available
on the box.

## Related

- [options.md](options.md) — `GenerateOption` / `LoadOption` and the `With*` functions
- [contracts.md](contracts.md) — extended capability interfaces (Scheduler, CacheService, EmbeddingModel, RerankModel)
- [discover.md](discover.md) — `Discover()` scans a directory for model dirs
- [service.md](service.md) — Core ServiceRuntime registration
- `go-mlx/docs/runtime/register_metal.md` — the canonical Backend implementation
