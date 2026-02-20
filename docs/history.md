# Project History — go-inference

## Origin

`go-inference` was created on 19 February 2026 to solve a dependency inversion problem in the Core Go ecosystem.

`go-mlx` (Apple Metal inference on darwin/arm64) and `go-rocm` (AMD ROCm inference on linux/amd64) both needed to expose the same `TextModel` interface so that `go-ml` and `go-ai` could treat them interchangeably. The two backends cannot import each other — each carries platform-specific CGO or subprocess dependencies that would break cross-platform compilation.

Three options were considered:

1. **Duplicate interfaces** — Each backend defines its own `TextModel`. Simple to start, but the interfaces diverge over time as backends evolve without a shared contract. Rejected.
2. **Shared interface package** (chosen) — A new package with zero dependencies defines the contract. ~100 LOC at inception, compiles on all platforms. All backends import it; it imports nothing.
3. **Define in go-ml** — `go-ml` already had `Backend` and `StreamingBackend` types. Rejected because `go-ml` carries heavy dependencies (DuckDB, Parquet) that backends should not import.

## Commit History

### `fca0ed8` — Initial commit

Repository scaffolding. `go.mod`, empty `README.md`.

### `07cd917` — feat: define shared TextModel, Backend, Token, Message interfaces

First substantive commit. Defined `TextModel`, `Backend`, `Token`, `Message`, the `Register`/`Get`/`List`/`Default`/`LoadModel` registry functions, `GenerateConfig`, `LoadConfig`, and all `With*` options. Established the zero-dependency constraint and the `Default()` priority order (metal > rocm > llama_cpp).

### `3719734` — feat: add ParallelSlots to LoadConfig for concurrent inference

Added `WithParallelSlots` to `LoadConfig`. Required for llama.cpp backends that allocate inference slots at load time. Metal backends ignore the field.

### `2517b77` — feat: add batch inference API (Classify, BatchGenerate)

Added `Classify` and `BatchGenerate` to `TextModel`, along with `ClassifyResult` and `BatchResult`. `Classify` is a prefill-only fast path (single forward pass, no autoregressive decoding) for domain classification tasks in `go-i18n`. `BatchGenerate` runs full autoregressive decoding across multiple prompts in parallel.

### `df17676` — feat: add GenerateMetrics type and Metrics() to TextModel

Added `GenerateMetrics` and `TextModel.Metrics()`. Provides per-operation performance data: token counts, prefill and decode durations, throughput, and GPU memory usage. Required by the LEM Lab dashboard and future monitoring integrations.

### `28f444c` — feat: add ModelInfo type and Info() to TextModel

Added `ModelInfo` and `TextModel.Info()`. Provides static metadata about a loaded model: architecture, vocabulary size, layer count, hidden dimension, and quantisation details. Required by `go-ai` MCP tools that surface model information to agents.

### `884225d` — feat: add Discover() for scanning model directories

Added `Discover(baseDir string) ([]DiscoveredModel, error)` and `DiscoveredModel`. Scans a directory tree (one level deep) for model directories identified by the presence of `config.json` and `.safetensors` weight files. Used by LEM Lab's model picker UI and `go-ai`'s model listing MCP tool.

### `c61ec9f` — docs: expand package doc with workflow examples

Expanded the package-level godoc comment in `inference.go` with complete examples: streaming generation, chat, classification, batch generation, functional options, and model discovery.

### `15ee86e` — fix: add json struct tags to Message for API serialization

Added `json:"role"` and `json:"content"` tags to `Message`. Required for correct serialisation through `go-ai`'s MCP tool payloads and the agentic portal's REST API.

### `d76448d` — test(inference): add comprehensive tests for all exported API

1,074 lines of Pest-style tests (using Go's `testing` package and `testify`). Comprehensive coverage of:

- `Register`, `Get`, `List`, `Default`, `LoadModel` — all happy paths, error paths, and edge cases
- `Default()` priority order (metal > rocm > llama_cpp > any available)
- All `GenerateOption` and `LoadOption` functions
- `ApplyGenerateOpts` and `ApplyLoadOpts` — nil options, empty options, last-option-wins
- `Discover` — single models, multiple models, quantised models, base-dir-as-model, missing files, invalid JSON
- All struct types: `Token`, `Message`, `ClassifyResult`, `BatchResult`, `ModelInfo`, `GenerateMetrics`
- Compile-time interface compliance assertions

Dispatched to Charon (Linux build agent). Commit hash recorded in TODO.md as Phase 1 foundation marker.

### `85f587a` — docs: mark Phase 1 foundation tests complete (Charon d76448d)

Updated TODO.md to record Phase 1 completion and Charon's commit hash.

### `c91e305` — docs: mark Phase 2 integration complete — all 3 backends migrated

Updated TODO.md to record Phase 2 integration completion across go-mlx, go-rocm, and go-ml.

## Phase Summary

### Phase 1 — Foundation (complete)

Established the interface contract, registry, functional options, model discovery, and comprehensive tests. All exported API covered. No backend implementations in this package.

### Phase 2 — Integration (complete)

All three backends migrated to implement `inference.TextModel` and register via `inference.Register()`:

- **go-mlx** (`register_metal.go`, darwin/arm64): `metalBackend{}` + `metalAdapter{}` wrap the internal Metal model. Full `TextModel` coverage including `Classify`, `BatchGenerate`, `Info`, `Metrics`. Build-tagged `darwin && arm64`.
- **go-rocm** (`register_rocm.go`, linux/amd64): `rocmBackend{}` spawns and manages a `llama-server` subprocess. 5,794 LOC. Build-tagged `linux && amd64`.
- **go-ml** (`adapter.go`, `backend_http_textmodel.go`): Two-way bridge. `adapter.go` (118 LOC) wraps `inference.TextModel` into `go-ml`'s internal `Backend`/`StreamingBackend` interfaces. `backend_http_textmodel.go` (135 LOC) provides the reverse: wraps an HTTP llama.cpp server as `inference.TextModel`. `backend_mlx.go` collapsed from 253 to 35 LOC after migration.

### Phase 3 — Extended Interfaces (deferred)

Two interfaces are specified but not yet implemented, pending concrete consumer demand:

**BatchModel** — For throughput-sensitive batch classification (e.g. `go-i18n` processing 5,000 sentences per second):

```go
type BatchModel interface {
    TextModel
    BatchGenerate(ctx context.Context, prompts []string, opts ...GenerateOption) iter.Seq2[int, Token]
}
```

Note: the current `BatchGenerate` on `TextModel` collects all tokens before returning. A streaming `BatchModel` with `iter.Seq2` would reduce peak memory for large batches.

**StatsModel** — For dashboard and monitoring integrations:

```go
type StatsModel interface {
    TextModel
    Stats() GenerateStats
}
```

Where `GenerateStats` aggregates `GenerateMetrics` across multiple calls (rolling averages, peak values, histograms).

Neither interface will be added until at least two consumers have a concrete need. The pattern for adding them is: define the interface in this package, update go-mlx and go-rocm to implement it, update go-ml's adapter, then update consumers.

## Known Limitations

**Metrics on CPU backends** — `GenerateMetrics.PeakMemoryBytes` and `ActiveMemoryBytes` are zero for CPU-only backends. There is no protocol for backends to report CPU RAM usage; this was considered unnecessary at the time of design.

**`Discover` scan depth** — `Discover` scans only one level deep. Deeply nested model hierarchies (e.g. `models/org/repo/revision/`) are not found. The consumer is expected to call `Discover` on the correct parent directory.

**`Discover` and invalid JSON** — A `config.json` containing invalid JSON is silently tolerated: the directory is included with an empty `ModelType`. This prevents a single malformed file from hiding all other models in a directory, but it means the returned `DiscoveredModel` may be incomplete.

**No de-registration** — `Register` overwrites silently; there is no `Unregister`. This is intentional for simplicity. Backends registered in `init()` live for the lifetime of the process.

**`Default()` error message** — When all registered backends are unavailable, the error says "no backends registered" rather than "no backends available". This is slightly misleading but matches the no-backends case exactly, which simplifies error handling in consumers that treat both cases identically.

**`ParallelSlots` ignored by Metal** — Apple Metal manages concurrency internally. `WithParallelSlots` is accepted by `go-mlx` but has no effect. This is documented in `options.go` but not enforced.

## Future Considerations

- A `StatsModel` interface, when two consumers require aggregated metrics.
- A streaming `BatchModel` with `iter.Seq2[int, Token]` for high-throughput classification.
- Licence headers on all source files (currently absent, tracked informally).
- A formal `CHANGELOG.md` if the package grows beyond its current single-package scope.
- Consideration of `errors.Is`/`errors.As` sentinel errors (e.g. `ErrNoBackend`, `ErrBackendUnavailable`) to allow consumers to handle specific failure modes without string matching.
