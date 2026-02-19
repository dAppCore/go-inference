# FINDINGS.md — go-inference Research & Discovery

---

## 2026-02-19: Package Creation (Virgil)

### Motivation

go-mlx (darwin/arm64) and go-rocm (linux/amd64) both need to implement the same TextModel interface, but go-rocm can't import go-mlx (platform-specific CGO dependency). A shared interface package solves this.

### Alternatives Considered

1. **Duplicate interfaces** — Each backend defines its own TextModel. Simple but diverges over time as backends evolve independently. Rejected.
2. **Shared interface package** (chosen) — `core/go-inference` defines the contract. ~100 LOC, zero deps, compiles everywhere.
3. **Define in go-ml** — go-ml already has Backend/StreamingBackend. But go-ml has heavy deps (DuckDB, Parquet) that backends shouldn't import. Rejected.

### Interface Design Decisions

- **`context.Context` on Generate/Chat**: Required for HTTP handler cancellation, timeouts, graceful shutdown. go-ml's current backend_mlx.go already uses ctx.
- **`Err() error` on TextModel**: iter.Seq can't carry errors. Consumers check Err() after the iterator stops. Pattern matches database/sql Row.Err().
- **`Chat()` on TextModel**: Models own their chat templates (Gemma3, Qwen3, Llama3 all have different formats). Keeping templates in consumers means every consumer duplicates model-specific formatting.
- **`Available() bool` on Backend**: Needed for Default() to skip unavailable backends (e.g. ROCm registered but no GPU present).
- **`GPULayers` in LoadConfig**: ROCm/llama.cpp support partial GPU offload. Metal always does full offload. Default -1 = all layers.
- **`RepeatPenalty` in GenerateConfig**: llama.cpp backends use this heavily. Metal backends can ignore it.

### Consumer Mapping

| Consumer | What it imports | How it uses TextModel |
|----------|----------------|----------------------|
| go-ml | go-inference | Wraps TextModel into its own Backend interface, adds scoring |
| go-ai | go-inference (via go-ml) | Exposes via MCP tools |
| go-i18n | go-inference | Direct: LoadModel → Generate(WithMaxTokens(1)) for classification |
| LEM Lab | go-inference (via go-ml) | Chat streaming for web UI |
