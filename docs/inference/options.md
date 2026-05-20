<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# options.go — GenerateOption + LoadOption

**Package**: `dappco.re/go/inference`
**File**: `go/options.go`

## What this is

Two functional-option families:

- **`GenerateOption`** — passed to Generate / Chat / Classify / BatchGenerate. Tunes sampling.
- **`LoadOption`** — passed to LoadModel / LoadTrainable. Tunes load.

Each is `func(*Config)`; backends call `ApplyGenerateOpts(opts)` / `ApplyLoadOpts(opts)` to flatten into a `GenerateConfig` / `LoadConfig`.

## GenerateConfig

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

`DefaultGenerateConfig()` — MaxTokens=256, Temperature=0.0 (greedy), RepeatPenalty=1.0, everything else zero.

## With* generators

| Function | Tunes | Typical |
|----------|-------|---------|
| `WithMaxTokens(n)` | output cap | 64 short, 256 medium, 2048 long-form |
| `WithTemperature(t)` | randomness | 0.0 greedy, 0.7 balanced, 1.5 high-variance |
| `WithTopK(k)` | top-k filter | 40 typical, 0 disabled |
| `WithTopP(p)` | nucleus | 0.9 typical, 0 disabled |
| `WithStopTokens(ids…)` | early halt | EOS id (model-specific) |
| `WithRepeatPenalty(p)` | repetition guard | 1.0 off, 1.1 mild, 1.5 strong |
| `WithLogits()` | capture logits | off by default — doubles classify memory |

## LoadConfig

```go
type LoadConfig struct {
    Backend       string  // "metal" | "rocm" | "llama_cpp" | "" (auto)
    ContextLen    int     // KV cache cap in tokens — 0 = model default
    GPULayers     int     // -1 = all (default), 0 = CPU, n = partial
    ParallelSlots int     // concurrent inference slots — 0 = backend default
    AdapterPath   string  // LoRA dir — empty = no adapter
}
```

`ApplyLoadOpts(opts)` starts with `GPULayers: -1` (full GPU); everything else zero.

## With* generators (load)

| Function | Tunes | Notes |
|----------|-------|-------|
| `WithBackend(name)` | explicit backend | overrides Default() preference order |
| `WithContextLen(n)` | KV cap | trade context vs VRAM |
| `WithGPULayers(n)` | offload | -1 all, 0 CPU, partial supported per-backend |
| `WithParallelSlots(n)` | concurrency | costs VRAM proportional to n |
| `WithAdapterPath(path)` | LoRA at load | weights stay separate from base |

## Why functional options

Backends grow option fields independently. Adding `WithFlashAttention(true)` doesn't touch any call site that doesn't pass it. `ApplyGenerateOpts` / `ApplyLoadOpts` flatten the chain so backends consume a plain struct internally.

## Related

- [inference.md](inference.md) — where GenerateOption / LoadOption are passed in
- [training.md](training.md) — `LoRAConfig` for fine-tuning loops
