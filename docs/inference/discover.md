<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# discover.go — model directory scanning

**Package**: `dappco.re/go/inference`
**File**: `go/discover.go`

## What this is

A backend-neutral filesystem scan that yields one `DiscoveredModel` per model directory under a root. Used by:

- CoreAgent / core/ide model picker UI
- `lab/` to enumerate available models
- Test harnesses that auto-find fixtures

Two entry points, with different coverage:

- **`Discover(baseDir)`** (this file) — a **lazy** `iter.Seq[DiscoveredModel]` over **safetensors** model directories only (`config.json` + at least one `*.safetensors`). This is the function documented below.
- **`DiscoverModels(basePath)`** (in [gguf.md](gguf.md) / `gguf.go`) — an **eager** `[]DiscoveredModel` that includes both safetensors dirs (via `Discover`) **and** GGUF files, sorted by path. Reach for this when you also need `.gguf` models.

Architecture + quantisation metadata is extracted at scan time so callers don't have to load each model to decide whether it's interesting.

## DiscoveredModel

```go
type DiscoveredModel struct {
    Path        string  // absolute path to dir or .gguf file
    ModelType   string  // architecture: gemma3, qwen3, llama, …
    QuantBits   int     // 0 = unknown / unquantised
    QuantGroup  int
    QuantType   string  // q4_k_m, q8_0, etc. (GGUF)
    QuantFamily string  // q4, q8 (coarse)
    NumFiles    int     // number of weight files
    Format      string  // "safetensors" or "gguf"
}
```

## Discover

```go
for m := range inference.Discover("/Volumes/Data/models") {
    fmt.Printf("%s  arch=%s  quant=%dbit\n", m.Path, m.ModelType, m.QuantBits)
}
```

Returns `iter.Seq[DiscoveredModel]`. Iteration is lazy — caller can break early on first match. It is a pre-order directory walk; siblings within each directory are visited in alphabetical name order.

## What it inspects (safetensors directories)

- `config.json` → `model_type`, `quantization` / `quantization_config` (`bits`, `group_size`)
- `NumFiles` = count of `*.safetensors` in the directory
- `Format` is always `"safetensors"` for `Discover` results

Detection is metadata-only — weight tensors are not loaded. (GGUF header parsing lives in `ReadGGUFInfo` / `DiscoverModels`; see [gguf.md](gguf.md).)

## What it emits vs skips

A directory yields a `DiscoveredModel` only when it contains **both** `config.json` and at least one `*.safetensors` file. Every other directory is walked but produces nothing. There is no explicit hidden-directory or symlink-loop handling — directories that lack the two markers are simply passed over, and the walk recurses into every subdirectory it can list.

## Why a generator not a slice

Large model trees with 100+ models would cost noticeable RAM if returned all-at-once. The generator pattern lets a UI render the first row immediately while the scan continues.

## Related

- [gguf.md](gguf.md) — `GGUFInfo` + `ReadGGUFInfo` + `DiscoverModels` (the GGUF-aware scan)
