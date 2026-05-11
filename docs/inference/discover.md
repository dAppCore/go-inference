<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# discover.go — model directory scanning

**Package**: `dappco.re/go/inference`
**File**: `go/discover.go`

## What this is

A backend-neutral filesystem scan that yields one `DiscoveredModel` per model directory under a root. Used by:

- CoreAgent / core/ide model picker UI
- `core/lab` to enumerate available models
- Test harnesses that auto-find fixtures

Detects both safetensors directories (`config.json` + `*.safetensors`) and GGUF files. Architecture + quantisation metadata extracted at scan time so callers don't have to load each model to decide whether it's interesting.

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

Returns `iter.Seq[DiscoveredModel]`. Iteration is lazy — caller can break early on first match. Sort order: alphabetical by path.

## What it inspects

For safetensors directories:
- `config.json` → `model_type`, `num_hidden_layers`, `vocab_size`, `quantization_config`
- File count = count of `*.safetensors`

For GGUF files:
- Magic + version header
- Architecture metadata key
- Quantisation type from tensor headers

Detection is metadata-only. Weight tensors are not loaded.

## What it skips

- Hidden directories (`.git`, `.cache`)
- Directories without `config.json` or matching `*.gguf`
- Symlink loops (basic loop detection)

## Why a generator not a slice

Large model trees with 100+ models would cost noticeable RAM if returned all-at-once. The generator pattern lets a UI render the first row immediately while the scan continues.

## Related

- [gguf.md](gguf.md) — `GGUFInfo` for the richer single-file scan
- `go-mlx/docs/model/model_pack.md` (planned) — full model-pack validation (uses Discover + Inspect)
- `go-ml/docs/scoring/inventory.md` (planned) — inventory persistence
