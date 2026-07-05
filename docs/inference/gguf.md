<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# gguf.go — GGUF metadata reader

**Package**: `dappco.re/go/inference`
**File**: `go/gguf.go`

## What this is

The discovery-side GGUF (llama.cpp model format) metadata mapping. `ReadGGUFInfo` reads the header + a *subset* of the key-value section without loading tensors — same intent as the safetensors path in `discover.go`. The wire parsing itself is delegated to the sibling `dappco.re/go/inference/model/gguf` package (`gguf.ResolveFile`, `gguf.MetadataSubset`); this file owns only the narrow `GGUFInfo` field mapping and the fixed `general.file_type` → quantisation table.

## GGUFInfo

```go
type GGUFInfo struct {
    Path             string
    Architecture     string
    VocabSize        int
    HiddenSize       int
    NumLayers        int
    ContextLength    int
    QuantBits        int
    QuantGroup       int
    QuantType        string  // q4_k_m, q8_0, f16, …
    QuantFamily      string  // q4, q8, f16
    TensorCount      int
    MetadataCount    int
    ValidationIssues []GGUFValidationIssue
}
```

`GGUFInfo.Valid()` reports true when no `ValidationIssues` carry `GGUFValidationError` severity. `GGUFValidationIssue` = `{Severity, Code, Message, Tensor}`; severity is `GGUFValidationWarning` or `GGUFValidationError`. The identity fields map cleanly onto `ModelIdentity`.

## Quantisation mapping

`general.file_type` is folded onto the discovery quant fields via a fixed table (deliberately simpler than the `model/gguf` package's per-tensor-type inference):

| file_type | bits | group | type | family |
|-----------|------|-------|------|--------|
| 0 | 32 | 0 | f32 | f32 |
| 1 | 16 | 0 | f16 | f16 |
| 7 | 8 | 32 | q8_0 | q8 |
| 15 | 4 | 32 | q4_k_m | q4 |
| other | 0 | 0 | "" | "" |

## Public API

```go
info, err := inference.ReadGGUFInfo("/models/foo.gguf")   // one file → GGUFInfo
models    := inference.DiscoverModels("/models")          // dir → []DiscoveredModel (safetensors + GGUF)
```

`DiscoverModels` combines `Discover` (safetensors) with a GGUF walk: any directory holding exactly one `*.gguf` is read via `ReadGGUFInfo` and folded into a `DiscoveredModel` with `Format: "gguf"`; results are sorted by path. A `.gguf` file passed directly (not a directory) yields a single-element slice.

## What it reads

Only the handful of discovery keys are decoded — `general.architecture`, `general.file_type`, and the `*.vocab_size` / `*.embedding_length` / `*.block_count` / `*.context_length` / `tokenizer.ggml.tokens` keys. Every other metadata entry's value bytes are skipped in place inside `gguf.MetadataSubset`, keeping this cheap enough for per-directory discovery sweeps.

## Why the mapping lives here (not in a llama-cpp binding)

- **No CGO.** The wire reader is pure-Go (`model/gguf`), not a llama-cpp cgo binding.
- **Narrow, pinned surface.** The `GGUFInfo` mapping + the fixed quantisation table are the discovery contract downstream backends were built against — kept stable by this package's alloc-budget and behaviour tests.
- **Cross-platform.** The same code compiles on every platform; backend-specific GGUF use (loading tensors) lives in the engines.

## Related

- [discover.md](discover.md) — `Discover()` (safetensors) vs `DiscoverModels()` (safetensors + GGUF)
- `go/model/gguf/` — the actual GGUF wire reader (`ResolveFile`, `MetadataSubset`)
- `go/model/quant/` — quantisation used when engines load GGUF tensors
