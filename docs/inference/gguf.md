<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# gguf.go — GGUF metadata reader

**Package**: `dappco.re/go/inference`
**File**: `go/gguf.go`

## What this is

A minimal GGUF (llama.cpp model format) metadata parser. Reads the header + key-value section without loading tensors — same intent as the safetensors path in `discover.go`. Used by Discover, by `model_pack.go` validation in go-mlx, and by the core/ide model picker.

## GGUFInfo

```go
type GGUFInfo struct {
    Path             string
    Architecture     string
    QuantType        string  // q4_k_m, q8_0, f16, …
    QuantFamily      string  // q4, q8, f16
    QuantBits        int
    QuantGroup       int
    ContextLength    int
    NumLayers        int
    HiddenSize       int
    VocabSize        int
    ChatTemplate     string
    NumTensors       int
    HeaderBytes      int64
    FileBytes        int64
    Metadata         map[string]any
}
```

Maps cleanly onto `ModelIdentity` + `TokenizerIdentity.ChatTemplate`.

## GGUF format constants

```go
ggufMagic      = 0x46554747   // "GGUF" little-endian
ggufVersion    = 3
ggufTypeUint32 = 4
ggufTypeString = 8
```

The parser handles v2 + v3 files. v1 is rare in the wild; not supported.

## Public API

```go
info, err := inference.ReadGGUFInfo("/models/foo.gguf")
infos     := inference.ScanGGUF(io.Reader)   // for streaming scenarios
```

## What it parses

Header → key-value section. Stops as soon as the architecture + quant + chat template are known. Tensor headers are scanned only when `NumTensors` is requested (default off — the scan is bounded to the metadata section).

## Why a local parser instead of llama-cpp-go binding

Three reasons:

1. **No CGO.** `inference` is zero-deps; pulling in a llama-cpp binding violates the package contract.
2. **Smaller surface.** We only need metadata, not inference — the parser is ~285 lines.
3. **Cross-platform.** The same code compiles on every platform; backend-specific GGUF use (loading tensors) lives in the backend.

## Related

- [discover.md](discover.md) — `Discover()` uses this for `.gguf` files
- `go-mlx/docs/model/gguf_info.md` (planned) — backend-specific GGUF tensor load
- `go-mlx/docs/model/gguf_quantize.md` (planned) — write-side GGUF quantisation
