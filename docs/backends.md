---
title: Backends
description: How the backend registry works and how to implement a new inference backend.
---

# Backends

go-inference uses a registry pattern to decouple consumers from GPU-specific implementations. Backends self-register at init time with build tags, so the right backend is available on each platform without any consumer-side configuration.

## Registry

The registry is a package-level `map[string]Backend` protected by a `sync.RWMutex`.

### Registry functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `Register` | `Register(b Backend)` | Add a backend to the registry (called from `init()`) |
| `Get` | `Get(name string) (Backend, bool)` | Retrieve a backend by name |
| `List` | `List() []string` | All registered backend names, sorted alphabetically |
| `All` | `All() iter.Seq2[string, Backend]` | Iterator over all registered backends |
| `Default` | `Default() (Backend, error)` | First available backend by platform preference |
| `LoadModel` | `LoadModel(path string, opts ...LoadOption) (TextModel, error)` | Load via specified or default backend |
| `LoadTrainable` | `LoadTrainable(path string, opts ...LoadOption) (TrainableModel, error)` | Load a training-capable model |

### Platform preference

`Default()` walks a priority list and returns the first available backend:

```
metal > rocm > llama_cpp > (any other registered backend)
```

Metal is preferred on Apple Silicon for direct GPU memory access. ROCm is preferred over llama.cpp on Linux because it avoids HTTP overhead. If none of the preferred backends are available, any registered backend that reports `Available() == true` is used.

If no backends are registered at all, `Default()` returns:

```
inference: no backends registered (import a backend package)
```

### LoadModel routing

`LoadModel` is the primary consumer entry point. It resolves the backend then delegates:

```go
// Explicit backend
m, err := inference.LoadModel("/path/to/model/", inference.WithBackend("rocm"))

// Auto-detect (uses Default())
m, err := inference.LoadModel("/path/to/model/")
```

When `WithBackend()` is set, `LoadModel` looks up the named backend directly and returns an error if it is not registered or not available. When no backend is specified, it calls `Default()`.

### Overwriting entries

Registering a name that already exists silently overwrites the previous entry. This allows test code to replace backends without a separate de-registration step.

---

## How backends register

Backends call `inference.Register()` from an `init()` function guarded by build tags. This ensures the registration only compiles on the target platform:

```go
// file: register_metal.go in go-mlx
//go:build darwin && arm64

package metal

import "dappco.re/go/inference"

func init() {
    inference.Register(NewBackend())
}
```

```go
// file: register_rocm.go in go-rocm
//go:build linux && amd64

package rocm

import "dappco.re/go/inference"

func init() {
    inference.Register(NewBackend())
}
```

The consumer imports the backend package with a blank import to trigger `init()`:

```go
import (
    "dappco.re/go/inference"
    _ "forge.lthn.ai/core/go-mlx/metal"  // registers "metal" backend
)
```

Because the import is guarded by build tags in the backend package, the blank import compiles to nothing on unsupported platforms.

---

## Implementing a new backend

To add a new inference backend (e.g. for a new GPU runtime or inference server), implement the `Backend` interface and optionally `TrainableModel`.

### Step 1: Implement Backend

```go
package mybackend

import "dappco.re/go/inference"

type myBackend struct{}

func NewBackend() inference.Backend {
    return &myBackend{}
}

func (b *myBackend) Name() string { return "mybackend" }

func (b *myBackend) Available() bool {
    // Check whether the runtime/hardware is present.
    // Return false if the GPU driver is missing, the server is unreachable, etc.
    return checkHardware()
}

func (b *myBackend) LoadModel(path string, opts ...inference.LoadOption) (inference.TextModel, error) {
    cfg := inference.ApplyLoadOpts(opts)
    // Load weights, allocate GPU memory, set up KV cache...
    return &myModel{config: cfg}, nil
}
```

### Step 2: Implement TextModel

Every method on the `TextModel` interface must be implemented. Key considerations:

**Generate and Chat** must return `iter.Seq[Token]`. The iterator pattern gives the backend control over token production:

```go
func (m *myModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
    cfg := inference.ApplyGenerateOpts(opts)
    return func(yield func(inference.Token) bool) {
        // Prefill the prompt...
        for i := 0; i < cfg.MaxTokens; i++ {
            if ctx.Err() != nil {
                m.lastErr = ctx.Err()
                return
            }
            tok := m.decodeNext()
            if !yield(tok) {
                return // caller broke out of range loop
            }
            if tok.ID == m.eosTokenID {
                return
            }
        }
    }
}
```

**Err** stores the error from the last Generate/Chat call:

```go
func (m *myModel) Err() error { return m.lastErr }
```

**Chat** should apply the model's native chat template before calling Generate internally. Do not expose template logic to the consumer.

**Classify** runs a single forward pass per prompt (no autoregressive loop). Only populate `ClassifyResult.Logits` when the config has `ReturnLogits == true`.

### Step 3: Register with build tags

Create a registration file with appropriate build constraints:

```go
// file: register.go
//go:build linux && amd64

package mybackend

import "dappco.re/go/inference"

func init() {
    inference.Register(NewBackend())
}
```

### Step 4 (optional): Support training

If your backend supports LoRA fine-tuning, have your model type also implement `TrainableModel`:

```go
func (m *myModel) ApplyLoRA(cfg inference.LoRAConfig) inference.Adapter {
    // Inject LoRA layers into cfg.TargetKeys projections.
    // Return an Adapter that wraps the trainable parameters.
    return &myAdapter{params: loraParams}
}

func (m *myModel) Encode(text string) []int32 {
    return m.tokeniser.Encode(text)
}

func (m *myModel) Decode(ids []int32) string {
    return m.tokeniser.Decode(ids)
}

func (m *myModel) NumLayers() int {
    return m.config.NumLayers
}
```

The `Adapter` returned by `ApplyLoRA` must implement `TotalParams()` and `Save()`:

```go
type myAdapter struct {
    params []trainableParam
}

func (a *myAdapter) TotalParams() int {
    total := 0
    for _, p := range a.params {
        total += p.NumElements()
    }
    return total
}

func (a *myAdapter) Save(path string) error {
    // Write adapter weights to safetensors format.
    return writeSafetensors(path, a.params)
}
```

### Step 5 (optional): Support attention inspection

If your backend can extract attention vectors from the KV cache, implement `AttentionInspector`:

```go
func (m *myModel) InspectAttention(ctx context.Context, prompt string, opts ...inference.GenerateOption) (*inference.AttentionSnapshot, error) {
    // Run prefill, then read Q/K vectors from the KV cache.
    return &inference.AttentionSnapshot{
        NumLayers:    m.numLayers,
        NumHeads:     m.numKVHeads,
        SeqLen:       seqLen,
        HeadDim:      m.headDim,
        Keys:         keys,    // [layer][head] -> flat []float32
        Architecture: m.arch,
    }, nil
}
```

---

## Model discovery

`Discover` scans a directory for model directories, useful for building model selection UIs or inventory tools.

```go
func Discover(baseDir string) iter.Seq[DiscoveredModel]
```

A valid model directory must contain:
- `config.json` — parsed for `model_type` and optional `quantization` fields
- At least one `.safetensors` file

The function scans one level deep (immediate subdirectories of `baseDir`). It also checks `baseDir` itself, so passing a direct model path works:

```go
// Scan a models directory
for m := range inference.Discover("/path/to/models/") {
    fmt.Printf("%s — %s (%d files)\n", m.Path, m.ModelType, m.NumFiles)
}

// Check a single model directory
for m := range inference.Discover("/path/to/models/gemma3-1b") {
    fmt.Printf("arch=%s quant=%d-bit\n", m.ModelType, m.QuantBits)
}
```

---

## Existing backends

| Backend | Package | Platform | Registration |
|---------|---------|----------|-------------|
| `metal` | go-mlx | darwin/arm64 | `//go:build darwin && arm64` |
| `rocm` | go-rocm | linux/amd64 | `//go:build linux && amd64` |
| `llama_cpp` | go-ml | any (HTTP) | No build tags (wraps llama.cpp HTTP server) |

**metal** — Native Apple Metal GPU inference via CGO bindings. Supports `TrainableModel` and `AttentionInspector`. Highest throughput on Apple Silicon.

**rocm** — AMD ROCm GPU inference via a managed `llama-server` subprocess. Direct GPU memory access without HTTP overhead.

**llama_cpp** — Wraps an external llama.cpp HTTP server as a `TextModel`. Works on any platform. Registered in go-ml's `backend_http_textmodel.go`.
