---
title: Backends
description: The in-tree GPU engines, how the backend registry works, and how to implement a new backend.
---

# Backends

go-inference uses a registry to decouple consumers from GPU-specific engines. Two engines live in this repository — `engine/metal` (Apple GPU) and `engine/hip` (AMD ROCm) — each gated by build tags so only the right one compiles on a given platform. A blank import registers the engine at `init` time; consumers program against the `Backend`/`TextModel` contract and never reference an engine's internals.

## The in-tree engines

### metal — Apple GPU, no cgo

Path `engine/metal`, package clause `native`, build tag `//go:build darwin && arm64`. "Metal" names the Apple Metal API this engine drives — it is **not** go-mlx's cgo `pkg/metal` (deleted, never ported). Verified in `engine/metal/device.go`:

- **No cgo, no mlx-c.** Kernels are dispatched from Go through the `github.com/tmc/apple` objc bridge (purego `objc_msgSend`).
- Loads the **same compiled `mlx.metallib`** the reference MLX build ships, located via the `MLX_METALLIB_PATH` environment variable, plus an optional sibling `lthn_kernels.metallib` (go-inference's own fused kernels; absent ⇒ those ops fall back to composed primitives).
- **The kernels are shared with MLX; the innovation is the encode path.** Decode and diffusion are fixed per-step command sequences, so the engine records the sequence once into an **Indirect Command Buffer (ICB)** and replays it per token — bypassing the host re-encode that dominates MLX's decode. A MoE arch falls back to the re-encode path (the ICB cannot host the router's host-side top-k).

Registers as `"metal"`. Loads a reactive native token model (dense / MoE / PLE, bf16 or 4-bit) with the directory's tokenizer attached; `WithContextLen` sizes the KV cache (default 4096). It implements `VisionModel` (`AcceptsImages`) and exposes LoRA SFT training through the `engine.TrainerModel` / `engine.Trainer` seam (`OpenTrainer`), not the root `TrainableModel.ApplyLoRA` interface.

#### Runtime levers

Environment variables read once at process start. The load-bearing ones (non-exhaustive — the full list greps as `LTHN_[A-Z0-9_]*` in `engine/metal`):

| Variable | Effect |
|----------|--------|
| `LTHN_KV_Q8=1` | int8 paged KV cache with f32 group scales, opted in per layer on gqa2 geometry (`nHeads == 2*kvHeads`, `headDim ≤ 256`) — half the KV bytes at parity tok/s. Errors loudly at load if no layer qualifies. |
| `LTHN_MTP_REENGAGE=0` | Restores the permanent low-acceptance bail for MTP speculative decoding. The default re-engagement gate is wall-clock-adaptive (probes plain-decode economics live), so paired runs are not byte-reproducible run-to-run; this switch is the reproducibility anchor. |
| `LTHN_MTP_VERIFY_FOLD` | MTP verify forward tier (#55). Default: the GREEDY verify runs the per-row lane (byte-identical to sequential plain decode — the exact-greedy contract) and the SAMPLED verify takes the small-K batched fold (qmm token-identity tier — weights swept once per block, batched numerics that can flip a near-tied argmax). `=1` forces the fold everywhere (the A/B lever that resurrects the pre-#55 greedy behaviour: faster, byte-inexact); `=0` forces the per-row lane everywhere. |
| `LTHN_MTP_ROWS_HEAD=1` | Re-arms the K-row fused verify rows head (qmm_t token-identity tier) in the byte-exact greedy verify — an A/B lever (#55). The exact lane otherwise scores verify rows through the per-row canonical qmv head, the tier plain decode picks with. |
| `LTHN_SDPA_SPLIT=N` | Overrides the paged-SDPA split-window grain (rows per pass-1 threadgroup window; default 256, halved on the GQA-shared route). A probe lever for kernel tuning, not a serving knob. |
| `LTHN_SDPA_GEMM_MINKV=N` | Overrides the attended-length knee at which prompt-scale SDPA switches from the multiQ vector kernel to the steel-GEMM composition (compiled default 2048; `=4096` restores the previous knee). A probe/A-B lever, not a serving knob. |
| `LTHN_FLASH_WIN=0` | Restores the multiQ ring kernel by disabling the sliding-window flash prompt-attention lane (an A/B lever). The window flash is otherwise gated by a 1024-row floor, so small chunks already stay on the ring kernel. |
| `LTHN_MTP_DIAG=1` | Per-cycle MTP draft/accept diagnostics on stderr (any non-empty value). |
| `LTHN_GPU_TRACE` | Per-stage GPU-time attribution for the batched dense pass (any non-empty value splits the pass's command buffer at named stage boundaries; the split serialises the stages, so the traced total runs slower than production and both the per-bucket shares and the total are printed). `=host` arms the host spans only (no command-buffer split), so wall-vs-seam decomposition reads at production GPU fidelity. |

### rocm — AMD ROCm

Path `engine/hip`, package `hip`. The default `linux && amd64` build is native-first: it registers the ROCm backend, reads GGUF metadata, and drives the native HIP runtime — the old OpenAI-compatible `llama-server` subprocess path survives only behind the `rocm_legacy_server` build tag and is not built by default. Three variants of the backend exist by build tag:

| Build tag | Behaviour |
|-----------|-----------|
| `linux && amd64 && !rocm_legacy_server` | native ROCm/HIP runtime (default) |
| `!linux \|\| !amd64` | portable stub: `Available()` returns `false`, `LoadModel` fails cleanly |
| `linux && amd64 && rocm_legacy_server` | legacy `llama-server` subprocess bridge |

Registers as `"rocm"`. GGUF loading works; safetensors model-pack loading is **not yet available** in the current quarantine landing (blocked on a missing upstream package — the load fails with an explicit message rather than guessing).

#### Quantisation lanes on AMD

Not every quantisation format is equally "native" on ROCm, and this shapes
what a ROCm quant contribution should target. The AMD-native lanes are:

- **GGUF** (`q4_0`, `q4_K`, `q8_0`) — MLX-style group-affine quantisation;
  `engine/hip`'s existing `rocm_mlx_q4_projection` kernel family already
  serves this.
- **FP8** (W8A8, via llm-compressor / AMD Quark) — 8-bit weight+activation.
- **MXFP4** (via AMD Quark) — OCP microscaled 4-bit (`model/quant/mxfp4`),
  hardware-accelerated from CDNA4/MI350 onward.

**GPTQ, AWQ, and Marlin are unsupported on AMD** — this is not a gap to
close by porting a kernel, it is the ecosystem's own shape: Marlin, the fast
kernel both GPTQ and AWQ lean on, is hand-written Nvidia tensor-core PTX with
no ROCm equivalent. Scoping ROCm quant work as "port GPTQ to HIP" swims
against that grain; the native-feeling targets are GGUF, FP8, and MXFP4
above. See `docs/design-rocm.md` §A.2 and the
[vLLM ROCm quantization compatibility matrix](https://docs.vllm.ai/en/latest/features/quantization/)
for the full picture.

### About `llama_cpp`

`llama_cpp` is still a slot in the preference order, but **no package in this repository registers it** as an `inference.Backend`. The serving layer provides `serving.HTTPBackend` (name `"http"`) and `serving.LlamaBackend` (name `"llama"`) that wrap an external llama.cpp HTTP server as a `TextModel` — but these are serving-level adapters, not registered inference backends.

---

## Registry

The registry is a package-level `map[string]Backend` guarded by a Core mutex.

### Registry functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `Register` | `Register(b Backend)` | Add a backend (called from `init()`); overwrites an existing same-named entry |
| `Get` | `Get(name string) (Backend, bool)` | Retrieve a backend by name |
| `List` | `List() []string` | All registered names, sorted alphabetically (nil when empty) |
| `All` | `All() iter.Seq2[string, Backend]` | Iterator over all registered backends, name order |
| `Default` | `Default() core.Result` | First available backend by preference order; Result's `Value` is the `Backend` |
| `LoadModel` | `LoadModel(path string, opts ...LoadOption) core.Result` | Load via specified or default backend |
| `LoadTrainable` | `LoadTrainable(path string, opts ...LoadOption) core.Result` | Load and assert `TrainableModel` |
| `BackendDeviceInfo` | `BackendDeviceInfo(name string) (DeviceInfo, bool)` | Accelerator info for a `DeviceInfoProvider` backend |

Fallible functions return `core.Result` — `OK bool` with the payload in `Value`, or the error in `Value` on failure.

### Platform preference

`Default()` walks a fixed preference order and returns the first backend whose `Available()` is true:

```
metal > rocm > llama_cpp > (any other registered available backend)
```

Metal is preferred on Apple Silicon for direct GPU-memory access; ROCm is preferred on Linux. If none of the preferred backends are available, any registered backend reporting `Available() == true` is used. With nothing registered, `Default()` returns a failed Result (`no backends registered`); with backends registered but none available, `no backends available`.

### LoadModel routing

`LoadModel` is the primary consumer entry point. It resolves the backend then delegates:

```go
// Explicit backend (bypasses Default())
r := inference.LoadModel("/models/model.gguf", inference.WithBackend("rocm"))

// Auto-detect (uses Default())
r := inference.LoadModel("/models/gemma-4-e2b-it-4bit")
if !r.OK {
    log.Fatal(r.Error())
}
m := r.Value.(inference.TextModel)
```

When `WithBackend()` is set, `LoadModel` looks up the named backend directly and fails if it is not registered or not available. Otherwise it calls `Default()`.

---

## How engines register

Each engine calls `inference.Register()` from an `init()` gated by its build tags, so registration only compiles on the target platform:

```go
// engine/metal/inference_register.go  —  //go:build darwin && arm64
package native

import "dappco.re/go/inference"

func init() { inference.Register(metalBackend{}) }
```

```go
// engine/hip/register_rocm.go  —  //go:build linux && amd64
package hip

import "dappco.re/go/inference"

func init() { inference.Register(&rocmBackend{}) }
```

The application blank-imports the engine to trigger `init()`:

```go
import (
    "dappco.re/go/inference"
    _ "dappco.re/go/inference/engine/metal" // registers "metal" on darwin/arm64
    _ "dappco.re/go/inference/engine/hip"   // registers "rocm" on linux/amd64
)
```

Because the engine package is guarded by build tags (with a portable stub for other platforms), the blank import stays satisfiable everywhere while only the matching engine compiles in.

---

## Implementing a new backend

To add a new engine (a new GPU runtime or inference server), implement the `Backend` interface and, optionally, `TrainableModel` / `AttentionInspector` / `VisionModel` / `DeviceInfoProvider`.

### Step 1: Implement Backend

```go
package mybackend

import (
    core "dappco.re/go"
    "dappco.re/go/inference"
)

type myBackend struct{}

func NewBackend() inference.Backend { return &myBackend{} }

func (b *myBackend) Name() string { return "mybackend" }

func (b *myBackend) Available() bool {
    return checkHardware() // false when the driver/hardware is absent
}

func (b *myBackend) LoadModel(path string, opts ...inference.LoadOption) core.Result {
    cfg := inference.ApplyLoadOpts(opts)
    model, err := loadWeights(path, cfg) // allocate GPU memory, set up KV cache...
    if err != nil {
        return core.Fail(core.E("mybackend.LoadModel", "load weights", err))
    }
    return core.Ok(model)
}
```

`LoadModel` returns `core.Ok(model)` on success and `core.Fail(core.E(...))` on failure — never a `(TextModel, error)` tuple.

### Step 2: Implement TextModel

Every method on the `TextModel` interface must be implemented. `Generate` and `Chat` return `iter.Seq[Token]`; `Classify`, `BatchGenerate`, `Err`, and `Close` return `core.Result`.

```go
func (m *myModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
    cfg := inference.ApplyGenerateOpts(opts)
    return func(yield func(inference.Token) bool) {
        for i := 0; cfg.MaxTokens == 0 || i < cfg.MaxTokens; i++ {
            if ctx.Err() != nil {
                m.lastErr = core.Fail(core.E("mybackend.Generate", "context", ctx.Err()))
                return
            }
            tok := m.decodeNext()
            if !yield(tok) {
                return // caller broke out of the range loop
            }
            if tok.ID == m.eosTokenID {
                return
            }
        }
    }
}

func (m *myModel) Err() core.Result { return m.lastErr } // OK Result on clean EOS
```

- **Chat** applies the model's native chat template before decoding — do not expose template logic to the consumer.
- **Classify** runs one forward pass per prompt (no autoregressive loop); populate `ClassifyResult.Logits` only when `cfg.ReturnLogits` is true. Return `core.Ok([]inference.ClassifyResult{...})`.
- **BatchGenerate** returns `core.Ok([]inference.BatchResult{...})`; per-prompt failures go in `BatchResult.Err`, not the outer Result.

### Step 3: Register with build tags

```go
// register.go  —  //go:build linux && amd64
package mybackend

import "dappco.re/go/inference"

func init() { inference.Register(NewBackend()) }
```

### Step 4 (optional): Support training

The in-tree engines expose LoRA SFT through the **`engine.TrainerModel`** seam (in `dappco.re/go/inference/engine`): the loaded model implements `OpenTrainer(cfg inference.TrainingConfig) (engine.Trainer, error)`, and the returned `engine.Trainer` owns the frozen base, the trainable LoRA weights, and the optimiser state — a caller drives `Step`/`Save`. The trained tensors never cross the package boundary; only the on-disk adapter does.

```go
tr, ok := model.(engine.TrainerModel)
if !ok { /* engine has no trainer */ }
trainer, err := tr.OpenTrainer(inference.TrainingConfig{LoRA: inference.LoRAConfig{Rank: 8, Alpha: 16}})
for step := 0; step < steps; step++ {
    loss, _ := trainer.Step(batch) // one AdamW step
}
_ = trainer.Save("/models/lora/domain-v1")
```

The root package also defines an older capability interface, `TrainableModel` (`ApplyLoRA`/`Encode`/`Decode`/`NumLayers`) with an `Adapter` return, which `LoadTrainable` asserts — but no in-tree engine implements it today. Prefer the `engine.Trainer` seam.

### Step 5 (optional): Support attention inspection

```go
func (m *myModel) InspectAttention(ctx context.Context, prompt string, opts ...inference.GenerateOption) (*inference.AttentionSnapshot, error) {
    // Run prefill, then read Q/K vectors from the KV cache.
    return &inference.AttentionSnapshot{
        NumLayers:    m.numLayers,
        NumHeads:     m.numKVHeads,
        SeqLen:       seqLen,
        HeadDim:      m.headDim,
        Keys:         keys, // [layer][head] → flat []float32
        Architecture: m.arch,
    }, nil
}
```

---

## Model discovery

`Discover` walks a directory tree for model directories — useful for model-selection UIs or inventory tools.

```go
func Discover(baseDir string) iter.Seq[DiscoveredModel]
```

A valid model directory contains `config.json` (parsed for `model_type` and optional quantisation fields) and at least one `.safetensors` file. The walk is **recursive** (every subdirectory under `baseDir`) and also probes `baseDir` itself, so a direct model path works. It is lazy — `break` stops the scan early.

```go
// Scan a models directory
for m := range inference.Discover("/path/to/models/") {
    fmt.Printf("%s — %s (%d files, %s)\n", m.Path, m.ModelType, m.NumFiles, m.Format)
}

// Check a single model directory
for m := range inference.Discover("/path/to/models/gemma3-1b") {
    fmt.Printf("arch=%s quant=%d-bit\n", m.ModelType, m.QuantBits)
}
```

---

## Registered backends in this repository

| Backend | Package | Platform | Registration tag |
|---------|---------|----------|------------------|
| `metal` | `engine/metal` (package `native`) | darwin/arm64 | `//go:build darwin && arm64` |
| `rocm`  | `engine/hip` (package `hip`)       | linux/amd64  | `//go:build linux && amd64` |

**metal** — no-cgo Apple GPU engine dispatching MLX's compiled Metal kernels via the objc runtime; ICB replay path for decode/diffusion. Implements `VisionModel`; trains via the `engine.Trainer` seam.

**rocm** — native ROCm/HIP engine on Linux/amd64 (GGUF today; the legacy `llama-server` bridge is behind a build tag). A portable stub reports unavailable on all other platforms.
</content>
