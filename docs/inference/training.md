<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# training.go — TrainableModel + Adapter contracts

**Package**: `dappco.re/go/inference`
**File**: `go/training.go`

## What this is

The contract surface for **fine-tuning** — LoRA adapter management, gradient steps, save/load. Backends that can train implement `TrainableModel`; the rest don't. Same pattern as the inspection interfaces in `contracts.go` — opt-in via type assertion.

## LoRAConfig

```go
type LoRAConfig struct {
    Rank       int       // decomposition rank (default 8)
    Alpha      float32   // scaling factor (default 16)
    TargetKeys []string  // projection suffixes (default: q_proj, v_proj)
    BFloat16   bool      // mixed-precision adapter weights
}
```

`DefaultLoRAConfig()` — Rank=8, Alpha=16, TargetKeys=["q_proj","v_proj"], BFloat16=false.

Backends that don't honour `BFloat16` ignore the field (still emit a probe event so the caller knows).

## Adapter

```go
type Adapter interface {
    // implementation-defined methods; the concrete type is backend-specific
    // (e.g. *metal.LoRAAdapter for go-mlx)
}
```

`Adapter` is intentionally **interface-empty** — the concrete type lives in each backend. Consumers hold an `Adapter` reference for save/load/swap but never inspect its methods directly. The backend exposes the operations through its `TrainableModel`.

## TrainableModel

```go
type TrainableModel interface {
    TextModel
    AttachAdapter(cfg LoRAConfig) (Adapter, error)
    DetachAdapter() error
    Step(ctx, batch) (StepResult, error)        // one optimiser step
    SaveAdapter(path string) error
    LoadAdapter(path string) error
}
```

(Exact method shapes are backend-defined; this file holds the umbrella interface signature.)

## LoadTrainable

```go
inference.LoadTrainable(path, opts...) core.Result
```

Top-level helper — same pattern as `LoadModel` but typed to `TrainableModel`. Backends that don't support training return a "trainable not supported on backend X" error.

## Why training is a separate interface

Most callers never train — they want inference. Forcing every backend to stub out training methods bloats the contract. Inference-only backends (HTTP, llama.cpp subprocess) literally cannot train; they implement `TextModel` and that's all anyone needs.

## Implemented by

- `go-mlx` — full training surface: SFT, LoRA, GRPO, distillation
- `go-rocm` — planned mirror
- `go-ml` does NOT implement TrainableModel — it consumes trainable models via go-mlx

## Related

- [capability.md](capability.md) — `CapabilityLoRATraining`, `CapabilityDistillation`, `CapabilityGRPO`
- `go-mlx/docs/training/sft.md` (planned) — reference SFT implementation
- `go-mlx/docs/training/lora_adapter.md` (planned) — LoRA Adapter concrete shape
- `go-mlx/docs/training/grpo.md` (planned) — reasoning training loop
- `go-mlx/docs/training/distill.md` (planned) — teacher/student distillation
- [../state/identity.md](../state/identity.md) — `AdapterIdentity` portable identity
