<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# training.go — TrainableModel + Adapter contracts

**Package**: `dappco.re/go/inference`
**File**: `go/training.go`

## What this is

The **low-level contract seam** for fine-tuning: attach a LoRA adapter, tokenise, and report layer count. Backends that can train implement `TrainableModel`; the rest don't. Same pattern as the inspection interfaces in `contracts.go` — opt-in via type assertion. The optimiser, gradient computation and tensor creation live in the backend package itself; the actual training pipelines (LoRA SFT, self-distillation, MTP tuning) run through the `train/` package and `cmd/lem sft` / `ssd` / `tune`, driving a model through this seam.

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

## Adapter

```go
type Adapter interface {
    TotalParams() int        // sum of injected adapter weight elements
    Save(path string) error  // persist adapter weights to a safetensors file
}
```

The concrete type lives in each backend; consumers hold an `Adapter` returned by `ApplyLoRA` to report parameter count and save weights.

## TrainableModel

```go
type TrainableModel interface {
    TextModel
    ApplyLoRA(cfg LoRAConfig) Adapter  // inject LoRA into target projections
    Encode(text string) []int32        // tokenise via the model's tokeniser
    Decode(ids []int32) string         // detokenise
    NumLayers() int                    // transformer depth (sizes per-layer LoRA)
}
```

`ApplyLoRA` returns the `Adapter`; the training loop in `train/` uses `Encode` / `Decode` to build batches and `NumLayers` to size per-layer matrices. Backend-specific training operations (optimisers, gradient computation, tensor creation) are provided by the backend package directly (e.g. `engine/metal` for Apple GPU, `engine/hip` for AMD).

## LoadTrainable

```go
inference.LoadTrainable(path, opts...) core.Result
```

Top-level helper — same pattern as `LoadModel` but typed to `TrainableModel`; on `r.OK` the `Value` is a `TrainableModel`. A model loaded from a backend that doesn't implement `TrainableModel` is closed and the Result fails with `backend %q does not support training` (where `%q` is the model type).

## Why training is a separate interface

Most callers never train — they want inference. Forcing every backend to stub out training methods bloats the contract. Inference-only backends (HTTP, llama.cpp subprocess) literally cannot train; they implement `TextModel` and that's all anyone needs.

## Implemented by

- `engine/metal` — the in-repo training surface (LoRA apply + tokenise + layer count) the `train/` pipelines drive
- `engine/hip` — the AMD/ROCm mirror

## Related

- [capability.md](capability.md) — `CapabilityLoRATraining`, `CapabilityDistillation`, `CapabilityGRPO`
- [dataset.md](dataset.md) — `DatasetStream` + the `TrainingConfig` / `DistillConfig` / `GRPOConfig` envelopes the pipelines consume
- `go/train/` — the SFT / self-distillation / MTP-tune implementations (`cmd/lem sft`, `ssd`, `tune`)
- [../state/identity.md](../state/identity.md) — `AdapterIdentity` portable identity
