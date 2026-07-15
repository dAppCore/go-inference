---
title: Interfaces
description: TextModel, Backend, TrainableModel, Adapter, and the optional capability interfaces.
---

# Interfaces

The root `inference` package defines the contract. Two interfaces are core (`TextModel`, `Backend`); the rest are optional capabilities an engine advertises and consumers discover by type assertion (`TrainableModel`, `AttentionInspector`, `VisionModel`, `DeviceInfoProvider`).

Fallible methods return `core.Result` (from `dappco.re/go`), not the Go `(T, error)` tuple. A `Result` has `OK bool` and `Value any`; on failure `Value` holds the error (also reachable via `r.Error()`).

## TextModel

The primary inference interface. Every loaded model satisfies this.

```go
type TextModel interface {
    Generate(ctx context.Context, prompt string, opts ...GenerateOption) iter.Seq[Token]
    Chat(ctx context.Context, messages []Message, opts ...GenerateOption) iter.Seq[Token]
    Classify(ctx context.Context, prompts []string, opts ...GenerateOption) core.Result
    BatchGenerate(ctx context.Context, prompts []string, opts ...GenerateOption) core.Result
    ModelType() string
    Info() ModelInfo
    Metrics() GenerateMetrics
    Err() core.Result
    Close() core.Result
}
```

### Generate

```go
Generate(ctx context.Context, prompt string, opts ...GenerateOption) iter.Seq[Token]
```

Streams tokens for a raw text prompt. The caller ranges over the returned iterator; the engine controls token production. The iterator stops on end-of-sequence (EOS), context cancellation, or reaching `MaxTokens`.

After the iterator is exhausted, call `Err()` to check for errors — `iter.Seq` cannot carry errors alongside values (the `database/sql` `Row.Err()` pattern).

```go
for tok := range m.Generate(ctx, "The capital of France is", inference.WithMaxTokens(32)) {
    fmt.Print(tok.Text)
}
if r := m.Err(); !r.OK {
    log.Fatal(r.Error())
}
```

### Chat

```go
Chat(ctx context.Context, messages []Message, opts ...GenerateOption) iter.Seq[Token]
```

Streams tokens from a multi-turn conversation. The engine applies the model's native chat template internally — Gemma, Qwen3, and Llama all use distinct formats, so template application belongs in the engine, not every consumer.

```go
msgs := []inference.Message{
    {Role: "system", Content: "You are a helpful assistant."},
    {Role: "user", Content: "What is 2+2?"},
}
for tok := range m.Chat(ctx, msgs, inference.WithMaxTokens(64)) {
    fmt.Print(tok.Text)
}
```

### Classify

```go
Classify(ctx context.Context, prompts []string, opts ...GenerateOption) core.Result
```

Runs batched prefill-only inference. Each prompt gets a single forward pass and the token at the last position is sampled — no autoregressive decoding loop. The Result carries `[]ClassifyResult` in `Value` when OK.

Set `WithLogits()` to receive the full vocab-sized logit array in each result. Off by default to avoid large allocations.

```go
cr := m.Classify(ctx, prompts, inference.WithTemperature(0))
if !cr.OK {
    log.Fatal(cr.Error())
}
for _, r := range cr.Value.([]inference.ClassifyResult) {
    fmt.Printf("predicted: %s (id=%d)\n", r.Token.Text, r.Token.ID)
}
```

### BatchGenerate

```go
BatchGenerate(ctx context.Context, prompts []string, opts ...GenerateOption) core.Result
```

Runs batched autoregressive generation — each prompt decoded up to `MaxTokens`. The Result carries `[]BatchResult` in `Value` when OK. Per-prompt errors (context cancellation, OOM) are captured in each `BatchResult.Err` rather than aborting the whole batch.

### ModelType

```go
ModelType() string
```

The architecture identifier: `"gemma3"`, `"qwen3"`, `"llama3"`, etc. Read from the model's `config.json` at load time.

### Info

```go
Info() ModelInfo
```

Static metadata about the loaded model — architecture, vocabulary size, layer count, hidden dimension, quantisation. Called once after load, typically for logging or display.

### Metrics

```go
Metrics() GenerateMetrics
```

Performance metrics from the most recent inference operation. Valid after `Generate` (once the iterator is exhausted), `Chat`, `Classify`, or `BatchGenerate`. Includes token counts, prefill/decode timing, throughput, GPU memory, and whether a thinking budget was force-closed.

### Err

```go
Err() core.Result
```

The error state from the last `Generate`/`Chat` call. Check after the iterator stops to distinguish normal EOS (returns an **OK** Result) from an error (a **failed** Result carrying the error in `Value`).

### Close

```go
Close() core.Result
```

Releases all resources — GPU memory, KV caches, subprocesses. Returns an OK Result on success, a failed Result carrying the error otherwise. Must be called when the model is no longer needed.

---

## Backend

A named inference engine that can load models.

```go
type Backend interface {
    Name() string
    LoadModel(path string, opts ...LoadOption) core.Result
    Available() bool
}
```

### Name

```go
Name() string
```

The registry key: `"metal"` or `"rocm"` for the in-tree engines. This is the string consumers pass to `WithBackend()`.

### LoadModel

```go
LoadModel(path string, opts ...LoadOption) core.Result
```

Loads a model from a filesystem path — a safetensors directory (`config.json` + `.safetensors`) for Metal, or a GGUF file for ROCm. Returns a ready `TextModel` in the Result's `Value` when OK.

### Available

```go
Available() bool
```

Reports whether this engine can run on the current hardware — `false` when the GPU runtime or device is absent, so `LoadModel`/`Default()` fail cleanly rather than crashing. The build tags govern whether the engine compiles in at all; `Available()` is the runtime gate.

---

## TrainableModel

Extends `TextModel` with a LoRA fine-tuning surface. Not every engine supports training — use a type assertion or `LoadTrainable()` to check.

```go
type TrainableModel interface {
    TextModel

    ApplyLoRA(cfg LoRAConfig) Adapter
    Encode(text string) []int32
    Decode(ids []int32) string
    NumLayers() int
}
```

> **Note.** This is the older capability interface. The in-tree engines expose LoRA SFT instead through the `engine.TrainerModel` seam (`OpenTrainer(cfg inference.TrainingConfig) (engine.Trainer, error)`, in `dappco.re/go/inference/engine`), where `engine.Trainer` holds the frozen base, the trainable weights, and the optimiser state and the caller drives `Step`/`Save`. No in-tree engine implements `TrainableModel.ApplyLoRA` today, so `LoadTrainable` will fail against the current metal/hip models — prefer probing `engine.TrainerModel`. The `TrainableModel` / `Adapter` / `LoRAConfig` types remain defined in the root contract.

### ApplyLoRA

```go
ApplyLoRA(cfg LoRAConfig) Adapter
```

Injects LoRA adapters into the target projection layers named by `cfg.TargetKeys`. Returns an `Adapter` holding references to the trainable parameters. The concrete type is engine-specific.

### Encode / Decode

```go
Encode(text string) []int32
Decode(ids []int32) string
```

Tokenise text into IDs and back, using the model's native tokeniser. Required by training pipelines that prepare input sequences.

### NumLayers

```go
NumLayers() int
```

The number of transformer layers — used to size per-layer LoRA matrices and validate target layers.

### Checking for training support

Via type assertion on an existing model:

```go
tm, ok := model.(inference.TrainableModel)
if !ok {
    log.Fatal("backend does not support training")
}
```

Via the convenience loader (loads, asserts, closes on mismatch):

```go
r := inference.LoadTrainable("/path/to/model/")
if !r.OK {
    log.Fatal(r.Error())
}
tm := r.Value.(inference.TrainableModel)
defer tm.Close()
```

`LoadTrainable` calls `LoadModel` internally and returns a failed Result if the resulting model does not implement `TrainableModel` — closing the model first, so there is no resource leak.

---

## Adapter

Holds the trainable LoRA parameters applied to a model. The concrete type is engine-specific. Note that `Adapter` uses the plain `(T, error)` convention, not `core.Result`.

```go
type Adapter interface {
    TotalParams() int
    Save(path string) error
}
```

### TotalParams

```go
TotalParams() int
```

The total number of trainable parameters across all LoRA adapter layers.

### Save

```go
Save(path string) error
```

Writes the adapter weights to a safetensors file. Used to checkpoint adapter state during training or export a fine-tuned adapter for later inference via `WithAdapterPath()`.

---

## AttentionInspector

Optional interface for extracting attention-level data — used for Q/K Bone Orientation analysis. Discovered by type assertion; engines that do not support it are unaffected.

```go
type AttentionInspector interface {
    InspectAttention(ctx context.Context, prompt string, opts ...GenerateOption) (*AttentionSnapshot, error)
}
```

Runs a prefill pass and extracts Q and/or K vectors from the KV cache, returned as an `AttentionSnapshot` indexed by layer, head, and position.

```go
if inspector, ok := model.(inference.AttentionInspector); ok {
    snap, err := inspector.InspectAttention(ctx, "analyse this prompt")
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("layers=%d heads=%d seq_len=%d\n", snap.NumLayers, snap.NumHeads, snap.SeqLen)
}
```

`AttentionInspector` is a defined optional capability: `serving.InferenceAdapter.InspectAttention` forwards to the underlying `TextModel` when it implements the interface. Note that the in-tree engines (`engine/metal`, `engine/hip`) do not implement `InspectAttention` today — the interface and the adapter delegation are in place for a producing engine.

---

## VisionModel

Optional interface a `TextModel` implements when the **loaded checkpoint** accepts image content. It is a live probe, not a static family declaration — a vision-capable family can ship a snapshot without the vision tower.

```go
type VisionModel interface {
    AcceptsImages() bool
}
```

The compat handlers use this to reject image requests against text-only models, and only engines reporting `true` serve the `Message.Images` on a turn.

---

## DeviceInfoProvider

Optional interface a `Backend` implements when it can describe its accelerator without loading a model.

```go
type DeviceInfoProvider interface {
    DeviceInfo() DeviceInfo
}
```

Reachable through the package helper, which returns `false` when the backend is unregistered or does not expose device information:

```go
if info, ok := inference.BackendDeviceInfo("metal"); ok {
    fmt.Printf("%s (%s), %d GiB\n", info.Name, info.Architecture, info.MemorySize>>30)
}
```
</content>
