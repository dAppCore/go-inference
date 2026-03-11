---
title: Interfaces
description: TextModel, Backend, TrainableModel, and AttentionInspector interface reference.
---

# Interfaces

go-inference defines four interfaces. Two are core (`TextModel`, `Backend`) and two are optional extensions (`TrainableModel`, `AttentionInspector`).

## TextModel

The primary inference interface. Every loaded model satisfies this.

```go
type TextModel interface {
    Generate(ctx context.Context, prompt string, opts ...GenerateOption) iter.Seq[Token]
    Chat(ctx context.Context, messages []Message, opts ...GenerateOption) iter.Seq[Token]
    Classify(ctx context.Context, prompts []string, opts ...GenerateOption) ([]ClassifyResult, error)
    BatchGenerate(ctx context.Context, prompts []string, opts ...GenerateOption) ([]BatchResult, error)
    ModelType() string
    Info() ModelInfo
    Metrics() GenerateMetrics
    Err() error
    Close() error
}
```

### Generate

```go
Generate(ctx context.Context, prompt string, opts ...GenerateOption) iter.Seq[Token]
```

Streams tokens for a raw text prompt. The caller ranges over the returned iterator; the backend controls token production. The iterator stops on end-of-sequence (EOS), context cancellation, or hitting `MaxTokens`.

After the iterator is exhausted, call `Err()` to check for errors. This follows the `database/sql` `Row.Err()` pattern — `iter.Seq` cannot carry errors alongside values.

```go
for tok := range m.Generate(ctx, "The capital of France is", inference.WithMaxTokens(32)) {
    fmt.Print(tok.Text)
}
if err := m.Err(); err != nil {
    log.Fatal(err)
}
```

### Chat

```go
Chat(ctx context.Context, messages []Message, opts ...GenerateOption) iter.Seq[Token]
```

Streams tokens from a multi-turn conversation. The model applies its native chat template internally — Gemma3, Qwen3, and Llama3 all use distinct formats, so template application belongs in the backend rather than in every consumer.

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
Classify(ctx context.Context, prompts []string, opts ...GenerateOption) ([]ClassifyResult, error)
```

Runs batched prefill-only inference. Each prompt gets a single forward pass and the token at the last position is sampled. This is the fast path for classification tasks — no autoregressive decoding loop. Used by go-i18n for domain labelling.

Set `WithLogits()` to receive the full vocab-sized logit array in each result. This is off by default to avoid large allocations.

```go
results, err := m.Classify(ctx, prompts, inference.WithTemperature(0))
for _, r := range results {
    fmt.Printf("predicted: %s (id=%d)\n", r.Token.Text, r.Token.ID)
}
```

### BatchGenerate

```go
BatchGenerate(ctx context.Context, prompts []string, opts ...GenerateOption) ([]BatchResult, error)
```

Runs batched autoregressive generation. Each prompt is decoded up to `MaxTokens`. Unlike `Classify`, this runs the full decoding loop for every prompt. Per-prompt errors (context cancellation, OOM) are captured in `BatchResult.Err` rather than aborting the entire batch.

### ModelType

```go
ModelType() string
```

Returns the architecture identifier: `"gemma3"`, `"qwen3"`, `"llama"`, etc. Read from the model's `config.json` at load time.

### Info

```go
Info() ModelInfo
```

Returns static metadata about the loaded model — architecture, vocabulary size, layer count, hidden dimension, and quantisation details. Called once after load, typically for logging or display.

### Metrics

```go
Metrics() GenerateMetrics
```

Returns performance metrics from the most recent inference operation. Valid after `Generate` (once the iterator is exhausted), `Chat`, `Classify`, or `BatchGenerate`. Includes token counts, prefill/decode timing, throughput, and GPU memory usage.

### Err

```go
Err() error
```

Returns the error from the last `Generate` or `Chat` call. Check this after the iterator stops to distinguish normal end-of-sequence from errors. Returns `nil` when generation completed successfully.

### Close

```go
Close() error
```

Releases all resources — GPU memory, KV caches, subprocesses. Must be called when the model is no longer needed.

---

## Backend

A named inference engine that can load models.

```go
type Backend interface {
    Name() string
    LoadModel(path string, opts ...LoadOption) (TextModel, error)
    Available() bool
}
```

### Name

```go
Name() string
```

Returns the registry key: `"metal"`, `"rocm"`, or `"llama_cpp"`. This is the string consumers pass to `WithBackend()`.

### LoadModel

```go
LoadModel(path string, opts ...LoadOption) (TextModel, error)
```

Loads a model from a filesystem path. The directory must contain `config.json` and one or more `.safetensors` weight files (HuggingFace safetensors layout). Returns a ready-to-use `TextModel`.

### Available

```go
Available() bool
```

Reports whether this backend can run on the current hardware. A backend may be registered unconditionally (in a shared binary) while still returning `false` on platforms where its GPU runtime is absent. The `Default()` function skips unavailable backends.

---

## TrainableModel

Extends `TextModel` with LoRA fine-tuning capabilities. Not all backends support training — use a type assertion or `LoadTrainable()` to check.

```go
type TrainableModel interface {
    TextModel

    ApplyLoRA(cfg LoRAConfig) Adapter
    Encode(text string) []int32
    Decode(ids []int32) string
    NumLayers() int
}
```

### ApplyLoRA

```go
ApplyLoRA(cfg LoRAConfig) Adapter
```

Injects LoRA adapters into the target projection layers specified by `cfg.TargetKeys`. Returns an `Adapter` that holds references to all trainable parameters. The concrete adapter type is backend-specific (e.g. `*metal.LoRAAdapter` for go-mlx).

### Encode

```go
Encode(text string) []int32
```

Tokenises text into token IDs using the model's native tokeniser. Required for training pipelines that need to prepare input sequences.

### Decode

```go
Decode(ids []int32) string
```

Converts token IDs back to text. The inverse of `Encode`.

### NumLayers

```go
NumLayers() int
```

Returns the number of transformer layers. Used by training code to configure layer-specific learning rates or to validate LoRA target layers.

### Checking for training support

Via type assertion on an existing model:

```go
tm, ok := model.(inference.TrainableModel)
if !ok {
    log.Fatal("backend does not support training")
}
```

Via the convenience function (loads and asserts in one step):

```go
tm, err := inference.LoadTrainable("/path/to/model/")
if err != nil {
    log.Fatal(err)
}
defer tm.Close()
```

`LoadTrainable` calls `LoadModel` internally and returns an error if the resulting model does not implement `TrainableModel`. It also closes the model before returning the error, so there is no resource leak.

---

## AttentionInspector

An optional interface for extracting attention-level data. Used for Q/K Bone Orientation analysis. Discovered via type assertion — backends that do not support attention inspection are entirely unaffected.

```go
type AttentionInspector interface {
    InspectAttention(ctx context.Context, prompt string, opts ...GenerateOption) (*AttentionSnapshot, error)
}
```

### InspectAttention

```go
InspectAttention(ctx context.Context, prompt string, opts ...GenerateOption) (*AttentionSnapshot, error)
```

Runs a prefill pass and extracts Q and/or K vectors from the KV cache. Returns an `AttentionSnapshot` containing the raw vectors indexed by layer, head, and position.

```go
if inspector, ok := model.(inference.AttentionInspector); ok {
    snap, err := inspector.InspectAttention(ctx, "analyse this prompt")
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("layers=%d heads=%d seq_len=%d\n", snap.NumLayers, snap.NumHeads, snap.SeqLen)
}
```

### Current implementations

- **go-mlx** — extracts post-RoPE K vectors (and optionally Q vectors) from the Metal KV cache after prefill
- **go-ml** — `InferenceAdapter.InspectAttention()` delegates via type assertion to the underlying `TextModel`

---

## Adapter

Holds trainable LoRA parameters applied to a model. The concrete type is backend-specific.

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

Returns the total number of trainable parameters across all LoRA adapter layers.

### Save

```go
Save(path string) error
```

Writes the adapter weights to a safetensors file at the given path. Used to checkpoint adapter state during training or to export a fine-tuned adapter for later inference via `WithAdapterPath()`.
