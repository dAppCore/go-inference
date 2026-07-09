<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# dataset.go — DatasetStream contract

**Package**: `dappco.re/go/inference`
**File**: `go/dataset.go`

## What this is

The smallest pull-based dataset contract shared by training, evaluation, distillation, and reasoning rollouts. One sample at a time, optional reset. Every package agrees on this shape so a dataset assembled in `eval/datapipe` flows directly into a `train/` loop without conversion.

## DatasetSample

```go
type DatasetSample struct {
    Text      string             // raw text (continuation pretraining)
    Prompt    string             // user prompt (SFT, instruct)
    Response  string             // assistant response (SFT target)
    Reasoning string             // chain-of-thought (GRPO, distillation)
    Messages  []Message          // multi-turn conversation
    Format    string             // source-corpus row shape it was normalised from
    Labels    map[string]string  // routing / filtering metadata
}
```

A sample carries whichever fields the task needs. SFT samples populate Prompt + Response. GRPO samples add Reasoning. Eval samples often only use Messages. `Format` records the source row shape (`"text"`, `"openai_messages"`, `"sharegpt"`, `"prompt_response"`, `"alpaca"`, `"reasoning"`) — stamped by `train/dataset.LoadJSONL`, empty for samples built directly.

## DatasetStream

```go
type DatasetStream interface {
    Next() (DatasetSample, bool, error)
}
```

`Next` returns `(sample, ok, err)`. `ok=false` + `err=nil` = end of stream. Errors are terminal — the caller stops consuming.

## DatasetResetter

```go
type DatasetResetter interface {
    Reset() error
}
```

Optional. Streams that wrap an in-memory list or a seekable file implement Reset so training loops can run multiple epochs. Streaming-only sources (HF datasets streaming mode) don't.

## Eval / bench / training envelope

`dataset.go` no longer holds only the stream contract — it also carries the backend-neutral **config + report DTOs** the training and eval pipelines exchange (all plain JSON-tagged structs):

- **Batching**: `LossMask`, `Batch` (token IDs + attention/loss masks + samples)
- **Evaluation**: `EvalConfig`, `EvalMetrics`, `QualityProbe`, `QualityProbeResult`, `EvalReport`, and the `Evaluator` interface (`Evaluate(ctx, DatasetStream, EvalConfig) (*EvalReport, error)`)
- **Benchmark**: `BenchConfig`, `BenchReport`
- **Memory planning**: `MemoryPlan`, `ModelFitReport`
- **Training**: `TrainingConfig`, `TrainingMetrics`, `TrainingResult`, `DistillConfig` (embeds `TrainingConfig`), `GRPOConfig` (embeds `TrainingConfig`)

These are wire-stable shapes; the loops that produce and consume them live in `train/` and `eval/`.

## Why one interface for everything

The temptation is to have `TrainingDataset`, `EvalDataset`, `DistillDataset` — different shapes per task. We resist. A single `DatasetStream.Next() → DatasetSample` covers every task because `DatasetSample` is wide enough that each consumer reads the fields it cares about. New tasks add fields to DatasetSample without churning consumers.

## Implemented by

- `train/dataset/` — JSONL / corpus ingestion → `DatasetStream` (`LoadJSONL` stamps `DatasetSample.Format`)
- `eval/datapipe/` — evaluation data pipelines
- test fixtures via in-memory slice wrappers

## Consumed by

- `train/` — `sft.go` (supervised fine-tuning), `ssd.go` (self-distillation), `grpo/`, `distill/`
- `eval/` — evaluation + benchmark runners
- `agent/` — scoring/eval agent loop

## Related

- [training.md](training.md) — `TrainableModel`; the `train/` pipelines drive it over a `DatasetStream`
- `go/train/dataset/` — the reference JSONL loader
