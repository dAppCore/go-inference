<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# dataset.go — DatasetStream contract

**Package**: `dappco.re/go/inference`
**File**: `go/dataset.go`

## What this is

The smallest possible pull-based dataset contract shared by training, evaluation, distillation, and reasoning rollouts. One sample at a time, optional reset, optional length. Backends and consumers agree on this shape so a dataset assembled in go-ml flows directly into go-mlx training without conversion.

## DatasetSample

```go
type DatasetSample struct {
    Text      string             // raw text (continuation pretraining)
    Prompt    string             // user prompt (SFT, instruct)
    Response  string             // assistant response (SFT target)
    Reasoning string             // chain-of-thought (GRPO, distillation)
    Messages  []Message          // multi-turn conversation
    Labels    map[string]string  // routing / filtering metadata
}
```

A sample carries whichever fields the task needs. SFT samples populate Prompt + Response. GRPO samples add Reasoning. Eval samples often only use Messages.

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

## DatasetSized

Optional. Streams that know their length up-front report it for progress UI / cosine LR schedules.

## DatasetConfig (planned umbrella)

The capability surface in `capability.go` mentions `CapabilityEvaluation` + `CapabilityDistillation` + `CapabilityGRPO`. Each consumes a DatasetStream. The eval/bench/distill/grpo config DTOs live in the consuming packages (go-mlx, go-ml) rather than here — this file is just the stream contract.

## Why one interface for everything

The temptation is to have `TrainingDataset`, `EvalDataset`, `DistillDataset` — different shapes per task. We resist. A single `DatasetStream.Next() → DatasetSample` covers every task because `DatasetSample` is wide enough that each consumer reads the fields it cares about. New tasks add fields to DatasetSample without churning consumers.

## Implemented by

- `go-mlx/dataset_stream.go` — in-process iterator over MLX-format files
- `go-ml/ingest.go` — DuckDB / Parquet ingestion → DatasetStream
- `go-mlx/cmd/violet` — wraps an HTTP-streamed dataset
- test fixtures via in-memory slice wrappers

## Consumed by

- `go-mlx/sft.go` — supervised fine-tuning loop
- `go-mlx/grpo.go` — reasoning training loop
- `go-mlx/distill.go` — teacher/student distillation
- `go-mlx/eval.go` — evaluation runner
- `go-ml/agent_eval.go` — scoring engine eval

## Related

- [training.md](training.md) — TrainableModel consumes DatasetStream in Step
- `go-mlx/docs/training/dataset_stream.md` (planned) — reference iterator
- `go-ml/docs/scoring/ingest.md` (planned) — go-ml's dataset assembly path
