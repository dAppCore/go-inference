<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# metrics-sink

WithMetricsSink delivers ONE generation's final GenerateMetrics as its
stream completes — a request-scoped alternative to the global m.Metrics()
read, which is last-writer-wins under concurrent generations against the
same model (see the doc comment on GenerateMetrics.MetricsSink).

Running two Generate/Chat calls concurrently against one loaded model is a
real, supported shape here: LoadConfig.ParallelSlots exists precisely for
it, and the engine's session layer (go/engine/prompt_reuse.go) opens an
independent session per call rather than serialising through one. What is
NOT safe to read concurrently is the model's shared last-error/last-metrics
state — hence the sink. This example demonstrates two goroutines generating
at once, each capturing its OWN usage through its own sink; it checks
m.Err() only once, after both have joined, and notes below why that single
check is a best-effort wrap-up rather than a per-request guarantee.

## Run

```sh
go run ./pkg/metrics-sink -model ~/models/gemma-4-e2b-it-4bit
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
