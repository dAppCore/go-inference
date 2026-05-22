<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# tuning.go — local discovery and autotune contracts

**Package**: `dappco.re/go/inference`
**File**: `go/tuning.go`

## What this is

Portable DTOs and interfaces for local setup UIs. Backends use these to expose
what a machine can do, propose model-load settings for different workloads, and
stream optional smoke-test results without leaking backend-specific types.

The important interfaces are:

```go
type MachineDiscoverer interface {
    DiscoverMachine(context.Context, MachineDiscoveryRequest) (*MachineDiscoveryReport, error)
}

type TuningPlanner interface {
    PlanTuning(context.Context, TuningPlanRequest) (*TuningPlan, error)
}
```

Discovery should be metadata-first: device facts, capabilities, cache modes,
and model-pack metadata where available. It should not load weights. Tuning is
separate and opt-in.

## Workloads

`TuningWorkload` is a stable string used in UI and persisted profiles:

- `chat`
- `coding`
- `long_context`
- `agent_state`
- `throughput`
- `low_latency`

## Candidate and profile

`TuningCandidate` records the concrete settings a UI can try or save: context
length, cache policy/mode, batch size, prefill chunk size, parallel slots,
allocator limits, model identity, adapter identity, and runtime identity.

After a smoke run, callers persist `TuningProfile`: key, candidate,
measurements, score, and labels.

## Model replace

`PlanModelReplace` is the conservative state decision helper:

- same model/runtime/adapter: reuse state
- same model/adapter but runtime settings changed: checkpoint state
- model or adapter changed: compact to summary/new window

This lets a UI change models or settings quickly while keeping the state flow
honest.

