<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# probe.go — observability bus DTOs

**Package**: `dappco.re/go/inference`
**File**: `go/probe.go`

## What this is

The portable shape for **runtime telemetry events** that backends emit during a session. Probes are the "what's happening inside the model right now" signal — used by go-ml's scoring engine, the core/ide attention inspector, and the eval/bench pipelines.

A backend implements `ProbeSink` to receive probes, or emits via package-injected sink for in-process subscribers. No transport policy in this file — just the DTOs.

## Event kinds

```go
ProbeEventToken          // every generated token
ProbeEventLogits         // raw logits (when ReturnLogits set)
ProbeEventEntropy        // per-step sampling entropy
ProbeEventSelectedHeads  // which attention heads fired
ProbeEventLayerCoherence // per-layer activation alignment
ProbeEventRouterDecision // MoE expert routing decisions
ProbeEventResidual       // residual-stream magnitude
ProbeEventCachePressure  // KV cache fill / eviction
ProbeEventMemoryPressure // GPU allocator state
ProbeEventTraining       // SFT/LoRA/GRPO step events
```

## Phases

```go
ProbePhasePrefill   // initial prompt forward pass
ProbePhaseDecode    // autoregressive generation
ProbePhaseTraining  // SFT/LoRA/GRPO loop
```

## Event payload

`ProbeEvent` carries `Kind` + `Phase` + per-event payload (numeric + label maps). The full shape is small and self-describing — `ProbeEventToken` includes the token id/text; `ProbeEventLayerCoherence` includes a per-layer float; `ProbeEventRouterDecision` includes expert indices and weights.

## ProbeSink

```go
type ProbeSink interface {
    EmitProbe(event ProbeEvent)
}
```

Implemented by:

- `go-ml/agent_eval.go` — collects probes into eval reports
- `core/api` SSE handler — streams probes to core/ide
- in-process test fixtures that just accumulate events

A backend with no `ProbeSink` injected emits to a no-op default.

## Why a separate file

Probes are an extension surface, not a core capability. A minimal backend (CPU llama fallback) emits nothing but still satisfies TextModel. A research-grade backend (go-mlx with attention inspection + MoE routing) emits dozens of events per generated token. The shape is portable so consumers don't pin to one backend.

## Related

- [capability.md](capability.md) — `CapabilityProbeEvents` / `CapabilityAttentionProbe` / `CapabilityLogitProbe`
- `go-mlx/docs/observability/probe.md` (planned) — backend wiring
- `go-ml/docs/agent/agent_eval.md` (planned) — probe collection in eval
