<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# contracts.go — extension interfaces

**Package**: `dappco.re/go/inference`
**File**: `go/contracts.go`

## What this is

The "everything beyond TextModel" surface. Each capability that some
backends support but not all is its own interface, discovered by type
assertion. A backend implements only the interfaces it can deliver; a
consumer probes via `if x, ok := model.(inference.Y); ok { ... }`.

This file is the source of truth for what extensions exist; the
implementations live in backends.

## Capability interfaces

| Interface | What it adds |
|-----------|--------------|
| `SchedulerModel` | queue-aware Schedule(req) → handle + token stream — for serving loops with cancellation + batching |
| `CancellableModel` | CancelRequest(id) — abort an in-flight generation |
| `CacheService` | CacheStats + WarmCache + ClearCache — prompt-cache management |
| `EmbeddingModel` | Embed(req) — vector embeddings |
| `RerankModel` | Rerank(req) — cross-encoder document scoring |
| `ReasoningParser` | ParseReasoning(tokens, text) — extract chain-of-thought from `<think>` channels |
| `ToolParser` | ParseTools(tokens, text) — extract structured tool-call output |
| `ModelPackInspector` | InspectModelPack(path) — validate a model dir without loading weights |

## Request / Result DTOs

| Type | Role |
|------|------|
| `RequestHandle` | id + model identity + labels — what a Schedule call returns to track a request |
| `RequestCancelResult` | id + cancelled bool + reason |
| `ScheduledRequest` | id + model + prompt/messages + sampler + labels — input to a scheduler |
| `ScheduledToken` | request_id + token + per-request metrics + labels — what the scheduler streams |
| `CacheBlockRef` | portable handle for one cache block — id, kind, model/adapter/tokenizer hash, token range, size, encoding |
| `CacheStats` | block count + memory/disk bytes + hits/misses/evictions + hit rate + restore latency |
| `CacheWarmRequest` / `CacheWarmResult` | warm a prompt's cache + report which blocks are ready |
| `EmbeddingRequest` / `EmbeddingResult` / `EmbeddingUsage` | input strings → vectors + token accounting |
| `RerankRequest` / `RerankScore` / `RerankResult` | query + documents → scored documents |
| `ReasoningSegment` / `ReasoningParseResult` | visible text vs reasoning channels |
| `ToolCall` / `ToolParseResult` | visible text vs tool calls |
| `ModelPackInspection` | path, format, model identity, supported bool, capabilities, notes |

## Agent memory aliases (live here for import convenience)

```go
type AgentMemoryRef          = state.Ref
type AgentMemoryWakeRequest  = state.WakeRequest
type AgentMemoryWakeResult   = state.WakeResult
type AgentMemorySleepRequest = state.SleepRequest
type AgentMemorySleepResult  = state.SleepResult
type AgentMemorySession      = state.Session
type AgentMemoryForker       = state.Forker
```

Importing `dappco.re/go/inference` gives you the memory lifecycle
shape without needing a separate `inference/model/state` import. The
state package owns the real types; this file just re-exports them.

## How a consumer probes capabilities

```go
m, _ := inference.LoadModel(path).Value.(inference.TextModel)

if sched, ok := m.(inference.SchedulerModel); ok {
    handle, tokens, err := sched.Schedule(ctx, req)
    // serve queue
}
if cancel, ok := m.(inference.CancellableModel); ok {
    _ = cancel.CancelRequest(ctx, oldRequestID)
}
if cache, ok := m.(inference.CacheService); ok {
    stats, _ := cache.CacheStats(ctx)
}
if embed, ok := m.(inference.EmbeddingModel); ok {
    result, _ := embed.Embed(ctx, req)
}
```

## How a backend opts in

In `engine/metal` (example):

```go
// the native text model already implements TextModel
// — add Schedule to also implement SchedulerModel:
func (m *nativeTextModel) Schedule(ctx, req) (RequestHandle, <-chan ScheduledToken, error) {
    // …
}
```

No registration step. The type assertion at the call site is the only
discovery mechanism. Backends that *don't* implement an interface
simply fail the type check; consumers fall back to whatever default
they have.

## Why type-assertion not method-set

Different engines are at different stages. `engine/metal` may have
SchedulerModel before `engine/hip`; `engine/hip` may ship CacheService
earlier than `engine/metal`. Forcing every backend to stub out every
interface would make TextModel a 50-method monster and silently degrade
— type assertion lets each engine grow at its own pace and the consumer
explicitly handles the "not available" path.

## Related

- [inference.md](inference.md) — the base TextModel + Backend
- [capability.md](capability.md) — `CapabilityReport` for static
  introspection of what a backend claims to support
- [../state/agent_memory.md](../state/agent_memory.md) — the real
  agent-memory types (these are aliases)
- [../openai/services.md](../openai/services.md) — wire types that
  carry EmbeddingResult / RerankResult / CacheStats over HTTP
