<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# state/agent_memory.go — Wake / Sleep / Fork lifecycle

**Package**: `dappco.re/go/inference/state`
**File**: `go/state/agent_memory.go`
**Aliased into**: `dappco.re/go/inference` (as `AgentMemory*` for the
historical naming consumers expect)

## What this is

The portable contract for **persisting and restoring live model state**
without binding to a concrete storage backend. A runtime that implements
`Session` can be told to write its current KV/context as a durable
"bundle", and a runtime that implements `Forker` can re-spawn a session
from a bundle written earlier — possibly on a different machine, possibly
much later, possibly from a knowledge-pack `.mp4` that was scanned in by
phone camera.

Three lifecycle verbs, four DTOs, two interfaces. Nothing else.

## DTOs

| Type | Role |
|------|------|
| `Ref` | URI-first identity for a durable state span — bundle + index + sampler/model identity + token/byte ranges. The thing you keep in your filesystem / DB / cold-storage index to point at one wake target. |
| `WakeRequest` | "Restore prefix from this URI into this session." Carries the model + tokenizer + adapter + runtime identity for compatibility checking; `Store` is an opaque runtime handle (deliberately not JSON-serialised). |
| `WakeResult` | "I restored N prefix tokens from this bundle/index, B blocks, K block size." Returned by `Session.WakeState`. |
| `SleepRequest` | "Persist the current session state to this URI, parented to that earlier URI." `ReuseParentPrefix` enables append-mode: a new bundle that shares prefix blocks with its parent — `O(delta)` writes, not full re-encode. |
| `SleepResult` | "I wrote N tokens across B blocks (R reused from parent), here is the new Ref." |

`Store any` on both Wake/Sleep requests is the explicit escape hatch for
backend-owned handles (State video encoder, file log writer, S3 client) that
the JSON serialisation layer doesn't need to see.

`Adapter` and `Runtime` are metadata fields, not dependency hooks. They let
orchestration decide whether waking a saved prefix is safe after adapter or
runtime settings change; the concrete backend still owns the final restore.

## Interfaces

```go
type Session interface {
    WakeState(ctx, WakeRequest) (*WakeResult, error)
    SleepState(ctx, SleepRequest) (*SleepResult, error)
}

type Forker interface {
    ForkState(ctx, WakeRequest) (Session, *WakeResult, error)
}
```

`Session.WakeState` restores into an **existing** session. `Forker.ForkState`
**creates** a new live session from durable state — used when you want
two divergent continuations from the same parent prefix without disturbing
the original. ForkState returns both the new Session and the wake result
so callers can either keep operating on the fork directly or hand it back
through a registry.

## Aliases

Consumers historically used `AgentMemory*` names (the concept predates
the package split). These are kept as type aliases so existing callers
compile without rewriting:

```go
type AgentMemoryRef         = Ref
type AgentMemoryWakeRequest = WakeRequest
type AgentMemoryWakeResult  = WakeResult
type AgentMemorySleepRequest = SleepRequest
type AgentMemorySleepResult = SleepResult
type AgentMemorySession     = Session
type AgentMemoryForker      = Forker
```

The `inference` parent package re-exports these via `identity.go` so a
consumer importing only `dappco.re/go/inference` sees `AgentMemoryRef`
without needing the `state` subpackage import.

## Where it's implemented

- `go-mlx` — Metal-backed `Session` + `Forker`. The reference
  implementation, with KV-block-level append, parent-prefix reuse, and
  State video `.mp4` packaging. See `go-mlx/docs/memory/agent_memory.md`.
- `go-rocm` — planned mirror for AMD/ROCm.
- `go-cuda` — planned mirror for NVIDIA/CUDA.

## Why URI-first

Storage policy lives at the URI scheme, not in the contract.

- `state://aurelius/meditations` — QR-video knowledge pack
- `file:///var/lib/coreagent/bundles/abc123/` — local filestore
- `s3://lethean-bundles/2026-05/agent-7/` — object storage
- `memory://test/fixture-1` — in-memory test harness

A runtime that knows how to dial the URI handles the bytes; the contract
doesn't care which one ships first or which one ships best.

## Why no streaming Wake API

`WakeResult` reports counts (tokens / blocks / bytes), not a streaming
channel. The bytes go into the runtime's own KV cache before the result
returns — by the time you have a `WakeResult`, the session is ready to
generate. The streaming progress story is owned by `probe.go` (probe
events emitted during wake) rather than by this DTO.

## Used by

- `go-mlx/cmd/violet` — sidecar exposes Wake/Sleep/Fork over Unix socket
- LTHN project seeds — app/CLI orchestration can wake a per-project context,
  append observations, then sleep a child state or fall back to a text summary.
- `go-ai/ai/book_state_demo.go` — teacher/student demo uses WakeResult →
  `BookState` (the demo's user-facing context shape)
- `go-mlx/pkg/memvid` — deprecated compatibility path for older State video
  encoder/decoder imports
- `core/ide` (planned) — agent inspector panel reads bundle index for
  the "what's in my brain right now" UI

## Validated benchmark

92k-token book loaded into context from cold (runner not preloaded) in
**55.2s** including bundle decode + KV restore — see
`project_local_inference_topology.md`. The same bundle re-restored from
warm cache: **998ms** for a chapter, **2.15s** for the full book.
