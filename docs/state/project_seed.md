<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# state/project_seed.go — project-seed workflow helpers

**Package**: `dappco.re/go/inference/model/state`
**File**: `go/model/state/project_seed.go`
**Aliased into**: `dappco.re/go/inference`

## What this is

Small backend-neutral helpers for the LTHN project-memory flow. They do not
load models or write bytes. They produce consistent `WakeRequest` and
`SleepRequest` values, decide whether a continuation should persist state or
fall back to summary text, and compare a saved `Bundle` with a wake request
before a runtime tries to restore KV.

The concrete runtime still owns wake/sleep. `engine/metal` restores KV blocks
on Apple GPU; `engine/hip` and future engines can implement the same `Session`
and `Forker` contracts without copying app policy.

## ProjectSeed

`NewProjectSeed` normalises the URI set for a project (defaulting `BaseURI` to
`state://projects` and `ProjectID` to `default` when unset):

```go
seed := state.NewProjectSeed(state.ProjectSeedOptions{
    BaseURI:   "state://lthn/projects",
    ProjectID: "core/go-inference",
})
```

The default seed entry becomes:

```text
state://lthn/projects/core/go-inference/seed
state://lthn/projects/core/go-inference/seed/bundle
state://lthn/projects/core/go-inference/seed/index
```

`seed.WakeRequest(...)` carries model, tokenizer, adapter, runtime, and labels
into a normal `WakeRequest`.

## Continuation modes

`seed.PlanContinuation(...)` lowers product policy into concrete request shape:

| Mode | Result |
|------|--------|
| `ProjectSeedStateCheckpoint` | returns a `SleepRequest` with parent refs and `ReuseParentPrefix=true` |
| `ProjectSeedReuseCurrent` | no sleep request; caller records findings elsewhere and keeps the current seed |
| `ProjectSeedSummaryWindow` | no sleep request; caller writes summary text and starts a fresh window |
| `ProjectSeedHybrid` | returns a sleep request and marks that summary text should also be written |

This keeps "reply" separate from persistence. A background agent can wake,
append observations, sleep a new child state, and never emit an operator-facing
answer.

## Compatibility

`CheckWakeCompatibility(bundle, req)` checks the high-risk identity fields
before a wake:

- model hash, architecture, layer count, quantisation, and context capacity
- tokenizer hash and chat template
- adapter presence/hash/path/rank
- runtime backend/cache-mode changes as warnings, not hard blockers

When the report is incompatible, orchestration should prefer summary/new-window
or hybrid fallback. `SkipCompatibilityCheck` is still available for explicit
research runs and returns a compatible report with a warning.
