<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# Continuous batching for `lem serve` — design memo + landable slice-1 skeleton

Status: design + landable skeleton (`serving/interleave`, engine-free). This
memo maps "continuous batching" onto the serving topology go-inference
actually ships, records what already exists versus what is genuinely missing,
scopes the slice ladder, and hands over the ranked open questions a wiring
round needs answered. Style follows `docs/design-tiered-kv.md`.

## The one-line ask vs the one-line reality

"Continuous batching" (vLLM/oMLX sense): a new request joins an
already-decoding batch mid-stream, per token, so N in-flight generations keep
a GPU saturated instead of draining to empty between batches.

Today's `lem serve` has **no batch, and no scheduler, at all.** `serving/serve.go`
mounts `serving/compat.NewMuxWithAdmin(resolver, ...)` directly; every OpenAI /
Anthropic / Ollama handler in `serving/compat/mux.go` calls
`resolver.ResolveModel` then the model's `Chat`/`Generate` **inline in that
request's own `net/http` connection goroutine** — no queue, no concurrency
cap, no fairness, no backpressure. Whatever `net/http`'s goroutine-per-connection
scheduling and the engine's own locking happen to produce is what you get.
Two scheduler packages already exist in the tree and are wired into **nothing**
(`grep` for their import paths outside their own package returns zero hits) —
see the reconciliation in §b.

## (a) What "continuous batching" means against OUR engine, honestly

| CB mechanism | what it needs | what we have |
|---|---|---|
| N sessions share ONE GPU forward pass per decode step (KV caches co-resident, one dispatch advances all N) | an engine primitive that batches across the **session** dimension | **Not built.** `engine/metal/decode_batched_session.go`'s `stepTokensBatchedDense` batches **tokens of one session** (prefill / MTP-draft verification) — a different axis. No code path folds N *different* sessions' single-token steps into one dispatch. |
| N independent sessions decode concurrently, interleaved for fairness | scheduling only, no engine change | **Already safe at the engine level.** `engine/metal/attention.go:375` — the resident-weight-buffer mutex explicitly "guards concurrent sessions; the decode itself is single-goroutine" — i.e. multiple sessions running concurrently is an anticipated, protected shape; each session already owns its own `engine.Session` (`engine/model.go:openSession`/`stream`). What was missing is the **serve-layer scheduler** — §b, slice 1. |
| a new request admitted mid-decode without waiting for the batch to drain | live/async admission | **Not built at serve.** Two dormant candidates exist (`serving/schedule`, `serving/scheduler`); neither does live async admission *and* per-token fairness together — see below. |

The dense/gemma decode path already pays close to the cheapest possible
per-token dispatch cost: `engine/metal/decode_forward_arch_icb.go` records the
whole per-token forward pass as an indirect command buffer **once** and
replays it — roughly one CB submission per token. The COMPOSED/hybrid path
(gated-delta, Qwen3.5) does not have this yet: `docs/handover.md`'s
2026-07-12 entry (`#23 COMPOSED DEVICE SEAM`) measures **~70+ per-projection
CB round-trips/token at ~330µs each (~10µs of it real compute)** on that path
and names "GPU-resident whole-token orchestration" as the unbuilt next rung.
**True batched-decode-across-sessions (slice 3) is a strictly bigger version
of exactly that unbuilt rung** — it needs N sessions' KV *and* activations
co-resident with ONE dispatch stepping all of them, harder than the
single-session whole-token fuse the handover doc already flags as
"design-worthy, brief a research pass before building." Nothing in this round
attempts it — see the requirement list in §b.

## (b) The slice ladder

### Reconciliation first — two dormant schedulers already exist

Before adding a third scheduler package, both existing ones need naming and
an honest gap analysis, because their overlap is the actual design decision
here, not a detail:

| | `serving/schedule` | `serving/scheduler` | `serving/interleave` (this round) |
|---|---|---|---|
| origin | Tier-B lift from go-ai (`680f3e0`), perf-tuned (259→4 allocs/op) | built in this repo, wraps `inference.TextModel` | new, this round |
| shape | `Engine.Run(ctx, []Request, Stepper, onToken)` — **one blocking call over a static, pre-known slice** | `New(model, Config)` — persistent `MaxConcurrent`-worker goroutine pool draining a shared job queue | `New(Config)` — persistent admission loop; `Submit` is async, any time |
| admission | dual limit: `MaxConcurrency` **and** `MaxBatchTokens` (a real token budget) | count only: `MaxConcurrent` | dual limit, mirrors `schedule`'s shape (count **and** an optional token budget) |
| live mid-decode admission | **No** — `requests` is fixed at the `Run` call; nothing can join after it starts except from that same pre-known queue | **Yes** — `Schedule` can be called any time, a worker picks it up | **Yes** — `Submit` any time |
| per-request cancel | No (only whole-`Run` ctx) | Yes (`CancelRequest`) | Yes (`Cancel`) |
| per-request backpressure | N/A (single `onToken` callback, no per-request channel) | Yes (`StreamBuffer`-bounded channel per job, isolated by the worker's own goroutine) | Yes (same isolation — see below) |
| fairness mechanism | the `Stepper` you inject decides — pluggable point for a future batched-GPU stepper (slice 3) | Go's own goroutine scheduler (N workers race) | budgeted admission + FIFO queue (see the round-robin note below) |
| stats | none | rich (`ProbeSink` events: queued/start/first_token/cancelled/complete, queue depth, latencies) | counters (Submitted/Admitted/Completed/Cancelled/Active/Queued) |
| consumers today | **zero** | **zero** | zero (new) |

`schedule.Engine`'s `Stepper` seam is genuinely well-shaped for slice 3 later
(a `Stepper.Step(ctx, running []*Seq)` that does one batched GPU call), and a
*naive* Stepper that internally loops each running `Seq`'s own per-session
generator would already satisfy slice 1 semantics with **zero changes** to
that package — but its `Run` call shape (a fixed slice, one blocking call)
cannot serve live HTTP traffic without a wrapper that keeps re-invoking it,
which is most of the work anyway. `serving/scheduler` is closer to a live
server today (persistent workers, real cancellation, real per-job
backpressure, rich probes) but has no token budget and, being a full
`inference.TextModel` wrapper, cannot directly drive a `continuity`-woken
session's `handle.Generate` (that's a `model/state/session.Session`, not a
`TextModel`).

**Neither is deleted or modified this round** (scope discipline — this is a
design + a new, additive package). The reconciliation itself is the
deliverable for the operator: §"Ranked open questions" below asks which of
the three survives wiring.

### Slice 1 — request-level scheduler, no engine change (`serving/interleave`, landed this round)

The brief's language is literal per-token lockstep round-robin. Building that
literally — one central loop calling `next()` on every active generator once
per round, in fixed order — was tried on paper and **rejected**, for a
concrete reason found while designing it, not a hunch:

> a newly-admitted generator's **first** pull also pays its **prefill**
> (`engine.TextModel.stream` calls `sess.PrefillTokens(ids)` *before* the
> decode loop starts yielding). A synchronous per-round `next()` call on a
> lockstep driver would block **every other active session's turn** for the
> whole prefill duration of whichever session was just admitted — for a long
> prompt, that is 10-100x a single decode step. Literal lockstep round-robin
> is only safe once §"Slice 2" (prefill riding off the round-robin path) also
> exists; building it first would ship a real serving regression, not a win.

`serving/interleave.Engine` instead gives **each admitted request its own
goroutine** (the same isolation pattern `serving/scheduler.worker`/`run`
already uses and that this repo has already proven): the request's `Source`
(a ctx-bound `iter.Seq[inference.Token]` — literally the existing per-session
stream, `engine.TextModel.Generate`/`Chat` or a `continuity`-woken session's
`handle.Generate`, unmodified) is ranged over on its own goroutine, and each
produced token is delivered to that request's own bounded output channel via
a `select` against the request's own context. Consequences:

- **No engine change** — `Source` wraps whatever the caller already calls.
- **Fairness is admission-level, not token-level lockstep**: a dual budget
  (`MaxActive` concurrent sessions **and** an optional `MaxBatchTokens`
  running-prompt-token budget, mirroring `schedule.Engine`'s admission shape)
  gates a FIFO queue, so no request can starve another out of ever being
  admitted, and no unbounded pile of concurrent decode loops can be created.
  This is a materially weaker guarantee than lockstep ("session A never gets
  token *N+1* before session B gets token *N*") — flagged as open question 1.
- **Backpressure is per-request, not global**: one slow consumer's channel
  filling up blocks only *that* request's own goroutine (the `select` against
  its own `ctx.Done()`), never another request's tokens. This is strictly
  better than a shared-loop lockstep design would have given for free.
- **Cancellation** is a `context.CancelFunc` per request (`Submit`'s ctx is
  the parent); `Cancel(id)` cancels a queued request before it ever runs
  (its channel closes with zero tokens, `Stats().Cancelled` counts it) or an
  active one mid-stream (observed both by the `select` here and, one layer
  deeper, by the engine's own per-token `ctx.Err()` check inside
  `decodeFromPrefilled`'s `emit` — belt and braces, no engine change needed
  either way).
- **Stats** are atomic counters (`Submitted`/`Admitted`/`Completed`/
  `Cancelled`/`Active`/`Queued`), safe to read concurrently — smaller than
  `scheduler`'s `ProbeSink` event stream by design (a skeleton, not a
  replacement for that probe machinery).

See `go/serving/interleave/interleave.go` for the implementation and
`go/serving/interleave/interleave_test.go` for the fake-generator test suite
(admission budget, cancellation of both queued and active requests,
backpressure isolation, clean `Close()` shutdown — race-clean).

**Not wired into `serve.go`** — that is deliberately the next slice, per the
brief. Wiring it requires picking a winner from the reconciliation table
above (open question 1) and deciding how `continuity`'s per-conversation
`busy` lock interacts with a `Source` that can now be cancelled out from under
a resident conversation mid-turn (open question 3).

### Slice 2 — batched-prefill admission (not built, requirements only)

New request's prefill must ride alongside other sessions' decode steps
without stalling them. Slice 1's goroutine-per-request design already gets
this **for free** in its async form (a new request's prefill happens on its
own goroutine, so it never blocks anyone else's `select`) — the requirement
that remains open is specifically for a design that wants **literal lockstep
round-robin** (open question 1): if the operator wants that stronger fairness
property, it needs an explicit two-phase admission — prefill a newly-admitted
session on its own goroutine first, and only splice it into the round-robin
rotation once its first token is ready. Requirements:

1. A "ready" signal per newly-admitted session (prefill-complete plus first
   token in hand) distinct from "actively rotating."
2. The round-robin driver must poll/select non-blockingly across N
   in-flight prefills without reflection-based dynamic `select` (a fan-in
   channel, or the per-request-goroutine model this round already uses,
   feeding a shared "arrived" channel instead of per-request output channels
   directly — a real design fork, not sketched further here).
3. Decide whether "prefill occupies a `MaxActive` slot" or a separate,
   smaller prefill-concurrency limit — a large prefill is a real GPU
   resource user (its own PrefillTokens dispatch, see `engine/metal/decode_forward_arch.go`)
   and unlimited concurrent prefills would defeat the K-tier engine-level
   protections currently guarding only sessions, not the prefill dispatch
   itself.

### Slice 3 — true batched decode step across sessions (the engine campaign, not built)

Enumerated requirements, deliberately not attempted this round (hard fence:
`engine/metal`, `engine/hip`, `model/*` are out of scope):

1. **Multi-session KV co-residency.** Each session's KV cache currently lives
   in its own `engine.Session`; a batched step needs N sessions' KV blocks
   addressable in one dispatch — closer to `kv/kvtier`'s tier/placement
   machinery (unwired today, see §c) than to anything in `engine/metal`.
2. **A batched-Stepper.** `schedule.Engine`'s `Stepper` interface is already
   the right seam (`Step(ctx, running []*Seq) (StepResult, error)` — one call
   advances the whole running set); slice 3 is "write a `Stepper` whose
   `Step` is one real batched GPU dispatch," not a scheduler change.
3. **Ragged-batch handling.** Sessions decode at different positions (KV
   length) and request different `MaxNewTokens` — the GPU op needs to accept
   a ragged batch (padding, or per-lane position vectors) the way
   `stepTokensBatchedDense` already does for *tokens within one session*
   (`embs [][]byte, basePos int` — note it takes ONE `basePos`, not one per
   lane; extending to N independent per-lane positions is new surface, not a
   parameter tweak).
4. **The whole-token GPU-resident orchestration this needs already has a
   named, unbuilt prerequisite** one layer down: `docs/handover.md #23`'s
   "GPU-resident whole-token orchestration — weights uploaded once, whole
   token encoded in ONE CB, recurrence on device" for a *single* session.
   Slice 3 is that, plus N-way session multiplexing inside the same CB. Build
   the single-session version first; it is the smaller, already-flagged
   research pass.
5. **A cost model that says it's worth it.** The ICB dense path already pays
   ~1 CB/token — the ceiling slice-3 chases is CB-count reduction under
   concurrent *sessions*, which only matters if per-session concurrent
   decode (available today, per §a) is actually CB-count-bound rather than
   compute-bound at realistic concurrency. Nobody has measured this — see §d.

### Slice 3 implementation audit (2026-07-12, task #35 slice 3)

The implementation audit reached the brief's explicit deep-surgery stop. No
optional capability is bound and the scheduler remains on its existing
single-request path. In particular, this audit does **not** call K ordinary
session steps behind a method named "batch": that would increment an
engagement counter without performing one shared forward.

The blocking ownership boundary is evidenced in the current tree:

1. `go/engine/metal/arch_session.go:35-86` gives each `ArchSession` one embedded
   `archDecodeState` and one scalar `pos`. Session construction creates that
   state and its paged KV allocation together at
   `go/engine/metal/arch_session.go:645-668`.
2. The available batched primitive is a method on **one** such state and takes
   one scalar position: `stepTokensBatchedDense(embs, basePos)` at
   `go/engine/metal/decode_batched_session.go:381-387`. Its documented and
   implemented contract is consecutive positions
   `[basePos, basePos+K)` written into that state's per-layer caches, not K
   independent positions in K independent caches.
3. The attention plumbing derives every row's live length and cache landing
   from that shared `basePos` (for example
   `go/engine/metal/decode_batched_session.go:1137-1140` and
   `go/engine/metal/decode_batched_session.go:1321-1350`). Substituting a
   position vector alone is insufficient: those calls still bind cache buffers
   through the receiver's single `archDecodeState`.
4. The existing engine wrapper opens a fresh independent engine session for
   every generation (`go/engine/model.go:195-201`). There is no owner above
   those sessions that can encode one command buffer against all of their KV
   allocations.

The smallest honest engine seam remains an optional capability with three
operations: prepare/prefill an independent decode lane, advance a slice of
prepared lanes in one `Step(ctx, lanes)` forward, and retire a lane. A step
result must contain one token-or-terminal result per input lane and expose a
monotonic batched-forward counter so tests can prove the K-way implementation
fired. `serving/scheduler` interleave mode can probe that capability and use a
single step coordinator only when it is present; serial, batch, and the
existing interleave source path need no changes. Capability construction must
also check `LTHN_CB_STEP`: the exact value `0` returns the ordinary model
without the optional wrapper, so an interface assertion fails rather than a
claimed batch method silently falling back internally.

Implementing that seam in Metal first requires a multi-session state owner
which separates shared immutable weights from lane-owned mutable state, then
changes the whole batched attention/cache path to accept per-lane KV buffer
bindings, positions, live lengths, ring slots, and optional PLE state. Only
after that owner exists can the required varied-fill fixture compare K
independent serial sessions byte-for-byte with one counter-guarded K-way
forward. That is the deep surgery named by the brief; adding the public
interface or scheduler probe before any engine can truthfully bind it would be
dead API rather than a buildable partial.

## (c) Memory / KV budget interaction with the multi-model + tiered work

Two KV-budget primitives exist and are **unwired into anything** (same status
`docs/design-tiered-kv.md` recorded for them): `kv/budget` (`FitsMemory` /
`FitsWindow` — token-count-to-placement decisions, RFC §6.2/§6.11/§6.16) and
`kv/kvtier` (GPU/CPU/Disk tier eviction policy, pin, promote-on-touch).
`serving/interleave`'s admission budget (`MaxBatchTokens`) is a **token-count
proxy**, not a byte budget — it does not know how many KV bytes a session
actually costs (that depends on architecture: hidden size, layer count,
quant, and whether the arch shares KV across layers, per the E4B skip
mentioned in `docs/release-v0.12.0-DRAFT.md`). Any slice-2/3 wiring that
raises `MaxActive` meaningfully (more concurrent sessions = more resident KV)
should gate through `kv/budget.FitsMemory` per admission candidate rather
than trust a token-count ceiling alone — otherwise a serve process running
multiple models (the multi-model work) or several long-context conversations
concurrently can admit past the device's real memory budget while every
token-count check still reports "fits." This round's skeleton does **not**
wire that in (it would be a fabricated byte-cost model with no receipt) —
flagged as open question 2.

## (d) Measured estimates where possible, stated assumptions where not

Measured, from the existing handover receipts (cited, not re-derived):

- ICB-replayed dense decode: ≈1 CB submission/token (from the ICB-replay
  design itself — `decode_forward_arch_icb.go`'s `stepBody` replays a
  **recorded** command buffer per token rather than re-encoding).
- COMPOSED/hybrid decode: ~70 CB round-trips/token × ~330µs/CB ≈ 23ms of pure
  CB dispatch overhead per token, of which ~10µs/CB (~0.7ms total) is real
  compute — i.e. that path is currently **CPU-dispatch-bound**, not
  compute-bound (`docs/handover.md #23`).

Assumed, not measured (no engine benchmarking was run this round — the fence
excludes `engine/*`, and a real number needs a live multi-session harness
this round did not build):

- Whether N *dense-path* concurrent sessions (ICB-replay, ~1 CB/token each)
  saturate the Metal command queue's real concurrent-dispatch capacity before
  saturating the GPU's compute, at realistic `MaxActive` values (4-8). If the
  queue already pipelines N small CBs efficiently, slice 1's goroutine
  concurrency alone captures most of the *dense-path* win, and slice 3's
  value is concentrated on the COMPOSED path (where the 70-CB/token tax is
  large enough that ANY session sharing GPU idle time between CBs is a real
  throughput opportunity slice 1 cannot reach without true session-batching).
- Whether Go's goroutine scheduler introduces measurable unfairness between
  concurrent sessions under `serving/interleave` in practice (the `residentBufMu`
  contention path in `engine/metal/attention.go` is the only shared-mutable
  state identified between concurrent sessions; it is locked only on a
  weight-buffer cache miss, not per-token, so contention should be rare, but
  "should be" is an assumption, not a receipt).

**A follow-up instrumented experiment (N concurrent `serving/interleave`
sessions against a live dense checkpoint, xctrace GPU-interval capture per
`lethean-perf`'s trace-forensics recipe) is the natural next round if slice 1
is wired — it would convert both assumptions above into receipts and tell
slice 3's actual priority.**

## Ranked open questions for the operator

1. **Which scheduler survives wiring?** `serving/schedule`, `serving/scheduler`,
   and this round's `serving/interleave` now all sit dormant with materially
   different shapes (see the reconciliation table). Wiring one into
   `serve.go` without a decision here risks landing a fourth ad-hoc glue
   layer. My read: `interleave` for live HTTP admission (it is the only one
   that is both async-live and has a token budget), with `schedule.Engine`'s
   `Stepper` seam reserved for slice 3 — but this is the operator's call, not
   mine to make unilaterally by picking a wiring order.
2. **Byte-budget admission (§c) — build now or defer?** `kv/budget.FitsMemory`
   exists and is a small integration; not wiring it means `MaxBatchTokens` is
   an honest-but-soft ceiling until it is.
3. **`continuity.Manager`'s `busy` lock vs. cancellable sessions.** Today a
   resident conversation cannot be cancelled mid-turn (continuity's `acquire`
   returns an error if `conv.busy`, and there is no cancel path at all — the
   turn always runs to completion or failure). If `interleave.Cancel` is
   wired to a continuity-backed request, the resident session's `busy` flag
   and eventual `finishTurn` sleep-and-evict need to handle "the turn was
   cut short mid-generation, but the client got a partial reply" — the
   `continuity.go` doc comment already treats a mid-stream disconnect as
   "correct, not a compromise" for the *stateless-client-drop* case; an
   explicit `Cancel` is the same shape but currently has no caller in that
   package to exercise it.
4. **Literal lockstep vs. admission-fairness (§b slice 1 vs slice 2).** Is
   admission-level fairness (this round's `interleave.Engine`) an acceptable
   reading of "round-robin/fair token-interleave," or is the stronger
   per-token lockstep guarantee load-bearing enough to justify building
   slice 2's two-phase admission before wiring anything into `serve.go`?

## What shipped this round

- `docs/design-continuous-batching.md` (this memo).
- `go/serving/interleave/` — `Engine`, `Source`/`Stream`, `Config`, `Stats`,
  `Submit`/`Cancel`/`Close`; unit-tested with fake generators (admission
  budget, both cancellation paths, per-request backpressure isolation, clean
  shutdown), `-race` clean; **not wired into `serve.go`**.

## Resolution — one scheduler, three modes, wired (supersedes open question 1)

Open question 1 ("which scheduler survives wiring?") was answered **modes, not a
survivor pick**. The three dormant packages collapsed into ONE —
`go/serving/scheduler` (the generic name, the most-tested base) — with a `Mode`
selecting the discipline. `serving/schedule` and `serving/interleave` are
DELETED; their behaviour + tests moved into the unified package.

| Mode | source package | what it kept | admission | fairness | backpressure |
|---|---|---|---|---|---|
| `serial` | `serving/scheduler` (verbatim) | bounded-queue worker pool, `ProbeSink` events, `CancellableModel` fallback, all the perf-tuned alloc contracts | count only (`MaxConcurrent` workers + `MaxQueue`) | Go scheduler across N workers | per-request `StreamBuffer` channel |
| `batch` | `serving/schedule` (algorithm + tests) | dual concurrency+token budget, oversize-prompt retirement, continuous top-up, the `MaxRunning` witness | dual: `MaxConcurrent` **and** `MaxBatchTokens` | lockstep — one coordinator advances the whole running set one token/round | **global** (one shared lane) |
| `interleave` | `serving/interleave` (verbatim) | live async admission, per-request goroutine isolation, `findAndCancel`, close-drain | dual: `MaxConcurrent` **and** optional `MaxBatchTokens` | admission-level (FIFO queue) | **per-request** (isolated) |

**Best-version-wins, losers deleted:**
- The one submission surface is `inference.SchedulerModel.Schedule(ctx, req)` —
  the mux already prefers it (`serving/compat/mux.go` `forEachCompatToken`), so
  wiring is a resolver decorator, not a mux change.
- The `source`/`stream` per-request-stream abstraction and the FIFO
  `MaxQueue` bounded-queue backpressure are **shared** by batch and interleave —
  batch adopted interleave's bounded queue (the better of the two; the former
  `schedule.Engine` had none, its slice was the queue). `serial`'s
  `generateOptions`/`cloneLabels`/`millis` request helpers are the package's one
  copy, used by every mode.
- Mode-specific requirement, **probed at construction, fail-closed**: `batch`
  (and `interleave` with `MaxBatchTokens > 0`) needs the model to expose
  `inference.TokenizerModel` so prompt tokens can be counted for the budget — a
  model without `Encode` returns an error from `scheduler.New` (and, in serve,
  from `ResolveModel`) rather than silently running an inert budget.

**Serve wiring (`-scheduler serial|batch|interleave`):** a `Scheduler` field on
`serving.ServeConfig` fed by the flag. Unset (default) leaves the request path
**byte-for-byte unchanged** — `RunServe` builds NO wrapper, `host.resolver`
stays the bare hot-swap resolver, nothing new runs on the hot path (the wrap is
inside `if core.Trim(cfg.Scheduler) != ""`). Set = `host.resolver` is decorated
with a `schedulerResolver` that presents every resolved model through a
persistent `scheduler.Model` in the chosen mode; a boot notice prints the active
mode. The scheduler is built once per underlying model and reused, so concurrent
requests share its queue/running-set; `CloseEngine` tears the engine down
without touching the model (whose lifecycle stays with the resolver beneath).

**Multi-model (`-models-config`) — noted, not built.** The scheduler wrap lives
only in `RunServe`'s single-model path; `runMultiModelServe` is untouched.
Wiring it there needs a scheduler instance **per resident model** (the
`multiModelResolver` returns a different model per id and evicts on LRU/idle-TTL),
so a single shared `schedulerResolver` (one `lastModel` slot) is the wrong shape:
the decorator would need a per-model-id scheduler cache keyed to the registry's
residency, building on admit and calling `CloseEngine` on eviction so an evicted
model's scheduler goroutine does not outlive it. That is a registry-lifecycle
change (a scheduler is a live goroutine, unlike the stateless welfare/policy
resolver wraps), deliberately out of this single-model slice.

**Multi-model resolution — wired (#35, supersedes the paragraph above).**
`multiModelResolver` gained exactly the per-model-id scheduler cache the
paragraph above anticipated: a `schedCfg *scheduler.Config` set once
(`setScheduler`, wired from `runMultiModelServe` when `-scheduler` is set) and
a `sched *scheduler.Model` field on `modelEntry`. `ensureResident` builds the
scheduler right after a model loads (fail-closed: a capability-probe failure
rolls the entry back to non-resident and Closes the just-loaded model rather
than serving it unscheduled), and `evict` — the ONE function LRU/budget
eviction, idle-TTL sweep, and explicit unload all already funnelled through —
calls `sched.CloseEngine()` before `model.Close()`, so a request genuinely
in flight on the evicting model is cancelled and its stream ends cleanly
before the engine underneath it is torn down. `closeSchedulers` (deferred by
`runMultiModelServe`) drains whatever is still resident at serve shutdown, the
multi-model twin of the single-model path's `sched.close()`.
One pre-existing gap this surfaced and fixed in `serving/scheduler` itself:
serial mode's `CloseEngine` was a no-op (nothing ever closed the worker
queue) — harmless for single-model serve (the process exits anyway) but a
real per-eviction goroutine leak once a scheduler is built and torn down
repeatedly over a serve process's life. It now joins its worker pool
synchronously, the same guarantee batch/interleave already held.

**One semantics the unification could NOT fully collapse (evidenced):** the
former `schedule.Engine`'s int-based `Stepper` seam (`StepResult.Tokens
map[string]int`, `go/serving/schedule/schedule.go:71` in the pre-merge tree) is a
token-**count** scheduler; the live serve path needs token-**payload**
(`inference.Token`) delivery per request. A static, slab-allocated count
scheduler (`schedule.go`'s `seqStore`/`tokenSlab`, sized from `len(requests)` up
front) and a live payload coordinator with unbounded submission cannot share
buffer management without one regressing the other, so batch mode is a live
coordinator (`go/serving/scheduler/batch.go`) that keeps schedule's admission
**policy** (dual budget, oversize retirement, continuous top-up, cap clamping —
all re-tested) while delivering real tokens via `iter.Pull`. Batch's lockstep
also inherits schedule's structural cost the memo flagged above: a
newly-admitted request's first pull pays its prefill inside the round (briefly
stalling peers), and a stalled consumer backpressures the whole set — which is
exactly why `interleave` (per-request goroutines) remains the other mode rather
than being folded into batch.
