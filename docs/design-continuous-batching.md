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

## Slice 3 — the multi-session owner, BUILT (#35, 2026-07-12)

The slice-3 audit above stopped at the deep-surgery fence; this round builds
across it. What shipped is a real multi-session state owner in `engine/metal`, a
neutral capability the scheduler probes, and receipts.

### The owner — `engine/metal/lane_set.go` (`laneSet`)

Shared vs lane-owned, the separation the audit demanded:

- **Shared immutable (weights).** Each lane is an `ArchSession` opened from the
  SAME `NativeTokenModel` (`openLaneSession` → `model.openSession(shards, headEnc)`),
  so the resident weight buffers — keyed by mmap address in the process-global
  `residentBufs` cache (`attention.go`), read-only during decode — are the SAME
  device buffers for every lane. No new sharing mechanism; the engine already
  shared weights across sessions, it lacked an owner ABOVE them.
- **Lane-owned mutable.** Each lane owns its `archDecodeState` (recorded-ICB
  decode caches, per-lane scratch), its scalar `pos`, its sampler/greedy state,
  its per-lane embed/PLE scratch.

The forward (`Step`) advances every active lane by one token through ONE shared
command buffer. The seam is `archICBReplay.encodeStepBody(enc, emb, pos, pli)`
(`decode_forward_arch_icb.go:663`), which replays a lane's recorded ICB into an
EXTERNAL encoder without committing (the MTP chained-decode path already uses
it). `Step` phase 1 produces each lane's token from its current hidden
(`greedyFromHiddenInPool` — the same per-lane op the serial loop runs); phase 2
replays every still-live lane's ICB into a single encoder and commits + waits
ONCE. K lanes cost one GPU submission, not K commit/wait round-trips, and the
GPU may pipeline the lanes' disjoint executions.

**Why it is byte-identical, not a K-serial-steps fake:** a lane replays its OWN
recorded ICB over its OWN caches at its OWN position — the exact GPU commands
`ArchSession.Step` runs for that lane alone. Lanes touch disjoint device buffers
(only read-only weights are shared), so fusing them into one command buffer
changes submission and scheduling, never arithmetic. A monotonic
`BatchForwardCount` (+1 per `Step`) proves the K-way path fired: the batched run
advances K lanes with the forward count of a SINGLE lane, not K× it.

Ragged admission is supported (`Prepare` between `Step`s). Prefill rides the same
per-token ICB `stepBody` the decode replays, so a lane's caches are populated
exactly as `Step` reads them regardless of which production prefill route a
batched prefill would take (batched-prefill admission is the pinned slice-2 rung).

### The capability seam

- `inference.BatchStepModel` (`go/batch_step.go`) — neutral, zero-dep, alongside
  `SchedulerModel`/`TokenizerModel`: `BatchStepAvailable() bool` +
  `OpenLaneSet(cfg) (LaneSet, error)` + the `LaneSet`/`LaneSpec`/`LaneStep`
  contract. `OpenLaneSet` REFUSES with a clear error when unavailable — never a
  silent serial degrade.
- `engine.TextModel` (`go/engine/batch_step.go`) implements it, delegating to the
  metal model's `LaneSetOpener` (structural — the backend never imports engine).
- **Kill switch:** `engine.batchStepKillSwitch()` reads `LTHN_CB_STEP`; the EXACT
  value `0` makes `BatchStepAvailable` report false and `OpenLaneSet` refuse. The
  gate is an availability method, NOT a wrapper type, deliberately: wrapping the
  model to add/remove the interface would strip its other optional capabilities
  (the registry capability-stripping bug class) — probe-then-check keeps every
  capability reachable on one value.

### Scheduler wiring — interleave mode probes it

`serving/scheduler` interleave mode probes `base.(inference.BatchStepModel)` +
`BatchStepAvailable()` + `TokenizerModel` at `New`; present ⇒ it builds a
`cbStepEngine` (`cb_step.go`) — a single goroutine that owns one `LaneSet` and
drives every admitted request through it (ragged `Prepare`, one shared `Step`
per round, per-request token channels, cancel, close). CB-eligible requests (raw
prompt, greedy) route there; chat (the neutral `TextModel` surface exposes no
template render) and non-greedy fall to the UNCHANGED plain interleave engine.
When the capability is absent — the common case, and whenever `LTHN_CB_STEP=0` —
`cbEngine` is nil and interleave mode is byte-for-byte the plain per-request
engine (52 existing scheduler tests still green; serial/batch modes untouched).

### Receipts

- **Byte-identity (counter-guarded), synthetic dense fixture** —
  `lane_set_test.go`: the same lane specs (varied prompt content AND length)
  produce the SAME per-lane token streams run alone (K=1) or all together (K>1);
  `BatchForwardCount` advances by one per step, not K; the owner also matches
  production `ArchSession.Generate` greedy token-for-token. Both PASS on GPU.
- **Coordinator (race-clean), fake lane set** — `cb_step_test.go`: interleave
  mode drives K raw-prompt greedy requests through one shared lane set (one
  forward per round, far fewer than K×maxNew); chat and unavailable/kill-switch
  fall back with no lane admitted. All PASS `-race`.
- **Throughput A/B on real E2B** — `lane_set_ab_test.go` (4 concurrent, temp 0,
  maxNew 64, prompt 24 tok, prefill included): serial one-lane-at-a-time
  **45.9 tok/s** aggregate vs batched four-lanes-one-forward **118.3 tok/s**
  aggregate = **2.58× aggregate speedup**, output byte-identical, 63 batched
  forwards (one per step). This CONVERTS §d's open assumption into a receipt: the
  dense ICB path IS submission-bound enough at K=4 that fusing lanes into one
  command buffer is a real win, even before weight-read-once GEMM batching.

## Weight-read-once GEMM batching — BUILT (bf16), evidenced-BLOCKED (quant) (#35 rung 2)

The compute-density rung on top of the 2.58× CB-count win. `engine/metal/lane_set_gemm.go`
adds a phase-2 forward (`batchedGEMMForward`) that sweeps each weight matrix ONCE for
all K lanes instead of K times.

**The seam is the projection, and only the projection.** A dense decode layer's seven
matmuls (qkv/o + mlp gate/up/down) are the only weight-heavy ops; rms, qk-norm+rope,
value-norm, SDPA, the residuals, the swiglu are cheap per-row work. So the forward keeps
EVERY non-projection op per-lane on its own single-row kernel — byte-identical to the ICB
the merged path replays — and lifts ONLY the projections to the batched `projector.projectRows`:
it gathers the K lanes' per-lane rms outputs into one `[K,D]` slab, sweeps the weight once,
scatters the `[K,N]` result back. Attention stays per-lane: each lane ropes/stores/attends
over its OWN icb KV caches at its OWN position. Byte-identity is per-op by construction —
only the projection's DISPATCH shape changes (one weight sweep vs K), never a row's
accumulation order. Kill switch `LTHN_CB_GEMM=0` restores the per-lane replay.

**Receipts** (`engine/metal/lane_set_gemm_test.go`):
- **Byte-identity, counter-guarded** — `TestLaneSetGEMMByteIdentityHiddens`: two lane sets
  (GEMM armed vs replay) advanced in lockstep over varied-fill specs produce byte-for-byte
  identical POST-STACK HIDDENS every step (not just argmax tokens); `gemmFwdCount>0` proves
  the GEMM path fired, `==0` on the replay set.
- **Weight-read-once A/B** — `TestLaneSetGEMMThroughputAB` (synthetic bf16, E2B-shape
  dModel 1536 / 16 layers / qDim 2048 / dFF 8192, K=4): serial replay **173 tok/s** →
  batched replay **178 tok/s** (1.03× — this bf16 decode is weight-bandwidth-bound, not
  CB-count-bound, so the CB-count amortisation is small here) → **batched GEMM 267 tok/s =
  1.51× vs batched replay, 1.55× vs serial**, output byte-identical across all three modes.
  The win is real and dominant exactly where decode is weight-bound.

**BLOCKED for the 4-bit quant path (evidenced STOP).** The quant ICB FUSES the entry/MLP
rms INTO the qmv (`setRMSQMV`, `decode_forward_arch_icb_quant.go:677-694`), keeping the
normed activation in fp32 through the matmul. Batching the weight read across lanes needs
a separately materialised (bf16-rounded) normed slab fed to `qmv-rows`, which rounds one
ulp differently and DIVERGES — a bisect showed byte-identical layer 0, one-ulp drift at
layer 1, exploding by the first global layer. There is NO batched rms-qmv-rows kernel
(`rms_qmv.go` is single-row only) and the metallib is read-only, so byte-identity AND
weight-read-once cannot both hold for the fused-rms quant path this rung. `gemmEligible`
gates on `bf16Projector` (the non-fused path) plus the proven envelope (no PLE tower, no
KV-share, no sliding window — those single-row encoders are mirrored but await a bf16
checkpoint to prove; both installed models are quant). E2B and every 4-bit model fall back
to the per-lane ICB replay — the 2.58× stays safe (`TestLaneSetGEMMQuantFallsBackToReplay`).

**The batched dense path already carries the same trade** (`decode_batched_session.go` uses
`encRMSNormRowsBF16`+`projectRows`, separate rms — token-identity tier for the MTP verify);
a real quant weight-read-once decode needs a **batched fused rms-qmv-rows Metal kernel**,
the pinned next rung one level down.

### Pinned gaps (evidenced, the next rungs)

1. **Weight-read-once GEMM batching — DONE for bf16, quant needs a fused kernel.** See the
   section above. The remaining quant work is a **batched rms-qmv-rows Metal shader** (fused
   rms + quantised matvec over K rows, weight read once) — the only way to get byte-identity
   AND weight-read-once for the fused-rms 4-bit path; needs a metallib rebuild (read-only here).
2. **Non-ICB arches.** The external-encoder fusion needs the ICB `encodeStepBody`
   seam; a non-ICB `stepToken` opens its own command buffer. MoE (12B/26B) and
   COMPOSED/hybrid decode fall outside the owner today — `Prepare` refuses them.
3. **Batched prefill admission** (slice 2) and a **batched head** (phase 1 runs K
   per-lane greedy heads; small vs the fused body, but foldable).
4. **Serve integration.** The `cbStepEngine` serves raw-prompt greedy; chat
   template rendering, incremental detokenisation, and per-request non-greedy
   sampler state are the serve-layer rungs before the CB path is the default
   interleave discipline.
