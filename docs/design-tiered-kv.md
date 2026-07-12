<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# Tiered KV for `lem serve` — design memo (oMLX-parity probe)

Status: exploration. This memo maps oMLX's hot-RAM / cold-SSD KV-cache design
onto the machinery go-inference already ships, records the honest gaps, and
scopes the cheapest real slice. It is the study deliverable for the tiered-KV
lane; the measurement receipts it points at live in the lane report.

## oMLX's design, in one line

A hot RAM tier plus a cold SSD tier with **block-level prefix sharing**,
transparently persisting **every** context (not just named conversations)
across requests and restarts, so a returning prefix is a load, never a
re-prefill.

## The mapping — oMLX semantic → what we already have

| oMLX tier semantic | Our machinery | Status |
|---|---|---|
| Hot tier: contexts resident in fast memory | `continuity.Manager.resident` — RAM-resident conversation sessions, LRU-evicted at `defaultMaxResident = 4` | **Exists** |
| Cold tier: contexts spilled to persistent storage | Every turn calls `SleepAgentMemory` → KV blocks written to the `state.Store` (`serving/continuity/continuity.go:finishTurn`) | **Exists** |
| Evicted-then-revived = load, not re-prefill | `acquire` probes the store index; a hit calls `WakeAgentMemory` + `RestoreKV` and prefills **only the new turn** (`AppendPrompt`), never the prefix | **Exists** |
| Persist across restarts | The store is disk-backed (`filestore.Store`) when `-state-store` is set | **Exists (opt-in)** |
| Persist *every* context, not just named | The conversation key is a hash of the whole message prefix — anonymous by construction, every accepted text chat is cached | **Exists** (declines: media / mid-turn / thinking-override / one-shot completion) |
| Block-level prefix sharing *within* a conversation | Incremental sleep chains each turn to its parent bundle (`ReuseParentPrefix{,Trusted}`, `BlocksReused`) — only the new turn's blocks are written | **Exists** |
| Block-level prefix sharing *across* conversations | The key is a **whole-prefix hash**, so two conversations sharing a system prompt store two independent bundles — no shared blocks | **Not built** |
| GPU→CPU→SSD placement of *live* KV under budget | `kv/kvtier` (LRU tier policy, GPU/CPU/Disk budgets, pin, promote-on-touch) is fully built but **imported by nothing** | **Built, unwired** |
| Radix/prefix tree over token blocks | `kv/radix` is built but **imported by nothing** | **Built, unwired** |

**Headline:** the hot/cold tier, per-turn spill, and wake-not-re-prefill that
the lane brief proposed to *prototype* are already the shipped behaviour, gated
by `-state-conversations` (default on). The remaining oMLX distance is
cross-conversation block sharing (the two unwired packages), not the spill.

## The default is inverted (the real cheap slice)

Today, with `-state-conversations` on and `-state-store` **unset**, serve still
writes a **file**: `resolveStateStorePath("")` returns the home-relative
`conversations.kv` and `openConversationStore` calls `filestore.Create`
(O_TRUNC, wiped each run). So a MacBook-class box that never asked for
durability pays a disk write on every sleep and a disk read on every wake, for
a cache it throws away at shutdown.

Invert it:

- `-state-store` **unset** → hot tier is a pure-RAM `state.Store`
  (`state.InMemoryStore`, which already exists). No file, no per-turn disk
  round-trip. State lives as long as the serve process — the right default for
  a long-lived server on a laptop.
- `-state-store` **set** → durable `filestore.Store`, exactly as today. Chats
  that asked for durability keep their `.kv` semantics untouched.

This composes with, rather than replaces, the existing tiers: hot = RAM store
(default), warm = resident conversations (as now), cold = spill to `.kv` only
when durability is requested (or, as future work, when memory pressure demands).

### One prerequisite: `state.InMemoryStore` was not thread-safe

`filestore.Store` guards its maps with a `sync.Mutex`; `InMemoryStore` had no
lock. Serve is concurrent and `continuity` touches the store **outside** its own
manager lock (wake in `acquire`, sleep in `finishTurn`), so the bare in-memory
store as a serve default would risk a `concurrent map read and map write` panic.
Giving `InMemoryStore` an `RWMutex` is completing the `state.Store` contract the
concurrent serve path already assumes — not new scope.

### Why `lem generate -state` keeps the file default

`lem generate` is a one-shot process: the point of `-state chat1` is that a
*second* invocation wakes `chat1` from disk. A RAM store cannot survive across
processes, so RAM-by-default there would silently break cross-call continuity.
The RAM tier only helps a repeated-generate loop **inside one process** (the
bench harness). Generate therefore keeps the file default; the RAM path is a
harness/library option, not a flag flip.

## What was unknown → now measured

Whether the per-turn disk tax actually matters at realistic sizes, and how big
the wake-vs-re-prefill win is. Neither is knowable by reading. The receipt
harness (`serving/continuity/receipts_metal_test.go`, opt-in, real e2b
checkpoint on engine/metal) drives the shipped `continuity.Manager` and reports:

Receipt — gemma-4-e2b-it-4bit, ctx 4096, ~2K-token conversation, 3 reps (min):

| Path | TTFT | Total turn |
|---|---|---|
| re-prefill (cold, full prefill) | 309 ms | 339 ms |
| wake (file store) | 128 ms | 159 ms |
| wake (RAM store) | 123 ms | 151 ms |
| resident (warm, no store) | 41 ms | 68 ms |

Two readings:

1. **Wake vs re-prefill — REAL, and the bigger win.** Wake avoids ~186 ms of
   re-prefill here (309 → 123 ms), and this grows with context: the prefill it
   skips is O(tokens) while the restore is roughly flat (the continuity header
   records 12 s of prefill at 32 K). This is the value of the continuity
   machinery — and it already ships.
2. **RAM store vs file store — MARGINAL on latency.** The store-backend tax is
   ~5 ms TTFT / ~8 ms total-turn / ~3 ms sleep-write. It is small because the
   file read hits the OS page cache within a live process — a cold read
   (post-restart, cache evicted) or slower storage would widen it, which this
   in-process harness cannot show.

### Verdict

- The continuity machinery (hot resident + cold spill + wake-not-re-prefill) is
  a real win and is already shipped; the lane brief's proposed prototype was
  already the behaviour.
- The **RAM-store default is still the right default** — but for simplicity and
  hygiene more than raw latency: it removes a per-turn disk write (SSD wear), the
  whole ephemeral-file lifecycle (wipe-on-launch, remove-on-shutdown,
  crash-leftover handling), and is strictly ≥ 0 versus the file (never a
  regression). The latency win is single-digit ms warm, larger cold.

### `lem generate -state`

The same `SleepAgentMemory`/`WakeAgentMemory` tax (~5–8 ms/turn) applies, but
generate is a one-shot process, so RAM cannot be its default without breaking
cross-invocation continuity (see above). A RAM tier only helps a repeated-generate
loop inside one process; the explicit per-invocation file-vs-RAM number is a
follow-up harness, not shipped here.

## Shipped in this lane

- `state.InMemoryStore` made concurrency-safe (`RWMutex`).
- `lem serve` defaults the conversation store to RAM when `-state-store` is
  unset; any explicit path is the durable file, as before.

## Ranked follow-ups

1. **Cross-conversation block sharing** — wire `kv/radix` so a shared system-prompt
   prefix is one bundle across conversations. Biggest oMLX-distance closer; a real
   build (block-addressed keys instead of whole-prefix hashes), not a slice.
2. **Memory-pressure spill** — evict the RAM store's coldest bundles to `.kv`
   under a byte budget instead of dropping, so the RAM default degrades to the
   cold tier gracefully. `kv/kvtier` is the ready-made policy; it needs a
   `state.Store`-backed `Store` mover.
3. **Cache the declined paths** — one-shot completions and mid-turn requests get
   no caching today; oMLX caches every context.
