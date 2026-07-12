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

## What is genuinely unknown → measured

Whether the per-turn disk tax actually matters at realistic sizes, and how big
the wake-vs-re-prefill win is. Neither is knowable by reading; both are in the
receipt tables in the lane report (real e2b checkpoint, engine/metal):

1. **Serve, RAM store vs file store** — per-turn wall + TTFT across a multi-turn
   conversation (the sleep tax while resident; the wake+sleep tax once evicted).
2. **Wake vs re-prefill** — TTFT for a revived long conversation (the value of
   the continuity machinery itself, independent of the store backend).

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
