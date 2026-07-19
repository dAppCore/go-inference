# Design: the MTP boundary-greedy carry (#53 lever 3 — design only)

Status: **designed, deliberately not implemented in this lane.** Lever pack 1
parked it ("wide stale-token invalidation surface = output-correctness risk");
this note maps that surface precisely, gives the correctness argument, and
names the one mechanism that would make it safe. The verdict is at the end.

## What is recomputed today, per round

Every **accepting** assistant-verify round
(`verifyDraftBlockFromSessionWithSuppress`, engine/metal/assistant_load.go)
ends with:

```
hidden := hiddens[accepted-1]                     // last accepted row, already resident
logits := target.headLogitsScratch(hidden, false) // FULL head: RMSNorm + vocab×dModel qmv + softcap
target.rememberRetainedHidden(hidden)
target.rememberRetainedLogits(logits)
```

and the **next** round's verify opens with:

```
boundaryLogits := target.BoundaryLogits()               // the retained row
first := greedyBF16Suppressed(boundaryLogits, vocab, s) // 512KB host argmax scan
```

At 26B geometry (vocab 262144 × dModel 2816, 4-bit) the head recompute is a
~370MB weight read plus a command-buffer submit+wait, and the argmax scan
walks 512KB of bf16 on the host — together roughly 1–2 ms of every ~19 ms
round, every accepting round. Live diag (2026-07-19, 26B pair): verify totals
15–20 ms of which the batched forward is 11–18 ms and the K-row rows-head
2.5 ms; the boundary head + bookkeeping is most of the remainder.

## What the verify already knows

`verifyAssistantDraftRows` runs the K-row greedy head over every verified row
— `rows[i] = argmax(head(hiddens[i]))`, suppression applied. So
`rows[accepted-1]` **is** the greedy token at the exact hidden that becomes
the retained boundary. The recomputed `first` of the next round is the same
argmax of the same hidden. Carrying `rows[accepted-1]` forward as the next
round's `first` would skip both the head recompute and the argmax scan.

## The correctness argument — and its one dependency

The K-row quant head scores rows through the **qmm_t token-identity tier**
(`greedyRowsQuantFusedInPool`), while `headLogitsScratch` runs the per-row
**qmv byte tier**. The two produce different logits *bytes*; the carry is
token-exact iff their **argmax** agrees at the boundary hidden.

That cross-tier argmax equality is not a new assumption: the existing
carry-lead mechanism already stakes the emitted stream on it. On a partial
accept the loop emits `ReplacementToken = rows[accepted-1]` (rows tier) and
re-verifies it next round against `first` (qmv tier) computed from the same
retained hidden — if the tiers ever disagreed there, the already-emitted
carry token would diverge from plain decode today. The carry therefore adds
**no new numeric assumption**; it changes which tier's evaluation is *named*
the boundary truth. Softcap does not disturb it: `tanh(x/c)·c` is strictly
monotone, so pre-/post-softcap argmax are identical (the K-row head already
skips softcap for exactly this reason).

## The invalidation surface (why lever 1 parked it)

A carried greedy is a *derived* value of `retainedHidden`. It must die the
instant the boundary moves by any path other than the verify that produced
it. Writers of the retained boundary today (engine/metal, non-test):

- `rememberRetainedHidden` / `rememberRetainedHiddenFrom` — the verify accept
  and reject paths, the sampled verify, the dflash lane, snapshot restore.
- `retainHiddenDirectFromICB` — the ICB decode's zero-copy retain.
- `resetRetainedLogits` + state-block/state-snapshot restores
  (`session_kv_snapshot.go`, `session_state_blocks.go`).
- The re-engagement plain stretches (`nativeAssistantPlainRunFromTargetCache`)
  which advance the boundary through plain decode between drafting windows.

Consumers of the retained **logits row** (which the carry would stop
materialising eagerly): `BoundaryLogits()` (public API), state snapshot/block
serialisation, `speculative_model.go`'s logit export, and the sampled verify
(needs the full row to sample, not an argmax). A carry that only stores the
token would break every one of them unless the row stays reconstructible.

## The safe mechanism (if/when implemented)

1. **Lazy retained logits.** Replace the eager
   `headLogitsScratch + rememberRetainedLogits` with a
   `retainedLogits = nil` + on-demand materialisation inside
   `BoundaryLogits()` from `retainedHidden` — the recompute produces the
   *identical* qmv-tier bytes the eager path stored, so every logits consumer
   (state save, sampled lane, API) is byte-unaffected; only the *timing* of
   the head call moves from every accepting round to actual demand.
2. **Version-guarded carry at one choke point.** Add a monotonically
   increasing `retainedVersion` bumped inside `rememberRetainedHidden` (and
   the ICB retain) — the ONLY places a boundary is ever installed. The verify
   stores `(carriedGreedy, versionAtCompute)`; the next round consumes it only
   when the versions match, else it falls back to `BoundaryLogits()` + argmax
   exactly as today. Staleness then cannot exist by construction rather than
   by audit.
3. **Suppress gate.** Carry only when the suppress set is empty (the standard
   serve shape). `rows[]` and `first` are computed under the same per-request
   suppress list, but gating on empty removes the need to prove list identity
   across rounds.

## Verdict

Worth ~1–2 ms + one GPU sync per **accepting** round (~5–8% of the 26B round
at current acceptance) — real, but the smallest of the three levers, and it
touches the retained-boundary machinery that state snapshots, the sampled
lane, and the dflash lane all share. Implement as its OWN lane with the
version counter landing first and the greedy-parity + state-snapshot suites
as the gate; do not fold it into an unrelated perf lane. Until then the eager
recompute stays — it is the byte-authoritative boundary and every consumer's
contract holds unchanged.
