# TQ on MoE + hybrid arches — design (#48 follow-on)

TurboQuant KV (`-kv-cache turboquant[:N]`) currently arms only on sessions the
arch ICB can record: dense, standard-attention, no trace. This design extends
it to (a) the MoE family (gemma4 26B-A4B shape) and (b) mixed-layer-kind
(hybrid) arches, without touching the proven ICB lane. Decisions only; the
in-tree code is the survey.

## The core insight

**MoE is an FFN property, not a cache kind.** The old blanket decline
(`tqKVArchServable`: "declines MoE archs") conflated *cannot record the ICB*
(true — host router breaks the recorded chain) with *cannot serve the KV
contract* (false — a MoE layer's attention side is standard K/V rows). The MoE
half of this design is therefore a **second TQ carrier**: the same codes+γ
caches, the same #41/#48 kernels, driven per token by `stepToken`'s encoder
instead of by ICB replay. No new kernels; no change to the recorded lane.

## Layer-kind → cache-kind matrix

Under an armed TQ mode, each layer's cache kind is decided by its KIND, never
by the arch's family name:

| Layer kind (spec-derived)                                        | Cache kind        | Why |
|------------------------------------------------------------------|-------------------|-----|
| Attention mixer, GLOBAL owner, geometry OK, ungated, no sinks    | **TQ codes + γ**  | the unbounded-growth cache TQ exists for |
| Attention mixer, SLIDING owner                                   | native bf16 ring  | bounded residency — nothing to win; ring slot rebind is unaddressable over packed codes |
| Attention sharer (`KVShareFrom ≠ self`)                          | reads owner's kind| ICB lane: TQ read wired (sliding sharers already force the owner bf16). State lane v1: ANY sharer forces the owner native — the shared-attention emitters are not TQ-wired there |
| `MixerGatedDelta` (linear attention / conv / recurrent state)    | native state      | no KV rows exist; nothing to quantise. `CacheIndex == -1` by construction — it can never be handed a KV cache of any kind |
| Gated full attention (`arch.AttnOutputGate` — qwen3_5/next)      | native (own lane) | KV lives in the gated/fused lane's resident state (`gatedAttnLayer`/`arch_qwen_fused.go`), not in `lb`/ICB caches — a different seam (see non-goals) |
| Attention sinks (gpt_oss)                                        | arch-wide decline | the TQ read kernels carry no sinks lane (unchanged v1 rule) |

Geometry OK = both bit widths instantiated and head dim ∈ {128, 256, 512}
(`tqKVGeometryOK`, unchanged).

## Where the arming decision lives

Unchanged homes, one qualification function:

- **Mode parse**: `parseTurboQuantCacheMode` (tq_kv_mode.go) — untouched.
- **Arch qualification**: `tqKVArchServable` (tq_kv_mode.go) reworked to the
  per-layer-kind matrix above. It stops blanket-declining MoE and hybrid
  mixers; it still refuses loudly when the matrix leaves NO qualifying layer
  ("a turboquant session that would quantise nothing must not pretend"), and
  that refusal is what a qwen3_5/next stack hits today (every attention layer
  gated) — with the message naming the gated-attention seam, not a generic no.
- **Per-layer enablement at cache construction**: `allocArchICBCachesTQ`
  (ICB lane, untouched) and its state-lane mirror `allocArchStateKVTQ` (new).
  The session constructors pick the carrier: ICB-eligible → recorded lane
  (exactly as today); ICB-ineligible **because of MoE / gated-delta mixers**
  → the state lane; trace → decline (host code reads bf16 cache bytes).

## The state-lane decode (MoE half)

`stepToken` is the 26B's decode. The TQ session takes the LINEAR-cache route:

- **No paged pool.** A state-TQ session declines `initDevicePagedKV` wholesale
  (the attention-sinks precedent): paged TQ would need a per-page code
  addressing kernel family that does not exist. Linear caches are the lane.
- **TQ owners hold code caches + γ planes** allocated by `allocArchStateKVTQ`
  (maxLen rows — global owners only); their `lb` bf16 caches are never
  allocated. Every other owner keeps today's `lb` caches byte-identically.
- **The attention half** for a TQ owner is `encAttnHalfKVTQ` (new, mirroring
  `encAttnHalfKVInputAt` + the ICB recorder's TQ block): K rope/norm and the V
  projection land in fixed bf16 staging rows, one `lthn_tq_kv_store` each
  rotates+quantises staging → code row + γ row at `pos`, and the SDPA reads
  codes — the FUSED single-pass kernel below the 2-pass knee, the
  rot→pass1→pass2 trio at/past it (per-token encode picks per token; no
  rebind constraint). Same emitters, same kernels, same op economics as the
  recorded lane — the house O(output)-only fusion rule is inherited, not
  re-derived.
- **Prefill** rides the existing fall-through: the batched dense pass DECLINES
  a state-TQ session up front (one guard — it is not TQ-aware for this
  carrier and would land bf16 rows into code caches), so prefill runs the
  per-token path, which is TQ-correct by construction. Slower prefill,
  correct bytes; batched state-lane prefill is a named perf follow-up.
- **Decline choke points** (`laneSet`, MTP pairing, prompt reuse, CaptureKV)
  extend from `s.state.icb.hasKVTQ()` to "either carrier armed" via one
  session-level helper. MTP+TQ already declines at load; the helper is belt
  and braces.

## Snapshot codec for MIXED cache kinds

The block codec (`SessionStateBlockSource` / `RestoreStateBlocks`,
session_state_blocks.go) becomes kind-aware; the monolithic bf16-shaped APIs
(`CaptureKV`) keep their v1 decline — they cannot represent codes.

- `sessionStateLayerView` / `SessionStateLayerBlock` gain a second cache mode:
  `"turboquant"` beside `"fixed"`, plus additive fields — per-side row strides
  (K and V differ under `turboquant:3.5`), the two γ planes, and the bit
  widths. Zero values everywhere for native layers: the wire shape of a
  native layer's block is byte-identical to today.
- A TQ layer's block payload is its RAW code rows (Key/ValueBytes at code row
  stride) + γ rows — bytes preserved exactly, never dequantised in transit.
  Row-range slicing works unchanged: codes and γ are row-addressed exactly
  like bf16 rows, just with different strides.
- **Kind boundaries are enforced on restore**: the existing per-layer
  CacheMode check makes a TQ block landing on a native layer (or the reverse)
  a loud `RestoreStateBlocks` error — reinterpreting bytes across kinds is
  structurally impossible, which is the failure mode this design exists to
  forbid. Bits/strides/geometry are validated per layer the same way.
- The kind-aware view builder is a NEW entry used by the block codec;
  `stateLayerViewsRefreshing` keeps its wholesale TQ decline so the MTP
  drafter export path (which reads bf16-shaped views) can never receive code
  bytes even if a future caller wires it wrongly.
- A gemma4 dense TQ session already yields MIXED kinds naturally (sliding
  ring layers native + global layers TQ) — the round-trip gate exercises
  exactly that, plus the MoE session on the state carrier.

## Decline rules (the contract)

1. Unknown mode string → error (unchanged).
2. Sinks anywhere → error (unchanged; kernel limitation).
3. Trace + TQ → error (host reads bf16 cache bytes).
4. No qualifying layer after per-kind classification → error, message names
   what disqualified (head-dim geometry / gated attention). This is qwen3_5
   and qwen3_next TODAY: their attention layers are all gated → loud decline,
   never silent-native, never a TQ cache behind a lane that reads bf16.
5. State-TQ carrier: batched dense prefill, laneSet, MTP, prompt reuse,
   CaptureKV all decline exactly as the ICB carrier does.
6. Snapshot restore across kinds → per-layer error (codec section).
7. TQ unset → zero behaviour change: every new branch gates on the parsed
   mode / an armed carrier; native lanes stay byte-stable.

## What this design will NOT do (and why)

- **No TQ paged-KV kernels.** Per-page code addressing is a new kernel family
  with its own gate matrix; the linear lane serves the MoE decode correctly
  today. Paged TQ is a perf follow-up with its own receipts, not a
  correctness prerequisite.
- **No TQ in the gated-attention / fused qwen chain lane.** That lane's KV is
  resident state inside `arch_qwen_fused.go`/`gatedAttnLayer` (host f32 or
  device bf16), CB-recorded with its own replay bookkeeping — arming TQ there
  without wiring that seam is precisely the silent-wrong-cache failure this
  design forbids. Hybrids whose attention is all gated decline loudly (rule
  4). Wiring TQ through the fused chain is the follow-on campaign; the
  layer-kind matrix and mixed-kind codec landed here are its prerequisites.
- **No host-KV TQ.** The gated host fallback keeps f32 KV on the host — there
  is no device residency to win; quantising it would spend accuracy on
  nothing.
- **No batched prefill for the state carrier (v1).** Correct-but-sequential
  beats fast-but-uncertain; the ICB carrier's batched TQ prefill is untouched.
- **No band changes, no new kernels.** Every dispatch this design adds reuses
  the shipped #41/#48 kernel set with its measured bands; a new path that
  needed a wider band would be a bug in the path, not a tolerance update.
