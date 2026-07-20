# Design: owned-weight device binding (#60)

Status: implemented with this change.
Scope: `engine/metal` zero-copy bind path + the `model` load pipeline.

## The gap

`shardBuffers.bufForAligned` (engine/metal/nocopy_weights.go) resolves a weight `[]byte` to a
no-copy Metal buffer + offset by locating the slice inside a mapped checkpoint shard. Any weight
that is an OWNED heap buffer — synthesised at load rather than viewed from the mmap — falls off
the end of the shard scan and fails with "weight is not a view into any mapped shard". Two real
consumers are blocked:

1. **The exact b1→b2 repack** (`model/quant/mlxaffine.RepackB1ToB2`). Bonsai-27B-mlx-1bit ships
   1-bit affine codes; the metallib carries qmv kernels for bits {2,3,4,5,6,8}. The repack is an
   exact format bridge, its packed output is an owned buffer, and the probe (2026-07-20, see
   0bddd98d) showed it reaches the kernels and then dies at bind. `subTwoBitQuantDecline`
   currently refuses these checkpoints up front.
2. **Every synthesised pack** — the `packExperts` MoE tensors (mixtral, dbrx, olmoe, qwenmoe,
   llama4) allocate owned buffers at NormalizeConfig time. On the strict quant bind
   (`viewQuantWeight` → `mustBufFor4`) they hit the same wall.

## How the resolver's ownership model already works

- `newShardBuffers` wraps each shard's page-aligned mmap in ONE no-copy buffer; `shardBuffers`
  owns the `DirMapping` and MUST outlive every bound view; `Close` evicts + unmaps.
- `residentBytes(b)` is the existing owned-bytes device cache: keyed by `&b[0]`, each entry
  **holds the slice** (GC-alive) and a `runtime.Pinner` (immovable) beside its no-copy buffer —
  or falls back to an upload copy (`sharedBytes`) when the no-copy wrap is refused (odd base,
  binding failure). Entries release device memory via `evictResidentBufsForRanges` at
  `shardBuffers.Close`, or `resetResidentBufsForTest` between test loads.
- `bufForAligned` already carries two sanctioned owned-bytes fallbacks through `residentBytes`:
  misaligned shard views, and F16→BF16-**widened** tensors that the `DirMapping` positively
  registered at load (`IsWidened` — the precedent this design generalises).
- The strict tail — error on an UNREGISTERED off-shard weight — is the wrong-mapping guard: a
  weight from a different mapping must fail at load, not decode coherently wrong. It stays.

## Decision

**Registered-owned-range adoption, resident binding, session-scoped eviction.** The `widened`
mechanism generalised:

1. `safetensors.DirMapping` gains an `owned` range set beside `widened`:
   `AdoptOwnedTensors()` sweeps `dm.Tensors` once at the end of `model.Load` and records the
   heap range of every non-empty `Tensor.Data` that is neither a shard view nor widened —
   i.e. every load-time synthesis that landed in the tensor map (packExperts packs, the
   repacked b1 weight, future normalisers), with no per-arch registration code.
   `IsOwned(b)` answers the binder; `OwnedRanges()` hands the spans to eviction.
2. `bufForAligned` accepts a registered owned weight exactly as it accepts a widened one:
   `s.dm.IsOwned(weight)` → `residentBytes(weight)` (cache keyed by data pointer, entry pins
   the slice, no-copy where the wrap is accepted, upload copy otherwise).
3. `shardBuffers.Close` evicts the owned ranges alongside the shard ranges before `dm.Close`,
   so a repacked/synthesised model's device buffers are released with the session — owned
   packs can be GBs (a fully repacked 27B), unlike the small widened norm vectors.
   `DirMapping.Close` clears the owned set so stale ranges can never match recycled addresses.
4. The b1→b2 repack re-hooks at `model.LoadLinear` (the probed change): after `affineGeometry`,
   `Bits == 1` → `RepackB1ToB2`, `Bits = 2`, pass through unmodified on error. The repacked
   WEIGHT tensor is written back into the tensor map (Data + packed shape) so the adoption
   sweep sees it, a tied second read loads the b2 form instead of repacking twice, and the map
   stays the truthful "checkpoint as loaded". Scales/biases keep the ORIGINAL mmap views
   (byte-identical under the widening) — they stay on the zero-copy fast path.
5. `subTwoBitQuantDecline` is deleted; `TestRealCheckpointGPU_Bonsai1BitRepack_Bad` flips back
   to the `_Good` serving form.

## Lifetime contract

A GPU read of freed heap must be impossible:

- **Bind → decode:** every owned bind goes through `residentBytes`; the cache entry holds the
  slice reference AND the pin, so the backing array can neither be collected nor moved while
  the entry exists. Nothing evicts entries mid-session.
- **Eviction:** only `shardBuffers.Close` (the session teardown — same contract as the shard
  buffers: called after every command buffer completed) and `resetResidentBufsForTest` (tests,
  same quiesced contract). Both unpin and release; after that the model's slices die with the
  `LoadedModel`.
- **Key stability:** an address-keyed cache is only sound while addresses can't be re-issued
  for different bytes — guaranteed because the entry itself keeps the slice alive (see the
  `residentBuf` doc). Owned ranges on the `DirMapping` are cleared at `Close`, so a later
  model's allocations can never false-match a dead registration.

## What this does NOT do

- **No behaviour change for view-bound weights.** The owned check sits after the shard scan
  and beside the widened check, on the error path only; aligned shard views bind byte-identically
  as before (the existing metal suite is the guard).
- **No blanket acceptance of off-shard pointers.** An unregistered off-shard projection still
  fails the wrong-mapping guard.
- **No synthetic-shard staging.** Re-materialising owned tensors into a mmap-backed file adds
  I/O and a second lifetime for no benefit over the proven pin+cache path.
- **No widened-lifetime change.** Widened ranges keep their current (process-lifetime cache)
  behaviour; only the new owned class gets session-scoped eviction.
- **No MoE decode-shape work.** Binding is the class fix this change ships. The packExperts
  archs still have SEPARATE, named serve gaps past the binder that are out of scope here:
  the bf16 MoE step decode (`moeBlockBF16…`) requires gemma's five sandwich norms + local MLP
  (llama-family MoE layers carry one pre-FF norm), and `packExperts` packs `.weight` only, so
  a quantised pack has no packed scales/biases triple yet. The new synthetic receipts therefore
  prove (a) end-to-end Generate through owned repacked weights on the strict quant bind path
  (a synthetic b1 checkpoint — the exact lane Bonsai serves through), and (b) the packExperts
  tensors of a factory-loaded synthetic MoE checkpoint adopting as owned and resolving through
  the strict binder that refused them before.
- `totalMappedBytes` (RAM budgeting) still counts mapped shards only; owned bytes are not
  budgeted. Known under-count, small relative to checkpoints today; revisit if owned packs grow.
