# Gated-delta recurrent state in the block codec — design (#62)

`stateLayerViewsRefreshingKinds` (session_state_blocks.go) builds the block
codec's view list by walking `spec.OwnsCache()` layers only. A `MixerGatedDelta`
layer's `CacheIndex` is `-1` by construction (it owns no KV cache — the recurrence
lives on the layer struct, not in a cache slot), so it never appears in that
walk. Its recurrent state (`gatedDeltaLayer.conv`/`.delta`, host Go fields
threaded token to token in `arch_gated_delta.go`) is consequently invisible to
`StateBlockSource` / `RangeStateBlocks` / `RestoreStateBlocks`: a restored
hybrid session's gated-delta layers start from a fresh, empty recurrence, not
the saved one's accumulated history. Pinned by
`TestStateBlocksTurboQuantHybridGatedDeltaStateNotRestored_Ugly` (now flipped
to `_Good`, see below). This design closes the gap for the block codec only —
the monolithic bf16-shaped codecs (`SerializeState`, `CaptureKV`,
session_kv_snapshot.go) keep the same silent omission they have today; they
are out of fence and out of scope (see Non-goals).

## The core insight

The recurrence is a **single running total, not a per-token row**. A KV cache
layer's block payload is naturally sliced by `[start, end)` token range — one
row per token. `conv`/`delta` have no such structure: they are the state *as
of* the session's current position, full stop. There is nothing to slice.

So the host-state kind cannot reuse the KV row-range machinery
(`stateBlockLayerBytes`/`restoreStateBlockLayer`) at all — it needs its own
fill/restore pair, exactly as TurboQuant's `turboquant-codes` kind got its own
(`tqFillStateBlockLayer`/`tqRestoreStateBlockLayer`, session_state_tq.go) — but
where TQ's payload is still row-addressed (just at a different stride), gated-
delta's payload is **carried exactly once**: on the block whose span reaches
the stream's captured position (`end == position`). Every earlier block
declares the layer explicitly **absent**.

## The new kind

`nativeStateCacheModeGatedDeltaState = "gated-delta-state"` — a third
`CacheMode`, alongside `"fixed"` (native KV) and `"turboquant-codes"` (TQ
KV). `sessionStateLayerView` gains one field, `gd *gatedDeltaLayer` — the
bound recurrence holder, read live at fill time (never copied into the view
up front, matching the KV views' own contract: they are pointers/slices into
live session state, valid only until the source session next mutates).

`SessionStateLayerBlock` gains one field, `HostStatePresent bool`, and reuses
`KeyBytes`/`ValueBytes` for conv/delta bytes (no new byte fields — same reuse
TQ already established for its own K/V-at-different-strides payload).
`RowBytes`/`ValueRowBytes`/`KVHeads`/`HeadDim`/`MaxSize` stay zero for this
kind; they describe KV geometry this kind has none of.

### Absent vs empty

`HostStatePresent` is the explicit signal, decoupled from byte length
entirely:

- The block whose span ends at the stream's position (`end == position`) sets
  `HostStatePresent = true` and carries the real conv/delta bytes — whatever
  their length, including a legitimately empty/all-zero state if the layer
  never accumulated one. Presence is not inferred from length.
- Every other block sets `HostStatePresent = false` with nil bytes — this
  layer is simply not part of this block.
- Restore is symmetric: `HostStatePresent == false` is a no-op (the layer's
  state doesn't move on this block) UNLESS the block still carries bytes
  anyway, which is refused loudly (`gated-delta block declares absent but
  carries bytes`) — a shape that can only come from a malformed or
  hand-corrupted stream, never from this codec's own fill path.

`end == position` is computable independently on both sides (fill has `end`
and `position` in scope already; restore has `block.TokenStart` +
`block.TokenCount` vs `position`) without needing to inspect payload bytes —
and it's robust across `StateBlockSourceFrom`/`RangeStateBlocksFrom`'s
`startToken` trimming and the sliding-window boundary insertions, because
`position` (the source session's `s.pos` at capture time) is always the last
boundary in the stream regardless of what gets trimmed off the front or
inserted in the middle (`stateBlockBoundaries` always appends it last).
`RestoreStateBlocks` requires full coverage to `source.Position`
(`block coverage does not match position` otherwise), so the final block —
the one that writes the state — is always reached when restore succeeds at
all.

Trusted-prefix streams (`TrustPrefixBlocks`/`TrustPrefixTokens`, `BlockCount
== 0`) never touch gated-delta state, exactly as they never touch KV cache
bytes today (`restoreStateBlockMetadata`'s early return) — trusting a prefix
already means "the target holds this data from whatever established the
trust". Not a new decision; inherited from the existing KV precedent.

## Where the view comes from

`stateLayerViewsRefreshingKinds`'s second parameter is renamed `allowTQ` →
`mixedKinds` (mechanical, in-file, two call sites, no external callers): it
already meant exactly "the kind-aware block codec, not a legacy bf16-shaped
caller" — TQ was just the first kind to use it. The loop gains one branch,
checked **before** the existing `OwnsCache()` skip (a `MixerGatedDelta` layer
always fails that check, so it would otherwise be skipped unconditionally,
exactly as it is today):

```go
if spec.Mixer == model.MixerGatedDelta {
    if mixedKinds {
        if view, ok := s.gatedDeltaStateLayerView(li, spec); ok {
            views = append(views, view)
        }
    }
    continue
}
if !spec.OwnsCache() {
    continue
}
```

`mixedKinds = false` (every legacy caller — `stateLayerViews`,
`stateLayerViewsRefreshing`, used by `SerializeState`/`CaptureKV`/the MTP
drafter export/`diffusion_session.go`/`turboquant_capture_tap.go`) takes the
`continue` unconditionally: **byte-identical to today**, gated-delta layers
stay invisible to those surfaces exactly as they are now. Only
`StateBlockSource`/`RangeStateBlocks`/`RestoreStateBlocks` pass
`mixedKinds = true`, unchanged from today's TQ-only meaning.

The cache-freshness fast path (`len(s.stateBlockViews) == ownerCount`) gets a
matching `ownerCount` adjustment (`+= s.gatedDeltaOwnerLayers()` when
`mixedKinds`), so repeated `StateBlockSource` calls on the same session (the
codec's own stated use case — "streams... without first assembling a
monolithic blob") still hit the cache instead of always rebuilding every KV
view from scratch just because the count no longer matched a KV-only
baseline. Unlike the paged-KV views, a gated-delta view never goes stale on a
cache hit — it holds a **pointer** to the layer holder, not a snapshotted
buffer, so it reads current state at fill time regardless of how long ago the
view was built. No refresh path needed for this kind.

### Device-resident state (the fused/ICB chain)

`icbEligible()` (arch_session.go) declines the *old* per-step ICB replay
outright whenever any layer is `MixerGatedDelta` — every hybrid session
decodes via `stepToken`, and `s.state.icb` is always nil for one. That is
**not** the same thing as the recurrent state always being host-resident,
though: the newer whole-token chain walk (`arch_qwen_fused.go`,
`gatedDeltaQuantChainLayerDevice`/`gatedDeltaQuantChainMoELayerDevice`) can
still engage a **device-resident** handle for a gated-delta layer
(`gd.sc.Device`, primed once by `lthn_gated_delta.go`'s
`gatedDeltaBlockDeviceHook`/`gatedDeltaQuantLayerRun`), independently of the
old ICB gate, for real (servable-geometry) hybrids. `RestoreStateBlocks`
already anticipates ICB-recorded sessions reaching it (`if s.state.icb == nil
{ reloadPagedStateLayerViews }`), so a gated-delta host-state view must not
silently read stale host fields when the true state has moved to that device
handle.

`attn.GatedDeltaDeviceStateExport` (model/attn/gated_delta.go) is exactly the
seam for this — wired at engine init
(`lthn_gated_delta.go:gatedDeltaDeviceStateExportHook`), documented as "the
snapshot/clone seam", and independently unit-tested
(`TestGatedDeltaBlockDeviceRun_*`'s export/prime round trip). It has had no
live caller since #50 retired `model/composed`'s `CloneState`. `snapshotState`
(new, on `*gatedDeltaLayer`) is a thin pass-through: try the export hook when
`gd.sc.Device != nil`, fall back to `gd.conv`/`gd.delta` otherwise (`ok=false`
from the hook means the same, per its own contract — "the caller's host
slices stand").

**Honest boundary**: this fallback is defensive and correct by construction
(a 4-line call into an independently-tested primitive), but it is **not**
exercised end-to-end by this patch's tests. Every fixture available inside
this task's file fence decodes via the host path only — the state-carrier TQ
hybrid (`kvCacheMode: "turboquant:4"`, proven `state.icb == nil` and
`stepToken`-only by `TestArchQuantSessionTurboQuantHybrid_Good`) never
engages `gd.sc.Device`. Building a fused-chain-eligible hybrid fixture
touches `arch_qwen_fused.go`/`decode_forward_arch.go`, both outside this
task's fence. The branch exists so a real qwen3.5 hybrid's fused decode does
not silently regress to stale host bytes; it is not claimed as tested by this
patch.

## The fill/restore pair

`gdFillStateBlockLayer(view, end, position)`: `end != position` → return
`{Layer, CacheIndex, CacheMode}` only, `HostStatePresent` stays `false`. `end
== position` → `conv, delta := view.gd.snapshotState()`, wrap both with the
existing `float32Bytes` (zero-copy view — same convention as every other
KeyBytes/ValueBytes in this codec).

`gdRestoreStateBlockLayer(view, layer)`, dispatched from `restoreStateBlock`
**before** the generic bf16 KV checks (same reason TQ's check runs first: kv-
head/head-dim/cache-mode/max-size/row-byte checks are meaningless for this
kind and must never accidentally pass or fail on it):

1. `layer.CacheMode`/`view.cacheMode` both `gated-delta-state`, or refuse
   (`gated-delta cache-kind mismatch`) — **absolute kind equality**, mirroring
   `tqRestoreStateBlockLayer`'s own first check. Never a reinterpret.
2. `!HostStatePresent` → no-op, unless bytes are present anyway (refuse —
   see Absent vs empty above).
3. Length check against `view.gd.stateLengths()` — computed from the bound
   `GatedDeltaConfig` (`(ConvKernel-1)·ConvDim()` float32s for conv,
   `ValueHeads·HeadDim·HeadDim` for delta), available before any decode ever
   runs, so a snapshot's size can be validated even onto a session that
   hasn't stepped yet. Mismatch refuses (`gated-delta state size mismatch`).
4. Copy (never alias) into `gd.conv`/`gd.delta` — `bytesFloat32`, the mirror
   of `float32Bytes`, always a fresh copy since the source bytes are a
   transient wire payload.

## Decline rules (the contract)

1. Legacy (bf16-shaped) callers: zero behaviour change. Gated-delta layers
   stay invisible, exactly as today.
2. Kind mismatch on restore (either direction) → loud, structural refusal.
   Never a reinterpret.
3. Absent-but-carries-bytes → loud refusal (malformed/corrupted stream
   shape).
4. Length mismatch against the target's own bound geometry → loud refusal.
5. A `MixerGatedDelta` spec with no bound holder (`bindGatedDeltaQuant`/
   `BF16` never ran, or the checkpoint's weights didn't bind) → the view is
   silently omitted from the block stream (`gatedDeltaStateLayerView` returns
   `ok=false`), not a hard error for the whole codec call. Save and restore
   both derive this the same way from the same model, so a broken-model
   session stays internally consistent (both sides omit the same layer) —
   this is a pre-existing construction-time gap, not one this design
   introduces or can detect further without touching the loader (out of
   fence).
6. Non-hybrid sessions: zero behaviour change. No `MixerGatedDelta` layer ⇒
   the new branch never fires ⇒ `views`/`ownerCount` identical to before this
   design.

## What this design will NOT do (and why)

- **No change to the legacy bf16-shaped codecs** (`SerializeState`,
  `CaptureKV`, session_kv_snapshot.go, the MTP drafter export in
  assistant_load.go). They keep silently omitting gated-delta state, exactly
  as before this patch. Those files are outside this task's fence; fixing
  them is a separate, explicitly out-of-scope follow-on.
- **No change to `gatedAttnLayer` / gated full-attention state**
  (`arch.AttnOutputGate`, qwen3_5/next's *attention* gating). That is a
  different layer kind with its own resident-state seam
  (`arch_qwen_fused.go`), unrelated to `MixerGatedDelta`'s recurrence, and
  docs/design-tq-moe-hybrid.md already scopes TQ away from it for the same
  reason. #62 is about the gated-**delta** mixer's conv/delta only.
  Restoring a gated-attention hybrid's KV through this codec is unaffected
  either way — its KV still rides the ordinary owner-layer path.
- **No end-to-end test of the device-resident (`gd.sc.Device`) fallback.**
  See "Device-resident state" above — implemented defensively over a tested
  primitive, not verified end-to-end within this task's fence.
- **No attempt to reconstruct gated-delta state from a partial/trusted-prefix
  stream.** A trusted prefix means "already resident, believe it" — the same
  assumption the KV side already makes; not strengthened or weakened here.
- **No wire/serialisation format.** `SessionStateBlock`/`SessionStateLayerBlock`
  remain in-process Go structs (confirmed: no consumer outside
  engine/metal exists today); this design changes their Go shape, not any
  encoded form.
