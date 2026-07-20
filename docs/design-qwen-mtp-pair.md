# Design: qwen hybrid MTP pair — factory-native drafter, v1 scope (#59 item 2)

The #50 composed strip deleted `model/composed` and with it the qwen family's
working MTP (multi-token-prediction) speculative pair. Two stacked declines
are live today: `LoadSpeculativePair` (`go/engine/metal/speculative_model.go`)
refuses any gated-delta hybrid TARGET by name
(`qwen35.HybridModelType`), and the `qwen3_5_mtp` / `qwen3_5_mtp_text` /
`qwen3_6_mtp` DRAFTER checkpoint itself refuses at `Parse`
(`go/model/arch/Qwen/qwen35/register.go`). This is the deepest remaining
post-composed capability gap named in the strip inventory
(`go/docs/composed-strip-inventory.md`: *"qwen MTP speculative pair (`lem
pair`/`-draft` on hybrid targets) — typed decline at LoadSpeculativePair:
named 'factory pair route pending (#50)'"*).

The deleted reference lives at `b1f6c21a^:go/model/composed/mtp.go` (750
lines) + `mtp_loader_test.go`/`mtp_test.go`/`config.go`/`mixers.go`/`moe.go`,
and is the port source for part 1 below. The working factory precedent —
gemma4's assistant pair — lives in `go/model/gemma4/assistant.go` +
`go/engine/metal/{assistant_load.go,speculative_model.go,mtp.go}`, studied
for part 2.

**Bonus find**: the real drafter checkpoint this design cites is locally
cached on this box — `mlx-community/Qwen3.6-27B-MTP-4bit`
(`~/.cache/huggingface/hub`, snapshot `83795d546e9d328160e593fb0bf10b2bf2fe637e`),
alongside its base `mlx-community/Qwen3.6-27B-4bit`. Every architectural claim
below about "the real checkpoint" is read from that snapshot's actual
`config.json` and safetensors header (parsed header-only — no tensor data
touched), not reconstructed from the historical Go source alone. The two
agree exactly, which is itself useful corroboration.

## 1. What the composed qwen MTP drafter actually was

**Architecture** — one (the real checkpoint: `mtp_num_hidden_layers: 1`)
full-attention transformer layer per drafter, structurally identical to one
of the base's own `full_attention` layers: gated attention
(`attn_output_gate: true` — inherited, because the checkpoint's
`text_config` block *is* the base's own `text_config`, reused verbatim, not
a distinct drafter schema), per-head `q_norm`/`k_norm`, dense SwiGLU FFN (no
MoE tensors — `mlp.{gate,up,down}_proj`, width 17408, the SAME
`intermediate_size` as the base). Real tensor evidence (31 tensors, header
read via `safetensors.IndexFiles` — no data loaded):

```
fc.{weight,scales,biases}                         [5120, 1280] U32 (packed 4-bit; unpacked InDim 1280·8=10240=2·5120)
pre_fc_norm_embedding.weight                       [5120] BF16
pre_fc_norm_hidden.weight                          [5120] BF16
layers.0.input_layernorm.weight                    [5120] BF16
layers.0.self_attn.{q,k,v,o}_proj.{weight,scales,biases}
  q_proj OutDim 12288 = 2·(24·256)  — gated: q_proj emits [q ; gate], doubling rows (qwen35.go's own comment, confirmed here)
  k_proj/v_proj OutDim 1024 = 4·256 — num_key_value_heads=4, head_dim=256, matches the base
  o_proj OutDim 5120 = hidden_size
layers.0.self_attn.{q,k}_norm.weight               [256] BF16  (= head_dim)
layers.0.post_attention_layernorm.weight            [5120] BF16
layers.0.mlp.{gate,up,down}_proj.{weight,scales,biases}
  gate/up OutDim 17408 = intermediate_size; down OutDim 5120 = hidden_size
norm.weight                                         [5120] BF16
```

No embedding table, no LM head — the drafter shares the base's
(`mtp_use_dedicated_embeddings` / `tie_word_embeddings` both `false`). No
ordered-embedding / centroid head option either (unlike gemma4's assistant):
logits come straight off the base's own shared LM head.

**The combiner.** `fc` ([D, 2D], D = the base's hidden size — qwen's drafter
hidden size *equals* the base's, unlike gemma4's smaller assistant backbone)
projects the concatenation of two SEPARATELY RMS-normed D-vectors —
`pre_fc_norm_embedding(embed(t_{i+1}))` and `pre_fc_norm_hidden(h_i)`, the
base's hidden state that *produced* token `t_i` — into the head's own input
row: `x = fc([RMSNorm(embed(t), Enorm) ; RMSNorm(h, Hnorm)])`.

**State — the load-bearing difference.** The historical `headDrafter`
(`composed/mtp.go`) keeps a PERSISTENT session over the head's own `Stack`
— its own attention K/V, built purely from the sequence of committed
`(t_{i+1}, h_i)` pairs (`reset`/`observe`) — and clones that session
(`Snapshot`/`Restore`) to draft `k` tokens ahead without touching the live
state (`draftBlock`). It never reads or shares the base's own K/V cache. This
is a genuinely separate, self-contained decode session, not an attachment to
the base's.

**Verify discipline — and an important economics finding.** Composed's
default (non-`BlockVerify`) lane commits the base ONE canonical token at a
time: a draft is compared against the *already-known* next canonical greedy
(cheap — a head-logits matmul on a hidden the previous commit's REAL forward
already produced), so the base's own state never absorbs a speculative,
unverified token and nothing is ever rolled back. Composed's own doc comment
calls this **"a correctness-first slice"** and names the real throughput
lever explicitly: `generateBlockVerify`, which forwards the WHOLE draft block
through the base in ONE pass and needs `sess.Snapshot()`/`sess.Restore()` to
roll back a partial reject, is called out as **"deliberately out of this
correctness-first slice"** — a documented, deferred follow-up in the
reference implementation itself. In other words: even the shipped composed
lane only bought real speed in the optional batched-verify path; its default
path measured acceptance and saved nothing. This matters for part 4/5 below.

## 2. How the factory pair (gemma4, the studied precedent) differs — and why that's the blocker

`AssistantPair`/`AssistantModel` (`go/engine/metal/assistant_load.go`, 3603
lines) is built for a drafter with **no KV of its own at all**: it attends
directly into the TARGET's own K/V rows, matched by layer type + head_dim
(`validateNativeAssistantTargetTypes`, `targetKVByLayerTypeFromSession`) —
every drafter layer's type must resolve to a target KV stream of the same
head_dim. Rollback on a partial verify-reject is free there: a plain KV row
is simply overwritten by the next real write at the same position (`mtp.go`
lines 40-46, "Cache rollback on reject"). The combiner shape differs too:
gemma4's `pre_projection`/`post_projection` combine RAW
`[token_embed(backbone); previous_hidden(backbone)]` (no per-half norm) at
width `backbone*2 → hidden`; qwen's `fc` combines two independently-normed
halves at `2D → D` (D = the base's own hidden, not a separate smaller
backbone).

**The MTPMethod gap.** `mtp.MTPMethod` (`go/model/mtp/assistant_spec.go`)
declares exactly two values today: `MTPDraftModel` — documented explicitly
as projecting `[token embed ⊕ target hidden]` **"while sharing the target's
KV streams"** (gemma4's shape, word for word) — and `MTPDFlash`
(block-diffusion). Neither honestly describes qwen's own-KV, per-half-normed
shape. `AssistantSpec.Method` has no "unset/other" escape: an unstamped spec
(and a bare zero-value `Method` field) defaults to `MTPDraftModel` via
`resolveMTPMethod`/`AssistantPair.Method()`'s own fallback. Registering
qwen's drafter through `mtp.RegisterAssistant` today, even purely for its
`Parse` capability, would therefore SILENTLY claim the shared-KV contract —
a coherent-but-wrong declaration. Nothing currently reads it (see part 5 —
`LoadSpeculativePair`'s hybrid-target guard fires first, so no code path
reaches a qwen `AssistantPair` today), but it is a live trap for whoever
removes that guard later without reading this doc. **This lane deliberately
does not call `mtp.RegisterAssistant`** for exactly that reason. A third,
honest `MTPMethod` value is the correct fix, and it belongs in
`model/mtp` — outside this lane's fence (`model/` outside `qwen35` is
forbidden).

## 3. What verify lane hybrids get (once, or if, a pair exists)

This is architecture-neutral good news, worth stating plainly so it is not
mistaken for a further gap:

- The layer-major BATCHED verify row driver
  (`mtpRowsDriverEligible`, `go/engine/metal/mtp_rows_driver.go:81-88`)
  declines a `MixerGatedDelta` layer and a gated-attention layer
  (`s.gatedAttn[li] != nil`) **by design** — its own header: *"Any model
  that departs from that shape... declines the WHOLE block up front, before
  any GPU work starts, and the caller keeps today's row-major lane
  unchanged. Never a wrong-but-silent answer."* This is a scheduling/batching
  optimisation decline, not a correctness one.
- Greedy verify over a qwen hybrid target therefore rides the ordinary
  ROW-MAJOR per-token verify lane automatically whenever a pair exists —
  byte-identical to plain decode, simply un-batched (slower than gemma4's
  batched MoE verify on a uniform-MoE target, never wrong).
- `mtpVerifyFoldArmed(exact)` (`mtp.go:101-109`) — the sampled-vs-greedy fold
  routing rule — is already architecture-neutral: the batched small-K fold
  arms only for the non-byte-exact (sampled) lane by default, never the
  greedy lane unless the `LTHN_MTP_VERIFY_FOLD=1` A/B lever forces it. A
  hybrid pair would inherit this exact rule unchanged.

**Stated plainly**: nothing about serving a hybrid needs a new verify-lane
decision. The gap is entirely upstream of verify — no pair loads at all yet.

## 4. Why a genuinely *accelerating* pair needs target-side rollback — and why that's out of fence

A hybrid target's gated-delta layers thread a recurrent conv+delta state
(`arch_gated_delta.go`), not a KV cache. The only verify lane that gives real
throughput (part 1's finding) speculatively steps the target through K
drafted tokens before knowing which prefix is accepted, then must roll back
the REJECTED SUFFIX on a partial accept. For a plain-KV target that's a
free `pos` reset; for gated-delta layers the recurrent state must be
explicitly snapshotted and restored — it cannot be sliced by token range
(`docs/design-gd-state-blocks.md`, #62: *"a single running total, not a
per-token row... there is nothing to slice"*).

That primitive **already exists**: `gatedDeltaLayer.snapshotState()`
(`session_state_blocks.go:689-696`) and `(*gatedDeltaDeviceState).prime()`
(`lthn_gated_delta.go:375`), independently unit-tested
(`TestGatedDeltaBlockDeviceRun_*`'s "export/prime round trip"), and
documented in BOTH `session_state_blocks.go`'s own comment and
`design-gd-state-blocks.md` as **"orphaned as a live caller since #50
retired model/composed's `CloneState`"**. Wiring a new caller — snapshot
before a verify block, restore on partial reject — is real, scoped,
plausible follow-up work. But it lives in `session_state_blocks.go`,
explicitly **forbidden by this lane's fence**.

So, independent of the own-KV drafter shape (part 2) and the `MTPMethod` gap:
a per-token verify pair (mirroring composed's own default lane) is
fence-legal but — per part 1's economics finding — delivers **no throughput
win** over plain decode, only acceptance-rate telemetry; the batched lane
that would deliver one needs forbidden territory. Building new engine/metal
machinery (an own-KV decode stack for the head, a new fc-combiner op) to
land a lane that the reference implementation's own design notes say saves
nothing is a poor trade against this lane's remaining scope.

## 5. v1 scope: what this lane ships

- **qwen35 gains a real, tested drafter config parser + Arch derivation +
  weight-name mapping** (`go/model/arch/Qwen/qwen35/mtp_drafter.go`):
  `ParseDrafterConfig`, `Config.DrafterArch()`, `DrafterWeightNames()`,
  `DrafterTensorNames()`. `DrafterArch()` reuses `Config.Arch()` completely
  unchanged, called on a shallow copy that overrides only the layer count
  (`MTPNumHiddenLayers`) and forces the schedule to all-`full_attention`
  (`FullAttentionInterval = 1`) — every other dimension (hidden, heads,
  kv_heads, head_dim, rope, eps, FF, `AttnOutputGate`) is the base's own,
  because the real checkpoint's `text_config` *is* the base's `text_config`.
  Validated against BOTH a synthetic fixture (`model.Assemble` round-trip)
  and the REAL locally-cached checkpoint's header (31/31 tensor names
  reconciled, geometry cross-checked against the numbers in part 1) — this
  is genuine, checkable groundwork: a future lane wiring a real pair (via a
  new `MTPMethod`, or a bespoke own-KV engine loader) starts from a
  proven-correct config/weight map instead of re-deriving it from the
  historical composed source under time pressure.
- **This capability is deliberately NOT wired into any registry.**
  `model.RegisterArch`'s existing entry in `register.go` keeps its friendly
  standalone-load refusal **unchanged** — it is protected by the pre-existing
  `TestMTPDrafterRefusal_Bad`, which asserts the refusal names "MTP drafter"
  and directs to "lem pair"; replacing it with the new real `Parse` would
  make a standalone `lem generate <mtp-checkpoint>` fail deep inside
  `model.Assemble` instead (a clean, non-crashing, but far less helpful
  "model.embed_tokens absent" — confirmed safe: `model/assemble.go:128-129`
  hard-checks a nil `Embed` and returns a typed `core.NewError`, never a
  crash — but strictly worse UX than the existing named refusal for no
  functional gain, since nothing can serve the checkpoint standalone
  either way). `mtp.RegisterAssistant` is deliberately not called either —
  see part 2's `MTPMethod` mismatch. The new functions are plain,
  directly-callable, directly-testable package API, exactly the shape a
  follow-up pair-loader (in `engine/metal`, or wherever the `MTPMethod`
  question resolves) will want to call.
- **`LoadSpeculativePair`'s hybrid-target decline stays in force** for all
  seven released hybrid ids — nothing here makes any hybrid pair loadable —
  reworded to name PRECISELY what now exists (the drafter checkpoint parses
  as a real, weight-validated architecture) and what remains (an own-KV head
  forward with no target-KV sharing; a new `MTPMethod` or equivalent
  dispatch; batched-verify rollback wiring in forbidden `session_state_*`
  territory), replacing the previous vaguer "the factory pair route is
  pending".
- **Tests**: drafter config/Arch round-trip on synthetic tensors
  (`mtp_drafter_test.go`); a `model.Assemble` weight-name round-trip on a
  synthetic tensor set built from `DrafterTensorNames()` (plus a
  test-only synthetic embed/lm_head, clearly commented as not present on a
  real checkpoint, added solely to exercise `Assemble`'s per-layer
  resolution); a real-checkpoint header-only reconciliation receipt
  (`mtp_drafter_real_test.go`, behind the local-HF-cache skip pattern —
  confirmed present on this box: `mlx-community/Qwen3.6-27B-MTP-4bit`,
  snapshot `83795d546e9d328160e593fb0bf10b2bf2fe637e`); a clean-error (never
  crashes) receipt for the realistic no-embed tensor set, matching what a
  hypothetical standalone Assemble attempt would hit; and an updated
  refusal-contract test on `LoadSpeculativePair`'s reworded hybrid decline
  (`go/engine/metal/qwen_mtp_pair_test.go`).

## 6. What this lane will NOT do

- **No live pair load.** No drafting, no verifying, no `-draft` working on a
  qwen hybrid target. `lem generate -draft <mtp> <hybrid-base>` still
  declines exactly as before this lane (reworded message only, same
  functional refusal for all seven hybrid ids).
- **No new `mtp.MTPMethod` value, no `model/mtp` edits** — out of fence.
- **No own-KV decode stack for the drafter head on engine/metal** (no
  fc-combiner kernel, no small-transformer decode loop over the head's own
  K/V) — the real engineering content of a working pair, left to a follow-up
  lane once the `MTPMethod` question is resolved.
- **No gated-delta recurrent-state snapshot/restore wiring** for speculative
  rollback — the primitive exists (#62), a live caller does not;
  `session_state_blocks.go` is out of fence.
- **No change to the qwen35 standalone-load refusal message or behaviour** —
  `TestMTPDrafterRefusal_Bad` stays green, unedited.
- **No change to the verify-lane routing rules**
  (`mtpRowsDriverEligible`, `mtpVerifyFoldArmed`) — confirmed already
  architecture-neutral in part 3; nothing to change for hybrids specifically.
- **No MoE-base MTP drafter validated.** Only the dense 27B's drafter is
  locally cached; whether a real MoE-base (`qwen3_5_moe`) drafter checkpoint
  even carries `num_experts` in its own (base-reused) `text_config`, and
  whether `DrafterArch()` should honour it if so, is untested and unverified
  either way — `DrafterArch()` inherits the historical loader's behaviour
  (pass the effective config to the same generic FFN builder unconditionally)
  without independent evidence it is correct for that case.
