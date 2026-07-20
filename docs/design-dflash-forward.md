# DFlash engine forward — the real z-lab architecture (#52)

**Verdict up front: implementable now, as a block-forward primitive.** The real
z-lab DFlash drafter maths is fully pinned in-tree (`decode/dflash/zlab.go`, the
host-f32 reference the #37 survey cross-validated against the real
`z-lab/Qwen3-4B-DFlash-b16` checkpoint's own `modeling_dflash.py` executed
through transformers — see `docs/design-dflash-survey.md` §4-§5), and the
engine needs **no new kernels** to run it: every op is either a matmul (the
steel f32 GEMM above the established work floor, host f64 below — the qwen
vision port's exact two-tier pattern, `engine/metal/qwen_vision_encoder.go`) or
cheap host maths (RMSNorm, RoPE, softmax, SiLU). What is NOT implementable in
this lane is the live serve glue — every glue file is owned by a sibling
(`assistant_dflash*.go`, `speculative_model.go`, `serving/serve_draft.go`), and
one piece of its semantics (the reference `spec_generate` loop) is not fully
derivable in-tree. So the v1 here is: **arch package + engine forward,
oracle-gated; serving stays the honest typed decline** (`DFlashEngineProbe`
remains `false`, untouched), with the remainder named at the end.

## 1. Corrected premise: what actually exists in-tree

The task board said "find the dflash arch package under `go/model/arch/`".
**No such package exists** — that is this lane's first finding, not a blocker.
DFlash recognition-and-decline is spread across:

| Where | What |
|---|---|
| `go/decode/dflash/dflash.go` | The model-free contract: `ParseConfig` (recognises BOTH real export conventions), `BlockProposer` seam, the fuzz-proven lossless verify driver (`AcceptBlock`/`Generate`). |
| `go/decode/dflash/zlab.go` | The host-f32 reference forward for the real z-lab architecture — THE oracle. Pinned to real weights by `testdata/zlab_qwen3_4b_oracle.json` (env-gated `LTHN_DFLASH_ZLAB_CKPT`). |
| `go/model/gemma4/assistant_dflash.go` | Registers `model_type == "gemma4_dflash_assistant"` — a type **no published checkpoint uses** (survey §3). Sibling-owned; left alone. |
| `go/engine/metal/assistant_dflash*.go` | The pre-checkpoint engine forward the survey **falsified** (gemma4 sandwich layer, GELU, one shared anchor embedding, numAux-as-context-length, invented reduced head — survey §4's table). Sibling-owned; left alone. Its non-corrupting live aux tap (`assistant_dflash_livetap.go`) is real and engine-general — the future glue reuses it. |
| `go/serving/serve_draft.go` | `DetectDFlashDraft` + `DFlashDraftNotice` + `DFlashEngineProbe` (hardcoded `false`) — the honest decline. Sibling-owned; stays exactly as is until a forward is correct end-to-end. |

So the checkpoint parses and identifies, and cannot generate — the board is
right — and the missing pieces this lane CAN own are (a) a real arch package
(config + weight mapping + typed payload) and (b) a real engine forward.

## 2. What the real architecture computes per block

One drafter forward proposes a whole block. Inputs: `noiseEmbedding`
`[blockLen, hidden]` (the block's own candidate/mask-token embeddings, taken
from the TARGET's embedding table — the drafter has none) and
`targetHiddenRaw` `[ctxLen, numAux*hidden]` (the target's hidden states at
`dflash_config.target_layer_ids`, concatenated feature-wise per context
token). Then:

1. **Fused context, once per call**: `fc` (Linear `[hidden, numAux*hidden]`,
   bias-free) over each context row, then `hidden_norm` (plain RMSNorm).
   Context length is the number of target TOKENS fused in — it grows with the
   conversation; `numAux` only sizes the concatenated feature width. (The old
   engine forward conflated these — survey §4.)
2. **Five qwen3-style decoder layers** (`num_hidden_layers` = 5 in every
   published checkpoint), each plain pre-norm, TWO RMSNorms total:
   - `input_layernorm` → attention → residual add;
   - `post_attention_layernorm` → SwiGLU MLP (**SiLU**-gated, never GELU) →
     residual add.
3. **The attention is the diffusion mechanism**: q comes from the block's own
   normed hidden `[blockLen]`; k/v are the CONCATENATION of the shared fused
   context rows and the block's own normed rows `[ctxLen+blockLen]` — cross-
   and self-attention in ONE softmax, **no causal mask** (every block position
   sees every context row and every other block position). Per-head `q_norm`,
   and `k_norm` applied to the **concatenated** k. RoPE (standard split-half
   rotate-half, `rope_theta` 1e6): q at absolute positions
   `[ctxLen, ctxLen+blockLen)`, k across `[0, ctxLen+blockLen)`. GQA
   (heads 32 / kv 8 on the 4B drafter), scale `1/sqrt(head_dim)`.
4. **Final `norm`** (plain RMSNorm) → `[blockLen, hidden]` — the input the
   **borrowed TARGET lm_head** consumes. The checkpoint has no head and no
   embedding of its own; there is no d2t remap in this convention.

`decode/dflash/zlab.go` implements exactly this and is cross-validated; the
engine forward's contract is to reproduce it.

## 3. What the tensor inventory implies

Real tensor set (z-lab/Qwen3-4B-DFlash-b16, names verbatim — no `model.` or
`dflash.` prefix anywhere):

```
fc.weight                                   [hidden, numAux*hidden]   ([2560, 12800])
hidden_norm.weight                          [hidden]
norm.weight                                 [hidden]
layers.{0..4}.input_layernorm.weight        [hidden]
layers.{0..4}.post_attention_layernorm.weight [hidden]
layers.{0..4}.self_attn.q_proj.weight       [heads*head_dim, hidden]  ([4096, 2560])
layers.{0..4}.self_attn.k_proj.weight       [kv*head_dim, hidden]     ([1024, 2560])
layers.{0..4}.self_attn.v_proj.weight       [kv*head_dim, hidden]
layers.{0..4}.self_attn.o_proj.weight       [hidden, heads*head_dim]
layers.{0..4}.self_attn.q_norm.weight       [head_dim]
layers.{0..4}.self_attn.k_norm.weight       [head_dim]
layers.{0..4}.mlp.gate_proj.weight          [intermediate, hidden]    ([9728, 2560])
layers.{0..4}.mlp.up_proj.weight            [intermediate, hidden]
layers.{0..4}.mlp.down_proj.weight          [hidden, intermediate]
```

All bf16, all bias-free. **No** `embed_tokens`, **no** `lm_head`, **no**
`d2t` — the absence is architectural (borrowed target head), so the arch
package must NOT demand them and the serve glue must source embedding + head
from the target. Dims for the 4B drafter (from the in-tree oracle fixture,
which was computed from the real checkpoint): hidden 2560, heads 32, kv_heads
8, head_dim 128, intermediate 9728, layers 5, numAux 5, eps 1e-6, rope_theta
1e6; config extras (verbatim real config.json bytes, `decode/dflash`'s
`TestParseConfig_ZLabNative`): block_size 16, mask_token_id 151669,
target_layer_ids [1, 9, 17, 25, 33], model_type "qwen3", architectures
["DFlashDraftModel"].

~538M parameters total → ~1.1GB bf16. The v1 payload widens to owned host f32
(~2.2GB) — the same posture as mamba2/rwkv7/the qwen vision tower (payload is
ALWAYS host f32; the mmap is released after load). Resident bf16 device
buffers are a named perf follow-up, not v1.

## 4. What the engine forward needs

**No new kernels.** Op-by-op against what `engine/metal` already has:

| Op | Engine primitive |
|---|---|
| All projections (fc, q/k/v/o, gate/up/down) | `MatMulF32NT` (steel f32 GEMM) above the `1<<20` M·K·N work floor; ported host f64-accumulation GEMM below it — `qwenVisionMatNT`'s exact tier split, reused. |
| RMSNorm (plain, no gemma +1), per-head q/k norm | host f32/f64 — same tier the qwen vision port runs norms at. |
| RoPE (split-half, cos/sin table) | host — table computed per call over `ctxLen+blockLen`. |
| Joint cross+self softmax attention (non-causal, GQA) | host f64 attention core — the qwen vision tower's bidirectional attention shape (`qwenVisionAttentionForward`), with DFlash's asymmetric q-rows≠kv-rows. |
| SiLU | host (`qwenVisionSilu` shape). |

The block is small (blockLen ≤ 16, hidden 2560) — per call the dominant cost
is the ctx-row k/v projections and the MLP on 16 rows; the two-tier GEMM
covers the only shapes that matter. A fused device path (ICB-recorded chain,
resident weights) is the perf follow-up once the maths is proven live.

## 5. Session-contract analysis: does block-diffusion fit?

Yes — because DFlash is a **drafter**, not a primary decode mode. The
generation loop the engine session contract hosts stays plain autoregressive;
the diffusion happens entirely inside `BlockProposer.ProposeBlock`, and the
fuzz-proven `decode/dflash.AcceptBlock`/`Generate` verify driver guarantees
the emitted sequence is byte-identical to plain greedy decode regardless of
what the proposer returns. No session-contract extension is needed: the
existing propose→verify seam (`assistant_dflash_proposer.go`'s
`DFlashAuxSource` → `dflash.BlockProposer` bridge) is the right shape — it is
the CONTENTS of the drafter forward that were wrong, not the seam.

**One genuine evidence gap, named honestly**: the reference serve loop
(`spec_generate` in the checkpoint's `modeling_dflash.py`) — how many denoise
rounds per block, how the block re-seeds between rounds, and its KV-cache
bookkeeping across rounds — is NOT fully derivable in-tree. The survey
executed that file but vendored only the per-call forward semantics (which
`zlab.go` pins, including per-position noise seeding and position
convention). The block-forward primitive built here is correct per call
against real weights; the serve-loop policy above it needs the reference
re-inspected when the glue lane opens. That is a serving-lane concern and
does not block the forward.

## 6. v1 scope (this lane)

1. **`go/model/arch/z-lab/dflash/`** (new — vendor dir matches the HF org, as
   `zai-org/`, `rednote-hilab/` do): `Config` (decoder dims + the dflash
   block, recognition delegated to `decode/dflash.ParseConfig` — SPOR),
   `DraftModel`/`DraftLayer` typed payload (owned f32), `Assemble` (tensor
   map → validated payload, exact checkpoint names, honest missing/mis-shape
   errors), `Load(dir)`. **No `model.RegisterArch`**: a drafter is not a
   standalone servable model — registering it would invite a misload; its
   `model_type` is literally `"qwen3"`, which must keep resolving to the real
   qwen3 text arch.
2. **`go/engine/metal/dflash_zlab.go`** (new): `DFlashZLabForward(payload,
   noiseEmbedding, targetHiddenRaw, ctxLen, blockLen) → [blockLen, hidden]`
   — the §2 maths on the §4 tiers.
3. **Oracle gates**: synthetic-weight parity vs `decode/dflash.ZLabForward`
   (the cross-validated reference) below the GEMM floor AND above it (device
   tier exercised); shape/determinism/refusal suites Good/Bad/Ugly; an
   env-gated real-checkpoint test against the pinned fixture (skips cleanly —
   the local HF cache holds only a pruned stub of the checkpoint now, and this
   lane does not download).
4. **One additive seam edit** (`engine/metal/load.go`, `loadRegistered`): a
   typed decline when the MAIN-model path is handed a DFlash drafter dir —
   today that fails with a bewildering missing-`embed_tokens` error after
   resolving as plain qwen3. Blast radius: only DFlash-marked dirs, which
   cannot load today anyway; the assistant/-draft paths do not route through
   `loadRegistered` (verified: `LoadAssistantDir` reads config + mmap
   directly; `LoadSpeculativePair` uses `LoadDir` only for the TARGET;
   serving detects DFlash before any load).

## 7. Remainder (the glue lane — sibling-owned files, in order)

1. Live `DFlashAuxSource` for the real convention: per-layer aux hiddens at
   `target_layer_ids` via the existing non-corrupting live tap, PLUS the
   borrowed target embedding for block seeding (per-position mask-token
   embeddings — not the old single-anchor broadcast).
2. Re-point `DFlashDrafter`/`ProposeBlock` (assistant_dflash*.go) at this
   forward + the borrowed target lm_head for block readout (no reduced head,
   no d2t for z-lab); keep the speculators-convention reduced-head posture as
   a separate arm if that family is ever wanted.
3. Reconfirm the `spec_generate` round policy against the reference (§5's
   named gap) before arming.
4. Flip `DFlashEngineProbe` ONLY once a live draft→verify round trip against
   a real paired target lands an accept-rate receipt (survey §6 item 6-7's
   posture, unchanged).
5. Perf follow-ups: resident bf16 weights + fused device chain; skip re-fusing
   unchanged context rows across rounds.
