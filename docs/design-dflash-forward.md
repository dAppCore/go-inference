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

**Round policy reconfirmed 2026-07-20 (§7a) — no contradiction with the landed
v1 forward's single-call-per-block shape. Arming proceeds.**

1. ~~Live `DFlashAuxSource` for the real convention: per-layer aux hiddens at
   `target_layer_ids` via the existing non-corrupting live tap, PLUS the
   borrowed target embedding for block seeding (per-position mask-token
   embeddings — not the old single-anchor broadcast).~~ **Done, with one
   caveat — see §7b: the live tap (`ExtractAuxHiddensLive`) is single-boundary-
   position-shaped and cannot supply the reference's multi-row per-round need
   without a session-stepping redesign paired to the verify oracle's own
   posture; the landed `ExtractAuxHiddensAllRaw` instead widens the EXISTING
   throwaway `ForwardCaptureHiddens` primitive `ExtractAuxHiddens` already
   used (same full-prefix-replay tier `generateDFlash`'s own verify oracle
   already has) to return every row. Per-position mask-token seeding IS
   real (`zLabDFlashProposer.ProposeBlock`, `assistant_dflash_zlab.go`).**
2. ~~Re-point `DFlashDrafter`/`ProposeBlock` (assistant_dflash*.go) at this
   forward + the borrowed target lm_head for block readout (no reduced head,
   no d2t for z-lab); keep the speculators-convention reduced-head posture as
   a separate arm if that family is ever wanted.~~ **Done — a new,
   z-lab-specific `zLabDFlashDrafter`/`zLabDFlashProposer`
   (`assistant_dflash_zlab.go`), NOT a repoint of the speculators-convention
   `DFlashDrafter` (kept exactly as landed, the separate arm this item
   asked for). Reads each proposed position off the TARGET's own
   `headLogitsScratch`, never a reduced head.**
3. ~~Reconfirm the `spec_generate` round policy against the reference (§5's
   named gap) before arming.~~ **Done — see §7a.**
4. **Still not flipped — re-measured, see §7c.** §7b's 0.00 receipt pre-dated
   the target hidden-capture fix that item named as the blocker. That fix has
   since landed; re-measuring against it (§7c) surfaced and fixed one further,
   genuinely in-lane bug (an anchor-token phantom context row), but the
   resulting accept-rate — real and non-zero now, unlike §7b — is still well
   under a healthy bar. `DFlashEngineProbe` stays `false`.
5. Perf follow-ups: resident bf16 weights + fused device chain; skip re-fusing
   unchanged context rows across rounds. (Now a shared item with §7b's fix —
   both point at the same incremental-capture direction.)

## 7a. Round-policy reconfirmation (banked evidence, 2026-07-20)

Fetched directly from the hub repo `z-lab/Qwen3-4B-DFlash-b16` (the checkpoint's
own current files, via the HF filesystem tool — not the stale local cache,
which held only a weightless `refs/main` stub). Repo listing: `assets/`,
`.gitattributes`, `README.md`, `config.json`, `dflash.py` (12,340 bytes),
`model.safetensors` (1,074,860,568 bytes ≈ 1.075GB — confirms design doc §3's
"~538M parameters → ~1.1GB bf16" estimate), `modeling_dflash.py` (13,271
bytes), `utils.py` (5,731 bytes). No separate eval/generate script exists in
the repo — `spec_generate` is the only generation loop.

`config.json`'s `"auto_map": {"AutoModel": "dflash.DFlashDraftModel"}` makes
**`dflash.py`** the config-declared canonical load path for
`trust_remote_code=True`. The repo also carries a byte-different but
semantically-identical `modeling_dflash.py` (imports inlined instead of
sourced from `utils.py`; `dflash.py` reads `target_layer_ids`/`mask_token_id`
from `config.dflash_config` with a computed `build_target_layer_ids` fallback,
`modeling_dflash.py` always recomputes `target_layer_ids` and takes
`mask_token_id` as an explicit `spec_generate` argument). For this checkpoint's
config the fallback formula reproduces the declared `[1, 9, 17, 25, 33]`
exactly, so both files behave identically here; both were read in full and the
round policy below holds in both. `README.md`'s Transformers quick-start
confirms `spec_generate` as the checkpoint's own documented entry point:

> ```python
> generate_ids = model.spec_generate(
>     input_ids=model_inputs["input_ids"], max_new_tokens=2048,
>     temperature=0.0, target=target, stop_token_ids=[tokenizer.eos_token_id]
> )
> ```

Answering §5's three named unknowns, from `DFlashDraftModel.spec_generate`:

1. **Denoise rounds per block: exactly ONE.** The decode loop
   (`while start < max_length:`) calls the drafter forward exactly once per
   block:
   > `draft_logits = target.lm_head(self(target_hidden=target_hidden,`
   > `noise_embedding=noise_embedding, ..., use_cache=True,`
   > `is_causal=False)[:, -block_size+1:, :])`
   > `block_output_ids[:, 1:] = sample(draft_logits)`
   No inner loop re-feeds the block through the drafter — "block diffusion"
   here is one parallel forward predicting the whole masked block at once
   (single-shot block-parallel drafting, closer to Medusa/EAGLE-style block
   drafting than iterative denoising), exactly the single-call shape
   `DFlashZLabForward` already has. **Confirms, not contradicts, the landed
   v1 primitive** (and matches `decode/dflash.BlockProposer.ProposeBlock`,
   already a single `context → []int` call, and `Generate`'s one
   `ProposeBlock` call per loop iteration).
2. **Re-seeding is between BLOCKS, not "between rounds" within one** (there is
   one round per block). `output_ids` is globally pre-filled with
   `mask_token_id` (`torch.full((1, max_length+block_size), mask_token_id,
   ...)`); each new block window `block_output_ids =
   output_ids[:, start:start+block_size].clone()` has position 0 = the
   just-verified real token (previous round's accepted/corrected/bonus
   token), positions `1..block_size-1` = the literal, unwritten
   `mask_token_id`. `noise_embedding = target.model.embed_tokens(
   block_output_ids)` — sourced from the TARGET's own embedding table,
   confirming §2/§3's "no embedding of its own; borrowed from target".
3. **KV-cache bookkeeping**: target KV (`past_key_values_target`) accumulates
   through prefill + every verify call, then `.crop(start)` drops
   rejected-tail entries to the new accept boundary each round. Draft KV
   (`past_key_values_draft`) is genuinely incremental, not reset per round:
   `target_hidden` is reassigned each round to only
   `[:acceptance_length+1]` of the just-verified target hidden states (NOT
   the full growing context), fused via `fc`/`hidden_norm` and appended to
   the draft cache alongside the block's own noise rows; then
   > `past_key_values_draft.crop(start)`
   (using `start` BEFORE this round's `start += acceptance_length + 1`)
   discards exactly the current block's speculative noise-row K/V while
   KEEPING the newly-fused context-gap K/V. The persistent draft cache thus
   accumulates one round's worth of context rows at a time — the mechanism
   §7 item 5 already names as the perf follow-up ("skip re-fusing unchanged
   context rows across rounds"). The v1 engine forward instead recomputes the
   full context every call; **the two are mathematically equivalent** given
   matching absolute RoPE positions (context rows `[0, ctxLen)`, block rows
   `[ctxLen, ctxLen+blockLen)` — confirmed against `apply_rotary_pos_emb`'s
   tail-slice for q, `cos[..., -q_len:, :]`, vs full-width for k), just less
   efficient. **No correction to the landed maths is needed.**

**Acceptance-rule cross-check** (reference vs the already-landed Go driver):
the reference's
> `acceptance_length = (block_output_ids[:, 1:] ==`
> `posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()`
> `output_ids[:, start:start+acceptance_length+1] = block_output_ids[:, :acceptance_length+1]`
> `output_ids[:, start+acceptance_length+1] = posterior[:, acceptance_length]`
is longest-common-prefix-then-bonus-token — the same rule `decode/dflash.
AcceptBlock` (`go/decode/dflash/dflash.go:230-245`) already implements: walk
while `proposed[i] == next(seq)`, commit the target's own token each step,
stop and commit the correction at first divergence, else commit one bonus
token after a full match.

**Verdict: no contradiction anywhere in the chain (reference round policy ↔
landed engine-forward call shape ↔ landed Go verify driver). Arming (§7
items 1, 2, 4) proceeds.**

## 7b. Live glue landed; DFlashEngineProbe withheld (evidence, 2026-07-20)

**What landed** (`assistant_dflash_zlab.go`, new; `assistant_dflash_proposer.go`
+ `ExtractAuxHiddensAllRaw`; `speculative_model.go`'s `zlab` field +
`generateDFlashZLab`): `LoadSpeculativePair` recognises the z-lab convention
from `draftPath/config.json` alone (`zlabdflash.ParseConfig`, before
`LoadAssistantPairDirs` would even be attempted — that loader demands
`model.embed_tokens.weight`, which this convention architecturally lacks) and
routes to a wholly separate load path. `zLabDFlashProposer.ProposeBlock` seeds
each block from the TARGET's own embedding table (the real anchor at position
0, `MaskTokenID`'s embedding at every other position) and reads each proposed
position off the TARGET's own borrowed `headLogitsScratch` + argmax — never a
reduced draft vocab. The speculators-convention `DFlashDrafter`
(`assistant_dflash.go`) is untouched, the separate arm §7 item 2 asked for.

**The receipt** (`TestSpeculativeModel_DFlashZLab_RealPairedGenerate`,
`assistant_dflash_zlab_test.go`, env-gated, real
`z-lab/Qwen3-4B-DFlash-b16` + real `Qwen/Qwen3-4B`):

- **Determinism**: PASS — the same prompt twice yields byte-identical token
  sequences.
- **Losslessness**: PASS — the paired sequence is byte-identical to the SAME
  target's plain greedy decode (`ArchSession.Generate`), confirmed a second,
  independent way too: loading the target PLAINLY through
  `inference.LoadModel` (no drafter at all) and running the identical Chat
  request reproduces the exact same tokens.
- **Accept-rate**: **0.00** (180 proposed, 0 accepted, over 12 rounds on a
  12-token greedy completion of "The capital of France is"). Lossless, but
  not yet accelerating.

**Root cause of the 0% accept-rate — NOT this lane's wiring.**
`ExtractAuxHiddensAllRaw` widens `ForwardCaptureHiddens`'s existing per-layer
capture to every context row (`assistant_dflash_proposer.go`'s doc); the bug
is one layer down, in `ForwardCaptureHiddens` itself
(`train_session.go`/`decode_forward_arch.go` — outside this lane's fence:
`dflash*`/`assistant_dflash*`/z-lab only). Two independent checks, both
against the real Qwen/Qwen3-4B target:

1. **External cross-validation**: an independent `transformers`/`torch`
   extraction (venv at `/Users/snider/PyCharmMiscProject/.venv`, torch 2.13.0
   + transformers 5.5.4, MPS) of the SAME 5-token prompt's `hidden_states` at
   the SAME `target_layer_ids` (`[1, 9, 17, 25, 33]`, `extract_context_
   feature`'s own `+1` HF-embedding-slot convention) diverges from
   `ExtractAuxHiddensAllRaw`'s output by ~2× total magnitude (sumAbs got
   ≈221,719 vs want ≈117,451; max abs diff ≈4,837 on values with hidden std
   ~120+ at that depth) — reproduced identically with the ICB replay path
   AND with it disabled (`LTHN_DECODE_ICB=0`), so it is not ICB-specific. The
   divergence is worst at the FIRST token and at the DEEPEST sampled layers —
   consistent with a per-layer capture-bookkeeping fault, not a wrong
   `target_layer_ids` list (the literal `[1,9,17,25,33]` values were shared
   byte-for-byte between the Python reference and the Go call in this check,
   bypassing config-parsing entirely).
2. **Within-engine cross-check**: `ForwardCaptureHiddens`' OWN final-layer
   row, read through the SAME session's head, agrees EXACTLY with the
   ordinary `PrefillTokens`+`BoundaryLogits` decode path's greedy prediction
   for the identical prefix (both pick token 43614). So the FINAL output is
   right; only the INTERMEDIATE per-layer captures — exactly what a DFlash
   aux tap needs — are wrong. `decode_forward_arch.go` already carries a
   comment naming this exact bug CLASS from a prior incident (#391, "the
   captured hiddens disagreed with the serving forward" from a recorded-
   op-boundary misalignment) — this looks like the same family recurring for
   a real, large (36-layer, GQA) architecture, not a new defect this lane
   introduced.

**Consequence**: `DFlashEngineProbe` (`serving/serve_draft.go`) stays
`false` — §7 item 4's own gate ("ONLY once a live draft→verify round trip
against a real paired target lands an accept-rate receipt") is not met. The
architecture, the load path, and the proposer are all correct and proven
lossless; the drafter is fed the wrong numbers by a primitive this lane does
not own. **Follow-up** (a different lane, `train_session.go`/
`decode_forward_arch.go`'s owner): re-validate `ForwardCaptureHiddens`'
intermediate-layer capture against a real large model (repro: the exact
prompt tokens `[785, 6722, 315, 9625, 374]` — "The capital of France is" —
target_layer_ids `[1, 9, 17, 25, 33]`, expected sums logged above), fix, then
re-run `TestSpeculativeModel_DFlashZLab_RealPairedGenerate` and flip the
probe once accept-rate is meaningfully above 0.

## 7c. Accept-rate re-measured post-capture-fix; a second, in-lane bug found and fixed

The follow-up §7b named (`ForwardCaptureHiddens`'s intermediate-layer capture)
has since landed: every one of the 5 aux layers `[1, 9, 17, 25, 33]` — and
every OTHER decoder layer, 0 through 35, not just those spot-checked — now
sits within ±0.25% of an independent transformers/torch oracle
(`capture_hidden_qwen3_oracle_test.go`'s `TestForwardCaptureHiddensQwen3
AllLayersVsRealOracle`, both the ICB-replay and plain capture routes), and a
plain greedy decode of "The capital of France is" now completes correctly
(" Paris"). §7b's own named re-measurement step is due.

**Re-measurement.** `TestSpeculativeModel_DFlashZLab_RealPairedGenerate`
(the single-prompt, 12-token receipt) now reads a non-zero accept-rate for
the first time — a real change from §7b's 0.00, and direct evidence the
capture fix was the blocker that test named. A single 12-token sample is too
small to read as a stable number, so a second, purpose-built instrument
(`TestSpeculativeModel_DFlashZLab_AcceptRateSurvey`,
`assistant_dflash_zlab_acceptrate_test.go`) drives the same real pair over
several hundred draft/verify rounds across a mixed prompt set (prose
continuation, narrative, code, arithmetic, a repetitive pangram) and reports
the accepted-per-round distribution alongside the aggregate.

**Bisection before accepting that number.** Two remaining candidate fault
classes were checked and both come back clean, ruling them out as
explanations for a still-low accept-rate:

1. **Target hidden fidelity at the drafter's own tap layers** — already
   covered by the all-layer re-measurement above; ±0.25% at every one of the
   5 `target_layer_ids`, not a coarse spot-check.
2. **The drafter's own forward maths** — `TestDFlashZLabForward_RealCheckpoint`
   (`dflash_zlab_test.go`) still holds the engine forward within its 0.02
   relative band against `decode/dflash.ZLabForward`'s independently
   cross-validated oracle fixture, computed from the same real checkpoint's
   weights. Untouched by anything in this pass.

**A third, genuinely in-lane bug, found by reading the checkpoint's own
`spec_generate`/`modeling_dflash.py` line by line rather than re-deriving its
position convention from description.** Its decode loop tracks one absolute
position index per already-committed token (`start`, advanced by
`acceptance_length + 1` each round) and fuses ONLY already-verified hidden
states strictly BEFORE that index into `target_hidden`
(`extract_context_feature(...)[:, :acceptance_length+1, :]` — a span that
ends at the anchor's predecessor; the prefill-time seed is the same
convention over just the prompt positions). The anchor token — the most
recently committed one — enters the forward SOLELY through the block's own
position-0 noise embedding, re-embedded fresh; `apply_rotary_pos_emb`'s
query-tail slice (`cos[..., -q_len:, :]`) confirms that embedding's RoPE
position is the anchor's own absolute index, i.e. one past the LAST true
context row, never overlapping it.

The landed live glue (`zLabDFlashProposer.source`, wired in
`speculativeModel.generateDFlashZLab`) instead extracts hidden states for
EVERY position of the running sequence via `ExtractAuxHiddensAllRaw`,
including the anchor as `targetHiddenRaw`'s own final row, and passed the
resulting full count straight through as `ctxLen`. Two compounding
consequences: a phantom context row representing the anchor's mid-stack
hidden state that the checkpoint's own convention never produces, and —
because `DFlashZLabForward` ROPEs the block starting exactly at `ctxLen`
(`dflash_zlab.go`'s own documented convention, itself correct) — the block
seeded one absolute position further along than the checkpoint places it.

**Fix** (`zLabDFlashProposer.ProposeBlock`, `assistant_dflash_zlab.go`):
trim that final context row — and decrement the `ctxLen` count that came with
it — once, at the seam, before the forward call. `ExtractAuxHiddensAllRaw`
and every `zLabDFlashSource` implementation keep their existing "every
position of the running sequence" contract unchanged; only this call site
now knows the anchor's row is not part of the checkpoint's own notion of
context. `DFlashZLabForward`'s own RoPE maths is untouched — it was already
correct given a properly-sized `ctxLen`.

**Receipt** (`TestSpeculativeModel_DFlashZLab_AcceptRateSurvey`, the same 8
prompts, temperature 0, 48 tokens each, before vs after, real
z-lab/Qwen3-4B-DFlash-b16 + real Qwen/Qwen3-4B, losslessness holding both
ways — every emitted token byte-identical either side of this change):

| | rounds | proposed | accepted | mean accepted/round | accept-rate | tokens/round |
|---|---|---|---|---|---|---|
| before this fix | 204 | 3,060 | 190 | 0.931 | 6.21% | 1.882 |
| after this fix | 196 | 2,940 | 197 | 1.005 | 6.70% | 1.959 |

Fewer verify rounds needed for the same 384 output tokens either way the
comparison is read; the fix wins the aggregate and 5 of the 8 individual
prompts (most clearly on the more deterministic ones — code completion,
counting — where per-token noise is lowest and a real signal is easiest to
see through it).

**Honest conclusion.** The fix is real and kept — grounded in the
checkpoint's own reference code, not a guess, and it measurably helps. But
mean accepted tokens per round (~1.0) remains well under a healthy bar
(≥2 of a block) even after it: `DFlashEngineProbe` stays `false`. Both
remaining candidate fault classes this pass could check — target hidden
fidelity at every layer, and the drafter's own forward maths against its
real-checkpoint oracle — are independently clean, so the residual gap is not
attributable to either. What is left open, named honestly rather than
guessed at: whether this specific 4B drafter/target pairing simply has low
inherent quality on open-ended continuation, or some other factor this pass's
fence did not reach. The v1 forward's full-context-replay-every-round posture
(vs the checkpoint's own persistent incremental cache) is NOT itself
suspected — the two are per-row mathematically equivalent operations
(`fc`/`hidden_norm`/projections are row-independent; only the softmax
combines rows, over the identical row set either way) — but that equivalence
has not been checked against a live accept-rate number from the reference
implementation itself, because none exists to compare against.

## 7d. The residual accept-rate gap bisected to numeric regime, not structure (evidence, 2026-07-21)

**The missing comparator now exists.** §7c ended honestly open on whether
~1.0 accepted/round was this pairing's inherent quality or an unreached
fault. The reference implementation's own accept-rate, measured for the
first time (the checkpoint's `spec_generate` run verbatim on torch/MPS —
scratchpad `dflash_ref_survey.py`, its method source re-exec'd with only the
return widened to expose `acceptance_lengths`; same 8 prompts, 48 greedy
tokens, no early stop): **rounds=115, accept-rate 17.10%, 3.565
tokens/round** vs our survey's 7.8% / 2.087 (re-run same day). The pairing
is healthy; the gap was real. Per-prompt, the gap concentrates on
deterministic sequences (counting: reference 9.33 tokens/round vs ours
1.82) while repetition/code nearly match (fox 6.00 vs 5.78, fib 3.71 vs
3.43).

**The hunt, stage by stage (instruments committed:
`assistant_dflash_zlab_round1_test.go`, `assistant_dflash_zlab_xfeed_test.go`;
torch dumps via scratchpad `dflash_dump_ctx.py`):**

- Round-1 proposal diff (identical frames, anchors match 7/8): fib matches
  the reference 9 positions deep — the forward is fundamentally right —
  while prose prompts mismatch at slot 0.
- Torch-exact context features fed to OUR forward change almost nothing vs
  our live capture (colours: byte-identical proposal lists) — capture
  fidelity is NOT the gap, closing §7b's last suspicion about
  `ExtractAuxHiddensAllRaw`.
- Stage isolation on torch-exact inputs: fused context (fc+hidden_norm)
  within 0.3%; layer-0 attention on torch's own normed input within 0.6%;
  full per-layer bisect within 0.7–2.5% at every position through all 5
  layers; the target embedding row byte-exact (2560/2560 elements equal).
  No structural fault exists in the engine forward. (A first bisect run
  showed 55–112% divergence at position 0 — that was THIS harness's own
  bug, the embedID shared-scratch aliasing the survey test's comment warns
  about, pin-fixed in both instruments; the corrected numbers are the ones
  above.)

**Mechanism.** The drafter's proposal argmaxes are knife-edge: the SAME
forward flips fib's match depth 9 → 1 between two context inputs that
differ by ±0.25%. The checkpoint's tie-breaks were shaped by torch-bf16
arithmetic (training and reference inference share it); our forward —
f32/f64 host + steel f32, MORE precise but differently rounded — lands on
the other side of enough ties to halve the accept-rate, worst where the
reference locks onto a deterministic pattern and we fall off it.

**Consequence.** `DFlashEngineProbe` stays `false`. The one honest fix
direction is a torch-bf16-equivalent numeric mode for the drafter forward
(bf16 rounding after each op, torch reduction orders) — a
fidelity-to-reference lane, not a correctness fix; the engine forward is
already correct by every structural measure above. Whether that lane is
worth building for a non-gemma4 side drafter is a prioritisation call, not
an engineering unknown.
