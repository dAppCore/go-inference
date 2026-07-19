# DFlash survey (#37): from recognised scaffolding to a verified drafter

**Verdict.** DFlash is a **real, distinct external drafter family** — not a stale
backlog entry and not an internal codename. z-lab (the arXiv paper's own lab) and
RedHatAI publish real checkpoints on Hugging Face, one of z-lab's own with over
226,000 downloads. The engine-side forward this repo already carries
(`engine/metal/assistant_dflash*.go`, landed 2026-07-12 → 2026-07-16) was built
from the arXiv description and a design memo **before any real checkpoint had
been inspected** — its own commit message says so ("no public gemma-4 DFlash
checkpoint exists to measure accept-length against"). That is no longer true.
Checked against three real checkpoints across the two conventions the wild
actually uses, the built forward's **maths shape does not match either
convention** in several concrete, evidenced ways (below) — not a stub, not
phantom, but built against an assumed architecture now falsifiable and falsified.

This session (a) confirms the family is real with hub evidence, (b) itemises
every mismatch between the existing engine forward and the two real
conventions, and (c) ships a **new, independently oracle-gated host-f32
forward** for the real (dominant) z-lab convention, validated against a
downloaded real checkpoint's actual weights — proof the *correct* architecture
is now understood and implemented, even though it is not yet the one wired to
the GPU engine. The GPU engine forward is UNCHANGED this session (see
"What's not done" below for why, and the exact punch list).

## 1. What exists in the tree

| File | What it is |
|---|---|
| `go/decode/dflash/dflash.go` | Model-free, GPU-free drafter contract: config recognition (`ParseConfig`), the `BlockProposer` seam, the greedy block-verify driver (`AcceptBlock`/`Generate`) — proven lossless by a 20,000-case adversarial fuzz (spec-on == spec-off) per its landing commit. Untouched by the mismatch below: the verify driver's losslessness does not depend on what the proposer proposes. |
| `go/model/mtp/assistant_spec.go` | The reactive attached-drafter registry (`AssistantSpec`/`AssistantConfig`/`MTPMethod`). Declares `MTPDFlash` as a first-class method alongside `MTPDraftModel`. |
| `go/model/gemma4/assistant_dflash.go` | Registers `model_type == "gemma4_dflash_assistant"` → `mtp.MTPDFlash`, assuming a gemma4-family decoder (nested-or-flat `text_config`, `backbone_hidden_size`). Its own doc comment already flagged this as provisional: *"A z-lab / RedHatAI DFlash checkpoint declares its base model_type (qwen3, llama, …) with the dflash marker beside it; recognising THAT … is the follow-up once such a checkpoint is on the box."* That checkpoint is now on the box (see §2) — no published checkpoint uses `gemma4_dflash_assistant`, or a gemma4 decoder, at all (§3). |
| `go/engine/metal/assistant_dflash*.go` | The block-parallel GPU draft forward: fused-context projection (`dflash.aux_projection`), per-layer KV injection reusing the gemma4 sandwich-norm layer (`draftLayerIntoScratch`), block-parallel readout, reduced-vocab head + `d2t` remap. Oracle-gated against a **synthetic** bf16 host reference (parity + lossless-engagement + loader tests) — internally consistent, never checked against a real checkpoint's shape. `assistant_dflash_livetap.go` (the non-corrupting live aux-hidden tap) is real, engine-general, and unaffected by the architecture mismatch — it taps *any* model's per-layer hiddens, gemma4 or otherwise. |
| `go/serving/serve_draft.go` | `DetectDFlashDraft` (recognises + parses), `DFlashDraftNotice` (honest decline), `DFlashEngineProbe` (**hardcoded `return false`**), `ArmDFlash` = `IsDFlash() && DFlashEngineProbe()`. |

The `docs/design-dflash.md` memo `engine/metal/assistant_dflash*.go`'s own doc
comments still cite was **retired** 2026-07-14 (`290811cf`, "implemented work
verified, remainders tasked") once the contract slice + engine forward
landed — its retirement note says the live lane was "gated on the
non-corrupting aux tap exactly as documented → task #4". `ExtractAuxHiddensLive`
(the non-corrupting tap) landed the same day, closing that specific stated
blocker, but `DFlashEngineProbe` was never flipped afterwards — it is still
`false` as of this survey (`e3e9c521`, 2026-07-19). So even by the *old*,
now-falsified architecture assumption, the last documented step (flip the
probe) was simply never taken; this survey supersedes that plan anyway, since
the architecture it would have armed does not match a real checkpoint.

## 2. Is DFlash a real external family? (yes — hub evidence)

```
huggingface_hub.HfApi().list_models(search='dflash')
```

Real, actively published checkpoints, several very widely downloaded:

| Repo | Downloads | Publisher |
|---|---|---|
| `z-lab/Qwen3.6-35B-A3B-DFlash` | 226,098 | z-lab (arXiv 2602.06036's own lab) |
| `z-lab/Qwen3-8B-DFlash-b16` | 113,210 | z-lab |
| `z-lab/Qwen3.5-27B-DFlash` | 93,362 | z-lab |
| `z-lab/Qwen3.6-27B-DFlash` | 70,404 | z-lab |
| `z-lab/gemma-4-26B-A4B-it-DFlash` | 18,809 | z-lab (drafts for gemma-4 26B-A4B) |
| `z-lab/Qwen3-4B-DFlash-b16` | 25,351 | z-lab — **the checkpoint this survey downloaded and validated against** |
| `RedHatAI/NVIDIA-Nemotron-3-Super-120B-A12B-speculator.dflash` | 249 | RedHatAI (vLLM "speculators" library convention) |
| `RedHatAI/NVIDIA-Nemotron-3-Ultra-550B-A55B-speculator.dflash` | 109 | RedHatAI |

z-lab's own convention dominates real-world usage by a wide margin (its
smallest-target checkpoint alone has 100x RedHatAI's largest). Every z-lab
checkpoint — **including the ones drafting for gemma-4 targets** — ships a
**qwen3-architecture decoder**, never a gemma4 one (§3).

## 3. What loads today vs what refuses

- **Via `mtp.ParseAssistantConfig`** (the reactive dispatch `LoadAssistantDir`
  uses): keys on top-level `model_type`. `model/gemma4/assistant_dflash.go`
  claims only `"gemma4_dflash_assistant"`. **No checkpoint found by this survey
  uses that model_type or a gemma4 decoder** — z-lab's own config.json sets
  `"model_type": "qwen3"` even for its gemma-4-targeting checkpoints;
  RedHatAI's speculators-convention config carries **no top-level `model_type`
  at all** (the decoder arch is nested under `transformer_layer_config`, a
  third, different nesting key from both `text_config` and `dflash_config`).
  Net: **no real checkpoint this survey found resolves through the existing
  reactive assistant-spec registration.**
- **Via `LoadDFlashDrafter`** (the model-free `dflash.ParseConfig` + ordinary
  pack loader): before this session, `ParseConfig` recognised only
  `speculators_model_type == "dflash"` — matching RedHatAI's convention
  (mostly — see §4) but not z-lab's (no such field in any z-lab config.json).
  After this session (§5), `ParseConfig` recognises **either** real convention.
  Loading still ultimately depends on the decoder arch resolving through
  `mtp.ParseAssistantConfig`, so a real checkpoint still fails at that stage
  today — the gap is now precisely located, not vague.
- **Via `serving`**: `DFlashEngineProbe` hardcoded `false` — any detected
  DFlash checkpoint (whichever convention) degrades to plain autoregressive
  with `DFlashDraftNotice`, by design ("the honest decline, never a misload").
  That posture is correct and should **stay** false until the engine forward
  actually matches a real checkpoint's maths (§4) — arming today's forward
  against a real checkpoint would silently produce wrong numbers (GELU where
  the checkpoint needs SiLU, a 4-norm sandwich layer where it has 2, etc.),
  which is a *worse* outcome than the current honest decline.

## 4. The gap: engine/metal's assumed architecture vs the two real conventions

Checked against `z-lab/Qwen3-4B-DFlash-b16` (downloaded; config + real
safetensors tensor names/shapes) and `z-lab/gemma-4-26B-A4B-it-DFlash` /
`z-lab/gemma4-12B-it-DFlash` (config only) for the z-lab convention, and
`RedHatAI/NVIDIA-Nemotron-3-Super-120B-A12B-speculator.dflash` (config only)
for the speculators convention. The z-lab checkpoint's own `modeling_dflash.py`
was additionally **executed** (transformers, `trust_remote_code=True`) and
compared against an independent numpy re-implementation of the same maths —
max abs diff 7.3e-5 at final-norm output scale (std ≈ 2.6) — so the reading
below is cross-validated against real code execution, not just against source.

### Config-level

| | Existing code assumed | z-lab real (dominant) | RedHatAI real (speculators) |
|---|---|---|---|
| Marker | `speculators_model_type=="dflash"` (`decode/dflash`) or `model_type=="gemma4_dflash_assistant"` (`model/gemma4`) | `architectures:["DFlashDraftModel"]`, no `speculators_model_type` | **Both** `speculators_model_type:"dflash"` AND `architectures:["DFlashDraftModel"]` |
| Fused layers field | `aux_hidden_state_layer_ids` (top-level) | `dflash_config.target_layer_ids` (nested) | `aux_hidden_state_layer_ids` (top-level) ✓ |
| Mask token | *(field did not exist in `Config`)* | `dflash_config.mask_token_id` (nested) | `mask_token_id` (top-level) |
| Verifier name | `speculators_config.verifier.name` | *(field does not exist)* | `speculators_config.verifier.**name_or_path**` — the code read the wrong key; every real checkpoint uses `name_or_path` |
| Decoder arch nesting | flat or `text_config` (`model/gemma4`) | flat, `model_type:"qwen3"` | `transformer_layer_config` — a third key |

### Forward-maths level (the part that actually computes numbers)

| | `engine/metal/assistant_dflash.go` (built pre-checkpoint) | Real (z-lab, verified against executed `modeling_dflash.py`) |
|---|---|---|
| Decoder layer | gemma4 **sandwich-norm** (4 layernorms), reused verbatim from the MTP path (`draftLayerIntoScratch`) | Plain **pre-norm**, 2 layernorms (`input_layernorm`, `post_attention_layernorm`) — standard qwen3 |
| MLP gate | **GELU**-gated (gemma4's `GeluGateMulBF16Into`) | **SiLU**-gated (SwiGLU) — the exact activation-mismatch bug class this repo's own CLAUDE.md already names for the qwen hybrid lane ("coherent-but-wrong text below the greedy argmax threshold") |
| K normalisation | none applied to injected K rows | `k_norm` (RMSNorm) applied to the **concatenated** K (context rows + block's own rows together) |
| Fused-context construction | one `aux_projection` MatVec, no following norm | `fc` (Linear) **then** `hidden_norm` (RMSNorm) — the norm step is missing entirely in the built code |
| Cross-attention context length | `numAux` (the **count of fused layers**, e.g. 2–6) | the number of **target TOKENS** fused in (arbitrarily large, grows with context) — the built code conflates "how many layers we fuse" with "how many context positions we attend over" |
| Intra-block attention | none — every block position reads only the shared injected context, differentiated solely by rope position (explicitly documented in the file as a deferred "refinement") | **genuine self-attention among the block's own positions**, concatenated into the SAME softmax as the cross-attention context (`k = cat(k_ctx, k_noise)`) — this is the actual "diffusion" mechanism (every masked position sees every other), not a documented-later refinement but the core of the algorithm |
| Block seed | ONE shared anchor-embedding, broadcast to every block position | **per-position** "noise" embeddings — each block position starts as its own (initially mostly mask-token) embedding, denoised jointly |
| Draft LM head | reduced-vocab `dflash.lm_head.weight` + `dflash.d2t` remap (both invented tensor names) | **no head of its own at all** — z-lab's checkpoint has neither `lm_head` nor `embed_tokens`; the drafter borrows the TARGET's tied embedding/lm_head at serve time. (RedHatAI's speculators convention *does* carry `draft_vocab_size` — a real reduced head — so the two real conventions differ from each other here too.) |

None of this is a criticism of the original build: it was an honest,
well-tested implementation of the *paper's description*, built explicitly
because no checkpoint existed yet to check it against ("the engine fill-in the
#33 contract slice designed" — a memo-driven build, by design, with its own
receipts calling out exactly that limitation). The gap is now precisely
evidenced rather than assumed.

## 5. What this session built (real, oracle-gated, receipted)

Scope: the config-parsing layer (`go/decode/dflash/dflash.go`, unfenced) and a
**new host-f32 reference forward** for the real z-lab architecture
(`go/decode/dflash/zlab.go`, new file, unfenced — no engine/metal changes).

- **`decode/dflash.Config`**: added `MaskTokenID` (real generation bookkeeping
  the old struct had no field for at all); `ParseConfig` now recognises EITHER
  real convention (`speculators_model_type=="dflash"` OR `architectures`
  contains `"DFlashDraftModel"`), reads `dflash_config.target_layer_ids` as a
  fallback for `aux_hidden_state_layer_ids`, reads `mask_token_id` from either
  nesting, and reads `speculators_config.verifier.name_or_path` (falling back
  to the old `.name` key for compatibility). New tests
  (`TestParseConfig_ZLabNative`, `TestParseConfig_SpeculatorsRealShape`) use
  the **actual byte content** of real `z-lab/Qwen3-4B-DFlash-b16` and
  `RedHatAI/NVIDIA-Nemotron-3-Super-120B-A12B-speculator.dflash` config.json
  files, trimmed to the fields read, cited by source repo in the test doc
  comments.
- **`decode/dflash.ZLabForward`** (+ `ZLabArch`, `ZLabWeights`,
  `ZLabWidenBF16`): a from-scratch, pure-Go, engine-free host-f32
  implementation of `DFlashDraftModel.forward()` — the real 5-layer qwen3-style
  decoder, correctly shaped per §4 (2-norm layers, k_norm on concatenated K,
  intra-block self+cross attention in one softmax, SiLU-gated MLP, `fc` +
  `hidden_norm` fusion). Returns every layer's hidden state (not just the
  final one) so a caller can oracle-gate at depth.
- **Oracle gate, real weights, 3 depths**: `go/decode/dflash/zlab_test.go`
  carries a synthetic-weight Good/Bad/Ugly suite (always runs, no download)
  plus `TestZLabForward_RealCheckpoint`, env-gated on
  `LTHN_DFLASH_ZLAB_CKPT` (the `model/quant/awq` `LEM_AWQ_REFERENCE_DIR`
  pattern — skips cleanly without it). `testdata/zlab_qwen3_4b_oracle.json`
  pins the exact inputs and expected per-depth outputs an **independent numpy
  re-implementation** computed from the real downloaded
  `z-lab/Qwen3-4B-DFlash-b16` weights — and that numpy implementation was
  itself cross-checked against the checkpoint's own `modeling_dflash.py`
  executed through `transformers` (not just against my own reading of its
  source). Result, run against the real checkpoint:

  ```
  after_layer_0: max abs diff 0.079  across 10240 values (hidden std ≈ 120 at this depth)
  after_layer_2: max abs diff 0.100  across 10240 values (hidden std ≈ 135 at this depth)
  final_norm:    max abs diff 0.0023 across 10240 values (hidden std ≈ 2.6  at this depth)
  ```

  Reproduction:

  ```bash
  python -c "from huggingface_hub import snapshot_download; \
    print(snapshot_download('z-lab/Qwen3-4B-DFlash-b16', \
    allow_patterns=['config.json','model.safetensors']))"
  LTHN_DFLASH_ZLAB_CKPT=<printed dir> go test ./decode/dflash/... -run RealCheckpoint -v
  ```

This proves the *architecture reading* in §4 is correct against real trained
weights, independent of the GPU engine. It does **not** by itself change what
serve or the metal engine do — see below.

## 6. What's NOT done, and why

**The GPU engine forward (`engine/metal/assistant_dflash*.go`) is unchanged.**
Correcting it to match §4 is an architecture-level rewrite, not a parameter
tweak: a new draft-layer forward (2-norm, k_norm, intra-block self-attention
concatenated into the SAME softmax as the cross-attention context, SiLU gate)
that cannot reuse `draftLayerIntoScratch`/`draftAttentionIntoScratch`
(gemma4-sandwich-specific, and outside this session's file fence besides), a
decoder-arch registration for the z-lab convention's qwen3 decoder (which
belongs in a `model/qwen3`-family package, not `model/gemma4`, per the
house rule that format/arch ownership stays with the owning architecture
package), and a decision about the two real conventions' *different* vocab
posture (z-lab: borrow the target's tied head, no remap; speculators: a real
reduced `draft_vocab_size` head). Attempting that rewrite in the time
remaining this session, without a second real-target checkpoint to pair
against for a true draft→verify accept-rate, risked exactly the "invented
wiring" the task brief warned against. The host-f32 reference in §5 is the
de-risking step the `model/needle` package's own doc comment describes for
exactly this situation: *"de-risk the … direction before any accelerated
port — a readable oracle whose only claim to correctness is that it reproduces
the reference model's tokens."*

**Punch list for the next pass** (engine-side, `engine/metal/assistant_dflash*.go`
+ a new `model/qwen3`-family registration — outside this session's fence in
spirit even where not in letter):

1. Register a decoder-arch parser for the z-lab qwen3-decoder DFlash
   convention (`model_type:"qwen3"` + `dflash_config` present) — home it beside
   wherever this repo's other qwen3 arch derivation already lives, not in
   `model/gemma4`.
2. Add a qwen3-shaped draft-layer forward to the engine (2 norms, k_norm on
   concatenated K, SiLU MLP) — a sibling to `draftLayerIntoScratch`, not a
   patch to it (that function stays gemma4-correct for the MTP `-assistant`
   lane, which is real and unaffected by any of this).
3. Implement genuine intra-block self-attention (concatenate the block's own
   K/V into the SAME softmax as the injected context, per §4) — this is the
   part `assistant_dflash.go`'s own comment correctly flagged as deferred, and
   it is not optional: it is the actual "diffusion" mechanism, not a
   refinement on top of a working approximation.
4. Fix the numAux/context-length conflation: cross-attention context length is
   the number of target tokens fused in, not the number of fused layers.
5. Decide the vocab posture per convention (borrowed target head for z-lab;
   real reduced head + `d2t` for speculators) rather than assuming one
   universally.
6. Once (1)-(5) land and are oracle-gated the way §5's host forward already
   is, flip `DFlashEngineProbe` — and only then, since arming today's forward
   against a real checkpoint would silently compute wrong numbers.
7. `serving/serve_draft.go`'s doc comments still cite the retired
   `docs/design-dflash.md` (deleted 2026-07-14, `290811cf`) — repoint them at
   this file when that area is next touched (out of this session's fence:
   `serving/` is explicitly off-limits here).

**Not attempted at all**: a live draft-propose→verify round trip against a
paired real target checkpoint (e.g. `Qwen3-4B-Instruct` verifying
`z-lab/Qwen3-4B-DFlash-b16`'s proposals) — that needs the engine-side forward
from the punch list above to exist first; measuring a real accept-rate before
the forward is correct would not be a meaningful number.

## Files touched this session

- `go/decode/dflash/dflash.go` — `Config.MaskTokenID`; `ParseConfig` recognises
  both real conventions; verifier key fix.
- `go/decode/dflash/dflash_test.go` — two new tests against real checkpoint
  config.json byte content (z-lab, RedHatAI).
- `go/decode/dflash/zlab.go` — new: the real z-lab architecture host-f32
  forward.
- `go/decode/dflash/zlab_test.go` — new: synthetic Good/Bad/Ugly suite +
  env-gated real-checkpoint oracle test.
- `go/decode/dflash/testdata/zlab_qwen3_4b_oracle.json` — new: the pinned
  oracle fixture (inputs + expected outputs at 3 depths), reproducible via the
  recipe in §5.

No changes to `engine/metal/`, `model/gemma4/`, `model/mtp/`, `serving/`, or
any other package — the existing MTP `-assistant` lane (gemma4's own drafters)
and everything else in the tree is untouched and unaffected.
