<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# DFlash speculative decoding — design memo (oMLX feature-gap #33)

Status: research + honest slice. This memo establishes what **DFlash** is from
primary sources, maps it onto the speculative machinery go-inference already
ships, records the honest engine gap, and scopes the plain-Go slice that lands
now. The measurement receipts (accept-rate / latency on real models) belong to
the orchestrator's measuring session, not here.

## What DFlash is, in one line

A speculative-decoding **method** whose drafter is a small **block-diffusion**
language model: it proposes a whole block of *K* tokens in a single parallel
forward pass (no autoregression), conditioned on fused hidden states drawn from
several layers of the target ("verifier") model, and the target verifies the
block with the ordinary greedy/rejection rule — so acceleration is **lossless**.

## Primary sources

- **Paper:** *DFlash: Block Diffusion for Flash Speculative Decoding*, arXiv
  [2602.06036](https://arxiv.org/abs/2602.06036) (ICML 2026 poster; Z-Lab).
- **Reference impl / format:** vLLM **Speculators** v0.5.0 added DFlash support
  ([Red Hat, 2026-06-04](https://developers.redhat.com/articles/2026/06/04/speculators-v050-dflash-support-and-online-training);
  [vLLM blog](https://vllm.ai/blog/2026-05-28-speculators-v050)); parsing landed
  in vLLM via [PR #38300](https://github.com/vllm-project/vllm/pull/38300) and
  llama.cpp via [PR #22105](https://github.com/ggml-org/llama.cpp/pull/22105).
- **Engineering write-up:** [LMSYS, 2026-06-15](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)
  (SGLang integration; block-size / drafter-latency trade-offs).
- **Checkpoints:** [z-lab](https://huggingface.co/z-lab) (Qwen3, Qwen3.5,
  LLaMA), RedHatAI `*-speculator.dflash` (e.g. DeepSeek-V4-Flash).

## What a user actually downloads

A DFlash drafter is a **draft-model checkpoint**, not merely a method flag. Its
`config.json` (verified against RedHatAI/DeepSeek-V4-Flash-speculator.dflash)
carries the Speculators contract:

```jsonc
{
  "speculators_model_type": "dflash",          // <- the identifying marker
  "speculators_version": "0.5.0.dev163",
  "speculators_config": {
    "algorithm": "dflash",
    "default_proposal_method": "greedy",
    "speculative_tokens": 7,
    "verifier_accept_k": 1,
    "accept_tolerance": 0.0,
    "verifier": { "name": "deepseek-ai/DeepSeek-V4-Flash" }  // the target
  },
  "block_size": 8,                              // γ — the diffusion block
  "max_anchors": 3072,
  "aux_hidden_state_layer_ids": [3, 13, 23, 32, 42],  // fused verifier layers
  "draft_vocab_size": 32000,                    // draft vocab + d2t/t2d maps
  "transformer_layer_config": { "...": "the draft's own decoder arch" },
  "torch_dtype": "bfloat16"
}
```

The weights add the drafter's decoder stack, a projection that fuses the
verifier hidden states, and `d2t`/`t2d` tables mapping the reduced draft vocab
to/from the target vocab. Enabling it (SGLang) is
`--speculative-algorithm DFLASH --speculative-draft-model-path <hf>
--speculative-dflash-block-size 8`.

## How it works (from the paper)

1. **Block-parallel drafting.** The drafter is a block-diffusion model (Arriola
   et al. 2025, adapted). Given an *anchor* — a clean token from the last
   verified step — it masks the next `block_size − 1` positions and **decodes
   them all in parallel in a single forward pass** (γ ≈ 8–16). This is the
   headline difference from EAGLE/Medusa/MTP, which draft autoregressively.
2. **Target conditioning.** Hidden states from a fixed set of verifier layers
   (`aux_hidden_state_layer_ids`, uniformly sampled shallow→deep) are
   concatenated, passed through a lightweight projection into a "fused target
   context feature", and **injected as the Key/Value projections of every draft
   layer**, persisting across drafting iterations.
3. **Verification.** The target runs one forward over the proposed block and
   accepts the longest prefix matching its own greedy argmax (default proposal
   "greedy", `verifier_accept_k = 1`, `accept_tolerance = 0.0`); the first
   divergence commits the target's correction token, the rest is discarded, KV
   rolls back. **Lossless** — the emitted sequence equals plain decode.
4. **Training.** Target frozen; draft stack + projection trained (shared frozen
   embedding + LM head), exponentially-decaying per-position loss weighting.

## The mapping — DFlash need → what we already have

| DFlash need | Our machinery | Status |
|---|---|---|
| A drafter proposes, the target verifies and accepts a prefix | `decode/ngram` (model-free proposer) + the engine verify loop | **Exists** (the seam) |
| Adaptive number of speculative tokens | `decode/specctl` (accept-rate EMA → draft length) | **Exists** |
| Extensible speculative-method identity | `model.MTPMethod` enum ("EAGLE… each earn their own constant + a decode branch") | **Exists** (extension point) |
| A separate draft model conditioned on target hidden state, sharing target KV | metal `AssistantModel.DraftAttention/DraftLayer(layerIdx, hidden, targetKV)` — the MTP `draft-model` method | **Exists but single-hidden + autoregressive** |
| Reactive drafter detection by checkpoint shape / config | `serving.ResolveServeDraft` / `DetectGemma4DraftPath` (path-shape ladder) | **Exists** (MTP-shaped) |
| Block-diffusion primitives: canvas mask, denoising steps, KV rollback, confidence/entropy thresholds | hip `model.DiffusionSamplerRoute` (`diffusion_gemma`, block-diffusion generation) | **Declared, kernel not linked** (`KernelStatusNotLinked`, metadata-only) |
| **Fused MULTI-layer verifier hidden extraction → per-draft-layer KV injection** | — | **Not built** |
| **Block-parallel (diffusion) draft forward** (K tokens/pass) | — | **Not built** (MTP path is autoregressive) |
| Reduced draft vocab with `d2t`/`t2d` maps | — | **Not built** |

## Headline

DFlash's **drafter/verify _contract_** is compatible with what we ship: it is a
separate-draft-model method (the family the MTP `AssistantModel` already
serves), its acceptance is the ordinary greedy prefix-accept our verify loop
already implements, and `MTPMethod` was explicitly built to grow a new method +
branch. What DFlash needs that the **metal engine lacks** is three engine-side
shapes: (a) fused multi-layer verifier-hidden extraction injected as KV into
every draft layer — the metal MTP path conditions on a *single* final hidden and
runs *autoregressively*; (b) a block-parallel diffusion draft forward; (c) the
reduced-vocab `d2t`/`t2d` mapping. The hip `diffusion_gemma` route shows the
house already models block-diffusion *metadata* (canvas mask, denoising steps,
KV rollback) with the kernel honestly **not linked** — the same posture DFlash
takes here.

## The slice that lands now

Model-free, plain-Go, provably lossless — sitting beside `ngram` and `specctl`,
exercised without a GPU exactly as they are:

1. **`decode/dflash`** — the DFlash drafter-side contract:
   - `Config` + `ParseConfig([]byte)` — recognises a real DFlash checkpoint by
     `speculators_model_type == "dflash"` and reads `block_size`,
     `aux_hidden_state_layer_ids`, and the verifier target (config only, never
     weights — the `isGemma4FamilyConfig` posture).
   - `BlockProposer` — the seam the engine's diffusion draft forward will
     implement (propose up to `BlockSize` candidates in one shot); a model-free
     `LookupProposer` (ngram-backed block lookup) stands in for the real
     diffusion drafter so the contract is exercisable, and proves a non-zero
     accept-rate on structured text.
   - `AcceptBlock` / `Generate` — the **greedy block-verify driver**: every
     committed token is the target's own next token, so the output is
     byte-identical to plain greedy autoregression **whatever the drafter
     proposes**. This is the executable specification of DFlash's losslessness —
     the invariant the metal `decode_verify_icb` must honour — proven by a
     20 000-case adversarial fuzz (`spec-on == spec-off`).
2. **`model.MTPDFlash`** — the registered method identity for the enum.
3. **`serving`** — `DetectDFlashDraft` recognises a DFlash checkpoint and
   `DFlashDraftNotice` reports the truth: drafter detected, the engine's
   block-diffusion draft forward is not linked, serving plain autoregressive
   (the `diffusion_gemma` honesty posture, not a faked lane).

## The engine forward — what #60 landed

The block-parallel draft forward is now built on engine/metal, against the seams
above, no cgo (`go/engine/metal/assistant_dflash.go`). It is a SIBLING of the MTP
draft step, not a modification of it — it reuses the parity-gated draft-layer
machinery (`draftLayerIntoScratch`, the SDPA inside `draftAttentionIntoScratch`)
verbatim and adds only the DFlash-specific maths:

- **KV injection (1b).** Each verifier hidden projects through every draft layer's
  own `k_proj`/`v_proj` into that layer's injected `AssistantTargetKV` (numAux rows
  of target-context memory) — the exact seam the MTP `draftLayer` already consumes.
  `DFlashDrafter.injectedKV` (`assistant_dflash.go`).
- **Block-parallel readout (1c).** The block's positions decode in one sweep against
  the SAME shared injected context, differentiated by rope position (anchorPos+j) —
  no position depends on another's PREDICTION, so it is genuinely non-autoregressive.
  `DFlashDrafter.ProposeBlock`.
- **Reduced-vocab `d2t` (1d).** `dflash.lm_head` → argmax → `dflash.d2t` maps the
  drafted id into the target vocab. `DFlashDrafter.headArgmax` / `loadDFlashD2T`.
- **Loader.** The drafter loads through the ordinary reactive pack loader
  (`LoadDFlashDrafter` → `LoadAssistantDir` → the registered `gemma4_dflash_assistant`
  spec, `go/model/gemma4/assistant_dflash.go`); no hand-rolled loader.
- **Verify bridge.** Wrapped behind `decode/dflash.BlockProposer`
  (`assistant_dflash_proposer.go`) and driven through `dflash.Generate` (the
  fuzz-proven verify driver) — output stays byte-identical to plain decode.

Receipts (`go/engine/metal/assistant_dflash_test.go`, no public checkpoint exists so
these ARE the done-bar, not a real-model accept rate): the metal forward's proposals
match a bf16 host reference of the same maths (`TestDFlashProposeBlockParity`); the
proposer fires through the verify driver and the emitted sequence equals plain decode
(`TestDFlashProposeBlockLosslessEngagement`); a pack on disk loads and proposes
(`TestLoadDFlashDrafter`).

## The evidenced gap (deliberately not forced)

Two steps could not be expressed cleanly against the current engine seams and are
NOT faked — the block forward takes the fused verifier hiddens as INPUT, so parity +
losslessness are provable now while these remain open:

- **A cheap, non-corrupting live-session aux tap (1a).** The extraction SEAM exists
  and is demonstrated (`ExtractAuxHiddens` in `assistant_dflash_proposer.go`, over the
  real per-layer capture `ForwardCaptureHiddens` / `captureLayerHiddens`), but that
  capture RESETS the session to pos 0 and re-runs the whole sequence, OVERWRITING the
  KV cache. On a live incrementally-decoding serving session that would corrupt the
  target cache and BREAK losslessness — not merely slow it. A boundary tap that
  captures the aux-layer hiddens during the ordinary decode step (the way
  `retainedHidden` already captures the final one, `decode_forward_arch.go:1592`), or
  a separate resident tap session, is the seam DFlash still needs before the live lane
  can extract cheaply and correctly.
- **The live HTTP lane (`speculativeModel`).** `speculative_model.go` always runs the
  autoregressive MTP loop (`GenerateFromSessionEach`); `pair.Method()` has no consumer
  yet. Routing `MTPDFlash` to a DFlash generate loop (proposer + verify) waits on the
  tap above. Until then `serving.DFlashEngineProbe` stays false and serve declines
  honestly (`serve_draft.go`) — the arming SEAM is wired and tested (toggle the probe),
  ready to flip in one place.
- **Joint intra-block self-attention.** The shipped readout treats block positions as
  independent given the shared injected context (differentiated by rope). Positions
  attending each other's hidden (true block-diffusion denoise) is the refinement; the
  losslessness invariant holds either way, so this only affects accept-rate.

The contract fixed the interface so the engine work here was a fill-in, not a redesign;
the losslessness fuzz remains the acceptance gate the whole lane passes.
