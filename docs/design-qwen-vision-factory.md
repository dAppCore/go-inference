# Design: the qwen35 factory-native vision tower (#59 item 1)

The #50 composed strip deleted `model/composed` — and with it the qwen-family
vision tower (image encode → soft tokens spliced into the text prefill). The
factory route serves the qwen3_5/3_6 family's TEXT stack only; an image turn on
a vision-towered qwen checkpoint hits the engine's clean
`engine.TextModel.Chat: model does not accept image input` refusal
(`go/engine/model.go`). This lane gives the qwen35 family a factory-native
tower. The deleted reference lives at `b1f6c21a^:go/model/composed/{vision.go,
vision_loader.go,vision_token_model.go}` and is the port source — every formula
below is what that code actually computed, not a fresh derivation.

## 1. What the composed tower actually did

**Architecture** (the REAL layout, verified against
mlx-community/Qwen3.6-27B-4bit — 333 `vision_tower.*` tensors, 21 name
patterns; the reconciliation receipt was
`vision_loader_real_test.go`):

- **Patch embed** — a dense `[Hidden, PatchDim]` projection + bias
  (`vision_tower.patch_embed.proj.*`; the dense weight may ship >2-D
  `[Hidden,T,P,P,C]` — row-major flatten is byte-identical). PatchDim =
  InChannels·PatchSize²·TemporalPatchSize (a still image feeds
  TemporalPatchSize identical frames).
- **Learned positions** — `vision_tower.pos_embed.weight`
  `[NumPositions, Hidden]`, a SQUARE table added once after the patch embed;
  a non-native grid resamples it with separable bilinear interpolation in the
  reference convention (`linspace(0, side-1, n)` per axis, ceil corner
  clamped).
- **2-D rotary** — applied to q/k in EVERY block for BOTH layout families,
  ADDITIVE with the learned table (the pre-fix XOR gate is what made the first
  live 27B image turn read 'E' for a giant Q). Pair-angle table
  `visionRotaryTable(HeadDim, theta)`, row drives the first quarter of pair
  angles, column the second, rotate-half doubling.
- **27 pre-norm bidirectional blocks** — LayerNorm WITH BIAS (norm1/norm2,
  weight+bias — not the text stack's RMSNorm), fused `attn.qkv`
  `[3·Hidden, Hidden]` split into equal Q/K/V output-row bands (plain MHA — no
  GQA signal exists in the fused form), `attn.proj` output projection, no
  q/k-norm in the real layout (the loader probes rather than assumes), and a
  plain 2-linear GELU MLP (`mlp.linear_fc1/linear_fc2`,
  `gelu_pytorch_tanh`). 1/√HeadDim attention scale. No KV cache, no mask —
  stateless, once per image.
- **Merger** — LayerNorm(weight+bias) → spatial MergeSize×MergeSize concat
  (raster-order M² block gather) → linear_fc1 → GELU(tanh) → linear_fc2 into
  TEXT hidden width (5120 on the 27B). Soft tokens =
  (gridH/M)·(gridW/M).
- **The GUESSED layout** (separate q/k/v/o + SwiGLU + optional per-head RMS
  q/k-norm + no pos_embed) is the second convention the loader resolves purely
  by which tensor names are present — never a model_type switch. Both layouts
  run the SAME forward; which names resolved sets load-time facts
  (GELU flag, LearnedPositions flag).
- **Geometry is weight-derived**: Hidden/PatchDim/Depth/HeadDim/KVHeads/FF/
  MergeSize come from tensor shapes; only PatchSize (pixel geometry) is
  config-load-bearing; NumHeads is a fallback when no q_norm exists to read
  HeadDim off. `spatial_merge_size`, when present in config, is
  cross-validated against the derived value; merger output width is
  cross-validated against text hidden.

**Preprocessing** (`imageToPatchGrid`): decode PNG/JPEG, TOP-LEFT CROP to the
nearest PatchSize·MergeSize multiple per axis (no resample), patchify to
`[gridH·gridW, PatchDim]` channel-last rows repeated TemporalPatchSize times,
pixels as `v/255` in [0,1]. **No image_mean/std normalisation** — the HF
processor applies (x−0.5)/0.5; the composed lane did not, and it served the
live 27B image turns. The port keeps the reference behaviour (parity target =
the composed tower); revisiting normalisation against the HF processor is a
separate, measured follow-up.

**Splice + prefill** (`vision_token_model.go`): the
`engine.VisionTokenModel` method set — AcceptsImageInput (a live tower probe),
ImagePlaceholderTokenID (config.json top-level `image_token_id`),
ImagePlaceholderBlock (`<|vision_start|>` + `<|image_pad|>`×n +
`<|vision_end|>` — family-constant spellings, ids from config),
ProjectImage (patchify → tower → bf16 feature rows), and
TokenEmbeddingsWithFeatures (embed ids, overwrite placeholder rows with
feature rows, N-th placeholder ← N-th row, count mismatch fails loud). The
engine's neutral `chatMultimodal` (go/engine/vision.go) drives it and
`PrefillTokenEmbeddings` lands the rows.

**Numeric tier**: host f32 weights, f64-accumulated projections
(`matNT`/`matNTCols`), with a device seam (`ProjMatMulInto` → steel f32 GEMM)
engaged for shapes ≥ 2^20 M·K·N. Packed (quant) projections dequantised via
`mlxaffine.DequantizeTensor` per row and dotted in f64.

## 2. The factory shape this port lands in

**Where qwen35 actually serves.** Since #18/#50 the qwen3_5/3_6 hybrids load
through the ONE factory route (`model.Load` → `loadedToQuant`/`loadedToBF16` →
`NewQuantTokenModel`/`NewBF16TokenModel`) and serve as **`*NativeTokenModel`**
with `ArchSession` decode (gated-delta layers bind via
`bindGatedDeltaQuant/BF16`). They do NOT ride
`sessionTextModel`/`composedEngineSession` — that arm serves rwkv7/mamba2
only. The `visionModel()` seam kept in `inference_register.go` therefore needs
**no edit for this port**: the NativeTokenModel lane already carries the whole
multimodal machinery gemma4 uses —
`TokenEmbeddingsWithFeatures` (generic splice over any placeholder id),
`ArchSession.PrefillTokenEmbeddings` (generic embeddings prefill), and the
ChatML dialect (`DetectChatTemplate` picks `<|im_start|>` off the qwen
tokenizer). What is missing is exactly one thing: a loaded tower behind
`AcceptsImageInput`/`ProjectImage`.

**Why not `ArchSpec.Vision` → `vision.Loaded`.** The factory's declarative
vision hook returns `*vision.Loaded` (model/vision), whose Layer shape is the
SigLIP tower's: single-tensor norms (no bias slots), Gate/Up/Down MLP, no
merge-size/temporal geometry, no learned-table-plus-rope semantics. The qwen
tower cannot express itself in it honestly (norm biases and the GELU/merge
facts have no true home), and `model/` outside qwen35 is out of this lane's
fence — extending the neutral payload for qwen is a follow-up decision, not a
contortion to make today. The dotsocr/glmocr packages set the in-tree
precedent for an arch-owned vision weight set.

**The landing:**

- **`go/model/arch/Qwen/qwen35/`** owns the tower payload + assembly (the arch
  owns its tensor-name mapping — house rule):
  - `vision.go` — exported payload types (`VisionTower`, `VisionLinear` (f32 +
    optional bias), `VisionBlock`, `VisionMerger`, `VisionTowerConfig` incl.
    ImageTokenID + the family token spellings + TextHidden), the
    `ImageToPatchGrid` preprocessing, and `InterpolatePosEmbed` — all ported.
  - `vision_loader.go` — `LoadVisionTower(tensors, configJSON)`: the ported
    probe/derive/validate assembly (both layouts, fused-QKV split,
    weight-derived geometry, config cross-validation, `(nil, nil)` for a
    text-only checkpoint). **Every projection lands as host f32**: dense
    tensors widen via `safetensors.DecodeFloatData`; packed tensors (`.scales`
    sibling) dequantise at load via `mlxaffine.DequantizeTensor` — the SAME
    primitive the reference's host quant matvec called per row, so the
    dequantised values (and the f64 dot that follows) are identical to the
    reference's host path. The 27B's tower ships dense bf16 (no vision tensor
    carries scales), so the packed path is a compatibility lane, not the
    receipt lane.
  - `qwen35.go` — Config gains the multimodal wrapper fields it already
    tolerates but did not parse: `image_token_id`, `video_token_id`,
    `vision_start/end_token_id` (polymorphic scalar-or-list `tokenID`,
    ported) and `vision_config` (patch_size/in_channels/num_heads/
    num_key_value_heads/spatial_merge_size/rms_norm_eps/rope_theta). Text
    `Arch()` untouched.
- **`go/engine/metal/`** owns the forward + the serve glue (NEW files):
  - `qwen_vision_encoder.go` — `QwenVisionTowerForward(patches, gridH, gridW,
    tower)`: the ported reference maths (LayerNorm-with-bias, 2-D rope +
    learned-table addition, bidirectional attention, GELU/SwiGLU MLP, merger).
    Projections dispatch to the engine's `MatMulF32NT` steel f32 GEMM above
    the reference's 2^20 work floor and fall back to the ported host
    f64-accumulation `matNT` below it / on device error — the same two-tier
    structure (and the same numeric tiers) the composed lane served with.
    Reuses `geluTanhScalar` (byte-identical formula already in the engine);
    ports the nil-guarded per-head RMS norm (the engine's `rmsNormVec`
    normalises even with nil scale — different contract).
  - `qwen_vision.go` — the glue: `loadQwenVisionTower` (probe arch → parse →
    assemble from `dm.Tensors`) and `projectQwenImage` (patchify → forward →
    `f32ToBf16Slice` rows).
- **Existing-file edits (all additive, each justified in the lane report):**
  - `token_model.go`: one field (`qwenVision *qwen35.VisionTower`) + one
    branch each in `AcceptsImageInput`, `ImagePlaceholderTokenID`,
    `ImagePlaceholderBlock`, `projectImageWithCfg` — the vision entry points.
    The qwen branch ignores the gemma `VisionImageFeatureConfig` (qwen's
    preprocessing is its own crop policy, ported).
  - `load.go`: one call per arm (quant + bf16) at the existing vision seam
    (`tm.vision = lm.Vision; …`) to attach the qwen tower when the factory
    payloads are nil.
  - `inference_register.go`: **zero edits** (see above; the load-routing
    receipt is in the lane report).

Weights stay host-resident f32 for the model's life — the same widen-to-f32
cost the composed lane paid (~1.7 GB on the 27B tower). Device-resident bf16
tower weights are a measured follow-up, not v1.

## 3. Scope of THIS lane (v1 bounds)

- A servable text+image path on the qwen35 family: **single image per turn**
  semantics proven; the neutral splice supports multiple images per prompt
  (N-th placeholder ← N-th feature row) and inherits its behaviour untested
  beyond unit level.
- The same v1 perf bounds gemma4 vision shipped with: **no prompt cache for
  vision turns** (an image turn re-prefills; `PrefillTokenEmbeddings` resets
  retained state), no ICB replay of the tower, tower runs once per image at
  prefill.
- Refusal contract preserved: a qwen checkpoint WITHOUT `vision_tower.*`
  tensors loads text-only, `AcceptsImageInput() == false`, and an image turn
  gets the engine's existing clean refusal — never a crash, never a fake
  answer.
- Receipts: config/weight round-trip on synthetic tensors (both layouts +
  packed), encoder shape/determinism + an independent naive-oracle golden on
  synthetic weights, the real-checkpoint tensor-name reconciliation
  (header-only, behind the local-HF-cache skip), and — the local snapshot
  being present on this box — a live end-to-end image-turn receipt on
  mlx-community/Qwen3.6-27B-4bit plus the live refusal receipt on a text-only
  qwen snapshot.

## 4. What this lane will NOT do

- **No video, no audio** — `VideoPlaceholderTokenID` stays 0 for qwen
  (a video turn hits chatMultimodal's clean "model declares no video
  placeholder tokens"); the qwen omni audio family is out of scope.
- **No multi-tower variants** (deepstack visual indexes ship empty on the
  released checkpoints and are ignored), no Qwen2-VL-era checkpoints beyond
  what the GUESSED layout already resolves, no GGUF vision.
- **No hip** — darwin/arm64 metal lane only; `engine/hip` untouched.
- **No prompt-cache / continuity for image turns**, no soft-token budget
  interface (`VisionBudgetTokenModel` — qwen's grid is image-native, not
  budgeted; a request's `VisionBudget` is ignored by the fallback path
  exactly as the interface contract allows).
- ~~**No HF-processor parity work** (mean/std normalisation, min/max_pixels
  resize policy) — the reference's crop policy is ported verbatim; aligning
  preprocessing with `Qwen2VLImageProcessorFast` is a separate measured
  follow-up.~~ **Done — see §5.** The named follow-up landed (#59's
  normalisation follow-up): normalisation AND the resize policy, not just
  normalisation, because the real checkpoint's declared bounds turned out to
  diverge from the crop policy on ordinary inputs, not just an edge case.
- **No `model/` edits outside qwen35** — the neutral `vision.Loaded`
  extension (folding the qwen tower into the declarative `ArchSpec.Vision`
  hook properly) is named as the natural follow-up once the payload shape is
  settled by a second consumer.

## 5. The normalisation + resize follow-up (#59, addendum)

§4 named this as future work and expected it to be normalisation only. It
turned out to be both, and the "measured" part of "separate, measured
follow-up" mattered: the real checkpoint's `preprocessor_config.json`
(`mlx-community/Qwen3.6-27B-4bit`) declares

```json
{"size": {"shortest_edge": 65536, "longest_edge": 16777216},
 "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
 "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]}
```

— `Qwen2VLImageProcessorFast`'s `smart_resize`: both axes rounded to the
nearest `patch_size·merge_size` (=32) multiple, the resulting pixel COUNT
clamped into `[shortest_edge, longest_edge]` via a scale-and-round-again
step, resampled (not cropped) with an antialiased bicubic filter. Checked
against this port's own 64×64 live-receipt fixture, that bound
(`shortest_edge`=65536=256²) forces a **4×upscale to 256×256** — not a
rare/large-image-only edge case, the common case for anything smaller than
a 256px-per-side photo. Top-left crop and smart_resize therefore disagree on
ordinary inputs, not just extremes, so item 3 of the follow-up brief ("if
resize semantics diverge, implement what the config declares") applied.

**What shipped** (`model/arch/Qwen/qwen35/vision_preprocess.go`,
`vision.go`'s `ImageToPatchGrid`, `engine/metal/qwen_vision.go`):

- `VisionPreprocessConfig` (ImageMean/ImageStd/MinPixels/MaxPixels) +
  `LoadVisionPreprocessConfig(dir)`, reading `preprocessor_config.json` —
  the filename `model/arch/rednote-hilab/dotsocr` and
  `model/arch/zai-org/glmocr` already use for the same Qwen2VL-derived
  processor family (gemma4's OWN convention is the sibling
  `processor_config.json`, a different file — qwen35 follows its closer
  architectural cousins here, not gemma4's filename). HF-standard defaults
  (`transformers.image_utils.OPENAI_CLIP_MEAN/STD`, `size` 56²/28²·1280 —
  `Qwen2VLImageProcessor`'s own class defaults, confirmed against a local
  transformers 5.6.0.dev install) apply when the file is absent — never an
  error, mirroring `LoadGemma4ImageFeatureConfigs`'s missing-file contract.
  A present-but-malformed file fails loud, matching `LoadVisionTower`'s own
  present-but-broken contract for the tower weights. The load seam
  (`loadQwenVisionTower`) traces the fallback via `nativeTraceLog` when no
  file is found — the one gemma-neighbourhood load path that already logs a
  processor-config fallback is `buildAudioExtractor`, not the vision path
  (which is silent); this port took the logged idiom since the brief asked
  for it explicitly.
- `qwenSmartResizeTarget` — a verbatim port of `smart_resize`'s dimension
  arithmetic, using `math.RoundToEven` (NOT `math.Round`) for the
  round-to-nearest-multiple step: Python's builtin `round()` is
  round-half-to-even, and disagrees with Go's round-half-away-from-zero
  `math.Round` at exact `.5` boundaries (confirmed against the reference:
  `smart_resize(2,2,factor=4,...)` rounds `2/4=0.5` DOWN to the even 0, not
  up to 1).
- `resizeBicubicRGB`/`resampleAxis`/`cubicKernel` — a separable two-pass
  antialiased bicubic resample (a=-0.5 kernel, support widened by the
  downsample factor when shrinking), independently written for this
  package. It was NOT ported from `engine/metal`'s gemma4 vision resize
  (`vision_features.go`'s `visionResizeBicubicAA`, out of this lane's file
  fence, read-only precedent) but lands on the same well-known
  Pillow-compatible algorithm that file already uses — unsurprising, since
  both are implementing the same publicly documented antialiasing
  convention torchvision added specifically to match Pillow's C resize.
- Normalisation applies exactly where the reference applies it: after the
  resize (which itself runs in the 0–255 domain and rounds back to an
  integer pixel value, matching the reference's `uint8` resize output —
  confirmed via the oracle), THEN `/255`, THEN `(v-mean[c])/std[c]` per
  channel.
- `ImageToPatchGrid` gained a `VisionPreprocessConfig` parameter. A
  zero-value config (a caller with no policy to declare — e.g. the seam
  dispatch test's tiny synthetic tower) degrades safely to the pre-#59
  behaviour: `MinPixels`/`MaxPixels` <=0 disable the resize bounds (only
  the parameter-free round-to-nearest-multiple step still applies), and a
  zero `ImageStd[c]` is treated as 1 by `normalized()`, making
  "normalisation" a no-op plain `/255` rescale. Production callers always
  route through `LoadVisionPreprocessConfig`, which never returns a
  zero-value config.

**Verification.** No Python dependency ships in this repo, so the oracle
comparisons below are reproducible, not re-checked by `go test`: a local
`transformers`+`torch`+`torchvision` install (`AutoImageProcessor` resolves
the checkpoint's own `Qwen2VLImageProcessor`) fed the exact 64×64
live-receipt fixture (and a synthetic 4001×5003 non-aligned image for the
downscale branch) through the real pipeline.
`qwenSmartResizeTarget`/`smart_resize` dimension arithmetic matched exactly
on every case tried (upscale, downscale, no-op, the round-half-to-even
boundary, the >200:1 aspect-ratio refusal and its exact-200 boundary).
`resizeBicubicRGB` matched the real resize to max abs diff 1/255 (uint8
domain) — 0.02%–0.09% of pixels differing by that single LSB, the rest
exact — on both the upscale and downscale cases: quantisation-noise-level
agreement, not a byte-identical port. The committed Go tests
(`vision_preprocess_test.go`, `vision_test.go`) pin the dimension
arithmetic and a handful of oracle-cross-checked pixel values (both
uniform-region "no blending" points and genuinely-blended edge-transition
points) as Go-native receipts that don't need the Python environment to
re-run.

**Live effect.** `TestQwenVisionImageTurn_RealCheckpoint`
(`mlx-community/Qwen3.6-27B-4bit`, the 64×64 navy-background/yellow-square
fixture, "what shape is on the image?"): pre-#59 (top-left crop, no
normalisation) answered **"rectangle"**; post-#59 (smart_resize to 256×256
+ 0.5/0.5/0.5 normalisation) answers **"square"** — the geometrically
precise answer. Soft-token count for this fixture rose from 4 to 64
((16/2)·(16/2) after the forced 256×256 upscale) — still trivial prefill
cost for a 27B model; the test stayed "minutes-cheap" (~10s wall,
including model load).

## 6. The device-resident bf16 tower (#59's vision follow-up)

§2 named this as the natural next step and bounded it in advance: "Device-resident
bf16 tower weights are a measured follow-up, not v1." This section is that
follow-up — upload the tower's weights once, encode on-GPU, and let the v1 host
f32 copy go.

### 6.1 The precedent, and why qwen didn't start there

gemma4's SigLIP tower (`engine/metal/vision.go`) is already device-resident, and
it got there almost for free: its weights are bf16 **views straight into the
mapped checkpoint** (`gemma4.AssembleVision`'s `visionWeight` returns
`t.Data` — a `[]byte` alias of the mmap, never widened), so `residentBytes`
(the #60 owned-weight pin-cache, `attention.go`) no-copy-binds them to a Metal
buffer on first use and the "upload" costs nothing beyond that first bind — there
is no separate f32 copy to free because one was never made. Every projection
rides `MatRowsBF16`/`MatMulBF16NT` (the fused bf16 steel GEMM), attention rides
`VisionSDPA` (decomposed score/softmax/weighted-sum, the two matmuls on-device,
GQA-aware via a `nHeads/nKVHeads` ratio and a caller-supplied scale — both already
parameters, not gemma constants), and the only host-side work is the tiny
per-head QK-norm + rope and the elementwise glue (bias-add, GELU-gate-mul) —
too small for a command-buffer round trip to pay for itself.

qwen's v1 tower could not start there: `qwen35.LoadVisionTower` **widens or
dequantises every weight into an OWNED f32 heap copy** at load (dense tensors via
`safetensors.DecodeFloatData`, packed ones via `mlxaffine.DequantizeTensor`) —
deliberately, to match the retired composed engine's own host f32/f64 numeric
tier byte-for-byte (§1's "Numeric tier" note). That copy is real, additional
memory (not an mmap alias), it is what made the v1 tower's ~1.7 GB cost
concrete rather than nominal, and it is exactly what this follow-up now
downcasts to bf16 and releases.

### 6.2 What moves device, what stays host

Everything GEMM/LayerNorm/attention/GELU-shaped moves onto the engine's
existing bf16 kernels — the same reuse gemma's tower already proved out, just
applied to qwen's LayerNorm-with-bias / fused-QKV / merger shape instead of
SigLIP's RMSNorm / pooling shape:

| stage | kernel reused | new code |
|---|---|---|
| patch embed | `MatRowsBF16` | bf16 patch upcast (`f32ToBf16Slice`, already used for output features) |
| per-block LayerNorm (×2) | `LayerNormBF16` | a bias-optional wrapper (§6.4) |
| Q/K/V/O projections | `MatRowsBF16` | host bias-add (no equal-shape broadcast-add kernel exists; gemma's own tower computes its linear biases the same way) |
| attention core | `VisionSDPA` (gemma's kernel, unmodified — GQA ratio + scale are call parameters) | token-major↔head-major bf16 reshape (below) |
| GELU MLP (the REAL layout — what the live 27B ships) | `MatRowsBF16` + `GeluBF16` | none |
| merger LayerNorm + both linears + GELU | `LayerNormBF16`, `MatRowsBF16`, `GeluBF16` | none |

Stays host, deliberately, matching where gemma's OWN tower already keeps this
same slice of work:

- **Per-head QK-norm + 2-D rope.** HeadDim-sized (tens to low hundreds of
  values), applied once per token per head — the existing scalar helpers
  (`qwenVisionRMSNormHead`, `qwenVisionRope2D`, `qwenVisionRotaryTable`) are
  reused byte-for-byte, wrapped only in new bf16↔f32 data-movement glue
  (`qwenVisionSplitHeadsRoped`) — no new host maths, exactly the edge gemma's
  own `qkNormRoPEHeadMajor` sits at.
- **The merger's spatial-block gather.** A pure row permutation (M×M blocks →
  concatenated rows), zero arithmetic. There is no device gather primitive of
  this shape anywhere in this engine, and gemma4's tower has no merger to have
  set a precedent for one — `qwenVisionMergeSpatialBF16` is
  `qwenVisionMergeSpatial`'s existing f32 gather re-expressed as byte-wide
  copies, not a new algorithm.
- **The GUESSED layout's SwiGLU gate** (`silu(gate)·up`). No device kernel
  computes this activation — `GeluGateMul`/`GeluGateMulBF16` are GELU-gated
  (gemma's nonlinearity, wrong shape), and the decode path's `MoEExpertsQuantSiLU`
  is a MoE-block primitive with no vision-tower-shaped sibling. A new metallib
  kernel was **not** written for it: it is O(L·FF) elementwise (the same tier as
  the bias-add/gather glue above, not a compute-bound GEMM) and it serves a
  compatibility lane no live receipt exercises — the real 27B checkpoint ships
  the REAL layout's plain GELU MLP, which is fully device end-to-end.
  `qwenVisionSiluGateMulBF16` stays a host bf16↔f64↔bf16 loop, reusing
  `qwenVisionSilu`'s existing formula unchanged.

This is a truthful, not a forced, partial: every stage that has a genuine
device twin uses it; the three that don't are all either too small to be worth
a kernel, structurally gather-only, or unexercised by any live checkpoint.

### 6.3 The seam and the lifetime

`qwen35.VisionTower` gained one field: `DeviceSeam any` — an opaque hook
`qwen35` never reads (AX-8: this package never imports engine).
`engine/metal/qwen_vision_device.go` stores its bf16 mirror
(`*qwenVisionDeviceTower`) there once built. This — not a package-level cache
keyed by tower pointer — is the load-bearing design choice: the mirror's
lifetime becomes **exactly** the tower's, so a model unload frees both
together with no separate eviction hook to wire up or forget. (An earlier
draft of this follow-up used a `sync.Map` cache; it was dropped specifically
because it would have kept every unloaded model's ~0.9 GB bf16 tower alive
for the rest of the process — the `DeviceSeam` field has no such failure
mode.)

The individual `residentBytes` binds each bf16 buffer picks up on first kernel
dispatch are a SEPARATE, narrower lifetime question this follow-up does not
change the shape of: qwen's vision tower has never been registered in
`safetensors.DirMapping.OwnedRanges()` (unlike the b1→b2 repack / packExperts
class `docs/design-owned-weight-binding.md` describes), so those individual
buffer binds are process-lifetime-cached, same tier as this engine's widened
F16→BF16 tensors. This was already true of the v1 tower's f32 weights for any
projection shape that crossed the old 2^20 device-GEMM floor; it is not a new
category, only a wider one now that every stage (including attention) routes
through a device kernel. Wiring these into `OwnedRanges()` for true
session-scoped eviction is the natural next follow-up, not silently accepted
here — it needs a `model.Load`-sequencing change (the tower loads after
`AdoptOwnedTensors()`'s sweep already ran), out of this lane's file fence.

### 6.4 Two rough edges the port hit

- **`LayerNormBF16` requires a real bias buffer**, even for a "no bias" norm —
  its own convention (gemma4's audio subsampler passes an explicit zero
  vector) differs from `qwen35.VisionBlock`/`VisionMerger`'s convention (a nil
  `NormB` means no bias, tolerated by the host `qwenVisionLayerNorm`). The
  GUESSED layout's synthetic test tower has exactly this shape (no merger/block
  bias) and caught it immediately. `qwenVisionLayerNormBF16` restores qwen35's
  convention by substituting a fresh zero bf16 buffer — exact, not an
  approximation, since bf16 zero is the same all-zero byte pattern as f32 zero.
- **`VisionSDPA` wants head-major** `[heads, L, headDim]`; every projection
  naturally produces token-major `[L, heads·headDim]`. `qwenVisionSplitHeadsBF16`
  / `qwenVisionSplitHeadsRoped` / `qwenVisionMergeHeadsBF16` do the reshape
  (the norm+rope variant folds the per-head host maths into the same pass, so
  there is no separate transpose step) — proven as exact inverses in isolation
  (`TestQwenVisionSplitMergeHeadsBF16_RoundTrip_Good`) rather than trusted to
  the end-to-end parity number alone.

### 6.5 The precision change, measured

This is a precision change, not a free lunch: the host tier keeps f32 weights
with f64-accumulated projections; the device tier is bf16 (≈3 decimal digits,
round-to-nearest-even, MLX's own convention — `f32ToBF16`) **throughout**,
weights and activations alike. `TestQwenVisionTowerForwardDevice_HostParity_Good`
runs the same synthetic weights and patches through both tiers and pins a
max-abs-diff band, the same method `TestQwenVisionTowerForward_OracleParity_Good`
already uses against the independent f64 oracle (that test pins `1e-5` —
f32-vs-f32 rounding noise; this one needs a much looser band, bf16's real cost).

A depth sweep on the REAL-layout synthetic tower (2/4/8/16/27 blocks; logged
during development, not a committed test) showed the divergence does **not**
grow unboundedly with depth — each block's LayerNorm re-centres the
activations it hands to the next one:

| blocks | max abs diff | max reference magnitude |
|---|---|---|
| 2 | 0.00099 | 0.44 |
| 4 | 0.00139 | 0.35 |
| 8 | 0.00173 | 0.30 |
| 16 | 0.00116 | 0.27 |
| 27 (the live checkpoint's actual depth) | 0.00265 | 0.30 |

The committed test therefore runs the REAL layout at 27 blocks — the live
checkpoint's own depth, not a token-sized stand-in — and pins **0.01**, ~4×
the deepest measured case, tight enough to still catch a structural porting
slip (a transposed weight or a wrong fused band diverges by order-1, not by
bf16 noise) by orders of magnitude. The GUESSED layout (1 block, its existing
synthetic shape) measured 0.00023, comfortably inside the same band.

### 6.6 The memory story, measured on the real 27B tower

`LTHN_VISION_DIAG=1` on `TestQwenVisionImageTurn_RealCheckpoint`
(`mlx-community/Qwen3.6-27B-4bit`) logs the built mirror's exact byte count:

```
qwen-vision: device tower resident, bf16 bytes=921460192 (~879 MiB); host f32 copy freed (~1758 MiB)
```

879 MiB resident, ≈1758 MiB (≈1.72 GB) freed — matching §2's original "~1.7 GB"
estimate almost exactly, and confirming the bf16 mirror really is half: same
element counts, 2 bytes/element instead of 4. `TestFreeQwenVisionHostWeights_Good`
pins the release itself (every large f32 slice is nil after
`freeQwenVisionHostWeights`, on both tensor layouts) as a Go-native receipt
that doesn't need the real checkpoint to re-run; the number above is the
measured confirmation on the actual 27B tower.

### 6.7 The wall-time story, measured on the real 27B tower

Same test, same fixture (64×64 → smart-resize 256×256 → grid 16×16, 64 soft
tokens), timed with `LTHN_VISION_DIAG=1` around `projectQwenImage`'s tower-forward
call only (excludes model load and text decode):

| path | encode wall | reply |
|---|---|---|
| host (`LTHN_QWEN_VISION_DEVICE=0`) | 4.69 s | `"square"` |
| device (default) | 0.72–0.73 s (two runs) | `"Square"` |

≈6.4× faster. The dominant cost this removes is the host attention core
(`qwenVisionAttentionForward`'s triple-nested f64 loop, O(L²·HeadDim·H) with
**zero** device dispatch of its own — only the projections either side of it
crossed the old 2^20 GEMM floor) against `VisionSDPA`'s device-matmul
decomposition; the effect grows with L² (patch count squared), so it will be
more pronounced still on larger images than this 64-soft-token fixture.

Both replies are correct ("square" / "Square" — a capitalisation difference
from the downstream text decode, not a content error); they are not expected
to be byte-identical given §6.5 — the image EMBEDDINGS differ by the pinned
band, so the greedy token stream downstream of them is free to diverge in
inconsequential ways while remaining semantically right. This is the
correctness receipt for the device path specifically: pre-#59-normalisation
this same fixture answered "rectangle" (§5); every measurement in this section
answers "square"/"Square" — the device tower did not regress the normalisation
follow-up's fix.

### 6.8 The kill-switch

`LTHN_QWEN_VISION_DEVICE=0` restores the v1 host-only tower exactly (f32
weights populated and never freed, `DeviceSeam` left nil, `projectQwenImage`
falls through to `QwenVisionTowerForward`) — the same "one lever per commit"
env kill-switch shape this engine's other wall-clock-adaptive features use
(`LTHN_MTP_REENGAGE`, `LTHN_MTP_DRAFTLEN`). It is a real safety valve, not
just a test convenience: both `TestLoadQwenVisionTower_DeviceResident_Good`
and `TestLoadQwenVisionTower_DeviceDisabled_Good` exercise it end-to-end
through the real load seam.
