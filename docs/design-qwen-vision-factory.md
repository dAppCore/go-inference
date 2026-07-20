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
