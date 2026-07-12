# Design memo: quantisation-format landscape for go-inference's next tier

Research only — no code changes in this memo's production. Scope: where `lem quant`'s
output-format decision should go next, grounded in the actual repo + actual hardware,
not the format landscape in the abstract.

## 1. Repo reality (what go-inference has today)

- **`go/model/quant/mlxaffine`** — the cross-engine NATIVE quant format. Group-affine
  INT2/4/8, packed uint32 LSB-first, bf16 scale+bias per group, byte-verified against
  `mlx_lm.convert`'s own output (`oracle_test.go`). Both `engine/metal` (Apple) and
  `engine/hip` consume it **directly on-device** — confirmed via the
  `rocm_mlx_q4_projection*` kernel family in `engine/hip/kernels/rocm_kernels.hip`
  (packed-row q4/q6 variants at group16/32/64, batched, greedy, triple-projection,
  GELU-tanh-fused). hip's own capability labels (`gemma4_mlx_affine_decode`,
  `quant_family = "mlx_affine"`, `production_quant_policy = "gemma4_mlx_affine"`) mark
  this as the production policy on *both* backends — it is not a "Mac format", it's
  the shared native runtime format.
- **`go/model/gguf`** — the full k-quant suite (Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K, Q4_0/
  Q5_0/Q8_0), byte-pinned against ggml. It's genuinely bidirectional: a WRITER
  (`quantize_writer.go`, `gguf.QuantizeModelPack`) and a READER that dequants on load
  (`tensors.go`: `ggufDequantizeQ8_0ToF16`, `ggufDequantizeQ4_0ToF16`). q4_k_m is
  llama-validated for gemma4. `lem quant -gguf <format>` is the CLI escape hatch from
  the native-affine default (`go/cmd/lem/quant.go`).
- **`go/model/quant/jang`** — MiniMax M2's JANGTQ/MXTQ profile: per-role (attention /
  shared-expert / routed-expert / embed / lm_head) mixed bit-width, 1/2/4/8-bit packed,
  LSB0, affine encoding. Reader + reference-dequant only, no writer — mirrors
  quantfmt's read-only posture, just for a packed-tensor format instead of a config.
- **`go/model/quant/autoround`** — a real SignRound-style rounding engine: symmetric /
  asymmetric affine int quantise (`QuantizeWeights`), gradient-directed rounding
  (`signRoundAdjust`) when a caller supplies gradients (`calibration.go`: "Calibration
  planning is native metadata only; model-gradient capture is supplied by the caller
  before SignRound quantization"), three built-in profiles (W4A16/W2A16 at
  auto-round / auto-round-best / auto-round-light), plus a native safetensors pack
  writer and an optional GGUF Q4_K_M export path.
  - **Finding:** `SchemeMXFP4` / `SchemeNVFP4` exist as `ResolveScheme` entries
    (bits=4, group=32/16) but are **metadata stubs only** — the numeric core
    (`quantParams` / `quantizeOne`) always does plain affine integer rounding
    regardless of scheme. Calling `QuantizeWeights` with `SchemeMXFP4` today silently
    produces an INT4 grouped-affine tensor, **not** a true E2M1-element /
    E8M0-shared-exponent MX tensor. No MX bit-packing or float-element quantiser
    exists anywhere in the repo.
- **`go/model/quant/quantfmt`** — a pure-Go READER for HF `quantization_config`
  (GPTQ, AWQ, compressed-tensors, fp8, bitsandbytes) — normalises method/bits/group/
  scheme for logging and loader dispatch. **There is no writer for any of these four
  formats anywhere in the repo** (`grep -rln 'gptq|awq|bitsandbytes' go/model`
  outside `quantfmt.go` returns nothing but golden test fixtures).
- **HIP quant kernel entry points** (`engine/hip/kernels/rocm_kernels.hip`, grep on
  `__global__`): `rocm_mlx_q4_projection*` (affine, incl. q6 group16/32/64 variants),
  `rocm_gguf_q4_0_*` / `rocm_gguf_q4_k_*` / `rocm_gguf_q8_0_*` (the GGUF-affine
  adapter, incl. MoE selected-expert gate/up/down). **No FP8 or MX-format kernel
  exists in HIP today** — the entire HIP quant surface is INT4/INT6/INT8 affine.

## 2. Ground truth — the hardware (homelab box, gfx1101)

`ssh homelab` (findings, not assumptions):

- `rocminfo`: GPU is **gfx1101** (RDNA3, Radeon RX 7800 XT). ROCm **7.2.0** installed.
  hipBLASLt, hipBLAS, MIGraphX present. No `amd-quark` / `quark` Python package
  installed.
- `strings libhipblaslt.so.1` shows Tensile kernel libraries for `gfx1101` — but the
  datacenter-tier libraries (`gfx940H/941H/942H/950H` — CDNA3/CDNA3.5/CDNA4) are a
  visibly separate set from the RDNA tier (`gfx1100/1101/1102/1103/1150/1151/1200/
  1201`). Consistent with the precision table below; I could **not** confirm from
  `strings` alone which dtype kernels the "H" libraries carry specifically — flagged
  as an open question, not asserted as fact.

**ROCm's own precision-support matrix** (rocm.docs.amd.com/precision-support, via
WebFetch) is unambiguous on **matrix-core** acceleration (not just "the C++ type
exists in HIP"):

| Format | CDNA1–2 | CDNA3 (MI300) | CDNA4 (MI350/355) | RDNA2 | RDNA3 (**ours**, gfx1101) | RDNA4 |
|---|---|---|---|---|---|---|
| float8 (E4M3 / E5M2) | no | **yes** | **yes** | no | **no** | **yes** |
| float6 (E3M2 / E2M3) | no | no | **yes** | no | no | no |
| float4 (E2M1) | no | no | **yes** | no | no | no |

**Our exact box (gfx1101) has zero native matrix-core acceleration for fp8, fp6, or
fp4.** Not even MI300-class (CDNA3) accelerates fp4/fp6 — that's CDNA4 (MI350/355)
exclusively. fp8 matrix-core acceleration is CDNA3+/RDNA4 only; RDNA3 (our box, and
the 7900XT/XTX) is the architecture ROCm's own docs explicitly mark unsupported.

The **OCP Microscaling (MX) v1.0 spec** (queried via a secondary source — the primary
PDF on opencompute.org 403'd): a block of *k*=32 elements sharing one E8M0 (8-bit,
exponent-only, powers of 2 from 2⁻¹²⁷ to 2¹²⁷) scale. MXFP8 = E5M2 or E4M3 elements;
MXFP6 = E3M2 or E2M3; MXFP4 = E2M1. This is a genuinely **different tensor layout**
from our affine format (per-group *float* scale + *float* bias vs. per-32-block
*power-of-two-only* shared exponent, no bias/zero-point term at all) — an MX writer
or reader would be new code, not a repack of `mlxaffine`.

**AMD Quark** (quark.docs.amd.com) is AMD's own calibrated quantiser: PyTorch/ONNX
front end, fp8/MX/int4/int8 targets, SmoothQuant/AWQ/GPTQ algorithms, exports to
ONNX/safetensors/GGUF. Not installed on the homelab box. Its whole reason to exist is
feeding the CDNA3+/RDNA4 fp8 and MI350 MX matrix cores above — i.e. it's the tool for
hardware we don't have, not something we'd be reimplementing for gfx1101 gain.

**Genuinely useful finding, found on the homelab box specifically:**
`bitsandbytes`'s official ROCm (preview) wheels
(huggingface.co/docs/bitsandbytes/installation) target **`gfx1101` by name**, across
ROCm 6.2.4 through 7.2.4 — our exact GPU is a first-class NF4/FP4 target *today*, not
a maybe. bitsandbytes: "all features are supported for both consumer RDNA and Data
Center CDNA products" (still preview-labelled).

## 3. NVIDIA / ZLUDA seam (research question 2)

`go/engine/hip/kernels/README.md`'s `ZLUDA_CUDA_TESTS` target pins
`ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85` — the exact UUID `rocminfo` reports for
our gfx1101 card. **This is not real NVIDIA hardware.** ZLUDA here is a CUDA-source-
portability compile/run check on the *same AMD box*: does our HIP kernel source,
taken through the CUDA toolchain path, still execute correctly under ZLUDA
translation on gfx1101. It answers "is our kernel source CUDA-shape-portable", not
"do we get NVIDIA tensor cores."

If go-inference ever runs on real NVIDIA silicon (not via ZLUDA), fp8 tensor cores
are Hopper/Ada+ (H100, RTX 40-series and up — per NVIDIA's own FP8 blog), and fp4/
fp6/MXFP8 are Blackwell-only. **fp8/fp4 do not change the portable-kernel story via
the ZLUDA seam today** — there's no real Hopper or Blackwell hardware in the loop,
and translating a hardware-matrix-core CUDA intrinsic onto silicon with no matching
matrix unit (our RDNA3 box) isn't something to bank a design on.

## 4. HF-ecosystem exports (research question 3)

| Format | Algorithm | Typical consumers | Hardware need | go-inference gap |
|---|---|---|---|---|
| **GPTQ** | post-training affine int4 (+ optional desc_act reordering), calibrated on a small sample set | vLLM, GPTQModel/AutoGPTQ, ExLlama(v2), TGI | none intrinsically — INT4 dequant-then-matmul runs anywhere; fused/fast kernels (Marlin, ExLlama) are CUDA-first, ROCm support is partial | **numeric core already exists** — `autoround.QuantizeWeights` is the same scale/zero-point affine int quantisation family GPTQ writes. Missing piece is GPTQ's int32-word packing + `quantization_config.json` shape (bits/group_size/desc_act/sym) — a format/layout task, not a new algorithm. |
| **AWQ** | activation-aware per-channel salient-weight protection + affine int4 | transformers, vLLM, ExLlama(v2 — HF's own docs say AMD-supported), AutoAWQ | same as GPTQ — dequant is hardware-agnostic; fused GEMM kernels are again CUDA-first | needs a genuinely new algorithmic step (per-channel activation-scale search from calibration data) layered on the existing affine core — more effort than GPTQ. |
| **bitsandbytes NF4/FP4** | non-uniform 4-bit codebook (NF4: 16 fixed quantile-fitted levels) ± double-quantisation of the scales | transformers/PEFT/QLoRA — typically quantised **at load time** inside the framework; less often distributed as a pre-quantised Hub artefact than GPTQ/AWQ | CUDA CC 6.0+; **ROCm preview wheels target gfx1101 (ours) explicitly**, ROCm 6.2–7.2.4 | a genuinely different algorithm from anything in the repo (nearest-codebook, not affine) — small and well-specified, but new; and the "export a static artefact" use case is a smaller slice of NF4's real-world usage than GPTQ/AWQ's "share a pre-quantised checkpoint" pattern. |
| **fp8, plain, non-MX** (e4m3/e5m2, per-tensor or per-block static/dynamic) | direct cast + one scale (no packing) | vLLM (first-class W8A8 fp8), TensorRT-LLM | matrix-core accel is CDNA3+/RDNA4/Hopper+ only (§2) — **our own box gets nothing** from it | trivial to WRITE (config already read by quantfmt); the addressable audience skews datacenter/high-end-consumer, not "the non-Mac user" our GGUF lane already serves. |
| **MXFP4/MXFP6/MXFP8** | OCP MX: 32-element block, E8M0 shared exponent, E2M1/E3M2/E2M3/E4M3/E5M2 elements | vLLM/TensorRT-LLM/Quark-exported checkpoints for MI350- or Blackwell-class serving | CDNA4 (fp4/fp6/fp8) or RDNA4 (fp8 only) matrix cores — **nothing we own accelerates any of this** | an entirely new tensor layout (shared power-of-two exponent, no bias term) — not a repack of the affine format; no reference implementation on hand to byte-verify against, the way mlxaffine had `mlx_lm.convert`'s own output. Highest effort, targets hardware tiers we can't validate on. |

`vLLM`'s GGUF-on-ROCm path is itself rough today (open GitHub issues on load
failures and ROCm CI instability) — reinforcing that GGUF's real practical home is
the llama.cpp family of consumers, not vLLM, and that a vLLM-shaped export (GPTQ/AWQ/
fp8/MX — vLLM's native quant formats) is a genuinely different audience than GGUF's.

## 5. Recommendation

**Confirm** "affine default for Mac+hip, GGUF for everyone else" — with one framing
correction: affine isn't "Mac's" default, it's the shared **native runtime** format
for both `engine/metal` and `engine/hip` (both consume packed MLX-affine tensors
directly, on-device, no host dequant step). GGUF is correctly the portable
**interchange** lane — the one format go-inference both reads (dequant-on-load) and
writes, for consumers outside our own engine (the llama.cpp family). Nothing in this
research overturns that resting position.

**The single best next format: a GPTQ writer, reusing `autoround`'s existing
affine-int quantiser core.**

Ranked, sorted portable-now vs hardware-tier-later per the kernel doctrine:

1. **GPTQ export** (portable-now, low effort) — the numeric core (symmetric/
   asymmetric affine int4, `autoround.QuantizeWeights`) already exists; the gap is
   purely format/layout (int32-word packing + `quantization_config.json` shape), the
   same class of work `mlxaffine` already proved out (reverse-engineer + byte-verify
   a foreign packed format). Broadest current non-Mac/non-our-engine consumer base
   (vLLM, GPTQModel, ExLlama, TGI). Zero hardware dependency to *write* it — dequant-
   then-matmul runs everywhere; fused kernels are a consumer-side concern.
2. **AWQ export** (portable-now, medium effort) — same destination format family as
   GPTQ but needs a genuinely new calibration step (per-channel activation-aware
   scaling) on top of the existing affine core. Do this after GPTQ, not instead of
   it — most of the safetensors/config plumbing is shared.
3. **Plain fp8 (e4m3/e5m2, non-MX)** (portable-now, trivial effort, narrow audience)
   — the cheapest possible thing to write (one scale, one cast, no packing at all),
   but the payoff needs CDNA3+/RDNA4/Hopper+ hardware to matter, which is a smaller
   slice of "the non-Mac user" than GPTQ/AWQ's audience. Low priority unless a
   specific datacenter-serving ask shows up.
4. **bitsandbytes NF4** (portable-now on our *exact* hardware per the wheel table,
   medium effort) — the genuinely surprising finding of this research: gfx1101 has
   first-class preview support today. But it's a new algorithm (codebook, not
   affine), and NF4's Hub usage pattern (quantise-at-load inside transformers) makes
   a static-artefact writer less obviously useful than GPTQ/AWQ's "share a
   pre-quantised checkpoint" pattern. Worth a second look if a concrete QLoRA/PEFT
   consumer asks for it.
5. **MXFP4/MXFP6/MXFP8** (hardware-tier-later, highest effort) — hold. Every
   accelerating architecture (CDNA4 for fp4/fp6, CDNA4/RDNA4 for fp8) is hardware we
   don't have in the fleet; the tensor layout is a genuinely new format (shared
   power-of-two exponent block, not a repack of affine); and there's no reference
   implementation on hand to byte-verify against, unlike mlxaffine's `mlx_lm.convert`
   anchor. Don't build this speculatively — revisit when either real CDNA4/RDNA4
   hardware or a concrete consumer ask exists.

## 6. Open questions for the operator

- Do we actually want go-inference to be a **general-purpose HF-ecosystem exporter**
  (produce GPTQ/AWQ artefacts for other people's serving stacks), or is the
  "non-Mac user" framing really about **our own engine's** portable path — in which
  case GGUF already covers it and GPTQ/AWQ add no runtime value to go-inference
  itself? The ranking above assumes the former (export-tool value: someone already
  running `lem quant` locally, offline, without a Python/CUDA calibration toolchain,
  gets a checkpoint vLLM/TGI can serve). If the latter is the actual goal, GPTQ/AWQ
  drop off the list entirely and the honest answer becomes "ship nothing new — GGUF
  is already the right everyone-else format."
- The hipBLASLt "H"-suffixed Tensile libraries (`gfx940H/941H/942H/950H`) sit
  alongside the plain RDNA/CDNA libraries in the same `.so`. I read this as a
  CDNA3+/CDNA4-only fp8/MX kernel set (consistent with the precision table) but
  couldn't confirm the dtype split from `strings` alone. Worth a `hipblaslt-bench`
  capability query on the homelab box if this ever becomes load-bearing.
- AMD Quark isn't installed anywhere in the fleet, and its exact RDNA-vs-CDNA
  hardware gating per algorithm isn't nailed down from the docs alone — if MXFP4/fp8
  work is ever prioritised, Quark (not a hand-rolled quantiser) is very likely the
  right calibration front end to sit in front of our export writers, given it already
  emits GGUF/safetensors.

## Sources

- ROCm precision support matrix — https://rocm.docs.amd.com/en/latest/compatibility/precision-support.html
- ROCm GPU architecture specs — https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html
- hipBLASLt docs — https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/index.html
- AMD Quark — https://quark.docs.amd.com/latest/
- OCP Microscaling formats (secondary source, primary PDF 403'd) — https://en.wikipedia.org/wiki/Block_floating_point ; primary: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- NVIDIA FP8 introduction — https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/
- HF Transformers AWQ guide — https://huggingface.co/docs/transformers/en/quantization/awq
- bitsandbytes installation guide (ROCm wheel target table) — https://huggingface.co/docs/bitsandbytes/main/en/installation
- vLLM GGUF/ROCm issue tracker (queried via WebFetch, not individually cited) — https://github.com/vllm-project/vllm/issues
- Homelab ground truth — `ssh homelab 'rocminfo'`, `/opt/rocm/.info/version`, `strings /opt/rocm/lib/libhipblaslt.so.1` (2026-07-12)
