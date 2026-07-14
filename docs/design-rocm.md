---
title: ROCm/AMD quant design — engine/hip vs the shared quant/scheme layers
description: Splits the ROCm quant-serving problem along the fence — HIP-only compute kernels vs engine-neutral scheme/format protocols — so the two halves dispatch independently.
---

# ROCm/AMD quant design

AMD are shipping serious accelerators (MI300/MI350 CDNA, RDNA4 consumer) and good
ROCm support is strategic for go-inference. This document maps a "ROCm quant
variant" onto this repo's actual seams, splits the work along the fence between
`engine/hip` (fenced, HIP-owner-only) and the general go-inference layers
(shared, any-engine), and ends with a dispatch table so the split can be worked
independently.

**Grounding method**: every file/symbol cited below was read in this tree at
commit `fb3bfb2` (worktree `goinf-rocm-design`). Where a claim is about AMD
hardware rather than this repo, a source URL is given; anything I could not
verify is flagged in **Open questions**, not asserted.

---

## The headline finding

`engine/hip` is an **island**. It was landed as a wholesale port from a
formerly-independent repo (`dappco.re/go/rocm`, see the package doc in
`go/engine/hip/rocm.go:1-14`: *"quarantined into go-mlx as the Tier-4 pkg/hip
engine"*) and has not yet been reconciled with go-inference's shared
engine-neutral layers. Concretely, for every general-purpose registry
`engine/metal` consumes, `engine/hip` has grown its **own parallel copy**
instead:

| Shared, engine-neutral home | `engine/metal` consumes it | `engine/hip` has its own copy instead |
|---|---|---|
| `engine/scheme` (mixer/cache/quant identity registry) | yes — `engine/metal/scheme.go` imports `engine/scheme` | yes — `engine/hip/scheme/{scheme,builtin}.go`, a byte-for-byte reimplementation of the same interfaces |
| `model.RegisterBackendQuant` (backend↔kind compute cross-section) | yes — `engine/metal/model_quant.go:25` | no — nothing in `engine/hip` calls it |
| `model/gguf` (shared GGUF metadata parser) | yes — `engine/metal/assistant_gguf.go` | no — `engine/hip/internal/gguf/gguf.go` (968 lines) is a separate, internal parser |
| `model.Arch` / `model.LookupArch` (architecture declarations) | yes | **yes** — `engine/hip/gemma4_architecture_adapter.go:84` — hip does correctly consume this one |

That last row matters: hip is not *incapable* of consuming the shared layers —
it already does for architecture declarations. The scheme/quant-catalogue/GGUF
duplication is a specific, addressable gap left over from the quarantine
landing, not an architectural necessity. This is the shape of the "general"
half of the work below (Part B): reconnect the island, don't build a bridge
that doesn't need to exist.

---

## Part A — engine/hip specifics (fenced; DESIGN ONLY, no code written here)

### A.1 What ROCm compute already exists for quantised inference

`engine/hip/kernels/rocm_kernels.hip` (12,512 lines) is the single HIP
translation unit; exported kernel names are catalogued in
`engine/hip/kernels/README.md`. For weight-quant specifically, the linked
kernels are:

- `rocm_mlx_q4_projection` / `_batch` / `_greedy` / `_triple_projection` /
  `_pair_projection` — MLX-style group-affine 4/6/8-bit packed row projection.
  `_batch` is the M>1 (prefill) path, the un-suffixed and `_greedy` variants
  are M=1 (decode). This is the **same quant family** `engine/metal` serves
  (`scheme.RegisterQuant(quantInfo{"affine", 0})` in
  `engine/scheme/builtin.go:104`) — HIP already has an affine matvec/matmul
  pair, it is just not wired through the shared registry (see Part B).
- `rocm_jangtq_projection`, `rocm_codebook_lookup` — fixture kernels for
  JANGTQ and codebook/VQ; per `native.go:2543-2546` these are
  `rocmFixtureKernelCapability` entries: **"fixture kernel is linked; ...model
  integration remains pending"** — i.e. the kernel exists and runs on a
  synthetic fixture, but no loaded model routes through it yet.
- `rocm_embedding_lookup` — supports f32, BF16, and MLX-affine 4/6/8-bit
  embedding tables (README.md:82), including Gemma4's packed U32 weights.

**Read the kernel body, not just the name** — I read
`rocm_mlx_q4_projection` (`rocm_kernels.hip:5522-5599`). It is a **hand-rolled,
warp-shuffle-reduction, scalar dot-product** kernel: one thread group per
output-row block, each thread unpacks 4-bit nibbles from a `uint32_t` word,
multiplies against `float` activations, and reduces across the row's lane
group with `rocm_shfl_down`. There is **no MFMA, no WMMA, no hipBLASLt/rocBLAS
call, no Composable Kernel template** anywhere in the 12.5k-line file
(`grep -n "mfma\|wmma\|__builtin_amdgcn\|rocblas\|hipblas\|composable_kernel"
rocm_kernels.hip` — zero hits). This is architecturally equivalent to what a
naive/reference Metal `qmv` kernel looks like *before* the house fusion work
(`engine/metal/kernels/lthn_qmv_rows.metal`, `lthn_rms_qmv.metal`) — but
`engine/metal` has since grown megakernels (`lthn_attn_megakernel.metal`,
`lthn_ffn_megakernel.metal`, `lthn_gemv2_megakernel.metal`,
`lthn_layer_megakernel.metal` — `grep -c "^extern \"C\" __global__"
rocm_kernels.hip` = 28 free-standing kernels, `grep -n "megakernel"
rocm_kernels.hip` = zero). **HIP's quant compute is at the pre-fusion stage
metal was at before the O(output)-only-fold campaign** — that campaign's
lessons (see `reference_metal_engine_perf_working_rules` memory: small-K
physics, occupancy-bound, no grid barriers) are directly re-applicable once a
matrix-core path exists to fuse around.

The **quantise-direction** (float32 → packed) side is further behind than the
decode direction: `engine/hip/hip_autoround_quant_launch.go` defines a full Go
request/response/launch-args shape for AutoRound quantisation into
MXFP4/NVFP4/FP8/MXFP8/INT2 (`hipAutoRoundFormat*` constants,
`hip_autoround_quant_launch.go:20-26`), but the capability report is explicit
that this is metadata-only: `native.go:2583` sets
`labels["autoround_hip_kernel"] = hipKernelStatusNotLinked` and
`labels["autoround_runtime"] = "planned_hip"`. There is no
`rocm_autoround_quantize` symbol in the kernels/README.md export list. **This
is Go-side plumbing waiting for a kernel, not a kernel with a Go binding
gap.**

### A.2 How AMD actually accelerates low precision — hardware reality by architecture

Two distinct matrix-core instruction families exist, and AMD's own docs are
sparse on cross-generation deltas, so the table below is assembled from
several primary/near-primary sources (cited per row) — verify against the
[AMD Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)
before committing to a specific instruction mnemonic in kernel code.

| Architecture | Instruction family | Native low-precision in matrix cores | Source |
|---|---|---|---|
| CDNA2 (MI200) | MFMA | fp16/bf16/fp32/fp64 only — **no** native FP8 | ROCm precision-support docs |
| CDNA3 (MI300) | MFMA | **FP8 E4M3/E5M2, FNUZ variant** (differs from Nvidia H100's OCP FP8) | ROCm precision-support docs; `rocm.docs.amd.com/en/latest/reference/precision-support.html` |
| CDNA4 (MI350/MI355) | MFMA | FP8 (E4M3/E5M2, OCP this time) **+ FP4 (E2M1) + FP6 (E2M3/E3M2)**, all **with OCP microscaling (MX) block-shared-exponent scaling** | AMD MI350 product page; `rocm.blogs.amd.com/software-tools-optimization/mxfp4-mxfp6-quantization/` |
| RDNA3 (RX 7000) | WMMA | fp16/bf16/fp32, **int8 (IU8)**, int4 (IU4) — no FP8 | `gpuopen.com/learn/wmma_on_rdna3/` |
| RDNA4 (RX 9000, gfx12) | WMMA | fp16/bf16/int8 confirmed by AMD's own "Using Matrix Core" guide (`gpuopen.com/learn/using_matrix_core_amd_rdna4/`, doubles RDNA3 throughput: int8 512→2048 FLOPS/clock/CU); **native FP8 E4M3 also present** — instruction `v_wmma_f32_16x16x16_fp8_fp8`, independently confirmed by a vLLM enablement writeup with real throughput deltas (Qwen3-30B: 52→85 tok/s) | See **Open questions** — AMD's own GPUOpen "matrix core" intro article for RDNA4 does not itself list FP8 in its data-type table, only vLLM/community sources currently confirm the FP8 WMMA tile |

**MXFP4/OCP microscaling, concretely** (relevant to the low-end + Bonsai
1-bit ambition): a block of 32 elements shares one 8-bit E8M0 exponent; MXFP4
elements are E2M1 (range ±6), MXFP6 is E2M3 (range ±7.5). This is a **block
scale format, not a sub-4-bit format** — the narrowest OCP MX element AMD
accelerates in hardware today (MI350/CDNA4) is 4-bit. **No AMD part
accelerates sub-4-bit (2-bit/1-bit) in matrix cores** — a Bonsai 1-bit /
TurboQuant-style scheme on ROCm is necessarily a **dequant-to-fp8-or-bf16 then
MFMA/WMMA** path, exactly the shape `engine/hip`'s existing TurboQuant KV
codec already takes (`native.go:2587-2592`: `kv_compression_runtime =
"cpu_reference"` — it isn't even GPU-resident yet). This mirrors what
`engine/hip/model/quant.go:207-217`'s `builtinQuantSchemes()` already encodes:
`mxfp4`/`mxfp8`/`nvfp4` are `QuantSchemeRuntimePlannedHIP` +
`FeatureRuntimePlanned` — the Go-side scheme catalogue already anticipated
this hardware fact before I went looking for it.

**The vLLM-ROCm quant compatibility matrix is a directly-relevant signal for
where to spend effort.** vLLM's own AMD support table
(`docs.vllm.ai/en/latest/features/quantization/`) marks **GPTQ, AWQ, and
Marlin as unsupported on AMD**, while **GGUF and FP8 (W8A8, via llm-compressor)
are supported**. This is not a policy choice — GPTQ/AWQ's fast-path kernels
(Marlin) are hand-written for Nvidia tensor-core PTX and have no ROCm
equivalent; the ROCm ecosystem's answer to "4-bit weight-only" is GGUF-style
group quantisation (which `rocm_mlx_q4_projection` already serves, being MLX's
own group-affine format) plus, going forward, AMD Quark's MXFP4/FP8 output. **I
read this as validation, not a coincidence, that HIP's kernel surface today has
MLX-affine + fixture JANGTQ/codebook and nothing for GPTQ/AWQ** — building a
GPTQ/AWQ decode kernel for HIP would be swimming against the ecosystem's own
kernel-library reality; MXFP4 (Quark), FP8 (W8A8), and GGUF q4_0/q4_K/q8_0 are
the native-feeling lanes. See `docs.vllm.ai` and `quark.docs.amd.com/latest/`
(AMD Quark: PTQ/QAT toolkit, SmoothQuant/AWQ/GPTQ/Qronos algorithms, but its
**output formats** are `int4/uint4/int8/fp8(e4m3fn,e5m2)/MX` — GPTQ/AWQ are
input recipes Quark can *re-emit* as FP8/MX for AMD hardware, not something
Quark ships a ROCm GPTQ kernel for).

### A.3 The engine/hip build/test reality

- Default build: `linux && amd64 && !rocm_legacy_server` (native HIP runtime);
  `!linux || !amd64` compiles a stub that always reports `Available() ==
  false` (`rocm_stub.go`); a third `rocm_legacy_server` tag keeps the old
  llama-server subprocess bridge alive but unbuilt by default
  (`docs/backends.md:38-46`, verified against `register_rocm.go` /
  `rocm_stub.go` build tags).
- **No `MLX_METALLIB_PATH`-equivalent gate exists yet.** Metal's kernels are
  gated by an environment variable pointing at a pre-built `.metallib`, wired
  into the Taskfile (`task metallib`, `task test:metal`). HIP's kernel testing
  is currently **manual, undocumented in the Taskfile**, and lives entirely in
  `engine/hip/kernels/README.md`'s env-var recipes:
  `GO_ROCM_RUN_HIP_TESTS=1 GO_ROCM_KERNEL_HSACO=<path> go test ./go -run
  'TestHIPHardware.*KernelSource'`, plus opt-in compile-portability tests
  (`GO_ROCM_RUN_AMD_HIP_COMPILE_TESTS`, `..._NVIDIA_HIP_COMPILE_TESTS` via
  `CUDA_PATH`, `..._HIP_CPU_COMPILE_TESTS`/`RUNTIME_TESTS` via HIP-CPU). There
  **is** one Taskfile hook: `bench:hip:gemma4-sweep`
  (`Taskfile.yml:168-172`, `GO_ROCM_RUN_GEMMA4_SWEEP_RECEIPT=1`), so the
  PASS-count receipt culture has a toehold but not the build-gate discipline
  metal has.
- **Safetensors model-pack loading is not yet available** on HIP — GGUF
  loading works, but per `docs/backends.md:46`: *"safetensors model-pack
  loading is not yet available in the current quarantine landing (blocked on
  a missing upstream package — the load fails with an explicit message rather
  than guessing)"*. Any quant format whose checkpoints ship as safetensors
  (GPTQ/AWQ/compressed-tensors/most HF exports) is blocked upstream of the
  quant question entirely until this lands — **this is likely the single
  highest-leverage engine/hip item on the list**, since it gates every
  format below GGUF.
- The hardware test matrix (`hip_hardware_test.go`, 327KB) is real HIP-device
  hardware testing gated behind env vars, mirroring the intent of metal's
  discipline but not (yet) its Taskfile ergonomics — bringing `task
  test:hip` (gated on `linux && amd64` + a HIP toolchain probe) into the
  Taskfile the way `task metallib` / `task test:metal` exist for metal is a
  small, mechanical win.

### A.4 What Part A is NOT

Nothing above should be read as "build MFMA/WMMA kernels now." The honest
state is: HIP has a working, scalar reference implementation for the one
quant family that matters most today (MLX-affine, matching what ships in
production GGUF/MLX checkpoints), fixture kernels for two more, and Go-side
scheme metadata anticipating three formats (MXFP4/MXFP8/NVFP4) with zero
matching kernel code. The fenced HIP owner's next real decision is **which
matrix-core path to target first** (RDNA4 consumer WMMA int8/fp8, or CDNA3/4
MFMA fp8/mx) — that decision needs a real device to benchmark against and is
explicitly out of scope for this design-only doc.

---

## Part B — general go-inference protocols (engine-neutral; shared territory)

### B.1 Quant scheme registration — the fork that needs healing, not a new mechanism

The general quant-identity registry **already exists** and is **already
designed** to be engine-neutral:

```go
// engine/scheme/scheme.go
type QuantScheme interface {
    Kind() string // the quantization.kind a model declares ("affine", "q4_0", …)
    Bits() int    // nominal bit-width; 0 means "the model's config declares it"
}
func RegisterQuant(q QuantScheme) core.Result
func QuantFor(kind string) (QuantScheme, bool)
```

`engine/scheme/builtin.go:104` registers exactly one entry today: `affine`.
`engine/metal` consumes this package directly
(`engine/metal/scheme.go`/`engine/metal/attention.go`). **`engine/hip` does
not import `engine/scheme` at all** — instead `engine/hip/scheme/{scheme,builtin}.go`
is an independently-maintained package with the **identical interface shape**
(`Kind()`/`Bits()`, `core.NewRegistry[QuantScheme]()`, even the same
`Compatible(mixer, cache)` helper) but registers **seven** quant kinds
(`affine, bf16, mxfp4, mxfp8, nvfp4, q4_0, jangtq` —
`engine/hip/scheme/builtin.go:66-76`) and **eleven** mixer kinds including
`mamba2/rwkv7/gla/retnet/deltanet/gsa/nsa/moba/mla` that `engine/scheme`'s
builtin doesn't know about at all (`engine/scheme/builtin.go:76` registers
only `softmax-hybrid`).

**This is the finding, not a hypothesis to validate**: HIP's scheme registry
is currently *more complete* than the general one for both mixers and quant
kinds, and none of that richer information is visible to any caller that only
knows `engine/scheme`. Two consequences:

1. A caller asking `scheme.QuantKinds()` (the general "what can I load"
   catalogue, per its own doc comment) gets a **wrong, incomplete answer**
   whenever HIP is the loaded backend — it doesn't know `mxfp4` exists.
2. When the FLA/SSM mixer work lands (the memory index's "flash-linear-attention
   mixers" item), whoever writes `RegisterMixer(mamba2{})` against
   `engine/scheme` will be duplicating work `engine/hip/scheme/builtin.go:37`
   already did — from the *wrong* registry's perspective.

**Where should a new ROCm-relevant scheme (AMD Quark / compressed-tensors /
MXFP4 checkpoint) register so it is recognised regardless of engine?**
`engine/scheme` — that is the package whose own doc comment says *"every
Engine (metal on Apple, rocm on AMD/CUDA/CPU) inherits this one scheme
catalogue"* (`engine/scheme/scheme.go:16-21`). It does not need to move to the
`dappco.re/go` core module (the generic `core.NewRegistry[T]` primitive it's
built on is already correctly SPOR'd there and reused by both scheme
packages) — it needs `engine/hip` to **import and register against it
instead of forking it**. The `engine/hip/model/quant.go` `QuantScheme` type
(richer: `Loader`, `Source`, `Runtime`, `RuntimeStatus`, `Labels` — a
capability-reporting/route-resolution shape, not just identity) is a
different, legitimate concern from `engine/scheme.QuantScheme` (a bare
identity contract) and can stay hip-local **as long as it also calls
`scheme.RegisterQuant`** for the identity half — the two are not mutually
exclusive, they answer different questions ("what quant kinds exist" vs "how
does this loaded model's route resolve").

### B.2 The capability contract — how an engine advertises what it can serve

`inference.CapabilityReport` (`go/capability.go:179-190`) already carries a
`Quantizations []string` field and a `Capabilities []Capability` list keyed
by stable `CapabilityID`s (`CapabilityQuantization` among them,
`go/inference.go:66`). Both `engine/metal` and `engine/hip` populate this —
HIP's is notably rich (`rocmCapabilityQuantizations` at
`engine/hip/native.go:2908-2930` lists 20 tokens including `mxfp4`, `nvfp4`,
`jangtq`, `codebook`; `rocmQuantizationCapabilityLabels`,
`native.go:2564-2610`, attaches ~30 fine-grained labels per report, e.g.
`weight_quantization_runtime = "metadata"`, `kv_compression_runtime =
"cpu_reference"`). This is the **discovery** half of the contract and it
already works without either engine importing the other — a caller does
`inference.CapabilitiesOf(backend)` or `report.Supports(CapabilityQuantization)`
and never sees `engine/metal` or `engine/hip` package names.

The **compute-dispatch** half — how a loaded backend's actual matvec/matmul
gets called for a given quant kind — already has a clean, existing mechanism
too, and it is the one HIP is missing:

```go
// model/quant.go (root, engine-neutral)
type QuantMatVec interface {
    scheme.QuantScheme // Kind() + Bits()
    MatVec(x, packed, scales, biases []byte, outDim, inDim, groupSize, bits int) ([]byte, error)
}
func RegisterBackendQuant(backend string, q QuantMatVec) core.Result // keyed "backend/kind"
func BackendQuant(backend, kind string) (QuantMatVec, bool)
```

`engine/metal/model_quant.go:25` is the **entire** metal-side registration:
`func init() { model.RegisterBackendQuant("native", affineQMV{}) }`. This is
precisely the "capability contract" the design brief asked to locate: **a
caller resolves `model.BackendQuant("rocm", "affine")` and gets back the
loaded ROCm backend's `MatVec`, without either package importing the other's
concrete types.** The interface trades in raw bytes (bf16 activations in/out,
packed weight + scales + biases bytes) specifically so it doesn't need a
shared tensor type — it is already engine-agnostic by construction. **HIP
registers nothing here.** This is the second, more consequential half of the
"reconnect the island" work (B.1 was identity/discovery; this is compute
routing) — and it's a small, mechanical addition once `rocm_mlx_q4_projection`
has a Go-side `MatVec`-shaped wrapper (it likely already has one internally,
given the kernel exists and is exercised by tests — the work is exposing that
wrapper through this seam, not writing new compute).

### B.3 Format/loader neutrality — what's already engine-neutral, what leaks

- **`model/quant/quantfmt` (root)** — genuinely engine-neutral today. Pure Go,
  zero engine imports, parses the *HuggingFace* `quantization_config` block
  (GPTQ/AWQ/compressed-tensors/FP8/bitsandbytes — `quantfmt.go:33-50`) into a
  normalised `QuantInfo{Method, Bits, GroupSize, Scheme, Symmetric, DescAct,
  Activation}`. Neither `engine/metal` nor `engine/hip` currently calls it
  (unverified beyond a grep of the two engine trees for `quantfmt` — zero
  hits) — it may be wired only at a model-conversion/export boundary today,
  not the load path. **Gap**: no `MethodMXFP4`/`Method` case for AMD Quark's
  MX-format `quant_method` string — worth adding once Quark's actual
  `config.json` shape is confirmed (see Open questions).
- **`model.QuantConfig` (root `model/quant_config.go`)** — a *different*
  parser for a *different* config shape: the **MLX-native** `quantization`
  block (`group_size`, `bits`, `mode` ∈ `{affine, mxfp4, mxfp8, nvfp4}` —
  `quant_config.go:100-128`). This is already MX-format-aware at the
  *validation* level (it accepts `mxfp4`/`mxfp8`/`nvfp4` modes with their
  correct group sizes — 32 for mxfp4/mxfp8, 16 for nvfp4) even though no
  engine has a kernel for them yet — the Go-side plumbing is ahead of the
  compute, consistently with what I found in `engine/hip/model/quant.go`'s
  builtin schemes (A.1). **This confirms the MX-format story is a
  cross-cutting, already-anticipated piece of the design, not a green-field
  addition** — the work is kernels + wiring, not new schema.
  `quantfmt.QuantInfo` and `model.QuantConfig` do not share code today and
  answer genuinely different questions (HF checkpoint provenance vs MLX
  runtime block) — that's a defensible split, not obviously a duplication to
  collapse, but worth naming so nobody builds a third parser for a third
  format assuming neither existing one applies.
- **The exporter side** (`model/quant/{gptq,awq,fp8,nf4,autoround,jang,codebook,mlxaffine}/`)
  is genuinely engine-neutral pure-Go pack/export code (confirmed no engine
  imports via the package tree — each is `export.go`/`pack.go`/`load.go`
  shaped, producing packed weight bytes + a snapshot format). **None of these
  formats except MLX-affine (and the fixture-stage JANGTQ/codebook) have a
  matching decode kernel in *either* engine** — GPTQ/AWQ/FP8/NF4/AutoRound
  are export/conversion tooling with no runtime consumer yet on metal *or*
  hip. This is not a ROCm-specific gap; it's worth flagging so ROCm work
  doesn't get scoped as "catch up to metal" when metal doesn't serve these
  either.
- **`engine/hip/internal/gguf/gguf.go`** (968 lines) vs the shared
  `model/gguf` package (imported by root `inference` (`gguf.go:10`) and
  `engine/metal/assistant_gguf.go`) — confirmed independent implementations;
  `engine/hip/internal/gguf` does not import `model/gguf` (grep for the
  import path in the hip tree returns zero hits). This is the same
  "quarantine landing left an island" pattern as the scheme fork (B.1), just
  for GGUF metadata instead of quant identity. Out of scope for *this* quant
  design doc's dispatch table (it's not quant-specific), but flagged because
  a ROCm-relevant scheme detector that reads GGUF tensor types (`q4_0`,
  `q4_K_M`, `q8_0` — all present in `rocmCapabilityQuantizations`,
  `native.go:2919-2927`) will hit this fork immediately.

### B.4 The externals boundary — where does a ROCm scheme definition home?

go-inference consumes `dappco.re/go` (core), `dappco.re/go/api`,
`dappco.re/go/cgo`, `dappco.re/go/io`, `dappco.re/go/process` etc. as
versioned externals (`go/go.mod:6-17`) — the SPOR discipline that applies
across that boundary (best-version-wins homed in the shared external, engine
repos consume) is **not the relevant boundary here**. A ROCm quant scheme
definition is inference-domain-specific (it names a `QuantScheme.Kind()` a
*model* declares) — it has no business in the generic `core` module alongside
`core.NewRegistry[T]`/`core.Result`. **It homes in `engine/scheme`**, this
repo's own already-correctly-positioned shared package (B.1). The externals
boundary is relevant only one level down: if a future MXFP4/OCP-microscaling
*codec* (the actual scale/dequant math) turns out to be identical bit-twiddling
useful outside inference too, *that* utility — not the scheme registration —
would be a `dappco.re/go` core candidate, exactly as `core.NewRegistry[T]`
already is.

### B.5 Cross-engine parity — testing a contract without a physical AMD box in CI

`engine/enginetest` (`engine/enginetest.go`) already exists and is exactly the
right shape for this: a `SessionFactory func(t *testing.T)
inference.SessionHandle` — an engine imports the suite in its own tests and
hands it a factory; the suite exercises lifecycle/error/shape invariants
**deliberately independent of model quality or output content**
(`enginetest.go:19-22`). Today it covers session/textmodel contract
conformance, not quant-output equivalence — extending it (or a sibling
package) with a **byte-identity or tolerance-bounded parity check**, given a
shared small quantised fixture checkpoint and a reference dequant computed in
pure Go (no engine), is the natural next step: each engine's `MatVec`
(B.2's `model.BackendQuant`) gets asserted against the same reference output.
This runs entirely on CI without ROCm hardware because it validates the
*dequant arithmetic*, not throughput — the actual GPU kernel still needs
Snider's physical AMD box for the hardware-truth pass (mirroring how metal's
own kernels get validated against a pure-Go reference before/alongside GPU
runs, per the `qmv_fast_impl` byte-identity port pattern in memory). **This is
squarely a "general" item** — the harness belongs in `engine/enginetest`
where both engines' tests already look for it, not duplicated per-engine.

---

## Dispatch boundary

Each item tagged for the fenced HIP owner or for shared-layer ("general")
work. Counts: **7 `[engine/hip]`**, **8 `[general]`**.

| # | Item | Tag | Notes |
|---|---|---|---|
| 1 | Land safetensors model-pack loading on HIP | `[engine/hip]` | Blocks every quant format shipped as safetensors (most HF exports); currently fails with an explicit "not available" error (`docs/backends.md:46`). Highest-leverage single item. |
| 2 | Decide + prototype a matrix-core path (RDNA4 WMMA int8/fp8, or CDNA3/4 MFMA fp8/mx) for the affine quant matvec/matmul | `[engine/hip]` | Needs the physical AMD box; current kernel is scalar/warp-shuffle only (A.1). Do this *after* item 1 unblocks real checkpoints to benchmark against. |
| 3 | Land a real GPU kernel for AutoRound MXFP4/NVFP4/FP8/MXFP8/INT2 quantise-direction | `[engine/hip]` | Go-side request/launch-args shape already exists (`hip_autoround_quant_launch.go`); no kernel. |
| 4 | Route JANGTQ + codebook fixture kernels through a loaded model (not just synthetic fixtures) | `[engine/hip]` | Kernels are linked and tested (`hip_jangtq_launch.go`, `hip_codebook_launch.go`); production model integration is the pending piece per the capability labels. |
| 5 | GPU-resident TurboQuant KV codec (currently `kv_compression_runtime = "cpu_reference"`) | `[engine/hip]` | Out of scope for weight quant but flagged since it shares the "no sub-4-bit hardware accelerate" constraint from A.2. |
| 6 | Bring HIP kernel/hardware testing into the Taskfile (`task test:hip`, mirroring `task metallib`/`task test:metal`) | `[engine/hip]` | Mechanical; env vars already documented in `kernels/README.md`, just not Taskfile-wired. |
| 7 | Expose `rocm_mlx_q4_projection`'s Go wrapper as a `model.QuantMatVec` implementation | `[engine/hip]` | The kernel-calling code already exists somewhere in the HIP tree (the kernel is tested); this is packaging that call to satisfy item 9's interface, so it is fenced (touches HIP-internal call sites) even though the *target* interface is general. |
| 8 | Reconnect `engine/hip/scheme` into `engine/scheme` — either delete the fork and call `scheme.Register{Mixer,Cache,Quant}` directly, or keep the hip-local richer metadata type but *also* register identity against the shared registry | `[general]` | B.1. Also lands the 8 mixer kinds (`mamba2`/`rwkv7`/`gla`/… ) the shared registry currently lacks — direct unblock for the FLA/SSM mixer work mentioned in the current-state tracker. |
| 9 | `engine/hip` calls `model.RegisterBackendQuant("rocm", ...)` for each linked kernel (starting with affine) | `[general]` | B.2. The interface (`model.QuantMatVec`) already exists and is engine-agnostic by construction (raw-bytes contract); this is the missing registration call, not new API. |
| 10 | Add an AMD-Quark-shaped `Method` (or confirm an existing one covers it) to `model/quant/quantfmt` | `[general]` | B.3. Needs Quark's actual emitted `quantization_config` JSON shape verified (Open questions) before the `Method` constant is added. |
| 11 | Decide whether `quantfmt.QuantInfo` (HF-checkpoint detection) and `model.QuantConfig` (MLX-runtime block) should be unified, cross-referenced, or intentionally kept separate | `[general]` | B.3. Not urgent — flagged so a third parser isn't built assuming neither covers a new format. |
| 12 | Reconcile `engine/hip/internal/gguf` with the shared `model/gguf` package | `[general]` | B.3. Same "quarantine island" pattern as item 8, for metadata parsing rather than quant identity; needed before any GGUF-tensor-type-driven scheme detection (`q4_0`/`q4_K_M`/`q8_0`) can be engine-neutral. |
| 13 | Extend `engine/enginetest` with a quant-parity check (pure-Go reference dequant vs each engine's `model.BackendQuant` output) | `[general]` | B.5. Runs in CI without AMD hardware; validates arithmetic, not throughput. |
| 14 | Land `engine/scheme` registration for MXFP4/MXFP8/NVFP4/q4_0/jangtq identities (currently only in the hip-local fork) | `[general]` | Follows directly from item 8 once the fork is resolved — listed separately because it's a content change (what's registered) vs item 8's structural change (where registration happens). |
| 15 | Document the "GGUF/FP8/MX are the AMD-native quant lanes; GPTQ/AWQ are not" ecosystem finding (A.2) somewhere durable (`docs/backends.md` rocm section, or a follow-up to this doc) | `[general]` | Prevents future work from being scoped as "port GPTQ to HIP" against the ecosystem's own grain. |

---

## Open questions

1. **RDNA4 native FP8 WMMA — confirmed or not?** AMD's own GPUOpen
   "Using the Matrix Cores of AMD RDNA 4" intro article's data-type table
   (fp16/bf16/int8) does not mention FP8, but a community/vLLM enablement
   writeup cites a specific instruction (`v_wmma_f32_16x16x16_fp8_fp8`) with
   real throughput numbers. I could not find an AMD RDNA4 ISA reference
   manual page to settle this definitively in the time available — before
   committing kernel design to "RDNA4 has native FP8 WMMA," check the
   [AMD RDNA4 ISA reference](https://www.amd.com/en/support/tech-docs) or the
   [AMD Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)
   output for gfx12 directly.
2. **AMD Quark's actual checkpoint `quantization_config` shape** — I read
   Quark's doc index describing its algorithms and output *formats*
   (ONNX/JSON-safetensors/GGUF) but did not find a concrete example
   `config.json` for its MX/FP8 output to confirm what `quant_method` string
   (if any) it writes, which `quantfmt.go` would need to recognise (dispatch
   item 10).
3. **Does anything currently call `quantfmt.Parse`?** A grep of `engine/`
   for `quantfmt` returned zero hits — I could not find the load-path caller
   in the time available; it's possible this package is wired at a
   conversion-tool boundary (`cli/` or `model/modelmgmt/`) rather than the
   inference load path. Worth confirming before assuming it's "the" HF
   checkpoint detector callers actually hit at load time.
4. **Is there a `MatVec`-shaped wrapper for `rocm_mlx_q4_projection` already
   in the HIP tree, or does dispatch item 7 need one written from scratch?**
   The kernel is clearly called from somewhere (it's exercised by tests per
   the kernels/README.md), but I did not trace the exact Go call site that
   invokes it during a real generate — worth a quick `grep -rn
   "rocm_mlx_q4_projection"` across the hip `.go` files before scoping item 7
   as "just add a wrapper."
5. **hipBLASLt / Composable Kernel** — I could not get substantive detail
   from either project's docs in the time available (both pages returned
   thin/navigational content to the fetch tool). Before scoping dispatch item
   2's matrix-core path, a fenced-owner-side spike should establish whether
   calling into `hipBLASLt` (a vendor GEMM library, analogous to calling into
   MLX's own kernels the way `engine/metal` loads `mlx.metallib`) is viable
   for this repo's no-vendor-library-dependency posture, or whether a
   hand-written MFMA/WMMA kernel (matching `engine/metal`'s own house-kernel
   strategy) is the more consistent choice.
