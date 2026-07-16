<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# Design — hybrid recurrence device port (the Qwen lane campaign)

**Problem.** The composed lane serves Qwen 3.5/3.6 hybrids at 3.41 tok/s (27B-4bit, M3 Ultra)
where mlx-lm serves the same snapshot at 41.3 — a ×12 architecture gap, not a tuning gap. Every
gated-delta layer round-trips host↔device: device GEMMs for the projections, then the causal conv,
the α/β gate transform, the delta-rule recurrence and the gated RMSNorm all run in host Go loops
(`model/arch/deltanet/deltanet.go` — a triple loop over [Hv,128,128] f64-accumulated state, ~48
layers × ~3 MiB of state read+written per token by the CPU), then more device GEMMs — ~368 command
buffers per token. Every reference engine (mlx-lm, omlx, llama.cpp, vllm-metal) runs the recurrence
as ONE kernel dispatch with the state resident in GPU registers across the whole token loop, and the
per-layer state living permanently on device.

**Target.** 27B-4bit decode ≥ 40 tok/s single-stream. The physics: 48 gated-delta layers × 3 MiB
f32 state = 144 MiB resident, ~288 MiB/token of state traffic ≈ 0.4 ms at M3 Ultra bandwidth; the
4-bit weights (~14 GB streamed per token) set a ~17 ms floor ≈ mid-50s tok/s ceiling. mlx-lm's 41.3
with zero cross-op fusion confirms the budget: state residency + few CBs/layer is enough; whole-token
chaining (the ArchSession trick) is upside beyond it.

## Reference readings (what to port, verified in source)

- **mlx-lm `models/gated_delta.py`** — the port source. Kernel `gated_delta_step`: grid
  `(32, Dv, B·Hv)`, tg `(32,4,1)`; one simdgroup per (head, dv-row); lane x owns `Dk/32` state
  floats in registers; the whole T-loop inside the kernel (T=1 decode, T=chunk prefill — same
  kernel); per step: decay `state·g` → `kv_mem = Σ state·k` (simd_sum) → `delta = (v−kv_mem)·β` →
  `state += k·delta` → `y = Σ state·q` (simd_sum, lane 0 writes). State `[B,Hv,Dv,Dk]` f32,
  Dk-contiguous rows. GQA repeat is index arithmetic (`hk = hv/(Hv/Hk)`), never materialised.
  q/k arrive pre-normalised: `q = (1/Dk)·rms_norm(q)`, `k = (1/√Dk)·rms_norm(k)` — algebraically
  our host's ℓ2-norm(k), ℓ2-norm(q)·(1/√Dk) factoring. g/β precomputed outside
  (`g = exp(−exp(A_log)·softplus(a+dt_bias))`, `β = sigmoid(b)`).
- **llama.cpp `ggml-metal.metal` `kernel_gated_delta_net_impl`** — two steals: (1) **snapshot
  slots**: kernel writes K trailing per-token states in-pass (`target_slot = T−1−t`; slot s = state
  s tokens back) so a speculative verify rolls back the recurrence for free — the lever that
  un-parks MTP block-verify on this arch; (2) scale applied to the OUTPUT (`y·scale`), not to q —
  one fewer vector pass. Also proves one kernel body serves scalar-g (GDN) and vector-g (KDA/Kimi)
  via a function constant — keep the seam, defer the KDA variant.
- **omlx `custom_kernels/qwen35_prefill/gdn.py`** — the measured verdict: their chunked-parallel
  WY kernels lost E2E to the sequential register-resident form on Apple Silicon
  (`qwen35_gdn_chunked.py`: blocked-seq 14.9 ms vs stock 29.7 ms at 16k, chunked slower still).
  Port the dumb-fast shape; revisit their Kernel S staging (threadgroup k/q/v tiles,
  `simd_shuffle_down` reduction) only if the prefill instrument demands it.
- **vllm-metal `attention/caches/gdn_cache.py`** — slot-structured state pools for continuous
  batching of hybrids. Not this campaign's first slices, but the state-buffer shape below keeps a
  leading batch/slot axis so it composes later.

## Geometry (Qwen3.6-27B, `text_config`)

64 layers, `full_attention_interval 4` → 48 linear_attention + 16 full_attention. Gated-delta:
`Hk=16, Hv=48, Dk=Dv=128` (square), conv K=4, `convDim = 2·qDim+vDim = 10240`, `vDim = 6144`,
hidden 5120. `mamba_ssm_dtype: float32` — state is f32 by spec. Composed lane activations are f32
end-to-end, so the kernels are f32-in/f32-out (no bf16 tier needed on this lane today).

## Decisions

1. **Kernel boundary = mlx-lm's**: recurrence only; conv/norms/gates/z-gating are separate tiny
   kernels in the same command buffer. Every reference factors it this way; it keeps the recurrence
   kernel shape-pure and lets each pre/post stage gate against its host twin independently.
2. **State layout `[slots, Hv, Dv, Dk]` f32, device-resident** (mlx-lm/llama.cpp transposed
   layout: a simdgroup reads one contiguous 512 B Dk-row). Slot 0 = live state; slots 1..K−1 =
   trailing snapshots (llama.cpp mapping), allocated from day one (48 layers × 4 slots × 3 MiB =
   576 MiB is fine; slots beyond 0 cost nothing until block-verify uses them). Conv ring
   `[(K−1), convDim]` f32 rides in the same per-layer state object.
3. **Scale on output** (llama.cpp form): q gets rms_norm only; `y·(1/√Dk)` at write.
4. **Quant-first**: the 27B checkpoint is 4-bit — the device block consumes the packed projections
   through the existing affine qmv/qmm emitters inside the block's CB. The f32 fused-input hook
   (`GatedDeltaInputDevice`) stays for dense checkpoints; the quant path stops bouncing through
   per-projection `ProjQuantMatMulInto` round-trips.
5. **Declare/bind (AX-8)**: `qwen3`/`composed` declare block-level device hooks exactly as they
   declare `GatedDeltaInputDevice` today; `engine/metal` binds them. The host path stays intact and
   authoritative — it is the parity reference and the non-metal build.
6. **State ownership**: `gatedDeltaState` grows an opaque device-state handle beside its host
   slices. Device-resident is the fast path; `CloneState` blits device-side; host export (readback)
   serves tests and the non-metal fallbacks. The handle is per-session, pooled like every other
   scratch.

## Slices (each lands with its receipt; lethean-perf discipline)

- **S0 — instrument. DONE.** `TestComposedDecodeProfileReal` (engine/metal) decodes 24 real
  tokens for `-cpuprofile`; `TestComposedDecodeRoundTripCensus` counts CBs. Baseline receipt
  (27B-4bit, M3 Ultra, 2026-07-16): **215.75 ms/token = 4.63 tok/s** (3.41 in the gap report —
  #8-B's fused quant tail already moved it). CB census: **368/token = 304 per-projection quant
  seams** (48 gd × 5 [qkv,z,a,b,out] + 16 attn × 4 [q,k,v,o], exact) **+ 64 fused quant tails**
  (#8-B, one per layer). CPU profile over the run: objc round-trip machinery ~1.9 s
  (`runtime.cgocall` — CB submit/wait/copy), quant-seam cum ~1.5 s (`MatMulQuantF32NTInto`,
  incl. per-call f32→bf16 upload converts), host recurrence ~1.1 s (`GatedDeltaRuleF32`),
  alloc/copy churn ~1.7 s (`madvise`/`memmove`/`memclr`/`tensorF32`). No single wall — the
  architecture is the wall, confirming the port shape: device-resident state + one CB per layer
  attacks all four lines at once.
- **S1 — the recurrence kernel.** `lthn_gated_delta_step` in `lthn_kernels`: mlx-lm shape +
  snapshot slots + scale-on-output, scalar-g, f32. Parity vs `deltanet.GatedDeltaRuleF32` (host
  pre-norms q/k so both sides see identical inputs; tolerance-gated — f32 state vs the host's f64
  accumulation) at the real 27B shape + one ragged shape; snapshot slots proven by stepping T=4 and
  matching slot s against the host state s tokens back. Bench: kernel µs at L=1 and L=64 vs the
  host loop (expect ~ms → ~µs).
- **S2 — the device block. DONE.** One CB from qkv/z/a/b to gated output: conv-ring update + SiLU
  + split + ℓ2-norms (one fused kernel) → α/β transform → recurrence → gated RMSNorm·SiLU(z);
  state device-resident across tokens (primed once, exported only for snapshots/clones). Engaged at
  the MIXER level (`GatedDeltaBlockDeviceTry` — the mixer owns state threading; a nil-scratch legacy
  caller can never engage and lose continuity), host block untouched as the parity reference; the
  engine binds via `composed_backend.go` (`LTHN_GD_BLOCK=0` = the A/B off switch). Receipts
  (2026-07-16): block parity vs the host block ~2e-7 scaled across carried sequences (fixture +
  real 27B shapes, L=1 and L=4); export/prime round-trip byte-exact; **27B-4bit decode 215.75 →
  133.46 ms/token = 4.63 → 7.49 tok/s (+62%)**; real generation sane (Asimov prompt, coherent).
  Kernel receipt from S1 stands: 48 layers in one CB = 1.17 ms vs 78.7 ms host (×67).
- **S3 — whole-layer CB. DONE.** One packed gated-delta layer = ONE command buffer: input RMSNorm
  → the five packed projections (affine qmv over the checkpoint codes, cast-once/project-many) →
  the block stages → the #8-B FFN tail — x [L,D] the only upload, y the only readback (the
  per-stage path paid 7 CBs + 6 host crossings). Built from two extractions
  (`encResidualNormMLPQuantTail`, `encGatedDeltaBlockStages` — explicit barriers throughout, the
  #8-B encoder discipline) + `gatedDeltaQuantLayerRun`; engagement at the composed quant branch
  through `GatedDeltaQuantLayerDeviceTry` (same never-fall-back-mid-sequence contract). Receipts
  (2026-07-16): session-level A/B vs the per-stage path **bit-identical (0.000e+00)** over carried
  steps; CB census 368 → **~129**/token (quant projections 304 → 64, tails 64 → 16 — only the 16
  attention layers remain on per-stage seams); **27B-4bit decode 133.46 → 66.42 ms/token = 7.49 →
  15.05 tok/s (+101%; ×3.25 over the campaign baseline)**; generation sane. Full gate green.
- **S4 — prefill T>1.** The same kernel with T=chunk (it already loops); chunk the prompt through
  the conv/norm pre-stages. Receipt: prefill t/s vs mlx-lm on the 27B. Only if the instrument says
  prefill matters after decode lands: omlx Kernel S staging.
- **S5 — MTP block-verify rollback.** Wire the snapshot slots into the composed speculative pair
  (#7's parked `BlockVerify`): verify a K-block, roll the recurrence back to the accept boundary by
  slot copy instead of recompute. Receipt: the pair's tok/s with block-verify ON vs OFF.
- **S6+ (separate tasks)** — whole-token chaining across layers (attention layers included), then
  the serving sophistication (#17 MoE fusion for the 35B; paged/slotted state for batching à la
  vllm-metal).

## Gates

Every slice: `task metallib` fresh, `task test:metal` green, the slice's parity test in-tree
(one test per new symbol, house rules), and the receipt number in the commit message. No slice
ships on synthetic-only evidence: S2 onward must show the number on the real 27B snapshot via
`lem bench`.
