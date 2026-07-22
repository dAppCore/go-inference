<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# HIP production readiness

This file defines the release boundary and repeatable evidence for the HIP++
engine. It is deliberately narrower than the complete capability registry:
production readiness applies to the Gemma 4 text-serving lane, while research
and incomplete model-family surfaces remain accurately marked experimental.

## Production scope

The production lane is native HIP++ text generation on Linux with these cached
GGUF model families:

- `unsloth/gemma-4-E2B-it-GGUF`
- `ggml-org/gemma-4-E4B-it-GGUF`
- `ggml-org/gemma-4-12B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

MLX repositories present in the shared Hugging Face cache are not inputs to
this lane. The loader may synthesize an affine device layout internally, but
that implementation detail does not make the Linux engine or its model
artifacts MLX.

Production readiness covers:

- GGUF load, tokenization, generation, and chat without MTP.
- E2B, E4B, 12B, and sparse 26B-A4B model geometry.
- Device-resident attention and KV, with VRAM-aware host expert residency for
  26B-A4B.
- Retained session state, exact KV capture/restore, prompt continuation, and
  context growth.
- Continuous batching, bounded fallback scheduling, request cancellation,
  probe delivery, and synchronous shutdown.
- One portable HIP++ kernel source compiled for AMD, HIP/CUDA, HIP-CPU x86_64,
  and HIP-CPU AArch64.
- A local release build whose runtime dependencies match the packaged
  sidecars.

The following do not block this release and must remain experimental unless
they independently earn production evidence: MTP, diffusion, training,
multimodal towers, speculative decoding, non-Gemma model families, and
quantization formats that are only declared or fixture-tested.

## Baseline at `fc7b2a06`

Recorded on 2026-07-16 on the RX 7800 XT development host:

- `go test ./engine/hip -count=1`: PASS.
- The six stale failures named in `docs/handover.md`: PASS when run together.
- `go vet ./engine/hip`: PASS.
- `go test -race ./engine/hip -count=1`: FAIL. The fallback scheduler emits
  `queued` from the admitting goroutine while its worker emits `started` into
  the same `ProbeSink`; `TestScheduler_Good_EmitsProbeEvents` races while
  appending to its event slice.
- `GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go vet ./engine/hip`: blocked outside
  HIP because `model/safetensors` does not define `float16SliceToFloat32` on
  Darwin. The exact repository-level issue is tracked in `docs/todo.md`.
- The 26B non-MTP retained-depth curve is 56.68, 56.28, 55.18, 54.24, and
  50.91 tok/s at 2K, 4K, 8K, 12K, and 32K. The detailed receipt remains in
  `GEMMA4_PERFORMANCE.md`.

## Release blockers

- [x] Serialize probe delivery across admission, fallback workers, continuous
  scheduling, and the wrapped model without requiring every consumer sink to
  implement its own locking.
- [x] Make fallback scheduler shutdown cancel work, join its worker, and only
  then close the loaded model. `Close` must not return while generation can
  still touch device resources.
- [x] Add deterministic tests for concurrent probe delivery and synchronous
  shutdown, then make the complete HIP package race-clean.
- [x] Add one unattended production gate that runs the focused host, AMD
  hardware, state, and batching receipts against a caller-selected GGUF.
- [x] Keep sparse continuous batching deterministic beyond the first decode
  step by routing K=1 and K>1 sparse QKV through the same batched projection
  family. Dense models retain their separately verified fused K=1 route.
- [x] Re-run the final AMD, HIP/CUDA, HIP-CPU x86_64, and HIP-CPU AArch64
  compile matrix once after the implementation is stable.
- [ ] Build the release artifacts and verify the dependency guard before
  calling the lane production-ready. This checkout has empty, uninitialized
  `external/rocr-runtime`, `external/rocm-clr`, and `external/rocm-hip`
  submodules, and the host does not provide replacement static ROCr/HIP
  archives. The guard now fails early with this prerequisite instead of an
  opaque CMake error.

## Verified at `52435236`

Recorded on 2026-07-16 on the RX 7800 XT development host:

- `go test ./engine/hip -count=1`: PASS.
- `go test -race ./engine/hip -count=1`: PASS in 20.990 seconds.
- `go vet ./engine/hip`: PASS.
- `make test-hip-production HIP_PRODUCTION_GGUF=<E2B Q4_K_M>`: PASS. The
  required, non-skipped hardware receipts were native package/public decode,
  two-lane continuous batching parity, and exact KV capture/restore
  continuity; the hardware group completed in 63.110 seconds.
- E4B and 12B `TestNativeDecodeSmokeKernelStatus_Good`: PASS in 21.41 and
  19.93 seconds. This found and fixed package retained-state geometry that
  counted `HeadDim` rather than `KeyHeads * HeadDim` for multi-KV-head models.
- 26B-A4B native smoke and the explicitly enabled MoE lane-set receipt: PASS
  in 6.91 and 7.47 seconds.
- `make hip`: PASS for AMD gfx1100 C++23, HIP/CUDA sm_75, HIP-CPU x86_64, and
  HIP-CPU AArch64. CUDA deprecation warnings and GCC HIP-CPU SIMD ABI warnings
  remain warnings only.
- `make test-hip-cpu-runtime test-hip-cpu-kernel-runtime test-zluda-cuda`:
  PASS.
- `make release-dependency-guard`: BLOCKED before compilation because the
  three pinned ROCm source submodules above are not initialized. No fetch,
  pull, submodule update, or push was performed.

## Final AMD production matrix

Recorded on 2026-07-16 on the RX 7800 XT development host after the sparse QKV
lane determinism fix:

- `go test ./engine/hip -count=1`: PASS in 4.292 seconds.
- `go test -race ./engine/hip -count=1`: PASS in 21.365 seconds.
- `go vet ./engine/hip`: PASS.
- Every `test-hip-production` row rebuilt the gfx1100 sidecar, passed the
  scheduler gate, and executed every selected hardware receipt without a skip.

| GGUF | native decode | K=2 lane parity | sparse K=2 parity | exact state | hardware total |
| --- | ---: | ---: | ---: | ---: | ---: |
| E2B Q4_K_M | 21.01s | 21.33s | n/a | 20.87s | 63.521s |
| E4B Q4_K_M | 21.01s | 21.47s | n/a | 20.77s | 63.590s |
| 12B Q4_K_M | 20.38s | 20.02s | n/a | 21.84s | 62.444s |
| 26B-A4B UD-Q4_K_M | 6.95s | 7.64s | 7.51s | 8.86s | 31.116s |

The original 26B receipt only generated two tokens, which exercised one shared
forward and missed the fault. The production receipt now generates four tokens.
It caught K=1 sparse decode using fused triple-QKV while K>1 used three batched
projections; their small rounding delta entered retained KV at layer zero and
changed both streams on the second forward. Sparse layers now force the batched
QKV route for every lane count. The same broad policy changed 12B dense output,
so dense layers deliberately keep their proven fused K=1 route.

## Acceptance commands

Run host checks from the Go module:

```sh
cd go
go test ./engine/hip -count=1
go test -race ./engine/hip -count=1
go vet ./engine/hip
```

Run the focused AMD production receipt from the repository root. It builds the
gfx1100 sidecar, runs the scheduler gate, requires every selected hardware test
to execute without skipping, and accepts an actual GGUF file:

```sh
make test-hip-production HIP_PRODUCTION_GGUF="$GGUF"
```

The 26B sparse batching receipt adds its four-token MoE-specific lane check:

```sh
make test-hip-production HIP_PRODUCTION_GGUF="$GGUF" HIP_PRODUCTION_MOE=1
```

Run portability and release packaging once at the end:

```sh
make hip
make test-hip-cpu-runtime test-hip-cpu-kernel-runtime test-zluda-cuda
make release-dependency-guard
```

Missing optional toolchains may be reported as unavailable, but a toolchain
that is installed on the production host must compile cleanly. AMD hardware
generation and state receipts may not skip on this host.

## Implementation plan

1. [x] Add deterministic scheduler tests that fail on concurrent sink entry and
   model close before worker retirement.
2. [x] Introduce a package-local serialized probe sink and a fallback worker join;
   route the same wrapped sink into the HIP scheduler, shared continuous
   scheduler, and loaded model.
3. [x] Re-run focused scheduler tests and the package race detector, fixing any
   additional engine-owned races it exposes.
4. [x] Add a Makefile production target that validates required model/sidecar
   inputs and runs the exact AMD GGUF serving receipts without broad hidden
   work.
5. [x] Exercise E2B state and continuous batching plus the 26B sparse lane on the
   RX 7800 XT, preserving exact failures here if a blocker remains.
6. [ ] Initialize the pinned ROCm source submodules without changing their
   recorded commits, then run the release dependency guard and commit the
   verified packaging state locally. The portability and runtime portions of
   this step already pass.
