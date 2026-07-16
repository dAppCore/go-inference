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

- [ ] Serialize probe delivery across admission, fallback workers, continuous
  scheduling, and the wrapped model without requiring every consumer sink to
  implement its own locking.
- [ ] Make fallback scheduler shutdown cancel work, join its worker, and only
  then close the loaded model. `Close` must not return while generation can
  still touch device resources.
- [ ] Add deterministic tests for concurrent probe delivery and synchronous
  shutdown, then make the complete HIP package race-clean.
- [ ] Add one unattended production gate that runs the focused host, AMD
  hardware, state, and batching receipts against a caller-selected GGUF.
- [ ] Re-run the final AMD, HIP/CUDA, HIP-CPU x86_64, and HIP-CPU AArch64
  compile matrix once after the implementation is stable.
- [ ] Build the release artifacts and verify the dependency guard before
  calling the lane production-ready.

## Acceptance commands

Run host checks from the Go module:

```sh
cd go
go test ./engine/hip -count=1
go test -race ./engine/hip -count=1
go vet ./engine/hip
```

Run the focused AMD production receipt from the repository root, using the
actual GGUF file and the built gfx1100 sidecar:

```sh
make hip-amd
HIP_VISIBLE_DEVICES=0 \
GO_ROCM_RUN_MODEL_TESTS=1 \
GO_ROCM_MODEL_PATH="$GGUF" \
GO_ROCM_KERNEL_HSACO="$PWD/build/kernels/rocm_kernels_gfx1100.hsaco" \
go -C go test ./engine/hip -count=1 -v \
  -run '^(TestNativeDecodeSmokeKernelStatus_Good|TestHIPGemma4ExactStateContinuityHardware_Good|TestHIPLaneSetE2BHardwareMatchesSingleLanes_Good)$'
```

The 26B sparse batching receipt uses the same command with
`TestHIPLaneSet26BMoEHardwareMatchesSingleLanes_Good` and its GGUF path.

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

1. Add deterministic scheduler tests that fail on concurrent sink entry and
   model close before worker retirement.
2. Introduce a package-local serialized probe sink and a fallback worker join;
   route the same wrapped sink into the HIP scheduler, shared continuous
   scheduler, and loaded model.
3. Re-run focused scheduler tests and the package race detector, fixing any
   additional engine-owned races it exposes.
4. Add a Makefile production target that validates required model/sidecar
   inputs and runs the exact AMD GGUF serving receipts without broad hidden
   work.
5. Exercise E2B state and continuous batching plus the 26B sparse lane on the
   RX 7800 XT, preserving exact failures here if a blocker remains.
6. Run the one final portability matrix, HIP-CPU/ZLUDA runtime smokes, release
   dependency guard, and commit the verified production state locally.
