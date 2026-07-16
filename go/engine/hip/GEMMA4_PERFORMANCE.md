# Gemma4 HIP performance baseline

This file is the durable performance board for the RX 7800 XT development
host. Do not rerun an unchanged row merely to rediscover its baseline. Update a
row only when the engine, model artifact, runtime, or benchmark geometry has
changed, and record the new revision and date.

All results below are non-MTP. `HIP_VISIBLE_DEVICES=0` excludes the integrated
`AMD Radeon Graphics` device and its shared-memory aperture; only the 16 GB
discrete card is measured.

## Host and revisions

| item | value |
|---|---|
| date | 2026-07-16 |
| GPU | AMD Radeon RX 7800 XT, gfx1100, 16,368 MiB reported VRAM |
| CPU / RAM | AMD Ryzen 9 9950X / 128 GB |
| ROCm | 7.2.26015 |
| go-inference | `50372990b5dde3b82612c4e92208d5d24023f90b` |
| llama.cpp oracle | `c7d8722922a2599dc4d77f8808d8e6c2fde5e7a2` |

## Short-context board

HIP uses a two-token prompt, context 1024, and one `tg512` measured run. Model
load is outside the timer. llama.cpp uses its warmup plus three measured
`pp512` and `tg512` runs, flash attention, and full layer offload to `ROCm0`.
The same-GGUF columns use the exact file in both engines. The native HIP pack
column records the fastest already-supported Q4 representation and is not a
numerical parity comparison against Q4_K_M.

| target | HIP native Q4 tg512 | HIP same-GGUF tg512 | llama.cpp GGUF pp512 | llama.cpp GGUF tg512 | same-GGUF delta | decision |
|---|---:|---:|---:|---:|---:|---|
| E2B | **159.5** | 138.7 | 5,611.78 | 150.25 | -7.7% | native lane frozen above 100 tok/s |
| E4B | 79.88 | **82.17** | 3,202.18 | 94.61 | -13.2% | active short-context gap |
| 12B | **53.74** | 36.93 | 1,425.00 | 50.20 | -26.4% | native floor met; Q4_K path remains active |
| 26B-A4B | n/a | **56.80** | 806.55 | 62.25 | -8.8% | 50 tok/s floor met; deep-context row remains active |

Native HIP artifacts are `mlx-community/gemma-4-e2b-it-4bit`,
`lmstudio-community/gemma-4-E4B-it-MLX-4bit`, and
`mlx-community/gemma-4-12b-it-4bit`. The standard 26B receipt uses GGUF host
experts with attention and KV on the discrete card.

The llama.cpp files are pinned by Hugging Face snapshot:

| target | repository and snapshot | file size |
|---|---|---:|
| E2B | `unsloth/gemma-4-E2B-it-GGUF@739965d7`, `Q4_K_M` | 3,090,917,516 bytes |
| E4B | `ggml-org/gemma-4-E4B-it-GGUF@2714b551`, `Q4_K_M` | 5,319,465,128 bytes |
| 12B | `ggml-org/gemma-4-12B-it-GGUF@44ee90c4`, `Q4_K_M` | 7,365,558,464 bytes |
| 26B-A4B | `unsloth/gemma-4-26B-A4B-it-GGUF@c462057f`, `UD-Q4_K_M` | 16,931,716,216 bytes |

For 26B, llama.cpp's `--fit-target 512` kept all 31 layers assigned to the
GPU, used a 15,430.53 MiB ROCm model buffer, and placed selected layer 28/29
expert tensors in `ROCm_Host`. KV and attention stayed on the GPU. This is the
relevant architecture oracle for HIP host-offload work.

Before `845e4354`, HIP retained each raw GGUF tensor after uploading its
canonical affine replacement. The duplicated 12B allocations exhausted VRAM
on the first KV page pair. HIP now releases each source after successful
synthesis; the same 12B GGUF completes `tg512` instead of returning
`rocm.hip.hipMalloc: HIP returned 2`.

## Context growth

The established E4B and 12B HIP receipts use a prompt of `context - 512`
tokens followed by `tg512`. They are not the same geometry as the
short-context llama.cpp rows above.

| target | 2K | 4K | 8K | 12K | 32K | 2K to 32K loss |
|---|---:|---:|---:|---:|---:|---:|
| E4B | 74.72 | 73.11 | 70.28 | 68.24 | 61.11 | 18.2% |
| 12B | 47.307 | 46.325 | 44.607 | 43.121 | 38.046 | 19.6% |

E4B's 32K receipt and 12B's 8K/32K receipts also passed exact state
continuation. Preserve that state/continuity check whenever a context row is
updated.

### 26B-A4B retained-depth decode

These rows separate decode from cold prefill. llama.cpp prefills its depth
outside the `tg512` timer. HIP materialises the same `context - 512` retained
depth through an engine session, then times only the following 512 non-MTP
tokens. Both engines use the same `UD-Q4_K_M` GGUF. llama.cpp holds one
32K-capable placement plan across the sweep with `--fit-target 512 --fit-ctx
32768`.

| engine / revision | 2K | 4K | 8K | 12K | 32K | 2K to 32K loss |
|---|---:|---:|---:|---:|---:|---:|
| llama.cpp `c7d8722` | 59.58 | 56.74 | 56.05 | 55.56 | 51.17 | 14.1% |
| HIP before wide incremental GQA | 56.14 | 54.48 | 50.93 | 47.86 | not run | n/a |
| HIP `50372990` | **56.67** | **56.06** | **54.71** | **53.66** | **49.37** | **12.9%** |

`50372990` connects the already-validated grouped batch-attention path to
single-token decode for 16-head GQA lanes. At 32K, the existing GQA8 kernel
measures 0.594 ms per 26B global-attention call versus 1.916 ms for the generic
per-head scan, a 3.23x kernel improvement. Whole-model gains grow with depth:
0.9% at 2K, 2.9% at 4K, 7.4% at 8K, and 12.1% at 12K. The 4K, 8K, 12K, and
32K receipts each used 2,560 GQA8 launches and zero old per-head launches.

The 32K result is 3.5% behind llama.cpp and misses the 50 tok/s requirement by
0.63 tok/s, so it remains active work. Its expert cache retained 1,814 entries
and 7,005,408,256 bytes; the timed decode saw 530 misses, 105 evictions, and
1,971,932,160 host-to-device bytes. The complete 32K test took 563.5 seconds,
including model load, cold state materialisation, `tg512`, and teardown. Do not
present that wall time as decode latency or as an isolated prefill measurement.

## DiffusionGemma diagnostic

The cached `mlx-community/diffusiongemma-26B-A4B-it-4bit` model is block
diffusion, not the standard autoregressive 26B-A4B model used by the llama.cpp
row. Its current production `tg512` result is about 17.50 tok/s. A cold `tg64`
diagnostic is 8.32 tok/s because the timed run also populates an empty expert
cache with 10,216,876,032 bytes of host-to-device traffic. Neither number is a
like-for-like replacement for the standard autoregressive 26B HIP row.

## Reproduction

HIP short-context row, from the `go` module:

```sh
HIP_VISIBLE_DEVICES=0 \
GO_ROCM_RUN_BENCHMARKS=1 \
GO_ROCM_PRODUCTION_MODEL_PATH="$MODEL" \
GO_ROCM_KERNEL_HSACO="$HSACO" \
GO_ROCM_BENCH_CONTEXT_LEN=1024 \
GO_ROCM_BENCH_TOKENS=512 \
GO_ROCM_BENCH_PROMPT_TOKEN_COUNT=2 \
go test ./engine/hip -run '^$' \
  -bench '^BenchmarkInferenceGemma4Q4Generate$' -benchtime=1x -count=1
```

llama.cpp card-oracle row:

```sh
HIP_VISIBLE_DEVICES=0 llama-bench -m "$GGUF" \
  -p 512 -n 512 -r 3 -ngl 999 -sm none -dev ROCm0 \
  -fa on -t 16 -o json
```

Add `-fitt 512 -fitc 1024` for the 26B-A4B row.

llama.cpp retained-depth 26B sweep:

```sh
HIP_VISIBLE_DEVICES=0 llama-bench -m "$GGUF" \
  -p 0 -n 512 -d 1536,3584,7680,11776,32256 \
  -r 3 -ngl 999 -sm none -dev ROCm0 -fa on -t 16 \
  -fitt 512 -fitc 32768 -o json
```

HIP retained-depth row, where `$DEPTH` is one less than the llama.cpp depth so
the untimed materialisation token lands on the same boundary:

```sh
HIP_VISIBLE_DEVICES=0 \
GO_ROCM_RUN_RETAINED_DEPTH_BENCHMARK=1 \
GO_ROCM_PRODUCTION_MODEL_PATH="$GGUF" \
GO_ROCM_KERNEL_HSACO="$HSACO" \
GO_ROCM_RETAINED_DEPTH_TOKENS="$DEPTH" \
GO_ROCM_RETAINED_DEPTH_DECODE_TOKENS=512 \
GO_ROCM_RETAINED_DEPTH_CONTEXT_LEN="$CONTEXT" \
go test ./engine/hip -run '^$' \
  -bench '^BenchmarkInferenceGemma4Q4RetainedDepthDecode$' \
  -benchtime=1x -count=1
```
