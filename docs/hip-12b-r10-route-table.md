# HIP 12B round 10 decode route table

## Verdict

The proposed device-route miss is not present. In one instrumented real decode
step, both the clean 12B and clean E2B packs routed every observed transformer
operation to a HIP device kernel. There were **no host-fallback entries**, so
there is no operation that is host on 12B but device on E2B and no host
operation that can explain the earlier 923 s / 120-token run.

The 12B 20-token transcript still collapses into repetition, but the current
checkout did not reproduce the old latency: the armed test completed in 1.70 s
(2.13 s including `go test` startup and model load). The dominant measured
device operation was `rocm_mlx_q4_projection` (28.46% of synchronized kernel
wall time), followed by `rocm_mlx_q4_gelu_tanh_multiply` (20.93%). This is a
device-kernel cost distribution, not evidence of a host fallback.

## Method

- Box tree: `/tmp/hip-12b-r10`, cloned from `dev` at `c014998` (later than the
  required `ea73a3a` baseline), with this worktree's instrument files copied in.
- GPU code: `build/kernels/rocm_kernels_gfx1101.hsaco`, supplied through
  `GO_ROCM_KERNEL_HSACO`.
- Models: `/tmp/models/gemma-4-12B-it-4bit-clean` and
  `/tmp/models/gemma-4-E2B-it-4bit-clean`.
- Test: `TestHIPGemma4Q4DecodeKVMode`, `GO_ROCM_GEN_KV_MODE=fp16`, prompt
  `why the sky is blue`, 20 generated tokens.
- Gate: `GO_ROCM_HIP_DECODE_ROUTE_METRICS=1` arms exactly one one-token batched
  decode forward. Normal runs retain a nil atomic fast path. While armed, each
  kernel launch is device-synchronized so its wall time is attributable, then
  recorded with the active layer and layer type.
- Percentages below use the sum of synchronized kernel wall times in that one
  step. They are diagnostic shares, not uninstrumented throughput timings.

## Layer-type route table

| Layer group | E2B route | E2B wall ms | E2B share | 12B route | 12B wall ms | 12B share |
|---|---:|---:|---:|---:|---:|---:|
| Sliding attention | device-kernel | 13.157 | 76.58% | device-kernel | 29.884 | 81.51% |
| Full attention | device-kernel | 3.621 | 21.08% | device-kernel | 6.685 | 18.23% |
| Global / outside layer | device-kernel | 0.403 | 2.35% | device-kernel | 0.094 | 0.26% |
| Host fallback | none | 0 | 0% | none | 0 | 0% |
| **Total** |  | **17.181** | **100%** |  | **36.663** | **100%** |

## Per-operation route table

Every populated cell below is `device-kernel`. A dash means that model did not
invoke that kernel in the measured step; it does not mean host fallback.

| Operation | E2B ms (share) | 12B ms (share) |
|---|---:|---:|
| `rocm_mlx_q4_projection` | 4.231 (24.63%) | 10.436 (28.46%) |
| `rocm_mlx_q4_gelu_tanh_multiply` | 2.155 (12.54%) | 7.675 (20.93%) |
| `rocm_rms_norm_residual_add_norm` | 2.678 (15.59%) | 5.187 (14.15%) |
| `rocm_attention_heads_batch_causal` | 1.140 (6.64%) | 3.554 (9.69%) |
| `rocm_attention_heads_batch_causal_query_rms_rope` | 1.629 (9.48%) | — |
| `rocm_rms_norm_residual_add_mlx_q4_gelu_tanh_projection` | 1.503 (8.75%) | — |
| `rocm_mlx_q4_projection_cols256` | 1.192 (6.94%) | — |
| `rocm_mlx_q4_triple_projection` | 0.644 (3.75%) | 3.204 (8.74%) |
| `rocm_kv_encode_token_value_norm` | 0.505 (2.94%) | 2.213 (6.04%) |
| `rocm_rms_norm_rope_heads_pair` | 0.511 (2.97%) | 1.800 (4.91%) |
| `rocm_kv_descriptor_append` | 0.511 (2.97%) | 1.726 (4.71%) |
| `rocm_mlx_q4_pair_projection` | — | 0.666 (1.82%) |
| `rocm_projection_batch` | 0.152 (0.88%) | — |
| `rocm_embedding_lookup_greedy_token` | 0.101 (0.59%) | 0.094 (0.26%) |
| `rocm_rms_norm_heads` | 0.069 (0.40%) | 0.054 (0.15%) |
| `rocm_rms_norm_residual_add` | 0.043 (0.25%) | 0.054 (0.15%) |
| `rocm_vector_scale` | 0.055 (0.32%) | — |
| `rocm_per_layer_input_transpose` | 0.032 (0.19%) | — |
| `rocm_vector_add_scaled` | 0.030 (0.17%) | — |
| **Host-only on 12B** | **none** | **none** |

The geometry-specific difference is itself device-routed: 12B full-attention
layers use `rocm_mlx_q4_pair_projection`; E2B instead has its per-layer-input
and fused query-RMS/RoPE kernels. Full-attention in both models uses
`rocm_attention_heads_batch_causal` on device.

## 20-token 12B receipt

The route markers fired before this transcript in the same armed test output:

```text
HIP_DECODE_ROUTE_TABLE_BEGIN
layer  layer_type         route          op                              calls  wall_ms
11     full_attention     device-kernel  rocm_mlx_q4_projection          2      0.271
...
HIP_DECODE_ROUTE_TABLE_END

=== DECODE kv=fp16 model=/tmp/models/gemma-4-12B-it-4bit-clean prompt="why the sky is blue" maxTokens=20 ===

thought
thoughtthoughtthoughtthoughtthoughtthoughtthoughtthoughtthoughtthoughtthoughtthoughtthoughtthoughtthoughtthoughtthought
=== end (128 chars) ===
--- PASS: TestHIPGemma4Q4DecodeKVMode (1.70s)
ELAPSED=2.13
```

The corresponding E2B transcript was coherent:

```text
.

This is a simple statement of fact.

It describes the color of the sky.
```

## Reproduction commands

```sh
export PATH=$PATH:/usr/local/go/bin:/usr/lib/go/bin
export GO_ROCM_RUN_HIP_TESTS=1
export GO_ROCM_KERNEL_HSACO=/tmp/hip-12b-r10/build/kernels/rocm_kernels_gfx1101.hsaco
export GO_ROCM_HIP_DECODE_ROUTE_METRICS=1
export GO_ROCM_GEN_MAX_TOKENS=20
export GO_ROCM_GEN_PROMPT='why the sky is blue'
export GO_ROCM_GEN_KV_MODE=fp16
export GO_ROCM_ORACLE_MODEL_PATH=/tmp/models/gemma-4-12B-it-4bit-clean
cd /tmp/hip-12b-r10/go
go test -v -count=1 ./engine/hip -run '^TestHIPGemma4Q4DecodeKVMode$'
```

Repeat with `GO_ROCM_ORACLE_MODEL_PATH=/tmp/models/gemma-4-E2B-it-4bit-clean`
for the contrast table.
