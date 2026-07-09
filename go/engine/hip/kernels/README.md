<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# go-rocm HIP Kernels

`rocm_kernels.hip` contains the first native kernel source for the launch ABI used by `go/hip_launch.go`.

Build a gfx1100 HSACO on a ROCm machine:

```bash
mkdir -p build
hipcc --std=c++23 --genco --offload-arch=gfx1100 -O2 kernels/rocm_kernels.hip -o build/rocm_kernels_gfx1100.hsaco
GO_ROCM_RUN_HIP_TESTS=1 GO_ROCM_KERNEL_HSACO=$PWD/build/rocm_kernels_gfx1100.hsaco go test ./go -run 'TestHIPHardware.*KernelSource' -count=1 -v
```

The source portability matrix is covered by opt-in tests:

```bash
GO_ROCM_RUN_AMD_HIP_COMPILE_TESTS=1 go test ./go -run '^TestHIPKernelSource_AMDHIPCompile_Good$' -count=1 -v
CUDA_PATH=/usr/local/cuda-12.8 GO_ROCM_RUN_NVIDIA_HIP_COMPILE_TESTS=1 go test ./go -run '^TestHIPKernelSource_NVIDIAHIPCompile_Good$' -count=1 -v
GO_ROCM_RUN_HIP_CPU_COMPILE_TESTS=1 go test ./go -run '^TestHIPKernelSource_HIPCPUCompile_Good$' -count=1 -v
GO_ROCM_RUN_HIP_CPU_RUNTIME_TESTS=1 go test ./go -run '^TestHIPKernelSource_HIPCPURuntimeSmoke_Good$' -count=1 -v
GO_ROCM_RUN_HIP_CPU_KERNEL_RUNTIME_TESTS=1 go test ./go -run '^TestHIPKernelSource_HIPCPUProductionKernelRuntimeSmoke_Good$' -count=1 -v
CUDA_PATH=/usr/local/cuda-12.8 GO_ROCM_RUN_ZLUDA_CUDA_TESTS=1 ROCR_VISIBLE_DEVICES=GPU-880ed6479d653a85 go test ./go -run '^TestHIPKernelSource_ZLUDACUDARuntimeSmoke_Good$' -count=1 -v
```

The compile tests use `ccache` when it is available in `PATH`: direct
C++/CUDA compiler checks launch through `ccache`, and HIP driver checks prepend
`/usr/lib/ccache` so subprocess compiler calls can hit the cache. Set
`GO_ROCM_USE_CCACHE=0` to force direct compiler execution, or
`GO_ROCM_CCACHE=/path/to/ccache` to pin a specific launcher.

HIP-CPU is discovered through `GO_ROCM_HIP_CPU_INCLUDE`,
`GO_ROCM_HIP_CPU_ROOT`, or `/opt/hip-cpu/include`. The CPU compile test defaults
to `x86_64,aarch64`; set `GO_ROCM_HIP_CPU_TARGETS=x86_64` for host-only checks.
The production-kernel runtime smoke compiles `rocm_kernels.hip` into a HIP-CPU
host binary and launches `rocm_embedding_mean_pool` on the CPU.

The HIP source is built as C++23. The Go cgo bridge uses `dappco.re/go/cgo`
and `core.PinnedView` for retained Go-owned buffers; direct HIP use of the
`go-cgo` `cgo_pinned_view.hpp` mdspan companion requires a ROCm host toolchain
that provides `<mdspan>`.

The exported symbols must stay in sync with the Go launcher names:

- `rocm_prefill`
- `rocm_decode`
- `rocm_kv_encode_token`
- `rocm_kv_descriptor_append`
- `rocm_projection`
- `rocm_mlx_q4_projection`
- `rocm_mlx_q4_projection_batch`
- `rocm_mlx_q4_projection_greedy`
- `rocm_mlx_q4_triple_projection`
- `rocm_mlx_q4_pair_projection`
- `rocm_mlx_q4_gelu_tanh_multiply`
- `rocm_mlx_q4_gelu_tanh_multiply_batch`
- `rocm_mlx_q4_gelu_tanh_projection`
- `rocm_mlx_q4_gelu_tanh_projection_batch`
- `rocm_rms_norm`
- `rocm_rms_norm_residual_add`
- `rocm_rms_norm_residual_add_norm`
- `rocm_rms_norm_heads`
- `rocm_rms_norm_rope_heads`
- `rocm_rms_norm_rope_heads_batch`
- `rocm_rope`
- `rocm_rope_heads`
- `rocm_greedy_sample`
- `rocm_softcap_greedy_sample`
- `rocm_attention`
- `rocm_attention_heads`
- `rocm_attention_heads_batch_causal`
- `rocm_vector_add`
- `rocm_vector_scale`
- `rocm_swiglu`
- `rocm_gelu_tanh_multiply`
- `rocm_moe_router`
- `rocm_moe_lazy_experts`
- `rocm_jangtq_projection`
- `rocm_codebook_lookup`
- `rocm_lora_projection`
- `rocm_embedding_lookup`
- `rocm_embedding_mean_pool`
- `rocm_rerank_cosine`
- `rocm_tiny_prefill`
- `rocm_tiny_decode`
- `rocm_cross_entropy_loss`
- `rocm_distillation_kl_loss`
- `rocm_grpo_advantage`

The prefill and decode kernels currently validate and consume their launch packets, referenced device memory, and optional status-output pointers in the reserved packet fields; the hardware smoke covers fp16, q8, and k-q8-v-q4 cache-mode descriptors. The KV encode and descriptor append kernels support the loaded-model device KV cache path. The projection kernels perform the toy fp16/q8/BF16 row projections, MLX affine 4/6/8-bit packed row projection, batched MLX affine prompt-row projection, fused MLX affine greedy projection, batched MLX affine GELU-tanh projection for Gemma4 per-layer inputs, JANGTQ projection, codebook lookup, and LoRA projection used by the Go fake-driver fixtures and loaded-model projection smoke. `rocm_embedding_lookup` supports f32, BF16, and MLX affine 4/6/8-bit embedding tables, including loaded Gemma4 packed U32 weights with BF16 scales/biases. The RMSNorm, batched Q/K RMSNorm+RoPE, RoPE, greedy sampler, softcap greedy sampler, single-head attention, multi-head q attention, batched causal prefill attention, vector-add, vector-scale, SwiGLU, GELU-tanh multiply, batched MLX affine GELU-tanh multiply, MoE, training-loss, and GRPO kernels execute deterministic primitive fixtures. `rocm_tiny_prefill` is a toy embedding-attention-output fixture that writes toy KV buffers, logits, final-token attention weights, and a greedy result buffer. `rocm_tiny_decode` consumes those toy prior KV vectors, appends the decoded token embedding, and writes updated KV, logits, attention, and greedy result buffers. The tiny kernels accept fp32, fp16, or q8 output-head weights. These tiny kernels are not yet the production loaded-model generation path.
