// SPDX-Licence-Identifier: EUPL-1.2

// lthn_steel_attn_256 — MLX's steel flash-attention template instantiated at
// the head dim NOBODY ships (#375). Upstream instantiates steel_attention at
// BD 64/80/128 only, which is why every engine (mlx-lm included) materialises
// S for gemma4's 256/512-dim global attention and pays the #367 cache-poison
// tax. The template itself is BD-parameterised and its threadgroup arithmetic
// FITS at BD=256 with upstream's own bq32/bk16/wm4/wn1 shape:
//   Q_smem 32×(256+8)×2B = 16.9KB + KV_smem 12.3KB = 29.2KB < 32KB.
// So the prompt lane's flash upgrade is an instantiation + host plumbing, not
// a new kernel body — Apple's proven MMA/loader/softmax fragments, our BD.
// (BD=512 blows the 32KB budget at any legal warp shape — the 31B lane needs
// the split-D treatment, phase 2b.)
//
// Function constants resolved host-side (flash_prompt.go): 200 align_Q,
// 201 align_K, 300 has_mask=false, 301 do_causal=true, 302 has_sinks=false.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"

#define instantiate_lthn_attn(tname, dtype, bq, bk, bd, wm, wn, mname, mtype) \
  instantiate_kernel(                                                         \
      "steel_attention_" #tname "_bq" #bq "_bk" #bk "_bd" #bd                 \
      "_wm" #wm "_wn" #wn "_mask" #mname,                                     \
  attention, dtype, bq, bk, bd, wm, wn, mtype, float)

instantiate_lthn_attn(bfloat16, bfloat16_t, 32, 16, 256, 4, 1, bool_, bool);
// clang-format on
