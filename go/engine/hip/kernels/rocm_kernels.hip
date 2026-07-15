// SPDX-Licence-Identifier: EUPL-1.2

#include <hip/hip_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr uint32_t ROCM_PREFILL_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_PREFILL_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_PREFILL_LAUNCH_STATUS_OK = 0x5052464cu;
constexpr uint32_t ROCM_DECODE_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_DECODE_LAUNCH_ARGS_HEADER_BYTES = 32;
constexpr uint32_t ROCM_DECODE_LAUNCH_ARGS_BYTES = 96;
constexpr uint32_t ROCM_DEVICE_KV_LAUNCH_DESCRIPTOR_BYTES = 64;
constexpr uint32_t ROCM_DEVICE_KV_DESCRIPTOR_VERSION = 1;
constexpr uint32_t ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES = 32;
constexpr uint32_t ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES = 64;
constexpr uint32_t ROCM_DEVICE_KV_DESCRIPTOR_MODE_KQ8VQ4 = 3;
constexpr uint32_t ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_FP16 = 1;
constexpr uint32_t ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8 = 2;
constexpr uint32_t ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4 = 3;
constexpr uint32_t ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS = 4;
constexpr uint32_t ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4_ROWS = 5;
constexpr uint32_t ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS_INTERLEAVED = 6;
constexpr uint32_t ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4_ROWS_INTERLEAVED = 7;
constexpr uint32_t ROCM_KV_ENCODE_TOKEN_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_KV_ENCODE_TOKEN_LAUNCH_ARGS_BYTES = 96;
constexpr uint32_t ROCM_KV_ENCODE_TOKEN_VALUE_NORM_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_KV_ENCODE_TOKEN_VALUE_NORM_LAUNCH_ARGS_BYTES = 112;
constexpr uint32_t ROCM_KV_ENCODE_TOKEN_VALUE_NORM_DESCRIPTOR_APPEND_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_KV_ENCODE_TOKEN_VALUE_NORM_DESCRIPTOR_APPEND_LAUNCH_ARGS_BYTES = 248;
constexpr uint32_t ROCM_KV_ENCODE_TOKEN_VALUE_NORM_MAX_HEADS = 64;
constexpr uint32_t ROCM_KV_ENCODE_TOKEN_BLOCK_SIZE = 256;
constexpr uint32_t ROCM_KV_DESCRIPTOR_APPEND_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_KV_DESCRIPTOR_APPEND_LAUNCH_ARGS_BYTES = 128;
constexpr uint32_t ROCM_KV_DESCRIPTOR_APPEND_BLOCK_SIZE = 64;
constexpr uint64_t ROCM_KV_DESCRIPTOR_APPEND_MODE_GROW_LAST_PAGE = 1;
constexpr uint64_t ROCM_KV_DESCRIPTOR_APPEND_MODE_BUILD_SINGLE_PAGE = 2;
constexpr uint32_t ROCM_DECODE_LAUNCH_STATUS_OK = 0x4445434fu;
constexpr uint32_t ROCM_PROJECTION_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_PROJECTION_LAUNCH_ARGS_BYTES = 96;
constexpr uint32_t ROCM_PROJECTION_BATCH_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_PROJECTION_BATCH_LAUNCH_ARGS_BYTES = 104;
constexpr uint32_t ROCM_PROJECTION_WEIGHT_ENCODING_FP16 = 1;
constexpr uint32_t ROCM_PROJECTION_WEIGHT_ENCODING_Q8 = 2;
constexpr uint32_t ROCM_PROJECTION_WEIGHT_ENCODING_F32 = 3;
constexpr uint32_t ROCM_PROJECTION_WEIGHT_ENCODING_BF16 = 4;
constexpr uint32_t ROCM_PROJECTION_LAUNCH_FLAG_BIAS = 1;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_LAUNCH_ARGS_BYTES = 96;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_BATCH_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_BATCH_LAUNCH_ARGS_BYTES = 96;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_GREEDY_BATCH_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_GREEDY_BATCH_LAUNCH_ARGS_BYTES = 104;
constexpr uint32_t ROCM_MLX_Q4_TRIPLE_PROJECTION_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_MLX_Q4_TRIPLE_PROJECTION_LAUNCH_ARGS_BYTES = 168;
constexpr uint32_t ROCM_MLX_Q4_GELU_TANH_MUL_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_MLX_Q4_GELU_TANH_MUL_LAUNCH_ARGS_BYTES = 128;
constexpr uint32_t ROCM_MLX_Q4_GELU_TANH_MUL_BATCH_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_MLX_Q4_GELU_TANH_MUL_BATCH_LAUNCH_ARGS_BYTES = 128;
constexpr uint32_t ROCM_MLX_Q4_GELU_TANH_MLP_PERSISTENT_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_MLX_Q4_GELU_TANH_MLP_PERSISTENT_LAUNCH_ARGS_BYTES = 184;
constexpr uint32_t ROCM_MLX_Q4_GELU_TANH_PROJ_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_MLX_Q4_GELU_TANH_PROJ_LAUNCH_ARGS_BYTES = 96;
constexpr uint32_t ROCM_MLX_Q4_GELU_TANH_PROJ_BATCH_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_MLX_Q4_GELU_TANH_PROJ_BATCH_LAUNCH_ARGS_BYTES = 104;
constexpr uint32_t ROCM_RMS_NORM_RESIDUAL_ADD_GELU_TANH_PROJ_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_RMS_NORM_RESIDUAL_ADD_GELU_TANH_PROJ_LAUNCH_ARGS_BYTES = 160;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_BITS = 4;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE = 256;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_ROWS_PER_BLOCK = 8;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW = ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE / ROCM_MLX_Q4_PROJECTION_ROWS_PER_BLOCK;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_ROW16_ROWS_PER_BLOCK = 16;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_ROW16_THREADS_PER_ROW = ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE / ROCM_MLX_Q4_PROJECTION_ROW16_ROWS_PER_BLOCK;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_12B_DOWN_ROWS = 3840;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_12B_DOWN_COLS = 15360;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_12B_DOWN_GROUP_SIZE = 32;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_12B_DOWN_GROUPS_PER_ROW = ROCM_MLX_Q4_PROJECTION_12B_DOWN_COLS / ROCM_MLX_Q4_PROJECTION_12B_DOWN_GROUP_SIZE;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_12B_DOWN_PACKED_PER_ROW = ROCM_MLX_Q4_PROJECTION_12B_DOWN_COLS / 8;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_COLS256_ROWS_PER_BLOCK = 32;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_COLS256_THREADS_PER_ROW = ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE / ROCM_MLX_Q4_PROJECTION_COLS256_ROWS_PER_BLOCK;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_Q6_ROW16_ROWS_PER_BLOCK = 16;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_Q6_ROW16_THREADS_PER_ROW = ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE / ROCM_MLX_Q4_PROJECTION_Q6_ROW16_ROWS_PER_BLOCK;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_Q6_ROW32_ROWS_PER_BLOCK = 32;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_Q6_ROW32_THREADS_PER_ROW = ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE / ROCM_MLX_Q4_PROJECTION_Q6_ROW32_ROWS_PER_BLOCK;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_Q6_ROW64_ROWS_PER_BLOCK = 64;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_Q6_ROW64_THREADS_PER_ROW = ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE / ROCM_MLX_Q4_PROJECTION_Q6_ROW64_ROWS_PER_BLOCK;
constexpr uint32_t ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_ROWS_PER_BLOCK = 16;
constexpr uint32_t ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_THREADS_PER_ROW = ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE / ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_ROWS_PER_BLOCK;
constexpr uint32_t ROCM_MLX_Q4_GELU_TANH_Q4_G32_COLS1536_ROW16_ROWS_PER_BLOCK = 16;
constexpr uint32_t ROCM_MLX_Q4_GELU_TANH_Q4_G32_COLS1536_ROW16_THREADS_PER_ROW = ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE / ROCM_MLX_Q4_GELU_TANH_Q4_G32_COLS1536_ROW16_ROWS_PER_BLOCK;
constexpr uint32_t ROCM_MLX_Q4_GELU_TANH_12B_ROWS = 15360;
constexpr uint32_t ROCM_MLX_Q4_GELU_TANH_12B_COLS = 3840;
constexpr uint32_t ROCM_MLX_Q4_GELU_TANH_12B_GROUP_SIZE = 32;
constexpr uint32_t ROCM_MLX_Q4_GELU_TANH_12B_GROUPS_PER_ROW = ROCM_MLX_Q4_GELU_TANH_12B_COLS / ROCM_MLX_Q4_GELU_TANH_12B_GROUP_SIZE;
constexpr uint32_t ROCM_MLX_Q4_GELU_TANH_12B_PACKED_PER_ROW = ROCM_MLX_Q4_GELU_TANH_12B_COLS / 8;
constexpr uint32_t ROCM_MLX_Q4_GELU_TANH_12B_ROW8_ROWS_PER_BLOCK = 8;
constexpr uint32_t ROCM_MLX_Q4_GELU_TANH_12B_ROW8_THREADS_PER_ROW = ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE / ROCM_MLX_Q4_GELU_TANH_12B_ROW8_ROWS_PER_BLOCK;
constexpr uint32_t ROCM_MLX_Q4_GELU_TANH_MLP_PERSISTENT_BLOCKS = 120;
constexpr uint32_t ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_ROW32_ROWS_PER_BLOCK = 32;
constexpr uint32_t ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_ROW32_THREADS_PER_ROW = ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE / ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_ROW32_ROWS_PER_BLOCK;
constexpr uint32_t ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_ROW64_ROWS_PER_BLOCK = 64;
constexpr uint32_t ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_ROW64_THREADS_PER_ROW = ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE / ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_ROW64_ROWS_PER_BLOCK;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK = 8;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_BATCH_WIDE_TOKENS_PER_BLOCK = 16;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_GREEDY_ROWS_PER_BLOCK = 32;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_GREEDY_THREADS_PER_ROW = ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE / ROCM_MLX_Q4_PROJECTION_GREEDY_ROWS_PER_BLOCK;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_ROWS_PER_BLOCK = 64;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_THREADS_PER_ROW = ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE / ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_ROWS_PER_BLOCK;
constexpr uint32_t ROCM_MLX_Q4_PROJECTION_BEST_BYTES = 8;
constexpr uint32_t ROCM_PACKED_TOPK_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_PACKED_TOPK_LAUNCH_ARGS_BYTES = 48;
constexpr uint32_t ROCM_PACKED_TOPK_SAMPLE_LAUNCH_ARGS_VERSION = 2;
constexpr uint32_t ROCM_PACKED_TOPK_SAMPLE_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_ORDERED_EMBEDDING_CANDIDATES_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_ORDERED_EMBEDDING_CANDIDATES_LAUNCH_ARGS_BYTES = 80;
constexpr uint32_t ROCM_ORDERED_EMBEDDING_CANDIDATES_BLOCK_SIZE = 256;
constexpr uint32_t ROCM_PACKED_TOPK_MAX_K = 128;
constexpr uint32_t ROCM_PACKED_TOPK_BLOCK_SIZE = 256;
constexpr uint32_t ROCM_PACKED_TOPK_CHUNK_SIZE = 4096;
constexpr uint32_t ROCM_RMS_NORM_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_RMS_NORM_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_RMS_NORM_RESIDUAL_ADD_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_RMS_NORM_RESIDUAL_ADD_LAUNCH_ARGS_BYTES = 80;
constexpr uint32_t ROCM_RMS_NORM_RESIDUAL_ADD_NORM_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_RMS_NORM_RESIDUAL_ADD_NORM_LAUNCH_ARGS_BYTES = 128;
constexpr uint32_t ROCM_RMS_NORM_HEADS_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_RMS_NORM_HEADS_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_RMS_NORM_ROPE_HEADS_LAUNCH_ARGS_VERSION = 2;
constexpr uint32_t ROCM_RMS_NORM_ROPE_HEADS_LAUNCH_ARGS_BYTES = 88;
constexpr uint32_t ROCM_RMS_NORM_ROPE_HEADS_PAIR_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_RMS_NORM_ROPE_HEADS_PAIR_LAUNCH_ARGS_BYTES = 144;
constexpr uint32_t ROCM_RMS_NORM_ROPE_HEADS_BATCH_LAUNCH_ARGS_VERSION = 2;
constexpr uint32_t ROCM_RMS_NORM_ROPE_HEADS_BATCH_LAUNCH_ARGS_BYTES = 96;
constexpr uint32_t ROCM_RMS_NORM_ROPE_HEADS_PAIR_LANE_BATCH_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_RMS_NORM_ROPE_HEADS_PAIR_LANE_BATCH_LAUNCH_ARGS_BYTES = 152;
constexpr uint32_t ROCM_MOE_COMBINE_NORMS_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_MOE_COMBINE_NORMS_LAUNCH_ARGS_BYTES = 96;
constexpr uint32_t ROCM_RMS_NORM_WEIGHT_ENCODING_NONE = 0;
constexpr uint32_t ROCM_RMS_NORM_WEIGHT_ENCODING_F32 = 1;
constexpr uint32_t ROCM_RMS_NORM_WEIGHT_ENCODING_BF16 = 2;
constexpr uint32_t ROCM_RMS_NORM_LAUNCH_FLAG_ADD_UNIT_WEIGHT = 1;
constexpr uint32_t ROCM_RMS_NORM_LAUNCH_FLAG_ROPE_NEOX = 2;
constexpr uint32_t ROCM_RMS_NORM_LAUNCH_FLAG_MASK = ROCM_RMS_NORM_LAUNCH_FLAG_ADD_UNIT_WEIGHT | ROCM_RMS_NORM_LAUNCH_FLAG_ROPE_NEOX;
constexpr uint32_t ROCM_ROPE_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_ROPE_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_ROPE_HEADS_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_ROPE_HEADS_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_GREEDY_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_GREEDY_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_SOFTCAP_GREEDY_LAUNCH_ARGS_VERSION = 2;
constexpr uint32_t ROCM_SOFTCAP_GREEDY_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_GREEDY_RESULT_BYTES = 8;
constexpr uint32_t ROCM_ATTENTION_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_ATTENTION_LAUNCH_ARGS_BYTES = 104;
constexpr uint32_t ROCM_ATTENTION_HEADS_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_ATTENTION_HEADS_LAUNCH_ARGS_BYTES = 128;
constexpr uint32_t ROCM_ATTENTION_HEADS_BATCH_CAUSAL_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_ATTENTION_HEADS_BATCH_CAUSAL_LAUNCH_ARGS_BYTES = 144;
constexpr uint32_t ROCM_ATTENTION_HEADS_LANE_BATCH_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_ATTENTION_HEADS_LANE_BATCH_LAUNCH_ARGS_BYTES = 112;
constexpr uint32_t ROCM_ATTENTION_HEADS_LANE_DESCRIPTOR_BYTES = 32;
constexpr uint32_t ROCM_ATTENTION_HEADS_BATCH_CAUSAL_QUERY_RMS_ROPE_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_ATTENTION_HEADS_BATCH_CAUSAL_QUERY_RMS_ROPE_LAUNCH_ARGS_BYTES = 184;
constexpr uint32_t ROCM_ATTENTION_HEADS_SHARED_MAX_TOKENS = 2048;
constexpr uint32_t ROCM_ATTENTION_HEADS_CHUNKED_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_ATTENTION_HEADS_CHUNKED_LAUNCH_ARGS_BYTES = 128;
constexpr uint32_t ROCM_ATTENTION_HEADS_BATCH_CHUNKED_LAUNCH_ARGS_VERSION = 2;
constexpr uint32_t ROCM_ATTENTION_HEADS_BATCH_CHUNKED_LAUNCH_ARGS_BYTES = 136;
constexpr uint32_t ROCM_ATTENTION_HEADS_CHUNKED_BLOCK_SIZE = 512;
constexpr uint32_t ROCM_ATTENTION_HEADS_CHUNK_SIZE = 64;
constexpr uint32_t ROCM_ATTENTION_KV_SOURCE_CONTIGUOUS = 0;
constexpr uint32_t ROCM_ATTENTION_KV_SOURCE_DEVICE = 1;
constexpr uint32_t ROCM_VECTOR_ADD_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_VECTOR_ADD_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_VECTOR_ADD_SCALED_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_VECTOR_ADD_SCALED_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_VECTOR_SCALE_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_VECTOR_SCALE_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_PER_LAYER_INPUT_TRANSPOSE_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_PER_LAYER_INPUT_TRANSPOSE_LAUNCH_ARGS_BYTES = 56;
constexpr uint32_t ROCM_SWIGLU_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_SWIGLU_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_GELU_TANH_MUL_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_GELU_TANH_MUL_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_MOE_ROUTER_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_MOE_ROUTER_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_MOE_ROUTER_LAUNCH_STATUS_OK = 0x4d4f4552u;
constexpr uint32_t ROCM_MOE_LAZY_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_MOE_LAZY_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_MOE_BATCH_ROUTE_ROWS_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_MOE_BATCH_ROUTE_ROWS_LAUNCH_ARGS_BYTES = 72;
constexpr uint32_t ROCM_MOE_BATCH_REDUCE_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_MOE_BATCH_REDUCE_LAUNCH_ARGS_BYTES = 56;
constexpr uint32_t ROCM_MOE_BATCH_ROUTE_METADATA_BYTES = 16;
constexpr uint32_t ROCM_MOE_BATCH_ROUTE_BLOCK_SIZE = 256;
constexpr uint32_t ROCM_MOE_MLX_AFFINE_ROUTES_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_MOE_MLX_AFFINE_ROUTES_LAUNCH_ARGS_BYTES = 80;
constexpr uint32_t ROCM_MOE_MLX_AFFINE_ROUTES_CHUNK_BYTES = 152;
constexpr uint32_t ROCM_MOE_MLX_AFFINE_ROUTES_PER_CHUNK = 8;
constexpr uint32_t ROCM_MOE_MLX_AFFINE_ROUTES_FLAG_GATE_UP = 1;
constexpr uint32_t ROCM_GGUF_Q4_0_PROJECTION_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_GGUF_Q4_0_PROJECTION_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_GGUF_Q4_K_EXPAND_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_GGUF_Q4_K_EXPAND_LAUNCH_ARGS_BYTES = 48;
constexpr uint32_t ROCM_GGUF_Q4_0_PROJECTION_BLOCK_SIZE = 256;
constexpr uint32_t ROCM_GGUF_Q4_0_PROJECTION_ROWS_PER_BLOCK = 8;
constexpr uint32_t ROCM_GGUF_Q4_0_PROJECTION_THREADS_PER_ROW = ROCM_GGUF_Q4_0_PROJECTION_BLOCK_SIZE / ROCM_GGUF_Q4_0_PROJECTION_ROWS_PER_BLOCK;
constexpr uint32_t ROCM_GGUF_Q4_0_SELECTED_EXPERT_PAIR16_ROWS_PER_BLOCK = 16;
constexpr uint32_t ROCM_GGUF_Q4_0_SELECTED_EXPERT_PAIR16_THREADS_PER_ROW = ROCM_GGUF_Q4_0_PROJECTION_BLOCK_SIZE / ROCM_GGUF_Q4_0_SELECTED_EXPERT_PAIR16_ROWS_PER_BLOCK;
constexpr uint32_t ROCM_GGUF_Q4_K_SELECTED_EXPERT_GATE_UP_SPLIT_ROWS_PER_BLOCK = 8;
constexpr uint32_t ROCM_GGUF_Q4_K_EXPANDED_BLOCK_BYTES = 152;
constexpr uint32_t ROCM_GGUF_Q5_1_SELECTED_EXPERT_DOWN_EXPERT8_ROWS_PER_BLOCK = 2;
constexpr uint32_t ROCM_GGUF_Q5_1_SELECTED_EXPERT_DOWN_EXPERT8_EXPERTS = 8;
constexpr uint32_t ROCM_GGUF_Q4_0_SELECTED_EXPERTS_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_GGUF_Q4_0_SELECTED_EXPERTS_LAUNCH_ARGS_BYTES = 240;
constexpr uint32_t ROCM_GGUF_Q4_0_SELECTED_EXPERTS_MAX_TOP_K = 8;
constexpr uint32_t ROCM_GGUF_EXPERT_FORMAT_Q4_0 = 1;
constexpr uint32_t ROCM_GGUF_EXPERT_FORMAT_Q4_K = 2;
constexpr uint32_t ROCM_GGUF_EXPERT_FORMAT_Q5_1 = 3;
constexpr uint32_t ROCM_GGUF_EXPERT_FORMAT_Q8_0 = 4;
constexpr uint32_t ROCM_GGUF_EXPERT_FORMAT_Q4_K_EXPANDED = 5;
constexpr uint32_t ROCM_JANGTQ_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_JANGTQ_LAUNCH_ARGS_BYTES = 96;
constexpr uint32_t ROCM_JANGTQ_LAUNCH_FLAG_BIAS = 1;
constexpr uint32_t ROCM_CODEBOOK_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_CODEBOOK_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_LORA_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_LORA_LAUNCH_ARGS_BYTES = 128;
constexpr uint32_t ROCM_LORA_LAUNCH_FLAG_BIAS = 1;
constexpr uint32_t ROCM_EMBEDDING_LOOKUP_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_EMBEDDING_LOOKUP_LAUNCH_ARGS_BYTES = 104;
constexpr uint32_t ROCM_DIFFUSION_EXPECTED_EMBEDDING_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_DIFFUSION_EXPECTED_EMBEDDING_LAUNCH_ARGS_BYTES = 120;
constexpr uint32_t ROCM_DIFFUSION_SAMPLE_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_DIFFUSION_SAMPLE_LAUNCH_ARGS_BYTES = 72;
constexpr uint32_t ROCM_DIFFUSION_SAMPLE_RESULT_BYTES = 16;
constexpr uint32_t ROCM_DIFFUSION_SAMPLE_BLOCK_SIZE = 32;
constexpr uint32_t ROCM_DIFFUSION_SAMPLE_STATUS_OK = 0x44534653u;
constexpr uint32_t ROCM_EMBEDDING_TABLE_ENCODING_F32 = 1;
constexpr uint32_t ROCM_EMBEDDING_TABLE_ENCODING_BF16 = 2;
constexpr uint32_t ROCM_EMBEDDING_TABLE_ENCODING_MLX_Q4 = 3;
constexpr uint32_t ROCM_DIFFUSION_EXPECTED_EMBEDDING_AFFINE_G64_ROWS_PER_BLOCK = 16;
constexpr uint32_t ROCM_DIFFUSION_EXPECTED_EMBEDDING_Q8_G64_DIMS_PER_THREAD = 4;
constexpr uint32_t ROCM_DIFFUSION_EXPECTED_EMBEDDING_Q8_G64_ROWS_PER_THREAD = 4;
constexpr uint32_t ROCM_EMBEDDING_MEAN_POOL_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_EMBEDDING_MEAN_POOL_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_EMBEDDING_MEAN_POOL_LAUNCH_FLAG_NORMALIZE = 1;
constexpr uint32_t ROCM_RERANK_COSINE_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_RERANK_COSINE_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_TINY_PREFILL_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_TINY_PREFILL_LAUNCH_ARGS_BYTES = 160;
constexpr uint32_t ROCM_TINY_DECODE_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_TINY_DECODE_LAUNCH_ARGS_BYTES = 160;
constexpr uint32_t ROCM_TINY_OUTPUT_WEIGHT_ENCODING_FP32 = 1;
constexpr uint32_t ROCM_TINY_OUTPUT_WEIGHT_ENCODING_FP16 = 2;
constexpr uint32_t ROCM_TINY_OUTPUT_WEIGHT_ENCODING_Q8 = 3;
constexpr uint32_t ROCM_CROSS_ENTROPY_LOSS_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_CROSS_ENTROPY_LOSS_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_CROSS_ENTROPY_LOSS_OUTPUT_BYTES = 16;
constexpr uint32_t ROCM_DISTILLATION_KL_LOSS_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_DISTILLATION_KL_LOSS_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_DISTILLATION_KL_LOSS_OUTPUT_BYTES = 8;
constexpr uint32_t ROCM_GRPO_ADVANTAGE_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_GRPO_ADVANTAGE_LAUNCH_ARGS_BYTES = 64;
constexpr uint32_t ROCM_AUTOROUND_QUANTIZE_LAUNCH_ARGS_VERSION = 1;
constexpr uint32_t ROCM_AUTOROUND_QUANTIZE_LAUNCH_ARGS_BYTES = 96;
constexpr uint32_t ROCM_AUTOROUND_FORMAT_MXFP4 = 1;
constexpr uint32_t ROCM_AUTOROUND_FORMAT_NVFP4 = 2;
constexpr uint32_t ROCM_AUTOROUND_FORMAT_FP8 = 3;
constexpr uint32_t ROCM_AUTOROUND_FORMAT_MXFP8 = 4;
constexpr uint32_t ROCM_AUTOROUND_FORMAT_INT2 = 5;

struct rocm_prefill_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t token_pointer;
	uint64_t token_count;
	uint64_t token_bytes;
	uint32_t mode_code;
	uint32_t block_size;
	uint32_t key_width;
	uint32_t value_width;
	uint64_t status_pointer;
	uint32_t status_value;
	uint32_t reserved;
};

struct rocm_device_kv_launch_descriptor {
	uint64_t descriptor_pointer;
	uint64_t descriptor_bytes;
	uint32_t descriptor_version;
	uint32_t mode_code;
	uint32_t block_size;
	uint32_t page_count;
	uint64_t token_count;
	uint32_t key_width;
	uint32_t value_width;
	uint64_t status_pointer;
	uint32_t status_value;
	uint32_t reserved;
};

struct rocm_device_kv_descriptor_header {
	uint32_t version;
	uint32_t header_bytes;
	uint32_t page_bytes;
	uint32_t mode_code;
	uint32_t page_count;
	uint32_t block_size;
	uint64_t token_count;
};

struct rocm_device_kv_page_descriptor {
	uint64_t token_start;
	uint64_t token_count;
	uint32_t key_width;
	uint32_t value_width;
	uint32_t key_encoding;
	uint32_t value_encoding;
	uint64_t key_pointer;
	uint64_t value_pointer;
	uint64_t key_bytes;
	uint64_t value_bytes;
};

struct rocm_kv_encode_token_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t key_input_pointer;
	uint64_t value_input_pointer;
	uint64_t key_output_pointer;
	uint64_t value_output_pointer;
	uint32_t key_count;
	uint32_t value_count;
	uint32_t key_input_bytes;
	uint32_t value_input_bytes;
	uint32_t key_output_bytes;
	uint32_t value_output_bytes;
	uint32_t key_encoding;
	uint32_t value_encoding;
	uint64_t reserved0;
	uint64_t reserved1;
	uint64_t reserved2;
};

struct rocm_kv_encode_token_value_norm_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t key_input_pointer;
	uint64_t value_input_pointer;
	uint64_t key_output_pointer;
	uint64_t value_output_pointer;
	uint32_t key_count;
	uint32_t value_count;
	uint32_t key_input_bytes;
	uint32_t value_input_bytes;
	uint32_t key_output_bytes;
	uint32_t value_output_bytes;
	uint32_t key_encoding;
	uint32_t value_encoding;
	uint64_t reserved0;
	uint64_t reserved1;
	uint64_t reserved2;
	uint32_t value_head_dim;
	uint32_t value_head_count;
	uint32_t value_epsilon_bits;
	uint32_t reserved3;
};

struct rocm_kv_descriptor_append_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t previous_descriptor_pointer;
	uint64_t output_descriptor_pointer;
	uint64_t new_key_pointer;
	uint64_t new_value_pointer;
	uint64_t previous_descriptor_bytes;
	uint64_t output_descriptor_bytes;
	uint64_t new_key_bytes;
	uint64_t new_value_bytes;
	uint32_t mode_code;
	uint32_t block_size;
	uint32_t output_page_count;
	uint32_t output_token_count;
	uint32_t key_width;
	uint32_t value_width;
	uint32_t new_key_encoding;
	uint32_t new_value_encoding;
	uint64_t trim_start;
	uint64_t reserved0;
	uint64_t reserved1;
};

struct rocm_kv_encode_token_value_norm_descriptor_append_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	rocm_kv_encode_token_value_norm_launch_args encode;
	rocm_kv_descriptor_append_launch_args descriptor;
};

struct rocm_decode_launch_args {
	uint32_t version;
	uint32_t header_bytes;
	uint32_t total_bytes;
	uint32_t token_id;
	uint64_t position;
	uint32_t kv_descriptor_bytes;
	uint32_t reserved;
	rocm_device_kv_launch_descriptor kv;
};

struct rocm_projection_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint32_t input_count;
	uint32_t input_bytes;
	uint64_t weight_pointer;
	uint64_t weight_bytes;
	uint64_t bias_pointer;
	uint64_t bias_bytes;
	uint64_t output_pointer;
	uint64_t output_bytes;
	uint32_t rows;
	uint32_t cols;
	uint32_t weight_encoding;
	uint32_t flags;
	uint32_t q8_scale_bits;
	uint32_t reserved;
};

struct rocm_projection_batch_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t weight_pointer;
	uint64_t weight_bytes;
	uint64_t bias_pointer;
	uint64_t bias_bytes;
	uint64_t output_pointer;
	uint64_t output_bytes;
	uint32_t rows;
	uint32_t cols;
	uint32_t batch;
	uint32_t weight_encoding;
	uint32_t flags;
	uint32_t q8_scale_bits;
	uint64_t input_bytes;
	uint64_t reserved;
};

struct rocm_mlx_q4_projection_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t weight_pointer;
	uint64_t scale_pointer;
	uint64_t bias_pointer;
	uint64_t output_pointer;
	uint32_t rows;
	uint32_t cols;
	uint32_t group_size;
	uint32_t bits;
	uint32_t input_bytes;
	uint32_t weight_bytes;
	uint32_t scale_bytes;
	uint32_t bias_bytes;
	uint32_t output_bytes;
	uint32_t suppress_count;
	uint64_t suppress_pointer;
};

struct rocm_mlx_q4_projection_batch_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t weight_pointer;
	uint64_t scale_pointer;
	uint64_t bias_pointer;
	uint64_t output_pointer;
	uint32_t rows;
	uint32_t cols;
	uint32_t batch;
	uint32_t group_size;
	uint32_t bits;
	uint32_t input_bytes;
	uint32_t weight_bytes;
	uint32_t scale_bytes;
	uint32_t bias_bytes;
	uint32_t output_bytes;
	uint32_t reserved0;
};

struct rocm_mlx_q4_projection_greedy_batch_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t weight_pointer;
	uint64_t scale_pointer;
	uint64_t bias_pointer;
	uint64_t output_pointer;
	uint64_t suppress_pointer;
	uint32_t rows;
	uint32_t cols;
	uint32_t batch;
	uint32_t group_size;
	uint32_t bits;
	uint32_t input_bytes;
	uint32_t weight_bytes;
	uint32_t scale_bytes;
	uint32_t bias_bytes;
	uint32_t output_bytes;
	uint32_t suppress_count;
	uint32_t reserved0;
};

struct rocm_packed_topk_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t output_pointer;
	uint32_t input_count;
	uint32_t output_count;
	uint32_t top_k;
	uint32_t chunk_size;
	uint32_t input_bytes;
	uint32_t output_bytes;
};

struct rocm_packed_topk_sample_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t output_pointer;
	uint32_t input_count;
	uint32_t top_k;
	uint32_t input_bytes;
	uint32_t output_bytes;
	uint32_t temperature_bits;
	uint32_t top_p_bits;
	uint64_t draw_bits;
	uint32_t softcap_bits;
	uint32_t reserved0;
};

struct rocm_ordered_embedding_candidates_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t topk_pointer;
	uint64_t token_ordering_pointer;
	uint64_t output_pointer;
	uint64_t suppress_pointer;
	uint32_t topk_count;
	uint32_t num_centroids;
	uint32_t tokens_per_centroid;
	uint32_t token_ordering_element_bytes;
	uint32_t token_ordering_count;
	uint32_t output_count;
	uint32_t suppress_count;
	uint32_t topk_bytes;
	uint32_t token_ordering_bytes;
	uint32_t output_bytes;
};

struct rocm_mlx_q4_triple_projection_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t output_pointer;
	uint64_t first_weight_pointer;
	uint64_t first_scale_pointer;
	uint64_t first_bias_pointer;
	uint64_t second_weight_pointer;
	uint64_t second_scale_pointer;
	uint64_t second_bias_pointer;
	uint64_t third_weight_pointer;
	uint64_t third_scale_pointer;
	uint64_t third_bias_pointer;
	uint32_t first_rows;
	uint32_t second_rows;
	uint32_t third_rows;
	uint32_t cols;
	uint32_t group_size;
	uint32_t bits;
	uint32_t input_bytes;
	uint32_t output_bytes;
	uint32_t first_weight_bytes;
	uint32_t first_scale_bytes;
	uint32_t first_bias_bytes;
	uint32_t second_weight_bytes;
	uint32_t second_scale_bytes;
	uint32_t second_bias_bytes;
	uint32_t third_weight_bytes;
	uint32_t third_scale_bytes;
	uint32_t third_bias_bytes;
	uint32_t reserved0;
};

struct rocm_mlx_q4_gelu_tanh_mul_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t gate_weight_pointer;
	uint64_t gate_scale_pointer;
	uint64_t gate_bias_pointer;
	uint64_t up_weight_pointer;
	uint64_t up_scale_pointer;
	uint64_t up_bias_pointer;
	uint64_t output_pointer;
	uint32_t rows;
	uint32_t cols;
	uint32_t group_size;
	uint32_t bits;
	uint32_t input_bytes;
	uint32_t gate_weight_bytes;
	uint32_t gate_scale_bytes;
	uint32_t gate_bias_bytes;
	uint32_t up_weight_bytes;
	uint32_t up_scale_bytes;
	uint32_t up_bias_bytes;
	uint32_t output_bytes;
	uint32_t reserved0;
	uint32_t reserved1;
};

struct rocm_mlx_q4_gelu_tanh_mul_batch_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t gate_weight_pointer;
	uint64_t gate_scale_pointer;
	uint64_t gate_bias_pointer;
	uint64_t up_weight_pointer;
	uint64_t up_scale_pointer;
	uint64_t up_bias_pointer;
	uint64_t output_pointer;
	uint32_t rows;
	uint32_t cols;
	uint32_t group_size;
	uint32_t bits;
	uint32_t input_bytes;
	uint32_t gate_weight_bytes;
	uint32_t gate_scale_bytes;
	uint32_t gate_bias_bytes;
	uint32_t up_weight_bytes;
	uint32_t up_scale_bytes;
	uint32_t up_bias_bytes;
	uint32_t output_bytes;
	uint32_t batch;
	uint32_t reserved0;
};

struct rocm_mlx_q4_gelu_tanh_mlp_persistent_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t gate_weight_pointer;
	uint64_t gate_scale_pointer;
	uint64_t gate_bias_pointer;
	uint64_t up_weight_pointer;
	uint64_t up_scale_pointer;
	uint64_t up_bias_pointer;
	uint64_t down_weight_pointer;
	uint64_t down_scale_pointer;
	uint64_t down_bias_pointer;
	uint64_t activation_pointer;
	uint64_t output_pointer;
	uint64_t barrier_pointer;
	uint32_t rows;
	uint32_t cols;
	uint32_t down_rows;
	uint32_t group_size;
	uint32_t bits;
	uint32_t input_bytes;
	uint32_t gate_weight_bytes;
	uint32_t gate_scale_bytes;
	uint32_t gate_bias_bytes;
	uint32_t up_weight_bytes;
	uint32_t up_scale_bytes;
	uint32_t up_bias_bytes;
	uint32_t down_weight_bytes;
	uint32_t down_scale_bytes;
	uint32_t down_bias_bytes;
	uint32_t activation_bytes;
	uint32_t output_bytes;
	uint32_t barrier_bytes;
};

struct rocm_mlx_q4_gelu_tanh_proj_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t weight_pointer;
	uint64_t scale_pointer;
	uint64_t bias_pointer;
	uint64_t multiplier_pointer;
	uint64_t output_pointer;
	uint32_t rows;
	uint32_t cols;
	uint32_t group_size;
	uint32_t bits;
	uint32_t input_bytes;
	uint32_t weight_bytes;
	uint32_t scale_bytes;
	uint32_t bias_bytes;
	uint32_t multiplier_bytes;
	uint32_t output_bytes;
};

struct rocm_mlx_q4_gelu_tanh_proj_batch_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t weight_pointer;
	uint64_t scale_pointer;
	uint64_t bias_pointer;
	uint64_t multiplier_pointer;
	uint64_t output_pointer;
	uint32_t rows;
	uint32_t cols;
	uint32_t batch;
	uint32_t group_size;
	uint32_t bits;
	uint32_t input_bytes;
	uint32_t weight_bytes;
	uint32_t scale_bytes;
	uint32_t bias_bytes;
	uint32_t multiplier_bytes;
	uint32_t output_bytes;
};

struct rocm_rms_norm_residual_add_gelu_tanh_proj_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t rms_weight_pointer;
	uint64_t residual_pointer;
	uint64_t weight_pointer;
	uint64_t scale_pointer;
	uint64_t bias_pointer;
	uint64_t multiplier_pointer;
	uint64_t residual_output_pointer;
	uint64_t activation_output_pointer;
	uint32_t rows;
	uint32_t cols;
	uint32_t group_size;
	uint32_t bits;
	uint32_t input_bytes;
	uint32_t rms_weight_bytes;
	uint32_t residual_bytes;
	uint32_t weight_bytes;
	uint32_t scale_bytes;
	uint32_t bias_bytes;
	uint32_t multiplier_bytes;
	uint32_t residual_output_bytes;
	uint32_t activation_output_bytes;
	uint32_t epsilon_bits;
	uint32_t rms_weight_encoding;
	uint32_t rms_flags;
	uint32_t output_scale_bits;
	uint32_t reserved0;
	uint32_t reserved1;
	uint32_t reserved2;
};

struct rocm_lora_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t base_weight_pointer;
	uint64_t lora_a_pointer;
	uint64_t lora_b_pointer;
	uint64_t bias_pointer;
	uint64_t output_pointer;
	uint32_t input_count;
	uint32_t rows;
	uint32_t cols;
	uint32_t rank;
	uint32_t input_bytes;
	uint32_t base_weight_bytes;
	uint32_t lora_a_bytes;
	uint32_t lora_b_bytes;
	uint32_t bias_bytes;
	uint32_t output_bytes;
	uint32_t alpha_bits;
	uint32_t flags;
	uint64_t reserved1;
	uint64_t reserved2;
	uint64_t reserved3;
};

struct rocm_embedding_lookup_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t token_pointer;
	uint64_t embedding_pointer;
	uint64_t output_pointer;
	uint32_t token_count;
	uint32_t vocab_size;
	uint32_t hidden_size;
	uint32_t token_bytes;
	uint64_t embedding_bytes;
	uint64_t output_bytes;
	uint32_t table_encoding;
	uint32_t group_size;
	uint64_t scale_pointer;
	uint64_t bias_pointer;
	uint32_t scale_bytes;
	uint32_t bias_bytes;
	uint32_t output_scale_bits;
	uint32_t bits;
};

struct rocm_diffusion_expected_embedding_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t probability_pointer;
	uint64_t embedding_pointer;
	uint64_t scale_pointer;
	uint64_t bias_pointer;
	uint64_t output_pointer;
	uint32_t rows;
	uint32_t vocab_size;
	uint32_t hidden_size;
	uint32_t table_encoding;
	uint32_t group_size;
	uint32_t bits;
	uint64_t probability_bytes;
	uint64_t embedding_bytes;
	uint64_t scale_bytes;
	uint64_t bias_bytes;
	uint64_t output_bytes;
	uint32_t output_scale_bits;
	uint32_t reserved;
};

struct rocm_diffusion_sample_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t logits_pointer;
	uint64_t draws_pointer;
	uint64_t output_pointer;
	uint32_t rows;
	uint32_t vocab_size;
	uint32_t temperature_bits;
	uint32_t softcap_bits;
	uint64_t logits_bytes;
	uint64_t draws_bytes;
	uint64_t output_bytes;
};

struct rocm_diffusion_sample_result {
	int32_t sampled;
	int32_t greedy;
	float entropy;
	uint32_t status;
};

struct rocm_embedding_mean_pool_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t token_pointer;
	uint64_t output_pointer;
	uint32_t token_count;
	uint32_t dim;
	uint32_t token_bytes;
	uint32_t output_bytes;
	uint32_t flags;
	uint32_t reserved;
	uint64_t reserved2;
	uint64_t reserved3;
};

struct rocm_rerank_cosine_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t query_pointer;
	uint64_t document_pointer;
	uint64_t output_pointer;
	uint32_t document_count;
	uint32_t dim;
	uint32_t query_bytes;
	uint32_t document_bytes;
	uint32_t output_bytes;
	uint32_t reserved;
	uint64_t reserved2;
};

struct rocm_rms_norm_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t weight_pointer;
	uint64_t output_pointer;
	uint32_t count;
	uint32_t input_bytes;
	uint32_t weight_bytes;
	uint32_t output_bytes;
	uint32_t epsilon_bits;
	uint32_t weight_encoding;
	uint32_t flags;
	uint32_t output_scale_bits;
};

struct rocm_moe_combine_norms_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t local_input_pointer;
	uint64_t local_weight_pointer;
	uint64_t expert_input_pointer;
	uint64_t expert_weight_pointer;
	uint64_t output_pointer;
	uint32_t count;
	uint32_t local_input_bytes;
	uint32_t local_weight_bytes;
	uint32_t expert_input_bytes;
	uint32_t expert_weight_bytes;
	uint32_t output_bytes;
	uint32_t local_epsilon_bits;
	uint32_t local_weight_encoding;
	uint32_t local_flags;
	uint32_t expert_epsilon_bits;
	uint32_t expert_weight_encoding;
	uint32_t expert_flags;
};

struct rocm_rms_norm_residual_add_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t weight_pointer;
	uint64_t residual_pointer;
	uint64_t output_pointer;
	uint32_t count;
	uint32_t input_bytes;
	uint32_t weight_bytes;
	uint32_t residual_bytes;
	uint32_t output_bytes;
	uint32_t epsilon_bits;
	uint32_t weight_encoding;
	uint32_t flags;
	uint32_t output_scale_bits;
};

struct rocm_rms_norm_residual_add_norm_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t weight_pointer;
	uint64_t residual_pointer;
	uint64_t residual_output_pointer;
	uint64_t norm_weight_pointer;
	uint64_t norm_output_pointer;
	uint32_t count;
	uint32_t input_bytes;
	uint32_t weight_bytes;
	uint32_t residual_bytes;
	uint32_t residual_output_bytes;
	uint32_t norm_weight_bytes;
	uint32_t norm_output_bytes;
	uint32_t epsilon_bits;
	uint32_t weight_encoding;
	uint32_t flags;
	uint32_t norm_epsilon_bits;
	uint32_t norm_weight_encoding;
	uint32_t norm_flags;
	uint32_t output_scale_bits;
	uint32_t reserved;
	uint64_t reserved2;
};

struct rocm_rms_norm_heads_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t weight_pointer;
	uint64_t output_pointer;
	uint32_t head_dim;
	uint32_t head_count;
	uint32_t input_bytes;
	uint32_t weight_bytes;
	uint32_t output_bytes;
	uint32_t epsilon_bits;
	uint32_t weight_encoding;
	uint32_t flags;
};

struct rocm_rms_norm_rope_heads_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t weight_pointer;
	uint64_t output_pointer;
	uint32_t head_dim;
	uint32_t head_count;
	uint32_t input_bytes;
	uint32_t weight_bytes;
	uint32_t output_bytes;
	uint32_t epsilon_bits;
	uint32_t weight_encoding;
	uint32_t flags;
	uint32_t position;
	uint32_t base_bits;
	uint32_t frequency_dim;
	uint32_t rotary_count;
	uint32_t frequency_scale_bits;
	uint32_t reserved0;
};

struct rocm_rms_norm_rope_heads_pair_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t query_input_pointer;
	uint64_t query_weight_pointer;
	uint64_t query_output_pointer;
	uint64_t key_input_pointer;
	uint64_t key_weight_pointer;
	uint64_t key_output_pointer;
	uint32_t head_dim;
	uint32_t query_head_count;
	uint32_t key_head_count;
	uint32_t query_input_bytes;
	uint32_t query_weight_bytes;
	uint32_t query_output_bytes;
	uint32_t key_input_bytes;
	uint32_t key_weight_bytes;
	uint32_t key_output_bytes;
	uint32_t query_epsilon_bits;
	uint32_t query_weight_encoding;
	uint32_t query_flags;
	uint32_t key_epsilon_bits;
	uint32_t key_weight_encoding;
	uint32_t key_flags;
	uint32_t position;
	uint32_t base_bits;
	uint32_t frequency_dim;
	uint32_t rotary_count;
	uint32_t frequency_scale_bits;
	uint32_t reserved0;
	uint32_t reserved1;
};

struct rocm_rms_norm_rope_heads_batch_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t weight_pointer;
	uint64_t output_pointer;
	uint32_t head_dim;
	uint32_t head_count;
	uint32_t batch;
	uint32_t input_bytes;
	uint32_t weight_bytes;
	uint32_t output_bytes;
	uint32_t epsilon_bits;
	uint32_t weight_encoding;
	uint32_t flags;
	uint32_t start_position;
	uint32_t base_bits;
	uint32_t frequency_dim;
	uint32_t rotary_count;
	uint32_t frequency_scale_bits;
	uint32_t reserved1;
	uint32_t reserved2;
};

struct rocm_rms_norm_rope_heads_pair_lane_batch_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t query_input_pointer;
	uint64_t query_weight_pointer;
	uint64_t query_output_pointer;
	uint64_t key_input_pointer;
	uint64_t key_weight_pointer;
	uint64_t key_output_pointer;
	uint64_t positions_pointer;
	uint32_t head_dim;
	uint32_t query_head_count;
	uint32_t key_head_count;
	uint32_t batch;
	uint32_t query_input_bytes;
	uint32_t query_weight_bytes;
	uint32_t query_output_bytes;
	uint32_t key_input_bytes;
	uint32_t key_weight_bytes;
	uint32_t key_output_bytes;
	uint32_t query_epsilon_bits;
	uint32_t query_weight_encoding;
	uint32_t query_flags;
	uint32_t key_epsilon_bits;
	uint32_t key_weight_encoding;
	uint32_t key_flags;
	uint32_t positions_bytes;
	uint32_t base_bits;
	uint32_t frequency_dim;
	uint32_t rotary_count;
	uint32_t frequency_scale_bits;
	uint32_t reserved;
};

struct rocm_rope_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t output_pointer;
	uint32_t count;
	uint32_t input_bytes;
	uint32_t output_bytes;
	uint32_t position;
	uint32_t base_bits;
	uint32_t frequency_dim;
	uint64_t reserved2;
	uint64_t reserved3;
};

struct rocm_rope_heads_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t output_pointer;
	uint32_t head_dim;
	uint32_t head_count;
	uint32_t input_bytes;
	uint32_t output_bytes;
	uint32_t position;
	uint32_t base_bits;
	uint32_t frequency_dim;
	uint32_t rotary_count;
	uint64_t reserved2;
};

struct rocm_greedy_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t logits_pointer;
	uint64_t output_pointer;
	uint32_t count;
	uint32_t logits_bytes;
	uint32_t output_bytes;
	uint32_t reserved;
	uint64_t reserved2;
	uint64_t reserved3;
	uint64_t reserved4;
};

struct rocm_softcap_greedy_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t logits_pointer;
	uint64_t output_pointer;
	uint32_t count;
	uint32_t logits_bytes;
	uint32_t output_bytes;
	uint32_t softcap_bits;
	uint64_t suppress_pointer;
	uint32_t suppress_count;
	uint32_t reserved3;
	uint64_t reserved4;
};

struct rocm_attention_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t query_pointer;
	uint64_t key_pointer;
	uint64_t value_pointer;
	uint64_t output_pointer;
	uint64_t weight_pointer;
	uint32_t dim;
	uint32_t token_count;
	uint32_t query_bytes;
	uint32_t key_bytes;
	uint32_t value_bytes;
	uint32_t output_bytes;
	uint32_t weight_bytes;
	uint32_t kv_source;
	uint32_t scale_bits;
	uint64_t descriptor_pointer;
	uint64_t descriptor_bytes;
};

struct rocm_attention_heads_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t query_pointer;
	uint64_t key_pointer;
	uint64_t value_pointer;
	uint64_t output_pointer;
	uint64_t weight_pointer;
	uint32_t dim;
	uint32_t token_count;
	uint32_t head_count;
	uint32_t query_bytes;
	uint32_t key_bytes;
	uint32_t value_bytes;
	uint32_t output_bytes;
	uint32_t weight_bytes;
	uint32_t kv_source;
	uint32_t scale_bits;
	uint64_t descriptor_pointer;
	uint64_t descriptor_bytes;
	uint64_t shared_mem_bytes;
	uint32_t window_size;
	uint32_t reserved0;
	uint64_t reserved1;
};

struct rocm_attention_heads_batch_causal_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t query_pointer;
	uint64_t key_pointer;
	uint64_t value_pointer;
	uint64_t output_pointer;
	uint64_t weight_pointer;
	uint32_t dim;
	uint32_t token_count;
	uint32_t head_count;
	uint32_t query_count;
	uint32_t query_start_token;
	uint32_t query_bytes;
	uint32_t key_bytes;
	uint32_t value_bytes;
	uint32_t output_bytes;
	uint32_t weight_bytes;
	uint32_t kv_source;
	uint32_t scale_bits;
	uint64_t descriptor_pointer;
	uint64_t descriptor_bytes;
	uint64_t shared_mem_bytes;
	uint32_t window_size;
	uint32_t reserved0;
	uint64_t reserved1;
	uint64_t reserved2;
};

struct rocm_attention_heads_lane_descriptor {
	uint64_t descriptor_pointer;
	uint64_t descriptor_bytes;
	uint32_t token_count;
	uint32_t window_size;
	uint64_t reserved0;
};

struct rocm_attention_heads_lane_batch_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t query_pointer;
	uint64_t output_pointer;
	uint64_t weight_pointer;
	uint64_t lane_pointer;
	uint32_t dim;
	uint32_t lane_count;
	uint32_t head_count;
	uint32_t key_heads;
	uint32_t max_token_count;
	uint32_t query_bytes;
	uint32_t output_bytes;
	uint32_t weight_bytes;
	uint32_t lane_bytes;
	uint32_t scale_bits;
	uint64_t shared_mem_bytes;
	uint64_t reserved0;
	uint64_t reserved1;
	uint64_t reserved2;
};

struct rocm_attention_heads_batch_causal_query_rms_rope_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t query_input_pointer;
	uint64_t query_weight_pointer;
	uint64_t key_pointer;
	uint64_t value_pointer;
	uint64_t output_pointer;
	uint64_t attention_weight_pointer;
	uint32_t dim;
	uint32_t token_count;
	uint32_t head_count;
	uint32_t query_count;
	uint32_t query_start_token;
	uint32_t query_input_bytes;
	uint32_t query_weight_bytes;
	uint32_t key_bytes;
	uint32_t value_bytes;
	uint32_t output_bytes;
	uint32_t attention_weight_bytes;
	uint32_t kv_source;
	uint32_t scale_bits;
	uint32_t key_heads;
	uint64_t descriptor_pointer;
	uint64_t descriptor_bytes;
	uint64_t shared_mem_bytes;
	uint32_t window_size;
	uint32_t query_epsilon_bits;
	uint32_t query_weight_encoding;
	uint32_t query_flags;
	uint32_t rope_start_position;
	uint32_t rope_base_bits;
	uint32_t rope_frequency_dim;
	uint32_t rope_rotary_count;
	uint32_t rope_frequency_scale_bits;
	uint32_t reserved0;
	uint64_t reserved1;
};

struct rocm_attention_heads_chunked_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t query_pointer;
	uint64_t descriptor_pointer;
	uint64_t partial_pointer;
	uint64_t stats_pointer;
	uint64_t output_pointer;
	uint32_t dim;
	uint32_t token_count;
	uint32_t head_count;
	uint32_t chunk_size;
	uint32_t chunk_count;
	uint32_t query_bytes;
	uint64_t descriptor_bytes;
	uint32_t partial_bytes;
	uint32_t stats_bytes;
	uint32_t output_bytes;
	uint32_t scale_bits;
	uint32_t window_size;
	uint32_t key_heads;
	uint64_t reserved1;
	uint64_t reserved2;
	uint64_t reserved3;
};

struct rocm_attention_heads_batch_chunked_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t query_pointer;
	uint64_t descriptor_pointer;
	uint64_t partial_pointer;
	uint64_t stats_pointer;
	uint64_t output_pointer;
	uint32_t dim;
	uint32_t token_count;
	uint32_t head_count;
	uint32_t query_count;
	uint32_t query_start_token;
	uint32_t chunk_size;
	uint32_t chunk_count;
	uint32_t query_bytes;
	uint64_t descriptor_bytes;
	uint32_t partial_bytes;
	uint32_t stats_bytes;
	uint32_t output_bytes;
	uint32_t scale_bits;
	uint32_t window_size;
	uint32_t chunk_start_token;
	uint32_t key_heads;
	uint32_t reserved0;
	uint64_t reserved2;
	uint64_t reserved3;
};

struct rocm_vector_add_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t left_pointer;
	uint64_t right_pointer;
	uint64_t output_pointer;
	uint32_t count;
	uint32_t left_bytes;
	uint32_t right_bytes;
	uint32_t output_bytes;
	uint32_t reserved;
	uint64_t reserved2;
};

struct rocm_vector_add_scaled_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t left_pointer;
	uint64_t right_pointer;
	uint64_t output_pointer;
	uint32_t count;
	uint32_t left_bytes;
	uint32_t right_bytes;
	uint32_t output_bytes;
	uint32_t scale_bits;
	uint32_t reserved;
	uint64_t reserved2;
};

struct rocm_vector_scale_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t output_pointer;
	uint32_t count;
	uint32_t input_bytes;
	uint32_t output_bytes;
	uint32_t scale_bits;
	uint64_t reserved2;
	uint64_t reserved3;
	uint64_t reserved4;
};

struct rocm_per_layer_input_transpose_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t output_pointer;
	uint64_t input_bytes;
	uint64_t output_bytes;
	uint32_t batch;
	uint32_t layer_count;
	uint32_t input_size;
	uint32_t reserved;
};

struct rocm_swiglu_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t gate_pointer;
	uint64_t up_pointer;
	uint64_t output_pointer;
	uint32_t count;
	uint32_t gate_bytes;
	uint32_t up_bytes;
	uint32_t output_bytes;
	uint32_t reserved;
	uint64_t reserved2;
};

struct rocm_gelu_tanh_mul_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t gate_pointer;
	uint64_t up_pointer;
	uint64_t output_pointer;
	uint32_t count;
	uint32_t gate_bytes;
	uint32_t up_bytes;
	uint32_t output_bytes;
	uint32_t reserved;
	uint64_t reserved2;
};

struct rocm_moe_router_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t logit_pointer;
	uint64_t id_pointer;
	uint64_t prob_pointer;
	uint32_t expert_count;
	uint32_t top_k;
	uint32_t logit_bytes;
	uint32_t id_bytes;
	uint32_t prob_bytes;
	uint32_t layer;
	uint64_t status_pointer;
};

struct rocm_moe_lazy_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t id_pointer;
	uint64_t resident_pointer;
	uint32_t selected_count;
	uint32_t expert_count;
	uint32_t id_bytes;
	uint32_t resident_bytes;
	uint64_t reserved;
	uint64_t reserved2;
	uint64_t reserved3;
};

struct rocm_moe_batch_route_metadata {
	uint32_t row;
	uint32_t pair;
	float weight;
	uint32_t reserved;
};

struct rocm_moe_batch_route_rows_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t metadata_pointer;
	uint64_t output_pointer;
	uint32_t route_count;
	uint32_t row_width;
	uint32_t input_rows;
	uint32_t output_rows;
	uint64_t input_bytes;
	uint64_t metadata_bytes;
	uint64_t output_bytes;
};

struct rocm_moe_batch_reduce_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t output_pointer;
	uint32_t rows;
	uint32_t top_k;
	uint32_t row_width;
	uint32_t reserved;
	uint64_t input_bytes;
	uint64_t output_bytes;
};

struct rocm_moe_mlx_affine_routes_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t chunk_pointer;
	uint64_t output_pointer;
	uint32_t rows;
	uint32_t cols;
	uint32_t input_rows;
	uint32_t pair_count;
	uint32_t chunk_count;
	uint32_t group_size;
	uint32_t bits;
	uint32_t input_bytes;
	uint32_t chunk_bytes;
	uint32_t output_bytes;
	uint32_t flags;
	uint32_t reserved0;
};

struct rocm_moe_mlx_affine_routes_chunk {
	uint64_t gate_up_weight_pointer;
	uint64_t gate_up_scale_pointer;
	uint64_t gate_up_bias_pointer;
	uint64_t down_weight_pointer;
	uint64_t down_scale_pointer;
	uint64_t down_bias_pointer;
	uint32_t route_count;
	uint32_t reserved0;
	uint32_t token_rows[ROCM_MOE_MLX_AFFINE_ROUTES_PER_CHUNK];
	uint32_t pair_indices[ROCM_MOE_MLX_AFFINE_ROUTES_PER_CHUNK];
	float route_weights[ROCM_MOE_MLX_AFFINE_ROUTES_PER_CHUNK];
};

struct rocm_gguf_q4_0_projection_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t weight_pointer;
	uint64_t output_pointer;
	uint32_t rows;
	uint32_t cols;
	uint32_t row_offset;
	uint32_t weight_rows;
	uint32_t input_bytes;
	uint32_t weight_bytes;
	uint32_t output_bytes;
	uint32_t reserved;
};

struct rocm_gguf_q4_k_expand_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t raw_pointer;
	uint64_t expanded_pointer;
	uint32_t block_count;
	uint32_t reserved;
	uint64_t raw_bytes;
	uint64_t expanded_bytes;
};

struct rocm_gguf_q4_0_selected_experts_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t activation_pointer;
	uint64_t output_pointer;
	uint64_t gate_up_pointers[ROCM_GGUF_Q4_0_SELECTED_EXPERTS_MAX_TOP_K];
	uint64_t down_pointers[ROCM_GGUF_Q4_0_SELECTED_EXPERTS_MAX_TOP_K];
	float route_weights[ROCM_GGUF_Q4_0_SELECTED_EXPERTS_MAX_TOP_K];
	uint32_t top_k;
	uint32_t hidden_size;
	uint32_t expert_ff;
	uint32_t gate_up_rows;
	uint32_t down_rows;
	uint32_t input_bytes;
	uint32_t activation_bytes;
	uint32_t output_bytes;
	uint32_t gate_up_bytes;
	uint32_t down_bytes;
	uint32_t gate_up_format;
	uint32_t down_format;
};

struct rocm_jangtq_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t input_pointer;
	uint64_t packed_pointer;
	uint64_t bias_pointer;
	uint64_t output_pointer;
	uint32_t input_count;
	uint32_t rows;
	uint32_t cols;
	uint32_t bits;
	uint32_t group_size;
	uint32_t input_bytes;
	uint32_t packed_bytes;
	uint32_t bias_bytes;
	uint32_t output_bytes;
	uint32_t scale_bits;
	uint32_t flags;
	uint64_t reserved;
};

struct rocm_codebook_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t code_pointer;
	uint64_t codebook_pointer;
	uint64_t output_pointer;
	uint32_t code_count;
	uint32_t codebook_count;
	uint32_t code_dim;
	uint32_t code_bytes;
	uint32_t codebook_bytes;
	uint32_t output_bytes;
	uint64_t reserved;
};

struct rocm_tiny_prefill_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t token_pointer;
	uint64_t embedding_pointer;
	uint64_t output_weight_pointer;
	uint64_t logit_pointer;
	uint64_t attention_pointer;
	uint64_t result_pointer;
	uint64_t key_pointer;
	uint64_t value_pointer;
	uint32_t token_count;
	uint32_t vocab_size;
	uint32_t hidden_size;
	uint32_t token_bytes;
	uint32_t embedding_bytes;
	uint32_t output_weight_bytes;
	uint32_t logit_bytes;
	uint32_t attention_bytes;
	uint32_t result_bytes;
	uint32_t key_bytes;
	uint32_t value_bytes;
	uint32_t output_weight_encoding;
	uint32_t q8_scale_bits;
	uint32_t reserved;
	uint64_t reserved2;
	uint64_t reserved3;
	uint64_t reserved4;
	uint64_t reserved5;
};

struct rocm_tiny_decode_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t prior_key_pointer;
	uint64_t prior_value_pointer;
	uint64_t embedding_pointer;
	uint64_t output_weight_pointer;
	uint64_t logit_pointer;
	uint64_t attention_pointer;
	uint64_t updated_key_pointer;
	uint64_t updated_value_pointer;
	uint64_t result_pointer;
	uint32_t token_id;
	uint32_t prior_token_count;
	uint32_t vocab_size;
	uint32_t hidden_size;
	uint32_t prior_key_bytes;
	uint32_t prior_value_bytes;
	uint32_t embedding_bytes;
	uint32_t output_weight_bytes;
	uint32_t logit_bytes;
	uint32_t attention_bytes;
	uint32_t updated_key_bytes;
	uint32_t updated_value_bytes;
	uint32_t result_bytes;
	uint32_t output_weight_encoding;
	uint32_t q8_scale_bits;
	uint32_t reserved;
	uint64_t reserved2;
	uint64_t reserved3;
};

struct rocm_cross_entropy_loss_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t logit_pointer;
	uint64_t target_pointer;
	uint64_t output_pointer;
	uint32_t batch;
	uint32_t vocab;
	uint32_t logit_bytes;
	uint32_t target_bytes;
	uint32_t output_bytes;
	uint32_t reserved;
	uint64_t reserved2;
};

struct rocm_distillation_kl_loss_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t student_pointer;
	uint64_t teacher_pointer;
	uint64_t output_pointer;
	uint32_t batch;
	uint32_t vocab;
	uint32_t student_bytes;
	uint32_t teacher_bytes;
	uint32_t output_bytes;
	uint32_t reserved;
	uint64_t temperature_bits;
};

struct rocm_grpo_advantage_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t reward_pointer;
	uint64_t output_pointer;
	uint32_t count;
	uint32_t reward_bytes;
	uint32_t output_bytes;
	uint32_t reserved;
	uint64_t reserved2;
	uint64_t reserved3;
	uint64_t reserved4;
};

struct rocm_autoround_quantize_launch_args {
	uint32_t version;
	uint32_t total_bytes;
	uint64_t weight_pointer;
	uint64_t packed_pointer;
	uint64_t scale_pointer;
	uint32_t rows;
	uint32_t cols;
	uint32_t format_code;
	uint32_t bits;
	uint32_t group_size;
	uint32_t groups_per_row;
	uint32_t weight_bytes;
	uint32_t packed_bytes;
	uint32_t scale_bytes;
	uint32_t nsamples;
	uint32_t seqlen;
	uint32_t iters;
	uint64_t reserved0;
	uint64_t reserved1;
};

static_assert(sizeof(rocm_prefill_launch_args) == ROCM_PREFILL_LAUNCH_ARGS_BYTES, "prefill launch ABI drift");
static_assert(sizeof(rocm_device_kv_launch_descriptor) == ROCM_DEVICE_KV_LAUNCH_DESCRIPTOR_BYTES, "KV launch ABI drift");
static_assert(sizeof(rocm_device_kv_descriptor_header) == ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES, "KV descriptor header ABI drift");
static_assert(sizeof(rocm_device_kv_page_descriptor) == ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES, "KV descriptor page ABI drift");
static_assert(sizeof(rocm_kv_encode_token_launch_args) == ROCM_KV_ENCODE_TOKEN_LAUNCH_ARGS_BYTES, "KV encode token launch ABI drift");
static_assert(sizeof(rocm_kv_encode_token_value_norm_launch_args) == ROCM_KV_ENCODE_TOKEN_VALUE_NORM_LAUNCH_ARGS_BYTES, "KV encode value norm launch ABI drift");
static_assert(sizeof(rocm_kv_encode_token_value_norm_descriptor_append_launch_args) == ROCM_KV_ENCODE_TOKEN_VALUE_NORM_DESCRIPTOR_APPEND_LAUNCH_ARGS_BYTES, "KV encode value norm descriptor append launch ABI drift");
static_assert(sizeof(rocm_kv_descriptor_append_launch_args) == ROCM_KV_DESCRIPTOR_APPEND_LAUNCH_ARGS_BYTES, "KV descriptor append launch ABI drift");
static_assert(sizeof(rocm_decode_launch_args) == ROCM_DECODE_LAUNCH_ARGS_BYTES, "decode launch ABI drift");
static_assert(sizeof(rocm_projection_launch_args) == ROCM_PROJECTION_LAUNCH_ARGS_BYTES, "projection launch ABI drift");
static_assert(sizeof(rocm_projection_batch_launch_args) == ROCM_PROJECTION_BATCH_LAUNCH_ARGS_BYTES, "projection batch launch ABI drift");
static_assert(sizeof(rocm_mlx_q4_projection_launch_args) == ROCM_MLX_Q4_PROJECTION_LAUNCH_ARGS_BYTES, "MLX q4 projection launch ABI drift");
static_assert(sizeof(rocm_mlx_q4_projection_batch_launch_args) == ROCM_MLX_Q4_PROJECTION_BATCH_LAUNCH_ARGS_BYTES, "MLX q4 projection batch launch ABI drift");
static_assert(sizeof(rocm_mlx_q4_projection_greedy_batch_launch_args) == ROCM_MLX_Q4_PROJECTION_GREEDY_BATCH_LAUNCH_ARGS_BYTES, "MLX q4 projection greedy batch launch ABI drift");
static_assert(sizeof(rocm_mlx_q4_triple_projection_launch_args) == ROCM_MLX_Q4_TRIPLE_PROJECTION_LAUNCH_ARGS_BYTES, "MLX q4 triple projection launch ABI drift");
static_assert(sizeof(rocm_mlx_q4_gelu_tanh_mul_launch_args) == ROCM_MLX_Q4_GELU_TANH_MUL_LAUNCH_ARGS_BYTES, "MLX q4 GELU tanh multiply launch ABI drift");
static_assert(sizeof(rocm_mlx_q4_gelu_tanh_mul_batch_launch_args) == ROCM_MLX_Q4_GELU_TANH_MUL_BATCH_LAUNCH_ARGS_BYTES, "MLX q4 GELU tanh multiply batch launch ABI drift");
static_assert(sizeof(rocm_mlx_q4_gelu_tanh_mlp_persistent_launch_args) == ROCM_MLX_Q4_GELU_TANH_MLP_PERSISTENT_LAUNCH_ARGS_BYTES, "MLX q4 GELU tanh MLP persistent launch ABI drift");
static_assert(sizeof(rocm_mlx_q4_gelu_tanh_proj_launch_args) == ROCM_MLX_Q4_GELU_TANH_PROJ_LAUNCH_ARGS_BYTES, "MLX q4 GELU tanh projection launch ABI drift");
static_assert(sizeof(rocm_mlx_q4_gelu_tanh_proj_batch_launch_args) == ROCM_MLX_Q4_GELU_TANH_PROJ_BATCH_LAUNCH_ARGS_BYTES, "MLX q4 GELU tanh projection batch launch ABI drift");
static_assert(sizeof(rocm_rms_norm_residual_add_gelu_tanh_proj_launch_args) == ROCM_RMS_NORM_RESIDUAL_ADD_GELU_TANH_PROJ_LAUNCH_ARGS_BYTES, "RMS residual-add GELU tanh projection launch ABI drift");
static_assert(sizeof(rocm_packed_topk_sample_launch_args) == ROCM_PACKED_TOPK_SAMPLE_LAUNCH_ARGS_BYTES, "packed top-k sample launch ABI drift");
static_assert(sizeof(rocm_ordered_embedding_candidates_launch_args) == ROCM_ORDERED_EMBEDDING_CANDIDATES_LAUNCH_ARGS_BYTES, "ordered embedding candidates launch ABI drift");
static_assert(sizeof(rocm_rms_norm_launch_args) == ROCM_RMS_NORM_LAUNCH_ARGS_BYTES, "RMSNorm launch ABI drift");
static_assert(sizeof(rocm_moe_combine_norms_launch_args) == ROCM_MOE_COMBINE_NORMS_LAUNCH_ARGS_BYTES, "MoE combine norms launch ABI drift");
static_assert(sizeof(rocm_rms_norm_residual_add_launch_args) == ROCM_RMS_NORM_RESIDUAL_ADD_LAUNCH_ARGS_BYTES, "RMSNorm residual-add launch ABI drift");
static_assert(sizeof(rocm_rms_norm_residual_add_norm_launch_args) == ROCM_RMS_NORM_RESIDUAL_ADD_NORM_LAUNCH_ARGS_BYTES, "RMSNorm residual-add-norm launch ABI drift");
static_assert(sizeof(rocm_rms_norm_heads_launch_args) == ROCM_RMS_NORM_HEADS_LAUNCH_ARGS_BYTES, "RMSNorm heads launch ABI drift");
static_assert(sizeof(rocm_rms_norm_rope_heads_launch_args) == ROCM_RMS_NORM_ROPE_HEADS_LAUNCH_ARGS_BYTES, "RMSNorm RoPE heads launch ABI drift");
static_assert(sizeof(rocm_rms_norm_rope_heads_pair_launch_args) == ROCM_RMS_NORM_ROPE_HEADS_PAIR_LAUNCH_ARGS_BYTES, "RMSNorm RoPE heads pair launch ABI drift");
static_assert(sizeof(rocm_rms_norm_rope_heads_batch_launch_args) == ROCM_RMS_NORM_ROPE_HEADS_BATCH_LAUNCH_ARGS_BYTES, "RMSNorm RoPE heads batch launch ABI drift");
static_assert(sizeof(rocm_rms_norm_rope_heads_pair_lane_batch_launch_args) == ROCM_RMS_NORM_ROPE_HEADS_PAIR_LANE_BATCH_LAUNCH_ARGS_BYTES, "RMSNorm RoPE heads pair lane batch launch ABI drift");
static_assert(sizeof(rocm_rope_launch_args) == ROCM_ROPE_LAUNCH_ARGS_BYTES, "RoPE launch ABI drift");
static_assert(sizeof(rocm_rope_heads_launch_args) == ROCM_ROPE_HEADS_LAUNCH_ARGS_BYTES, "RoPE heads launch ABI drift");
static_assert(sizeof(rocm_greedy_launch_args) == ROCM_GREEDY_LAUNCH_ARGS_BYTES, "greedy launch ABI drift");
static_assert(sizeof(rocm_softcap_greedy_launch_args) == ROCM_SOFTCAP_GREEDY_LAUNCH_ARGS_BYTES, "softcap greedy launch ABI drift");
static_assert(sizeof(rocm_attention_launch_args) == ROCM_ATTENTION_LAUNCH_ARGS_BYTES, "attention launch ABI drift");
static_assert(sizeof(rocm_attention_heads_launch_args) == ROCM_ATTENTION_HEADS_LAUNCH_ARGS_BYTES, "attention heads launch ABI drift");
static_assert(sizeof(rocm_attention_heads_batch_causal_launch_args) == ROCM_ATTENTION_HEADS_BATCH_CAUSAL_LAUNCH_ARGS_BYTES, "attention heads batch causal launch ABI drift");
static_assert(sizeof(rocm_attention_heads_lane_descriptor) == ROCM_ATTENTION_HEADS_LANE_DESCRIPTOR_BYTES, "attention heads lane descriptor ABI drift");
static_assert(sizeof(rocm_attention_heads_lane_batch_launch_args) == ROCM_ATTENTION_HEADS_LANE_BATCH_LAUNCH_ARGS_BYTES, "attention heads lane batch launch ABI drift");
static_assert(sizeof(rocm_attention_heads_batch_causal_query_rms_rope_launch_args) == ROCM_ATTENTION_HEADS_BATCH_CAUSAL_QUERY_RMS_ROPE_LAUNCH_ARGS_BYTES, "attention heads batch causal query RMS RoPE launch ABI drift");
static_assert(sizeof(rocm_attention_heads_chunked_launch_args) == ROCM_ATTENTION_HEADS_CHUNKED_LAUNCH_ARGS_BYTES, "attention heads chunked launch ABI drift");
static_assert(sizeof(rocm_attention_heads_batch_chunked_launch_args) == ROCM_ATTENTION_HEADS_BATCH_CHUNKED_LAUNCH_ARGS_BYTES, "attention heads batch chunked launch ABI drift");
static_assert(sizeof(rocm_vector_add_launch_args) == ROCM_VECTOR_ADD_LAUNCH_ARGS_BYTES, "vector add launch ABI drift");
static_assert(sizeof(rocm_vector_add_scaled_launch_args) == ROCM_VECTOR_ADD_SCALED_LAUNCH_ARGS_BYTES, "vector add-scaled launch ABI drift");
static_assert(sizeof(rocm_vector_scale_launch_args) == ROCM_VECTOR_SCALE_LAUNCH_ARGS_BYTES, "vector scale launch ABI drift");
static_assert(sizeof(rocm_per_layer_input_transpose_launch_args) == ROCM_PER_LAYER_INPUT_TRANSPOSE_LAUNCH_ARGS_BYTES, "per-layer input transpose launch ABI drift");
static_assert(sizeof(rocm_swiglu_launch_args) == ROCM_SWIGLU_LAUNCH_ARGS_BYTES, "SwiGLU launch ABI drift");
static_assert(sizeof(rocm_gelu_tanh_mul_launch_args) == ROCM_GELU_TANH_MUL_LAUNCH_ARGS_BYTES, "GELU tanh multiply launch ABI drift");
static_assert(sizeof(rocm_moe_router_launch_args) == ROCM_MOE_ROUTER_LAUNCH_ARGS_BYTES, "MoE router launch ABI drift");
static_assert(sizeof(rocm_moe_lazy_launch_args) == ROCM_MOE_LAZY_LAUNCH_ARGS_BYTES, "MoE lazy expert launch ABI drift");
static_assert(sizeof(rocm_moe_batch_route_metadata) == ROCM_MOE_BATCH_ROUTE_METADATA_BYTES, "MoE batch route metadata ABI drift");
static_assert(sizeof(rocm_moe_batch_route_rows_launch_args) == ROCM_MOE_BATCH_ROUTE_ROWS_LAUNCH_ARGS_BYTES, "MoE batch route rows launch ABI drift");
static_assert(sizeof(rocm_moe_batch_reduce_launch_args) == ROCM_MOE_BATCH_REDUCE_LAUNCH_ARGS_BYTES, "MoE batch reduce launch ABI drift");
static_assert(sizeof(rocm_moe_mlx_affine_routes_launch_args) == ROCM_MOE_MLX_AFFINE_ROUTES_LAUNCH_ARGS_BYTES, "MoE MLX affine routes launch ABI drift");
static_assert(sizeof(rocm_moe_mlx_affine_routes_chunk) == ROCM_MOE_MLX_AFFINE_ROUTES_CHUNK_BYTES, "MoE MLX affine routes chunk ABI drift");
static_assert(sizeof(rocm_gguf_q4_0_projection_launch_args) == ROCM_GGUF_Q4_0_PROJECTION_LAUNCH_ARGS_BYTES, "GGUF Q4_0 projection launch ABI drift");
static_assert(sizeof(rocm_gguf_q4_k_expand_launch_args) == ROCM_GGUF_Q4_K_EXPAND_LAUNCH_ARGS_BYTES, "GGUF Q4_K expansion launch ABI drift");
static_assert(sizeof(rocm_gguf_q4_0_selected_experts_launch_args) == ROCM_GGUF_Q4_0_SELECTED_EXPERTS_LAUNCH_ARGS_BYTES, "GGUF Q4_0 selected experts launch ABI drift");
static_assert(sizeof(rocm_jangtq_launch_args) == ROCM_JANGTQ_LAUNCH_ARGS_BYTES, "JANGTQ launch ABI drift");
static_assert(sizeof(rocm_codebook_launch_args) == ROCM_CODEBOOK_LAUNCH_ARGS_BYTES, "codebook launch ABI drift");
static_assert(sizeof(rocm_lora_launch_args) == ROCM_LORA_LAUNCH_ARGS_BYTES, "LoRA launch ABI drift");
static_assert(sizeof(rocm_embedding_lookup_launch_args) == ROCM_EMBEDDING_LOOKUP_LAUNCH_ARGS_BYTES, "embedding lookup launch ABI drift");
static_assert(sizeof(rocm_diffusion_expected_embedding_launch_args) == ROCM_DIFFUSION_EXPECTED_EMBEDDING_LAUNCH_ARGS_BYTES, "diffusion expected embedding launch ABI drift");
static_assert(sizeof(rocm_diffusion_sample_launch_args) == ROCM_DIFFUSION_SAMPLE_LAUNCH_ARGS_BYTES, "diffusion sample launch ABI drift");
static_assert(sizeof(rocm_diffusion_sample_result) == ROCM_DIFFUSION_SAMPLE_RESULT_BYTES, "diffusion sample result ABI drift");
static_assert(sizeof(rocm_embedding_mean_pool_launch_args) == ROCM_EMBEDDING_MEAN_POOL_LAUNCH_ARGS_BYTES, "embedding mean-pool launch ABI drift");
static_assert(sizeof(rocm_rerank_cosine_launch_args) == ROCM_RERANK_COSINE_LAUNCH_ARGS_BYTES, "rerank cosine launch ABI drift");
static_assert(sizeof(rocm_tiny_prefill_launch_args) == ROCM_TINY_PREFILL_LAUNCH_ARGS_BYTES, "tiny prefill launch ABI drift");
static_assert(sizeof(rocm_tiny_decode_launch_args) == ROCM_TINY_DECODE_LAUNCH_ARGS_BYTES, "tiny decode launch ABI drift");
static_assert(sizeof(rocm_cross_entropy_loss_launch_args) == ROCM_CROSS_ENTROPY_LOSS_LAUNCH_ARGS_BYTES, "cross entropy loss launch ABI drift");
static_assert(sizeof(rocm_distillation_kl_loss_launch_args) == ROCM_DISTILLATION_KL_LOSS_LAUNCH_ARGS_BYTES, "distillation KL loss launch ABI drift");
static_assert(sizeof(rocm_grpo_advantage_launch_args) == ROCM_GRPO_ADVANTAGE_LAUNCH_ARGS_BYTES, "GRPO advantage launch ABI drift");
static_assert(sizeof(rocm_autoround_quantize_launch_args) == ROCM_AUTOROUND_QUANTIZE_LAUNCH_ARGS_BYTES, "AutoRound quantize launch ABI drift");

__device__ float rocm_float_from_bits(uint32_t bits)
{
	union {
		uint32_t u;
		float f;
	} value;
	value.u = bits;
	return value.f;
}

__device__ uint32_t rocm_float_bits(float value)
{
	union {
		float f;
		uint32_t u;
	} bits;
	bits.f = value;
	return bits.u;
}

__device__ double rocm_double_from_bits(uint64_t bits)
{
	union {
		uint64_t u;
		double f;
	} value;
	value.u = bits;
	return value.f;
}

__device__ float rocm_half_to_float(uint16_t half)
{
	const uint32_t sign = (static_cast<uint32_t>(half & 0x8000u)) << 16;
	uint32_t exponent = (half >> 10) & 0x1fu;
	uint32_t mantissa = half & 0x03ffu;

	if (exponent == 0) {
		if (mantissa == 0) {
			return rocm_float_from_bits(sign);
		}
		while ((mantissa & 0x0400u) == 0) {
			mantissa <<= 1;
			--exponent;
		}
		++exponent;
		mantissa &= 0x03ffu;
	} else if (exponent == 31) {
		return rocm_float_from_bits(sign | 0x7f800000u | (mantissa << 13));
	}

	exponent = exponent + (127u - 15u);
	return rocm_float_from_bits(sign | (exponent << 23) | (mantissa << 13));
}

__device__ uint16_t rocm_float_to_half(float value)
{
	const uint32_t bits = rocm_float_bits(value);
	const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
	const int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
	const uint32_t mantissa = bits & 0x7fffffu;
	if (exponent <= 0) {
		return sign;
	}
	if (exponent >= 0x1f) {
		return static_cast<uint16_t>(sign | 0x7c00u);
	}
	return static_cast<uint16_t>(sign | static_cast<uint16_t>(exponent << 10) | static_cast<uint16_t>(mantissa >> 13));
}

__device__ float rocm_bfloat16_to_float(uint16_t value)
{
	return rocm_float_from_bits(static_cast<uint32_t>(value) << 16);
}

__device__ uint16_t rocm_float_to_bfloat16(float value)
{
	uint32_t bits = rocm_float_bits(value);
	bits += 0x7fffu + ((bits >> 16u) & 1u);
	return static_cast<uint16_t>(bits >> 16u);
}

__device__ uint32_t rocm_ordered_float_key(float value)
{
	const uint32_t bits = rocm_float_bits(value);
	if ((bits & 0x80000000u) != 0u) {
		return ~bits;
	}
	return bits ^ 0x80000000u;
}

__device__ uint64_t rocm_pack_score_index(float score, uint32_t index)
{
	return (static_cast<uint64_t>(rocm_ordered_float_key(score)) << 32) | static_cast<uint64_t>(~index);
}

__device__ float rocm_score_from_ordered_key(uint32_t key)
{
	if ((key & 0x80000000u) != 0u) {
		return rocm_float_from_bits(key ^ 0x80000000u);
	}
	return rocm_float_from_bits(~key);
}

__device__ double rocm_packed_topk_sample_score(unsigned long long packed, double softcap)
{
	const double score = static_cast<double>(rocm_score_from_ordered_key(static_cast<uint32_t>(packed >> 32)));
	return softcap > 0.0 ? tanh(score / softcap) * softcap : score;
}

__device__ bool rocm_supported_kv_mode(uint32_t mode_code)
{
	return mode_code >= 1 && mode_code <= 3;
}

__device__ bool rocm_positive_finite(float value)
{
	return value > 0.0f && isfinite(value);
}

__device__ bool rocm_valid_prefill_args(const rocm_prefill_launch_args &args)
{
	return args.version == ROCM_PREFILL_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_PREFILL_LAUNCH_ARGS_BYTES &&
		args.token_pointer != 0 &&
		args.token_count > 0 &&
		args.token_bytes == args.token_count * sizeof(int32_t) &&
		rocm_supported_kv_mode(args.mode_code) &&
		args.block_size > 0 &&
		args.key_width > 0 &&
		args.value_width > 0;
}

__device__ bool rocm_valid_decode_args(const rocm_decode_launch_args &args)
{
	return args.version == ROCM_DECODE_LAUNCH_ARGS_VERSION &&
		args.header_bytes == ROCM_DECODE_LAUNCH_ARGS_HEADER_BYTES &&
		args.total_bytes == ROCM_DECODE_LAUNCH_ARGS_BYTES &&
		args.kv_descriptor_bytes == ROCM_DEVICE_KV_LAUNCH_DESCRIPTOR_BYTES &&
		args.kv.descriptor_pointer != 0 &&
		args.kv.descriptor_bytes >= 32 &&
		args.kv.descriptor_version == ROCM_DEVICE_KV_DESCRIPTOR_VERSION &&
		rocm_supported_kv_mode(args.kv.mode_code) &&
		args.kv.block_size > 0 &&
		args.kv.page_count > 0 &&
		args.kv.token_count > 0 &&
		args.position == args.kv.token_count &&
		args.kv.key_width > 0 &&
		args.kv.value_width > 0;
}

__device__ bool rocm_valid_projection_args(const rocm_projection_launch_args &args)
{
	if (args.version != ROCM_PROJECTION_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_PROJECTION_LAUNCH_ARGS_BYTES ||
		args.input_pointer == 0 ||
		args.weight_pointer == 0 ||
		args.output_pointer == 0 ||
		args.rows == 0 ||
		args.cols == 0 ||
		args.input_count != args.cols ||
		args.input_bytes != args.cols * sizeof(float) ||
		args.output_bytes != args.rows * sizeof(float)) {
		return false;
	}
	if (args.weight_encoding == ROCM_PROJECTION_WEIGHT_ENCODING_FP16 ||
		args.weight_encoding == ROCM_PROJECTION_WEIGHT_ENCODING_BF16) {
		return args.weight_bytes == static_cast<uint64_t>(args.rows) * args.cols * sizeof(uint16_t);
	}
	if (args.weight_encoding == ROCM_PROJECTION_WEIGHT_ENCODING_Q8) {
		const float scale = rocm_float_from_bits(args.q8_scale_bits);
		return args.weight_bytes == static_cast<uint64_t>(args.rows) * args.cols &&
			rocm_positive_finite(scale);
	}
	if (args.weight_encoding == ROCM_PROJECTION_WEIGHT_ENCODING_F32) {
		return args.weight_bytes == static_cast<uint64_t>(args.rows) * args.cols * sizeof(float);
	}
	return false;
}

__device__ bool rocm_valid_projection_batch_args(const rocm_projection_batch_launch_args &args)
{
	if (args.version != ROCM_PROJECTION_BATCH_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_PROJECTION_BATCH_LAUNCH_ARGS_BYTES ||
		args.input_pointer == 0 ||
		args.weight_pointer == 0 ||
		args.output_pointer == 0 ||
		args.rows == 0 ||
		args.cols == 0 ||
		args.batch == 0 ||
		args.input_bytes != static_cast<uint64_t>(args.batch) * args.cols * sizeof(float) ||
		args.output_bytes != static_cast<uint64_t>(args.batch) * args.rows * sizeof(float)) {
		return false;
	}
	if ((args.flags & ROCM_PROJECTION_LAUNCH_FLAG_BIAS) != 0 &&
		(args.bias_pointer == 0 || args.bias_bytes != args.rows * sizeof(float))) {
		return false;
	}
	if ((args.flags & ROCM_PROJECTION_LAUNCH_FLAG_BIAS) == 0 &&
		(args.bias_pointer != 0 || args.bias_bytes != 0)) {
		return false;
	}
	if (args.weight_encoding == ROCM_PROJECTION_WEIGHT_ENCODING_FP16 ||
		args.weight_encoding == ROCM_PROJECTION_WEIGHT_ENCODING_BF16) {
		return args.weight_bytes == static_cast<uint64_t>(args.rows) * args.cols * sizeof(uint16_t);
	}
	if (args.weight_encoding == ROCM_PROJECTION_WEIGHT_ENCODING_Q8) {
		const float scale = rocm_float_from_bits(args.q8_scale_bits);
		return args.weight_bytes == static_cast<uint64_t>(args.rows) * args.cols &&
			rocm_positive_finite(scale);
	}
	if (args.weight_encoding == ROCM_PROJECTION_WEIGHT_ENCODING_F32) {
		return args.weight_bytes == static_cast<uint64_t>(args.rows) * args.cols * sizeof(float);
	}
	return false;
}

__device__ bool rocm_mlx_affine_bits_supported(uint32_t bits);
__device__ uint64_t rocm_mlx_affine_packed_per_row(uint32_t cols, uint32_t bits);
__device__ bool rocm_mlx_affine_shape_valid(
	uint32_t rows,
	uint32_t cols,
	uint32_t group_size,
	uint32_t bits,
	uint64_t weight_bytes,
	uint64_t scale_bytes,
	uint64_t bias_bytes);

__device__ bool rocm_valid_mlx_q4_projection_args(const rocm_mlx_q4_projection_launch_args &args)
{
	if (args.version != ROCM_MLX_Q4_PROJECTION_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_MLX_Q4_PROJECTION_LAUNCH_ARGS_BYTES ||
		args.input_pointer == 0 ||
		args.weight_pointer == 0 ||
		args.scale_pointer == 0 ||
		args.bias_pointer == 0 ||
		args.output_pointer == 0 ||
		args.rows == 0 ||
		args.cols == 0 ||
		args.group_size == 0 ||
		args.input_bytes != args.cols * sizeof(float) ||
		args.output_bytes != args.rows * sizeof(float)) {
		return false;
	}
	return rocm_mlx_affine_shape_valid(args.rows, args.cols, args.group_size, args.bits, args.weight_bytes, args.scale_bytes, args.bias_bytes);
}

__device__ bool rocm_valid_mlx_q4_projection_batch_args(const rocm_mlx_q4_projection_batch_launch_args &args)
{
	if (args.version != ROCM_MLX_Q4_PROJECTION_BATCH_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_MLX_Q4_PROJECTION_BATCH_LAUNCH_ARGS_BYTES ||
		args.input_pointer == 0 ||
		args.weight_pointer == 0 ||
		args.scale_pointer == 0 ||
		args.bias_pointer == 0 ||
		args.output_pointer == 0 ||
		args.rows == 0 ||
		args.cols == 0 ||
		args.batch == 0 ||
		args.group_size == 0 ||
		args.input_bytes != args.batch * args.cols * sizeof(float) ||
		args.output_bytes != args.batch * args.rows * sizeof(float)) {
		return false;
	}
	return rocm_mlx_affine_shape_valid(args.rows, args.cols, args.group_size, args.bits, args.weight_bytes, args.scale_bytes, args.bias_bytes);
}

__device__ bool rocm_valid_mlx_q4_projection_greedy_args(const rocm_mlx_q4_projection_launch_args &args)
{
	if (args.version != ROCM_MLX_Q4_PROJECTION_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_MLX_Q4_PROJECTION_LAUNCH_ARGS_BYTES ||
		args.input_pointer == 0 ||
		args.weight_pointer == 0 ||
		args.scale_pointer == 0 ||
		args.bias_pointer == 0 ||
		args.output_pointer == 0 ||
		args.rows == 0 ||
		args.cols == 0 ||
		args.group_size == 0 ||
		args.input_bytes != args.cols * sizeof(float) ||
		args.output_bytes != ROCM_MLX_Q4_PROJECTION_BEST_BYTES) {
		return false;
	}
	if (args.suppress_count != 0 && args.suppress_pointer == 0) {
		return false;
	}
	return rocm_mlx_affine_shape_valid(args.rows, args.cols, args.group_size, args.bits, args.weight_bytes, args.scale_bytes, args.bias_bytes);
}

__device__ bool rocm_valid_mlx_q4_projection_greedy_batch_args(const rocm_mlx_q4_projection_greedy_batch_launch_args &args)
{
	if (args.version != ROCM_MLX_Q4_PROJECTION_GREEDY_BATCH_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_MLX_Q4_PROJECTION_GREEDY_BATCH_LAUNCH_ARGS_BYTES ||
		args.input_pointer == 0 ||
		args.weight_pointer == 0 ||
		args.scale_pointer == 0 ||
		args.bias_pointer == 0 ||
		args.output_pointer == 0 ||
		args.rows == 0 ||
		args.cols == 0 ||
		args.batch == 0 ||
		args.group_size == 0 ||
		args.input_bytes != args.batch * args.cols * sizeof(float) ||
		args.output_bytes != args.batch * ROCM_MLX_Q4_PROJECTION_BEST_BYTES) {
		return false;
	}
	if (args.suppress_count != 0 && args.suppress_pointer == 0) {
		return false;
	}
	return rocm_mlx_affine_shape_valid(args.rows, args.cols, args.group_size, args.bits, args.weight_bytes, args.scale_bytes, args.bias_bytes);
}

__device__ bool rocm_valid_mlx_q4_projection_scores_args(const rocm_mlx_q4_projection_launch_args &args)
{
	if (args.version != ROCM_MLX_Q4_PROJECTION_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_MLX_Q4_PROJECTION_LAUNCH_ARGS_BYTES ||
		args.input_pointer == 0 ||
		args.weight_pointer == 0 ||
		args.scale_pointer == 0 ||
		args.bias_pointer == 0 ||
		args.output_pointer == 0 ||
		args.rows == 0 ||
		args.cols == 0 ||
		args.group_size == 0 ||
		args.input_bytes != args.cols * sizeof(float) ||
		args.output_bytes != args.rows * ROCM_MLX_Q4_PROJECTION_BEST_BYTES) {
		return false;
	}
	if (args.suppress_count != 0 && args.suppress_pointer == 0) {
		return false;
	}
	return rocm_mlx_affine_shape_valid(args.rows, args.cols, args.group_size, args.bits, args.weight_bytes, args.scale_bytes, args.bias_bytes);
}

__device__ bool rocm_valid_mlx_q4_projection_selected_greedy_args(const rocm_mlx_q4_projection_launch_args &args)
{
	if (args.version != ROCM_MLX_Q4_PROJECTION_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_MLX_Q4_PROJECTION_LAUNCH_ARGS_BYTES ||
		args.input_pointer == 0 ||
		args.weight_pointer == 0 ||
		args.scale_pointer == 0 ||
		args.bias_pointer == 0 ||
		args.output_pointer == 0 ||
		args.suppress_pointer == 0 ||
		args.suppress_count == 0 ||
		args.rows == 0 ||
		args.cols == 0 ||
		args.group_size == 0 ||
		args.input_bytes != args.cols * sizeof(float) ||
		args.output_bytes != ROCM_MLX_Q4_PROJECTION_BEST_BYTES) {
		return false;
	}
	return rocm_mlx_affine_shape_valid(args.rows, args.cols, args.group_size, args.bits, args.weight_bytes, args.scale_bytes, args.bias_bytes);
}

__device__ bool rocm_valid_packed_topk_args(const rocm_packed_topk_launch_args &args)
{
	if (args.version != ROCM_PACKED_TOPK_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_PACKED_TOPK_LAUNCH_ARGS_BYTES ||
		args.input_pointer == 0 ||
		args.output_pointer == 0 ||
		args.input_count == 0 ||
		args.output_count == 0 ||
		args.top_k == 0 ||
		args.top_k > ROCM_PACKED_TOPK_MAX_K ||
		args.chunk_size != ROCM_PACKED_TOPK_CHUNK_SIZE ||
		args.input_bytes != args.input_count * ROCM_MLX_Q4_PROJECTION_BEST_BYTES ||
		args.output_bytes != args.output_count * ROCM_MLX_Q4_PROJECTION_BEST_BYTES) {
		return false;
	}
	const uint32_t chunk_count = (args.input_count + args.chunk_size - 1u) / args.chunk_size;
	return args.output_count == chunk_count * args.top_k;
}

__device__ bool rocm_valid_ordered_embedding_candidates_args(const rocm_ordered_embedding_candidates_launch_args &args)
{
	if (args.version != ROCM_ORDERED_EMBEDDING_CANDIDATES_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_ORDERED_EMBEDDING_CANDIDATES_LAUNCH_ARGS_BYTES ||
		args.topk_pointer == 0 ||
		args.token_ordering_pointer == 0 ||
		args.output_pointer == 0 ||
		args.topk_count == 0 ||
		args.topk_count > ROCM_PACKED_TOPK_MAX_K ||
		args.num_centroids == 0 ||
		args.tokens_per_centroid == 0 ||
		args.token_ordering_count == 0 ||
		args.output_count == 0 ||
		args.output_count != args.topk_count * args.tokens_per_centroid ||
		args.token_ordering_count != args.num_centroids * args.tokens_per_centroid ||
		args.topk_bytes != args.topk_count * ROCM_MLX_Q4_PROJECTION_BEST_BYTES ||
		args.output_bytes != args.output_count * sizeof(int32_t)) {
		return false;
	}
	if (args.token_ordering_element_bytes != sizeof(int32_t) && args.token_ordering_element_bytes != sizeof(int64_t)) {
		return false;
	}
	if (args.token_ordering_bytes != args.token_ordering_count * args.token_ordering_element_bytes) {
		return false;
	}
	if (args.suppress_count != 0 && args.suppress_pointer == 0) {
		return false;
	}
	return true;
}

__device__ bool rocm_valid_packed_topk_sample_args(const rocm_packed_topk_sample_launch_args &args)
{
	const float temperature = rocm_float_from_bits(args.temperature_bits);
	const float top_p = rocm_float_from_bits(args.top_p_bits);
	const double draw = rocm_double_from_bits(args.draw_bits);
	const float softcap = rocm_float_from_bits(args.softcap_bits);
	return args.version == ROCM_PACKED_TOPK_SAMPLE_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_PACKED_TOPK_SAMPLE_LAUNCH_ARGS_BYTES &&
		args.input_pointer != 0 &&
		args.output_pointer != 0 &&
		args.input_count > 0 &&
		args.top_k > 0 &&
		args.top_k <= args.input_count &&
		args.top_k <= ROCM_PACKED_TOPK_MAX_K &&
		args.input_bytes == args.input_count * ROCM_MLX_Q4_PROJECTION_BEST_BYTES &&
		args.output_bytes == ROCM_MLX_Q4_PROJECTION_BEST_BYTES &&
		temperature >= 0.0f &&
		isfinite(temperature) &&
		top_p >= 0.0f &&
		top_p <= 1.0f &&
		isfinite(top_p) &&
		softcap >= 0.0f &&
		isfinite(softcap) &&
		isfinite(draw);
}

__device__ bool rocm_mlx_q4_token_suppressed(uint32_t row, const int32_t *suppress_tokens, uint32_t suppress_count)
{
	if (suppress_tokens == nullptr || suppress_count == 0) {
		return false;
	}
	const int32_t token = static_cast<int32_t>(row);
	for (uint32_t index = 0; index < suppress_count; ++index) {
		if (suppress_tokens[index] == token) {
			return true;
		}
	}
	return false;
}

__device__ bool rocm_valid_mlx_q4_triple_projection_part(uint32_t rows, uint32_t cols, uint32_t group_size, uint32_t bits, uint32_t weight_bytes, uint32_t scale_bytes, uint32_t bias_bytes)
{
	if (rows == 0) {
		return false;
	}
	return rocm_mlx_affine_shape_valid(rows, cols, group_size, bits, weight_bytes, scale_bytes, bias_bytes);
}

__device__ bool rocm_valid_mlx_q4_triple_projection_args(const rocm_mlx_q4_triple_projection_launch_args &args)
{
	const uint64_t total_rows = static_cast<uint64_t>(args.first_rows) + args.second_rows + args.third_rows;
	if (args.version != ROCM_MLX_Q4_TRIPLE_PROJECTION_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_MLX_Q4_TRIPLE_PROJECTION_LAUNCH_ARGS_BYTES ||
		args.input_pointer == 0 ||
		args.output_pointer == 0 ||
		args.first_weight_pointer == 0 ||
		args.first_scale_pointer == 0 ||
		args.first_bias_pointer == 0 ||
		args.second_weight_pointer == 0 ||
		args.second_scale_pointer == 0 ||
		args.second_bias_pointer == 0 ||
		args.cols == 0 ||
		args.group_size == 0 ||
		args.input_bytes != args.cols * sizeof(float) ||
		static_cast<uint64_t>(args.output_bytes) != total_rows * sizeof(float)) {
		return false;
	}
	if (args.third_rows > 0 &&
		(args.third_weight_pointer == 0 ||
			args.third_scale_pointer == 0 ||
			args.third_bias_pointer == 0)) {
		return false;
	}
	if (args.third_rows == 0 &&
		(args.third_weight_bytes != 0 ||
			args.third_scale_bytes != 0 ||
			args.third_bias_bytes != 0)) {
		return false;
	}
	return rocm_valid_mlx_q4_triple_projection_part(args.first_rows, args.cols, args.group_size, args.bits, args.first_weight_bytes, args.first_scale_bytes, args.first_bias_bytes) &&
		rocm_valid_mlx_q4_triple_projection_part(args.second_rows, args.cols, args.group_size, args.bits, args.second_weight_bytes, args.second_scale_bytes, args.second_bias_bytes) &&
		(args.third_rows == 0 || rocm_valid_mlx_q4_triple_projection_part(args.third_rows, args.cols, args.group_size, args.bits, args.third_weight_bytes, args.third_scale_bytes, args.third_bias_bytes));
}

__device__ bool rocm_valid_mlx_q4_gelu_tanh_mul_args(const rocm_mlx_q4_gelu_tanh_mul_launch_args &args)
{
	if (args.version != ROCM_MLX_Q4_GELU_TANH_MUL_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_MLX_Q4_GELU_TANH_MUL_LAUNCH_ARGS_BYTES ||
		args.input_pointer == 0 ||
		args.gate_weight_pointer == 0 ||
		args.gate_scale_pointer == 0 ||
		args.gate_bias_pointer == 0 ||
		args.up_weight_pointer == 0 ||
		args.up_scale_pointer == 0 ||
		args.up_bias_pointer == 0 ||
		args.output_pointer == 0 ||
		args.rows == 0 ||
		args.cols == 0 ||
		args.group_size == 0 ||
		args.input_bytes != args.cols * sizeof(float) ||
		args.output_bytes != args.rows * sizeof(float)) {
		return false;
	}
	return rocm_mlx_affine_shape_valid(args.rows, args.cols, args.group_size, args.bits, args.gate_weight_bytes, args.gate_scale_bytes, args.gate_bias_bytes) &&
		rocm_mlx_affine_shape_valid(args.rows, args.cols, args.group_size, args.bits, args.up_weight_bytes, args.up_scale_bytes, args.up_bias_bytes);
}

__device__ bool rocm_valid_mlx_q4_gelu_tanh_mul_batch_args(const rocm_mlx_q4_gelu_tanh_mul_batch_launch_args &args)
{
	if (args.version != ROCM_MLX_Q4_GELU_TANH_MUL_BATCH_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_MLX_Q4_GELU_TANH_MUL_BATCH_LAUNCH_ARGS_BYTES ||
		args.input_pointer == 0 ||
		args.gate_weight_pointer == 0 ||
		args.gate_scale_pointer == 0 ||
		args.gate_bias_pointer == 0 ||
		args.up_weight_pointer == 0 ||
		args.up_scale_pointer == 0 ||
		args.up_bias_pointer == 0 ||
		args.output_pointer == 0 ||
		args.rows == 0 ||
		args.cols == 0 ||
		args.batch == 0 ||
		args.group_size == 0 ||
		args.input_bytes != args.batch * args.cols * sizeof(float) ||
		args.output_bytes != args.batch * args.rows * sizeof(float)) {
		return false;
	}
	return rocm_mlx_affine_shape_valid(args.rows, args.cols, args.group_size, args.bits, args.gate_weight_bytes, args.gate_scale_bytes, args.gate_bias_bytes) &&
		rocm_mlx_affine_shape_valid(args.rows, args.cols, args.group_size, args.bits, args.up_weight_bytes, args.up_scale_bytes, args.up_bias_bytes);
}

__device__ bool rocm_valid_mlx_q4_gelu_tanh_mlp_persistent_args(const rocm_mlx_q4_gelu_tanh_mlp_persistent_launch_args &args)
{
	if (args.version != ROCM_MLX_Q4_GELU_TANH_MLP_PERSISTENT_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_MLX_Q4_GELU_TANH_MLP_PERSISTENT_LAUNCH_ARGS_BYTES ||
		args.input_pointer == 0 ||
		args.gate_weight_pointer == 0 ||
		args.gate_scale_pointer == 0 ||
		args.gate_bias_pointer == 0 ||
		args.up_weight_pointer == 0 ||
		args.up_scale_pointer == 0 ||
		args.up_bias_pointer == 0 ||
		args.down_weight_pointer == 0 ||
		args.down_scale_pointer == 0 ||
		args.down_bias_pointer == 0 ||
		args.activation_pointer == 0 ||
		args.output_pointer == 0 ||
		args.barrier_pointer == 0 ||
		args.rows == 0 ||
		args.cols != 1536u ||
		args.down_rows != 1536u ||
		args.group_size != 32u ||
		args.bits != 4u ||
		args.input_bytes != args.cols * sizeof(float) ||
		args.activation_bytes != args.rows * sizeof(float) ||
		args.output_bytes != args.down_rows * sizeof(float) ||
		args.barrier_bytes != 8u) {
		return false;
	}
	return rocm_mlx_affine_shape_valid(args.rows, args.cols, args.group_size, args.bits, args.gate_weight_bytes, args.gate_scale_bytes, args.gate_bias_bytes) &&
		rocm_mlx_affine_shape_valid(args.rows, args.cols, args.group_size, args.bits, args.up_weight_bytes, args.up_scale_bytes, args.up_bias_bytes) &&
		rocm_mlx_affine_shape_valid(args.down_rows, args.rows, args.group_size, args.bits, args.down_weight_bytes, args.down_scale_bytes, args.down_bias_bytes);
}

__device__ bool rocm_valid_mlx_q4_gelu_tanh_proj_args(const rocm_mlx_q4_gelu_tanh_proj_launch_args &args)
{
	if (args.version != ROCM_MLX_Q4_GELU_TANH_PROJ_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_MLX_Q4_GELU_TANH_PROJ_LAUNCH_ARGS_BYTES ||
		args.input_pointer == 0 ||
		args.weight_pointer == 0 ||
		args.scale_pointer == 0 ||
		args.bias_pointer == 0 ||
		args.multiplier_pointer == 0 ||
		args.output_pointer == 0 ||
		args.rows == 0 ||
		args.cols == 0 ||
		args.group_size == 0 ||
		args.input_bytes != args.cols * sizeof(float) ||
		args.multiplier_bytes != args.rows * sizeof(float) ||
		args.output_bytes != args.rows * sizeof(float)) {
		return false;
	}
	return rocm_mlx_affine_shape_valid(args.rows, args.cols, args.group_size, args.bits, args.weight_bytes, args.scale_bytes, args.bias_bytes);
}

__device__ bool rocm_valid_mlx_q4_gelu_tanh_proj_batch_args(const rocm_mlx_q4_gelu_tanh_proj_batch_launch_args &args)
{
	if (args.version != ROCM_MLX_Q4_GELU_TANH_PROJ_BATCH_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_MLX_Q4_GELU_TANH_PROJ_BATCH_LAUNCH_ARGS_BYTES ||
		args.input_pointer == 0 ||
		args.weight_pointer == 0 ||
		args.scale_pointer == 0 ||
		args.bias_pointer == 0 ||
		args.multiplier_pointer == 0 ||
		args.output_pointer == 0 ||
		args.rows == 0 ||
		args.cols == 0 ||
		args.batch == 0 ||
		args.group_size == 0 ||
		args.input_bytes != args.batch * args.cols * sizeof(float) ||
		args.multiplier_bytes != args.batch * args.rows * sizeof(float) ||
		args.output_bytes != args.batch * args.rows * sizeof(float)) {
		return false;
	}
	return rocm_mlx_affine_shape_valid(args.rows, args.cols, args.group_size, args.bits, args.weight_bytes, args.scale_bytes, args.bias_bytes);
}

__device__ bool rocm_valid_rms_norm_weight_config(uint64_t weight_pointer, uint32_t weight_bytes, uint32_t count, uint32_t weight_encoding, uint32_t flags);

__device__ bool rocm_valid_rms_norm_residual_add_gelu_tanh_proj_args(const rocm_rms_norm_residual_add_gelu_tanh_proj_launch_args &args)
{
	const float epsilon = rocm_float_from_bits(args.epsilon_bits);
	const float output_scale = rocm_float_from_bits(args.output_scale_bits);
	if (args.version != ROCM_RMS_NORM_RESIDUAL_ADD_GELU_TANH_PROJ_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_RMS_NORM_RESIDUAL_ADD_GELU_TANH_PROJ_LAUNCH_ARGS_BYTES ||
		args.input_pointer == 0 ||
		args.residual_pointer == 0 ||
		args.weight_pointer == 0 ||
		args.scale_pointer == 0 ||
		args.bias_pointer == 0 ||
		args.multiplier_pointer == 0 ||
		args.residual_output_pointer == 0 ||
		args.activation_output_pointer == 0 ||
		args.rows == 0 ||
		args.cols == 0 ||
		args.group_size == 0 ||
		args.input_bytes != args.cols * sizeof(float) ||
		args.residual_bytes != args.cols * sizeof(float) ||
		args.residual_output_bytes != args.cols * sizeof(float) ||
		args.multiplier_bytes != args.rows * sizeof(float) ||
		args.activation_output_bytes != args.rows * sizeof(float) ||
		!(epsilon >= 0.0f) ||
		!isfinite(output_scale)) {
		return false;
	}
	return rocm_valid_rms_norm_weight_config(args.rms_weight_pointer, args.rms_weight_bytes, args.cols, args.rms_weight_encoding, args.rms_flags) &&
		rocm_mlx_affine_shape_valid(args.rows, args.cols, args.group_size, args.bits, args.weight_bytes, args.scale_bytes, args.bias_bytes);
}

__device__ bool rocm_valid_lora_args(const rocm_lora_launch_args &args)
{
	const bool has_bias = (args.flags & ROCM_LORA_LAUNCH_FLAG_BIAS) != 0;
	bool valid_bias = args.bias_pointer == 0 && args.bias_bytes == 0;
	if (has_bias) {
		valid_bias = args.bias_pointer != 0 && args.bias_bytes == args.rows * sizeof(float);
	}
	return args.version == ROCM_LORA_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_LORA_LAUNCH_ARGS_BYTES &&
		args.input_pointer != 0 &&
		args.base_weight_pointer != 0 &&
		args.lora_a_pointer != 0 &&
		args.lora_b_pointer != 0 &&
		args.output_pointer != 0 &&
		args.rows > 0 &&
		args.cols > 0 &&
		args.rank > 0 &&
		args.input_count == args.cols &&
		args.input_bytes == args.cols * sizeof(float) &&
		args.base_weight_bytes == args.rows * args.cols * sizeof(float) &&
		args.lora_a_bytes == args.rank * args.cols * sizeof(float) &&
		args.lora_b_bytes == args.rows * args.rank * sizeof(float) &&
		args.output_bytes == args.rows * sizeof(float) &&
		rocm_positive_finite(rocm_float_from_bits(args.alpha_bits)) &&
		valid_bias;
}

__device__ bool rocm_valid_embedding_mean_pool_args(const rocm_embedding_mean_pool_launch_args &args)
{
	return args.version == ROCM_EMBEDDING_MEAN_POOL_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_EMBEDDING_MEAN_POOL_LAUNCH_ARGS_BYTES &&
		args.token_pointer != 0 &&
		args.output_pointer != 0 &&
		args.token_count > 0 &&
		args.dim > 0 &&
		args.token_bytes == args.token_count * args.dim * sizeof(float) &&
		args.output_bytes == args.dim * sizeof(float);
}

__device__ bool rocm_mlx_affine_bits_supported(uint32_t bits)
{
	return bits == 4u || bits == 6u || bits == 8u;
}

__device__ uint64_t rocm_mlx_affine_packed_per_row(uint32_t cols, uint32_t bits)
{
	return static_cast<uint64_t>(cols) * bits / 32u;
}

__device__ bool rocm_mlx_affine_shape_valid(
	uint32_t rows,
	uint32_t cols,
	uint32_t group_size,
	uint32_t bits,
	uint64_t weight_bytes,
	uint64_t scale_bytes,
	uint64_t bias_bytes)
{
	if (rows == 0 || cols == 0 || group_size == 0 || !rocm_mlx_affine_bits_supported(bits)) {
		return false;
	}
	const uint64_t total_bits = static_cast<uint64_t>(cols) * bits;
	if ((total_bits % 32u) != 0 || (cols % group_size) != 0) {
		return false;
	}
	const uint64_t packed_per_row = total_bits / 32u;
	const uint64_t groups_per_row = cols / group_size;
	return weight_bytes == static_cast<uint64_t>(rows) * packed_per_row * sizeof(uint32_t) &&
		scale_bytes == static_cast<uint64_t>(rows) * groups_per_row * sizeof(uint16_t) &&
		bias_bytes == static_cast<uint64_t>(rows) * groups_per_row * sizeof(uint16_t);
}

__device__ bool rocm_valid_embedding_lookup_args(const rocm_embedding_lookup_launch_args &args)
{
	const float output_scale = args.output_scale_bits == 0 ? 1.0f : rocm_float_from_bits(args.output_scale_bits);
	if (args.version != ROCM_EMBEDDING_LOOKUP_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_EMBEDDING_LOOKUP_LAUNCH_ARGS_BYTES ||
		args.token_pointer == 0 ||
		args.embedding_pointer == 0 ||
		args.output_pointer == 0 ||
		args.token_count == 0 ||
		args.vocab_size == 0 ||
		args.hidden_size == 0 ||
		args.token_bytes != args.token_count * sizeof(int32_t) ||
		args.output_bytes != static_cast<uint64_t>(args.token_count) * args.hidden_size * sizeof(float) ||
		!isfinite(output_scale)) {
		return false;
	}
	if (args.table_encoding == ROCM_EMBEDDING_TABLE_ENCODING_F32) {
		return args.embedding_bytes == static_cast<uint64_t>(args.vocab_size) * args.hidden_size * sizeof(float);
	}
	if (args.table_encoding == ROCM_EMBEDDING_TABLE_ENCODING_BF16) {
		return args.embedding_bytes == static_cast<uint64_t>(args.vocab_size) * args.hidden_size * sizeof(uint16_t);
	}
	if (args.table_encoding == ROCM_EMBEDDING_TABLE_ENCODING_MLX_Q4) {
		const uint32_t bits = args.bits == 0u ? ROCM_MLX_Q4_PROJECTION_BITS : args.bits;
		return args.scale_pointer != 0 &&
			args.bias_pointer != 0 &&
			rocm_mlx_affine_shape_valid(args.vocab_size, args.hidden_size, args.group_size, bits, args.embedding_bytes, args.scale_bytes, args.bias_bytes);
	}
	return false;
}

__device__ bool rocm_valid_embedding_lookup_greedy_token_args(const rocm_embedding_lookup_launch_args &args)
{
	if (args.token_count != 1u || args.token_bytes != sizeof(uint64_t)) {
		return false;
	}
	rocm_embedding_lookup_launch_args normalized = args;
	normalized.token_bytes = sizeof(int32_t);
	return rocm_valid_embedding_lookup_args(normalized);
}

__device__ bool rocm_valid_diffusion_expected_embedding_args(const rocm_diffusion_expected_embedding_launch_args &args)
{
	const float output_scale = args.output_scale_bits == 0 ? 1.0f : rocm_float_from_bits(args.output_scale_bits);
	if (args.version != ROCM_DIFFUSION_EXPECTED_EMBEDDING_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_DIFFUSION_EXPECTED_EMBEDDING_LAUNCH_ARGS_BYTES ||
		args.probability_pointer == 0 ||
		args.embedding_pointer == 0 ||
		args.output_pointer == 0 ||
		args.rows == 0 ||
		args.vocab_size == 0 ||
		args.hidden_size == 0 ||
		args.probability_bytes != static_cast<uint64_t>(args.rows) * args.vocab_size * sizeof(float) ||
		args.output_bytes != static_cast<uint64_t>(args.rows) * args.hidden_size * sizeof(float) ||
		!isfinite(output_scale)) {
		return false;
	}
	if (args.table_encoding == ROCM_EMBEDDING_TABLE_ENCODING_F32) {
		return args.embedding_bytes == static_cast<uint64_t>(args.vocab_size) * args.hidden_size * sizeof(float);
	}
	if (args.table_encoding == ROCM_EMBEDDING_TABLE_ENCODING_BF16) {
		return args.embedding_bytes == static_cast<uint64_t>(args.vocab_size) * args.hidden_size * sizeof(uint16_t);
	}
	if (args.table_encoding == ROCM_EMBEDDING_TABLE_ENCODING_MLX_Q4) {
		const uint32_t bits = args.bits == 0u ? ROCM_MLX_Q4_PROJECTION_BITS : args.bits;
		return args.scale_pointer != 0 &&
			args.bias_pointer != 0 &&
			rocm_mlx_affine_shape_valid(args.vocab_size, args.hidden_size, args.group_size, bits, args.embedding_bytes, args.scale_bytes, args.bias_bytes);
	}
	return false;
}

__device__ bool rocm_valid_diffusion_sample_args(const rocm_diffusion_sample_launch_args &args)
{
	const float temperature = rocm_float_from_bits(args.temperature_bits);
	const float softcap = rocm_float_from_bits(args.softcap_bits);
	return args.version == ROCM_DIFFUSION_SAMPLE_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_DIFFUSION_SAMPLE_LAUNCH_ARGS_BYTES &&
		args.logits_pointer != 0 &&
		args.draws_pointer != 0 &&
		args.output_pointer != 0 &&
		args.rows > 0 &&
		args.vocab_size > 0 &&
		args.logits_bytes == static_cast<uint64_t>(args.rows) * args.vocab_size * sizeof(float) &&
		args.draws_bytes == static_cast<uint64_t>(args.rows) * sizeof(float) &&
		args.output_bytes == static_cast<uint64_t>(args.rows) * ROCM_DIFFUSION_SAMPLE_RESULT_BYTES &&
		temperature > 0.0f &&
		isfinite(temperature) &&
		softcap >= 0.0f &&
		isfinite(softcap);
}

__device__ bool rocm_valid_rerank_cosine_args(const rocm_rerank_cosine_launch_args &args)
{
	return args.version == ROCM_RERANK_COSINE_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_RERANK_COSINE_LAUNCH_ARGS_BYTES &&
		args.query_pointer != 0 &&
		args.document_pointer != 0 &&
		args.output_pointer != 0 &&
		args.document_count > 0 &&
		args.dim > 0 &&
		args.query_bytes == args.dim * sizeof(float) &&
		args.document_bytes == args.document_count * args.dim * sizeof(float) &&
		args.output_bytes == args.document_count * sizeof(float);
}

__device__ bool rocm_valid_rms_norm_args(const rocm_rms_norm_launch_args &args)
{
	const float epsilon = rocm_float_from_bits(args.epsilon_bits);
	if (args.version != ROCM_RMS_NORM_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_RMS_NORM_LAUNCH_ARGS_BYTES ||
		args.input_pointer == 0 ||
		args.output_pointer == 0 ||
		args.count == 0 ||
		args.input_bytes != args.count * sizeof(float) ||
		args.output_bytes != args.count * sizeof(float) ||
		!(epsilon >= 0.0f) ||
		(args.flags & ~ROCM_RMS_NORM_LAUNCH_FLAG_ADD_UNIT_WEIGHT) != 0) {
		return false;
	}
	if (args.weight_encoding == ROCM_RMS_NORM_WEIGHT_ENCODING_NONE) {
		return args.weight_pointer == 0 && args.weight_bytes == 0 && args.flags == 0;
	}
	if (args.weight_encoding == ROCM_RMS_NORM_WEIGHT_ENCODING_F32) {
		return args.weight_pointer != 0 && args.weight_bytes == args.count * sizeof(float);
	}
	if (args.weight_encoding == ROCM_RMS_NORM_WEIGHT_ENCODING_BF16) {
		return args.weight_pointer != 0 && args.weight_bytes == args.count * sizeof(uint16_t);
	}
	return false;
}

__device__ bool rocm_valid_moe_combine_norms_args(const rocm_moe_combine_norms_launch_args &args)
{
	const float local_epsilon = rocm_float_from_bits(args.local_epsilon_bits);
	const float expert_epsilon = rocm_float_from_bits(args.expert_epsilon_bits);
	if (args.version != ROCM_MOE_COMBINE_NORMS_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_MOE_COMBINE_NORMS_LAUNCH_ARGS_BYTES ||
		args.local_input_pointer == 0 ||
		args.expert_input_pointer == 0 ||
		args.output_pointer == 0 ||
		args.count == 0 ||
		args.local_input_bytes != args.count * sizeof(float) ||
		args.expert_input_bytes != args.count * sizeof(float) ||
		args.output_bytes != args.count * sizeof(float) ||
		!(local_epsilon >= 0.0f) ||
		!(expert_epsilon >= 0.0f) ||
		(args.local_flags & ~ROCM_RMS_NORM_LAUNCH_FLAG_ADD_UNIT_WEIGHT) != 0 ||
		(args.expert_flags & ~ROCM_RMS_NORM_LAUNCH_FLAG_ADD_UNIT_WEIGHT) != 0) {
		return false;
	}
	const bool local_weight_ok =
		(args.local_weight_encoding == ROCM_RMS_NORM_WEIGHT_ENCODING_NONE && args.local_weight_pointer == 0 && args.local_weight_bytes == 0 && args.local_flags == 0) ||
		(args.local_weight_encoding == ROCM_RMS_NORM_WEIGHT_ENCODING_F32 && args.local_weight_pointer != 0 && args.local_weight_bytes == args.count * sizeof(float)) ||
		(args.local_weight_encoding == ROCM_RMS_NORM_WEIGHT_ENCODING_BF16 && args.local_weight_pointer != 0 && args.local_weight_bytes == args.count * sizeof(uint16_t));
	const bool expert_weight_ok =
		(args.expert_weight_encoding == ROCM_RMS_NORM_WEIGHT_ENCODING_NONE && args.expert_weight_pointer == 0 && args.expert_weight_bytes == 0 && args.expert_flags == 0) ||
		(args.expert_weight_encoding == ROCM_RMS_NORM_WEIGHT_ENCODING_F32 && args.expert_weight_pointer != 0 && args.expert_weight_bytes == args.count * sizeof(float)) ||
		(args.expert_weight_encoding == ROCM_RMS_NORM_WEIGHT_ENCODING_BF16 && args.expert_weight_pointer != 0 && args.expert_weight_bytes == args.count * sizeof(uint16_t));
	return local_weight_ok && expert_weight_ok;
}

__device__ bool rocm_valid_rms_norm_weight_config(uint64_t weight_pointer, uint32_t weight_bytes, uint32_t count, uint32_t weight_encoding, uint32_t flags)
{
	if ((flags & ~ROCM_RMS_NORM_LAUNCH_FLAG_MASK) != 0) {
		return false;
	}
	if (weight_encoding == ROCM_RMS_NORM_WEIGHT_ENCODING_NONE) {
		return weight_pointer == 0 && weight_bytes == 0 && (flags & ROCM_RMS_NORM_LAUNCH_FLAG_ADD_UNIT_WEIGHT) == 0;
	}
	if (weight_encoding == ROCM_RMS_NORM_WEIGHT_ENCODING_F32) {
		return weight_pointer != 0 && weight_bytes == count * sizeof(float);
	}
	if (weight_encoding == ROCM_RMS_NORM_WEIGHT_ENCODING_BF16) {
		return weight_pointer != 0 && weight_bytes == count * sizeof(uint16_t);
	}
	return false;
}

__device__ bool rocm_valid_rms_norm_residual_add_args(const rocm_rms_norm_residual_add_launch_args &args)
{
	const float epsilon = rocm_float_from_bits(args.epsilon_bits);
	const float output_scale = args.output_scale_bits == 0 ? 1.0f : rocm_float_from_bits(args.output_scale_bits);
	if (args.version != ROCM_RMS_NORM_RESIDUAL_ADD_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_RMS_NORM_RESIDUAL_ADD_LAUNCH_ARGS_BYTES ||
		args.input_pointer == 0 ||
		args.residual_pointer == 0 ||
		args.output_pointer == 0 ||
		args.count == 0 ||
		args.input_bytes != args.count * sizeof(float) ||
		args.residual_bytes != args.count * sizeof(float) ||
		args.output_bytes != args.count * sizeof(float) ||
		!(epsilon >= 0.0f) ||
		!isfinite(output_scale) ||
		(args.flags & ~ROCM_RMS_NORM_LAUNCH_FLAG_ADD_UNIT_WEIGHT) != 0) {
		return false;
	}
	if (args.weight_encoding == ROCM_RMS_NORM_WEIGHT_ENCODING_NONE) {
		return args.weight_pointer == 0 && args.weight_bytes == 0 && args.flags == 0;
	}
	if (args.weight_encoding == ROCM_RMS_NORM_WEIGHT_ENCODING_F32) {
		return args.weight_pointer != 0 && args.weight_bytes == args.count * sizeof(float);
	}
	if (args.weight_encoding == ROCM_RMS_NORM_WEIGHT_ENCODING_BF16) {
		return args.weight_pointer != 0 && args.weight_bytes == args.count * sizeof(uint16_t);
	}
	return false;
}

__device__ bool rocm_valid_rms_norm_residual_add_norm_args(const rocm_rms_norm_residual_add_norm_launch_args &args)
{
	const float epsilon = rocm_float_from_bits(args.epsilon_bits);
	const float norm_epsilon = rocm_float_from_bits(args.norm_epsilon_bits);
	const float output_scale = args.output_scale_bits == 0 ? 1.0f : rocm_float_from_bits(args.output_scale_bits);
	return args.version == ROCM_RMS_NORM_RESIDUAL_ADD_NORM_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_RMS_NORM_RESIDUAL_ADD_NORM_LAUNCH_ARGS_BYTES &&
		args.input_pointer != 0 &&
		args.residual_pointer != 0 &&
		args.residual_output_pointer != 0 &&
		args.norm_output_pointer != 0 &&
		args.count > 0 &&
		args.input_bytes == args.count * sizeof(float) &&
		args.residual_bytes == args.count * sizeof(float) &&
		args.residual_output_bytes == args.count * sizeof(float) &&
		args.norm_output_bytes == args.count * sizeof(float) &&
		(epsilon >= 0.0f) &&
		isfinite(epsilon) &&
		(norm_epsilon >= 0.0f) &&
		isfinite(norm_epsilon) &&
		isfinite(output_scale) &&
		rocm_valid_rms_norm_weight_config(args.weight_pointer, args.weight_bytes, args.count, args.weight_encoding, args.flags) &&
		rocm_valid_rms_norm_weight_config(args.norm_weight_pointer, args.norm_weight_bytes, args.count, args.norm_weight_encoding, args.norm_flags);
}

__device__ bool rocm_valid_rms_norm_heads_args(const rocm_rms_norm_heads_launch_args &args)
{
	const float epsilon = rocm_float_from_bits(args.epsilon_bits);
	if (args.version != ROCM_RMS_NORM_HEADS_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_RMS_NORM_HEADS_LAUNCH_ARGS_BYTES ||
		args.input_pointer == 0 ||
		args.output_pointer == 0 ||
		args.head_dim == 0 ||
		args.head_count == 0 ||
		args.input_bytes != args.head_dim * args.head_count * sizeof(float) ||
		args.output_bytes != args.head_dim * args.head_count * sizeof(float) ||
		!(epsilon >= 0.0f) ||
		(args.flags & ~ROCM_RMS_NORM_LAUNCH_FLAG_ADD_UNIT_WEIGHT) != 0) {
		return false;
	}
	if (args.weight_encoding == ROCM_RMS_NORM_WEIGHT_ENCODING_NONE) {
		return args.weight_pointer == 0 && args.weight_bytes == 0 && args.flags == 0;
	}
	if (args.weight_encoding == ROCM_RMS_NORM_WEIGHT_ENCODING_F32) {
		return args.weight_pointer != 0 && args.weight_bytes == args.head_dim * sizeof(float);
	}
	if (args.weight_encoding == ROCM_RMS_NORM_WEIGHT_ENCODING_BF16) {
		return args.weight_pointer != 0 && args.weight_bytes == args.head_dim * sizeof(uint16_t);
	}
	return false;
}

__device__ bool rocm_valid_rms_norm_rope_heads_args(const rocm_rms_norm_rope_heads_launch_args &args)
{
	const float epsilon = rocm_float_from_bits(args.epsilon_bits);
	const float base = rocm_float_from_bits(args.base_bits);
	const float frequency_scale = rocm_float_from_bits(args.frequency_scale_bits);
	const uint32_t rotary_count = args.rotary_count == 0u ? args.head_dim : args.rotary_count;
	return args.version == ROCM_RMS_NORM_ROPE_HEADS_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_RMS_NORM_ROPE_HEADS_LAUNCH_ARGS_BYTES &&
		args.input_pointer != 0 &&
		args.output_pointer != 0 &&
		args.head_dim > 0 &&
		(args.head_dim % 2u) == 0u &&
		args.head_count > 0 &&
		args.input_bytes == args.head_dim * args.head_count * sizeof(float) &&
		args.output_bytes == args.head_dim * args.head_count * sizeof(float) &&
		(epsilon >= 0.0f) &&
		isfinite(epsilon) &&
		base > 0.0f &&
		isfinite(base) &&
		frequency_scale > 0.0f &&
		isfinite(frequency_scale) &&
		(args.flags & ~ROCM_RMS_NORM_LAUNCH_FLAG_MASK) == 0u &&
		(args.frequency_dim == 0u || args.frequency_dim >= args.head_dim) &&
		(rotary_count % 2u) == 0u &&
		rotary_count <= args.head_dim &&
		rocm_valid_rms_norm_weight_config(args.weight_pointer, args.weight_bytes, args.head_dim, args.weight_encoding, args.flags);
}

__device__ bool rocm_valid_rms_norm_rope_heads_pair_args(const rocm_rms_norm_rope_heads_pair_launch_args &args)
{
	const float query_epsilon = rocm_float_from_bits(args.query_epsilon_bits);
	const float key_epsilon = rocm_float_from_bits(args.key_epsilon_bits);
	const float base = rocm_float_from_bits(args.base_bits);
	const float frequency_scale = rocm_float_from_bits(args.frequency_scale_bits);
	const uint32_t rotary_count = args.rotary_count == 0u ? args.head_dim : args.rotary_count;
	const uint64_t query_count = static_cast<uint64_t>(args.head_dim) * static_cast<uint64_t>(args.query_head_count);
	const uint64_t key_count = static_cast<uint64_t>(args.head_dim) * static_cast<uint64_t>(args.key_head_count);
	return args.version == ROCM_RMS_NORM_ROPE_HEADS_PAIR_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_RMS_NORM_ROPE_HEADS_PAIR_LAUNCH_ARGS_BYTES &&
		args.query_input_pointer != 0 &&
		args.query_output_pointer != 0 &&
		args.key_input_pointer != 0 &&
		args.key_output_pointer != 0 &&
		args.head_dim > 0 &&
		(args.head_dim % 2u) == 0u &&
		args.query_head_count > 0 &&
		args.key_head_count > 0 &&
		query_count <= 0xffffffffu &&
		key_count <= 0xffffffffu &&
		args.query_input_bytes == query_count * sizeof(float) &&
		args.query_output_bytes == query_count * sizeof(float) &&
		args.key_input_bytes == key_count * sizeof(float) &&
		args.key_output_bytes == key_count * sizeof(float) &&
		(query_epsilon >= 0.0f) &&
		isfinite(query_epsilon) &&
		(key_epsilon >= 0.0f) &&
		isfinite(key_epsilon) &&
		base > 0.0f &&
		isfinite(base) &&
		frequency_scale > 0.0f &&
		isfinite(frequency_scale) &&
		(args.query_flags & ~ROCM_RMS_NORM_LAUNCH_FLAG_MASK) == 0u &&
		(args.key_flags & ~ROCM_RMS_NORM_LAUNCH_FLAG_MASK) == 0u &&
		(args.frequency_dim == 0u || args.frequency_dim >= args.head_dim) &&
		(rotary_count % 2u) == 0u &&
		rotary_count <= args.head_dim &&
		rocm_valid_rms_norm_weight_config(args.query_weight_pointer, args.query_weight_bytes, args.head_dim, args.query_weight_encoding, args.query_flags) &&
		rocm_valid_rms_norm_weight_config(args.key_weight_pointer, args.key_weight_bytes, args.head_dim, args.key_weight_encoding, args.key_flags);
}

__device__ bool rocm_valid_rms_norm_rope_heads_batch_args(const rocm_rms_norm_rope_heads_batch_launch_args &args)
{
	const float epsilon = rocm_float_from_bits(args.epsilon_bits);
	const float base = rocm_float_from_bits(args.base_bits);
	const float frequency_scale = rocm_float_from_bits(args.frequency_scale_bits);
	const uint32_t rotary_count = args.rotary_count == 0u ? args.head_dim : args.rotary_count;
	const uint64_t total_count = static_cast<uint64_t>(args.head_dim) * static_cast<uint64_t>(args.head_count) * static_cast<uint64_t>(args.batch);
	const uint64_t last_position = static_cast<uint64_t>(args.start_position) + static_cast<uint64_t>(args.batch == 0u ? 0u : args.batch - 1u);
	return args.version == ROCM_RMS_NORM_ROPE_HEADS_BATCH_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_RMS_NORM_ROPE_HEADS_BATCH_LAUNCH_ARGS_BYTES &&
		args.input_pointer != 0 &&
		args.output_pointer != 0 &&
		args.head_dim > 0 &&
		(args.head_dim % 2u) == 0u &&
		args.head_count > 0 &&
		args.batch > 0 &&
		total_count <= 0xffffffffu &&
		args.input_bytes == total_count * sizeof(float) &&
		args.output_bytes == total_count * sizeof(float) &&
		(epsilon >= 0.0f) &&
		isfinite(epsilon) &&
		base > 0.0f &&
		isfinite(base) &&
		frequency_scale > 0.0f &&
		isfinite(frequency_scale) &&
		last_position <= 0xffffffffu &&
		(args.flags & ~ROCM_RMS_NORM_LAUNCH_FLAG_MASK) == 0u &&
		(args.frequency_dim == 0u || args.frequency_dim >= args.head_dim) &&
		(rotary_count % 2u) == 0u &&
		rotary_count <= args.head_dim &&
		rocm_valid_rms_norm_weight_config(args.weight_pointer, args.weight_bytes, args.head_dim, args.weight_encoding, args.flags);
}

__device__ bool rocm_valid_rms_norm_rope_heads_pair_lane_batch_args(const rocm_rms_norm_rope_heads_pair_lane_batch_launch_args &args)
{
	const float query_epsilon = rocm_float_from_bits(args.query_epsilon_bits);
	const float key_epsilon = rocm_float_from_bits(args.key_epsilon_bits);
	const float base = rocm_float_from_bits(args.base_bits);
	const float frequency_scale = rocm_float_from_bits(args.frequency_scale_bits);
	const uint32_t rotary_count = args.rotary_count == 0u ? args.head_dim : args.rotary_count;
	const uint64_t query_count = static_cast<uint64_t>(args.head_dim) * static_cast<uint64_t>(args.query_head_count) * static_cast<uint64_t>(args.batch);
	const uint64_t key_count = static_cast<uint64_t>(args.head_dim) * static_cast<uint64_t>(args.key_head_count) * static_cast<uint64_t>(args.batch);
	return args.version == ROCM_RMS_NORM_ROPE_HEADS_PAIR_LANE_BATCH_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_RMS_NORM_ROPE_HEADS_PAIR_LANE_BATCH_LAUNCH_ARGS_BYTES &&
		args.query_input_pointer != 0 &&
		args.query_output_pointer != 0 &&
		args.key_input_pointer != 0 &&
		args.key_output_pointer != 0 &&
		args.positions_pointer != 0 &&
		args.head_dim > 0 &&
		(args.head_dim % 2u) == 0u &&
		args.query_head_count > 0 &&
		args.key_head_count > 0 &&
		args.batch > 0 &&
		query_count <= 0xffffffffu &&
		key_count <= 0xffffffffu &&
		args.query_input_bytes == query_count * sizeof(float) &&
		args.query_output_bytes == query_count * sizeof(float) &&
		args.key_input_bytes == key_count * sizeof(float) &&
		args.key_output_bytes == key_count * sizeof(float) &&
		args.positions_bytes == args.batch * sizeof(uint32_t) &&
		(query_epsilon >= 0.0f) &&
		isfinite(query_epsilon) &&
		(key_epsilon >= 0.0f) &&
		isfinite(key_epsilon) &&
		base > 0.0f &&
		isfinite(base) &&
		frequency_scale > 0.0f &&
		isfinite(frequency_scale) &&
		(args.query_flags & ~ROCM_RMS_NORM_LAUNCH_FLAG_MASK) == 0u &&
		(args.key_flags & ~ROCM_RMS_NORM_LAUNCH_FLAG_MASK) == 0u &&
		(args.frequency_dim == 0u || args.frequency_dim >= args.head_dim) &&
		(rotary_count % 2u) == 0u &&
		rotary_count <= args.head_dim &&
		rocm_valid_rms_norm_weight_config(args.query_weight_pointer, args.query_weight_bytes, args.head_dim, args.query_weight_encoding, args.query_flags) &&
		rocm_valid_rms_norm_weight_config(args.key_weight_pointer, args.key_weight_bytes, args.head_dim, args.key_weight_encoding, args.key_flags);
}

__device__ bool rocm_valid_rope_args(const rocm_rope_launch_args &args)
{
	const bool rotary_override = args.reserved2 != 0;
	const bool rotary_fits = !rotary_override || args.reserved2 <= static_cast<uint64_t>(args.count);
	const uint32_t rotary_count = rotary_override ? static_cast<uint32_t>(args.reserved2) : args.count;
	return args.version == ROCM_ROPE_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_ROPE_LAUNCH_ARGS_BYTES &&
		args.input_pointer != 0 &&
		args.output_pointer != 0 &&
		args.count > 0 &&
		rotary_fits &&
		(args.count % 2) == 0 &&
		(rotary_count % 2) == 0 &&
		rotary_count <= args.count &&
		args.input_bytes == args.count * sizeof(float) &&
		args.output_bytes == args.count * sizeof(float) &&
		rocm_float_from_bits(args.base_bits) > 0.0f &&
		(args.frequency_dim == 0 || args.frequency_dim >= args.count);
}

__device__ bool rocm_valid_rope_heads_args(const rocm_rope_heads_launch_args &args)
{
	const bool rotary_override = args.rotary_count != 0;
	const uint32_t rotary_count = rotary_override ? args.rotary_count : args.head_dim;
	return args.version == ROCM_ROPE_HEADS_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_ROPE_HEADS_LAUNCH_ARGS_BYTES &&
		args.input_pointer != 0 &&
		args.output_pointer != 0 &&
		args.head_dim > 0 &&
		args.head_count > 0 &&
		(args.head_dim % 2) == 0 &&
		(rotary_count % 2) == 0 &&
		rotary_count <= args.head_dim &&
		args.input_bytes == args.head_dim * args.head_count * sizeof(float) &&
		args.output_bytes == args.head_dim * args.head_count * sizeof(float) &&
		rocm_float_from_bits(args.base_bits) > 0.0f &&
		(args.frequency_dim == 0 || args.frequency_dim >= args.head_dim);
}

__device__ bool rocm_valid_greedy_args(const rocm_greedy_launch_args &args)
{
	return args.version == ROCM_GREEDY_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_GREEDY_LAUNCH_ARGS_BYTES &&
		args.logits_pointer != 0 &&
		args.output_pointer != 0 &&
		args.count > 0 &&
		args.logits_bytes == args.count * sizeof(float) &&
		args.output_bytes == ROCM_GREEDY_RESULT_BYTES;
}

__device__ bool rocm_valid_softcap_greedy_args(const rocm_softcap_greedy_launch_args &args)
{
	const float softcap = rocm_float_from_bits(args.softcap_bits);
	return args.version == ROCM_SOFTCAP_GREEDY_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_SOFTCAP_GREEDY_LAUNCH_ARGS_BYTES &&
		args.logits_pointer != 0 &&
		args.output_pointer != 0 &&
		args.count > 0 &&
		args.logits_bytes == args.count * sizeof(float) &&
		args.output_bytes == ROCM_GREEDY_RESULT_BYTES &&
		(args.suppress_count == 0 || (args.suppress_pointer != 0 && args.suppress_count < args.count)) &&
		softcap >= 0.0f &&
		isfinite(softcap);
}

__device__ uint64_t rocm_device_kv_tensor_bytes(uint32_t encoding, uint64_t length);
__device__ uint64_t rocm_device_kv_tensor_bytes_rows(uint32_t encoding, uint64_t length, uint64_t rows);

__device__ uint32_t rocm_attention_kv_head_count(uint32_t encoded, uint32_t query_heads)
{
	if (encoded == 0u) {
		return 1u;
	}
	return encoded > query_heads ? query_heads : encoded;
}

__device__ uint32_t rocm_attention_kv_head_for_query(uint32_t query_head, uint32_t query_heads, uint32_t kv_heads)
{
	if (kv_heads <= 1u || query_heads <= kv_heads) {
		return kv_heads <= 1u ? 0u : query_head;
	}
	const uint32_t group = query_heads / kv_heads;
	if (group == 0u) {
		return 0u;
	}
	const uint32_t kv_head = query_head / group;
	return kv_head >= kv_heads ? kv_heads - 1u : kv_head;
}

__device__ bool rocm_valid_device_kv_attention_descriptor_header(const rocm_attention_launch_args &args)
{
	if (args.kv_source != ROCM_ATTENTION_KV_SOURCE_DEVICE ||
		args.descriptor_pointer == 0 ||
		args.descriptor_bytes < ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES ||
		args.key_pointer != 0 ||
		args.value_pointer != 0 ||
		args.key_bytes != 0 ||
		args.value_bytes != 0) {
		return false;
	}
	const rocm_device_kv_descriptor_header *header = reinterpret_cast<const rocm_device_kv_descriptor_header *>(static_cast<uintptr_t>(args.descriptor_pointer));
	if (header->version != ROCM_DEVICE_KV_DESCRIPTOR_VERSION ||
		header->header_bytes != ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES ||
		header->page_bytes != ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES ||
		header->page_count == 0 ||
		header->token_count != args.token_count ||
		args.descriptor_bytes != static_cast<uint64_t>(ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES) + static_cast<uint64_t>(header->page_count) * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES) {
		return false;
	}
	return true;
}

__device__ bool rocm_valid_device_kv_attention_descriptor(const rocm_attention_launch_args &args)
{
	if (!rocm_valid_device_kv_attention_descriptor_header(args)) {
		return false;
	}
	const rocm_device_kv_descriptor_header *header = reinterpret_cast<const rocm_device_kv_descriptor_header *>(static_cast<uintptr_t>(args.descriptor_pointer));
	const unsigned char *base = reinterpret_cast<const unsigned char *>(header);
	for (uint32_t page_index = 0; page_index < header->page_count; ++page_index) {
		const rocm_device_kv_page_descriptor *page = reinterpret_cast<const rocm_device_kv_page_descriptor *>(base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + page_index * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES);
		if (page->token_count == 0 ||
			page->token_start + page->token_count > args.token_count ||
			page->key_width != args.dim ||
			page->value_width != args.dim ||
			page->key_pointer == 0 ||
			page->value_pointer == 0 ||
			rocm_device_kv_tensor_bytes_rows(page->key_encoding, static_cast<uint64_t>(page->token_count) * page->key_width, page->token_count) == 0 ||
			rocm_device_kv_tensor_bytes_rows(page->value_encoding, static_cast<uint64_t>(page->token_count) * page->value_width, page->token_count) == 0 ||
			page->key_bytes != rocm_device_kv_tensor_bytes_rows(page->key_encoding, static_cast<uint64_t>(page->token_count) * page->key_width, page->token_count) ||
			page->value_bytes != rocm_device_kv_tensor_bytes_rows(page->value_encoding, static_cast<uint64_t>(page->token_count) * page->value_width, page->token_count)) {
			return false;
		}
	}
	return true;
}

__device__ uint64_t rocm_device_kv_tensor_bytes(uint32_t encoding, uint64_t length)
{
	if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_FP16) {
		return length * sizeof(uint16_t);
	}
	if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8) {
		return sizeof(uint32_t) + length * sizeof(int8_t);
	}
	if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4) {
		return sizeof(uint32_t) + ((length + 1u) / 2u);
	}
	if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS) {
		return sizeof(uint32_t) + length * sizeof(int8_t);
	}
	if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4_ROWS) {
		return sizeof(uint32_t) + ((length + 1u) / 2u);
	}
	if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS_INTERLEAVED) {
		return sizeof(uint32_t) + length;
	}
	if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4_ROWS_INTERLEAVED) {
		return sizeof(uint32_t) + ((length + 1u) / 2u);
	}
	return 0;
}

__device__ uint64_t rocm_device_kv_tensor_bytes_rows(uint32_t encoding, uint64_t length, uint64_t rows)
{
	if (rows == 0u) {
		return 0;
	}
	if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS) {
		return rows * sizeof(uint32_t) + length * sizeof(int8_t);
	}
	if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4_ROWS) {
		return rows * sizeof(uint32_t) + ((length + 1u) / 2u);
	}
	if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS_INTERLEAVED) {
		if ((length % rows) != 0u) {
			return 0u;
		}
		const uint64_t row_width = length / rows;
		return rows * (sizeof(uint32_t) + row_width);
	}
	if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4_ROWS_INTERLEAVED) {
		if ((length % rows) != 0u) {
			return 0u;
		}
		const uint64_t row_width = length / rows;
		return rows * (sizeof(uint32_t) + ((row_width + 1u) / 2u));
	}
	return rocm_device_kv_tensor_bytes(encoding, length);
}

__device__ bool rocm_valid_kv_tensor_encoding(uint32_t encoding)
{
	return encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_FP16 ||
		encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8 ||
		encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4 ||
		encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS ||
		encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4_ROWS ||
		encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS_INTERLEAVED ||
		encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4_ROWS_INTERLEAVED;
}

__device__ bool rocm_valid_kv_encode_token_args(const rocm_kv_encode_token_launch_args &args)
{
	const uint64_t row_count = args.reserved2 == 0u ? 1u : args.reserved2;
	const uint64_t key_width = args.reserved0 == 0u ? args.key_count : args.reserved0;
	const uint64_t value_width = args.reserved1 == 0u ? args.value_count : args.reserved1;
	return args.version == ROCM_KV_ENCODE_TOKEN_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_KV_ENCODE_TOKEN_LAUNCH_ARGS_BYTES &&
		args.key_input_pointer != 0 &&
		args.value_input_pointer != 0 &&
		args.key_output_pointer != 0 &&
		args.value_output_pointer != 0 &&
		args.key_count > 0 &&
		args.value_count > 0 &&
		args.key_input_bytes == args.key_count * sizeof(float) &&
		args.value_input_bytes == args.value_count * sizeof(float) &&
		rocm_valid_kv_tensor_encoding(args.key_encoding) &&
		rocm_valid_kv_tensor_encoding(args.value_encoding) &&
		row_count > 0u &&
		key_width > 0u &&
		value_width > 0u &&
		static_cast<uint64_t>(args.key_count) == key_width * row_count &&
		static_cast<uint64_t>(args.value_count) == value_width * row_count &&
		(args.key_encoding != ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4_ROWS || (key_width & 1u) == 0u) &&
		(args.value_encoding != ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4_ROWS || (value_width & 1u) == 0u) &&
		(args.key_encoding != ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4_ROWS_INTERLEAVED || (key_width & 1u) == 0u) &&
		(args.value_encoding != ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4_ROWS_INTERLEAVED || (value_width & 1u) == 0u) &&
		args.key_output_bytes == rocm_device_kv_tensor_bytes_rows(args.key_encoding, args.key_count, row_count) &&
		args.value_output_bytes == rocm_device_kv_tensor_bytes_rows(args.value_encoding, args.value_count, row_count);
}

__device__ bool rocm_valid_kv_encode_token_value_norm_args(const rocm_kv_encode_token_value_norm_launch_args &args)
{
	const uint64_t row_count = args.reserved2 == 0u ? 1u : args.reserved2;
	const uint64_t key_width = args.reserved0 == 0u ? args.key_count : args.reserved0;
	const uint64_t value_width = args.reserved1 == 0u ? args.value_count : args.reserved1;
	const float epsilon = rocm_float_from_bits(args.value_epsilon_bits);
	return args.version == ROCM_KV_ENCODE_TOKEN_VALUE_NORM_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_KV_ENCODE_TOKEN_VALUE_NORM_LAUNCH_ARGS_BYTES &&
		args.key_input_pointer != 0 &&
		args.value_input_pointer != 0 &&
		args.key_output_pointer != 0 &&
		args.value_output_pointer != 0 &&
		args.key_count > 0 &&
		args.value_count > 0 &&
		args.key_input_bytes == args.key_count * sizeof(float) &&
		args.value_input_bytes == args.value_count * sizeof(float) &&
		rocm_valid_kv_tensor_encoding(args.key_encoding) &&
		rocm_valid_kv_tensor_encoding(args.value_encoding) &&
		row_count > 0u &&
		key_width > 0u &&
		value_width > 0u &&
		args.value_head_dim > 0u &&
		args.value_head_count > 0u &&
		args.value_head_count <= ROCM_KV_ENCODE_TOKEN_VALUE_NORM_MAX_HEADS &&
		value_width == static_cast<uint64_t>(args.value_head_dim) * args.value_head_count &&
		static_cast<uint64_t>(args.key_count) == key_width * row_count &&
		static_cast<uint64_t>(args.value_count) == value_width * row_count &&
		(args.key_encoding != ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4_ROWS || (key_width & 1u) == 0u) &&
		(args.value_encoding != ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4_ROWS || (value_width & 1u) == 0u) &&
		(args.key_encoding != ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4_ROWS_INTERLEAVED || (key_width & 1u) == 0u) &&
		(args.value_encoding != ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4_ROWS_INTERLEAVED || (value_width & 1u) == 0u) &&
		isfinite(epsilon) &&
		epsilon >= 0.0f &&
		args.key_output_bytes == rocm_device_kv_tensor_bytes_rows(args.key_encoding, args.key_count, row_count) &&
		args.value_output_bytes == rocm_device_kv_tensor_bytes_rows(args.value_encoding, args.value_count, row_count);
}

__device__ uint8_t rocm_pack_signed_q4(int value)
{
	if (value < 0) {
		value += 16;
	}
	return static_cast<uint8_t>(value) & 0x0fu;
}

__device__ int rocm_clamp_int(int value, int low, int high)
{
	if (value < low) {
		return low;
	}
	if (value > high) {
		return high;
	}
	return value;
}

__device__ int rocm_quantize_kv_value(float value, float scale, uint32_t encoding)
{
	const int quantized = static_cast<int>(roundf(value / scale));
	if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8 || encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS || encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS_INTERLEAVED) {
		return rocm_clamp_int(quantized, -127, 127);
	}
	return rocm_clamp_int(quantized, -8, 7);
}

__device__ void rocm_encode_kv_token_tensor(const float *input, unsigned char *output, uint32_t count, uint32_t encoding, uint32_t row_width, uint32_t row_count)
{
	const uint32_t tid = threadIdx.x;
	const uint32_t threads = blockDim.x;
	if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_FP16) {
		uint16_t *halves = reinterpret_cast<uint16_t *>(output);
		for (uint32_t index = tid; index < count; index += threads) {
			halves[index] = rocm_float_to_half(input[index]);
		}
		return;
	}

	__shared__ float scratch[ROCM_KV_ENCODE_TOKEN_BLOCK_SIZE];
	const bool row_interleaved = encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS_INTERLEAVED || encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4_ROWS_INTERLEAVED;
	if (row_interleaved) {
		const uint32_t row_payload = encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS_INTERLEAVED
			? row_width
			: ((row_width + 1u) >> 1u);
		const uint32_t row_stride = sizeof(uint32_t) + row_payload;
		for (uint32_t row = 0; row < row_count; ++row) {
			const uint32_t row_start = row * row_width;
			unsigned char *row_output = output + static_cast<uint64_t>(row) * row_stride;
			float max_abs = 0.0f;
			for (uint32_t dim = tid; dim < row_width; dim += threads) {
				const float magnitude = fabsf(input[row_start + dim]);
				if (magnitude > max_abs) {
					max_abs = magnitude;
				}
			}
			scratch[tid] = max_abs;
			__syncthreads();
			for (uint32_t stride = threads >> 1; stride > 0; stride >>= 1) {
				if (tid < stride && scratch[tid + stride] > scratch[tid]) {
					scratch[tid] = scratch[tid + stride];
				}
				__syncthreads();
			}
			const float scale = scratch[0] == 0.0f ? 1.0f : scratch[0] / (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS_INTERLEAVED ? 127.0f : 7.0f);
			if (tid == 0) {
				*reinterpret_cast<uint32_t *>(row_output) = rocm_float_bits(scale);
			}
			__syncthreads();
			unsigned char *payload = row_output + sizeof(uint32_t);
			if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS_INTERLEAVED) {
				int8_t *quantized = reinterpret_cast<int8_t *>(payload);
				for (uint32_t dim = tid; dim < row_width; dim += threads) {
					quantized[dim] = static_cast<int8_t>(rocm_quantize_kv_value(input[row_start + dim], scale, encoding));
				}
			} else {
				const uint32_t packed_count = row_width >> 1u;
				for (uint32_t packed = tid; packed < packed_count; packed += threads) {
					const uint32_t first = row_start + packed * 2u;
					const int low = rocm_quantize_kv_value(input[first], scale, encoding);
					const int high = rocm_quantize_kv_value(input[first + 1u], scale, encoding);
					payload[packed] = static_cast<unsigned char>(rocm_pack_signed_q4(low) | (rocm_pack_signed_q4(high) << 4));
				}
			}
			__syncthreads();
		}
		return;
	}
	const bool row_scaled = encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS || encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4_ROWS;
	if (row_scaled) {
		unsigned char *payload = output + row_count * sizeof(uint32_t);
		for (uint32_t row = 0; row < row_count; ++row) {
			const uint32_t row_start = row * row_width;
			float max_abs = 0.0f;
			for (uint32_t dim = tid; dim < row_width; dim += threads) {
				const float magnitude = fabsf(input[row_start + dim]);
				if (magnitude > max_abs) {
					max_abs = magnitude;
				}
			}
			scratch[tid] = max_abs;
			__syncthreads();
			for (uint32_t stride = threads >> 1; stride > 0; stride >>= 1) {
				if (tid < stride && scratch[tid + stride] > scratch[tid]) {
					scratch[tid] = scratch[tid + stride];
				}
				__syncthreads();
			}
			const float scale = scratch[0] == 0.0f ? 1.0f : scratch[0] / (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS ? 127.0f : 7.0f);
			if (tid == 0) {
				reinterpret_cast<uint32_t *>(output)[row] = rocm_float_bits(scale);
			}
			__syncthreads();
			if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS) {
				int8_t *quantized = reinterpret_cast<int8_t *>(payload);
				for (uint32_t dim = tid; dim < row_width; dim += threads) {
					const uint32_t index = row_start + dim;
					quantized[index] = static_cast<int8_t>(rocm_quantize_kv_value(input[index], scale, encoding));
				}
			} else {
				const uint32_t packed_start = row_start >> 1u;
				const uint32_t packed_count = row_width >> 1u;
				for (uint32_t packed = tid; packed < packed_count; packed += threads) {
					const uint32_t first = row_start + packed * 2u;
					const int low = rocm_quantize_kv_value(input[first], scale, encoding);
					const int high = rocm_quantize_kv_value(input[first + 1u], scale, encoding);
					payload[packed_start + packed] = static_cast<unsigned char>(rocm_pack_signed_q4(low) | (rocm_pack_signed_q4(high) << 4));
				}
			}
			__syncthreads();
		}
		return;
	}

	float max_abs = 0.0f;
	for (uint32_t index = tid; index < count; index += threads) {
		const float magnitude = fabsf(input[index]);
		if (magnitude > max_abs) {
			max_abs = magnitude;
		}
	}
	scratch[tid] = max_abs;
	__syncthreads();
	for (uint32_t stride = threads >> 1; stride > 0; stride >>= 1) {
		if (tid < stride && scratch[tid + stride] > scratch[tid]) {
			scratch[tid] = scratch[tid + stride];
		}
		__syncthreads();
	}
	const float scale = scratch[0] == 0.0f ? 1.0f : scratch[0] / (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8 ? 127.0f : 7.0f);
	if (tid == 0) {
		*reinterpret_cast<uint32_t *>(output) = rocm_float_bits(scale);
	}
	__syncthreads();

	unsigned char *payload = output + sizeof(uint32_t);
	if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8) {
		int8_t *quantized = reinterpret_cast<int8_t *>(payload);
		for (uint32_t index = tid; index < count; index += threads) {
			quantized[index] = static_cast<int8_t>(rocm_quantize_kv_value(input[index], scale, encoding));
		}
		return;
	}

	const uint32_t packed_count = (count + 1u) / 2u;
	for (uint32_t packed_index = tid; packed_index < packed_count; packed_index += threads) {
		const uint32_t first = packed_index * 2u;
		const int low = rocm_quantize_kv_value(input[first], scale, encoding);
		int high = 0;
		if (first + 1u < count) {
			high = rocm_quantize_kv_value(input[first + 1u], scale, encoding);
		}
		payload[packed_index] = static_cast<unsigned char>(rocm_pack_signed_q4(low) | (rocm_pack_signed_q4(high) << 4));
	}
}

__device__ float rocm_kv_encode_block_reduce_sum(float value, float *scratch)
{
	const uint32_t tid = threadIdx.x;
	const uint32_t threads = blockDim.x;
	scratch[tid] = value;
	__syncthreads();
	for (uint32_t stride = threads >> 1; stride > 0; stride >>= 1) {
		if (tid < stride) {
			scratch[tid] += scratch[tid + stride];
		}
		__syncthreads();
	}
	return scratch[0];
}

__device__ float rocm_kv_encode_block_reduce_max(float value, float *scratch)
{
	const uint32_t tid = threadIdx.x;
	const uint32_t threads = blockDim.x;
	scratch[tid] = value;
	__syncthreads();
	for (uint32_t stride = threads >> 1; stride > 0; stride >>= 1) {
		if (tid < stride && scratch[tid + stride] > scratch[tid]) {
			scratch[tid] = scratch[tid + stride];
		}
		__syncthreads();
	}
	return scratch[0];
}

__device__ void rocm_kv_encode_value_norm_prepare_row(const float *input, uint32_t row_start, uint32_t head_dim, uint32_t head_count, float epsilon, float *scratch, float *head_rms)
{
	const uint32_t tid = threadIdx.x;
	for (uint32_t head = 0; head < head_count; ++head) {
		const uint32_t head_start = row_start + head * head_dim;
		float sum_squares = 0.0f;
		for (uint32_t dim = tid; dim < head_dim; dim += blockDim.x) {
			const float value = input[head_start + dim];
			sum_squares += value * value;
		}
		const float total = rocm_kv_encode_block_reduce_sum(sum_squares, scratch);
		if (tid == 0) {
			const float rms = sqrtf(total / static_cast<float>(head_dim) + epsilon);
			head_rms[head] = rms == 0.0f ? 1.0f : rms;
		}
		__syncthreads();
	}
}

__device__ float rocm_kv_encode_value_norm_value(const float *input, uint32_t row_start, uint32_t dim, uint32_t head_dim, const float *head_rms)
{
	const uint32_t head = dim / head_dim;
	return input[row_start + dim] / head_rms[head];
}

__device__ void rocm_encode_kv_token_tensor_value_norm(const float *input, unsigned char *output, uint32_t count, uint32_t encoding, uint32_t row_width, uint32_t row_count, uint32_t head_dim, uint32_t head_count, float epsilon)
{
	const uint32_t tid = threadIdx.x;
	const uint32_t threads = blockDim.x;
	__shared__ float scratch[ROCM_KV_ENCODE_TOKEN_BLOCK_SIZE];
	__shared__ float head_rms[ROCM_KV_ENCODE_TOKEN_VALUE_NORM_MAX_HEADS];
	if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_FP16) {
		uint16_t *halves = reinterpret_cast<uint16_t *>(output);
		for (uint32_t row = 0; row < row_count; ++row) {
			const uint32_t row_start = row * row_width;
			rocm_kv_encode_value_norm_prepare_row(input, row_start, head_dim, head_count, epsilon, scratch, head_rms);
			for (uint32_t dim = tid; dim < row_width; dim += threads) {
				halves[row_start + dim] = rocm_float_to_half(rocm_kv_encode_value_norm_value(input, row_start, dim, head_dim, head_rms));
			}
			__syncthreads();
		}
		return;
	}

	const bool row_interleaved = encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS_INTERLEAVED || encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4_ROWS_INTERLEAVED;
	if (row_interleaved) {
		const uint32_t row_payload = encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS_INTERLEAVED
			? row_width
			: ((row_width + 1u) >> 1u);
		const uint32_t row_stride = sizeof(uint32_t) + row_payload;
		for (uint32_t row = 0; row < row_count; ++row) {
			const uint32_t row_start = row * row_width;
			unsigned char *row_output = output + static_cast<uint64_t>(row) * row_stride;
			rocm_kv_encode_value_norm_prepare_row(input, row_start, head_dim, head_count, epsilon, scratch, head_rms);
			float max_abs = 0.0f;
			for (uint32_t dim = tid; dim < row_width; dim += threads) {
				const float magnitude = fabsf(rocm_kv_encode_value_norm_value(input, row_start, dim, head_dim, head_rms));
				if (magnitude > max_abs) {
					max_abs = magnitude;
				}
			}
			const float row_max = rocm_kv_encode_block_reduce_max(max_abs, scratch);
			const float scale = row_max == 0.0f ? 1.0f : row_max / (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS_INTERLEAVED ? 127.0f : 7.0f);
			if (tid == 0) {
				*reinterpret_cast<uint32_t *>(row_output) = rocm_float_bits(scale);
			}
			__syncthreads();
			unsigned char *payload = row_output + sizeof(uint32_t);
			if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS_INTERLEAVED) {
				int8_t *quantized = reinterpret_cast<int8_t *>(payload);
				for (uint32_t dim = tid; dim < row_width; dim += threads) {
					quantized[dim] = static_cast<int8_t>(rocm_quantize_kv_value(rocm_kv_encode_value_norm_value(input, row_start, dim, head_dim, head_rms), scale, encoding));
				}
			} else {
				const uint32_t packed_count = row_width >> 1u;
				for (uint32_t packed = tid; packed < packed_count; packed += threads) {
					const uint32_t dim = packed * 2u;
					const int low = rocm_quantize_kv_value(rocm_kv_encode_value_norm_value(input, row_start, dim, head_dim, head_rms), scale, encoding);
					const int high = rocm_quantize_kv_value(rocm_kv_encode_value_norm_value(input, row_start, dim + 1u, head_dim, head_rms), scale, encoding);
					payload[packed] = static_cast<unsigned char>(rocm_pack_signed_q4(low) | (rocm_pack_signed_q4(high) << 4));
				}
			}
			__syncthreads();
		}
		return;
	}

	const bool row_scaled = encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS || encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4_ROWS;
	if (row_scaled) {
		unsigned char *payload = output + row_count * sizeof(uint32_t);
		for (uint32_t row = 0; row < row_count; ++row) {
			const uint32_t row_start = row * row_width;
			rocm_kv_encode_value_norm_prepare_row(input, row_start, head_dim, head_count, epsilon, scratch, head_rms);
			float max_abs = 0.0f;
			for (uint32_t dim = tid; dim < row_width; dim += threads) {
				const float magnitude = fabsf(rocm_kv_encode_value_norm_value(input, row_start, dim, head_dim, head_rms));
				if (magnitude > max_abs) {
					max_abs = magnitude;
				}
			}
			const float row_max = rocm_kv_encode_block_reduce_max(max_abs, scratch);
			const float scale = row_max == 0.0f ? 1.0f : row_max / (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS ? 127.0f : 7.0f);
			if (tid == 0) {
				reinterpret_cast<uint32_t *>(output)[row] = rocm_float_bits(scale);
			}
			__syncthreads();
			if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS) {
				int8_t *quantized = reinterpret_cast<int8_t *>(payload);
				for (uint32_t dim = tid; dim < row_width; dim += threads) {
					quantized[row_start + dim] = static_cast<int8_t>(rocm_quantize_kv_value(rocm_kv_encode_value_norm_value(input, row_start, dim, head_dim, head_rms), scale, encoding));
				}
			} else {
				const uint32_t packed_start = row_start >> 1u;
				const uint32_t packed_count = row_width >> 1u;
				for (uint32_t packed = tid; packed < packed_count; packed += threads) {
					const uint32_t dim = packed * 2u;
					const int low = rocm_quantize_kv_value(rocm_kv_encode_value_norm_value(input, row_start, dim, head_dim, head_rms), scale, encoding);
					const int high = rocm_quantize_kv_value(rocm_kv_encode_value_norm_value(input, row_start, dim + 1u, head_dim, head_rms), scale, encoding);
					payload[packed_start + packed] = static_cast<unsigned char>(rocm_pack_signed_q4(low) | (rocm_pack_signed_q4(high) << 4));
				}
			}
			__syncthreads();
		}
		return;
	}

	float max_abs = 0.0f;
	for (uint32_t row = 0; row < row_count; ++row) {
		const uint32_t row_start = row * row_width;
		rocm_kv_encode_value_norm_prepare_row(input, row_start, head_dim, head_count, epsilon, scratch, head_rms);
		for (uint32_t dim = tid; dim < row_width; dim += threads) {
			const float magnitude = fabsf(rocm_kv_encode_value_norm_value(input, row_start, dim, head_dim, head_rms));
			if (magnitude > max_abs) {
				max_abs = magnitude;
			}
		}
		__syncthreads();
	}
	const float global_max = rocm_kv_encode_block_reduce_max(max_abs, scratch);
	const float scale = global_max == 0.0f ? 1.0f : global_max / (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8 ? 127.0f : 7.0f);
	if (tid == 0) {
		*reinterpret_cast<uint32_t *>(output) = rocm_float_bits(scale);
	}
	__syncthreads();

	unsigned char *payload = output + sizeof(uint32_t);
	if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8) {
		int8_t *quantized = reinterpret_cast<int8_t *>(payload);
		for (uint32_t row = 0; row < row_count; ++row) {
			const uint32_t row_start = row * row_width;
			rocm_kv_encode_value_norm_prepare_row(input, row_start, head_dim, head_count, epsilon, scratch, head_rms);
			for (uint32_t dim = tid; dim < row_width; dim += threads) {
				quantized[row_start + dim] = static_cast<int8_t>(rocm_quantize_kv_value(rocm_kv_encode_value_norm_value(input, row_start, dim, head_dim, head_rms), scale, encoding));
			}
			__syncthreads();
		}
		return;
	}

	const uint32_t packed_count = (count + 1u) / 2u;
	for (uint32_t row = 0; row < row_count; ++row) {
		const uint32_t row_start = row * row_width;
		rocm_kv_encode_value_norm_prepare_row(input, row_start, head_dim, head_count, epsilon, scratch, head_rms);
		for (uint32_t packed_index = tid; packed_index < packed_count; packed_index += threads) {
			const uint32_t first = packed_index * 2u;
			if (first < row_start || first >= row_start + row_width) {
				continue;
			}
			const uint32_t dim = first - row_start;
			const int low = rocm_quantize_kv_value(rocm_kv_encode_value_norm_value(input, row_start, dim, head_dim, head_rms), scale, encoding);
			int high = 0;
			if (dim + 1u < row_width) {
				high = rocm_quantize_kv_value(rocm_kv_encode_value_norm_value(input, row_start, dim + 1u, head_dim, head_rms), scale, encoding);
			}
			payload[packed_index] = static_cast<unsigned char>(rocm_pack_signed_q4(low) | (rocm_pack_signed_q4(high) << 4));
		}
		__syncthreads();
	}
}

extern "C" __global__ void rocm_kv_encode_token(const unsigned char *packet)
{
	if (packet == nullptr || blockIdx.x > 1u || blockDim.x == 0 || blockDim.x > ROCM_KV_ENCODE_TOKEN_BLOCK_SIZE) {
		return;
	}
	const rocm_kv_encode_token_launch_args &args = *reinterpret_cast<const rocm_kv_encode_token_launch_args *>(packet);
	if (!rocm_valid_kv_encode_token_args(args)) {
		return;
	}
	const bool encode_key = blockIdx.x == 0;
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(encode_key ? args.key_input_pointer : args.value_input_pointer));
	unsigned char *output = reinterpret_cast<unsigned char *>(static_cast<uintptr_t>(encode_key ? args.key_output_pointer : args.value_output_pointer));
	const uint32_t count = encode_key ? args.key_count : args.value_count;
	const uint32_t encoding = encode_key ? args.key_encoding : args.value_encoding;
	const uint32_t row_count = args.reserved2 == 0u ? 1u : static_cast<uint32_t>(args.reserved2);
	const uint32_t row_width = encode_key
		? (args.reserved0 == 0u ? args.key_count : static_cast<uint32_t>(args.reserved0))
		: (args.reserved1 == 0u ? args.value_count : static_cast<uint32_t>(args.reserved1));
	rocm_encode_kv_token_tensor(input, output, count, encoding, row_width, row_count);
}

extern "C" __global__ void rocm_kv_encode_token_value_norm(const unsigned char *packet)
{
	if (packet == nullptr || blockIdx.x > 1u || blockDim.x == 0 || blockDim.x > ROCM_KV_ENCODE_TOKEN_BLOCK_SIZE) {
		return;
	}
	const rocm_kv_encode_token_value_norm_launch_args &args = *reinterpret_cast<const rocm_kv_encode_token_value_norm_launch_args *>(packet);
	if (!rocm_valid_kv_encode_token_value_norm_args(args)) {
		return;
	}
	const bool encode_key = blockIdx.x == 0;
	const uint32_t row_count = args.reserved2 == 0u ? 1u : static_cast<uint32_t>(args.reserved2);
	if (encode_key) {
		const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.key_input_pointer));
		unsigned char *output = reinterpret_cast<unsigned char *>(static_cast<uintptr_t>(args.key_output_pointer));
		const uint32_t row_width = args.reserved0 == 0u ? args.key_count : static_cast<uint32_t>(args.reserved0);
		rocm_encode_kv_token_tensor(input, output, args.key_count, args.key_encoding, row_width, row_count);
		return;
	}
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.value_input_pointer));
	unsigned char *output = reinterpret_cast<unsigned char *>(static_cast<uintptr_t>(args.value_output_pointer));
	const uint32_t row_width = args.reserved1 == 0u ? args.value_count : static_cast<uint32_t>(args.reserved1);
	rocm_encode_kv_token_tensor_value_norm(input, output, args.value_count, args.value_encoding, row_width, row_count, args.value_head_dim, args.value_head_count, rocm_float_from_bits(args.value_epsilon_bits));
}

__device__ bool rocm_valid_kv_descriptor_append_args(const rocm_kv_descriptor_append_launch_args &args)
{
	return args.version == ROCM_KV_DESCRIPTOR_APPEND_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_KV_DESCRIPTOR_APPEND_LAUNCH_ARGS_BYTES &&
		(args.reserved0 == ROCM_KV_DESCRIPTOR_APPEND_MODE_BUILD_SINGLE_PAGE || args.previous_descriptor_pointer != 0) &&
		args.output_descriptor_pointer != 0 &&
		args.new_key_pointer != 0 &&
		args.new_value_pointer != 0 &&
		(args.reserved0 == ROCM_KV_DESCRIPTOR_APPEND_MODE_BUILD_SINGLE_PAGE || args.previous_descriptor_bytes >= ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES) &&
		args.output_page_count > 0 &&
		args.output_token_count > 0 &&
		args.output_descriptor_bytes == static_cast<uint64_t>(ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES) + static_cast<uint64_t>(args.output_page_count) * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES &&
		rocm_supported_kv_mode(args.mode_code) &&
		args.block_size > 0 &&
		args.key_width > 0 &&
		args.value_width > 0 &&
		rocm_valid_kv_tensor_encoding(args.new_key_encoding) &&
		rocm_valid_kv_tensor_encoding(args.new_value_encoding) &&
		args.new_key_bytes > 0 &&
		args.new_value_bytes > 0;
}

__device__ bool rocm_kv_descriptor_encoding_is_row_interleaved(uint32_t encoding)
{
	return encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS_INTERLEAVED ||
		encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4_ROWS_INTERLEAVED;
}

__device__ uint64_t rocm_kv_descriptor_interleaved_row_stride(uint32_t encoding, uint32_t width)
{
	if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS_INTERLEAVED) {
		return sizeof(uint32_t) + width;
	}
	if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4_ROWS_INTERLEAVED) {
		return sizeof(uint32_t) + ((static_cast<uint64_t>(width) + 1u) / 2u);
	}
	return 0u;
}

__device__ bool rocm_kv_descriptor_trim_page(const rocm_device_kv_page_descriptor *page, uint64_t trim_start, rocm_device_kv_page_descriptor *out_page)
{
	if (page == nullptr || out_page == nullptr) {
		return false;
	}
	const uint64_t page_end = page->token_start + page->token_count;
	if (page_end <= trim_start) {
		return false;
	}
	*out_page = *page;
	if (page->token_start >= trim_start) {
		out_page->token_start = page->token_start - trim_start;
		return true;
	}
	if (!rocm_kv_descriptor_encoding_is_row_interleaved(page->key_encoding) ||
		!rocm_kv_descriptor_encoding_is_row_interleaved(page->value_encoding)) {
		return false;
	}
	const uint64_t skip_tokens = trim_start - page->token_start;
	const uint64_t key_stride = rocm_kv_descriptor_interleaved_row_stride(page->key_encoding, page->key_width);
	const uint64_t value_stride = rocm_kv_descriptor_interleaved_row_stride(page->value_encoding, page->value_width);
	if (key_stride == 0u ||
		value_stride == 0u ||
		page->key_bytes != key_stride * page->token_count ||
		page->value_bytes != value_stride * page->token_count ||
		skip_tokens >= page->token_count) {
		return false;
	}
	out_page->token_start = 0u;
	out_page->token_count = page_end - trim_start;
	out_page->key_pointer = page->key_pointer + key_stride * skip_tokens;
	out_page->value_pointer = page->value_pointer + value_stride * skip_tokens;
	out_page->key_bytes = key_stride * out_page->token_count;
	out_page->value_bytes = value_stride * out_page->token_count;
	return true;
}

__device__ void rocm_kv_descriptor_append_execute(const rocm_kv_descriptor_append_launch_args &args)
{
	unsigned char *output_base = reinterpret_cast<unsigned char *>(static_cast<uintptr_t>(args.output_descriptor_pointer));
	if (args.reserved0 == ROCM_KV_DESCRIPTOR_APPEND_MODE_BUILD_SINGLE_PAGE) {
		if (threadIdx.x != 0 ||
			args.output_page_count != 1u ||
			args.trim_start != 0u) {
			return;
		}
		rocm_device_kv_page_descriptor *page = reinterpret_cast<rocm_device_kv_page_descriptor *>(output_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES);
		page->token_start = 0u;
		page->token_count = args.output_token_count;
		page->key_width = args.key_width;
		page->value_width = args.value_width;
		page->key_encoding = args.new_key_encoding;
		page->value_encoding = args.new_value_encoding;
		page->key_pointer = args.new_key_pointer;
		page->value_pointer = args.new_value_pointer;
		page->key_bytes = args.new_key_bytes;
		page->value_bytes = args.new_value_bytes;

		rocm_device_kv_descriptor_header *header = reinterpret_cast<rocm_device_kv_descriptor_header *>(output_base);
		header->version = ROCM_DEVICE_KV_DESCRIPTOR_VERSION;
		header->header_bytes = ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES;
		header->page_bytes = ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES;
		header->mode_code = args.mode_code;
		header->page_count = args.output_page_count;
		header->block_size = args.block_size;
		header->token_count = args.output_token_count;
		return;
	}
	const rocm_device_kv_descriptor_header *previous = reinterpret_cast<const rocm_device_kv_descriptor_header *>(static_cast<uintptr_t>(args.previous_descriptor_pointer));
	const uint64_t append_count = args.trim_start + static_cast<uint64_t>(args.output_token_count) - previous->token_count;
	if (previous->version != ROCM_DEVICE_KV_DESCRIPTOR_VERSION ||
		previous->header_bytes != ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES ||
		previous->page_bytes != ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES ||
		previous->mode_code != args.mode_code ||
		previous->block_size != args.block_size ||
		previous->page_count == 0 ||
		previous->token_count == 0 ||
			args.previous_descriptor_bytes != static_cast<uint64_t>(ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES) + static_cast<uint64_t>(previous->page_count) * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES ||
			args.trim_start + static_cast<uint64_t>(args.output_token_count) <= previous->token_count ||
			append_count == 0u ||
			append_count > args.block_size) {
		return;
	}

	const unsigned char *previous_base = reinterpret_cast<const unsigned char *>(previous);
	if (args.reserved0 == ROCM_KV_DESCRIPTOR_APPEND_MODE_GROW_LAST_PAGE &&
		args.trim_start == 0u &&
		args.output_page_count == previous->page_count &&
		args.output_token_count == previous->token_count + append_count) {
		const uint32_t last_index = previous->page_count - 1u;
		if (args.previous_descriptor_pointer != args.output_descriptor_pointer) {
			for (uint32_t page_index = threadIdx.x; page_index < previous->page_count; page_index += blockDim.x) {
				const rocm_device_kv_page_descriptor *page = reinterpret_cast<const rocm_device_kv_page_descriptor *>(previous_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + page_index * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES);
				rocm_device_kv_page_descriptor *out_page = reinterpret_cast<rocm_device_kv_page_descriptor *>(output_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + page_index * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES);
				*out_page = *page;
			}
			__syncthreads();
		}
		if (threadIdx.x != 0) {
			return;
		}
		rocm_device_kv_page_descriptor *last_page = reinterpret_cast<rocm_device_kv_page_descriptor *>(output_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + last_index * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES);
		const rocm_device_kv_page_descriptor *previous_last = reinterpret_cast<const rocm_device_kv_page_descriptor *>(previous_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + last_index * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES);
		if (previous_last->token_start + previous_last->token_count != previous->token_count ||
			previous_last->key_pointer != args.new_key_pointer ||
			previous_last->value_pointer != args.new_value_pointer ||
			previous_last->key_width != args.key_width ||
			previous_last->value_width != args.value_width ||
			previous_last->key_encoding != args.new_key_encoding ||
			previous_last->value_encoding != args.new_value_encoding ||
			args.new_key_bytes <= previous_last->key_bytes ||
			args.new_value_bytes <= previous_last->value_bytes) {
			return;
		}
		last_page->token_count = previous_last->token_count + append_count;
		last_page->key_bytes = args.new_key_bytes;
		last_page->value_bytes = args.new_value_bytes;
		rocm_device_kv_descriptor_header *header = reinterpret_cast<rocm_device_kv_descriptor_header *>(output_base);
		header->version = ROCM_DEVICE_KV_DESCRIPTOR_VERSION;
		header->header_bytes = ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES;
		header->page_bytes = ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES;
		header->mode_code = args.mode_code;
		header->page_count = args.output_page_count;
		header->block_size = args.block_size;
		header->token_count = args.output_token_count;
		return;
	}
	if (args.reserved0 == ROCM_KV_DESCRIPTOR_APPEND_MODE_GROW_LAST_PAGE &&
		args.trim_start > 0u &&
		args.output_page_count <= previous->page_count) {
		if (threadIdx.x != 0) {
			return;
		}
		const uint32_t last_index = previous->page_count - 1u;
		const uint32_t output_last_index = args.output_page_count - 1u;
		uint32_t output_index = 0u;
		for (uint32_t page_index = 0u; page_index < last_index; ++page_index) {
			const rocm_device_kv_page_descriptor *page = reinterpret_cast<const rocm_device_kv_page_descriptor *>(previous_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + page_index * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES);
			rocm_device_kv_page_descriptor local_page {};
			if (!rocm_kv_descriptor_trim_page(page, args.trim_start, &local_page)) {
				continue;
			}
			if (output_index >= output_last_index) {
				return;
			}
			rocm_device_kv_page_descriptor *out_page = reinterpret_cast<rocm_device_kv_page_descriptor *>(output_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + output_index * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES);
			*out_page = local_page;
			++output_index;
		}
		if (output_index != output_last_index) {
			return;
		}
		const rocm_device_kv_page_descriptor *previous_last = reinterpret_cast<const rocm_device_kv_page_descriptor *>(previous_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + last_index * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES);
		rocm_device_kv_page_descriptor retained_last {};
		if (!rocm_kv_descriptor_trim_page(previous_last, args.trim_start, &retained_last) ||
			previous_last->token_start + previous_last->token_count != previous->token_count ||
			retained_last.key_pointer != args.new_key_pointer ||
			retained_last.value_pointer != args.new_value_pointer ||
			retained_last.key_width != args.key_width ||
			retained_last.value_width != args.value_width ||
			retained_last.key_encoding != args.new_key_encoding ||
			retained_last.value_encoding != args.new_value_encoding ||
			args.new_key_bytes <= retained_last.key_bytes ||
			args.new_value_bytes <= retained_last.value_bytes) {
			return;
		}
		rocm_device_kv_page_descriptor *last_page = reinterpret_cast<rocm_device_kv_page_descriptor *>(output_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + output_last_index * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES);
		*last_page = retained_last;
		last_page->token_count = retained_last.token_count + append_count;
		last_page->key_bytes = args.new_key_bytes;
		last_page->value_bytes = args.new_value_bytes;

		rocm_device_kv_descriptor_header *header = reinterpret_cast<rocm_device_kv_descriptor_header *>(output_base);
		header->version = ROCM_DEVICE_KV_DESCRIPTOR_VERSION;
		header->header_bytes = ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES;
		header->page_bytes = ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES;
		header->mode_code = args.mode_code;
		header->page_count = args.output_page_count;
		header->block_size = args.block_size;
		header->token_count = args.output_token_count;
		return;
	}
	if (args.trim_start == 0u &&
		args.previous_descriptor_pointer == args.output_descriptor_pointer &&
		args.output_page_count == previous->page_count + 1u) {
		if (threadIdx.x != 0) {
			return;
		}
		rocm_device_kv_page_descriptor *new_page = reinterpret_cast<rocm_device_kv_page_descriptor *>(output_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + previous->page_count * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES);
		new_page->token_start = static_cast<uint64_t>(args.output_token_count) - append_count;
		new_page->token_count = append_count;
		new_page->key_width = args.key_width;
		new_page->value_width = args.value_width;
		new_page->key_encoding = args.new_key_encoding;
		new_page->value_encoding = args.new_value_encoding;
		new_page->key_pointer = args.new_key_pointer;
		new_page->value_pointer = args.new_value_pointer;
		new_page->key_bytes = args.new_key_bytes;
		new_page->value_bytes = args.new_value_bytes;

		rocm_device_kv_descriptor_header *header = reinterpret_cast<rocm_device_kv_descriptor_header *>(output_base);
		header->version = ROCM_DEVICE_KV_DESCRIPTOR_VERSION;
		header->header_bytes = ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES;
		header->page_bytes = ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES;
		header->mode_code = args.mode_code;
		header->page_count = args.output_page_count;
		header->block_size = args.block_size;
		header->token_count = args.output_token_count;
		return;
	}
	if (args.trim_start == 0u && args.output_page_count == previous->page_count + 1u) {
		for (uint32_t page_index = threadIdx.x; page_index < previous->page_count; page_index += blockDim.x) {
			const rocm_device_kv_page_descriptor *page = reinterpret_cast<const rocm_device_kv_page_descriptor *>(previous_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + page_index * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES);
			rocm_device_kv_page_descriptor *out_page = reinterpret_cast<rocm_device_kv_page_descriptor *>(output_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + page_index * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES);
			*out_page = *page;
		}
		__syncthreads();
		if (threadIdx.x != 0) {
			return;
		}
		rocm_device_kv_page_descriptor *new_page = reinterpret_cast<rocm_device_kv_page_descriptor *>(output_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + previous->page_count * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES);
		new_page->token_start = static_cast<uint64_t>(args.output_token_count) - append_count;
		new_page->token_count = append_count;
		new_page->key_width = args.key_width;
		new_page->value_width = args.value_width;
		new_page->key_encoding = args.new_key_encoding;
		new_page->value_encoding = args.new_value_encoding;
		new_page->key_pointer = args.new_key_pointer;
		new_page->value_pointer = args.new_value_pointer;
		new_page->key_bytes = args.new_key_bytes;
		new_page->value_bytes = args.new_value_bytes;

		rocm_device_kv_descriptor_header *header = reinterpret_cast<rocm_device_kv_descriptor_header *>(output_base);
		header->version = ROCM_DEVICE_KV_DESCRIPTOR_VERSION;
		header->header_bytes = ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES;
		header->page_bytes = ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES;
		header->mode_code = args.mode_code;
		header->page_count = args.output_page_count;
		header->block_size = args.block_size;
		header->token_count = args.output_token_count;
		return;
	}
	if (args.trim_start > 0u &&
		args.previous_descriptor_pointer == args.output_descriptor_pointer &&
		previous->page_count == previous->token_count &&
		args.output_page_count == args.output_token_count &&
		args.output_page_count + args.trim_start == previous->page_count + 1u) {
		const uint32_t copied_pages = args.output_page_count - 1u;
		for (uint32_t tile_start = 0u; tile_start < copied_pages; tile_start += blockDim.x) {
			const uint32_t output_index = tile_start + threadIdx.x;
			rocm_device_kv_page_descriptor local_page {};
			const bool active = output_index < copied_pages;
			if (active) {
				const uint64_t source_index = static_cast<uint64_t>(output_index) + args.trim_start;
				const rocm_device_kv_page_descriptor *page = reinterpret_cast<const rocm_device_kv_page_descriptor *>(previous_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + source_index * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES);
				local_page = *page;
				local_page.token_start = output_index;
			}
			__syncthreads();
			if (active) {
				rocm_device_kv_page_descriptor *out_page = reinterpret_cast<rocm_device_kv_page_descriptor *>(output_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + output_index * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES);
				*out_page = local_page;
			}
			__syncthreads();
		}
		if (threadIdx.x != 0) {
			return;
		}
		rocm_device_kv_page_descriptor *new_page = reinterpret_cast<rocm_device_kv_page_descriptor *>(output_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + copied_pages * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES);
		new_page->token_start = static_cast<uint64_t>(args.output_token_count - 1u);
		new_page->token_count = 1;
		new_page->key_width = args.key_width;
		new_page->value_width = args.value_width;
		new_page->key_encoding = args.new_key_encoding;
		new_page->value_encoding = args.new_value_encoding;
		new_page->key_pointer = args.new_key_pointer;
		new_page->value_pointer = args.new_value_pointer;
		new_page->key_bytes = args.new_key_bytes;
		new_page->value_bytes = args.new_value_bytes;

		rocm_device_kv_descriptor_header *header = reinterpret_cast<rocm_device_kv_descriptor_header *>(output_base);
		header->version = ROCM_DEVICE_KV_DESCRIPTOR_VERSION;
		header->header_bytes = ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES;
		header->page_bytes = ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES;
		header->mode_code = args.mode_code;
		header->page_count = args.output_page_count;
		header->block_size = args.block_size;
		header->token_count = args.output_token_count;
		return;
	}
	if (args.trim_start > 0u &&
		args.previous_descriptor_pointer != args.output_descriptor_pointer &&
		previous->page_count == previous->token_count &&
		args.output_page_count == args.output_token_count &&
		args.output_page_count + args.trim_start == previous->page_count + 1u) {
		const uint32_t copied_pages = args.output_page_count - 1u;
		for (uint32_t output_index = threadIdx.x; output_index < copied_pages; output_index += blockDim.x) {
			const uint64_t source_index = static_cast<uint64_t>(output_index) + args.trim_start;
			const rocm_device_kv_page_descriptor *page = reinterpret_cast<const rocm_device_kv_page_descriptor *>(previous_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + source_index * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES);
			rocm_device_kv_page_descriptor *out_page = reinterpret_cast<rocm_device_kv_page_descriptor *>(output_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + output_index * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES);
			*out_page = *page;
			out_page->token_start = output_index;
		}
		__syncthreads();
		if (threadIdx.x != 0) {
			return;
		}
		rocm_device_kv_page_descriptor *new_page = reinterpret_cast<rocm_device_kv_page_descriptor *>(output_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + copied_pages * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES);
		new_page->token_start = static_cast<uint64_t>(args.output_token_count - 1u);
		new_page->token_count = 1;
		new_page->key_width = args.key_width;
		new_page->value_width = args.value_width;
		new_page->key_encoding = args.new_key_encoding;
		new_page->value_encoding = args.new_value_encoding;
		new_page->key_pointer = args.new_key_pointer;
		new_page->value_pointer = args.new_value_pointer;
		new_page->key_bytes = args.new_key_bytes;
		new_page->value_bytes = args.new_value_bytes;

		rocm_device_kv_descriptor_header *header = reinterpret_cast<rocm_device_kv_descriptor_header *>(output_base);
		header->version = ROCM_DEVICE_KV_DESCRIPTOR_VERSION;
		header->header_bytes = ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES;
		header->page_bytes = ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES;
		header->mode_code = args.mode_code;
		header->page_count = args.output_page_count;
		header->block_size = args.block_size;
		header->token_count = args.output_token_count;
		return;
	}
	if (threadIdx.x != 0) {
		return;
	}
	uint32_t output_index = 0;
	for (uint32_t page_index = 0; page_index < previous->page_count; ++page_index) {
		const rocm_device_kv_page_descriptor *page = reinterpret_cast<const rocm_device_kv_page_descriptor *>(previous_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + page_index * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES);
		const uint64_t page_end = page->token_start + page->token_count;
		if (page_end <= args.trim_start) {
			continue;
		}
		if (output_index + 1u >= args.output_page_count) {
			return;
		}
		rocm_device_kv_page_descriptor local_page {};
		if (!rocm_kv_descriptor_trim_page(page, args.trim_start, &local_page)) {
			return;
		}
		rocm_device_kv_page_descriptor *out_page = reinterpret_cast<rocm_device_kv_page_descriptor *>(output_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + output_index * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES);
		*out_page = local_page;
		++output_index;
	}
	if (output_index + 1u != args.output_page_count) {
		return;
	}
	rocm_device_kv_page_descriptor *new_page = reinterpret_cast<rocm_device_kv_page_descriptor *>(output_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + output_index * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES);
	new_page->token_start = static_cast<uint64_t>(args.output_token_count) - append_count;
	new_page->token_count = append_count;
	new_page->key_width = args.key_width;
	new_page->value_width = args.value_width;
	new_page->key_encoding = args.new_key_encoding;
	new_page->value_encoding = args.new_value_encoding;
	new_page->key_pointer = args.new_key_pointer;
	new_page->value_pointer = args.new_value_pointer;
	new_page->key_bytes = args.new_key_bytes;
	new_page->value_bytes = args.new_value_bytes;

	rocm_device_kv_descriptor_header *header = reinterpret_cast<rocm_device_kv_descriptor_header *>(output_base);
	header->version = ROCM_DEVICE_KV_DESCRIPTOR_VERSION;
	header->header_bytes = ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES;
	header->page_bytes = ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES;
	header->mode_code = args.mode_code;
	header->page_count = args.output_page_count;
	header->block_size = args.block_size;
	header->token_count = args.output_token_count;
}

extern "C" __global__ void rocm_kv_descriptor_append(const unsigned char *packet)
{
	if (packet == nullptr || blockIdx.x != 0) {
		return;
	}
	const rocm_kv_descriptor_append_launch_args &args = *reinterpret_cast<const rocm_kv_descriptor_append_launch_args *>(packet);
	if (!rocm_valid_kv_descriptor_append_args(args)) {
		return;
	}
	rocm_kv_descriptor_append_execute(args);
}

extern "C" __global__ void rocm_kv_encode_token_value_norm_descriptor_append(const unsigned char *packet)
{
	if (packet == nullptr || blockIdx.x > 2u || blockDim.x == 0 || blockDim.x > ROCM_KV_ENCODE_TOKEN_BLOCK_SIZE) {
		return;
	}
	const rocm_kv_encode_token_value_norm_descriptor_append_launch_args &args = *reinterpret_cast<const rocm_kv_encode_token_value_norm_descriptor_append_launch_args *>(packet);
	if (args.version != ROCM_KV_ENCODE_TOKEN_VALUE_NORM_DESCRIPTOR_APPEND_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_KV_ENCODE_TOKEN_VALUE_NORM_DESCRIPTOR_APPEND_LAUNCH_ARGS_BYTES) {
		return;
	}
	if (blockIdx.x < 2u) {
		if (!rocm_valid_kv_encode_token_value_norm_args(args.encode)) {
			return;
		}
		const bool encode_key = blockIdx.x == 0;
		const uint32_t row_count = args.encode.reserved2 == 0u ? 1u : static_cast<uint32_t>(args.encode.reserved2);
		if (encode_key) {
			const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.encode.key_input_pointer));
			unsigned char *output = reinterpret_cast<unsigned char *>(static_cast<uintptr_t>(args.encode.key_output_pointer));
			const uint32_t row_width = args.encode.reserved0 == 0u ? args.encode.key_count : static_cast<uint32_t>(args.encode.reserved0);
			rocm_encode_kv_token_tensor(input, output, args.encode.key_count, args.encode.key_encoding, row_width, row_count);
			return;
		}
		const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.encode.value_input_pointer));
		unsigned char *output = reinterpret_cast<unsigned char *>(static_cast<uintptr_t>(args.encode.value_output_pointer));
		const uint32_t row_width = args.encode.reserved1 == 0u ? args.encode.value_count : static_cast<uint32_t>(args.encode.reserved1);
		rocm_encode_kv_token_tensor_value_norm(input, output, args.encode.value_count, args.encode.value_encoding, row_width, row_count, args.encode.value_head_dim, args.encode.value_head_count, rocm_float_from_bits(args.encode.value_epsilon_bits));
		return;
	}
	if (!rocm_valid_kv_descriptor_append_args(args.descriptor)) {
		return;
	}
	rocm_kv_descriptor_append_execute(args.descriptor);
}

__device__ bool rocm_valid_attention_args(const rocm_attention_launch_args &args)
{
	const float scale = rocm_float_from_bits(args.scale_bits);
	if (args.version != ROCM_ATTENTION_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_ATTENTION_LAUNCH_ARGS_BYTES ||
		args.query_pointer == 0 ||
		args.output_pointer == 0 ||
		args.weight_pointer == 0 ||
		args.dim == 0 ||
		args.token_count == 0 ||
		args.query_bytes != args.dim * sizeof(float) ||
		args.output_bytes != args.dim * sizeof(float) ||
		args.weight_bytes != args.token_count * sizeof(float) ||
		!(scale >= 0.0f) ||
		!isfinite(scale)) {
		return false;
	}
	if (args.kv_source == ROCM_ATTENTION_KV_SOURCE_CONTIGUOUS) {
		return args.key_pointer != 0 &&
			args.value_pointer != 0 &&
			args.key_bytes == args.dim * args.token_count * sizeof(float) &&
			args.value_bytes == args.dim * args.token_count * sizeof(float) &&
			args.descriptor_pointer == 0 &&
			args.descriptor_bytes == 0;
	}
	return rocm_valid_device_kv_attention_descriptor(args);
}

__device__ bool rocm_valid_attention_heads_args(const rocm_attention_heads_launch_args &args)
{
	const float scale = rocm_float_from_bits(args.scale_bits);
	const uint32_t kv_head_count = rocm_attention_kv_head_count(args.reserved0, args.head_count);
	const bool global_weights = args.weight_pointer != 0 &&
		args.weight_bytes == args.head_count * args.token_count * sizeof(float);
	const bool shared_weights = args.weight_pointer == 0 &&
		args.weight_bytes == 0 &&
		args.token_count <= ROCM_ATTENTION_HEADS_SHARED_MAX_TOKENS;
	if (args.version != ROCM_ATTENTION_HEADS_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_ATTENTION_HEADS_LAUNCH_ARGS_BYTES ||
		args.query_pointer == 0 ||
		args.output_pointer == 0 ||
		args.dim == 0 ||
		args.token_count == 0 ||
		args.head_count == 0 ||
		args.query_bytes != args.head_count * args.dim * sizeof(float) ||
		args.output_bytes != args.head_count * args.dim * sizeof(float) ||
		(!global_weights && !shared_weights) ||
		!(scale >= 0.0f) ||
		!isfinite(scale)) {
		return false;
	}
	if (args.kv_source == ROCM_ATTENTION_KV_SOURCE_CONTIGUOUS) {
		return args.key_pointer != 0 &&
			args.value_pointer != 0 &&
			kv_head_count > 0 &&
			args.head_count % kv_head_count == 0 &&
			args.key_bytes == args.dim * args.token_count * kv_head_count * sizeof(float) &&
			args.value_bytes == args.dim * args.token_count * kv_head_count * sizeof(float) &&
			args.descriptor_pointer == 0 &&
			args.descriptor_bytes == 0;
	}
	if (args.kv_source != ROCM_ATTENTION_KV_SOURCE_DEVICE ||
		args.key_pointer != 0 ||
		args.value_pointer != 0 ||
		args.key_bytes != 0 ||
		args.value_bytes != 0 ||
		args.descriptor_pointer == 0 ||
		args.descriptor_bytes < ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES) {
		return false;
	}
	rocm_attention_launch_args single = {};
	single.version = ROCM_ATTENTION_LAUNCH_ARGS_VERSION;
	single.total_bytes = ROCM_ATTENTION_LAUNCH_ARGS_BYTES;
	single.query_pointer = args.query_pointer;
	single.output_pointer = args.output_pointer;
	single.weight_pointer = args.weight_pointer;
	single.dim = args.dim;
	single.token_count = args.token_count;
	single.query_bytes = args.dim * sizeof(float);
	single.output_bytes = args.dim * sizeof(float);
	single.weight_bytes = args.token_count * sizeof(float);
	single.kv_source = args.kv_source;
	single.scale_bits = args.scale_bits;
	single.descriptor_pointer = args.descriptor_pointer;
	single.descriptor_bytes = args.descriptor_bytes;
	return rocm_valid_device_kv_attention_descriptor_header(single);
}

__device__ bool rocm_valid_attention_heads_batch_causal_args(const rocm_attention_heads_batch_causal_launch_args &args)
{
	const float scale = rocm_float_from_bits(args.scale_bits);
	const uint32_t kv_head_count = rocm_attention_kv_head_count(args.reserved0, args.head_count);
	const uint64_t query_elements = static_cast<uint64_t>(args.query_count) * args.head_count * args.dim;
	const bool causal_window_ok = static_cast<uint64_t>(args.query_start_token) + args.query_count <= args.token_count;
	const bool global_weights = args.weight_pointer != 0 &&
		static_cast<uint64_t>(args.weight_bytes) == static_cast<uint64_t>(args.query_count) * args.head_count * args.token_count * sizeof(float);
	const bool shared_weights = args.weight_pointer == 0 &&
		args.weight_bytes == 0 &&
		args.token_count <= ROCM_ATTENTION_HEADS_SHARED_MAX_TOKENS;
	if (args.version != ROCM_ATTENTION_HEADS_BATCH_CAUSAL_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_ATTENTION_HEADS_BATCH_CAUSAL_LAUNCH_ARGS_BYTES ||
		args.query_pointer == 0 ||
		args.output_pointer == 0 ||
		args.dim == 0 ||
		args.token_count == 0 ||
		args.head_count == 0 ||
		args.query_count == 0 ||
		!causal_window_ok ||
		static_cast<uint64_t>(args.query_bytes) != query_elements * sizeof(float) ||
		static_cast<uint64_t>(args.output_bytes) != query_elements * sizeof(float) ||
		(!global_weights && !shared_weights) ||
		!(scale >= 0.0f) ||
		!isfinite(scale)) {
		return false;
	}
	if (args.kv_source == ROCM_ATTENTION_KV_SOURCE_CONTIGUOUS) {
		return args.key_pointer != 0 &&
			args.value_pointer != 0 &&
			kv_head_count > 0 &&
			args.head_count % kv_head_count == 0 &&
			static_cast<uint64_t>(args.key_bytes) == static_cast<uint64_t>(args.dim) * args.token_count * kv_head_count * sizeof(float) &&
			static_cast<uint64_t>(args.value_bytes) == static_cast<uint64_t>(args.dim) * args.token_count * kv_head_count * sizeof(float) &&
			args.descriptor_pointer == 0 &&
			args.descriptor_bytes == 0;
	}
	if (args.kv_source != ROCM_ATTENTION_KV_SOURCE_DEVICE ||
		args.key_pointer != 0 ||
		args.value_pointer != 0 ||
		args.key_bytes != 0 ||
		args.value_bytes != 0 ||
		args.descriptor_pointer == 0 ||
		args.descriptor_bytes < ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES) {
		return false;
	}
	rocm_attention_launch_args single = {};
	single.version = ROCM_ATTENTION_LAUNCH_ARGS_VERSION;
	single.total_bytes = ROCM_ATTENTION_LAUNCH_ARGS_BYTES;
	single.query_pointer = args.query_pointer;
	single.output_pointer = args.output_pointer;
	single.weight_pointer = args.weight_pointer;
	single.dim = args.dim;
	single.token_count = args.token_count;
	single.query_bytes = args.dim * sizeof(float);
	single.output_bytes = args.dim * sizeof(float);
	single.weight_bytes = args.token_count * sizeof(float);
	single.kv_source = args.kv_source;
	single.scale_bits = args.scale_bits;
	single.descriptor_pointer = args.descriptor_pointer;
	single.descriptor_bytes = args.descriptor_bytes;
	return rocm_valid_device_kv_attention_descriptor_header(single);
}

__device__ bool rocm_valid_attention_heads_batch_capped_args(const rocm_attention_heads_batch_causal_launch_args &args)
{
	return rocm_valid_attention_heads_batch_causal_args(args) &&
		args.reserved1 != 0 &&
		args.reserved2 == static_cast<uint64_t>(args.query_count) * sizeof(uint32_t);
}

__device__ bool rocm_valid_attention_heads_lane_batch_args(const rocm_attention_heads_lane_batch_launch_args &args)
{
	const float scale = rocm_float_from_bits(args.scale_bits);
	const uint64_t query_elements = static_cast<uint64_t>(args.lane_count) * args.head_count * args.dim;
	const bool global_weights = args.weight_pointer != 0 &&
		static_cast<uint64_t>(args.weight_bytes) == static_cast<uint64_t>(args.lane_count) * args.head_count * args.max_token_count * sizeof(float);
	const bool shared_weights = args.weight_pointer == 0 &&
		args.weight_bytes == 0 &&
		args.max_token_count <= ROCM_ATTENTION_HEADS_SHARED_MAX_TOKENS;
	return args.version == ROCM_ATTENTION_HEADS_LANE_BATCH_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_ATTENTION_HEADS_LANE_BATCH_LAUNCH_ARGS_BYTES &&
		args.query_pointer != 0 &&
		args.output_pointer != 0 &&
		args.lane_pointer != 0 &&
		args.dim > 0 &&
		args.lane_count > 0 &&
		args.head_count > 0 &&
		args.key_heads > 0 &&
		args.key_heads <= args.head_count &&
		args.head_count % args.key_heads == 0 &&
		args.max_token_count > 0 &&
		static_cast<uint64_t>(args.query_bytes) == query_elements * sizeof(float) &&
		static_cast<uint64_t>(args.output_bytes) == query_elements * sizeof(float) &&
		static_cast<uint64_t>(args.lane_bytes) == static_cast<uint64_t>(args.lane_count) * ROCM_ATTENTION_HEADS_LANE_DESCRIPTOR_BYTES &&
		(global_weights || shared_weights) &&
		(scale >= 0.0f) &&
		isfinite(scale);
}

__device__ bool rocm_valid_attention_heads_batch_causal_query_rms_rope_args(const rocm_attention_heads_batch_causal_query_rms_rope_launch_args &args)
{
	const float scale = rocm_float_from_bits(args.scale_bits);
	const float query_epsilon = rocm_float_from_bits(args.query_epsilon_bits);
	const float rope_base = rocm_float_from_bits(args.rope_base_bits);
	const float rope_frequency_scale = rocm_float_from_bits(args.rope_frequency_scale_bits);
	const uint32_t kv_head_count = args.key_heads == 0u ? 1u : args.key_heads;
	const uint64_t query_elements = static_cast<uint64_t>(args.query_count) * args.head_count * args.dim;
	const bool causal_window_ok = static_cast<uint64_t>(args.query_start_token) + args.query_count <= args.token_count;
	const bool global_weights = args.attention_weight_pointer != 0 &&
		static_cast<uint64_t>(args.attention_weight_bytes) == static_cast<uint64_t>(args.query_count) * args.head_count * args.token_count * sizeof(float);
	const bool shared_weights = args.attention_weight_pointer == 0 &&
		args.attention_weight_bytes == 0 &&
		args.token_count <= ROCM_ATTENTION_HEADS_SHARED_MAX_TOKENS;
	bool query_weight_ok = false;
	if ((args.query_flags & ~ROCM_RMS_NORM_LAUNCH_FLAG_MASK) == 0u) {
		if (args.query_weight_encoding == ROCM_RMS_NORM_WEIGHT_ENCODING_NONE) {
			query_weight_ok = args.query_weight_pointer == 0 &&
				args.query_weight_bytes == 0 &&
				(args.query_flags & ROCM_RMS_NORM_LAUNCH_FLAG_ADD_UNIT_WEIGHT) == 0u;
		} else if (args.query_weight_encoding == ROCM_RMS_NORM_WEIGHT_ENCODING_F32) {
			query_weight_ok = args.query_weight_pointer != 0 &&
				static_cast<uint64_t>(args.query_weight_bytes) == static_cast<uint64_t>(args.dim) * sizeof(float);
		} else if (args.query_weight_encoding == ROCM_RMS_NORM_WEIGHT_ENCODING_BF16) {
			query_weight_ok = args.query_weight_pointer != 0 &&
				static_cast<uint64_t>(args.query_weight_bytes) == static_cast<uint64_t>(args.dim) * sizeof(uint16_t);
		}
	}
	if (args.version != ROCM_ATTENTION_HEADS_BATCH_CAUSAL_QUERY_RMS_ROPE_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_ATTENTION_HEADS_BATCH_CAUSAL_QUERY_RMS_ROPE_LAUNCH_ARGS_BYTES ||
		args.query_input_pointer == 0 ||
		args.output_pointer == 0 ||
		args.dim == 0 ||
		(args.dim & 1u) != 0u ||
		args.token_count == 0 ||
		args.head_count == 0 ||
		args.query_count == 0 ||
		kv_head_count == 0 ||
		args.head_count % kv_head_count != 0 ||
		!causal_window_ok ||
		static_cast<uint64_t>(args.query_input_bytes) != query_elements * sizeof(float) ||
		static_cast<uint64_t>(args.output_bytes) != query_elements * sizeof(float) ||
		(!global_weights && !shared_weights) ||
		!query_weight_ok ||
		!(scale >= 0.0f) ||
		!isfinite(scale) ||
		!(query_epsilon >= 0.0f) ||
		!isfinite(query_epsilon) ||
		!(rope_base > 0.0f) ||
		!isfinite(rope_base) ||
		!(rope_frequency_scale > 0.0f) ||
		!isfinite(rope_frequency_scale) ||
		(args.rope_frequency_dim != 0u && args.rope_frequency_dim < args.dim) ||
		args.rope_rotary_count > args.dim ||
		(args.rope_rotary_count & 1u) != 0u) {
		return false;
	}
	if (args.kv_source == ROCM_ATTENTION_KV_SOURCE_CONTIGUOUS) {
		return args.key_pointer != 0 &&
			args.value_pointer != 0 &&
			static_cast<uint64_t>(args.key_bytes) == static_cast<uint64_t>(args.dim) * args.token_count * kv_head_count * sizeof(float) &&
			static_cast<uint64_t>(args.value_bytes) == static_cast<uint64_t>(args.dim) * args.token_count * kv_head_count * sizeof(float) &&
			args.descriptor_pointer == 0 &&
			args.descriptor_bytes == 0;
	}
	if (args.kv_source != ROCM_ATTENTION_KV_SOURCE_DEVICE ||
		args.key_pointer != 0 ||
		args.value_pointer != 0 ||
		args.key_bytes != 0 ||
		args.value_bytes != 0 ||
		args.descriptor_pointer == 0 ||
		args.descriptor_bytes < ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES) {
		return false;
	}
	rocm_attention_launch_args single = {};
	single.version = ROCM_ATTENTION_LAUNCH_ARGS_VERSION;
	single.total_bytes = ROCM_ATTENTION_LAUNCH_ARGS_BYTES;
	single.query_pointer = args.query_input_pointer;
	single.output_pointer = args.output_pointer;
	single.weight_pointer = args.attention_weight_pointer == 0 ? 0 : args.attention_weight_pointer;
	single.dim = args.dim;
	single.token_count = args.token_count;
	single.query_bytes = args.dim * sizeof(float);
	single.output_bytes = args.dim * sizeof(float);
	single.weight_bytes = args.token_count * sizeof(float);
	single.kv_source = args.kv_source;
	single.scale_bits = args.scale_bits;
	single.descriptor_pointer = args.descriptor_pointer;
	single.descriptor_bytes = args.descriptor_bytes;
	return rocm_valid_device_kv_attention_descriptor_header(single);
}

__device__ bool rocm_valid_attention_heads_chunked_args(const rocm_attention_heads_chunked_launch_args &args)
{
	const float scale = rocm_float_from_bits(args.scale_bits);
	if (args.version != ROCM_ATTENTION_HEADS_CHUNKED_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_ATTENTION_HEADS_CHUNKED_LAUNCH_ARGS_BYTES ||
		args.query_pointer == 0 ||
		args.descriptor_pointer == 0 ||
		args.partial_pointer == 0 ||
		args.stats_pointer == 0 ||
		args.output_pointer == 0 ||
		args.dim == 0 ||
		args.dim > ROCM_ATTENTION_HEADS_CHUNKED_BLOCK_SIZE ||
		args.token_count == 0 ||
		args.head_count == 0 ||
		args.key_heads == 0 ||
		args.key_heads > args.head_count ||
		args.head_count % args.key_heads != 0 ||
		args.dim > (~0u) / args.key_heads ||
		(args.key_heads > 1u && (args.dim & 1u) != 0u) ||
		args.chunk_size == 0 ||
		args.chunk_count == 0 ||
		args.chunk_count != (args.token_count + args.chunk_size - 1u) / args.chunk_size ||
		static_cast<uint64_t>(args.query_bytes) != static_cast<uint64_t>(args.head_count) * args.dim * sizeof(float) ||
		static_cast<uint64_t>(args.partial_bytes) != static_cast<uint64_t>(args.head_count) * args.chunk_count * args.dim * sizeof(float) ||
		static_cast<uint64_t>(args.stats_bytes) != static_cast<uint64_t>(args.head_count) * args.chunk_count * 2u * sizeof(float) ||
		static_cast<uint64_t>(args.output_bytes) != static_cast<uint64_t>(args.head_count) * args.dim * sizeof(float) ||
		!(scale >= 0.0f) ||
		!isfinite(scale)) {
		return false;
	}
	if (args.descriptor_bytes < ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES) {
		return false;
	}
	const rocm_device_kv_descriptor_header *header = reinterpret_cast<const rocm_device_kv_descriptor_header *>(static_cast<uintptr_t>(args.descriptor_pointer));
	return header->version == ROCM_DEVICE_KV_DESCRIPTOR_VERSION &&
		header->header_bytes == ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES &&
		header->page_bytes == ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES &&
		header->mode_code == ROCM_DEVICE_KV_DESCRIPTOR_MODE_KQ8VQ4 &&
		header->page_count > 0 &&
		header->token_count == args.token_count &&
		args.descriptor_bytes == static_cast<uint64_t>(ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES) + static_cast<uint64_t>(header->page_count) * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES;
}

__device__ bool rocm_valid_attention_heads_batch_chunked_args(const rocm_attention_heads_batch_chunked_launch_args &args)
{
	const float scale = rocm_float_from_bits(args.scale_bits);
	const uint64_t query_elements = static_cast<uint64_t>(args.query_count) * args.head_count * args.dim;
	const bool causal_window_ok = static_cast<uint64_t>(args.query_start_token) + args.query_count <= args.token_count;
	const uint32_t active_end = args.query_start_token + args.query_count;
	const uint32_t expected_chunk_count = args.chunk_size != 0u && active_end > args.chunk_start_token
		? (active_end - args.chunk_start_token + args.chunk_size - 1u) / args.chunk_size
		: 0u;
	if (args.version != ROCM_ATTENTION_HEADS_BATCH_CHUNKED_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_ATTENTION_HEADS_BATCH_CHUNKED_LAUNCH_ARGS_BYTES ||
		args.query_pointer == 0 ||
		args.descriptor_pointer == 0 ||
		args.partial_pointer == 0 ||
		args.stats_pointer == 0 ||
		args.output_pointer == 0 ||
		args.dim == 0 ||
		args.dim > ROCM_ATTENTION_HEADS_CHUNKED_BLOCK_SIZE ||
		args.token_count == 0 ||
		args.head_count == 0 ||
		args.key_heads == 0 ||
		args.key_heads > args.head_count ||
		args.head_count % args.key_heads != 0 ||
		args.dim > (~0u) / args.key_heads ||
		(args.key_heads > 1u && (args.dim & 1u) != 0u) ||
		args.query_count == 0 ||
		!causal_window_ok ||
		args.chunk_size == 0 ||
		args.chunk_count == 0 ||
		args.chunk_start_token > active_end ||
		args.chunk_count != expected_chunk_count ||
		static_cast<uint64_t>(args.query_bytes) != query_elements * sizeof(float) ||
		static_cast<uint64_t>(args.partial_bytes) != query_elements * args.chunk_count * sizeof(float) ||
		static_cast<uint64_t>(args.stats_bytes) != static_cast<uint64_t>(args.query_count) * args.head_count * args.chunk_count * 2u * sizeof(float) ||
		static_cast<uint64_t>(args.output_bytes) != query_elements * sizeof(float) ||
		!(scale >= 0.0f) ||
		!isfinite(scale)) {
		return false;
	}
	if (args.descriptor_bytes < ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES) {
		return false;
	}
	const rocm_device_kv_descriptor_header *header = reinterpret_cast<const rocm_device_kv_descriptor_header *>(static_cast<uintptr_t>(args.descriptor_pointer));
	return header->version == ROCM_DEVICE_KV_DESCRIPTOR_VERSION &&
		header->header_bytes == ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES &&
		header->page_bytes == ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES &&
		header->mode_code == ROCM_DEVICE_KV_DESCRIPTOR_MODE_KQ8VQ4 &&
		header->page_count > 0 &&
		header->token_count == args.token_count &&
		args.descriptor_bytes == static_cast<uint64_t>(ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES) + static_cast<uint64_t>(header->page_count) * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES;
}

__device__ bool rocm_valid_vector_add_args(const rocm_vector_add_launch_args &args)
{
	return args.version == ROCM_VECTOR_ADD_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_VECTOR_ADD_LAUNCH_ARGS_BYTES &&
		args.left_pointer != 0 &&
		args.right_pointer != 0 &&
		args.output_pointer != 0 &&
		args.count > 0 &&
		args.left_bytes == args.count * sizeof(float) &&
		args.right_bytes == args.count * sizeof(float) &&
		args.output_bytes == args.count * sizeof(float);
}

__device__ bool rocm_valid_vector_add_scaled_args(const rocm_vector_add_scaled_launch_args &args)
{
	return args.version == ROCM_VECTOR_ADD_SCALED_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_VECTOR_ADD_SCALED_LAUNCH_ARGS_BYTES &&
		args.left_pointer != 0 &&
		args.right_pointer != 0 &&
		args.output_pointer != 0 &&
		args.count > 0 &&
		args.left_bytes == args.count * sizeof(float) &&
		args.right_bytes == args.count * sizeof(float) &&
		args.output_bytes == args.count * sizeof(float) &&
		isfinite(rocm_float_from_bits(args.scale_bits));
}

__device__ bool rocm_valid_vector_scale_args(const rocm_vector_scale_launch_args &args)
{
	return args.version == ROCM_VECTOR_SCALE_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_VECTOR_SCALE_LAUNCH_ARGS_BYTES &&
		args.input_pointer != 0 &&
		args.output_pointer != 0 &&
		args.count > 0 &&
		args.input_bytes == args.count * sizeof(float) &&
		args.output_bytes == args.count * sizeof(float) &&
		isfinite(rocm_float_from_bits(args.scale_bits));
}

__device__ bool rocm_valid_per_layer_input_transpose_args(const rocm_per_layer_input_transpose_launch_args &args)
{
	if (args.version != ROCM_PER_LAYER_INPUT_TRANSPOSE_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_PER_LAYER_INPUT_TRANSPOSE_LAUNCH_ARGS_BYTES ||
		args.input_pointer == 0 ||
		args.output_pointer == 0 ||
		args.batch == 0 ||
		args.layer_count == 0 ||
		args.input_size == 0) {
		return false;
	}
	const uint64_t count = static_cast<uint64_t>(args.batch) * args.layer_count * args.input_size;
	return args.input_bytes == count * sizeof(float) &&
		args.output_bytes == count * sizeof(float);
}

__device__ bool rocm_valid_swiglu_args(const rocm_swiglu_launch_args &args)
{
	return args.version == ROCM_SWIGLU_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_SWIGLU_LAUNCH_ARGS_BYTES &&
		args.gate_pointer != 0 &&
		args.up_pointer != 0 &&
		args.output_pointer != 0 &&
		args.count > 0 &&
		args.gate_bytes == args.count * sizeof(float) &&
		args.up_bytes == args.count * sizeof(float) &&
		args.output_bytes == args.count * sizeof(float);
}

__device__ bool rocm_valid_gelu_tanh_mul_args(const rocm_gelu_tanh_mul_launch_args &args)
{
	return args.version == ROCM_GELU_TANH_MUL_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_GELU_TANH_MUL_LAUNCH_ARGS_BYTES &&
		args.gate_pointer != 0 &&
		args.up_pointer != 0 &&
		args.output_pointer != 0 &&
		args.count > 0 &&
		args.gate_bytes == args.count * sizeof(float) &&
		args.up_bytes == args.count * sizeof(float) &&
		args.output_bytes == args.count * sizeof(float);
}

__device__ bool rocm_valid_moe_router_args(const rocm_moe_router_launch_args &args)
{
	return args.version == ROCM_MOE_ROUTER_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_MOE_ROUTER_LAUNCH_ARGS_BYTES &&
		args.logit_pointer != 0 &&
		args.id_pointer != 0 &&
		args.prob_pointer != 0 &&
		args.expert_count > 0 &&
		args.top_k > 0 &&
		args.top_k <= args.expert_count &&
		args.logit_bytes == args.expert_count * sizeof(float) &&
		args.id_bytes == args.top_k * sizeof(int32_t) &&
		args.prob_bytes == args.top_k * sizeof(float);
}

__device__ bool rocm_valid_moe_lazy_args(const rocm_moe_lazy_launch_args &args)
{
	return args.version == ROCM_MOE_LAZY_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_MOE_LAZY_LAUNCH_ARGS_BYTES &&
		args.id_pointer != 0 &&
		args.resident_pointer != 0 &&
		args.selected_count > 0 &&
		args.expert_count > 0 &&
		args.id_bytes == args.selected_count * sizeof(int32_t) &&
		args.resident_bytes == args.expert_count * sizeof(uint8_t);
}

__device__ bool rocm_valid_moe_batch_route_rows_args(const rocm_moe_batch_route_rows_launch_args &args)
{
	return args.version == ROCM_MOE_BATCH_ROUTE_ROWS_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_MOE_BATCH_ROUTE_ROWS_LAUNCH_ARGS_BYTES &&
		args.input_pointer != 0 &&
		args.metadata_pointer != 0 &&
		args.output_pointer != 0 &&
		args.route_count > 0 &&
		args.row_width > 0 &&
		args.input_rows > 0 &&
		args.output_rows > 0 &&
		args.input_bytes == static_cast<uint64_t>(args.input_rows) * args.row_width * sizeof(float) &&
		args.metadata_bytes == static_cast<uint64_t>(args.route_count) * ROCM_MOE_BATCH_ROUTE_METADATA_BYTES &&
		args.output_bytes == static_cast<uint64_t>(args.output_rows) * args.row_width * sizeof(float);
}

__device__ bool rocm_valid_moe_batch_reduce_args(const rocm_moe_batch_reduce_launch_args &args)
{
	return args.version == ROCM_MOE_BATCH_REDUCE_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_MOE_BATCH_REDUCE_LAUNCH_ARGS_BYTES &&
		args.input_pointer != 0 &&
		args.output_pointer != 0 &&
		args.rows > 0 &&
		args.top_k > 0 &&
		args.row_width > 0 &&
		args.input_bytes == static_cast<uint64_t>(args.rows) * args.top_k * args.row_width * sizeof(float) &&
		args.output_bytes == static_cast<uint64_t>(args.rows) * args.row_width * sizeof(float);
}

__device__ bool rocm_valid_moe_mlx_affine_routes_args(const rocm_moe_mlx_affine_routes_launch_args &args)
{
	if (args.version != ROCM_MOE_MLX_AFFINE_ROUTES_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_MOE_MLX_AFFINE_ROUTES_LAUNCH_ARGS_BYTES ||
		args.input_pointer == 0 ||
		args.chunk_pointer == 0 ||
		args.output_pointer == 0 ||
		args.rows == 0 ||
		args.cols == 0 ||
		args.input_rows == 0 ||
		args.pair_count == 0 ||
		args.chunk_count == 0 ||
		args.group_size == 0 ||
		(args.flags & ~ROCM_MOE_MLX_AFFINE_ROUTES_FLAG_GATE_UP) != 0 ||
		!rocm_mlx_affine_bits_supported(args.bits)) {
		return false;
	}
	const uint64_t total_bits = static_cast<uint64_t>(args.cols) * args.bits;
	if ((total_bits % 32u) != 0u || (args.cols % args.group_size) != 0u) {
		return false;
	}
	return args.input_bytes == static_cast<uint64_t>(args.input_rows) * args.cols * sizeof(float) &&
		args.chunk_bytes == static_cast<uint64_t>(args.chunk_count) * ROCM_MOE_MLX_AFFINE_ROUTES_CHUNK_BYTES &&
		args.output_bytes == static_cast<uint64_t>(args.pair_count) * args.rows * sizeof(float);
}

__device__ bool rocm_valid_gguf_q4_0_projection_args(const rocm_gguf_q4_0_projection_launch_args &args, bool fused_gate_up)
{
	if (args.version != ROCM_GGUF_Q4_0_PROJECTION_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_GGUF_Q4_0_PROJECTION_LAUNCH_ARGS_BYTES ||
		args.input_pointer == 0 ||
		args.weight_pointer == 0 ||
		args.output_pointer == 0 ||
		args.rows == 0 ||
		args.cols == 0 ||
		(args.cols & 31u) != 0u ||
		args.weight_rows == 0 ||
		args.input_bytes != args.cols * sizeof(float) ||
		args.output_bytes != args.rows * sizeof(float) ||
		args.weight_bytes != static_cast<uint64_t>(args.weight_rows) * (args.cols >> 5u) * 18u) {
		return false;
	}
	const uint64_t required_rows = static_cast<uint64_t>(args.row_offset) + (fused_gate_up ? 2u : 1u) * args.rows;
	return required_rows <= args.weight_rows;
}

__device__ bool rocm_valid_gguf_q4_k_expand_args(const rocm_gguf_q4_k_expand_launch_args &args)
{
	return args.version == ROCM_GGUF_Q4_K_EXPAND_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_GGUF_Q4_K_EXPAND_LAUNCH_ARGS_BYTES &&
		args.raw_pointer != 0 &&
		args.expanded_pointer != 0 &&
		args.block_count > 0 &&
		args.raw_bytes == static_cast<uint64_t>(args.block_count) * 144u &&
		args.expanded_bytes == static_cast<uint64_t>(args.block_count) * ROCM_GGUF_Q4_K_EXPANDED_BLOCK_BYTES;
}

__device__ bool rocm_valid_gguf_q4_0_selected_experts_args(const rocm_gguf_q4_0_selected_experts_launch_args &args)
{
	const bool q4_0_pair = args.gate_up_format == ROCM_GGUF_EXPERT_FORMAT_Q4_0 && args.down_format == ROCM_GGUF_EXPERT_FORMAT_Q4_0;
	const bool q4_k_gate = args.gate_up_format == ROCM_GGUF_EXPERT_FORMAT_Q4_K || args.gate_up_format == ROCM_GGUF_EXPERT_FORMAT_Q4_K_EXPANDED;
	const bool mixed_pair = q4_k_gate &&
		(args.down_format == ROCM_GGUF_EXPERT_FORMAT_Q5_1 || args.down_format == ROCM_GGUF_EXPERT_FORMAT_Q8_0);
	const bool gate_bytes_valid =
		(args.gate_up_format == ROCM_GGUF_EXPERT_FORMAT_Q4_0 && (args.hidden_size & 31u) == 0u && args.gate_up_bytes == args.gate_up_rows * (args.hidden_size >> 5u) * 18u) ||
		(args.gate_up_format == ROCM_GGUF_EXPERT_FORMAT_Q4_K && (args.hidden_size & 255u) == 0u && args.gate_up_bytes == args.gate_up_rows * (args.hidden_size >> 8u) * 144u) ||
		(args.gate_up_format == ROCM_GGUF_EXPERT_FORMAT_Q4_K_EXPANDED && (args.hidden_size & 255u) == 0u && args.gate_up_bytes == args.gate_up_rows * (args.hidden_size >> 8u) * ROCM_GGUF_Q4_K_EXPANDED_BLOCK_BYTES);
	const bool down_bytes_valid =
		(args.down_format == ROCM_GGUF_EXPERT_FORMAT_Q4_0 && (args.expert_ff & 31u) == 0u && args.down_bytes == args.down_rows * (args.expert_ff >> 5u) * 18u) ||
		(args.down_format == ROCM_GGUF_EXPERT_FORMAT_Q5_1 && (args.expert_ff & 31u) == 0u && args.down_bytes == args.down_rows * (args.expert_ff >> 5u) * 24u) ||
		(args.down_format == ROCM_GGUF_EXPERT_FORMAT_Q8_0 && (args.expert_ff & 31u) == 0u && args.down_bytes == args.down_rows * (args.expert_ff >> 5u) * 34u);
	if (args.version != ROCM_GGUF_Q4_0_SELECTED_EXPERTS_LAUNCH_ARGS_VERSION ||
		args.total_bytes != ROCM_GGUF_Q4_0_SELECTED_EXPERTS_LAUNCH_ARGS_BYTES ||
		args.input_pointer == 0 || args.activation_pointer == 0 || args.output_pointer == 0 ||
		args.top_k == 0 || args.top_k > ROCM_GGUF_Q4_0_SELECTED_EXPERTS_MAX_TOP_K ||
		args.hidden_size == 0 || args.expert_ff == 0 ||
		(!q4_0_pair && !mixed_pair) || !gate_bytes_valid || !down_bytes_valid ||
		args.gate_up_rows != 2u * args.expert_ff || args.down_rows != args.hidden_size ||
		args.input_bytes != args.hidden_size * sizeof(float) ||
		args.activation_bytes != args.top_k * args.expert_ff * sizeof(float) ||
		args.output_bytes != args.hidden_size * sizeof(float)) {
		return false;
	}
	for (uint32_t expert = 0; expert < args.top_k; ++expert) {
		if (args.gate_up_pointers[expert] == 0 || args.down_pointers[expert] == 0 || !isfinite(args.route_weights[expert])) {
			return false;
		}
	}
	return true;
}

__device__ bool rocm_valid_jangtq_args(const rocm_jangtq_launch_args &args)
{
	bool valid_bits = args.bits == 2 || args.bits == 4 || args.bits == 8;
	const uint64_t required_packed = (static_cast<uint64_t>(args.bits) * args.rows * args.cols + 7u) / 8u;
	const bool has_bias = (args.flags & ROCM_JANGTQ_LAUNCH_FLAG_BIAS) != 0;
	bool valid_bias = args.bias_pointer == 0 && args.bias_bytes == 0;
	if (has_bias) {
		valid_bias = args.bias_pointer != 0 && args.bias_bytes == args.rows * sizeof(float);
	}
	return args.version == ROCM_JANGTQ_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_JANGTQ_LAUNCH_ARGS_BYTES &&
		args.input_pointer != 0 &&
		args.packed_pointer != 0 &&
		args.output_pointer != 0 &&
		args.input_count == args.cols &&
		args.rows > 0 &&
		args.cols > 0 &&
		valid_bits &&
		args.group_size > 0 &&
		(args.group_size & (args.group_size - 1u)) == 0 &&
		args.input_bytes == args.cols * sizeof(float) &&
		args.packed_bytes >= required_packed &&
		args.output_bytes == args.rows * sizeof(float) &&
		rocm_positive_finite(rocm_float_from_bits(args.scale_bits)) &&
		valid_bias;
}

__device__ int8_t rocm_unpack_signed_bits(const uint8_t *packed, uint32_t bits, uint32_t index)
{
	const uint32_t bit_offset = index * bits;
	const uint32_t byte_index = bit_offset / 8u;
	const uint32_t shift = bit_offset % 8u;
	uint32_t raw = static_cast<uint32_t>(packed[byte_index] >> shift);
	if (shift + bits > 8u) {
		raw |= static_cast<uint32_t>(packed[byte_index + 1u]) << (8u - shift);
	}
	const uint32_t mask = (1u << bits) - 1u;
	const uint32_t sign_bit = 1u << (bits - 1u);
	raw &= mask;
	int32_t value = static_cast<int32_t>(raw);
	if ((raw & sign_bit) != 0) {
		value -= static_cast<int32_t>(1u << bits);
	}
	return static_cast<int8_t>(value);
}

__device__ bool rocm_valid_codebook_args(const rocm_codebook_launch_args &args)
{
	return args.version == ROCM_CODEBOOK_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_CODEBOOK_LAUNCH_ARGS_BYTES &&
		args.code_pointer != 0 &&
		args.codebook_pointer != 0 &&
		args.output_pointer != 0 &&
		args.code_count > 0 &&
		args.codebook_count > 0 &&
		args.code_dim > 0 &&
		args.code_bytes == args.code_count * sizeof(uint8_t) &&
		args.codebook_bytes == args.codebook_count * args.code_dim * sizeof(float) &&
		args.output_bytes == args.code_count * args.code_dim * sizeof(float);
}

__device__ bool rocm_valid_tiny_prefill_args(const rocm_tiny_prefill_launch_args &args)
{
	const uint32_t table_count = args.vocab_size * args.hidden_size;
	bool valid_output_weights = false;
	if (args.output_weight_encoding == ROCM_TINY_OUTPUT_WEIGHT_ENCODING_FP32) {
		valid_output_weights = args.output_weight_bytes == table_count * sizeof(float);
	} else if (args.output_weight_encoding == ROCM_TINY_OUTPUT_WEIGHT_ENCODING_FP16) {
		valid_output_weights = args.output_weight_bytes == table_count * sizeof(uint16_t);
	} else if (args.output_weight_encoding == ROCM_TINY_OUTPUT_WEIGHT_ENCODING_Q8) {
		const float scale = rocm_float_from_bits(args.q8_scale_bits);
		valid_output_weights = args.output_weight_bytes == table_count &&
			rocm_positive_finite(scale);
	}
	return args.version == ROCM_TINY_PREFILL_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_TINY_PREFILL_LAUNCH_ARGS_BYTES &&
		args.token_pointer != 0 &&
		args.embedding_pointer != 0 &&
		args.output_weight_pointer != 0 &&
		args.logit_pointer != 0 &&
		args.attention_pointer != 0 &&
		args.result_pointer != 0 &&
		args.key_pointer != 0 &&
		args.value_pointer != 0 &&
		args.token_count > 0 &&
		args.vocab_size > 0 &&
		args.hidden_size > 0 &&
		args.token_bytes == args.token_count * sizeof(int32_t) &&
		args.embedding_bytes == args.vocab_size * args.hidden_size * sizeof(float) &&
		valid_output_weights &&
		args.logit_bytes == args.vocab_size * sizeof(float) &&
		args.attention_bytes == args.token_count * sizeof(float) &&
		args.result_bytes == ROCM_GREEDY_RESULT_BYTES &&
		args.key_bytes == args.token_count * args.hidden_size * sizeof(float) &&
		args.value_bytes == args.token_count * args.hidden_size * sizeof(float);
}

__device__ bool rocm_valid_tiny_decode_args(const rocm_tiny_decode_launch_args &args)
{
	const uint32_t table_count = args.vocab_size * args.hidden_size;
	bool valid_output_weights = false;
	if (args.output_weight_encoding == ROCM_TINY_OUTPUT_WEIGHT_ENCODING_FP32) {
		valid_output_weights = args.output_weight_bytes == table_count * sizeof(float);
	} else if (args.output_weight_encoding == ROCM_TINY_OUTPUT_WEIGHT_ENCODING_FP16) {
		valid_output_weights = args.output_weight_bytes == table_count * sizeof(uint16_t);
	} else if (args.output_weight_encoding == ROCM_TINY_OUTPUT_WEIGHT_ENCODING_Q8) {
		const float scale = rocm_float_from_bits(args.q8_scale_bits);
		valid_output_weights = args.output_weight_bytes == table_count &&
			rocm_positive_finite(scale);
	}
	return args.version == ROCM_TINY_DECODE_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_TINY_DECODE_LAUNCH_ARGS_BYTES &&
		args.prior_key_pointer != 0 &&
		args.prior_value_pointer != 0 &&
		args.embedding_pointer != 0 &&
		args.output_weight_pointer != 0 &&
		args.logit_pointer != 0 &&
		args.attention_pointer != 0 &&
		args.updated_key_pointer != 0 &&
		args.updated_value_pointer != 0 &&
		args.result_pointer != 0 &&
		args.prior_token_count > 0 &&
		args.vocab_size > 0 &&
		args.hidden_size > 0 &&
		args.token_id < args.vocab_size &&
		args.prior_key_bytes == args.prior_token_count * args.hidden_size * sizeof(float) &&
		args.prior_value_bytes == args.prior_token_count * args.hidden_size * sizeof(float) &&
		args.embedding_bytes == args.vocab_size * args.hidden_size * sizeof(float) &&
		valid_output_weights &&
		args.logit_bytes == args.vocab_size * sizeof(float) &&
		args.attention_bytes == (args.prior_token_count + 1u) * sizeof(float) &&
		args.updated_key_bytes == (args.prior_token_count + 1u) * args.hidden_size * sizeof(float) &&
		args.updated_value_bytes == (args.prior_token_count + 1u) * args.hidden_size * sizeof(float) &&
		args.result_bytes == ROCM_GREEDY_RESULT_BYTES;
}

__device__ bool rocm_valid_cross_entropy_loss_args(const rocm_cross_entropy_loss_launch_args &args)
{
	return args.version == ROCM_CROSS_ENTROPY_LOSS_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_CROSS_ENTROPY_LOSS_LAUNCH_ARGS_BYTES &&
		args.logit_pointer != 0 &&
		args.target_pointer != 0 &&
		args.output_pointer != 0 &&
		args.batch > 0 &&
		args.vocab > 0 &&
		args.logit_bytes == args.batch * args.vocab * sizeof(float) &&
		args.target_bytes == args.batch * sizeof(int32_t) &&
		args.output_bytes == ROCM_CROSS_ENTROPY_LOSS_OUTPUT_BYTES;
}

__device__ bool rocm_valid_distillation_kl_loss_args(const rocm_distillation_kl_loss_launch_args &args)
{
	const double temperature = rocm_double_from_bits(args.temperature_bits);
	return args.version == ROCM_DISTILLATION_KL_LOSS_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_DISTILLATION_KL_LOSS_LAUNCH_ARGS_BYTES &&
		args.student_pointer != 0 &&
		args.teacher_pointer != 0 &&
		args.output_pointer != 0 &&
		args.batch > 0 &&
		args.vocab > 0 &&
		args.student_bytes == args.batch * args.vocab * sizeof(float) &&
		args.teacher_bytes == args.batch * args.vocab * sizeof(float) &&
		args.output_bytes == ROCM_DISTILLATION_KL_LOSS_OUTPUT_BYTES &&
		temperature > 0.0 &&
		isfinite(temperature);
}

__device__ bool rocm_valid_grpo_advantage_args(const rocm_grpo_advantage_launch_args &args)
{
	return args.version == ROCM_GRPO_ADVANTAGE_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_GRPO_ADVANTAGE_LAUNCH_ARGS_BYTES &&
		args.reward_pointer != 0 &&
		args.output_pointer != 0 &&
		args.count > 0 &&
		args.reward_bytes == args.count * sizeof(double) &&
		args.output_bytes == args.count * sizeof(double);
}

__device__ bool rocm_valid_autoround_format(uint32_t format_code, uint32_t bits)
{
	if (bits == 4) {
		return format_code == ROCM_AUTOROUND_FORMAT_MXFP4 || format_code == ROCM_AUTOROUND_FORMAT_NVFP4;
	}
	if (bits == 8) {
		return format_code == ROCM_AUTOROUND_FORMAT_FP8 || format_code == ROCM_AUTOROUND_FORMAT_MXFP8;
	}
	if (bits == 2) {
		return format_code == ROCM_AUTOROUND_FORMAT_INT2;
	}
	return false;
}

__device__ bool rocm_valid_autoround_quantize_args(const rocm_autoround_quantize_launch_args &args)
{
	const uint64_t value_count = static_cast<uint64_t>(args.rows) * static_cast<uint64_t>(args.cols);
	const uint64_t scale_count = static_cast<uint64_t>(args.rows) * static_cast<uint64_t>(args.groups_per_row);
	const uint64_t packed_bits = value_count * static_cast<uint64_t>(args.bits);
	return args.version == ROCM_AUTOROUND_QUANTIZE_LAUNCH_ARGS_VERSION &&
		args.total_bytes == ROCM_AUTOROUND_QUANTIZE_LAUNCH_ARGS_BYTES &&
		args.weight_pointer != 0 &&
		args.packed_pointer != 0 &&
		args.scale_pointer != 0 &&
		args.rows > 0 &&
		args.cols > 0 &&
		args.group_size > 0 &&
		args.cols % args.group_size == 0 &&
		args.groups_per_row == args.cols / args.group_size &&
		rocm_valid_autoround_format(args.format_code, args.bits) &&
		args.weight_bytes == value_count * sizeof(float) &&
		args.packed_bytes == (packed_bits + 7u) / 8u &&
		args.scale_bytes == scale_count * sizeof(float) &&
		args.nsamples > 0 &&
		args.seqlen > 0 &&
		args.iters > 0;
}

__device__ void rocm_autoround_pack_signed(uint8_t *packed, uint32_t bits, uint32_t index, int32_t value)
{
	const uint32_t mask = (1u << bits) - 1u;
	const uint32_t encoded = static_cast<uint32_t>(value) & mask;
	const uint32_t bit_offset = index * bits;
	const uint32_t byte_offset = bit_offset >> 3;
	const uint32_t shift = bit_offset & 7u;
	const uint32_t shifted = encoded << shift;
	packed[byte_offset] = static_cast<uint8_t>(packed[byte_offset] | (shifted & 0xffu));
	if (shift + bits > 8u) {
		packed[byte_offset + 1u] = static_cast<uint8_t>(packed[byte_offset + 1u] | ((shifted >> 8u) & 0xffu));
	}
}

__device__ float rocm_tiny_output_weight_value(uint64_t pointer, uint32_t encoding, uint32_t index, float q8_scale)
{
	if (encoding == ROCM_TINY_OUTPUT_WEIGHT_ENCODING_FP32) {
		const float *weights = reinterpret_cast<const float *>(static_cast<uintptr_t>(pointer));
		return weights[index];
	}
	if (encoding == ROCM_TINY_OUTPUT_WEIGHT_ENCODING_FP16) {
		const uint16_t *weights = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(pointer));
		return rocm_half_to_float(weights[index]);
	}
	const int8_t *weights = reinterpret_cast<const int8_t *>(static_cast<uintptr_t>(pointer));
	return static_cast<float>(weights[index]) * q8_scale;
}

}

extern "C" __global__ void rocm_prefill(const unsigned char *packet)
{
	if (blockIdx.x != 0 || threadIdx.x != 0 || packet == nullptr) {
		return;
	}
	const rocm_prefill_launch_args &args = *reinterpret_cast<const rocm_prefill_launch_args *>(packet);
	if (!rocm_valid_prefill_args(args)) {
		return;
	}
	const int32_t *tokens = reinterpret_cast<const int32_t *>(static_cast<uintptr_t>(args.token_pointer));
	volatile int32_t first_token = tokens[0];
	(void)first_token;
	if (args.status_pointer != 0) {
		uint32_t *status = reinterpret_cast<uint32_t *>(static_cast<uintptr_t>(args.status_pointer));
		*status = args.status_value == 0 ? ROCM_PREFILL_LAUNCH_STATUS_OK : args.status_value;
	}
}

extern "C" __global__ void rocm_decode(const unsigned char *packet)
{
	if (blockIdx.x != 0 || threadIdx.x != 0 || packet == nullptr) {
		return;
	}
	const rocm_decode_launch_args &args = *reinterpret_cast<const rocm_decode_launch_args *>(packet);
	if (!rocm_valid_decode_args(args)) {
		return;
	}
	const uint32_t *descriptor = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.kv.descriptor_pointer));
	volatile uint32_t descriptor_version = descriptor[0];
	volatile uint32_t descriptor_mode = descriptor[3];
	(void)descriptor_version;
	(void)descriptor_mode;
	if (args.kv.status_pointer != 0) {
		uint32_t *status = reinterpret_cast<uint32_t *>(static_cast<uintptr_t>(args.kv.status_pointer));
		*status = args.kv.status_value == 0 ? ROCM_DECODE_LAUNCH_STATUS_OK : args.kv.status_value;
	}
}

__device__ float rocm_block_reduce_sum(float value, float *scratch);

extern "C" __global__ void rocm_projection(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_projection_launch_args &args = *reinterpret_cast<const rocm_projection_launch_args *>(packet);
	if (!rocm_valid_projection_args(args)) {
		return;
	}

	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row = blockIdx.x;
	if (row >= args.rows) {
		return;
	}

	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float sum = 0.0f;
	if (args.weight_encoding == ROCM_PROJECTION_WEIGHT_ENCODING_FP16) {
		const uint16_t *weights = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.weight_pointer));
		for (uint32_t col = threadIdx.x; col < args.cols; col += blockDim.x) {
			sum += input[col] * rocm_half_to_float(weights[row * args.cols + col]);
		}
	} else if (args.weight_encoding == ROCM_PROJECTION_WEIGHT_ENCODING_BF16) {
		const uint16_t *weights = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.weight_pointer));
		for (uint32_t col = threadIdx.x; col < args.cols; col += blockDim.x) {
			sum += input[col] * rocm_bfloat16_to_float(weights[row * args.cols + col]);
		}
	} else if (args.weight_encoding == ROCM_PROJECTION_WEIGHT_ENCODING_Q8) {
		const int8_t *weights = reinterpret_cast<const int8_t *>(static_cast<uintptr_t>(args.weight_pointer));
		const float scale = rocm_float_from_bits(args.q8_scale_bits);
		for (uint32_t col = threadIdx.x; col < args.cols; col += blockDim.x) {
			sum += input[col] * static_cast<float>(weights[row * args.cols + col]) * scale;
		}
	} else {
		const float *weights = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.weight_pointer));
		for (uint32_t col = threadIdx.x; col < args.cols; col += blockDim.x) {
			sum += input[col] * weights[row * args.cols + col];
		}
	}
	__shared__ float scratch[ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE / 32u];
	sum = rocm_block_reduce_sum(sum, scratch);
	if (threadIdx.x == 0u && (args.flags & ROCM_PROJECTION_LAUNCH_FLAG_BIAS) != 0 && args.bias_pointer != 0) {
		const float *bias = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.bias_pointer));
		sum += bias[row];
	}
	if (threadIdx.x == 0u) {
		output[row] = sum;
	}
}

__device__ float rocm_mlx_q4_row_reduce(float value);

extern "C" __global__ void rocm_projection_batch(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_projection_batch_launch_args &args = *reinterpret_cast<const rocm_projection_batch_launch_args *>(packet);
	if (!rocm_valid_projection_batch_args(args) || blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}

	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_ROWS_PER_BLOCK + row_lane;
	const uint32_t batch_base = blockIdx.y * ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK;
	if (batch_base >= args.batch) {
		return;
	}

	const float *input_base = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	float *output_base = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float sums[ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK] = {};
	if (row < args.rows) {
		if (args.weight_encoding == ROCM_PROJECTION_WEIGHT_ENCODING_FP16) {
			const uint16_t *weights = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.weight_pointer));
			for (uint32_t col = col_lane; col < args.cols; col += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
				const float weight = rocm_half_to_float(weights[row * args.cols + col]);
#pragma unroll
				for (uint32_t token_lane = 0; token_lane < ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK; ++token_lane) {
					const uint32_t batch = batch_base + token_lane;
					if (batch < args.batch) {
						sums[token_lane] += input_base[static_cast<uint64_t>(batch) * args.cols + col] * weight;
					}
				}
			}
		} else if (args.weight_encoding == ROCM_PROJECTION_WEIGHT_ENCODING_BF16) {
			const uint16_t *weights = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.weight_pointer));
			for (uint32_t col = col_lane; col < args.cols; col += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
				const float weight = rocm_bfloat16_to_float(weights[row * args.cols + col]);
#pragma unroll
				for (uint32_t token_lane = 0; token_lane < ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK; ++token_lane) {
					const uint32_t batch = batch_base + token_lane;
					if (batch < args.batch) {
						sums[token_lane] += input_base[static_cast<uint64_t>(batch) * args.cols + col] * weight;
					}
				}
			}
		} else if (args.weight_encoding == ROCM_PROJECTION_WEIGHT_ENCODING_Q8) {
			const int8_t *weights = reinterpret_cast<const int8_t *>(static_cast<uintptr_t>(args.weight_pointer));
			const float scale = rocm_float_from_bits(args.q8_scale_bits);
			for (uint32_t col = col_lane; col < args.cols; col += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
				const float weight = static_cast<float>(weights[row * args.cols + col]) * scale;
#pragma unroll
				for (uint32_t token_lane = 0; token_lane < ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK; ++token_lane) {
					const uint32_t batch = batch_base + token_lane;
					if (batch < args.batch) {
						sums[token_lane] += input_base[static_cast<uint64_t>(batch) * args.cols + col] * weight;
					}
				}
			}
		} else {
			const float *weights = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.weight_pointer));
			for (uint32_t col = col_lane; col < args.cols; col += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
				const float weight = weights[row * args.cols + col];
#pragma unroll
				for (uint32_t token_lane = 0; token_lane < ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK; ++token_lane) {
					const uint32_t batch = batch_base + token_lane;
					if (batch < args.batch) {
						sums[token_lane] += input_base[static_cast<uint64_t>(batch) * args.cols + col] * weight;
					}
				}
			}
		}
	}
	const float *bias = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.bias_pointer));
#pragma unroll
	for (uint32_t token_lane = 0; token_lane < ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK; ++token_lane) {
		const uint32_t batch = batch_base + token_lane;
		float sum = rocm_mlx_q4_row_reduce(sums[token_lane]);
		if ((args.flags & ROCM_PROJECTION_LAUNCH_FLAG_BIAS) != 0 && args.bias_pointer != 0 && row < args.rows) {
			sum += bias[row];
		}
		if (col_lane == 0 && row < args.rows && batch < args.batch) {
			output_base[static_cast<uint64_t>(batch) * args.rows + row] = sum;
		}
	}
}

__device__ float rocm_gelu_tanh_value(float value)
{
	const float value2 = value * value;
	const float value3 = value2 * value;
	const float tanh_arg = 0.7978845608028654f * (value + 0.044715f * value3);
	return 0.5f * value * (1.0f + tanhf(tanh_arg));
}

__device__ float rocm_fast_expf(float value)
{
#if defined(__HIP_CPU_RT__)
	return expf(value);
#else
	return __expf(value);
#endif
}

__device__ float rocm_rsqrtf(float value)
{
#if defined(__HIP_CPU_RT__)
	return 1.0f / sqrtf(value);
#else
	return rsqrtf(value);
#endif
}

__device__ float rocm_shfl_down(float value, uint32_t delta, int width)
{
#if defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIP_PLATFORM_NVCC__) || defined(__CUDACC__)
	return __shfl_down_sync(0xffffffffu, value, delta, width);
#else
	return __shfl_down(value, delta, width);
#endif
}

__device__ uint32_t rocm_shfl_u32(uint32_t value, int source_lane, int width)
{
#if defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIP_PLATFORM_NVCC__) || defined(__CUDACC__)
	return __shfl_sync(0xffffffffu, value, source_lane, width);
#else
	return __shfl(value, source_lane, width);
#endif
}

__device__ float rocm_shfl_float(float value, int source_lane, int width)
{
#if defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIP_PLATFORM_NVCC__) || defined(__CUDACC__)
	return __shfl_sync(0xffffffffu, value, source_lane, width);
#else
	return __shfl(value, source_lane, width);
#endif
}

__device__ uint64_t rocm_shfl_u64(uint64_t value, int source_lane, int width)
{
	const uint32_t lo = rocm_shfl_u32(static_cast<uint32_t>(value), source_lane, width);
	const uint32_t hi = rocm_shfl_u32(static_cast<uint32_t>(value >> 32u), source_lane, width);
	return (static_cast<uint64_t>(hi) << 32u) | static_cast<uint64_t>(lo);
}

__device__ float rocm_mlx_q4_row_reduce(float value)
{
	for (uint32_t stride = ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW / 2u; stride > 0u; stride >>= 1u) {
		value += rocm_shfl_down(value, stride, ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW);
	}
	return value;
}

__device__ float rocm_mlx_q4_projection_row16_reduce(float value)
{
	for (uint32_t stride = ROCM_MLX_Q4_PROJECTION_ROW16_THREADS_PER_ROW / 2u; stride > 0u; stride >>= 1u) {
		value += rocm_shfl_down(value, stride, ROCM_MLX_Q4_PROJECTION_ROW16_THREADS_PER_ROW);
	}
	return value;
}

__device__ uint32_t rocm_mlx_affine_q4_quantized_value(const uint32_t *weights, uint64_t row_packed_base, uint32_t col)
{
	const uint32_t word = weights[row_packed_base + (col >> 3u)];
	return (word >> ((col & 7u) << 2u)) & 0x0fu;
}

__device__ uint32_t rocm_mlx_affine_q6_quantized_value(const uint32_t *weights, uint64_t row_packed_base, uint32_t col)
{
	const uint32_t block = col >> 4u;
	const uint32_t lane = col & 15u;
	const uint64_t packed = row_packed_base + static_cast<uint64_t>(block) * 3u;
	const uint32_t word0 = weights[packed];
	const uint32_t word1 = weights[packed + 1u];
	const uint32_t word2 = weights[packed + 2u];
	switch (lane) {
	case 0u:
		return word0 & 0x3fu;
	case 1u:
		return (word0 >> 6u) & 0x3fu;
	case 2u:
		return (word0 >> 12u) & 0x3fu;
	case 3u:
		return (word0 >> 18u) & 0x3fu;
	case 4u:
		return (word0 >> 24u) & 0x3fu;
	case 5u:
		return ((word0 >> 30u) | (word1 << 2u)) & 0x3fu;
	case 6u:
		return (word1 >> 4u) & 0x3fu;
	case 7u:
		return (word1 >> 10u) & 0x3fu;
	case 8u:
		return (word1 >> 16u) & 0x3fu;
	case 9u:
		return (word1 >> 22u) & 0x3fu;
	case 10u:
		return ((word1 >> 28u) | (word2 << 4u)) & 0x3fu;
	case 11u:
		return (word2 >> 2u) & 0x3fu;
	case 12u:
		return (word2 >> 8u) & 0x3fu;
	case 13u:
		return (word2 >> 14u) & 0x3fu;
	case 14u:
		return (word2 >> 20u) & 0x3fu;
	default:
		return (word2 >> 26u) & 0x3fu;
	}
}

__device__ uint32_t rocm_mlx_affine_q8_quantized_value(const uint32_t *weights, uint64_t row_packed_base, uint32_t col)
{
	const uint32_t word = weights[row_packed_base + (col >> 2u)];
	return (word >> ((col & 3u) << 3u)) & 0xffu;
}

__device__ void rocm_mlx_affine_q4_64_pair_dot(
	const float *input,
	const uint32_t *first_weights,
	const uint32_t *second_weights,
	uint64_t row_packed_base,
	uint32_t first_packed,
	uint32_t first_col,
	float *first_q_dot_sum,
	float *second_q_dot_sum,
	float *input_sum)
{
#pragma unroll
	for (uint32_t packed_lane = 0; packed_lane < 8u; ++packed_lane) {
		const uint32_t first_word = first_weights[row_packed_base + first_packed + packed_lane];
		const uint32_t second_word = second_weights[row_packed_base + first_packed + packed_lane];
		const uint32_t col = first_col + (packed_lane << 3u);
		const float in0 = input[col];
		const float in1 = input[col + 1u];
		const float in2 = input[col + 2u];
		const float in3 = input[col + 3u];
		const float in4 = input[col + 4u];
		const float in5 = input[col + 5u];
		const float in6 = input[col + 6u];
		const float in7 = input[col + 7u];
		*input_sum += in0 + in1 + in2 + in3 + in4 + in5 + in6 + in7;
		*first_q_dot_sum +=
			static_cast<float>(first_word & 0x0fu) * in0 +
			static_cast<float>((first_word >> 4u) & 0x0fu) * in1 +
			static_cast<float>((first_word >> 8u) & 0x0fu) * in2 +
			static_cast<float>((first_word >> 12u) & 0x0fu) * in3 +
			static_cast<float>((first_word >> 16u) & 0x0fu) * in4 +
			static_cast<float>((first_word >> 20u) & 0x0fu) * in5 +
			static_cast<float>((first_word >> 24u) & 0x0fu) * in6 +
			static_cast<float>((first_word >> 28u) & 0x0fu) * in7;
		*second_q_dot_sum +=
			static_cast<float>(second_word & 0x0fu) * in0 +
			static_cast<float>((second_word >> 4u) & 0x0fu) * in1 +
			static_cast<float>((second_word >> 8u) & 0x0fu) * in2 +
			static_cast<float>((second_word >> 12u) & 0x0fu) * in3 +
			static_cast<float>((second_word >> 16u) & 0x0fu) * in4 +
			static_cast<float>((second_word >> 20u) & 0x0fu) * in5 +
			static_cast<float>((second_word >> 24u) & 0x0fu) * in6 +
			static_cast<float>((second_word >> 28u) & 0x0fu) * in7;
	}
}

__device__ void rocm_mlx_affine_q8_32_dot(
	const float *input,
	const uint32_t *weights,
	uint64_t row_packed_base,
	uint32_t first_packed,
	uint32_t first_col,
	float *q_dot_sum,
	float *input_sum)
{
#pragma unroll
	for (uint32_t packed_lane = 0; packed_lane < 8u; ++packed_lane) {
		const uint32_t word = weights[row_packed_base + first_packed + packed_lane];
		const uint32_t col = first_col + (packed_lane << 2u);
		const float in0 = input[col];
		const float in1 = input[col + 1u];
		const float in2 = input[col + 2u];
		const float in3 = input[col + 3u];
		*input_sum += in0 + in1 + in2 + in3;
		*q_dot_sum +=
			static_cast<float>(word & 0xffu) * in0 +
			static_cast<float>((word >> 8u) & 0xffu) * in1 +
			static_cast<float>((word >> 16u) & 0xffu) * in2 +
			static_cast<float>(word >> 24u) * in3;
	}
}

__device__ void rocm_mlx_affine_q8_32_pair_dot(
	const float *input,
	const uint32_t *first_weights,
	const uint32_t *second_weights,
	uint64_t row_packed_base,
	uint32_t first_packed,
	uint32_t first_col,
	float *first_q_dot_sum,
	float *second_q_dot_sum,
	float *input_sum)
{
#pragma unroll
	for (uint32_t packed_lane = 0; packed_lane < 8u; ++packed_lane) {
		const uint32_t first_word = first_weights[row_packed_base + first_packed + packed_lane];
		const uint32_t second_word = second_weights[row_packed_base + first_packed + packed_lane];
		const uint32_t col = first_col + (packed_lane << 2u);
		const float in0 = input[col];
		const float in1 = input[col + 1u];
		const float in2 = input[col + 2u];
		const float in3 = input[col + 3u];
		*input_sum += in0 + in1 + in2 + in3;
		*first_q_dot_sum +=
			static_cast<float>(first_word & 0xffu) * in0 +
			static_cast<float>((first_word >> 8u) & 0xffu) * in1 +
			static_cast<float>((first_word >> 16u) & 0xffu) * in2 +
			static_cast<float>(first_word >> 24u) * in3;
		*second_q_dot_sum +=
			static_cast<float>(second_word & 0xffu) * in0 +
			static_cast<float>((second_word >> 8u) & 0xffu) * in1 +
			static_cast<float>((second_word >> 16u) & 0xffu) * in2 +
			static_cast<float>(second_word >> 24u) * in3;
	}
}

__device__ uint32_t rocm_mlx_affine_quantized_value(const uint32_t *weights, uint64_t row_packed_base, uint32_t col, uint32_t bits)
{
	if (bits == 4u) {
		return rocm_mlx_affine_q4_quantized_value(weights, row_packed_base, col);
	}
	if (bits == 6u) {
		return rocm_mlx_affine_q6_quantized_value(weights, row_packed_base, col);
	}
	if (bits == 8u) {
		return rocm_mlx_affine_q8_quantized_value(weights, row_packed_base, col);
	}
	const uint64_t bit_offset = static_cast<uint64_t>(col) * bits;
	const uint64_t word_index = bit_offset >> 5u;
	const uint32_t shift = static_cast<uint32_t>(bit_offset & 31u);
	uint32_t value = weights[row_packed_base + word_index] >> shift;
	if (shift + bits > 32u) {
		value |= weights[row_packed_base + word_index + 1u] << (32u - shift);
	}
	return value & ((1u << bits) - 1u);
}

__device__ void rocm_mlx_affine_q6_16_dot(
	const float *input,
	const uint32_t *weights,
	uint64_t row_packed_base,
	uint32_t packed,
	uint32_t col,
	float *q_dot_sum,
	float *input_sum)
{
	const uint32_t word0 = weights[row_packed_base + packed];
	const uint32_t word1 = weights[row_packed_base + packed + 1u];
	const uint32_t word2 = weights[row_packed_base + packed + 2u];
	const float in0 = input[col];
	const float in1 = input[col + 1u];
	const float in2 = input[col + 2u];
	const float in3 = input[col + 3u];
	const float in4 = input[col + 4u];
	const float in5 = input[col + 5u];
	const float in6 = input[col + 6u];
	const float in7 = input[col + 7u];
	const float in8 = input[col + 8u];
	const float in9 = input[col + 9u];
	const float in10 = input[col + 10u];
	const float in11 = input[col + 11u];
	const float in12 = input[col + 12u];
	const float in13 = input[col + 13u];
	const float in14 = input[col + 14u];
	const float in15 = input[col + 15u];
	*input_sum += in0 + in1 + in2 + in3 + in4 + in5 + in6 + in7 + in8 + in9 + in10 + in11 + in12 + in13 + in14 + in15;
	*q_dot_sum +=
		static_cast<float>(word0 & 0x3fu) * in0 +
		static_cast<float>((word0 >> 6) & 0x3fu) * in1 +
		static_cast<float>((word0 >> 12) & 0x3fu) * in2 +
		static_cast<float>((word0 >> 18) & 0x3fu) * in3 +
		static_cast<float>((word0 >> 24) & 0x3fu) * in4 +
		static_cast<float>(((word0 >> 30) | (word1 << 2)) & 0x3fu) * in5 +
		static_cast<float>((word1 >> 4) & 0x3fu) * in6 +
		static_cast<float>((word1 >> 10) & 0x3fu) * in7 +
		static_cast<float>((word1 >> 16) & 0x3fu) * in8 +
		static_cast<float>((word1 >> 22) & 0x3fu) * in9 +
		static_cast<float>(((word1 >> 28) | (word2 << 4)) & 0x3fu) * in10 +
		static_cast<float>((word2 >> 2) & 0x3fu) * in11 +
		static_cast<float>((word2 >> 8) & 0x3fu) * in12 +
		static_cast<float>((word2 >> 14) & 0x3fu) * in13 +
		static_cast<float>((word2 >> 20) & 0x3fu) * in14 +
		static_cast<float>((word2 >> 26) & 0x3fu) * in15;
}

__device__ void rocm_mlx_affine_q6_16_pair_dot(
	const float *input,
	const uint32_t *gate_weights,
	const uint32_t *up_weights,
	uint64_t row_packed_base,
	uint32_t packed,
	uint32_t col,
	float *gate_q_dot_sum,
	float *up_q_dot_sum,
	float *input_sum)
{
	const uint32_t gate0 = gate_weights[row_packed_base + packed];
	const uint32_t gate1 = gate_weights[row_packed_base + packed + 1u];
	const uint32_t gate2 = gate_weights[row_packed_base + packed + 2u];
	const uint32_t up0 = up_weights[row_packed_base + packed];
	const uint32_t up1 = up_weights[row_packed_base + packed + 1u];
	const uint32_t up2 = up_weights[row_packed_base + packed + 2u];
	const float in0 = input[col];
	const float in1 = input[col + 1u];
	const float in2 = input[col + 2u];
	const float in3 = input[col + 3u];
	const float in4 = input[col + 4u];
	const float in5 = input[col + 5u];
	const float in6 = input[col + 6u];
	const float in7 = input[col + 7u];
	const float in8 = input[col + 8u];
	const float in9 = input[col + 9u];
	const float in10 = input[col + 10u];
	const float in11 = input[col + 11u];
	const float in12 = input[col + 12u];
	const float in13 = input[col + 13u];
	const float in14 = input[col + 14u];
	const float in15 = input[col + 15u];
	*input_sum += in0 + in1 + in2 + in3 + in4 + in5 + in6 + in7 + in8 + in9 + in10 + in11 + in12 + in13 + in14 + in15;
	*gate_q_dot_sum +=
		static_cast<float>(gate0 & 0x3fu) * in0 +
		static_cast<float>((gate0 >> 6) & 0x3fu) * in1 +
		static_cast<float>((gate0 >> 12) & 0x3fu) * in2 +
		static_cast<float>((gate0 >> 18) & 0x3fu) * in3 +
		static_cast<float>((gate0 >> 24) & 0x3fu) * in4 +
		static_cast<float>(((gate0 >> 30) | (gate1 << 2)) & 0x3fu) * in5 +
		static_cast<float>((gate1 >> 4) & 0x3fu) * in6 +
		static_cast<float>((gate1 >> 10) & 0x3fu) * in7 +
		static_cast<float>((gate1 >> 16) & 0x3fu) * in8 +
		static_cast<float>((gate1 >> 22) & 0x3fu) * in9 +
		static_cast<float>(((gate1 >> 28) | (gate2 << 4)) & 0x3fu) * in10 +
		static_cast<float>((gate2 >> 2) & 0x3fu) * in11 +
		static_cast<float>((gate2 >> 8) & 0x3fu) * in12 +
		static_cast<float>((gate2 >> 14) & 0x3fu) * in13 +
		static_cast<float>((gate2 >> 20) & 0x3fu) * in14 +
		static_cast<float>((gate2 >> 26) & 0x3fu) * in15;
	*up_q_dot_sum +=
		static_cast<float>(up0 & 0x3fu) * in0 +
		static_cast<float>((up0 >> 6) & 0x3fu) * in1 +
		static_cast<float>((up0 >> 12) & 0x3fu) * in2 +
		static_cast<float>((up0 >> 18) & 0x3fu) * in3 +
		static_cast<float>((up0 >> 24) & 0x3fu) * in4 +
		static_cast<float>(((up0 >> 30) | (up1 << 2)) & 0x3fu) * in5 +
		static_cast<float>((up1 >> 4) & 0x3fu) * in6 +
		static_cast<float>((up1 >> 10) & 0x3fu) * in7 +
		static_cast<float>((up1 >> 16) & 0x3fu) * in8 +
		static_cast<float>((up1 >> 22) & 0x3fu) * in9 +
		static_cast<float>(((up1 >> 28) | (up2 << 4)) & 0x3fu) * in10 +
		static_cast<float>((up2 >> 2) & 0x3fu) * in11 +
		static_cast<float>((up2 >> 8) & 0x3fu) * in12 +
		static_cast<float>((up2 >> 14) & 0x3fu) * in13 +
		static_cast<float>((up2 >> 20) & 0x3fu) * in14 +
		static_cast<float>((up2 >> 26) & 0x3fu) * in15;
}

__device__ void rocm_mlx_affine_q6_16_batch_dot(
	const float *input_base,
	const uint32_t *weights,
	uint64_t row_packed_base,
	uint32_t packed,
	uint32_t col,
	uint32_t cols,
	uint32_t batch_base,
	uint32_t batch_count,
	float *q_dot_sums,
	float *input_sums)
{
	const uint32_t word0 = weights[row_packed_base + packed];
	const uint32_t word1 = weights[row_packed_base + packed + 1u];
	const uint32_t word2 = weights[row_packed_base + packed + 2u];
	const float q0 = static_cast<float>(word0 & 0x3fu);
	const float q1 = static_cast<float>((word0 >> 6) & 0x3fu);
	const float q2 = static_cast<float>((word0 >> 12) & 0x3fu);
	const float q3 = static_cast<float>((word0 >> 18) & 0x3fu);
	const float q4 = static_cast<float>((word0 >> 24) & 0x3fu);
	const float q5 = static_cast<float>(((word0 >> 30) | (word1 << 2)) & 0x3fu);
	const float q6 = static_cast<float>((word1 >> 4) & 0x3fu);
	const float q7 = static_cast<float>((word1 >> 10) & 0x3fu);
	const float q8 = static_cast<float>((word1 >> 16) & 0x3fu);
	const float q9 = static_cast<float>((word1 >> 22) & 0x3fu);
	const float q10 = static_cast<float>(((word1 >> 28) | (word2 << 4)) & 0x3fu);
	const float q11 = static_cast<float>((word2 >> 2) & 0x3fu);
	const float q12 = static_cast<float>((word2 >> 8) & 0x3fu);
	const float q13 = static_cast<float>((word2 >> 14) & 0x3fu);
	const float q14 = static_cast<float>((word2 >> 20) & 0x3fu);
	const float q15 = static_cast<float>((word2 >> 26) & 0x3fu);
#pragma unroll
	for (uint32_t token_lane = 0; token_lane < ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK; ++token_lane) {
		const uint32_t batch = batch_base + token_lane;
		if (batch >= batch_count) {
			continue;
		}
		const float *input = input_base + batch * cols;
		const float in0 = input[col];
		const float in1 = input[col + 1u];
		const float in2 = input[col + 2u];
		const float in3 = input[col + 3u];
		const float in4 = input[col + 4u];
		const float in5 = input[col + 5u];
		const float in6 = input[col + 6u];
		const float in7 = input[col + 7u];
		const float in8 = input[col + 8u];
		const float in9 = input[col + 9u];
		const float in10 = input[col + 10u];
		const float in11 = input[col + 11u];
		const float in12 = input[col + 12u];
		const float in13 = input[col + 13u];
		const float in14 = input[col + 14u];
		const float in15 = input[col + 15u];
		input_sums[token_lane] += in0 + in1 + in2 + in3 + in4 + in5 + in6 + in7 + in8 + in9 + in10 + in11 + in12 + in13 + in14 + in15;
		q_dot_sums[token_lane] +=
			q0 * in0 +
			q1 * in1 +
			q2 * in2 +
			q3 * in3 +
			q4 * in4 +
			q5 * in5 +
			q6 * in6 +
			q7 * in7 +
			q8 * in8 +
			q9 * in9 +
			q10 * in10 +
			q11 * in11 +
			q12 * in12 +
			q13 * in13 +
			q14 * in14 +
			q15 * in15;
	}
}

__device__ void rocm_mlx_affine_q6_16_pair_batch_dot(
	const float *input_base,
	const uint32_t *gate_weights,
	const uint32_t *up_weights,
	uint64_t row_packed_base,
	uint32_t packed,
	uint32_t col,
	uint32_t cols,
	uint32_t batch_base,
	uint32_t batch_count,
	float *gate_q_dot_sums,
	float *up_q_dot_sums,
	float *input_sums)
{
	const uint32_t gate0 = gate_weights[row_packed_base + packed];
	const uint32_t gate1 = gate_weights[row_packed_base + packed + 1u];
	const uint32_t gate2 = gate_weights[row_packed_base + packed + 2u];
	const uint32_t up0 = up_weights[row_packed_base + packed];
	const uint32_t up1 = up_weights[row_packed_base + packed + 1u];
	const uint32_t up2 = up_weights[row_packed_base + packed + 2u];
	const float gate_q0 = static_cast<float>(gate0 & 0x3fu);
	const float gate_q1 = static_cast<float>((gate0 >> 6) & 0x3fu);
	const float gate_q2 = static_cast<float>((gate0 >> 12) & 0x3fu);
	const float gate_q3 = static_cast<float>((gate0 >> 18) & 0x3fu);
	const float gate_q4 = static_cast<float>((gate0 >> 24) & 0x3fu);
	const float gate_q5 = static_cast<float>(((gate0 >> 30) | (gate1 << 2)) & 0x3fu);
	const float gate_q6 = static_cast<float>((gate1 >> 4) & 0x3fu);
	const float gate_q7 = static_cast<float>((gate1 >> 10) & 0x3fu);
	const float gate_q8 = static_cast<float>((gate1 >> 16) & 0x3fu);
	const float gate_q9 = static_cast<float>((gate1 >> 22) & 0x3fu);
	const float gate_q10 = static_cast<float>(((gate1 >> 28) | (gate2 << 4)) & 0x3fu);
	const float gate_q11 = static_cast<float>((gate2 >> 2) & 0x3fu);
	const float gate_q12 = static_cast<float>((gate2 >> 8) & 0x3fu);
	const float gate_q13 = static_cast<float>((gate2 >> 14) & 0x3fu);
	const float gate_q14 = static_cast<float>((gate2 >> 20) & 0x3fu);
	const float gate_q15 = static_cast<float>((gate2 >> 26) & 0x3fu);
	const float up_q0 = static_cast<float>(up0 & 0x3fu);
	const float up_q1 = static_cast<float>((up0 >> 6) & 0x3fu);
	const float up_q2 = static_cast<float>((up0 >> 12) & 0x3fu);
	const float up_q3 = static_cast<float>((up0 >> 18) & 0x3fu);
	const float up_q4 = static_cast<float>((up0 >> 24) & 0x3fu);
	const float up_q5 = static_cast<float>(((up0 >> 30) | (up1 << 2)) & 0x3fu);
	const float up_q6 = static_cast<float>((up1 >> 4) & 0x3fu);
	const float up_q7 = static_cast<float>((up1 >> 10) & 0x3fu);
	const float up_q8 = static_cast<float>((up1 >> 16) & 0x3fu);
	const float up_q9 = static_cast<float>((up1 >> 22) & 0x3fu);
	const float up_q10 = static_cast<float>(((up1 >> 28) | (up2 << 4)) & 0x3fu);
	const float up_q11 = static_cast<float>((up2 >> 2) & 0x3fu);
	const float up_q12 = static_cast<float>((up2 >> 8) & 0x3fu);
	const float up_q13 = static_cast<float>((up2 >> 14) & 0x3fu);
	const float up_q14 = static_cast<float>((up2 >> 20) & 0x3fu);
	const float up_q15 = static_cast<float>((up2 >> 26) & 0x3fu);
#pragma unroll
	for (uint32_t token_lane = 0; token_lane < ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK; ++token_lane) {
		const uint32_t batch = batch_base + token_lane;
		if (batch >= batch_count) {
			continue;
		}
		const float *input = input_base + batch * cols;
		const float in0 = input[col];
		const float in1 = input[col + 1u];
		const float in2 = input[col + 2u];
		const float in3 = input[col + 3u];
		const float in4 = input[col + 4u];
		const float in5 = input[col + 5u];
		const float in6 = input[col + 6u];
		const float in7 = input[col + 7u];
		const float in8 = input[col + 8u];
		const float in9 = input[col + 9u];
		const float in10 = input[col + 10u];
		const float in11 = input[col + 11u];
		const float in12 = input[col + 12u];
		const float in13 = input[col + 13u];
		const float in14 = input[col + 14u];
		const float in15 = input[col + 15u];
		input_sums[token_lane] += in0 + in1 + in2 + in3 + in4 + in5 + in6 + in7 + in8 + in9 + in10 + in11 + in12 + in13 + in14 + in15;
		gate_q_dot_sums[token_lane] +=
			gate_q0 * in0 +
			gate_q1 * in1 +
			gate_q2 * in2 +
			gate_q3 * in3 +
			gate_q4 * in4 +
			gate_q5 * in5 +
			gate_q6 * in6 +
			gate_q7 * in7 +
			gate_q8 * in8 +
			gate_q9 * in9 +
			gate_q10 * in10 +
			gate_q11 * in11 +
			gate_q12 * in12 +
			gate_q13 * in13 +
			gate_q14 * in14 +
			gate_q15 * in15;
		up_q_dot_sums[token_lane] +=
			up_q0 * in0 +
			up_q1 * in1 +
			up_q2 * in2 +
			up_q3 * in3 +
			up_q4 * in4 +
			up_q5 * in5 +
			up_q6 * in6 +
			up_q7 * in7 +
			up_q8 * in8 +
			up_q9 * in9 +
			up_q10 * in10 +
			up_q11 * in11 +
			up_q12 * in12 +
			up_q13 * in13 +
			up_q14 * in14 +
			up_q15 * in15;
	}
}

__device__ float rocm_mlx_q4_projection_row_sum(
	const float *input,
	const uint32_t *weights,
	const uint16_t *scales,
	const uint16_t *biases,
	uint32_t row,
	uint32_t cols,
	uint32_t group_size,
	uint32_t bits,
	uint32_t col_lane,
	uint32_t threads_per_row)
{
	const uint64_t packed_per_row = rocm_mlx_affine_packed_per_row(cols, bits);
	const uint64_t row_packed_base = static_cast<uint64_t>(row) * packed_per_row;
	float sum = 0.0f;
	if (bits == 4u && group_size == 64u) {
		const uint32_t groups_per_row = cols >> 6u;
		const uint32_t row_group_base = row * groups_per_row;
		for (uint32_t group_col = col_lane; group_col < groups_per_row; group_col += threads_per_row) {
			const uint32_t group = row_group_base + group_col;
			const float scale = rocm_bfloat16_to_float(scales[group]);
			const float bias = rocm_bfloat16_to_float(biases[group]);
			const uint32_t first_packed = group_col << 3u;
			float q_dot_sum = 0.0f;
			float input_sum = 0.0f;
			for (uint32_t group_packed = 0; group_packed < 8u; ++group_packed) {
				const uint32_t packed = first_packed + group_packed;
				const uint32_t word = weights[row_packed_base + packed];
				const uint32_t col = packed << 3;
				const float in0 = input[col];
				const float in1 = input[col + 1u];
				const float in2 = input[col + 2u];
				const float in3 = input[col + 3u];
				const float in4 = input[col + 4u];
				const float in5 = input[col + 5u];
				const float in6 = input[col + 6u];
				const float in7 = input[col + 7u];
				input_sum += in0 + in1 + in2 + in3 + in4 + in5 + in6 + in7;
				q_dot_sum +=
					static_cast<float>(word & 0x0fu) * in0 +
					static_cast<float>((word >> 4) & 0x0fu) * in1 +
					static_cast<float>((word >> 8) & 0x0fu) * in2 +
					static_cast<float>((word >> 12) & 0x0fu) * in3 +
					static_cast<float>((word >> 16) & 0x0fu) * in4 +
					static_cast<float>((word >> 20) & 0x0fu) * in5 +
					static_cast<float>((word >> 24) & 0x0fu) * in6 +
					static_cast<float>((word >> 28) & 0x0fu) * in7;
			}
			sum += scale * q_dot_sum + bias * input_sum;
		}
		return sum;
	}
	if (bits == 6u && group_size == 64u) {
		const uint32_t groups_per_row = cols >> 6u;
		const uint32_t row_group_base = row * groups_per_row;
		for (uint32_t group_col = col_lane; group_col < groups_per_row; group_col += threads_per_row) {
			const uint32_t group = row_group_base + group_col;
			const float scale = rocm_bfloat16_to_float(scales[group]);
			const float bias = rocm_bfloat16_to_float(biases[group]);
			const uint32_t first_packed = group_col * 12u;
			const uint32_t first_col = group_col << 6u;
			float q_dot_sum = 0.0f;
			float input_sum = 0.0f;
			for (uint32_t block = 0; block < 4u; ++block) {
				rocm_mlx_affine_q6_16_dot(input, weights, row_packed_base, first_packed + block * 3u, first_col + block * 16u, &q_dot_sum, &input_sum);
			}
			sum += scale * q_dot_sum + bias * input_sum;
		}
		return sum;
	}
	const uint32_t groups_per_row = cols / group_size;
	const uint32_t row_group_base = row * groups_per_row;
	if (bits == 8u && group_size == 32u) {
		for (uint32_t group_col = col_lane; group_col < groups_per_row; group_col += threads_per_row) {
			const uint32_t group = row_group_base + group_col;
			const float scale = rocm_bfloat16_to_float(scales[group]);
			const float bias = rocm_bfloat16_to_float(biases[group]);
			float q_dot_sum = 0.0f;
			float input_sum = 0.0f;
			rocm_mlx_affine_q8_32_dot(input, weights, row_packed_base, group_col << 3u, group_col << 5u, &q_dot_sum, &input_sum);
			sum += scale * q_dot_sum + bias * input_sum;
		}
		return sum;
	}
	if (bits == 8u && group_size == 64u) {
		for (uint32_t group_col = col_lane; group_col < groups_per_row; group_col += threads_per_row) {
			const uint32_t group = row_group_base + group_col;
			const float scale = rocm_bfloat16_to_float(scales[group]);
			const float bias = rocm_bfloat16_to_float(biases[group]);
			const uint32_t first_packed = group_col << 4u;
			const uint32_t first_col = group_col << 6u;
			float q_dot_sum = 0.0f;
			float input_sum = 0.0f;
			rocm_mlx_affine_q8_32_dot(input, weights, row_packed_base, first_packed, first_col, &q_dot_sum, &input_sum);
			rocm_mlx_affine_q8_32_dot(input, weights, row_packed_base, first_packed + 8u, first_col + 32u, &q_dot_sum, &input_sum);
			sum += scale * q_dot_sum + bias * input_sum;
		}
		return sum;
	}
	if (bits == 6u && group_size == 16u) {
		for (uint32_t group_col = col_lane; group_col < groups_per_row; group_col += threads_per_row) {
			const uint32_t group = row_group_base + group_col;
			const float scale = rocm_bfloat16_to_float(scales[group]);
			const float bias = rocm_bfloat16_to_float(biases[group]);
			float q_dot_sum = 0.0f;
			float input_sum = 0.0f;
			rocm_mlx_affine_q6_16_dot(input, weights, row_packed_base, group_col * 3u, group_col << 4u, &q_dot_sum, &input_sum);
			sum += scale * q_dot_sum + bias * input_sum;
		}
		return sum;
	}
	if (bits == 4u && (group_size & 7u) == 0u) {
		const uint32_t packed_per_group = group_size >> 3u;
		for (uint32_t group_col = col_lane; group_col < groups_per_row; group_col += threads_per_row) {
			const uint32_t group = row_group_base + group_col;
			const float scale = rocm_bfloat16_to_float(scales[group]);
			const float bias = rocm_bfloat16_to_float(biases[group]);
			const uint32_t first_packed = group_col * packed_per_group;
			float q_dot_sum = 0.0f;
			float input_sum = 0.0f;
			for (uint32_t group_packed = 0; group_packed < packed_per_group; ++group_packed) {
				const uint32_t packed = first_packed + group_packed;
				const uint32_t word = weights[row_packed_base + packed];
				const uint32_t col = packed << 3;
				const float in0 = input[col];
				const float in1 = input[col + 1u];
				const float in2 = input[col + 2u];
				const float in3 = input[col + 3u];
				const float in4 = input[col + 4u];
				const float in5 = input[col + 5u];
				const float in6 = input[col + 6u];
				const float in7 = input[col + 7u];
				input_sum += in0 + in1 + in2 + in3 + in4 + in5 + in6 + in7;
				q_dot_sum +=
					static_cast<float>(word & 0x0fu) * in0 +
					static_cast<float>((word >> 4) & 0x0fu) * in1 +
					static_cast<float>((word >> 8) & 0x0fu) * in2 +
					static_cast<float>((word >> 12) & 0x0fu) * in3 +
					static_cast<float>((word >> 16) & 0x0fu) * in4 +
					static_cast<float>((word >> 20) & 0x0fu) * in5 +
					static_cast<float>((word >> 24) & 0x0fu) * in6 +
					static_cast<float>((word >> 28) & 0x0fu) * in7;
			}
			sum += scale * q_dot_sum + bias * input_sum;
		}
		return sum;
	}
	for (uint32_t col = col_lane; col < cols; col += threads_per_row) {
		const uint32_t q = rocm_mlx_affine_quantized_value(weights, row_packed_base, col, bits);
		const uint32_t group = row_group_base + col / group_size;
		const float scale = rocm_bfloat16_to_float(scales[group]);
		const float bias = rocm_bfloat16_to_float(biases[group]);
		sum += input[col] * (static_cast<float>(q) * scale + bias);
	}
	return sum;
}

__device__ float rocm_mlx_q4_projection_row_sum(
	const float *input,
	const uint32_t *weights,
	const uint16_t *scales,
	const uint16_t *biases,
	uint32_t row,
	uint32_t cols,
	uint32_t group_size,
	uint32_t bits,
	uint32_t col_lane)
{
	return rocm_mlx_q4_projection_row_sum(input, weights, scales, biases, row, cols, group_size, bits, col_lane, ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW);
}

__device__ float rocm_mlx_q4_cols256_row_reduce(float value)
{
	for (uint32_t stride = ROCM_MLX_Q4_PROJECTION_COLS256_THREADS_PER_ROW / 2u; stride > 0u; stride >>= 1u) {
		value += rocm_shfl_down(value, stride, ROCM_MLX_Q4_PROJECTION_COLS256_THREADS_PER_ROW);
	}
	return value;
}

__device__ float rocm_mlx_q4_projection_q6_row16_reduce(float value)
{
	for (uint32_t stride = ROCM_MLX_Q4_PROJECTION_Q6_ROW16_THREADS_PER_ROW / 2u; stride > 0u; stride >>= 1u) {
		value += rocm_shfl_down(value, stride, ROCM_MLX_Q4_PROJECTION_Q6_ROW16_THREADS_PER_ROW);
	}
	return value;
}

__device__ float rocm_mlx_q4_projection_q6_row32_reduce(float value)
{
	for (uint32_t stride = ROCM_MLX_Q4_PROJECTION_Q6_ROW32_THREADS_PER_ROW / 2u; stride > 0u; stride >>= 1u) {
		value += rocm_shfl_down(value, stride, ROCM_MLX_Q4_PROJECTION_Q6_ROW32_THREADS_PER_ROW);
	}
	return value;
}

__device__ float rocm_mlx_q4_projection_q6_row64_reduce(float value)
{
	for (uint32_t stride = ROCM_MLX_Q4_PROJECTION_Q6_ROW64_THREADS_PER_ROW / 2u; stride > 0u; stride >>= 1u) {
		value += rocm_shfl_down(value, stride, ROCM_MLX_Q4_PROJECTION_Q6_ROW64_THREADS_PER_ROW);
	}
	return value;
}

__device__ float rocm_mlx_q4_gelu_tanh_q6_cols1536_row_reduce(float value)
{
	for (uint32_t stride = ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_THREADS_PER_ROW / 2u; stride > 0u; stride >>= 1u) {
		value += rocm_shfl_down(value, stride, ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_THREADS_PER_ROW);
	}
	return value;
}

__device__ float rocm_mlx_q4_gelu_tanh_q4_g32_cols1536_row16_reduce(float value)
{
	for (uint32_t stride = ROCM_MLX_Q4_GELU_TANH_Q4_G32_COLS1536_ROW16_THREADS_PER_ROW / 2u; stride > 0u; stride >>= 1u) {
		value += rocm_shfl_down(value, stride, ROCM_MLX_Q4_GELU_TANH_Q4_G32_COLS1536_ROW16_THREADS_PER_ROW);
	}
	return value;
}

__device__ float rocm_mlx_q4_gelu_tanh_q6_cols1536_row32_reduce(float value)
{
	for (uint32_t stride = ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_ROW32_THREADS_PER_ROW / 2u; stride > 0u; stride >>= 1u) {
		value += rocm_shfl_down(value, stride, ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_ROW32_THREADS_PER_ROW);
	}
	return value;
}

__device__ float rocm_mlx_q4_gelu_tanh_q6_cols1536_row64_reduce(float value)
{
	for (uint32_t stride = ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_ROW64_THREADS_PER_ROW / 2u; stride > 0u; stride >>= 1u) {
		value += rocm_shfl_down(value, stride, ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_ROW64_THREADS_PER_ROW);
	}
	return value;
}

__device__ float rocm_mlx_q4_greedy_row_reduce(float value)
{
	for (uint32_t stride = ROCM_MLX_Q4_PROJECTION_GREEDY_THREADS_PER_ROW / 2u; stride > 0u; stride >>= 1u) {
		value += rocm_shfl_down(value, stride, ROCM_MLX_Q4_PROJECTION_GREEDY_THREADS_PER_ROW);
	}
	return value;
}

__device__ float rocm_mlx_q4_greedy_q6_row_reduce(float value)
{
	for (uint32_t stride = ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_THREADS_PER_ROW / 2u; stride > 0u; stride >>= 1u) {
		value += rocm_shfl_down(value, stride, ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_THREADS_PER_ROW);
	}
	return value;
}

extern "C" __global__ void rocm_mlx_q4_projection(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_projection_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_projection_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_projection_args(args)) {
		return;
	}

	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_ROWS_PER_BLOCK + row_lane;

	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.weight_pointer));
	const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.scale_pointer));
	const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.bias_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float sum = 0.0f;
	if (row < args.rows) {
		sum = rocm_mlx_q4_projection_row_sum(input, weights, scales, biases, row, args.cols, args.group_size, args.bits, col_lane);
	}
	sum = rocm_mlx_q4_row_reduce(sum);
	if (col_lane == 0 && row < args.rows) {
		output[row] = sum;
	}
}

extern "C" __global__ void rocm_mlx_q4_projection_q4_g64_e4b_row8(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_projection_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_projection_launch_args *>(packet);
	const bool attention_q = args.rows == 2048u && args.cols == 2560u;
	const bool attention_out = args.rows == 2560u && args.cols == 2048u;
	if (!rocm_valid_mlx_q4_projection_args(args) ||
		args.bits != 4u ||
		args.group_size != 64u ||
		(!attention_q && !attention_out) ||
		blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}

	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_ROWS_PER_BLOCK + row_lane;
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.weight_pointer));
	const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.scale_pointer));
	const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.bias_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float sum = 0.0f;
	if (row < args.rows) {
		const uint64_t packed_per_row = args.cols >> 3u;
		const uint32_t groups_per_row = args.cols >> 6u;
		const uint64_t row_packed_base = static_cast<uint64_t>(row) * packed_per_row;
		const uint32_t row_group_base = row * groups_per_row;
		for (uint32_t group_col = col_lane; group_col < groups_per_row; group_col += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
			const uint32_t group = row_group_base + group_col;
			const float scale = rocm_bfloat16_to_float(scales[group]);
			const float bias = rocm_bfloat16_to_float(biases[group]);
			const uint32_t first_packed = group_col << 3u;
			float q_dot_sum = 0.0f;
			float input_sum = 0.0f;
#pragma unroll
			for (uint32_t group_packed = 0; group_packed < 8u; ++group_packed) {
				const uint32_t packed = first_packed + group_packed;
				const uint32_t word = weights[row_packed_base + packed];
				const uint32_t col = packed << 3u;
				const float in0 = input[col];
				const float in1 = input[col + 1u];
				const float in2 = input[col + 2u];
				const float in3 = input[col + 3u];
				const float in4 = input[col + 4u];
				const float in5 = input[col + 5u];
				const float in6 = input[col + 6u];
				const float in7 = input[col + 7u];
				input_sum += in0 + in1 + in2 + in3 + in4 + in5 + in6 + in7;
				q_dot_sum +=
					static_cast<float>(word & 0x0fu) * in0 +
					static_cast<float>((word >> 4u) & 0x0fu) * in1 +
					static_cast<float>((word >> 8u) & 0x0fu) * in2 +
					static_cast<float>((word >> 12u) & 0x0fu) * in3 +
					static_cast<float>((word >> 16u) & 0x0fu) * in4 +
					static_cast<float>((word >> 20u) & 0x0fu) * in5 +
					static_cast<float>((word >> 24u) & 0x0fu) * in6 +
					static_cast<float>((word >> 28u) & 0x0fu) * in7;
			}
			sum += scale * q_dot_sum + bias * input_sum;
		}
	}
	sum = rocm_mlx_q4_row_reduce(sum);
	if (col_lane == 0u && row < args.rows) {
		output[row] = sum;
	}
}

extern "C" __global__ void rocm_mlx_q4_projection_q8_g64_row8(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_projection_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_projection_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_projection_args(args) || args.bits != 8u || args.group_size != 64u || args.cols < 2560u) {
		return;
	}
	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}

	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_ROWS_PER_BLOCK + row_lane;
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.weight_pointer));
	const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.scale_pointer));
	const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.bias_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float sum = 0.0f;
	if (row < args.rows) {
		const uint64_t packed_per_row = rocm_mlx_affine_packed_per_row(args.cols, args.bits);
		const uint32_t groups_per_row = args.cols >> 6u;
		const uint64_t row_packed_base = static_cast<uint64_t>(row) * packed_per_row;
		const uint32_t row_group_base = row * groups_per_row;
		for (uint32_t group_col = col_lane; group_col < groups_per_row; group_col += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
			const uint32_t group = row_group_base + group_col;
			const float scale = rocm_bfloat16_to_float(scales[group]);
			const float bias = rocm_bfloat16_to_float(biases[group]);
			const uint32_t first_packed = group_col << 4u;
			const uint32_t first_col = group_col << 6u;
			float q_dot_sum = 0.0f;
			float input_sum = 0.0f;
			rocm_mlx_affine_q8_32_dot(input, weights, row_packed_base, first_packed, first_col, &q_dot_sum, &input_sum);
			rocm_mlx_affine_q8_32_dot(input, weights, row_packed_base, first_packed + 8u, first_col + 32u, &q_dot_sum, &input_sum);
			sum += scale * q_dot_sum + bias * input_sum;
		}
	}
	sum = rocm_mlx_q4_row_reduce(sum);
	if (col_lane == 0u && row < args.rows) {
		output[row] = sum;
	}
}

extern "C" __global__ void rocm_mlx_q4_projection_q4_g64_rows3840_cols15360_row16(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_projection_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_projection_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_projection_args(args) ||
		args.rows != 3840u ||
		args.cols != 15360u ||
		args.group_size != 64u ||
		args.bits != 4u ||
		blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}

	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_ROW16_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_ROW16_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_ROW16_ROWS_PER_BLOCK + row_lane;
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.weight_pointer));
	const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.scale_pointer));
	const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.bias_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float sum = rocm_mlx_q4_projection_row_sum(
		input,
		weights,
		scales,
		biases,
		row,
		args.cols,
		args.group_size,
		args.bits,
		col_lane,
		ROCM_MLX_Q4_PROJECTION_ROW16_THREADS_PER_ROW);
	sum = rocm_mlx_q4_projection_row16_reduce(sum);
	if (col_lane == 0u) {
		output[row] = sum;
	}
}

__device__ float rocm_mlx_q4_projection_q4_g32_rows3840_cols15360_row_sum(
	const float *input,
	const uint32_t *weights,
	const uint16_t *scales,
	const uint16_t *biases,
	uint32_t row,
	uint32_t col_lane)
{
	const uint64_t row_packed_base = static_cast<uint64_t>(row) * ROCM_MLX_Q4_PROJECTION_12B_DOWN_PACKED_PER_ROW;
	const uint32_t row_group_base = row * ROCM_MLX_Q4_PROJECTION_12B_DOWN_GROUPS_PER_ROW;
	float sum = 0.0f;
	for (uint32_t group_col = col_lane; group_col < ROCM_MLX_Q4_PROJECTION_12B_DOWN_GROUPS_PER_ROW; group_col += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
		const uint32_t group = row_group_base + group_col;
		const float scale = rocm_bfloat16_to_float(scales[group]);
		const float bias = rocm_bfloat16_to_float(biases[group]);
		const uint32_t first_packed = group_col * 4u;
		float q_dot_sum = 0.0f;
		float input_sum = 0.0f;
#pragma unroll
		for (uint32_t group_packed = 0; group_packed < 4u; ++group_packed) {
			const uint32_t packed = first_packed + group_packed;
			const uint32_t word = weights[row_packed_base + packed];
			const uint32_t col = packed << 3u;
			const float in0 = input[col];
			const float in1 = input[col + 1u];
			const float in2 = input[col + 2u];
			const float in3 = input[col + 3u];
			const float in4 = input[col + 4u];
			const float in5 = input[col + 5u];
			const float in6 = input[col + 6u];
			const float in7 = input[col + 7u];
			input_sum += in0 + in1 + in2 + in3 + in4 + in5 + in6 + in7;
			q_dot_sum +=
				static_cast<float>(word & 0x0fu) * in0 +
				static_cast<float>((word >> 4) & 0x0fu) * in1 +
				static_cast<float>((word >> 8) & 0x0fu) * in2 +
				static_cast<float>((word >> 12) & 0x0fu) * in3 +
				static_cast<float>((word >> 16) & 0x0fu) * in4 +
				static_cast<float>((word >> 20) & 0x0fu) * in5 +
				static_cast<float>((word >> 24) & 0x0fu) * in6 +
				static_cast<float>((word >> 28) & 0x0fu) * in7;
		}
		sum += scale * q_dot_sum + bias * input_sum;
	}
	return sum;
}

extern "C" __global__ void rocm_mlx_q4_projection_q4_g32_rows3840_cols15360(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_projection_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_projection_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_projection_args(args) ||
		args.rows != ROCM_MLX_Q4_PROJECTION_12B_DOWN_ROWS ||
		args.cols != ROCM_MLX_Q4_PROJECTION_12B_DOWN_COLS ||
		args.group_size != ROCM_MLX_Q4_PROJECTION_12B_DOWN_GROUP_SIZE ||
		args.bits != 4u ||
		blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}

	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_ROWS_PER_BLOCK + row_lane;
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.weight_pointer));
	const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.scale_pointer));
	const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.bias_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float sum = rocm_mlx_q4_projection_q4_g32_rows3840_cols15360_row_sum(input, weights, scales, biases, row, col_lane);
	sum = rocm_mlx_q4_row_reduce(sum);
	if (col_lane == 0u) {
		output[row] = sum;
	}
}

__device__ float rocm_gguf_q4_0_projection_row_sum(
	const float *input,
	const unsigned char *weights,
	uint32_t row,
	uint32_t cols,
	uint32_t col_lane)
{
	const uint32_t blocks_per_row = cols >> 5u;
	const uint64_t row_base = static_cast<uint64_t>(row) * blocks_per_row * 18u;
	float sum = 0.0f;
	for (uint32_t col = col_lane; col < cols; col += ROCM_GGUF_Q4_0_PROJECTION_THREADS_PER_ROW) {
		const uint32_t block_index = col >> 5u;
		const uint32_t quant_lane = col & 31u;
		const unsigned char *block = weights + row_base + static_cast<uint64_t>(block_index) * 18u;
		const uint16_t scale_bits = static_cast<uint16_t>(block[0]) | (static_cast<uint16_t>(block[1]) << 8u);
		const unsigned char packed = block[2u + (quant_lane & 15u)];
		const uint32_t quant = quant_lane < 16u ? (packed & 0x0fu) : (packed >> 4u);
		sum += input[col] * rocm_half_to_float(scale_bits) * (static_cast<float>(quant) - 8.0f);
	}
	return sum;
}

__device__ void rocm_gguf_q4_0_projection_row_pair16_sums(
	const float *input,
	const unsigned char *weights,
	uint32_t row,
	uint32_t cols,
	uint32_t col_lane,
	float &low_sum,
	float &high_sum)
{
	const uint32_t blocks_per_row = cols >> 5u;
	const uint64_t row_base = static_cast<uint64_t>(row) * blocks_per_row * 18u;
	low_sum = 0.0f;
	high_sum = 0.0f;
	for (uint32_t block_index = 0; block_index < blocks_per_row; ++block_index) {
		const unsigned char *block = weights + row_base + static_cast<uint64_t>(block_index) * 18u;
		const uint16_t scale_bits = static_cast<uint16_t>(block[0]) | (static_cast<uint16_t>(block[1]) << 8u);
		const unsigned char packed = block[2u + col_lane];
		const float scale = rocm_half_to_float(scale_bits);
		const uint32_t col_base = block_index << 5u;
		low_sum += input[col_base + col_lane] * scale * (static_cast<float>(packed & 0x0fu) - 8.0f);
		high_sum += input[col_base + 16u + col_lane] * scale * (static_cast<float>(packed >> 4u) - 8.0f);
	}
}

__device__ float rocm_gguf_q4_0_projection_row_pair16_reduce(float low_sum, float high_sum)
{
	float value = low_sum + high_sum;
	for (uint32_t stride = ROCM_GGUF_Q4_0_SELECTED_EXPERT_PAIR16_THREADS_PER_ROW / 2u; stride > 0u; stride >>= 1u) {
		value += rocm_shfl_down(value, stride, ROCM_GGUF_Q4_0_SELECTED_EXPERT_PAIR16_THREADS_PER_ROW);
	}
	return value;
}

__device__ void rocm_gguf_q4_k_scale_min(const unsigned char *scales, uint32_t group, uint32_t &scale, uint32_t &minimum)
{
	if (group < 4u) {
		scale = scales[group] & 63u;
		minimum = scales[group + 4u] & 63u;
		return;
	}
	scale = (scales[group + 4u] & 0x0fu) | ((scales[group - 4u] >> 6u) << 4u);
	minimum = (scales[group + 4u] >> 4u) | ((scales[group] >> 6u) << 4u);
}

__device__ float rocm_gguf_q4_k_projection_row_sum(
	const float *input,
	const unsigned char *weights,
	uint32_t row,
	uint32_t cols,
	uint32_t col_lane)
{
	const uint32_t blocks_per_row = cols >> 8u;
	const uint64_t row_base = static_cast<uint64_t>(row) * blocks_per_row * 144u;
	float sum = 0.0f;
	for (uint32_t col = col_lane; col < cols; col += ROCM_GGUF_Q4_0_PROJECTION_THREADS_PER_ROW) {
		const uint32_t block_index = col >> 8u;
		const uint32_t block_lane = col & 255u;
		const uint32_t group = block_lane >> 5u;
		const uint32_t quant_lane = block_lane & 31u;
		const unsigned char *block = weights + row_base + static_cast<uint64_t>(block_index) * 144u;
		const uint16_t d_bits = static_cast<uint16_t>(block[0]) | (static_cast<uint16_t>(block[1]) << 8u);
		const uint16_t dmin_bits = static_cast<uint16_t>(block[2]) | (static_cast<uint16_t>(block[3]) << 8u);
		uint32_t scale = 0;
		uint32_t minimum = 0;
		rocm_gguf_q4_k_scale_min(block + 4u, group, scale, minimum);
		const unsigned char packed = block[16u + (group >> 1u) * 32u + quant_lane];
		const uint32_t quant = (group & 1u) == 0u ? (packed & 0x0fu) : (packed >> 4u);
		const float value = rocm_half_to_float(d_bits) * static_cast<float>(scale * quant) - rocm_half_to_float(dmin_bits) * static_cast<float>(minimum);
		sum += input[col] * value;
	}
	return sum;
}

__device__ void rocm_gguf_q4_k_projection_row_pair16_sums(
	const float *input,
	const unsigned char *weights,
	uint32_t row,
	uint32_t cols,
	uint32_t col_lane,
	float &low_sum,
	float &high_sum)
{
	const uint32_t blocks_per_row = cols >> 8u;
	const uint64_t row_base = static_cast<uint64_t>(row) * blocks_per_row * 144u;
	low_sum = 0.0f;
	high_sum = 0.0f;
	for (uint32_t block_index = 0; block_index < blocks_per_row; ++block_index) {
		const unsigned char *block = weights + row_base + static_cast<uint64_t>(block_index) * 144u;
		const uint16_t d_bits = static_cast<uint16_t>(block[0]) | (static_cast<uint16_t>(block[1]) << 8u);
		const uint16_t dmin_bits = static_cast<uint16_t>(block[2]) | (static_cast<uint16_t>(block[3]) << 8u);
		const float d = rocm_half_to_float(d_bits);
		const float dmin = rocm_half_to_float(dmin_bits);
		const uint32_t block_col = block_index << 8u;
		for (uint32_t group = 0; group < 8u; ++group) {
			uint32_t scale = 0;
			uint32_t minimum = 0;
			rocm_gguf_q4_k_scale_min(block + 4u, group, scale, minimum);
			const uint32_t quant_base = 16u + (group >> 1u) * 32u;
			const unsigned char packed_low = block[quant_base + col_lane];
			const unsigned char packed_high = block[quant_base + 16u + col_lane];
			const uint32_t quant_low = (group & 1u) == 0u ? (packed_low & 0x0fu) : (packed_low >> 4u);
			const uint32_t quant_high = (group & 1u) == 0u ? (packed_high & 0x0fu) : (packed_high >> 4u);
			const float low_value = d * static_cast<float>(scale * quant_low) - dmin * static_cast<float>(minimum);
			const float high_value = d * static_cast<float>(scale * quant_high) - dmin * static_cast<float>(minimum);
			const uint32_t group_col = block_col + (group << 5u);
			low_sum += input[group_col + col_lane] * low_value;
			high_sum += input[group_col + 16u + col_lane] * high_value;
		}
	}
}

__device__ void rocm_gguf_q4_k_expanded_projection_row_pair16_sums(
	const float *input,
	const unsigned char *weights,
	uint32_t row,
	uint32_t cols,
	uint32_t col_lane,
	float &low_sum,
	float &high_sum)
{
	const uint32_t blocks_per_row = cols >> 8u;
	const uint64_t row_base = static_cast<uint64_t>(row) * blocks_per_row * ROCM_GGUF_Q4_K_EXPANDED_BLOCK_BYTES;
	low_sum = 0.0f;
	high_sum = 0.0f;
	for (uint32_t block_index = 0; block_index < blocks_per_row; ++block_index) {
		const unsigned char *block = weights + row_base + static_cast<uint64_t>(block_index) * ROCM_GGUF_Q4_K_EXPANDED_BLOCK_BYTES;
		const float *metadata = reinterpret_cast<const float *>(block);
		const float d = metadata[0];
		const float dmin = metadata[1];
		const uint32_t block_col = block_index << 8u;
		for (uint32_t group = 0; group < 8u; ++group) {
			const uint32_t scale = block[8u + group];
			const uint32_t minimum = block[16u + group];
			const uint32_t quant_base = 24u + (group >> 1u) * 32u;
			const unsigned char packed_low = block[quant_base + col_lane];
			const unsigned char packed_high = block[quant_base + 16u + col_lane];
			const uint32_t quant_low = (group & 1u) == 0u ? (packed_low & 0x0fu) : (packed_low >> 4u);
			const uint32_t quant_high = (group & 1u) == 0u ? (packed_high & 0x0fu) : (packed_high >> 4u);
			const float low_value = d * static_cast<float>(scale * quant_low) - dmin * static_cast<float>(minimum);
			const float high_value = d * static_cast<float>(scale * quant_high) - dmin * static_cast<float>(minimum);
			const uint32_t group_col = block_col + (group << 5u);
			low_sum += input[group_col + col_lane] * low_value;
			high_sum += input[group_col + 16u + col_lane] * high_value;
		}
	}
}

__device__ float rocm_gguf_q5_1_projection_row_sum(
	const float *input,
	const unsigned char *weights,
	uint32_t row,
	uint32_t cols,
	uint32_t col_lane)
{
	const uint32_t blocks_per_row = cols >> 5u;
	const uint64_t row_base = static_cast<uint64_t>(row) * blocks_per_row * 24u;
	float sum = 0.0f;
	for (uint32_t col = col_lane; col < cols; col += ROCM_GGUF_Q4_0_PROJECTION_THREADS_PER_ROW) {
		const uint32_t block_index = col >> 5u;
		const uint32_t quant_lane = col & 31u;
		const unsigned char *block = weights + row_base + static_cast<uint64_t>(block_index) * 24u;
		const uint16_t d_bits = static_cast<uint16_t>(block[0]) | (static_cast<uint16_t>(block[1]) << 8u);
		const uint16_t m_bits = static_cast<uint16_t>(block[2]) | (static_cast<uint16_t>(block[3]) << 8u);
		const uint32_t qh = static_cast<uint32_t>(block[4]) |
			(static_cast<uint32_t>(block[5]) << 8u) |
			(static_cast<uint32_t>(block[6]) << 16u) |
			(static_cast<uint32_t>(block[7]) << 24u);
		const unsigned char packed = block[8u + (quant_lane & 15u)];
		uint32_t quant = quant_lane < 16u ? (packed & 0x0fu) : (packed >> 4u);
		quant |= ((qh >> quant_lane) & 1u) << 4u;
		const float value = rocm_half_to_float(d_bits) * static_cast<float>(quant) + rocm_half_to_float(m_bits);
		sum += input[col] * value;
	}
	return sum;
}

__device__ void rocm_gguf_q5_1_projection_row_pair16_sums(
	const float *input,
	const unsigned char *weights,
	uint32_t row,
	uint32_t cols,
	uint32_t col_lane,
	float &low_sum,
	float &high_sum)
{
	const uint32_t blocks_per_row = cols >> 5u;
	const uint64_t row_base = static_cast<uint64_t>(row) * blocks_per_row * 24u;
	low_sum = 0.0f;
	high_sum = 0.0f;
	for (uint32_t block_index = 0; block_index < blocks_per_row; ++block_index) {
		const unsigned char *block = weights + row_base + static_cast<uint64_t>(block_index) * 24u;
		const uint16_t d_bits = static_cast<uint16_t>(block[0]) | (static_cast<uint16_t>(block[1]) << 8u);
		const uint16_t m_bits = static_cast<uint16_t>(block[2]) | (static_cast<uint16_t>(block[3]) << 8u);
		const uint32_t qh = static_cast<uint32_t>(block[4]) |
			(static_cast<uint32_t>(block[5]) << 8u) |
			(static_cast<uint32_t>(block[6]) << 16u) |
			(static_cast<uint32_t>(block[7]) << 24u);
		const unsigned char packed = block[8u + col_lane];
		uint32_t quant_low = packed & 0x0fu;
		uint32_t quant_high = packed >> 4u;
		quant_low |= ((qh >> col_lane) & 1u) << 4u;
		quant_high |= ((qh >> (16u + col_lane)) & 1u) << 4u;
		const float d = rocm_half_to_float(d_bits);
		const float m = rocm_half_to_float(m_bits);
		const uint32_t col_base = block_index << 5u;
		low_sum += input[col_base + col_lane] * (d * static_cast<float>(quant_low) + m);
		high_sum += input[col_base + 16u + col_lane] * (d * static_cast<float>(quant_high) + m);
	}
}

__device__ float rocm_gguf_q8_0_projection_row_sum(
	const float *input,
	const unsigned char *weights,
	uint32_t row,
	uint32_t cols,
	uint32_t col_lane)
{
	const uint32_t blocks_per_row = cols >> 5u;
	const uint64_t row_base = static_cast<uint64_t>(row) * blocks_per_row * 34u;
	float sum = 0.0f;
	for (uint32_t col = col_lane; col < cols; col += ROCM_GGUF_Q4_0_PROJECTION_THREADS_PER_ROW) {
		const uint32_t block_index = col >> 5u;
		const uint32_t quant_lane = col & 31u;
		const unsigned char *block = weights + row_base + static_cast<uint64_t>(block_index) * 34u;
		const uint16_t d_bits = static_cast<uint16_t>(block[0]) | (static_cast<uint16_t>(block[1]) << 8u);
		const uint32_t raw = block[2u + quant_lane];
		const int32_t quant = raw < 128u ? static_cast<int32_t>(raw) : static_cast<int32_t>(raw) - 256;
		sum += input[col] * rocm_half_to_float(d_bits) * static_cast<float>(quant);
	}
	return sum;
}

__device__ void rocm_gguf_q8_0_projection_row_pair16_sums(
	const float *input,
	const unsigned char *weights,
	uint32_t row,
	uint32_t cols,
	uint32_t col_lane,
	float &low_sum,
	float &high_sum)
{
	const uint32_t blocks_per_row = cols >> 5u;
	const uint64_t row_base = static_cast<uint64_t>(row) * blocks_per_row * 34u;
	low_sum = 0.0f;
	high_sum = 0.0f;
	for (uint32_t block_index = 0; block_index < blocks_per_row; ++block_index) {
		const unsigned char *block = weights + row_base + static_cast<uint64_t>(block_index) * 34u;
		const uint16_t d_bits = static_cast<uint16_t>(block[0]) | (static_cast<uint16_t>(block[1]) << 8u);
		const uint32_t raw_low = block[2u + col_lane];
		const uint32_t raw_high = block[18u + col_lane];
		const int32_t quant_low = raw_low < 128u ? static_cast<int32_t>(raw_low) : static_cast<int32_t>(raw_low) - 256;
		const int32_t quant_high = raw_high < 128u ? static_cast<int32_t>(raw_high) : static_cast<int32_t>(raw_high) - 256;
		const float d = rocm_half_to_float(d_bits);
		const uint32_t col_base = block_index << 5u;
		low_sum += input[col_base + col_lane] * d * static_cast<float>(quant_low);
		high_sum += input[col_base + 16u + col_lane] * d * static_cast<float>(quant_high);
	}
}

extern "C" __global__ void rocm_gguf_q4_k_expand_metadata(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_gguf_q4_k_expand_launch_args &args = *reinterpret_cast<const rocm_gguf_q4_k_expand_launch_args *>(packet);
	if (!rocm_valid_gguf_q4_k_expand_args(args) || (blockDim.x & 31u) != 0u) {
		return;
	}
	const uint32_t linear = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t block_index = linear >> 5u;
	if (block_index >= args.block_count) {
		return;
	}
	const uint32_t lane = linear & 31u;
	const unsigned char *raw = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(args.raw_pointer)) + static_cast<uint64_t>(block_index) * 144u;
	unsigned char *expanded = reinterpret_cast<unsigned char *>(static_cast<uintptr_t>(args.expanded_pointer)) + static_cast<uint64_t>(block_index) * ROCM_GGUF_Q4_K_EXPANDED_BLOCK_BYTES;
	if (lane == 0u) {
		const uint16_t d_bits = static_cast<uint16_t>(raw[0]) | (static_cast<uint16_t>(raw[1]) << 8u);
		const uint16_t dmin_bits = static_cast<uint16_t>(raw[2]) | (static_cast<uint16_t>(raw[3]) << 8u);
		float *metadata = reinterpret_cast<float *>(expanded);
		metadata[0] = rocm_half_to_float(d_bits);
		metadata[1] = rocm_half_to_float(dmin_bits);
		for (uint32_t group = 0; group < 8u; ++group) {
			uint32_t scale = 0;
			uint32_t minimum = 0;
			rocm_gguf_q4_k_scale_min(raw + 4u, group, scale, minimum);
			expanded[8u + group] = static_cast<unsigned char>(scale);
			expanded[16u + group] = static_cast<unsigned char>(minimum);
		}
	}
	for (uint32_t offset = lane; offset < 128u; offset += 32u) {
		expanded[24u + offset] = raw[16u + offset];
	}
}

extern "C" __global__ void rocm_gguf_q4_0_projection(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_gguf_q4_0_projection_launch_args &args = *reinterpret_cast<const rocm_gguf_q4_0_projection_launch_args *>(packet);
	if (!rocm_valid_gguf_q4_0_projection_args(args, false) || blockDim.x != ROCM_GGUF_Q4_0_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_GGUF_Q4_0_PROJECTION_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_GGUF_Q4_0_PROJECTION_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_GGUF_Q4_0_PROJECTION_ROWS_PER_BLOCK + row_lane;
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const unsigned char *weights = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(args.weight_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float sum = 0.0f;
	if (row < args.rows) {
		sum = rocm_gguf_q4_0_projection_row_sum(input, weights, args.row_offset + row, args.cols, col_lane);
	}
	sum = rocm_mlx_q4_row_reduce(sum);
	if (col_lane == 0u && row < args.rows) {
		output[row] = sum;
	}
}

extern "C" __global__ void rocm_gguf_q4_0_gelu_tanh_gate_up(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_gguf_q4_0_projection_launch_args &args = *reinterpret_cast<const rocm_gguf_q4_0_projection_launch_args *>(packet);
	if (!rocm_valid_gguf_q4_0_projection_args(args, true) || blockDim.x != ROCM_GGUF_Q4_0_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_GGUF_Q4_0_PROJECTION_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_GGUF_Q4_0_PROJECTION_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_GGUF_Q4_0_PROJECTION_ROWS_PER_BLOCK + row_lane;
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const unsigned char *weights = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(args.weight_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float gate_sum = 0.0f;
	float up_sum = 0.0f;
	if (row < args.rows) {
		gate_sum = rocm_gguf_q4_0_projection_row_sum(input, weights, args.row_offset + row, args.cols, col_lane);
		up_sum = rocm_gguf_q4_0_projection_row_sum(input, weights, args.row_offset + args.rows + row, args.cols, col_lane);
	}
	gate_sum = rocm_mlx_q4_row_reduce(gate_sum);
	up_sum = rocm_mlx_q4_row_reduce(up_sum);
	if (col_lane == 0u && row < args.rows) {
		output[row] = rocm_gelu_tanh_value(gate_sum) * up_sum;
	}
}

extern "C" __global__ void rocm_gguf_q4_0_selected_expert_gate_up(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_gguf_q4_0_selected_experts_launch_args &args = *reinterpret_cast<const rocm_gguf_q4_0_selected_experts_launch_args *>(packet);
	if (!rocm_valid_gguf_q4_0_selected_experts_args(args) || args.gate_up_format != ROCM_GGUF_EXPERT_FORMAT_Q4_0 || blockDim.x != ROCM_GGUF_Q4_0_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_GGUF_Q4_0_PROJECTION_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_GGUF_Q4_0_PROJECTION_THREADS_PER_ROW;
	const uint32_t selected_row = blockIdx.x * ROCM_GGUF_Q4_0_PROJECTION_ROWS_PER_BLOCK + row_lane;
	const uint32_t selected_rows = args.top_k * args.expert_ff;
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	float *activation = reinterpret_cast<float *>(static_cast<uintptr_t>(args.activation_pointer));
	float gate_sum = 0.0f;
	float up_sum = 0.0f;
	if (selected_row < selected_rows) {
		const uint32_t expert = selected_row / args.expert_ff;
		const uint32_t row = selected_row - expert * args.expert_ff;
		const unsigned char *weights = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(args.gate_up_pointers[expert]));
		gate_sum = rocm_gguf_q4_0_projection_row_sum(input, weights, row, args.hidden_size, col_lane);
		up_sum = rocm_gguf_q4_0_projection_row_sum(input, weights, args.expert_ff + row, args.hidden_size, col_lane);
	}
	gate_sum = rocm_mlx_q4_row_reduce(gate_sum);
	up_sum = rocm_mlx_q4_row_reduce(up_sum);
	if (col_lane == 0u && selected_row < selected_rows) {
		activation[selected_row] = rocm_gelu_tanh_value(gate_sum) * up_sum;
	}
}

extern "C" __global__ void rocm_gguf_q4_0_selected_expert_down(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_gguf_q4_0_selected_experts_launch_args &args = *reinterpret_cast<const rocm_gguf_q4_0_selected_experts_launch_args *>(packet);
	if (!rocm_valid_gguf_q4_0_selected_experts_args(args) || args.down_format != ROCM_GGUF_EXPERT_FORMAT_Q4_0 || blockDim.x != ROCM_GGUF_Q4_0_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_GGUF_Q4_0_PROJECTION_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_GGUF_Q4_0_PROJECTION_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_GGUF_Q4_0_PROJECTION_ROWS_PER_BLOCK + row_lane;
	const float *activation = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.activation_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float combined = 0.0f;
	for (uint32_t expert = 0; expert < args.top_k; ++expert) {
		float sum = 0.0f;
		if (row < args.hidden_size) {
			const unsigned char *weights = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(args.down_pointers[expert]));
			sum = rocm_gguf_q4_0_projection_row_sum(activation + expert * args.expert_ff, weights, row, args.expert_ff, col_lane);
		}
		sum = rocm_mlx_q4_row_reduce(sum);
		if (col_lane == 0u) {
			combined += args.route_weights[expert] * sum;
		}
	}
	if (col_lane == 0u && row < args.hidden_size) {
		output[row] = combined;
	}
}

extern "C" __global__ void rocm_gguf_q4_0_selected_expert_gate_up_pair16(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_gguf_q4_0_selected_experts_launch_args &args = *reinterpret_cast<const rocm_gguf_q4_0_selected_experts_launch_args *>(packet);
	if (!rocm_valid_gguf_q4_0_selected_experts_args(args) || args.gate_up_format != ROCM_GGUF_EXPERT_FORMAT_Q4_0 || blockDim.x != ROCM_GGUF_Q4_0_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_GGUF_Q4_0_SELECTED_EXPERT_PAIR16_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_GGUF_Q4_0_SELECTED_EXPERT_PAIR16_THREADS_PER_ROW;
	const uint32_t selected_row = blockIdx.x * ROCM_GGUF_Q4_0_SELECTED_EXPERT_PAIR16_ROWS_PER_BLOCK + row_lane;
	const uint32_t selected_rows = args.top_k * args.expert_ff;
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	float *activation = reinterpret_cast<float *>(static_cast<uintptr_t>(args.activation_pointer));
	float gate_low = 0.0f;
	float gate_high = 0.0f;
	float up_low = 0.0f;
	float up_high = 0.0f;
	if (selected_row < selected_rows) {
		const uint32_t expert = selected_row / args.expert_ff;
		const uint32_t row = selected_row - expert * args.expert_ff;
		const unsigned char *weights = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(args.gate_up_pointers[expert]));
		rocm_gguf_q4_0_projection_row_pair16_sums(input, weights, row, args.hidden_size, col_lane, gate_low, gate_high);
		rocm_gguf_q4_0_projection_row_pair16_sums(input, weights, args.expert_ff + row, args.hidden_size, col_lane, up_low, up_high);
	}
	const float gate_sum = rocm_gguf_q4_0_projection_row_pair16_reduce(gate_low, gate_high);
	const float up_sum = rocm_gguf_q4_0_projection_row_pair16_reduce(up_low, up_high);
	if (col_lane == 0u && selected_row < selected_rows) {
		activation[selected_row] = rocm_gelu_tanh_value(gate_sum) * up_sum;
	}
}

extern "C" __global__ void rocm_gguf_q4_0_selected_expert_down_pair16(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_gguf_q4_0_selected_experts_launch_args &args = *reinterpret_cast<const rocm_gguf_q4_0_selected_experts_launch_args *>(packet);
	if (!rocm_valid_gguf_q4_0_selected_experts_args(args) || args.down_format != ROCM_GGUF_EXPERT_FORMAT_Q4_0 || blockDim.x != ROCM_GGUF_Q4_0_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_GGUF_Q4_0_SELECTED_EXPERT_PAIR16_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_GGUF_Q4_0_SELECTED_EXPERT_PAIR16_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_GGUF_Q4_0_SELECTED_EXPERT_PAIR16_ROWS_PER_BLOCK + row_lane;
	const float *activation = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.activation_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float combined = 0.0f;
	for (uint32_t expert = 0; expert < args.top_k; ++expert) {
		float low_sum = 0.0f;
		float high_sum = 0.0f;
		if (row < args.hidden_size) {
			const unsigned char *weights = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(args.down_pointers[expert]));
			rocm_gguf_q4_0_projection_row_pair16_sums(activation + expert * args.expert_ff, weights, row, args.expert_ff, col_lane, low_sum, high_sum);
		}
		const float sum = rocm_gguf_q4_0_projection_row_pair16_reduce(low_sum, high_sum);
		if (col_lane == 0u) {
			combined += args.route_weights[expert] * sum;
		}
	}
	if (col_lane == 0u && row < args.hidden_size) {
		output[row] = combined;
	}
}

extern "C" __global__ void rocm_gguf_q4_k_selected_expert_gate_up(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_gguf_q4_0_selected_experts_launch_args &args = *reinterpret_cast<const rocm_gguf_q4_0_selected_experts_launch_args *>(packet);
	if (!rocm_valid_gguf_q4_0_selected_experts_args(args) || args.gate_up_format != ROCM_GGUF_EXPERT_FORMAT_Q4_K || blockDim.x != ROCM_GGUF_Q4_0_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_GGUF_Q4_0_PROJECTION_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_GGUF_Q4_0_PROJECTION_THREADS_PER_ROW;
	const uint32_t selected_row = blockIdx.x * ROCM_GGUF_Q4_0_PROJECTION_ROWS_PER_BLOCK + row_lane;
	const uint32_t selected_rows = args.top_k * args.expert_ff;
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	float *activation = reinterpret_cast<float *>(static_cast<uintptr_t>(args.activation_pointer));
	float gate_sum = 0.0f;
	float up_sum = 0.0f;
	if (selected_row < selected_rows) {
		const uint32_t expert = selected_row / args.expert_ff;
		const uint32_t row = selected_row - expert * args.expert_ff;
		const unsigned char *weights = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(args.gate_up_pointers[expert]));
		gate_sum = rocm_gguf_q4_k_projection_row_sum(input, weights, row, args.hidden_size, col_lane);
		up_sum = rocm_gguf_q4_k_projection_row_sum(input, weights, args.expert_ff + row, args.hidden_size, col_lane);
	}
	gate_sum = rocm_mlx_q4_row_reduce(gate_sum);
	up_sum = rocm_mlx_q4_row_reduce(up_sum);
	if (col_lane == 0u && selected_row < selected_rows) {
		activation[selected_row] = rocm_gelu_tanh_value(gate_sum) * up_sum;
	}
}

extern "C" __global__ void rocm_gguf_q5_1_selected_expert_down(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_gguf_q4_0_selected_experts_launch_args &args = *reinterpret_cast<const rocm_gguf_q4_0_selected_experts_launch_args *>(packet);
	if (!rocm_valid_gguf_q4_0_selected_experts_args(args) || args.down_format != ROCM_GGUF_EXPERT_FORMAT_Q5_1 || blockDim.x != ROCM_GGUF_Q4_0_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_GGUF_Q4_0_PROJECTION_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_GGUF_Q4_0_PROJECTION_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_GGUF_Q4_0_PROJECTION_ROWS_PER_BLOCK + row_lane;
	const float *activation = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.activation_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float combined = 0.0f;
	for (uint32_t expert = 0; expert < args.top_k; ++expert) {
		float sum = 0.0f;
		if (row < args.hidden_size) {
			const unsigned char *weights = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(args.down_pointers[expert]));
			sum = rocm_gguf_q5_1_projection_row_sum(activation + expert * args.expert_ff, weights, row, args.expert_ff, col_lane);
		}
		sum = rocm_mlx_q4_row_reduce(sum);
		if (col_lane == 0u) {
			combined += args.route_weights[expert] * sum;
		}
	}
	if (col_lane == 0u && row < args.hidden_size) {
		output[row] = combined;
	}
}

extern "C" __global__ void rocm_gguf_q8_0_selected_expert_down(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_gguf_q4_0_selected_experts_launch_args &args = *reinterpret_cast<const rocm_gguf_q4_0_selected_experts_launch_args *>(packet);
	if (!rocm_valid_gguf_q4_0_selected_experts_args(args) || args.down_format != ROCM_GGUF_EXPERT_FORMAT_Q8_0 || blockDim.x != ROCM_GGUF_Q4_0_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_GGUF_Q4_0_PROJECTION_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_GGUF_Q4_0_PROJECTION_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_GGUF_Q4_0_PROJECTION_ROWS_PER_BLOCK + row_lane;
	const float *activation = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.activation_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float combined = 0.0f;
	for (uint32_t expert = 0; expert < args.top_k; ++expert) {
		float sum = 0.0f;
		if (row < args.hidden_size) {
			const unsigned char *weights = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(args.down_pointers[expert]));
			sum = rocm_gguf_q8_0_projection_row_sum(activation + expert * args.expert_ff, weights, row, args.expert_ff, col_lane);
		}
		sum = rocm_mlx_q4_row_reduce(sum);
		if (col_lane == 0u) {
			combined += args.route_weights[expert] * sum;
		}
	}
	if (col_lane == 0u && row < args.hidden_size) {
		output[row] = combined;
	}
}

extern "C" __global__ void rocm_gguf_q4_k_selected_expert_gate_up_pair16(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_gguf_q4_0_selected_experts_launch_args &args = *reinterpret_cast<const rocm_gguf_q4_0_selected_experts_launch_args *>(packet);
	if (!rocm_valid_gguf_q4_0_selected_experts_args(args) || args.gate_up_format != ROCM_GGUF_EXPERT_FORMAT_Q4_K || blockDim.x != ROCM_GGUF_Q4_0_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_GGUF_Q4_0_SELECTED_EXPERT_PAIR16_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_GGUF_Q4_0_SELECTED_EXPERT_PAIR16_THREADS_PER_ROW;
	const uint32_t selected_row = blockIdx.x * ROCM_GGUF_Q4_0_SELECTED_EXPERT_PAIR16_ROWS_PER_BLOCK + row_lane;
	const uint32_t selected_rows = args.top_k * args.expert_ff;
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	float *activation = reinterpret_cast<float *>(static_cast<uintptr_t>(args.activation_pointer));
	float gate_low = 0.0f;
	float gate_high = 0.0f;
	float up_low = 0.0f;
	float up_high = 0.0f;
	if (selected_row < selected_rows) {
		const uint32_t expert = selected_row / args.expert_ff;
		const uint32_t row = selected_row - expert * args.expert_ff;
		const unsigned char *weights = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(args.gate_up_pointers[expert]));
		rocm_gguf_q4_k_projection_row_pair16_sums(input, weights, row, args.hidden_size, col_lane, gate_low, gate_high);
		rocm_gguf_q4_k_projection_row_pair16_sums(input, weights, args.expert_ff + row, args.hidden_size, col_lane, up_low, up_high);
	}
	const float gate_sum = rocm_gguf_q4_0_projection_row_pair16_reduce(gate_low, gate_high);
	const float up_sum = rocm_gguf_q4_0_projection_row_pair16_reduce(up_low, up_high);
	if (col_lane == 0u && selected_row < selected_rows) {
		activation[selected_row] = rocm_gelu_tanh_value(gate_sum) * up_sum;
	}
}

extern "C" __global__ void rocm_gguf_q4_k_selected_expert_gate_up_split_pair16(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_gguf_q4_0_selected_experts_launch_args &args = *reinterpret_cast<const rocm_gguf_q4_0_selected_experts_launch_args *>(packet);
	if (!rocm_valid_gguf_q4_0_selected_experts_args(args) ||
		args.gate_up_format != ROCM_GGUF_EXPERT_FORMAT_Q4_K ||
		blockDim.x != ROCM_GGUF_Q4_0_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t col_lane = threadIdx.x & (ROCM_GGUF_Q4_0_SELECTED_EXPERT_PAIR16_THREADS_PER_ROW - 1u);
	const uint32_t group = threadIdx.x / ROCM_GGUF_Q4_0_SELECTED_EXPERT_PAIR16_THREADS_PER_ROW;
	const uint32_t row_lane = group >> 1u;
	const uint32_t projection = group & 1u;
	const uint32_t selected_row = blockIdx.x * ROCM_GGUF_Q4_K_SELECTED_EXPERT_GATE_UP_SPLIT_ROWS_PER_BLOCK + row_lane;
	const uint32_t selected_rows = args.top_k * args.expert_ff;
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	float *activation = reinterpret_cast<float *>(static_cast<uintptr_t>(args.activation_pointer));
	float low_sum = 0.0f;
	float high_sum = 0.0f;
	if (selected_row < selected_rows) {
		const uint32_t expert = selected_row / args.expert_ff;
		const uint32_t row = selected_row - expert * args.expert_ff;
		const unsigned char *weights = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(args.gate_up_pointers[expert]));
		rocm_gguf_q4_k_projection_row_pair16_sums(input, weights, projection * args.expert_ff + row, args.hidden_size, col_lane, low_sum, high_sum);
	}
	const float projection_sum = rocm_gguf_q4_0_projection_row_pair16_reduce(low_sum, high_sum);
	__shared__ float projection_sums[ROCM_GGUF_Q4_K_SELECTED_EXPERT_GATE_UP_SPLIT_ROWS_PER_BLOCK * 2u];
	if (col_lane == 0u) {
		projection_sums[group] = projection_sum;
	}
	__syncthreads();
	if (col_lane == 0u && projection == 0u && selected_row < selected_rows) {
		activation[selected_row] = rocm_gelu_tanh_value(projection_sums[group]) * projection_sums[group + 1u];
	}
}

extern "C" __global__ void rocm_gguf_q4_k_expanded_selected_expert_gate_up_split_pair16(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_gguf_q4_0_selected_experts_launch_args &args = *reinterpret_cast<const rocm_gguf_q4_0_selected_experts_launch_args *>(packet);
	if (!rocm_valid_gguf_q4_0_selected_experts_args(args) ||
		args.gate_up_format != ROCM_GGUF_EXPERT_FORMAT_Q4_K_EXPANDED ||
		blockDim.x != ROCM_GGUF_Q4_0_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t col_lane = threadIdx.x & (ROCM_GGUF_Q4_0_SELECTED_EXPERT_PAIR16_THREADS_PER_ROW - 1u);
	const uint32_t group = threadIdx.x / ROCM_GGUF_Q4_0_SELECTED_EXPERT_PAIR16_THREADS_PER_ROW;
	const uint32_t row_lane = group >> 1u;
	const uint32_t projection = group & 1u;
	const uint32_t selected_row = blockIdx.x * ROCM_GGUF_Q4_K_SELECTED_EXPERT_GATE_UP_SPLIT_ROWS_PER_BLOCK + row_lane;
	const uint32_t selected_rows = args.top_k * args.expert_ff;
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	float *activation = reinterpret_cast<float *>(static_cast<uintptr_t>(args.activation_pointer));
	float low_sum = 0.0f;
	float high_sum = 0.0f;
	if (selected_row < selected_rows) {
		const uint32_t expert = selected_row / args.expert_ff;
		const uint32_t row = selected_row - expert * args.expert_ff;
		const unsigned char *weights = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(args.gate_up_pointers[expert]));
		rocm_gguf_q4_k_expanded_projection_row_pair16_sums(input, weights, projection * args.expert_ff + row, args.hidden_size, col_lane, low_sum, high_sum);
	}
	const float projection_sum = rocm_gguf_q4_0_projection_row_pair16_reduce(low_sum, high_sum);
	__shared__ float projection_sums[ROCM_GGUF_Q4_K_SELECTED_EXPERT_GATE_UP_SPLIT_ROWS_PER_BLOCK * 2u];
	if (col_lane == 0u) {
		projection_sums[group] = projection_sum;
	}
	__syncthreads();
	if (col_lane == 0u && projection == 0u && selected_row < selected_rows) {
		activation[selected_row] = rocm_gelu_tanh_value(projection_sums[group]) * projection_sums[group + 1u];
	}
}

extern "C" __global__ void rocm_gguf_q5_1_selected_expert_down_pair16(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_gguf_q4_0_selected_experts_launch_args &args = *reinterpret_cast<const rocm_gguf_q4_0_selected_experts_launch_args *>(packet);
	if (!rocm_valid_gguf_q4_0_selected_experts_args(args) || args.down_format != ROCM_GGUF_EXPERT_FORMAT_Q5_1 || blockDim.x != ROCM_GGUF_Q4_0_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_GGUF_Q4_0_SELECTED_EXPERT_PAIR16_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_GGUF_Q4_0_SELECTED_EXPERT_PAIR16_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_GGUF_Q4_0_SELECTED_EXPERT_PAIR16_ROWS_PER_BLOCK + row_lane;
	const float *activation = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.activation_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float combined = 0.0f;
	for (uint32_t expert = 0; expert < args.top_k; ++expert) {
		float low_sum = 0.0f;
		float high_sum = 0.0f;
		if (row < args.hidden_size) {
			const unsigned char *weights = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(args.down_pointers[expert]));
			rocm_gguf_q5_1_projection_row_pair16_sums(activation + expert * args.expert_ff, weights, row, args.expert_ff, col_lane, low_sum, high_sum);
		}
		const float sum = rocm_gguf_q4_0_projection_row_pair16_reduce(low_sum, high_sum);
		if (col_lane == 0u) {
			combined += args.route_weights[expert] * sum;
		}
	}
	if (col_lane == 0u && row < args.hidden_size) {
		output[row] = combined;
	}
}

extern "C" __global__ void rocm_gguf_q5_1_selected_expert_down_expert8_pair16(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_gguf_q4_0_selected_experts_launch_args &args = *reinterpret_cast<const rocm_gguf_q4_0_selected_experts_launch_args *>(packet);
	if (!rocm_valid_gguf_q4_0_selected_experts_args(args) ||
		args.down_format != ROCM_GGUF_EXPERT_FORMAT_Q5_1 ||
		args.top_k != ROCM_GGUF_Q5_1_SELECTED_EXPERT_DOWN_EXPERT8_EXPERTS ||
		blockDim.x != ROCM_GGUF_Q4_0_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t col_lane = threadIdx.x & (ROCM_GGUF_Q4_0_SELECTED_EXPERT_PAIR16_THREADS_PER_ROW - 1u);
	const uint32_t group = threadIdx.x / ROCM_GGUF_Q4_0_SELECTED_EXPERT_PAIR16_THREADS_PER_ROW;
	const uint32_t row_lane = group / ROCM_GGUF_Q5_1_SELECTED_EXPERT_DOWN_EXPERT8_EXPERTS;
	const uint32_t expert = group - row_lane * ROCM_GGUF_Q5_1_SELECTED_EXPERT_DOWN_EXPERT8_EXPERTS;
	const uint32_t row = blockIdx.x * ROCM_GGUF_Q5_1_SELECTED_EXPERT_DOWN_EXPERT8_ROWS_PER_BLOCK + row_lane;
	const float *activation = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.activation_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float low_sum = 0.0f;
	float high_sum = 0.0f;
	if (row < args.hidden_size) {
		const unsigned char *weights = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(args.down_pointers[expert]));
		rocm_gguf_q5_1_projection_row_pair16_sums(activation + expert * args.expert_ff, weights, row, args.expert_ff, col_lane, low_sum, high_sum);
	}
	const float expert_sum = rocm_gguf_q4_0_projection_row_pair16_reduce(low_sum, high_sum);
	__shared__ float expert_sums[ROCM_GGUF_Q5_1_SELECTED_EXPERT_DOWN_EXPERT8_ROWS_PER_BLOCK * ROCM_GGUF_Q5_1_SELECTED_EXPERT_DOWN_EXPERT8_EXPERTS];
	if (col_lane == 0u) {
		expert_sums[group] = expert_sum;
	}
	__syncthreads();
	if (col_lane == 0u && expert == 0u && row < args.hidden_size) {
		float combined = 0.0f;
		for (uint32_t selected = 0; selected < ROCM_GGUF_Q5_1_SELECTED_EXPERT_DOWN_EXPERT8_EXPERTS; ++selected) {
			combined += args.route_weights[selected] * expert_sums[row_lane * ROCM_GGUF_Q5_1_SELECTED_EXPERT_DOWN_EXPERT8_EXPERTS + selected];
		}
		output[row] = combined;
	}
}

extern "C" __global__ void rocm_gguf_q8_0_selected_expert_down_pair16(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_gguf_q4_0_selected_experts_launch_args &args = *reinterpret_cast<const rocm_gguf_q4_0_selected_experts_launch_args *>(packet);
	if (!rocm_valid_gguf_q4_0_selected_experts_args(args) || args.down_format != ROCM_GGUF_EXPERT_FORMAT_Q8_0 || blockDim.x != ROCM_GGUF_Q4_0_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_GGUF_Q4_0_SELECTED_EXPERT_PAIR16_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_GGUF_Q4_0_SELECTED_EXPERT_PAIR16_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_GGUF_Q4_0_SELECTED_EXPERT_PAIR16_ROWS_PER_BLOCK + row_lane;
	const float *activation = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.activation_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float combined = 0.0f;
	for (uint32_t expert = 0; expert < args.top_k; ++expert) {
		float low_sum = 0.0f;
		float high_sum = 0.0f;
		if (row < args.hidden_size) {
			const unsigned char *weights = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(args.down_pointers[expert]));
			rocm_gguf_q8_0_projection_row_pair16_sums(activation + expert * args.expert_ff, weights, row, args.expert_ff, col_lane, low_sum, high_sum);
		}
		const float sum = rocm_gguf_q4_0_projection_row_pair16_reduce(low_sum, high_sum);
		if (col_lane == 0u) {
			combined += args.route_weights[expert] * sum;
		}
	}
	if (col_lane == 0u && row < args.hidden_size) {
		output[row] = combined;
	}
}

extern "C" __global__ void rocm_mlx_q4_projection_cols256(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_projection_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_projection_launch_args *>(packet);
	const bool supported_cols256_shape =
		args.cols == 256u &&
		((args.bits == 8u && args.group_size == 32u) ||
		 ((args.bits == 4u || args.bits == 6u) && (args.group_size == 32u || args.group_size == 64u)));
	if (!rocm_valid_mlx_q4_projection_args(args) || !supported_cols256_shape) {
		return;
	}

	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_COLS256_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_COLS256_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_COLS256_ROWS_PER_BLOCK + row_lane;

	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.weight_pointer));
	const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.scale_pointer));
	const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.bias_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float sum = 0.0f;
	if (row < args.rows) {
		sum = rocm_mlx_q4_projection_row_sum(input, weights, scales, biases, row, args.cols, args.group_size, args.bits, col_lane, ROCM_MLX_Q4_PROJECTION_COLS256_THREADS_PER_ROW);
	}
	sum = rocm_mlx_q4_cols256_row_reduce(sum);
	if (col_lane == 0 && row < args.rows) {
		output[row] = sum;
	}
}

extern "C" __global__ void rocm_mlx_q4_projection_q6_g16_row16(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_projection_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_projection_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_projection_args(args) || args.bits != 6u || args.group_size != 16u || args.cols <= 2048u) {
		return;
	}
	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}

	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_Q6_ROW16_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_Q6_ROW16_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_Q6_ROW16_ROWS_PER_BLOCK + row_lane;
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.weight_pointer));
	const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.scale_pointer));
	const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.bias_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float sum = 0.0f;
	if (row < args.rows) {
		sum = rocm_mlx_q4_projection_row_sum(input, weights, scales, biases, row, args.cols, args.group_size, args.bits, col_lane, ROCM_MLX_Q4_PROJECTION_Q6_ROW16_THREADS_PER_ROW);
	}
	sum = rocm_mlx_q4_projection_q6_row16_reduce(sum);
	if (col_lane == 0u && row < args.rows) {
		output[row] = sum;
	}
}

extern "C" __global__ void rocm_mlx_q4_projection_q6_row16(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_projection_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_projection_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_projection_args(args) || args.bits != 6u || args.group_size != 64u || args.cols < 1536u) {
		return;
	}

	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_Q6_ROW16_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_Q6_ROW16_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_Q6_ROW16_ROWS_PER_BLOCK + row_lane;

	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.weight_pointer));
	const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.scale_pointer));
	const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.bias_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float sum = 0.0f;
	if (row < args.rows) {
		sum = rocm_mlx_q4_projection_row_sum(input, weights, scales, biases, row, args.cols, args.group_size, args.bits, col_lane, ROCM_MLX_Q4_PROJECTION_Q6_ROW16_THREADS_PER_ROW);
	}
	sum = rocm_mlx_q4_projection_q6_row16_reduce(sum);
	if (col_lane == 0 && row < args.rows) {
		output[row] = sum;
	}
}

extern "C" __global__ void rocm_mlx_q4_projection_q6_row32(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_projection_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_projection_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_projection_args(args) || args.bits != 6u || args.group_size != 64u || args.cols < 1536u || args.cols > 2048u) {
		return;
	}

	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_Q6_ROW32_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_Q6_ROW32_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_Q6_ROW32_ROWS_PER_BLOCK + row_lane;

	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.weight_pointer));
	const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.scale_pointer));
	const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.bias_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float sum = 0.0f;
	if (row < args.rows) {
		sum = rocm_mlx_q4_projection_row_sum(input, weights, scales, biases, row, args.cols, args.group_size, args.bits, col_lane, ROCM_MLX_Q4_PROJECTION_Q6_ROW32_THREADS_PER_ROW);
	}
	sum = rocm_mlx_q4_projection_q6_row32_reduce(sum);
	if (col_lane == 0 && row < args.rows) {
		output[row] = sum;
	}
}

extern "C" __global__ void rocm_mlx_q4_projection_q6_row64(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_projection_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_projection_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_projection_args(args) || args.bits != 6u || args.group_size != 64u || args.cols < 1536u || args.cols > 2048u) {
		return;
	}

	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_Q6_ROW64_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_Q6_ROW64_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_Q6_ROW64_ROWS_PER_BLOCK + row_lane;

	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.weight_pointer));
	const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.scale_pointer));
	const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.bias_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float sum = 0.0f;
	if (row < args.rows) {
		sum = rocm_mlx_q4_projection_row_sum(input, weights, scales, biases, row, args.cols, args.group_size, args.bits, col_lane, ROCM_MLX_Q4_PROJECTION_Q6_ROW64_THREADS_PER_ROW);
	}
	sum = rocm_mlx_q4_projection_q6_row64_reduce(sum);
	if (col_lane == 0 && row < args.rows) {
		output[row] = sum;
	}
}

extern "C" __global__ void rocm_mlx_q4_projection_batch(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_projection_batch_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_projection_batch_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_projection_batch_args(args)) {
		return;
	}

	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE || blockIdx.y >= args.batch) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_ROWS_PER_BLOCK + row_lane;
	const uint32_t batch_base = blockIdx.y * ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK;

	const float *input_base = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.weight_pointer));
	const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.scale_pointer));
	const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.bias_pointer));
	float *output_base = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float sums[ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK] = {};
	if (row < args.rows) {
		const uint64_t packed_per_row = rocm_mlx_affine_packed_per_row(args.cols, args.bits);
		const uint32_t groups_per_row = args.cols / args.group_size;
		if (args.bits == 4u && (args.group_size & 7u) == 0u) {
			const uint32_t packed_per_group = args.group_size >> 3u;
			for (uint32_t group_col = col_lane; group_col < groups_per_row; group_col += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
				const uint32_t group = row * groups_per_row + group_col;
				const float scale = rocm_bfloat16_to_float(scales[group]);
				const float bias = rocm_bfloat16_to_float(biases[group]);
				const uint32_t first_packed = group_col * packed_per_group;
				float q_dot_sums[ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK] = {};
				float input_sums[ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK] = {};
				for (uint32_t group_packed = 0; group_packed < packed_per_group; ++group_packed) {
					const uint32_t packed = first_packed + group_packed;
					const uint32_t word = weights[static_cast<uint64_t>(row) * packed_per_row + packed];
					const uint32_t col = packed << 3;
					const float q0 = static_cast<float>(word & 0x0fu);
					const float q1 = static_cast<float>((word >> 4) & 0x0fu);
					const float q2 = static_cast<float>((word >> 8) & 0x0fu);
					const float q3 = static_cast<float>((word >> 12) & 0x0fu);
					const float q4 = static_cast<float>((word >> 16) & 0x0fu);
					const float q5 = static_cast<float>((word >> 20) & 0x0fu);
					const float q6 = static_cast<float>((word >> 24) & 0x0fu);
					const float q7 = static_cast<float>((word >> 28) & 0x0fu);
#pragma unroll
					for (uint32_t token_lane = 0; token_lane < ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK; ++token_lane) {
						const uint32_t batch = batch_base + token_lane;
						if (batch >= args.batch) {
							continue;
						}
						const float *input = input_base + batch * args.cols;
						const float in0 = input[col];
						const float in1 = input[col + 1u];
						const float in2 = input[col + 2u];
						const float in3 = input[col + 3u];
						const float in4 = input[col + 4u];
						const float in5 = input[col + 5u];
						const float in6 = input[col + 6u];
						const float in7 = input[col + 7u];
						input_sums[token_lane] += in0 + in1 + in2 + in3 + in4 + in5 + in6 + in7;
						q_dot_sums[token_lane] +=
							q0 * in0 +
							q1 * in1 +
							q2 * in2 +
							q3 * in3 +
							q4 * in4 +
							q5 * in5 +
							q6 * in6 +
							q7 * in7;
					}
				}
#pragma unroll
				for (uint32_t token_lane = 0; token_lane < ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK; ++token_lane) {
					sums[token_lane] += scale * q_dot_sums[token_lane] + bias * input_sums[token_lane];
				}
			}
		} else if (args.bits == 6u && args.group_size == 64u) {
			const uint64_t row_packed_base = static_cast<uint64_t>(row) * packed_per_row;
			for (uint32_t group_col = col_lane; group_col < groups_per_row; group_col += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
				const uint32_t group = row * groups_per_row + group_col;
				const float scale = rocm_bfloat16_to_float(scales[group]);
				const float bias = rocm_bfloat16_to_float(biases[group]);
				const uint32_t first_packed = group_col * 12u;
				const uint32_t first_col = group_col << 6u;
				float q_dot_sums[ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK] = {};
				float input_sums[ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK] = {};
				for (uint32_t block = 0; block < 4u; ++block) {
					rocm_mlx_affine_q6_16_batch_dot(input_base, weights, row_packed_base, first_packed + block * 3u, first_col + block * 16u, args.cols, batch_base, args.batch, q_dot_sums, input_sums);
				}
#pragma unroll
				for (uint32_t token_lane = 0; token_lane < ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK; ++token_lane) {
					sums[token_lane] += scale * q_dot_sums[token_lane] + bias * input_sums[token_lane];
				}
			}
		} else {
			for (uint32_t col = col_lane; col < args.cols; col += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
				const uint32_t q = rocm_mlx_affine_quantized_value(weights, static_cast<uint64_t>(row) * packed_per_row, col, args.bits);
				const uint32_t group = row * groups_per_row + col / args.group_size;
				const float scale = rocm_bfloat16_to_float(scales[group]);
				const float bias = rocm_bfloat16_to_float(biases[group]);
#pragma unroll
				for (uint32_t token_lane = 0; token_lane < ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK; ++token_lane) {
					const uint32_t batch = batch_base + token_lane;
					if (batch < args.batch) {
						const float *input = input_base + batch * args.cols;
						sums[token_lane] += input[col] * (static_cast<float>(q) * scale + bias);
					}
				}
			}
		}
	}
#pragma unroll
	for (uint32_t token_lane = 0; token_lane < ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK; ++token_lane) {
		const uint32_t batch = batch_base + token_lane;
		const float sum = rocm_mlx_q4_row_reduce(sums[token_lane]);
		if (col_lane == 0 && row < args.rows && batch < args.batch) {
			float *output = output_base + batch * args.rows;
			output[row] = sum;
		}
	}
}

extern "C" __global__ void rocm_mlx_q4_projection_batch_q4_g64_tokens16(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_projection_batch_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_projection_batch_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_projection_batch_args(args) || args.bits != 4u || args.group_size != 64u) {
		return;
	}
	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE ||
		blockIdx.y * ROCM_MLX_Q4_PROJECTION_BATCH_WIDE_TOKENS_PER_BLOCK >= args.batch) {
		return;
	}

	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_ROWS_PER_BLOCK + row_lane;
	const uint32_t batch_base = blockIdx.y * ROCM_MLX_Q4_PROJECTION_BATCH_WIDE_TOKENS_PER_BLOCK;
	const float *input_base = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.weight_pointer));
	const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.scale_pointer));
	const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.bias_pointer));
	float *output_base = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float sums[ROCM_MLX_Q4_PROJECTION_BATCH_WIDE_TOKENS_PER_BLOCK] = {};
	if (row < args.rows) {
		const uint64_t packed_per_row = rocm_mlx_affine_packed_per_row(args.cols, args.bits);
		const uint32_t groups_per_row = args.cols >> 6u;
		constexpr uint32_t packed_per_group = 8u;
		for (uint32_t group_col = col_lane; group_col < groups_per_row; group_col += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
			const uint32_t group = row * groups_per_row + group_col;
			const float scale = rocm_bfloat16_to_float(scales[group]);
			const float bias = rocm_bfloat16_to_float(biases[group]);
			const uint32_t first_packed = group_col * packed_per_group;
			float q_dot_sums[ROCM_MLX_Q4_PROJECTION_BATCH_WIDE_TOKENS_PER_BLOCK] = {};
			float input_sums[ROCM_MLX_Q4_PROJECTION_BATCH_WIDE_TOKENS_PER_BLOCK] = {};
			for (uint32_t group_packed = 0; group_packed < packed_per_group; ++group_packed) {
				const uint32_t packed = first_packed + group_packed;
				const uint32_t word = weights[static_cast<uint64_t>(row) * packed_per_row + packed];
				const uint32_t col = packed << 3u;
				const float q0 = static_cast<float>(word & 0x0fu);
				const float q1 = static_cast<float>((word >> 4u) & 0x0fu);
				const float q2 = static_cast<float>((word >> 8u) & 0x0fu);
				const float q3 = static_cast<float>((word >> 12u) & 0x0fu);
				const float q4 = static_cast<float>((word >> 16u) & 0x0fu);
				const float q5 = static_cast<float>((word >> 20u) & 0x0fu);
				const float q6 = static_cast<float>((word >> 24u) & 0x0fu);
				const float q7 = static_cast<float>((word >> 28u) & 0x0fu);
#pragma unroll
				for (uint32_t token_lane = 0; token_lane < ROCM_MLX_Q4_PROJECTION_BATCH_WIDE_TOKENS_PER_BLOCK; ++token_lane) {
					const uint32_t batch = batch_base + token_lane;
					if (batch >= args.batch) {
						continue;
					}
					const float *input = input_base + batch * args.cols;
					const float in0 = input[col];
					const float in1 = input[col + 1u];
					const float in2 = input[col + 2u];
					const float in3 = input[col + 3u];
					const float in4 = input[col + 4u];
					const float in5 = input[col + 5u];
					const float in6 = input[col + 6u];
					const float in7 = input[col + 7u];
					input_sums[token_lane] += in0 + in1 + in2 + in3 + in4 + in5 + in6 + in7;
					q_dot_sums[token_lane] +=
						q0 * in0 +
						q1 * in1 +
						q2 * in2 +
						q3 * in3 +
						q4 * in4 +
						q5 * in5 +
						q6 * in6 +
						q7 * in7;
				}
			}
#pragma unroll
			for (uint32_t token_lane = 0; token_lane < ROCM_MLX_Q4_PROJECTION_BATCH_WIDE_TOKENS_PER_BLOCK; ++token_lane) {
				sums[token_lane] += scale * q_dot_sums[token_lane] + bias * input_sums[token_lane];
			}
		}
	}
#pragma unroll
	for (uint32_t token_lane = 0; token_lane < ROCM_MLX_Q4_PROJECTION_BATCH_WIDE_TOKENS_PER_BLOCK; ++token_lane) {
		const uint32_t batch = batch_base + token_lane;
		const float sum = rocm_mlx_q4_row_reduce(sums[token_lane]);
		if (col_lane == 0 && row < args.rows && batch < args.batch) {
			float *output = output_base + batch * args.rows;
			output[row] = sum;
		}
	}
}

extern "C" __global__ void rocm_mlx_q4_projection_batch_q8_g64_row16_tokens16(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_projection_batch_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_projection_batch_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_projection_batch_args(args) || args.bits != 8u || args.group_size != 64u) {
		return;
	}
	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE ||
		blockIdx.y * ROCM_MLX_Q4_PROJECTION_BATCH_WIDE_TOKENS_PER_BLOCK >= args.batch) {
		return;
	}

	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_ROW16_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_ROW16_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_ROW16_ROWS_PER_BLOCK + row_lane;
	const uint32_t batch_base = blockIdx.y * ROCM_MLX_Q4_PROJECTION_BATCH_WIDE_TOKENS_PER_BLOCK;
	const float *input_base = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.weight_pointer));
	const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.scale_pointer));
	const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.bias_pointer));
	float *output_base = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float sums[ROCM_MLX_Q4_PROJECTION_BATCH_WIDE_TOKENS_PER_BLOCK] = {};
	if (row < args.rows) {
		const uint64_t packed_per_row = rocm_mlx_affine_packed_per_row(args.cols, args.bits);
		const uint32_t groups_per_row = args.cols >> 6u;
		const uint64_t row_packed_base = static_cast<uint64_t>(row) * packed_per_row;
		for (uint32_t packed = col_lane; packed < packed_per_row; packed += ROCM_MLX_Q4_PROJECTION_ROW16_THREADS_PER_ROW) {
			const uint32_t word = weights[row_packed_base + packed];
			const uint32_t col = packed << 2u;
			const uint32_t group = row * groups_per_row + (packed >> 4u);
			const float scale = rocm_bfloat16_to_float(scales[group]);
			const float bias = rocm_bfloat16_to_float(biases[group]);
			const float value0 = static_cast<float>(word & 0xffu) * scale + bias;
			const float value1 = static_cast<float>((word >> 8u) & 0xffu) * scale + bias;
			const float value2 = static_cast<float>((word >> 16u) & 0xffu) * scale + bias;
			const float value3 = static_cast<float>(word >> 24u) * scale + bias;
#pragma unroll
			for (uint32_t token_lane = 0; token_lane < ROCM_MLX_Q4_PROJECTION_BATCH_WIDE_TOKENS_PER_BLOCK; ++token_lane) {
				const uint32_t batch = batch_base + token_lane;
				if (batch < args.batch) {
					const float *input = input_base + batch * args.cols;
					sums[token_lane] = fmaf(input[col], value0, sums[token_lane]);
					sums[token_lane] = fmaf(input[col + 1u], value1, sums[token_lane]);
					sums[token_lane] = fmaf(input[col + 2u], value2, sums[token_lane]);
					sums[token_lane] = fmaf(input[col + 3u], value3, sums[token_lane]);
				}
			}
		}
	}
#pragma unroll
	for (uint32_t token_lane = 0; token_lane < ROCM_MLX_Q4_PROJECTION_BATCH_WIDE_TOKENS_PER_BLOCK; ++token_lane) {
		const uint32_t batch = batch_base + token_lane;
		const float sum = rocm_mlx_q4_projection_row16_reduce(sums[token_lane]);
		if (col_lane == 0 && row < args.rows && batch < args.batch) {
			float *output = output_base + batch * args.rows;
			output[row] = sum;
		}
	}
}

extern "C" __global__ void rocm_mlx_q4_projection_batch_q6_row16(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_projection_batch_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_projection_batch_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_projection_batch_args(args) || args.bits != 6u || args.group_size != 64u || args.cols < 1536u) {
		return;
	}

	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE || blockIdx.y >= args.batch) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_Q6_ROW16_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_Q6_ROW16_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_Q6_ROW16_ROWS_PER_BLOCK + row_lane;
	const uint32_t batch_base = blockIdx.y * ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK;

	const float *input_base = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.weight_pointer));
	const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.scale_pointer));
	const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.bias_pointer));
	float *output_base = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float sums[ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK] = {};
	if (row < args.rows) {
		const uint64_t packed_per_row = rocm_mlx_affine_packed_per_row(args.cols, args.bits);
		const uint32_t groups_per_row = args.cols >> 6u;
		const uint64_t row_packed_base = static_cast<uint64_t>(row) * packed_per_row;
		const uint32_t row_group_base = row * groups_per_row;
		for (uint32_t group_col = col_lane; group_col < groups_per_row; group_col += ROCM_MLX_Q4_PROJECTION_Q6_ROW16_THREADS_PER_ROW) {
			const uint32_t group = row_group_base + group_col;
			const float scale = rocm_bfloat16_to_float(scales[group]);
			const float bias = rocm_bfloat16_to_float(biases[group]);
			const uint32_t first_packed = group_col * 12u;
			const uint32_t first_col = group_col << 6u;
			float q_dot_sums[ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK] = {};
			float input_sums[ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK] = {};
			for (uint32_t block = 0; block < 4u; ++block) {
				rocm_mlx_affine_q6_16_batch_dot(input_base, weights, row_packed_base, first_packed + block * 3u, first_col + block * 16u, args.cols, batch_base, args.batch, q_dot_sums, input_sums);
			}
#pragma unroll
			for (uint32_t token_lane = 0; token_lane < ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK; ++token_lane) {
				sums[token_lane] += scale * q_dot_sums[token_lane] + bias * input_sums[token_lane];
			}
		}
	}
#pragma unroll
	for (uint32_t token_lane = 0; token_lane < ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK; ++token_lane) {
		const uint32_t batch = batch_base + token_lane;
		const float sum = rocm_mlx_q4_projection_q6_row16_reduce(sums[token_lane]);
		if (col_lane == 0 && row < args.rows && batch < args.batch) {
			float *output = output_base + batch * args.rows;
			output[row] = sum;
		}
	}
}

extern "C" __global__ void rocm_mlx_q4_triple_projection(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_triple_projection_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_triple_projection_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_triple_projection_args(args)) {
		return;
	}

	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_ROWS_PER_BLOCK + row_lane;
	const uint32_t total_rows = args.first_rows + args.second_rows + args.third_rows;
	if (row >= total_rows) {
		return;
	}

	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *weights = nullptr;
	const uint16_t *scales = nullptr;
	const uint16_t *biases = nullptr;
	uint32_t local_row = row;
	if (row < args.first_rows) {
		weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.first_weight_pointer));
		scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.first_scale_pointer));
		biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.first_bias_pointer));
	} else if (row < args.first_rows + args.second_rows) {
		local_row = row - args.first_rows;
		weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.second_weight_pointer));
		scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.second_scale_pointer));
		biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.second_bias_pointer));
	} else {
		local_row = row - args.first_rows - args.second_rows;
		weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.third_weight_pointer));
		scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.third_scale_pointer));
		biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.third_bias_pointer));
	}
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float sum = rocm_mlx_q4_projection_row_sum(input, weights, scales, biases, local_row, args.cols, args.group_size, args.bits, col_lane);
	sum = rocm_mlx_q4_row_reduce(sum);
	if (col_lane == 0) {
		output[row] = sum;
	}
}

extern "C" __global__ void rocm_mlx_q4_triple_projection_q6_row16(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_triple_projection_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_triple_projection_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_triple_projection_args(args) || args.bits != 6u || args.group_size != 64u || args.cols < 1536u) {
		return;
	}

	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_Q6_ROW16_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_Q6_ROW16_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_Q6_ROW16_ROWS_PER_BLOCK + row_lane;
	const uint32_t total_rows = args.first_rows + args.second_rows + args.third_rows;
	if (row >= total_rows) {
		return;
	}

	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *weights = nullptr;
	const uint16_t *scales = nullptr;
	const uint16_t *biases = nullptr;
	uint32_t local_row = row;
	if (row < args.first_rows) {
		weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.first_weight_pointer));
		scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.first_scale_pointer));
		biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.first_bias_pointer));
	} else if (row < args.first_rows + args.second_rows) {
		local_row = row - args.first_rows;
		weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.second_weight_pointer));
		scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.second_scale_pointer));
		biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.second_bias_pointer));
	} else {
		local_row = row - args.first_rows - args.second_rows;
		weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.third_weight_pointer));
		scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.third_scale_pointer));
		biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.third_bias_pointer));
	}
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float sum = rocm_mlx_q4_projection_row_sum(input, weights, scales, biases, local_row, args.cols, args.group_size, args.bits, col_lane, ROCM_MLX_Q4_PROJECTION_Q6_ROW16_THREADS_PER_ROW);
	sum = rocm_mlx_q4_projection_q6_row16_reduce(sum);
	if (col_lane == 0) {
		output[row] = sum;
	}
}

extern "C" __global__ void rocm_mlx_q4_triple_projection_q6_row64(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_triple_projection_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_triple_projection_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_triple_projection_args(args) || args.bits != 6u || args.group_size != 64u || args.cols != 1536u) {
		return;
	}

	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_Q6_ROW64_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_Q6_ROW64_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_Q6_ROW64_ROWS_PER_BLOCK + row_lane;
	const uint32_t total_rows = args.first_rows + args.second_rows + args.third_rows;
	if (row >= total_rows) {
		return;
	}

	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *weights = nullptr;
	const uint16_t *scales = nullptr;
	const uint16_t *biases = nullptr;
	uint32_t local_row = row;
	if (row < args.first_rows) {
		weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.first_weight_pointer));
		scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.first_scale_pointer));
		biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.first_bias_pointer));
	} else if (row < args.first_rows + args.second_rows) {
		local_row = row - args.first_rows;
		weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.second_weight_pointer));
		scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.second_scale_pointer));
		biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.second_bias_pointer));
	} else {
		local_row = row - args.first_rows - args.second_rows;
		weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.third_weight_pointer));
		scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.third_scale_pointer));
		biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.third_bias_pointer));
	}
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float sum = rocm_mlx_q4_projection_row_sum(input, weights, scales, biases, local_row, args.cols, args.group_size, args.bits, col_lane, ROCM_MLX_Q4_PROJECTION_Q6_ROW64_THREADS_PER_ROW);
	sum = rocm_mlx_q4_projection_q6_row64_reduce(sum);
	if (col_lane == 0) {
		output[row] = sum;
	}
}

extern "C" __global__ void rocm_mlx_q4_pair_projection(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_triple_projection_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_triple_projection_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_triple_projection_args(args) || args.third_rows != 0) {
		return;
	}

	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_ROWS_PER_BLOCK + row_lane;
	const uint32_t total_rows = args.first_rows + args.second_rows;
	if (row >= total_rows) {
		return;
	}

	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.first_weight_pointer));
	const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.first_scale_pointer));
	const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.first_bias_pointer));
	uint32_t local_row = row;
	if (row >= args.first_rows) {
		local_row = row - args.first_rows;
		weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.second_weight_pointer));
		scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.second_scale_pointer));
		biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.second_bias_pointer));
	}
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float sum = rocm_mlx_q4_projection_row_sum(input, weights, scales, biases, local_row, args.cols, args.group_size, args.bits, col_lane);
	sum = rocm_mlx_q4_row_reduce(sum);
	if (col_lane == 0) {
		output[row] = sum;
	}
}

extern "C" __global__ void rocm_mlx_q4_projection_greedy(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_projection_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_projection_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_projection_greedy_args(args)) {
		return;
	}

	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_GREEDY_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_GREEDY_THREADS_PER_ROW;
	const uint32_t row_stride = gridDim.x * ROCM_MLX_Q4_PROJECTION_GREEDY_ROWS_PER_BLOCK;
	__shared__ unsigned long long block_best[ROCM_MLX_Q4_PROJECTION_GREEDY_ROWS_PER_BLOCK];

	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.weight_pointer));
	const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.scale_pointer));
	const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.bias_pointer));
	const int32_t *suppress_tokens = reinterpret_cast<const int32_t *>(static_cast<uintptr_t>(args.suppress_pointer));
	unsigned long long persistent_best = 0;
	for (uint32_t row_base = blockIdx.x * ROCM_MLX_Q4_PROJECTION_GREEDY_ROWS_PER_BLOCK; row_base < args.rows; row_base += row_stride) {
		const uint32_t row = row_base + row_lane;
		float sum = 0.0f;
		if (row < args.rows) {
			sum = rocm_mlx_q4_projection_row_sum(input, weights, scales, biases, row, args.cols, args.group_size, args.bits, col_lane, ROCM_MLX_Q4_PROJECTION_GREEDY_THREADS_PER_ROW);
		}
		sum = rocm_mlx_q4_greedy_row_reduce(sum);
		if (col_lane == 0) {
			unsigned long long packed = 0;
			if (row < args.rows && isfinite(sum) && !rocm_mlx_q4_token_suppressed(row, suppress_tokens, args.suppress_count)) {
				packed = static_cast<unsigned long long>(rocm_pack_score_index(sum, row));
			}
			if (packed > persistent_best) {
				persistent_best = packed;
			}
		}
	}
	if (col_lane == 0) {
		block_best[row_lane] = persistent_best;
	}
	__syncthreads();
	if (threadIdx.x == 0) {
		unsigned long long best_value = block_best[0];
		for (uint32_t index = 1u; index < ROCM_MLX_Q4_PROJECTION_GREEDY_ROWS_PER_BLOCK; ++index) {
			const unsigned long long other = block_best[index];
			if (other > best_value) {
				best_value = other;
			}
		}
		if (best_value != 0) {
			unsigned long long *best = reinterpret_cast<unsigned long long *>(static_cast<uintptr_t>(args.output_pointer));
			atomicMax(best, best_value);
		}
	}
}

extern "C" __global__ void rocm_mlx_q4_projection_greedy_q6_row64(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_projection_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_projection_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_projection_greedy_args(args) || args.bits != 6u || args.group_size != 64u || args.cols < 1536u) {
		return;
	}

	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_ROWS_PER_BLOCK + row_lane;
	__shared__ unsigned long long block_best[ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_ROWS_PER_BLOCK];

	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.weight_pointer));
	const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.scale_pointer));
	const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.bias_pointer));
	const int32_t *suppress_tokens = reinterpret_cast<const int32_t *>(static_cast<uintptr_t>(args.suppress_pointer));
	float sum = 0.0f;
	if (row < args.rows) {
		sum = rocm_mlx_q4_projection_row_sum(input, weights, scales, biases, row, args.cols, args.group_size, args.bits, col_lane, ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_THREADS_PER_ROW);
	}
	sum = rocm_mlx_q4_greedy_q6_row_reduce(sum);
	if (col_lane == 0) {
		unsigned long long packed = 0;
		if (row < args.rows && isfinite(sum) && !rocm_mlx_q4_token_suppressed(row, suppress_tokens, args.suppress_count)) {
			packed = static_cast<unsigned long long>(rocm_pack_score_index(sum, row));
		}
		block_best[row_lane] = packed;
	}
	__syncthreads();
	if (threadIdx.x == 0) {
		unsigned long long best_value = block_best[0];
		for (uint32_t index = 1u; index < ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_ROWS_PER_BLOCK; ++index) {
			const unsigned long long other = block_best[index];
			if (other > best_value) {
				best_value = other;
			}
		}
		if (best_value != 0) {
			unsigned long long *best = reinterpret_cast<unsigned long long *>(static_cast<uintptr_t>(args.output_pointer));
			atomicMax(best, best_value);
		}
	}
}

extern "C" __global__ void rocm_mlx_q4_projection_greedy_batch(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_projection_greedy_batch_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_projection_greedy_batch_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_projection_greedy_batch_args(args)) {
		return;
	}

	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t batch_index = blockIdx.y;
	if (batch_index >= args.batch) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_GREEDY_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_GREEDY_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_GREEDY_ROWS_PER_BLOCK + row_lane;
	__shared__ unsigned long long block_best[ROCM_MLX_Q4_PROJECTION_GREEDY_ROWS_PER_BLOCK];

	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer)) + static_cast<uint64_t>(batch_index) * args.cols;
	const uint32_t *weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.weight_pointer));
	const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.scale_pointer));
	const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.bias_pointer));
	const int32_t *suppress_tokens = reinterpret_cast<const int32_t *>(static_cast<uintptr_t>(args.suppress_pointer));
	float sum = 0.0f;
	if (row < args.rows) {
		sum = rocm_mlx_q4_projection_row_sum(input, weights, scales, biases, row, args.cols, args.group_size, args.bits, col_lane, ROCM_MLX_Q4_PROJECTION_GREEDY_THREADS_PER_ROW);
	}
	sum = rocm_mlx_q4_greedy_row_reduce(sum);
	if (col_lane == 0) {
		unsigned long long packed = 0;
		if (row < args.rows && isfinite(sum) && !rocm_mlx_q4_token_suppressed(row, suppress_tokens, args.suppress_count)) {
			packed = static_cast<unsigned long long>(rocm_pack_score_index(sum, row));
		}
		block_best[row_lane] = packed;
	}
	__syncthreads();
	if (threadIdx.x == 0) {
		unsigned long long best_value = block_best[0];
		for (uint32_t index = 1u; index < ROCM_MLX_Q4_PROJECTION_GREEDY_ROWS_PER_BLOCK; ++index) {
			const unsigned long long other = block_best[index];
			if (other > best_value) {
				best_value = other;
			}
		}
		if (best_value != 0) {
			unsigned long long *best = reinterpret_cast<unsigned long long *>(static_cast<uintptr_t>(args.output_pointer));
			atomicMax(&best[batch_index], best_value);
		}
	}
}

extern "C" __global__ void rocm_mlx_q4_projection_greedy_batch_q6_row64(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_projection_greedy_batch_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_projection_greedy_batch_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_projection_greedy_batch_args(args) || args.bits != 6u || args.group_size != 64u || args.cols < 1536u) {
		return;
	}

	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t batch_index = blockIdx.y;
	if (batch_index >= args.batch) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_ROWS_PER_BLOCK + row_lane;
	__shared__ unsigned long long block_best[ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_ROWS_PER_BLOCK];

	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer)) + static_cast<uint64_t>(batch_index) * args.cols;
	const uint32_t *weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.weight_pointer));
	const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.scale_pointer));
	const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.bias_pointer));
	const int32_t *suppress_tokens = reinterpret_cast<const int32_t *>(static_cast<uintptr_t>(args.suppress_pointer));
	float sum = 0.0f;
	if (row < args.rows) {
		sum = rocm_mlx_q4_projection_row_sum(input, weights, scales, biases, row, args.cols, args.group_size, args.bits, col_lane, ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_THREADS_PER_ROW);
	}
	sum = rocm_mlx_q4_greedy_q6_row_reduce(sum);
	if (col_lane == 0) {
		unsigned long long packed = 0;
		if (row < args.rows && isfinite(sum) && !rocm_mlx_q4_token_suppressed(row, suppress_tokens, args.suppress_count)) {
			packed = static_cast<unsigned long long>(rocm_pack_score_index(sum, row));
		}
		block_best[row_lane] = packed;
	}
	__syncthreads();
	if (threadIdx.x == 0) {
		unsigned long long best_value = block_best[0];
		for (uint32_t index = 1u; index < ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_ROWS_PER_BLOCK; ++index) {
			const unsigned long long other = block_best[index];
			if (other > best_value) {
				best_value = other;
			}
		}
		if (best_value != 0) {
			unsigned long long *best = reinterpret_cast<unsigned long long *>(static_cast<uintptr_t>(args.output_pointer));
			atomicMax(&best[batch_index], best_value);
		}
	}
}

extern "C" __global__ void rocm_mlx_q4_projection_scores(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_projection_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_projection_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_projection_scores_args(args)) {
		return;
	}

	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_GREEDY_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_GREEDY_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_GREEDY_ROWS_PER_BLOCK + row_lane;

	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.weight_pointer));
	const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.scale_pointer));
	const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.bias_pointer));
	const int32_t *suppress_tokens = reinterpret_cast<const int32_t *>(static_cast<uintptr_t>(args.suppress_pointer));
	unsigned long long *scores = reinterpret_cast<unsigned long long *>(static_cast<uintptr_t>(args.output_pointer));
	float sum = 0.0f;
	if (row < args.rows) {
		sum = rocm_mlx_q4_projection_row_sum(input, weights, scales, biases, row, args.cols, args.group_size, args.bits, col_lane, ROCM_MLX_Q4_PROJECTION_GREEDY_THREADS_PER_ROW);
	}
	sum = rocm_mlx_q4_greedy_row_reduce(sum);
	if (col_lane == 0 && row < args.rows) {
		unsigned long long packed = 0;
		if (isfinite(sum) && !rocm_mlx_q4_token_suppressed(row, suppress_tokens, args.suppress_count)) {
			packed = static_cast<unsigned long long>(rocm_pack_score_index(sum, row));
		}
		scores[row] = packed;
	}
}

extern "C" __global__ void rocm_mlx_q4_projection_scores_q6_row64(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_projection_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_projection_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_projection_scores_args(args) || args.bits != 6u || args.group_size != 64u || args.cols < 1536u) {
		return;
	}

	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_ROWS_PER_BLOCK + row_lane;

	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.weight_pointer));
	const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.scale_pointer));
	const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.bias_pointer));
	const int32_t *suppress_tokens = reinterpret_cast<const int32_t *>(static_cast<uintptr_t>(args.suppress_pointer));
	unsigned long long *scores = reinterpret_cast<unsigned long long *>(static_cast<uintptr_t>(args.output_pointer));
	float sum = 0.0f;
	if (row < args.rows) {
		sum = rocm_mlx_q4_projection_row_sum(input, weights, scales, biases, row, args.cols, args.group_size, args.bits, col_lane, ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_THREADS_PER_ROW);
	}
	sum = rocm_mlx_q4_greedy_q6_row_reduce(sum);
	if (col_lane == 0 && row < args.rows) {
		unsigned long long packed = 0;
		if (isfinite(sum) && !rocm_mlx_q4_token_suppressed(row, suppress_tokens, args.suppress_count)) {
			packed = static_cast<unsigned long long>(rocm_pack_score_index(sum, row));
		}
		scores[row] = packed;
	}
}

extern "C" __global__ void rocm_mlx_q4_projection_selected_greedy(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_projection_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_projection_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_projection_selected_greedy_args(args)) {
		return;
	}

	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_GREEDY_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_GREEDY_THREADS_PER_ROW;
	const uint32_t selected_index = blockIdx.x * ROCM_MLX_Q4_PROJECTION_GREEDY_ROWS_PER_BLOCK + row_lane;
	__shared__ unsigned long long block_best[ROCM_MLX_Q4_PROJECTION_GREEDY_ROWS_PER_BLOCK];

	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.weight_pointer));
	const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.scale_pointer));
	const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.bias_pointer));
	const int32_t *selected_rows = reinterpret_cast<const int32_t *>(static_cast<uintptr_t>(args.suppress_pointer));
	float sum = 0.0f;
	uint32_t row = args.rows;
	if (selected_index < args.suppress_count) {
		const int32_t selected = selected_rows[selected_index];
		if (selected >= 0 && static_cast<uint32_t>(selected) < args.rows) {
			row = static_cast<uint32_t>(selected);
			sum = rocm_mlx_q4_projection_row_sum(input, weights, scales, biases, row, args.cols, args.group_size, args.bits, col_lane, ROCM_MLX_Q4_PROJECTION_GREEDY_THREADS_PER_ROW);
		}
	}
	sum = rocm_mlx_q4_greedy_row_reduce(sum);
	if (col_lane == 0) {
		unsigned long long packed = 0;
		if (row < args.rows && isfinite(sum)) {
			packed = static_cast<unsigned long long>(rocm_pack_score_index(sum, row));
		}
		block_best[row_lane] = packed;
	}
	__syncthreads();
	if (threadIdx.x == 0) {
		unsigned long long best_value = block_best[0];
		for (uint32_t index = 1u; index < ROCM_MLX_Q4_PROJECTION_GREEDY_ROWS_PER_BLOCK; ++index) {
			const unsigned long long other = block_best[index];
			if (other > best_value) {
				best_value = other;
			}
		}
		if (best_value != 0) {
			unsigned long long *best = reinterpret_cast<unsigned long long *>(static_cast<uintptr_t>(args.output_pointer));
			atomicMax(best, best_value);
		}
	}
}

extern "C" __global__ void rocm_mlx_q4_projection_selected_greedy_q6_row64(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_projection_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_projection_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_projection_selected_greedy_args(args) || args.bits != 6u || args.group_size != 64u || args.cols < 1536u) {
		return;
	}

	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_THREADS_PER_ROW;
	const uint32_t selected_index = blockIdx.x * ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_ROWS_PER_BLOCK + row_lane;
	__shared__ unsigned long long block_best[ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_ROWS_PER_BLOCK];

	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.weight_pointer));
	const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.scale_pointer));
	const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.bias_pointer));
	const int32_t *selected_rows = reinterpret_cast<const int32_t *>(static_cast<uintptr_t>(args.suppress_pointer));
	float sum = 0.0f;
	uint32_t row = args.rows;
	if (selected_index < args.suppress_count) {
		const int32_t selected = selected_rows[selected_index];
		if (selected >= 0 && static_cast<uint32_t>(selected) < args.rows) {
			row = static_cast<uint32_t>(selected);
			sum = rocm_mlx_q4_projection_row_sum(input, weights, scales, biases, row, args.cols, args.group_size, args.bits, col_lane, ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_THREADS_PER_ROW);
		}
	}
	sum = rocm_mlx_q4_greedy_q6_row_reduce(sum);
	if (col_lane == 0) {
		unsigned long long packed = 0;
		if (row < args.rows && isfinite(sum)) {
			packed = static_cast<unsigned long long>(rocm_pack_score_index(sum, row));
		}
		block_best[row_lane] = packed;
	}
	__syncthreads();
	if (threadIdx.x == 0) {
		unsigned long long best_value = block_best[0];
		for (uint32_t index = 1u; index < ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_ROWS_PER_BLOCK; ++index) {
			const unsigned long long other = block_best[index];
			if (other > best_value) {
				best_value = other;
			}
		}
		if (best_value != 0) {
			unsigned long long *best = reinterpret_cast<unsigned long long *>(static_cast<uintptr_t>(args.output_pointer));
			atomicMax(best, best_value);
		}
	}
}

extern "C" __global__ void rocm_ordered_embedding_candidates(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_ordered_embedding_candidates_launch_args &args = *reinterpret_cast<const rocm_ordered_embedding_candidates_launch_args *>(packet);
	if (!rocm_valid_ordered_embedding_candidates_args(args) || blockDim.x != ROCM_ORDERED_EMBEDDING_CANDIDATES_BLOCK_SIZE) {
		return;
	}

	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= args.output_count) {
		return;
	}
	const uint32_t centroid_rank = index / args.tokens_per_centroid;
	const uint32_t token_offset = index - centroid_rank * args.tokens_per_centroid;
	const unsigned long long *topk = reinterpret_cast<const unsigned long long *>(static_cast<uintptr_t>(args.topk_pointer));
	const int32_t *suppress_tokens = reinterpret_cast<const int32_t *>(static_cast<uintptr_t>(args.suppress_pointer));
	int32_t *output = reinterpret_cast<int32_t *>(static_cast<uintptr_t>(args.output_pointer));

	int32_t selected_token = -1;
	const unsigned long long packed_centroid = topk[centroid_rank];
	if (packed_centroid != 0) {
		const uint32_t centroid = ~static_cast<uint32_t>(packed_centroid);
		if (centroid < args.num_centroids) {
			const uint32_t ordering_index = centroid * args.tokens_per_centroid + token_offset;
			int64_t ordered = -1;
			if (args.token_ordering_element_bytes == sizeof(int32_t)) {
				const int32_t *ordering = reinterpret_cast<const int32_t *>(static_cast<uintptr_t>(args.token_ordering_pointer));
				ordered = ordering[ordering_index];
			} else {
				const int64_t *ordering = reinterpret_cast<const int64_t *>(static_cast<uintptr_t>(args.token_ordering_pointer));
				ordered = ordering[ordering_index];
			}
			if (ordered >= 0 && ordered <= INT32_MAX && !rocm_mlx_q4_token_suppressed(static_cast<uint32_t>(ordered), suppress_tokens, args.suppress_count)) {
				selected_token = static_cast<int32_t>(ordered);
			}
		}
	}
	output[index] = selected_token;
}

extern "C" __global__ void rocm_packed_topk(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_packed_topk_launch_args &args = *reinterpret_cast<const rocm_packed_topk_launch_args *>(packet);
	if (!rocm_valid_packed_topk_args(args) || blockDim.x != ROCM_PACKED_TOPK_BLOCK_SIZE) {
		return;
	}
	__shared__ unsigned long long scratch[ROCM_PACKED_TOPK_CHUNK_SIZE];
	const uint32_t chunk_begin = blockIdx.x * args.chunk_size;
	const unsigned long long *input = reinterpret_cast<const unsigned long long *>(static_cast<uintptr_t>(args.input_pointer));
	unsigned long long *output = reinterpret_cast<unsigned long long *>(static_cast<uintptr_t>(args.output_pointer));
	for (uint32_t local = threadIdx.x; local < ROCM_PACKED_TOPK_CHUNK_SIZE; local += blockDim.x) {
		const uint32_t index = chunk_begin + local;
		scratch[local] = index < args.input_count ? input[index] : 0;
	}
	__syncthreads();

	for (uint32_t width = 2u; width <= ROCM_PACKED_TOPK_CHUNK_SIZE; width <<= 1u) {
		for (uint32_t stride = width >> 1u; stride > 0u; stride >>= 1u) {
			for (uint32_t local = threadIdx.x; local < ROCM_PACKED_TOPK_CHUNK_SIZE; local += blockDim.x) {
				const uint32_t other = local ^ stride;
				if (other > local) {
					const bool descending = (local & width) == 0u;
					const unsigned long long left = scratch[local];
					const unsigned long long right = scratch[other];
					if ((descending && left < right) || (!descending && left > right)) {
						scratch[local] = right;
						scratch[other] = left;
					}
				}
			}
			__syncthreads();
		}
	}
	if (threadIdx.x < args.top_k) {
		output[blockIdx.x * args.top_k + threadIdx.x] = scratch[threadIdx.x];
	}
}

extern "C" __global__ void rocm_packed_topk_sample(const unsigned char *packet)
{
	if (packet == nullptr || blockIdx.x != 0 || threadIdx.x != 0) {
		return;
	}
	const rocm_packed_topk_sample_launch_args &args = *reinterpret_cast<const rocm_packed_topk_sample_launch_args *>(packet);
	if (!rocm_valid_packed_topk_sample_args(args)) {
		return;
	}
	const unsigned long long *input = reinterpret_cast<const unsigned long long *>(static_cast<uintptr_t>(args.input_pointer));
	unsigned long long *output = reinterpret_cast<unsigned long long *>(static_cast<uintptr_t>(args.output_pointer));
	output[0] = 0ull;

	uint32_t first_valid = args.top_k;
	for (uint32_t index = 0; index < args.top_k; ++index) {
		if (input[index] != 0ull) {
			first_valid = index;
			break;
		}
	}
	if (first_valid >= args.top_k) {
		return;
	}
	const float raw_temperature = rocm_float_from_bits(args.temperature_bits);
	const float raw_top_p = rocm_float_from_bits(args.top_p_bits);
	if (raw_temperature <= 0.0f && raw_top_p <= 0.0f) {
		output[0] = input[first_valid];
		return;
	}
	const double temperature = raw_temperature == 0.0f ? 1.0 : static_cast<double>(raw_temperature);
	const double top_p = raw_top_p == 0.0f ? 1.0 : static_cast<double>(raw_top_p);
	if (!(temperature > 0.0) || !(top_p > 0.0 && top_p <= 1.0)) {
		return;
	}
	const double softcap = static_cast<double>(rocm_float_from_bits(args.softcap_bits));
	const double max_value = rocm_packed_topk_sample_score(input[first_valid], softcap) / temperature;
	double total = 0.0;
	for (uint32_t index = first_valid; index < args.top_k; ++index) {
		const unsigned long long packed = input[index];
		if (packed == 0ull) {
			continue;
		}
		const double score = rocm_packed_topk_sample_score(packed, softcap);
		const double weight = exp(score / temperature - max_value);
		if (!isfinite(weight)) {
			return;
		}
		total += weight;
	}
	if (!(total > 0.0) || !isfinite(total)) {
		return;
	}
	uint32_t limit = args.top_k;
	if (top_p < 1.0) {
		double cumulative = 0.0;
		for (uint32_t index = first_valid; index < args.top_k; ++index) {
			const unsigned long long packed = input[index];
			if (packed == 0ull) {
				continue;
			}
			const double score = rocm_packed_topk_sample_score(packed, softcap);
			cumulative += exp(score / temperature - max_value);
			if (cumulative / total >= top_p) {
				limit = index + 1u;
				break;
			}
		}
	}
	double selected_total = 0.0;
	unsigned long long fallback = input[first_valid];
	for (uint32_t index = first_valid; index < limit; ++index) {
		const unsigned long long packed = input[index];
		if (packed == 0ull) {
			continue;
		}
		fallback = packed;
		const double score = rocm_packed_topk_sample_score(packed, softcap);
		selected_total += exp(score / temperature - max_value);
	}
	if (!(selected_total > 0.0) || !isfinite(selected_total)) {
		return;
	}
	double draw = rocm_double_from_bits(args.draw_bits);
	if (draw < 0.0) {
		draw = 0.0;
	}
	if (draw >= 1.0) {
		draw = nextafter(1.0, 0.0);
	}
	const double target = draw * selected_total;
	double cumulative = 0.0;
	for (uint32_t index = first_valid; index < limit; ++index) {
		const unsigned long long packed = input[index];
		if (packed == 0ull) {
			continue;
		}
		const double score = rocm_packed_topk_sample_score(packed, softcap);
		cumulative += exp(score / temperature - max_value);
		if (target <= cumulative) {
			output[0] = packed;
			return;
		}
	}
	output[0] = fallback;
}

extern "C" __global__ void rocm_mlx_q4_gelu_tanh_multiply(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_gelu_tanh_mul_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_gelu_tanh_mul_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_gelu_tanh_mul_args(args)) {
		return;
	}

	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_ROWS_PER_BLOCK + row_lane;

	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *gate_weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.gate_weight_pointer));
	const uint16_t *gate_scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.gate_scale_pointer));
	const uint16_t *gate_biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.gate_bias_pointer));
	const uint32_t *up_weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.up_weight_pointer));
	const uint16_t *up_scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.up_scale_pointer));
	const uint16_t *up_biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.up_bias_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	const uint64_t packed_per_row = rocm_mlx_affine_packed_per_row(args.cols, args.bits);
	const uint32_t groups_per_row = args.cols / args.group_size;
	const uint64_t row_packed_base = static_cast<uint64_t>(row) * packed_per_row;
	const uint32_t row_group_base = row * groups_per_row;
	float gate_sum = 0.0f;
	float up_sum = 0.0f;
	if (row < args.rows && args.bits == 4u && args.group_size == 64u) {
		for (uint32_t packed = col_lane; packed < packed_per_row; packed += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
			const uint32_t gate_word = gate_weights[row_packed_base + packed];
			const uint32_t up_word = up_weights[row_packed_base + packed];
			const uint32_t col = packed << 3;
			const uint32_t group = row_group_base + (packed >> 3u);
			const float gate_scale = rocm_bfloat16_to_float(gate_scales[group]);
			const float gate_bias = rocm_bfloat16_to_float(gate_biases[group]);
			const float up_scale = rocm_bfloat16_to_float(up_scales[group]);
			const float up_bias = rocm_bfloat16_to_float(up_biases[group]);
			const float in0 = input[col];
			const float in1 = input[col + 1u];
			const float in2 = input[col + 2u];
			const float in3 = input[col + 3u];
			const float in4 = input[col + 4u];
			const float in5 = input[col + 5u];
			const float in6 = input[col + 6u];
			const float in7 = input[col + 7u];
			const float input_sum = in0 + in1 + in2 + in3 + in4 + in5 + in6 + in7;
			const float gate_q_dot =
				static_cast<float>(gate_word & 0x0fu) * in0 +
				static_cast<float>((gate_word >> 4) & 0x0fu) * in1 +
				static_cast<float>((gate_word >> 8) & 0x0fu) * in2 +
				static_cast<float>((gate_word >> 12) & 0x0fu) * in3 +
				static_cast<float>((gate_word >> 16) & 0x0fu) * in4 +
				static_cast<float>((gate_word >> 20) & 0x0fu) * in5 +
				static_cast<float>((gate_word >> 24) & 0x0fu) * in6 +
				static_cast<float>((gate_word >> 28) & 0x0fu) * in7;
			const float up_q_dot =
				static_cast<float>(up_word & 0x0fu) * in0 +
				static_cast<float>((up_word >> 4) & 0x0fu) * in1 +
				static_cast<float>((up_word >> 8) & 0x0fu) * in2 +
				static_cast<float>((up_word >> 12) & 0x0fu) * in3 +
				static_cast<float>((up_word >> 16) & 0x0fu) * in4 +
				static_cast<float>((up_word >> 20) & 0x0fu) * in5 +
				static_cast<float>((up_word >> 24) & 0x0fu) * in6 +
				static_cast<float>((up_word >> 28) & 0x0fu) * in7;
			gate_sum += gate_scale * gate_q_dot + gate_bias * input_sum;
			up_sum += up_scale * up_q_dot + up_bias * input_sum;
		}
	} else if (row < args.rows && args.bits == 4u && (args.group_size & 7u) == 0u) {
		for (uint32_t packed = col_lane; packed < packed_per_row; packed += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
			const uint32_t gate_word = gate_weights[row_packed_base + packed];
			const uint32_t up_word = up_weights[row_packed_base + packed];
			const uint32_t col = packed << 3;
			const uint32_t group = row_group_base + col / args.group_size;
			const float gate_scale = rocm_bfloat16_to_float(gate_scales[group]);
			const float gate_bias = rocm_bfloat16_to_float(gate_biases[group]);
			const float up_scale = rocm_bfloat16_to_float(up_scales[group]);
			const float up_bias = rocm_bfloat16_to_float(up_biases[group]);
			const float in0 = input[col];
			const float in1 = input[col + 1u];
			const float in2 = input[col + 2u];
			const float in3 = input[col + 3u];
			const float in4 = input[col + 4u];
			const float in5 = input[col + 5u];
			const float in6 = input[col + 6u];
			const float in7 = input[col + 7u];
			const float input_sum = in0 + in1 + in2 + in3 + in4 + in5 + in6 + in7;
			const float gate_q_dot =
				static_cast<float>(gate_word & 0x0fu) * in0 +
				static_cast<float>((gate_word >> 4) & 0x0fu) * in1 +
				static_cast<float>((gate_word >> 8) & 0x0fu) * in2 +
				static_cast<float>((gate_word >> 12) & 0x0fu) * in3 +
				static_cast<float>((gate_word >> 16) & 0x0fu) * in4 +
				static_cast<float>((gate_word >> 20) & 0x0fu) * in5 +
				static_cast<float>((gate_word >> 24) & 0x0fu) * in6 +
				static_cast<float>((gate_word >> 28) & 0x0fu) * in7;
			const float up_q_dot =
				static_cast<float>(up_word & 0x0fu) * in0 +
				static_cast<float>((up_word >> 4) & 0x0fu) * in1 +
				static_cast<float>((up_word >> 8) & 0x0fu) * in2 +
				static_cast<float>((up_word >> 12) & 0x0fu) * in3 +
				static_cast<float>((up_word >> 16) & 0x0fu) * in4 +
				static_cast<float>((up_word >> 20) & 0x0fu) * in5 +
				static_cast<float>((up_word >> 24) & 0x0fu) * in6 +
				static_cast<float>((up_word >> 28) & 0x0fu) * in7;
			gate_sum += gate_scale * gate_q_dot + gate_bias * input_sum;
			up_sum += up_scale * up_q_dot + up_bias * input_sum;
		}
	} else if (row < args.rows && args.bits == 8u && args.group_size == 32u) {
		for (uint32_t group_col = col_lane; group_col < groups_per_row; group_col += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
			const uint32_t group = row_group_base + group_col;
			const float gate_scale = rocm_bfloat16_to_float(gate_scales[group]);
			const float gate_bias = rocm_bfloat16_to_float(gate_biases[group]);
			const float up_scale = rocm_bfloat16_to_float(up_scales[group]);
			const float up_bias = rocm_bfloat16_to_float(up_biases[group]);
			float gate_q_dot = 0.0f;
			float up_q_dot = 0.0f;
			float input_sum = 0.0f;
			rocm_mlx_affine_q8_32_pair_dot(input, gate_weights, up_weights, row_packed_base, group_col << 3u, group_col << 5u, &gate_q_dot, &up_q_dot, &input_sum);
			gate_sum += gate_scale * gate_q_dot + gate_bias * input_sum;
			up_sum += up_scale * up_q_dot + up_bias * input_sum;
		}
	} else if (row < args.rows && args.bits == 8u && args.group_size == 64u) {
		for (uint32_t group_col = col_lane; group_col < groups_per_row; group_col += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
			const uint32_t group = row_group_base + group_col;
			const float gate_scale = rocm_bfloat16_to_float(gate_scales[group]);
			const float gate_bias = rocm_bfloat16_to_float(gate_biases[group]);
			const float up_scale = rocm_bfloat16_to_float(up_scales[group]);
			const float up_bias = rocm_bfloat16_to_float(up_biases[group]);
			const uint32_t first_packed = group_col << 4u;
			const uint32_t first_col = group_col << 6u;
			float gate_q_dot = 0.0f;
			float up_q_dot = 0.0f;
			float input_sum = 0.0f;
			rocm_mlx_affine_q8_32_pair_dot(input, gate_weights, up_weights, row_packed_base, first_packed, first_col, &gate_q_dot, &up_q_dot, &input_sum);
			rocm_mlx_affine_q8_32_pair_dot(input, gate_weights, up_weights, row_packed_base, first_packed + 8u, first_col + 32u, &gate_q_dot, &up_q_dot, &input_sum);
			gate_sum += gate_scale * gate_q_dot + gate_bias * input_sum;
			up_sum += up_scale * up_q_dot + up_bias * input_sum;
		}
	} else if (row < args.rows && args.bits == 6u && args.group_size == 64u) {
		for (uint32_t group_col = col_lane; group_col < groups_per_row; group_col += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
			const uint32_t group = row_group_base + group_col;
			const float gate_scale = rocm_bfloat16_to_float(gate_scales[group]);
			const float gate_bias = rocm_bfloat16_to_float(gate_biases[group]);
			const float up_scale = rocm_bfloat16_to_float(up_scales[group]);
			const float up_bias = rocm_bfloat16_to_float(up_biases[group]);
			const uint32_t first_packed = group_col * 12u;
			const uint32_t first_col = group_col << 6u;
			float gate_q_dot = 0.0f;
			float up_q_dot = 0.0f;
			float input_sum = 0.0f;
			for (uint32_t block = 0; block < 4u; ++block) {
				rocm_mlx_affine_q6_16_pair_dot(input, gate_weights, up_weights, row_packed_base, first_packed + block * 3u, first_col + block * 16u, &gate_q_dot, &up_q_dot, &input_sum);
			}
			gate_sum += gate_scale * gate_q_dot + gate_bias * input_sum;
			up_sum += up_scale * up_q_dot + up_bias * input_sum;
		}
	} else if (row < args.rows) {
		for (uint32_t col = col_lane; col < args.cols; col += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
			const uint32_t gate_q = rocm_mlx_affine_quantized_value(gate_weights, row_packed_base, col, args.bits);
			const uint32_t up_q = rocm_mlx_affine_quantized_value(up_weights, row_packed_base, col, args.bits);
			const uint32_t group = row_group_base + col / args.group_size;
			const float gate_scale = rocm_bfloat16_to_float(gate_scales[group]);
			const float gate_bias = rocm_bfloat16_to_float(gate_biases[group]);
			const float up_scale = rocm_bfloat16_to_float(up_scales[group]);
			const float up_bias = rocm_bfloat16_to_float(up_biases[group]);
			const float value = input[col];
			gate_sum += value * (static_cast<float>(gate_q) * gate_scale + gate_bias);
			up_sum += value * (static_cast<float>(up_q) * up_scale + up_bias);
		}
	}
	gate_sum = rocm_mlx_q4_row_reduce(gate_sum);
	up_sum = rocm_mlx_q4_row_reduce(up_sum);
	if (col_lane == 0 && row < args.rows) {
		output[row] = rocm_gelu_tanh_value(gate_sum) * up_sum;
	}
}

extern "C" __global__ void rocm_mlx_q4_gelu_tanh_multiply_q8_g64_row8(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_gelu_tanh_mul_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_gelu_tanh_mul_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_gelu_tanh_mul_args(args) || args.bits != 8u || args.group_size != 64u || args.cols < 2560u) {
		return;
	}
	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}

	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_ROWS_PER_BLOCK + row_lane;
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *gate_weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.gate_weight_pointer));
	const uint16_t *gate_scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.gate_scale_pointer));
	const uint16_t *gate_biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.gate_bias_pointer));
	const uint32_t *up_weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.up_weight_pointer));
	const uint16_t *up_scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.up_scale_pointer));
	const uint16_t *up_biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.up_bias_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float gate_sum = 0.0f;
	float up_sum = 0.0f;
	if (row < args.rows) {
		const uint64_t packed_per_row = rocm_mlx_affine_packed_per_row(args.cols, args.bits);
		const uint32_t groups_per_row = args.cols >> 6u;
		const uint32_t half_groups_per_row = groups_per_row << 1u;
		const uint64_t row_packed_base = static_cast<uint64_t>(row) * packed_per_row;
		const uint32_t row_group_base = row * groups_per_row;
		for (uint32_t half_group = col_lane; half_group < half_groups_per_row; half_group += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
			const uint32_t group_col = half_group >> 1u;
			const uint32_t group = row_group_base + group_col;
			const float gate_scale = rocm_bfloat16_to_float(gate_scales[group]);
			const float gate_bias = rocm_bfloat16_to_float(gate_biases[group]);
			const float up_scale = rocm_bfloat16_to_float(up_scales[group]);
			const float up_bias = rocm_bfloat16_to_float(up_biases[group]);
			const uint32_t first_packed = half_group << 3u;
			const uint32_t first_col = half_group << 5u;
			float gate_q_dot = 0.0f;
			float up_q_dot = 0.0f;
			float input_sum = 0.0f;
			rocm_mlx_affine_q8_32_pair_dot(input, gate_weights, up_weights, row_packed_base, first_packed, first_col, &gate_q_dot, &up_q_dot, &input_sum);
			gate_sum += gate_scale * gate_q_dot + gate_bias * input_sum;
			up_sum += up_scale * up_q_dot + up_bias * input_sum;
		}
	}
	gate_sum = rocm_mlx_q4_row_reduce(gate_sum);
	up_sum = rocm_mlx_q4_row_reduce(up_sum);
	if (col_lane == 0u && row < args.rows) {
		output[row] = rocm_gelu_tanh_value(gate_sum) * up_sum;
	}
}

extern "C" __global__ void rocm_mlx_q4_gelu_tanh_multiply_q4_g64_rows15360_cols3840_row8(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_gelu_tanh_mul_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_gelu_tanh_mul_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_gelu_tanh_mul_args(args) ||
		args.rows != 15360u ||
		args.cols != 3840u ||
		args.group_size != 64u ||
		args.bits != 4u ||
		blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}

	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_ROWS_PER_BLOCK + row_lane;
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *gate_weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.gate_weight_pointer));
	const uint16_t *gate_scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.gate_scale_pointer));
	const uint16_t *gate_biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.gate_bias_pointer));
	const uint32_t *up_weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.up_weight_pointer));
	const uint16_t *up_scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.up_scale_pointer));
	const uint16_t *up_biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.up_bias_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	const uint64_t packed_per_row = args.cols >> 3u;
	const uint32_t groups_per_row = args.cols >> 6u;
	const uint64_t row_packed_base = static_cast<uint64_t>(row) * packed_per_row;
	const uint32_t row_group_base = row * groups_per_row;
	float gate_sum = 0.0f;
	float up_sum = 0.0f;
	for (uint32_t group_col = col_lane; group_col < groups_per_row; group_col += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
		const uint32_t group = row_group_base + group_col;
		const float gate_scale = rocm_bfloat16_to_float(gate_scales[group]);
		const float gate_bias = rocm_bfloat16_to_float(gate_biases[group]);
		const float up_scale = rocm_bfloat16_to_float(up_scales[group]);
		const float up_bias = rocm_bfloat16_to_float(up_biases[group]);
		const uint32_t first_packed = group_col << 3u;
		const uint32_t first_col = group_col << 6u;
		float gate_q_dot = 0.0f;
		float up_q_dot = 0.0f;
		float input_sum = 0.0f;
		rocm_mlx_affine_q4_64_pair_dot(
			input,
			gate_weights,
			up_weights,
			row_packed_base,
			first_packed,
			first_col,
			&gate_q_dot,
			&up_q_dot,
			&input_sum);
		gate_sum += gate_scale * gate_q_dot + gate_bias * input_sum;
		up_sum += up_scale * up_q_dot + up_bias * input_sum;
	}
	gate_sum = rocm_mlx_q4_row_reduce(gate_sum);
	up_sum = rocm_mlx_q4_row_reduce(up_sum);
	if (col_lane == 0u) {
		output[row] = rocm_gelu_tanh_value(gate_sum) * up_sum;
	}
}

extern "C" __global__ void rocm_mlx_q4_gelu_tanh_multiply_q4_g32_cols1536_row16(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_gelu_tanh_mul_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_gelu_tanh_mul_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_gelu_tanh_mul_args(args) || args.bits != 4u || args.group_size != 32u || args.cols < 1536u || (args.cols % 32u) != 0u) {
		return;
	}

	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_GELU_TANH_Q4_G32_COLS1536_ROW16_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_GELU_TANH_Q4_G32_COLS1536_ROW16_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_GELU_TANH_Q4_G32_COLS1536_ROW16_ROWS_PER_BLOCK + row_lane;

	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *gate_weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.gate_weight_pointer));
	const uint16_t *gate_scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.gate_scale_pointer));
	const uint16_t *gate_biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.gate_bias_pointer));
	const uint32_t *up_weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.up_weight_pointer));
	const uint16_t *up_scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.up_scale_pointer));
	const uint16_t *up_biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.up_bias_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	const uint64_t packed_per_row = args.cols >> 3u;
	const uint32_t groups_per_row = args.cols >> 5u;
	const uint64_t row_packed_base = static_cast<uint64_t>(row) * packed_per_row;
	const uint32_t row_group_base = row * groups_per_row;
	float gate_sum = 0.0f;
	float up_sum = 0.0f;
	if (row < args.rows) {
		for (uint32_t group_col = col_lane; group_col < groups_per_row; group_col += ROCM_MLX_Q4_GELU_TANH_Q4_G32_COLS1536_ROW16_THREADS_PER_ROW) {
			const uint32_t group = row_group_base + group_col;
			const float gate_scale = rocm_bfloat16_to_float(gate_scales[group]);
			const float gate_bias = rocm_bfloat16_to_float(gate_biases[group]);
			const float up_scale = rocm_bfloat16_to_float(up_scales[group]);
			const float up_bias = rocm_bfloat16_to_float(up_biases[group]);
			const uint32_t first_packed = group_col << 2u;
			float gate_q_dot = 0.0f;
			float up_q_dot = 0.0f;
			float input_sum = 0.0f;
			for (uint32_t group_packed = 0; group_packed < 4u; ++group_packed) {
				const uint32_t packed = first_packed + group_packed;
				const uint32_t gate_word = gate_weights[row_packed_base + packed];
				const uint32_t up_word = up_weights[row_packed_base + packed];
				const uint32_t col = packed << 3u;
				const float in0 = input[col];
				const float in1 = input[col + 1u];
				const float in2 = input[col + 2u];
				const float in3 = input[col + 3u];
				const float in4 = input[col + 4u];
				const float in5 = input[col + 5u];
				const float in6 = input[col + 6u];
				const float in7 = input[col + 7u];
				input_sum += in0 + in1 + in2 + in3 + in4 + in5 + in6 + in7;
				gate_q_dot +=
					static_cast<float>(gate_word & 0x0fu) * in0 +
					static_cast<float>((gate_word >> 4) & 0x0fu) * in1 +
					static_cast<float>((gate_word >> 8) & 0x0fu) * in2 +
					static_cast<float>((gate_word >> 12) & 0x0fu) * in3 +
					static_cast<float>((gate_word >> 16) & 0x0fu) * in4 +
					static_cast<float>((gate_word >> 20) & 0x0fu) * in5 +
					static_cast<float>((gate_word >> 24) & 0x0fu) * in6 +
					static_cast<float>((gate_word >> 28) & 0x0fu) * in7;
				up_q_dot +=
					static_cast<float>(up_word & 0x0fu) * in0 +
					static_cast<float>((up_word >> 4) & 0x0fu) * in1 +
					static_cast<float>((up_word >> 8) & 0x0fu) * in2 +
					static_cast<float>((up_word >> 12) & 0x0fu) * in3 +
					static_cast<float>((up_word >> 16) & 0x0fu) * in4 +
					static_cast<float>((up_word >> 20) & 0x0fu) * in5 +
					static_cast<float>((up_word >> 24) & 0x0fu) * in6 +
					static_cast<float>((up_word >> 28) & 0x0fu) * in7;
			}
			gate_sum += gate_scale * gate_q_dot + gate_bias * input_sum;
			up_sum += up_scale * up_q_dot + up_bias * input_sum;
		}
	}
	gate_sum = rocm_mlx_q4_gelu_tanh_q4_g32_cols1536_row16_reduce(gate_sum);
	up_sum = rocm_mlx_q4_gelu_tanh_q4_g32_cols1536_row16_reduce(up_sum);
	if (col_lane == 0 && row < args.rows) {
		output[row] = rocm_gelu_tanh_value(gate_sum) * up_sum;
	}
}

extern "C" __global__ void rocm_mlx_q4_gelu_tanh_multiply_q4_g32_rows15360_cols3840(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_gelu_tanh_mul_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_gelu_tanh_mul_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_gelu_tanh_mul_args(args) ||
		args.rows != ROCM_MLX_Q4_GELU_TANH_12B_ROWS ||
		args.cols != ROCM_MLX_Q4_GELU_TANH_12B_COLS ||
		args.group_size != ROCM_MLX_Q4_GELU_TANH_12B_GROUP_SIZE ||
		args.bits != 4u ||
		blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}

	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_GELU_TANH_Q4_G32_COLS1536_ROW16_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_GELU_TANH_Q4_G32_COLS1536_ROW16_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_GELU_TANH_Q4_G32_COLS1536_ROW16_ROWS_PER_BLOCK + row_lane;
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *gate_weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.gate_weight_pointer));
	const uint16_t *gate_scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.gate_scale_pointer));
	const uint16_t *gate_biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.gate_bias_pointer));
	const uint32_t *up_weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.up_weight_pointer));
	const uint16_t *up_scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.up_scale_pointer));
	const uint16_t *up_biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.up_bias_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	const uint64_t row_packed_base = static_cast<uint64_t>(row) * ROCM_MLX_Q4_GELU_TANH_12B_PACKED_PER_ROW;
	const uint32_t row_group_base = row * ROCM_MLX_Q4_GELU_TANH_12B_GROUPS_PER_ROW;
	float gate_sum = 0.0f;
	float up_sum = 0.0f;
	for (uint32_t group_col = col_lane; group_col < ROCM_MLX_Q4_GELU_TANH_12B_GROUPS_PER_ROW; group_col += ROCM_MLX_Q4_GELU_TANH_Q4_G32_COLS1536_ROW16_THREADS_PER_ROW) {
		const uint32_t group = row_group_base + group_col;
		const float gate_scale = rocm_bfloat16_to_float(gate_scales[group]);
		const float gate_bias = rocm_bfloat16_to_float(gate_biases[group]);
		const float up_scale = rocm_bfloat16_to_float(up_scales[group]);
		const float up_bias = rocm_bfloat16_to_float(up_biases[group]);
		const uint32_t first_packed = group_col << 2u;
		float gate_q_dot = 0.0f;
		float up_q_dot = 0.0f;
		float input_sum = 0.0f;
#pragma unroll
		for (uint32_t group_packed = 0; group_packed < 4u; ++group_packed) {
			const uint32_t packed = first_packed + group_packed;
			const uint32_t gate_word = gate_weights[row_packed_base + packed];
			const uint32_t up_word = up_weights[row_packed_base + packed];
			const uint32_t col = packed << 3u;
			const float in0 = input[col];
			const float in1 = input[col + 1u];
			const float in2 = input[col + 2u];
			const float in3 = input[col + 3u];
			const float in4 = input[col + 4u];
			const float in5 = input[col + 5u];
			const float in6 = input[col + 6u];
			const float in7 = input[col + 7u];
			input_sum += in0 + in1 + in2 + in3 + in4 + in5 + in6 + in7;
			gate_q_dot +=
				static_cast<float>(gate_word & 0x0fu) * in0 +
				static_cast<float>((gate_word >> 4) & 0x0fu) * in1 +
				static_cast<float>((gate_word >> 8) & 0x0fu) * in2 +
				static_cast<float>((gate_word >> 12) & 0x0fu) * in3 +
				static_cast<float>((gate_word >> 16) & 0x0fu) * in4 +
				static_cast<float>((gate_word >> 20) & 0x0fu) * in5 +
				static_cast<float>((gate_word >> 24) & 0x0fu) * in6 +
				static_cast<float>((gate_word >> 28) & 0x0fu) * in7;
			up_q_dot +=
				static_cast<float>(up_word & 0x0fu) * in0 +
				static_cast<float>((up_word >> 4) & 0x0fu) * in1 +
				static_cast<float>((up_word >> 8) & 0x0fu) * in2 +
				static_cast<float>((up_word >> 12) & 0x0fu) * in3 +
				static_cast<float>((up_word >> 16) & 0x0fu) * in4 +
				static_cast<float>((up_word >> 20) & 0x0fu) * in5 +
				static_cast<float>((up_word >> 24) & 0x0fu) * in6 +
				static_cast<float>((up_word >> 28) & 0x0fu) * in7;
		}
		gate_sum += gate_scale * gate_q_dot + gate_bias * input_sum;
		up_sum += up_scale * up_q_dot + up_bias * input_sum;
	}
	gate_sum = rocm_mlx_q4_gelu_tanh_q4_g32_cols1536_row16_reduce(gate_sum);
	up_sum = rocm_mlx_q4_gelu_tanh_q4_g32_cols1536_row16_reduce(up_sum);
	if (col_lane == 0u) {
		output[row] = rocm_gelu_tanh_value(gate_sum) * up_sum;
	}
}

extern "C" __global__ void rocm_mlx_q4_gelu_tanh_multiply_q4_g32_rows15360_cols3840_row8(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_gelu_tanh_mul_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_gelu_tanh_mul_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_gelu_tanh_mul_args(args) ||
		args.rows != ROCM_MLX_Q4_GELU_TANH_12B_ROWS ||
		args.cols != ROCM_MLX_Q4_GELU_TANH_12B_COLS ||
		args.group_size != ROCM_MLX_Q4_GELU_TANH_12B_GROUP_SIZE ||
		args.bits != 4u ||
		blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}

	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_GELU_TANH_12B_ROW8_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_GELU_TANH_12B_ROW8_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_GELU_TANH_12B_ROW8_ROWS_PER_BLOCK + row_lane;
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *gate_weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.gate_weight_pointer));
	const uint16_t *gate_scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.gate_scale_pointer));
	const uint16_t *gate_biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.gate_bias_pointer));
	const uint32_t *up_weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.up_weight_pointer));
	const uint16_t *up_scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.up_scale_pointer));
	const uint16_t *up_biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.up_bias_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	const uint64_t row_packed_base = static_cast<uint64_t>(row) * ROCM_MLX_Q4_GELU_TANH_12B_PACKED_PER_ROW;
	const uint32_t row_group_base = row * ROCM_MLX_Q4_GELU_TANH_12B_GROUPS_PER_ROW;
	float gate_sum = 0.0f;
	float up_sum = 0.0f;
	for (uint32_t group_col = col_lane; group_col < ROCM_MLX_Q4_GELU_TANH_12B_GROUPS_PER_ROW; group_col += ROCM_MLX_Q4_GELU_TANH_12B_ROW8_THREADS_PER_ROW) {
		const uint32_t group = row_group_base + group_col;
		const float gate_scale = rocm_bfloat16_to_float(gate_scales[group]);
		const float gate_bias = rocm_bfloat16_to_float(gate_biases[group]);
		const float up_scale = rocm_bfloat16_to_float(up_scales[group]);
		const float up_bias = rocm_bfloat16_to_float(up_biases[group]);
		const uint32_t first_packed = group_col << 2u;
		float gate_q_dot = 0.0f;
		float up_q_dot = 0.0f;
		float input_sum = 0.0f;
#pragma unroll
		for (uint32_t group_packed = 0; group_packed < 4u; ++group_packed) {
			const uint32_t packed = first_packed + group_packed;
			const uint32_t gate_word = gate_weights[row_packed_base + packed];
			const uint32_t up_word = up_weights[row_packed_base + packed];
			const uint32_t col = packed << 3u;
			const float in0 = input[col];
			const float in1 = input[col + 1u];
			const float in2 = input[col + 2u];
			const float in3 = input[col + 3u];
			const float in4 = input[col + 4u];
			const float in5 = input[col + 5u];
			const float in6 = input[col + 6u];
			const float in7 = input[col + 7u];
			input_sum += in0 + in1 + in2 + in3 + in4 + in5 + in6 + in7;
			gate_q_dot +=
				static_cast<float>(gate_word & 0x0fu) * in0 +
				static_cast<float>((gate_word >> 4) & 0x0fu) * in1 +
				static_cast<float>((gate_word >> 8) & 0x0fu) * in2 +
				static_cast<float>((gate_word >> 12) & 0x0fu) * in3 +
				static_cast<float>((gate_word >> 16) & 0x0fu) * in4 +
				static_cast<float>((gate_word >> 20) & 0x0fu) * in5 +
				static_cast<float>((gate_word >> 24) & 0x0fu) * in6 +
				static_cast<float>((gate_word >> 28) & 0x0fu) * in7;
			up_q_dot +=
				static_cast<float>(up_word & 0x0fu) * in0 +
				static_cast<float>((up_word >> 4) & 0x0fu) * in1 +
				static_cast<float>((up_word >> 8) & 0x0fu) * in2 +
				static_cast<float>((up_word >> 12) & 0x0fu) * in3 +
				static_cast<float>((up_word >> 16) & 0x0fu) * in4 +
				static_cast<float>((up_word >> 20) & 0x0fu) * in5 +
				static_cast<float>((up_word >> 24) & 0x0fu) * in6 +
				static_cast<float>((up_word >> 28) & 0x0fu) * in7;
		}
		gate_sum += gate_scale * gate_q_dot + gate_bias * input_sum;
		up_sum += up_scale * up_q_dot + up_bias * input_sum;
	}
	gate_sum = rocm_mlx_q4_row_reduce(gate_sum);
	up_sum = rocm_mlx_q4_row_reduce(up_sum);
	if (col_lane == 0u) {
		output[row] = rocm_gelu_tanh_value(gate_sum) * up_sum;
	}
}

__device__ void rocm_mlx_q4_gelu_tanh_mlp_persistent_barrier(uint32_t *barrier, uint32_t block_count)
{
	__syncthreads();
	__threadfence();
	if (threadIdx.x == 0u) {
		const uint32_t sense = atomicAdd(barrier + 1u, 0u);
		const uint32_t ticket = atomicAdd(barrier, 1u);
		if (ticket + 1u == block_count) {
			atomicExch(barrier, 0u);
			__threadfence();
			atomicAdd(barrier + 1u, 1u);
		} else {
			while (atomicAdd(barrier + 1u, 0u) == sense) {
			}
		}
	}
	__syncthreads();
}

extern "C" __global__ void rocm_mlx_q4_gelu_tanh_mlp_q4_g32_cols1536_persistent(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_gelu_tanh_mlp_persistent_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_gelu_tanh_mlp_persistent_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_gelu_tanh_mlp_persistent_args(args)) {
		return;
	}
	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE || gridDim.x == 0u) {
		return;
	}

	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *gate_weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.gate_weight_pointer));
	const uint16_t *gate_scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.gate_scale_pointer));
	const uint16_t *gate_biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.gate_bias_pointer));
	const uint32_t *up_weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.up_weight_pointer));
	const uint16_t *up_scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.up_scale_pointer));
	const uint16_t *up_biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.up_bias_pointer));
	const uint32_t *down_weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.down_weight_pointer));
	const uint16_t *down_scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.down_scale_pointer));
	const uint16_t *down_biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.down_bias_pointer));
	float *activation = reinterpret_cast<float *>(static_cast<uintptr_t>(args.activation_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	uint32_t *barrier = reinterpret_cast<uint32_t *>(static_cast<uintptr_t>(args.barrier_pointer));

	const uint32_t activation_blocks = (args.rows + ROCM_MLX_Q4_GELU_TANH_Q4_G32_COLS1536_ROW16_ROWS_PER_BLOCK - 1u) / ROCM_MLX_Q4_GELU_TANH_Q4_G32_COLS1536_ROW16_ROWS_PER_BLOCK;
	const uint32_t activation_row_lane = threadIdx.x / ROCM_MLX_Q4_GELU_TANH_Q4_G32_COLS1536_ROW16_THREADS_PER_ROW;
	const uint32_t activation_col_lane = threadIdx.x - activation_row_lane * ROCM_MLX_Q4_GELU_TANH_Q4_G32_COLS1536_ROW16_THREADS_PER_ROW;
	const uint64_t gate_packed_per_row = args.cols >> 3u;
	const uint32_t gate_groups_per_row = args.cols >> 5u;
	for (uint32_t activation_block = blockIdx.x; activation_block < activation_blocks; activation_block += gridDim.x) {
		const uint32_t row = activation_block * ROCM_MLX_Q4_GELU_TANH_Q4_G32_COLS1536_ROW16_ROWS_PER_BLOCK + activation_row_lane;
		const uint64_t row_packed_base = static_cast<uint64_t>(row) * gate_packed_per_row;
		const uint32_t row_group_base = row * gate_groups_per_row;
		float gate_sum = 0.0f;
		float up_sum = 0.0f;
		if (row < args.rows) {
			for (uint32_t group_col = activation_col_lane; group_col < gate_groups_per_row; group_col += ROCM_MLX_Q4_GELU_TANH_Q4_G32_COLS1536_ROW16_THREADS_PER_ROW) {
				const uint32_t group = row_group_base + group_col;
				const float gate_scale = rocm_bfloat16_to_float(gate_scales[group]);
				const float gate_bias = rocm_bfloat16_to_float(gate_biases[group]);
				const float up_scale = rocm_bfloat16_to_float(up_scales[group]);
				const float up_bias = rocm_bfloat16_to_float(up_biases[group]);
				const uint32_t first_packed = group_col << 2u;
				float gate_q_dot = 0.0f;
				float up_q_dot = 0.0f;
				float input_sum = 0.0f;
				for (uint32_t group_packed = 0; group_packed < 4u; ++group_packed) {
					const uint32_t packed = first_packed + group_packed;
					const uint32_t gate_word = gate_weights[row_packed_base + packed];
					const uint32_t up_word = up_weights[row_packed_base + packed];
					const uint32_t col = packed << 3u;
					const float in0 = input[col];
					const float in1 = input[col + 1u];
					const float in2 = input[col + 2u];
					const float in3 = input[col + 3u];
					const float in4 = input[col + 4u];
					const float in5 = input[col + 5u];
					const float in6 = input[col + 6u];
					const float in7 = input[col + 7u];
					input_sum += in0 + in1 + in2 + in3 + in4 + in5 + in6 + in7;
					gate_q_dot +=
						static_cast<float>(gate_word & 0x0fu) * in0 +
						static_cast<float>((gate_word >> 4) & 0x0fu) * in1 +
						static_cast<float>((gate_word >> 8) & 0x0fu) * in2 +
						static_cast<float>((gate_word >> 12) & 0x0fu) * in3 +
						static_cast<float>((gate_word >> 16) & 0x0fu) * in4 +
						static_cast<float>((gate_word >> 20) & 0x0fu) * in5 +
						static_cast<float>((gate_word >> 24) & 0x0fu) * in6 +
						static_cast<float>((gate_word >> 28) & 0x0fu) * in7;
					up_q_dot +=
						static_cast<float>(up_word & 0x0fu) * in0 +
						static_cast<float>((up_word >> 4) & 0x0fu) * in1 +
						static_cast<float>((up_word >> 8) & 0x0fu) * in2 +
						static_cast<float>((up_word >> 12) & 0x0fu) * in3 +
						static_cast<float>((up_word >> 16) & 0x0fu) * in4 +
						static_cast<float>((up_word >> 20) & 0x0fu) * in5 +
						static_cast<float>((up_word >> 24) & 0x0fu) * in6 +
						static_cast<float>((up_word >> 28) & 0x0fu) * in7;
				}
				gate_sum += gate_scale * gate_q_dot + gate_bias * input_sum;
				up_sum += up_scale * up_q_dot + up_bias * input_sum;
			}
		}
		gate_sum = rocm_mlx_q4_gelu_tanh_q4_g32_cols1536_row16_reduce(gate_sum);
		up_sum = rocm_mlx_q4_gelu_tanh_q4_g32_cols1536_row16_reduce(up_sum);
		if (activation_col_lane == 0u && row < args.rows) {
			activation[row] = rocm_gelu_tanh_value(gate_sum) * up_sum;
		}
	}

	rocm_mlx_q4_gelu_tanh_mlp_persistent_barrier(barrier, gridDim.x);

	const uint32_t projection_blocks = (args.down_rows + ROCM_MLX_Q4_PROJECTION_ROWS_PER_BLOCK - 1u) / ROCM_MLX_Q4_PROJECTION_ROWS_PER_BLOCK;
	const uint32_t projection_row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t projection_col_lane = threadIdx.x - projection_row_lane * ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	for (uint32_t projection_block = blockIdx.x; projection_block < projection_blocks; projection_block += gridDim.x) {
		const uint32_t row = projection_block * ROCM_MLX_Q4_PROJECTION_ROWS_PER_BLOCK + projection_row_lane;
		float sum = 0.0f;
		if (row < args.down_rows) {
			sum = rocm_mlx_q4_projection_row_sum(activation, down_weights, down_scales, down_biases, row, args.rows, args.group_size, args.bits, projection_col_lane);
		}
		sum = rocm_mlx_q4_row_reduce(sum);
		if (projection_col_lane == 0u && row < args.down_rows) {
			output[row] = sum;
		}
	}
}

extern "C" __global__ void rocm_mlx_q4_gelu_tanh_multiply_q6_cols1536(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_gelu_tanh_mul_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_gelu_tanh_mul_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_gelu_tanh_mul_args(args) || args.bits != 6u || args.group_size != 64u || args.cols != 1536u) {
		return;
	}

	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_ROWS_PER_BLOCK + row_lane;

	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *gate_weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.gate_weight_pointer));
	const uint16_t *gate_scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.gate_scale_pointer));
	const uint16_t *gate_biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.gate_bias_pointer));
	const uint32_t *up_weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.up_weight_pointer));
	const uint16_t *up_scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.up_scale_pointer));
	const uint16_t *up_biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.up_bias_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	const uint64_t packed_per_row = rocm_mlx_affine_packed_per_row(args.cols, args.bits);
	const uint32_t groups_per_row = args.cols >> 6u;
	const uint64_t row_packed_base = static_cast<uint64_t>(row) * packed_per_row;
	const uint32_t row_group_base = row * groups_per_row;
	float gate_sum = 0.0f;
	float up_sum = 0.0f;
	if (row < args.rows) {
		for (uint32_t group_col = col_lane; group_col < groups_per_row; group_col += ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_THREADS_PER_ROW) {
			const uint32_t group = row_group_base + group_col;
			const float gate_scale = rocm_bfloat16_to_float(gate_scales[group]);
			const float gate_bias = rocm_bfloat16_to_float(gate_biases[group]);
			const float up_scale = rocm_bfloat16_to_float(up_scales[group]);
			const float up_bias = rocm_bfloat16_to_float(up_biases[group]);
			const uint32_t first_packed = group_col * 12u;
			const uint32_t first_col = group_col << 6u;
			float gate_q_dot = 0.0f;
			float up_q_dot = 0.0f;
			float input_sum = 0.0f;
			for (uint32_t block = 0; block < 4u; ++block) {
				rocm_mlx_affine_q6_16_pair_dot(input, gate_weights, up_weights, row_packed_base, first_packed + block * 3u, first_col + block * 16u, &gate_q_dot, &up_q_dot, &input_sum);
			}
			gate_sum += gate_scale * gate_q_dot + gate_bias * input_sum;
			up_sum += up_scale * up_q_dot + up_bias * input_sum;
		}
	}
	gate_sum = rocm_mlx_q4_gelu_tanh_q6_cols1536_row_reduce(gate_sum);
	up_sum = rocm_mlx_q4_gelu_tanh_q6_cols1536_row_reduce(up_sum);
	if (col_lane == 0 && row < args.rows) {
		output[row] = rocm_gelu_tanh_value(gate_sum) * up_sum;
	}
}

extern "C" __global__ void rocm_mlx_q4_gelu_tanh_multiply_q6_cols1536_row32(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_gelu_tanh_mul_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_gelu_tanh_mul_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_gelu_tanh_mul_args(args) || args.bits != 6u || args.group_size != 64u || args.cols != 1536u || args.rows > 6144u) {
		return;
	}

	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_ROW32_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_ROW32_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_ROW32_ROWS_PER_BLOCK + row_lane;

	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *gate_weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.gate_weight_pointer));
	const uint16_t *gate_scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.gate_scale_pointer));
	const uint16_t *gate_biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.gate_bias_pointer));
	const uint32_t *up_weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.up_weight_pointer));
	const uint16_t *up_scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.up_scale_pointer));
	const uint16_t *up_biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.up_bias_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	const uint64_t packed_per_row = rocm_mlx_affine_packed_per_row(args.cols, args.bits);
	const uint32_t groups_per_row = args.cols >> 6u;
	const uint64_t row_packed_base = static_cast<uint64_t>(row) * packed_per_row;
	const uint32_t row_group_base = row * groups_per_row;
	float gate_sum = 0.0f;
	float up_sum = 0.0f;
	if (row < args.rows) {
		for (uint32_t group_col = col_lane; group_col < groups_per_row; group_col += ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_ROW32_THREADS_PER_ROW) {
			const uint32_t group = row_group_base + group_col;
			const float gate_scale = rocm_bfloat16_to_float(gate_scales[group]);
			const float gate_bias = rocm_bfloat16_to_float(gate_biases[group]);
			const float up_scale = rocm_bfloat16_to_float(up_scales[group]);
			const float up_bias = rocm_bfloat16_to_float(up_biases[group]);
			const uint32_t first_packed = group_col * 12u;
			const uint32_t first_col = group_col << 6u;
			float gate_q_dot = 0.0f;
			float up_q_dot = 0.0f;
			float input_sum = 0.0f;
			for (uint32_t block = 0; block < 4u; ++block) {
				rocm_mlx_affine_q6_16_pair_dot(input, gate_weights, up_weights, row_packed_base, first_packed + block * 3u, first_col + block * 16u, &gate_q_dot, &up_q_dot, &input_sum);
			}
			gate_sum += gate_scale * gate_q_dot + gate_bias * input_sum;
			up_sum += up_scale * up_q_dot + up_bias * input_sum;
		}
	}
	gate_sum = rocm_mlx_q4_gelu_tanh_q6_cols1536_row32_reduce(gate_sum);
	up_sum = rocm_mlx_q4_gelu_tanh_q6_cols1536_row32_reduce(up_sum);
	if (col_lane == 0 && row < args.rows) {
		output[row] = rocm_gelu_tanh_value(gate_sum) * up_sum;
	}
}

extern "C" __global__ void rocm_mlx_q4_gelu_tanh_multiply_q6_cols1536_row64(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_gelu_tanh_mul_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_gelu_tanh_mul_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_gelu_tanh_mul_args(args) || args.bits != 6u || args.group_size != 64u || args.cols != 1536u || args.rows > 6144u) {
		return;
	}

	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_ROW64_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_ROW64_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_ROW64_ROWS_PER_BLOCK + row_lane;

	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *gate_weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.gate_weight_pointer));
	const uint16_t *gate_scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.gate_scale_pointer));
	const uint16_t *gate_biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.gate_bias_pointer));
	const uint32_t *up_weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.up_weight_pointer));
	const uint16_t *up_scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.up_scale_pointer));
	const uint16_t *up_biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.up_bias_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	const uint64_t packed_per_row = rocm_mlx_affine_packed_per_row(args.cols, args.bits);
	const uint32_t groups_per_row = args.cols >> 6u;
	const uint64_t row_packed_base = static_cast<uint64_t>(row) * packed_per_row;
	const uint32_t row_group_base = row * groups_per_row;
	float gate_sum = 0.0f;
	float up_sum = 0.0f;
	if (row < args.rows) {
		for (uint32_t group_col = col_lane; group_col < groups_per_row; group_col += ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_ROW64_THREADS_PER_ROW) {
			const uint32_t group = row_group_base + group_col;
			const float gate_scale = rocm_bfloat16_to_float(gate_scales[group]);
			const float gate_bias = rocm_bfloat16_to_float(gate_biases[group]);
			const float up_scale = rocm_bfloat16_to_float(up_scales[group]);
			const float up_bias = rocm_bfloat16_to_float(up_biases[group]);
			const uint32_t first_packed = group_col * 12u;
			const uint32_t first_col = group_col << 6u;
			float gate_q_dot = 0.0f;
			float up_q_dot = 0.0f;
			float input_sum = 0.0f;
			for (uint32_t block = 0; block < 4u; ++block) {
				rocm_mlx_affine_q6_16_pair_dot(input, gate_weights, up_weights, row_packed_base, first_packed + block * 3u, first_col + block * 16u, &gate_q_dot, &up_q_dot, &input_sum);
			}
			gate_sum += gate_scale * gate_q_dot + gate_bias * input_sum;
			up_sum += up_scale * up_q_dot + up_bias * input_sum;
		}
	}
	gate_sum = rocm_mlx_q4_gelu_tanh_q6_cols1536_row64_reduce(gate_sum);
	up_sum = rocm_mlx_q4_gelu_tanh_q6_cols1536_row64_reduce(up_sum);
	if (col_lane == 0 && row < args.rows) {
		output[row] = rocm_gelu_tanh_value(gate_sum) * up_sum;
	}
}

extern "C" __global__ void rocm_mlx_q4_gelu_tanh_multiply_batch(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_gelu_tanh_mul_batch_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_gelu_tanh_mul_batch_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_gelu_tanh_mul_batch_args(args)) {
		return;
	}

	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_ROWS_PER_BLOCK + row_lane;
	const uint32_t batch_base = blockIdx.y * ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK;
	if (batch_base >= args.batch) {
		return;
	}

	const float *input_base = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *gate_weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.gate_weight_pointer));
	const uint16_t *gate_scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.gate_scale_pointer));
	const uint16_t *gate_biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.gate_bias_pointer));
	const uint32_t *up_weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.up_weight_pointer));
	const uint16_t *up_scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.up_scale_pointer));
	const uint16_t *up_biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.up_bias_pointer));
	float *output_base = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	const uint64_t packed_per_row = rocm_mlx_affine_packed_per_row(args.cols, args.bits);
	const uint32_t groups_per_row = args.cols / args.group_size;
	float gate_sums[ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK] = {};
	float up_sums[ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK] = {};
	if (row < args.rows && args.bits == 4u && (args.group_size & 7u) == 0u) {
		for (uint32_t packed = col_lane; packed < packed_per_row; packed += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
			const uint32_t gate_word = gate_weights[static_cast<uint64_t>(row) * packed_per_row + packed];
			const uint32_t up_word = up_weights[static_cast<uint64_t>(row) * packed_per_row + packed];
			const uint32_t col = packed << 3;
			const uint32_t group = row * groups_per_row + col / args.group_size;
			const float gate_scale = rocm_bfloat16_to_float(gate_scales[group]);
			const float gate_bias = rocm_bfloat16_to_float(gate_biases[group]);
			const float up_scale = rocm_bfloat16_to_float(up_scales[group]);
			const float up_bias = rocm_bfloat16_to_float(up_biases[group]);
			const float gate_q0 = static_cast<float>(gate_word & 0x0fu);
			const float gate_q1 = static_cast<float>((gate_word >> 4) & 0x0fu);
			const float gate_q2 = static_cast<float>((gate_word >> 8) & 0x0fu);
			const float gate_q3 = static_cast<float>((gate_word >> 12) & 0x0fu);
			const float gate_q4 = static_cast<float>((gate_word >> 16) & 0x0fu);
			const float gate_q5 = static_cast<float>((gate_word >> 20) & 0x0fu);
			const float gate_q6 = static_cast<float>((gate_word >> 24) & 0x0fu);
			const float gate_q7 = static_cast<float>((gate_word >> 28) & 0x0fu);
			const float up_q0 = static_cast<float>(up_word & 0x0fu);
			const float up_q1 = static_cast<float>((up_word >> 4) & 0x0fu);
			const float up_q2 = static_cast<float>((up_word >> 8) & 0x0fu);
			const float up_q3 = static_cast<float>((up_word >> 12) & 0x0fu);
			const float up_q4 = static_cast<float>((up_word >> 16) & 0x0fu);
			const float up_q5 = static_cast<float>((up_word >> 20) & 0x0fu);
			const float up_q6 = static_cast<float>((up_word >> 24) & 0x0fu);
			const float up_q7 = static_cast<float>((up_word >> 28) & 0x0fu);
#pragma unroll
			for (uint32_t token_lane = 0; token_lane < ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK; ++token_lane) {
				const uint32_t batch = batch_base + token_lane;
				if (batch >= args.batch) {
					continue;
				}
				const float *input = input_base + batch * args.cols;
				const float in0 = input[col];
				const float in1 = input[col + 1u];
				const float in2 = input[col + 2u];
				const float in3 = input[col + 3u];
				const float in4 = input[col + 4u];
				const float in5 = input[col + 5u];
				const float in6 = input[col + 6u];
				const float in7 = input[col + 7u];
				const float input_sum = in0 + in1 + in2 + in3 + in4 + in5 + in6 + in7;
				const float gate_q_dot =
					gate_q0 * in0 +
					gate_q1 * in1 +
					gate_q2 * in2 +
					gate_q3 * in3 +
					gate_q4 * in4 +
					gate_q5 * in5 +
					gate_q6 * in6 +
					gate_q7 * in7;
				const float up_q_dot =
					up_q0 * in0 +
					up_q1 * in1 +
					up_q2 * in2 +
					up_q3 * in3 +
					up_q4 * in4 +
					up_q5 * in5 +
					up_q6 * in6 +
					up_q7 * in7;
				gate_sums[token_lane] += gate_scale * gate_q_dot + gate_bias * input_sum;
				up_sums[token_lane] += up_scale * up_q_dot + up_bias * input_sum;
			}
		}
	} else if (row < args.rows && args.bits == 8u && args.group_size == 64u) {
		for (uint32_t packed = col_lane; packed < packed_per_row; packed += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
			const uint32_t gate_word = gate_weights[static_cast<uint64_t>(row) * packed_per_row + packed];
			const uint32_t up_word = up_weights[static_cast<uint64_t>(row) * packed_per_row + packed];
			const uint32_t col = packed << 2u;
			const uint32_t group = row * groups_per_row + (packed >> 4u);
			const float gate_scale = rocm_bfloat16_to_float(gate_scales[group]);
			const float gate_bias = rocm_bfloat16_to_float(gate_biases[group]);
			const float up_scale = rocm_bfloat16_to_float(up_scales[group]);
			const float up_bias = rocm_bfloat16_to_float(up_biases[group]);
			const float gate_q0 = static_cast<float>(gate_word & 0xffu);
			const float gate_q1 = static_cast<float>((gate_word >> 8u) & 0xffu);
			const float gate_q2 = static_cast<float>((gate_word >> 16u) & 0xffu);
			const float gate_q3 = static_cast<float>(gate_word >> 24u);
			const float up_q0 = static_cast<float>(up_word & 0xffu);
			const float up_q1 = static_cast<float>((up_word >> 8u) & 0xffu);
			const float up_q2 = static_cast<float>((up_word >> 16u) & 0xffu);
			const float up_q3 = static_cast<float>(up_word >> 24u);
#pragma unroll
			for (uint32_t token_lane = 0; token_lane < ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK; ++token_lane) {
				const uint32_t batch = batch_base + token_lane;
				if (batch >= args.batch) {
					continue;
				}
				const float *input = input_base + batch * args.cols;
				const float in0 = input[col];
				const float in1 = input[col + 1u];
				const float in2 = input[col + 2u];
				const float in3 = input[col + 3u];
				const float input_sum = in0 + in1 + in2 + in3;
				const float gate_q_dot = gate_q0 * in0 + gate_q1 * in1 + gate_q2 * in2 + gate_q3 * in3;
				const float up_q_dot = up_q0 * in0 + up_q1 * in1 + up_q2 * in2 + up_q3 * in3;
				gate_sums[token_lane] += gate_scale * gate_q_dot + gate_bias * input_sum;
				up_sums[token_lane] += up_scale * up_q_dot + up_bias * input_sum;
			}
		}
	} else if (row < args.rows && args.bits == 6u && args.group_size == 64u) {
		const uint64_t row_packed_base = static_cast<uint64_t>(row) * packed_per_row;
		for (uint32_t group_col = col_lane; group_col < groups_per_row; group_col += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
			const uint32_t group = row * groups_per_row + group_col;
			const float gate_scale = rocm_bfloat16_to_float(gate_scales[group]);
			const float gate_bias = rocm_bfloat16_to_float(gate_biases[group]);
			const float up_scale = rocm_bfloat16_to_float(up_scales[group]);
			const float up_bias = rocm_bfloat16_to_float(up_biases[group]);
			const uint32_t first_packed = group_col * 12u;
			const uint32_t first_col = group_col << 6u;
			float gate_q_dot_sums[ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK] = {};
			float up_q_dot_sums[ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK] = {};
			float input_sums[ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK] = {};
			for (uint32_t block = 0; block < 4u; ++block) {
				rocm_mlx_affine_q6_16_pair_batch_dot(input_base, gate_weights, up_weights, row_packed_base, first_packed + block * 3u, first_col + block * 16u, args.cols, batch_base, args.batch, gate_q_dot_sums, up_q_dot_sums, input_sums);
			}
#pragma unroll
			for (uint32_t token_lane = 0; token_lane < ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK; ++token_lane) {
				gate_sums[token_lane] += gate_scale * gate_q_dot_sums[token_lane] + gate_bias * input_sums[token_lane];
				up_sums[token_lane] += up_scale * up_q_dot_sums[token_lane] + up_bias * input_sums[token_lane];
			}
		}
	} else if (row < args.rows) {
		for (uint32_t col = col_lane; col < args.cols; col += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
			const uint64_t row_packed_base = static_cast<uint64_t>(row) * packed_per_row;
			const uint32_t gate_q = rocm_mlx_affine_quantized_value(gate_weights, row_packed_base, col, args.bits);
			const uint32_t up_q = rocm_mlx_affine_quantized_value(up_weights, row_packed_base, col, args.bits);
			const uint32_t group = row * groups_per_row + col / args.group_size;
			const float gate_scale = rocm_bfloat16_to_float(gate_scales[group]);
			const float gate_bias = rocm_bfloat16_to_float(gate_biases[group]);
			const float up_scale = rocm_bfloat16_to_float(up_scales[group]);
			const float up_bias = rocm_bfloat16_to_float(up_biases[group]);
#pragma unroll
			for (uint32_t token_lane = 0; token_lane < ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK; ++token_lane) {
				const uint32_t batch = batch_base + token_lane;
				if (batch < args.batch) {
					const float *input = input_base + batch * args.cols;
					const float value = input[col];
					gate_sums[token_lane] += value * (static_cast<float>(gate_q) * gate_scale + gate_bias);
					up_sums[token_lane] += value * (static_cast<float>(up_q) * up_scale + up_bias);
				}
			}
		}
	}
#pragma unroll
	for (uint32_t token_lane = 0; token_lane < ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK; ++token_lane) {
		const uint32_t batch = batch_base + token_lane;
		const float gate_sum = rocm_mlx_q4_row_reduce(gate_sums[token_lane]);
		const float up_sum = rocm_mlx_q4_row_reduce(up_sums[token_lane]);
		if (col_lane == 0 && row < args.rows && batch < args.batch) {
			float *output = output_base + batch * args.rows;
			output[row] = rocm_gelu_tanh_value(gate_sum) * up_sum;
		}
	}
}

extern "C" __global__ void rocm_mlx_q4_gelu_tanh_multiply_batch_q8_g64_row16(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_gelu_tanh_mul_batch_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_gelu_tanh_mul_batch_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_gelu_tanh_mul_batch_args(args) || args.bits != 8u || args.group_size != 64u) {
		return;
	}
	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_ROW16_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_ROW16_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_ROW16_ROWS_PER_BLOCK + row_lane;
	const uint32_t batch_base = blockIdx.y * ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK;
	if (batch_base >= args.batch) {
		return;
	}

	const float *input_base = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *gate_weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.gate_weight_pointer));
	const uint16_t *gate_scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.gate_scale_pointer));
	const uint16_t *gate_biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.gate_bias_pointer));
	const uint32_t *up_weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.up_weight_pointer));
	const uint16_t *up_scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.up_scale_pointer));
	const uint16_t *up_biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.up_bias_pointer));
	float *output_base = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float gate_sums[ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK] = {};
	float up_sums[ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK] = {};
	if (row < args.rows) {
		const uint64_t packed_per_row = rocm_mlx_affine_packed_per_row(args.cols, args.bits);
		const uint32_t groups_per_row = args.cols >> 6u;
		const uint64_t row_packed_base = static_cast<uint64_t>(row) * packed_per_row;
		for (uint32_t packed = col_lane; packed < packed_per_row; packed += ROCM_MLX_Q4_PROJECTION_ROW16_THREADS_PER_ROW) {
			const uint32_t gate_word = gate_weights[row_packed_base + packed];
			const uint32_t up_word = up_weights[row_packed_base + packed];
			const uint32_t col = packed << 2u;
			const uint32_t group = row * groups_per_row + (packed >> 4u);
			const float gate_scale = rocm_bfloat16_to_float(gate_scales[group]);
			const float gate_bias = rocm_bfloat16_to_float(gate_biases[group]);
			const float up_scale = rocm_bfloat16_to_float(up_scales[group]);
			const float up_bias = rocm_bfloat16_to_float(up_biases[group]);
			const float gate_value0 = static_cast<float>(gate_word & 0xffu) * gate_scale + gate_bias;
			const float gate_value1 = static_cast<float>((gate_word >> 8u) & 0xffu) * gate_scale + gate_bias;
			const float gate_value2 = static_cast<float>((gate_word >> 16u) & 0xffu) * gate_scale + gate_bias;
			const float gate_value3 = static_cast<float>(gate_word >> 24u) * gate_scale + gate_bias;
			const float up_value0 = static_cast<float>(up_word & 0xffu) * up_scale + up_bias;
			const float up_value1 = static_cast<float>((up_word >> 8u) & 0xffu) * up_scale + up_bias;
			const float up_value2 = static_cast<float>((up_word >> 16u) & 0xffu) * up_scale + up_bias;
			const float up_value3 = static_cast<float>(up_word >> 24u) * up_scale + up_bias;
#pragma unroll
			for (uint32_t token_lane = 0; token_lane < ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK; ++token_lane) {
				const uint32_t batch = batch_base + token_lane;
				if (batch >= args.batch) {
					continue;
				}
				const float *input = input_base + batch * args.cols;
				const float in0 = input[col];
				const float in1 = input[col + 1u];
				const float in2 = input[col + 2u];
				const float in3 = input[col + 3u];
				gate_sums[token_lane] = fmaf(in0, gate_value0, gate_sums[token_lane]);
				gate_sums[token_lane] = fmaf(in1, gate_value1, gate_sums[token_lane]);
				gate_sums[token_lane] = fmaf(in2, gate_value2, gate_sums[token_lane]);
				gate_sums[token_lane] = fmaf(in3, gate_value3, gate_sums[token_lane]);
				up_sums[token_lane] = fmaf(in0, up_value0, up_sums[token_lane]);
				up_sums[token_lane] = fmaf(in1, up_value1, up_sums[token_lane]);
				up_sums[token_lane] = fmaf(in2, up_value2, up_sums[token_lane]);
				up_sums[token_lane] = fmaf(in3, up_value3, up_sums[token_lane]);
			}
		}
	}
#pragma unroll
	for (uint32_t token_lane = 0; token_lane < ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK; ++token_lane) {
		const uint32_t batch = batch_base + token_lane;
		const float gate_sum = rocm_mlx_q4_projection_row16_reduce(gate_sums[token_lane]);
		const float up_sum = rocm_mlx_q4_projection_row16_reduce(up_sums[token_lane]);
		if (col_lane == 0 && row < args.rows && batch < args.batch) {
			float *output = output_base + batch * args.rows;
			output[row] = rocm_gelu_tanh_value(gate_sum) * up_sum;
		}
	}
}

extern "C" __global__ void rocm_mlx_q4_gelu_tanh_projection(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_gelu_tanh_proj_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_gelu_tanh_proj_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_gelu_tanh_proj_args(args)) {
		return;
	}

	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_ROWS_PER_BLOCK + row_lane;

	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.weight_pointer));
	const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.scale_pointer));
	const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.bias_pointer));
	const float *multiplier = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.multiplier_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float sum = 0.0f;
	if (row < args.rows) {
		sum = rocm_mlx_q4_projection_row_sum(input, weights, scales, biases, row, args.cols, args.group_size, args.bits, col_lane);
	}
	sum = rocm_mlx_q4_row_reduce(sum);
	if (col_lane == 0 && row < args.rows) {
		output[row] = rocm_gelu_tanh_value(sum) * multiplier[row];
	}
}

extern "C" __global__ void rocm_mlx_q4_gelu_tanh_projection_q6_row16(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_gelu_tanh_proj_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_gelu_tanh_proj_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_gelu_tanh_proj_args(args) || args.bits != 6u || args.group_size != 64u || args.cols < 1536u) {
		return;
	}

	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_Q6_ROW16_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_Q6_ROW16_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_Q6_ROW16_ROWS_PER_BLOCK + row_lane;

	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.weight_pointer));
	const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.scale_pointer));
	const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.bias_pointer));
	const float *multiplier = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.multiplier_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float sum = 0.0f;
	if (row < args.rows) {
		sum = rocm_mlx_q4_projection_row_sum(input, weights, scales, biases, row, args.cols, args.group_size, args.bits, col_lane, ROCM_MLX_Q4_PROJECTION_Q6_ROW16_THREADS_PER_ROW);
	}
	sum = rocm_mlx_q4_projection_q6_row16_reduce(sum);
	if (col_lane == 0 && row < args.rows) {
		output[row] = rocm_gelu_tanh_value(sum) * multiplier[row];
	}
}

extern "C" __global__ void rocm_mlx_q4_gelu_tanh_projection_batch(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_mlx_q4_gelu_tanh_proj_batch_launch_args &args = *reinterpret_cast<const rocm_mlx_q4_gelu_tanh_proj_batch_launch_args *>(packet);
	if (!rocm_valid_mlx_q4_gelu_tanh_proj_batch_args(args)) {
		return;
	}

	if (blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_ROWS_PER_BLOCK + row_lane;
	const uint32_t batch_base = blockIdx.y * ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK;
	if (batch_base >= args.batch) {
		return;
	}

	const float *input_base = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint32_t *weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.weight_pointer));
	const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.scale_pointer));
	const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.bias_pointer));
	const float *multiplier_base = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.multiplier_pointer));
	float *output_base = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float sums[ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK] = {};
	if (row < args.rows) {
		const uint64_t packed_per_row = rocm_mlx_affine_packed_per_row(args.cols, args.bits);
		const uint32_t groups_per_row = args.cols / args.group_size;
		if (args.bits == 4u && (args.group_size & 7u) == 0u) {
			const uint32_t packed_per_group = args.group_size >> 3u;
			for (uint32_t group_col = col_lane; group_col < groups_per_row; group_col += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
				const uint32_t group = row * groups_per_row + group_col;
				const float scale = rocm_bfloat16_to_float(scales[group]);
				const float bias = rocm_bfloat16_to_float(biases[group]);
				const uint32_t first_packed = group_col * packed_per_group;
				float q_dot_sums[ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK] = {};
				float input_sums[ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK] = {};
				for (uint32_t group_packed = 0; group_packed < packed_per_group; ++group_packed) {
					const uint32_t packed = first_packed + group_packed;
					const uint32_t word = weights[static_cast<uint64_t>(row) * packed_per_row + packed];
					const uint32_t col = packed << 3;
					const float q0 = static_cast<float>(word & 0x0fu);
					const float q1 = static_cast<float>((word >> 4) & 0x0fu);
					const float q2 = static_cast<float>((word >> 8) & 0x0fu);
					const float q3 = static_cast<float>((word >> 12) & 0x0fu);
					const float q4 = static_cast<float>((word >> 16) & 0x0fu);
					const float q5 = static_cast<float>((word >> 20) & 0x0fu);
					const float q6 = static_cast<float>((word >> 24) & 0x0fu);
					const float q7 = static_cast<float>((word >> 28) & 0x0fu);
#pragma unroll
					for (uint32_t token_lane = 0; token_lane < ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK; ++token_lane) {
						const uint32_t batch = batch_base + token_lane;
						if (batch >= args.batch) {
							continue;
						}
						const float *input = input_base + batch * args.cols;
						const float in0 = input[col];
						const float in1 = input[col + 1u];
						const float in2 = input[col + 2u];
						const float in3 = input[col + 3u];
						const float in4 = input[col + 4u];
						const float in5 = input[col + 5u];
						const float in6 = input[col + 6u];
						const float in7 = input[col + 7u];
						input_sums[token_lane] += in0 + in1 + in2 + in3 + in4 + in5 + in6 + in7;
						q_dot_sums[token_lane] +=
							q0 * in0 +
							q1 * in1 +
							q2 * in2 +
							q3 * in3 +
							q4 * in4 +
							q5 * in5 +
							q6 * in6 +
							q7 * in7;
					}
				}
#pragma unroll
				for (uint32_t token_lane = 0; token_lane < ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK; ++token_lane) {
					sums[token_lane] += scale * q_dot_sums[token_lane] + bias * input_sums[token_lane];
				}
			}
		} else if (args.bits == 6u && args.group_size == 64u) {
			const uint64_t row_packed_base = static_cast<uint64_t>(row) * packed_per_row;
			for (uint32_t group_col = col_lane; group_col < groups_per_row; group_col += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
				const uint32_t group = row * groups_per_row + group_col;
				const float scale = rocm_bfloat16_to_float(scales[group]);
				const float bias = rocm_bfloat16_to_float(biases[group]);
				const uint32_t first_packed = group_col * 12u;
				const uint32_t first_col = group_col << 6u;
				float q_dot_sums[ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK] = {};
				float input_sums[ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK] = {};
				for (uint32_t block = 0; block < 4u; ++block) {
					rocm_mlx_affine_q6_16_batch_dot(input_base, weights, row_packed_base, first_packed + block * 3u, first_col + block * 16u, args.cols, batch_base, args.batch, q_dot_sums, input_sums);
				}
#pragma unroll
				for (uint32_t token_lane = 0; token_lane < ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK; ++token_lane) {
					sums[token_lane] += scale * q_dot_sums[token_lane] + bias * input_sums[token_lane];
				}
			}
		} else {
			for (uint32_t col = col_lane; col < args.cols; col += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
				const uint32_t q = rocm_mlx_affine_quantized_value(weights, static_cast<uint64_t>(row) * packed_per_row, col, args.bits);
				const uint32_t group = row * groups_per_row + col / args.group_size;
				const float scale = rocm_bfloat16_to_float(scales[group]);
				const float bias = rocm_bfloat16_to_float(biases[group]);
#pragma unroll
				for (uint32_t token_lane = 0; token_lane < ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK; ++token_lane) {
					const uint32_t batch = batch_base + token_lane;
					if (batch < args.batch) {
						const float *input = input_base + batch * args.cols;
						sums[token_lane] += input[col] * (static_cast<float>(q) * scale + bias);
					}
				}
			}
		}
	}
#pragma unroll
	for (uint32_t token_lane = 0; token_lane < ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK; ++token_lane) {
		const uint32_t batch = batch_base + token_lane;
		const float sum = rocm_mlx_q4_row_reduce(sums[token_lane]);
		if (col_lane == 0 && row < args.rows && batch < args.batch) {
			const float *multiplier = multiplier_base + batch * args.rows;
			float *output = output_base + batch * args.rows;
			output[row] = rocm_gelu_tanh_value(sum) * multiplier[row];
		}
	}
}

extern "C" __global__ void rocm_lora_projection(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_lora_launch_args &args = *reinterpret_cast<const rocm_lora_launch_args *>(packet);
	if (!rocm_valid_lora_args(args)) {
		return;
	}

	const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= args.rows) {
		return;
	}

	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const float *base = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.base_weight_pointer));
	const float *lora_a = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.lora_a_pointer));
	const float *lora_b = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.lora_b_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));

	float sum = 0.0f;
	for (uint32_t col = 0; col < args.cols; ++col) {
		sum += input[col] * base[row * args.cols + col];
	}
	if ((args.flags & ROCM_LORA_LAUNCH_FLAG_BIAS) != 0 && args.bias_pointer != 0) {
		const float *bias = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.bias_pointer));
		sum += bias[row];
	}
	float delta = 0.0f;
	for (uint32_t r = 0; r < args.rank; ++r) {
		float down = 0.0f;
		for (uint32_t col = 0; col < args.cols; ++col) {
			down += lora_a[r * args.cols + col] * input[col];
		}
		delta += lora_b[row * args.rank + r] * down;
	}
	const float scale = rocm_float_from_bits(args.alpha_bits) / static_cast<float>(args.rank);
	output[row] = sum + scale * delta;
}

__device__ void rocm_embedding_lookup_store(const rocm_embedding_lookup_launch_args &args, uint32_t index, int32_t token_id, uint32_t dim, float output_scale)
{
	if (token_id < 0 || static_cast<uint32_t>(token_id) >= args.vocab_size) {
		return;
	}
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float value;
	if (args.table_encoding == ROCM_EMBEDDING_TABLE_ENCODING_MLX_Q4) {
		const uint32_t *embedding = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.embedding_pointer));
		const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.scale_pointer));
		const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.bias_pointer));
		const uint32_t bits = args.bits == 0u ? ROCM_MLX_Q4_PROJECTION_BITS : args.bits;
		const uint64_t packed_per_row = rocm_mlx_affine_packed_per_row(args.hidden_size, bits);
		const uint32_t groups_per_row = args.hidden_size / args.group_size;
		const uint32_t token_u = static_cast<uint32_t>(token_id);
		const uint32_t quantized = rocm_mlx_affine_quantized_value(embedding, static_cast<uint64_t>(token_u) * packed_per_row, dim, bits);
		const uint64_t group = static_cast<uint64_t>(token_u) * groups_per_row + dim / args.group_size;
		value = static_cast<float>(quantized) * rocm_bfloat16_to_float(scales[group]) + rocm_bfloat16_to_float(biases[group]);
	} else {
		const uint64_t table_index = static_cast<uint64_t>(token_id) * args.hidden_size + dim;
		if (args.table_encoding == ROCM_EMBEDDING_TABLE_ENCODING_BF16) {
			const uint16_t *embedding = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.embedding_pointer));
			value = rocm_bfloat16_to_float(embedding[table_index]);
		} else {
			const float *embedding = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.embedding_pointer));
			value = embedding[table_index];
		}
	}
	output[index] = value * output_scale;
}

extern "C" __global__ void rocm_embedding_lookup(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_embedding_lookup_launch_args &args = *reinterpret_cast<const rocm_embedding_lookup_launch_args *>(packet);
	if (!rocm_valid_embedding_lookup_args(args)) {
		return;
	}
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t total = static_cast<uint64_t>(args.token_count) * args.hidden_size;
	if (index >= total) {
		return;
	}
	const int32_t *tokens = reinterpret_cast<const int32_t *>(static_cast<uintptr_t>(args.token_pointer));
	const uint32_t token_index = index / args.hidden_size;
	const uint32_t dim = index - token_index * args.hidden_size;
	const int32_t token_id = tokens[token_index];
	const float output_scale = args.output_scale_bits == 0 ? 1.0f : rocm_float_from_bits(args.output_scale_bits);
	rocm_embedding_lookup_store(args, index, token_id, dim, output_scale);
}

extern "C" __global__ void rocm_embedding_lookup_greedy_token(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_embedding_lookup_launch_args &args = *reinterpret_cast<const rocm_embedding_lookup_launch_args *>(packet);
	if (!rocm_valid_embedding_lookup_greedy_token_args(args)) {
		return;
	}
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= args.hidden_size) {
		return;
	}
	const uint64_t *best = reinterpret_cast<const uint64_t *>(static_cast<uintptr_t>(args.token_pointer));
	const int32_t token_id = static_cast<int32_t>(~static_cast<uint32_t>(*best));
	const float output_scale = args.output_scale_bits == 0 ? 1.0f : rocm_float_from_bits(args.output_scale_bits);
	rocm_embedding_lookup_store(args, index, token_id, index, output_scale);
}

extern "C" __global__ void rocm_diffusion_expected_embedding(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_diffusion_expected_embedding_launch_args &args = *reinterpret_cast<const rocm_diffusion_expected_embedding_launch_args *>(packet);
	if (!rocm_valid_diffusion_expected_embedding_args(args)) {
		return;
	}
	const uint32_t dim = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t row = blockIdx.y;
	if (dim >= args.hidden_size || row >= args.rows) {
		return;
	}
	const float *probabilities = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.probability_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	const uint64_t probability_base = static_cast<uint64_t>(row) * args.vocab_size;
	float sum = 0.0f;
	if (args.table_encoding == ROCM_EMBEDDING_TABLE_ENCODING_MLX_Q4) {
		const uint32_t *embedding = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.embedding_pointer));
		const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.scale_pointer));
		const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.bias_pointer));
		const uint32_t bits = args.bits == 0u ? ROCM_MLX_Q4_PROJECTION_BITS : args.bits;
		const uint64_t packed_per_row = rocm_mlx_affine_packed_per_row(args.hidden_size, bits);
		const uint32_t groups_per_row = args.hidden_size / args.group_size;
		for (uint32_t token = 0; token < args.vocab_size; ++token) {
			const uint32_t quantized = rocm_mlx_affine_quantized_value(embedding, static_cast<uint64_t>(token) * packed_per_row, dim, bits);
			const uint64_t group = static_cast<uint64_t>(token) * groups_per_row + dim / args.group_size;
			const float value = static_cast<float>(quantized) * rocm_bfloat16_to_float(scales[group]) + rocm_bfloat16_to_float(biases[group]);
			sum = fmaf(probabilities[probability_base + token], value, sum);
		}
	} else if (args.table_encoding == ROCM_EMBEDDING_TABLE_ENCODING_BF16) {
		const uint16_t *embedding = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.embedding_pointer));
		for (uint32_t token = 0; token < args.vocab_size; ++token) {
			const uint64_t index = static_cast<uint64_t>(token) * args.hidden_size + dim;
			sum = fmaf(probabilities[probability_base + token], rocm_bfloat16_to_float(embedding[index]), sum);
		}
	} else {
		const float *embedding = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.embedding_pointer));
		for (uint32_t token = 0; token < args.vocab_size; ++token) {
			const uint64_t index = static_cast<uint64_t>(token) * args.hidden_size + dim;
			sum = fmaf(probabilities[probability_base + token], embedding[index], sum);
		}
	}
	const float output_scale = args.output_scale_bits == 0 ? 1.0f : rocm_float_from_bits(args.output_scale_bits);
	output[static_cast<uint64_t>(row) * args.hidden_size + dim] = sum * output_scale;
}

extern "C" __global__ void rocm_diffusion_sample_probabilities(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_diffusion_sample_launch_args &args = *reinterpret_cast<const rocm_diffusion_sample_launch_args *>(packet);
	if (!rocm_valid_diffusion_sample_args(args) ||
		blockDim.x != ROCM_DIFFUSION_SAMPLE_BLOCK_SIZE ||
		blockIdx.x >= args.rows) {
		return;
	}

	const uint32_t lane = threadIdx.x;
	const uint32_t row = blockIdx.x;
	float *logits = reinterpret_cast<float *>(static_cast<uintptr_t>(args.logits_pointer)) + static_cast<uint64_t>(row) * args.vocab_size;
	const float *draws = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.draws_pointer));
	rocm_diffusion_sample_result *results = reinterpret_cast<rocm_diffusion_sample_result *>(static_cast<uintptr_t>(args.output_pointer));
	const float temperature = rocm_float_from_bits(args.temperature_bits);
	const float softcap = rocm_float_from_bits(args.softcap_bits);

	float local_maximum = -FLT_MAX;
	uint32_t local_greedy = args.vocab_size;
	uint32_t local_valid = 1u;
	for (uint32_t col = lane; col < args.vocab_size; col += ROCM_DIFFUSION_SAMPLE_BLOCK_SIZE) {
		float value = logits[col];
		if (softcap > 0.0f) {
			value = tanhf(value / softcap) * softcap;
		}
		value = rocm_bfloat16_to_float(rocm_float_to_bfloat16(value / temperature));
		logits[col] = value;
		if (!isfinite(value)) {
			local_valid = 0u;
		}
		if (value > local_maximum || (value == local_maximum && col < local_greedy)) {
			local_maximum = value;
			local_greedy = col;
		}
	}

	float maximum = local_maximum;
	uint32_t greedy = local_greedy;
	uint32_t valid = local_valid;
	for (uint32_t source = 0; source < ROCM_DIFFUSION_SAMPLE_BLOCK_SIZE; ++source) {
		const float candidate = rocm_shfl_float(local_maximum, static_cast<int>(source), ROCM_DIFFUSION_SAMPLE_BLOCK_SIZE);
		const uint32_t candidate_index = rocm_shfl_u32(local_greedy, static_cast<int>(source), ROCM_DIFFUSION_SAMPLE_BLOCK_SIZE);
		const uint32_t candidate_valid = rocm_shfl_u32(local_valid, static_cast<int>(source), ROCM_DIFFUSION_SAMPLE_BLOCK_SIZE);
		if (lane == 0u) {
			valid &= candidate_valid;
			if (candidate_index < args.vocab_size &&
				(candidate > maximum || (candidate == maximum && candidate_index < greedy))) {
				maximum = candidate;
				greedy = candidate_index;
			}
		}
	}
	maximum = rocm_shfl_float(maximum, 0, ROCM_DIFFUSION_SAMPLE_BLOCK_SIZE);
	greedy = rocm_shfl_u32(greedy, 0, ROCM_DIFFUSION_SAMPLE_BLOCK_SIZE);
	valid = rocm_shfl_u32(valid, 0, ROCM_DIFFUSION_SAMPLE_BLOCK_SIZE);

	float total = 0.0f;
	float weighted = 0.0f;
	for (uint32_t base = 0; base < args.vocab_size; base += ROCM_DIFFUSION_SAMPLE_BLOCK_SIZE) {
		const uint32_t col = base + lane;
		const float value = col < args.vocab_size ? logits[col] : 0.0f;
		const float probability = col < args.vocab_size ? expf(value - maximum) : 0.0f;
		if (col < args.vocab_size) {
			logits[col] = probability;
		}
		for (uint32_t source = 0; source < ROCM_DIFFUSION_SAMPLE_BLOCK_SIZE; ++source) {
			const float ordered_probability = rocm_shfl_float(probability, static_cast<int>(source), ROCM_DIFFUSION_SAMPLE_BLOCK_SIZE);
			const float ordered_value = rocm_shfl_float(value, static_cast<int>(source), ROCM_DIFFUSION_SAMPLE_BLOCK_SIZE);
			if (lane == 0u && base + source < args.vocab_size) {
				total += ordered_probability;
				weighted += ordered_probability * ordered_value;
			}
		}
	}
	total = rocm_shfl_float(total, 0, ROCM_DIFFUSION_SAMPLE_BLOCK_SIZE);
	weighted = rocm_shfl_float(weighted, 0, ROCM_DIFFUSION_SAMPLE_BLOCK_SIZE);

	const float draw = draws[row];
	const float target = draw * total;
	float accumulated = 0.0f;
	uint32_t sampled = args.vocab_size - 1u;
	uint32_t found = 0u;
	for (uint32_t base = 0; base < args.vocab_size; base += ROCM_DIFFUSION_SAMPLE_BLOCK_SIZE) {
		const uint32_t col = base + lane;
		const float probability = col < args.vocab_size ? logits[col] : 0.0f;
		if (col < args.vocab_size) {
			logits[col] = probability / total;
		}
		for (uint32_t source = 0; source < ROCM_DIFFUSION_SAMPLE_BLOCK_SIZE; ++source) {
			const float ordered_probability = rocm_shfl_float(probability, static_cast<int>(source), ROCM_DIFFUSION_SAMPLE_BLOCK_SIZE);
			if (lane == 0u && found == 0u && base + source < args.vocab_size) {
				accumulated += ordered_probability;
				if (accumulated >= target) {
					sampled = base + source;
					found = 1u;
				}
			}
		}
	}

	if (lane == 0u) {
		rocm_diffusion_sample_result result = {};
		result.sampled = static_cast<int32_t>(sampled);
		result.greedy = static_cast<int32_t>(greedy);
		result.entropy = maximum + static_cast<float>(log(static_cast<double>(total))) - weighted / total;
		if (valid != 0u && draw >= 0.0f && draw < 1.0f && isfinite(draw) && total > 0.0f && isfinite(total) && isfinite(result.entropy)) {
			result.status = ROCM_DIFFUSION_SAMPLE_STATUS_OK;
		}
		results[row] = result;
	}
}

extern "C" __global__ void rocm_diffusion_expected_embedding_affine_g64_rows16(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_diffusion_expected_embedding_launch_args &args = *reinterpret_cast<const rocm_diffusion_expected_embedding_launch_args *>(packet);
	if (!rocm_valid_diffusion_expected_embedding_args(args) ||
		args.table_encoding != ROCM_EMBEDDING_TABLE_ENCODING_MLX_Q4 ||
		args.group_size != 64u ||
		(args.bits != 4u && args.bits != 8u) ||
		blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t dim = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t row_base = blockIdx.y * ROCM_DIFFUSION_EXPECTED_EMBEDDING_AFFINE_G64_ROWS_PER_BLOCK;
	if (dim >= args.hidden_size || row_base >= args.rows) {
		return;
	}

	const float *probabilities = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.probability_pointer));
	const uint32_t *embedding = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.embedding_pointer));
	const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.scale_pointer));
	const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.bias_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	const uint64_t packed_per_row = rocm_mlx_affine_packed_per_row(args.hidden_size, args.bits);
	const uint32_t groups_per_row = args.hidden_size / args.group_size;
	float sums[ROCM_DIFFUSION_EXPECTED_EMBEDDING_AFFINE_G64_ROWS_PER_BLOCK] = {};
	for (uint32_t token = 0; token < args.vocab_size; ++token) {
		const uint32_t quantized = rocm_mlx_affine_quantized_value(embedding, static_cast<uint64_t>(token) * packed_per_row, dim, args.bits);
		const uint64_t group = static_cast<uint64_t>(token) * groups_per_row + dim / args.group_size;
		const float value = static_cast<float>(quantized) * rocm_bfloat16_to_float(scales[group]) + rocm_bfloat16_to_float(biases[group]);
#pragma unroll
		for (uint32_t row_lane = 0; row_lane < ROCM_DIFFUSION_EXPECTED_EMBEDDING_AFFINE_G64_ROWS_PER_BLOCK; ++row_lane) {
			const uint32_t row = row_base + row_lane;
			if (row < args.rows) {
				sums[row_lane] = fmaf(probabilities[static_cast<uint64_t>(row) * args.vocab_size + token], value, sums[row_lane]);
			}
		}
	}
	const float output_scale = args.output_scale_bits == 0 ? 1.0f : rocm_float_from_bits(args.output_scale_bits);
#pragma unroll
	for (uint32_t row_lane = 0; row_lane < ROCM_DIFFUSION_EXPECTED_EMBEDDING_AFFINE_G64_ROWS_PER_BLOCK; ++row_lane) {
		const uint32_t row = row_base + row_lane;
		if (row < args.rows) {
			output[static_cast<uint64_t>(row) * args.hidden_size + dim] = sums[row_lane] * output_scale;
		}
	}
}

extern "C" __global__ void rocm_diffusion_expected_embedding_q8_g64_dims4_rows4(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_diffusion_expected_embedding_launch_args &args = *reinterpret_cast<const rocm_diffusion_expected_embedding_launch_args *>(packet);
	if (!rocm_valid_diffusion_expected_embedding_args(args) ||
		args.table_encoding != ROCM_EMBEDDING_TABLE_ENCODING_MLX_Q4 ||
		args.group_size != 64u ||
		args.bits != 8u ||
		blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const uint32_t packed_dim = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t dim_base = packed_dim * ROCM_DIFFUSION_EXPECTED_EMBEDDING_Q8_G64_DIMS_PER_THREAD;
	const uint32_t row_base = blockIdx.y * ROCM_DIFFUSION_EXPECTED_EMBEDDING_Q8_G64_ROWS_PER_THREAD;
	if (dim_base >= args.hidden_size || row_base >= args.rows) {
		return;
	}

	const float *probabilities = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.probability_pointer));
	const uint32_t *embedding = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.embedding_pointer));
	const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.scale_pointer));
	const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.bias_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	const uint64_t packed_per_row = rocm_mlx_affine_packed_per_row(args.hidden_size, args.bits);
	const uint32_t groups_per_row = args.hidden_size / args.group_size;
	float sums[ROCM_DIFFUSION_EXPECTED_EMBEDDING_Q8_G64_ROWS_PER_THREAD][ROCM_DIFFUSION_EXPECTED_EMBEDDING_Q8_G64_DIMS_PER_THREAD] = {};
	for (uint32_t token = 0; token < args.vocab_size; ++token) {
		const uint32_t word = embedding[static_cast<uint64_t>(token) * packed_per_row + packed_dim];
		const uint64_t group = static_cast<uint64_t>(token) * groups_per_row + dim_base / args.group_size;
		const float scale = rocm_bfloat16_to_float(scales[group]);
		const float bias = rocm_bfloat16_to_float(biases[group]);
		const float values[ROCM_DIFFUSION_EXPECTED_EMBEDDING_Q8_G64_DIMS_PER_THREAD] = {
			static_cast<float>(word & 0xffu) * scale + bias,
			static_cast<float>((word >> 8u) & 0xffu) * scale + bias,
			static_cast<float>((word >> 16u) & 0xffu) * scale + bias,
			static_cast<float>(word >> 24u) * scale + bias,
		};
#pragma unroll
		for (uint32_t row_lane = 0; row_lane < ROCM_DIFFUSION_EXPECTED_EMBEDDING_Q8_G64_ROWS_PER_THREAD; ++row_lane) {
			const uint32_t row = row_base + row_lane;
			if (row >= args.rows) {
				continue;
			}
			const float probability = probabilities[static_cast<uint64_t>(row) * args.vocab_size + token];
#pragma unroll
			for (uint32_t dim_lane = 0; dim_lane < ROCM_DIFFUSION_EXPECTED_EMBEDDING_Q8_G64_DIMS_PER_THREAD; ++dim_lane) {
				sums[row_lane][dim_lane] = fmaf(probability, values[dim_lane], sums[row_lane][dim_lane]);
			}
		}
	}
	const float output_scale = args.output_scale_bits == 0 ? 1.0f : rocm_float_from_bits(args.output_scale_bits);
#pragma unroll
	for (uint32_t row_lane = 0; row_lane < ROCM_DIFFUSION_EXPECTED_EMBEDDING_Q8_G64_ROWS_PER_THREAD; ++row_lane) {
		const uint32_t row = row_base + row_lane;
		if (row >= args.rows) {
			continue;
		}
#pragma unroll
		for (uint32_t dim_lane = 0; dim_lane < ROCM_DIFFUSION_EXPECTED_EMBEDDING_Q8_G64_DIMS_PER_THREAD; ++dim_lane) {
			const uint32_t dim = dim_base + dim_lane;
			if (dim < args.hidden_size) {
				output[static_cast<uint64_t>(row) * args.hidden_size + dim] = sums[row_lane][dim_lane] * output_scale;
			}
		}
	}
}

extern "C" __global__ void rocm_embedding_mean_pool(const unsigned char *packet)
{
	if (blockIdx.x != 0 || threadIdx.x != 0 || packet == nullptr) {
		return;
	}
	const rocm_embedding_mean_pool_launch_args &args = *reinterpret_cast<const rocm_embedding_mean_pool_launch_args *>(packet);
	if (!rocm_valid_embedding_mean_pool_args(args)) {
		return;
	}
	const float *tokens = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.token_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	for (uint32_t dim = 0; dim < args.dim; ++dim) {
		float sum = 0.0f;
		for (uint32_t token = 0; token < args.token_count; ++token) {
			sum += tokens[token * args.dim + dim];
		}
		output[dim] = sum / static_cast<float>(args.token_count);
	}
	if ((args.flags & ROCM_EMBEDDING_MEAN_POOL_LAUNCH_FLAG_NORMALIZE) != 0) {
		float norm = 0.0f;
		for (uint32_t dim = 0; dim < args.dim; ++dim) {
			norm += output[dim] * output[dim];
		}
		if (norm > 0.0f) {
			const float scale = rocm_rsqrtf(norm);
			for (uint32_t dim = 0; dim < args.dim; ++dim) {
				output[dim] *= scale;
			}
		}
	}
}

extern "C" __global__ void rocm_rerank_cosine(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_rerank_cosine_launch_args &args = *reinterpret_cast<const rocm_rerank_cosine_launch_args *>(packet);
	if (!rocm_valid_rerank_cosine_args(args)) {
		return;
	}
	const uint32_t doc = blockIdx.x * blockDim.x + threadIdx.x;
	if (doc >= args.document_count) {
		return;
	}
	const float *query = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.query_pointer));
	const float *documents = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.document_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float dot = 0.0f;
	float query_norm = 0.0f;
	float doc_norm = 0.0f;
	const float *document = documents + doc * args.dim;
	for (uint32_t dim = 0; dim < args.dim; ++dim) {
		const float q = query[dim];
		const float d = document[dim];
		dot += q * d;
		query_norm += q * q;
		doc_norm += d * d;
	}
	if (query_norm == 0.0f || doc_norm == 0.0f) {
		output[doc] = 0.0f;
		return;
	}
	output[doc] = dot * rocm_rsqrtf(query_norm) * rocm_rsqrtf(doc_norm);
}

__device__ float rocm_block_reduce_sum(float value, float *scratch)
{
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t wave = threadIdx.x >> 5u;
	const uint32_t wave_count = (blockDim.x + 31u) >> 5u;
	for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
		value += rocm_shfl_down(value, stride, 32);
	}
	if (lane == 0u) {
		scratch[wave] = value;
	}
	__syncthreads();
	float reduced = lane < wave_count ? scratch[lane] : 0.0f;
	if (wave == 0u) {
		for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
			reduced += rocm_shfl_down(reduced, stride, 32);
		}
		if (lane == 0u) {
			scratch[0] = reduced;
		}
	}
	__syncthreads();
	return scratch[0];
}

extern "C" __global__ void rocm_rms_norm(const unsigned char *packet)
{
	if (blockIdx.x != 0 || packet == nullptr) {
		return;
	}
	const rocm_rms_norm_launch_args &args = *reinterpret_cast<const rocm_rms_norm_launch_args *>(packet);
	if (!rocm_valid_rms_norm_args(args)) {
		return;
	}
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	__shared__ float partial[32];
	float sum_squares = 0.0f;
	for (uint32_t i = threadIdx.x; i < args.count; i += blockDim.x) {
		sum_squares += input[i] * input[i];
	}
	const float total = rocm_block_reduce_sum(sum_squares, partial);
	const float rms = sqrtf(total / static_cast<float>(args.count) + rocm_float_from_bits(args.epsilon_bits));
	if (rms == 0.0f) {
		return;
	}
	if (args.weight_encoding == ROCM_RMS_NORM_WEIGHT_ENCODING_NONE) {
		for (uint32_t i = threadIdx.x; i < args.count; i += blockDim.x) {
			output[i] = input[i] / rms;
		}
	} else if (args.weight_encoding == ROCM_RMS_NORM_WEIGHT_ENCODING_BF16) {
		const uint16_t *weight = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.weight_pointer));
		for (uint32_t i = threadIdx.x; i < args.count; i += blockDim.x) {
			float w = rocm_bfloat16_to_float(weight[i]);
			if ((args.flags & ROCM_RMS_NORM_LAUNCH_FLAG_ADD_UNIT_WEIGHT) != 0) {
				w += 1.0f;
			}
			output[i] = input[i] / rms * w;
		}
	} else {
		const float *weight = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.weight_pointer));
		for (uint32_t i = threadIdx.x; i < args.count; i += blockDim.x) {
			float w = weight[i];
			if ((args.flags & ROCM_RMS_NORM_LAUNCH_FLAG_ADD_UNIT_WEIGHT) != 0) {
				w += 1.0f;
			}
			output[i] = input[i] / rms * w;
		}
	}
}

extern "C" __global__ void rocm_rms_norm_residual_add(const unsigned char *packet)
{
	if (blockIdx.x != 0 || packet == nullptr) {
		return;
	}
	const rocm_rms_norm_residual_add_launch_args &args = *reinterpret_cast<const rocm_rms_norm_residual_add_launch_args *>(packet);
	if (!rocm_valid_rms_norm_residual_add_args(args)) {
		return;
	}
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const float *residual = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.residual_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	__shared__ float partial[32];
	float sum_squares = 0.0f;
	for (uint32_t i = threadIdx.x; i < args.count; i += blockDim.x) {
		sum_squares += input[i] * input[i];
	}
	const float total = rocm_block_reduce_sum(sum_squares, partial);
	const float mean_square = total / static_cast<float>(args.count) + rocm_float_from_bits(args.epsilon_bits);
	if (mean_square == 0.0f) {
		return;
	}
	const float inv_rms = rocm_rsqrtf(mean_square);
	const float output_scale = args.output_scale_bits == 0 ? 1.0f : rocm_float_from_bits(args.output_scale_bits);
	if (args.weight_encoding == ROCM_RMS_NORM_WEIGHT_ENCODING_NONE) {
		for (uint32_t i = threadIdx.x; i < args.count; i += blockDim.x) {
			output[i] = (residual[i] + input[i] * inv_rms) * output_scale;
		}
	} else if (args.weight_encoding == ROCM_RMS_NORM_WEIGHT_ENCODING_BF16) {
		const uint16_t *weight = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.weight_pointer));
		for (uint32_t i = threadIdx.x; i < args.count; i += blockDim.x) {
			float w = rocm_bfloat16_to_float(weight[i]);
			if ((args.flags & ROCM_RMS_NORM_LAUNCH_FLAG_ADD_UNIT_WEIGHT) != 0) {
				w += 1.0f;
			}
			output[i] = (residual[i] + input[i] * inv_rms * w) * output_scale;
		}
	} else {
		const float *weight = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.weight_pointer));
		for (uint32_t i = threadIdx.x; i < args.count; i += blockDim.x) {
			float w = weight[i];
			if ((args.flags & ROCM_RMS_NORM_LAUNCH_FLAG_ADD_UNIT_WEIGHT) != 0) {
				w += 1.0f;
			}
			output[i] = (residual[i] + input[i] * inv_rms * w) * output_scale;
		}
	}
}

__device__ float rocm_rms_norm_weight_value(uint64_t pointer, uint32_t encoding, uint32_t flags, uint32_t index)
{
	float weight_value = 1.0f;
	if (encoding == ROCM_RMS_NORM_WEIGHT_ENCODING_BF16) {
		const uint16_t *weight = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(pointer));
		weight_value = rocm_bfloat16_to_float(weight[index]);
	} else if (encoding == ROCM_RMS_NORM_WEIGHT_ENCODING_F32) {
		const float *weight = reinterpret_cast<const float *>(static_cast<uintptr_t>(pointer));
		weight_value = weight[index];
	}
	if ((flags & ROCM_RMS_NORM_LAUNCH_FLAG_ADD_UNIT_WEIGHT) != 0) {
		weight_value += 1.0f;
	}
	return weight_value;
}

extern "C" __global__ void rocm_moe_combine_norms(const unsigned char *packet)
{
	if (blockIdx.x != 0 || packet == nullptr) {
		return;
	}
	const rocm_moe_combine_norms_launch_args &args = *reinterpret_cast<const rocm_moe_combine_norms_launch_args *>(packet);
	if (!rocm_valid_moe_combine_norms_args(args)) {
		return;
	}
	const float *local_input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.local_input_pointer));
	const float *expert_input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.expert_input_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	__shared__ float partial[32];
	float local_sum_squares = 0.0f;
	float expert_sum_squares = 0.0f;
	for (uint32_t index = threadIdx.x; index < args.count; index += blockDim.x) {
		const float local_value = local_input[index];
		const float expert_value = expert_input[index];
		local_sum_squares += local_value * local_value;
		expert_sum_squares += expert_value * expert_value;
	}
	const float local_total = rocm_block_reduce_sum(local_sum_squares, partial);
	const float expert_total = rocm_block_reduce_sum(expert_sum_squares, partial);
	const float local_rms = sqrtf(local_total / static_cast<float>(args.count) + rocm_float_from_bits(args.local_epsilon_bits));
	const float expert_rms = sqrtf(expert_total / static_cast<float>(args.count) + rocm_float_from_bits(args.expert_epsilon_bits));
	if (local_rms == 0.0f || expert_rms == 0.0f) {
		return;
	}
	for (uint32_t index = threadIdx.x; index < args.count; index += blockDim.x) {
		const float local_weight = rocm_rms_norm_weight_value(args.local_weight_pointer, args.local_weight_encoding, args.local_flags, index);
		const float expert_weight = rocm_rms_norm_weight_value(args.expert_weight_pointer, args.expert_weight_encoding, args.expert_flags, index);
		output[index] = local_input[index] / local_rms * local_weight + expert_input[index] / expert_rms * expert_weight;
	}
}

extern "C" __global__ void rocm_rms_norm_residual_add_norm(const unsigned char *packet)
{
	if (blockIdx.x != 0 || packet == nullptr) {
		return;
	}
	const rocm_rms_norm_residual_add_norm_launch_args &args = *reinterpret_cast<const rocm_rms_norm_residual_add_norm_launch_args *>(packet);
	if (!rocm_valid_rms_norm_residual_add_norm_args(args)) {
		return;
	}
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const float *residual = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.residual_pointer));
	float *residual_output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.residual_output_pointer));
	float *norm_output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.norm_output_pointer));
	__shared__ float partial[32];

	float sum_squares = 0.0f;
	for (uint32_t i = threadIdx.x; i < args.count; i += blockDim.x) {
		sum_squares += input[i] * input[i];
	}
	const float total = rocm_block_reduce_sum(sum_squares, partial);
	const float mean_square = total / static_cast<float>(args.count) + rocm_float_from_bits(args.epsilon_bits);
	if (mean_square == 0.0f) {
		return;
	}
	const float inv_rms = rocm_rsqrtf(mean_square);

	const float output_scale = args.output_scale_bits == 0 ? 1.0f : rocm_float_from_bits(args.output_scale_bits);
	float residual_sum_squares = 0.0f;
	for (uint32_t i = threadIdx.x; i < args.count; i += blockDim.x) {
		const float weight = rocm_rms_norm_weight_value(args.weight_pointer, args.weight_encoding, args.flags, i);
		const float value = (residual[i] + input[i] * inv_rms * weight) * output_scale;
		residual_output[i] = value;
		residual_sum_squares += value * value;
	}
	const float residual_total = rocm_block_reduce_sum(residual_sum_squares, partial);
	const float norm_mean_square = residual_total / static_cast<float>(args.count) + rocm_float_from_bits(args.norm_epsilon_bits);
	if (norm_mean_square == 0.0f) {
		return;
	}
	const float norm_inv_rms = rocm_rsqrtf(norm_mean_square);
	for (uint32_t i = threadIdx.x; i < args.count; i += blockDim.x) {
		const float weight = rocm_rms_norm_weight_value(args.norm_weight_pointer, args.norm_weight_encoding, args.norm_flags, i);
		norm_output[i] = residual_output[i] * norm_inv_rms * weight;
	}
}

extern "C" __global__ void rocm_rms_norm_heads(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_rms_norm_heads_launch_args &args = *reinterpret_cast<const rocm_rms_norm_heads_launch_args *>(packet);
	if (!rocm_valid_rms_norm_heads_args(args) || blockIdx.x >= args.head_count) {
		return;
	}
	const uint32_t head = blockIdx.x;
	const uint32_t head_offset = head * args.head_dim;
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer)) + head_offset;
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer)) + head_offset;
	__shared__ float partial[32];
	float sum_squares = 0.0f;
	for (uint32_t i = threadIdx.x; i < args.head_dim; i += blockDim.x) {
		sum_squares += input[i] * input[i];
	}
	const float total = rocm_block_reduce_sum(sum_squares, partial);
	const float rms = sqrtf(total / static_cast<float>(args.head_dim) + rocm_float_from_bits(args.epsilon_bits));
	if (rms == 0.0f) {
		return;
	}
	if (args.weight_encoding == ROCM_RMS_NORM_WEIGHT_ENCODING_NONE) {
		for (uint32_t i = threadIdx.x; i < args.head_dim; i += blockDim.x) {
			output[i] = input[i] / rms;
		}
	} else if (args.weight_encoding == ROCM_RMS_NORM_WEIGHT_ENCODING_BF16) {
		const uint16_t *weight = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.weight_pointer));
		for (uint32_t i = threadIdx.x; i < args.head_dim; i += blockDim.x) {
			float w = rocm_bfloat16_to_float(weight[i]);
			if ((args.flags & ROCM_RMS_NORM_LAUNCH_FLAG_ADD_UNIT_WEIGHT) != 0) {
				w += 1.0f;
			}
			output[i] = input[i] / rms * w;
		}
	} else {
		const float *weight = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.weight_pointer));
		for (uint32_t i = threadIdx.x; i < args.head_dim; i += blockDim.x) {
			float w = weight[i];
			if ((args.flags & ROCM_RMS_NORM_LAUNCH_FLAG_ADD_UNIT_WEIGHT) != 0) {
				w += 1.0f;
			}
			output[i] = input[i] / rms * w;
		}
	}
}

extern "C" __global__ void rocm_rms_norm_rope_heads(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_rms_norm_rope_heads_launch_args &args = *reinterpret_cast<const rocm_rms_norm_rope_heads_launch_args *>(packet);
	if (!rocm_valid_rms_norm_rope_heads_args(args) || blockIdx.x >= args.head_count) {
		return;
	}
	const uint32_t head = blockIdx.x;
	const uint32_t head_offset = head * args.head_dim;
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer)) + head_offset;
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer)) + head_offset;
	__shared__ float partial[32];
	float sum_squares = 0.0f;
	for (uint32_t i = threadIdx.x; i < args.head_dim; i += blockDim.x) {
		sum_squares += input[i] * input[i];
	}
	const float total = rocm_block_reduce_sum(sum_squares, partial);
	const float rms = sqrtf(total / static_cast<float>(args.head_dim) + rocm_float_from_bits(args.epsilon_bits));
	if (rms == 0.0f) {
		return;
	}
	const uint32_t frequency_dim = args.frequency_dim == 0u ? args.head_dim : args.frequency_dim;
		const uint32_t rotary_count = args.rotary_count == 0u ? args.head_dim : args.rotary_count;
		const float base = rocm_float_from_bits(args.base_bits);
		const float frequency_scale = rocm_float_from_bits(args.frequency_scale_bits);
		const float dim = static_cast<float>(frequency_dim);
		const uint32_t half_head_dim = args.head_dim >> 1u;
		const bool neox = (args.flags & ROCM_RMS_NORM_LAUNCH_FLAG_ROPE_NEOX) != 0u;
		if (neox) {
			const uint32_t active_pairs = rotary_count >> 1u;
			for (uint32_t pair = threadIdx.x; pair < half_head_dim; pair += blockDim.x) {
				const uint32_t i = pair;
				const uint32_t j = pair + half_head_dim;
				const float weight_x = rocm_rms_norm_weight_value(args.weight_pointer, args.weight_encoding, args.flags, i);
				const float weight_y = rocm_rms_norm_weight_value(args.weight_pointer, args.weight_encoding, args.flags, j);
				const float x = input[i] / rms * weight_x;
				const float y = input[j] / rms * weight_y;
				if (pair < active_pairs) {
					const float frequency = 1.0f / powf(base, static_cast<float>(pair << 1u) / dim);
					const float angle = static_cast<float>(args.position) * frequency * frequency_scale;
					const float cosine = cosf(angle);
					const float sine = sinf(angle);
					output[i] = x * cosine - y * sine;
					output[j] = x * sine + y * cosine;
				} else {
					output[i] = x;
					output[j] = y;
				}
			}
			return;
		}
		for (uint32_t pair = threadIdx.x; pair < half_head_dim; pair += blockDim.x) {
			const uint32_t i = pair << 1u;
			const float weight_x = rocm_rms_norm_weight_value(args.weight_pointer, args.weight_encoding, args.flags, i);
			const float weight_y = rocm_rms_norm_weight_value(args.weight_pointer, args.weight_encoding, args.flags, i + 1u);
			const float x = input[i] / rms * weight_x;
			const float y = input[i + 1u] / rms * weight_y;
			if (i < rotary_count) {
				const float frequency = 1.0f / powf(base, static_cast<float>(i) / dim);
				const float angle = static_cast<float>(args.position) * frequency * frequency_scale;
				const float cosine = cosf(angle);
				const float sine = sinf(angle);
			output[i] = x * cosine - y * sine;
			output[i + 1u] = x * sine + y * cosine;
		} else {
			output[i] = x;
			output[i + 1u] = y;
		}
	}
}

extern "C" __global__ void rocm_rms_norm_rope_heads_pair(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_rms_norm_rope_heads_pair_launch_args &args = *reinterpret_cast<const rocm_rms_norm_rope_heads_pair_launch_args *>(packet);
	if (!rocm_valid_rms_norm_rope_heads_pair_args(args) || blockIdx.x >= args.query_head_count + args.key_head_count) {
		return;
	}
	const bool query = blockIdx.x < args.query_head_count;
	const uint32_t head = query ? blockIdx.x : blockIdx.x - args.query_head_count;
	const uint32_t head_offset = head * args.head_dim;
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(query ? args.query_input_pointer : args.key_input_pointer)) + head_offset;
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(query ? args.query_output_pointer : args.key_output_pointer)) + head_offset;
	const uint64_t weight_pointer = query ? args.query_weight_pointer : args.key_weight_pointer;
	const uint32_t weight_encoding = query ? args.query_weight_encoding : args.key_weight_encoding;
	const uint32_t flags = query ? args.query_flags : args.key_flags;
	const uint32_t epsilon_bits = query ? args.query_epsilon_bits : args.key_epsilon_bits;
	__shared__ float partial[32];
	float sum_squares = 0.0f;
	for (uint32_t i = threadIdx.x; i < args.head_dim; i += blockDim.x) {
		sum_squares += input[i] * input[i];
	}
	const float total = rocm_block_reduce_sum(sum_squares, partial);
	const float rms = sqrtf(total / static_cast<float>(args.head_dim) + rocm_float_from_bits(epsilon_bits));
	if (rms == 0.0f) {
		return;
	}
	const uint32_t frequency_dim = args.frequency_dim == 0u ? args.head_dim : args.frequency_dim;
	const uint32_t rotary_count = args.rotary_count == 0u ? args.head_dim : args.rotary_count;
	const float base = rocm_float_from_bits(args.base_bits);
	const float frequency_scale = rocm_float_from_bits(args.frequency_scale_bits);
	const float dim = static_cast<float>(frequency_dim);
	const uint32_t half_head_dim = args.head_dim >> 1u;
	const bool neox = (flags & ROCM_RMS_NORM_LAUNCH_FLAG_ROPE_NEOX) != 0u;
	if (neox) {
		const uint32_t active_pairs = rotary_count >> 1u;
		for (uint32_t pair = threadIdx.x; pair < half_head_dim; pair += blockDim.x) {
			const uint32_t i = pair;
			const uint32_t j = pair + half_head_dim;
			const float weight_x = rocm_rms_norm_weight_value(weight_pointer, weight_encoding, flags, i);
			const float weight_y = rocm_rms_norm_weight_value(weight_pointer, weight_encoding, flags, j);
			const float x = input[i] / rms * weight_x;
			const float y = input[j] / rms * weight_y;
			if (pair < active_pairs) {
				const float frequency = 1.0f / powf(base, static_cast<float>(pair << 1u) / dim);
				const float angle = static_cast<float>(args.position) * frequency * frequency_scale;
				const float cosine = cosf(angle);
				const float sine = sinf(angle);
				output[i] = x * cosine - y * sine;
				output[j] = x * sine + y * cosine;
			} else {
				output[i] = x;
				output[j] = y;
			}
		}
		return;
	}
	for (uint32_t pair = threadIdx.x; pair < half_head_dim; pair += blockDim.x) {
		const uint32_t i = pair << 1u;
		const float weight_x = rocm_rms_norm_weight_value(weight_pointer, weight_encoding, flags, i);
		const float weight_y = rocm_rms_norm_weight_value(weight_pointer, weight_encoding, flags, i + 1u);
		const float x = input[i] / rms * weight_x;
		const float y = input[i + 1u] / rms * weight_y;
		if (i < rotary_count) {
			const float frequency = 1.0f / powf(base, static_cast<float>(i) / dim);
			const float angle = static_cast<float>(args.position) * frequency * frequency_scale;
			const float cosine = cosf(angle);
			const float sine = sinf(angle);
			output[i] = x * cosine - y * sine;
			output[i + 1u] = x * sine + y * cosine;
		} else {
			output[i] = x;
			output[i + 1u] = y;
		}
	}
}

__device__ void rocm_rms_norm_rope_heads_pair_lane_batch_execute(
	const float *input,
	float *output,
	uint64_t weight_pointer,
	uint32_t weight_encoding,
	uint32_t flags,
	uint32_t epsilon_bits,
	uint32_t head_dim,
	uint32_t position,
	float base,
	uint32_t frequency_dim,
	uint32_t rotary_count,
	float frequency_scale,
	float *partial)
{
	float sum_squares = 0.0f;
	for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
		sum_squares += input[i] * input[i];
	}
	const float total = rocm_block_reduce_sum(sum_squares, partial);
	const float rms = sqrtf(total / static_cast<float>(head_dim) + rocm_float_from_bits(epsilon_bits));
	if (rms == 0.0f) {
		return;
	}
	const uint32_t effective_frequency_dim = frequency_dim == 0u ? head_dim : frequency_dim;
	const uint32_t effective_rotary_count = rotary_count == 0u ? head_dim : rotary_count;
	const float dim = static_cast<float>(effective_frequency_dim);
	const uint32_t half_head_dim = head_dim >> 1u;
	const bool neox = (flags & ROCM_RMS_NORM_LAUNCH_FLAG_ROPE_NEOX) != 0u;
	if (neox) {
		const uint32_t active_pairs = effective_rotary_count >> 1u;
		for (uint32_t pair = threadIdx.x; pair < half_head_dim; pair += blockDim.x) {
			const uint32_t i = pair;
			const uint32_t j = pair + half_head_dim;
			const float weight_x = rocm_rms_norm_weight_value(weight_pointer, weight_encoding, flags, i);
			const float weight_y = rocm_rms_norm_weight_value(weight_pointer, weight_encoding, flags, j);
			const float x = input[i] / rms * weight_x;
			const float y = input[j] / rms * weight_y;
			if (pair < active_pairs) {
				const float frequency = 1.0f / powf(base, static_cast<float>(pair << 1u) / dim);
				const float angle = static_cast<float>(position) * frequency * frequency_scale;
				const float cosine = cosf(angle);
				const float sine = sinf(angle);
				output[i] = x * cosine - y * sine;
				output[j] = x * sine + y * cosine;
			} else {
				output[i] = x;
				output[j] = y;
			}
		}
		return;
	}
	for (uint32_t pair = threadIdx.x; pair < half_head_dim; pair += blockDim.x) {
		const uint32_t i = pair << 1u;
		const float weight_x = rocm_rms_norm_weight_value(weight_pointer, weight_encoding, flags, i);
		const float weight_y = rocm_rms_norm_weight_value(weight_pointer, weight_encoding, flags, i + 1u);
		const float x = input[i] / rms * weight_x;
		const float y = input[i + 1u] / rms * weight_y;
		if (i < effective_rotary_count) {
			const float frequency = 1.0f / powf(base, static_cast<float>(i) / dim);
			const float angle = static_cast<float>(position) * frequency * frequency_scale;
			const float cosine = cosf(angle);
			const float sine = sinf(angle);
			output[i] = x * cosine - y * sine;
			output[i + 1u] = x * sine + y * cosine;
		} else {
			output[i] = x;
			output[i + 1u] = y;
		}
	}
}

extern "C" __global__ void rocm_rms_norm_rope_heads_pair_lane_batch(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_rms_norm_rope_heads_pair_lane_batch_launch_args &args = *reinterpret_cast<const rocm_rms_norm_rope_heads_pair_lane_batch_launch_args *>(packet);
	if (!rocm_valid_rms_norm_rope_heads_pair_lane_batch_args(args) || blockIdx.y >= args.batch) {
		return;
	}
	const uint32_t head = blockIdx.x;
	if (head >= args.query_head_count && head >= args.key_head_count) {
		return;
	}
	const uint32_t batch = blockIdx.y;
	const uint32_t *positions = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.positions_pointer));
	const uint32_t position = positions[batch];
	const float base = rocm_float_from_bits(args.base_bits);
	const float frequency_scale = rocm_float_from_bits(args.frequency_scale_bits);
	__shared__ float partial[32];
	if (head < args.query_head_count) {
		const uint64_t offset = (static_cast<uint64_t>(batch) * args.query_head_count + head) * args.head_dim;
		const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.query_input_pointer)) + offset;
		float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.query_output_pointer)) + offset;
		rocm_rms_norm_rope_heads_pair_lane_batch_execute(input, output, args.query_weight_pointer, args.query_weight_encoding, args.query_flags, args.query_epsilon_bits, args.head_dim, position, base, args.frequency_dim, args.rotary_count, frequency_scale, partial);
	}
	__syncthreads();
	if (head < args.key_head_count) {
		const uint64_t offset = (static_cast<uint64_t>(batch) * args.key_head_count + head) * args.head_dim;
		const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.key_input_pointer)) + offset;
		float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.key_output_pointer)) + offset;
		rocm_rms_norm_rope_heads_pair_lane_batch_execute(input, output, args.key_weight_pointer, args.key_weight_encoding, args.key_flags, args.key_epsilon_bits, args.head_dim, position, base, args.frequency_dim, args.rotary_count, frequency_scale, partial);
	}
}

extern "C" __global__ void rocm_rms_norm_rope_heads_batch(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_rms_norm_rope_heads_batch_launch_args &args = *reinterpret_cast<const rocm_rms_norm_rope_heads_batch_launch_args *>(packet);
	if (!rocm_valid_rms_norm_rope_heads_batch_args(args) || blockIdx.x >= args.head_count || blockIdx.y >= args.batch) {
		return;
	}
	const uint32_t head = blockIdx.x;
	const uint32_t batch = blockIdx.y;
	const uint32_t head_offset = (batch * args.head_count + head) * args.head_dim;
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer)) + head_offset;
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer)) + head_offset;
	__shared__ float partial[32];
	float sum_squares = 0.0f;
	for (uint32_t i = threadIdx.x; i < args.head_dim; i += blockDim.x) {
		sum_squares += input[i] * input[i];
	}
	const float total = rocm_block_reduce_sum(sum_squares, partial);
	const float rms = sqrtf(total / static_cast<float>(args.head_dim) + rocm_float_from_bits(args.epsilon_bits));
	if (rms == 0.0f) {
		return;
	}
	const uint32_t frequency_dim = args.frequency_dim == 0u ? args.head_dim : args.frequency_dim;
		const uint32_t rotary_count = args.rotary_count == 0u ? args.head_dim : args.rotary_count;
		const float base = rocm_float_from_bits(args.base_bits);
		const float frequency_scale = rocm_float_from_bits(args.frequency_scale_bits);
		const float dim = static_cast<float>(frequency_dim);
		const uint32_t position = args.start_position + batch;
		const uint32_t half_head_dim = args.head_dim >> 1u;
		const bool neox = (args.flags & ROCM_RMS_NORM_LAUNCH_FLAG_ROPE_NEOX) != 0u;
		if (neox) {
			const uint32_t active_pairs = rotary_count >> 1u;
			for (uint32_t pair = threadIdx.x; pair < half_head_dim; pair += blockDim.x) {
				const uint32_t i = pair;
				const uint32_t j = pair + half_head_dim;
				const float weight_x = rocm_rms_norm_weight_value(args.weight_pointer, args.weight_encoding, args.flags, i);
				const float weight_y = rocm_rms_norm_weight_value(args.weight_pointer, args.weight_encoding, args.flags, j);
				const float x = input[i] / rms * weight_x;
				const float y = input[j] / rms * weight_y;
				if (pair < active_pairs) {
					const float frequency = 1.0f / powf(base, static_cast<float>(pair << 1u) / dim);
					const float angle = static_cast<float>(position) * frequency * frequency_scale;
					const float cosine = cosf(angle);
					const float sine = sinf(angle);
					output[i] = x * cosine - y * sine;
					output[j] = x * sine + y * cosine;
				} else {
					output[i] = x;
					output[j] = y;
				}
			}
			return;
		}
		for (uint32_t pair = threadIdx.x; pair < half_head_dim; pair += blockDim.x) {
			const uint32_t i = pair << 1u;
			const float weight_x = rocm_rms_norm_weight_value(args.weight_pointer, args.weight_encoding, args.flags, i);
			const float weight_y = rocm_rms_norm_weight_value(args.weight_pointer, args.weight_encoding, args.flags, i + 1u);
			const float x = input[i] / rms * weight_x;
			const float y = input[i + 1u] / rms * weight_y;
			if (i < rotary_count) {
				const float frequency = 1.0f / powf(base, static_cast<float>(i) / dim);
				const float angle = static_cast<float>(position) * frequency * frequency_scale;
				const float cosine = cosf(angle);
				const float sine = sinf(angle);
			output[i] = x * cosine - y * sine;
			output[i + 1u] = x * sine + y * cosine;
		} else {
			output[i] = x;
			output[i + 1u] = y;
		}
	}
}

__device__ float rocm_block_reduce_sum(float value, float *scratch);
__device__ float rocm_rms_norm_weight_value(uint64_t pointer, uint32_t encoding, uint32_t flags, uint32_t index);

__device__ float rocm_rms_residual_added_value(
	const float *input,
	const float *residual,
	uint64_t weight_pointer,
	uint32_t weight_encoding,
	uint32_t flags,
	float rms,
	float output_scale,
	uint32_t index)
{
	const float weight = rocm_rms_norm_weight_value(weight_pointer, weight_encoding, flags, index);
	return (residual[index] + input[index] / rms * weight) * output_scale;
}

__device__ float rocm_mlx_q4_projection_rms_residual_row_sum(
	const float *input,
	const float *residual,
	const uint32_t *weights,
	const uint16_t *scales,
	const uint16_t *biases,
	uint64_t rms_weight_pointer,
	uint32_t rms_weight_encoding,
	uint32_t rms_flags,
	float rms,
	float output_scale,
	uint32_t row,
	uint32_t cols,
	uint32_t group_size,
	uint32_t bits,
	uint32_t col_lane)
{
	const uint64_t packed_per_row = rocm_mlx_affine_packed_per_row(cols, bits);
	const uint64_t row_packed_base = static_cast<uint64_t>(row) * packed_per_row;
	float sum = 0.0f;
	if (bits == 4u && (group_size & 7u) == 0u) {
		const uint32_t groups_per_row = cols / group_size;
		const uint32_t packed_per_group = group_size >> 3u;
		const uint32_t row_group_base = row * groups_per_row;
		for (uint32_t group_col = col_lane; group_col < groups_per_row; group_col += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
			const uint32_t group = row_group_base + group_col;
			const float scale = rocm_bfloat16_to_float(scales[group]);
			const float bias = rocm_bfloat16_to_float(biases[group]);
			const uint32_t first_packed = group_col * packed_per_group;
			float q_dot_sum = 0.0f;
			float input_sum = 0.0f;
			for (uint32_t group_packed = 0; group_packed < packed_per_group; ++group_packed) {
				const uint32_t packed = first_packed + group_packed;
				const uint32_t word = weights[row_packed_base + packed];
				const uint32_t col = packed << 3;
				const float in0 = rocm_rms_residual_added_value(input, residual, rms_weight_pointer, rms_weight_encoding, rms_flags, rms, output_scale, col);
				const float in1 = rocm_rms_residual_added_value(input, residual, rms_weight_pointer, rms_weight_encoding, rms_flags, rms, output_scale, col + 1u);
				const float in2 = rocm_rms_residual_added_value(input, residual, rms_weight_pointer, rms_weight_encoding, rms_flags, rms, output_scale, col + 2u);
				const float in3 = rocm_rms_residual_added_value(input, residual, rms_weight_pointer, rms_weight_encoding, rms_flags, rms, output_scale, col + 3u);
				const float in4 = rocm_rms_residual_added_value(input, residual, rms_weight_pointer, rms_weight_encoding, rms_flags, rms, output_scale, col + 4u);
				const float in5 = rocm_rms_residual_added_value(input, residual, rms_weight_pointer, rms_weight_encoding, rms_flags, rms, output_scale, col + 5u);
				const float in6 = rocm_rms_residual_added_value(input, residual, rms_weight_pointer, rms_weight_encoding, rms_flags, rms, output_scale, col + 6u);
				const float in7 = rocm_rms_residual_added_value(input, residual, rms_weight_pointer, rms_weight_encoding, rms_flags, rms, output_scale, col + 7u);
				input_sum += in0 + in1 + in2 + in3 + in4 + in5 + in6 + in7;
				q_dot_sum +=
					static_cast<float>(word & 0x0fu) * in0 +
					static_cast<float>((word >> 4) & 0x0fu) * in1 +
					static_cast<float>((word >> 8) & 0x0fu) * in2 +
					static_cast<float>((word >> 12) & 0x0fu) * in3 +
					static_cast<float>((word >> 16) & 0x0fu) * in4 +
					static_cast<float>((word >> 20) & 0x0fu) * in5 +
					static_cast<float>((word >> 24) & 0x0fu) * in6 +
					static_cast<float>((word >> 28) & 0x0fu) * in7;
			}
			sum += scale * q_dot_sum + bias * input_sum;
		}
		return sum;
	}
	const uint32_t groups_per_row = cols / group_size;
	const uint32_t row_group_base = row * groups_per_row;
	for (uint32_t col = col_lane; col < cols; col += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
		const uint32_t q = rocm_mlx_affine_quantized_value(weights, row_packed_base, col, bits);
		const uint32_t group = row_group_base + col / group_size;
		const float scale = rocm_bfloat16_to_float(scales[group]);
		const float bias = rocm_bfloat16_to_float(biases[group]);
		const float value = rocm_rms_residual_added_value(input, residual, rms_weight_pointer, rms_weight_encoding, rms_flags, rms, output_scale, col);
		sum += value * (static_cast<float>(q) * scale + bias);
	}
	return sum;
}

extern "C" __global__ void rocm_rms_norm_residual_add_mlx_q4_gelu_tanh_projection(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_rms_norm_residual_add_gelu_tanh_proj_launch_args &args = *reinterpret_cast<const rocm_rms_norm_residual_add_gelu_tanh_proj_launch_args *>(packet);
	if (!rocm_valid_rms_norm_residual_add_gelu_tanh_proj_args(args) || blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE) {
		return;
	}
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const float *residual = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.residual_pointer));
	float *residual_output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.residual_output_pointer));
	float *activation_output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.activation_output_pointer));
	const uint32_t *weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.weight_pointer));
	const uint16_t *scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.scale_pointer));
	const uint16_t *biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(args.bias_pointer));
	const float *multiplier = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.multiplier_pointer));

	__shared__ float partial[32];
	float sum_squares = 0.0f;
	for (uint32_t i = threadIdx.x; i < args.cols; i += blockDim.x) {
		sum_squares += input[i] * input[i];
	}
	const float total = rocm_block_reduce_sum(sum_squares, partial);
	const float rms = sqrtf(total / static_cast<float>(args.cols) + rocm_float_from_bits(args.epsilon_bits));
	if (rms == 0.0f) {
		return;
	}
	const float output_scale = rocm_float_from_bits(args.output_scale_bits);
	if (blockIdx.x == 0) {
		for (uint32_t i = threadIdx.x; i < args.cols; i += blockDim.x) {
			residual_output[i] = rocm_rms_residual_added_value(input, residual, args.rms_weight_pointer, args.rms_weight_encoding, args.rms_flags, rms, output_scale, i);
		}
	}

	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_ROWS_PER_BLOCK + row_lane;
	float sum = 0.0f;
	if (row < args.rows) {
		sum = rocm_mlx_q4_projection_rms_residual_row_sum(input, residual, weights, scales, biases, args.rms_weight_pointer, args.rms_weight_encoding, args.rms_flags, rms, output_scale, row, args.cols, args.group_size, args.bits, col_lane);
	}
	sum = rocm_mlx_q4_row_reduce(sum);
	if (col_lane == 0 && row < args.rows) {
		activation_output[row] = rocm_gelu_tanh_value(sum) * multiplier[row];
	}
}

extern "C" __global__ void rocm_rope(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_rope_launch_args &args = *reinterpret_cast<const rocm_rope_launch_args *>(packet);
	if (!rocm_valid_rope_args(args)) {
		return;
	}
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	const float base = rocm_float_from_bits(args.base_bits);
	const uint32_t frequency_dim = args.frequency_dim == 0 ? args.count : args.frequency_dim;
	const uint32_t rotary_count = args.reserved2 == 0 ? args.count : static_cast<uint32_t>(args.reserved2);
	const float dim = static_cast<float>(frequency_dim);
	const uint32_t pair_count = args.count >> 1;
	for (uint32_t pair = blockIdx.x * blockDim.x + threadIdx.x; pair < pair_count; pair += blockDim.x * gridDim.x) {
		const uint32_t i = pair << 1;
		const float x = input[i];
		const float y = input[i + 1];
		if (i < rotary_count) {
			const float frequency = 1.0f / powf(base, static_cast<float>(i) / dim);
			const float angle = static_cast<float>(args.position) * frequency;
			const float cosine = cosf(angle);
			const float sine = sinf(angle);
			output[i] = x * cosine - y * sine;
			output[i + 1] = x * sine + y * cosine;
		} else {
			output[i] = x;
			output[i + 1] = y;
		}
	}
}

extern "C" __global__ void rocm_rope_heads(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_rope_heads_launch_args &args = *reinterpret_cast<const rocm_rope_heads_launch_args *>(packet);
	if (!rocm_valid_rope_heads_args(args)) {
		return;
	}
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	const float base = rocm_float_from_bits(args.base_bits);
	const uint32_t frequency_dim = args.frequency_dim == 0 ? args.head_dim : args.frequency_dim;
	const uint32_t rotary_count = args.rotary_count == 0 ? args.head_dim : args.rotary_count;
	const float dim = static_cast<float>(frequency_dim);
	const uint32_t pairs_per_head = args.head_dim >> 1;
	const uint32_t total_pairs = args.head_count * pairs_per_head;
	for (uint32_t global_pair = blockIdx.x * blockDim.x + threadIdx.x; global_pair < total_pairs; global_pair += blockDim.x * gridDim.x) {
		const uint32_t head = global_pair / pairs_per_head;
		const uint32_t pair = global_pair - head * pairs_per_head;
		const uint32_t local_i = pair << 1;
		const uint32_t i = head * args.head_dim + local_i;
		const float x = input[i];
		const float y = input[i + 1];
		if (local_i < rotary_count) {
			const float frequency = 1.0f / powf(base, static_cast<float>(local_i) / dim);
			const float angle = static_cast<float>(args.position) * frequency;
			const float cosine = cosf(angle);
			const float sine = sinf(angle);
			output[i] = x * cosine - y * sine;
			output[i + 1] = x * sine + y * cosine;
		} else {
			output[i] = x;
			output[i + 1] = y;
		}
	}
}

extern "C" __global__ void rocm_greedy_sample(const unsigned char *packet)
{
	if (blockIdx.x != 0 || threadIdx.x != 0 || packet == nullptr) {
		return;
	}
	const rocm_greedy_launch_args &args = *reinterpret_cast<const rocm_greedy_launch_args *>(packet);
	if (!rocm_valid_greedy_args(args)) {
		return;
	}
	const float *logits = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.logits_pointer));
	int32_t best_index = 0;
	float best_score = logits[0];
	for (uint32_t i = 1; i < args.count; ++i) {
		if (logits[i] > best_score) {
			best_index = static_cast<int32_t>(i);
			best_score = logits[i];
		}
	}
	const uintptr_t output_pointer = static_cast<uintptr_t>(args.output_pointer);
	int32_t *output_index = reinterpret_cast<int32_t *>(output_pointer);
	float *output_score = reinterpret_cast<float *>(output_pointer + sizeof(int32_t));
	*output_index = best_index;
	*output_score = best_score;
}

extern "C" __global__ void rocm_softcap_greedy_sample(const unsigned char *packet)
{
	if (blockIdx.x != 0 || packet == nullptr) {
		return;
	}
	const rocm_softcap_greedy_launch_args &args = *reinterpret_cast<const rocm_softcap_greedy_launch_args *>(packet);
	if (!rocm_valid_softcap_greedy_args(args)) {
		return;
	}
	const uint32_t tid = threadIdx.x;
	const uint32_t threads = blockDim.x;
	if (threads == 0 || threads > 256) {
		return;
	}
	__shared__ float scores[256];
	__shared__ int32_t indices[256];
	float *logits = reinterpret_cast<float *>(static_cast<uintptr_t>(args.logits_pointer));
	const int32_t *suppress_tokens = reinterpret_cast<const int32_t *>(static_cast<uintptr_t>(args.suppress_pointer));
	for (uint32_t suppress_index = tid; suppress_index < args.suppress_count; suppress_index += threads) {
		const int32_t token = suppress_tokens[suppress_index];
		if (token >= 0 && static_cast<uint32_t>(token) < args.count) {
			logits[token] = -FLT_MAX;
		}
	}
	__syncthreads();
	float best_score = -FLT_MAX;
	int32_t best_index = -1;
	for (uint32_t index = tid; index < args.count; index += threads) {
		const float score = logits[index];
		if (best_index < 0 || score > best_score || (score == best_score && static_cast<int32_t>(index) < best_index)) {
			best_score = score;
			best_index = static_cast<int32_t>(index);
		}
	}
	scores[tid] = best_score;
	indices[tid] = best_index;
	__syncthreads();
	for (uint32_t stride = threads >> 1; stride > 0; stride >>= 1) {
		if (tid < stride) {
			const float other_score = scores[tid + stride];
			const int32_t other_index = indices[tid + stride];
			if (other_index >= 0 && (indices[tid] < 0 || other_score > scores[tid] || (other_score == scores[tid] && other_index < indices[tid]))) {
				scores[tid] = other_score;
				indices[tid] = other_index;
			}
		}
		__syncthreads();
	}
	if (tid == 0) {
		float score = scores[0];
		const float softcap = rocm_float_from_bits(args.softcap_bits);
		if (softcap > 0.0f) {
			score = tanhf(score / softcap) * softcap;
		}
		const uintptr_t output_pointer = static_cast<uintptr_t>(args.output_pointer);
		int32_t *output_index = reinterpret_cast<int32_t *>(output_pointer);
		float *output_score = reinterpret_cast<float *>(output_pointer + sizeof(int32_t));
		*output_index = indices[0];
		*output_score = score;
	}
}

extern "C" __global__ void rocm_cross_entropy_loss(const unsigned char *packet)
{
	if (blockIdx.x != 0 || threadIdx.x != 0 || packet == nullptr) {
		return;
	}
	const rocm_cross_entropy_loss_launch_args &args = *reinterpret_cast<const rocm_cross_entropy_loss_launch_args *>(packet);
	if (!rocm_valid_cross_entropy_loss_args(args)) {
		return;
	}
	const float *logits = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.logit_pointer));
	const int32_t *targets = reinterpret_cast<const int32_t *>(static_cast<uintptr_t>(args.target_pointer));
	double total = 0.0;
	for (uint32_t batch = 0; batch < args.batch; ++batch) {
		const float *row = logits + batch * args.vocab;
		const int32_t target = targets[batch];
		if (target < 0 || static_cast<uint32_t>(target) >= args.vocab) {
			return;
		}
		float max_value = row[0];
		for (uint32_t col = 1; col < args.vocab; ++col) {
			max_value = fmaxf(max_value, row[col]);
		}
		double sum = 0.0;
		for (uint32_t col = 0; col < args.vocab; ++col) {
			sum += exp(static_cast<double>(row[col] - max_value));
		}
		total += static_cast<double>(max_value) + log(sum) - static_cast<double>(row[target]);
	}
	const double loss = total / static_cast<double>(args.batch);
	double *output = reinterpret_cast<double *>(static_cast<uintptr_t>(args.output_pointer));
	output[0] = loss;
	output[1] = exp(loss);
}

extern "C" __global__ void rocm_distillation_kl_loss(const unsigned char *packet)
{
	if (blockIdx.x != 0 || threadIdx.x != 0 || packet == nullptr) {
		return;
	}
	const rocm_distillation_kl_loss_launch_args &args = *reinterpret_cast<const rocm_distillation_kl_loss_launch_args *>(packet);
	if (!rocm_valid_distillation_kl_loss_args(args)) {
		return;
	}
	const float *student = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.student_pointer));
	const float *teacher = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.teacher_pointer));
	const double temperature = rocm_double_from_bits(args.temperature_bits);
	double total = 0.0;
	for (uint32_t batch = 0; batch < args.batch; ++batch) {
		const float *student_row = student + batch * args.vocab;
		const float *teacher_row = teacher + batch * args.vocab;
		double student_max = static_cast<double>(student_row[0]) / temperature;
		double teacher_max = static_cast<double>(teacher_row[0]) / temperature;
		for (uint32_t col = 1; col < args.vocab; ++col) {
			student_max = fmax(student_max, static_cast<double>(student_row[col]) / temperature);
			teacher_max = fmax(teacher_max, static_cast<double>(teacher_row[col]) / temperature);
		}
		double student_sum = 0.0;
		double teacher_sum = 0.0;
		for (uint32_t col = 0; col < args.vocab; ++col) {
			student_sum += exp(static_cast<double>(student_row[col]) / temperature - student_max);
			teacher_sum += exp(static_cast<double>(teacher_row[col]) / temperature - teacher_max);
		}
		const double student_norm = student_max + log(student_sum);
		const double teacher_norm = teacher_max + log(teacher_sum);
		for (uint32_t col = 0; col < args.vocab; ++col) {
			const double student_log_prob = static_cast<double>(student_row[col]) / temperature - student_norm;
			const double teacher_log_prob = static_cast<double>(teacher_row[col]) / temperature - teacher_norm;
			const double teacher_prob = exp(teacher_log_prob);
			total += teacher_prob * (teacher_log_prob - student_log_prob);
		}
	}
	double *output = reinterpret_cast<double *>(static_cast<uintptr_t>(args.output_pointer));
	output[0] = total * temperature * temperature / static_cast<double>(args.batch);
}

extern "C" __global__ void rocm_grpo_advantage(const unsigned char *packet)
{
	if (blockIdx.x != 0 || threadIdx.x != 0 || packet == nullptr) {
		return;
	}
	const rocm_grpo_advantage_launch_args &args = *reinterpret_cast<const rocm_grpo_advantage_launch_args *>(packet);
	if (!rocm_valid_grpo_advantage_args(args)) {
		return;
	}
	const double *rewards = reinterpret_cast<const double *>(static_cast<uintptr_t>(args.reward_pointer));
	double *output = reinterpret_cast<double *>(static_cast<uintptr_t>(args.output_pointer));
	double mean = 0.0;
	for (uint32_t index = 0; index < args.count; ++index) {
		const double reward = rewards[index];
		if (!isfinite(reward)) {
			return;
		}
		mean += reward;
	}
	mean /= static_cast<double>(args.count);
	double variance = 0.0;
	for (uint32_t index = 0; index < args.count; ++index) {
		const double diff = rewards[index] - mean;
		variance += diff * diff;
	}
	variance /= static_cast<double>(args.count);
	if (variance == 0.0) {
		for (uint32_t index = 0; index < args.count; ++index) {
			output[index] = 0.0;
		}
		return;
	}
	const double stddev = sqrt(variance);
	for (uint32_t index = 0; index < args.count; ++index) {
		output[index] = (rewards[index] - mean) / stddev;
	}
}

extern "C" __global__ void rocm_autoround_quantize(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_autoround_quantize_launch_args &args = *reinterpret_cast<const rocm_autoround_quantize_launch_args *>(packet);
	if (!rocm_valid_autoround_quantize_args(args)) {
		return;
	}
	const uint32_t group_index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t group_count = args.rows * args.groups_per_row;
	if (group_index >= group_count) {
		return;
	}
	const uint32_t row = group_index / args.groups_per_row;
	const uint32_t group = group_index % args.groups_per_row;
	const uint32_t value_start = row * args.cols + group * args.group_size;
	const uint32_t packed_start_bit = value_start * args.bits;
	const uint32_t packed_start_byte = packed_start_bit >> 3;
	const uint32_t packed_group_bytes = (args.group_size * args.bits + 7u) / 8u;
	const float *weights = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.weight_pointer));
	uint8_t *packed = reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(args.packed_pointer));
	float *scales = reinterpret_cast<float *>(static_cast<uintptr_t>(args.scale_pointer));
	float max_abs = 0.0f;
	for (uint32_t offset = 0; offset < args.group_size; ++offset) {
		const float value = weights[value_start + offset];
		if (!isfinite(value)) {
			return;
		}
		max_abs = fmaxf(max_abs, fabsf(value));
	}
	const int32_t qmin = -(1 << (args.bits - 1u));
	const int32_t qmax = (1 << (args.bits - 1u)) - 1;
	const float scale = max_abs == 0.0f ? 1.0f : max_abs / static_cast<float>(qmax);
	scales[group_index] = scale;
	for (uint32_t byte = 0; byte < packed_group_bytes; ++byte) {
		packed[packed_start_byte + byte] = 0;
	}
	for (uint32_t offset = 0; offset < args.group_size; ++offset) {
		int32_t quantized = static_cast<int32_t>(roundf(weights[value_start + offset] / scale));
		if (quantized < qmin) {
			quantized = qmin;
		}
		if (quantized > qmax) {
			quantized = qmax;
		}
		rocm_autoround_pack_signed(packed, args.bits, value_start + offset, quantized);
	}
}

__device__ const rocm_device_kv_page_descriptor *rocm_attention_device_kv_page_from_descriptor(uint64_t descriptor_pointer, uint32_t token)
{
	if (descriptor_pointer == 0u) {
		return nullptr;
	}
	const rocm_device_kv_descriptor_header *header = reinterpret_cast<const rocm_device_kv_descriptor_header *>(static_cast<uintptr_t>(descriptor_pointer));
	const unsigned char *base = reinterpret_cast<const unsigned char *>(header);
	if (header->block_size > 0u) {
		const uint32_t page_index = token / header->block_size;
		if (page_index < header->page_count) {
			const rocm_device_kv_page_descriptor *page = reinterpret_cast<const rocm_device_kv_page_descriptor *>(base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + page_index * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES);
			if (token >= page->token_start && token < page->token_start + page->token_count) {
				return page;
			}
			const uint64_t page_end = page->token_start + page->token_count;
			if (token >= page_end) {
				const uint64_t suffix_index = static_cast<uint64_t>(page_index) + (static_cast<uint64_t>(token) - page_end) + 1u;
				if (suffix_index < header->page_count) {
					const rocm_device_kv_page_descriptor *suffix_page = reinterpret_cast<const rocm_device_kv_page_descriptor *>(base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + suffix_index * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES);
					if (suffix_page->token_start == token && suffix_page->token_count == 1u) {
						return suffix_page;
					}
				}
			}
		}
	}
	if (token < header->page_count) {
		const rocm_device_kv_page_descriptor *page = reinterpret_cast<const rocm_device_kv_page_descriptor *>(base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + token * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES);
		if (page->token_start == token && page->token_count == 1) {
			return page;
		}
	}
	uint32_t left = 0;
	uint32_t right = header->page_count;
	while (left < right) {
		const uint32_t page_index = left + ((right - left) >> 1u);
		const rocm_device_kv_page_descriptor *page = reinterpret_cast<const rocm_device_kv_page_descriptor *>(base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + page_index * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES);
		const uint64_t page_start = page->token_start;
		const uint64_t page_end = page_start + page->token_count;
		if (token < page_start) {
			right = page_index;
			continue;
		}
		if (token >= page_end) {
			left = page_index + 1u;
			continue;
		}
		if (page->token_count > 0) {
			return page;
		}
		break;
	}
	return nullptr;
}

__device__ const rocm_device_kv_page_descriptor *rocm_attention_device_kv_page(const rocm_attention_launch_args &args, uint32_t token)
{
	return rocm_attention_device_kv_page_from_descriptor(args.descriptor_pointer, token);
}

__device__ const rocm_device_kv_page_descriptor *rocm_attention_heads_chunked_device_kv_page(const rocm_attention_heads_chunked_launch_args &args, uint32_t token)
{
	return rocm_attention_device_kv_page_from_descriptor(args.descriptor_pointer, token);
}

__device__ const rocm_device_kv_page_descriptor *rocm_attention_heads_batch_chunked_device_kv_page(const rocm_attention_heads_batch_chunked_launch_args &args, uint32_t token)
{
	return rocm_attention_device_kv_page_from_descriptor(args.descriptor_pointer, token);
}

__device__ bool rocm_device_kv_encoding_is_q8(uint32_t encoding)
{
	return encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8 ||
		encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS ||
		encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS_INTERLEAVED;
}

__device__ bool rocm_device_kv_encoding_is_q4(uint32_t encoding)
{
	return encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4 ||
		encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4_ROWS ||
		encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4_ROWS_INTERLEAVED;
}

__device__ bool rocm_device_kv_encoding_is_row_scaled(uint32_t encoding)
{
	return encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS ||
		encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4_ROWS;
}

__device__ bool rocm_device_kv_encoding_is_row_interleaved(uint32_t encoding)
{
	return encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS_INTERLEAVED ||
		encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4_ROWS_INTERLEAVED;
}

__device__ uint64_t rocm_device_kv_tensor_payload_offset(uint32_t encoding, uint64_t rows)
{
	if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_FP16) {
		return 0u;
	}
	if (rocm_device_kv_encoding_is_row_scaled(encoding)) {
		return rows * sizeof(uint32_t);
	}
	return sizeof(uint32_t);
}

__device__ float rocm_device_kv_tensor_scale(const unsigned char *bytes, uint32_t encoding, uint64_t local_token)
{
	if (bytes == nullptr || encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_FP16) {
		return 1.0f;
	}
	if (rocm_device_kv_encoding_is_row_interleaved(encoding)) {
		return 1.0f;
	}
	const uint64_t scale_index = rocm_device_kv_encoding_is_row_scaled(encoding) ? local_token : 0u;
	return rocm_float_from_bits(reinterpret_cast<const uint32_t *>(bytes)[scale_index]);
}

__device__ uint64_t rocm_device_kv_interleaved_row_stride(uint32_t encoding, uint32_t width)
{
	if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS_INTERLEAVED) {
		return sizeof(uint32_t) + width;
	}
	if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4_ROWS_INTERLEAVED) {
		return sizeof(uint32_t) + ((static_cast<uint64_t>(width) + 1u) >> 1u);
	}
	return 0u;
}

__device__ const unsigned char *rocm_device_kv_row_payload_pointer(const unsigned char *bytes, uint32_t encoding, uint64_t rows, uint32_t width, uint64_t local_token)
{
	if (bytes == nullptr) {
		return nullptr;
	}
	if (rocm_device_kv_encoding_is_row_interleaved(encoding)) {
		const uint64_t stride = rocm_device_kv_interleaved_row_stride(encoding, width);
		return bytes + local_token * stride + sizeof(uint32_t);
	}
	return bytes + rocm_device_kv_tensor_payload_offset(encoding, rows);
}

__device__ float rocm_device_kv_row_scale(const unsigned char *bytes, uint32_t encoding, uint64_t local_token, uint32_t width)
{
	if (bytes == nullptr || encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_FP16) {
		return 1.0f;
	}
	if (rocm_device_kv_encoding_is_row_interleaved(encoding)) {
		const uint64_t stride = rocm_device_kv_interleaved_row_stride(encoding, width);
		return rocm_float_from_bits(*reinterpret_cast<const uint32_t *>(bytes + local_token * stride));
	}
	return rocm_device_kv_tensor_scale(bytes, encoding, local_token);
}

__device__ bool rocm_attention_kq8vq4_page_valid(const rocm_device_kv_page_descriptor *page, uint32_t token, uint32_t dim)
{
	if (page == nullptr ||
		page->token_count == 0 ||
		token < page->token_start ||
		token >= page->token_start + page->token_count ||
		page->key_width != dim ||
		page->value_width != dim ||
		(page->value_width & 1u) != 0u ||
		!rocm_device_kv_encoding_is_q8(page->key_encoding) ||
		!rocm_device_kv_encoding_is_q4(page->value_encoding) ||
		page->key_pointer == 0 ||
		page->value_pointer == 0) {
		return false;
	}
	const uint64_t key_count = static_cast<uint64_t>(page->token_count) * page->key_width;
	const uint64_t value_count = static_cast<uint64_t>(page->token_count) * page->value_width;
	return page->key_bytes == rocm_device_kv_tensor_bytes_rows(page->key_encoding, key_count, page->token_count) &&
		page->value_bytes == rocm_device_kv_tensor_bytes_rows(page->value_encoding, value_count, page->token_count);
}

__device__ float rocm_attention_device_kv_value_from_page(const rocm_device_kv_page_descriptor *page, bool key, uint32_t token, uint32_t dim)
{
	if (page == nullptr) {
		return 0.0f;
	}
	const uint32_t width = key ? page->key_width : page->value_width;
	const uint64_t pointer = key ? page->key_pointer : page->value_pointer;
	const uint32_t encoding = key ? page->key_encoding : page->value_encoding;
	const uint64_t local_token = static_cast<uint64_t>(token) - page->token_start;
	const uint64_t index = local_token * width + dim;
	if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_FP16) {
		const uint16_t *values = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(pointer));
		return rocm_half_to_float(values[index]);
	}
	const unsigned char *bytes = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(pointer));
	const float scale = rocm_device_kv_row_scale(bytes, encoding, local_token, width);
	const unsigned char *payload = rocm_device_kv_row_payload_pointer(bytes, encoding, page->token_count, width, local_token);
	const uint64_t row_index = rocm_device_kv_encoding_is_row_interleaved(encoding) ? dim : index;
	if (rocm_device_kv_encoding_is_q8(encoding)) {
		const int8_t *values = reinterpret_cast<const int8_t *>(payload);
		return static_cast<float>(values[row_index]) * scale;
	}
	if (rocm_device_kv_encoding_is_q4(encoding)) {
		const unsigned char *values = payload;
		unsigned char packed = values[row_index / 2u];
		if ((row_index & 1u) != 0u) {
			packed >>= 4;
		}
		int quantized = static_cast<int>(packed & 0x0fu);
		if (quantized >= 8) {
			quantized -= 16;
		}
		return static_cast<float>(quantized) * scale;
	}
	return 0.0f;
}

__device__ float rocm_attention_device_kv_dot_from_page_offset(const rocm_device_kv_page_descriptor *page, bool key, uint32_t token, const float *query, uint32_t dim_count, uint32_t dim_offset)
{
	if (page == nullptr) {
		return 0.0f;
	}
	const uint32_t width = key ? page->key_width : page->value_width;
	const uint64_t pointer = key ? page->key_pointer : page->value_pointer;
	const uint32_t encoding = key ? page->key_encoding : page->value_encoding;
	const uint64_t local_token = static_cast<uint64_t>(token) - page->token_start;
	const uint64_t base_index = local_token * width + dim_offset;
	float dot = 0.0f;
	if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_FP16) {
		const uint16_t *values = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(pointer));
		for (uint32_t dim = 0; dim < dim_count; ++dim) {
			dot += query[dim] * rocm_half_to_float(values[base_index + dim]);
		}
		return dot;
	}
	const unsigned char *bytes = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(pointer));
	const float scale = rocm_device_kv_row_scale(bytes, encoding, local_token, width);
	const unsigned char *payload = rocm_device_kv_row_payload_pointer(bytes, encoding, page->token_count, width, local_token);
	const uint64_t row_base_index = rocm_device_kv_encoding_is_row_interleaved(encoding) ? dim_offset : base_index;
	if (rocm_device_kv_encoding_is_q8(encoding)) {
		const int8_t *values = reinterpret_cast<const int8_t *>(payload);
		float quantized_dot = 0.0f;
		uint32_t dim = 0;
		if ((row_base_index & 3u) == 0u) {
			const int32_t *packed_values = reinterpret_cast<const int32_t *>(values + row_base_index);
			for (; dim + 16u <= dim_count; dim += 16u) {
				const uint32_t packed_index = dim >> 2u;
				const int32_t packed0 = packed_values[packed_index];
				const int32_t packed1 = packed_values[packed_index + 1u];
				const int32_t packed2 = packed_values[packed_index + 2u];
				const int32_t packed3 = packed_values[packed_index + 3u];
				const int8_t q0 = static_cast<int8_t>(packed0 & 0xff);
				const int8_t q1 = static_cast<int8_t>((packed0 >> 8) & 0xff);
				const int8_t q2 = static_cast<int8_t>((packed0 >> 16) & 0xff);
				const int8_t q3 = static_cast<int8_t>((packed0 >> 24) & 0xff);
				const int8_t q4 = static_cast<int8_t>(packed1 & 0xff);
				const int8_t q5 = static_cast<int8_t>((packed1 >> 8) & 0xff);
				const int8_t q6 = static_cast<int8_t>((packed1 >> 16) & 0xff);
				const int8_t q7 = static_cast<int8_t>((packed1 >> 24) & 0xff);
				const int8_t q8 = static_cast<int8_t>(packed2 & 0xff);
				const int8_t q9 = static_cast<int8_t>((packed2 >> 8) & 0xff);
				const int8_t q10 = static_cast<int8_t>((packed2 >> 16) & 0xff);
				const int8_t q11 = static_cast<int8_t>((packed2 >> 24) & 0xff);
				const int8_t q12 = static_cast<int8_t>(packed3 & 0xff);
				const int8_t q13 = static_cast<int8_t>((packed3 >> 8) & 0xff);
				const int8_t q14 = static_cast<int8_t>((packed3 >> 16) & 0xff);
				const int8_t q15 = static_cast<int8_t>((packed3 >> 24) & 0xff);
				quantized_dot += query[dim] * static_cast<float>(q0);
				quantized_dot += query[dim + 1u] * static_cast<float>(q1);
				quantized_dot += query[dim + 2u] * static_cast<float>(q2);
				quantized_dot += query[dim + 3u] * static_cast<float>(q3);
				quantized_dot += query[dim + 4u] * static_cast<float>(q4);
				quantized_dot += query[dim + 5u] * static_cast<float>(q5);
				quantized_dot += query[dim + 6u] * static_cast<float>(q6);
				quantized_dot += query[dim + 7u] * static_cast<float>(q7);
				quantized_dot += query[dim + 8u] * static_cast<float>(q8);
				quantized_dot += query[dim + 9u] * static_cast<float>(q9);
				quantized_dot += query[dim + 10u] * static_cast<float>(q10);
				quantized_dot += query[dim + 11u] * static_cast<float>(q11);
				quantized_dot += query[dim + 12u] * static_cast<float>(q12);
				quantized_dot += query[dim + 13u] * static_cast<float>(q13);
				quantized_dot += query[dim + 14u] * static_cast<float>(q14);
				quantized_dot += query[dim + 15u] * static_cast<float>(q15);
			}
			for (; dim + 8u <= dim_count; dim += 8u) {
				const uint32_t packed_index = dim >> 2u;
				const int32_t packed0 = packed_values[packed_index];
				const int32_t packed1 = packed_values[packed_index + 1u];
				const int8_t q0 = static_cast<int8_t>(packed0 & 0xff);
				const int8_t q1 = static_cast<int8_t>((packed0 >> 8) & 0xff);
				const int8_t q2 = static_cast<int8_t>((packed0 >> 16) & 0xff);
				const int8_t q3 = static_cast<int8_t>((packed0 >> 24) & 0xff);
				const int8_t q4 = static_cast<int8_t>(packed1 & 0xff);
				const int8_t q5 = static_cast<int8_t>((packed1 >> 8) & 0xff);
				const int8_t q6 = static_cast<int8_t>((packed1 >> 16) & 0xff);
				const int8_t q7 = static_cast<int8_t>((packed1 >> 24) & 0xff);
				quantized_dot += query[dim] * static_cast<float>(q0);
				quantized_dot += query[dim + 1u] * static_cast<float>(q1);
				quantized_dot += query[dim + 2u] * static_cast<float>(q2);
				quantized_dot += query[dim + 3u] * static_cast<float>(q3);
				quantized_dot += query[dim + 4u] * static_cast<float>(q4);
				quantized_dot += query[dim + 5u] * static_cast<float>(q5);
				quantized_dot += query[dim + 6u] * static_cast<float>(q6);
				quantized_dot += query[dim + 7u] * static_cast<float>(q7);
			}
			for (; dim + 4u <= dim_count; dim += 4u) {
				const int32_t packed = packed_values[dim >> 2u];
				const int8_t q0 = static_cast<int8_t>(packed & 0xff);
				const int8_t q1 = static_cast<int8_t>((packed >> 8) & 0xff);
				const int8_t q2 = static_cast<int8_t>((packed >> 16) & 0xff);
				const int8_t q3 = static_cast<int8_t>((packed >> 24) & 0xff);
				quantized_dot += query[dim] * static_cast<float>(q0);
				quantized_dot += query[dim + 1u] * static_cast<float>(q1);
				quantized_dot += query[dim + 2u] * static_cast<float>(q2);
				quantized_dot += query[dim + 3u] * static_cast<float>(q3);
			}
		}
		for (; dim < dim_count; ++dim) {
			quantized_dot += query[dim] * static_cast<float>(values[row_base_index + dim]);
		}
		return quantized_dot * scale;
	}
	if (rocm_device_kv_encoding_is_q4(encoding)) {
		const unsigned char *values = payload;
		float quantized_dot = 0.0f;
		for (uint32_t dim = 0; dim < dim_count; ++dim) {
			const uint64_t index = row_base_index + dim;
			unsigned char packed = values[index / 2u];
			if ((index & 1u) != 0u) {
				packed >>= 4;
			}
			int quantized = static_cast<int>(packed & 0x0fu);
			if (quantized >= 8) {
				quantized -= 16;
			}
			quantized_dot += query[dim] * static_cast<float>(quantized);
		}
		dot = quantized_dot * scale;
	}
	return dot;
}

__device__ float rocm_attention_device_kv_dot_from_page(const rocm_device_kv_page_descriptor *page, bool key, uint32_t token, const float *query, uint32_t dim_count)
{
	return rocm_attention_device_kv_dot_from_page_offset(page, key, token, query, dim_count, 0u);
}

__device__ float rocm_attention_device_kv_value_from_tensor(uint64_t pointer, uint32_t encoding, uint64_t base_index, uint32_t dim, float scale)
{
	if (pointer == 0) {
		return 0.0f;
	}
	const uint64_t index = base_index + dim;
	if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_FP16) {
		const uint16_t *values = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(pointer));
		return rocm_half_to_float(values[index]);
	}
	const unsigned char *bytes = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(pointer));
	if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8) {
		const int8_t *values = reinterpret_cast<const int8_t *>(bytes + sizeof(uint32_t));
		return static_cast<float>(values[index]) * scale;
	}
	if (encoding == ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4) {
		const unsigned char *values = bytes + sizeof(uint32_t);
		unsigned char packed = values[index / 2u];
		if ((index & 1u) != 0u) {
			packed >>= 4;
		}
		int quantized = static_cast<int>(packed & 0x0fu);
		if (quantized >= 8) {
			quantized -= 16;
		}
		return static_cast<float>(quantized) * scale;
	}
	return 0.0f;
}

__device__ float rocm_attention_device_kv_value(const rocm_attention_launch_args &args, bool key, uint32_t token, uint32_t dim)
{
	const uint32_t dim_offset = args.kv_source == ROCM_ATTENTION_KV_SOURCE_DEVICE ? args.key_bytes : 0u;
	return rocm_attention_device_kv_value_from_page(rocm_attention_device_kv_page(args, token), key, token, dim + dim_offset);
}

__device__ uint32_t rocm_attention_contiguous_kv_stride(const rocm_attention_launch_args &args)
{
	if (args.kv_source != ROCM_ATTENTION_KV_SOURCE_CONTIGUOUS || args.token_count == 0u) {
		return args.dim;
	}
	const uint32_t elements = args.key_bytes / sizeof(float);
	const uint32_t stride = elements / args.token_count;
	return stride >= args.dim ? stride : args.dim;
}

__device__ uint32_t rocm_attention_device_kv_dim_offset(const rocm_attention_launch_args &args)
{
	return args.kv_source == ROCM_ATTENTION_KV_SOURCE_DEVICE ? args.key_bytes : 0u;
}

__device__ void rocm_run_single_head_attention(const rocm_attention_launch_args &args)
{
	const float *query = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.query_pointer));
	const float *keys = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.key_pointer));
	const float *values = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.value_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	extern __shared__ float shared_attention_weights[];
	float *weights = args.weight_pointer == 0
		? shared_attention_weights
		: reinterpret_cast<float *>(static_cast<uintptr_t>(args.weight_pointer));
	const float requested_scale = rocm_float_from_bits(args.scale_bits);
	const float scale = requested_scale == 0.0f ? 1.0f / sqrtf(static_cast<float>(args.dim)) : requested_scale;
	const bool device_kv = args.kv_source == ROCM_ATTENTION_KV_SOURCE_DEVICE;
	const uint32_t kv_stride = rocm_attention_contiguous_kv_stride(args);
	const uint32_t device_kv_dim_offset = rocm_attention_device_kv_dim_offset(args);

	float max_score = -3.4028234663852886e38f;
	for (uint32_t token = 0; token < args.token_count; ++token) {
		float dot = 0.0f;
		const uint32_t base = token * kv_stride;
		if (device_kv) {
			dot = rocm_attention_device_kv_dot_from_page_offset(rocm_attention_device_kv_page(args, token), true, token, query, args.dim, device_kv_dim_offset);
		} else {
			for (uint32_t dim = 0; dim < args.dim; ++dim) {
				dot += query[dim] * keys[base + dim];
			}
		}
		const float score = dot * scale;
		weights[token] = score;
		if (score > max_score) {
			max_score = score;
		}
	}

	float sum = 0.0f;
	for (uint32_t token = 0; token < args.token_count; ++token) {
		const float value = rocm_fast_expf(weights[token] - max_score);
		weights[token] = value;
		sum += value;
	}
	if (sum == 0.0f) {
		return;
	}
	for (uint32_t dim = 0; dim < args.dim; ++dim) {
		output[dim] = 0.0f;
	}
	for (uint32_t token = 0; token < args.token_count; ++token) {
		const float weight = weights[token] / sum;
		weights[token] = weight;
		const uint32_t base = token * kv_stride;
		for (uint32_t dim = 0; dim < args.dim; ++dim) {
			const float value = device_kv ? rocm_attention_device_kv_value(args, false, token, dim) : values[base + dim];
			output[dim] += weight * value;
		}
	}
}

__device__ uint64_t rocm_attention_align_shared_offset(uint64_t value, uint64_t alignment)
{
	const uint64_t mask = alignment - 1u;
	return (value + mask) & ~mask;
}

__device__ float rocm_attention_block_reduce_sum(float value, float *scratch)
{
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t wave = threadIdx.x >> 5u;
	const uint32_t wave_count = (blockDim.x + 31u) >> 5u;
	for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
		value += rocm_shfl_down(value, stride, 32);
	}
	if (lane == 0u) {
		scratch[wave] = value;
	}
	__syncthreads();
	float reduced = lane < wave_count ? scratch[lane] : 0.0f;
	if (wave == 0u) {
		for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
			reduced += rocm_shfl_down(reduced, stride, 32);
		}
		if (lane == 0u) {
			scratch[0] = reduced;
		}
	}
	__syncthreads();
	return scratch[0];
}

__device__ float rocm_attention_block_reduce_max(float value, float *scratch)
{
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t wave = threadIdx.x >> 5u;
	const uint32_t wave_count = (blockDim.x + 31u) >> 5u;
	for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
		const float other = rocm_shfl_down(value, stride, 32);
		if (other > value) {
			value = other;
		}
	}
	if (lane == 0u) {
		scratch[wave] = value;
	}
	__syncthreads();
	float reduced = lane < wave_count ? scratch[lane] : -FLT_MAX;
	if (wave == 0u) {
		for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
			const float other = rocm_shfl_down(reduced, stride, 32);
			if (other > reduced) {
				reduced = other;
			}
		}
		if (lane == 0u) {
			scratch[0] = reduced;
		}
	}
	__syncthreads();
	return scratch[0];
}

__device__ void rocm_run_single_head_attention_range_token_parallel(const rocm_attention_launch_args &args, uint32_t token_start, uint32_t token_end)
{
	const uint32_t tid = threadIdx.x;
	const uint32_t threads = blockDim.x;
	const bool device_kv = args.kv_source == ROCM_ATTENTION_KV_SOURCE_DEVICE;
	const uint32_t kv_stride = rocm_attention_contiguous_kv_stride(args);
	const uint32_t device_kv_dim_offset = rocm_attention_device_kv_dim_offset(args);
	if (token_start >= token_end || token_end > args.token_count) {
		for (uint32_t dim = tid; dim < args.dim; dim += threads) {
			float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
			output[dim] = 0.0f;
		}
		return;
	}
	if (threads > 512 || (threads & (threads - 1u)) != 0u) {
		if (tid != 0) {
			return;
		}
		const float *query = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.query_pointer));
		const float *keys = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.key_pointer));
		const float *values = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.value_pointer));
		float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
		extern __shared__ float shared_attention_weights[];
		float *weights = args.weight_pointer == 0
			? shared_attention_weights
			: reinterpret_cast<float *>(static_cast<uintptr_t>(args.weight_pointer));
		const float requested_scale = rocm_float_from_bits(args.scale_bits);
		const float scale = requested_scale == 0.0f ? 1.0f / sqrtf(static_cast<float>(args.dim)) : requested_scale;
		float max_score = -FLT_MAX;
		for (uint32_t token = token_start; token < token_end; ++token) {
			float dot = 0.0f;
			const uint32_t base = token * kv_stride;
			if (device_kv) {
				dot = rocm_attention_device_kv_dot_from_page_offset(rocm_attention_device_kv_page(args, token), true, token, query, args.dim, device_kv_dim_offset);
			} else {
				for (uint32_t dim = 0; dim < args.dim; ++dim) {
					dot += query[dim] * keys[base + dim];
				}
			}
			const float score = dot * scale;
			weights[token] = score;
			if (score > max_score) {
				max_score = score;
			}
		}
		float sum = 0.0f;
		for (uint32_t token = token_start; token < token_end; ++token) {
			const float value = rocm_fast_expf(weights[token] - max_score);
			weights[token] = value;
			sum += value;
		}
		if (sum == 0.0f) {
			for (uint32_t dim = 0; dim < args.dim; ++dim) {
				output[dim] = 0.0f;
			}
			return;
		}
		const float inv_sum = 1.0f / sum;
		for (uint32_t dim = 0; dim < args.dim; ++dim) {
			float out = 0.0f;
			for (uint32_t token = token_start; token < token_end; ++token) {
				const uint32_t base = token * kv_stride;
				const float value = device_kv ? rocm_attention_device_kv_value(args, false, token, dim) : values[base + dim];
				out += weights[token] * inv_sum * value;
			}
			output[dim] = out;
		}
		return;
	}

	__shared__ float scratch[512];
	__shared__ float shared_max_score;
	__shared__ float shared_sum;

	const float *query = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.query_pointer));
	const float *keys = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.key_pointer));
	const float *values = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.value_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	extern __shared__ float shared_attention_weights[];
	float *weights = args.weight_pointer == 0
		? shared_attention_weights
		: reinterpret_cast<float *>(static_cast<uintptr_t>(args.weight_pointer));
	const float requested_scale = rocm_float_from_bits(args.scale_bits);
	const float scale = requested_scale == 0.0f ? 1.0f / sqrtf(static_cast<float>(args.dim)) : requested_scale;

	float local_max = -FLT_MAX;
	for (uint32_t token = token_start + tid; token < token_end; token += threads) {
		float dot = 0.0f;
		const uint32_t base = token * kv_stride;
		if (device_kv) {
			dot = rocm_attention_device_kv_dot_from_page_offset(rocm_attention_device_kv_page(args, token), true, token, query, args.dim, device_kv_dim_offset);
		} else {
			for (uint32_t dim = 0; dim < args.dim; ++dim) {
				dot += query[dim] * keys[base + dim];
			}
		}
		const float score = dot * scale;
		weights[token] = score;
		if (score > local_max) {
			local_max = score;
		}
	}
	__syncthreads();
	const float max_score = rocm_attention_block_reduce_max(local_max, scratch);
	if (tid == 0) {
		shared_max_score = max_score;
	}
	__syncthreads();

	float local_sum = 0.0f;
	for (uint32_t token = token_start + tid; token < token_end; token += threads) {
		const float value = rocm_fast_expf(weights[token] - shared_max_score);
		weights[token] = value;
		local_sum += value;
	}
	const float sum = rocm_attention_block_reduce_sum(local_sum, scratch);
	if (tid == 0) {
		shared_sum = sum;
	}
	__syncthreads();
	if (shared_sum == 0.0f) {
		for (uint32_t dim = tid; dim < args.dim; dim += threads) {
			output[dim] = 0.0f;
		}
		return;
	}
	const float inv_sum = 1.0f / shared_sum;
	for (uint32_t dim = tid; dim < args.dim; dim += threads) {
		float out = 0.0f;
		for (uint32_t token = token_start; token < token_end; ++token) {
			const uint32_t base = token * kv_stride;
			const float value = device_kv ? rocm_attention_device_kv_value(args, false, token, dim) : values[base + dim];
			out += weights[token] * inv_sum * value;
		}
		output[dim] = out;
	}
}

__device__ void rocm_run_single_head_attention_token_parallel(const rocm_attention_launch_args &args, uint64_t dynamic_shared_bytes)
{
	const uint32_t tid = threadIdx.x;
	const uint32_t threads = blockDim.x;
	if (threads > 512 || (threads & (threads - 1u)) != 0u) {
		if (tid == 0) {
			rocm_run_single_head_attention(args);
		}
		return;
	}
	__shared__ float scratch[512];
	__shared__ float shared_max_score;
	__shared__ float shared_sum;

	const float *query = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.query_pointer));
	const float *keys = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.key_pointer));
	const float *values = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.value_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	extern __shared__ float shared_attention_weights[];
	float *weights = args.weight_pointer == 0
		? shared_attention_weights
		: reinterpret_cast<float *>(static_cast<uintptr_t>(args.weight_pointer));
	const float requested_scale = rocm_float_from_bits(args.scale_bits);
	const float scale = requested_scale == 0.0f ? 1.0f / sqrtf(static_cast<float>(args.dim)) : requested_scale;
	const bool device_kv = args.kv_source == ROCM_ATTENTION_KV_SOURCE_DEVICE;
	const uint32_t kv_stride = rocm_attention_contiguous_kv_stride(args);
	const uint32_t device_kv_dim_offset = rocm_attention_device_kv_dim_offset(args);
	const rocm_device_kv_descriptor_header *device_kv_header = device_kv
		? reinterpret_cast<const rocm_device_kv_descriptor_header *>(static_cast<uintptr_t>(args.descriptor_pointer))
		: nullptr;
	const unsigned char *device_kv_base = reinterpret_cast<const unsigned char *>(device_kv_header);
	const bool direct_token_pages =
		device_kv_header != nullptr &&
		device_kv_header->block_size == 1u &&
		device_kv_header->page_count == args.token_count;
	const rocm_device_kv_page_descriptor *first_device_kv_page = device_kv_header != nullptr && device_kv_header->page_count > 0u
		? reinterpret_cast<const rocm_device_kv_page_descriptor *>(device_kv_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES)
		: nullptr;
	const bool single_head_device_kv_width = first_device_kv_page != nullptr &&
		first_device_kv_page->key_width == args.dim &&
		first_device_kv_page->value_width == args.dim;
	const uint64_t shared_weight_bytes = args.weight_pointer == 0 ? static_cast<uint64_t>(args.token_count) * sizeof(float) : 0u;
	uint64_t shared_offset = rocm_attention_align_shared_offset(shared_weight_bytes, sizeof(uint64_t));
	const uint64_t pointer_bytes = static_cast<uint64_t>(args.token_count) * sizeof(uint64_t);
	uint64_t scale_offset = rocm_attention_align_shared_offset(shared_offset + pointer_bytes, sizeof(float));
	const uint64_t scale_bytes = static_cast<uint64_t>(args.token_count) * sizeof(float);
	const bool cache_value_metadata =
		device_kv &&
		device_kv_dim_offset == 0u &&
		single_head_device_kv_width &&
		args.token_count >= 16u &&
		dynamic_shared_bytes >= scale_offset + scale_bytes;
	unsigned char *shared_bytes = reinterpret_cast<unsigned char *>(shared_attention_weights);
	uint64_t *cached_value_pointers = cache_value_metadata ? reinterpret_cast<uint64_t *>(shared_bytes + shared_offset) : nullptr;
	float *cached_value_scales = cache_value_metadata ? reinterpret_cast<float *>(shared_bytes + scale_offset) : nullptr;
	const bool kq8vq4_direct_token_pages =
		direct_token_pages &&
		device_kv_dim_offset == 0u &&
		single_head_device_kv_width &&
		device_kv_header != nullptr &&
		device_kv_header->mode_code == ROCM_DEVICE_KV_DESCRIPTOR_MODE_KQ8VQ4;
	const bool q4_direct_value_pages = cache_value_metadata && kq8vq4_direct_token_pages;
	const bool cache_query = args.dim <= threads;
	if (cache_query && tid < args.dim) {
		scratch[tid] = query[tid];
	}
	__syncthreads();
	const float *query_values = cache_query ? scratch : query;

	float local_max = -FLT_MAX;
	for (uint32_t token = tid; token < args.token_count; token += threads) {
		float dot = 0.0f;
		const uint32_t base = token * kv_stride;
		const rocm_device_kv_page_descriptor *page = nullptr;
		if (device_kv) {
			page = direct_token_pages
				? reinterpret_cast<const rocm_device_kv_page_descriptor *>(device_kv_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + token * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES)
				: rocm_attention_device_kv_page(args, token);
		}
		if (cache_value_metadata) {
			uint64_t cached_pointer = 0;
			float cached_scale = 0.0f;
			if (rocm_attention_kq8vq4_page_valid(page, token, args.dim)) {
				const uint64_t local_token = static_cast<uint64_t>(token) - page->token_start;
				const unsigned char *bytes = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(page->value_pointer));
				const uint64_t value_base = rocm_device_kv_encoding_is_row_interleaved(page->value_encoding) ? 0u : local_token * page->value_width;
				cached_pointer = reinterpret_cast<uint64_t>(rocm_device_kv_row_payload_pointer(bytes, page->value_encoding, page->token_count, page->value_width, local_token)) + (value_base >> 1u);
				cached_scale = rocm_device_kv_row_scale(bytes, page->value_encoding, local_token, page->value_width);
			}
			cached_value_pointers[token] = cached_pointer;
			cached_value_scales[token] = cached_scale;
		}
		if (device_kv) {
			dot = rocm_attention_device_kv_dot_from_page_offset(page, true, token, query_values, args.dim, device_kv_dim_offset);
		} else {
			for (uint32_t dim = 0; dim < args.dim; ++dim) {
				dot += query_values[dim] * keys[base + dim];
			}
		}
		const float score = dot * scale;
		weights[token] = score;
		if (score > local_max) {
			local_max = score;
		}
	}
	__syncthreads();
	const float max_score = rocm_attention_block_reduce_max(local_max, scratch);
	if (tid == 0) {
		shared_max_score = max_score;
	}
	__syncthreads();

	float local_sum = 0.0f;
	for (uint32_t token = tid; token < args.token_count; token += threads) {
		const float value = rocm_fast_expf(weights[token] - shared_max_score);
		weights[token] = value;
		local_sum += value;
	}
	const float sum = rocm_attention_block_reduce_sum(local_sum, scratch);
	if (tid == 0) {
		shared_sum = sum;
	}
	__syncthreads();
	if (shared_sum == 0.0f) {
		return;
	}
	const float inv_sum = 1.0f / shared_sum;
	for (uint32_t token = tid; token < args.token_count; token += threads) {
		weights[token] *= inv_sum;
	}
	__syncthreads();

		if (device_kv && threads >= 512u && args.dim <= threads) {
			const uint32_t pair_count = (args.dim + 1u) >> 1u;
			const uint32_t value_groups = pair_count == 0u ? 0u : threads / pair_count;
			const bool q4_direct_uncached_value_pages = !cache_value_metadata && kq8vq4_direct_token_pages;
			if (q4_direct_value_pages && value_groups > 1u) {
				const uint32_t pair = tid % pair_count;
				const uint32_t group = tid / pair_count;
				const uint32_t dim0 = pair << 1u;
				const uint32_t dim1 = dim0 + 1u;
				float partial0 = 0.0f;
				float partial1 = 0.0f;
				if (group < value_groups && dim0 < args.dim) {
					for (uint32_t token = group; token < args.token_count; token += value_groups) {
						const uint64_t cached_pointer = cached_value_pointers[token];
						if (cached_pointer == 0u) {
							continue;
						}
						const unsigned char *values = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(cached_pointer));
						const unsigned char packed = values[dim0 >> 1u];
						const float weighted_scale = weights[token] * cached_value_scales[token];
						int q0 = static_cast<int>(packed & 0x0fu);
						if (q0 >= 8) {
							q0 -= 16;
						}
						partial0 += weighted_scale * static_cast<float>(q0);
						if (dim1 < args.dim) {
							int q1 = static_cast<int>((packed >> 4) & 0x0fu);
							if (q1 >= 8) {
								q1 -= 16;
							}
							partial1 += weighted_scale * static_cast<float>(q1);
						}
					}
				}
				scratch[tid] = partial0;
				__syncthreads();
				if (tid < pair_count) {
					float out0 = 0.0f;
					for (uint32_t value_group = 0; value_group < value_groups; ++value_group) {
						out0 += scratch[value_group * pair_count + tid];
					}
					output[tid << 1u] = out0;
				}
				__syncthreads();
				scratch[tid] = partial1;
				__syncthreads();
				if (tid < pair_count) {
					const uint32_t output_dim = (tid << 1u) + 1u;
					if (output_dim < args.dim) {
						float out1 = 0.0f;
						for (uint32_t value_group = 0; value_group < value_groups; ++value_group) {
							out1 += scratch[value_group * pair_count + tid];
						}
						output[output_dim] = out1;
					}
				}
				return;
			}
			if (q4_direct_uncached_value_pages && value_groups > 1u) {
				const uint32_t pair = tid % pair_count;
				const uint32_t group = tid / pair_count;
				const uint32_t dim0 = pair << 1u;
				const uint32_t dim1 = dim0 + 1u;
				float partial0 = 0.0f;
				float partial1 = 0.0f;
				if (group < value_groups && dim0 < args.dim) {
					for (uint32_t token = group; token < args.token_count; token += value_groups) {
						const rocm_device_kv_page_descriptor *page = reinterpret_cast<const rocm_device_kv_page_descriptor *>(device_kv_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + token * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES);
						const uint64_t pointer = page->value_pointer;
						if (pointer == 0u) {
							continue;
						}
						const unsigned char *bytes = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(pointer));
						const unsigned char *values = bytes + sizeof(uint32_t);
						const unsigned char packed = values[dim0 >> 1u];
						const float weighted_scale = weights[token] * rocm_float_from_bits(*reinterpret_cast<const uint32_t *>(bytes));
						int q0 = static_cast<int>(packed & 0x0fu);
						if (q0 >= 8) {
							q0 -= 16;
						}
						partial0 += weighted_scale * static_cast<float>(q0);
						if (dim1 < args.dim) {
							int q1 = static_cast<int>((packed >> 4) & 0x0fu);
							if (q1 >= 8) {
								q1 -= 16;
							}
							partial1 += weighted_scale * static_cast<float>(q1);
						}
					}
				}
				scratch[tid] = partial0;
				__syncthreads();
				if (tid < pair_count) {
					float out0 = 0.0f;
					for (uint32_t value_group = 0; value_group < value_groups; ++value_group) {
						out0 += scratch[value_group * pair_count + tid];
					}
					output[tid << 1u] = out0;
				}
				__syncthreads();
				scratch[tid] = partial1;
				__syncthreads();
				if (tid < pair_count) {
					const uint32_t output_dim = (tid << 1u) + 1u;
					if (output_dim < args.dim) {
						float out1 = 0.0f;
						for (uint32_t value_group = 0; value_group < value_groups; ++value_group) {
							out1 += scratch[value_group * pair_count + tid];
						}
						output[output_dim] = out1;
					}
				}
				return;
			}
			if (value_groups > 1u) {
				const uint32_t pair = tid % pair_count;
				const uint32_t group = tid / pair_count;
				const uint32_t dim0 = pair << 1u;
				const uint32_t dim1 = dim0 + 1u;
				float partial0 = 0.0f;
				float partial1 = 0.0f;
				if (group < value_groups && dim0 < args.dim) {
					for (uint32_t token = group; token < args.token_count; token += value_groups) {
						const float weight = weights[token];
						const uint64_t cached_pointer = cache_value_metadata ? cached_value_pointers[token] : 0u;
						if (cached_pointer != 0u) {
							const unsigned char *values = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(cached_pointer));
							const unsigned char packed = values[dim0 >> 1u];
							const float weighted_scale = weight * cached_value_scales[token];
							int q0 = static_cast<int>(packed & 0x0fu);
							if (q0 >= 8) {
								q0 -= 16;
							}
							partial0 += weighted_scale * static_cast<float>(q0);
							if (dim1 < args.dim) {
								int q1 = static_cast<int>((packed >> 4) & 0x0fu);
								if (q1 >= 8) {
									q1 -= 16;
								}
								partial1 += weighted_scale * static_cast<float>(q1);
							}
							continue;
						}
						const rocm_device_kv_page_descriptor *page = direct_token_pages
							? reinterpret_cast<const rocm_device_kv_page_descriptor *>(device_kv_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + token * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES)
							: rocm_attention_device_kv_page(args, token);
						if (page == nullptr) {
							continue;
						}
						const uint64_t pointer = page->value_pointer;
						const uint32_t encoding = page->value_encoding;
						const uint64_t local_token = static_cast<uint64_t>(token) - page->token_start;
						const uint64_t base_index = (rocm_device_kv_encoding_is_row_interleaved(encoding) ? 0u : local_token * page->value_width) + device_kv_dim_offset;
						uint64_t value_payload_offset = sizeof(uint32_t);
						float value_scale = 0.0f;
						if (encoding != ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_FP16 && pointer != 0) {
							const unsigned char *bytes = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(pointer));
							value_scale = rocm_device_kv_row_scale(bytes, encoding, local_token, page->value_width);
							value_payload_offset = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rocm_device_kv_row_payload_pointer(bytes, encoding, page->token_count, page->value_width, local_token))) - pointer;
						}
						const uint64_t index0 = base_index + dim0;
					if (pointer != 0 && rocm_device_kv_encoding_is_q4(encoding) && ((index0 & 1u) == 0u)) {
						const unsigned char *values = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(pointer) + value_payload_offset);
						const unsigned char packed = values[index0 >> 1u];
						int q0 = static_cast<int>(packed & 0x0fu);
						if (q0 >= 8) {
							q0 -= 16;
						}
						partial0 += weight * value_scale * static_cast<float>(q0);
						if (dim1 < args.dim) {
							int q1 = static_cast<int>((packed >> 4) & 0x0fu);
							if (q1 >= 8) {
								q1 -= 16;
							}
							partial1 += weight * value_scale * static_cast<float>(q1);
						}
					} else if (pointer != 0 && rocm_device_kv_encoding_is_q8(encoding)) {
						const int8_t *values = reinterpret_cast<const int8_t *>(static_cast<uintptr_t>(pointer) + value_payload_offset);
						partial0 += weight * value_scale * static_cast<float>(values[index0]);
						if (dim1 < args.dim) {
							partial1 += weight * value_scale * static_cast<float>(values[base_index + dim1]);
						}
					} else {
						partial0 += weight * rocm_attention_device_kv_value_from_tensor(pointer, encoding, base_index, dim0, value_scale);
						if (dim1 < args.dim) {
							partial1 += weight * rocm_attention_device_kv_value_from_tensor(pointer, encoding, base_index, dim1, value_scale);
						}
					}
				}
			}
			scratch[tid] = partial0;
			__syncthreads();
			if (tid < pair_count) {
				float out0 = 0.0f;
				for (uint32_t value_group = 0; value_group < value_groups; ++value_group) {
					out0 += scratch[value_group * pair_count + tid];
				}
				output[tid << 1u] = out0;
			}
			__syncthreads();
			scratch[tid] = partial1;
			__syncthreads();
			if (tid < pair_count) {
				const uint32_t output_dim = (tid << 1u) + 1u;
				if (output_dim < args.dim) {
					float out1 = 0.0f;
					for (uint32_t value_group = 0; value_group < value_groups; ++value_group) {
						out1 += scratch[value_group * pair_count + tid];
					}
					output[output_dim] = out1;
				}
			}
			return;
		}
	}

	if (device_kv && args.dim <= threads) {
		const uint32_t value_groups = threads / args.dim;
		if (value_groups > 1u) {
			const uint32_t dim = tid % args.dim;
			const uint32_t group = tid / args.dim;
				float partial = 0.0f;
				if (group < value_groups) {
					for (uint32_t token = group; token < args.token_count; token += value_groups) {
					const rocm_device_kv_page_descriptor *page = direct_token_pages
						? reinterpret_cast<const rocm_device_kv_page_descriptor *>(device_kv_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + token * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES)
						: rocm_attention_device_kv_page(args, token);
					if (page == nullptr) {
						continue;
					}
					const uint64_t pointer = page->value_pointer;
						const uint32_t encoding = page->value_encoding;
						const uint64_t local_token = static_cast<uint64_t>(token) - page->token_start;
						const uint64_t base_index = (rocm_device_kv_encoding_is_row_interleaved(encoding) ? 0u : local_token * page->value_width) + device_kv_dim_offset;
						const float weight = weights[token];
					float value_scale = 0.0f;
					uint64_t value_payload_offset = sizeof(uint32_t);
					if (encoding != ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_FP16 && pointer != 0) {
						const unsigned char *bytes = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(pointer));
						value_scale = rocm_device_kv_row_scale(bytes, encoding, local_token, page->value_width);
						value_payload_offset = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rocm_device_kv_row_payload_pointer(bytes, encoding, page->token_count, page->value_width, local_token))) - pointer;
					}
					const uint64_t index = base_index + dim;
					if (pointer != 0 && rocm_device_kv_encoding_is_q4(encoding)) {
						const unsigned char *values = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(pointer) + value_payload_offset);
						unsigned char packed = values[index / 2u];
						if ((index & 1u) != 0u) {
							packed >>= 4;
						}
						int quantized = static_cast<int>(packed & 0x0fu);
						if (quantized >= 8) {
							quantized -= 16;
						}
						partial += weight * value_scale * static_cast<float>(quantized);
					} else if (pointer != 0 && rocm_device_kv_encoding_is_q8(encoding)) {
						const int8_t *values = reinterpret_cast<const int8_t *>(static_cast<uintptr_t>(pointer) + value_payload_offset);
						partial += weight * value_scale * static_cast<float>(values[index]);
					} else {
						partial += weight * rocm_attention_device_kv_value_from_tensor(pointer, encoding, base_index, dim, value_scale);
					}
				}
			}
			scratch[tid] = partial;
			__syncthreads();
			if (tid < args.dim) {
				float out = 0.0f;
				for (uint32_t value_group = 0; value_group < value_groups; ++value_group) {
					out += scratch[value_group * args.dim + tid];
				}
				output[tid] = out;
			}
			return;
		}
	}

	if (device_kv && args.dim <= threads * 2u) {
		const uint32_t dim0 = tid;
		const uint32_t dim1 = tid + threads;
		float out0 = 0.0f;
		float out1 = 0.0f;
		if (dim0 < args.dim || dim1 < args.dim) {
				for (uint32_t token = 0; token < args.token_count; ++token) {
				const rocm_device_kv_page_descriptor *page = direct_token_pages
					? reinterpret_cast<const rocm_device_kv_page_descriptor *>(device_kv_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + token * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES)
					: rocm_attention_device_kv_page(args, token);
				if (page == nullptr) {
					continue;
				}
				const uint64_t pointer = page->value_pointer;
				const uint32_t encoding = page->value_encoding;
				const uint64_t local_token = static_cast<uint64_t>(token) - page->token_start;
				const uint64_t base_index = (rocm_device_kv_encoding_is_row_interleaved(encoding) ? 0u : local_token * page->value_width) + device_kv_dim_offset;
				float value_scale = 0.0f;
				uint64_t value_payload_offset = sizeof(uint32_t);
				if (encoding != ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_FP16 && pointer != 0) {
					const unsigned char *bytes = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(pointer));
					value_scale = rocm_device_kv_row_scale(bytes, encoding, local_token, page->value_width);
					value_payload_offset = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rocm_device_kv_row_payload_pointer(bytes, encoding, page->token_count, page->value_width, local_token))) - pointer;
				}
					const float weight = weights[token];
					const float weighted_scale = weight * value_scale;
				if (dim0 < args.dim) {
					float value0 = 0.0f;
					const uint64_t index0 = base_index + dim0;
					if (pointer != 0 && rocm_device_kv_encoding_is_q4(encoding)) {
						const unsigned char *values = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(pointer) + value_payload_offset);
					unsigned char packed = values[index0 / 2u];
					if ((index0 & 1u) != 0u) {
						packed >>= 4;
					}
					int quantized = static_cast<int>(packed & 0x0fu);
						if (quantized >= 8) {
							quantized -= 16;
						}
						out0 += weighted_scale * static_cast<float>(quantized);
					} else if (pointer != 0 && rocm_device_kv_encoding_is_q8(encoding)) {
						const int8_t *values = reinterpret_cast<const int8_t *>(static_cast<uintptr_t>(pointer) + value_payload_offset);
						out0 += weighted_scale * static_cast<float>(values[index0]);
					} else {
						value0 = rocm_attention_device_kv_value_from_tensor(pointer, encoding, base_index, dim0, value_scale);
						out0 += weight * value0;
					}
				}
				if (dim1 < args.dim) {
					float value1 = 0.0f;
					const uint64_t index1 = base_index + dim1;
					if (pointer != 0 && rocm_device_kv_encoding_is_q4(encoding)) {
					const unsigned char *values = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(pointer) + value_payload_offset);
					unsigned char packed = values[index1 / 2u];
					if ((index1 & 1u) != 0u) {
						packed >>= 4;
					}
					int quantized = static_cast<int>(packed & 0x0fu);
						if (quantized >= 8) {
							quantized -= 16;
						}
						out1 += weighted_scale * static_cast<float>(quantized);
					} else if (pointer != 0 && rocm_device_kv_encoding_is_q8(encoding)) {
						const int8_t *values = reinterpret_cast<const int8_t *>(static_cast<uintptr_t>(pointer) + value_payload_offset);
						out1 += weighted_scale * static_cast<float>(values[index1]);
					} else {
						value1 = rocm_attention_device_kv_value_from_tensor(pointer, encoding, base_index, dim1, value_scale);
						out1 += weight * value1;
					}
				}
			}
		}
		if (dim0 < args.dim) {
			output[dim0] = out0;
		}
		if (dim1 < args.dim) {
			output[dim1] = out1;
		}
		return;
	}

		for (uint32_t dim = tid; dim < args.dim; dim += threads) {
			float out = 0.0f;
			for (uint32_t token = 0; token < args.token_count; ++token) {
			const uint32_t base = token * kv_stride;
			const rocm_device_kv_page_descriptor *page = device_kv ? rocm_attention_device_kv_page(args, token) : nullptr;
			const float value = device_kv ? rocm_attention_device_kv_value(args, false, token, dim) : values[base + dim];
			out += weights[token] * value;
		}
			output[dim] = out;
		}
	}

__device__ void rocm_run_single_head_attention_parallel(const rocm_attention_launch_args &args)
{
	const uint32_t tid = threadIdx.x;
	const uint32_t threads = blockDim.x;
	if (threads > 512 || (threads & (threads - 1u)) != 0u) {
		if (tid == 0) {
			rocm_run_single_head_attention(args);
		}
		return;
	}
	__shared__ float scratch[512];
	__shared__ float shared_max_score;
	__shared__ float shared_sum;

	const float *query = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.query_pointer));
	const float *keys = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.key_pointer));
	const float *values = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.value_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	float *weights = reinterpret_cast<float *>(static_cast<uintptr_t>(args.weight_pointer));
	const float requested_scale = rocm_float_from_bits(args.scale_bits);
	const float scale = requested_scale == 0.0f ? 1.0f / sqrtf(static_cast<float>(args.dim)) : requested_scale;
	const bool device_kv = args.kv_source == ROCM_ATTENTION_KV_SOURCE_DEVICE;
	const uint32_t kv_stride = rocm_attention_contiguous_kv_stride(args);

	if (tid == 0) {
		shared_max_score = -FLT_MAX;
	}
	__syncthreads();

	for (uint32_t token = 0; token < args.token_count; ++token) {
		float partial = 0.0f;
		const uint32_t base = token * kv_stride;
		for (uint32_t dim = tid; dim < args.dim; dim += threads) {
			const float key = device_kv ? rocm_attention_device_kv_value(args, true, token, dim) : keys[base + dim];
			partial += query[dim] * key;
		}
		scratch[tid] = partial;
		__syncthreads();
		for (uint32_t stride = threads >> 1; stride > 0; stride >>= 1) {
			if (tid < stride) {
				scratch[tid] += scratch[tid + stride];
			}
			__syncthreads();
		}
		if (tid == 0) {
			const float score = scratch[0] * scale;
			weights[token] = score;
			if (score > shared_max_score) {
				shared_max_score = score;
			}
		}
		__syncthreads();
	}

	float partial_sum = 0.0f;
	for (uint32_t token = tid; token < args.token_count; token += threads) {
		const float value = rocm_fast_expf(weights[token] - shared_max_score);
		weights[token] = value;
		partial_sum += value;
	}
	scratch[tid] = partial_sum;
	__syncthreads();
	for (uint32_t stride = threads >> 1; stride > 0; stride >>= 1) {
		if (tid < stride) {
			scratch[tid] += scratch[tid + stride];
		}
		__syncthreads();
	}
	if (tid == 0) {
		shared_sum = scratch[0];
	}
	__syncthreads();
	if (shared_sum == 0.0f) {
		return;
	}
	for (uint32_t token = tid; token < args.token_count; token += threads) {
		weights[token] /= shared_sum;
	}
	__syncthreads();
	for (uint32_t dim = tid; dim < args.dim; dim += threads) {
		float out = 0.0f;
		for (uint32_t token = 0; token < args.token_count; ++token) {
			const uint32_t base = token * kv_stride;
			const float value = device_kv ? rocm_attention_device_kv_value(args, false, token, dim) : values[base + dim];
			out += weights[token] * value;
		}
		output[dim] = out;
	}
}

extern "C" __global__ void rocm_attention(const unsigned char *packet)
{
	if (blockIdx.x != 0 || threadIdx.x != 0 || packet == nullptr) {
		return;
	}
	const rocm_attention_launch_args &args = *reinterpret_cast<const rocm_attention_launch_args *>(packet);
	if (!rocm_valid_attention_args(args)) {
		return;
	}
	rocm_run_single_head_attention(args);
}

extern "C" __global__ void rocm_attention_heads_chunked_stage1(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_attention_heads_chunked_launch_args &args = *reinterpret_cast<const rocm_attention_heads_chunked_launch_args *>(packet);
	if (!rocm_valid_attention_heads_chunked_args(args)) {
		return;
	}
	const uint32_t chunk_count = args.chunk_count;
	const uint32_t head = blockIdx.x / chunk_count;
	const uint32_t chunk = blockIdx.x - head * chunk_count;
	if (head >= args.head_count || chunk >= chunk_count) {
		return;
	}
	const uint32_t tid = threadIdx.x;
	const uint32_t threads = blockDim.x;
	if (threads != ROCM_ATTENTION_HEADS_CHUNKED_BLOCK_SIZE) {
		return;
	}
	const uint32_t kv_head = rocm_attention_kv_head_for_query(head, args.head_count, args.key_heads);
	const uint32_t kv_dim_offset = kv_head * args.dim;
	const uint32_t kv_width = args.key_heads * args.dim;

	extern __shared__ unsigned char shared_bytes[];
	float *scores = reinterpret_cast<float *>(shared_bytes);
	uint64_t value_pointer_offset = rocm_attention_align_shared_offset(static_cast<uint64_t>(args.chunk_size) * sizeof(float), sizeof(uint64_t));
	uint64_t value_scale_offset = rocm_attention_align_shared_offset(value_pointer_offset + static_cast<uint64_t>(args.chunk_size) * sizeof(uint64_t), sizeof(float));
	uint64_t query_cache_offset = rocm_attention_align_shared_offset(value_scale_offset + static_cast<uint64_t>(args.chunk_size) * sizeof(float), sizeof(float));
	uint64_t *value_pointers = reinterpret_cast<uint64_t *>(shared_bytes + value_pointer_offset);
	float *value_scales = reinterpret_cast<float *>(shared_bytes + value_scale_offset);
	float *query_values = reinterpret_cast<float *>(shared_bytes + query_cache_offset);
	__shared__ float scratch[ROCM_ATTENTION_HEADS_CHUNKED_BLOCK_SIZE];
	__shared__ float value_scratch1[ROCM_ATTENTION_HEADS_CHUNKED_BLOCK_SIZE];
	__shared__ float shared_max_score;
	__shared__ float shared_sum;

	const float *query = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.query_pointer)) + static_cast<uint64_t>(head) * args.dim;
	float *partials = reinterpret_cast<float *>(static_cast<uintptr_t>(args.partial_pointer)) + (static_cast<uint64_t>(head) * chunk_count + chunk) * args.dim;
	float *stats = reinterpret_cast<float *>(static_cast<uintptr_t>(args.stats_pointer)) + (static_cast<uint64_t>(head) * chunk_count + chunk) * 2u;
	const float requested_scale = rocm_float_from_bits(args.scale_bits);
	const float scale = requested_scale == 0.0f ? 1.0f / sqrtf(static_cast<float>(args.dim)) : requested_scale;
	const uint32_t visible_tokens = args.token_count;
	const uint32_t window_start = args.window_size > 0u && visible_tokens > args.window_size
		? visible_tokens - args.window_size
		: 0u;
	const uint32_t chunk_start = chunk * args.chunk_size;
	const uint32_t chunk_end = chunk_start + args.chunk_size < visible_tokens
		? chunk_start + args.chunk_size
		: visible_tokens;
	const uint32_t start = chunk_start < window_start ? window_start : chunk_start;
	if (chunk_start >= visible_tokens || chunk_end <= window_start || start >= chunk_end) {
		if (tid == 0) {
			stats[0] = -FLT_MAX;
			stats[1] = 0.0f;
		}
		for (uint32_t dim = tid; dim < args.dim; dim += threads) {
			partials[dim] = 0.0f;
		}
		return;
	}
	const uint32_t local_count = chunk_end - start;
	const rocm_device_kv_descriptor_header *device_kv_header = reinterpret_cast<const rocm_device_kv_descriptor_header *>(static_cast<uintptr_t>(args.descriptor_pointer));
	const unsigned char *device_kv_base = reinterpret_cast<const unsigned char *>(device_kv_header);
	const bool direct_kq8vq4_token_pages =
		device_kv_header != nullptr &&
		device_kv_header->block_size == 1u &&
		device_kv_header->page_count == args.token_count &&
		device_kv_header->mode_code == ROCM_DEVICE_KV_DESCRIPTOR_MODE_KQ8VQ4;

	if (tid < args.dim) {
		query_values[tid] = query[tid];
	}
	__syncthreads();

	float local_max = -FLT_MAX;
	const uint32_t score_lanes = args.chunk_size <= threads && (threads % args.chunk_size) == 0u
		? threads / args.chunk_size
		: 1u;
	if (score_lanes > 1u && score_lanes <= 8u && (score_lanes & (score_lanes - 1u)) == 0u && args.dim >= score_lanes) {
		const uint32_t local = tid / score_lanes;
		const uint32_t lane = tid - local * score_lanes;
		float partial_dot = 0.0f;
		uint64_t key_pointer = 0;
		float key_scale = 0.0f;
		uint64_t value_pointer = 0;
		float value_scale = 0.0f;
		uint32_t page_valid = 0;
		if (local < local_count && lane == 0u) {
			const uint32_t token = start + local;
			const rocm_device_kv_page_descriptor *page = direct_kq8vq4_token_pages
				? reinterpret_cast<const rocm_device_kv_page_descriptor *>(device_kv_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + token * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES)
				: rocm_attention_heads_chunked_device_kv_page(args, token);
			if (direct_kq8vq4_token_pages) {
				page_valid = page != nullptr && page->key_pointer != 0 && page->value_pointer != 0 ? 1u : 0u;
			} else {
				page_valid = rocm_attention_kq8vq4_page_valid(page, token, kv_width) ? 1u : 0u;
			}
			if (page_valid) {
				const uint64_t local_token = direct_kq8vq4_token_pages ? 0u : static_cast<uint64_t>(token) - page->token_start;
				const unsigned char *key_bytes = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(page->key_pointer));
				const uint64_t key_base = (direct_kq8vq4_token_pages || rocm_device_kv_encoding_is_row_interleaved(page->key_encoding)) ? 0u : local_token * page->key_width;
				key_pointer = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rocm_device_kv_row_payload_pointer(key_bytes, page->key_encoding, page->token_count, page->key_width, local_token))) + key_base + kv_dim_offset;
				key_scale = rocm_device_kv_row_scale(key_bytes, page->key_encoding, local_token, page->key_width);
				const unsigned char *value_bytes = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(page->value_pointer));
				const uint64_t value_base = (direct_kq8vq4_token_pages || rocm_device_kv_encoding_is_row_interleaved(page->value_encoding)) ? 0u : local_token * page->value_width;
				value_pointer = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rocm_device_kv_row_payload_pointer(value_bytes, page->value_encoding, page->token_count, page->value_width, local_token))) + ((value_base + kv_dim_offset) >> 1u);
				value_scale = rocm_device_kv_row_scale(value_bytes, page->value_encoding, local_token, page->value_width);
			}
		}
		page_valid = rocm_shfl_u32(page_valid, 0, static_cast<int>(score_lanes));
		key_pointer = rocm_shfl_u64(key_pointer, 0, static_cast<int>(score_lanes));
		key_scale = rocm_shfl_float(key_scale, 0, static_cast<int>(score_lanes));
		if (local < local_count && page_valid != 0u) {
			const int8_t *key_values = reinterpret_cast<const int8_t *>(static_cast<uintptr_t>(key_pointer));
			float quantized_dot = 0.0f;
			for (uint32_t dim = lane; dim < args.dim; dim += score_lanes) {
				quantized_dot += query_values[dim] * static_cast<float>(key_values[dim]);
			}
			partial_dot = quantized_dot * key_scale;
		}
		float dot = partial_dot;
		for (uint32_t score_lane = 1u; score_lane < score_lanes; ++score_lane) {
			dot += rocm_shfl_down(partial_dot, score_lane, static_cast<int>(score_lanes));
		}
		if (lane == 0u && local < local_count) {
			float score = -FLT_MAX;
			if (page_valid != 0u) {
				score = dot * scale;
				if (score > local_max) {
					local_max = score;
				}
			}
			scores[local] = score;
			value_pointers[local] = value_pointer;
			value_scales[local] = value_scale;
		}
	} else {
		for (uint32_t local = tid; local < local_count; local += threads) {
			const uint32_t token = start + local;
			const rocm_device_kv_page_descriptor *page = direct_kq8vq4_token_pages
				? reinterpret_cast<const rocm_device_kv_page_descriptor *>(device_kv_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + token * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES)
				: rocm_attention_heads_chunked_device_kv_page(args, token);
			bool page_valid = false;
			if (direct_kq8vq4_token_pages) {
				page_valid = page != nullptr && page->key_pointer != 0 && page->value_pointer != 0;
			} else {
				page_valid = rocm_attention_kq8vq4_page_valid(page, token, kv_width);
			}
			float score = -FLT_MAX;
			uint64_t value_pointer = 0;
			float value_scale = 0.0f;
			if (page_valid) {
				const uint64_t local_token = direct_kq8vq4_token_pages ? 0u : static_cast<uint64_t>(token) - page->token_start;
				const unsigned char *value_bytes = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(page->value_pointer));
				const uint64_t value_base = (direct_kq8vq4_token_pages || rocm_device_kv_encoding_is_row_interleaved(page->value_encoding)) ? 0u : local_token * page->value_width;
				value_pointer = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rocm_device_kv_row_payload_pointer(value_bytes, page->value_encoding, page->token_count, page->value_width, local_token))) + ((value_base + kv_dim_offset) >> 1u);
				value_scale = rocm_device_kv_row_scale(value_bytes, page->value_encoding, local_token, page->value_width);
				const float dot = rocm_attention_device_kv_dot_from_page_offset(page, true, token, query_values, args.dim, kv_dim_offset);
				score = dot * scale;
				if (score > local_max) {
					local_max = score;
				}
			}
			scores[local] = score;
			value_pointers[local] = value_pointer;
			value_scales[local] = value_scale;
		}
	}
	__syncthreads();

	const float max_score = rocm_attention_block_reduce_max(local_max, scratch);
	if (tid == 0) {
		shared_max_score = max_score;
	}
	__syncthreads();

	float local_sum = 0.0f;
	for (uint32_t local = tid; local < local_count; local += threads) {
		float value = 0.0f;
		if (shared_max_score > -FLT_MAX && scores[local] > -FLT_MAX) {
			value = rocm_fast_expf(scores[local] - shared_max_score);
			local_sum += value;
		}
		scores[local] = value;
	}
	const float sum = rocm_attention_block_reduce_sum(local_sum, scratch);
	if (tid == 0) {
		shared_sum = sum;
		stats[0] = shared_max_score;
		stats[1] = sum;
	}
	__syncthreads();

	const uint32_t pair_count = (args.dim + 1u) >> 1u;
	const uint32_t value_groups = pair_count == 0u ? 0u : threads / pair_count;
	const uint32_t pair = pair_count == 0u ? 0u : tid % pair_count;
	const uint32_t group = pair_count == 0u ? 0u : tid / pair_count;
	const uint32_t dim0 = pair << 1u;
	const uint32_t dim1 = dim0 + 1u;
	float partial0 = 0.0f;
	float partial1 = 0.0f;
	if (value_groups > 0u && group < value_groups && dim0 < args.dim && shared_sum > 0.0f) {
		for (uint32_t local = group; local < local_count; local += value_groups) {
			const uint64_t value_pointer = value_pointers[local];
			if (value_pointer == 0u) {
				continue;
			}
			const unsigned char *values = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(value_pointer));
			const unsigned char packed = values[dim0 >> 1u];
			const float weighted_scale = scores[local] * value_scales[local];
			int q0 = static_cast<int>(packed & 0x0fu);
			if (q0 >= 8) {
				q0 -= 16;
			}
			partial0 += weighted_scale * static_cast<float>(q0);
			if (dim1 < args.dim) {
				int q1 = static_cast<int>((packed >> 4) & 0x0fu);
				if (q1 >= 8) {
					q1 -= 16;
				}
				partial1 += weighted_scale * static_cast<float>(q1);
			}
		}
	}
	scratch[tid] = partial0;
	value_scratch1[tid] = partial1;
	__syncthreads();
	if (tid < pair_count) {
		float out0 = 0.0f;
		for (uint32_t value_group = 0; value_group < value_groups; ++value_group) {
			out0 += scratch[value_group * pair_count + tid];
		}
		partials[tid << 1u] = out0;
		const uint32_t output_dim = (tid << 1u) + 1u;
		if (output_dim < args.dim) {
			float out1 = 0.0f;
			for (uint32_t value_group = 0; value_group < value_groups; ++value_group) {
				out1 += value_scratch1[value_group * pair_count + tid];
			}
			partials[output_dim] = out1;
		}
	}
}

extern "C" __global__ void rocm_attention_heads_chunked_stage2(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_attention_heads_chunked_launch_args &args = *reinterpret_cast<const rocm_attention_heads_chunked_launch_args *>(packet);
	if (!rocm_valid_attention_heads_chunked_args(args) || blockIdx.x >= args.head_count) {
		return;
	}
	const uint32_t tid = threadIdx.x;
	const uint32_t threads = blockDim.x;
	if (threads != ROCM_ATTENTION_HEADS_CHUNKED_BLOCK_SIZE) {
		return;
	}
	__shared__ float scratch[ROCM_ATTENTION_HEADS_CHUNKED_BLOCK_SIZE];
	__shared__ float shared_max_score;
	__shared__ float shared_sum;

	const uint32_t head = blockIdx.x;
	const uint32_t chunk_count = args.chunk_count;
	const float *partials = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.partial_pointer)) + static_cast<uint64_t>(head) * chunk_count * args.dim;
	const float *stats = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.stats_pointer)) + static_cast<uint64_t>(head) * chunk_count * 2u;
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer)) + static_cast<uint64_t>(head) * args.dim;

	float local_max = -FLT_MAX;
	for (uint32_t chunk = tid; chunk < chunk_count; chunk += threads) {
		const float chunk_max = stats[chunk * 2u];
		const float chunk_sum = stats[chunk * 2u + 1u];
		if (chunk_sum > 0.0f && chunk_max > local_max) {
			local_max = chunk_max;
		}
	}
	const float max_score = rocm_attention_block_reduce_max(local_max, scratch);
	if (tid == 0) {
		shared_max_score = max_score;
	}
	__syncthreads();

	float local_sum = 0.0f;
	if (shared_max_score > -FLT_MAX) {
		for (uint32_t chunk = tid; chunk < chunk_count; chunk += threads) {
			const float chunk_sum = stats[chunk * 2u + 1u];
			if (chunk_sum > 0.0f) {
				local_sum += chunk_sum * rocm_fast_expf(stats[chunk * 2u] - shared_max_score);
			}
		}
	}
	const float total_sum = rocm_attention_block_reduce_sum(local_sum, scratch);
	if (tid == 0) {
		shared_sum = total_sum;
	}
	__syncthreads();
	if (shared_sum == 0.0f || !(shared_max_score > -FLT_MAX)) {
		for (uint32_t dim = tid; dim < args.dim; dim += threads) {
			output[dim] = 0.0f;
		}
		return;
	}
	const float inv_sum = 1.0f / shared_sum;
	const bool cached_chunk_weights = chunk_count <= threads;
	if (cached_chunk_weights) {
		if (tid < chunk_count) {
			const float chunk_sum = stats[tid * 2u + 1u];
			scratch[tid] = chunk_sum == 0.0f ? 0.0f : rocm_fast_expf(stats[tid * 2u] - shared_max_score);
		}
		__syncthreads();
	}
	for (uint32_t dim = tid; dim < args.dim; dim += threads) {
		float out = 0.0f;
		for (uint32_t chunk = 0; chunk < chunk_count; ++chunk) {
			float chunk_weight = 0.0f;
			if (cached_chunk_weights) {
				chunk_weight = scratch[chunk];
			} else {
				const float chunk_sum = stats[chunk * 2u + 1u];
				if (chunk_sum == 0.0f) {
					continue;
				}
				chunk_weight = rocm_fast_expf(stats[chunk * 2u] - shared_max_score);
			}
			out += partials[chunk * args.dim + dim] * chunk_weight;
		}
		output[dim] = out * inv_sum;
	}
}

extern "C" __global__ void rocm_attention_heads_batch_chunked_stage1_v2(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_attention_heads_batch_chunked_launch_args &args = *reinterpret_cast<const rocm_attention_heads_batch_chunked_launch_args *>(packet);
	if (!rocm_valid_attention_heads_batch_chunked_args(args)) {
		return;
	}
	const uint32_t chunk_count = args.chunk_count;
	const uint32_t row = blockIdx.x / chunk_count;
	const uint32_t chunk = blockIdx.x - row * chunk_count;
	const uint32_t query_row = row / args.head_count;
	const uint32_t head = row - query_row * args.head_count;
	if (query_row >= args.query_count || head >= args.head_count || chunk >= chunk_count) {
		return;
	}
	const uint32_t tid = threadIdx.x;
	const uint32_t threads = blockDim.x;
	if (threads != ROCM_ATTENTION_HEADS_CHUNKED_BLOCK_SIZE) {
		return;
	}
	const uint32_t kv_head_count = args.key_heads;
	const uint32_t kv_head = rocm_attention_kv_head_for_query(head, args.head_count, kv_head_count);
	const uint32_t kv_dim_offset = kv_head * args.dim;
	const uint32_t kv_width = kv_head_count * args.dim;

	float *partials = reinterpret_cast<float *>(static_cast<uintptr_t>(args.partial_pointer)) + (static_cast<uint64_t>(row) * chunk_count + chunk) * args.dim;
	float *stats = reinterpret_cast<float *>(static_cast<uintptr_t>(args.stats_pointer)) + (static_cast<uint64_t>(row) * chunk_count + chunk) * 2u;
	const uint32_t visible_tokens = args.query_start_token + query_row + 1u;
	const uint32_t window_start = args.window_size > 0u && visible_tokens > args.window_size
		? visible_tokens - args.window_size
		: 0u;
	const uint32_t chunk_start = args.chunk_start_token + chunk * args.chunk_size;
	const uint32_t chunk_end = chunk_start + args.chunk_size < visible_tokens
		? chunk_start + args.chunk_size
		: visible_tokens;
	const uint32_t effective_start = chunk_start < window_start ? window_start : chunk_start;
	if (chunk_start >= visible_tokens || chunk_end <= window_start || effective_start >= chunk_end) {
		if (tid == 0) {
			stats[0] = -FLT_MAX;
			stats[1] = 0.0f;
		}
		for (uint32_t dim = tid; dim < args.dim; dim += threads) {
			partials[dim] = 0.0f;
		}
		return;
	}

	extern __shared__ unsigned char shared_bytes[];
	float *scores = reinterpret_cast<float *>(shared_bytes);
	uint64_t value_pointer_offset = rocm_attention_align_shared_offset(static_cast<uint64_t>(args.chunk_size) * sizeof(float), sizeof(uint64_t));
	uint64_t value_scale_offset = rocm_attention_align_shared_offset(value_pointer_offset + static_cast<uint64_t>(args.chunk_size) * sizeof(uint64_t), sizeof(float));
	uint64_t query_cache_offset = rocm_attention_align_shared_offset(value_scale_offset + static_cast<uint64_t>(args.chunk_size) * sizeof(float), sizeof(float));
	uint64_t *value_pointers = reinterpret_cast<uint64_t *>(shared_bytes + value_pointer_offset);
	float *value_scales = reinterpret_cast<float *>(shared_bytes + value_scale_offset);
	float *query_values = reinterpret_cast<float *>(shared_bytes + query_cache_offset);
	__shared__ float scratch[ROCM_ATTENTION_HEADS_CHUNKED_BLOCK_SIZE];
	__shared__ float value_scratch1[ROCM_ATTENTION_HEADS_CHUNKED_BLOCK_SIZE];
	__shared__ float shared_max_score;
	__shared__ float shared_sum;

	const float *query = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.query_pointer)) + static_cast<uint64_t>(row) * args.dim;
	const float requested_scale = rocm_float_from_bits(args.scale_bits);
	const float scale = requested_scale == 0.0f ? 1.0f / sqrtf(static_cast<float>(args.dim)) : requested_scale;
	const rocm_device_kv_descriptor_header *device_kv_header = reinterpret_cast<const rocm_device_kv_descriptor_header *>(static_cast<uintptr_t>(args.descriptor_pointer));
	const unsigned char *device_kv_base = reinterpret_cast<const unsigned char *>(device_kv_header);
	const bool direct_kq8vq4_token_pages =
		device_kv_header != nullptr &&
		device_kv_header->block_size == 1u &&
		device_kv_header->page_count == args.token_count &&
		device_kv_header->mode_code == ROCM_DEVICE_KV_DESCRIPTOR_MODE_KQ8VQ4;

	if (tid < args.dim) {
		query_values[tid] = query[tid];
	}
	__syncthreads();

	float local_max = -FLT_MAX;
	const uint32_t local_count = chunk_end - effective_start;
	const uint32_t score_lanes = args.chunk_size <= threads && (threads % args.chunk_size) == 0u
		? threads / args.chunk_size
		: 1u;
	if (score_lanes > 1u && score_lanes <= 8u && (score_lanes & (score_lanes - 1u)) == 0u && args.dim >= score_lanes) {
		const uint32_t local = tid / score_lanes;
		const uint32_t lane = tid - local * score_lanes;
		float partial_dot = 0.0f;
		uint64_t key_pointer = 0;
		float key_scale = 0.0f;
		uint64_t value_pointer = 0;
		float value_scale = 0.0f;
		uint32_t page_valid = 0;
		if (local < local_count && lane == 0u) {
			const uint32_t token = effective_start + local;
			const rocm_device_kv_page_descriptor *page = direct_kq8vq4_token_pages
				? reinterpret_cast<const rocm_device_kv_page_descriptor *>(device_kv_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + token * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES)
				: rocm_attention_heads_batch_chunked_device_kv_page(args, token);
			if (direct_kq8vq4_token_pages) {
				page_valid = page != nullptr && page->key_pointer != 0 && page->value_pointer != 0 ? 1u : 0u;
			} else {
				page_valid = rocm_attention_kq8vq4_page_valid(page, token, kv_width) ? 1u : 0u;
			}
			if (page_valid) {
				const uint64_t local_token = direct_kq8vq4_token_pages ? 0u : static_cast<uint64_t>(token) - page->token_start;
				const unsigned char *key_bytes = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(page->key_pointer));
				const uint64_t key_base = (direct_kq8vq4_token_pages || rocm_device_kv_encoding_is_row_interleaved(page->key_encoding)) ? 0u : local_token * page->key_width;
				key_pointer = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rocm_device_kv_row_payload_pointer(key_bytes, page->key_encoding, page->token_count, page->key_width, local_token))) + key_base + kv_dim_offset;
				key_scale = rocm_device_kv_row_scale(key_bytes, page->key_encoding, local_token, page->key_width);
				const unsigned char *value_bytes = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(page->value_pointer));
				const uint64_t value_base = (direct_kq8vq4_token_pages || rocm_device_kv_encoding_is_row_interleaved(page->value_encoding)) ? 0u : local_token * page->value_width;
				value_pointer = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rocm_device_kv_row_payload_pointer(value_bytes, page->value_encoding, page->token_count, page->value_width, local_token))) + ((value_base + kv_dim_offset) >> 1u);
				value_scale = rocm_device_kv_row_scale(value_bytes, page->value_encoding, local_token, page->value_width);
			}
		}
		page_valid = rocm_shfl_u32(page_valid, 0, static_cast<int>(score_lanes));
		key_pointer = rocm_shfl_u64(key_pointer, 0, static_cast<int>(score_lanes));
		key_scale = rocm_shfl_float(key_scale, 0, static_cast<int>(score_lanes));
		if (local < local_count && page_valid != 0u) {
			const int8_t *key_values = reinterpret_cast<const int8_t *>(static_cast<uintptr_t>(key_pointer));
			float quantized_dot = 0.0f;
			for (uint32_t dim = lane; dim < args.dim; dim += score_lanes) {
				quantized_dot += query_values[dim] * static_cast<float>(key_values[dim]);
			}
			partial_dot = quantized_dot * key_scale;
		}
		float dot = partial_dot;
		for (uint32_t score_lane = 1u; score_lane < score_lanes; ++score_lane) {
			dot += rocm_shfl_down(partial_dot, score_lane, static_cast<int>(score_lanes));
		}
		if (lane == 0u && local < local_count) {
			float score = -FLT_MAX;
			if (page_valid != 0u) {
				score = dot * scale;
				if (score > local_max) {
					local_max = score;
				}
			}
			scores[local] = score;
			value_pointers[local] = value_pointer;
			value_scales[local] = value_scale;
		}
	} else {
		for (uint32_t local = tid; local < local_count; local += threads) {
			const uint32_t token = effective_start + local;
			const rocm_device_kv_page_descriptor *page = direct_kq8vq4_token_pages
				? reinterpret_cast<const rocm_device_kv_page_descriptor *>(device_kv_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + token * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES)
				: rocm_attention_heads_batch_chunked_device_kv_page(args, token);
			bool page_valid = false;
			if (direct_kq8vq4_token_pages) {
				page_valid = page != nullptr && page->key_pointer != 0 && page->value_pointer != 0;
			} else {
				page_valid = rocm_attention_kq8vq4_page_valid(page, token, kv_width);
			}
			float score = -FLT_MAX;
			uint64_t value_pointer = 0;
			float value_scale = 0.0f;
			if (page_valid) {
				const uint64_t local_token = direct_kq8vq4_token_pages ? 0u : static_cast<uint64_t>(token) - page->token_start;
				const unsigned char *value_bytes = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(page->value_pointer));
				const uint64_t value_base = (direct_kq8vq4_token_pages || rocm_device_kv_encoding_is_row_interleaved(page->value_encoding)) ? 0u : local_token * page->value_width;
				value_pointer = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rocm_device_kv_row_payload_pointer(value_bytes, page->value_encoding, page->token_count, page->value_width, local_token))) + ((value_base + kv_dim_offset) >> 1u);
				value_scale = rocm_device_kv_row_scale(value_bytes, page->value_encoding, local_token, page->value_width);
				const float dot = rocm_attention_device_kv_dot_from_page_offset(page, true, token, query_values, args.dim, kv_dim_offset);
				score = dot * scale;
				if (score > local_max) {
					local_max = score;
				}
			}
			scores[local] = score;
			value_pointers[local] = value_pointer;
			value_scales[local] = value_scale;
		}
	}
	__syncthreads();

	const float max_score = rocm_attention_block_reduce_max(local_max, scratch);
	if (tid == 0) {
		shared_max_score = max_score;
	}
	__syncthreads();

	float local_sum = 0.0f;
	for (uint32_t local = tid; local < local_count; local += threads) {
		float value = 0.0f;
		if (shared_max_score > -FLT_MAX && scores[local] > -FLT_MAX) {
			value = rocm_fast_expf(scores[local] - shared_max_score);
			local_sum += value;
		}
		scores[local] = value;
	}
	const float sum = rocm_attention_block_reduce_sum(local_sum, scratch);
	if (tid == 0) {
		shared_sum = sum;
		stats[0] = shared_max_score;
		stats[1] = sum;
	}
	__syncthreads();

	const uint32_t pair_count = (args.dim + 1u) >> 1u;
	const uint32_t value_groups = pair_count == 0u ? 0u : threads / pair_count;
	const uint32_t pair = pair_count == 0u ? 0u : tid % pair_count;
	const uint32_t group = pair_count == 0u ? 0u : tid / pair_count;
	const uint32_t dim0 = pair << 1u;
	const uint32_t dim1 = dim0 + 1u;
	float partial0 = 0.0f;
	float partial1 = 0.0f;
	if (value_groups > 0u && group < value_groups && dim0 < args.dim && shared_sum > 0.0f) {
		for (uint32_t local = group; local < local_count; local += value_groups) {
			const uint64_t value_pointer = value_pointers[local];
			if (value_pointer == 0u) {
				continue;
			}
			const unsigned char *values = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(value_pointer));
			const unsigned char packed = values[dim0 >> 1u];
			const float weighted_scale = scores[local] * value_scales[local];
			int q0 = static_cast<int>(packed & 0x0fu);
			if (q0 >= 8) {
				q0 -= 16;
			}
			partial0 += weighted_scale * static_cast<float>(q0);
			if (dim1 < args.dim) {
				int q1 = static_cast<int>((packed >> 4) & 0x0fu);
				if (q1 >= 8) {
					q1 -= 16;
				}
				partial1 += weighted_scale * static_cast<float>(q1);
			}
		}
	}
	scratch[tid] = partial0;
	value_scratch1[tid] = partial1;
	__syncthreads();
	if (tid < pair_count) {
		float out0 = 0.0f;
		for (uint32_t value_group = 0; value_group < value_groups; ++value_group) {
			out0 += scratch[value_group * pair_count + tid];
		}
		partials[tid << 1u] = out0;
		const uint32_t output_dim = (tid << 1u) + 1u;
		if (output_dim < args.dim) {
			float out1 = 0.0f;
			for (uint32_t value_group = 0; value_group < value_groups; ++value_group) {
				out1 += value_scratch1[value_group * pair_count + tid];
			}
			partials[output_dim] = out1;
		}
	}
}

extern "C" __global__ void rocm_attention_heads_batch_chunked_stage1_gqa2(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_attention_heads_batch_chunked_launch_args &args = *reinterpret_cast<const rocm_attention_heads_batch_chunked_launch_args *>(packet);
	if (!rocm_valid_attention_heads_batch_chunked_args(args) || args.head_count < 2u || (args.head_count & 1u) != 0u) {
		return;
	}
	const uint32_t query_heads_per_kv = args.head_count / args.key_heads;
	if (query_heads_per_kv < 2u || (query_heads_per_kv & 1u) != 0u) {
		return;
	}
	const uint32_t chunk_count = args.chunk_count;
	const uint32_t pairs_per_query = args.head_count >> 1u;
	const uint32_t pair_row = blockIdx.x / chunk_count;
	const uint32_t chunk = blockIdx.x - pair_row * chunk_count;
	const uint32_t query_row = pair_row / pairs_per_query;
	const uint32_t pair_index = pair_row - query_row * pairs_per_query;
	const uint32_t head0 = pair_index << 1u;
	const uint32_t head1 = head0 + 1u;
	if (query_row >= args.query_count || head1 >= args.head_count || chunk >= chunk_count) {
		return;
	}
	const uint32_t tid = threadIdx.x;
	const uint32_t threads = blockDim.x;
	if (threads != ROCM_ATTENTION_HEADS_CHUNKED_BLOCK_SIZE) {
		return;
	}
	const uint32_t kv_head0 = rocm_attention_kv_head_for_query(head0, args.head_count, args.key_heads);
	const uint32_t kv_head1 = rocm_attention_kv_head_for_query(head1, args.head_count, args.key_heads);
	if (kv_head0 != kv_head1) {
		return;
	}
	const uint32_t kv_dim_offset = kv_head0 * args.dim;
	const uint32_t kv_width = args.key_heads * args.dim;
	const uint32_t row0 = query_row * args.head_count + head0;
	const uint32_t row1 = row0 + 1u;

	float *partials0 = reinterpret_cast<float *>(static_cast<uintptr_t>(args.partial_pointer)) + (static_cast<uint64_t>(row0) * chunk_count + chunk) * args.dim;
	float *partials1 = reinterpret_cast<float *>(static_cast<uintptr_t>(args.partial_pointer)) + (static_cast<uint64_t>(row1) * chunk_count + chunk) * args.dim;
	float *stats0 = reinterpret_cast<float *>(static_cast<uintptr_t>(args.stats_pointer)) + (static_cast<uint64_t>(row0) * chunk_count + chunk) * 2u;
	float *stats1 = reinterpret_cast<float *>(static_cast<uintptr_t>(args.stats_pointer)) + (static_cast<uint64_t>(row1) * chunk_count + chunk) * 2u;
	const uint32_t visible_tokens = args.query_start_token + query_row + 1u;
	const uint32_t window_start = args.window_size > 0u && visible_tokens > args.window_size
		? visible_tokens - args.window_size
		: 0u;
	const uint32_t chunk_start = args.chunk_start_token + chunk * args.chunk_size;
	const uint32_t chunk_end = chunk_start + args.chunk_size < visible_tokens
		? chunk_start + args.chunk_size
		: visible_tokens;
	const uint32_t effective_start = chunk_start < window_start ? window_start : chunk_start;
	if (chunk_start >= visible_tokens || chunk_end <= window_start || effective_start >= chunk_end) {
		if (tid == 0u) {
			stats0[0] = -FLT_MAX;
			stats0[1] = 0.0f;
			stats1[0] = -FLT_MAX;
			stats1[1] = 0.0f;
		}
		for (uint32_t dim = tid; dim < args.dim; dim += threads) {
			partials0[dim] = 0.0f;
			partials1[dim] = 0.0f;
		}
		return;
	}

	extern __shared__ unsigned char shared_bytes[];
	float *scores0 = reinterpret_cast<float *>(shared_bytes);
	const uint64_t scores1_offset = rocm_attention_align_shared_offset(static_cast<uint64_t>(args.chunk_size) * sizeof(float), sizeof(float));
	const uint64_t value_pointer_offset = rocm_attention_align_shared_offset(scores1_offset + static_cast<uint64_t>(args.chunk_size) * sizeof(float), sizeof(uint64_t));
	const uint64_t value_scale_offset = rocm_attention_align_shared_offset(value_pointer_offset + static_cast<uint64_t>(args.chunk_size) * sizeof(uint64_t), sizeof(float));
	const uint64_t query0_cache_offset = rocm_attention_align_shared_offset(value_scale_offset + static_cast<uint64_t>(args.chunk_size) * sizeof(float), sizeof(float));
	const uint64_t query1_cache_offset = rocm_attention_align_shared_offset(query0_cache_offset + static_cast<uint64_t>(args.dim) * sizeof(float), sizeof(float));
	float *scores1 = reinterpret_cast<float *>(shared_bytes + scores1_offset);
	uint64_t *value_pointers = reinterpret_cast<uint64_t *>(shared_bytes + value_pointer_offset);
	float *value_scales = reinterpret_cast<float *>(shared_bytes + value_scale_offset);
	float *query_values0 = reinterpret_cast<float *>(shared_bytes + query0_cache_offset);
	float *query_values1 = reinterpret_cast<float *>(shared_bytes + query1_cache_offset);
	__shared__ float scratch[ROCM_ATTENTION_HEADS_CHUNKED_BLOCK_SIZE];
	__shared__ float value_scratch1[ROCM_ATTENTION_HEADS_CHUNKED_BLOCK_SIZE];
	__shared__ float shared_max_score0;
	__shared__ float shared_max_score1;
	__shared__ float shared_sum0;
	__shared__ float shared_sum1;

	const float *query0 = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.query_pointer)) + static_cast<uint64_t>(row0) * args.dim;
	const float *query1 = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.query_pointer)) + static_cast<uint64_t>(row1) * args.dim;
	const float requested_scale = rocm_float_from_bits(args.scale_bits);
	const float scale = requested_scale == 0.0f ? 1.0f / sqrtf(static_cast<float>(args.dim)) : requested_scale;
	const rocm_device_kv_descriptor_header *device_kv_header = reinterpret_cast<const rocm_device_kv_descriptor_header *>(static_cast<uintptr_t>(args.descriptor_pointer));
	const unsigned char *device_kv_base = reinterpret_cast<const unsigned char *>(device_kv_header);
	const bool direct_kq8vq4_token_pages =
		device_kv_header != nullptr &&
		device_kv_header->block_size == 1u &&
		device_kv_header->page_count == args.token_count &&
		device_kv_header->mode_code == ROCM_DEVICE_KV_DESCRIPTOR_MODE_KQ8VQ4;

	if (tid < args.dim) {
		query_values0[tid] = query0[tid];
		query_values1[tid] = query1[tid];
	}
	__syncthreads();

	float local_max0 = -FLT_MAX;
	float local_max1 = -FLT_MAX;
	const uint32_t local_count = chunk_end - effective_start;
	const uint32_t score_lanes = args.chunk_size <= threads && (threads % args.chunk_size) == 0u
		? threads / args.chunk_size
		: 1u;
	if (score_lanes > 1u && score_lanes <= 8u && (score_lanes & (score_lanes - 1u)) == 0u && args.dim >= score_lanes) {
		const uint32_t local = tid / score_lanes;
		const uint32_t lane = tid - local * score_lanes;
		float partial_dot0 = 0.0f;
		float partial_dot1 = 0.0f;
		uint64_t key_pointer = 0u;
		float key_scale = 0.0f;
		uint64_t value_pointer = 0u;
		float value_scale = 0.0f;
		uint32_t page_valid = 0u;
		if (local < local_count && lane == 0u) {
			const uint32_t token = effective_start + local;
			const rocm_device_kv_page_descriptor *page = direct_kq8vq4_token_pages
				? reinterpret_cast<const rocm_device_kv_page_descriptor *>(device_kv_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + token * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES)
				: rocm_attention_heads_batch_chunked_device_kv_page(args, token);
			if (direct_kq8vq4_token_pages) {
				page_valid = page != nullptr && page->key_pointer != 0u && page->value_pointer != 0u ? 1u : 0u;
			} else {
				page_valid = rocm_attention_kq8vq4_page_valid(page, token, kv_width) ? 1u : 0u;
			}
			if (page_valid != 0u) {
				const uint64_t local_token = direct_kq8vq4_token_pages ? 0u : static_cast<uint64_t>(token) - page->token_start;
				const unsigned char *key_bytes = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(page->key_pointer));
				const uint64_t key_base = (direct_kq8vq4_token_pages || rocm_device_kv_encoding_is_row_interleaved(page->key_encoding)) ? 0u : local_token * page->key_width;
				key_pointer = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rocm_device_kv_row_payload_pointer(key_bytes, page->key_encoding, page->token_count, page->key_width, local_token))) + key_base + kv_dim_offset;
				key_scale = rocm_device_kv_row_scale(key_bytes, page->key_encoding, local_token, page->key_width);
				const unsigned char *value_bytes = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(page->value_pointer));
				const uint64_t value_base = (direct_kq8vq4_token_pages || rocm_device_kv_encoding_is_row_interleaved(page->value_encoding)) ? 0u : local_token * page->value_width;
				value_pointer = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rocm_device_kv_row_payload_pointer(value_bytes, page->value_encoding, page->token_count, page->value_width, local_token))) + ((value_base + kv_dim_offset) >> 1u);
				value_scale = rocm_device_kv_row_scale(value_bytes, page->value_encoding, local_token, page->value_width);
			}
		}
		page_valid = rocm_shfl_u32(page_valid, 0, static_cast<int>(score_lanes));
		key_pointer = rocm_shfl_u64(key_pointer, 0, static_cast<int>(score_lanes));
		key_scale = rocm_shfl_float(key_scale, 0, static_cast<int>(score_lanes));
		if (local < local_count && page_valid != 0u) {
			const int8_t *key_values = reinterpret_cast<const int8_t *>(static_cast<uintptr_t>(key_pointer));
			float quantized_dot0 = 0.0f;
			float quantized_dot1 = 0.0f;
			for (uint32_t dim = lane; dim < args.dim; dim += score_lanes) {
				const float quantized_key = static_cast<float>(key_values[dim]);
				quantized_dot0 += query_values0[dim] * quantized_key;
				quantized_dot1 += query_values1[dim] * quantized_key;
			}
			partial_dot0 = quantized_dot0 * key_scale;
			partial_dot1 = quantized_dot1 * key_scale;
		}
		float dot0 = partial_dot0;
		float dot1 = partial_dot1;
		for (uint32_t score_lane = 1u; score_lane < score_lanes; ++score_lane) {
			dot0 += rocm_shfl_down(partial_dot0, score_lane, static_cast<int>(score_lanes));
			dot1 += rocm_shfl_down(partial_dot1, score_lane, static_cast<int>(score_lanes));
		}
		if (lane == 0u && local < local_count) {
			float score0 = -FLT_MAX;
			float score1 = -FLT_MAX;
			if (page_valid != 0u) {
				score0 = dot0 * scale;
				score1 = dot1 * scale;
				if (score0 > local_max0) {
					local_max0 = score0;
				}
				if (score1 > local_max1) {
					local_max1 = score1;
				}
			}
			scores0[local] = score0;
			scores1[local] = score1;
			value_pointers[local] = value_pointer;
			value_scales[local] = value_scale;
		}
	} else {
		for (uint32_t local = tid; local < local_count; local += threads) {
			const uint32_t token = effective_start + local;
			const rocm_device_kv_page_descriptor *page = direct_kq8vq4_token_pages
				? reinterpret_cast<const rocm_device_kv_page_descriptor *>(device_kv_base + ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES + token * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES)
				: rocm_attention_heads_batch_chunked_device_kv_page(args, token);
			bool page_valid = false;
			if (direct_kq8vq4_token_pages) {
				page_valid = page != nullptr && page->key_pointer != 0u && page->value_pointer != 0u;
			} else {
				page_valid = rocm_attention_kq8vq4_page_valid(page, token, kv_width);
			}
			float score0 = -FLT_MAX;
			float score1 = -FLT_MAX;
			uint64_t value_pointer = 0u;
			float value_scale = 0.0f;
			if (page_valid) {
				const uint64_t local_token = direct_kq8vq4_token_pages ? 0u : static_cast<uint64_t>(token) - page->token_start;
				const unsigned char *key_bytes = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(page->key_pointer));
				const uint64_t key_base = (direct_kq8vq4_token_pages || rocm_device_kv_encoding_is_row_interleaved(page->key_encoding)) ? 0u : local_token * page->key_width;
				const int8_t *key_values = reinterpret_cast<const int8_t *>(reinterpret_cast<uintptr_t>(rocm_device_kv_row_payload_pointer(key_bytes, page->key_encoding, page->token_count, page->key_width, local_token)) + key_base + kv_dim_offset);
				const float key_scale = rocm_device_kv_row_scale(key_bytes, page->key_encoding, local_token, page->key_width);
				float quantized_dot0 = 0.0f;
				float quantized_dot1 = 0.0f;
				for (uint32_t dim = 0u; dim < args.dim; ++dim) {
					const float quantized_key = static_cast<float>(key_values[dim]);
					quantized_dot0 += query_values0[dim] * quantized_key;
					quantized_dot1 += query_values1[dim] * quantized_key;
				}
				score0 = quantized_dot0 * key_scale * scale;
				score1 = quantized_dot1 * key_scale * scale;
				if (score0 > local_max0) {
					local_max0 = score0;
				}
				if (score1 > local_max1) {
					local_max1 = score1;
				}
				const unsigned char *value_bytes = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(page->value_pointer));
				const uint64_t value_base = (direct_kq8vq4_token_pages || rocm_device_kv_encoding_is_row_interleaved(page->value_encoding)) ? 0u : local_token * page->value_width;
				value_pointer = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rocm_device_kv_row_payload_pointer(value_bytes, page->value_encoding, page->token_count, page->value_width, local_token))) + ((value_base + kv_dim_offset) >> 1u);
				value_scale = rocm_device_kv_row_scale(value_bytes, page->value_encoding, local_token, page->value_width);
			}
			scores0[local] = score0;
			scores1[local] = score1;
			value_pointers[local] = value_pointer;
			value_scales[local] = value_scale;
		}
	}
	__syncthreads();

	const float max_score0 = rocm_attention_block_reduce_max(local_max0, scratch);
	if (tid == 0u) {
		shared_max_score0 = max_score0;
	}
	__syncthreads();
	const float max_score1 = rocm_attention_block_reduce_max(local_max1, scratch);
	if (tid == 0u) {
		shared_max_score1 = max_score1;
	}
	__syncthreads();

	float local_sum0 = 0.0f;
	for (uint32_t local = tid; local < local_count; local += threads) {
		float value = 0.0f;
		if (shared_max_score0 > -FLT_MAX && scores0[local] > -FLT_MAX) {
			value = rocm_fast_expf(scores0[local] - shared_max_score0);
			local_sum0 += value;
		}
		scores0[local] = value;
	}
	const float sum0 = rocm_attention_block_reduce_sum(local_sum0, scratch);
	if (tid == 0u) {
		shared_sum0 = sum0;
		stats0[0] = shared_max_score0;
		stats0[1] = sum0;
	}
	__syncthreads();

	float local_sum1 = 0.0f;
	for (uint32_t local = tid; local < local_count; local += threads) {
		float value = 0.0f;
		if (shared_max_score1 > -FLT_MAX && scores1[local] > -FLT_MAX) {
			value = rocm_fast_expf(scores1[local] - shared_max_score1);
			local_sum1 += value;
		}
		scores1[local] = value;
	}
	const float sum1 = rocm_attention_block_reduce_sum(local_sum1, scratch);
	if (tid == 0u) {
		shared_sum1 = sum1;
		stats1[0] = shared_max_score1;
		stats1[1] = sum1;
	}
	__syncthreads();

	const uint32_t dim_pair_count = (args.dim + 1u) >> 1u;
	const uint32_t value_groups = dim_pair_count == 0u ? 0u : threads / dim_pair_count;
	const uint32_t dim_pair = dim_pair_count == 0u ? 0u : tid % dim_pair_count;
	const uint32_t group = dim_pair_count == 0u ? 0u : tid / dim_pair_count;
	const uint32_t dim0 = dim_pair << 1u;
	const uint32_t dim1 = dim0 + 1u;
	float head0_partial0 = 0.0f;
	float head0_partial1 = 0.0f;
	float head1_partial0 = 0.0f;
	float head1_partial1 = 0.0f;
	if (value_groups > 0u && group < value_groups && dim0 < args.dim && (shared_sum0 > 0.0f || shared_sum1 > 0.0f)) {
		for (uint32_t local = group; local < local_count; local += value_groups) {
			const uint64_t value_pointer = value_pointers[local];
			if (value_pointer == 0u) {
				continue;
			}
			const unsigned char *values = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(value_pointer));
			const unsigned char packed = values[dim0 >> 1u];
			int quantized0 = static_cast<int>(packed & 0x0fu);
			if (quantized0 >= 8) {
				quantized0 -= 16;
			}
			int quantized1 = static_cast<int>((packed >> 4) & 0x0fu);
			if (quantized1 >= 8) {
				quantized1 -= 16;
			}
			const float value_scale = value_scales[local];
			const float weighted_scale0 = scores0[local] * value_scale;
			const float weighted_scale1 = scores1[local] * value_scale;
			if (shared_sum0 > 0.0f) {
				head0_partial0 += weighted_scale0 * static_cast<float>(quantized0);
				if (dim1 < args.dim) {
					head0_partial1 += weighted_scale0 * static_cast<float>(quantized1);
				}
			}
			if (shared_sum1 > 0.0f) {
				head1_partial0 += weighted_scale1 * static_cast<float>(quantized0);
				if (dim1 < args.dim) {
					head1_partial1 += weighted_scale1 * static_cast<float>(quantized1);
				}
			}
		}
	}

	scratch[tid] = head0_partial0;
	value_scratch1[tid] = head0_partial1;
	__syncthreads();
	if (tid < dim_pair_count) {
		float out0 = 0.0f;
		float out1 = 0.0f;
		for (uint32_t value_group = 0u; value_group < value_groups; ++value_group) {
			out0 += scratch[value_group * dim_pair_count + tid];
			out1 += value_scratch1[value_group * dim_pair_count + tid];
		}
		partials0[tid << 1u] = out0;
		const uint32_t output_dim = (tid << 1u) + 1u;
		if (output_dim < args.dim) {
			partials0[output_dim] = out1;
		}
	}
	__syncthreads();

	scratch[tid] = head1_partial0;
	value_scratch1[tid] = head1_partial1;
	__syncthreads();
	if (tid < dim_pair_count) {
		float out0 = 0.0f;
		float out1 = 0.0f;
		for (uint32_t value_group = 0u; value_group < value_groups; ++value_group) {
			out0 += scratch[value_group * dim_pair_count + tid];
			out1 += value_scratch1[value_group * dim_pair_count + tid];
		}
		partials1[tid << 1u] = out0;
		const uint32_t output_dim = (tid << 1u) + 1u;
		if (output_dim < args.dim) {
			partials1[output_dim] = out1;
		}
	}
}

extern "C" __global__ void rocm_attention_heads_batch_chunked_stage2(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_attention_heads_batch_chunked_launch_args &args = *reinterpret_cast<const rocm_attention_heads_batch_chunked_launch_args *>(packet);
	if (!rocm_valid_attention_heads_batch_chunked_args(args) || blockIdx.x >= args.query_count * args.head_count) {
		return;
	}
	const uint32_t tid = threadIdx.x;
	const uint32_t threads = blockDim.x;
	if (threads != ROCM_ATTENTION_HEADS_CHUNKED_BLOCK_SIZE) {
		return;
	}
	__shared__ float scratch[ROCM_ATTENTION_HEADS_CHUNKED_BLOCK_SIZE];
	__shared__ float shared_max_score;
	__shared__ float shared_sum;

	const uint32_t row = blockIdx.x;
	const uint32_t chunk_count = args.chunk_count;
	const float *partials = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.partial_pointer)) + static_cast<uint64_t>(row) * chunk_count * args.dim;
	const float *stats = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.stats_pointer)) + static_cast<uint64_t>(row) * chunk_count * 2u;
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer)) + static_cast<uint64_t>(row) * args.dim;

	float local_max = -FLT_MAX;
	for (uint32_t chunk = tid; chunk < chunk_count; chunk += threads) {
		const float chunk_max = stats[chunk * 2u];
		const float chunk_sum = stats[chunk * 2u + 1u];
		if (chunk_sum > 0.0f && chunk_max > local_max) {
			local_max = chunk_max;
		}
	}
	const float max_score = rocm_attention_block_reduce_max(local_max, scratch);
	if (tid == 0) {
		shared_max_score = max_score;
	}
	__syncthreads();

	float local_sum = 0.0f;
	if (shared_max_score > -FLT_MAX) {
		for (uint32_t chunk = tid; chunk < chunk_count; chunk += threads) {
			const float chunk_sum = stats[chunk * 2u + 1u];
			if (chunk_sum > 0.0f) {
				local_sum += chunk_sum * rocm_fast_expf(stats[chunk * 2u] - shared_max_score);
			}
		}
	}
	const float total_sum = rocm_attention_block_reduce_sum(local_sum, scratch);
	if (tid == 0) {
		shared_sum = total_sum;
	}
	__syncthreads();
	if (shared_sum == 0.0f || !(shared_max_score > -FLT_MAX)) {
		for (uint32_t dim = tid; dim < args.dim; dim += threads) {
			output[dim] = 0.0f;
		}
		return;
	}
	const float inv_sum = 1.0f / shared_sum;
	const bool cached_chunk_weights = chunk_count <= threads;
	if (cached_chunk_weights) {
		if (tid < chunk_count) {
			const float chunk_sum = stats[tid * 2u + 1u];
			scratch[tid] = chunk_sum == 0.0f ? 0.0f : rocm_fast_expf(stats[tid * 2u] - shared_max_score);
		}
		__syncthreads();
	}
	for (uint32_t dim = tid; dim < args.dim; dim += threads) {
		float out = 0.0f;
		for (uint32_t chunk = 0; chunk < chunk_count; ++chunk) {
			float chunk_weight = 0.0f;
			if (cached_chunk_weights) {
				chunk_weight = scratch[chunk];
			} else {
				const float chunk_sum = stats[chunk * 2u + 1u];
				if (chunk_sum == 0.0f) {
					continue;
				}
				chunk_weight = rocm_fast_expf(stats[chunk * 2u] - shared_max_score);
			}
			out += partials[chunk * args.dim + dim] * chunk_weight;
		}
		output[dim] = out * inv_sum;
	}
}

extern "C" __global__ void rocm_attention_heads(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_attention_heads_launch_args &args = *reinterpret_cast<const rocm_attention_heads_launch_args *>(packet);
	if (!rocm_valid_attention_heads_args(args) || blockIdx.x >= args.head_count) {
		return;
	}
	const uint64_t head = blockIdx.x;
	const uint32_t kv_head_count = rocm_attention_kv_head_count(args.reserved0, args.head_count);
	const uint32_t kv_head = rocm_attention_kv_head_for_query(static_cast<uint32_t>(head), args.head_count, kv_head_count);
	const uint64_t kv_pointer_offset = static_cast<uint64_t>(kv_head) * args.dim * sizeof(float);
	rocm_attention_launch_args single = {};
	single.version = ROCM_ATTENTION_LAUNCH_ARGS_VERSION;
	single.total_bytes = ROCM_ATTENTION_LAUNCH_ARGS_BYTES;
	single.query_pointer = args.query_pointer + head * args.dim * sizeof(float);
	single.key_pointer = args.kv_source == ROCM_ATTENTION_KV_SOURCE_CONTIGUOUS ? args.key_pointer + kv_pointer_offset : 0u;
	single.value_pointer = args.kv_source == ROCM_ATTENTION_KV_SOURCE_CONTIGUOUS ? args.value_pointer + kv_pointer_offset : 0u;
	single.output_pointer = args.output_pointer + head * args.dim * sizeof(float);
	single.weight_pointer = args.weight_pointer == 0
		? 0
		: args.weight_pointer + head * args.token_count * sizeof(float);
	single.dim = args.dim;
	single.token_count = args.token_count;
	single.query_bytes = args.dim * sizeof(float);
	single.key_bytes = args.kv_source == ROCM_ATTENTION_KV_SOURCE_DEVICE ? kv_head * args.dim : args.key_bytes;
	single.value_bytes = args.kv_source == ROCM_ATTENTION_KV_SOURCE_DEVICE ? kv_head * args.dim : args.value_bytes;
	single.output_bytes = args.dim * sizeof(float);
	single.weight_bytes = args.token_count * sizeof(float);
	single.kv_source = args.kv_source;
	single.scale_bits = args.scale_bits;
	single.descriptor_pointer = args.descriptor_pointer;
	single.descriptor_bytes = args.descriptor_bytes;
	const uint32_t window_start = args.window_size > 0u && args.token_count > args.window_size
		? args.token_count - args.window_size
		: 0u;
	if (window_start > 0u) {
		rocm_run_single_head_attention_range_token_parallel(single, window_start, args.token_count);
		return;
	}
	if (blockDim.x <= 1) {
		if (threadIdx.x == 0) {
			rocm_run_single_head_attention(single);
		}
		return;
	}
	if (args.weight_pointer == 0 || args.token_count >= 16) {
		rocm_run_single_head_attention_token_parallel(single, args.shared_mem_bytes);
		return;
	}
	rocm_run_single_head_attention_parallel(single);
}

extern "C" __global__ void rocm_attention_heads_batch_causal(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_attention_heads_batch_causal_launch_args &args = *reinterpret_cast<const rocm_attention_heads_batch_causal_launch_args *>(packet);
	if (!rocm_valid_attention_heads_batch_causal_args(args) ||
		blockIdx.x >= args.head_count ||
		blockIdx.y >= args.query_count) {
		return;
	}
	const uint64_t head = blockIdx.x;
	const uint64_t query_row = blockIdx.y;
	const uint32_t kv_head_count = rocm_attention_kv_head_count(args.reserved0, args.head_count);
	const uint32_t kv_head = rocm_attention_kv_head_for_query(static_cast<uint32_t>(head), args.head_count, kv_head_count);
	const uint64_t kv_pointer_offset = static_cast<uint64_t>(kv_head) * args.dim * sizeof(float);
	const uint32_t visible_tokens = args.query_start_token + static_cast<uint32_t>(query_row) + 1u;
	if (visible_tokens == 0 || visible_tokens > args.token_count) {
		return;
	}
	const uint32_t window_start = args.window_size > 0u && visible_tokens > args.window_size
		? visible_tokens - args.window_size
		: 0u;
	const uint64_t batch_head_index = query_row * args.head_count + head;
	rocm_attention_launch_args single = {};
	single.version = ROCM_ATTENTION_LAUNCH_ARGS_VERSION;
	single.total_bytes = ROCM_ATTENTION_LAUNCH_ARGS_BYTES;
	single.query_pointer = args.query_pointer + batch_head_index * args.dim * sizeof(float);
	single.key_pointer = args.kv_source == ROCM_ATTENTION_KV_SOURCE_CONTIGUOUS ? args.key_pointer + kv_pointer_offset : 0u;
	single.value_pointer = args.kv_source == ROCM_ATTENTION_KV_SOURCE_CONTIGUOUS ? args.value_pointer + kv_pointer_offset : 0u;
	single.output_pointer = args.output_pointer + batch_head_index * args.dim * sizeof(float);
	single.weight_pointer = args.weight_pointer == 0
		? 0
		: args.weight_pointer + batch_head_index * args.token_count * sizeof(float);
	single.dim = args.dim;
	single.token_count = visible_tokens;
	single.query_bytes = args.dim * sizeof(float);
	single.key_bytes = args.kv_source == ROCM_ATTENTION_KV_SOURCE_DEVICE ? kv_head * args.dim : args.key_bytes;
	single.value_bytes = args.kv_source == ROCM_ATTENTION_KV_SOURCE_DEVICE ? kv_head * args.dim : args.value_bytes;
	single.output_bytes = args.dim * sizeof(float);
	single.weight_bytes = visible_tokens * sizeof(float);
	single.kv_source = args.kv_source;
	single.scale_bits = args.scale_bits;
	single.descriptor_pointer = args.descriptor_pointer;
	single.descriptor_bytes = args.descriptor_bytes;
	// Contiguous KV derives its per-token stride from the full buffer (key_bytes / token_count),
	// so the causal window must be iterated as an explicit [window_start, visible_tokens) range with
	// the full token_count in scope. The visible-count fast paths below reduce single.token_count to
	// the window, which would divide the full buffer by the window and mis-stride every query row that
	// sees fewer than all tokens. The device KV path addresses via descriptors and is unaffected.
	if (window_start > 0u || args.kv_source == ROCM_ATTENTION_KV_SOURCE_CONTIGUOUS) {
		single.token_count = args.token_count;
		single.weight_bytes = args.token_count * sizeof(float);
		rocm_run_single_head_attention_range_token_parallel(single, window_start, visible_tokens);
		return;
	}
	if (blockDim.x <= 1) {
		if (threadIdx.x == 0) {
			rocm_run_single_head_attention(single);
		}
		return;
	}
	if (args.weight_pointer == 0 || visible_tokens >= 16) {
		rocm_run_single_head_attention_token_parallel(single, args.shared_mem_bytes);
		return;
	}
	rocm_run_single_head_attention_parallel(single);
}

// rocm_attention_heads_batch_capped is the multimodal sibling of causal batch
// attention. reserved1 points at one absolute visible-token count per query
// row and reserved2 carries its byte size. The whole K/V microbatch is landed
// before this kernel, so image/video rows may safely see through to their span
// end while ordinary rows retain their causal cap.
extern "C" __global__ void rocm_attention_heads_batch_capped(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_attention_heads_batch_causal_launch_args &args = *reinterpret_cast<const rocm_attention_heads_batch_causal_launch_args *>(packet);
	if (!rocm_valid_attention_heads_batch_capped_args(args) ||
		blockIdx.x >= args.head_count ||
		blockIdx.y >= args.query_count) {
		return;
	}
	const uint64_t head = blockIdx.x;
	const uint64_t query_row = blockIdx.y;
	const uint32_t *visible_caps = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args.reserved1));
	const uint32_t visible_tokens = visible_caps[query_row];
	if (visible_tokens == 0 || visible_tokens > args.token_count) {
		return;
	}
	const uint32_t kv_head_count = rocm_attention_kv_head_count(args.reserved0, args.head_count);
	const uint32_t kv_head = rocm_attention_kv_head_for_query(static_cast<uint32_t>(head), args.head_count, kv_head_count);
	const uint64_t kv_pointer_offset = static_cast<uint64_t>(kv_head) * args.dim * sizeof(float);
	const uint32_t window_start = args.window_size > 0u && visible_tokens > args.window_size
		? visible_tokens - args.window_size
		: 0u;
	const uint64_t batch_head_index = query_row * args.head_count + head;
	rocm_attention_launch_args single = {};
	single.version = ROCM_ATTENTION_LAUNCH_ARGS_VERSION;
	single.total_bytes = ROCM_ATTENTION_LAUNCH_ARGS_BYTES;
	single.query_pointer = args.query_pointer + batch_head_index * args.dim * sizeof(float);
	single.key_pointer = args.kv_source == ROCM_ATTENTION_KV_SOURCE_CONTIGUOUS ? args.key_pointer + kv_pointer_offset : 0u;
	single.value_pointer = args.kv_source == ROCM_ATTENTION_KV_SOURCE_CONTIGUOUS ? args.value_pointer + kv_pointer_offset : 0u;
	single.output_pointer = args.output_pointer + batch_head_index * args.dim * sizeof(float);
	single.weight_pointer = args.weight_pointer == 0
		? 0
		: args.weight_pointer + batch_head_index * args.token_count * sizeof(float);
	single.dim = args.dim;
	single.token_count = visible_tokens;
	single.query_bytes = args.dim * sizeof(float);
	single.key_bytes = args.kv_source == ROCM_ATTENTION_KV_SOURCE_DEVICE ? kv_head * args.dim : args.key_bytes;
	single.value_bytes = args.kv_source == ROCM_ATTENTION_KV_SOURCE_DEVICE ? kv_head * args.dim : args.value_bytes;
	single.output_bytes = args.dim * sizeof(float);
	single.weight_bytes = visible_tokens * sizeof(float);
	single.kv_source = args.kv_source;
	single.scale_bits = args.scale_bits;
	single.descriptor_pointer = args.descriptor_pointer;
	single.descriptor_bytes = args.descriptor_bytes;
	if (window_start > 0u || args.kv_source == ROCM_ATTENTION_KV_SOURCE_CONTIGUOUS) {
		single.token_count = args.token_count;
		single.weight_bytes = args.token_count * sizeof(float);
		rocm_run_single_head_attention_range_token_parallel(single, window_start, visible_tokens);
		return;
	}
	if (blockDim.x <= 1) {
		if (threadIdx.x == 0) {
			rocm_run_single_head_attention(single);
		}
		return;
	}
	if (args.weight_pointer == 0 || visible_tokens >= 16) {
		rocm_run_single_head_attention_token_parallel(single, args.shared_mem_bytes);
		return;
	}
	rocm_run_single_head_attention_parallel(single);
}

extern "C" __global__ void rocm_attention_heads_lane_batch(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_attention_heads_lane_batch_launch_args &args = *reinterpret_cast<const rocm_attention_heads_lane_batch_launch_args *>(packet);
	if (!rocm_valid_attention_heads_lane_batch_args(args) ||
		blockIdx.x >= args.head_count ||
		blockIdx.y >= args.lane_count) {
		return;
	}
	const uint64_t head = blockIdx.x;
	const uint64_t lane_index = blockIdx.y;
	const rocm_attention_heads_lane_descriptor *lanes = reinterpret_cast<const rocm_attention_heads_lane_descriptor *>(static_cast<uintptr_t>(args.lane_pointer));
	const rocm_attention_heads_lane_descriptor lane = lanes[lane_index];
	if (lane.descriptor_pointer == 0 ||
		lane.descriptor_bytes < ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES ||
		lane.token_count == 0 ||
		lane.token_count > args.max_token_count) {
		return;
	}
	const uint32_t kv_head = rocm_attention_kv_head_for_query(static_cast<uint32_t>(head), args.head_count, args.key_heads);
	const uint64_t batch_head_index = lane_index * args.head_count + head;
	rocm_attention_launch_args single = {};
	single.version = ROCM_ATTENTION_LAUNCH_ARGS_VERSION;
	single.total_bytes = ROCM_ATTENTION_LAUNCH_ARGS_BYTES;
	single.query_pointer = args.query_pointer + batch_head_index * args.dim * sizeof(float);
	single.output_pointer = args.output_pointer + batch_head_index * args.dim * sizeof(float);
	single.weight_pointer = args.weight_pointer == 0
		? 0
		: args.weight_pointer + batch_head_index * args.max_token_count * sizeof(float);
	single.dim = args.dim;
	single.token_count = lane.token_count;
	single.query_bytes = args.dim * sizeof(float);
	single.output_bytes = args.dim * sizeof(float);
	single.weight_bytes = lane.token_count * sizeof(float);
	single.kv_source = ROCM_ATTENTION_KV_SOURCE_DEVICE;
	single.scale_bits = args.scale_bits;
	single.descriptor_pointer = lane.descriptor_pointer;
	single.descriptor_bytes = lane.descriptor_bytes;
	if (!rocm_valid_device_kv_attention_descriptor_header(single)) {
		return;
	}
	// Device-KV attention reuses these fields as element offsets after the
	// descriptor header has been validated with the canonical zero values.
	single.key_bytes = kv_head * args.dim;
	single.value_bytes = kv_head * args.dim;
	const uint32_t window_start = lane.window_size > 0u && lane.token_count > lane.window_size
		? lane.token_count - lane.window_size
		: 0u;
	if (window_start > 0u) {
		rocm_run_single_head_attention_range_token_parallel(single, window_start, lane.token_count);
		return;
	}
	if (blockDim.x <= 1) {
		if (threadIdx.x == 0) {
			rocm_run_single_head_attention(single);
		}
		return;
	}
	if (args.weight_pointer == 0 || lane.token_count >= 16) {
		rocm_run_single_head_attention_token_parallel(single, args.shared_mem_bytes);
		return;
	}
	rocm_run_single_head_attention_parallel(single);
}

#if defined(__HIP_CPU_RT__)
// HIP-CPU does not provide the dynamically-sized extern shared-memory symbol
// required by this fused kernel. CPU callers must use the existing unfused
// RMSNorm/RoPE and attention kernels; abort if this entry point is launched so
// an unsupported fused path cannot silently return incorrect output.
extern "C" __global__ void rocm_attention_heads_batch_causal_query_rms_rope(const unsigned char *)
{
	__builtin_trap();
}
#else
extern "C" __global__ void rocm_attention_heads_batch_causal_query_rms_rope(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_attention_heads_batch_causal_query_rms_rope_launch_args &args = *reinterpret_cast<const rocm_attention_heads_batch_causal_query_rms_rope_launch_args *>(packet);
	if (!rocm_valid_attention_heads_batch_causal_query_rms_rope_args(args) ||
		blockIdx.x >= args.head_count ||
		blockIdx.y >= args.query_count) {
		return;
	}
	const uint64_t head = blockIdx.x;
	const uint64_t query_row = blockIdx.y;
	const uint32_t kv_head_count = args.key_heads == 0u ? 1u : args.key_heads;
	const uint32_t kv_head = rocm_attention_kv_head_for_query(static_cast<uint32_t>(head), args.head_count, kv_head_count);
	const uint64_t kv_pointer_offset = static_cast<uint64_t>(kv_head) * args.dim * sizeof(float);
	const uint32_t visible_tokens = args.query_start_token + static_cast<uint32_t>(query_row) + 1u;
	if (visible_tokens == 0 || visible_tokens > args.token_count) {
		return;
	}
	const uint32_t window_start = args.window_size > 0u && visible_tokens > args.window_size
		? visible_tokens - args.window_size
		: 0u;
	const uint64_t batch_head_index = query_row * args.head_count + head;
	const uint64_t query_shared_offset = rocm_attention_align_shared_offset(args.shared_mem_bytes, sizeof(float));
	extern __shared__ unsigned char fused_attention_shared[];
	float *query_shared = reinterpret_cast<float *>(fused_attention_shared + query_shared_offset);
	const float *query_input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.query_input_pointer)) + batch_head_index * args.dim;
	__shared__ float partial[32];
	float sum_squares = 0.0f;
	for (uint32_t i = threadIdx.x; i < args.dim; i += blockDim.x) {
		sum_squares += query_input[i] * query_input[i];
	}
	const float total = rocm_block_reduce_sum(sum_squares, partial);
	const float rms = sqrtf(total / static_cast<float>(args.dim) + rocm_float_from_bits(args.query_epsilon_bits));
	if (rms == 0.0f) {
		return;
	}
	const uint32_t frequency_dim = args.rope_frequency_dim == 0u ? args.dim : args.rope_frequency_dim;
	const uint32_t rotary_count = args.rope_rotary_count == 0u ? args.dim : args.rope_rotary_count;
	const float base = rocm_float_from_bits(args.rope_base_bits);
	const float frequency_scale = rocm_float_from_bits(args.rope_frequency_scale_bits);
	const float frequency_dim_f = static_cast<float>(frequency_dim);
	const uint32_t half_dim = args.dim >> 1u;
	const uint32_t position = args.rope_start_position + static_cast<uint32_t>(query_row);
	if ((args.query_flags & ROCM_RMS_NORM_LAUNCH_FLAG_ROPE_NEOX) != 0u) {
		const uint32_t active_pairs = rotary_count >> 1u;
		for (uint32_t pair = threadIdx.x; pair < half_dim; pair += blockDim.x) {
			const uint32_t i = pair;
			const uint32_t j = pair + half_dim;
			const float weight_x = rocm_rms_norm_weight_value(args.query_weight_pointer, args.query_weight_encoding, args.query_flags, i);
			const float weight_y = rocm_rms_norm_weight_value(args.query_weight_pointer, args.query_weight_encoding, args.query_flags, j);
			const float x = query_input[i] / rms * weight_x;
			const float y = query_input[j] / rms * weight_y;
			if (pair < active_pairs) {
				const float frequency = 1.0f / powf(base, static_cast<float>(pair << 1u) / frequency_dim_f);
				const float angle = static_cast<float>(position) * frequency * frequency_scale;
				const float cosine = cosf(angle);
				const float sine = sinf(angle);
				query_shared[i] = x * cosine - y * sine;
				query_shared[j] = x * sine + y * cosine;
			} else {
				query_shared[i] = x;
				query_shared[j] = y;
			}
		}
	} else {
		for (uint32_t pair = threadIdx.x; pair < half_dim; pair += blockDim.x) {
			const uint32_t i = pair << 1u;
			const float weight_x = rocm_rms_norm_weight_value(args.query_weight_pointer, args.query_weight_encoding, args.query_flags, i);
			const float weight_y = rocm_rms_norm_weight_value(args.query_weight_pointer, args.query_weight_encoding, args.query_flags, i + 1u);
			const float x = query_input[i] / rms * weight_x;
			const float y = query_input[i + 1u] / rms * weight_y;
			if (i < rotary_count) {
				const float frequency = 1.0f / powf(base, static_cast<float>(i) / frequency_dim_f);
				const float angle = static_cast<float>(position) * frequency * frequency_scale;
				const float cosine = cosf(angle);
				const float sine = sinf(angle);
				query_shared[i] = x * cosine - y * sine;
				query_shared[i + 1u] = x * sine + y * cosine;
			} else {
				query_shared[i] = x;
				query_shared[i + 1u] = y;
			}
		}
	}
	__syncthreads();
	rocm_attention_launch_args single = {};
	single.version = ROCM_ATTENTION_LAUNCH_ARGS_VERSION;
	single.total_bytes = ROCM_ATTENTION_LAUNCH_ARGS_BYTES;
	single.query_pointer = reinterpret_cast<uint64_t>(query_shared);
	single.key_pointer = args.kv_source == ROCM_ATTENTION_KV_SOURCE_CONTIGUOUS ? args.key_pointer + kv_pointer_offset : 0u;
	single.value_pointer = args.kv_source == ROCM_ATTENTION_KV_SOURCE_CONTIGUOUS ? args.value_pointer + kv_pointer_offset : 0u;
	single.output_pointer = args.output_pointer + batch_head_index * args.dim * sizeof(float);
	single.weight_pointer = args.attention_weight_pointer == 0
		? 0
		: args.attention_weight_pointer + batch_head_index * args.token_count * sizeof(float);
	single.dim = args.dim;
	single.token_count = visible_tokens;
	single.query_bytes = args.dim * sizeof(float);
	single.key_bytes = args.kv_source == ROCM_ATTENTION_KV_SOURCE_DEVICE ? kv_head * args.dim : args.key_bytes;
	single.value_bytes = args.kv_source == ROCM_ATTENTION_KV_SOURCE_DEVICE ? kv_head * args.dim : args.value_bytes;
	single.output_bytes = args.dim * sizeof(float);
	single.weight_bytes = visible_tokens * sizeof(float);
	single.kv_source = args.kv_source;
	single.scale_bits = args.scale_bits;
	single.descriptor_pointer = args.descriptor_pointer;
	single.descriptor_bytes = args.descriptor_bytes;
	// Contiguous KV derives its per-token stride from the full buffer (key_bytes / token_count),
	// so the causal window must be iterated as an explicit [window_start, visible_tokens) range with
	// the full token_count in scope. The visible-count fast paths below reduce single.token_count to
	// the window, which would divide the full buffer by the window and mis-stride every query row that
	// sees fewer than all tokens. The device KV path addresses via descriptors and is unaffected.
	if (window_start > 0u || args.kv_source == ROCM_ATTENTION_KV_SOURCE_CONTIGUOUS) {
		single.token_count = args.token_count;
		single.weight_bytes = args.token_count * sizeof(float);
		rocm_run_single_head_attention_range_token_parallel(single, window_start, visible_tokens);
		return;
	}
	if (blockDim.x <= 1) {
		if (threadIdx.x == 0) {
			rocm_run_single_head_attention(single);
		}
		return;
	}
	if (args.attention_weight_pointer == 0 || visible_tokens >= 16) {
		rocm_run_single_head_attention_token_parallel(single, args.shared_mem_bytes);
		return;
	}
	rocm_run_single_head_attention_parallel(single);
}
#endif

extern "C" __global__ void rocm_vector_add(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_vector_add_launch_args &args = *reinterpret_cast<const rocm_vector_add_launch_args *>(packet);
	if (!rocm_valid_vector_add_args(args)) {
		return;
	}
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= args.count) {
		return;
	}
	const float *left = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.left_pointer));
	const float *right = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.right_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	output[index] = left[index] + right[index];
}

extern "C" __global__ void rocm_vector_add_scaled(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_vector_add_scaled_launch_args &args = *reinterpret_cast<const rocm_vector_add_scaled_launch_args *>(packet);
	if (!rocm_valid_vector_add_scaled_args(args)) {
		return;
	}
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= args.count) {
		return;
	}
	const float *left = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.left_pointer));
	const float *right = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.right_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	output[index] = (left[index] + right[index]) * rocm_float_from_bits(args.scale_bits);
}

extern "C" __global__ void rocm_vector_scale(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_vector_scale_launch_args &args = *reinterpret_cast<const rocm_vector_scale_launch_args *>(packet);
	if (!rocm_valid_vector_scale_args(args)) {
		return;
	}
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= args.count) {
		return;
	}
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	output[index] = input[index] * rocm_float_from_bits(args.scale_bits);
}

extern "C" __global__ void rocm_per_layer_input_transpose(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_per_layer_input_transpose_launch_args &args = *reinterpret_cast<const rocm_per_layer_input_transpose_launch_args *>(packet);
	if (!rocm_valid_per_layer_input_transpose_args(args)) {
		return;
	}
	const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	const uint64_t input_size = args.input_size;
	const uint64_t layer_count = args.layer_count;
	const uint64_t batch = args.batch;
	const uint64_t total = batch * layer_count * input_size;
	if (index >= total) {
		return;
	}
	const uint64_t input_value = index % input_size;
	const uint64_t layer = (index / input_size) % layer_count;
	const uint64_t token = index / (input_size * layer_count);
	const uint64_t output_index = (layer * batch + token) * input_size + input_value;
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	output[output_index] = input[index];
}

extern "C" __global__ void rocm_swiglu(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_swiglu_launch_args &args = *reinterpret_cast<const rocm_swiglu_launch_args *>(packet);
	if (!rocm_valid_swiglu_args(args)) {
		return;
	}
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= args.count) {
		return;
	}
	const float *gate = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.gate_pointer));
	const float *up = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.up_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	const float value = gate[index];
	output[index] = value / (1.0f + expf(-value)) * up[index];
}

extern "C" __global__ void rocm_gelu_tanh_multiply(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_gelu_tanh_mul_launch_args &args = *reinterpret_cast<const rocm_gelu_tanh_mul_launch_args *>(packet);
	if (!rocm_valid_gelu_tanh_mul_args(args)) {
		return;
	}
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= args.count) {
		return;
	}
	const float *gate = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.gate_pointer));
	const float *up = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.up_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	const float value = gate[index];
	const float value2 = value * value;
	const float value3 = value2 * value;
	const float tanh_arg = 0.7978845608028654f * (value + 0.044715f * value3);
	const float gelu = 0.5f * value * (1.0f + tanhf(tanh_arg));
	output[index] = gelu * up[index];
}

extern "C" __global__ void rocm_moe_router(const unsigned char *packet)
{
	if (blockIdx.x != 0 || packet == nullptr) {
		return;
	}
	const rocm_moe_router_launch_args &args = *reinterpret_cast<const rocm_moe_router_launch_args *>(packet);
	if (!rocm_valid_moe_router_args(args)) {
		return;
	}
	const float *logits = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.logit_pointer));
	int32_t *ids = reinterpret_cast<int32_t *>(static_cast<uintptr_t>(args.id_pointer));
	float *probs = reinterpret_cast<float *>(static_cast<uintptr_t>(args.prob_pointer));

	if (args.expert_count <= blockDim.x) {
		const uint32_t expert = threadIdx.x;
		if (expert < args.expert_count) {
			const float score = logits[expert];
			uint32_t rank = 0;
			for (uint32_t candidate = 0; candidate < args.expert_count; ++candidate) {
				const float candidate_score = logits[candidate];
				if (candidate_score > score || (candidate_score == score && candidate < expert)) {
					++rank;
				}
			}
			if (rank < args.top_k) {
				ids[rank] = static_cast<int32_t>(expert);
			}
		}
	} else if (threadIdx.x == 0) {
		for (uint32_t rank = 0; rank < args.top_k; ++rank) {
			int32_t best = -1;
			float best_score = -FLT_MAX;
			for (uint32_t expert = 0; expert < args.expert_count; ++expert) {
				bool used = false;
				for (uint32_t previous = 0; previous < rank; ++previous) {
					if (ids[previous] == static_cast<int32_t>(expert)) {
						used = true;
					}
				}
				if (used) {
					continue;
				}
				const float score = logits[expert];
				if (best < 0 || score > best_score || (score == best_score && expert < static_cast<uint32_t>(best))) {
					best = static_cast<int32_t>(expert);
					best_score = score;
				}
			}
			if (best < 0) {
				return;
			}
			ids[rank] = best;
		}
	}
	__syncthreads();
	if (threadIdx.x != 0) {
		return;
	}
	float max_logit = logits[ids[0]];
	for (uint32_t rank = 1; rank < args.top_k; ++rank) {
		max_logit = fmaxf(max_logit, logits[ids[rank]]);
	}
	float sum = 0.0f;
	for (uint32_t rank = 0; rank < args.top_k; ++rank) {
		sum += expf(logits[ids[rank]] - max_logit);
	}
	if (sum == 0.0f) {
		return;
	}
	for (uint32_t rank = 0; rank < args.top_k; ++rank) {
		probs[rank] = expf(logits[ids[rank]] - max_logit) / sum;
	}
	if (args.status_pointer != 0) {
		uint32_t *status = reinterpret_cast<uint32_t *>(static_cast<uintptr_t>(args.status_pointer));
		*status = ROCM_MOE_ROUTER_LAUNCH_STATUS_OK;
	}
}

extern "C" __global__ void rocm_moe_batch_gather_rows(const unsigned char *packet)
{
	if (packet == nullptr || blockDim.x != ROCM_MOE_BATCH_ROUTE_BLOCK_SIZE) {
		return;
	}
	const rocm_moe_batch_route_rows_launch_args &args = *reinterpret_cast<const rocm_moe_batch_route_rows_launch_args *>(packet);
	if (!rocm_valid_moe_batch_route_rows_args(args) || args.output_rows != args.route_count) {
		return;
	}
	const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	const uint64_t count = static_cast<uint64_t>(args.route_count) * args.row_width;
	if (index >= count) {
		return;
	}
	const uint32_t route = static_cast<uint32_t>(index / args.row_width);
	const uint32_t column = static_cast<uint32_t>(index - static_cast<uint64_t>(route) * args.row_width);
	const rocm_moe_batch_route_metadata *metadata = reinterpret_cast<const rocm_moe_batch_route_metadata *>(static_cast<uintptr_t>(args.metadata_pointer));
	const uint32_t row = metadata[route].row;
	if (row >= args.input_rows) {
		return;
	}
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	output[index] = input[static_cast<uint64_t>(row) * args.row_width + column];
}

extern "C" __global__ void rocm_moe_batch_scatter_routes(const unsigned char *packet)
{
	if (packet == nullptr || blockDim.x != ROCM_MOE_BATCH_ROUTE_BLOCK_SIZE) {
		return;
	}
	const rocm_moe_batch_route_rows_launch_args &args = *reinterpret_cast<const rocm_moe_batch_route_rows_launch_args *>(packet);
	if (!rocm_valid_moe_batch_route_rows_args(args) || args.input_rows != args.route_count) {
		return;
	}
	const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	const uint64_t count = static_cast<uint64_t>(args.route_count) * args.row_width;
	if (index >= count) {
		return;
	}
	const uint32_t route = static_cast<uint32_t>(index / args.row_width);
	const uint32_t column = static_cast<uint32_t>(index - static_cast<uint64_t>(route) * args.row_width);
	const rocm_moe_batch_route_metadata *metadata = reinterpret_cast<const rocm_moe_batch_route_metadata *>(static_cast<uintptr_t>(args.metadata_pointer));
	const uint32_t pair = metadata[route].pair;
	const float weight = metadata[route].weight;
	if (pair >= args.output_rows || !isfinite(weight)) {
		return;
	}
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	output[static_cast<uint64_t>(pair) * args.row_width + column] = input[index] * weight;
}

extern "C" __global__ void rocm_moe_batch_reduce_routes(const unsigned char *packet)
{
	if (packet == nullptr || blockDim.x != ROCM_MOE_BATCH_ROUTE_BLOCK_SIZE) {
		return;
	}
	const rocm_moe_batch_reduce_launch_args &args = *reinterpret_cast<const rocm_moe_batch_reduce_launch_args *>(packet);
	if (!rocm_valid_moe_batch_reduce_args(args)) {
		return;
	}
	const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	const uint64_t count = static_cast<uint64_t>(args.rows) * args.row_width;
	if (index >= count) {
		return;
	}
	const uint32_t row = static_cast<uint32_t>(index / args.row_width);
	const uint32_t column = static_cast<uint32_t>(index - static_cast<uint64_t>(row) * args.row_width);
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	float sum = 0.0f;
	for (uint32_t rank = 0; rank < args.top_k; ++rank) {
		const uint64_t pair = static_cast<uint64_t>(row) * args.top_k + rank;
		sum += input[pair * args.row_width + column];
	}
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	output[index] = sum;
}

extern "C" __global__ void rocm_moe_mlx_affine_routes(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_moe_mlx_affine_routes_launch_args &args = *reinterpret_cast<const rocm_moe_mlx_affine_routes_launch_args *>(packet);
	if (!rocm_valid_moe_mlx_affine_routes_args(args) ||
		blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE ||
		gridDim.x != ((args.rows - 1u) / ROCM_MLX_Q4_PROJECTION_ROWS_PER_BLOCK) + 1u ||
		gridDim.y != args.chunk_count ||
		blockIdx.y >= args.chunk_count) {
		return;
	}

	const rocm_moe_mlx_affine_routes_chunk *chunks = reinterpret_cast<const rocm_moe_mlx_affine_routes_chunk *>(static_cast<uintptr_t>(args.chunk_pointer));
	const rocm_moe_mlx_affine_routes_chunk &chunk = chunks[blockIdx.y];
	const bool gate_up = (args.flags & ROCM_MOE_MLX_AFFINE_ROUTES_FLAG_GATE_UP) != 0u;
	if (chunk.route_count == 0u || chunk.route_count > ROCM_MOE_MLX_AFFINE_ROUTES_PER_CHUNK ||
		(gate_up && (chunk.gate_up_weight_pointer == 0 || chunk.gate_up_scale_pointer == 0 || chunk.gate_up_bias_pointer == 0)) ||
		(!gate_up && (chunk.down_weight_pointer == 0 || chunk.down_scale_pointer == 0 || chunk.down_bias_pointer == 0))) {
		return;
	}
	for (uint32_t route_lane = 0; route_lane < chunk.route_count; ++route_lane) {
		if (chunk.pair_indices[route_lane] >= args.pair_count ||
			!isfinite(chunk.route_weights[route_lane]) ||
			(gate_up && chunk.token_rows[route_lane] >= args.input_rows) ||
			(!gate_up && chunk.pair_indices[route_lane] >= args.input_rows)) {
			return;
		}
	}

	const uint32_t row_lane = threadIdx.x / ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t col_lane = threadIdx.x - row_lane * ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW;
	const uint32_t row = blockIdx.x * ROCM_MLX_Q4_PROJECTION_ROWS_PER_BLOCK + row_lane;
	if (row >= args.rows) {
		return;
	}

	const float *input_base = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	float *output_base = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	const uint64_t packed_per_row = rocm_mlx_affine_packed_per_row(args.cols, args.bits);
	const uint32_t groups_per_row = args.cols / args.group_size;
	const uint32_t row_group_base = row * groups_per_row;

	if (gate_up) {
		const uint32_t *gate_up_weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(chunk.gate_up_weight_pointer));
		const uint16_t *gate_up_scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(chunk.gate_up_scale_pointer));
		const uint16_t *gate_up_biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(chunk.gate_up_bias_pointer));
		const uint32_t *up_weights = gate_up_weights + static_cast<uint64_t>(args.rows) * packed_per_row;
		const uint16_t *up_scales = gate_up_scales + static_cast<uint64_t>(args.rows) * groups_per_row;
		const uint16_t *up_biases = gate_up_biases + static_cast<uint64_t>(args.rows) * groups_per_row;
		float gate_sums[ROCM_MOE_MLX_AFFINE_ROUTES_PER_CHUNK] = {};
		float up_sums[ROCM_MOE_MLX_AFFINE_ROUTES_PER_CHUNK] = {};

		if (args.bits == 4u && (args.group_size & 7u) == 0u) {
			const uint32_t packed_per_group = args.group_size >> 3u;
			for (uint32_t group_col = col_lane; group_col < groups_per_row; group_col += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
				const uint32_t group = row_group_base + group_col;
				const float gate_scale = rocm_bfloat16_to_float(gate_up_scales[group]);
				const float gate_bias = rocm_bfloat16_to_float(gate_up_biases[group]);
				const float up_scale = rocm_bfloat16_to_float(up_scales[group]);
				const float up_bias = rocm_bfloat16_to_float(up_biases[group]);
				float gate_q_dots[ROCM_MOE_MLX_AFFINE_ROUTES_PER_CHUNK] = {};
				float up_q_dots[ROCM_MOE_MLX_AFFINE_ROUTES_PER_CHUNK] = {};
				float input_sums[ROCM_MOE_MLX_AFFINE_ROUTES_PER_CHUNK] = {};
				const uint32_t first_packed = group_col * packed_per_group;
				for (uint32_t group_packed = 0; group_packed < packed_per_group; ++group_packed) {
					const uint32_t packed = first_packed + group_packed;
					const uint32_t gate_word = gate_up_weights[static_cast<uint64_t>(row) * packed_per_row + packed];
					const uint32_t up_word = up_weights[static_cast<uint64_t>(row) * packed_per_row + packed];
					const uint32_t col = packed << 3u;
					for (uint32_t route_lane = 0; route_lane < chunk.route_count; ++route_lane) {
						const float *input = input_base + static_cast<uint64_t>(chunk.token_rows[route_lane]) * args.cols;
						const float in0 = input[col];
						const float in1 = input[col + 1u];
						const float in2 = input[col + 2u];
						const float in3 = input[col + 3u];
						const float in4 = input[col + 4u];
						const float in5 = input[col + 5u];
						const float in6 = input[col + 6u];
						const float in7 = input[col + 7u];
						input_sums[route_lane] += in0 + in1 + in2 + in3 + in4 + in5 + in6 + in7;
						gate_q_dots[route_lane] +=
							static_cast<float>(gate_word & 0x0fu) * in0 +
							static_cast<float>((gate_word >> 4u) & 0x0fu) * in1 +
							static_cast<float>((gate_word >> 8u) & 0x0fu) * in2 +
							static_cast<float>((gate_word >> 12u) & 0x0fu) * in3 +
							static_cast<float>((gate_word >> 16u) & 0x0fu) * in4 +
							static_cast<float>((gate_word >> 20u) & 0x0fu) * in5 +
							static_cast<float>((gate_word >> 24u) & 0x0fu) * in6 +
							static_cast<float>((gate_word >> 28u) & 0x0fu) * in7;
						up_q_dots[route_lane] +=
							static_cast<float>(up_word & 0x0fu) * in0 +
							static_cast<float>((up_word >> 4u) & 0x0fu) * in1 +
							static_cast<float>((up_word >> 8u) & 0x0fu) * in2 +
							static_cast<float>((up_word >> 12u) & 0x0fu) * in3 +
							static_cast<float>((up_word >> 16u) & 0x0fu) * in4 +
							static_cast<float>((up_word >> 20u) & 0x0fu) * in5 +
							static_cast<float>((up_word >> 24u) & 0x0fu) * in6 +
							static_cast<float>((up_word >> 28u) & 0x0fu) * in7;
					}
				}
				for (uint32_t route_lane = 0; route_lane < chunk.route_count; ++route_lane) {
					gate_sums[route_lane] += gate_scale * gate_q_dots[route_lane] + gate_bias * input_sums[route_lane];
					up_sums[route_lane] += up_scale * up_q_dots[route_lane] + up_bias * input_sums[route_lane];
				}
			}
		} else if (args.bits == 6u && args.group_size == 64u) {
			const uint64_t row_packed_base = static_cast<uint64_t>(row) * packed_per_row;
			for (uint32_t group_col = col_lane; group_col < groups_per_row; group_col += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
				const uint32_t group = row_group_base + group_col;
				const float gate_scale = rocm_bfloat16_to_float(gate_up_scales[group]);
				const float gate_bias = rocm_bfloat16_to_float(gate_up_biases[group]);
				const float up_scale = rocm_bfloat16_to_float(up_scales[group]);
				const float up_bias = rocm_bfloat16_to_float(up_biases[group]);
				const uint32_t first_packed = group_col * 12u;
				const uint32_t first_col = group_col << 6u;
				for (uint32_t route_lane = 0; route_lane < chunk.route_count; ++route_lane) {
					const float *input = input_base + static_cast<uint64_t>(chunk.token_rows[route_lane]) * args.cols;
					float gate_q_dot = 0.0f;
					float up_q_dot = 0.0f;
					float input_sum = 0.0f;
					for (uint32_t block = 0; block < 4u; ++block) {
						rocm_mlx_affine_q6_16_pair_dot(input, gate_up_weights, up_weights, row_packed_base, first_packed + block * 3u, first_col + block * 16u, &gate_q_dot, &up_q_dot, &input_sum);
					}
					gate_sums[route_lane] += gate_scale * gate_q_dot + gate_bias * input_sum;
					up_sums[route_lane] += up_scale * up_q_dot + up_bias * input_sum;
				}
			}
		} else {
			const uint64_t row_packed_base = static_cast<uint64_t>(row) * packed_per_row;
			for (uint32_t col = col_lane; col < args.cols; col += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
				const uint32_t gate_q = rocm_mlx_affine_quantized_value(gate_up_weights, row_packed_base, col, args.bits);
				const uint32_t up_q = rocm_mlx_affine_quantized_value(up_weights, row_packed_base, col, args.bits);
				const uint32_t group = row_group_base + col / args.group_size;
				const float gate_scale = rocm_bfloat16_to_float(gate_up_scales[group]);
				const float gate_bias = rocm_bfloat16_to_float(gate_up_biases[group]);
				const float up_scale = rocm_bfloat16_to_float(up_scales[group]);
				const float up_bias = rocm_bfloat16_to_float(up_biases[group]);
				for (uint32_t route_lane = 0; route_lane < chunk.route_count; ++route_lane) {
					const float *input = input_base + static_cast<uint64_t>(chunk.token_rows[route_lane]) * args.cols;
					const float value = input[col];
					gate_sums[route_lane] += value * (static_cast<float>(gate_q) * gate_scale + gate_bias);
					up_sums[route_lane] += value * (static_cast<float>(up_q) * up_scale + up_bias);
				}
			}
		}

		for (uint32_t route_lane = 0; route_lane < chunk.route_count; ++route_lane) {
			const float gate_sum = rocm_mlx_q4_row_reduce(gate_sums[route_lane]);
			const float up_sum = rocm_mlx_q4_row_reduce(up_sums[route_lane]);
			if (col_lane == 0u) {
				output_base[static_cast<uint64_t>(chunk.pair_indices[route_lane]) * args.rows + row] = rocm_gelu_tanh_value(gate_sum) * up_sum;
			}
		}
		return;
	}

	const uint32_t *down_weights = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(chunk.down_weight_pointer));
	const uint16_t *down_scales = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(chunk.down_scale_pointer));
	const uint16_t *down_biases = reinterpret_cast<const uint16_t *>(static_cast<uintptr_t>(chunk.down_bias_pointer));
	float projection_sums[ROCM_MOE_MLX_AFFINE_ROUTES_PER_CHUNK] = {};
	if (args.bits == 4u && (args.group_size & 7u) == 0u) {
		const uint32_t packed_per_group = args.group_size >> 3u;
		for (uint32_t group_col = col_lane; group_col < groups_per_row; group_col += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
			const uint32_t group = row_group_base + group_col;
			const float scale = rocm_bfloat16_to_float(down_scales[group]);
			const float bias = rocm_bfloat16_to_float(down_biases[group]);
			float q_dots[ROCM_MOE_MLX_AFFINE_ROUTES_PER_CHUNK] = {};
			float input_sums[ROCM_MOE_MLX_AFFINE_ROUTES_PER_CHUNK] = {};
			const uint32_t first_packed = group_col * packed_per_group;
			for (uint32_t group_packed = 0; group_packed < packed_per_group; ++group_packed) {
				const uint32_t packed = first_packed + group_packed;
				const uint32_t word = down_weights[static_cast<uint64_t>(row) * packed_per_row + packed];
				const uint32_t col = packed << 3u;
				for (uint32_t route_lane = 0; route_lane < chunk.route_count; ++route_lane) {
					const float *input = input_base + static_cast<uint64_t>(chunk.pair_indices[route_lane]) * args.cols;
					const float in0 = input[col];
					const float in1 = input[col + 1u];
					const float in2 = input[col + 2u];
					const float in3 = input[col + 3u];
					const float in4 = input[col + 4u];
					const float in5 = input[col + 5u];
					const float in6 = input[col + 6u];
					const float in7 = input[col + 7u];
					input_sums[route_lane] += in0 + in1 + in2 + in3 + in4 + in5 + in6 + in7;
					q_dots[route_lane] +=
						static_cast<float>(word & 0x0fu) * in0 +
						static_cast<float>((word >> 4u) & 0x0fu) * in1 +
						static_cast<float>((word >> 8u) & 0x0fu) * in2 +
						static_cast<float>((word >> 12u) & 0x0fu) * in3 +
						static_cast<float>((word >> 16u) & 0x0fu) * in4 +
						static_cast<float>((word >> 20u) & 0x0fu) * in5 +
						static_cast<float>((word >> 24u) & 0x0fu) * in6 +
						static_cast<float>((word >> 28u) & 0x0fu) * in7;
				}
			}
			for (uint32_t route_lane = 0; route_lane < chunk.route_count; ++route_lane) {
				projection_sums[route_lane] += scale * q_dots[route_lane] + bias * input_sums[route_lane];
			}
		}
	} else if (args.bits == 6u && args.group_size == 64u) {
		const uint64_t row_packed_base = static_cast<uint64_t>(row) * packed_per_row;
		for (uint32_t group_col = col_lane; group_col < groups_per_row; group_col += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
			const uint32_t group = row_group_base + group_col;
			const float scale = rocm_bfloat16_to_float(down_scales[group]);
			const float bias = rocm_bfloat16_to_float(down_biases[group]);
			const uint32_t first_packed = group_col * 12u;
			const uint32_t first_col = group_col << 6u;
			for (uint32_t route_lane = 0; route_lane < chunk.route_count; ++route_lane) {
				const float *input = input_base + static_cast<uint64_t>(chunk.pair_indices[route_lane]) * args.cols;
				float q_dot = 0.0f;
				float input_sum = 0.0f;
				for (uint32_t block = 0; block < 4u; ++block) {
					rocm_mlx_affine_q6_16_dot(input, down_weights, row_packed_base, first_packed + block * 3u, first_col + block * 16u, &q_dot, &input_sum);
				}
				projection_sums[route_lane] += scale * q_dot + bias * input_sum;
			}
		}
	} else {
		const uint64_t row_packed_base = static_cast<uint64_t>(row) * packed_per_row;
		for (uint32_t col = col_lane; col < args.cols; col += ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW) {
			const uint32_t q = rocm_mlx_affine_quantized_value(down_weights, row_packed_base, col, args.bits);
			const uint32_t group = row_group_base + col / args.group_size;
			const float scale = rocm_bfloat16_to_float(down_scales[group]);
			const float bias = rocm_bfloat16_to_float(down_biases[group]);
			for (uint32_t route_lane = 0; route_lane < chunk.route_count; ++route_lane) {
				const float *input = input_base + static_cast<uint64_t>(chunk.pair_indices[route_lane]) * args.cols;
				projection_sums[route_lane] += input[col] * (static_cast<float>(q) * scale + bias);
			}
		}
	}

	for (uint32_t route_lane = 0; route_lane < chunk.route_count; ++route_lane) {
		const float projection_sum = rocm_mlx_q4_row_reduce(projection_sums[route_lane]);
		if (col_lane == 0u) {
			output_base[static_cast<uint64_t>(chunk.pair_indices[route_lane]) * args.rows + row] = projection_sum * chunk.route_weights[route_lane];
		}
	}
}

extern "C" __global__ void rocm_moe_lazy_experts(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_moe_lazy_launch_args &args = *reinterpret_cast<const rocm_moe_lazy_launch_args *>(packet);
	if (!rocm_valid_moe_lazy_args(args)) {
		return;
	}
	const uint32_t expert = blockIdx.x * blockDim.x + threadIdx.x;
	if (expert >= args.expert_count) {
		return;
	}
	const int32_t *ids = reinterpret_cast<const int32_t *>(static_cast<uintptr_t>(args.id_pointer));
	uint8_t *resident = reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(args.resident_pointer));
	uint8_t value = 0;
	for (uint32_t index = 0; index < args.selected_count; ++index) {
		if (ids[index] == static_cast<int32_t>(expert)) {
			value = 1;
		}
	}
	resident[expert] = value;
}

extern "C" __global__ void rocm_jangtq_projection(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_jangtq_launch_args &args = *reinterpret_cast<const rocm_jangtq_launch_args *>(packet);
	if (!rocm_valid_jangtq_args(args)) {
		return;
	}
	const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= args.rows) {
		return;
	}
	const float *input = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.input_pointer));
	const uint8_t *packed = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(args.packed_pointer));
	const float *bias = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.bias_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	const float scale = rocm_float_from_bits(args.scale_bits);
	float sum = 0.0f;
	if ((args.flags & ROCM_JANGTQ_LAUNCH_FLAG_BIAS) != 0) {
		sum = bias[row];
	}
	for (uint32_t col = 0; col < args.cols; ++col) {
		const uint32_t index = row * args.cols + col;
		const int8_t quantized = rocm_unpack_signed_bits(packed, args.bits, index);
		sum += input[col] * static_cast<float>(quantized) * scale;
	}
	output[row] = sum;
}

extern "C" __global__ void rocm_codebook_lookup(const unsigned char *packet)
{
	if (packet == nullptr) {
		return;
	}
	const rocm_codebook_launch_args &args = *reinterpret_cast<const rocm_codebook_launch_args *>(packet);
	if (!rocm_valid_codebook_args(args)) {
		return;
	}
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t output_count = args.code_count * args.code_dim;
	if (index >= output_count) {
		return;
	}
	const uint8_t *codes = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(args.code_pointer));
	const float *codebook = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.codebook_pointer));
	float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(args.output_pointer));
	const uint32_t code_index = index / args.code_dim;
	const uint32_t dim = index % args.code_dim;
	const uint32_t code = static_cast<uint32_t>(codes[code_index]);
	if (code >= args.codebook_count) {
		return;
	}
	output[index] = codebook[code * args.code_dim + dim];
}

extern "C" __global__ void rocm_tiny_prefill(const unsigned char *packet)
{
	if (blockIdx.x != 0 || threadIdx.x != 0 || packet == nullptr) {
		return;
	}
	const rocm_tiny_prefill_launch_args &args = *reinterpret_cast<const rocm_tiny_prefill_launch_args *>(packet);
	if (!rocm_valid_tiny_prefill_args(args)) {
		return;
	}
	const int32_t *tokens = reinterpret_cast<const int32_t *>(static_cast<uintptr_t>(args.token_pointer));
	const float *embedding = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.embedding_pointer));
	float *logits = reinterpret_cast<float *>(static_cast<uintptr_t>(args.logit_pointer));
	float *attention = reinterpret_cast<float *>(static_cast<uintptr_t>(args.attention_pointer));
	float *state_keys = reinterpret_cast<float *>(static_cast<uintptr_t>(args.key_pointer));
	float *state_values = reinterpret_cast<float *>(static_cast<uintptr_t>(args.value_pointer));
	const float q8_scale = rocm_float_from_bits(args.q8_scale_bits);

	const int32_t query_token = tokens[args.token_count - 1];
	if (query_token < 0 || static_cast<uint32_t>(query_token) >= args.vocab_size) {
		return;
	}
	const uint32_t query_base = static_cast<uint32_t>(query_token) * args.hidden_size;
	const float scale = 1.0f / sqrtf(static_cast<float>(args.hidden_size));
	float max_score = -3.4028234663852886e38f;
	for (uint32_t token = 0; token < args.token_count; ++token) {
		const int32_t token_id = tokens[token];
		if (token_id < 0 || static_cast<uint32_t>(token_id) >= args.vocab_size) {
			return;
		}
		const uint32_t key_base = static_cast<uint32_t>(token_id) * args.hidden_size;
		const uint32_t state_base = token * args.hidden_size;
		float score = 0.0f;
		for (uint32_t dim = 0; dim < args.hidden_size; ++dim) {
			const float value = embedding[key_base + dim];
			state_keys[state_base + dim] = value;
			state_values[state_base + dim] = value;
			score += embedding[query_base + dim] * value;
		}
		score *= scale;
		attention[token] = score;
		if (score > max_score) {
			max_score = score;
		}
	}

	float attention_sum = 0.0f;
	for (uint32_t token = 0; token < args.token_count; ++token) {
		const float value = expf(attention[token] - max_score);
		attention[token] = value;
		attention_sum += value;
	}
	if (attention_sum == 0.0f) {
		return;
	}
	for (uint32_t token = 0; token < args.token_count; ++token) {
		attention[token] = attention[token] / attention_sum;
	}

	int32_t best_index = 0;
	float best_score = 0.0f;
	for (uint32_t vocab = 0; vocab < args.vocab_size; ++vocab) {
		float logit = 0.0f;
		const uint32_t weight_base = vocab * args.hidden_size;
		for (uint32_t dim = 0; dim < args.hidden_size; ++dim) {
			float context_value = 0.0f;
			for (uint32_t token = 0; token < args.token_count; ++token) {
				const uint32_t value_base = static_cast<uint32_t>(tokens[token]) * args.hidden_size;
				context_value += attention[token] * embedding[value_base + dim];
			}
			logit += context_value * rocm_tiny_output_weight_value(args.output_weight_pointer, args.output_weight_encoding, weight_base + dim, q8_scale);
		}
		logits[vocab] = logit;
		if (vocab == 0 || logit > best_score) {
			best_index = static_cast<int32_t>(vocab);
			best_score = logit;
		}
	}

	const uintptr_t result_pointer = static_cast<uintptr_t>(args.result_pointer);
	int32_t *result_index = reinterpret_cast<int32_t *>(result_pointer);
	float *result_score = reinterpret_cast<float *>(result_pointer + sizeof(int32_t));
	*result_index = best_index;
	*result_score = best_score;
}

extern "C" __global__ void rocm_tiny_decode(const unsigned char *packet)
{
	if (blockIdx.x != 0 || threadIdx.x != 0 || packet == nullptr) {
		return;
	}
	const rocm_tiny_decode_launch_args &args = *reinterpret_cast<const rocm_tiny_decode_launch_args *>(packet);
	if (!rocm_valid_tiny_decode_args(args)) {
		return;
	}
	const float *prior_keys = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.prior_key_pointer));
	const float *prior_values = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.prior_value_pointer));
	const float *embedding = reinterpret_cast<const float *>(static_cast<uintptr_t>(args.embedding_pointer));
	float *logits = reinterpret_cast<float *>(static_cast<uintptr_t>(args.logit_pointer));
	float *attention = reinterpret_cast<float *>(static_cast<uintptr_t>(args.attention_pointer));
	float *updated_keys = reinterpret_cast<float *>(static_cast<uintptr_t>(args.updated_key_pointer));
	float *updated_values = reinterpret_cast<float *>(static_cast<uintptr_t>(args.updated_value_pointer));
	const float q8_scale = rocm_float_from_bits(args.q8_scale_bits);

	const uint32_t token_base = args.token_id * args.hidden_size;
	for (uint32_t index = 0; index < args.prior_token_count * args.hidden_size; ++index) {
		updated_keys[index] = prior_keys[index];
		updated_values[index] = prior_values[index];
	}
	const uint32_t updated_base = args.prior_token_count * args.hidden_size;
	for (uint32_t dim = 0; dim < args.hidden_size; ++dim) {
		updated_keys[updated_base + dim] = embedding[token_base + dim];
		updated_values[updated_base + dim] = embedding[token_base + dim];
	}

	const float scale = 1.0f / sqrtf(static_cast<float>(args.hidden_size));
	float max_score = -3.4028234663852886e38f;
	const uint32_t total_tokens = args.prior_token_count + 1u;
	for (uint32_t token = 0; token < total_tokens; ++token) {
		const uint32_t key_base = token * args.hidden_size;
		float score = 0.0f;
		for (uint32_t dim = 0; dim < args.hidden_size; ++dim) {
			score += embedding[token_base + dim] * updated_keys[key_base + dim];
		}
		score *= scale;
		attention[token] = score;
		if (score > max_score) {
			max_score = score;
		}
	}

	float attention_sum = 0.0f;
	for (uint32_t token = 0; token < total_tokens; ++token) {
		const float value = expf(attention[token] - max_score);
		attention[token] = value;
		attention_sum += value;
	}
	if (attention_sum == 0.0f) {
		return;
	}
	for (uint32_t token = 0; token < total_tokens; ++token) {
		attention[token] = attention[token] / attention_sum;
	}

	int32_t best_index = 0;
	float best_score = 0.0f;
	for (uint32_t vocab = 0; vocab < args.vocab_size; ++vocab) {
		float logit = 0.0f;
		const uint32_t weight_base = vocab * args.hidden_size;
		for (uint32_t dim = 0; dim < args.hidden_size; ++dim) {
			float context_value = 0.0f;
			for (uint32_t token = 0; token < total_tokens; ++token) {
				context_value += attention[token] * updated_values[token * args.hidden_size + dim];
			}
			logit += context_value * rocm_tiny_output_weight_value(args.output_weight_pointer, args.output_weight_encoding, weight_base + dim, q8_scale);
		}
		logits[vocab] = logit;
		if (vocab == 0 || logit > best_score) {
			best_index = static_cast<int32_t>(vocab);
			best_score = logit;
		}
	}

	const uintptr_t result_pointer = static_cast<uintptr_t>(args.result_pointer);
	int32_t *result_index = reinterpret_cast<int32_t *>(result_pointer);
	float *result_score = reinterpret_cast<float *>(result_pointer + sizeof(int32_t));
	*result_index = best_index;
	*result_score = best_score;
}
