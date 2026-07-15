// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	core "dappco.re/go"
)

const hipKernelSourcePathForTest = "kernels/rocm_kernels.hip"
const hipKernelMakefilePathForTest = "../../../Makefile"

func TestHIPKernelSource_ExportsLaunchABI_Good(t *testing.T) {
	sourceBytes, err := os.ReadFile(hipKernelSourcePathForTest)
	core.RequireNoError(t, err)
	source := string(sourceBytes)

	for _, symbol := range []string{
		`extern "C" __global__ void rocm_prefill`,
		`extern "C" __global__ void rocm_decode`,
		`extern "C" __global__ void rocm_kv_encode_token`,
		`extern "C" __global__ void rocm_kv_encode_token_value_norm`,
		`extern "C" __global__ void rocm_kv_encode_token_value_norm_descriptor_append`,
		`extern "C" __global__ void rocm_kv_descriptor_append`,
		`extern "C" __global__ void rocm_projection`,
		`extern "C" __global__ void rocm_projection_batch`,
		`extern "C" __global__ void rocm_mlx_q4_projection`,
		`extern "C" __global__ void rocm_mlx_q4_projection_q4_g32_rows3840_cols15360`,
		`extern "C" __global__ void rocm_mlx_q4_projection_cols256`,
		`extern "C" __global__ void rocm_mlx_q4_projection_q6_g16_row16`,
		`extern "C" __global__ void rocm_mlx_q4_projection_q6_row16`,
		`extern "C" __global__ void rocm_mlx_q4_projection_q6_row32`,
		`extern "C" __global__ void rocm_mlx_q4_projection_q6_row64`,
		`extern "C" __global__ void rocm_mlx_q4_projection_batch`,
		`extern "C" __global__ void rocm_mlx_q4_projection_batch_q4_g64_tokens16`,
		`extern "C" __global__ void rocm_mlx_q4_projection_batch_q8_g64_row16_tokens16`,
		`extern "C" __global__ void rocm_mlx_q4_projection_batch_q6_row16`,
		`extern "C" __global__ void rocm_mlx_q4_projection_greedy`,
		`extern "C" __global__ void rocm_mlx_q4_projection_greedy_q6_row64`,
		`extern "C" __global__ void rocm_mlx_q4_projection_greedy_batch`,
		`extern "C" __global__ void rocm_mlx_q4_projection_greedy_batch_q6_row64`,
		`extern "C" __global__ void rocm_mlx_q4_projection_scores`,
		`extern "C" __global__ void rocm_mlx_q4_projection_scores_q6_row64`,
		`extern "C" __global__ void rocm_mlx_q4_projection_selected_greedy`,
		`extern "C" __global__ void rocm_mlx_q4_projection_selected_greedy_q6_row64`,
		`extern "C" __global__ void rocm_ordered_embedding_candidates`,
		`extern "C" __global__ void rocm_packed_topk`,
		`extern "C" __global__ void rocm_packed_topk_sample`,
		`extern "C" __global__ void rocm_mlx_q4_triple_projection`,
		`extern "C" __global__ void rocm_mlx_q4_triple_projection_q6_row16`,
		`extern "C" __global__ void rocm_mlx_q4_triple_projection_q6_row64`,
		`extern "C" __global__ void rocm_mlx_q4_pair_projection`,
		`extern "C" __global__ void rocm_mlx_q4_gelu_tanh_multiply`,
		`extern "C" __global__ void rocm_mlx_q4_gelu_tanh_multiply_q4_g32_cols1536_row16`,
		`extern "C" __global__ void rocm_mlx_q4_gelu_tanh_multiply_q4_g32_rows15360_cols3840`,
		`extern "C" __global__ void rocm_mlx_q4_gelu_tanh_multiply_q4_g32_rows15360_cols3840_row8`,
		`extern "C" __global__ void rocm_mlx_q4_gelu_tanh_mlp_q4_g32_cols1536_persistent`,
		`extern "C" __global__ void rocm_mlx_q4_gelu_tanh_multiply_q6_cols1536`,
		`extern "C" __global__ void rocm_mlx_q4_gelu_tanh_multiply_q6_cols1536_row32`,
		`extern "C" __global__ void rocm_mlx_q4_gelu_tanh_multiply_q6_cols1536_row64`,
		`extern "C" __global__ void rocm_mlx_q4_gelu_tanh_multiply_batch`,
		`extern "C" __global__ void rocm_moe_mlx_affine_routes`,
		`extern "C" __global__ void rocm_mlx_q4_gelu_tanh_projection`,
		`extern "C" __global__ void rocm_mlx_q4_gelu_tanh_projection_q6_row16`,
		`extern "C" __global__ void rocm_mlx_q4_gelu_tanh_projection_batch`,
		`extern "C" __global__ void rocm_rms_norm_residual_add_mlx_q4_gelu_tanh_projection`,
		`extern "C" __global__ void rocm_rms_norm`,
		`extern "C" __global__ void rocm_rms_norm_residual_add`,
		`extern "C" __global__ void rocm_rms_norm_residual_add_norm`,
		`extern "C" __global__ void rocm_rms_norm_heads`,
		`extern "C" __global__ void rocm_rms_norm_rope_heads`,
		`extern "C" __global__ void rocm_rms_norm_rope_heads_pair`,
		`extern "C" __global__ void rocm_rms_norm_rope_heads_batch`,
		`extern "C" __global__ void rocm_rms_norm_rope_heads_pair_lane_batch`,
		`extern "C" __global__ void rocm_moe_combine_norms`,
		`extern "C" __global__ void rocm_moe_batch_gather_rows`,
		`extern "C" __global__ void rocm_moe_batch_scatter_routes`,
		`extern "C" __global__ void rocm_moe_batch_reduce_routes`,
		`extern "C" __global__ void rocm_rope`,
		`extern "C" __global__ void rocm_rope_heads`,
		`extern "C" __global__ void rocm_greedy_sample`,
		`extern "C" __global__ void rocm_softcap_greedy_sample`,
		`extern "C" __global__ void rocm_attention`,
		`extern "C" __global__ void rocm_attention_heads`,
		`extern "C" __global__ void rocm_attention_heads_batch_causal`,
		`extern "C" __global__ void rocm_attention_heads_batch_capped`,
		`extern "C" __global__ void rocm_attention_heads_lane_batch`,
		`extern "C" __global__ void rocm_attention_heads_batch_causal_query_rms_rope`,
		`extern "C" __global__ void rocm_attention_heads_chunked_stage1`,
		`extern "C" __global__ void rocm_attention_heads_chunked_stage2`,
		`extern "C" __global__ void rocm_attention_heads_batch_chunked_stage1_v2`,
		`extern "C" __global__ void rocm_attention_heads_batch_chunked_stage1_gqa2`,
		`extern "C" __global__ void rocm_attention_heads_batch_chunked_stage2`,
		`extern "C" __global__ void rocm_vector_add`,
		`extern "C" __global__ void rocm_vector_add_scaled`,
		`extern "C" __global__ void rocm_vector_scale`,
		`extern "C" __global__ void rocm_per_layer_input_transpose`,
		`extern "C" __global__ void rocm_swiglu`,
		`extern "C" __global__ void rocm_gelu_tanh_multiply`,
		`extern "C" __global__ void rocm_moe_router`,
		`extern "C" __global__ void rocm_moe_lazy_experts`,
		`extern "C" __global__ void rocm_gguf_q4_0_projection`,
		`extern "C" __global__ void rocm_gguf_q4_0_gelu_tanh_gate_up`,
		`extern "C" __global__ void rocm_gguf_q4_0_selected_expert_gate_up`,
		`extern "C" __global__ void rocm_gguf_q4_0_selected_expert_down`,
		`extern "C" __global__ void rocm_gguf_q4_0_selected_expert_gate_up_pair16`,
		`extern "C" __global__ void rocm_gguf_q4_0_selected_expert_down_pair16`,
		`extern "C" __global__ void rocm_gguf_q4_k_selected_expert_gate_up`,
		`extern "C" __global__ void rocm_gguf_q5_1_selected_expert_down`,
		`extern "C" __global__ void rocm_gguf_q8_0_selected_expert_down`,
		`extern "C" __global__ void rocm_gguf_q4_k_selected_expert_gate_up_pair16`,
		`extern "C" __global__ void rocm_gguf_q4_k_selected_expert_gate_up_split_pair16`,
		`extern "C" __global__ void rocm_gguf_q4_k_expand_metadata`,
		`extern "C" __global__ void rocm_gguf_q4_k_expanded_selected_expert_gate_up_split_pair16`,
		`extern "C" __global__ void rocm_gguf_q5_1_selected_expert_down_pair16`,
		`extern "C" __global__ void rocm_gguf_q5_1_selected_expert_down_expert8_pair16`,
		`extern "C" __global__ void rocm_gguf_q8_0_selected_expert_down_pair16`,
		`extern "C" __global__ void rocm_jangtq_projection`,
		`extern "C" __global__ void rocm_codebook_lookup`,
		`extern "C" __global__ void rocm_lora_projection`,
		`extern "C" __global__ void rocm_embedding_lookup`,
		`extern "C" __global__ void rocm_embedding_lookup_greedy_token`,
		`extern "C" __global__ void rocm_diffusion_expected_embedding`,
		`extern "C" __global__ void rocm_diffusion_expected_embedding_affine_g64_rows16`,
		`extern "C" __global__ void rocm_embedding_mean_pool`,
		`extern "C" __global__ void rocm_rerank_cosine`,
		`extern "C" __global__ void rocm_tiny_prefill`,
		`extern "C" __global__ void rocm_tiny_decode`,
		`extern "C" __global__ void rocm_cross_entropy_loss`,
		`extern "C" __global__ void rocm_distillation_kl_loss`,
		`extern "C" __global__ void rocm_grpo_advantage`,
		`extern "C" __global__ void rocm_autoround_quantize`,
	} {
		core.AssertTrue(t, strings.Contains(source, symbol), symbol)
	}

	for _, abi := range []string{
		core.Sprintf("ROCM_PREFILL_LAUNCH_ARGS_VERSION = %d", hipPrefillLaunchArgsVersion),
		core.Sprintf("ROCM_PREFILL_LAUNCH_ARGS_BYTES = %d", hipPrefillLaunchArgsBytes),
		core.Sprintf("ROCM_DECODE_LAUNCH_ARGS_VERSION = %d", hipDecodeLaunchArgsVersion),
		core.Sprintf("ROCM_DECODE_LAUNCH_ARGS_HEADER_BYTES = %d", hipDecodeLaunchArgsHeaderBytes),
		core.Sprintf("ROCM_DECODE_LAUNCH_ARGS_BYTES = %d", hipDecodeLaunchArgsBytes),
		core.Sprintf("ROCM_DEVICE_KV_LAUNCH_DESCRIPTOR_BYTES = %d", rocmDeviceKVLaunchDescriptorBytes),
		core.Sprintf("ROCM_DEVICE_KV_DESCRIPTOR_VERSION = %d", rocmDeviceKVDescriptorVersion),
		core.Sprintf("ROCM_DEVICE_KV_DESCRIPTOR_HEADER_BYTES = %d", rocmDeviceKVDescriptorHeaderBytes),
		core.Sprintf("ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES = %d", rocmDeviceKVDescriptorPageBytes),
		core.Sprintf("ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_FP16 = %d", rocmDeviceKVDescriptorEncodingFP16),
		core.Sprintf("ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8 = %d", rocmDeviceKVDescriptorEncodingQ8),
		core.Sprintf("ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4 = %d", rocmDeviceKVDescriptorEncodingQ4),
		core.Sprintf("ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS = %d", rocmDeviceKVDescriptorEncodingQ8Rows),
		core.Sprintf("ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4_ROWS = %d", rocmDeviceKVDescriptorEncodingQ4Rows),
		core.Sprintf("ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q8_ROWS_INTERLEAVED = %d", rocmDeviceKVDescriptorEncodingQ8RowsI),
		core.Sprintf("ROCM_DEVICE_KV_DESCRIPTOR_ENCODING_Q4_ROWS_INTERLEAVED = %d", rocmDeviceKVDescriptorEncodingQ4RowsI),
		core.Sprintf("ROCM_KV_ENCODE_TOKEN_LAUNCH_ARGS_VERSION = %d", hipKVEncodeTokenLaunchArgsVersion),
		core.Sprintf("ROCM_KV_ENCODE_TOKEN_LAUNCH_ARGS_BYTES = %d", hipKVEncodeTokenLaunchArgsBytes),
		core.Sprintf("ROCM_KV_ENCODE_TOKEN_VALUE_NORM_LAUNCH_ARGS_VERSION = %d", hipKVEncodeTokenValueNormLaunchArgsVersion),
		core.Sprintf("ROCM_KV_ENCODE_TOKEN_VALUE_NORM_LAUNCH_ARGS_BYTES = %d", hipKVEncodeTokenValueNormLaunchArgsBytes),
		core.Sprintf("ROCM_KV_ENCODE_TOKEN_VALUE_NORM_DESCRIPTOR_APPEND_LAUNCH_ARGS_VERSION = %d", hipKVEncodeTokenValueNormDescriptorAppendLaunchArgsVersion),
		core.Sprintf("ROCM_KV_ENCODE_TOKEN_VALUE_NORM_DESCRIPTOR_APPEND_LAUNCH_ARGS_BYTES = %d", hipKVEncodeTokenValueNormDescriptorAppendLaunchArgsBytes),
		core.Sprintf("ROCM_KV_ENCODE_TOKEN_VALUE_NORM_MAX_HEADS = %d", hipKVEncodeTokenValueNormMaxHeads),
		core.Sprintf("ROCM_KV_ENCODE_TOKEN_BLOCK_SIZE = %d", hipKVEncodeTokenBlockSize),
		core.Sprintf("ROCM_KV_DESCRIPTOR_APPEND_LAUNCH_ARGS_VERSION = %d", hipKVDescriptorAppendLaunchArgsVersion),
		core.Sprintf("ROCM_KV_DESCRIPTOR_APPEND_LAUNCH_ARGS_BYTES = %d", hipKVDescriptorAppendLaunchArgsBytes),
		core.Sprintf("ROCM_KV_DESCRIPTOR_APPEND_BLOCK_SIZE = %d", hipKVDescriptorAppendBlockSize),
		core.Sprintf("ROCM_KV_DESCRIPTOR_APPEND_MODE_GROW_LAST_PAGE = %d", rocmKVDescriptorAppendModeGrowLastPage),
		core.Sprintf("ROCM_KV_DESCRIPTOR_APPEND_MODE_BUILD_SINGLE_PAGE = %d", rocmKVDescriptorAppendModeBuildSinglePage),
		core.Sprintf("ROCM_PROJECTION_LAUNCH_ARGS_VERSION = %d", hipProjectionLaunchArgsVersion),
		core.Sprintf("ROCM_PROJECTION_LAUNCH_ARGS_BYTES = %d", hipProjectionLaunchArgsBytes),
		core.Sprintf("ROCM_PROJECTION_BATCH_LAUNCH_ARGS_VERSION = %d", hipProjectionBatchLaunchArgsVersion),
		core.Sprintf("ROCM_PROJECTION_BATCH_LAUNCH_ARGS_BYTES = %d", hipProjectionBatchLaunchArgsBytes),
		core.Sprintf("ROCM_PROJECTION_WEIGHT_ENCODING_FP16 = %d", hipProjectionWeightEncodingFP16),
		core.Sprintf("ROCM_PROJECTION_WEIGHT_ENCODING_Q8 = %d", hipProjectionWeightEncodingQ8),
		core.Sprintf("ROCM_PROJECTION_WEIGHT_ENCODING_F32 = %d", hipProjectionWeightEncodingF32),
		core.Sprintf("ROCM_PROJECTION_WEIGHT_ENCODING_BF16 = %d", hipProjectionWeightEncodingBF16),
		core.Sprintf("ROCM_MLX_Q4_PROJECTION_LAUNCH_ARGS_VERSION = %d", hipMLXQ4ProjectionLaunchArgsVersion),
		core.Sprintf("ROCM_MLX_Q4_PROJECTION_LAUNCH_ARGS_BYTES = %d", hipMLXQ4ProjectionLaunchArgsBytes),
		core.Sprintf("ROCM_MLX_Q4_PROJECTION_BATCH_LAUNCH_ARGS_VERSION = %d", hipMLXQ4ProjectionBatchLaunchArgsVersion),
		core.Sprintf("ROCM_MLX_Q4_PROJECTION_BATCH_LAUNCH_ARGS_BYTES = %d", hipMLXQ4ProjectionBatchLaunchArgsBytes),
		core.Sprintf("ROCM_MLX_Q4_PROJECTION_GREEDY_BATCH_LAUNCH_ARGS_VERSION = %d", hipMLXQ4ProjectionGreedyBatchLaunchArgsVersion),
		core.Sprintf("ROCM_MLX_Q4_PROJECTION_GREEDY_BATCH_LAUNCH_ARGS_BYTES = %d", hipMLXQ4ProjectionGreedyBatchLaunchArgsBytes),
		core.Sprintf("ROCM_MLX_Q4_TRIPLE_PROJECTION_LAUNCH_ARGS_VERSION = %d", hipMLXQ4TripleProjLaunchArgsVersion),
		core.Sprintf("ROCM_MLX_Q4_TRIPLE_PROJECTION_LAUNCH_ARGS_BYTES = %d", hipMLXQ4TripleProjLaunchArgsBytes),
		core.Sprintf("ROCM_MLX_Q4_GELU_TANH_MUL_LAUNCH_ARGS_VERSION = %d", hipMLXQ4GELUTanhMulLaunchArgsVersion),
		core.Sprintf("ROCM_MLX_Q4_GELU_TANH_MUL_LAUNCH_ARGS_BYTES = %d", hipMLXQ4GELUTanhMulLaunchArgsBytes),
		core.Sprintf("ROCM_MLX_Q4_GELU_TANH_MUL_BATCH_LAUNCH_ARGS_VERSION = %d", hipMLXQ4GELUTanhMulBatchLaunchArgsVersion),
		core.Sprintf("ROCM_MLX_Q4_GELU_TANH_MUL_BATCH_LAUNCH_ARGS_BYTES = %d", hipMLXQ4GELUTanhMulBatchLaunchArgsBytes),
		core.Sprintf("ROCM_MLX_Q4_GELU_TANH_MLP_PERSISTENT_LAUNCH_ARGS_VERSION = %d", hipMLXQ4GELUTanhMLPPersistentLaunchArgsVersion),
		core.Sprintf("ROCM_MLX_Q4_GELU_TANH_MLP_PERSISTENT_LAUNCH_ARGS_BYTES = %d", hipMLXQ4GELUTanhMLPPersistentLaunchArgsBytes),
		core.Sprintf("ROCM_MLX_Q4_GELU_TANH_PROJ_LAUNCH_ARGS_VERSION = %d", hipMLXQ4GELUTanhProjLaunchArgsVersion),
		core.Sprintf("ROCM_MLX_Q4_GELU_TANH_PROJ_LAUNCH_ARGS_BYTES = %d", hipMLXQ4GELUTanhProjLaunchArgsBytes),
		core.Sprintf("ROCM_MLX_Q4_GELU_TANH_PROJ_BATCH_LAUNCH_ARGS_VERSION = %d", hipMLXQ4GELUTanhProjBatchLaunchArgsVersion),
		core.Sprintf("ROCM_MLX_Q4_GELU_TANH_PROJ_BATCH_LAUNCH_ARGS_BYTES = %d", hipMLXQ4GELUTanhProjBatchLaunchArgsBytes),
		core.Sprintf("ROCM_RMS_NORM_RESIDUAL_ADD_GELU_TANH_PROJ_LAUNCH_ARGS_VERSION = %d", hipRMSResidualAddGELUTanhProjLaunchArgsVersion),
		core.Sprintf("ROCM_RMS_NORM_RESIDUAL_ADD_GELU_TANH_PROJ_LAUNCH_ARGS_BYTES = %d", hipRMSResidualAddGELUTanhProjLaunchArgsBytes),
		core.Sprintf("ROCM_MLX_Q4_PROJECTION_BITS = %d", hipMLXQ4ProjectionBits),
		core.Sprintf("ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE = %d", hipMLXQ4ProjectionBlockSize),
		core.Sprintf("ROCM_MLX_Q4_PROJECTION_ROWS_PER_BLOCK = %d", hipMLXQ4ProjectionRowsPerBlock),
		core.Sprintf("ROCM_MLX_Q4_PROJECTION_GREEDY_ROWS_PER_BLOCK = %d", hipMLXQ4ProjectionGreedyRowsPerBlock),
		core.Sprintf("ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_ROWS_PER_BLOCK = %d", hipMLXQ4ProjectionGreedyQ6RowsPerBlock),
		core.Sprintf("ROCM_MLX_Q4_PROJECTION_BEST_BYTES = %d", hipMLXQ4ProjectionBestBytes),
		core.Sprintf("ROCM_PACKED_TOPK_LAUNCH_ARGS_VERSION = %d", hipPackedTopKLaunchArgsVersion),
		core.Sprintf("ROCM_PACKED_TOPK_LAUNCH_ARGS_BYTES = %d", hipPackedTopKLaunchArgsBytes),
		core.Sprintf("ROCM_PACKED_TOPK_SAMPLE_LAUNCH_ARGS_VERSION = %d", hipPackedTopKSampleLaunchArgsVersion),
		core.Sprintf("ROCM_PACKED_TOPK_SAMPLE_LAUNCH_ARGS_BYTES = %d", hipPackedTopKSampleLaunchArgsBytes),
		core.Sprintf("ROCM_ORDERED_EMBEDDING_CANDIDATES_LAUNCH_ARGS_VERSION = %d", hipOrderedEmbeddingCandidatesLaunchArgsVersion),
		core.Sprintf("ROCM_ORDERED_EMBEDDING_CANDIDATES_LAUNCH_ARGS_BYTES = %d", hipOrderedEmbeddingCandidatesLaunchArgsBytes),
		core.Sprintf("ROCM_ORDERED_EMBEDDING_CANDIDATES_BLOCK_SIZE = %d", hipOrderedEmbeddingCandidatesBlockSize),
		core.Sprintf("ROCM_PACKED_TOPK_MAX_K = %d", hipPackedTopKMaxK),
		core.Sprintf("ROCM_PACKED_TOPK_BLOCK_SIZE = %d", hipPackedTopKBlockSize),
		core.Sprintf("ROCM_PACKED_TOPK_CHUNK_SIZE = %d", hipPackedTopKChunkSize),
		core.Sprintf("ROCM_RMS_NORM_LAUNCH_ARGS_VERSION = %d", hipRMSNormLaunchArgsVersion),
		core.Sprintf("ROCM_RMS_NORM_LAUNCH_ARGS_BYTES = %d", hipRMSNormLaunchArgsBytes),
		core.Sprintf("ROCM_RMS_NORM_RESIDUAL_ADD_LAUNCH_ARGS_VERSION = %d", hipRMSNormResidualAddArgsVersion),
		core.Sprintf("ROCM_RMS_NORM_RESIDUAL_ADD_LAUNCH_ARGS_BYTES = %d", hipRMSNormResidualAddArgsBytes),
		core.Sprintf("ROCM_RMS_NORM_RESIDUAL_ADD_NORM_LAUNCH_ARGS_VERSION = %d", hipRMSNormResAddNormArgsVersion),
		core.Sprintf("ROCM_RMS_NORM_RESIDUAL_ADD_NORM_LAUNCH_ARGS_BYTES = %d", hipRMSNormResAddNormArgsBytes),
		core.Sprintf("ROCM_RMS_NORM_HEADS_LAUNCH_ARGS_VERSION = %d", hipRMSNormHeadsLaunchArgsVersion),
		core.Sprintf("ROCM_RMS_NORM_HEADS_LAUNCH_ARGS_BYTES = %d", hipRMSNormHeadsLaunchArgsBytes),
		core.Sprintf("ROCM_RMS_NORM_ROPE_HEADS_LAUNCH_ARGS_VERSION = %d", hipRMSNormRoPEHeadsLaunchArgsVersion),
		core.Sprintf("ROCM_RMS_NORM_ROPE_HEADS_LAUNCH_ARGS_BYTES = %d", hipRMSNormRoPEHeadsLaunchArgsBytes),
		core.Sprintf("ROCM_RMS_NORM_ROPE_HEADS_PAIR_LAUNCH_ARGS_VERSION = %d", hipRMSNormRoPEHeadsPairLaunchArgsVersion),
		core.Sprintf("ROCM_RMS_NORM_ROPE_HEADS_PAIR_LAUNCH_ARGS_BYTES = %d", hipRMSNormRoPEHeadsPairLaunchArgsBytes),
		core.Sprintf("ROCM_RMS_NORM_ROPE_HEADS_BATCH_LAUNCH_ARGS_VERSION = %d", hipRMSNormRoPEHeadsBatchLaunchArgsVersion),
		core.Sprintf("ROCM_RMS_NORM_ROPE_HEADS_BATCH_LAUNCH_ARGS_BYTES = %d", hipRMSNormRoPEHeadsBatchLaunchArgsBytes),
		core.Sprintf("ROCM_RMS_NORM_ROPE_HEADS_PAIR_LANE_BATCH_LAUNCH_ARGS_VERSION = %d", hipRMSNormRoPEHeadsPairLaneBatchLaunchArgsVersion),
		core.Sprintf("ROCM_RMS_NORM_ROPE_HEADS_PAIR_LANE_BATCH_LAUNCH_ARGS_BYTES = %d", hipRMSNormRoPEHeadsPairLaneBatchLaunchArgsBytes),
		core.Sprintf("ROCM_MOE_COMBINE_NORMS_LAUNCH_ARGS_VERSION = %d", hipMoECombineNormsLaunchArgsVersion),
		core.Sprintf("ROCM_MOE_COMBINE_NORMS_LAUNCH_ARGS_BYTES = %d", hipMoECombineNormsLaunchArgsBytes),
		core.Sprintf("ROCM_MOE_BATCH_ROUTE_ROWS_LAUNCH_ARGS_VERSION = %d", hipMoEBatchRouteRowsLaunchArgsVersion),
		core.Sprintf("ROCM_MOE_BATCH_ROUTE_ROWS_LAUNCH_ARGS_BYTES = %d", hipMoEBatchRouteRowsLaunchArgsBytes),
		core.Sprintf("ROCM_MOE_BATCH_REDUCE_LAUNCH_ARGS_VERSION = %d", hipMoEBatchReduceLaunchArgsVersion),
		core.Sprintf("ROCM_MOE_BATCH_REDUCE_LAUNCH_ARGS_BYTES = %d", hipMoEBatchReduceLaunchArgsBytes),
		core.Sprintf("ROCM_MOE_BATCH_ROUTE_METADATA_BYTES = %d", hipMoEBatchRouteMetadataBytes),
		core.Sprintf("ROCM_MOE_BATCH_ROUTE_BLOCK_SIZE = %d", hipMoEBatchRouteBlockSize),
		core.Sprintf("ROCM_RMS_NORM_WEIGHT_ENCODING_NONE = %d", hipRMSNormWeightEncodingNone),
		core.Sprintf("ROCM_RMS_NORM_WEIGHT_ENCODING_F32 = %d", hipRMSNormWeightEncodingF32),
		core.Sprintf("ROCM_RMS_NORM_WEIGHT_ENCODING_BF16 = %d", hipRMSNormWeightEncodingBF16),
		core.Sprintf("ROCM_RMS_NORM_LAUNCH_FLAG_ADD_UNIT_WEIGHT = %d", hipRMSNormLaunchFlagAddUnitWeight),
		core.Sprintf("ROCM_RMS_NORM_LAUNCH_FLAG_ROPE_NEOX = %d", hipRMSNormLaunchFlagRoPENeoX),
		core.Sprintf("ROCM_ROPE_LAUNCH_ARGS_VERSION = %d", hipRoPELaunchArgsVersion),
		core.Sprintf("ROCM_ROPE_LAUNCH_ARGS_BYTES = %d", hipRoPELaunchArgsBytes),
		core.Sprintf("ROCM_ROPE_HEADS_LAUNCH_ARGS_VERSION = %d", hipRoPEHeadsLaunchArgsVersion),
		core.Sprintf("ROCM_ROPE_HEADS_LAUNCH_ARGS_BYTES = %d", hipRoPEHeadsLaunchArgsBytes),
		core.Sprintf("ROCM_GREEDY_LAUNCH_ARGS_VERSION = %d", hipGreedyLaunchArgsVersion),
		core.Sprintf("ROCM_GREEDY_LAUNCH_ARGS_BYTES = %d", hipGreedyLaunchArgsBytes),
		core.Sprintf("ROCM_SOFTCAP_GREEDY_LAUNCH_ARGS_VERSION = %d", hipSoftcapGreedyLaunchArgsVersion),
		core.Sprintf("ROCM_SOFTCAP_GREEDY_LAUNCH_ARGS_BYTES = %d", hipSoftcapGreedyLaunchArgsBytes),
		core.Sprintf("ROCM_GREEDY_RESULT_BYTES = %d", hipGreedyResultBytes),
		core.Sprintf("ROCM_ATTENTION_LAUNCH_ARGS_VERSION = %d", hipAttentionLaunchArgsVersion),
		core.Sprintf("ROCM_ATTENTION_LAUNCH_ARGS_BYTES = %d", hipAttentionLaunchArgsBytes),
		core.Sprintf("ROCM_ATTENTION_HEADS_LAUNCH_ARGS_VERSION = %d", hipAttentionHeadsLaunchArgsVersion),
		core.Sprintf("ROCM_ATTENTION_HEADS_LAUNCH_ARGS_BYTES = %d", hipAttentionHeadsLaunchArgsBytes),
		core.Sprintf("ROCM_ATTENTION_HEADS_BATCH_CAUSAL_LAUNCH_ARGS_VERSION = %d", hipAttentionHeadsBatchCausalLaunchArgsVersion),
		core.Sprintf("ROCM_ATTENTION_HEADS_BATCH_CAUSAL_LAUNCH_ARGS_BYTES = %d", hipAttentionHeadsBatchCausalLaunchArgsBytes),
		core.Sprintf("ROCM_ATTENTION_HEADS_LANE_BATCH_LAUNCH_ARGS_VERSION = %d", hipAttentionHeadsLaneBatchLaunchArgsVersion),
		core.Sprintf("ROCM_ATTENTION_HEADS_LANE_BATCH_LAUNCH_ARGS_BYTES = %d", hipAttentionHeadsLaneBatchLaunchArgsBytes),
		core.Sprintf("ROCM_ATTENTION_HEADS_LANE_DESCRIPTOR_BYTES = %d", hipAttentionHeadsLaneDescriptorBytes),
		core.Sprintf("ROCM_ATTENTION_HEADS_BATCH_CAUSAL_QUERY_RMS_ROPE_LAUNCH_ARGS_VERSION = %d", hipAttentionHeadsBatchCausalQueryRMSRoPELaunchArgsVersion),
		core.Sprintf("ROCM_ATTENTION_HEADS_BATCH_CAUSAL_QUERY_RMS_ROPE_LAUNCH_ARGS_BYTES = %d", hipAttentionHeadsBatchCausalQueryRMSRoPELaunchArgsBytes),
		core.Sprintf("ROCM_ATTENTION_HEADS_SHARED_MAX_TOKENS = %d", hipAttentionHeadsSharedMaxTokens),
		core.Sprintf("ROCM_ATTENTION_HEADS_CHUNKED_LAUNCH_ARGS_VERSION = %d", hipAttentionHeadsChunkedLaunchArgsVersion),
		core.Sprintf("ROCM_ATTENTION_HEADS_CHUNKED_LAUNCH_ARGS_BYTES = %d", hipAttentionHeadsChunkedLaunchArgsBytes),
		core.Sprintf("ROCM_ATTENTION_HEADS_BATCH_CHUNKED_LAUNCH_ARGS_VERSION = %d", hipAttentionHeadsBatchChunkedLaunchArgsVersion),
		core.Sprintf("ROCM_ATTENTION_HEADS_BATCH_CHUNKED_LAUNCH_ARGS_BYTES = %d", hipAttentionHeadsBatchChunkedLaunchArgsBytes),
		core.Sprintf("ROCM_ATTENTION_HEADS_CHUNKED_BLOCK_SIZE = %d", hipAttentionHeadsChunkedBlockSize),
		core.Sprintf("ROCM_ATTENTION_HEADS_CHUNK_SIZE = %d", hipAttentionHeadsChunkSize),
		core.Sprintf("ROCM_ATTENTION_KV_SOURCE_CONTIGUOUS = %d", hipAttentionKVSourceContiguous),
		core.Sprintf("ROCM_ATTENTION_KV_SOURCE_DEVICE = %d", hipAttentionKVSourceDevice),
		core.Sprintf("ROCM_VECTOR_ADD_LAUNCH_ARGS_VERSION = %d", hipVectorAddLaunchArgsVersion),
		core.Sprintf("ROCM_VECTOR_ADD_LAUNCH_ARGS_BYTES = %d", hipVectorAddLaunchArgsBytes),
		core.Sprintf("ROCM_VECTOR_ADD_SCALED_LAUNCH_ARGS_VERSION = %d", hipVectorAddScaledLaunchArgsVersion),
		core.Sprintf("ROCM_VECTOR_ADD_SCALED_LAUNCH_ARGS_BYTES = %d", hipVectorAddScaledLaunchArgsBytes),
		core.Sprintf("ROCM_VECTOR_SCALE_LAUNCH_ARGS_VERSION = %d", hipVectorScaleLaunchArgsVersion),
		core.Sprintf("ROCM_VECTOR_SCALE_LAUNCH_ARGS_BYTES = %d", hipVectorScaleLaunchArgsBytes),
		core.Sprintf("ROCM_PER_LAYER_INPUT_TRANSPOSE_LAUNCH_ARGS_VERSION = %d", hipPerLayerInputTransposeLaunchArgsVersion),
		core.Sprintf("ROCM_PER_LAYER_INPUT_TRANSPOSE_LAUNCH_ARGS_BYTES = %d", hipPerLayerInputTransposeLaunchArgsBytes),
		core.Sprintf("ROCM_SWIGLU_LAUNCH_ARGS_VERSION = %d", hipSwiGLULaunchArgsVersion),
		core.Sprintf("ROCM_SWIGLU_LAUNCH_ARGS_BYTES = %d", hipSwiGLULaunchArgsBytes),
		core.Sprintf("ROCM_GELU_TANH_MUL_LAUNCH_ARGS_VERSION = %d", hipGELUTanhMulLaunchArgsVersion),
		core.Sprintf("ROCM_GELU_TANH_MUL_LAUNCH_ARGS_BYTES = %d", hipGELUTanhMulLaunchArgsBytes),
		core.Sprintf("ROCM_MOE_ROUTER_LAUNCH_ARGS_VERSION = %d", hipMoERouterLaunchArgsVersion),
		core.Sprintf("ROCM_MOE_ROUTER_LAUNCH_ARGS_BYTES = %d", hipMoERouterLaunchArgsBytes),
		core.Sprintf("ROCM_MOE_LAZY_LAUNCH_ARGS_VERSION = %d", hipMoELazyLaunchArgsVersion),
		core.Sprintf("ROCM_MOE_LAZY_LAUNCH_ARGS_BYTES = %d", hipMoELazyLaunchArgsBytes),
		core.Sprintf("ROCM_GGUF_Q4_0_PROJECTION_LAUNCH_ARGS_VERSION = %d", hipGGUFQ4_0ProjectionLaunchArgsVersion),
		core.Sprintf("ROCM_GGUF_Q4_0_PROJECTION_LAUNCH_ARGS_BYTES = %d", hipGGUFQ4_0ProjectionLaunchArgsBytes),
		core.Sprintf("ROCM_GGUF_Q4_K_EXPAND_LAUNCH_ARGS_VERSION = %d", hipGGUFQ4KExpandLaunchArgsVersion),
		core.Sprintf("ROCM_GGUF_Q4_K_EXPAND_LAUNCH_ARGS_BYTES = %d", hipGGUFQ4KExpandLaunchArgsBytes),
		core.Sprintf("ROCM_GGUF_Q4_0_SELECTED_EXPERTS_LAUNCH_ARGS_VERSION = %d", hipGGUFQ4_0SelectedExpertsLaunchArgsVersion),
		core.Sprintf("ROCM_GGUF_Q4_0_SELECTED_EXPERTS_LAUNCH_ARGS_BYTES = %d", hipGGUFQ4_0SelectedExpertsLaunchArgsBytes),
		core.Sprintf("ROCM_GGUF_Q4_0_SELECTED_EXPERTS_MAX_TOP_K = %d", hipGGUFQ4_0SelectedExpertsMaxTopK),
		core.Sprintf("ROCM_GGUF_EXPERT_FORMAT_Q4_0 = %d", hipGGUFExpertFormatQ4_0),
		core.Sprintf("ROCM_GGUF_EXPERT_FORMAT_Q4_K = %d", hipGGUFExpertFormatQ4K),
		core.Sprintf("ROCM_GGUF_EXPERT_FORMAT_Q5_1 = %d", hipGGUFExpertFormatQ5_1),
		core.Sprintf("ROCM_GGUF_EXPERT_FORMAT_Q8_0 = %d", hipGGUFExpertFormatQ8_0),
		core.Sprintf("ROCM_GGUF_EXPERT_FORMAT_Q4_K_EXPANDED = %d", hipGGUFExpertFormatQ4KExpanded),
		core.Sprintf("ROCM_JANGTQ_LAUNCH_ARGS_VERSION = %d", hipJANGTQLaunchArgsVersion),
		core.Sprintf("ROCM_JANGTQ_LAUNCH_ARGS_BYTES = %d", hipJANGTQLaunchArgsBytes),
		core.Sprintf("ROCM_CODEBOOK_LAUNCH_ARGS_VERSION = %d", hipCodebookLaunchArgsVersion),
		core.Sprintf("ROCM_CODEBOOK_LAUNCH_ARGS_BYTES = %d", hipCodebookLaunchArgsBytes),
		core.Sprintf("ROCM_LORA_LAUNCH_ARGS_VERSION = %d", hipLoRALaunchArgsVersion),
		core.Sprintf("ROCM_LORA_LAUNCH_ARGS_BYTES = %d", hipLoRALaunchArgsBytes),
		core.Sprintf("ROCM_EMBEDDING_LOOKUP_LAUNCH_ARGS_VERSION = %d", hipEmbeddingLookupLaunchArgsVersion),
		core.Sprintf("ROCM_EMBEDDING_LOOKUP_LAUNCH_ARGS_BYTES = %d", hipEmbeddingLookupLaunchArgsBytes),
		core.Sprintf("ROCM_EMBEDDING_TABLE_ENCODING_F32 = %d", hipEmbeddingTableEncodingF32),
		core.Sprintf("ROCM_EMBEDDING_TABLE_ENCODING_BF16 = %d", hipEmbeddingTableEncodingBF16),
		core.Sprintf("ROCM_EMBEDDING_TABLE_ENCODING_MLX_Q4 = %d", hipEmbeddingTableEncodingMLXQ4),
		core.Sprintf("ROCM_EMBEDDING_MEAN_POOL_LAUNCH_ARGS_VERSION = %d", hipEmbeddingMeanPoolLaunchArgsVersion),
		core.Sprintf("ROCM_EMBEDDING_MEAN_POOL_LAUNCH_ARGS_BYTES = %d", hipEmbeddingMeanPoolLaunchArgsBytes),
		core.Sprintf("ROCM_RERANK_COSINE_LAUNCH_ARGS_VERSION = %d", hipRerankCosineLaunchArgsVersion),
		core.Sprintf("ROCM_RERANK_COSINE_LAUNCH_ARGS_BYTES = %d", hipRerankCosineLaunchArgsBytes),
		core.Sprintf("ROCM_TINY_PREFILL_LAUNCH_ARGS_VERSION = %d", hipTinyPrefillLaunchArgsVersion),
		core.Sprintf("ROCM_TINY_PREFILL_LAUNCH_ARGS_BYTES = %d", hipTinyPrefillLaunchArgsBytes),
		core.Sprintf("ROCM_TINY_DECODE_LAUNCH_ARGS_VERSION = %d", hipTinyDecodeLaunchArgsVersion),
		core.Sprintf("ROCM_TINY_DECODE_LAUNCH_ARGS_BYTES = %d", hipTinyDecodeLaunchArgsBytes),
		core.Sprintf("ROCM_AUTOROUND_QUANTIZE_LAUNCH_ARGS_VERSION = %d", hipAutoRoundQuantizeLaunchArgsVersion),
		core.Sprintf("ROCM_AUTOROUND_QUANTIZE_LAUNCH_ARGS_BYTES = %d", hipAutoRoundQuantizeLaunchArgsBytes),
		core.Sprintf("ROCM_AUTOROUND_FORMAT_MXFP4 = %d", hipAutoRoundFormatMXFP4),
		core.Sprintf("ROCM_AUTOROUND_FORMAT_NVFP4 = %d", hipAutoRoundFormatNVFP4),
		core.Sprintf("ROCM_AUTOROUND_FORMAT_FP8 = %d", hipAutoRoundFormatFP8),
		core.Sprintf("ROCM_AUTOROUND_FORMAT_MXFP8 = %d", hipAutoRoundFormatMXFP8),
		core.Sprintf("ROCM_AUTOROUND_FORMAT_INT2 = %d", hipAutoRoundFormatINT2),
		core.Sprintf("ROCM_TINY_OUTPUT_WEIGHT_ENCODING_FP32 = %d", hipTinyOutputWeightEncodingFP32),
		core.Sprintf("ROCM_TINY_OUTPUT_WEIGHT_ENCODING_FP16 = %d", hipTinyOutputWeightEncodingFP16),
		core.Sprintf("ROCM_TINY_OUTPUT_WEIGHT_ENCODING_Q8 = %d", hipTinyOutputWeightEncodingQ8),
		core.Sprintf("ROCM_CROSS_ENTROPY_LOSS_LAUNCH_ARGS_VERSION = %d", hipCrossEntropyLossLaunchArgsVersion),
		core.Sprintf("ROCM_CROSS_ENTROPY_LOSS_LAUNCH_ARGS_BYTES = %d", hipCrossEntropyLossLaunchArgsBytes),
		core.Sprintf("ROCM_CROSS_ENTROPY_LOSS_OUTPUT_BYTES = %d", hipCrossEntropyLossOutputBytes),
		core.Sprintf("ROCM_DISTILLATION_KL_LOSS_LAUNCH_ARGS_VERSION = %d", hipDistillationKLLossLaunchArgsVersion),
		core.Sprintf("ROCM_DISTILLATION_KL_LOSS_LAUNCH_ARGS_BYTES = %d", hipDistillationKLLossLaunchArgsBytes),
		core.Sprintf("ROCM_DISTILLATION_KL_LOSS_OUTPUT_BYTES = %d", hipDistillationKLLossOutputBytes),
		core.Sprintf("ROCM_GRPO_ADVANTAGE_LAUNCH_ARGS_VERSION = %d", hipGRPOAdvantageLaunchArgsVersion),
		core.Sprintf("ROCM_GRPO_ADVANTAGE_LAUNCH_ARGS_BYTES = %d", hipGRPOAdvantageLaunchArgsBytes),
		"ROCM_PREFILL_LAUNCH_STATUS_OK",
		"ROCM_DECODE_LAUNCH_STATUS_OK",
		"ROCM_MOE_ROUTER_LAUNCH_STATUS_OK",
	} {
		core.AssertTrue(t, strings.Contains(source, abi), abi)
	}
}

func TestHIPKernelSource_DiffusionExpectedEmbeddingAffineG64Rows16_Good(t *testing.T) {
	sourceBytes, err := os.ReadFile(hipKernelSourcePathForTest)
	core.RequireNoError(t, err)
	kernel := hipKernelSourceFunctionBodyForTest(t, string(sourceBytes), `extern "C" __global__ void rocm_diffusion_expected_embedding_affine_g64_rows16`)

	core.AssertTrue(t, strings.Contains(kernel, `float sums[ROCM_DIFFUSION_EXPECTED_EMBEDDING_AFFINE_G64_ROWS_PER_BLOCK]`), "row-batched expected embedding must retain one ordered sum per row")
	core.AssertTrue(t, strings.Contains(kernel, `args.bits != 4u && args.bits != 8u`), "row-batched expected embedding must accept the production Q8 table and Q4 tables")
	core.AssertTrue(t, strings.Contains(kernel, `row_base + row_lane`), "row-batched expected embedding must address each row in its block")
	core.AssertTrue(t, strings.Contains(kernel, `for (uint32_t token = 0; token < args.vocab_size; ++token)`), "row-batched expected embedding must preserve vocabulary accumulation order")
	core.AssertTrue(t, !strings.Contains(kernel, `__shared__`), "row-batched expected embedding must not stage the vocabulary through shared memory")
	core.AssertTrue(t, !strings.Contains(kernel, `__syncthreads()`), "row-batched expected embedding must not add tile barriers")
}

func TestHIPKernelSource_AMDBuildDefaultsO3_Good(t *testing.T) {
	makefileBytes, err := os.ReadFile(hipKernelMakefilePathForTest)
	core.RequireNoError(t, err)
	makefile := string(makefileBytes)

	core.AssertTrue(t, strings.Contains(makefile, "AMD_HIP_OPT ?= -O3"), "AMD HIP kernels should default to the measured O3 optimization level")
	core.AssertTrue(t, strings.Contains(makefile, "$(AMD_HIP_OPT)"), "hip-amd target should use the configurable AMD HIP optimization flag")
}

func TestHIPKernelSource_MoECombineNormsFusesIndependentRMSNorms_Good(t *testing.T) {
	sourceBytes, err := os.ReadFile(hipKernelSourcePathForTest)
	core.RequireNoError(t, err)
	source := string(sourceBytes)

	kernel := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_moe_combine_norms`)
	core.AssertTrue(t, strings.Contains(kernel, `local_sum_squares`), "fused kernel must reduce the local vector independently")
	core.AssertTrue(t, strings.Contains(kernel, `expert_sum_squares`), "fused kernel must reduce the expert vector independently")
	core.AssertTrue(t, strings.Count(kernel, `rocm_block_reduce_sum`) == 2, "fused kernel must perform one reduction per input")
	core.AssertTrue(t, strings.Contains(kernel, `args.local_weight_pointer`), "fused kernel must use local RMS weights")
	core.AssertTrue(t, strings.Contains(kernel, `args.expert_weight_pointer`), "fused kernel must use expert RMS weights")
}

func TestHIPKernelSource_MLXQ4ProjectionGeometryMatchesLaunchConfig_Good(t *testing.T) {
	sourceBytes, err := os.ReadFile(hipKernelSourcePathForTest)
	core.RequireNoError(t, err)
	source := string(sourceBytes)

	core.AssertTrue(t, strings.Contains(source, `if (bits == 4u && group_size == 64u)`), "q4 row-sum must keep the Gemma group64 fast path")
	core.AssertTrue(t, strings.Contains(source, `const uint32_t groups_per_row = cols >> 6u`), "q4 group64 fast path must use shift-derived group count")
	core.AssertTrue(t, strings.Contains(source, `for (uint32_t group_packed = 0; group_packed < 8u; ++group_packed)`), "q4 group64 fast path must use fixed packed-word groups")
	core.AssertTrue(t, strings.Contains(source, `if (bits == 6u && group_size == 64u)`), "q6 row-sum must keep the Gemma group64 fast path")
	core.AssertTrue(t, strings.Contains(source, `rocm_mlx_affine_q6_16_dot`), "q6 row-sum must use fixed 16-value unpack blocks")
	core.AssertTrue(t, strings.Contains(source, `rocm_mlx_affine_q6_16_pair_dot`), "q6 fused gate/up path must share fixed 16-value unpack blocks")
	core.AssertTrue(t, strings.Contains(source, `rocm_mlx_affine_q6_16_batch_dot`), "q6 batch projection must use fixed 16-value unpack blocks")
	core.AssertTrue(t, strings.Contains(source, `rocm_mlx_affine_q6_16_pair_batch_dot`), "q6 batch fused gate/up path must share fixed 16-value unpack blocks")
	core.AssertTrue(t, strings.Contains(source, `rocm_mlx_affine_q6_quantized_value`), "q6 embedding/generic affine lookup must use specialized value extraction")
	core.AssertTrue(t, strings.Contains(source, `rocm_mlx_affine_q8_quantized_value`), "q8 embedding/generic affine lookup must use specialized value extraction")

	projection := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_mlx_q4_projection`)
	core.AssertTrue(t, strings.Contains(projection, `threadIdx.x / ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW`), "projection rows use normal row geometry")
	core.AssertTrue(t, strings.Contains(projection, `blockIdx.x * ROCM_MLX_Q4_PROJECTION_ROWS_PER_BLOCK + row_lane`), "projection grid uses normal row blocks")
	core.AssertTrue(t, !strings.Contains(projection, `ROCM_MLX_Q4_PROJECTION_GREEDY_ROWS_PER_BLOCK`), "projection must not use greedy row blocks")
	core.AssertTrue(t, !strings.Contains(projection, `ROCM_MLX_Q4_PROJECTION_GREEDY_THREADS_PER_ROW`), "projection must not use greedy row threads")

	cols256 := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_mlx_q4_projection_cols256`)
	core.AssertTrue(t, strings.Contains(cols256, `args.bits == 8u && args.group_size == 32u`), "cols256 projection must allow GGUF q8 group32 tensors")
	core.AssertTrue(t, strings.Contains(cols256, `args.bits == 4u || args.bits == 6u`), "cols256 projection must retain q4/q6 tensor support")
	core.AssertTrue(t, strings.Contains(cols256, `threadIdx.x / ROCM_MLX_Q4_PROJECTION_COLS256_THREADS_PER_ROW`), "cols256 projection rows use narrow row geometry")
	core.AssertTrue(t, strings.Contains(cols256, `blockIdx.x * ROCM_MLX_Q4_PROJECTION_COLS256_ROWS_PER_BLOCK + row_lane`), "cols256 projection grid uses narrow row blocks")

	q6Row16 := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_mlx_q4_projection_q6_row16`)
	core.AssertTrue(t, strings.Contains(q6Row16, `args.bits != 6u || args.group_size != 64u || args.cols < 1536u`), "q6 row16 projection must guard its specialized tensor shape")
	core.AssertTrue(t, strings.Contains(q6Row16, `threadIdx.x / ROCM_MLX_Q4_PROJECTION_Q6_ROW16_THREADS_PER_ROW`), "q6 row16 projection rows use narrow row geometry")
	core.AssertTrue(t, strings.Contains(q6Row16, `blockIdx.x * ROCM_MLX_Q4_PROJECTION_Q6_ROW16_ROWS_PER_BLOCK + row_lane`), "q6 row16 projection grid uses narrow row blocks")
	core.AssertTrue(t, strings.Contains(q6Row16, `rocm_mlx_q4_projection_q6_row16_reduce`), "q6 row16 projection uses matching row reduction width")

	q6Row32 := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_mlx_q4_projection_q6_row32`)
	core.AssertTrue(t, strings.Contains(q6Row32, `args.bits != 6u || args.group_size != 64u || args.cols < 1536u || args.cols > 2048u`), "q6 row32 projection must guard its specialized tensor shape")
	core.AssertTrue(t, strings.Contains(q6Row32, `threadIdx.x / ROCM_MLX_Q4_PROJECTION_Q6_ROW32_THREADS_PER_ROW`), "q6 row32 projection rows use narrow row geometry")
	core.AssertTrue(t, strings.Contains(q6Row32, `blockIdx.x * ROCM_MLX_Q4_PROJECTION_Q6_ROW32_ROWS_PER_BLOCK + row_lane`), "q6 row32 projection grid uses narrow row blocks")
	core.AssertTrue(t, strings.Contains(q6Row32, `rocm_mlx_q4_projection_q6_row32_reduce`), "q6 row32 projection uses matching row reduction width")

	q6Row64 := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_mlx_q4_projection_q6_row64`)
	core.AssertTrue(t, strings.Contains(q6Row64, `args.bits != 6u || args.group_size != 64u || args.cols < 1536u || args.cols > 2048u`), "q6 row64 projection must guard its specialized tensor shape")
	core.AssertTrue(t, strings.Contains(q6Row64, `threadIdx.x / ROCM_MLX_Q4_PROJECTION_Q6_ROW64_THREADS_PER_ROW`), "q6 row64 projection rows use row64 geometry")
	core.AssertTrue(t, strings.Contains(q6Row64, `blockIdx.x * ROCM_MLX_Q4_PROJECTION_Q6_ROW64_ROWS_PER_BLOCK + row_lane`), "q6 row64 projection grid uses row64 blocks")
	core.AssertTrue(t, strings.Contains(q6Row64, `rocm_mlx_q4_projection_q6_row64_reduce`), "q6 row64 projection uses matching row reduction width")

	batch := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_mlx_q4_projection_batch`)
	core.AssertTrue(t, strings.Contains(batch, `threadIdx.x / ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW`), "batch projection rows use normal row geometry")
	core.AssertTrue(t, strings.Contains(batch, `blockIdx.x * ROCM_MLX_Q4_PROJECTION_ROWS_PER_BLOCK + row_lane`), "batch projection grid uses normal row blocks")
	core.AssertTrue(t, strings.Contains(batch, `blockIdx.y * ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK`), "batch projection must use grid Y for token blocks")
	core.AssertTrue(t, strings.Contains(batch, `batch >= args.batch`), "batch projection must guard partial token blocks")
	core.AssertTrue(t, strings.Contains(batch, `+ batch * args.cols`), "batch projection input must be row-offset by batch")
	core.AssertTrue(t, strings.Contains(batch, `+ batch * args.rows`), "batch projection output must be row-offset by batch")
	core.AssertTrue(t, strings.Contains(batch, `args.bits == 6u && args.group_size == 64u`), "batch projection must keep the q6 group64 fast path")

	batchQ4G64Tokens16 := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_mlx_q4_projection_batch_q4_g64_tokens16`)
	core.AssertTrue(t, strings.Contains(batchQ4G64Tokens16, `args.bits != 4u || args.group_size != 64u`), "q4 group64 tokens16 batch projection must guard its quantization shape")
	core.AssertTrue(t, strings.Contains(batchQ4G64Tokens16, `ROCM_MLX_Q4_PROJECTION_BATCH_WIDE_TOKENS_PER_BLOCK`), "q4 group64 tokens16 batch projection must use its wider token tile")
	core.AssertTrue(t, strings.Contains(batchQ4G64Tokens16, `rocm_mlx_q4_row_reduce`), "q4 group64 tokens16 batch projection must retain ordered row reduction")

	batchQ8G64Tokens16 := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_mlx_q4_projection_batch_q8_g64_row16_tokens16`)
	core.AssertTrue(t, strings.Contains(batchQ8G64Tokens16, `args.bits != 8u || args.group_size != 64u`), "q8 group64 tokens16 batch projection must guard its quantization shape")
	core.AssertTrue(t, strings.Contains(batchQ8G64Tokens16, `ROCM_MLX_Q4_PROJECTION_BATCH_WIDE_TOKENS_PER_BLOCK`), "q8 group64 tokens16 batch projection must use the wider token tile")
	core.AssertTrue(t, strings.Contains(batchQ8G64Tokens16, `ROCM_MLX_Q4_PROJECTION_ROW16_THREADS_PER_ROW`), "q8 group64 tokens16 batch projection must use row16 lane geometry")
	core.AssertTrue(t, strings.Contains(batchQ8G64Tokens16, `rocm_mlx_q4_projection_row16_reduce`), "q8 group64 tokens16 batch projection must use the matching row16 reduction")

	batchQ6Row16 := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_mlx_q4_projection_batch_q6_row16`)
	core.AssertTrue(t, strings.Contains(batchQ6Row16, `args.bits != 6u || args.group_size != 64u || args.cols < 1536u`), "q6 row16 batch projection must guard its specialized tensor shape")
	core.AssertTrue(t, strings.Contains(batchQ6Row16, `threadIdx.x / ROCM_MLX_Q4_PROJECTION_Q6_ROW16_THREADS_PER_ROW`), "q6 row16 batch projection rows use row16 geometry")
	core.AssertTrue(t, strings.Contains(batchQ6Row16, `blockIdx.x * ROCM_MLX_Q4_PROJECTION_Q6_ROW16_ROWS_PER_BLOCK + row_lane`), "q6 row16 batch projection grid uses row16 row blocks")
	core.AssertTrue(t, strings.Contains(batchQ6Row16, `rocm_mlx_affine_q6_16_batch_dot`), "q6 row16 batch projection must keep fixed q6 unpacking")
	core.AssertTrue(t, strings.Contains(batchQ6Row16, `rocm_mlx_q4_projection_q6_row16_reduce`), "q6 row16 batch projection uses matching row reduction width")

	tripleQ6Row16 := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_mlx_q4_triple_projection_q6_row16`)
	core.AssertTrue(t, strings.Contains(tripleQ6Row16, `args.bits != 6u || args.group_size != 64u || args.cols < 1536u`), "q6 row16 triple projection must guard its specialized tensor shape")
	core.AssertTrue(t, strings.Contains(tripleQ6Row16, `threadIdx.x / ROCM_MLX_Q4_PROJECTION_Q6_ROW16_THREADS_PER_ROW`), "q6 row16 triple projection rows use narrow row geometry")
	core.AssertTrue(t, strings.Contains(tripleQ6Row16, `blockIdx.x * ROCM_MLX_Q4_PROJECTION_Q6_ROW16_ROWS_PER_BLOCK + row_lane`), "q6 row16 triple projection grid uses narrow row blocks")
	core.AssertTrue(t, strings.Contains(tripleQ6Row16, `rocm_mlx_q4_projection_q6_row16_reduce`), "q6 row16 triple projection uses matching row reduction width")

	tripleQ6Row64 := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_mlx_q4_triple_projection_q6_row64`)
	core.AssertTrue(t, strings.Contains(tripleQ6Row64, `args.bits != 6u || args.group_size != 64u || args.cols != 1536u`), "q6 row64 triple projection must guard its specialized tensor shape")
	core.AssertTrue(t, strings.Contains(tripleQ6Row64, `threadIdx.x / ROCM_MLX_Q4_PROJECTION_Q6_ROW64_THREADS_PER_ROW`), "q6 row64 triple projection rows use row64 geometry")
	core.AssertTrue(t, strings.Contains(tripleQ6Row64, `blockIdx.x * ROCM_MLX_Q4_PROJECTION_Q6_ROW64_ROWS_PER_BLOCK + row_lane`), "q6 row64 triple projection grid uses row64 blocks")
	core.AssertTrue(t, strings.Contains(tripleQ6Row64, `rocm_mlx_q4_projection_q6_row64_reduce`), "q6 row64 triple projection uses matching row reduction width")

	gelu := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_mlx_q4_gelu_tanh_multiply`)
	core.AssertTrue(t, strings.Contains(gelu, `threadIdx.x / ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW`), "GELU multiply rows use projection row geometry")
	core.AssertTrue(t, strings.Contains(gelu, `blockIdx.x * ROCM_MLX_Q4_PROJECTION_ROWS_PER_BLOCK + row_lane`), "GELU multiply grid uses projection row blocks")
	core.AssertTrue(t, strings.Contains(gelu, `args.group_size == 64u`), "GELU multiply must keep the Gemma group64 index fast path")
	core.AssertTrue(t, strings.Contains(gelu, `const uint32_t row_group_base = row * groups_per_row`), "GELU multiply must hoist the row group base")
	core.AssertTrue(t, strings.Contains(gelu, `row_group_base + (packed >> 3u)`), "GELU multiply group64 path must avoid runtime group division")

	geluQ4G32Cols1536Row16 := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_mlx_q4_gelu_tanh_multiply_q4_g32_cols1536_row16`)
	core.AssertTrue(t, strings.Contains(geluQ4G32Cols1536Row16, `args.bits != 4u || args.group_size != 32u || args.cols < 1536u || (args.cols % 32u) != 0u`), "q4 group32 row16 GELU multiply must guard its specialized tensor shape")
	core.AssertTrue(t, strings.Contains(geluQ4G32Cols1536Row16, `threadIdx.x / ROCM_MLX_Q4_GELU_TANH_Q4_G32_COLS1536_ROW16_THREADS_PER_ROW`), "q4 group32 cols1536 row16 GELU multiply rows use row16 geometry")
	core.AssertTrue(t, strings.Contains(geluQ4G32Cols1536Row16, `blockIdx.x * ROCM_MLX_Q4_GELU_TANH_Q4_G32_COLS1536_ROW16_ROWS_PER_BLOCK + row_lane`), "q4 group32 cols1536 row16 GELU multiply grid uses row16 blocks")
	core.AssertTrue(t, strings.Contains(geluQ4G32Cols1536Row16, `rocm_mlx_q4_gelu_tanh_q4_g32_cols1536_row16_reduce`), "q4 group32 cols1536 row16 GELU multiply uses matching row reduction width")

	geluQ6Cols1536 := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_mlx_q4_gelu_tanh_multiply_q6_cols1536`)
	core.AssertTrue(t, strings.Contains(geluQ6Cols1536, `args.bits != 6u || args.group_size != 64u || args.cols != 1536u`), "q6 cols1536 GELU multiply must guard its specialized tensor shape")
	core.AssertTrue(t, strings.Contains(geluQ6Cols1536, `threadIdx.x / ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_THREADS_PER_ROW`), "q6 cols1536 GELU multiply rows use narrow row geometry")
	core.AssertTrue(t, strings.Contains(geluQ6Cols1536, `blockIdx.x * ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_ROWS_PER_BLOCK + row_lane`), "q6 cols1536 GELU multiply grid uses narrow row blocks")
	core.AssertTrue(t, strings.Contains(geluQ6Cols1536, `rocm_mlx_q4_gelu_tanh_q6_cols1536_row_reduce`), "q6 cols1536 GELU multiply uses matching row reduction width")

	geluQ6Cols1536Row32 := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_mlx_q4_gelu_tanh_multiply_q6_cols1536_row32`)
	core.AssertTrue(t, strings.Contains(geluQ6Cols1536Row32, `args.bits != 6u || args.group_size != 64u || args.cols != 1536u || args.rows > 6144u`), "q6 cols1536 row32 GELU multiply must guard its specialized tensor shape")
	core.AssertTrue(t, strings.Contains(geluQ6Cols1536Row32, `threadIdx.x / ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_ROW32_THREADS_PER_ROW`), "q6 cols1536 row32 GELU multiply rows use row32 geometry")
	core.AssertTrue(t, strings.Contains(geluQ6Cols1536Row32, `blockIdx.x * ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_ROW32_ROWS_PER_BLOCK + row_lane`), "q6 cols1536 row32 GELU multiply grid uses row32 blocks")
	core.AssertTrue(t, strings.Contains(geluQ6Cols1536Row32, `rocm_mlx_q4_gelu_tanh_q6_cols1536_row32_reduce`), "q6 cols1536 row32 GELU multiply uses matching row reduction width")

	geluQ6Cols1536Row64 := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_mlx_q4_gelu_tanh_multiply_q6_cols1536_row64`)
	core.AssertTrue(t, strings.Contains(geluQ6Cols1536Row64, `args.bits != 6u || args.group_size != 64u || args.cols != 1536u || args.rows > 6144u`), "q6 cols1536 row64 GELU multiply must guard its specialized tensor shape")
	core.AssertTrue(t, strings.Contains(geluQ6Cols1536Row64, `threadIdx.x / ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_ROW64_THREADS_PER_ROW`), "q6 cols1536 row64 GELU multiply rows use row64 geometry")
	core.AssertTrue(t, strings.Contains(geluQ6Cols1536Row64, `blockIdx.x * ROCM_MLX_Q4_GELU_TANH_Q6_COLS1536_ROW64_ROWS_PER_BLOCK + row_lane`), "q6 cols1536 row64 GELU multiply grid uses row64 blocks")
	core.AssertTrue(t, strings.Contains(geluQ6Cols1536Row64, `rocm_mlx_q4_gelu_tanh_q6_cols1536_row64_reduce`), "q6 cols1536 row64 GELU multiply uses matching row reduction width")

	geluProjQ6Row16 := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_mlx_q4_gelu_tanh_projection_q6_row16`)
	core.AssertTrue(t, strings.Contains(geluProjQ6Row16, `args.bits != 6u || args.group_size != 64u || args.cols < 1536u`), "q6 row16 GELU projection must guard its specialized tensor shape")
	core.AssertTrue(t, strings.Contains(geluProjQ6Row16, `threadIdx.x / ROCM_MLX_Q4_PROJECTION_Q6_ROW16_THREADS_PER_ROW`), "q6 row16 GELU projection rows use narrow row geometry")
	core.AssertTrue(t, strings.Contains(geluProjQ6Row16, `blockIdx.x * ROCM_MLX_Q4_PROJECTION_Q6_ROW16_ROWS_PER_BLOCK + row_lane`), "q6 row16 GELU projection grid uses narrow row blocks")
	core.AssertTrue(t, strings.Contains(geluProjQ6Row16, `rocm_mlx_q4_projection_q6_row16_reduce`), "q6 row16 GELU projection uses matching row reduction width")

	geluBatch := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_mlx_q4_gelu_tanh_multiply_batch`)
	core.AssertTrue(t, strings.Contains(geluBatch, `threadIdx.x / ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW`), "batch GELU multiply rows use projection row geometry")
	core.AssertTrue(t, strings.Contains(geluBatch, `blockIdx.x * ROCM_MLX_Q4_PROJECTION_ROWS_PER_BLOCK + row_lane`), "batch GELU multiply grid uses projection row blocks")
	core.AssertTrue(t, strings.Contains(geluBatch, `blockIdx.y * ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK`), "batch GELU multiply must use grid Y for token blocks")
	core.AssertTrue(t, strings.Contains(geluBatch, `batch >= args.batch`), "batch GELU multiply must guard partial token blocks")
	core.AssertTrue(t, strings.Contains(geluBatch, `+ batch * args.cols`), "batch GELU multiply input must be row-offset by batch")
	core.AssertTrue(t, strings.Contains(geluBatch, `+ batch * args.rows`), "batch GELU multiply output must be row-offset by batch")
	core.AssertTrue(t, strings.Contains(geluBatch, `args.bits == 6u && args.group_size == 64u`), "batch GELU multiply must keep the q6 group64 fast path")

	geluProjBatch := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_mlx_q4_gelu_tanh_projection_batch`)
	core.AssertTrue(t, strings.Contains(geluProjBatch, `blockIdx.y * ROCM_MLX_Q4_PROJECTION_BATCH_TOKENS_PER_BLOCK`), "batch GELU projection must use grid Y for token blocks")
	core.AssertTrue(t, strings.Contains(geluProjBatch, `batch >= args.batch`), "batch GELU projection must guard partial token blocks")
	core.AssertTrue(t, strings.Contains(geluProjBatch, `+ batch * args.cols`), "batch GELU projection input must be row-offset by batch")
	core.AssertTrue(t, strings.Contains(geluProjBatch, `+ batch * args.rows`), "batch GELU projection output must be row-offset by batch")
	core.AssertTrue(t, strings.Contains(geluProjBatch, `args.bits == 6u && args.group_size == 64u`), "batch GELU projection must keep the q6 group64 fast path")

	greedy := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_mlx_q4_projection_greedy`)
	core.AssertTrue(t, strings.Contains(greedy, `threadIdx.x / ROCM_MLX_Q4_PROJECTION_GREEDY_THREADS_PER_ROW`), "greedy rows use greedy row geometry")
	core.AssertTrue(t, strings.Contains(greedy, `const uint32_t row_stride = gridDim.x * ROCM_MLX_Q4_PROJECTION_GREEDY_ROWS_PER_BLOCK`), "greedy blocks must stride over row chunks")
	core.AssertTrue(t, strings.Contains(greedy, `for (uint32_t row_base = blockIdx.x * ROCM_MLX_Q4_PROJECTION_GREEDY_ROWS_PER_BLOCK; row_base < args.rows; row_base += row_stride)`), "greedy blocks must visit their assigned row chunks")
	core.AssertTrue(t, strings.Contains(greedy, `const uint32_t row = row_base + row_lane`), "greedy rows must be derived from each row chunk")
	core.AssertTrue(t, strings.Contains(greedy, `args.suppress_pointer`), "greedy fallback must accept device suppress tokens")
	core.AssertTrue(t, strings.Contains(greedy, `!rocm_mlx_q4_token_suppressed(row, suppress_tokens, args.suppress_count)`), "greedy fallback must filter suppressed rows on device")
	core.AssertTrue(t, strings.Contains(greedy, `for (uint32_t index = 1u; index < ROCM_MLX_Q4_PROJECTION_GREEDY_ROWS_PER_BLOCK; ++index)`), "greedy best reduction uses one post-sync serial pass")
	core.AssertEqual(t, 1, strings.Count(greedy, `__syncthreads();`), "greedy must synchronize only for the final shared-memory reduction")
	core.AssertTrue(t, strings.Contains(greedy, `unsigned long long persistent_best = 0;`), "greedy must retain the exact packed maximum across chunks")
	core.AssertTrue(t, strings.Contains(greedy, `if (packed > persistent_best)`), "each row leader must update its packed maximum exactly")
	core.AssertTrue(t, strings.Contains(greedy, `block_best[row_lane] = persistent_best;`), "row leaders must publish their maximum after visiting every assigned chunk")
	core.AssertEqual(t, 1, strings.Count(greedy, `atomicMax(best, best_value)`), "greedy must issue one atomic maximum per block")
	core.AssertTrue(t, !strings.Contains(greedy, `ROCM_MLX_Q4_PROJECTION_GREEDY_ROWS_PER_BLOCK / 2u`), "greedy best reduction must not reintroduce repeated block barriers")

	greedyQ6Row64 := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_mlx_q4_projection_greedy_q6_row64`)
	core.AssertTrue(t, strings.Contains(greedyQ6Row64, `args.bits != 6u || args.group_size != 64u || args.cols < 1536u`), "q6 row64 greedy must guard its specialized tensor shape")
	core.AssertTrue(t, strings.Contains(greedyQ6Row64, `threadIdx.x / ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_THREADS_PER_ROW`), "q6 row64 greedy rows use narrow row geometry")
	core.AssertTrue(t, strings.Contains(greedyQ6Row64, `blockIdx.x * ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_ROWS_PER_BLOCK + row_lane`), "q6 row64 greedy grid uses narrow row blocks")
	core.AssertTrue(t, strings.Contains(greedyQ6Row64, `rocm_mlx_q4_greedy_q6_row_reduce`), "q6 row64 greedy uses matching row reduction width")
	core.AssertTrue(t, strings.Contains(greedyQ6Row64, `for (uint32_t index = 1u; index < ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_ROWS_PER_BLOCK; ++index)`), "q6 row64 greedy best reduction scans the matching per-block row count")

	greedyBatch := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_mlx_q4_projection_greedy_batch`)
	core.AssertTrue(t, strings.Contains(greedyBatch, `const uint32_t batch_index = blockIdx.y`), "batch greedy must map grid Y to input rows")
	core.AssertTrue(t, strings.Contains(greedyBatch, `static_cast<uint64_t>(batch_index) * args.cols`), "batch greedy must offset each input row")
	core.AssertTrue(t, strings.Contains(greedyBatch, `atomicMax(&best[batch_index], best_value)`), "batch greedy must write one best result per input row")
	core.AssertTrue(t, strings.Contains(greedyBatch, `!rocm_mlx_q4_token_suppressed(row, suppress_tokens, args.suppress_count)`), "batch greedy must filter suppressed rows on device")

	greedyBatchQ6Row64 := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_mlx_q4_projection_greedy_batch_q6_row64`)
	core.AssertTrue(t, strings.Contains(greedyBatchQ6Row64, `args.bits != 6u || args.group_size != 64u || args.cols < 1536u`), "q6 row64 batch greedy must guard its specialized tensor shape")
	core.AssertTrue(t, strings.Contains(greedyBatchQ6Row64, `threadIdx.x / ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_THREADS_PER_ROW`), "q6 row64 batch greedy rows use narrow row geometry")
	core.AssertTrue(t, strings.Contains(greedyBatchQ6Row64, `blockIdx.x * ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_ROWS_PER_BLOCK + row_lane`), "q6 row64 batch greedy grid uses narrow row blocks")
	core.AssertTrue(t, strings.Contains(greedyBatchQ6Row64, `atomicMax(&best[batch_index], best_value)`), "q6 row64 batch greedy must write one best result per input row")

	scores := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_mlx_q4_projection_scores`)
	core.AssertTrue(t, strings.Contains(scores, `threadIdx.x / ROCM_MLX_Q4_PROJECTION_GREEDY_THREADS_PER_ROW`), "score rows use greedy row geometry")
	core.AssertTrue(t, strings.Contains(scores, `blockIdx.x * ROCM_MLX_Q4_PROJECTION_GREEDY_ROWS_PER_BLOCK + row_lane`), "score grid uses greedy row blocks")
	core.AssertTrue(t, strings.Contains(scores, `scores[row] = packed`), "score projection writes one packed score per vocab row")
	core.AssertTrue(t, strings.Contains(scores, `!rocm_mlx_q4_token_suppressed(row, suppress_tokens, args.suppress_count)`), "score projection filters suppressed rows on device")

	scoresQ6Row64 := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_mlx_q4_projection_scores_q6_row64`)
	core.AssertTrue(t, strings.Contains(scoresQ6Row64, `args.bits != 6u || args.group_size != 64u || args.cols < 1536u`), "q6 row64 score projection must guard its specialized tensor shape")
	core.AssertTrue(t, strings.Contains(scoresQ6Row64, `threadIdx.x / ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_THREADS_PER_ROW`), "q6 row64 score projection rows use q6 greedy row geometry")
	core.AssertTrue(t, strings.Contains(scoresQ6Row64, `blockIdx.x * ROCM_MLX_Q4_PROJECTION_GREEDY_Q6_ROWS_PER_BLOCK + row_lane`), "q6 row64 score projection grid uses q6 greedy row blocks")
	core.AssertTrue(t, strings.Contains(scoresQ6Row64, `rocm_mlx_q4_greedy_q6_row_reduce`), "q6 row64 score projection uses matching row reduction width")
	core.AssertTrue(t, strings.Contains(scoresQ6Row64, `scores[row] = packed`), "q6 row64 score projection writes one packed score per vocab row")

	topK := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_packed_topk`)
	core.AssertTrue(t, strings.Contains(topK, `__shared__ unsigned long long scratch[ROCM_PACKED_TOPK_CHUNK_SIZE]`), "packed top-k uses shared chunk sort")
	core.AssertTrue(t, strings.Contains(topK, `blockIdx.x * args.chunk_size`), "packed top-k partitions scores by chunk")
	core.AssertTrue(t, strings.Contains(topK, `local ^ stride`), "packed top-k uses parallel compare-swap passes")
	core.AssertTrue(t, strings.Contains(topK, `output[blockIdx.x * args.top_k + threadIdx.x] = scratch[threadIdx.x]`), "packed top-k writes chunk-local candidates")
}

func TestHIPKernelSource_MoEMLXAffineRoutesABIAndSemantics_Good(t *testing.T) {
	sourceBytes, err := os.ReadFile(hipKernelSourcePathForTest)
	core.RequireNoError(t, err)
	source := string(sourceBytes)

	for _, abi := range []string{
		`ROCM_MOE_MLX_AFFINE_ROUTES_LAUNCH_ARGS_VERSION = 1`,
		`ROCM_MOE_MLX_AFFINE_ROUTES_LAUNCH_ARGS_BYTES = 80`,
		`ROCM_MOE_MLX_AFFINE_ROUTES_CHUNK_BYTES = 152`,
		`ROCM_MOE_MLX_AFFINE_ROUTES_PER_CHUNK = 8`,
		`ROCM_MOE_MLX_AFFINE_ROUTES_FLAG_GATE_UP = 1`,
		`struct rocm_moe_mlx_affine_routes_launch_args`,
		`struct rocm_moe_mlx_affine_routes_chunk`,
		`static_assert(sizeof(rocm_moe_mlx_affine_routes_launch_args) == ROCM_MOE_MLX_AFFINE_ROUTES_LAUNCH_ARGS_BYTES`,
		`static_assert(sizeof(rocm_moe_mlx_affine_routes_chunk) == ROCM_MOE_MLX_AFFINE_ROUTES_CHUNK_BYTES`,
	} {
		core.AssertContains(t, source, abi)
	}

	kernel := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_moe_mlx_affine_routes`)
	core.AssertContains(t, kernel, `rocm_valid_moe_mlx_affine_routes_args(args)`)
	core.AssertContains(t, kernel, `blockDim.x != ROCM_MLX_Q4_PROJECTION_BLOCK_SIZE`)
	core.AssertContains(t, kernel, `gridDim.x != ((args.rows - 1u) / ROCM_MLX_Q4_PROJECTION_ROWS_PER_BLOCK) + 1u`)
	core.AssertContains(t, kernel, `gridDim.y != args.chunk_count`)
	core.AssertContains(t, kernel, `chunk.route_count == 0u || chunk.route_count > ROCM_MOE_MLX_AFFINE_ROUTES_PER_CHUNK`)
	core.AssertContains(t, kernel, `chunk.token_rows[route_lane]`)
	core.AssertContains(t, kernel, `chunk.pair_indices[route_lane]`)
	core.AssertContains(t, kernel, `gate_up_weights + static_cast<uint64_t>(args.rows) * packed_per_row`)
	core.AssertContains(t, kernel, `if (args.bits == 4u && (args.group_size & 7u) == 0u)`)
	core.AssertContains(t, kernel, `rocm_mlx_affine_q6_16_pair_dot`)
	core.AssertContains(t, kernel, `rocm_mlx_affine_q6_16_dot`)
	core.AssertContains(t, kernel, `rocm_mlx_q4_row_reduce`)
	core.AssertContains(t, kernel, `rocm_gelu_tanh_value(gate_sum) * up_sum`)
	core.AssertContains(t, kernel, `projection_sum * chunk.route_weights[route_lane]`)
}

func TestHIPKernelSource_MLXQ8Group32UsesPackedDot_Good(t *testing.T) {
	source, err := os.ReadFile("kernels/rocm_kernels.hip")
	core.RequireNoError(t, err)
	text := string(source)
	core.AssertContains(t, text, "rocm_mlx_affine_q8_32_dot")
	core.AssertContains(t, text, "if (bits == 8u && group_size == 32u)")
}

func TestHIPKernelSource_MLXQ8Group32FusedGELUUsesPackedPairDot_Good(t *testing.T) {
	source, err := os.ReadFile("kernels/rocm_kernels.hip")
	core.RequireNoError(t, err)
	text := string(source)
	core.AssertContains(t, text, "rocm_mlx_affine_q8_32_pair_dot")
	gelu := hipKernelSourceFunctionBodyForTest(t, text, `extern "C" __global__ void rocm_mlx_q4_gelu_tanh_multiply`)
	core.AssertContains(t, gelu, "args.bits == 8u && args.group_size == 32u")
}

func TestHIPKernelSource_MLXQ8Group64UsesPackedDot_Good(t *testing.T) {
	source, err := os.ReadFile("kernels/rocm_kernels.hip")
	core.RequireNoError(t, err)
	text := string(source)
	projection := hipKernelSourceFunctionBodyForTest(t, text, `__device__ float rocm_mlx_q4_projection_row_sum(`)
	core.AssertContains(t, projection, "if (bits == 8u && group_size == 64u)")
	core.AssertContains(t, projection, "rocm_mlx_affine_q8_32_dot")
}

func TestHIPKernelSource_MLXQ8Group64FusedGELUUsesPackedPairDot_Good(t *testing.T) {
	source, err := os.ReadFile("kernels/rocm_kernels.hip")
	core.RequireNoError(t, err)
	text := string(source)
	gelu := hipKernelSourceFunctionBodyForTest(t, text, `extern "C" __global__ void rocm_mlx_q4_gelu_tanh_multiply`)
	core.AssertContains(t, gelu, "args.bits == 8u && args.group_size == 64u")
	core.AssertContains(t, gelu, "rocm_mlx_affine_q8_32_pair_dot")
}

func TestHIPKernelSource_MLXQ4Group64Rows3840Cols15360UsesRow16_Good(t *testing.T) {
	source, err := os.ReadFile("kernels/rocm_kernels.hip")
	core.RequireNoError(t, err)
	kernel := hipKernelSourceFunctionBodyForTest(t, string(source), `extern "C" __global__ void rocm_mlx_q4_projection_q4_g64_rows3840_cols15360_row16`)
	core.AssertContains(t, kernel, "args.rows != 3840u")
	core.AssertContains(t, kernel, "args.cols != 15360u")
	core.AssertContains(t, kernel, "args.group_size != 64u")
	core.AssertContains(t, kernel, "args.bits != 4u")
	core.AssertContains(t, kernel, "ROCM_MLX_Q4_PROJECTION_ROW16_THREADS_PER_ROW")
}

func TestHIPKernelSource_MLXQ4Group64Rows15360Cols3840FusedGELUUsesRow8_Good(t *testing.T) {
	source, err := os.ReadFile("kernels/rocm_kernels.hip")
	core.RequireNoError(t, err)
	kernel := hipKernelSourceFunctionBodyForTest(t, string(source), `extern "C" __global__ void rocm_mlx_q4_gelu_tanh_multiply_q4_g64_rows15360_cols3840_row8`)
	core.AssertContains(t, kernel, "args.rows != 15360u")
	core.AssertContains(t, kernel, "args.cols != 3840u")
	core.AssertContains(t, kernel, "args.group_size != 64u")
	core.AssertContains(t, kernel, "args.bits != 4u")
	core.AssertContains(t, kernel, "ROCM_MLX_Q4_PROJECTION_THREADS_PER_ROW")
	core.AssertContains(t, kernel, "rocm_mlx_affine_q4_64_pair_dot")
}

func TestHIPKernelSource_ProjectionUsesCoalescedBlockPerRow_Good(t *testing.T) {
	source, err := os.ReadFile(hipKernelSourcePathForTest)
	core.RequireNoError(t, err)
	kernel := hipKernelSourceFunctionBodyForTest(t, string(source), `extern "C" __global__ void rocm_projection`)
	core.AssertContains(t, kernel, "const uint32_t row = blockIdx.x")
	core.AssertContains(t, kernel, "for (uint32_t col = threadIdx.x; col < args.cols; col += blockDim.x)")
	core.AssertContains(t, kernel, "rocm_block_reduce_sum")
}

func TestHIPKernelSource_AutoRoundQuantizeGroupPacking_Good(t *testing.T) {
	sourceBytes, err := os.ReadFile(hipKernelSourcePathForTest)
	core.RequireNoError(t, err)
	source := string(sourceBytes)

	kernel := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_autoround_quantize`)
	core.AssertTrue(t, strings.Contains(source, `static_assert(sizeof(rocm_autoround_quantize_launch_args) == ROCM_AUTOROUND_QUANTIZE_LAUNCH_ARGS_BYTES`), "AutoRound launch ABI must be statically checked")
	core.AssertTrue(t, strings.Contains(source, `__device__ bool rocm_valid_autoround_quantize_args`), "AutoRound launch args must have a source-side validator")
	core.AssertTrue(t, strings.Contains(kernel, `const uint32_t group_index = blockIdx.x * blockDim.x + threadIdx.x`), "AutoRound quantize must launch over row/group work items")
	core.AssertTrue(t, strings.Contains(kernel, `scales[group_index] = scale`), "AutoRound quantize must emit one scale per row/group")
	core.AssertTrue(t, strings.Contains(kernel, `rocm_autoround_pack_signed`), "AutoRound quantize must pack signed quantized values")
}

func TestHIPKernelSource_EmbeddingGreedyTokenReadsPackedBest_Good(t *testing.T) {
	sourceBytes, err := os.ReadFile(hipKernelSourcePathForTest)
	core.RequireNoError(t, err)
	source := string(sourceBytes)

	embedding := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_embedding_lookup_greedy_token`)
	core.AssertTrue(t, strings.Contains(embedding, `rocm_valid_embedding_lookup_greedy_token_args(args)`), "greedy-token embedding must validate the packed-token launch shape")
	core.AssertTrue(t, strings.Contains(embedding, `const uint64_t *best`), "greedy-token embedding must read the packed q4 greedy result")
	core.AssertTrue(t, strings.Contains(embedding, `~static_cast<uint32_t>(*best)`), "greedy-token embedding must unpack the token ID from the q4 greedy result")
	core.AssertTrue(t, strings.Contains(embedding, `args.output_scale_bits == 0 ? 1.0f : rocm_float_from_bits(args.output_scale_bits)`), "greedy-token embedding must support fused output scaling")
	core.AssertTrue(t, strings.Contains(embedding, `rocm_embedding_lookup_store(args, index, token_id, index, output_scale)`), "greedy-token embedding must reuse the normal embedding table path")
}

func TestHIPKernelSource_MoERouterRanksExpertsInParallel_Good(t *testing.T) {
	sourceBytes, err := os.ReadFile(hipKernelSourcePathForTest)
	core.RequireNoError(t, err)
	router := hipKernelSourceFunctionBodyForTest(t, string(sourceBytes), `extern "C" __global__ void rocm_moe_router`)
	core.AssertTrue(t, strings.Contains(router, "const uint32_t expert = threadIdx.x;"))
	core.AssertTrue(t, strings.Contains(router, "for (uint32_t candidate = 0; candidate < args.expert_count; ++candidate)"))
	core.AssertTrue(t, strings.Contains(router, "rank < args.top_k"))
	core.AssertTrue(t, strings.Contains(router, "__syncthreads();"))
}

func TestHIPKernelSource_MoEBatchRoutesPreserveRouterRankOrder_Good(t *testing.T) {
	sourceBytes, err := os.ReadFile(hipKernelSourcePathForTest)
	core.RequireNoError(t, err)
	source := string(sourceBytes)
	scatter := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_moe_batch_scatter_routes`)
	reduce := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_moe_batch_reduce_routes`)

	core.AssertTrue(t, strings.Contains(scatter, `const uint32_t pair = metadata[route].pair;`), "grouped expert output must return to its original router pair")
	core.AssertTrue(t, strings.Contains(scatter, `output[static_cast<uint64_t>(pair) * args.row_width + column] = input[index] * weight;`), "scatter must assign each pair exactly once")
	core.AssertTrue(t, !strings.Contains(scatter, "atomic"), "scatter must not introduce unordered floating-point atomics")
	core.AssertTrue(t, strings.Contains(reduce, `for (uint32_t rank = 0; rank < args.top_k; ++rank)`), "reduction must visit router ranks in order")
	core.AssertTrue(t, strings.Contains(reduce, `const uint64_t pair = static_cast<uint64_t>(row) * args.top_k + rank;`), "reduction must index the original row/rank pair")
}

func TestHIPDriverCGOSource_HotOutputPointersUseResultWrappers_Good(t *testing.T) {
	sourceBytes, err := os.ReadFile("hip_driver_cgo.go")
	core.RequireNoError(t, err)
	source := string(sourceBytes)

	for _, symbol := range []string{
		`core_rocm_hip_malloc_result`,
		`core_rocm_hip_host_malloc_mapped_result`,
		`core_rocm_hip_host_malloc_pinned_result`,
		`core_rocm_hip_event_create_result`,
		`core_rocm_hip_module_load_data_result`,
		`core_rocm_hip_module_get_function_result`,
	} {
		core.AssertTrue(t, strings.Contains(source, symbol), "cgo driver must keep result-return wrapper "+symbol)
	}
	for _, goSideCall := range []string{
		`C.core_rocm_hip_malloc(&`,
		`C.core_rocm_hip_host_malloc_mapped(&`,
		`C.core_rocm_hip_host_malloc_pinned(&`,
		`C.core_rocm_hip_event_create(&`,
		`C.core_rocm_hip_module_load_data(&`,
		`C.core_rocm_hip_module_get_function(&`,
	} {
		core.AssertTrue(t, !strings.Contains(source, goSideCall), "hot cgo output pointer call must stay inside C wrapper: "+goSideCall)
	}
}

func TestHIPKernelSource_KVDescriptorAppendInPlaceSkipsSelfCopy_Good(t *testing.T) {
	sourceBytes, err := os.ReadFile(hipKernelSourcePathForTest)
	core.RequireNoError(t, err)
	source := string(sourceBytes)

	appendKernel := hipKernelSourceFunctionBodyForTest(t, source, `__device__ void rocm_kv_descriptor_append_execute`)
	core.AssertTrue(t, strings.Contains(appendKernel, `ROCM_KV_DESCRIPTOR_APPEND_MODE_BUILD_SINGLE_PAGE`), "descriptor append must build single-page tables on device")
	core.AssertTrue(t, strings.Contains(appendKernel, `args.previous_descriptor_pointer == args.output_descriptor_pointer`), "descriptor append must detect in-place table reuse")
	core.AssertTrue(t, strings.Contains(appendKernel, `args.previous_descriptor_pointer != args.output_descriptor_pointer`), "trimmed descriptor append must avoid parallel self-copy")
	core.AssertTrue(t, strings.Contains(appendKernel, `args.output_page_count == previous->page_count + 1u`), "descriptor append must keep the no-trim append shape guard")
	core.AssertTrue(t, strings.Contains(appendKernel, `previous->page_count * ROCM_DEVICE_KV_DESCRIPTOR_PAGE_BYTES`), "descriptor append must write only the appended page in-place")
}

func TestHIPKernelSource_AttentionChunkedStage1ScoreLaneReduction_Good(t *testing.T) {
	sourceBytes, err := os.ReadFile(hipKernelSourcePathForTest)
	core.RequireNoError(t, err)
	source := string(sourceBytes)

	chunkedLookup := hipKernelSourceFunctionBodyForTest(t, source, `__device__ const rocm_device_kv_page_descriptor *rocm_attention_heads_chunked_device_kv_page`)
	core.AssertTrue(t, strings.Contains(chunkedLookup, `rocm_attention_device_kv_page_from_descriptor(args.descriptor_pointer, token)`), "chunked lookup must call the descriptor-only lookup directly")
	core.AssertTrue(t, !strings.Contains(chunkedLookup, `rocm_attention_launch_args lookup`), "chunked lookup must not rebuild generic launch args per token")
	batchChunkedLookup := hipKernelSourceFunctionBodyForTest(t, source, `__device__ const rocm_device_kv_page_descriptor *rocm_attention_heads_batch_chunked_device_kv_page`)
	core.AssertTrue(t, strings.Contains(batchChunkedLookup, `rocm_attention_device_kv_page_from_descriptor(args.descriptor_pointer, token)`), "batch chunked lookup must call the descriptor-only lookup directly")
	core.AssertTrue(t, !strings.Contains(batchChunkedLookup, `rocm_attention_launch_args lookup`), "batch chunked lookup must not rebuild generic launch args per token")

	stage1 := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_attention_heads_chunked_stage1`)
	core.AssertTrue(t, strings.Contains(stage1, `device_kv_header->block_size == 1u`), "stage1 direct token-page fast path must reject mixed block/page MP4 KV streams")
	core.AssertTrue(t, strings.Contains(stage1, `rocm_attention_kv_head_for_query(head, args.head_count, args.key_heads)`), "stage1 must map each query head to its KV head")
	core.AssertTrue(t, strings.Contains(stage1, `const uint32_t kv_dim_offset = kv_head * args.dim`), "stage1 must offset K/V rows to the mapped KV head")
	core.AssertTrue(t, strings.Contains(stage1, `const uint32_t kv_width = args.key_heads * args.dim`), "stage1 must validate the complete multi-head KV row width")
	core.AssertTrue(t, strings.Contains(stage1, `rocm_attention_device_kv_dot_from_page_offset(page, true, token, query_values, args.dim, kv_dim_offset)`), "stage1 fallback must apply the mapped KV-head offset")
	core.AssertTrue(t, strings.Contains(stage1, `(score_lanes & (score_lanes - 1u)) == 0u`), "stage1 score lanes must stay shuffle-width safe")
	core.AssertTrue(t, strings.Contains(stage1, `local < local_count && lane == 0u`), "stage1 score lanes must resolve KV pages once per token")
	core.AssertTrue(t, strings.Contains(stage1, `key_pointer = rocm_shfl_u64(key_pointer, 0, static_cast<int>(score_lanes))`), "stage1 score lanes must broadcast key pointers from lane zero")
	core.AssertTrue(t, strings.Contains(stage1, `key_scale = rocm_shfl_float(key_scale, 0, static_cast<int>(score_lanes))`), "stage1 score lanes must broadcast key scales from lane zero")
	core.AssertTrue(t, strings.Contains(stage1, `rocm_shfl_down(partial_dot, score_lane, static_cast<int>(score_lanes))`), "stage1 score lane reduction must use ordered lane shuffles")
	core.AssertTrue(t, !strings.Contains(stage1, `scratch[tid] = partial_dot`), "stage1 score lane reduction must not reintroduce shared-memory score scratch")
	core.AssertTrue(t, strings.Contains(stage1, `__shared__ float value_scratch1[ROCM_ATTENTION_HEADS_CHUNKED_BLOCK_SIZE]`), "stage1 value reduction must keep a second value scratch buffer")
	core.AssertTrue(t, strings.Contains(stage1, `value_scratch1[tid] = partial1`), "stage1 value reduction must write dim1 before the shared barrier")
	core.AssertTrue(t, strings.Contains(stage1, `out1 += value_scratch1[value_group * pair_count + tid]`), "stage1 value reduction must reduce dim1 from the second scratch buffer")
	core.AssertTrue(t, !strings.Contains(stage1, `scratch[tid] = partial1`), "stage1 value reduction must not reintroduce the second shared-memory pass")
	validation := hipKernelSourceFunctionBodyForTest(t, source, `__device__ bool rocm_valid_attention_heads_chunked_args`)
	core.AssertTrue(t, strings.Contains(validation, `args.key_heads == 0`), "chunked validation must reject an empty KV-head topology")
	core.AssertTrue(t, strings.Contains(validation, `args.key_heads > args.head_count`), "chunked validation must reject more KV heads than query heads")
	core.AssertTrue(t, strings.Contains(validation, `args.head_count % args.key_heads != 0`), "chunked validation must require an integral GQA ratio")
	core.AssertTrue(t, strings.Contains(validation, `args.dim > (~0u) / args.key_heads`), "chunked validation must reject KV-row width overflow")
	core.AssertTrue(t, strings.Contains(validation, `args.key_heads > 1u && (args.dim & 1u) != 0u`), "chunked validation must reject odd packed-Q4 KV-head offsets")

	stage2 := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_attention_heads_chunked_stage2`)
	core.AssertTrue(t, strings.Contains(stage2, `const bool cached_chunk_weights = chunk_count <= threads`), "stage2 must cache per-chunk softmax weights when they fit in shared scratch")
	core.AssertTrue(t, strings.Contains(stage2, `scratch[tid] = chunk_sum == 0.0f ? 0.0f : rocm_fast_expf`), "stage2 must compute each cached chunk weight once")
	batchStage1 := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_attention_heads_batch_chunked_stage1_v2`)
	core.AssertTrue(t, strings.Contains(batchStage1, `device_kv_header->block_size == 1u`), "batch stage1 direct token-page fast path must reject mixed block/page MP4 KV streams")
	core.AssertTrue(t, strings.Contains(source, `uint32_t key_heads;`), "batch chunked launch ABI must expose the host-written key-head count")
	core.AssertTrue(t, strings.Contains(batchStage1, `rocm_attention_kv_head_for_query(head, args.head_count, kv_head_count)`), "batch stage1 must map each query head to its KV head")
	core.AssertTrue(t, strings.Contains(batchStage1, `const uint32_t kv_dim_offset = kv_head * args.dim`), "batch stage1 must offset K/V rows to the mapped KV head")
	gqa2Stage1 := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_attention_heads_batch_chunked_stage1_gqa2`)
	core.AssertTrue(t, strings.Contains(gqa2Stage1, `const uint32_t head0 = pair_index << 1u`), "GQA2 stage1 must map each block to two adjacent query heads")
	core.AssertTrue(t, strings.Contains(gqa2Stage1, `if (kv_head0 != kv_head1)`), "GQA2 stage1 must reject adjacent heads that do not share one KV head")
	core.AssertTrue(t, strings.Contains(gqa2Stage1, `float *partials1 =`), "GQA2 stage1 must preserve a distinct partial workspace row for the second head")
	core.AssertTrue(t, strings.Contains(gqa2Stage1, `float *stats1 =`), "GQA2 stage1 must preserve a distinct stats workspace row for the second head")
	core.AssertEqual(t, 2, strings.Count(gqa2Stage1, `rocm_attention_heads_batch_chunked_device_kv_page(args, token)`))
	core.AssertTrue(t, strings.Contains(gqa2Stage1, `query_values0[dim] * quantized_key`), "GQA2 stage1 must reuse each K element for the first query head")
	core.AssertTrue(t, strings.Contains(gqa2Stage1, `query_values1[dim] * quantized_key`), "GQA2 stage1 must reuse each K element for the second query head")
	core.AssertTrue(t, strings.Contains(gqa2Stage1, `const unsigned char packed = values[dim0 >> 1u]`), "GQA2 stage1 must load each packed V pair once")
	batchStage2 := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_attention_heads_batch_chunked_stage2`)
	core.AssertTrue(t, strings.Contains(batchStage2, `const bool cached_chunk_weights = chunk_count <= threads`), "batch stage2 must cache per-chunk softmax weights when they fit in shared scratch")
	heads := hipKernelSourceFunctionBodyForTest(t, source, `__device__ void rocm_run_single_head_attention_token_parallel`)
	core.AssertTrue(t, strings.Contains(heads, `device_kv_header->block_size == 1u`), "shared attention direct token-page fast path must reject mixed block/page MP4 KV streams")
	core.AssertTrue(t, strings.Contains(heads, `first_device_kv_page->key_width == args.dim`), "shared attention metadata caching must reject multi-KV descriptor rows")
	core.AssertTrue(t, strings.Contains(heads, `single_head_device_kv_width &&`), "shared attention direct Q4 fast paths must require a one-head-wide descriptor row")
	core.AssertTrue(t, strings.Contains(heads, `cached_pointer = reinterpret_cast<uint64_t>(rocm_device_kv_row_payload_pointer(bytes, page->value_encoding, page->token_count, page->value_width, local_token)) + (value_base >> 1u)`), "shared attention must cache MP4 block q4 value row payload pointers")
	core.AssertTrue(t, strings.Contains(heads, `const unsigned char *values = reinterpret_cast<const unsigned char *>(static_cast<uintptr_t>(cached_pointer));`), "shared attention cached value pointers must already point at the q4 row payload")
}

func TestHIPKernelSource_RMSNormRoPEHeadsPairLaneBatchUsesDevicePositions_Good(t *testing.T) {
	sourceBytes, err := os.ReadFile(hipKernelSourcePathForTest)
	core.RequireNoError(t, err)
	source := string(sourceBytes)
	body := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_rms_norm_rope_heads_pair_lane_batch`)

	core.AssertTrue(t, strings.Contains(body, `const uint32_t position = positions[batch]`), "paired lane batch must load each position from the device buffer")
	core.AssertTrue(t, strings.Contains(body, `if (head < args.query_head_count)`), "paired lane batch must support query heads independently")
	core.AssertTrue(t, strings.Contains(body, `if (head < args.key_head_count)`), "paired lane batch must support key heads independently")
	core.AssertTrue(t, !strings.Contains(body, `start_position + batch`), "paired lane batch must not derive positions from a consecutive range")
	core.AssertTrue(t, strings.Contains(source, `static_assert(sizeof(rocm_rms_norm_rope_heads_pair_lane_batch_launch_args) == ROCM_RMS_NORM_ROPE_HEADS_PAIR_LANE_BATCH_LAUNCH_ARGS_BYTES`), "paired lane batch launch ABI must be statically checked")
}

func TestHIPKernelSource_HIPCPUFusedAttentionRouting_Good(t *testing.T) {
	sourceBytes, err := os.ReadFile(hipKernelSourcePathForTest)
	core.RequireNoError(t, err)
	source := string(sourceBytes)

	guardMarker := "#if defined(__HIP_CPU_RT__)\n// HIP-CPU does not provide the dynamically-sized extern shared-memory symbol"
	guardStart := strings.Index(source, guardMarker)
	if guardStart < 0 {
		t.Fatalf("HIP-CPU fused-attention guard not found")
	}
	elseOffset := strings.Index(source[guardStart:], "#else")
	if elseOffset < 0 {
		t.Fatalf("HIP-CPU fused-attention guard has no non-CPU branch")
	}
	elseStart := guardStart + elseOffset
	endOffset := strings.Index(source[elseStart:], "#endif")
	if endOffset < 0 {
		t.Fatalf("HIP-CPU fused-attention guard has no terminator")
	}
	end := elseStart + endOffset
	cpuBranch := source[guardStart:elseStart]
	nonCPUBranch := source[elseStart:end]

	core.AssertTrue(t, strings.Contains(cpuBranch, `extern "C" __global__ void rocm_attention_heads_batch_causal_query_rms_rope(const unsigned char *)`), "HIP-CPU must retain the fused-attention symbol for ABI lookup")
	core.AssertTrue(t, strings.Contains(cpuBranch, `__builtin_trap();`), "HIP-CPU fused-attention entry must decline execution")
	core.AssertTrue(t, !strings.Contains(cpuBranch, `extern __shared__`), "HIP-CPU fused-attention branch must not reference dynamic shared memory")
	core.AssertTrue(t, strings.Contains(nonCPUBranch, `extern "C" __global__ void rocm_attention_heads_batch_causal_query_rms_rope(const unsigned char *packet)`), "AMD/NVIDIA must retain the fused-attention implementation")
	core.AssertTrue(t, strings.Contains(nonCPUBranch, `extern __shared__ unsigned char fused_attention_shared[]`), "AMD/NVIDIA fused attention must own its dynamic shared-memory path")
	core.AssertTrue(t, strings.Contains(nonCPUBranch, `rocm_attention_align_shared_offset(args.shared_mem_bytes, sizeof(float))`), "fused attention must align the dynamic shared query region")
}

func TestHIPKernelSource_AttentionLaneBatchUsesIndependentDescriptors_Good(t *testing.T) {
	sourceBytes, err := os.ReadFile(hipKernelSourcePathForTest)
	core.RequireNoError(t, err)
	source := string(sourceBytes)
	body := hipKernelSourceFunctionBodyForTest(t, source, `extern "C" __global__ void rocm_attention_heads_lane_batch`)

	for _, marker := range []string{
		`const uint64_t lane_index = blockIdx.y`,
		`const rocm_attention_heads_lane_descriptor lane = lanes[lane_index]`,
		`single.token_count = lane.token_count`,
		`single.descriptor_pointer = lane.descriptor_pointer`,
		`single.descriptor_bytes = lane.descriptor_bytes`,
		`const uint64_t batch_head_index = lane_index * args.head_count + head`,
	} {
		core.AssertTrue(t, strings.Contains(body, marker), "lane attention must route each row through its own KV descriptor")
	}
	core.AssertTrue(t, strings.Contains(source, `static_assert(sizeof(rocm_attention_heads_lane_descriptor) == ROCM_ATTENTION_HEADS_LANE_DESCRIPTOR_BYTES`), "lane descriptor ABI must be statically checked")
	core.AssertTrue(t, strings.Contains(source, `static_assert(sizeof(rocm_attention_heads_lane_batch_launch_args) == ROCM_ATTENTION_HEADS_LANE_BATCH_LAUNCH_ARGS_BYTES`), "lane launch ABI must be statically checked")
}

func TestHIPKernelSource_AttentionReductionBlockSizeGuards_Good(t *testing.T) {
	sourceBytes, err := os.ReadFile(hipKernelSourcePathForTest)
	core.RequireNoError(t, err)
	source := string(sourceBytes)

	for _, marker := range []string{
		`__device__ void rocm_run_single_head_attention_range_token_parallel`,
		`__device__ void rocm_run_single_head_attention_token_parallel`,
		`__device__ void rocm_run_single_head_attention_parallel`,
	} {
		body := hipKernelSourceFunctionBodyForTest(t, source, marker)
		core.AssertTrue(t, strings.Contains(body, `if (threads > 512 || (threads & (threads - 1u)) != 0u)`), "attention reduction path must reject non-power-of-two block sizes")
	}
	for _, marker := range []string{
		`__device__ void rocm_run_single_head_attention_token_parallel`,
		`__device__ void rocm_run_single_head_attention_parallel`,
	} {
		body := hipKernelSourceFunctionBodyForTest(t, source, marker)
		core.AssertTrue(t, strings.Contains(body, `rocm_run_single_head_attention(args)`), "attention reduction guard must fall back to scalar attention")
	}

	for _, marker := range []string{
		`extern "C" __global__ void rocm_attention_heads_chunked_stage1`,
		`extern "C" __global__ void rocm_attention_heads_chunked_stage2`,
		`extern "C" __global__ void rocm_attention_heads_batch_chunked_stage1_v2`,
		`extern "C" __global__ void rocm_attention_heads_batch_chunked_stage1_gqa2`,
		`extern "C" __global__ void rocm_attention_heads_batch_chunked_stage2`,
	} {
		body := hipKernelSourceFunctionBodyForTest(t, source, marker)
		core.AssertTrue(t, strings.Contains(body, `if (threads != ROCM_ATTENTION_HEADS_CHUNKED_BLOCK_SIZE)`), "chunked attention reduction path must reject non-contract block sizes")
		core.AssertTrue(t, strings.Contains(body, "return;"), "chunked attention reduction guard must decline the invalid launch")
	}
}

func TestHIPKernelSource_NVIDIAHIPCompile_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_NVIDIA_HIP_COMPILE_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_NVIDIA_HIP_COMPILE_TESTS=1 to compile HIP source through the NVIDIA backend")
	}

	hipcc := rocmNVIDIATestLookPath(t, "hipcc")
	cudaPath := rocmNVIDIATestCUDAPath(t)
	arch := rocmNVIDIATestEnvDefault("GO_ROCM_NVIDIA_HIP_ARCH", "sm_75")
	std := rocmNVIDIATestEnvDefault("GO_ROCM_NVIDIA_HIP_STD", "c++20")
	outputPath := filepath.Join(t.TempDir(), "rocm_kernels_nvidia.o")
	cmd := rocmCompileTestCommand(t,
		hipcc,
		"--std="+std,
		"-c",
		"-x",
		"cu",
		"-I/opt/rocm/include",
		"-arch="+arch,
		hipKernelSourcePathForTest,
		"-o",
		outputPath,
	)
	cmd.Env = rocmCompileTestEnv(rocmNVIDIATestEnv(cudaPath, "HIP_PLATFORM=nvidia"))
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("compile HIP kernels through NVIDIA backend: %v\n%s", err, rocmNVIDIATestOutputTail(output))
	}
	info, err := os.Stat(outputPath)
	if err != nil {
		t.Fatalf("stat NVIDIA HIP object: %v", err)
	}
	if info.Size() == 0 {
		t.Fatalf("NVIDIA HIP object is empty: %s", outputPath)
	}
	t.Logf("compiled HIP kernels for NVIDIA backend std=%s arch=%s object_bytes=%d", std, arch, info.Size())
}

func TestHIPKernelSource_AMDHIPCompile_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_AMD_HIP_COMPILE_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_AMD_HIP_COMPILE_TESTS=1 to compile HIP source through the AMD backend")
	}

	hipcc := rocmNVIDIATestLookPath(t, "hipcc")
	arch := rocmNVIDIATestEnvDefault("GO_ROCM_AMD_HIP_ARCH", "gfx1100")
	std := rocmNVIDIATestEnvDefault("GO_ROCM_AMD_HIP_STD", "c++23")
	opt := rocmNVIDIATestEnvDefault("GO_ROCM_AMD_HIP_OPT", "-O3")
	outputPath := filepath.Join(t.TempDir(), "rocm_kernels_"+arch+".hsaco")
	cmd := rocmCompileTestCommand(t,
		hipcc,
		"--std="+std,
		"--genco",
		"--offload-arch="+arch,
		opt,
		hipKernelSourcePathForTest,
		"-o",
		outputPath,
	)
	cmd.Env = rocmCompileTestEnv(rocmNVIDIATestEnv("", "HIP_PLATFORM=amd"))
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("compile HIP kernels through AMD backend: %v\n%s", err, rocmNVIDIATestOutputTail(output))
	}
	info, err := os.Stat(outputPath)
	if err != nil {
		t.Fatalf("stat AMD HIP code object: %v", err)
	}
	if info.Size() == 0 {
		t.Fatalf("AMD HIP code object is empty: %s", outputPath)
	}
	t.Logf("compiled HIP kernels for AMD backend std=%s arch=%s opt=%s hsaco_bytes=%d", std, arch, opt, info.Size())
}

func TestHIPKernelSource_HIPCPUCompile_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_HIP_CPU_COMPILE_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_HIP_CPU_COMPILE_TESTS=1 to compile HIP source through HIP-CPU")
	}

	includeDir := rocmHIPCPUTestIncludeDir(t)
	for _, target := range rocmHIPCPUTestTargets() {
		target := target
		t.Run(target.name, func(t *testing.T) {
			compiler := rocmHIPCPUTestCompiler(t, target)
			outputPath := filepath.Join(t.TempDir(), "rocm_kernels_hip_cpu_"+target.name+".o")
			args := []string{
				"-std=c++20",
				"-O2",
				"-x",
				"c++",
				"-I" + includeDir,
			}
			args = append(args, target.extraCompileFlags...)
			args = append(args, "-c", hipKernelSourcePathForTest, "-o", outputPath)
			cmd := rocmCompileTestCommand(t, compiler, args...)
			output, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("compile HIP kernels through HIP-CPU target=%s compiler=%s: %v\n%s", target.name, compiler, err, rocmNVIDIATestOutputTail(output))
			}
			info, err := os.Stat(outputPath)
			if err != nil {
				t.Fatalf("stat HIP-CPU object: %v", err)
			}
			if info.Size() == 0 {
				t.Fatalf("HIP-CPU object is empty: %s", outputPath)
			}
			t.Logf("compiled HIP kernels for HIP-CPU target=%s compiler=%s object_bytes=%d include=%s", target.name, compiler, info.Size(), includeDir)
		})
	}
}

func TestHIPKernelSource_HIPCPURuntimeSmoke_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_HIP_CPU_RUNTIME_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_HIP_CPU_RUNTIME_TESTS=1 to compile and run a HIP-CPU runtime smoke")
	}

	includeDir := rocmHIPCPUTestIncludeDir(t)
	compiler := rocmHIPCPUTestCompiler(t, rocmHIPCPUTestTarget{name: "x86_64", compilerEnv: "GO_ROCM_HIP_CPU_CXX", compilerFallback: "g++"})
	tempDir := t.TempDir()
	sourcePath := filepath.Join(tempDir, "hip_cpu_smoke.cpp")
	binaryPath := filepath.Join(tempDir, "hip_cpu_smoke")
	core.RequireNoError(t, os.WriteFile(sourcePath, []byte(rocmHIPCPUSmokeSource), 0o644))

	compile := rocmCompileTestCommand(t,
		compiler,
		"-std=c++20",
		"-O2",
		"-I"+includeDir,
		sourcePath,
		"-ltbb",
		"-o",
		binaryPath,
	)
	output, err := compile.CombinedOutput()
	if err != nil {
		t.Fatalf("compile HIP-CPU smoke compiler=%s: %v\n%s", compiler, err, rocmNVIDIATestOutputTail(output))
	}

	run := exec.Command(binaryPath)
	output, err = run.CombinedOutput()
	if err != nil {
		t.Fatalf("run HIP-CPU smoke: %v\n%s", err, rocmNVIDIATestOutputTail(output))
	}
	if !strings.Contains(string(output), "hip_cpu_smoke_ok") {
		t.Fatalf("HIP-CPU smoke did not report success:\n%s", rocmNVIDIATestOutputTail(output))
	}
	t.Log(strings.TrimSpace(string(output)))
}

func TestHIPKernelSource_HIPCPUProductionKernelRuntimeSmoke_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_HIP_CPU_KERNEL_RUNTIME_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_HIP_CPU_KERNEL_RUNTIME_TESTS=1 to compile and run rocm_kernels.hip through HIP-CPU")
	}

	includeDir := rocmHIPCPUTestIncludeDir(t)
	compiler := rocmHIPCPUTestCompiler(t, rocmHIPCPUTestTarget{name: "x86_64", compilerEnv: "GO_ROCM_HIP_CPU_CXX", compilerFallback: "g++"})
	kernelPath, err := filepath.Abs(hipKernelSourcePathForTest)
	core.RequireNoError(t, err)
	tempDir := t.TempDir()
	sourcePath := filepath.Join(tempDir, "hip_cpu_rocm_kernel_smoke.cpp")
	binaryPath := filepath.Join(tempDir, "hip_cpu_rocm_kernel_smoke")
	source := rocmHIPCPUProductionKernelSmokeSource(kernelPath)
	core.RequireNoError(t, os.WriteFile(sourcePath, []byte(source), 0o644))

	compile := rocmCompileTestCommand(t,
		compiler,
		"-std=c++20",
		"-O0",
		"-I"+includeDir,
		sourcePath,
		"-ltbb",
		"-o",
		binaryPath,
	)
	output, err := compile.CombinedOutput()
	if err != nil {
		t.Fatalf("compile HIP-CPU production kernel smoke compiler=%s: %v\n%s", compiler, err, rocmNVIDIATestOutputTail(output))
	}

	run := exec.Command(binaryPath)
	output, err = run.CombinedOutput()
	if err != nil {
		t.Fatalf("run HIP-CPU production kernel smoke: %v\n%s", err, rocmNVIDIATestOutputTail(output))
	}
	if !strings.Contains(string(output), "hip_cpu_rocm_kernel_smoke_ok") {
		t.Fatalf("HIP-CPU production kernel smoke did not report success:\n%s", rocmNVIDIATestOutputTail(output))
	}
	t.Log(strings.TrimSpace(string(output)))
}

func TestHIPKernelSource_ZLUDACUDARuntimeSmoke_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_ZLUDA_CUDA_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_ZLUDA_CUDA_TESTS=1 to compile CUDA with nvcc and run it through ZLUDA")
	}

	cudaPath := rocmNVIDIATestCUDAPath(t)
	nvcc := filepath.Join(cudaPath, "bin", "nvcc")
	if _, err := os.Stat(nvcc); err != nil {
		nvcc = rocmNVIDIATestLookPath(t, "nvcc")
	}
	zludaDir := rocmZLUDATestDir(t)
	arch := rocmNVIDIATestEnvDefault("GO_ROCM_NVIDIA_CUDA_ARCH", "sm_75")
	tempDir := t.TempDir()
	sourcePath := filepath.Join(tempDir, "zluda_cuda_smoke.cu")
	binaryPath := filepath.Join(tempDir, "zluda_cuda_smoke")
	core.RequireNoError(t, os.WriteFile(sourcePath, []byte(rocmZLUDACUDASmokeSource), 0o644))

	compile := rocmCompileTestCommand(t,
		nvcc,
		"-std=c++17",
		"-arch="+arch,
		"-Wno-deprecated-gpu-targets",
		sourcePath,
		"-o",
		binaryPath,
	)
	compile.Env = rocmCompileTestEnv(rocmNVIDIATestEnv(cudaPath))
	output, err := compile.CombinedOutput()
	if err != nil {
		t.Fatalf("compile CUDA smoke with nvcc: %v\n%s", err, rocmNVIDIATestOutputTail(output))
	}

	run := exec.Command(binaryPath)
	run.Env = rocmZLUDATestEnv(t, cudaPath, zludaDir)
	output, err = run.CombinedOutput()
	if err != nil {
		t.Fatalf("run CUDA smoke through ZLUDA: %v\n%s", err, rocmNVIDIATestOutputTail(output))
	}
	if !strings.Contains(string(output), "zluda_cuda_smoke_ok") {
		t.Fatalf("ZLUDA smoke did not report success:\n%s", rocmNVIDIATestOutputTail(output))
	}
	t.Log(strings.TrimSpace(string(output)))
}

func hipKernelSourceFunctionBodyForTest(t *testing.T, source, marker string) string {
	t.Helper()
	start := strings.Index(source, marker)
	if start < 0 {
		t.Fatalf("kernel marker %q not found", marker)
	}
	open := strings.Index(source[start:], "{")
	if open < 0 {
		t.Fatalf("kernel marker %q has no body", marker)
	}
	index := start + open
	depth := 0
	for ; index < len(source); index++ {
		switch source[index] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return source[start : index+1]
			}
		}
	}
	t.Fatalf("kernel marker %q body did not close", marker)
	return ""
}

const rocmZLUDACUDASmokeSource = `
#include <cuda_runtime.h>
#include <cstdio>

__global__ void rocm_zluda_smoke_kernel(int *out) {
	const int index = threadIdx.x;
	out[index] = index + 7;
}

int main() {
	int count = 0;
	cudaError_t err = cudaGetDeviceCount(&count);
	if (err != cudaSuccess || count < 1) {
		std::printf("device_count_error=%s count=%d\n", cudaGetErrorString(err), count);
		return 10;
	}

	int *device = nullptr;
	int host[4] = {0, 0, 0, 0};
	err = cudaMalloc(reinterpret_cast<void **>(&device), sizeof(host));
	if (err != cudaSuccess) {
		std::printf("malloc_error=%s\n", cudaGetErrorString(err));
		return 11;
	}

	rocm_zluda_smoke_kernel<<<1, 4>>>(device);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::printf("launch_error=%s\n", cudaGetErrorString(err));
		cudaFree(device);
		return 12;
	}

	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		std::printf("sync_error=%s\n", cudaGetErrorString(err));
		cudaFree(device);
		return 13;
	}

	err = cudaMemcpy(host, device, sizeof(host), cudaMemcpyDeviceToHost);
	cudaFree(device);
	if (err != cudaSuccess) {
		std::printf("copy_error=%s\n", cudaGetErrorString(err));
		return 14;
	}
	for (int i = 0; i < 4; ++i) {
		if (host[i] != i + 7) {
			std::printf("value_error index=%d got=%d\n", i, host[i]);
			return 15;
		}
	}
	std::printf("zluda_cuda_smoke_ok count=%d values=%d,%d,%d,%d\n", count, host[0], host[1], host[2], host[3]);
	return 0;
}
`

const rocmHIPCPUSmokeSource = `
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void rocm_hip_cpu_smoke_kernel(float *out, const float *in, int count) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < count) {
		out[index] = in[index] * 2.0f + 1.0f;
	}
}

int main() {
	hipDeviceProp_t props{};
	hipError_t err = hipGetDeviceProperties(&props, 0);
	if (err != hipSuccess) {
		std::printf("props_error=%s\n", hipGetErrorString(err));
		return 10;
	}

	const int count = 8;
	float host_in[count] = {0, 1, 2, 3, 4, 5, 6, 7};
	float host_out[count] = {};
	float *device_in = nullptr;
	float *device_out = nullptr;
	err = hipMalloc(reinterpret_cast<void **>(&device_in), sizeof(host_in));
	if (err != hipSuccess) {
		std::printf("malloc_in_error=%s\n", hipGetErrorString(err));
		return 11;
	}
	err = hipMalloc(reinterpret_cast<void **>(&device_out), sizeof(host_out));
	if (err != hipSuccess) {
		std::printf("malloc_out_error=%s\n", hipGetErrorString(err));
		hipFree(device_in);
		return 12;
	}
	err = hipMemcpy(device_in, host_in, sizeof(host_in), hipMemcpyHostToDevice);
	if (err != hipSuccess) {
		std::printf("copy_in_error=%s\n", hipGetErrorString(err));
		hipFree(device_in);
		hipFree(device_out);
		return 13;
	}
	hipLaunchKernelGGL(rocm_hip_cpu_smoke_kernel, dim3(1), dim3(count), 0, nullptr, device_out, device_in, count);
	err = hipDeviceSynchronize();
	if (err != hipSuccess) {
		std::printf("sync_error=%s\n", hipGetErrorString(err));
		hipFree(device_in);
		hipFree(device_out);
		return 14;
	}
	err = hipMemcpy(host_out, device_out, sizeof(host_out), hipMemcpyDeviceToHost);
	hipFree(device_in);
	hipFree(device_out);
	if (err != hipSuccess) {
		std::printf("copy_out_error=%s\n", hipGetErrorString(err));
		return 15;
	}
	for (int i = 0; i < count; ++i) {
		const float want = host_in[i] * 2.0f + 1.0f;
		if (host_out[i] != want) {
			std::printf("value_error index=%d got=%.1f want=%.1f\n", i, host_out[i], want);
			return 16;
		}
	}
	std::printf("hip_cpu_smoke_ok device=%s values=%.1f,%.1f,%.1f,%.1f\n", props.name, host_out[0], host_out[1], host_out[2], host_out[3]);
	return 0;
}
`

const rocmHIPCPUProductionKernelSmokeSourceTemplate = `
#include <cmath>
#include <cstdio>

#include ${kernel_path}

#if defined(__HIP_CPU_RT__)
thread_local float shared_attention_weights[1];
thread_local unsigned char shared_bytes[1];
#endif

int main() {
	hipDeviceProp_t props{};
	hipError_t err = hipGetDeviceProperties(&props, 0);
	if (err != hipSuccess) {
		std::printf("props_error=%s\n", hipGetErrorString(err));
		return 10;
	}

	const uint32_t token_count = 3;
	const uint32_t dim = 4;
	float host_tokens[token_count * dim] = {
		1.0f, 2.0f, 3.0f, 4.0f,
		5.0f, 6.0f, 7.0f, 8.0f,
		9.0f, 10.0f, 11.0f, 12.0f,
	};
	float host_output[dim] = {};
	float *tokens = nullptr;
	float *output = nullptr;
	err = hipMalloc(reinterpret_cast<void **>(&tokens), sizeof(host_tokens));
	if (err != hipSuccess) {
		std::printf("malloc_tokens_error=%s\n", hipGetErrorString(err));
		return 11;
	}
	err = hipMalloc(reinterpret_cast<void **>(&output), sizeof(host_output));
	if (err != hipSuccess) {
		std::printf("malloc_output_error=%s\n", hipGetErrorString(err));
		hipFree(tokens);
		return 12;
	}
	err = hipMemcpy(tokens, host_tokens, sizeof(host_tokens), hipMemcpyHostToDevice);
	if (err != hipSuccess) {
		std::printf("copy_tokens_error=%s\n", hipGetErrorString(err));
		hipFree(tokens);
		hipFree(output);
		return 13;
	}

	rocm_embedding_mean_pool_launch_args args{};
	args.version = ROCM_EMBEDDING_MEAN_POOL_LAUNCH_ARGS_VERSION;
	args.total_bytes = ROCM_EMBEDDING_MEAN_POOL_LAUNCH_ARGS_BYTES;
	args.token_pointer = reinterpret_cast<uint64_t>(tokens);
	args.output_pointer = reinterpret_cast<uint64_t>(output);
	args.token_count = token_count;
	args.dim = dim;
	args.token_bytes = sizeof(host_tokens);
	args.output_bytes = sizeof(host_output);
	args.flags = 0;
	hipLaunchKernelGGL(rocm_embedding_mean_pool, dim3(1), dim3(1), 0, nullptr, reinterpret_cast<const unsigned char *>(&args));
	err = hipGetLastError();
	if (err != hipSuccess) {
		std::printf("launch_error=%s\n", hipGetErrorString(err));
		hipFree(tokens);
		hipFree(output);
		return 14;
	}
	err = hipDeviceSynchronize();
	if (err != hipSuccess) {
		std::printf("sync_error=%s\n", hipGetErrorString(err));
		hipFree(tokens);
		hipFree(output);
		return 15;
	}
	err = hipMemcpy(host_output, output, sizeof(host_output), hipMemcpyDeviceToHost);
	hipFree(tokens);
	hipFree(output);
	if (err != hipSuccess) {
		std::printf("copy_output_error=%s\n", hipGetErrorString(err));
		return 16;
	}

	const float want[dim] = {5.0f, 6.0f, 7.0f, 8.0f};
	for (uint32_t i = 0; i < dim; ++i) {
		if (std::fabs(host_output[i] - want[i]) > 0.00001f) {
			std::printf("value_error index=%u got=%.6f want=%.6f\n", i, host_output[i], want[i]);
			return 17;
		}
	}
	std::printf("hip_cpu_rocm_kernel_smoke_ok device=%s values=%.1f,%.1f,%.1f,%.1f\n", props.name, host_output[0], host_output[1], host_output[2], host_output[3]);
	return 0;
}
`

func rocmHIPCPUProductionKernelSmokeSource(kernelPath string) string {
	return strings.ReplaceAll(rocmHIPCPUProductionKernelSmokeSourceTemplate, "${kernel_path}", strconv.Quote(kernelPath))
}

func rocmNVIDIATestCUDAPath(t *testing.T) string {
	t.Helper()
	if cudaPath := os.Getenv("CUDA_PATH"); cudaPath != "" {
		return cudaPath
	}
	if cudaPath := os.Getenv("CUDA_HOME"); cudaPath != "" {
		return cudaPath
	}
	for _, candidate := range []string{"/usr/local/cuda", "/usr"} {
		if _, err := os.Stat(filepath.Join(candidate, "bin", "nvcc")); err == nil {
			return candidate
		}
	}
	t.Fatalf("CUDA toolkit with nvcc not found; install cuda-nvcc-12-8 or set CUDA_PATH")
	return ""
}

func rocmNVIDIATestLookPath(t *testing.T, name string) string {
	t.Helper()
	path, err := exec.LookPath(name)
	if err != nil {
		t.Fatalf("%s not found in PATH: %v", name, err)
	}
	return path
}

func rocmNVIDIATestEnv(cudaPath string, extra ...string) []string {
	env := append([]string{}, os.Environ()...)
	if cudaPath != "" {
		env = append(env, "CUDA_PATH="+cudaPath, "CUDA_HOME="+cudaPath)
	}
	env = append(env, extra...)
	return env
}

func rocmNVIDIATestEnvDefault(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}

func rocmCompileTestCommand(t *testing.T, compiler string, args ...string) *exec.Cmd {
	t.Helper()
	ccache, ok := rocmCompileTestCCache()
	if !ok {
		return exec.Command(compiler, args...)
	}
	if filepath.Base(compiler) == "hipcc" {
		cmd := exec.Command(compiler, args...)
		cmd.Env = rocmCompileTestEnv(os.Environ())
		t.Logf("using ccache PATH launcher for %s", compiler)
		return cmd
	}
	commandArgs := make([]string, 0, len(args)+1)
	commandArgs = append(commandArgs, compiler)
	commandArgs = append(commandArgs, args...)
	cmd := exec.Command(ccache, commandArgs...)
	cmd.Env = rocmCompileTestEnv(os.Environ())
	t.Logf("using ccache launcher for %s", compiler)
	return cmd
}

func rocmCompileTestCCache() (string, bool) {
	if os.Getenv("GO_ROCM_USE_CCACHE") == "0" {
		return "", false
	}
	ccache := os.Getenv("GO_ROCM_CCACHE")
	if ccache == "" {
		path, err := exec.LookPath("ccache")
		if err != nil {
			return "", false
		}
		ccache = path
	}
	return ccache, true
}

func rocmCompileTestEnv(env []string) []string {
	if _, ok := rocmCompileTestCCache(); !ok {
		return env
	}
	ccacheDir := "/usr/lib/ccache"
	if info, err := os.Stat(ccacheDir); err != nil || !info.IsDir() {
		return env
	}
	prefixed := make([]string, 0, len(env)+1)
	replaced := false
	for _, item := range env {
		if strings.HasPrefix(item, "PATH=") {
			prefixed = append(prefixed, "PATH="+ccacheDir+string(os.PathListSeparator)+strings.TrimPrefix(item, "PATH="))
			replaced = true
			continue
		}
		prefixed = append(prefixed, item)
	}
	if !replaced {
		prefixed = append(prefixed, "PATH="+ccacheDir)
	}
	return prefixed
}

func rocmNVIDIATestOutputTail(output []byte) string {
	const limit = 8192
	if len(output) <= limit {
		return string(output)
	}
	return string(output[len(output)-limit:])
}

func rocmZLUDATestDir(t *testing.T) string {
	t.Helper()
	candidates := []string{}
	if dir := os.Getenv("GO_ROCM_ZLUDA_DIR"); dir != "" {
		candidates = append(candidates, dir)
	}
	candidates = append(candidates, "/opt/zluda/v5/zluda", "/tmp/zluda-v5/zluda")
	for _, candidate := range candidates {
		if _, err := os.Stat(filepath.Join(candidate, "libcuda.so")); err == nil {
			return candidate
		}
	}
	t.Fatalf("ZLUDA directory not found; set GO_ROCM_ZLUDA_DIR to a v5 unpack containing libcuda.so")
	return ""
}

func rocmZLUDATestEnv(t *testing.T, cudaPath, zludaDir string) []string {
	t.Helper()
	paths := []string{zludaDir, filepath.Join(cudaPath, "lib64")}
	if _, err := os.Stat("/opt/rocm-6.4.4/lib/libamdhip64.so.6"); err == nil {
		paths = append(paths, "/opt/rocm-6.4.4/lib")
	}
	compatDir := rocmZLUDAHIPCompatDir(t)
	if compatDir != "" {
		paths = append(paths, compatDir)
	}
	paths = append(paths, "/opt/rocm/lib")
	if current := os.Getenv("LD_LIBRARY_PATH"); current != "" {
		paths = append(paths, current)
	}
	env := append([]string{}, os.Environ()...)
	env = append(env, "LD_LIBRARY_PATH="+strings.Join(paths, ":"))
	return env
}

func rocmZLUDAHIPCompatDir(t *testing.T) string {
	t.Helper()
	for _, candidate := range []string{
		"/opt/rocm-6.4.4/lib/libamdhip64.so.6",
		"/opt/rocm/lib/libamdhip64.so.6",
		"/usr/lib/x86_64-linux-gnu/libamdhip64.so.6",
	} {
		if _, err := os.Stat(candidate); err == nil {
			return ""
		}
	}
	target := ""
	for _, candidate := range []string{
		"/opt/rocm/lib/libamdhip64.so.7",
		"/opt/rocm-7.2.0/lib/libamdhip64.so.7",
	} {
		if _, err := os.Stat(candidate); err == nil {
			target = candidate
			break
		}
	}
	if target == "" {
		return ""
	}
	compatDir := filepath.Join(t.TempDir(), "zluda-hip-compat")
	core.RequireNoError(t, os.MkdirAll(compatDir, 0o755))
	core.RequireNoError(t, os.Symlink(target, filepath.Join(compatDir, "libamdhip64.so.6")))
	t.Logf("using local ZLUDA HIP ABI symlink libamdhip64.so.6 -> %s", target)
	return compatDir
}

type rocmHIPCPUTestTarget struct {
	name              string
	compilerEnv       string
	compilerFallback  string
	extraCompileFlags []string
}

func rocmHIPCPUTestTargets() []rocmHIPCPUTestTarget {
	targets := []rocmHIPCPUTestTarget{}
	names := "x86_64,aarch64"
	if configured := os.Getenv("GO_ROCM_HIP_CPU_TARGETS"); configured != "" {
		names = configured
	}
	for _, raw := range strings.Split(names, ",") {
		name := strings.TrimSpace(raw)
		switch name {
		case "":
			continue
		case "x86_64", "amd64":
			targets = append(targets, rocmHIPCPUTestTarget{
				name:             "x86_64",
				compilerEnv:      "GO_ROCM_HIP_CPU_CXX",
				compilerFallback: "g++",
			})
		case "aarch64", "arm64":
			targets = append(targets, rocmHIPCPUTestTarget{
				name:             "aarch64",
				compilerEnv:      "GO_ROCM_HIP_CPU_AARCH64_CXX",
				compilerFallback: "aarch64-linux-gnu-g++",
				extraCompileFlags: []string{
					"-DVALGRIND_STACK_REGISTER(a,b)=((void)0)",
				},
			})
		default:
			targets = append(targets, rocmHIPCPUTestTarget{
				name:             name,
				compilerEnv:      "GO_ROCM_HIP_CPU_CXX",
				compilerFallback: name + "-g++",
			})
		}
	}
	return targets
}

func rocmHIPCPUTestCompiler(t *testing.T, target rocmHIPCPUTestTarget) string {
	t.Helper()
	if configured := os.Getenv(target.compilerEnv); configured != "" {
		return configured
	}
	path, err := exec.LookPath(target.compilerFallback)
	if err != nil {
		t.Skipf("HIP-CPU compiler %s for target %s not found; set %s", target.compilerFallback, target.name, target.compilerEnv)
	}
	return path
}

func rocmHIPCPUTestIncludeDir(t *testing.T) string {
	t.Helper()
	candidates := []string{}
	if include := os.Getenv("GO_ROCM_HIP_CPU_INCLUDE"); include != "" {
		candidates = append(candidates, include)
	}
	if root := os.Getenv("GO_ROCM_HIP_CPU_ROOT"); root != "" {
		candidates = append(candidates, filepath.Join(root, "include"))
	}
	candidates = append(candidates, "/opt/hip-cpu/include", "/usr/local/include")
	for _, candidate := range candidates {
		header := filepath.Join(candidate, "hip", "hip_defines.h")
		bytes, err := os.ReadFile(header)
		if err == nil && strings.Contains(string(bytes), "__HIP_CPU_RT__") {
			return candidate
		}
	}
	t.Fatalf("HIP-CPU include directory not found; clone https://github.com/ROCm/HIP-CPU to /opt/hip-cpu or set GO_ROCM_HIP_CPU_INCLUDE")
	return ""
}
