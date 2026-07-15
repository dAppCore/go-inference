// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"sync"
	"time"

	core "dappco.re/go"
)

const (
	hipKernelNamePrefill                                    = "rocm_prefill"
	hipKernelNameDecode                                     = "rocm_decode"
	hipKernelNameKVEncodeToken                              = "rocm_kv_encode_token"
	hipKernelNameKVEncodeTokenValueNorm                     = "rocm_kv_encode_token_value_norm"
	hipKernelNameKVEncodeTokenValueNormDescriptorAppend     = "rocm_kv_encode_token_value_norm_descriptor_append"
	hipKernelNameKVDescriptorAppend                         = "rocm_kv_descriptor_append"
	hipKernelNameProjection                                 = "rocm_projection"
	hipKernelNameProjectionBatch                            = "rocm_projection_batch"
	hipKernelNameMLXQ4Proj                                  = "rocm_mlx_q4_projection"
	hipKernelNameMLXQ4ProjQ4G32Rows3840Cols15360            = "rocm_mlx_q4_projection_q4_g32_rows3840_cols15360"
	hipKernelNameMLXQ4ProjQ4G64Rows3840Cols15360Row16       = "rocm_mlx_q4_projection_q4_g64_rows3840_cols15360_row16"
	hipKernelNameMLXQ4ProjCols256                           = "rocm_mlx_q4_projection_cols256"
	hipKernelNameMLXQ4ProjQ6G16Row16                        = "rocm_mlx_q4_projection_q6_g16_row16"
	hipKernelNameMLXQ4ProjQ6Row16                           = "rocm_mlx_q4_projection_q6_row16"
	hipKernelNameMLXQ4ProjQ6Row32                           = "rocm_mlx_q4_projection_q6_row32"
	hipKernelNameMLXQ4ProjQ6Row64                           = "rocm_mlx_q4_projection_q6_row64"
	hipKernelNameMLXQ4ProjBatch                             = "rocm_mlx_q4_projection_batch"
	hipKernelNameMLXQ4ProjBatchQ4G64Tokens16                = "rocm_mlx_q4_projection_batch_q4_g64_tokens16"
	hipKernelNameMLXQ4ProjBatchQ8G64Row16Tokens16           = "rocm_mlx_q4_projection_batch_q8_g64_row16_tokens16"
	hipKernelNameMLXQ4ProjBatchQ6Row16                      = "rocm_mlx_q4_projection_batch_q6_row16"
	hipKernelNameMLXQ4ProjGreedy                            = "rocm_mlx_q4_projection_greedy"
	hipKernelNameMLXQ4ProjGreedyQ6Row64                     = "rocm_mlx_q4_projection_greedy_q6_row64"
	hipKernelNameMLXQ4ProjGreedyBatch                       = "rocm_mlx_q4_projection_greedy_batch"
	hipKernelNameMLXQ4ProjGreedyBatchQ6Row64                = "rocm_mlx_q4_projection_greedy_batch_q6_row64"
	hipKernelNameMLXQ4ProjScores                            = "rocm_mlx_q4_projection_scores"
	hipKernelNameMLXQ4ProjScoresQ6Row64                     = "rocm_mlx_q4_projection_scores_q6_row64"
	hipKernelNameMLXQ4ProjSelectedGreedy                    = "rocm_mlx_q4_projection_selected_greedy"
	hipKernelNameMLXQ4ProjSelectedGreedyQ6Row64             = "rocm_mlx_q4_projection_selected_greedy_q6_row64"
	hipKernelNameOrderedEmbeddingCandidates                 = "rocm_ordered_embedding_candidates"
	hipKernelNamePackedTopK                                 = "rocm_packed_topk"
	hipKernelNamePackedTopKSample                           = "rocm_packed_topk_sample"
	hipKernelNameMLXQ4TripleProj                            = "rocm_mlx_q4_triple_projection"
	hipKernelNameMLXQ4TripleProjQ6Row16                     = "rocm_mlx_q4_triple_projection_q6_row16"
	hipKernelNameMLXQ4TripleProjQ6Row64                     = "rocm_mlx_q4_triple_projection_q6_row64"
	hipKernelNameMLXQ4PairProj                              = "rocm_mlx_q4_pair_projection"
	hipKernelNameMLXQ4GELUTanhMul                           = "rocm_mlx_q4_gelu_tanh_multiply"
	hipKernelNameMLXQ4GELUTanhMulQ4G32Cols1536Row16         = "rocm_mlx_q4_gelu_tanh_multiply_q4_g32_cols1536_row16"
	hipKernelNameMLXQ4GELUTanhMulQ4G32Rows15360Cols3840     = "rocm_mlx_q4_gelu_tanh_multiply_q4_g32_rows15360_cols3840"
	hipKernelNameMLXQ4GELUTanhMulQ4G32Rows15360Cols3840Row8 = "rocm_mlx_q4_gelu_tanh_multiply_q4_g32_rows15360_cols3840_row8"
	hipKernelNameMLXQ4GELUTanhMulQ4G64Rows15360Cols3840Row8 = "rocm_mlx_q4_gelu_tanh_multiply_q4_g64_rows15360_cols3840_row8"
	hipKernelNameMLXQ4GELUTanhMLPQ4G32Cols1536Persistent    = "rocm_mlx_q4_gelu_tanh_mlp_q4_g32_cols1536_persistent"
	hipKernelNameMLXQ4GELUTanhMulQ6Cols1536                 = "rocm_mlx_q4_gelu_tanh_multiply_q6_cols1536"
	hipKernelNameMLXQ4GELUTanhMulQ6Cols1536Row32            = "rocm_mlx_q4_gelu_tanh_multiply_q6_cols1536_row32"
	hipKernelNameMLXQ4GELUTanhMulQ6Cols1536Row64            = "rocm_mlx_q4_gelu_tanh_multiply_q6_cols1536_row64"
	hipKernelNameMLXQ4GELUTanhMulBatch                      = "rocm_mlx_q4_gelu_tanh_multiply_batch"
	hipKernelNameMLXQ4GELUTanhMulBatchQ8G64Row16            = "rocm_mlx_q4_gelu_tanh_multiply_batch_q8_g64_row16"
	hipKernelNameMLXQ4GELUTanhProj                          = "rocm_mlx_q4_gelu_tanh_projection"
	hipKernelNameMLXQ4GELUTanhProjQ6Row16                   = "rocm_mlx_q4_gelu_tanh_projection_q6_row16"
	hipKernelNameMLXQ4GELUTanhProjBatch                     = "rocm_mlx_q4_gelu_tanh_projection_batch"
	hipKernelNameRMSNormResidualAddGELUTanhProj             = "rocm_rms_norm_residual_add_mlx_q4_gelu_tanh_projection"
	hipKernelNameRMSNorm                                    = "rocm_rms_norm"
	hipKernelNameRMSNormResidualAdd                         = "rocm_rms_norm_residual_add"
	hipKernelNameRMSNormResAddNorm                          = "rocm_rms_norm_residual_add_norm"
	hipKernelNameRMSNormHeads                               = "rocm_rms_norm_heads"
	hipKernelNameRMSNormRoPEHeads                           = "rocm_rms_norm_rope_heads"
	hipKernelNameRMSNormRoPEHeadsPair                       = "rocm_rms_norm_rope_heads_pair"
	hipKernelNameRMSNormRoPEHeadsBatch                      = "rocm_rms_norm_rope_heads_batch"
	hipKernelNameRMSNormRoPEHeadsPairLaneBatch              = "rocm_rms_norm_rope_heads_pair_lane_batch"
	hipKernelNameRoPE                                       = "rocm_rope"
	hipKernelNameRoPEHeads                                  = "rocm_rope_heads"
	hipKernelNameGreedy                                     = "rocm_greedy_sample"
	hipKernelNameSoftcapGreedy                              = "rocm_softcap_greedy_sample"
	hipKernelNameAttention                                  = "rocm_attention"
	hipKernelNameAttentionHeads                             = "rocm_attention_heads"
	hipKernelNameAttentionHeadsBatchCausal                  = "rocm_attention_heads_batch_causal"
	hipKernelNameAttentionHeadsBatchCapped                  = "rocm_attention_heads_batch_capped"
	hipKernelNameAttentionHeadsLaneBatch                    = "rocm_attention_heads_lane_batch"
	hipKernelNameAttentionHeadsBatchCausalQueryRMSRoPE      = "rocm_attention_heads_batch_causal_query_rms_rope"
	hipKernelNameAttentionHeadsChunkedStage1                = "rocm_attention_heads_chunked_stage1"
	hipKernelNameAttentionHeadsChunkedStage2                = "rocm_attention_heads_chunked_stage2"
	hipKernelNameAttentionHeadsBatchChunkedStage1           = "rocm_attention_heads_batch_chunked_stage1_v2"
	hipKernelNameAttentionHeadsBatchChunkedStage1GQA2       = "rocm_attention_heads_batch_chunked_stage1_gqa2"
	hipKernelNameAttentionHeadsBatchChunkedStage2           = "rocm_attention_heads_batch_chunked_stage2"
	hipKernelNameVectorAdd                                  = "rocm_vector_add"
	hipKernelNameVectorAddScaled                            = "rocm_vector_add_scaled"
	hipKernelNameVectorScale                                = "rocm_vector_scale"
	hipKernelNamePerLayerInputTranspose                     = "rocm_per_layer_input_transpose"
	hipKernelNameSwiGLU                                     = "rocm_swiglu"
	hipKernelNameGELUTanhMul                                = "rocm_gelu_tanh_multiply"
	hipKernelNameMoERouter                                  = "rocm_moe_router"
	hipKernelNameMoELazy                                    = "rocm_moe_lazy_experts"
	hipKernelNameMoECombineNorms                            = "rocm_moe_combine_norms"
	hipKernelNameMoEBatchGatherRows                         = "rocm_moe_batch_gather_rows"
	hipKernelNameMoEBatchScatterRoutes                      = "rocm_moe_batch_scatter_routes"
	hipKernelNameMoEBatchReduceRoutes                       = "rocm_moe_batch_reduce_routes"
	hipKernelNameMoEMLXAffineRoutes                         = "rocm_moe_mlx_affine_routes"
	hipKernelNameGGUFQ4_0Projection                         = "rocm_gguf_q4_0_projection"
	hipKernelNameGGUFQ4_0GELUTanhGateUp                     = "rocm_gguf_q4_0_gelu_tanh_gate_up"
	hipKernelNameGGUFQ4_0SelectedExpertGateUp               = "rocm_gguf_q4_0_selected_expert_gate_up"
	hipKernelNameGGUFQ4_0SelectedExpertDown                 = "rocm_gguf_q4_0_selected_expert_down"
	hipKernelNameGGUFQ4_0SelectedExpertGateUpPair16         = "rocm_gguf_q4_0_selected_expert_gate_up_pair16"
	hipKernelNameGGUFQ4_0SelectedExpertDownPair16           = "rocm_gguf_q4_0_selected_expert_down_pair16"
	hipKernelNameGGUFQ4KSelectedExpertGateUp                = "rocm_gguf_q4_k_selected_expert_gate_up"
	hipKernelNameGGUFQ5_1SelectedExpertDown                 = "rocm_gguf_q5_1_selected_expert_down"
	hipKernelNameGGUFQ8_0SelectedExpertDown                 = "rocm_gguf_q8_0_selected_expert_down"
	hipKernelNameGGUFQ4KSelectedExpertGateUpPair16          = "rocm_gguf_q4_k_selected_expert_gate_up_pair16"
	hipKernelNameGGUFQ4KSelectedExpertGateUpSplitPair16     = "rocm_gguf_q4_k_selected_expert_gate_up_split_pair16"
	hipKernelNameGGUFQ4KExpandMetadata                      = "rocm_gguf_q4_k_expand_metadata"
	hipKernelNameGGUFQ4KExpandedSelectedGateUpSplitPair16   = "rocm_gguf_q4_k_expanded_selected_expert_gate_up_split_pair16"
	hipKernelNameGGUFQ5_1SelectedExpertDownPair16           = "rocm_gguf_q5_1_selected_expert_down_pair16"
	hipKernelNameGGUFQ5_1SelectedExpertDownExpert8Pair16    = "rocm_gguf_q5_1_selected_expert_down_expert8_pair16"
	hipKernelNameGGUFQ8_0SelectedExpertDownPair16           = "rocm_gguf_q8_0_selected_expert_down_pair16"
	hipKernelNameJANGTQ                                     = "rocm_jangtq_projection"
	hipKernelNameCodebook                                   = "rocm_codebook_lookup"
	hipKernelNameLoRA                                       = "rocm_lora_projection"
	hipKernelNameEmbedLookup                                = "rocm_embedding_lookup"
	hipKernelNameEmbedLookupGreedyToken                     = "rocm_embedding_lookup_greedy_token"
	hipKernelNameDiffusionExpectedEmbedding                 = "rocm_diffusion_expected_embedding"
	hipKernelNameDiffusionExpectedEmbeddingAffineG64Rows16  = "rocm_diffusion_expected_embedding_affine_g64_rows16"
	hipKernelNameDiffusionExpectedEmbeddingQ8G64Dims4Rows4  = "rocm_diffusion_expected_embedding_q8_g64_dims4_rows4"
	hipKernelNameEmbedMean                                  = "rocm_embedding_mean_pool"
	hipKernelNameRerank                                     = "rocm_rerank_cosine"
	hipKernelNameTinyPrefill                                = "rocm_tiny_prefill"
	hipKernelNameTinyDecode                                 = "rocm_tiny_decode"
	hipKernelNameCrossEntropy                               = "rocm_cross_entropy_loss"
	hipKernelNameDistillKL                                  = "rocm_distillation_kl_loss"
	hipKernelNameGRPOAdvantage                              = "rocm_grpo_advantage"
	hipKernelNameAdamWUpdate                                = "rocm_adamw_update"
	hipKernelNameAutoRoundQuantize                          = "rocm_autoround_quantize"
)

type hipKernelLaunchConfig struct {
	Name           string
	Args           []byte
	GridX          uint32
	GridY          uint32
	GridZ          uint32
	BlockX         uint32
	BlockY         uint32
	BlockZ         uint32
	SharedMemBytes uint32
}

type nativeHIPKernelLauncher interface {
	LaunchKernel(config hipKernelLaunchConfig) error
}

type nativeHIPDeviceSynchronizer interface {
	DeviceSynchronize() error
}

type hipLaunchPacketPool struct {
	sync.Mutex
	packets [][]byte
}

var hipLaunchPacketPools sync.Map

const hipLaunchPacketPoolMaxPerSize = 512

func hipBorrowLaunchPacket(size int) []byte {
	if size <= 0 {
		return nil
	}
	poolValue, ok := hipLaunchPacketPools.Load(size)
	if !ok {
		pool := &hipLaunchPacketPool{}
		poolValue, _ = hipLaunchPacketPools.LoadOrStore(size, pool)
	}
	pool := poolValue.(*hipLaunchPacketPool)
	pool.Lock()
	if index := len(pool.packets) - 1; index >= 0 {
		packet := pool.packets[index]
		pool.packets[index] = nil
		pool.packets = pool.packets[:index]
		pool.Unlock()
		return packet[:size]
	}
	pool.Unlock()
	return make([]byte, size, size+1)
}

func hipPrewarmLaunchPacketPools(sizes []int, depth int) {
	if depth <= 0 {
		return
	}
	for _, size := range sizes {
		if size <= 0 {
			continue
		}
		packets := make([][]byte, 0, depth)
		for range depth {
			packets = append(packets, hipBorrowLaunchPacket(size))
		}
		for index := len(packets) - 1; index >= 0; index-- {
			hipReleaseLaunchPacket(packets[index])
		}
	}
}

func hipReleaseLaunchPacket(packet []byte) {
	if len(packet) == 0 || cap(packet) != len(packet)+1 {
		return
	}
	clear(packet)
	if poolValue, ok := hipLaunchPacketPools.Load(len(packet)); ok {
		pool := poolValue.(*hipLaunchPacketPool)
		pool.Lock()
		if len(pool.packets) < hipLaunchPacketPoolMaxPerSize {
			pool.packets = append(pool.packets, packet[:0])
		}
		pool.Unlock()
	}
}

func hipLaunchKernel(driver nativeHIPDriver, config hipKernelLaunchConfig) error {
	if err := config.Validate(); err != nil {
		return err
	}
	if driver == nil {
		return core.E("rocm.hip.LaunchKernel", "HIP driver is nil", nil)
	}
	if !driver.Available() {
		return core.E("rocm.hip.LaunchKernel", "HIP driver is not available", nil)
	}
	launcher, ok := driver.(nativeHIPKernelLauncher)
	if !ok {
		return core.E("rocm.hip.LaunchKernel", "native HIP kernel launcher is not linked yet", nil)
	}
	metrics := hipActiveDecodeRouteMetrics()
	if metrics == nil {
		return launcher.LaunchKernel(config)
	}
	start := time.Now()
	if err := launcher.LaunchKernel(config); err != nil {
		return err
	}
	if synchronizer, ok := driver.(nativeHIPDeviceSynchronizer); ok {
		if err := synchronizer.DeviceSynchronize(); err != nil {
			return err
		}
	}
	metrics.record(config.Name, hipDecodeRouteDevice, time.Since(start))
	return nil
}

func (config hipKernelLaunchConfig) Validate() error {
	if config.Name == "" {
		return core.E("rocm.hip.LaunchKernel", "kernel name is required", nil)
	}
	if len(config.Args) == 0 {
		return core.E("rocm.hip.LaunchKernel", "kernel launch args are required", nil)
	}
	if config.GridX == 0 || config.GridY == 0 || config.GridZ == 0 {
		return core.E("rocm.hip.LaunchKernel", "kernel grid dimensions must be positive", nil)
	}
	if config.BlockX == 0 || config.BlockY == 0 || config.BlockZ == 0 {
		return core.E("rocm.hip.LaunchKernel", "kernel block dimensions must be positive", nil)
	}
	return nil
}

func hipOneDimensionalLaunchConfig(name string, args []byte, workItems int) (hipKernelLaunchConfig, error) {
	work, err := rocmDeviceKVPositiveUint32("work items", workItems)
	if err != nil {
		return hipKernelLaunchConfig{}, err
	}
	const blockSize uint32 = 64
	gridX := (work + blockSize - 1) / blockSize
	config := hipKernelLaunchConfig{
		Name:   name,
		Args:   args,
		GridX:  gridX,
		GridY:  1,
		GridZ:  1,
		BlockX: blockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}

func hipSingleBlockLaunchConfig(name string, args []byte, blockSize uint32) (hipKernelLaunchConfig, error) {
	config := hipKernelLaunchConfig{
		Name:   name,
		Args:   args,
		GridX:  1,
		GridY:  1,
		GridZ:  1,
		BlockX: blockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	return config, config.Validate()
}
