// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// TestHIPGemma4Q4LayerOracle is the #52 layer-0 numerical oracle. It runs the
// real batched-prefill attention sub-block for a chosen layer on the GPU, reads
// every intermediate device buffer to host, and diffs each op against an
// independent float reference computed from HIP's OWN upstream buffer (so the
// first diverging op is isolated, not the accumulated error). The reference is
// convention-agnostic by construction: run it on a coherent model (E2B/E4B) and
// every op must match; any op that matches on the coherent model but diverges on
// the dense 12B is the bug's address.
//
// Env: GO_ROCM_RUN_HIP_TESTS=1, GO_ROCM_KERNEL_HSACO=<gfx1101 hsaco>,
// GO_ROCM_ORACLE_MODEL_PATH (or GO_ROCM_PRODUCTION_MODEL_PATH / GO_ROCM_MODEL_PATH).
// Optional: GO_ROCM_ORACLE_LAYER (default 0), GO_ROCM_ORACLE_TOKEN_COUNT
// (default 8), GO_ROCM_ORACLE_TOL (default 0.05 = 5% of reference RMS).
func TestHIPGemma4Q4LayerOracle(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_HIP_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_HIP_TESTS=1 to run ROCm hardware oracle tests")
	}
	if os.Getenv("GO_ROCM_KERNEL_HSACO") == "" {
		t.Skip("set GO_ROCM_KERNEL_HSACO to a compiled gfx1101 kernels/rocm_kernels.hip HSACO")
	}
	modelPath := hipOracleModelPath()
	if modelPath == "" {
		t.Skip("set GO_ROCM_ORACLE_MODEL_PATH / GO_ROCM_PRODUCTION_MODEL_PATH to a local Gemma4 MLX-affine pack")
	}
	layerIndex := hipOracleEnvInt("GO_ROCM_ORACLE_LAYER", 0)
	tokenCount := hipOracleEnvInt("GO_ROCM_ORACLE_TOKEN_COUNT", 8)
	tolRatio := float32(hipOracleEnvFloat("GO_ROCM_ORACLE_TOL", 0.05))
	const epsilon = float32(1e-6)

	runtime := newSystemNativeRuntime()
	if !runtime.Available() {
		t.Fatalf("native ROCm runtime is not available")
	}
	model, err := resultValue[inference.TextModel](newROCmBackendWithRuntime(runtime).LoadModel(modelPath, inference.WithContextLen(4096)))
	if err != nil {
		t.Fatalf("LoadModel(%q): %v", modelPath, err)
	}
	defer model.Close()
	rocmLoaded, ok := model.(*rocmModel)
	if !ok {
		t.Fatalf("LoadModel returned %T, want *rocmModel", model)
	}
	loaded, ok := rocmLoaded.native.(*hipLoadedModel)
	if !ok {
		t.Fatalf("native is %T, want *hipLoadedModel", rocmLoaded.native)
	}
	driver := loaded.driver
	ctx := context.Background()

	cfg, err := loaded.loadedGemma4Q4ForwardConfig(layerIndex + 1)
	if err != nil {
		t.Fatalf("loadedGemma4Q4ForwardConfig(%d): %v", layerIndex+1, err)
	}
	if layerIndex >= len(cfg.Layers) {
		t.Fatalf("layer index %d exceeds loaded layer count %d", layerIndex, len(cfg.Layers))
	}
	layer := cfg.Layers[layerIndex]
	hidden := layer.HiddenSize
	queryRows := layer.QueryHeads * layer.HeadDim
	kvDim := layer.KeyHeads * layer.HeadDim

	tokens := make([]int32, tokenCount)
	for i := range tokens {
		tokens[i] = int32((i*2654435761 + 12345) % layer.VocabSize)
		if tokens[i] < 0 {
			tokens[i] += int32(layer.VocabSize)
		}
	}

	t.Logf("ORACLE model=%s layer=%d type=%s k_eq_v=%v heads q=%d kv=%d headDim=%d hidden=%d window=%d ropeBase=%.0f rotaryDim=%d freqScale=%.4f tokens=%d",
		modelPath, layerIndex, layer.LayerType, layer.AttentionKEqV, layer.QueryHeads, layer.KeyHeads, layer.HeadDim, hidden, layer.SlidingWindow, layer.RoPEBase, layer.RoPERotaryDim, layer.effectiveRoPEFrequencyScale(), tokenCount)

	report := &hipOracleReport{t: t, tolRatio: tolRatio}

	// --- Embedding ---
	hiddenBuf, err := hipRunGemma4Q4PrefillEmbeddingBatch(ctx, driver, layer, tokens)
	if err != nil {
		t.Fatalf("embedding: %v", err)
	}
	defer hiddenBuf.Close()
	hipHidden := hipOracleReadF32(t, hiddenBuf, tokenCount*hidden)
	report.rmsOnly("embedding", hipHidden, hidden)

	// --- Input norm ---
	inputNormBuf, err := hipRunGemma4Q4PrefillInputNormBatch(ctx, driver, layer, hiddenBuf, tokenCount)
	if err != nil {
		t.Fatalf("input norm: %v", err)
	}
	defer inputNormBuf.Close()
	hipInputNorm := hipOracleReadF32(t, inputNormBuf, tokenCount*hidden)
	inputNormW := hipOracleReadNormWeight(t, driver, layer.InputNorm)
	refInputNorm := make([]float32, tokenCount*hidden)
	for tok := 0; tok < tokenCount; tok++ {
		row, err := hipReferenceRMSNorm(hipHidden[tok*hidden:(tok+1)*hidden], inputNormW, epsilon)
		core.RequireNoError(t, err)
		copy(refInputNorm[tok*hidden:], row)
	}
	report.diff("input_norm", refInputNorm, hipInputNorm, hidden)

	// --- Q/K/V projection ---
	qkv, err := hipRunGemma4Q4PrefillQKVProjectionBatch(ctx, driver, layer, inputNormBuf, tokenCount)
	if err != nil {
		t.Fatalf("qkv projection: %v", err)
	}
	defer qkv.Close()
	hipQuery := hipOracleReadF32(t, qkv.Query, tokenCount*queryRows)
	hipKey := hipOracleReadF32(t, qkv.Key, tokenCount*kvDim)
	hipValueRaw := hipOracleReadF32(t, qkv.Value, tokenCount*kvDim)

	// Projection oracle on token 0 only (plain quantised matmul, kernel shared
	// across all layers/models; one token detects a layout/correctness fault).
	report.projection("q_proj", t, driver, layer.QueryProjection, hipInputNorm[:hidden], hipQuery[:queryRows])
	report.projection("k_proj", t, driver, layer.KeyProjection, hipInputNorm[:hidden], hipKey[:kvDim])
	if !layer.AttentionKEqV {
		report.projection("v_proj", t, driver, layer.ValueProjection, hipInputNorm[:hidden], hipValueRaw[:kvDim])
	} else {
		report.note("v_proj", "k_eq_v: value derives from k_proj (checked via value_norm)")
	}

	// --- Q norm + RoPE / K norm + RoPE ---
	qk, err := hipRunGemma4Q4PrefillQKNormRoPEBatch(ctx, driver, layer, qkv, tokenCount, 0, epsilon)
	if err != nil {
		t.Fatalf("qk norm rope: %v", err)
	}
	defer qk.Close()
	hipRoPEQuery := hipOracleReadF32(t, qk.Query, tokenCount*queryRows)
	hipRoPEKey := hipOracleReadF32(t, qk.Key, tokenCount*kvDim)
	qNormW := hipOracleReadNormWeight(t, driver, layer.QueryNorm)
	kNormW := hipOracleReadNormWeight(t, driver, layer.KeyNorm)
	effFreqDim, effRotary := hipGemma4Q4RoPEKernelDims(layer)
	if effFreqDim == 0 {
		effFreqDim = layer.HeadDim
	}
	if effRotary == 0 {
		effRotary = layer.HeadDim
	}
	freqScale := float64(layer.effectiveRoPEFrequencyScale())
	refRoPEQuery := hipOracleNormRoPEHeads(t, hipQuery, tokenCount, layer.QueryHeads, layer.HeadDim, qNormW, epsilon, float64(layer.RoPEBase), effFreqDim, effRotary, freqScale)
	report.diff("q_norm_rope", refRoPEQuery, hipRoPEQuery, layer.HeadDim)
	refRoPEKey := hipOracleNormRoPEHeads(t, hipKey, tokenCount, layer.KeyHeads, layer.HeadDim, kNormW, epsilon, float64(layer.RoPEBase), effFreqDim, effRotary, freqScale)
	report.diff("k_norm_rope", refRoPEKey, hipRoPEKey, layer.HeadDim)

	// --- Value norm (per-head, no scale) ---
	valueBuf, err := hipRunGemma4Q4PrefillValueNormBatch(ctx, driver, layer, qkv, tokenCount, epsilon)
	if err != nil {
		t.Fatalf("value norm: %v", err)
	}
	defer valueBuf.Close()
	hipValue := hipOracleReadF32(t, valueBuf, tokenCount*kvDim)
	ones := make([]float32, layer.HeadDim)
	for i := range ones {
		ones[i] = 1
	}
	refValue := make([]float32, tokenCount*kvDim)
	for tok := 0; tok < tokenCount; tok++ {
		for h := 0; h < layer.KeyHeads; h++ {
			base := tok*kvDim + h*layer.HeadDim
			row, err := hipReferenceRMSNorm(hipValueRaw[base:base+layer.HeadDim], ones, epsilon)
			core.RequireNoError(t, err)
			copy(refValue[base:], row)
		}
	}
	report.diff("value_norm", refValue, hipValue, layer.HeadDim)

	// --- Attention (SDPA over the contiguous prefill KV) ---
	layerKV, err := hipRunGemma4Q4PrefillLayerKVBatch(ctx, driver, layer, hiddenBuf, tokenCount, 0, epsilon, rocmKVCacheModeKQ8VQ4)
	if err != nil {
		t.Fatalf("layer kv batch: %v", err)
	}
	defer layerKV.Close()
	attnBuf, err := hipRunGemma4Q4PrefillAttentionBatch(ctx, driver, layer, layerKV, tokenCount, 0)
	if err != nil {
		t.Fatalf("attention: %v", err)
	}
	defer attnBuf.Close()
	hipAttn := hipOracleReadF32(t, attnBuf, tokenCount*queryRows)
	// Reference SDPA from HIP's own rope query / rope key / value-norm buffers.
	kvHeadCount := layer.KeyHeads
	refAttn := make([]float32, tokenCount*queryRows)
	for tok := 0; tok < tokenCount; tok++ {
		visible := tok + 1
		windowStart := 0
		if layer.SlidingWindow > 0 && visible > layer.SlidingWindow {
			windowStart = visible - layer.SlidingWindow
		}
		for h := 0; h < layer.QueryHeads; h++ {
			kvHead := int(rocmOracleKVHeadForQuery(h, layer.QueryHeads, kvHeadCount))
			q := hipRoPEQuery[tok*queryRows+h*layer.HeadDim : tok*queryRows+(h+1)*layer.HeadDim]
			keys := make([][]float32, 0, visible-windowStart)
			values := make([][]float32, 0, visible-windowStart)
			for p := windowStart; p < visible; p++ {
				kb := p*kvDim + kvHead*layer.HeadDim
				keys = append(keys, hipRoPEKey[kb:kb+layer.HeadDim])
				values = append(values, hipValue[kb:kb+layer.HeadDim])
			}
			out, _, err := hipReferenceSingleHeadAttentionWithScale(q, keys, values, 1)
			core.RequireNoError(t, err)
			copy(refAttn[tok*queryRows+h*layer.HeadDim:], out)
		}
	}
	report.diff("attention_sdpa", refAttn, hipAttn, layer.HeadDim)

	// --- Post-attention sub-block: o_proj, residual, norms, MLP, final ---
	// Run the full layer body and diff each post-attention op (token 0) against
	// a reference fed HIP's OWN upstream buffer. This covers the half the
	// attention oracle above does not: o_proj layout, the two residual+norm
	// glues, the GeGLU MLP, and the per-layer output scalar.
	if os.Getenv("GO_ROCM_ORACLE_SKIP_MLP") != "1" {
		body, err := hipRunGemma4Q4PrefillLayerBodyBatchWithPerLayerInput(ctx, driver, layer, hiddenBuf, layerKV, nil, tokenCount, 0, epsilon)
		if err != nil {
			t.Fatalf("layer body: %v", err)
		}
		defer body.Close()
		attnOut0 := hipOracleReadF32(t, body.AttentionOutput, tokenCount*queryRows)[:queryRows]
		attnProj0 := hipOracleReadF32(t, body.AttentionProjection, tokenCount*hidden)[:hidden]
		attnResid0 := hipOracleReadF32(t, body.AttentionResidual, tokenCount*hidden)[:hidden]
		preFF0 := hipOracleReadF32(t, body.PreFeedForward, tokenCount*hidden)[:hidden]
		mlpOut0 := hipOracleReadF32(t, body.MLPOutput, tokenCount*hidden)[:hidden]
		final0 := hipOracleReadF32(t, body.FinalHidden, tokenCount*hidden)[:hidden]
		postAttnW := hipOracleReadNormWeight(t, driver, layer.PostAttentionNorm)
		preFFW := hipOracleReadNormWeight(t, driver, layer.PreFeedForwardNorm)
		postFFW := hipOracleReadNormWeight(t, driver, layer.PostFeedForwardNorm)
		layerScalar := layer.effectiveLayerScalar()
		report.note("layer_scalar", fmt.Sprintf("effectiveLayerScalar=%.6f (raw=%.6f)", layerScalar, layer.LayerScalar))

		report.projection("o_proj", t, driver, layer.OutputProjection, attnOut0, attnProj0)

		residRef := make([]float32, hidden)
		normedAttn, err := hipReferenceRMSNorm(attnProj0, postAttnW, epsilon)
		core.RequireNoError(t, err)
		for i := 0; i < hidden; i++ {
			residRef[i] = hipHidden[i] + normedAttn[i]
		}
		report.diff("attn_residual", residRef, attnResid0, hidden)

		preFFRef, err := hipReferenceRMSNorm(attnResid0, preFFW, epsilon)
		core.RequireNoError(t, err)
		report.diff("pre_ff_norm", preFFRef, preFF0, hidden)

		gateBits := layer.GateProjection.Bits
		if gateBits == 0 {
			gateBits = hipMLXQ4ProjectionBits
		}
		gateW := hipOracleReadUint32(t, driver, layer.GateProjection.WeightPointer, layer.GateProjection.WeightBytes)
		gateS := hipOracleReadUint16(t, driver, layer.GateProjection.ScalePointer, layer.GateProjection.ScaleBytes)
		gateB := hipOracleReadUint16(t, driver, layer.GateProjection.BiasPointer, layer.GateProjection.BiasBytes)
		gate, err := hipReferenceMLXAffineProjection(preFF0, gateW, gateS, gateB, layer.GateProjection.Rows, layer.GateProjection.Cols, layer.GateProjection.GroupSize, gateBits)
		core.RequireNoError(t, err)
		upW := hipOracleReadUint32(t, driver, layer.UpProjection.WeightPointer, layer.UpProjection.WeightBytes)
		upS := hipOracleReadUint16(t, driver, layer.UpProjection.ScalePointer, layer.UpProjection.ScaleBytes)
		upB := hipOracleReadUint16(t, driver, layer.UpProjection.BiasPointer, layer.UpProjection.BiasBytes)
		up, err := hipReferenceMLXAffineProjection(preFF0, upW, upS, upB, layer.UpProjection.Rows, layer.UpProjection.Cols, layer.UpProjection.GroupSize, gateBits)
		core.RequireNoError(t, err)
		gelu, err := hipGemma4Q4HostGELU(gate)
		core.RequireNoError(t, err)
		act, err := hipGemma4Q4HostMultiply(gelu, up)
		core.RequireNoError(t, err)
		downW := hipOracleReadUint32(t, driver, layer.DownProjection.WeightPointer, layer.DownProjection.WeightBytes)
		downS := hipOracleReadUint16(t, driver, layer.DownProjection.ScalePointer, layer.DownProjection.ScaleBytes)
		downB := hipOracleReadUint16(t, driver, layer.DownProjection.BiasPointer, layer.DownProjection.BiasBytes)
		mlpRef, err := hipReferenceMLXAffineProjection(act, downW, downS, downB, layer.DownProjection.Rows, layer.DownProjection.Cols, layer.DownProjection.GroupSize, gateBits)
		core.RequireNoError(t, err)
		report.diff("mlp", mlpRef, mlpOut0, hidden)

		// HIP/metal formula: final = (attnResidual + postFFNorm(mlp)) * layer_scalar
		// (the scalar multiplies the WHOLE residual output). Reference matches that.
		finalRef := make([]float32, hidden)
		normedMLP, err := hipReferenceRMSNorm(mlpOut0, postFFW, epsilon)
		core.RequireNoError(t, err)
		for i := 0; i < hidden; i++ {
			finalRef[i] = (attnResid0[i] + normedMLP[i]) * layerScalar
		}
		report.diff("final_hidden", finalRef, final0, hidden)
	}

	// --- Device-KV attention (the DECODE path) ---
	// The contiguous attention above is the prefill fast-path. Decode instead
	// attends a single query token against the descriptor-addressed device KV
	// cache. Build the cache from HIP's OWN exact rope-key / value-norm buffers
	// (so the ONLY new variable is the device-KV encode+read), then attend the
	// last token's query (tokenCount=1, queryStartToken=P-1 forces the device
	// branch). FP16 is near-exact — a divergence there is a genuine
	// geometry/stride/mapping fault; KQ8VQ4 adds quantisation (looser tol).
	{
		last := tokenCount - 1
		refLast := refAttn[last*queryRows:]
		keyPayload, err := hipFloat32Payload(hipRoPEKey)
		core.RequireNoError(t, err)
		keyBuf, err := hipUploadByteBuffer(driver, "rocm.hip.Oracle", "oracle devkv keys", keyPayload, tokenCount*kvDim)
		core.RequireNoError(t, err)
		defer keyBuf.Close()
		valPayload, err := hipFloat32Payload(hipValue)
		core.RequireNoError(t, err)
		valBuf, err := hipUploadByteBuffer(driver, "rocm.hip.Oracle", "oracle devkv values", valPayload, tokenCount*kvDim)
		core.RequireNoError(t, err)
		defer valBuf.Close()
		lastQuery := hipRoPEQuery[last*queryRows : (last+1)*queryRows]
		queryPayload, err := hipFloat32Payload(lastQuery)
		core.RequireNoError(t, err)
		queryBuf, err := hipUploadByteBuffer(driver, "rocm.hip.Oracle", "oracle devkv query", queryPayload, queryRows)
		core.RequireNoError(t, err)
		defer queryBuf.Close()

		blockSize := hipGemma4Q4DeviceKVBlockSizeForSlidingWindow(layer.SlidingWindow)
		// FP16 is near-exact and asserts (catches device-KV geometry/stride/mapping
		// faults). Q8/KQ8VQ4 carry inherent quantisation loss that coherent models
		// (E2B/E4B) also show, so they are reported informationally, not asserted.
		for _, tc := range []struct {
			mode   string
			tol    float32
			assert bool
		}{{rocmKVCacheModeFP16, tolRatio, true}, {rocmKVCacheModeQ8, tolRatio + 0.05, false}, {rocmKVCacheModeKQ8VQ4, tolRatio + 0.50, false}} {
			label := "attention_devkv_" + tc.mode
			empty := &rocmDeviceKVCache{driver: driver, mode: tc.mode, blockSize: blockSize}
			cache, err := empty.withAppendedDeviceRowsWindow(ctx, keyBuf, valBuf, kvDim, kvDim, tokenCount, 0)
			if err != nil {
				report.note(label, fmt.Sprintf("cache build error: %v", err))
				report.failed = append(report.failed, label)
				continue
			}
			table, err := cache.KernelDescriptorTable()
			if err != nil {
				cache.Close()
				report.note(label, fmt.Sprintf("descriptor error: %v", err))
				report.failed = append(report.failed, label)
				continue
			}
			outBuf, err := hipAllocateByteBuffer(driver, "rocm.hip.Oracle", "oracle devkv output", uint64(queryRows*4), queryRows)
			core.RequireNoError(t, err)
			req := hipAttentionHeadsBatchCausalDeviceRequest{
				Dim:             layer.HeadDim,
				DeviceKV:        cache,
				DescriptorTable: table,
				TokenCount:      cache.TokenCount(),
				HeadCount:       layer.QueryHeads,
				KeyHeads:        layer.KeyHeads,
				QueryCount:      1,
				QueryStartToken: last,
				WindowSize:      layer.SlidingWindow,
				Scale:           1,
			}
			if err := hipRunAttentionHeadsBatchCausalOutputFromDeviceQueryToDeviceKernel(ctx, driver, req, queryBuf, outBuf); err != nil {
				report.note(label, fmt.Sprintf("attention error: %v", err))
				if tc.assert {
					report.failed = append(report.failed, label)
				}
			} else if tc.assert {
				report.diffTol(label, refLast, hipOracleReadF32(t, outBuf, queryRows), tc.tol)
			} else {
				report.diffInfo(label, refLast, hipOracleReadF32(t, outBuf, queryRows))
			}
			outBuf.Close()
			table.Close()
			cache.Close()
		}
	}

	report.finish()
}

// TestHIPGemma4Q4DecodeKVMode drives a real end-to-end decode with a chosen
// device-KV mode and prints the transcript, so coherence can be compared across
// fp16 / q8 / k-q8-v-q4. Env: GO_ROCM_RUN_HIP_TESTS=1, GO_ROCM_KERNEL_HSACO,
// model path, GO_ROCM_GEN_KV_MODE (default k-q8-v-q4), GO_ROCM_GEN_PROMPT,
// GO_ROCM_GEN_MAX_TOKENS (default 120).
func TestHIPGemma4Q4DecodeKVMode(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_HIP_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_HIP_TESTS=1 to run ROCm decode coherence tests")
	}
	if os.Getenv("GO_ROCM_KERNEL_HSACO") == "" {
		t.Skip("set GO_ROCM_KERNEL_HSACO to a compiled gfx1101 HSACO")
	}
	modelPath := hipOracleModelPath()
	if modelPath == "" {
		t.Skip("set GO_ROCM_ORACLE_MODEL_PATH / GO_ROCM_PRODUCTION_MODEL_PATH")
	}
	mode := strings.TrimSpace(os.Getenv("GO_ROCM_GEN_KV_MODE"))
	if mode == "" {
		mode = rocmKVCacheModeKQ8VQ4
	}
	prompt := strings.TrimSpace(os.Getenv("GO_ROCM_GEN_PROMPT"))
	if prompt == "" {
		prompt = "why the sky is blue"
	}
	maxTokens := hipOracleEnvInt("GO_ROCM_GEN_MAX_TOKENS", 120)

	runtime := newSystemNativeRuntime()
	if !runtime.Available() {
		t.Fatalf("native ROCm runtime is not available")
	}
	model, err := newROCmBackendWithRuntime(runtime).LoadModelWithConfig(modelPath, ROCmLoadConfig{DeviceKVMode: mode}, inference.WithContextLen(4096))
	if err != nil {
		t.Fatalf("LoadModelWithConfig(%q, kv=%s): %v", modelPath, mode, err)
	}
	defer model.Close()

	var b strings.Builder
	for _, s := range collectTokenText(model.Generate(context.Background(), prompt, inference.WithMaxTokens(maxTokens))) {
		b.WriteString(s)
	}
	if err := resultError(model.Err()); err != nil {
		t.Fatalf("Generate(kv=%s): %v", mode, err)
	}
	text := b.String()
	t.Logf("=== DECODE kv=%s model=%s prompt=%q maxTokens=%d ===\n%s\n=== end (%d chars) ===", mode, modelPath, prompt, maxTokens, text, len(text))
}

type hipOracleReport struct {
	t        *testing.T
	tolRatio float32
	rows     []string
	failed   []string
}

// TestHIPGemma4Q4ChainedLayerOracle runs the complete transformer as two
// independent chains: the normal HIP forward and a float32 host reference.
// Unlike TestHIPGemma4Q4LayerOracle, every reference layer consumes the prior
// REFERENCE residual. This makes cross-layer accumulation visible and reports
// the first tensor at which the two chains separate.
//
// Env: the common HIP oracle variables above. GO_ROCM_ORACLE_TOKEN_COUNT
// defaults to 1 because a full 12B host reference is intentionally expensive.
// GO_ROCM_CHAIN_TOL defaults to 0.005 (0.5% of reference RMS).
func TestHIPGemma4Q4ChainedLayerOracle(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_HIP_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_HIP_TESTS=1 to run ROCm hardware oracle tests")
	}
	if os.Getenv("GO_ROCM_KERNEL_HSACO") == "" {
		t.Skip("set GO_ROCM_KERNEL_HSACO to a compiled gfx1101 kernels/rocm_kernels.hip HSACO")
	}
	modelPath := hipOracleModelPath()
	if modelPath == "" {
		t.Skip("set GO_ROCM_ORACLE_MODEL_PATH / GO_ROCM_PRODUCTION_MODEL_PATH to a local Gemma4 MLX-affine pack")
	}
	tokenCount := hipOracleEnvInt("GO_ROCM_ORACLE_TOKEN_COUNT", 1)
	tolRatio := float32(hipOracleEnvFloat("GO_ROCM_CHAIN_TOL", 0.005))
	const epsilon = float32(1e-6)

	runtime := newSystemNativeRuntime()
	if !runtime.Available() {
		t.Fatalf("native ROCm runtime is not available")
	}
	model, err := resultValue[inference.TextModel](newROCmBackendWithRuntime(runtime).LoadModel(modelPath, inference.WithContextLen(4096)))
	if err != nil {
		t.Fatalf("LoadModel(%q): %v", modelPath, err)
	}
	defer model.Close()
	loaded := model.(*rocmModel).native.(*hipLoadedModel)
	cfg, err := loaded.loadedGemma4Q4ForwardConfig(loaded.modelInfo.NumLayers)
	core.RequireNoError(t, err)
	tokens := make([]int32, tokenCount)
	for i := range tokens {
		tokens[i] = int32((i*2654435761 + 12345) % cfg.Layers[0].VocabSize)
		if tokens[i] < 0 {
			tokens[i] += int32(cfg.Layers[0].VocabSize)
		}
	}

	ctx := context.Background()
	forward, err := hipRunGemma4Q4PrefillForwardBatch(ctx, loaded.driver, cfg, tokens, 0, epsilon, rocmKVCacheModeFP16, nil, nil, nil)
	core.RequireNoError(t, err)
	defer forward.Close()
	ref := hipOracleReadF32(t, forward.Embedding, tokenCount*cfg.Layers[0].HiddenSize)
	perLayer, err := hipRunGemma4Q4PrefillPerLayerInputDeviceSetBatch(ctx, loaded.driver, cfg, tokens, forward.Embedding, epsilon)
	core.RequireNoError(t, err)
	if perLayer != nil {
		defer perLayer.Close()
	}

	rows := make([]string, 0, len(cfg.Layers))
	refLayers := make([]hipOracleLayerReference, 0, len(cfg.Layers))
	sharedSources := hipGemma4Q4SharedKVSourceByLayer(cfg)
	firstLayer := -1
	firstTensor := ""
	var firstMax, firstMean, firstRatio float32
	for index, layer := range cfg.Layers {
		var multiplier []float32
		if perLayer != nil {
			multiplier = hipOracleReadF32(t, perLayer.Layer(index), tokenCount*layer.PerLayerInput.InputSize)
		}
		var sharedKV *hipOracleLayerReference
		if sharedSources[index] != index {
			sharedKV = &refLayers[sharedSources[index]]
		}
		result := hipOracleReferenceLayer(t, loaded.driver, layer, ref, multiplier, sharedKV, tokenCount, epsilon)
		hipBody := forward.Layers[index].Body
		tensors := []struct {
			name string
			ref  []float32
			hip  *hipDeviceByteBuffer
		}{
			{"input_norm", result.inputNorm, forward.Layers[index].KV.InputNorm},
			{"attention", result.attention, hipBody.AttentionOutput},
			{"attention_projection", result.attentionProjection, hipBody.AttentionProjection},
			{"attention_residual", result.attentionResidual, hipBody.AttentionResidual},
			{"pre_ff_norm", result.preFeedForward, hipBody.PreFeedForward},
			{"mlp", result.mlp, hipBody.MLPOutput},
			{"post_ff", result.postFeedForward, hipBody.PostFeedForward},
			{"final_hidden", result.finalHidden, hipBody.FinalHidden},
		}
		layerTensor := "clean"
		var layerMax, layerMean, layerRatio float32
		for _, tensor := range tensors {
			hip := hipOracleReadF32(t, tensor.hip, len(tensor.ref))
			maxAbs, meanAbs := hipOracleMaxMeanDiff(tensor.ref, hip)
			ratio := float32(0)
			if rms := hipOracleRMS(tensor.ref); rms > 0 {
				ratio = maxAbs / rms
			}
			if ratio > layerRatio {
				layerMax, layerMean, layerRatio = maxAbs, meanAbs, ratio
			}
			if layerTensor == "clean" && ratio > tolRatio {
				layerTensor = tensor.name
				if firstLayer < 0 {
					firstLayer, firstTensor = index, tensor.name
					firstMax, firstMean, firstRatio = maxAbs, meanAbs, ratio
				}
			}
		}
		rows = append(rows, fmt.Sprintf("  L%02d %-22s maxAbs=%-11.6g meanAbs=%-11.6g refRMS=%-9.5g max/refRMS=%.6g", index, layerTensor, layerMax, layerMean, hipOracleRMS(result.finalHidden), layerRatio))
		ref = result.finalHidden
		refLayers = append(refLayers, result)
	}
	t.Logf("=== #52 chained float-reference divergence table model=%s tokens=%d tol=%.6g ===\n%s", modelPath, tokenCount, tolRatio, strings.Join(rows, "\n"))
	if firstLayer >= 0 {
		t.Fatalf("FIRST CHAIN DIVERGENCE layer=%d tensor=%s maxAbs=%.7g meanAbs=%.7g max/refRMS=%.7g", firstLayer, firstTensor, firstMax, firstMean, firstRatio)
	}
}

type hipOracleLayerReference struct {
	inputNorm           []float32
	attention           []float32
	attentionProjection []float32
	attentionResidual   []float32
	preFeedForward      []float32
	mlp                 []float32
	postFeedForward     []float32
	finalHidden         []float32
	key                 []float32
	value               []float32
}

func hipOracleReferenceLayer(t *testing.T, driver nativeHIPDriver, layer hipGemma4Q4Layer0Config, input, multiplier []float32, sharedKV *hipOracleLayerReference, tokenCount int, epsilon float32) hipOracleLayerReference {
	t.Helper()
	hidden := layer.HiddenSize
	queryRows := layer.QueryHeads * layer.HeadDim
	kvRows := layer.KeyHeads * layer.HeadDim
	result := hipOracleLayerReference{}
	result.inputNorm = hipOracleReferenceNormRows(t, input, hipOracleReadNormWeight(t, driver, layer.InputNorm), tokenCount, hidden, epsilon)
	query := hipOracleReferenceProjectionRows(t, driver, layer.QueryProjection, result.inputNorm, tokenCount)
	key := hipOracleReferenceProjectionRows(t, driver, layer.KeyProjection, result.inputNorm, tokenCount)
	valueRaw := key
	if !layer.AttentionKEqV {
		valueRaw = hipOracleReferenceProjectionRows(t, driver, layer.ValueProjection, result.inputNorm, tokenCount)
	}
	freqDim, rotary := hipGemma4Q4RoPEKernelDims(layer)
	if freqDim == 0 {
		freqDim = layer.HeadDim
	}
	if rotary == 0 {
		rotary = layer.HeadDim
	}
	query = hipOracleNormRoPEHeads(t, query, tokenCount, layer.QueryHeads, layer.HeadDim, hipOracleReadNormWeight(t, driver, layer.QueryNorm), epsilon, float64(layer.RoPEBase), freqDim, rotary, float64(layer.effectiveRoPEFrequencyScale()))
	key = hipOracleNormRoPEHeads(t, key, tokenCount, layer.KeyHeads, layer.HeadDim, hipOracleReadNormWeight(t, driver, layer.KeyNorm), epsilon, float64(layer.RoPEBase), freqDim, rotary, float64(layer.effectiveRoPEFrequencyScale()))
	ones := make([]float32, layer.HeadDim)
	for i := range ones {
		ones[i] = 1
	}
	value := hipOracleReferenceHeadNormRows(t, valueRaw, ones, tokenCount, layer.KeyHeads, layer.HeadDim, epsilon)
	result.key, result.value = key, value
	if sharedKV != nil {
		key, value = sharedKV.key, sharedKV.value
		result.key, result.value = key, value
	}
	result.attention = make([]float32, tokenCount*queryRows)
	for tok := 0; tok < tokenCount; tok++ {
		start := 0
		if layer.SlidingWindow > 0 && tok+1 > layer.SlidingWindow {
			start = tok + 1 - layer.SlidingWindow
		}
		for head := 0; head < layer.QueryHeads; head++ {
			kvHead := rocmOracleKVHeadForQuery(head, layer.QueryHeads, layer.KeyHeads)
			keys, values := make([][]float32, 0, tok+1-start), make([][]float32, 0, tok+1-start)
			for pos := start; pos <= tok; pos++ {
				base := pos*kvRows + kvHead*layer.HeadDim
				keys, values = append(keys, key[base:base+layer.HeadDim]), append(values, value[base:base+layer.HeadDim])
			}
			base := tok*queryRows + head*layer.HeadDim
			out, _, err := hipReferenceSingleHeadAttentionWithScale(query[base:base+layer.HeadDim], keys, values, 1)
			core.RequireNoError(t, err)
			copy(result.attention[base:], out)
		}
	}
	result.attentionProjection = hipOracleReferenceProjectionRows(t, driver, layer.OutputProjection, result.attention, tokenCount)
	postAttention := hipOracleReferenceNormRows(t, result.attentionProjection, hipOracleReadNormWeight(t, driver, layer.PostAttentionNorm), tokenCount, hidden, epsilon)
	result.attentionResidual = hipOracleAddRows(input, postAttention, 1)
	result.preFeedForward = hipOracleReferenceNormRows(t, result.attentionResidual, hipOracleReadNormWeight(t, driver, layer.PreFeedForwardNorm), tokenCount, hidden, epsilon)
	gate := hipOracleReferenceProjectionRows(t, driver, layer.GateProjection, result.preFeedForward, tokenCount)
	up := hipOracleReferenceProjectionRows(t, driver, layer.UpProjection, result.preFeedForward, tokenCount)
	activation := make([]float32, len(gate))
	for row := 0; row < tokenCount; row++ {
		start := row * layer.GateProjection.Rows
		gelu, err := hipGemma4Q4HostGELU(gate[start : start+layer.GateProjection.Rows])
		core.RequireNoError(t, err)
		product, err := hipGemma4Q4HostMultiply(gelu, up[start:start+layer.GateProjection.Rows])
		core.RequireNoError(t, err)
		copy(activation[start:], product)
	}
	result.mlp = hipOracleReferenceProjectionRows(t, driver, layer.DownProjection, activation, tokenCount)
	postFFNorm := hipOracleReferenceNormRows(t, result.mlp, hipOracleReadNormWeight(t, driver, layer.PostFeedForwardNorm), tokenCount, hidden, epsilon)
	result.postFeedForward = hipOracleAddRows(result.attentionResidual, postFFNorm, 1)
	result.finalHidden = result.postFeedForward
	if len(multiplier) == 0 {
		result.postFeedForward = hipOracleScale(result.postFeedForward, layer.effectiveLayerScalar())
		result.finalHidden = result.postFeedForward
		return result
	}
	inputGate := hipOracleReferenceProjectionRows(t, driver, layer.PerLayerInput.InputGate, result.postFeedForward, tokenCount)
	for tok := 0; tok < tokenCount; tok++ {
		start := tok * layer.PerLayerInput.InputGate.Rows
		gelu, err := hipGemma4Q4HostGELU(inputGate[start : start+layer.PerLayerInput.InputGate.Rows])
		core.RequireNoError(t, err)
		for i := range gelu {
			inputGate[start+i] = gelu[i] * multiplier[start+i]
		}
	}
	perLayerProjection := hipOracleReferenceProjectionRows(t, driver, layer.PerLayerInput.Projection, inputGate, tokenCount)
	perLayerNorm := hipOracleReferenceNormRows(t, perLayerProjection, hipOracleReadNormWeight(t, driver, layer.PerLayerInput.PostInputNorm), tokenCount, hidden, epsilon)
	result.finalHidden = hipOracleScale(hipOracleAddRows(result.postFeedForward, perLayerNorm, 1), layer.effectiveLayerScalar())
	return result
}

func hipOracleReferenceProjectionRows(t *testing.T, driver nativeHIPDriver, projection hipMLXQ4DeviceWeightConfig, input []float32, tokenCount int) []float32 {
	t.Helper()
	weights := hipOracleReadUint32(t, driver, projection.WeightPointer, projection.WeightBytes)
	scales := hipOracleReadUint16(t, driver, projection.ScalePointer, projection.ScaleBytes)
	biases := hipOracleReadUint16(t, driver, projection.BiasPointer, projection.BiasBytes)
	bits := projection.Bits
	if bits == 0 {
		bits = hipMLXQ4ProjectionBits
	}
	out := make([]float32, tokenCount*projection.Rows)
	for row := 0; row < tokenCount; row++ {
		projected, err := hipReferenceMLXAffineProjection(input[row*projection.Cols:(row+1)*projection.Cols], weights, scales, biases, projection.Rows, projection.Cols, projection.GroupSize, bits)
		core.RequireNoError(t, err)
		copy(out[row*projection.Rows:], projected)
	}
	return out
}

func hipOracleReferenceNormRows(t *testing.T, input, weight []float32, rows, width int, epsilon float32) []float32 {
	t.Helper()
	out := make([]float32, len(input))
	for row := 0; row < rows; row++ {
		normed, err := hipReferenceRMSNorm(input[row*width:(row+1)*width], weight, epsilon)
		core.RequireNoError(t, err)
		copy(out[row*width:], normed)
	}
	return out
}

func hipOracleReferenceHeadNormRows(t *testing.T, input, weight []float32, tokens, heads, width int, epsilon float32) []float32 {
	t.Helper()
	return hipOracleReferenceNormRows(t, input, weight, tokens*heads, width, epsilon)
}

func hipOracleAddRows(a, b []float32, scale float32) []float32 {
	out := make([]float32, len(a))
	for i := range out {
		out[i] = a[i] + b[i]*scale
	}
	return out
}

func hipOracleScale(input []float32, scale float32) []float32 {
	out := make([]float32, len(input))
	for i := range out {
		out[i] = input[i] * scale
	}
	return out
}

func (r *hipOracleReport) rmsOnly(name string, hip []float32, width int) {
	r.rows = append(r.rows, fmt.Sprintf("  %-16s   (input)   hipRMS=%.4f", name, hipOracleRMS(hip)))
}

func (r *hipOracleReport) note(name, msg string) {
	r.rows = append(r.rows, fmt.Sprintf("  %-16s   %s", name, msg))
}

func (r *hipOracleReport) diff(name string, ref, hip []float32, width int) {
	r.diffTol(name, ref, hip, r.tolRatio)
}

// diffInfo reports a diff row without ever failing the test (for inherently
// lossy comparisons such as quantised device-KV, whose loss coherent models
// tolerate — it localises quant cost, it is not a correctness gate).
func (r *hipOracleReport) diffInfo(name string, ref, hip []float32) {
	maxAbs, meanAbs := hipOracleMaxMeanDiff(ref, hip)
	refRMS := hipOracleRMS(ref)
	ratio := float32(0)
	if refRMS > 0 {
		ratio = maxAbs / refRMS
	}
	r.rows = append(r.rows, fmt.Sprintf("  %-16s   maxAbs=%.5f meanAbs=%.5f  refRMS=%.4f  max/refRMS=%.4f  info", name, maxAbs, meanAbs, refRMS, ratio))
}

func (r *hipOracleReport) diffTol(name string, ref, hip []float32, tol float32) {
	if len(ref) != len(hip) {
		r.rows = append(r.rows, fmt.Sprintf("  %-16s   LENGTH MISMATCH ref=%d hip=%d", name, len(ref), len(hip)))
		r.failed = append(r.failed, name)
		return
	}
	maxAbs, meanAbs := hipOracleMaxMeanDiff(ref, hip)
	refRMS := hipOracleRMS(ref)
	hipRMS := hipOracleRMS(hip)
	ratio := float32(0)
	if refRMS > 0 {
		ratio = maxAbs / refRMS
	}
	status := "ok"
	if ratio > tol {
		status = "FAIL"
		r.failed = append(r.failed, name)
	}
	r.rows = append(r.rows, fmt.Sprintf("  %-16s   maxAbs=%.5f meanAbs=%.5f  refRMS=%.4f hipRMS=%.4f  max/refRMS=%.4f  %s",
		name, maxAbs, meanAbs, refRMS, hipRMS, ratio, status))
}

func (r *hipOracleReport) projection(name string, t *testing.T, driver nativeHIPDriver, proj hipMLXQ4DeviceWeightConfig, input, hipOut []float32) {
	weights := hipOracleReadUint32(t, driver, proj.WeightPointer, proj.WeightBytes)
	scales := hipOracleReadUint16(t, driver, proj.ScalePointer, proj.ScaleBytes)
	biases := hipOracleReadUint16(t, driver, proj.BiasPointer, proj.BiasBytes)
	bits := proj.Bits
	if bits == 0 {
		bits = hipMLXQ4ProjectionBits
	}
	ref, err := hipReferenceMLXAffineProjection(input, weights, scales, biases, proj.Rows, proj.Cols, proj.GroupSize, bits)
	if err != nil {
		r.rows = append(r.rows, fmt.Sprintf("  %-16s   reference error: %v", name, err))
		r.failed = append(r.failed, name)
		return
	}
	r.diff(name, ref, hipOut, proj.Rows)
}

func (r *hipOracleReport) finish() {
	r.t.Logf("=== #52 layer oracle op-diff table ===\n%s", strings.Join(r.rows, "\n"))
	if len(r.failed) > 0 {
		r.t.Errorf("FIRST DIVERGING OP(s): %s (tol=%.3f of refRMS) — see table above", strings.Join(r.failed, ", "), r.tolRatio)
	}
}

// rocmOracleKVHeadForQuery mirrors rocm_attention_kv_head_for_query in the kernel.
func rocmOracleKVHeadForQuery(queryHead, queryHeads, kvHeads int) int {
	if kvHeads <= 1 || queryHeads <= kvHeads {
		if kvHeads <= 1 {
			return 0
		}
		return queryHead
	}
	group := queryHeads / kvHeads
	if group == 0 {
		return 0
	}
	kv := queryHead / group
	if kv >= kvHeads {
		return kvHeads - 1
	}
	return kv
}

func hipOracleNormRoPEHeads(t *testing.T, in []float32, tokenCount, heads, headDim int, weight []float32, epsilon float32, base float64, freqDim, rotary int, freqScale float64) []float32 {
	t.Helper()
	rows := heads * headDim
	out := make([]float32, tokenCount*rows)
	for tok := 0; tok < tokenCount; tok++ {
		for h := 0; h < heads; h++ {
			base0 := tok*rows + h*headDim
			normed, err := hipReferenceRMSNorm(in[base0:base0+headDim], weight, epsilon)
			core.RequireNoError(t, err)
			rotated, err := hipReferenceRoPENeoXWithFrequencyDimScale(normed, tok, base, freqDim, rotary, freqScale)
			core.RequireNoError(t, err)
			copy(out[base0:], rotated)
		}
	}
	return out
}

func hipOracleReadF32(t *testing.T, buf *hipDeviceByteBuffer, count int) []float32 {
	t.Helper()
	values, err := hipReadFloat32DeviceOutput(buf, "rocm.hip.Oracle", "oracle f32 read", count)
	core.RequireNoError(t, err)
	return values
}

func hipOracleReadNormWeight(t *testing.T, driver nativeHIPDriver, cfg hipRMSNormDeviceWeightConfig) []float32 {
	t.Helper()
	if cfg.Flags&hipRMSNormLaunchFlagAddUnitWeight != 0 {
		t.Fatalf("oracle expects raw Gemma4 norm weights (no unit-add flag)")
	}
	payload := make([]byte, cfg.WeightBytes)
	core.RequireNoError(t, driver.CopyDeviceToHost(cfg.WeightPointer, payload))
	out := make([]float32, cfg.Count)
	switch cfg.WeightEncoding {
	case hipRMSNormWeightEncodingBF16:
		for i := 0; i < cfg.Count; i++ {
			out[i] = hipBFloat16ToFloat32(uint16(payload[i*2]) | uint16(payload[i*2+1])<<8)
		}
	case hipRMSNormWeightEncodingF32:
		for i := 0; i < cfg.Count; i++ {
			bits := uint32(payload[i*4]) | uint32(payload[i*4+1])<<8 | uint32(payload[i*4+2])<<16 | uint32(payload[i*4+3])<<24
			out[i] = math.Float32frombits(bits)
		}
	default:
		t.Fatalf("unexpected norm weight encoding %d", cfg.WeightEncoding)
	}
	return out
}

func hipOracleReadUint32(t *testing.T, driver nativeHIPDriver, pointer nativeDevicePointer, byteCount uint64) []uint32 {
	t.Helper()
	payload := make([]byte, byteCount)
	core.RequireNoError(t, driver.CopyDeviceToHost(pointer, payload))
	out := make([]uint32, byteCount/4)
	for i := range out {
		out[i] = uint32(payload[i*4]) | uint32(payload[i*4+1])<<8 | uint32(payload[i*4+2])<<16 | uint32(payload[i*4+3])<<24
	}
	return out
}

func hipOracleReadUint16(t *testing.T, driver nativeHIPDriver, pointer nativeDevicePointer, byteCount uint64) []uint16 {
	t.Helper()
	payload := make([]byte, byteCount)
	core.RequireNoError(t, driver.CopyDeviceToHost(pointer, payload))
	out := make([]uint16, byteCount/2)
	for i := range out {
		out[i] = uint16(payload[i*2]) | uint16(payload[i*2+1])<<8
	}
	return out
}

func hipOracleMaxMeanDiff(a, b []float32) (maxAbs, meanAbs float32) {
	var sum float64
	for i := range a {
		d := float64(a[i] - b[i])
		if d < 0 {
			d = -d
		}
		if float32(d) > maxAbs {
			maxAbs = float32(d)
		}
		sum += d
	}
	if len(a) > 0 {
		meanAbs = float32(sum / float64(len(a)))
	}
	return maxAbs, meanAbs
}

func hipOracleRMS(a []float32) float32 {
	var sum float64
	for _, v := range a {
		sum += float64(v) * float64(v)
	}
	if len(a) == 0 {
		return 0
	}
	return float32(math.Sqrt(sum / float64(len(a))))
}

func hipOracleModelPath() string {
	for _, key := range []string{"GO_ROCM_ORACLE_MODEL_PATH", "GO_ROCM_PRODUCTION_MODEL_PATH", "GO_ROCM_MODEL_PATH"} {
		if v := strings.TrimSpace(os.Getenv(key)); v != "" {
			return v
		}
	}
	return ""
}

func hipOracleEnvInt(key string, def int) int {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func hipOracleEnvFloat(key string, def float64) float64 {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		if n, err := strconv.ParseFloat(v, 64); err == nil {
			return n
		}
	}
	return def
}

// TestHIPGemma4Q4IncrementalDecodeOracle is the #52 round-3 STATE-CARRY oracle.
// Round 2 proved every op on the batched-prefill path is float-clean and that
// prefill yields a correct first token; the 12B still garbles between decode
// steps. This oracle isolates the carry: it greedy-decodes N tokens through the
// real device-KV decode machinery (a 1-token prefill-with-prior per step, which
// is exactly what production batched decode appends with) and, after each step,
// diffs the incremental next-token logits against a FRESH full batched prefill
// of the identical prefix (prompt + generated[0..k-1]) — the path round 2
// proved op-correct. Both prefixes are token-identical, so the ONLY variable is
// carried-vs-fresh KV state; the first step whose argmax (or logits) diverges is
// the bug's address.
//
// When a divergence is found (or GO_ROCM_ORACLE_LOCALISE_STEP forces one), it
// re-derives the incremental and fresh device state at that step and dumps a
// per-layer KV-row divergence table, labelled sliding vs full, comparing both
// the newly appended row and all retained rows. The 5:1 interleave gives the
// signature: a sliding-window append/RoPE fault lights up the sliding layers, a
// full-attention/k_eq_v fault the 8 full layers (5,11,17,23,29,35,41,47 on the
// 12B), a global position fault every layer; new-row-only vs all-rows delta
// separates an append/RoPE-at-position bug from a retention/window bug.
//
// Calibrate on E2B/E4B first: they are coherent on hip and MUST match at every
// step. If they diverge too, the oracle harness is wrong, not the engine.
//
// FINDING (2026-07-12, this oracle): the state-carry theory is FALSIFIED. On the
// exact production garble trajectory (12B fp16 -> token 45518 "thought" on loop),
// incremental == recompute at every step (max/refRMS <= 0.006, argmax identical),
// and so does coherent E2B. The embedding, all 48 per-op layer transforms, and
// the final norm + tied LM head are each float-reference-clean; yet the 12B's
// full-forward top logit is a control token (107) and its residual-stream RMS
// collapses (peak ~4.4 @ L23 -> 0.22 @ L48). The bug is in the DENSE chained
// forward, not the KV carry. CRUCIAL: E2B/E4B are Gemma-3n (per-layer inputs +
// cross-layer KV sharing) while the 12B is standard DENSE (perLayerInputLayers=0,
// kvSharedLayers=0) — a structurally DIFFERENT path. So E2B coherence validates
// the ORACLE METHOD but NOT the dense forward; the 12B is the only model here on
// the dense path, and it is the one that garbles. The reference-clean components
// prove the corruption is cross-layer accumulation, invisible to per-op isolation.
// Strongest next instrument: a chained full-float reference forward (chain round
// 2's per-op references across all 48 layers) to pin the exact dense layer.
//
// Env: GO_ROCM_RUN_HIP_TESTS=1, GO_ROCM_KERNEL_HSACO, model path (as round 2).
// Optional: GO_ROCM_ORACLE_KV_MODE (default fp16 — near-exact, so any divergence
// is a genuine geometry/stride/mapping fault, not quant noise; round 2 showed
// fp16 also garbles), GO_ROCM_ORACLE_PROMPT_LEN (8), GO_ROCM_ORACLE_DECODE_STEPS
// (6), GO_ROCM_ORACLE_TOL (0.05 of refRMS — argmax is the hard gate),
// GO_ROCM_ORACLE_LOCALISE_STEP (-1 = auto: first divergent step),
// GO_ROCM_ORACLE_FORCE_BATCHED_PROJ (0 — set 1 to drive the incremental step
// through production's exact ForceBatchedProjection decode kernel).
func TestHIPGemma4Q4IncrementalDecodeOracle(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_HIP_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_HIP_TESTS=1 to run ROCm hardware oracle tests")
	}
	if os.Getenv("GO_ROCM_KERNEL_HSACO") == "" {
		t.Skip("set GO_ROCM_KERNEL_HSACO to a compiled gfx1101 kernels/rocm_kernels.hip HSACO")
	}
	modelPath := hipOracleModelPath()
	if modelPath == "" {
		t.Skip("set GO_ROCM_ORACLE_MODEL_PATH / GO_ROCM_PRODUCTION_MODEL_PATH to a local Gemma4 MLX-affine pack")
	}
	mode := strings.TrimSpace(os.Getenv("GO_ROCM_ORACLE_KV_MODE"))
	if mode == "" {
		mode = rocmKVCacheModeFP16
	}
	// A real prompt reproduces the production garble ("Sky"/"thought" then
	// collapse); synthetic gibberish merely repeats and both paths agree, which
	// says nothing. Default to the production repro prompt; set "synthetic" to
	// fall back to the round-2 deterministic token generator.
	promptText := os.Getenv("GO_ROCM_ORACLE_PROMPT")
	if promptText == "" {
		promptText = "why the sky is blue"
	}
	synthLen := hipOracleEnvInt("GO_ROCM_ORACLE_PROMPT_LEN", 8)
	steps := hipOracleEnvInt("GO_ROCM_ORACLE_DECODE_STEPS", 6)
	tolRatio := float32(hipOracleEnvFloat("GO_ROCM_ORACLE_TOL", 0.05))
	localiseStep := hipOracleEnvInt("GO_ROCM_ORACLE_LOCALISE_STEP", -1)
	forceBatchedProj := os.Getenv("GO_ROCM_ORACLE_FORCE_BATCHED_PROJ") == "1"
	const epsilon = float32(1e-6)

	runtime := newSystemNativeRuntime()
	if !runtime.Available() {
		t.Fatalf("native ROCm runtime is not available")
	}
	model, err := resultValue[inference.TextModel](newROCmBackendWithRuntime(runtime).LoadModel(modelPath, inference.WithContextLen(4096)))
	if err != nil {
		t.Fatalf("LoadModel(%q): %v", modelPath, err)
	}
	defer model.Close()
	rocmLoaded, ok := model.(*rocmModel)
	if !ok {
		t.Fatalf("LoadModel returned %T, want *rocmModel", model)
	}
	loaded, ok := rocmLoaded.native.(*hipLoadedModel)
	if !ok {
		t.Fatalf("native is %T, want *hipLoadedModel", rocmLoaded.native)
	}
	driver := loaded.driver
	ctx := context.Background()

	layerCount := loaded.modelInfo.NumLayers
	if layerCount <= 0 {
		t.Fatalf("loaded model reports %d layers", layerCount)
	}
	cfg, err := loaded.loadedGemma4Q4ForwardConfig(layerCount)
	if err != nil {
		t.Fatalf("loadedGemma4Q4ForwardConfig(%d): %v", layerCount, err)
	}
	if !hipGemma4Q4CanUseBatchedGeneratePrefill(cfg) {
		t.Skipf("model at %s cannot use batched generate prefill (MoE/per-layer-input); oracle targets the dense Q4 carry", modelPath)
	}
	hidden := cfg.Layers[0].HiddenSize
	vocab := cfg.Layers[0].VocabSize
	last := cfg.Layers[len(cfg.Layers)-1]

	// Match production's greedy trajectory: production suppresses control tokens
	// before argmax (without it, raw greedy picks low special tokens and walks a
	// different, non-garbling path — E2B then looks broken though production
	// decodes it coherently). Both compared paths share the same suppression, so
	// the incremental-vs-recompute diff stays apples-to-apples.
	suppress := map[int32]bool{}
	for _, id := range hipGemma4Q4GenerationSuppressTokenIDs(loaded, nil) {
		suppress[id] = true
	}

	fullLayers := 0
	perLayerInputLayers := 0
	for _, l := range cfg.Layers {
		if hipIncrOracleLayerKind(l) == "full" {
			fullLayers++
		}
		if l.PerLayerInput.hasLayerApply() {
			perLayerInputLayers++
		}
	}
	scalars := make([]float32, len(cfg.Layers))
	sMin, sMax := float32(math.Inf(1)), float32(math.Inf(-1))
	for i, l := range cfg.Layers {
		s := l.effectiveLayerScalar()
		scalars[i] = s
		if s < sMin {
			sMin = s
		}
		if s > sMax {
			sMax = s
		}
	}
	t.Logf("INCR-ORACLE structure: perLayerInputLayers=%d globalPrecompute=%v layer_scalar min=%.5f max=%.5f\n  layer_scalars=%v",
		perLayerInputLayers, cfg.Layers[0].PerLayerInput.hasGlobalPrecompute(), sMin, sMax, scalars)
	var prompt []int32
	if promptText == "synthetic" {
		prompt = make([]int32, synthLen)
		for i := range prompt {
			prompt[i] = int32((i*2654435761 + 12345) % vocab)
			if prompt[i] < 0 {
				prompt[i] += int32(vocab)
			}
		}
	} else {
		ids, matched, terr := hipGemma4Q4PromptTokenIDs(promptText, loaded)
		if terr != nil {
			t.Fatalf("tokenize prompt %q: %v", promptText, terr)
		}
		if !matched || len(ids) == 0 {
			t.Fatalf("prompt %q did not tokenize to Gemma4 IDs (matched=%v len=%d)", promptText, matched, len(ids))
		}
		prompt = ids
	}
	promptLen := len(prompt)
	t.Logf("INCR-ORACLE model=%s layers=%d (full=%d sliding=%d) hidden=%d vocab=%d kvMode=%s prompt=%q promptLen=%d steps=%d forceBatchedProj=%v kvSharedLayers=%d sharedKVSources=%v",
		modelPath, layerCount, fullLayers, layerCount-fullLayers, hidden, vocab, mode, promptText, promptLen, steps, forceBatchedProj, cfg.KVSharedLayers, cfg.SharedKVSources)

	// --- Prefill the prompt: builds the initial carried device state and the
	// logits that predict the first generated token (round 2: prefill-correct). ---
	prefillLogits, prefillGreedy, deviceState, err := hipIncrOraclePrefill(ctx, t, driver, cfg, last, prompt, mode, epsilon, suppress, true)
	if err != nil {
		t.Fatalf("prompt prefill: %v", err)
	}
	defer func() {
		if deviceState != nil {
			_ = deviceState.Close()
		}
	}()

	generated := []int32{int32(prefillGreedy)}
	incrLogits := [][]float32{prefillLogits}
	incrGreedy := []int{prefillGreedy}
	position := promptLen

	for k := 1; k <= steps; k++ {
		tok := generated[k-1]
		lg, gtok, next, stepErr := hipIncrOracleStep(ctx, t, driver, cfg, last, mode, forceBatchedProj, deviceState, tok, position, epsilon, suppress)
		if stepErr != nil {
			t.Fatalf("incremental decode step %d (token %d @ pos %d): %v", k, tok, position, stepErr)
		}
		hipReleaseClosedGemma4Q4DeviceDecodeState(deviceState)
		deviceState = next
		incrLogits = append(incrLogits, lg)
		incrGreedy = append(incrGreedy, gtok)
		generated = append(generated, int32(gtok))
		position++
	}

	// --- Per-step incremental-vs-recompute divergence table. ---
	rows := []string{"  step  pos   argmax(incr/recompute)  match  max|Δlogit|  refRMS   max/refRMS"}
	firstDivergent := -1
	for k := 1; k <= steps; k++ {
		seq := make([]int32, 0, promptLen+k)
		seq = append(seq, prompt...)
		seq = append(seq, generated[:k]...)
		recLogits, recGreedy, _, recErr := hipIncrOraclePrefill(ctx, t, driver, cfg, last, seq, mode, epsilon, suppress, false)
		if recErr != nil {
			t.Fatalf("recompute prefill (len %d) for step %d: %v", len(seq), k, recErr)
		}
		maxAbs, _ := hipOracleMaxMeanDiff(recLogits, incrLogits[k])
		refRMS := hipOracleRMS(recLogits)
		ratio := float32(0)
		if refRMS > 0 {
			ratio = maxAbs / refRMS
		}
		argMatch := recGreedy == incrGreedy[k]
		diverged := !argMatch || ratio > tolRatio
		if diverged && firstDivergent < 0 {
			firstDivergent = k
		}
		status := "ok"
		if !argMatch {
			status = "ARGMAX"
		} else if ratio > tolRatio {
			status = "logit"
		}
		rows = append(rows, fmt.Sprintf("  %-4d  %-4d  %-8d / %-8d       %-5s  %-11.5f  %-7.4f  %-9.4f %s",
			k, promptLen+k-1, incrGreedy[k], recGreedy, boolLabel(argMatch), maxAbs, refRMS, ratio, status))
	}
	t.Logf("=== #52 incremental-vs-recompute divergence table (kvMode=%s) ===\n%s", mode, strings.Join(rows, "\n"))

	// --- Surface (c): the retained final stage (final RMSNorm -> LM-head ->
	// softcap) is SHARED by incremental and recompute, so an incr==recompute
	// result cannot exonerate it. Round 2 never checked it. Diff the final norm
	// and the tied LM-head projection against round 2's proven float references
	// at the last prompt row (which predicts the first generated token). ---
	if os.Getenv("GO_ROCM_ORACLE_SKIP_FINAL_STAGE") != "1" {
		hipIncrOracleCheckFinalStage(ctx, t, driver, cfg, last, prompt, mode, epsilon, tolRatio)
	}

	// --- Logit lens: run the prompt through the first L layers only, then apply
	// the real final head. Since embedding, every per-op layer transform, and the
	// final stage are each reference-clean, the corruption lives in the CHAINED
	// residual stream; this pins the depth at which the top token turns into a
	// suppressed control token (E2B stays a real word). ---
	if os.Getenv("GO_ROCM_ORACLE_SKIP_LOGIT_LENS") != "1" {
		hipIncrOracleLogitLens(ctx, t, driver, cfg, last, prompt, mode, epsilon, suppress)
	}

	// --- Localise the pinned step to a KV surface. ---
	target := localiseStep
	if target < 0 {
		target = firstDivergent
	}
	if target >= 1 && target <= steps {
		hipIncrOracleLocaliseKV(ctx, t, driver, cfg, mode, forceBatchedProj, prompt, generated, target, promptLen, epsilon, suppress)
	} else if firstDivergent < 0 {
		t.Logf("INCR-ORACLE: no incremental/recompute divergence across %d steps — the state-carry theory is FALSIFIED for this config (kvMode=%s, promptLen=%d). Widen steps/prompt or switch KV mode; report the strongest next hypothesis.", steps, mode, promptLen)
	}

	if firstDivergent >= 1 {
		t.Errorf("INCR-ORACLE: incremental decode diverges from recompute at step %d (pos %d) — carried KV state != fresh prefill; see the divergence + per-layer KV tables above", firstDivergent, promptLen+firstDivergent-1)
	}
}

// hipIncrOracleLogitLens applies the real final head to the residual stream after
// each depth L (1..N layers), reporting the last-prompt-row argmax, whether that
// raw top token is a suppressed control token, and the top/2nd-logit margin. The
// depth at which a real-word top token turns into a control token localises where
// the chained residual stream corrupts — the surface per-op isolation can't see.
func hipIncrOracleLogitLens(ctx context.Context, t *testing.T, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, last hipGemma4Q4Layer0Config, prompt []int32, mode string, epsilon float32, suppress map[int32]bool) {
	t.Helper()
	ec := defaultHIPGemma4Q4EngineConfig()
	ec.DeviceKVMode = mode
	hidden := cfg.Layers[0].HiddenSize
	tokenCount := len(prompt)
	rows := []string{"  L     rawArgmax  ctrl?  suppArgmax  top1     top2     margin   hiddenRMS"}
	flip := -1
	for L := 1; L <= len(cfg.Layers); L++ {
		tcfg := cfg
		tcfg.Layers = cfg.Layers[:L]
		tcfg.KVSharedLayers = 0
		tcfg.SharedKVSources = nil
		forward, err := hipRunGemma4Q4PrefillForwardBatchWithPriorDescriptorWorkspaceOutputRowWithEngineConfig(ctx, driver, tcfg, prompt, 0, epsilon, mode, nil, nil, nil, nil, -1, nil, nil, ec)
		if err != nil {
			rows = append(rows, fmt.Sprintf("  %-4d  (prefill err: %v)", L, err))
			continue
		}
		all, rErr := hipReadFloat32DeviceOutput(forward.FinalHidden, "rocm.hip.IncrOracle", "lens hidden", tokenCount*hidden)
		_ = forward.Close()
		if rErr != nil {
			rows = append(rows, fmt.Sprintf("  %-4d  (read err: %v)", L, rErr))
			continue
		}
		rowH := all[(tokenCount-1)*hidden : tokenCount*hidden]
		hiddenRMS := hipOracleRMS(rowH)
		logits, lErr := hipIncrOracleFinalLogits(ctx, driver, last, rowH, epsilon)
		if lErr != nil {
			rows = append(rows, fmt.Sprintf("  %-4d  (final-stage err: %v)", L, lErr))
			continue
		}
		rawArg := hipIncrOracleArgmax(logits, nil)
		suppArg := hipIncrOracleArgmax(logits, suppress)
		ctrl := suppress[int32(rawArg)]
		top1, top2 := hipIncrOracleTop2(logits)
		ctrlMark := "no"
		if ctrl {
			ctrlMark = "YES"
			if flip < 0 {
				flip = L
			}
		} else {
			flip = -1 // reset: only a sustained flip near the output matters
		}
		rows = append(rows, fmt.Sprintf("  %-4d  %-9d  %-5s  %-10d  %-8.3f %-8.3f %-8.3f %-8.3f",
			L, rawArg, ctrlMark, suppArg, top1, top2, top1-top2, hiddenRMS))
	}
	t.Logf("=== #52 logit-lens per-depth argmax (prompt last row, kvMode=%s) ===\n%s\n  earliest sustained control-token depth (to output) = %d of %d layers",
		mode, strings.Join(rows, "\n"), flip, len(cfg.Layers))
}

// hipIncrOracleTop2 returns the top and second logit values.
func hipIncrOracleTop2(logits []float32) (top1, top2 float32) {
	top1 = float32(math.Inf(-1))
	top2 = float32(math.Inf(-1))
	for _, v := range logits {
		if v > top1 {
			top2 = top1
			top1 = v
		} else if v > top2 {
			top2 = v
		}
	}
	return top1, top2
}

// hipIncrOracleCheckFinalStage validates the shared final stage against round 2's
// proven float references: the final RMSNorm and the (tied) LM-head projection.
// It prefills the prompt, takes the last row's post-stack hidden, then (1) diffs
// hip's final norm vs hipReferenceRMSNorm and (2) diffs hip's LM-head projection
// vs hipReferenceMLXAffineProjection fed the SAME normed input (so only the
// projection differs). A divergence here is the 12B garble's address on a surface
// incremental-vs-recompute cannot see, because both paths share this stage.
func hipIncrOracleCheckFinalStage(ctx context.Context, t *testing.T, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, last hipGemma4Q4Layer0Config, prompt []int32, mode string, epsilon float32, tolRatio float32) {
	t.Helper()
	ec := defaultHIPGemma4Q4EngineConfig()
	ec.DeviceKVMode = mode
	forward, err := hipRunGemma4Q4PrefillForwardBatchWithPriorDescriptorWorkspaceOutputRowWithEngineConfig(ctx, driver, cfg, prompt, 0, epsilon, mode, nil, nil, nil, nil, -1, nil, nil, ec)
	if err != nil {
		t.Logf("final-stage: prompt prefill: %v", err)
		return
	}
	defer forward.Close()
	tokenCount := len(prompt)
	hidden := cfg.Layers[0].HiddenSize
	all, err := hipReadFloat32DeviceOutput(forward.FinalHidden, "rocm.hip.IncrOracle", "final-stage hidden", tokenCount*hidden)
	if err != nil {
		t.Logf("final-stage: read hidden: %v", err)
		return
	}
	row := all[(tokenCount-1)*hidden : tokenCount*hidden]

	// (0) Embedding: hip's scaled lookup vs float reference. Round 2 only prints
	// the embedding RMS (never reference-checks it), yet it is the ONE forward
	// input every per-op check is fed and so cannot validate: a wrong embedding
	// (bad dequant, wrong row, wrong sqrt(hidden) scale) makes every downstream
	// op "pass" while the residual stream — and thus the logits — are wrong. The
	// check is scale-invariant: direction (normalised) isolates the lookup, the
	// implied scale isolates the sqrt(hidden) embedding scale.
	emb := cfg.Layers[0].Embedding
	expScale := cfg.Layers[0].EmbeddingScale
	if expScale == 0 {
		expScale = hipGemma4Q4EmbeddingScale(hidden)
	}
	embBits := emb.QuantBits
	if embBits == 0 {
		embBits = hipMLXQ4ProjectionBits
	}
	embW := hipOracleReadUint32(t, driver, emb.EmbeddingPointer, emb.EmbeddingBytes)
	embS := hipOracleReadUint16(t, driver, emb.ScalePointer, emb.ScaleBytes)
	embB := hipOracleReadUint16(t, driver, emb.BiasPointer, emb.BiasBytes)
	embRefUnscaled, embErr := hipReferenceMLXAffineEmbeddingLookup(embW, embS, embB, emb.VocabSize, emb.HiddenSize, emb.GroupSize, prompt, embBits)
	if embErr != nil {
		t.Logf("final-stage: ref embedding lookup: %v", embErr)
	} else {
		embHip, hErr := hipReadFloat32DeviceOutput(forward.Embedding, "rocm.hip.IncrOracle", "scaled embedding", tokenCount*hidden)
		if hErr != nil {
			t.Logf("final-stage: read hip embedding: %v", hErr)
		} else {
			refRow := embRefUnscaled[:hidden]
			hipRow := embHip[:hidden]
			refRMS := hipOracleRMS(refRow)
			hipRMS := hipOracleRMS(hipRow)
			impliedScale := float32(0)
			if refRMS > 0 {
				impliedScale = hipRMS / refRMS
			}
			var dirMax float32
			if refRMS > 0 && hipRMS > 0 {
				for i := 0; i < hidden; i++ {
					d := refRow[i]/refRMS - hipRow[i]/hipRMS
					if d < 0 {
						d = -d
					}
					if d > dirMax {
						dirMax = d
					}
				}
			}
			embStatus := "ok"
			if dirMax > tolRatio || impliedScale < expScale*0.98 || impliedScale > expScale*1.02 {
				embStatus = "FAIL"
			}
			t.Logf("=== #52 embedding float-reference check (token0=%d, hidden=%d, group=%d bits=%d) ===\n"+
				"  embedding    dirMax(normalised)=%.5f  impliedScale=%.4f expScale=%.4f(sqrt%d)  %s",
				prompt[0], hidden, emb.GroupSize, embBits, dirMax, impliedScale, expScale, hidden, embStatus)
			if embStatus == "FAIL" {
				t.Errorf("EMBEDDING DIVERGES from float reference (dirMax=%.5f impliedScale=%.4f vs expScale=%.4f) — the 12B garble is in the embedding lookup/scale, upstream of every per-op-clean layer", dirMax, impliedScale, expScale)
			}
		}
	}

	// (1) Final RMSNorm: hip kernel vs float reference on the identical hidden row.
	finalNormCfg := last.FinalNorm
	finalNormCfg.Epsilon = epsilon
	normW := hipOracleReadNormWeight(t, driver, last.FinalNorm)
	normHip, err := hipRunRMSNormKernelWithDeviceWeightConfig(ctx, driver, row, finalNormCfg)
	if err != nil {
		t.Logf("final-stage: hip final norm: %v", err)
		return
	}
	normRef, err := hipReferenceRMSNorm(row, normW, epsilon)
	if err != nil {
		t.Logf("final-stage: ref final norm: %v", err)
		return
	}
	normMax, _ := hipOracleMaxMeanDiff(normRef, normHip)
	normRMS := hipOracleRMS(normRef)
	normRatio := float32(0)
	if normRMS > 0 {
		normRatio = normMax / normRMS
	}

	// (2) Tied LM-head projection: hip kernel vs float reference, both fed hip's
	// own normed row, so the ONLY variable is the projection kernel/weights.
	lmW := hipOracleReadUint32(t, driver, last.LMHeadProjection.WeightPointer, last.LMHeadProjection.WeightBytes)
	lmS := hipOracleReadUint16(t, driver, last.LMHeadProjection.ScalePointer, last.LMHeadProjection.ScaleBytes)
	lmB := hipOracleReadUint16(t, driver, last.LMHeadProjection.BiasPointer, last.LMHeadProjection.BiasBytes)
	bits := last.LMHeadProjection.Bits
	if bits == 0 {
		bits = hipMLXQ4ProjectionBits
	}
	logitsHip, err := hipRunMLXQ4ProjectionKernelWithDeviceWeightConfig(ctx, driver, normHip, last.LMHeadProjection)
	if err != nil {
		t.Logf("final-stage: hip LM-head: %v", err)
		return
	}
	logitsRef, err := hipReferenceMLXAffineProjection(normHip, lmW, lmS, lmB, last.LMHeadProjection.Rows, last.LMHeadProjection.Cols, last.LMHeadProjection.GroupSize, bits)
	if err != nil {
		t.Logf("final-stage: ref LM-head: %v", err)
		return
	}
	lmMax, lmMean := hipOracleMaxMeanDiff(logitsRef, logitsHip)
	lmRMS := hipOracleRMS(logitsRef)
	lmRatio := float32(0)
	if lmRMS > 0 {
		lmRatio = lmMax / lmRMS
	}
	argHip := hipIncrOracleArgmax(logitsHip, nil)
	argRef := hipIncrOracleArgmax(logitsRef, nil)

	normStatus := "ok"
	if normRatio > tolRatio {
		normStatus = "FAIL"
	}
	lmStatus := "ok"
	if lmRatio > tolRatio || argHip != argRef {
		lmStatus = "FAIL"
	}
	t.Logf("=== #52 final-stage float-reference check (row=%d, kvMode=%s, LM rows=%d cols=%d group=%d bits=%d) ===\n"+
		"  final_norm   maxAbs=%.5f refRMS=%.4f max/refRMS=%.5f  %s\n"+
		"  lm_head      maxAbs=%.5f meanAbs=%.5f refRMS=%.4f max/refRMS=%.5f argmax(hip/ref)=%d/%d  %s",
		tokenCount-1, mode, last.LMHeadProjection.Rows, last.LMHeadProjection.Cols, last.LMHeadProjection.GroupSize, bits,
		normMax, normRMS, normRatio, normStatus,
		lmMax, lmMean, lmRMS, lmRatio, argHip, argRef, lmStatus)
	if normStatus == "FAIL" || lmStatus == "FAIL" {
		t.Errorf("FINAL-STAGE DIVERGES from float reference (norm=%s lm_head=%s) — the 12B garble is in the shared retained final stage, not the KV carry", normStatus, lmStatus)
	}
}

// hipIncrOracleFinalLogits runs the retained final stage — final RMSNorm, LM-head
// projection, tanh softcap — on one hidden row, byte-identically to the decode
// else-branch in hipRunGemma4Q4SingleTokenForwardWithStateInternal. Sharing this
// stage across both compared paths controls surface (c): the only variable left
// between incremental and recompute logits is the hidden row each path produced.
func hipIncrOracleFinalLogits(ctx context.Context, driver nativeHIPDriver, last hipGemma4Q4Layer0Config, rowHidden []float32, epsilon float32) ([]float32, error) {
	finalNormCfg := last.FinalNorm
	finalNormCfg.Epsilon = epsilon
	finalNorm, err := hipRunRMSNormKernelWithDeviceWeightConfig(ctx, driver, rowHidden, finalNormCfg)
	if err != nil {
		return nil, err
	}
	logits, err := hipRunMLXQ4ProjectionKernelWithDeviceWeightConfig(ctx, driver, finalNorm, last.LMHeadProjection)
	if err != nil {
		return nil, err
	}
	return hipGemma4Q4SoftcapLogits(logits, last.FinalLogitSoftcap)
}

// hipIncrOraclePrefill runs a fresh full batched prefill (startPosition 0, no
// prior) over tokens and returns the last row's softcapped logits + greedy
// token. When returnState is set it also returns the finalised device KV state
// (positions 0..len(tokens)-1) for KV-surface localisation.
func hipIncrOraclePrefill(ctx context.Context, t *testing.T, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, last hipGemma4Q4Layer0Config, tokens []int32, mode string, epsilon float32, suppress map[int32]bool, returnState bool) ([]float32, int, *hipGemma4Q4DeviceDecodeState, error) {
	t.Helper()
	ec := defaultHIPGemma4Q4EngineConfig()
	ec.DeviceKVMode = mode
	forward, err := hipRunGemma4Q4PrefillForwardBatchWithPriorDescriptorWorkspaceOutputRowWithEngineConfig(ctx, driver, cfg, tokens, 0, epsilon, mode, nil, nil, nil, nil, -1, nil, nil, ec)
	if err != nil {
		return nil, 0, nil, err
	}
	tokenCount := len(tokens)
	hiddenSize := cfg.Layers[0].HiddenSize
	all, err := hipReadFloat32DeviceOutput(forward.FinalHidden, "rocm.hip.IncrOracle", "prefill final hidden", tokenCount*hiddenSize)
	if err != nil {
		_ = forward.Close()
		return nil, 0, nil, err
	}
	lastRow := all[(tokenCount-1)*hiddenSize : tokenCount*hiddenSize]
	logits, err := hipIncrOracleFinalLogits(ctx, driver, last, lastRow, epsilon)
	if err != nil {
		_ = forward.Close()
		return nil, 0, nil, err
	}
	greedy := hipIncrOracleArgmax(logits, suppress)
	if !returnState {
		return logits, greedy, nil, forward.Close()
	}
	state, stateErr := hipGemma4Q4DeviceDecodeStateFromPrefillForward(forward, mode)
	closeErr := forward.Close()
	if stateErr != nil {
		return nil, 0, nil, stateErr
	}
	if closeErr != nil {
		_ = state.Close()
		return nil, 0, nil, closeErr
	}
	return logits, greedy, state, nil
}

// hipIncrOracleStep advances the carried device KV state by one token exactly as
// production batched decode does (a 1-token prefill-with-prior; ForceBatchedProjection
// mirrors hipRunAttachedDrafterTargetAdvanceOneBatch when requested). It returns
// that step's softcapped logits + greedy token and the new device state; the
// caller releases the prior state after finalisation, as production does.
func hipIncrOracleStep(ctx context.Context, t *testing.T, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, last hipGemma4Q4Layer0Config, mode string, forceBatchedProj bool, prior *hipGemma4Q4DeviceDecodeState, tokenID int32, position int, epsilon float32, suppress map[int32]bool) ([]float32, int, *hipGemma4Q4DeviceDecodeState, error) {
	t.Helper()
	layerCount := len(cfg.Layers)
	priorLayerKV := hipGemma4Q4DeviceLayerCaches(prior, nil, layerCount)
	priorDesc, err := hipGemma4Q4DeviceLayerDescriptorTableAliases(prior, nil, layerCount)
	if err != nil {
		return nil, 0, nil, err
	}
	defer hipCloseGemma4Q4DeviceLayerDescriptorTables(priorDesc)
	ec := defaultHIPGemma4Q4EngineConfig()
	ec.DeviceKVMode = mode
	ec.ForceBatchedProjection = forceBatchedProj
	forward, err := hipRunGemma4Q4PrefillForwardBatchWithPriorDescriptorWorkspaceOutputRowWithEngineConfig(ctx, driver, cfg, []int32{tokenID}, position, epsilon, mode, priorLayerKV, priorDesc, nil, nil, -1, nil, nil, ec)
	if err != nil {
		return nil, 0, nil, err
	}
	hiddenSize := cfg.Layers[0].HiddenSize
	row, err := hipReadFloat32DeviceOutput(forward.FinalHidden, "rocm.hip.IncrOracle", "step final hidden", hiddenSize)
	if err != nil {
		_ = forward.Close()
		return nil, 0, nil, err
	}
	logits, err := hipIncrOracleFinalLogits(ctx, driver, last, row, epsilon)
	if err != nil {
		_ = forward.Close()
		return nil, 0, nil, err
	}
	greedy := hipIncrOracleArgmax(logits, suppress)
	next, stateErr := hipGemma4Q4DeviceDecodeStateFromPrefillForward(forward, mode)
	closeErr := forward.Close()
	if stateErr != nil {
		return nil, 0, nil, stateErr
	}
	if closeErr != nil {
		_ = next.Close()
		return nil, 0, nil, closeErr
	}
	if err := hipFinalizeGemma4Q4ForwardDeviceState(prior, next); err != nil {
		_ = next.Close()
		return nil, 0, nil, err
	}
	return logits, greedy, next, nil
}

// hipIncrOracleLocaliseKV re-derives the incremental carried state and a fresh
// recompute state at the pinned step (identical token prefix) and diffs their
// per-layer device KV, at the newly appended row and across all retained rows,
// so the first diverging layer + its type name the corrupted surface.
func hipIncrOracleLocaliseKV(ctx context.Context, t *testing.T, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, mode string, forceBatchedProj bool, prompt, generated []int32, step, promptLen int, epsilon float32, suppress map[int32]bool) {
	t.Helper()
	last := cfg.Layers[len(cfg.Layers)-1]
	newPos := promptLen + step - 1

	// Incremental carried state after feeding generated[0..step-1].
	_, _, incrState, err := hipIncrOraclePrefill(ctx, t, driver, cfg, last, prompt, mode, epsilon, suppress, true)
	if err != nil {
		t.Logf("localise: prompt prefill: %v", err)
		return
	}
	defer func() {
		if incrState != nil {
			_ = incrState.Close()
		}
	}()
	pos := promptLen
	for j := 0; j < step; j++ {
		_, _, next, stepErr := hipIncrOracleStep(ctx, t, driver, cfg, last, mode, forceBatchedProj, incrState, generated[j], pos, epsilon, suppress)
		if stepErr != nil {
			t.Logf("localise: incremental step %d: %v", j, stepErr)
			return
		}
		hipReleaseClosedGemma4Q4DeviceDecodeState(incrState)
		incrState = next
		pos++
	}

	// Fresh recompute state over the identical prefix prompt+generated[0..step-1].
	seq := make([]int32, 0, promptLen+step)
	seq = append(seq, prompt...)
	seq = append(seq, generated[:step]...)
	_, _, recState, err := hipIncrOraclePrefill(ctx, t, driver, cfg, last, seq, mode, epsilon, suppress, true)
	if err != nil {
		t.Logf("localise: recompute prefill: %v", err)
		return
	}
	defer func() {
		if recState != nil {
			_ = recState.Close()
		}
	}()

	rows := []string{fmt.Sprintf("  layer  type      window  newRow(pos=%d) maxΔkey maxΔval   allRows maxΔkey maxΔval", newPos)}
	firstDivLayer := -1
	firstDivKind := ""
	divByKind := map[string]int{}
	for L := range cfg.Layers {
		layer := cfg.Layers[L]
		kind := hipIncrOracleLayerKind(layer)
		incrCache := incrState.layerCache(L)
		recCache := recState.layerCache(L)
		if incrCache == nil || recCache == nil {
			rows = append(rows, fmt.Sprintf("  %-5d  %-8s  %-6d  (cache unavailable)", L, kind, layer.SlidingWindow))
			continue
		}
		ik, iv, iKW, iVW, iTok, ierr := hipIncrOracleDecodeCache(incrCache)
		rk, rv, rKW, rVW, rTok, rerr := hipIncrOracleDecodeCache(recCache)
		if ierr != nil || rerr != nil {
			rows = append(rows, fmt.Sprintf("  %-5d  %-8s  %-6d  (decode err incr=%v rec=%v)", L, kind, layer.SlidingWindow, ierr, rerr))
			continue
		}
		if iKW != rKW || iVW != rVW {
			rows = append(rows, fmt.Sprintf("  %-5d  %-8s  %-6d  (width mismatch key %d/%d val %d/%d)", L, kind, layer.SlidingWindow, iKW, rKW, iVW, rVW))
			continue
		}
		newKeyD := hipIncrOracleRowDelta(ik, rk, iKW, iTok, rTok, newPos)
		newValD := hipIncrOracleRowDelta(iv, rv, iVW, iTok, rTok, newPos)
		allKeyD := hipIncrOracleAllDelta(ik, rk)
		allValD := hipIncrOracleAllDelta(iv, rv)
		diverged := newKeyD > 1e-2 || newValD > 1e-2 || allKeyD > 1e-2 || allValD > 1e-2
		mark := ""
		if diverged {
			divByKind[kind]++
			mark = "  <== DIVERGES"
			if firstDivLayer < 0 {
				firstDivLayer = L
				firstDivKind = kind
			}
		}
		rows = append(rows, fmt.Sprintf("  %-5d  %-8s  %-6d  %-13.5f %-8.5f  %-13.5f %-8.5f%s",
			L, kind, layer.SlidingWindow, newKeyD, newValD, allKeyD, allValD, mark))
		_ = iTok
		_ = rTok
	}
	summary := fmt.Sprintf("first-diverging-layer=%d kind=%s  divergent-by-kind=%v", firstDivLayer, firstDivKind, divByKind)
	if firstDivLayer < 0 {
		summary = "NO per-layer KV divergence at this step — the corrupted surface is downstream of KV (final-stage/scratch) or the logit divergence is numeric-only"
	}
	t.Logf("=== #52 per-layer KV-row localisation @ step %d (pos %d, kvMode=%s) ===\n%s\n  %s", step, newPos, mode, strings.Join(rows, "\n"), summary)
}

// hipIncrOracleDecodeCache reads a device KV cache back to host and dequantises
// it to contiguous [tokens*width] key/value float slices, indexed by absolute
// token position.
func hipIncrOracleDecodeCache(cache *rocmDeviceKVCache) (keys, values []float32, keyWidth, valueWidth, tokenCount int, err error) {
	host, err := cache.hostCache()
	if err != nil {
		return nil, nil, 0, 0, 0, err
	}
	keyWidth = host.keyWidth
	valueWidth = host.valueWidth
	if keyWidth <= 0 || valueWidth <= 0 {
		return nil, nil, 0, 0, 0, core.E("rocm.hip.IncrOracle", "kv cache has zero vector width", nil)
	}
	for _, b := range host.blocks {
		end := b.tokenStart + b.tokenCount
		if end > tokenCount {
			tokenCount = end
		}
	}
	keys = make([]float32, tokenCount*keyWidth)
	values = make([]float32, tokenCount*valueWidth)
	for _, b := range host.blocks {
		if b.keyWidth != keyWidth || b.valueWidth != valueWidth {
			return nil, nil, 0, 0, 0, core.E("rocm.hip.IncrOracle", "kv block width mismatch", nil)
		}
		kd := b.key.decodeRows(b.keyWidth)
		vd := b.value.decodeRows(b.valueWidth)
		copy(keys[b.tokenStart*keyWidth:], kd)
		copy(values[b.tokenStart*valueWidth:], vd)
	}
	return keys, values, keyWidth, valueWidth, tokenCount, nil
}

// hipIncrOracleRowDelta returns max|Δ| over one token row shared by both caches.
func hipIncrOracleRowDelta(a, b []float32, width, aTok, bTok, pos int) float32 {
	if pos < 0 || pos >= aTok || pos >= bTok {
		return float32(math.NaN())
	}
	var maxAbs float32
	for i := 0; i < width; i++ {
		d := a[pos*width+i] - b[pos*width+i]
		if d < 0 {
			d = -d
		}
		if d > maxAbs {
			maxAbs = d
		}
	}
	return maxAbs
}

// hipIncrOracleAllDelta returns max|Δ| over every element both caches share.
func hipIncrOracleAllDelta(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var maxAbs float32
	for i := 0; i < n; i++ {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		if d > maxAbs {
			maxAbs = d
		}
	}
	return maxAbs
}

// hipIncrOracleLayerKind classifies a layer as full-attention or sliding-window
// (the 5:1 interleave), used to read the divergence signature.
func hipIncrOracleLayerKind(layer hipGemma4Q4Layer0Config) string {
	if layer.SlidingWindow > 0 {
		return "sliding"
	}
	return "full"
}

// hipIncrOracleArgmax is the host greedy over a softcapped logit vector, skipping
// suppressed control tokens exactly as production does before argmax.
func hipIncrOracleArgmax(logits []float32, suppress map[int32]bool) int {
	best := -1
	for i, v := range logits {
		if suppress[int32(i)] {
			continue
		}
		if best < 0 || v > logits[best] {
			best = i
		}
	}
	if best < 0 {
		best = 0
	}
	return best
}
