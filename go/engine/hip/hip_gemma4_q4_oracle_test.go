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
