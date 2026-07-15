// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"encoding/binary"
	"math"
	"testing"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference"
	inferdecode "dappco.re/go/inference/decode"
)

func TestHIPSmallDecode_Good_QwenGemmaSmoke(t *testing.T) {
	for _, architecture := range []string{"qwen3", "gemma3"} {
		t.Run(architecture, func(t *testing.T) {
			req := hipSmallDecodeFixture(architecture)
			want, err := hipReferenceSmallDecode(req)
			core.RequireNoError(t, err)

			driver := &fakeHIPDriver{available: true}
			got, err := hipRunSmallDecode(context.Background(), driver, req)
			core.RequireNoError(t, err)

			core.AssertEqual(t, want.TokenID, got.TokenID)
			assertFloat32Near(t, want.Score, got.Score)
			assertFloat32SlicesNear(t, want.Logits, got.Logits, 0.0001)
			assertFloat32SlicesNear(t, want.Attention, got.Attention, 0.0001)
			assertFloat32SlicesNear(t, want.UpdatedKeys, got.UpdatedKeys, 0.0001)
			assertFloat32SlicesNear(t, want.UpdatedValues, got.UpdatedValues, 0.0001)
			core.AssertEqual(t, architecture, got.Labels["decode_architecture"])
			core.AssertEqual(t, "rms_norm,projection,rope,attention,greedy", got.Labels["decode_primitives"])

			var launchNames []string
			for _, launch := range driver.launches {
				launchNames = append(launchNames, launch.Name)
			}
			joined := core.Join(",", launchNames...)
			core.AssertContains(t, joined, hipKernelNameRMSNorm)
			core.AssertContains(t, joined, hipKernelNameProjection)
			core.AssertContains(t, joined, hipKernelNameRoPE)
			core.AssertContains(t, joined, hipKernelNameAttention)
			core.AssertContains(t, joined, hipKernelNameGreedy)
		})
	}
}

func TestHIPRMSNormResidualAddNormBlockSize_E4B_Good(t *testing.T) {
	core.AssertEqual(t, uint32(512), hipRMSNormResidualAddNormBlockSize(2560))
}

func TestHIPRMSNormResidualAddNormBlockSize_Default_Good(t *testing.T) {
	core.AssertEqual(t, uint32(256), hipRMSNormResidualAddNormBlockSize(1536))
}

func TestHIPRMSNormResidualAddBlockSize_E4B_Good(t *testing.T) {
	core.AssertEqual(t, uint32(512), hipRMSNormResidualAddBlockSize(2560))
}

func TestHIPRMSNormResidualAddBlockSize_Default_Good(t *testing.T) {
	core.AssertEqual(t, uint32(256), hipRMSNormResidualAddBlockSize(1536))
}

func TestHIPGemma4Q4Layer0_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup()

	got, err := hipRunGemma4Q4Layer0(context.Background(), driver, cfg, hipGemma4Q4Layer0Request{
		TokenID:  1,
		Position: 1,
		RoPEBase: 10000,
		Epsilon:  1e-6,
	})
	core.RequireNoError(t, err)

	core.AssertEqual(t, cfg.HiddenSize, len(got.Embedding))
	core.AssertEqual(t, cfg.HiddenSize, len(got.LayerInput))
	core.AssertEqual(t, cfg.QueryHeads*cfg.HeadDim, len(got.AttentionOutput))
	core.AssertEqual(t, cfg.HiddenSize, len(got.FinalHidden))
	core.AssertEqual(t, cfg.VocabSize, len(got.Logits))
	core.AssertEqual(t, 0, got.Greedy.TokenID)
	assertFloat32Near(t, 0, got.Greedy.Score)
	core.AssertEqual(t, hipKernelStatusLinked, got.Labels["gemma4_q4_layer0_kernel"])
	core.AssertEqual(t, hipKernelStatusNotLinked, got.Labels["production_decode"])
	core.AssertEqual(t, "0", got.Labels["decode_layer"])
	core.AssertContains(t, got.Labels["decode_primitives"], "mlx_q4_projection")

	var launchNames []string
	for _, launch := range driver.launches {
		launchNames = append(launchNames, launch.Name)
	}
	joined := core.Join(",", launchNames...)
	core.AssertContains(t, joined, hipKernelNameEmbedLookup)
	core.AssertContains(t, joined, hipKernelNameVectorScale)
	core.AssertContains(t, joined, hipKernelNameRMSNorm)
	core.AssertContains(t, joined, hipKernelNameMLXQ4Proj)
	core.AssertContains(t, joined, hipKernelNameRMSNormRoPEHeads)
	core.AssertContains(t, joined, hipKernelNameAttentionHeads)
	core.AssertContains(t, joined, hipKernelNameRMSNormResidualAdd)
	core.AssertContains(t, joined, hipKernelNameMLXQ4GELUTanhMul)
	core.AssertContains(t, joined, hipKernelNameGreedy)
	core.AssertContains(t, got.Labels["decode_primitives"], "gelu_tanh_mlp")
	core.AssertEqual(t, "device_gelu_tanh_multiply", got.Labels["gemma4_mlp_activation"])
	attentionScales := 0
	for _, launch := range driver.launches {
		if launch.Name == hipKernelNameAttentionHeads {
			attentionScales++
			tokenCount := binary.LittleEndian.Uint32(launch.Args[52:])
			core.AssertEqual(t, uint64(0), binary.LittleEndian.Uint64(launch.Args[40:]))
			core.AssertEqual(t, uint32(0), binary.LittleEndian.Uint32(launch.Args[76:]))
			core.AssertEqual(t, tokenCount*4, launch.SharedMemBytes)
			assertFloat32Near(t, 1, math.Float32frombits(binary.LittleEndian.Uint32(launch.Args[84:])))
		}
	}
	if attentionScales == 0 {
		t.Fatalf("Gemma4 q4 layer did not launch attention")
	}

	layerOnly, err := hipRunGemma4Q4DecoderLayer(context.Background(), driver, cfg, got.ScaledEmbedding, hipGemma4Q4DecoderLayerRequest{
		Position: 1,
		Epsilon:  1e-6,
	})
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, got.FinalHidden, layerOnly.FinalHidden, 0.0001)

	nonZeroInput := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	residualLayer, err := hipRunGemma4Q4DecoderLayer(context.Background(), driver, cfg, nonZeroInput, hipGemma4Q4DecoderLayerRequest{
		Position: 1,
		Epsilon:  1e-6,
	})
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, nonZeroInput, residualLayer.AttentionResidual, 0.0001)
	assertFloat32SlicesNear(t, nonZeroInput, residualLayer.FinalHidden, 0.0001)

	scaledCfg := cfg
	scaledCfg.LayerScalar = 0.5
	scaledLayer, err := hipRunGemma4Q4DecoderLayer(context.Background(), driver, scaledCfg, nonZeroInput, hipGemma4Q4DecoderLayerRequest{
		Position: 1,
		Epsilon:  1e-6,
	})
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4}, scaledLayer.FinalHidden, 0.0001)

	gelu, err := hipGemma4Q4HostGELU([]float32{-1, 0, 1})
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{-0.1588, 0, 0.8412}, gelu, 0.0001)

	partialRoPEStart := len(driver.launches)
	partialRoPECfg := cfg
	partialRoPECfg.RoPERotaryDim = cfg.HeadDim / 2
	_, err = hipRunGemma4Q4DecoderLayer(context.Background(), driver, partialRoPECfg, nonZeroInput, hipGemma4Q4DecoderLayerRequest{
		Position: 1,
		Epsilon:  1e-6,
	})
	core.RequireNoError(t, err)
	partialRoPELaunches := 0
	for _, launch := range driver.launches[partialRoPEStart:] {
		if launch.Name == hipKernelNameRMSNormRoPEHeads {
			partialRoPELaunches++
			core.AssertEqual(t, uint32(cfg.HeadDim), binary.LittleEndian.Uint32(launch.Args[72:]))
			core.AssertEqual(t, uint32(cfg.HeadDim/2), binary.LittleEndian.Uint32(launch.Args[76:]))
		}
	}
	if partialRoPELaunches == 0 {
		t.Fatalf("partial Gemma4 q4 RoPE did not launch")
	}

	perLayerStart := len(driver.launches)
	perLayerLayer, err := hipRunGemma4Q4DecoderLayer(context.Background(), driver, cfg, nonZeroInput, hipGemma4Q4DecoderLayerRequest{
		Position:      1,
		Epsilon:       1e-6,
		PerLayerInput: []float32{0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2},
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, cfg.HiddenSize, len(perLayerLayer.FinalHidden))
	perLayerQ4Ops := 0
	perLayerTripleQ4Launches := 0
	for _, launch := range driver.launches[perLayerStart:] {
		switch launch.Name {
		case hipKernelNameMLXQ4Proj:
			perLayerQ4Ops++
		case hipKernelNameMLXQ4TripleProj:
			perLayerQ4Ops += 3
			perLayerTripleQ4Launches++
		}
	}
	core.AssertEqual(t, 6, perLayerQ4Ops)
	core.AssertEqual(t, 1, perLayerTripleQ4Launches)

	variable, variableCleanup := hipGemma4Q4FixtureConfig(t, driver, 1, 4, 2, 16)
	variable.RoPEBase = 1000000
	variable.RoPERotaryDim = 2
	variable.SlidingWindow = 0
	defer variableCleanup()
	variableLayer, err := hipRunGemma4Q4DecoderLayer(context.Background(), driver, variable, got.ScaledEmbedding, hipGemma4Q4DecoderLayerRequest{
		Position: 1,
		Epsilon:  1e-6,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, cfg.HiddenSize, len(variableLayer.FinalHidden))
	core.AssertEqual(t, variable.QueryHeads*variable.HeadDim, len(variableLayer.AttentionOutput))
	core.AssertEqual(t, variable.IntermediateSize, variable.GateProjection.Rows)

	forward, err := hipRunGemma4Q4SingleTokenForward(context.Background(), driver, hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{cfg, variable}}, hipGemma4Q4ForwardRequest{
		TokenID:  1,
		Position: 1,
		Epsilon:  1e-6,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, len(forward.LayerResults))
	core.AssertEqual(t, cfg.HiddenSize, len(forward.FinalHidden))
	core.AssertEqual(t, cfg.VocabSize, len(forward.Logits))
	core.AssertEqual(t, "2", forward.Labels["decode_layers"])
	core.AssertEqual(t, hipKernelStatusNotLinked, forward.Labels["production_decode"])

	sliding := cfg
	sliding.SlidingWindow = 2
	decodeCfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{sliding, variable}}
	decodeLaunchStart := len(driver.launches)
	decode, err := hipRunGemma4Q4GreedyDecode(context.Background(), driver, decodeCfg, hipGemma4Q4GreedyDecodeRequest{
		PromptTokenIDs:    []int32{1, 0},
		MaxNewTokens:      2,
		Position:          1,
		Epsilon:           1e-6,
		MirrorDeviceKV:    true,
		DeviceKVAttention: true,
		DeviceKVMode:      rocmKVCacheModeKQ8VQ4,
	})
	core.RequireNoError(t, err)
	defer decode.DeviceState.Close()
	core.AssertEqual(t, 2, len(decode.Generated))
	core.AssertEqual(t, 3, len(decode.StepResults))
	core.AssertEqual(t, 2, len(decode.State.Layers))
	core.AssertEqual(t, cfg.HeadDim*2, len(decode.State.Layers[0].Keys))
	core.AssertEqual(t, variable.HeadDim*3, len(decode.State.Layers[1].Keys))
	core.AssertEqual(t, "2", decode.Labels["decode_prompt_tokens"])
	core.AssertEqual(t, "2", decode.Labels["decode_generated_tokens"])
	core.AssertEqual(t, "3", decode.Labels["decode_forward_steps"])
	core.AssertEqual(t, "3", decode.Labels["decode_state_tokens"])
	core.AssertEqual(t, hipKernelStatusNotLinked, decode.Labels["production_decode"])
	core.AssertEqual(t, hipKernelStatusNotLinked, decode.Labels["production_kv_cache_backing"])
	core.AssertEqual(t, "hip_device_mirror", decode.Labels["gemma4_q4_device_kv_backing"])
	core.AssertEqual(t, "2", decode.Labels["gemma4_q4_device_kv_layers"])
	core.AssertEqual(t, "2", decode.Labels["gemma4_q4_device_kv_min_tokens"])
	core.AssertEqual(t, "3", decode.Labels["gemma4_q4_device_kv_max_tokens"])
	core.AssertEqual(t, "hip_device_descriptor", decode.StepResults[0].Labels["attention_kv_backing"])
	core.AssertEqual(t, rocmKVCacheModeKQ8VQ4, decode.StepResults[0].Labels["attention_kv_mode"])
	core.AssertEqual(t, "returned", decode.StepResults[0].Labels["gemma4_q4_forward_device_state"])
	core.AssertEqual(t, "0", decode.StepResults[0].Labels["attention_kv_append_layers"])
	core.AssertEqual(t, "2", decode.StepResults[0].Labels["attention_kv_remirror_layers"])
	core.AssertEqual(t, "2", decode.StepResults[1].Labels["attention_kv_append_layers"])
	core.AssertEqual(t, "0", decode.StepResults[1].Labels["attention_kv_remirror_layers"])
	core.AssertEqual(t, "1", decode.StepResults[2].Labels["attention_kv_append_layers"])
	core.AssertEqual(t, "1", decode.StepResults[2].Labels["attention_kv_remirror_layers"])
	core.AssertEqual(t, "1", decode.StepResults[2].Labels["gemma4_q4_device_kv_append_layers"])
	core.AssertEqual(t, "1", decode.StepResults[2].Labels["gemma4_q4_device_kv_remirror_layers"])
	if countDeviceAttentionLaunches(driver.launches[decodeLaunchStart:]) == 0 {
		t.Fatalf("Gemma4 q4 decode launched no descriptor-backed attention kernels")
	}

	deviceState := decode.DeviceState
	if deviceState == nil {
		t.Fatalf("Gemma4 q4 decode device state is nil, want carried HIP mirror")
	}
	core.AssertEqual(t, 2, deviceState.LayerCount())
	core.AssertEqual(t, []int{2, 3}, deviceState.LayerTokenCounts())
	deviceLabels := deviceState.Labels()
	core.AssertEqual(t, "hip_device_mirror", deviceLabels["gemma4_q4_device_kv_backing"])
	core.AssertEqual(t, "2", deviceLabels["gemma4_q4_device_kv_layers"])
	core.AssertEqual(t, "2", deviceLabels["gemma4_q4_device_kv_min_tokens"])
	core.AssertEqual(t, "3", deviceLabels["gemma4_q4_device_kv_max_tokens"])
	core.AssertEqual(t, rocmKVCacheModeKQ8VQ4, deviceLabels["gemma4_q4_device_kv_mode"])
	core.AssertEqual(t, "1", deviceLabels["gemma4_q4_device_kv_append_layers"])
	core.AssertEqual(t, "1", deviceLabels["gemma4_q4_device_kv_remirror_layers"])
	core.AssertEqual(t, hipKernelStatusNotLinked, deviceLabels["production_kv_cache_backing"])
	restoredState, err := deviceState.HostState()
	core.RequireNoError(t, err)
	assertGemma4Q4DeviceStateMatchesQuantizedHost(t, decodeCfg, decode.State, restoredState, deviceState, rocmKVCacheModeKQ8VQ4)
	freeStart := len(driver.frees)
	core.RequireNoError(t, deviceState.Close())
	if len(driver.frees)-freeStart <= 0 {
		t.Fatalf("device state close freed %d allocations, want at least one", len(driver.frees)-freeStart)
	}
	_, err = deviceState.HostState()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "closed")

	quantizedForwardStart := len(driver.launches)
	quantizedForward, err := hipRunGemma4Q4SingleTokenForward(context.Background(), driver, hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{cfg}}, hipGemma4Q4ForwardRequest{
		TokenID:           1,
		Position:          1,
		Epsilon:           1e-6,
		DeviceKVAttention: true,
		DeviceKVMode:      rocmKVCacheModeKQ8VQ4,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, "hip_device_descriptor", quantizedForward.Labels["attention_kv_backing"])
	core.AssertEqual(t, rocmKVCacheModeKQ8VQ4, quantizedForward.Labels["attention_kv_mode"])
	core.AssertEqual(t, "0", quantizedForward.Labels["attention_kv_append_layers"])
	core.AssertEqual(t, "1", quantizedForward.Labels["attention_kv_remirror_layers"])
	core.AssertEqual(t, hipKernelStatusNotLinked, quantizedForward.Labels["production_kv_cache_backing"])
	if countDeviceAttentionLaunches(driver.launches[quantizedForwardStart:]) == 0 {
		t.Fatalf("Gemma4 q4 k-q8-v-q4 forward launched no descriptor-backed attention kernels")
	}

	partialRoPE, err := hipRunGemma4Q4RoPEVector(context.Background(), driver, []float32{1, 0, 3, 4}, 1, 1, 2)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{float32(math.Cos(1)), float32(math.Sin(1)), 3, 4}, partialRoPE, 0.0001)

	softcapped, err := hipGemma4Q4SoftcapLogits([]float32{0, 30, -30}, 30)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{0, float32(math.Tanh(1) * 30), -float32(math.Tanh(1) * 30)}, softcapped, 0.0001)

	t.Setenv("GO_ROCM_GEMMA4_Q4_FORWARD_LAYERS", "2")
	layerCount, ok := gemma4Q4ForwardLayerCountFromEnv(t, 2)
	core.AssertEqual(t, true, ok)
	core.AssertEqual(t, 2, layerCount)

	t.Setenv("GO_ROCM_GEMMA4_Q4_DECODE_PROMPT_TOKENS", "1, 0")
	promptTokens := gemma4Q4DecodePromptTokensEnv(t, cfg.VocabSize)
	core.AssertEqual(t, []int32{1, 0}, promptTokens)

	parsedTokens, tokenPrompt, err := hipGemma4Q4TokenPromptIDs("tokens:1, 0", cfg.VocabSize)
	core.RequireNoError(t, err)
	core.AssertEqual(t, true, tokenPrompt)
	core.AssertEqual(t, []int32{1, 0}, parsedTokens)
	parsedTokens, tokenPrompt, err = hipGemma4Q4TokenPromptIDs(" TOKENS:1, 0", cfg.VocabSize)
	core.RequireNoError(t, err)
	core.AssertEqual(t, true, tokenPrompt)
	core.AssertEqual(t, []int32{1, 0}, parsedTokens)
	_, tokenPrompt, err = hipGemma4Q4TokenPromptIDs("hello", cfg.VocabSize)
	core.RequireNoError(t, err)
	core.AssertEqual(t, false, tokenPrompt)

	countEmbeddingLaunches := func(start int) int {
		t.Helper()
		var count int
		for _, launch := range driver.launches[start:] {
			if launch.Name == hipKernelNameEmbedLookup || launch.Name == hipKernelNameEmbedLookupGreedyToken {
				count++
			}
		}
		return count
	}

	launchStart := len(driver.launches)
	var retainedState *hipGemma4Q4DeviceDecodeState
	stream, streamErr := hipGemma4Q4GenerateTokenSeqWithState(context.Background(), &hipLoadedModel{driver: driver}, decodeCfg, []int32{1, 0}, inference.GenerateConfig{MaxTokens: 2}, defaultHIPGemma4Q4EngineConfig(), nil, func(state *hipGemma4Q4DeviceDecodeState) error {
		retainedState = state
		return nil
	})
	var generated []inference.Token
	for token := range stream {
		generated = append(generated, token)
	}
	core.RequireNoError(t, streamErr())
	if retainedState == nil {
		t.Fatal("Gemma4 q4 generate did not retain device state")
	}
	core.AssertEqual(t, false, retainedState.closed)
	core.AssertEqual(t, len(decodeCfg.Layers), retainedState.LayerCount())
	core.AssertGreater(t, retainedState.maxLayerTokenCount(), 0)
	core.RequireNoError(t, retainedState.Close())
	core.AssertEqual(t, 2, len(generated))
	core.AssertEqual(t, 3, countEmbeddingLaunches(launchStart))
	core.AssertEqual(t, 6, countKVEncodeTokenLaunches(driver.launches[launchStart:]))
	if countDeviceAttentionLaunches(driver.launches[launchStart:]) == 0 {
		t.Fatalf("Gemma4 q4 public generate launched no descriptor-backed attention kernels")
	}
	for _, token := range generated {
		core.AssertEqual(t, int32(0), token.ID)
		core.AssertEqual(t, "<token:0>", token.Text)
	}

	statefulModel := &rocmModel{}
	stream, streamErr = statefulModel.hipGemma4Q4GenerateTokenSeq(context.Background(), nil, &hipLoadedModel{driver: driver}, decodeCfg, []int32{1, 0}, inference.GenerateConfig{MaxTokens: 1})
	generated = nil
	for token := range stream {
		generated = append(generated, token)
	}
	core.RequireNoError(t, streamErr())
	core.AssertEqual(t, 1, len(generated))
	statefulRuntime, ok := statefulModel.state.runtime.(*hipGemma4Q4DeviceDecodeState)
	core.RequireTrue(t, ok)
	core.AssertEqual(t, false, statefulRuntime.closed)
	core.AssertEqual(t, len(decodeCfg.Layers), statefulRuntime.LayerCount())
	core.RequireNoError(t, resultError(statefulModel.Close()))

	tokenText := &hipTokenTextDecoder{
		vocab: map[string]int32{
			"h":                  10,
			"e":                  11,
			"he":                 12,
			"\u2581":             13,
			"z":                  14,
			"\u2581z":            15,
			"<0xE2>":             16,
			"<0x82>":             17,
			"<0xAC>":             18,
			"<unk>":              19,
			"\u2581zero":         0,
			"<eos>":              1,
			"<0xE2><0x82><0xAC>": 2,
		},
		pieces: map[int32]string{
			0:  "\u2581zero",
			1:  "<eos>",
			2:  "<0xE2><0x82><0xAC>",
			10: "h",
			11: "e",
			12: "he",
			13: "\u2581",
			14: "z",
			15: "\u2581z",
			16: "<0xE2>",
			17: "<0x82>",
			18: "<0xAC>",
			19: "<unk>",
		},
		mergeRanks:  map[string]int{"h e": 0, "\u2581 z": 1},
		special:     map[int32]bool{1: true},
		specialText: map[string]int32{"<eos>": 1},
		unknownID:   19,
		hasUnknown:  true,
	}
	core.AssertEqual(t, []int32{12, 15, 1}, tokenText.Encode("he z<eos>"))
	bosTokenText := &hipTokenTextDecoder{
		vocab: map[string]int32{
			"<bos>": 2,
			"h":     10,
			"e":     11,
			"he":    12,
		},
		pieces:      map[int32]string{2: "<bos>", 10: "h", 11: "e", 12: "he"},
		mergeRanks:  map[string]int{"h e": 0},
		special:     map[int32]bool{2: true},
		specialText: map[string]int32{"<bos>": 2},
		bosID:       2,
		hasBOS:      true,
	}
	core.AssertEqual(t, []int32{2, 12}, bosTokenText.Encode("he"))
	core.AssertEqual(t, []int32{2, 12}, bosTokenText.Encode("<bos>he"))
	textPromptTokens, textPrompt, err := hipGemma4Q4TextPromptIDs("text:he z", &hipLoadedModel{tokenText: tokenText})
	core.RequireNoError(t, err)
	core.AssertEqual(t, true, textPrompt)
	core.AssertEqual(t, []int32{12, 15}, textPromptTokens)
	textPromptTokens, textPrompt, err = hipGemma4Q4TextPromptIDs(" TEXT:he z", &hipLoadedModel{tokenText: tokenText})
	core.RequireNoError(t, err)
	core.AssertEqual(t, true, textPrompt)
	core.AssertEqual(t, []int32{12, 15}, textPromptTokens)
	_, textPrompt, err = hipGemma4Q4TextPromptIDs("he z", &hipLoadedModel{
		modelInfo:   inference.ModelInfo{Architecture: "gemma4", QuantBits: 4},
		modelLabels: linkedGemma4TestLabels("E2B", "q4"),
		tokenText:   tokenText,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, true, textPrompt)
	_, textPrompt, err = hipGemma4Q4TextPromptIDs("he z", &hipLoadedModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4", QuantBits: 16},
		tokenText: tokenText,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, false, textPrompt)
	textPromptTokens, textPrompt, err = hipGemma4Q4TextPromptIDs("he z", &hipLoadedModel{
		modelInfo:   inference.ModelInfo{Architecture: "gemma4", QuantBits: 4},
		modelLabels: linkedGemma4TestLabels("E2B", "q4"),
		tokenText:   tokenText,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, true, textPrompt)
	core.AssertEqual(t, []int32{12, 15}, textPromptTokens)
	textPromptTokens, textPrompt, err = hipGemma4Q4TextPromptIDs(" z", &hipLoadedModel{
		modelInfo:   inference.ModelInfo{Architecture: "gemma4", QuantBits: 4},
		modelLabels: linkedGemma4TestLabels("E2B", "q4"),
		tokenText:   tokenText,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, true, textPrompt)
	core.AssertEqual(t, []int32{15}, textPromptTokens)
	textPromptTokens, textPrompt, err = hipGemma4Q4TextPromptIDs("he", &hipLoadedModel{
		modelInfo:   inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 4},
		modelLabels: linkedGemma4TestLabels("E2B", "q4"),
		tokenText:   tokenText,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, true, textPrompt)
	core.AssertEqual(t, []int32{12}, textPromptTokens)
	core.AssertEqual(t, " zero", tokenText.DecodeToken(0))
	core.AssertEqual(t, "", tokenText.DecodeToken(1))
	core.AssertEqual(t, "\xe2\x82\xac", tokenText.DecodeToken(2))

	stream, streamErr = hipGemma4Q4GenerateTokenSeq(context.Background(), &hipLoadedModel{driver: driver, tokenText: tokenText}, decodeCfg, []int32{1, 0}, inference.GenerateConfig{MaxTokens: 1})
	generated = nil
	for token := range stream {
		generated = append(generated, token)
	}
	core.RequireNoError(t, streamErr())
	core.AssertEqual(t, 1, len(generated))
	core.AssertEqual(t, " zero", generated[0].Text)

	launchStart = len(driver.launches)
	freeStart = len(driver.frees)
	stream, streamErr = hipGemma4Q4GenerateTokenSeq(context.Background(), &hipLoadedModel{driver: driver, tokenText: tokenText}, decodeCfg, []int32{1, 0}, inference.GenerateConfig{MaxTokens: 2})
	generated = nil
	for token := range stream {
		generated = append(generated, token)
		break
	}
	core.RequireNoError(t, streamErr())
	core.AssertEqual(t, 1, len(generated))
	core.AssertEqual(t, 2, countEmbeddingLaunches(launchStart))
	core.AssertEqual(t, 4, countKVEncodeTokenLaunches(driver.launches[launchStart:]))
	if countDeviceAttentionLaunches(driver.launches[launchStart:]) == 0 {
		t.Fatalf("Gemma4 q4 early-stopped public generate launched no descriptor-backed attention kernels")
	}
	if len(driver.frees) == freeStart {
		t.Fatalf("Gemma4 q4 early-stopped public generate freed no device KV allocations")
	}
}

func TestHIPGemma4Q4GenerateTokenSeq_UsesBatchedPrefill_Good(t *testing.T) {
	hipDrainAttentionHeadsChunkedWorkspacePoolForTest(t)
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup0()
	layer1, cleanup1 := hipGemma4Q4FixtureConfig(t, driver, 1, 4, 2, 8)
	defer cleanup1()
	embeddingWeightsPayload, err := hipUint32Payload(make([]uint32, layer0.VocabSize*(layer0.HiddenSize/layer0.GroupSize)))
	core.RequireNoError(t, err)
	embeddingWeights, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "generate batched prefill embedding weights", embeddingWeightsPayload, layer0.VocabSize*(layer0.HiddenSize/layer0.GroupSize))
	core.RequireNoError(t, err)
	defer embeddingWeights.Close()
	embeddingScalesPayload, err := hipUint16Payload([]uint16{0x3f80, 0x3f80})
	core.RequireNoError(t, err)
	embeddingScales, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "generate batched prefill embedding scales", embeddingScalesPayload, layer0.VocabSize*(layer0.HiddenSize/layer0.GroupSize))
	core.RequireNoError(t, err)
	defer embeddingScales.Close()
	embeddingBiasesPayload, err := hipUint16Payload([]uint16{0x3f80, 0x3f80})
	core.RequireNoError(t, err)
	embeddingBiases, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "generate batched prefill embedding biases", embeddingBiasesPayload, layer0.VocabSize*(layer0.HiddenSize/layer0.GroupSize))
	core.RequireNoError(t, err)
	defer embeddingBiases.Close()
	layer0.Embedding = hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: embeddingWeights.Pointer(),
		EmbeddingBytes:   embeddingWeights.SizeBytes(),
		TableEncoding:    hipEmbeddingTableEncodingMLXQ4,
		VocabSize:        layer0.VocabSize,
		HiddenSize:       layer0.HiddenSize,
		GroupSize:        layer0.GroupSize,
		ScalePointer:     embeddingScales.Pointer(),
		BiasPointer:      embeddingBiases.Pointer(),
		ScaleBytes:       embeddingScales.SizeBytes(),
		BiasBytes:        embeddingBiases.SizeBytes(),
	}
	layers, cleanupPerLayer := hipGemma4Q4GlobalPerLayerInputFixture(t, driver, []hipGemma4Q4Layer0Config{layer0, layer1})
	defer cleanupPerLayer()
	cfg := hipGemma4Q4ForwardConfig{Layers: layers}
	core.AssertEqual(t, true, hipGemma4Q4CanUseBatchedGeneratePrefill(cfg))
	engineConfig := defaultHIPGemma4Q4EngineConfig()
	engineConfig.DeviceKVMode = rocmKVCacheModeFP16
	engineConfig.DisableInterleavedRowPages = true
	engineConfig.PrefillUBatchTokens = 2

	start := len(driver.launches)
	allocStart := len(driver.allocations)
	stream, streamErr := hipGemma4Q4GenerateTokenSeqWithEngineConfig(context.Background(), &hipLoadedModel{driver: driver}, cfg, []int32{0, 1, 0}, inference.GenerateConfig{MaxTokens: 2}, engineConfig)
	var generated []inference.Token
	for token := range stream {
		generated = append(generated, token)
	}

	core.RequireNoError(t, streamErr())
	core.AssertEqual(t, 2, len(generated))
	launches := driver.launches[start:]
	batchProjectionLaunches := countLaunchName(launches, hipKernelNameMLXQ4ProjBatch)
	batchAttentionLaunches := countLaunchName(launches, hipKernelNameAttentionHeadsBatchCausal)
	finalGreedyLaunches := countLaunchName(launches, hipKernelNameMLXQ4ProjGreedy)
	if batchProjectionLaunches == 0 || batchAttentionLaunches == 0 || finalGreedyLaunches == 0 {
		t.Fatalf("Gemma4 q4 generate batched prefill launches projection_batch=%d attention_batch=%d final_greedy=%d, want all nonzero", batchProjectionLaunches, batchAttentionLaunches, finalGreedyLaunches)
	}
	wantGreedySlots := hipProjectionGreedyRoundFirstSlabSlots(hipProjectionGreedyReserveSlots + 4)
	core.AssertEqual(t, 1, countUint64Value(driver.allocations[allocStart:], uint64(wantGreedySlots*hipMLXQ4ProjectionBestBytes)))

	engineConfig.BidirectionalSpanTokens = [2]int32{1, 0}
	start = len(driver.launches)
	stream, streamErr = hipGemma4Q4GenerateTokenSeqWithEngineConfig(context.Background(), &hipLoadedModel{driver: driver}, cfg, []int32{0, 1, 1, 0}, inference.GenerateConfig{MaxTokens: 1}, engineConfig)
	generated = nil
	for token := range stream {
		generated = append(generated, token)
	}
	core.RequireNoError(t, streamErr())
	core.AssertEqual(t, 1, len(generated))
	launches = driver.launches[start:]
	core.AssertEqual(t, len(cfg.Layers), countLaunchName(launches, hipKernelNameAttentionHeadsBatchCapped))

	engineConfig.BidirectionalSpanTokens = [2]int32{}

	engineConfig.DisableBatchedPrefill = true
	start = len(driver.launches)
	stream, streamErr = hipGemma4Q4GenerateTokenSeqWithEngineConfig(context.Background(), &hipLoadedModel{driver: driver}, cfg, []int32{0, 1, 0}, inference.GenerateConfig{MaxTokens: 2}, engineConfig)
	generated = nil
	for token := range stream {
		generated = append(generated, token)
	}
	core.RequireNoError(t, streamErr())
	core.AssertEqual(t, 2, len(generated))
	launches = driver.launches[start:]
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameAttentionHeadsBatchCausal))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameMLXQ4ProjBatch))
}

func TestHIPGemma4Q4GenerateTokenSeq_MoEDisablesBatchedPrefill_Good(t *testing.T) {
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{{
		MoE: &hipGemma4MoELayerConfig{},
	}}}

	core.AssertEqual(t, false, hipGemma4Q4CanUseBatchedGeneratePrefill(cfg))
}

func TestHIPAttachedDrafterTargetPrefillUsesBatchedPath_Good(t *testing.T) {
	hipDrainAttentionHeadsChunkedWorkspacePoolForTest(t)
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup0()
	layer1, cleanup1 := hipGemma4Q4FixtureConfig(t, driver, 1, 4, 2, 8)
	defer cleanup1()
	hipGemma4Q4InstallNonzeroEmbeddingFixture(t, driver, &layer0, "attached target batched prefill")
	layers, cleanupPerLayer := hipGemma4Q4GlobalPerLayerInputFixture(t, driver, []hipGemma4Q4Layer0Config{layer0, layer1})
	defer cleanupPerLayer()
	cfg := hipGemma4Q4ForwardConfig{Layers: layers}
	core.AssertEqual(t, true, hipGemma4Q4CanUseBatchedGeneratePrefill(cfg))
	engineConfig := defaultHIPGemma4Q4EngineConfig()
	engineConfig.PrefillUBatchTokens = 2
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	workspace.EnsureProjectionGreedyBestCapacity(4)
	greedyBuffer, err := workspace.BorrowProjectionGreedyBest(driver)
	core.RequireNoError(t, err)

	start := len(driver.launches)
	result, err := hipRunAttachedDrafterTargetPrefill(context.Background(), driver, hipAttachedDrafterTargetPrefillRequest{
		TargetForward: cfg,
		DeviceKVMode:  rocmKVCacheModeKQ8VQ4,
		EngineConfig:  engineConfig,
		InputTokenIDs: []int32{0, 1, 0},
		Position:      0,
		Epsilon:       1e-6,
		GreedyBuffer:  greedyBuffer,
		Workspace:     workspace,
	})
	core.RequireNoError(t, err)
	defer result.DeviceState.Close()
	defer hipReleaseForwardDeviceFinalHidden(&result.Current)

	core.AssertEqual(t, int32(0), result.LastToken)
	core.AssertEqual(t, 3, result.Position)
	core.AssertEqual(t, 2, result.TargetCalls)
	core.AssertEqual(t, []int{3, 3}, result.DeviceState.LayerTokenCounts())
	if result.Current.DeviceFinalHidden == nil || result.Current.DeviceFinalHidden.Pointer() == 0 {
		t.Fatalf("attached target prefill hidden = %#v, want cloned final hidden", result.Current.DeviceFinalHidden)
	}
	launches := driver.launches[start:]
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameEmbedLookupGreedyToken))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameMLXQ4Proj))
	if batchProjectionLaunches := countLaunchName(launches, hipKernelNameMLXQ4ProjBatch); batchProjectionLaunches == 0 {
		t.Fatalf("attached target prefill launched projection_batch=%d, want batched projection", batchProjectionLaunches)
	}
	if batchAttentionLaunches := countLaunchName(launches, hipKernelNameAttentionHeadsBatchCausal); batchAttentionLaunches == 0 {
		t.Fatalf("attached target prefill launched attention_batch=%d, want batched attention", batchAttentionLaunches)
	}
	if finalGreedyLaunches := countLaunchName(launches, hipKernelNameMLXQ4ProjGreedy); finalGreedyLaunches == 0 {
		t.Fatalf("attached target prefill launched final_greedy=%d, want final greedy projection", finalGreedyLaunches)
	}
}

func TestHIPGemma4Q4EffectiveSlidingWindow_Good(t *testing.T) {
	core.AssertEqual(t, 512, hipGemma4Q4EffectiveSlidingWindow(256, 0))
	core.AssertEqual(t, 128, hipGemma4Q4EffectiveSlidingWindow(256, 128))
	core.AssertEqual(t, 512, hipGemma4Q4EffectiveSlidingWindow(256, 2048))
	core.AssertEqual(t, 0, hipGemma4Q4EffectiveSlidingWindow(512, 128))
}

func TestHIPGemma4Q4ChunkedAttentionEnabled_Good(t *testing.T) {
	core.AssertEqual(t, true, hipGemma4Q4ChunkedAttentionEnabled(1))
	core.AssertEqual(t, true, hipGemma4Q4ChunkedAttentionEnabled(4000))
}

func TestHIPGemma4Q4AttentionWorkspaceNeeded_Good(t *testing.T) {
	greedy := inference.GenerateConfig{}
	sampled := inference.GenerateConfig{Temperature: 1, TopK: 64, TopP: 0.95, RepeatPenalty: 1}
	repeatPenalty := inference.GenerateConfig{Temperature: 1, TopK: 64, TopP: 0.95, RepeatPenalty: 2}

	core.AssertEqual(t, true, hipGemma4Q4AttentionWorkspaceNeeded(128, greedy))
	core.AssertEqual(t, true, hipGemma4Q4AttentionWorkspaceNeeded(4000, greedy))
	core.AssertEqual(t, true, hipGemma4Q4AttentionWorkspaceNeeded(128, sampled))
	core.AssertEqual(t, true, hipGemma4Q4AttentionWorkspaceNeeded(128, repeatPenalty))
}

func TestHIPGemma4Q4DeviceKVBlockSize_Good(t *testing.T) {
	core.AssertEqual(t, rocmGemma4Q4DeviceKVBlockSize, hipGemma4Q4DeviceKVBlockSize())

	cfg := defaultHIPGemma4Q4EngineConfig()
	cfg.DeviceKVBlockSize = 16
	core.AssertEqual(t, 16, cfg.deviceKVBlockSize())

	cfg.DeviceKVBlockSize = 0
	core.AssertEqual(t, rocmGemma4Q4DeviceKVBlockSize, cfg.deviceKVBlockSize())
}

func TestHIPGemma4Q4DeviceKVBlockSizeForSlidingWindow_Good(t *testing.T) {
	core.AssertEqual(t, rocmGemma4Q4DeviceKVBlockSize, hipGemma4Q4DeviceKVBlockSizeForSlidingWindow(512))
	core.AssertEqual(t, rocmGemma4Q4GlobalDeviceKVBlockSize, hipGemma4Q4DeviceKVBlockSizeForSlidingWindow(0))

	cfg := defaultHIPGemma4Q4EngineConfig()
	cfg.GlobalDeviceKVBlockSize = 256
	core.AssertEqual(t, 256, cfg.deviceKVBlockSizeForSlidingWindow(0))
	core.AssertEqual(t, rocmGemma4Q4DeviceKVBlockSize, cfg.deviceKVBlockSizeForSlidingWindow(1024))

	cfg = defaultHIPGemma4Q4EngineConfig()
	cfg.DeviceKVBlockSize = 16
	cfg.GlobalDeviceKVBlockSize = 0
	core.AssertEqual(t, 16, cfg.deviceKVBlockSizeForSlidingWindow(0))
	core.AssertEqual(t, 16, cfg.deviceKVBlockSizeForSlidingWindow(512))

	cfg.DisableInterleavedRowPages = true
	core.AssertEqual(t, rocmGemma4Q4DeviceKVBlockSize, cfg.deviceKVBlockSizeForSlidingWindow(512))

	cfg.DisableInterleavedRowPages = false
	core.AssertEqual(t, 16, cfg.deviceKVBlockSizeForSlidingWindow(512))
}

func TestHIPAttachedDrafterResolveDraftTokensCapsSlidingWindow_Good(t *testing.T) {
	full := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{
		{SlidingWindow: 0},
		{SlidingWindow: 0},
	}}
	core.AssertEqual(t, 4, hipAttachedDrafterResolveDraftTokensForTarget(full, 4, 8))
	core.AssertEqual(t, 4, hipAttachedDrafterResolveDraftTokensForTarget(full, 0, 8))
	core.AssertEqual(t, 6, hipAttachedDrafterResolveDraftTokensForTarget(full, 8, 8))

	hybrid := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{
		{SlidingWindow: 0},
		{SlidingWindow: 5},
		{SlidingWindow: 3},
	}}
	core.AssertEqual(t, 2, hipAttachedDrafterMaxDraftProposalsForTarget(hybrid))
	core.AssertEqual(t, 2, hipAttachedDrafterResolveDraftTokensForTarget(hybrid, 8, 8))
	core.AssertEqual(t, 1, hipAttachedDrafterResolveDraftTokensForTarget(hybrid, 8, 1))

	tiny := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{
		{SlidingWindow: 1},
		{SlidingWindow: 2},
	}}
	core.AssertEqual(t, 1, hipAttachedDrafterResolveDraftTokensForTarget(tiny, 4, 8))
}

func TestHIPAttachedDrafterAdaptDraftTokensFallsBackOnLowAcceptance_Good(t *testing.T) {
	core.AssertEqual(t, ProductionMTPFallbackDraftTokens, hipAttachedDrafterAdaptDraftTokens(ProductionMTPDefaultDraftTokens, 4, 1))
	core.AssertEqual(t, ProductionMTPDefaultDraftTokens, hipAttachedDrafterAdaptDraftTokens(ProductionMTPDefaultDraftTokens, 4, 2))
	core.AssertEqual(t, ProductionMTPFallbackDraftTokens, hipAttachedDrafterAdaptDraftTokens(ProductionMTPFallbackDraftTokens, 2, 0))
	core.AssertEqual(t, 1, hipAttachedDrafterAdaptDraftTokens(1, 1, 0))
}

func TestHIPGemma4Q4DeviceKVTensorPrewarmCounts_Good(t *testing.T) {
	cfg := hipGemma4Q4ForwardConfig{
		Layers: []hipGemma4Q4Layer0Config{
			{HeadDim: 256, SlidingWindow: 512},
			{HeadDim: 256, SlidingWindow: 512},
			{HeadDim: 512, SlidingWindow: 0},
			{HeadDim: 512, SlidingWindow: 0},
		},
		KVSharedLayers:  2,
		SharedKVSources: []int{0, 0, 2, 2},
	}

	counts := hipGemma4Q4DeviceKVTensorPrewarmCounts(cfg, rocmKVCacheModeKQ8VQ4)
	keyEncoding, valueEncoding, ok := rocmKVInterleavedEncodingsForMode(rocmKVCacheModeKQ8VQ4)
	core.RequireTrue(t, ok, "KQ8VQ4 interleaved encodings should be available")
	localKeyStride, err := rocmKVInterleavedRowStride(keyEncoding, 256)
	core.RequireNoError(t, err)
	localValueStride, err := rocmKVInterleavedRowStride(valueEncoding, 256)
	core.RequireNoError(t, err)
	globalKeyStride, err := rocmKVInterleavedRowStride(keyEncoding, 512)
	core.RequireNoError(t, err)
	globalValueStride, err := rocmKVInterleavedRowStride(valueEncoding, 512)
	core.RequireNoError(t, err)
	keyTokenEncoding, valueTokenEncoding := rocmKVEncodingsForMode(rocmKVCacheModeKQ8VQ4)
	localKeyTokenBytes, err := rocmKVTensorDeviceByteCount(keyTokenEncoding, 256)
	core.RequireNoError(t, err)
	localValueTokenBytes, err := rocmKVTensorDeviceByteCount(valueTokenEncoding, 256)
	core.RequireNoError(t, err)
	globalKeyTokenBytes, err := rocmKVTensorDeviceByteCount(keyTokenEncoding, 512)
	core.RequireNoError(t, err)
	globalValueTokenBytes, err := rocmKVTensorDeviceByteCount(valueTokenEncoding, 512)
	core.RequireNoError(t, err)

	core.AssertEqual(t, 2, counts[(localKeyStride+localValueStride)*uint64(rocmGemma4Q4DeviceKVBlockSize)])
	core.AssertEqual(t, 4, counts[(globalKeyStride+globalValueStride)*uint64(rocmGemma4Q4GlobalDeviceKVBlockSize)])
	core.AssertEqual(t, 4, counts[localKeyTokenBytes])
	core.AssertEqual(t, 2, counts[localValueTokenBytes])
	core.AssertEqual(t, 2, counts[globalKeyTokenBytes])
	core.AssertEqual(t, localKeyTokenBytes, globalValueTokenBytes)
	core.AssertEqual(t, 5, len(counts))
}

func TestHIPGemma4Q4DeviceKVTensorPrewarmCountsForContext_UsesSharedOwners(t *testing.T) {
	cfg := hipGemma4Q4ForwardConfig{
		Layers: []hipGemma4Q4Layer0Config{
			{HeadDim: 256, SlidingWindow: 512},
			{HeadDim: 256, SlidingWindow: 512},
			{HeadDim: 512, SlidingWindow: 0},
			{HeadDim: 512, SlidingWindow: 0},
		},
		KVSharedLayers:  2,
		SharedKVSources: []int{0, 0, 2, 2},
	}

	counts := hipGemma4Q4DeviceKVTensorPrewarmCountsForContext(cfg, rocmKVCacheModeKQ8VQ4, 2560)
	keyEncoding, valueEncoding, ok := rocmKVInterleavedEncodingsForMode(rocmKVCacheModeKQ8VQ4)
	core.RequireTrue(t, ok, "KQ8VQ4 interleaved encodings should be available")
	localKeyStride, err := rocmKVInterleavedRowStride(keyEncoding, 256)
	core.RequireNoError(t, err)
	localValueStride, err := rocmKVInterleavedRowStride(valueEncoding, 256)
	core.RequireNoError(t, err)
	globalKeyStride, err := rocmKVInterleavedRowStride(keyEncoding, 512)
	core.RequireNoError(t, err)
	globalValueStride, err := rocmKVInterleavedRowStride(valueEncoding, 512)
	core.RequireNoError(t, err)
	keyTokenEncoding, valueTokenEncoding := rocmKVEncodingsForMode(rocmKVCacheModeKQ8VQ4)
	localKeyTokenBytes, err := rocmKVTensorDeviceByteCount(keyTokenEncoding, 256)
	core.RequireNoError(t, err)
	localValueTokenBytes, err := rocmKVTensorDeviceByteCount(valueTokenEncoding, 256)
	core.RequireNoError(t, err)
	globalKeyTokenBytes, err := rocmKVTensorDeviceByteCount(keyTokenEncoding, 512)
	core.RequireNoError(t, err)
	globalValueTokenBytes, err := rocmKVTensorDeviceByteCount(valueTokenEncoding, 512)
	core.RequireNoError(t, err)

	core.AssertEqual(t, 3, counts[(localKeyStride+localValueStride)*uint64(rocmGemma4Q4DeviceKVBlockSize)])
	core.AssertEqual(t, 5, counts[(globalKeyStride+globalValueStride)*uint64(rocmGemma4Q4GlobalDeviceKVBlockSize)])
	core.AssertEqual(t, 4, counts[localKeyTokenBytes])
	core.AssertEqual(t, 2, counts[localValueTokenBytes])
	core.AssertEqual(t, 2, counts[globalKeyTokenBytes])
	core.AssertEqual(t, localKeyTokenBytes, globalValueTokenBytes)
	core.AssertEqual(t, 5, len(counts))
}

func TestHIPGemma4Q4PrefillPlan_Good(t *testing.T) {
	ubatchTokens, err := hipGemma4Q4PrefillUBatchTokens()
	core.RequireNoError(t, err)
	core.AssertEqual(t, hipGemma4Q4PrefillDefaultUBatchTokens, ubatchTokens)

	engineConfig := defaultHIPGemma4Q4EngineConfig()
	engineConfig.PrefillUBatchTokens = 2
	ubatchTokens, err = engineConfig.prefillUBatchTokens()
	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, ubatchTokens)

	plan, err := hipGemma4Q4PlanPromptPrefill([]int32{2, 10979, 2, 10979, 2}, 7, ubatchTokens)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 5, plan.PromptTokens)
	core.AssertEqual(t, 7, plan.StartPos)
	core.AssertEqual(t, 2, plan.UBatchTokens)
	core.AssertEqual(t, 1, plan.OutputTokens)
	core.AssertEqual(t, 12, plan.NextPosition())
	core.AssertEqual(t, 3, len(plan.Batches))
	core.AssertEqual(t, []int32{2, 10979}, plan.Batches[0].Tokens)
	core.AssertEqual(t, 0, len(plan.Batches[0].OutputTokens))
	core.AssertEqual(t, -1, plan.Batches[0].OutputRow)
	core.AssertEqual(t, false, plan.Batches[0].OutputToken(0))
	core.AssertEqual(t, 0, plan.Batches[0].Start)
	core.AssertEqual(t, 2, plan.Batches[0].End)
	core.AssertEqual(t, 7, plan.Batches[0].Position)
	core.AssertEqual(t, []int32{2}, plan.Batches[2].Tokens)
	core.AssertEqual(t, 0, len(plan.Batches[2].OutputTokens))
	core.AssertEqual(t, 0, plan.Batches[2].OutputRow)
	core.AssertEqual(t, true, plan.Batches[2].OutputToken(0))
	core.AssertEqual(t, 4, plan.Batches[2].Start)
	core.AssertEqual(t, 5, plan.Batches[2].End)
	core.AssertEqual(t, 11, plan.Batches[2].Position)
}

func TestHIPGemma4Q4PrefillPlan_Good_SingleBatchInline(t *testing.T) {
	plan, err := hipGemma4Q4PlanPromptPrefill([]int32{2, 10979}, 7, 512)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, plan.LenBatches())
	core.AssertEqual(t, 0, len(plan.Batches))
	batch := plan.Batch(0)
	core.AssertEqual(t, []int32{2, 10979}, batch.Tokens)
	core.AssertEqual(t, 1, batch.OutputRow)
	core.AssertEqual(t, false, batch.OutputToken(0))
	core.AssertEqual(t, true, batch.OutputToken(1))
	core.AssertEqual(t, 7, batch.Position)
}

func TestHIPGemma4Q4PrefillPlanInto_ReusesScratch_Good(t *testing.T) {
	tokens := []int32{2, 10979, 2, 10979, 2}
	scratch := make([]hipGemma4Q4PrefillUBatch, 0, 4)
	plan, reused, err := hipGemma4Q4PlanPromptPrefillInto(tokens, 7, 2, scratch)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 3, len(plan.Batches))
	core.AssertEqual(t, 3, len(reused))
	if cap(reused) != cap(scratch) {
		t.Fatalf("reused scratch cap = %d, want original cap %d", cap(reused), cap(scratch))
	}
	core.AssertEqual(t, []int32{2, 10979}, plan.Batches[0].Tokens)
	core.AssertEqual(t, []int32{2}, plan.Batches[2].Tokens)

	single, returned, err := hipGemma4Q4PlanPromptPrefillInto(tokens[:1], 12, 512, reused)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, single.LenBatches())
	core.AssertEqual(t, 0, len(single.Batches))
	core.AssertEqual(t, 0, len(returned))
}

func TestHIPGemma4Q4PrefillPlanBidirectionalSpans_Good(t *testing.T) {
	tokens := []int32{1, 22, 22, 22, 2, 23, 23, 3}
	spans := hipGemma4Q4BidirectionalTokenSpans(tokens, [2]int32{22, 23})
	core.AssertEqual(t, [][2]int{{1, 4}, {5, 7}}, spans)

	plan, _, err := hipGemma4Q4PlanPromptPrefillBidirectionalInto(tokens, 5, 3, spans, nil)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 4, plan.LenBatches())

	wantRanges := [][2]int{{0, 1}, {1, 4}, {4, 7}, {7, 8}}
	wantCaps := [][]int32{nil, {9, 9, 9}, {10, 12, 12}, nil}
	for index := range wantRanges {
		batch := plan.Batch(index)
		core.AssertEqual(t, wantRanges[index][0], batch.Start)
		core.AssertEqual(t, wantRanges[index][1], batch.End)
		core.AssertEqual(t, wantCaps[index], batch.AttentionCaps)
	}
	core.AssertEqual(t, 0, plan.Batch(plan.LenBatches()-1).OutputRow)
}

func TestHIPGemma4Q4PrefillPlanBidirectionalSpans_Ugly(t *testing.T) {
	tokens := []int32{22, 22, 22, 22, 7}
	spans := hipGemma4Q4BidirectionalTokenSpans(tokens, [2]int32{22, 23})
	plan, _, err := hipGemma4Q4PlanPromptPrefillBidirectionalInto(tokens, 0, 2, spans, nil)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, plan.LenBatches())
	core.AssertEqual(t, 4, len(plan.Batch(0).Tokens))
	core.AssertEqual(t, []int32{4, 4, 4, 4}, plan.Batch(0).AttentionCaps)
}

func BenchmarkHIPGemma4Q4PlanPromptPrefill_29K(b *testing.B) {
	tokens := make([]int32, 29000)
	for index := range tokens {
		tokens[index] = int32(index%32000 + 1)
	}
	wantBatches := (len(tokens) + hipGemma4Q4PrefillDefaultUBatchTokens - 1) / hipGemma4Q4PrefillDefaultUBatchTokens
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		plan, err := hipGemma4Q4PlanPromptPrefill(tokens, 0, hipGemma4Q4PrefillDefaultUBatchTokens)
		if err != nil {
			b.Fatalf("hipGemma4Q4PlanPromptPrefill: %v", err)
		}
		if plan.PromptTokens != len(tokens) || len(plan.Batches) != wantBatches {
			b.Fatalf("plan = tokens %d batches %d, want 29000/%d", plan.PromptTokens, len(plan.Batches), wantBatches)
		}
	}
}

func BenchmarkHIPGemma4Q4PlanPromptPrefillInto_29K_Reused(b *testing.B) {
	tokens := make([]int32, 29000)
	for index := range tokens {
		tokens[index] = int32(index%32000 + 1)
	}
	wantBatches := (len(tokens) + hipGemma4Q4PrefillDefaultUBatchTokens - 1) / hipGemma4Q4PrefillDefaultUBatchTokens
	var scratch []hipGemma4Q4PrefillUBatch
	plan, reused, err := hipGemma4Q4PlanPromptPrefillInto(tokens, 0, hipGemma4Q4PrefillDefaultUBatchTokens, scratch)
	if err != nil {
		b.Fatalf("hipGemma4Q4PlanPromptPrefillInto warmup: %v", err)
	}
	if plan.PromptTokens != len(tokens) || len(plan.Batches) != wantBatches {
		b.Fatalf("warmup plan = tokens %d batches %d, want 29000/%d", plan.PromptTokens, len(plan.Batches), wantBatches)
	}
	scratch = reused
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		plan, scratch, err = hipGemma4Q4PlanPromptPrefillInto(tokens, 0, hipGemma4Q4PrefillDefaultUBatchTokens, scratch)
		if err != nil {
			b.Fatalf("hipGemma4Q4PlanPromptPrefillInto: %v", err)
		}
		if plan.PromptTokens != len(tokens) || len(plan.Batches) != wantBatches {
			b.Fatalf("plan = tokens %d batches %d, want 29000/%d", plan.PromptTokens, len(plan.Batches), wantBatches)
		}
	}
}

func BenchmarkHIPGemma4Q4PlanPromptPrefillInto_29K_Pooled(b *testing.B) {
	tokens := make([]int32, 29000)
	for index := range tokens {
		tokens[index] = int32(index%32000 + 1)
	}
	wantBatches := (len(tokens) + hipGemma4Q4PrefillDefaultUBatchTokens - 1) / hipGemma4Q4PrefillDefaultUBatchTokens
	scratch := hipBorrowGemma4Q4PrefillUBatches(wantBatches)
	hipReleaseGemma4Q4PrefillUBatches(scratch)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		scratch = hipBorrowGemma4Q4PrefillUBatches(wantBatches)
		plan, reused, err := hipGemma4Q4PlanPromptPrefillInto(tokens, 0, hipGemma4Q4PrefillDefaultUBatchTokens, scratch)
		if err != nil {
			hipReleaseGemma4Q4PrefillUBatches(reused)
			b.Fatalf("hipGemma4Q4PlanPromptPrefillInto: %v", err)
		}
		if plan.PromptTokens != len(tokens) || len(plan.Batches) != wantBatches {
			hipReleaseGemma4Q4PrefillUBatches(reused)
			b.Fatalf("plan = tokens %d batches %d, want 29000/%d", plan.PromptTokens, len(plan.Batches), wantBatches)
		}
		hipReleaseGemma4Q4PrefillUBatches(reused)
	}
}

func BenchmarkHIPGemma4Q4PlanPromptPrefill_SingleBatch(b *testing.B) {
	tokens := []int32{2, 10979}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		plan, err := hipGemma4Q4PlanPromptPrefill(tokens, 0, hipGemma4Q4PrefillDefaultUBatchTokens)
		if err != nil {
			b.Fatalf("hipGemma4Q4PlanPromptPrefill: %v", err)
		}
		if plan.PromptTokens != len(tokens) || plan.LenBatches() != 1 || len(plan.Batches) != 0 {
			b.Fatalf("plan = tokens %d batches %d/%d, want 2 1/0", plan.PromptTokens, plan.LenBatches(), len(plan.Batches))
		}
	}
}

func BenchmarkHIPGemma4Q4TokenPromptIDs_2K(b *testing.B) {
	prompt := inferenceBenchmarkTokenPrompt(2048, []int{2, 10979})
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		tokens, matched, err := hipGemma4Q4TokenPromptIDs(prompt, 32000)
		if err != nil {
			b.Fatalf("hipGemma4Q4TokenPromptIDs: %v", err)
		}
		if !matched || len(tokens) != 2048 {
			b.Fatalf("tokens matched=%t len=%d, want true/2048", matched, len(tokens))
		}
	}
}

func BenchmarkHIPGemma4Q4DeviceLayerCaches_Reused(b *testing.B) {
	state := &hipGemma4Q4DeviceDecodeState{layers: make([]hipGemma4Q4DeviceLayerKVState, 35)}
	for index := range state.layers {
		state.layers[index].cache = &rocmDeviceKVCache{}
	}
	scratch := make([]*rocmDeviceKVCache, 0, len(state.layers))
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		scratch = hipGemma4Q4DeviceLayerCaches(state, scratch, len(state.layers))
		if len(scratch) != len(state.layers) || scratch[0] == nil {
			b.Fatalf("layer cache scratch len=%d first=%v", len(scratch), scratch[0])
		}
	}
}

func BenchmarkHIPGemma4Q4DeviceLayerDescriptorTables_Reused(b *testing.B) {
	state := &hipGemma4Q4DeviceDecodeState{layers: make([]hipGemma4Q4DeviceLayerKVState, 35)}
	for index := range state.layers {
		state.layers[index].descriptorTable = &rocmDeviceKVDescriptorTable{}
	}
	scratch := make([]*rocmDeviceKVDescriptorTable, 0, len(state.layers))
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		scratch = hipGemma4Q4DeviceLayerDescriptorTables(state, scratch, len(state.layers))
		if len(scratch) != len(state.layers) || scratch[0] == nil {
			b.Fatalf("layer descriptor scratch len=%d first=%v", len(scratch), scratch[0])
		}
	}
}

func TestHIPGemma4Q4PrefillPlan_Bad(t *testing.T) {
	engineConfig := defaultHIPGemma4Q4EngineConfig()
	engineConfig.PrefillUBatchTokens = 0
	if _, err := engineConfig.prefillUBatchTokens(); err == nil {
		t.Fatalf("prefillUBatchTokens succeeded, want invalid engine config error")
	}
	if _, err := hipGemma4Q4PlanPromptPrefill(nil, 0, 512); err == nil {
		t.Fatalf("hipGemma4Q4PlanPromptPrefill succeeded with empty prompt")
	}
	if _, err := hipGemma4Q4PlanPromptPrefill([]int32{1}, -1, 512); err == nil {
		t.Fatalf("hipGemma4Q4PlanPromptPrefill succeeded with negative start position")
	}
	if _, err := hipGemma4Q4PlanPromptPrefill([]int32{1}, 0, 0); err == nil {
		t.Fatalf("hipGemma4Q4PlanPromptPrefill succeeded with zero ubatch size")
	}
}

func TestHIPGemma4Q4PrefillEmbeddingBatch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup()

	tokens := []int32{1, 0, 1}
	start := len(driver.launches)
	output, err := hipRunGemma4Q4PrefillEmbeddingBatch(context.Background(), driver, cfg, tokens)
	core.RequireNoError(t, err)
	defer output.Close()

	wantCount := len(tokens) * cfg.HiddenSize
	core.AssertEqual(t, wantCount, output.Count())
	core.AssertEqual(t, uint64(wantCount*4), output.SizeBytes())

	launches := driver.launches[start:]
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameEmbedLookup))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameVectorScale))
	wantScale := hipBFloat16ToFloat32(hipFloat32ToBFloat16(float32(math.Sqrt(float64(cfg.HiddenSize)))))
	for _, launch := range launches {
		switch launch.Name {
		case hipKernelNameEmbedLookup:
			core.AssertEqual(t, uint32(len(tokens)), binary.LittleEndian.Uint32(launch.Args[32:]))
			core.AssertEqual(t, uint32(cfg.HiddenSize), binary.LittleEndian.Uint32(launch.Args[40:]))
			core.AssertEqual(t, uint64(wantCount*4), binary.LittleEndian.Uint64(launch.Args[56:]))
		case hipKernelNameVectorScale:
			core.AssertEqual(t, uint32(wantCount), binary.LittleEndian.Uint32(launch.Args[24:]))
			core.AssertEqual(t, uint32(wantCount*4), binary.LittleEndian.Uint32(launch.Args[28:]))
			core.AssertEqual(t, uint32(wantCount*4), binary.LittleEndian.Uint32(launch.Args[32:]))
			assertFloat32Near(t, wantScale, math.Float32frombits(binary.LittleEndian.Uint32(launch.Args[36:])))
		}
	}
}

func TestHIPGemma4Q4PrefillEmbeddingBatchWorkspace_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup()

	tokens := []int32{1, 0, 1}
	tokenBuffer, err := hipUploadTokenIDs(driver, tokens)
	core.RequireNoError(t, err)
	defer tokenBuffer.Close()

	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	start := len(driver.launches)
	output, err := hipRunGemma4Q4PrefillEmbeddingBatchTokenBufferWorkspace(context.Background(), driver, cfg, tokens, tokenBuffer, workspace)
	core.RequireNoError(t, err)
	defer output.Close()

	wantCount := len(tokens) * cfg.HiddenSize
	core.AssertEqual(t, wantCount, output.Count())
	core.AssertEqual(t, uint64(wantCount*4), output.SizeBytes())
	core.AssertEqual(t, true, output.borrowed)
	core.AssertEqual(t, output.Pointer(), workspace.ScaledEmbeddingView.Pointer())

	launches := driver.launches[start:]
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameEmbedLookup))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameVectorScale))
	wantScale := hipBFloat16ToFloat32(hipFloat32ToBFloat16(float32(math.Sqrt(float64(cfg.HiddenSize)))))
	for _, launch := range launches {
		if launch.Name != hipKernelNameEmbedLookup {
			continue
		}
		core.AssertEqual(t, uint32(len(tokens)), binary.LittleEndian.Uint32(launch.Args[32:]))
		core.AssertEqual(t, uint32(cfg.HiddenSize), binary.LittleEndian.Uint32(launch.Args[40:]))
		core.AssertEqual(t, uint64(wantCount*4), binary.LittleEndian.Uint64(launch.Args[56:]))
		assertFloat32Near(t, wantScale, math.Float32frombits(binary.LittleEndian.Uint32(launch.Args[96:])))
	}
}

func TestHIPAttentionHeadsChunkedWorkspace_ScaledEmbeddingReusesLarger_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	large, err := workspace.EnsureScaledEmbedding(driver, 3072)
	core.RequireNoError(t, err)
	small, err := workspace.EnsureScaledEmbedding(driver, 1536)
	core.RequireNoError(t, err)

	core.AssertEqual(t, 1, len(driver.allocations))
	core.AssertEqual(t, large.Pointer(), small.Pointer())
	core.AssertEqual(t, 1536, small.Count())
	core.AssertEqual(t, uint64(1536*4), small.SizeBytes())
	core.AssertEqual(t, true, small.borrowed)
	if _, ok := workspace.ScaledEmbeddings[1536]; ok {
		t.Fatalf("smaller scaled embedding got a dedicated allocation")
	}
}

func TestHIPGemma4Q4PrefillEmbeddingBatch_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup()

	if _, err := hipRunGemma4Q4PrefillEmbeddingBatch(context.Background(), driver, cfg, nil); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillEmbeddingBatch succeeded with empty tokens")
	}
	core.AssertEqual(t, 0, len(driver.launches))

	unavailable := &fakeHIPDriver{available: false}
	if _, err := hipRunGemma4Q4PrefillEmbeddingBatch(context.Background(), unavailable, cfg, []int32{1}); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillEmbeddingBatch succeeded with unavailable driver")
	}
	core.AssertEqual(t, 0, len(unavailable.launches))
}

func TestHIPGemma4Q4PrefillInputNormBatch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup()

	tokenCount := 3
	inputValues := make([]float32, tokenCount*cfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(index%cfg.HiddenSize + 1)
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill input norm fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()

	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	start := len(driver.launches)
	output, err := hipRunGemma4Q4PrefillInputNormBatchWorkspace(context.Background(), driver, cfg, input, tokenCount, workspace)
	core.RequireNoError(t, err)
	defer output.Close()

	wantCount := tokenCount * cfg.HiddenSize
	core.AssertEqual(t, wantCount, output.Count())
	core.AssertEqual(t, uint64(wantCount*4), output.SizeBytes())
	core.AssertEqual(t, true, output.borrowed)
	core.AssertEqual(t, output.Pointer(), workspace.PrefillInputNormView.Pointer())

	launches := driver.launches[start:]
	core.AssertEqual(t, 1, len(launches))
	launch := launches[0]
	core.AssertEqual(t, hipKernelNameRMSNormHeads, launch.Name)
	core.AssertEqual(t, uint32(tokenCount), launch.GridX)
	core.AssertEqual(t, uint32(cfg.HiddenSize), binary.LittleEndian.Uint32(launch.Args[32:]))
	core.AssertEqual(t, uint32(tokenCount), binary.LittleEndian.Uint32(launch.Args[36:]))
	core.AssertEqual(t, uint32(wantCount*4), binary.LittleEndian.Uint32(launch.Args[40:]))
	core.AssertEqual(t, uint32(wantCount*4), binary.LittleEndian.Uint32(launch.Args[48:]))
	assertFloat32Near(t, cfg.InputNorm.Epsilon, math.Float32frombits(binary.LittleEndian.Uint32(launch.Args[52:])))
	core.AssertEqual(t, hipRMSNormWeightEncodingBF16, binary.LittleEndian.Uint32(launch.Args[56:]))
}

func TestHIPGemma4Q4PrefillInputNormBatch_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup()

	inputValues := make([]float32, cfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(index + 1)
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill input norm bad fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()
	start := len(driver.launches)

	if _, err := hipRunGemma4Q4PrefillInputNormBatch(context.Background(), driver, cfg, input, 0); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillInputNormBatch succeeded with zero token count")
	}
	if _, err := hipRunGemma4Q4PrefillInputNormBatch(context.Background(), driver, cfg, input, 2); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillInputNormBatch succeeded with mismatched token count")
	}
	core.AssertEqual(t, start, len(driver.launches))
}

func TestHIPGemma4Q4PrefillQKVProjectionBatch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup()

	tokenCount := 2
	inputValues := make([]float32, tokenCount*cfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(index%cfg.HiddenSize + 1)
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill QKV fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()

	start := len(driver.launches)
	qkv, err := hipRunGemma4Q4PrefillQKVProjectionBatch(context.Background(), driver, cfg, input, tokenCount)
	core.RequireNoError(t, err)
	defer qkv.Close()

	core.AssertEqual(t, tokenCount*cfg.QueryProjection.Rows, qkv.Query.Count())
	core.AssertEqual(t, tokenCount*cfg.KeyProjection.Rows, qkv.Key.Count())
	core.AssertEqual(t, tokenCount*cfg.ValueProjection.Rows, qkv.Value.Count())
	launches := driver.launches[start:]
	core.AssertEqual(t, 3, countLaunchName(launches, hipKernelNameMLXQ4ProjBatch))
	wantRows := []int{cfg.QueryProjection.Rows, cfg.KeyProjection.Rows, cfg.ValueProjection.Rows}
	for index, launch := range launches {
		core.AssertEqual(t, hipKernelNameMLXQ4ProjBatch, launch.Name)
		core.AssertEqual(t, uint32((tokenCount+hipMLXQ4ProjectionBatchTokensPerBlock-1)/hipMLXQ4ProjectionBatchTokensPerBlock), launch.GridY)
		core.AssertEqual(t, uint32(wantRows[index]), binary.LittleEndian.Uint32(launch.Args[48:]))
		core.AssertEqual(t, uint32(cfg.HiddenSize), binary.LittleEndian.Uint32(launch.Args[52:]))
		core.AssertEqual(t, uint32(tokenCount), binary.LittleEndian.Uint32(launch.Args[56:]))
	}
}

func TestHIPGemma4Q4PrefillQKVProjectionBatch_BatchOneUsesTripleProjection_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup()

	tokenCount := 1
	inputValues := make([]float32, cfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(index + 1)
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill batch-one QKV fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()

	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	start := len(driver.launches)
	qkv, err := hipRunGemma4Q4PrefillQKVProjectionBatchWorkspace(context.Background(), driver, cfg, input, tokenCount, workspace)
	core.RequireNoError(t, err)
	defer qkv.Close()

	core.AssertEqual(t, cfg.QueryProjection.Rows, qkv.Query.Count())
	core.AssertEqual(t, cfg.KeyProjection.Rows, qkv.Key.Count())
	core.AssertEqual(t, cfg.ValueProjection.Rows, qkv.Value.Count())
	core.AssertEqual(t, true, qkv.Query.borrowed)
	core.AssertEqual(t, true, qkv.Key.borrowed)
	core.AssertEqual(t, true, qkv.Value.borrowed)
	launches := driver.launches[start:]
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameMLXQ4TripleProj))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameMLXQ4ProjBatch))
	if workspace.ProjectionOutputFixed.Count() < cfg.QueryProjection.Rows+cfg.KeyProjection.Rows+cfg.ValueProjection.Rows {
		t.Fatalf("workspace projection rows = %d, want at least %d", workspace.ProjectionOutputFixed.Count(), cfg.QueryProjection.Rows+cfg.KeyProjection.Rows+cfg.ValueProjection.Rows)
	}
	core.AssertEqual(t, qkv.Query.Pointer()+nativeDevicePointer(cfg.QueryProjection.Rows*4), qkv.Key.Pointer())
	core.AssertEqual(t, qkv.Key.Pointer()+nativeDevicePointer(cfg.KeyProjection.Rows*4), qkv.Value.Pointer())
}

func TestHIPGemma4Q4PrefillQKVProjectionBatchTripleCompatible_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup()

	core.AssertEqual(t, true, hipGemma4Q4PrefillQKVProjectionBatchTripleCompatible(cfg))
	cfg.ValueProjection.Bits = cfg.QueryProjection.quantBits() + 2
	core.AssertEqual(t, false, hipGemma4Q4PrefillQKVProjectionBatchTripleCompatible(cfg))
}

func TestHIPGemma4Q4PrefillQKVProjectionBatch_AttentionKEqVBorrowsKeyProjection_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup()
	cfg.LayerType = "full_attention"
	cfg.AttentionKEqV = true
	cfg.ValueProjection = cfg.KeyProjection

	tokenCount := 2
	inputValues := make([]float32, tokenCount*cfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(index%cfg.HiddenSize + 1)
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill K=V QKV fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()

	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	start := len(driver.launches)
	qkv, err := hipRunGemma4Q4PrefillQKVProjectionBatchWorkspace(context.Background(), driver, cfg, input, tokenCount, workspace)
	core.RequireNoError(t, err)
	defer qkv.Close()

	core.AssertEqual(t, tokenCount*cfg.QueryProjection.Rows, qkv.Query.Count())
	core.AssertEqual(t, tokenCount*cfg.KeyProjection.Rows, qkv.Key.Count())
	core.AssertEqual(t, tokenCount*cfg.ValueProjection.Rows, qkv.Value.Count())
	core.AssertEqual(t, true, qkv.Query.borrowed)
	core.AssertEqual(t, false, qkv.Key.borrowed)
	core.AssertEqual(t, qkv.Key.Pointer(), qkv.Value.Pointer())
	core.AssertEqual(t, true, qkv.Value.borrowed)
	core.AssertEqual(t, qkv.Query.Pointer(), workspace.ProjectionOutputFixed.Pointer())
	launches := driver.launches[start:]
	core.AssertEqual(t, 2, countLaunchName(launches, hipKernelNameMLXQ4ProjBatch))
	wantRows := []int{cfg.QueryProjection.Rows, cfg.KeyProjection.Rows}
	for index, launch := range launches {
		core.AssertEqual(t, hipKernelNameMLXQ4ProjBatch, launch.Name)
		core.AssertEqual(t, uint32(wantRows[index]), binary.LittleEndian.Uint32(launch.Args[48:]))
	}
}

func TestHIPGemma4Q4PrefillQKVProjectionBatch_AttentionKEqVBatchOneUsesPairProjection_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup()
	cfg.LayerType = "full_attention"
	cfg.AttentionKEqV = true
	cfg.ValueProjection = cfg.KeyProjection

	tokenCount := 1
	inputValues := make([]float32, cfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(index + 1)
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill K=V batch-one QKV fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()

	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	start := len(driver.launches)
	qkv, err := hipRunGemma4Q4PrefillQKVProjectionBatchWorkspace(context.Background(), driver, cfg, input, tokenCount, workspace)
	core.RequireNoError(t, err)
	defer qkv.Close()

	core.AssertEqual(t, cfg.QueryProjection.Rows, qkv.Query.Count())
	core.AssertEqual(t, cfg.KeyProjection.Rows, qkv.Key.Count())
	core.AssertEqual(t, cfg.ValueProjection.Rows, qkv.Value.Count())
	core.AssertEqual(t, true, qkv.Query.borrowed)
	core.AssertEqual(t, true, qkv.Key.borrowed)
	core.AssertEqual(t, true, qkv.Value.borrowed)
	core.AssertEqual(t, qkv.Key.Pointer(), qkv.Value.Pointer())
	launches := driver.launches[start:]
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameMLXQ4PairProj))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameMLXQ4ProjBatch))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameMLXQ4Proj))
}

func TestHIPGemma4Q4PrefillQKVProjectionBatch_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup()

	inputValues := make([]float32, cfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(index + 1)
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill QKV bad fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()
	start := len(driver.launches)

	if _, err := hipRunGemma4Q4PrefillQKVProjectionBatch(context.Background(), driver, cfg, input, 0); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillQKVProjectionBatch succeeded with zero token count")
	}
	if _, err := hipRunGemma4Q4PrefillQKVProjectionBatch(context.Background(), driver, cfg, input, 2); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillQKVProjectionBatch succeeded with mismatched token count")
	}
	core.AssertEqual(t, start, len(driver.launches))
}

func TestHIPGemma4Q4PrefillQKNormRoPEBatch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()
	cfg.RoPERotaryDim = 2
	cfg.RoPEFrequencyScale = 0.5

	tokenCount := 2
	queryValues := make([]float32, tokenCount*cfg.QueryHeads*cfg.HeadDim)
	for index := range queryValues {
		queryValues[index] = float32(index%cfg.HeadDim + 1)
	}
	keyValues := make([]float32, tokenCount*cfg.HeadDim)
	for index := range keyValues {
		keyValues[index] = float32(index%cfg.HeadDim + 1)
	}
	queryPayload, err := hipFloat32Payload(queryValues)
	core.RequireNoError(t, err)
	query, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill Q/K RoPE query fixture", queryPayload, len(queryValues))
	core.RequireNoError(t, err)
	defer query.Close()
	keyPayload, err := hipFloat32Payload(keyValues)
	core.RequireNoError(t, err)
	key, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill Q/K RoPE key fixture", keyPayload, len(keyValues))
	core.RequireNoError(t, err)
	defer key.Close()
	qkv := &hipGemma4Q4PrefillQKVBatch{Query: query, Key: key}

	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	start := len(driver.launches)
	output, err := hipRunGemma4Q4PrefillQKNormRoPEBatchWorkspace(context.Background(), driver, cfg, qkv, tokenCount, 5, 1e-6, workspace)
	core.RequireNoError(t, err)
	defer output.Close()

	core.AssertEqual(t, tokenCount*cfg.QueryHeads*cfg.HeadDim, output.Query.Count())
	core.AssertEqual(t, tokenCount*cfg.HeadDim, output.Key.Count())
	core.AssertEqual(t, true, output.Query.borrowed)
	core.AssertEqual(t, false, output.Key.borrowed)
	core.AssertEqual(t, output.Query.Pointer(), workspace.RMSRoPEOutputView.Pointer())
	launches := driver.launches[start:]
	core.AssertEqual(t, 2, len(launches))
	core.AssertEqual(t, hipKernelNameRMSNormRoPEHeadsBatch, launches[0].Name)
	core.AssertEqual(t, hipKernelNameRMSNormRoPEHeadsBatch, launches[1].Name)
	core.AssertEqual(t, uint32(cfg.QueryHeads), launches[0].GridX)
	core.AssertEqual(t, uint32(tokenCount), launches[0].GridY)
	core.AssertEqual(t, uint32(1), launches[1].GridX)
	core.AssertEqual(t, uint32(tokenCount), launches[1].GridY)
	for index, launch := range launches {
		wantHeads := cfg.QueryHeads
		if index == 1 {
			wantHeads = 1
		}
		core.AssertEqual(t, uint32(cfg.HeadDim), binary.LittleEndian.Uint32(launch.Args[32:]))
		core.AssertEqual(t, uint32(wantHeads), binary.LittleEndian.Uint32(launch.Args[36:]))
		core.AssertEqual(t, uint32(tokenCount), binary.LittleEndian.Uint32(launch.Args[40:]))
		assertFloat32Near(t, 1e-6, math.Float32frombits(binary.LittleEndian.Uint32(launch.Args[56:])))
		core.AssertEqual(t, hipRMSNormWeightEncodingBF16, binary.LittleEndian.Uint32(launch.Args[60:]))
		core.AssertEqual(t, uint32(5), binary.LittleEndian.Uint32(launch.Args[68:]))
		core.AssertEqual(t, uint32(cfg.HeadDim), binary.LittleEndian.Uint32(launch.Args[76:]))
		core.AssertEqual(t, uint32(cfg.RoPERotaryDim), binary.LittleEndian.Uint32(launch.Args[80:]))
		assertFloat32Near(t, cfg.RoPEFrequencyScale, math.Float32frombits(binary.LittleEndian.Uint32(launch.Args[84:])))
	}
}

func TestHIPGemma4Q4PrefillQKNormRoPEBatch_BatchOneUsesSingleRoPE_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()
	cfg.RoPERotaryDim = 2
	cfg.RoPEFrequencyScale = 0.5

	tokenCount := 1
	queryValues := make([]float32, cfg.QueryHeads*cfg.HeadDim)
	for index := range queryValues {
		queryValues[index] = float32(index%cfg.HeadDim + 1)
	}
	keyValues := make([]float32, cfg.HeadDim)
	for index := range keyValues {
		keyValues[index] = float32(index%cfg.HeadDim + 1)
	}
	queryPayload, err := hipFloat32Payload(queryValues)
	core.RequireNoError(t, err)
	query, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill batch-one Q/K RoPE query fixture", queryPayload, len(queryValues))
	core.RequireNoError(t, err)
	defer query.Close()
	keyPayload, err := hipFloat32Payload(keyValues)
	core.RequireNoError(t, err)
	key, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill batch-one Q/K RoPE key fixture", keyPayload, len(keyValues))
	core.RequireNoError(t, err)
	defer key.Close()
	qkv := &hipGemma4Q4PrefillQKVBatch{Query: query, Key: key}

	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	start := len(driver.launches)
	output, err := hipRunGemma4Q4PrefillQKNormRoPEBatchWorkspace(context.Background(), driver, cfg, qkv, tokenCount, 5, 1e-6, workspace)
	core.RequireNoError(t, err)
	defer output.Close()

	core.AssertEqual(t, cfg.QueryHeads*cfg.HeadDim, output.Query.Count())
	core.AssertEqual(t, cfg.HeadDim, output.Key.Count())
	core.AssertEqual(t, true, output.Query.borrowed)
	core.AssertEqual(t, false, output.Key.borrowed)
	launches := driver.launches[start:]
	core.AssertEqual(t, 2, len(launches))
	core.AssertEqual(t, 2, countLaunchName(launches, hipKernelNameRMSNormRoPEHeads))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameRMSNormRoPEHeadsBatch))
	core.AssertEqual(t, uint32(cfg.QueryHeads), launches[0].GridX)
	core.AssertEqual(t, uint32(1), launches[0].GridY)
	core.AssertEqual(t, uint32(1), launches[1].GridX)
	core.AssertEqual(t, uint32(1), launches[1].GridY)
}

func TestHIPGemma4Q4PrefillQKNormRoPEBatch_BatchOneBorrowsRawKVUsesPairedRoPE_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()
	cfg.RoPERotaryDim = 2
	cfg.RoPEFrequencyScale = 0.5

	tokenCount := 1
	queryValues := make([]float32, cfg.QueryHeads*cfg.HeadDim)
	for index := range queryValues {
		queryValues[index] = float32(index%cfg.HeadDim + 1)
	}
	keyValues := make([]float32, cfg.HeadDim)
	for index := range keyValues {
		keyValues[index] = float32(index%cfg.HeadDim + 1)
	}
	queryPayload, err := hipFloat32Payload(queryValues)
	core.RequireNoError(t, err)
	query, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill paired Q/K RoPE query fixture", queryPayload, len(queryValues))
	core.RequireNoError(t, err)
	defer query.Close()
	keyPayload, err := hipFloat32Payload(keyValues)
	core.RequireNoError(t, err)
	key, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill paired Q/K RoPE key fixture", keyPayload, len(keyValues))
	core.RequireNoError(t, err)
	defer key.Close()
	qkv := &hipGemma4Q4PrefillQKVBatch{Query: query, Key: key}

	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	start := len(driver.launches)
	output, err := hipRunGemma4Q4PrefillQKNormRoPEBatchWorkspaceTransient(context.Background(), driver, cfg, qkv, tokenCount, 5, 1e-6, workspace, true)
	core.RequireNoError(t, err)
	defer output.Close()

	core.AssertEqual(t, cfg.QueryHeads*cfg.HeadDim, output.Query.Count())
	core.AssertEqual(t, cfg.HeadDim, output.Key.Count())
	core.AssertEqual(t, true, output.Query.borrowed)
	core.AssertEqual(t, true, output.Key.borrowed)
	launches := driver.launches[start:]
	core.AssertEqual(t, 1, len(launches))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameRMSNormRoPEHeadsPair))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameRMSNormRoPEHeads))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameRMSNormRoPEHeadsBatch))
	core.AssertEqual(t, uint32(cfg.QueryHeads+1), launches[0].GridX)
	core.AssertEqual(t, uint32(1), launches[0].GridY)
}

func TestHIPGemma4Q4PrefillQKNormRoPEBatch_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()

	tokenCount := 1
	queryPayload, err := hipFloat32Payload(make([]float32, tokenCount*cfg.QueryHeads*cfg.HeadDim))
	core.RequireNoError(t, err)
	query, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill Q/K RoPE bad query fixture", queryPayload, tokenCount*cfg.QueryHeads*cfg.HeadDim)
	core.RequireNoError(t, err)
	defer query.Close()
	keyPayload, err := hipFloat32Payload(make([]float32, tokenCount*cfg.HeadDim))
	core.RequireNoError(t, err)
	key, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill Q/K RoPE bad key fixture", keyPayload, tokenCount*cfg.HeadDim)
	core.RequireNoError(t, err)
	defer key.Close()
	qkv := &hipGemma4Q4PrefillQKVBatch{Query: query, Key: key}
	start := len(driver.launches)

	if _, err := hipRunGemma4Q4PrefillQKNormRoPEBatch(context.Background(), driver, cfg, qkv, 0, 0, 1e-6); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillQKNormRoPEBatch succeeded with zero token count")
	}
	if _, err := hipRunGemma4Q4PrefillQKNormRoPEBatch(context.Background(), driver, cfg, qkv, 2, 0, 1e-6); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillQKNormRoPEBatch succeeded with mismatched token count")
	}
	cfg.RoPERotaryDim = 3
	if _, err := hipRunGemma4Q4PrefillQKNormRoPEBatch(context.Background(), driver, cfg, qkv, tokenCount, 0, 1e-6); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillQKNormRoPEBatch succeeded with odd rotary dimension")
	}
	core.AssertEqual(t, start, len(driver.launches))
}

func TestHIPGemma4Q4PrefillValueNormBatch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()

	tokenCount := 2
	valueValues := []float32{
		1, 0, 3, 4,
		0, 2, 5, 12,
	}
	valuePayload, err := hipFloat32Payload(valueValues)
	core.RequireNoError(t, err)
	value, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill value norm fixture", valuePayload, len(valueValues))
	core.RequireNoError(t, err)
	defer value.Close()
	qkv := &hipGemma4Q4PrefillQKVBatch{Value: value}

	start := len(driver.launches)
	output, err := hipRunGemma4Q4PrefillValueNormBatch(context.Background(), driver, cfg, qkv, tokenCount, 1e-6)
	core.RequireNoError(t, err)
	defer output.Close()
	values, err := hipReadFloat32DeviceOutput(output, hipGemma4Q4Layer0Operation, "prefill value norm output", len(valueValues))
	core.RequireNoError(t, err)

	var want []float32
	unitWeight := []float32{1, 1, 1, 1}
	for token := 0; token < tokenCount; token++ {
		offset := token * cfg.HeadDim
		normalized, err := hipReferenceRMSNorm(valueValues[offset:offset+cfg.HeadDim], unitWeight, 1e-6)
		core.RequireNoError(t, err)
		want = append(want, normalized...)
	}
	assertFloat32SlicesNear(t, want, values, 0.0001)

	launches := driver.launches[start:]
	core.AssertEqual(t, 1, len(launches))
	core.AssertEqual(t, hipKernelNameRMSNormHeads, launches[0].Name)
	core.AssertEqual(t, uint32(tokenCount), launches[0].GridX)
	core.AssertEqual(t, uint32(cfg.HeadDim), binary.LittleEndian.Uint32(launches[0].Args[32:]))
	core.AssertEqual(t, uint32(tokenCount), binary.LittleEndian.Uint32(launches[0].Args[36:]))
	core.AssertEqual(t, hipRMSNormWeightEncodingNone, binary.LittleEndian.Uint32(launches[0].Args[56:]))
}

func TestHIPGemma4Q4PrefillValueNormBatch_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()

	valuePayload, err := hipFloat32Payload(make([]float32, cfg.HeadDim))
	core.RequireNoError(t, err)
	value, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill value norm bad fixture", valuePayload, cfg.HeadDim)
	core.RequireNoError(t, err)
	defer value.Close()
	qkv := &hipGemma4Q4PrefillQKVBatch{Value: value}
	start := len(driver.launches)

	if _, err := hipRunGemma4Q4PrefillValueNormBatch(context.Background(), driver, cfg, qkv, 0, 1e-6); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillValueNormBatch succeeded with zero token count")
	}
	if _, err := hipRunGemma4Q4PrefillValueNormBatch(context.Background(), driver, cfg, qkv, 2, 1e-6); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillValueNormBatch succeeded with mismatched token count")
	}
	if _, err := hipRunGemma4Q4PrefillValueNormBatch(context.Background(), driver, cfg, &hipGemma4Q4PrefillQKVBatch{}, 1, 1e-6); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillValueNormBatch succeeded with missing value buffer")
	}
	core.AssertEqual(t, start, len(driver.launches))
}

func TestHIPGemma4Q4PrefillDeviceKVBatch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()

	tokenCount := 3
	keyRows := []float32{
		1, 0, 0, 1,
		0, 1, 1, 0,
		-1, 1, 0.5, -0.5,
	}
	valueRows := []float32{
		2, 0, 0, 2,
		0, 2, 2, 0,
		3, -3, 1, -1,
	}
	keyPayload, err := hipFloat32Payload(keyRows)
	core.RequireNoError(t, err)
	key, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill device KV key fixture", keyPayload, len(keyRows))
	core.RequireNoError(t, err)
	defer key.Close()
	valuePayload, err := hipFloat32Payload(valueRows)
	core.RequireNoError(t, err)
	value, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill device KV value fixture", valuePayload, len(valueRows))
	core.RequireNoError(t, err)
	defer value.Close()
	qk := &hipGemma4Q4PrefillRoPEQKBatch{Key: key}

	start := len(driver.launches)
	deviceKV, err := hipRunGemma4Q4PrefillDeviceKVBatch(context.Background(), driver, cfg, qk, value, tokenCount, rocmKVCacheModeKQ8VQ4)
	core.RequireNoError(t, err)
	defer deviceKV.Close()

	wantPages := gemma4Q4DeviceKVPagesForTokens(tokenCount)
	core.AssertEqual(t, wantPages, countLaunchName(driver.launches[start:], hipKernelNameKVEncodeToken))
	core.AssertEqual(t, tokenCount, deviceKV.Cache.TokenCount())
	core.AssertEqual(t, wantPages, deviceKV.Cache.PageCount())
	core.AssertEqual(t, min(tokenCount, hipGemma4Q4DeviceKVBlockSize()), deviceKV.Cache.pages[0].tokenCount)
	core.AssertEqual(t, cfg.HeadDim, deviceKV.Launch.KeyWidth)
	core.AssertEqual(t, cfg.HeadDim, deviceKV.Launch.ValueWidth)
	core.AssertEqual(t, tokenCount, deviceKV.Launch.TokenCount)
	core.AssertEqual(t, rocmKVCacheModeKQ8VQ4, deviceKV.Launch.Mode)

	descriptorPayload := make([]byte, deviceKV.DescriptorTable.SizeBytes())
	core.RequireNoError(t, driver.CopyDeviceToHost(deviceKV.DescriptorTable.Pointer(), descriptorPayload))
	core.AssertEqual(t, uint64(tokenCount), binary.LittleEndian.Uint64(descriptorPayload[24:]))
	pageOffset := rocmDeviceKVDescriptorHeaderBytes
	core.AssertEqual(t, uint64(0), binary.LittleEndian.Uint64(descriptorPayload[pageOffset:]))
	core.AssertEqual(t, uint64(min(tokenCount, hipGemma4Q4DeviceKVBlockSize())), binary.LittleEndian.Uint64(descriptorPayload[pageOffset+8:]))
	core.AssertEqual(t, uint32(cfg.HeadDim), binary.LittleEndian.Uint32(descriptorPayload[pageOffset+16:]))
	core.AssertEqual(t, uint32(cfg.HeadDim), binary.LittleEndian.Uint32(descriptorPayload[pageOffset+20:]))
}

func TestHIPGemma4Q4PrefillDeviceKVBatchFullAttentionUsesGlobalBlockSize_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()
	cfg.LayerType = "full_attention"
	cfg.SlidingWindow = 0
	tokenCount := rocmGemma4Q4GlobalDeviceKVBlockSize + 2
	keyRows := make([]float32, tokenCount*cfg.HeadDim)
	valueRows := make([]float32, tokenCount*cfg.HeadDim)
	for index := range keyRows {
		keyRows[index] = float32(index%11) - 5
		valueRows[index] = float32(index%7) - 3
	}
	keyPayload, err := hipFloat32Payload(keyRows)
	core.RequireNoError(t, err)
	key, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill full attention device KV key fixture", keyPayload, len(keyRows))
	core.RequireNoError(t, err)
	defer key.Close()
	valuePayload, err := hipFloat32Payload(valueRows)
	core.RequireNoError(t, err)
	value, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill full attention device KV value fixture", valuePayload, len(valueRows))
	core.RequireNoError(t, err)
	defer value.Close()
	qk := &hipGemma4Q4PrefillRoPEQKBatch{Key: key}

	start := len(driver.launches)
	deviceKV, err := hipRunGemma4Q4PrefillDeviceKVBatch(context.Background(), driver, cfg, qk, value, tokenCount, rocmKVCacheModeKQ8VQ4)
	core.RequireNoError(t, err)
	defer deviceKV.Close()

	wantBlockSize := hipGemma4Q4DeviceKVBlockSizeForSlidingWindow(cfg.SlidingWindow)
	wantPages := (tokenCount + wantBlockSize - 1) / wantBlockSize
	core.AssertEqual(t, wantPages, countLaunchName(driver.launches[start:], hipKernelNameKVEncodeToken))
	core.AssertEqual(t, wantBlockSize, deviceKV.Cache.blockSize)
	core.AssertEqual(t, tokenCount, deviceKV.Cache.TokenCount())
	core.AssertEqual(t, wantPages, deviceKV.Cache.PageCount())
	core.AssertEqual(t, wantBlockSize, deviceKV.Cache.pages[0].tokenCount)
	core.AssertEqual(t, 2, deviceKV.Cache.pages[1].tokenCount)

	descriptorPayload := make([]byte, deviceKV.DescriptorTable.SizeBytes())
	core.RequireNoError(t, driver.CopyDeviceToHost(deviceKV.DescriptorTable.Pointer(), descriptorPayload))
	core.AssertEqual(t, uint32(wantBlockSize), binary.LittleEndian.Uint32(descriptorPayload[20:]))
	core.AssertEqual(t, uint64(tokenCount), binary.LittleEndian.Uint64(descriptorPayload[24:]))
}

func TestHIPGemma4Q4PrefillDeviceKVBatch_UsesEngineConfigGlobalBlockSize_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()
	cfg.LayerType = "full_attention"
	cfg.SlidingWindow = 0
	tokenCount := 6
	keyRows := make([]float32, tokenCount*cfg.HeadDim)
	valueRows := make([]float32, tokenCount*cfg.HeadDim)
	for index := range keyRows {
		keyRows[index] = float32(index%11) - 5
		valueRows[index] = float32(index%7) - 3
	}
	keyPayload, err := hipFloat32Payload(keyRows)
	core.RequireNoError(t, err)
	key, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill config device KV key fixture", keyPayload, len(keyRows))
	core.RequireNoError(t, err)
	defer key.Close()
	valuePayload, err := hipFloat32Payload(valueRows)
	core.RequireNoError(t, err)
	value, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill config device KV value fixture", valuePayload, len(valueRows))
	core.RequireNoError(t, err)
	defer value.Close()
	qk := &hipGemma4Q4PrefillRoPEQKBatch{Key: key}
	engineConfig := defaultHIPGemma4Q4EngineConfig()
	engineConfig.GlobalDeviceKVBlockSize = 4

	deviceKV, err := hipRunGemma4Q4PrefillDeviceKVBatchWithPriorDescriptorIntoWithEngineConfig(context.Background(), driver, cfg, nil, nil, qk, value, tokenCount, rocmKVCacheModeKQ8VQ4, nil, engineConfig)
	core.RequireNoError(t, err)
	defer deviceKV.Close()

	core.AssertEqual(t, 4, deviceKV.Cache.blockSize)
	core.AssertEqual(t, tokenCount, deviceKV.Cache.TokenCount())
	core.AssertEqual(t, 2, deviceKV.Cache.PageCount())
	core.AssertEqual(t, 4, deviceKV.Cache.pages[0].tokenCount)
	core.AssertEqual(t, 2, deviceKV.Cache.pages[1].tokenCount)
}

func TestHIPGemma4Q4PrefillDeviceKVBatch_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()

	keyPayload, err := hipFloat32Payload(make([]float32, cfg.HeadDim))
	core.RequireNoError(t, err)
	key, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill device KV bad key fixture", keyPayload, cfg.HeadDim)
	core.RequireNoError(t, err)
	defer key.Close()
	valuePayload, err := hipFloat32Payload(make([]float32, cfg.HeadDim))
	core.RequireNoError(t, err)
	value, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill device KV bad value fixture", valuePayload, cfg.HeadDim)
	core.RequireNoError(t, err)
	defer value.Close()
	qk := &hipGemma4Q4PrefillRoPEQKBatch{Key: key}
	start := len(driver.launches)

	if _, err := hipRunGemma4Q4PrefillDeviceKVBatch(context.Background(), driver, cfg, qk, value, 0, rocmKVCacheModeKQ8VQ4); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillDeviceKVBatch succeeded with zero token count")
	}
	if _, err := hipRunGemma4Q4PrefillDeviceKVBatch(context.Background(), driver, cfg, qk, value, 2, rocmKVCacheModeKQ8VQ4); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillDeviceKVBatch succeeded with mismatched key/value rows")
	}
	if _, err := hipRunGemma4Q4PrefillDeviceKVBatch(context.Background(), driver, cfg, &hipGemma4Q4PrefillRoPEQKBatch{}, value, 1, rocmKVCacheModeKQ8VQ4); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillDeviceKVBatch succeeded with missing key buffer")
	}
	core.AssertEqual(t, start, len(driver.launches))
}

func TestHIPGemma4Q4PrefillLayerKVBatch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()
	cfg.RoPERotaryDim = 2

	tokenCount := 3
	inputValues := make([]float32, tokenCount*cfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(index%cfg.HiddenSize + 1)
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill layer KV input fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()

	start := len(driver.launches)
	layer, err := hipRunGemma4Q4PrefillLayerKVBatch(context.Background(), driver, cfg, input, tokenCount, 7, 1e-6, rocmKVCacheModeKQ8VQ4)
	core.RequireNoError(t, err)
	defer layer.Close()

	core.AssertEqual(t, tokenCount*cfg.HiddenSize, layer.InputNorm.Count())
	core.AssertEqual(t, tokenCount*cfg.QueryProjection.Rows, layer.QKV.Query.Count())
	core.AssertEqual(t, tokenCount*cfg.HeadDim, layer.QK.Key.Count())
	core.AssertEqual(t, tokenCount*cfg.HeadDim, layer.Value.Count())
	core.AssertEqual(t, tokenCount, layer.DeviceKV.Cache.TokenCount())
	core.AssertEqual(t, gemma4Q4DeviceKVPagesForTokens(tokenCount), layer.DeviceKV.Cache.PageCount())

	launches := driver.launches[start:]
	core.AssertEqual(t, 2, countLaunchName(launches, hipKernelNameRMSNormHeads))
	core.AssertEqual(t, 3, countLaunchName(launches, hipKernelNameMLXQ4ProjBatch))
	core.AssertEqual(t, 2, countLaunchName(launches, hipKernelNameRMSNormRoPEHeadsBatch))
	core.AssertEqual(t, gemma4Q4DeviceKVPagesForTokens(tokenCount), countLaunchName(launches, hipKernelNameKVEncodeToken))
	core.AssertEqual(t, tokenCount, layer.DeviceKV.Launch.TokenCount)
	core.AssertEqual(t, cfg.HeadDim, layer.DeviceKV.Launch.KeyWidth)
	core.AssertEqual(t, cfg.HeadDim, layer.DeviceKV.Launch.ValueWidth)
}

func TestHIPGemma4Q4PrefillAttentionBatch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()
	cfg.RoPERotaryDim = 2

	tokenCount := 3
	inputValues := make([]float32, tokenCount*cfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(index%cfg.HiddenSize + 1)
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill attention input fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()

	layer, err := hipRunGemma4Q4PrefillLayerKVBatch(context.Background(), driver, cfg, input, tokenCount, 0, 1e-6, rocmKVCacheModeKQ8VQ4)
	core.RequireNoError(t, err)
	defer layer.Close()
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	start := len(driver.launches)
	output, err := hipRunGemma4Q4PrefillAttentionBatchWorkspace(context.Background(), driver, cfg, layer, tokenCount, 0, workspace)
	core.RequireNoError(t, err)
	defer output.Close()

	core.AssertEqual(t, tokenCount*cfg.QueryHeads*cfg.HeadDim, output.Count())
	core.AssertEqual(t, true, output.borrowed)
	attentionOutput := &workspace.AttentionOutputFixed
	core.RequireTrue(t, attentionOutput.Pointer() != 0, "attention workspace output should exist")
	core.AssertEqual(t, output.Pointer(), attentionOutput.Pointer())
	launches := driver.launches[start:]
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameAttentionHeadsBatchCausal))
	core.AssertEqual(t, uint32(cfg.QueryHeads), launches[0].GridX)
	core.AssertEqual(t, uint32(tokenCount), launches[0].GridY)
	core.AssertEqual(t, hipAttentionHeadsBatchCausalLaunchArgsBytes, len(launches[0].Args))
	core.AssertEqual(t, uint32(tokenCount), binary.LittleEndian.Uint32(launches[0].Args[52:]))
	core.AssertEqual(t, uint32(cfg.QueryHeads), binary.LittleEndian.Uint32(launches[0].Args[56:]))
	core.AssertEqual(t, uint32(tokenCount), binary.LittleEndian.Uint32(launches[0].Args[60:]))
	core.AssertEqual(t, uint32(0), binary.LittleEndian.Uint32(launches[0].Args[64:]))
	core.AssertEqual(t, hipAttentionKVSourceContiguous, binary.LittleEndian.Uint32(launches[0].Args[88:]))

	sharedLayer := &hipGemma4Q4PrefillLayerKVBatch{
		QK:        &hipGemma4Q4PrefillRoPEQKBatch{Query: layer.QK.Query},
		DeviceKV:  layer.DeviceKV,
		SharedKey: layer.QK.Key,
		SharedVal: layer.Value,
	}
	start = len(driver.launches)
	sharedOutput, err := hipRunGemma4Q4PrefillAttentionBatchWorkspace(context.Background(), driver, cfg, sharedLayer, tokenCount, 0, workspace)
	core.RequireNoError(t, err)
	defer sharedOutput.Close()
	core.AssertEqual(t, true, sharedOutput.borrowed)
	core.AssertEqual(t, sharedOutput.Pointer(), attentionOutput.Pointer())
	sharedLaunches := driver.launches[start:]
	core.AssertEqual(t, 1, countLaunchName(sharedLaunches, hipKernelNameAttentionHeadsBatchCausal))
	core.AssertEqual(t, hipAttentionKVSourceContiguous, binary.LittleEndian.Uint32(sharedLaunches[0].Args[88:]))
}

func TestHIPGemma4Q4PrefillAttentionBatch_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()
	start := len(driver.launches)

	if _, err := hipRunGemma4Q4PrefillAttentionBatch(context.Background(), driver, cfg, nil, 0, 0); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillAttentionBatch succeeded with zero token count")
	}
	if _, err := hipRunGemma4Q4PrefillAttentionBatch(context.Background(), driver, cfg, nil, 1, -1); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillAttentionBatch succeeded with negative query start")
	}
	if _, err := hipRunGemma4Q4PrefillAttentionBatch(context.Background(), driver, cfg, nil, 1, 0); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillAttentionBatch succeeded with missing layer")
	}
	core.AssertEqual(t, start, len(driver.launches))
}

func TestHIPGemma4Q4PrefillLayerBodyBatch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()
	cfg.RoPERotaryDim = 2

	tokenCount := 3
	inputValues := make([]float32, tokenCount*cfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(index%cfg.HiddenSize + 1)
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill layer body input fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()

	layer, err := hipRunGemma4Q4PrefillLayerKVBatch(context.Background(), driver, cfg, input, tokenCount, 0, 1e-6, rocmKVCacheModeKQ8VQ4)
	core.RequireNoError(t, err)
	defer layer.Close()
	start := len(driver.launches)
	body, err := hipRunGemma4Q4PrefillLayerBodyBatch(context.Background(), driver, cfg, input, layer, tokenCount, 0, 1e-6)
	core.RequireNoError(t, err)
	defer body.Close()

	core.AssertEqual(t, tokenCount*cfg.QueryHeads*cfg.HeadDim, body.AttentionOutput.Count())
	core.AssertEqual(t, tokenCount*cfg.HiddenSize, body.AttentionProjection.Count())
	core.AssertEqual(t, tokenCount*cfg.HiddenSize, body.AttentionResidual.Count())
	core.AssertEqual(t, tokenCount*cfg.HiddenSize, body.PreFeedForward.Count())
	core.AssertEqual(t, tokenCount*cfg.HiddenSize, body.MLPOutput.Count())
	core.AssertEqual(t, tokenCount*cfg.HiddenSize, body.FinalHidden.Count())
	finalHidden, err := hipReadFloat32DeviceOutput(body.FinalHidden, hipGemma4Q4Layer0Operation, "prefill layer body final hidden", len(inputValues))
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, inputValues, finalHidden, 0.0001)

	launches := driver.launches[start:]
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameAttentionHeadsBatchCausal))
	core.AssertEqual(t, 2, countLaunchName(launches, hipKernelNameMLXQ4ProjBatch))
	core.AssertEqual(t, 3, countLaunchName(launches, hipKernelNameRMSNormHeads))
	core.AssertEqual(t, 2, countLaunchName(launches, hipKernelNameVectorAdd))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameMLXQ4GELUTanhMulBatch))
}

func TestHIPGemma4Q4PrefillLayerBodyBatchWithPerLayerInput_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()
	cfg.RoPERotaryDim = 2

	tokenCount := 3
	inputValues := make([]float32, tokenCount*cfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(index%cfg.HiddenSize + 1)
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill layer body per-layer input fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()

	perLayerValues := make([]float32, tokenCount*cfg.PerLayerInput.InputSize)
	for index := range perLayerValues {
		perLayerValues[index] = float32(index%cfg.PerLayerInput.InputSize + 1)
	}
	perLayerPayload, err := hipFloat32Payload(perLayerValues)
	core.RequireNoError(t, err)
	perLayerInput, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill layer body per-layer input multiplier", perLayerPayload, len(perLayerValues))
	core.RequireNoError(t, err)
	defer perLayerInput.Close()

	layer, err := hipRunGemma4Q4PrefillLayerKVBatch(context.Background(), driver, cfg, input, tokenCount, 0, 1e-6, rocmKVCacheModeKQ8VQ4)
	core.RequireNoError(t, err)
	defer layer.Close()
	start := len(driver.launches)
	body, err := hipRunGemma4Q4PrefillLayerBodyBatchWithPerLayerInput(context.Background(), driver, cfg, input, layer, perLayerInput, tokenCount, 0, 1e-6)
	core.RequireNoError(t, err)
	defer body.Close()

	core.AssertEqual(t, tokenCount*cfg.QueryHeads*cfg.HeadDim, body.AttentionOutput.Count())
	core.AssertEqual(t, tokenCount*cfg.HiddenSize, body.PostFeedForward.Count())
	core.AssertEqual(t, tokenCount*cfg.HiddenSize, body.PerLayerProjection.Count())
	core.AssertEqual(t, tokenCount*cfg.HiddenSize, body.FinalHidden.Count())
	finalHidden, err := hipReadFloat32DeviceOutput(body.FinalHidden, hipGemma4Q4Layer0Operation, "prefill layer body per-layer final hidden", len(inputValues))
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, inputValues, finalHidden, 0.0001)

	launches := driver.launches[start:]
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameAttentionHeadsBatchCausal))
	core.AssertEqual(t, 3, countLaunchName(launches, hipKernelNameMLXQ4ProjBatch))
	core.AssertEqual(t, 4, countLaunchName(launches, hipKernelNameRMSNormHeads))
	core.AssertEqual(t, 3, countLaunchName(launches, hipKernelNameVectorAdd))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameMLXQ4GELUTanhMulBatch))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameMLXQ4GELUTanhProjBatch))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameVectorScale))
}

func TestHIPGemma4Q4PrefillLayerBodyBatchWorkspaceReusesProjectionOutput_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()
	cfg.RoPERotaryDim = 2

	tokenCount := 2
	inputValues := make([]float32, tokenCount*cfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(index%cfg.HiddenSize + 1)
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill layer body workspace input fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()

	perLayerValues := make([]float32, tokenCount*cfg.PerLayerInput.InputSize)
	for index := range perLayerValues {
		perLayerValues[index] = float32(index%cfg.PerLayerInput.InputSize + 1)
	}
	perLayerPayload, err := hipFloat32Payload(perLayerValues)
	core.RequireNoError(t, err)
	perLayerInput, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill layer body workspace per-layer input", perLayerPayload, len(perLayerValues))
	core.RequireNoError(t, err)
	defer perLayerInput.Close()

	layer, err := hipRunGemma4Q4PrefillLayerKVBatch(context.Background(), driver, cfg, input, tokenCount, 0, 1e-6, rocmKVCacheModeKQ8VQ4)
	core.RequireNoError(t, err)
	defer layer.Close()
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	start := len(driver.launches)
	body, err := hipRunGemma4Q4PrefillLayerBodyBatchWithPerLayerInputWorkspace(context.Background(), driver, cfg, input, layer, perLayerInput, tokenCount, 0, 1e-6, workspace)
	core.RequireNoError(t, err)
	defer body.Close()

	count := tokenCount * cfg.HiddenSize
	core.AssertEqual(t, count, body.AttentionProjection.Count())
	core.AssertEqual(t, count, body.MLPOutput.Count())
	core.AssertEqual(t, count, body.PerLayerProjection.Count())
	core.AssertEqual(t, count, body.PostFeedForward.Count())
	core.AssertEqual(t, count, body.FinalHidden.Count())
	core.AssertEqual(t, true, body.AttentionProjection.borrowed)
	core.AssertEqual(t, true, body.MLPOutput.borrowed)
	core.AssertEqual(t, true, body.PerLayerProjection.borrowed)
	core.AssertEqual(t, true, body.PostFeedForward.borrowed)
	core.AssertEqual(t, true, body.FinalHidden.borrowed)
	core.AssertEqual(t, body.AttentionProjection.Pointer(), body.MLPOutput.Pointer())
	core.AssertEqual(t, body.AttentionProjection.Pointer(), body.PerLayerProjection.Pointer())
	core.AssertEqual(t, body.AttentionProjection.Pointer(), workspace.ProjectionOutputFixed.Pointer())
	core.AssertEqual(t, body.PostFeedForward.Pointer(), workspace.IntermediateOutputView.Pointer())
	finalHiddenPair := &workspace.FinalHiddenPairFixed
	if finalHiddenPair.Pointer() == 0 {
		t.Fatalf("final hidden pair workspace was not allocated")
	}
	core.AssertEqual(t, body.FinalHidden.Pointer(), finalHiddenPair.Pointer()+nativeDevicePointer((cfg.Layer&1)*workspace.FinalHiddenPairFixedCap*4))
	core.AssertNotEqual(t, body.PostFeedForward.Pointer(), body.FinalHidden.Pointer())
	activationCount := tokenCount * cfg.GateProjection.Rows
	perLayerActivationCount := tokenCount * cfg.PerLayerInput.InputGate.Rows
	activationOutput := &workspace.ActivationOutputFixed
	if activationOutput.Pointer() == 0 || workspace.ActivationOutputCap < activationCount {
		t.Fatalf("MLP activation workspace was not allocated")
	}
	if perLayerActivationCount < activationCount {
		core.AssertEqual(t, activationOutput.Pointer(), workspace.ActivationOutputView.Pointer())
		core.AssertEqual(t, perLayerActivationCount, workspace.ActivationOutputView.Count())
		if _, ok := workspace.ActivationOutputs[hipAttentionHeadsChunkedWorkspaceCapacityCount(perLayerActivationCount)]; ok && hipAttentionHeadsChunkedWorkspaceCapacityCount(perLayerActivationCount) != hipAttentionHeadsChunkedWorkspaceCapacityCount(activationCount) {
			t.Fatalf("per-layer GELU projection activation got its own workspace allocation")
		}
	} else {
		perLayerActivationOutput := &workspace.ActivationOutputFixed
		if perLayerActivationOutput.Pointer() == 0 || workspace.ActivationOutputCap < perLayerActivationCount {
			t.Fatalf("per-layer GELU projection activation workspace was not allocated")
		}
		core.AssertEqual(t, activationOutput.Pointer(), perLayerActivationOutput.Pointer())
	}
	finalHidden, err := hipReadFloat32DeviceOutput(body.FinalHidden, hipGemma4Q4Layer0Operation, "prefill layer body workspace final hidden", len(inputValues))
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, inputValues, finalHidden, 0.0001)

	launches := driver.launches[start:]
	core.AssertEqual(t, 3, countLaunchName(launches, hipKernelNameMLXQ4ProjBatch))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameVectorAdd))
}

func TestHIPGemma4Q4PrefillLayerBodyBatchWithPerLayerInput_BatchOneMaterializesPostMLPResidualOnce_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()
	cfg.RoPERotaryDim = 2

	tokenCount := 1
	inputValues := make([]float32, cfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(index + 1)
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill layer body fused per-layer input fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()

	perLayerValues := make([]float32, cfg.PerLayerInput.InputSize)
	for index := range perLayerValues {
		perLayerValues[index] = float32(index + 1)
	}
	perLayerPayload, err := hipFloat32Payload(perLayerValues)
	core.RequireNoError(t, err)
	perLayerInput, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill layer body fused per-layer multiplier", perLayerPayload, len(perLayerValues))
	core.RequireNoError(t, err)
	defer perLayerInput.Close()

	layer, err := hipRunGemma4Q4PrefillLayerKVBatch(context.Background(), driver, cfg, input, tokenCount, 0, 1e-6, rocmKVCacheModeKQ8VQ4)
	core.RequireNoError(t, err)
	defer layer.Close()
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	start := len(driver.launches)
	body, err := hipRunGemma4Q4PrefillLayerBodyBatchWithPerLayerInputWorkspace(context.Background(), driver, cfg, input, layer, perLayerInput, tokenCount, 0, 1e-6, workspace)
	core.RequireNoError(t, err)
	defer body.Close()

	core.AssertEqual(t, cfg.HiddenSize, body.PostFeedForward.Count())
	core.AssertEqual(t, cfg.HiddenSize, body.PerLayerProjection.Count())
	core.AssertEqual(t, cfg.HiddenSize, body.FinalHidden.Count())
	launches := driver.launches[start:]
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameRMSNormResidualAddGELUTanhProj))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameMLXQ4GELUTanhProj))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameMLXQ4GELUTanhProjBatch))
	core.AssertEqual(t, 2, countLaunchName(launches, hipKernelNameRMSNormResidualAdd))
}

func TestHIPGemma4Q4PrefillResidualSmallBatchUsesFusedKernels_Good(t *testing.T) {
	for _, tokenCount := range []int{1, 2} {
		t.Run(core.Sprintf("tokens_%d", tokenCount), func(t *testing.T) {
			driver := &fakeHIPDriver{available: true}
			cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
			defer cleanup()

			count := tokenCount * cfg.HiddenSize
			inputValues := make([]float32, count)
			residualValues := make([]float32, count)
			for index := range inputValues {
				inputValues[index] = float32(index%cfg.HiddenSize + 1)
				residualValues[index] = float32(cfg.HiddenSize - index%cfg.HiddenSize)
			}
			inputPayload, err := hipFloat32Payload(inputValues)
			core.RequireNoError(t, err)
			input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill residual small-batch input", inputPayload, len(inputValues))
			core.RequireNoError(t, err)
			defer input.Close()
			residualPayload, err := hipFloat32Payload(residualValues)
			core.RequireNoError(t, err)
			residual, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill residual small-batch residual", residualPayload, len(residualValues))
			core.RequireNoError(t, err)
			defer residual.Close()

			workspace := &hipAttentionHeadsChunkedWorkspace{}
			defer workspace.Close()
			start := len(driver.launches)
			residualOutput, normOutput, err := hipRunGemma4Q4PrefillResidualAddNormBatchWorkspace(context.Background(), driver, input, residual, cfg.PostAttentionNorm, cfg.PreFeedForwardNorm, tokenCount, 1, workspace)
			core.RequireNoError(t, err)
			defer residualOutput.Close()
			defer normOutput.Close()
			postFeedForward, err := hipRunGemma4Q4PrefillResidualAddBatch(context.Background(), driver, input, residual, cfg.PostFeedForwardNorm, tokenCount, 1)
			core.RequireNoError(t, err)
			defer postFeedForward.Close()

			core.AssertEqual(t, count, residualOutput.Count())
			core.AssertEqual(t, count, normOutput.Count())
			core.AssertEqual(t, count, postFeedForward.Count())
			core.AssertEqual(t, true, residualOutput.borrowed)
			core.AssertEqual(t, true, normOutput.borrowed)
			rmsPair := &workspace.RMSResidualNormFixed
			core.RequireTrue(t, rmsPair.Pointer() != 0, "RMS residual/norm pair workspace should exist")
			core.AssertEqual(t, residualOutput.Pointer(), rmsPair.Pointer())
			core.AssertEqual(t, normOutput.Pointer(), rmsPair.Pointer()+nativeDevicePointer(count*4))
			launches := driver.launches[start:]
			core.AssertEqual(t, tokenCount, countLaunchName(launches, hipKernelNameRMSNormResAddNorm))
			core.AssertEqual(t, tokenCount, countLaunchName(launches, hipKernelNameRMSNormResidualAdd))
			core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameRMSNormHeads))
			core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameVectorAdd))
			core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameVectorScale))
		})
	}
}

func TestHIPGemma4Q4PrefillFinalGreedyForRow_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()

	tokenCount := 3
	hiddenValues := make([]float32, tokenCount*cfg.HiddenSize)
	for index := range hiddenValues {
		hiddenValues[index] = float32(index%cfg.HiddenSize + 1)
	}
	payload, err := hipFloat32Payload(hiddenValues)
	core.RequireNoError(t, err)
	hidden, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill final greedy hidden fixture", payload, len(hiddenValues))
	core.RequireNoError(t, err)
	defer hidden.Close()
	start := len(driver.launches)

	greedy, err := hipRunGemma4Q4PrefillFinalGreedyForRow(context.Background(), driver, cfg, hidden, tokenCount, tokenCount-1, 1e-6, nil)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, greedy.TokenID)

	launches := driver.launches[start:]
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameRMSNorm))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameMLXQ4ProjGreedy))
}

func TestHIPGemma4Q4PrefillFinalGreedyForRowWorkspaceReusesRMSNormOutput_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()

	tokenCount := 3
	hiddenValues := make([]float32, tokenCount*cfg.HiddenSize)
	for index := range hiddenValues {
		hiddenValues[index] = float32(index%cfg.HiddenSize + 1)
	}
	payload, err := hipFloat32Payload(hiddenValues)
	core.RequireNoError(t, err)
	hidden, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill final greedy workspace hidden fixture", payload, len(hiddenValues))
	core.RequireNoError(t, err)
	defer hidden.Close()
	best, err := hipAllocateByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill final greedy workspace best fixture", hipMLXQ4ProjectionBestBytes, 1)
	core.RequireNoError(t, err)
	defer best.Close()
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	normOutput, err := workspace.EnsureRMSNormOutput(driver, cfg.HiddenSize)
	core.RequireNoError(t, err)
	allocStart := len(driver.allocations)
	start := len(driver.launches)

	greedy, err := hipRunGemma4Q4PrefillFinalGreedyForRowSuppressWorkspace(context.Background(), driver, cfg, hidden, tokenCount, tokenCount-1, 1e-6, best, nil, workspace)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, greedy.TokenID)
	core.AssertEqual(t, allocStart, len(driver.allocations))
	rmsPair := &workspace.RMSResidualNormFixed
	core.RequireTrue(t, rmsPair.Pointer() != 0, "RMS residual/norm pair workspace should exist")
	core.AssertEqual(t, normOutput.Pointer(), rmsPair.Pointer()+nativeDevicePointer(cfg.HiddenSize*4))

	launches := driver.launches[start:]
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameRMSNorm))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameMLXQ4ProjGreedy))
}

func TestHIPGemma4Q4PrefillFinalGreedyForRowBorrowedBestInitializesBest_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()

	tokenCount := 3
	hiddenValues := make([]float32, tokenCount*cfg.HiddenSize)
	for index := range hiddenValues {
		hiddenValues[index] = float32(index%cfg.HiddenSize + 1)
	}
	payload, err := hipFloat32Payload(hiddenValues)
	core.RequireNoError(t, err)
	hidden, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill final greedy borrowed hidden fixture", payload, len(hiddenValues))
	core.RequireNoError(t, err)
	defer hidden.Close()
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	best, err := workspace.BorrowProjectionGreedyBest(driver)
	core.RequireNoError(t, err)
	memsetStart := len(driver.memsets)

	greedy, err := hipRunGemma4Q4PrefillFinalGreedyForRowSuppressWorkspace(context.Background(), driver, cfg, hidden, tokenCount, tokenCount-1, 1e-6, best, nil, workspace)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, greedy.TokenID)
	core.AssertEqual(t, memsetStart+1, len(driver.memsets))
	core.AssertEqual(t, best.SizeBytes(), driver.memsets[len(driver.memsets)-1])
}

func TestHIPGemma4Q4PrefillFinalGreedyTokenForRowReadsUint32_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()

	tokenCount := 3
	hiddenValues := make([]float32, tokenCount*cfg.HiddenSize)
	for index := range hiddenValues {
		hiddenValues[index] = float32(index%cfg.HiddenSize + 1)
	}
	payload, err := hipFloat32Payload(hiddenValues)
	core.RequireNoError(t, err)
	hidden, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill final greedy token hidden fixture", payload, len(hiddenValues))
	core.RequireNoError(t, err)
	defer hidden.Close()
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	best, err := workspace.BorrowProjectionGreedyBest(driver)
	core.RequireNoError(t, err)
	copyStart := len(driver.copies)

	greedy, err := hipRunGemma4Q4PrefillFinalGreedyTokenForRowWorkspace(context.Background(), driver, cfg, hidden, tokenCount, tokenCount-1, 1e-6, best, workspace)
	core.RequireNoError(t, err)

	core.AssertEqual(t, 0, greedy.TokenID)
	assertFloat32Near(t, 0, greedy.Score)
	if len(driver.copies) != copyStart+1 {
		t.Fatalf("device copies = %d, want one token read after %d", len(driver.copies), copyStart)
	}
	core.AssertEqual(t, uint64(4), driver.copies[len(driver.copies)-1])
}

func TestHIPGemma4Q4PrefillForwardBatch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup0()
	layer1, cleanup1 := hipGemma4Q4FixtureConfig(t, driver, 1, 4, 2, 8)
	defer cleanup1()
	embeddingWeightsPayload, err := hipUint32Payload(make([]uint32, layer0.VocabSize*(layer0.HiddenSize/layer0.GroupSize)))
	core.RequireNoError(t, err)
	embeddingWeights, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill forward nonzero embedding weights", embeddingWeightsPayload, layer0.VocabSize*(layer0.HiddenSize/layer0.GroupSize))
	core.RequireNoError(t, err)
	defer embeddingWeights.Close()
	embeddingScalesPayload, err := hipUint16Payload([]uint16{0x3f80, 0x3f80})
	core.RequireNoError(t, err)
	embeddingScales, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill forward nonzero embedding scales", embeddingScalesPayload, layer0.VocabSize*(layer0.HiddenSize/layer0.GroupSize))
	core.RequireNoError(t, err)
	defer embeddingScales.Close()
	embeddingBiasesPayload, err := hipUint16Payload([]uint16{0x3f80, 0x3f80})
	core.RequireNoError(t, err)
	embeddingBiases, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill forward nonzero embedding biases", embeddingBiasesPayload, layer0.VocabSize*(layer0.HiddenSize/layer0.GroupSize))
	core.RequireNoError(t, err)
	defer embeddingBiases.Close()
	layer0.Embedding = hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: embeddingWeights.Pointer(),
		EmbeddingBytes:   embeddingWeights.SizeBytes(),
		TableEncoding:    hipEmbeddingTableEncodingMLXQ4,
		VocabSize:        layer0.VocabSize,
		HiddenSize:       layer0.HiddenSize,
		GroupSize:        layer0.GroupSize,
		ScalePointer:     embeddingScales.Pointer(),
		BiasPointer:      embeddingBiases.Pointer(),
		ScaleBytes:       embeddingScales.SizeBytes(),
		BiasBytes:        embeddingBiases.SizeBytes(),
	}
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0, layer1}}
	tokens := []int32{0, 1, 0}
	perLayerInputs := make([]*hipDeviceByteBuffer, len(cfg.Layers))
	for layerIndex, layer := range cfg.Layers {
		values := make([]float32, len(tokens)*layer.PerLayerInput.InputSize)
		for index := range values {
			values[index] = float32(layerIndex + 1)
		}
		payload, err := hipFloat32Payload(values)
		core.RequireNoError(t, err)
		buffer, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill forward per-layer input fixture", payload, len(values))
		core.RequireNoError(t, err)
		defer buffer.Close()
		perLayerInputs[layerIndex] = buffer
	}
	start := len(driver.launches)
	forward, err := hipRunGemma4Q4PrefillForwardBatch(context.Background(), driver, cfg, tokens, 0, 1e-6, rocmKVCacheModeKQ8VQ4, perLayerInputs, nil, nil)
	core.RequireNoError(t, err)
	defer forward.Close()

	core.AssertEqual(t, len(tokens)*layer0.HiddenSize, forward.Embedding.Count())
	core.AssertEqual(t, 2, len(forward.Layers))
	core.AssertEqual(t, len(tokens)*layer0.HiddenSize, forward.FinalHidden.Count())
	core.AssertEqual(t, 0, len(forward.Greedy))
	finalHidden, err := hipReadFloat32DeviceOutput(forward.FinalHidden, hipGemma4Q4Layer0Operation, "prefill forward final hidden", len(tokens)*layer0.HiddenSize)
	core.RequireNoError(t, err)
	expectedHidden := make([]float32, len(tokens)*layer0.HiddenSize)
	wantScale := hipBFloat16ToFloat32(hipFloat32ToBFloat16(float32(math.Sqrt(float64(layer0.HiddenSize)))))
	for index := range expectedHidden {
		expectedHidden[index] = wantScale
	}
	assertFloat32SlicesNear(t, expectedHidden, finalHidden, 0.0001)

	launches := driver.launches[start:]
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameEmbedLookup))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameVectorScale))
	core.AssertEqual(t, 12, countLaunchName(launches, hipKernelNameMLXQ4ProjBatch))
	core.AssertEqual(t, 12, countLaunchName(launches, hipKernelNameRMSNormHeads))
	core.AssertEqual(t, 4, countLaunchName(launches, hipKernelNameRMSNormRoPEHeadsBatch))
	core.AssertEqual(t, len(cfg.Layers)*gemma4Q4DeviceKVPagesForTokens(len(tokens)), countLaunchName(launches, hipKernelNameKVEncodeToken))
	core.AssertEqual(t, 2, countLaunchName(launches, hipKernelNameAttentionHeadsBatchCausal))
	core.AssertEqual(t, 2, countLaunchName(launches, hipKernelNameMLXQ4GELUTanhMulBatch))
	core.AssertEqual(t, 2, countLaunchName(launches, hipKernelNameMLXQ4GELUTanhProjBatch))
	core.AssertEqual(t, 6, countLaunchName(launches, hipKernelNameVectorAdd))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameRMSNorm))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameMLXQ4ProjGreedy))
}

func TestHIPGemma4Q4PrefillForwardBatchInitialHidden_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()
	tokens := []int32{0, 1}
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer}}

	initialValues := make([]float32, len(tokens)*layer.HiddenSize)
	for index := range initialValues {
		initialValues[index] = float32(index + 1)
	}
	initialPayload, err := hipFloat32Payload(initialValues)
	core.RequireNoError(t, err)
	initialHidden, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill initial hidden fixture", initialPayload, len(initialValues))
	core.RequireNoError(t, err)
	defer initialHidden.Close()

	perLayerValues := make([]float32, len(tokens)*layer.PerLayerInput.InputSize)
	perLayerPayload, err := hipFloat32Payload(perLayerValues)
	core.RequireNoError(t, err)
	perLayerInput, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill initial hidden per-layer input fixture", perLayerPayload, len(perLayerValues))
	core.RequireNoError(t, err)
	defer perLayerInput.Close()

	start := len(driver.launches)
	forward, err := hipRunGemma4Q4PrefillForwardBatchWithPriorDescriptorWorkspaceOutputRowInitialHiddenWithEngineConfig(
		context.Background(), driver, cfg, tokens, 0, 1e-6, rocmKVCacheModeKQ8VQ4,
		nil, nil, []*hipDeviceByteBuffer{perLayerInput}, nil, -1, nil, nil,
		defaultHIPGemma4Q4EngineConfig(), initialHidden, nil,
	)
	core.RequireNoError(t, err)
	defer forward.Close()
	core.AssertEqual(t, initialHidden.Pointer(), forward.Embedding.Pointer())
	launches := driver.launches[start:]
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameEmbedLookup))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameVectorScale))
}

func TestHIPGemma4Q4PrefillForwardBatchBidirectionalCaps_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()
	tokens := []int32{22, 22}
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer}}

	initialValues := make([]float32, len(tokens)*layer.HiddenSize)
	for index := range initialValues {
		initialValues[index] = float32(index + 1)
	}
	initialPayload, err := hipFloat32Payload(initialValues)
	core.RequireNoError(t, err)
	initialHidden, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "bidir initial hidden", initialPayload, len(initialPayload)/4)
	core.RequireNoError(t, err)
	defer initialHidden.Close()
	perLayerPayload, err := hipFloat32Payload(make([]float32, len(tokens)*layer.PerLayerInput.InputSize))
	core.RequireNoError(t, err)
	perLayerInput, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "bidir per-layer input", perLayerPayload, len(perLayerPayload)/4)
	core.RequireNoError(t, err)
	defer perLayerInput.Close()
	rowCaps, err := hipUploadTokenIDs(driver, []int32{2, 2})
	core.RequireNoError(t, err)
	defer rowCaps.Close()

	start := len(driver.launches)
	forward, err := hipRunGemma4Q4PrefillForwardBatchWithPriorDescriptorWorkspaceOutputRowInitialHiddenWithEngineConfig(
		context.Background(), driver, cfg, tokens, 0, 1e-6, rocmKVCacheModeKQ8VQ4,
		nil, nil, []*hipDeviceByteBuffer{perLayerInput}, nil, -1, nil, nil,
		defaultHIPGemma4Q4EngineConfig(), initialHidden, rowCaps,
	)
	core.RequireNoError(t, err)
	defer forward.Close()
	launches := driver.launches[start:]
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameAttentionHeadsBatchCapped))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameAttentionHeadsBatchCausal))
}

func TestHIPGemma4Q4PrefillForwardBatchWithPrior_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup0()
	layer1, cleanup1 := hipGemma4Q4FixtureConfig(t, driver, 1, 4, 2, 8)
	defer cleanup1()
	embeddingWeightsPayload, err := hipUint32Payload(make([]uint32, layer0.VocabSize*(layer0.HiddenSize/layer0.GroupSize)))
	core.RequireNoError(t, err)
	embeddingWeights, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill forward prior nonzero embedding weights", embeddingWeightsPayload, layer0.VocabSize*(layer0.HiddenSize/layer0.GroupSize))
	core.RequireNoError(t, err)
	defer embeddingWeights.Close()
	embeddingScalesPayload, err := hipUint16Payload([]uint16{0x3f80, 0x3f80})
	core.RequireNoError(t, err)
	embeddingScales, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill forward prior nonzero embedding scales", embeddingScalesPayload, layer0.VocabSize*(layer0.HiddenSize/layer0.GroupSize))
	core.RequireNoError(t, err)
	defer embeddingScales.Close()
	embeddingBiasesPayload, err := hipUint16Payload([]uint16{0x3f80, 0x3f80})
	core.RequireNoError(t, err)
	embeddingBiases, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill forward prior nonzero embedding biases", embeddingBiasesPayload, layer0.VocabSize*(layer0.HiddenSize/layer0.GroupSize))
	core.RequireNoError(t, err)
	defer embeddingBiases.Close()
	layer0.Embedding = hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: embeddingWeights.Pointer(),
		EmbeddingBytes:   embeddingWeights.SizeBytes(),
		TableEncoding:    hipEmbeddingTableEncodingMLXQ4,
		VocabSize:        layer0.VocabSize,
		HiddenSize:       layer0.HiddenSize,
		GroupSize:        layer0.GroupSize,
		ScalePointer:     embeddingScales.Pointer(),
		BiasPointer:      embeddingBiases.Pointer(),
		ScaleBytes:       embeddingScales.SizeBytes(),
		BiasBytes:        embeddingBiases.SizeBytes(),
	}
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0, layer1}}
	tokens := []int32{0, 1}
	makePerLayerInputs := func(label string) []*hipDeviceByteBuffer {
		t.Helper()
		perLayerInputs := make([]*hipDeviceByteBuffer, len(cfg.Layers))
		for layerIndex, layer := range cfg.Layers {
			values := make([]float32, len(tokens)*layer.PerLayerInput.InputSize)
			for index := range values {
				values[index] = float32(layerIndex + 1)
			}
			payload, err := hipFloat32Payload(values)
			core.RequireNoError(t, err)
			buffer, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, label, payload, len(values))
			core.RequireNoError(t, err)
			buf := buffer
			t.Cleanup(func() {
				_ = buf.Close()
			})
			perLayerInputs[layerIndex] = buffer
		}
		return perLayerInputs
	}

	first, err := hipRunGemma4Q4PrefillForwardBatch(context.Background(), driver, cfg, tokens, 0, 1e-6, rocmKVCacheModeKQ8VQ4, makePerLayerInputs("prefill forward prior first per-layer input"), nil, nil)
	core.RequireNoError(t, err)
	defer first.Close()
	prior := []*rocmDeviceKVCache{
		first.Layers[0].KV.DeviceKV.Cache,
		first.Layers[1].KV.DeviceKV.Cache,
	}

	start := len(driver.launches)
	second, err := hipRunGemma4Q4PrefillForwardBatchWithPrior(context.Background(), driver, cfg, tokens, len(tokens), 1e-6, rocmKVCacheModeKQ8VQ4, prior, makePerLayerInputs("prefill forward prior second per-layer input"), nil, nil)
	core.RequireNoError(t, err)
	defer second.Close()

	core.AssertEqual(t, 2, len(second.Layers))
	for index := range second.Layers {
		core.AssertEqual(t, len(tokens)*2, second.Layers[index].KV.DeviceKV.Cache.TokenCount())
	}

	launches := driver.launches[start:]
	var attentionLaunches []hipKernelLaunchConfig
	for _, launch := range launches {
		if launch.Name == hipKernelNameAttentionHeadsBatchCausal {
			attentionLaunches = append(attentionLaunches, launch)
		}
	}
	core.AssertEqual(t, 2, len(attentionLaunches))
	for _, launch := range attentionLaunches {
		core.AssertEqual(t, uint32(len(tokens)*2), binary.LittleEndian.Uint32(launch.Args[52:]))
		core.AssertEqual(t, uint32(len(tokens)), binary.LittleEndian.Uint32(launch.Args[60:]))
		core.AssertEqual(t, uint32(len(tokens)), binary.LittleEndian.Uint32(launch.Args[64:]))
	}
	core.AssertEqual(t, len(cfg.Layers)*gemma4Q4DeviceKVPagesForTokens(len(tokens)), countLaunchName(launches, hipKernelNameKVEncodeToken))
}

func TestHIPGemma4Q4PrefillLayerKVBatchWithPriorWorkspaceReusesValueNorm_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()
	hipGemma4Q4InstallNonzeroEmbeddingFixture(t, driver, &cfg, "prefill value norm workspace")
	tokens := []int32{0, 1}
	inputValues := make([]float32, len(tokens)*cfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(index%cfg.HiddenSize + 1)
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill value norm workspace input", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()

	first, err := hipRunGemma4Q4PrefillLayerKVBatch(context.Background(), driver, cfg, input, len(tokens), 0, 1e-6, rocmKVCacheModeKQ8VQ4)
	core.RequireNoError(t, err)
	defer first.Close()
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	second, err := hipRunGemma4Q4PrefillLayerKVBatchWithPriorWorkspaceTransient(context.Background(), driver, cfg, input, first.DeviceKV.Cache, len(tokens), len(tokens), 1e-6, rocmKVCacheModeKQ8VQ4, workspace, true, true)
	core.RequireNoError(t, err)
	defer second.Close()

	valueCount := len(tokens) * cfg.HeadDim
	output := &workspace.KeyValueNormFixed
	if output.Pointer() == 0 || workspace.KeyValueNormCap < valueCount {
		t.Fatalf("value norm workspace for %d floats was not allocated", valueCount)
	}
	core.AssertEqual(t, true, second.Value.borrowed)
	core.AssertEqual(t, output.Pointer()+nativeDevicePointer(valueCount*4), second.Value.Pointer())
	core.AssertEqual(t, valueCount, second.Value.Count())
	core.AssertEqual(t, len(tokens)*2, second.DeviceKV.Cache.TokenCount())
}

func TestHIPGemma4Q4PrefillLayerKVBatchWithPriorWorkspaceFusesValueNormEncode_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()
	hipGemma4Q4InstallNonzeroEmbeddingFixture(t, driver, &cfg, "prefill fused value norm encode")
	tokenCount := 1
	inputValues := make([]float32, tokenCount*cfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(index%cfg.HiddenSize + 1)
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill fused value norm encode input", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()
	first, err := hipRunGemma4Q4PrefillLayerKVBatch(context.Background(), driver, cfg, input, tokenCount, 0, 1e-6, rocmKVCacheModeKQ8VQ4)
	core.RequireNoError(t, err)
	defer first.Close()
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	start := len(driver.launches)
	second, err := hipRunGemma4Q4PrefillLayerKVBatchWithPriorWorkspaceTransient(context.Background(), driver, cfg, input, first.DeviceKV.Cache, tokenCount, tokenCount, 1e-6, rocmKVCacheModeKQ8VQ4, workspace, true, true)
	core.RequireNoError(t, err)
	defer second.Close()

	launches := driver.launches[start:]
	core.AssertEqual(t, 1, countLaunchName(launches, "rocm_kv_encode_token_value_norm"))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameKVEncodeToken))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameRMSNormHeads))
	core.AssertNil(t, second.Value)
	core.AssertEqual(t, tokenCount*2, second.DeviceKV.Cache.TokenCount())
}

func TestHIPGemma4Q4PrefillForwardWorkspaceBorrowsTransientKVForNonSharedLayers_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup0()
	layer1, cleanup1 := hipGemma4Q4FixtureConfig(t, driver, 1, 4, 2, 8)
	defer cleanup1()
	hipGemma4Q4InstallNonzeroEmbeddingFixture(t, driver, &layer0, "prefill transient KV workspace")
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0, layer1}}
	tokens := []int32{0, 1}
	perLayerInputs := hipGemma4Q4PrefillForwardTestPerLayerInputs(t, driver, cfg, tokens, "prefill transient KV workspace per-layer input")
	best, err := hipAllocateByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill transient KV workspace best fixture", hipMLXQ4ProjectionBestBytes, 1)
	core.RequireNoError(t, err)
	defer best.Close()
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	forward, err := hipRunGemma4Q4PrefillForwardBatchWithPriorWorkspace(context.Background(), driver, cfg, tokens, 0, 1e-6, rocmKVCacheModeKQ8VQ4, nil, perLayerInputs, []bool{false, true}, best, workspace)
	core.RequireNoError(t, err)
	defer forward.Close()
	core.AssertEqual(t, 1, len(forward.Greedy))

	lastLayer := cfg.Layers[len(cfg.Layers)-1]
	lastKV := forward.Layers[len(forward.Layers)-1].KV
	keyCount := len(tokens) * lastLayer.KeyProjection.Rows
	core.AssertEqual(t, true, lastKV.QKV.Key.borrowed)
	core.AssertEqual(t, true, lastKV.QKV.Value.borrowed)
	core.AssertEqual(t, true, lastKV.QK.Key.borrowed)
	core.AssertEqual(t, true, lastKV.Value.borrowed)
	kvProjectionCap := keyCount
	kvProjectionPair := &workspace.KVProjectionPairFixed
	if kvProjectionPair.Pointer() == 0 || workspace.KVProjectionPairCap < kvProjectionCap {
		t.Fatalf("KV projection pair workspace missing for cap %d", kvProjectionCap)
	}
	core.AssertEqual(t, kvProjectionPair.Pointer(), lastKV.QKV.Key.Pointer())
	core.AssertEqual(t, kvProjectionPair.Pointer()+nativeDevicePointer(kvProjectionCap*4), lastKV.QKV.Value.Pointer())
	keyValueNormPair := &workspace.KeyValueNormFixed
	core.RequireTrue(t, keyValueNormPair.Pointer() != 0, "key/value norm pair workspace should exist")
	core.AssertEqual(t, keyValueNormPair.Pointer(), lastKV.QK.Key.Pointer())
	core.AssertEqual(t, keyValueNormPair.Pointer()+nativeDevicePointer(len(tokens)*lastLayer.HeadDim*4), lastKV.Value.Pointer())
	core.AssertEqual(t, layer0.HiddenSize, workspace.RMSNormOutputView.Count())
	rmsPair := &workspace.RMSResidualNormFixed
	core.RequireTrue(t, rmsPair.Pointer() != 0, "RMS residual/norm pair workspace should exist")
	core.AssertEqual(t, rmsPair.Pointer()+nativeDevicePointer(len(tokens)*layer0.HiddenSize*4), workspace.RMSNormOutputView.Pointer())
}

func TestHIPGemma4Q4PrefillForwardBatch_BatchOneCarriesNextInputNorm_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup0()
	layer1, cleanup1 := hipGemma4Q4FixtureConfig(t, driver, 1, 4, 2, 8)
	defer cleanup1()
	hipGemma4Q4InstallNonzeroEmbeddingFixture(t, driver, &layer0, "prefill next input norm carry")
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0, layer1}}
	tokens := []int32{0}
	perLayerInputs := hipGemma4Q4PrefillForwardTestPerLayerInputs(t, driver, cfg, tokens, "prefill next input norm carry per-layer input")
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	start := len(driver.launches)
	forward, err := hipRunGemma4Q4PrefillForwardBatchWithPriorDescriptorWorkspace(context.Background(), driver, cfg, tokens, 0, 1e-6, rocmKVCacheModeKQ8VQ4, nil, nil, perLayerInputs, nil, nil, workspace)
	core.RequireNoError(t, err)
	defer forward.Close()

	core.AssertEqual(t, 2, len(forward.Layers))
	firstBody := forward.Layers[0].Body
	secondKV := forward.Layers[1].KV
	core.AssertNotNil(t, firstBody.NextInputNorm)
	core.AssertNotNil(t, secondKV.InputNorm)
	core.AssertEqual(t, layer1.HiddenSize, firstBody.NextInputNorm.Count())
	core.AssertEqual(t, firstBody.NextInputNorm.Pointer(), secondKV.InputNorm.Pointer())
	core.AssertEqual(t, true, firstBody.NextInputNorm.borrowed)
	core.AssertEqual(t, true, secondKV.InputNorm.borrowed)
	launches := driver.launches[start:]
	layer1InputNormLaunches := 0
	for _, launch := range launches {
		if launch.Name != hipKernelNameRMSNormHeads {
			continue
		}
		if binary.LittleEndian.Uint64(launch.Args[8:]) == uint64(firstBody.FinalHidden.Pointer()) &&
			binary.LittleEndian.Uint64(launch.Args[16:]) == uint64(layer1.InputNorm.WeightPointer) {
			layer1InputNormLaunches++
		}
	}
	core.AssertEqual(t, 0, layer1InputNormLaunches)
}

func TestHIPGemma4Q4PrefillAttentionBatchOverride_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	values := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	payload, err := hipFloat32Payload(values)
	core.RequireNoError(t, err)
	override, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "lane attention override", payload, len(values))
	core.RequireNoError(t, err)
	defer override.Close()
	layer := &hipGemma4Q4PrefillLayerKVBatch{AttentionOverride: override}
	cfg := hipGemma4Q4Layer0Config{HeadDim: 2, QueryHeads: 2, KeyHeads: 1}
	start := len(driver.launches)
	output, err := hipRunGemma4Q4PrefillAttentionBatch(context.Background(), driver, cfg, layer, 2, 0)
	core.RequireNoError(t, err)
	defer output.Close()
	core.AssertEqual(t, override.Pointer(), output.Pointer())
	core.AssertEqual(t, start, len(driver.launches))
	got, err := hipReadFloat32DeviceOutput(output, hipGemma4Q4Layer0Operation, "lane attention override output", len(values))
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, values, got, 0)
}

func TestHIPGemma4Q4PrefillForwardWorkspaceBorrowsSharedSourceRawKVRetainsSharedOutputs_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup0()
	layer1, cleanup1 := hipGemma4Q4FixtureConfig(t, driver, 1, 4, 2, 8)
	defer cleanup1()
	hipGemma4Q4InstallNonzeroEmbeddingFixture(t, driver, &layer0, "prefill shared source KV ownership")
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0, layer1}, KVSharedLayers: 1}
	sources := hipGemma4Q4SharedKVSourceByLayer(cfg)
	core.AssertEqual(t, 0, sources[1])
	tokens := []int32{0, 1}
	perLayerInputs := hipGemma4Q4PrefillForwardTestPerLayerInputs(t, driver, cfg, tokens, "prefill shared source KV ownership per-layer input")
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	forward, err := hipRunGemma4Q4PrefillForwardBatchWithPriorWorkspace(context.Background(), driver, cfg, tokens, 0, 1e-6, rocmKVCacheModeKQ8VQ4, nil, perLayerInputs, nil, nil, workspace)
	core.RequireNoError(t, err)
	defer forward.Close()

	sourceKV := forward.Layers[0].KV
	sharedKV := forward.Layers[1].KV
	keyCount := len(tokens) * layer0.KeyProjection.Rows
	core.AssertEqual(t, true, sourceKV.QKV.Key.borrowed)
	core.AssertEqual(t, true, sourceKV.QKV.Value.borrowed)
	core.AssertEqual(t, true, sourceKV.QK.Key.borrowed)
	core.AssertEqual(t, true, sourceKV.Value.borrowed)
	kvProjectionCap := keyCount
	kvProjectionPair := &workspace.KVProjectionPairFixed
	if kvProjectionPair.Pointer() == 0 || workspace.KVProjectionPairCap < kvProjectionCap {
		t.Fatalf("KV projection pair workspace missing for cap %d", kvProjectionCap)
	}
	core.AssertEqual(t, kvProjectionPair.Pointer(), sourceKV.QKV.Key.Pointer())
	core.AssertEqual(t, kvProjectionPair.Pointer()+nativeDevicePointer(kvProjectionCap*4), sourceKV.QKV.Value.Pointer())
	keyValueNormPair := &workspace.KeyValueNormFixed
	core.RequireTrue(t, keyValueNormPair.Pointer() != 0, "key/value norm pair workspace should exist")
	core.AssertEqual(t, keyValueNormPair.Pointer(), sourceKV.QK.Key.Pointer())
	core.AssertEqual(t, keyValueNormPair.Pointer()+nativeDevicePointer(len(tokens)*layer0.HeadDim*4), sourceKV.Value.Pointer())
	if sharedKV.SharedKey != nil || sharedKV.SharedVal != nil {
		t.Fatalf("shared layer retained borrowed raw KV pointers: key=%#v value=%#v", sharedKV.SharedKey, sharedKV.SharedVal)
	}
	if sharedKV.DeviceKV == nil || sharedKV.DeviceKV.Cache == nil || sharedKV.DeviceKV.DescriptorTable == nil {
		t.Fatalf("shared layer did not retain descriptor-backed device KV")
	}
}

func TestHIPGemma4Q4PrefillLayerSharedQueryBatchOneUsesSingleTokenKernels_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	sourceCfg, cleanupSource := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanupSource()
	sharedCfg, cleanupShared := hipGemma4Q4FixtureConfig(t, driver, 1, 4, 2, 8)
	defer cleanupShared()
	sourceCfg.RoPERotaryDim = 2
	sharedCfg.RoPERotaryDim = 2
	tokenCount := 1
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	inputValues := make([]float32, sourceCfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(index + 1)
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill shared query batch-one input fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()

	source, err := hipRunGemma4Q4PrefillLayerKVBatchWithPriorDescriptorWorkspaceTransientPrecomputedInputNormIntoWithEngineConfig(context.Background(), driver, sourceCfg, input, nil, nil, nil, tokenCount, 0, 1e-6, rocmKVCacheModeKQ8VQ4, workspace, true, true, nil, defaultHIPGemma4Q4EngineConfig())
	core.RequireNoError(t, err)
	defer source.Close()

	start := len(driver.launches)
	shared, err := hipRunGemma4Q4PrefillLayerQueryBatchWithSharedKVWorkspacePrecomputedInputNormInto(context.Background(), driver, sharedCfg, input, input, source, tokenCount, 0, 1e-6, workspace, nil)
	core.RequireNoError(t, err)
	defer shared.Close()

	core.AssertEqual(t, sharedCfg.QueryProjection.Rows, shared.QKV.Query.Count())
	if !shared.QueryRMSRoPEAttention.Enabled || shared.QueryRMSRoPEAttention.Input == nil {
		t.Fatalf("shared query did not retain raw query for fused attention")
	}
	core.AssertEqual(t, sharedCfg.QueryProjection.Rows, shared.QueryRMSRoPEAttention.Input.Count())
	launches := driver.launches[start:]
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameMLXQ4Proj))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameMLXQ4ProjBatch))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameRMSNormRoPEHeads))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameRMSNormRoPEHeadsBatch))

	start = len(driver.launches)
	output, err := hipRunGemma4Q4PrefillAttentionBatchWorkspace(context.Background(), driver, sharedCfg, shared, tokenCount, 0, workspace)
	core.RequireNoError(t, err)
	defer output.Close()
	attentionLaunches := driver.launches[start:]
	core.AssertEqual(t, 1, countLaunchName(attentionLaunches, hipKernelNameAttentionHeadsBatchCausalQueryRMSRoPE))
	core.AssertEqual(t, 0, countLaunchName(attentionLaunches, hipKernelNameAttentionHeadsBatchCausal))
}

func TestHIPGemma4Q4PrefillAttentionSharedQueryDeepUsesChunkedKernels_Good(t *testing.T) {
	testHIPGemma4Q4PrefillAttentionSharedQueryDeepUsesChunkedKernels(t, false, hipKernelNameAttentionHeadsBatchChunkedStage1)
}

func TestHIPGemma4Q4PrefillAttentionSharedQueryDeepUsesGQA2ChunkedKernel_Good(t *testing.T) {
	testHIPGemma4Q4PrefillAttentionSharedQueryDeepUsesChunkedKernels(t, true, hipKernelNameAttentionHeadsBatchChunkedStage1GQA2)
}

func testHIPGemma4Q4PrefillAttentionSharedQueryDeepUsesChunkedKernels(t *testing.T, enableGQA2 bool, expectedStage1 string) {
	t.Helper()
	previous := hipAttentionHeadsBatchChunkedGQA2Enabled
	hipAttentionHeadsBatchChunkedGQA2Enabled = enableGQA2
	t.Cleanup(func() {
		hipAttentionHeadsBatchChunkedGQA2Enabled = previous
	})
	const (
		dim             = 512
		tokenCount      = 513
		headCount       = 8
		queryStartToken = tokenCount - 1
	)
	driver := &fakeHIPDriver{available: true}
	keys := make([]float32, tokenCount*dim)
	values := make([]float32, tokenCount*dim)
	cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, hipGemma4Q4DeviceKVBlockSize())
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, dim, dim, keys, values))
	deviceKV, err := cache.MirrorToDevice(driver)
	core.RequireNoError(t, err)
	defer deviceKV.Close()
	descriptor, err := deviceKV.KernelDescriptorTable()
	core.RequireNoError(t, err)
	defer descriptor.Close()

	queryValues := make([]float32, headCount*dim)
	for index := range queryValues {
		queryValues[index] = float32(index%17-8) * 0.03125
	}
	queryPayload, err := hipFloat32Payload(queryValues)
	core.RequireNoError(t, err)
	query, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "deep shared query fixture", queryPayload, len(queryValues))
	core.RequireNoError(t, err)
	defer query.Close()
	normWeights := make([]float32, dim)
	normPayload, err := hipFloat32Payload(normWeights)
	core.RequireNoError(t, err)
	norm, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "deep shared query norm fixture", normPayload, len(normWeights))
	core.RequireNoError(t, err)
	defer norm.Close()

	layer := &hipGemma4Q4PrefillLayerKVBatch{
		QK: &hipGemma4Q4PrefillRoPEQKBatch{},
		DeviceKV: &hipGemma4Q4PrefillDeviceKVBatch{
			Cache:           deviceKV,
			DescriptorTable: descriptor,
		},
		QueryRMSRoPEAttention: hipGemma4Q4QueryRMSRoPEAttention{
			Enabled: true,
			Input:   query,
			NormConfig: hipRMSNormDeviceWeightConfig{
				WeightPointer:  norm.Pointer(),
				WeightBytes:    norm.SizeBytes(),
				Count:          dim,
				Epsilon:        1e-6,
				WeightEncoding: hipRMSNormWeightEncodingF32,
			},
			StartPosition:  queryStartToken,
			Base:           10000,
			FrequencyDim:   dim,
			RotaryCount:    dim,
			FrequencyScale: 1,
		},
	}
	cfg := hipGemma4Q4Layer0Config{HeadDim: dim, QueryHeads: headCount, KeyHeads: 1}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	start := len(driver.launches)
	output, err := hipRunGemma4Q4PrefillAttentionBatchWorkspace(context.Background(), driver, cfg, layer, 1, queryStartToken, workspace)
	core.RequireNoError(t, err)
	defer output.Close()
	launches := driver.launches[start:]
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameRMSNormRoPEHeadsBatch))
	core.AssertEqual(t, 1, countLaunchName(launches, expectedStage1))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameAttentionHeadsBatchChunkedStage2))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameAttentionHeadsBatchCausalQueryRMSRoPE))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameAttentionHeadsBatchCausal))
}

func TestHIPAttentionHeadsBatchChunkedGQA2Eligible_Good(t *testing.T) {
	previous := hipAttentionHeadsBatchChunkedGQA2Enabled
	hipAttentionHeadsBatchChunkedGQA2Enabled = true
	t.Cleanup(func() {
		hipAttentionHeadsBatchChunkedGQA2Enabled = previous
	})

	core.AssertTrue(t, hipAttentionHeadsBatchChunkedGQA2Eligible(16, 1), "16 query heads sharing one KV head must pair")
	core.AssertTrue(t, hipAttentionHeadsBatchChunkedGQA2Eligible(4, 2), "two query heads per KV head must pair")
	core.AssertTrue(t, !hipAttentionHeadsBatchChunkedGQA2Eligible(12, 4), "odd query-heads-per-KV ratios must stay on v2")
	core.AssertTrue(t, !hipAttentionHeadsBatchChunkedGQA2Eligible(1, 1), "single-head attention must stay on v2")
	core.AssertTrue(t, !hipAttentionHeadsBatchChunkedGQA2Eligible(4, 3), "invalid GQA topology must stay on v2")

	hipAttentionHeadsBatchChunkedGQA2Enabled = false
	core.AssertTrue(t, !hipAttentionHeadsBatchChunkedGQA2Eligible(16, 1), "disabled experimental route must stay on v2")
}

func hipGemma4Q4PrefillForwardTestPerLayerInputs(t *testing.T, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, tokens []int32, label string) []*hipDeviceByteBuffer {
	t.Helper()
	perLayerInputs := make([]*hipDeviceByteBuffer, len(cfg.Layers))
	for layerIndex, layer := range cfg.Layers {
		values := make([]float32, len(tokens)*layer.PerLayerInput.InputSize)
		for index := range values {
			values[index] = float32(layerIndex + 1)
		}
		payload, err := hipFloat32Payload(values)
		core.RequireNoError(t, err)
		buffer, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, label, payload, len(values))
		core.RequireNoError(t, err)
		buf := buffer
		t.Cleanup(func() {
			_ = buf.Close()
		})
		perLayerInputs[layerIndex] = buffer
	}
	return perLayerInputs
}

func TestHIPGemma4Q4PrefillDecodeStateTrimsSlidingWindow_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup0()
	layer1, cleanup1 := hipGemma4Q4FixtureConfig(t, driver, 1, 4, 2, 8)
	defer cleanup1()
	hipGemma4Q4InstallNonzeroEmbeddingFixture(t, driver, &layer0, "prefill decode trim")
	layer0.SlidingWindow = 1
	layer1.SlidingWindow = 0
	layer0.PerLayerInput = hipGemma4Q4PerLayerInputConfig{}
	layer1.PerLayerInput = hipGemma4Q4PerLayerInputConfig{}
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0, layer1}}
	tokens := []int32{0, 1}

	firstForward, err := hipRunGemma4Q4PrefillForwardBatch(context.Background(), driver, cfg, tokens, 0, 1e-6, rocmKVCacheModeKQ8VQ4, nil, nil, nil)
	core.RequireNoError(t, err)
	firstState, err := hipGemma4Q4DeviceDecodeStateFromPrefillForward(firstForward, rocmKVCacheModeKQ8VQ4)
	core.RequireNoError(t, err)
	closeErr := firstForward.Close()
	core.RequireNoError(t, closeErr)
	defer firstState.Close()
	core.AssertEqual(t, []int{1, len(tokens)}, firstState.LayerTokenCounts())

	priorLayerKV := hipGemma4Q4DeviceLayerCaches(firstState, nil, len(cfg.Layers))
	start := len(driver.launches)
	secondForward, err := hipRunGemma4Q4PrefillForwardBatchWithPrior(context.Background(), driver, cfg, tokens, len(tokens), 1e-6, rocmKVCacheModeKQ8VQ4, priorLayerKV, nil, nil, nil)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1+len(tokens), secondForward.Layers[0].KV.DeviceKV.Cache.TokenCount())
	core.AssertEqual(t, len(tokens)*2, secondForward.Layers[1].KV.DeviceKV.Cache.TokenCount())

	secondState, err := hipGemma4Q4DeviceDecodeStateFromPrefillForward(secondForward, rocmKVCacheModeKQ8VQ4)
	core.RequireNoError(t, err)
	closeErr = secondForward.Close()
	core.RequireNoError(t, closeErr)
	defer secondState.Close()
	core.AssertEqual(t, []int{1, len(tokens) * 2}, secondState.LayerTokenCounts())

	core.RequireNoError(t, hipFinalizeGemma4Q4ForwardDeviceState(firstState, secondState))
	hipReleaseClosedGemma4Q4DeviceDecodeState(firstState)
	firstState = nil

	launches := driver.launches[start:]
	var attentionLaunches []hipKernelLaunchConfig
	for _, launch := range launches {
		if launch.Name == hipKernelNameAttentionHeadsBatchCausal {
			attentionLaunches = append(attentionLaunches, launch)
		}
	}
	core.AssertEqual(t, 2, len(attentionLaunches))
	core.AssertEqual(t, uint32(1+len(tokens)), binary.LittleEndian.Uint32(attentionLaunches[0].Args[52:]))
	core.AssertEqual(t, uint32(len(tokens)), binary.LittleEndian.Uint32(attentionLaunches[0].Args[60:]))
	core.AssertEqual(t, uint32(1), binary.LittleEndian.Uint32(attentionLaunches[0].Args[64:]))
	core.AssertEqual(t, uint32(len(tokens)*2), binary.LittleEndian.Uint32(attentionLaunches[1].Args[52:]))
	core.AssertEqual(t, uint32(len(tokens)), binary.LittleEndian.Uint32(attentionLaunches[1].Args[60:]))
	core.AssertEqual(t, uint32(len(tokens)), binary.LittleEndian.Uint32(attentionLaunches[1].Args[64:]))
}

func TestHIPGemma4Q4PrefillForwardWithPriorDescriptorUsesAppend_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup0()
	hipGemma4Q4InstallNonzeroEmbeddingFixture(t, driver, &layer0, "prefill prior descriptor append")
	layer0.LayerType = "full_attention"
	layer0.SlidingWindow = 0
	layer0.PerLayerInput = hipGemma4Q4PerLayerInputConfig{}
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0}}

	firstForward, err := hipRunGemma4Q4PrefillForwardBatch(context.Background(), driver, cfg, []int32{0}, 0, 1e-6, rocmKVCacheModeKQ8VQ4, nil, nil, nil)
	core.RequireNoError(t, err)
	firstState, err := hipGemma4Q4DeviceDecodeStateFromPrefillForward(firstForward, rocmKVCacheModeKQ8VQ4)
	core.RequireNoError(t, err)
	closeErr := firstForward.Close()
	core.RequireNoError(t, closeErr)
	defer firstState.Close()

	priorDescriptor := firstState.layerDescriptorTable(0)
	priorPointer := priorDescriptor.Pointer()
	priorLayerKV := hipGemma4Q4DeviceLayerCaches(firstState, nil, len(cfg.Layers))
	priorLayerDescriptors := hipGemma4Q4DeviceLayerDescriptorTables(firstState, nil, len(cfg.Layers))
	startLaunches := len(driver.launches)
	secondForward, err := hipRunGemma4Q4PrefillForwardBatchWithPriorDescriptorWorkspace(context.Background(), driver, cfg, []int32{1}, 1, 1e-6, rocmKVCacheModeKQ8VQ4, priorLayerKV, priorLayerDescriptors, nil, nil, nil, nil)
	core.RequireNoError(t, err)
	appendLaunches := make([]hipKernelLaunchConfig, 0, 1)
	for _, launch := range driver.launches[startLaunches:] {
		if launch.Name == hipKernelNameKVDescriptorAppend {
			appendLaunches = append(appendLaunches, launch)
		}
	}
	core.AssertEqual(t, 1, len(appendLaunches))
	secondDeviceKV := secondForward.Layers[0].KV.DeviceKV
	core.AssertTrue(t, secondDeviceKV.DescriptorTable.Pointer() != 0, "second descriptor table should be device-backed")
	core.AssertEqual(t, uint64(priorPointer), binary.LittleEndian.Uint64(appendLaunches[0].Args[8:]))
	core.AssertEqual(t, uint64(secondDeviceKV.DescriptorTable.Pointer()), binary.LittleEndian.Uint64(appendLaunches[0].Args[16:]))
	core.AssertEqual(t, 2, secondDeviceKV.Cache.TokenCount())

	secondState, err := hipGemma4Q4DeviceDecodeStateFromPrefillForward(secondForward, rocmKVCacheModeKQ8VQ4)
	closeErr = secondForward.Close()
	core.RequireNoError(t, err)
	core.RequireNoError(t, closeErr)
	defer secondState.Close()
	core.RequireNoError(t, hipFinalizeGemma4Q4ForwardDeviceState(firstState, secondState))
	hipReleaseClosedGemma4Q4DeviceDecodeState(firstState)
	firstState = nil
	core.AssertEqual(t, []int{2}, secondState.LayerTokenCounts())
}

func TestHIPGemma4Q4PrefillForwardWithPriorDescriptorWorkspaceFusesValueNormAppend_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup0()
	hipGemma4Q4InstallNonzeroEmbeddingFixture(t, driver, &layer0, "prefill prior descriptor fused append")
	layer0.LayerType = "full_attention"
	layer0.SlidingWindow = 0
	layer0.PerLayerInput = hipGemma4Q4PerLayerInputConfig{}
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0}}

	firstForward, err := hipRunGemma4Q4PrefillForwardBatch(context.Background(), driver, cfg, []int32{0}, 0, 1e-6, rocmKVCacheModeKQ8VQ4, nil, nil, nil)
	core.RequireNoError(t, err)
	firstState, err := hipGemma4Q4DeviceDecodeStateFromPrefillForward(firstForward, rocmKVCacheModeKQ8VQ4)
	core.RequireNoError(t, err)
	closeErr := firstForward.Close()
	core.RequireNoError(t, closeErr)
	defer firstState.Close()

	priorDescriptor := firstState.layerDescriptorTable(0)
	priorPointer := priorDescriptor.Pointer()
	priorLayerKV := hipGemma4Q4DeviceLayerCaches(firstState, nil, len(cfg.Layers))
	priorLayerDescriptors := hipGemma4Q4DeviceLayerDescriptorTables(firstState, nil, len(cfg.Layers))
	workspace := hipNewAttentionHeadsChunkedWorkspace()
	defer workspace.Close()
	startLaunches := len(driver.launches)
	secondForward, err := hipRunGemma4Q4PrefillForwardBatchWithPriorDescriptorWorkspace(context.Background(), driver, cfg, []int32{1}, 1, 1e-6, rocmKVCacheModeKQ8VQ4, priorLayerKV, priorLayerDescriptors, nil, nil, nil, workspace)
	core.RequireNoError(t, err)

	fusedLaunches := make([]hipKernelLaunchConfig, 0, 1)
	encodeLaunches := 0
	appendLaunches := 0
	for _, launch := range driver.launches[startLaunches:] {
		switch launch.Name {
		case hipKernelNameKVEncodeTokenValueNormDescriptorAppend:
			fusedLaunches = append(fusedLaunches, launch)
		case hipKernelNameKVEncodeTokenValueNorm:
			encodeLaunches++
		case hipKernelNameKVDescriptorAppend:
			appendLaunches++
		}
	}
	core.AssertEqual(t, 1, len(fusedLaunches))
	core.AssertEqual(t, 0, encodeLaunches)
	core.AssertEqual(t, 0, appendLaunches)
	secondDeviceKV := secondForward.Layers[0].KV.DeviceKV
	core.AssertTrue(t, secondDeviceKV.DescriptorTable.Pointer() != 0, "second descriptor table should be device-backed")
	core.AssertEqual(t, uint64(priorPointer), binary.LittleEndian.Uint64(fusedLaunches[0].Args[120+8:]))
	core.AssertEqual(t, uint64(secondDeviceKV.DescriptorTable.Pointer()), binary.LittleEndian.Uint64(fusedLaunches[0].Args[120+16:]))
	core.AssertEqual(t, 2, secondDeviceKV.Cache.TokenCount())

	secondState, err := hipGemma4Q4DeviceDecodeStateFromPrefillForward(secondForward, rocmKVCacheModeKQ8VQ4)
	closeErr = secondForward.Close()
	core.RequireNoError(t, err)
	core.RequireNoError(t, closeErr)
	defer secondState.Close()
	core.RequireNoError(t, hipFinalizeGemma4Q4ForwardDeviceState(firstState, secondState))
	hipReleaseClosedGemma4Q4DeviceDecodeState(firstState)
	firstState = nil
	core.AssertEqual(t, []int{2}, secondState.LayerTokenCounts())
}

func TestHIPGemma4Q4PrefillDecodeStateSharedAliasesFollowTrimmedSource_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 8, 1, 8)
	defer cleanup0()
	layer1, cleanup1 := hipGemma4Q4FixtureConfig(t, driver, 1, 4, 2, 8)
	defer cleanup1()
	layer2, cleanup2 := hipGemma4Q4FixtureConfig(t, driver, 2, 8, 1, 8)
	defer cleanup2()
	layer3, cleanup3 := hipGemma4Q4FixtureConfig(t, driver, 3, 4, 2, 8)
	defer cleanup3()
	hipGemma4Q4InstallNonzeroEmbeddingFixture(t, driver, &layer0, "prefill shared trim")
	layer0.LayerType = "sliding_attention"
	layer0.SlidingWindow = 1
	layer1.LayerType = "full_attention"
	layer1.SlidingWindow = 0
	layer2.LayerType = "sliding_attention"
	layer2.SlidingWindow = 1
	layer3.LayerType = "full_attention"
	layer3.SlidingWindow = 0
	layer0.PerLayerInput = hipGemma4Q4PerLayerInputConfig{}
	layer1.PerLayerInput = hipGemma4Q4PerLayerInputConfig{}
	layer2.PerLayerInput = hipGemma4Q4PerLayerInputConfig{}
	layer3.PerLayerInput = hipGemma4Q4PerLayerInputConfig{}
	cfg := hipGemma4Q4ForwardConfig{
		Layers:         []hipGemma4Q4Layer0Config{layer0, layer1, layer2, layer3},
		KVSharedLayers: 2,
	}

	forward, err := hipRunGemma4Q4PrefillForwardBatch(context.Background(), driver, cfg, []int32{0, 1}, 0, 1e-6, rocmKVCacheModeKQ8VQ4, nil, nil, nil)
	core.RequireNoError(t, err)
	state, err := hipGemma4Q4DeviceDecodeStateFromPrefillForward(forward, rocmKVCacheModeKQ8VQ4)
	core.RequireNoError(t, err)
	closeErr := forward.Close()
	core.RequireNoError(t, closeErr)
	defer state.Close()

	core.AssertEqual(t, []int{1, 2, 1, 2}, state.LayerTokenCounts())
	core.AssertEqual(t, true, state.layers[2].borrowedCache)
	core.AssertEqual(t, true, state.layers[2].borrowedDescriptorTable)
	core.AssertEqual(t, true, state.layers[3].borrowedCache)
	core.AssertEqual(t, true, state.layers[3].borrowedDescriptorTable)
	core.AssertEqual(t, state.layers[0].cache, state.layers[2].cache)
	core.AssertEqual(t, state.layers[0].descriptorTable, state.layers[2].descriptorTable)
	core.AssertEqual(t, state.layers[1].cache, state.layers[3].cache)
	core.AssertEqual(t, state.layers[1].descriptorTable, state.layers[3].descriptorTable)
	core.AssertEqual(t, state.layers[0].cache.TokenCount(), state.layers[2].cache.TokenCount())
	core.AssertEqual(t, state.layers[1].cache.TokenCount(), state.layers[3].cache.TokenCount())
}

func hipGemma4Q4InstallNonzeroEmbeddingFixture(t *testing.T, driver nativeHIPDriver, layer *hipGemma4Q4Layer0Config, label string) {
	t.Helper()
	if layer == nil {
		t.Fatal("layer is nil")
	}
	count := layer.VocabSize * (layer.HiddenSize / layer.GroupSize)
	embeddingWeightsPayload, err := hipUint32Payload(make([]uint32, count))
	core.RequireNoError(t, err)
	embeddingWeights, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, label+" embedding weights", embeddingWeightsPayload, count)
	core.RequireNoError(t, err)
	t.Cleanup(func() {
		_ = embeddingWeights.Close()
	})
	scaleValues := make([]uint16, count)
	for index := range scaleValues {
		scaleValues[index] = 0x3f80
	}
	embeddingScalesPayload, err := hipUint16Payload(scaleValues)
	core.RequireNoError(t, err)
	embeddingScales, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, label+" embedding scales", embeddingScalesPayload, count)
	core.RequireNoError(t, err)
	t.Cleanup(func() {
		_ = embeddingScales.Close()
	})
	embeddingBiasesPayload, err := hipUint16Payload(scaleValues)
	core.RequireNoError(t, err)
	embeddingBiases, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, label+" embedding biases", embeddingBiasesPayload, count)
	core.RequireNoError(t, err)
	t.Cleanup(func() {
		_ = embeddingBiases.Close()
	})
	layer.Embedding = hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: embeddingWeights.Pointer(),
		EmbeddingBytes:   embeddingWeights.SizeBytes(),
		TableEncoding:    hipEmbeddingTableEncodingMLXQ4,
		VocabSize:        layer.VocabSize,
		HiddenSize:       layer.HiddenSize,
		GroupSize:        layer.GroupSize,
		ScalePointer:     embeddingScales.Pointer(),
		BiasPointer:      embeddingBiases.Pointer(),
		ScaleBytes:       embeddingScales.SizeBytes(),
		BiasBytes:        embeddingBiases.SizeBytes(),
	}
}

func BenchmarkHIPGemma4Q4PrefillForwardSharedSourceLayers_SharedSuffix(b *testing.B) {
	const layerCount = 42
	forward := &hipGemma4Q4PrefillForwardBatch{
		Layers: make([]hipGemma4Q4PrefillForwardLayerBatch, layerCount),
	}
	for index := 0; index < layerCount; index++ {
		pointerBase := nativeDevicePointer(0x100000 + index*0x100)
		pages := []rocmDeviceKVPage{{
			tokenStart: 0,
			tokenCount: 1,
			keyWidth:   512,
			valueWidth: 512,
			key:        rocmDeviceKVTensor{pointer: pointerBase + 1, sizeBytes: 516, encoding: rocmKVEncodingQ8},
			value:      rocmDeviceKVTensor{pointer: pointerBase + 2, sizeBytes: 260, encoding: rocmKVEncodingQ4},
		}}
		cache := &rocmDeviceKVCache{mode: rocmKVCacheModeKQ8VQ4, blockSize: 1, tokenCount: 1, pages: pages}
		forward.Layers[index].KV = &hipGemma4Q4PrefillLayerKVBatch{
			DeviceKV: &hipGemma4Q4PrefillDeviceKVBatch{Cache: cache},
		}
	}
	for index := 24; index < layerCount; index++ {
		source := index - 18
		forward.Layers[index].KV.DeviceKV.Cache = &rocmDeviceKVCache{
			mode:       rocmKVCacheModeKQ8VQ4,
			blockSize:  1,
			tokenCount: 1,
			pages:      forward.Layers[source].KV.DeviceKV.Cache.pages,
			borrowed:   true,
		}
	}
	scratch := make([]int, 0, layerCount)
	sources := hipGemma4Q4PrefillForwardSharedSourceLayers(forward, scratch)
	if len(sources) != layerCount || sources[24] != 6 || sources[41] != 23 {
		b.Fatalf("shared sources[24,41] = %d,%d", sources[24], sources[41])
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sources = hipGemma4Q4PrefillForwardSharedSourceLayers(forward, scratch)
		if len(sources) != layerCount || sources[24] != 6 || sources[41] != 23 {
			b.Fatalf("shared sources[24,41] = %d,%d", sources[24], sources[41])
		}
	}
}

func TestHIPGemma4Q4PrefillForwardBatchWithGeneratedPerLayerInput_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup0()
	layer1, cleanup1 := hipGemma4Q4FixtureConfig(t, driver, 1, 4, 2, 8)
	defer cleanup1()
	embeddingWeightsPayload, err := hipUint32Payload(make([]uint32, layer0.VocabSize*(layer0.HiddenSize/layer0.GroupSize)))
	core.RequireNoError(t, err)
	embeddingWeights, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill generated input nonzero embedding weights", embeddingWeightsPayload, layer0.VocabSize*(layer0.HiddenSize/layer0.GroupSize))
	core.RequireNoError(t, err)
	defer embeddingWeights.Close()
	embeddingScalesPayload, err := hipUint16Payload([]uint16{0x3f80, 0x3f80})
	core.RequireNoError(t, err)
	embeddingScales, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill generated input nonzero embedding scales", embeddingScalesPayload, layer0.VocabSize*(layer0.HiddenSize/layer0.GroupSize))
	core.RequireNoError(t, err)
	defer embeddingScales.Close()
	embeddingBiasesPayload, err := hipUint16Payload([]uint16{0x3f80, 0x3f80})
	core.RequireNoError(t, err)
	embeddingBiases, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill generated input nonzero embedding biases", embeddingBiasesPayload, layer0.VocabSize*(layer0.HiddenSize/layer0.GroupSize))
	core.RequireNoError(t, err)
	defer embeddingBiases.Close()
	layer0.Embedding = hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: embeddingWeights.Pointer(),
		EmbeddingBytes:   embeddingWeights.SizeBytes(),
		TableEncoding:    hipEmbeddingTableEncodingMLXQ4,
		VocabSize:        layer0.VocabSize,
		HiddenSize:       layer0.HiddenSize,
		GroupSize:        layer0.GroupSize,
		ScalePointer:     embeddingScales.Pointer(),
		BiasPointer:      embeddingBiases.Pointer(),
		ScaleBytes:       embeddingScales.SizeBytes(),
		BiasBytes:        embeddingBiases.SizeBytes(),
	}

	layerCount := 2
	inputSize := layer0.PerLayerInput.InputSize
	globalRows := layerCount * inputSize
	globalWeightsPayload, err := hipUint32Payload(make([]uint32, layer0.VocabSize*(globalRows/layer0.GroupSize)))
	core.RequireNoError(t, err)
	globalWeights, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill generated per-layer embedding weights", globalWeightsPayload, layer0.VocabSize*(globalRows/layer0.GroupSize))
	core.RequireNoError(t, err)
	defer globalWeights.Close()
	globalScalesPayload, err := hipUint16Payload([]uint16{0x3f80, 0x3f80, 0x3f80, 0x3f80})
	core.RequireNoError(t, err)
	globalScales, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill generated per-layer embedding scales", globalScalesPayload, layer0.VocabSize*(globalRows/layer0.GroupSize))
	core.RequireNoError(t, err)
	defer globalScales.Close()
	globalBiasesPayload, err := hipUint16Payload([]uint16{0x3f80, 0x3f80, 0x3f80, 0x3f80})
	core.RequireNoError(t, err)
	globalBiases, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill generated per-layer embedding biases", globalBiasesPayload, layer0.VocabSize*(globalRows/layer0.GroupSize))
	core.RequireNoError(t, err)
	defer globalBiases.Close()
	modelProjectionPayload, err := hipUint16Payload(repeatUint16(0x3f80, globalRows*layer0.HiddenSize))
	core.RequireNoError(t, err)
	modelProjection, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill generated model projection", modelProjectionPayload, globalRows*layer0.HiddenSize)
	core.RequireNoError(t, err)
	defer modelProjection.Close()
	projectionNormPayload, err := hipUint16Payload(repeatUint16(0x3f80, inputSize))
	core.RequireNoError(t, err)
	projectionNorm, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill generated projection norm", projectionNormPayload, inputSize)
	core.RequireNoError(t, err)
	defer projectionNorm.Close()
	layer0.PerLayerInput.Embedding = hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: globalWeights.Pointer(),
		EmbeddingBytes:   globalWeights.SizeBytes(),
		TableEncoding:    hipEmbeddingTableEncodingMLXQ4,
		VocabSize:        layer0.VocabSize,
		HiddenSize:       globalRows,
		GroupSize:        layer0.GroupSize,
		ScalePointer:     globalScales.Pointer(),
		BiasPointer:      globalBiases.Pointer(),
		ScaleBytes:       globalScales.SizeBytes(),
		BiasBytes:        globalBiases.SizeBytes(),
	}
	layer0.PerLayerInput.ModelProjection = hipBF16DeviceWeightConfig{
		WeightPointer: modelProjection.Pointer(),
		WeightBytes:   modelProjection.SizeBytes(),
		Rows:          globalRows,
		Cols:          layer0.HiddenSize,
	}
	layer0.PerLayerInput.ProjectionNorm = hipRMSNormDeviceWeightConfig{
		WeightPointer:  projectionNorm.Pointer(),
		WeightBytes:    projectionNorm.SizeBytes(),
		Count:          inputSize,
		WeightEncoding: hipRMSNormWeightEncodingBF16,
	}

	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0, layer1}}
	tokens := []int32{0, 1}
	start := len(driver.launches)
	allocStart := len(driver.allocations)
	copyStart := len(driver.copies)
	forward, err := hipRunGemma4Q4PrefillForwardBatch(context.Background(), driver, cfg, tokens, 0, 1e-6, rocmKVCacheModeKQ8VQ4, nil, nil, nil)
	core.RequireNoError(t, err)
	defer forward.Close()

	core.AssertEqual(t, len(tokens)*layer0.HiddenSize, forward.FinalHidden.Count())
	launches := driver.launches[start:]
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameProjectionBatch))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNamePerLayerInputTranspose))
	core.AssertEqual(t, 2, countLaunchName(launches, hipKernelNameMLXQ4GELUTanhProjBatch))
	core.AssertEqual(t, 1, countUint64Value(driver.allocations[allocStart:], uint64(len(tokens)*4)))
	core.AssertEqual(t, 1, countUint64Value(driver.copies[copyStart:], uint64(len(tokens)*4)))
}

func TestHIPGemma4Q4PrefillPerLayerInputDeviceSetBatchWorkspaceReusesFinalBacking_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup0()
	layer1, cleanup1 := hipGemma4Q4FixtureConfig(t, driver, 1, 4, 2, 8)
	defer cleanup1()
	layers, cleanupPerLayer := hipGemma4Q4GlobalPerLayerInputFixture(t, driver, []hipGemma4Q4Layer0Config{layer0, layer1})
	defer cleanupPerLayer()

	tokens := []int32{0, 1}
	hiddenValues := make([]float32, len(tokens)*layer0.HiddenSize)
	payload, err := hipFloat32Payload(hiddenValues)
	core.RequireNoError(t, err)
	hidden, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill per-layer workspace hidden", payload, len(hiddenValues))
	core.RequireNoError(t, err)
	defer hidden.Close()

	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	cfg := hipGemma4Q4ForwardConfig{Layers: layers}
	allocStart := len(driver.allocations)
	set, err := hipRunGemma4Q4PrefillPerLayerInputDeviceSetBatchWorkspace(context.Background(), driver, cfg, tokens, hidden, 1e-6, workspace)
	core.RequireNoError(t, err)
	defer set.Close()
	core.AssertEqual(t, len(layers), set.LayerCount())
	core.AssertEqual(t, len(tokens)*layer0.HiddenSize, set.Layer(0).Count())
	core.AssertEqual(t, 3, len(driver.allocations)-allocStart)
	if workspace.PerLayerScaled != nil {
		t.Fatalf("per-layer scaled workspace allocated: %+v", workspace.PerLayerScaled)
	}

	set, err = hipRunGemma4Q4PrefillPerLayerInputDeviceSetBatchWorkspace(context.Background(), driver, cfg, tokens, hidden, 1e-6, workspace)
	core.RequireNoError(t, err)
	defer set.Close()
	core.AssertEqual(t, len(layers), set.LayerCount())
	core.AssertEqual(t, len(tokens)*layer0.HiddenSize, set.Layer(1).Count())
	core.AssertEqual(t, 4, len(driver.allocations)-allocStart)
}

func TestHIPGemma4Q4PrefillLayerBodyBatch_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()

	inputValues := make([]float32, cfg.HiddenSize)
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill layer body bad input fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()
	start := len(driver.launches)

	if _, err := hipRunGemma4Q4PrefillLayerBodyBatch(context.Background(), driver, cfg, input, nil, 0, 0, 1e-6); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillLayerBodyBatch succeeded with zero token count")
	}
	if _, err := hipRunGemma4Q4PrefillLayerBodyBatch(context.Background(), driver, cfg, input, nil, 1, -1, 1e-6); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillLayerBodyBatch succeeded with negative query start")
	}
	if _, err := hipRunGemma4Q4PrefillLayerBodyBatch(context.Background(), driver, cfg, input, nil, 2, 0, 1e-6); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillLayerBodyBatch succeeded with mismatched input shape")
	}
	if _, err := hipRunGemma4Q4PrefillLayerBodyBatch(context.Background(), driver, cfg, input, nil, 1, 0, 1e-6); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillLayerBodyBatch succeeded with missing layer")
	}
	badPerLayerPayload, err := hipFloat32Payload([]float32{1})
	core.RequireNoError(t, err)
	badPerLayer, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill layer body bad per-layer input fixture", badPerLayerPayload, 1)
	core.RequireNoError(t, err)
	defer badPerLayer.Close()
	if _, err := hipRunGemma4Q4PrefillLayerBodyBatchWithPerLayerInput(context.Background(), driver, cfg, input, nil, badPerLayer, 1, 0, 1e-6); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillLayerBodyBatchWithPerLayerInput succeeded with mismatched per-layer input")
	}
	if _, err := hipRunGemma4Q4PrefillFinalGreedyForRow(context.Background(), driver, cfg, input, 1, 1, 1e-6, nil); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillFinalGreedyForRow succeeded with row outside token batch")
	}
	if _, err := hipRunGemma4Q4PrefillFinalGreedyForRow(context.Background(), driver, cfg, input, 2, 0, 1e-6, nil); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillFinalGreedyForRow succeeded with mismatched hidden batch shape")
	}
	forwardCfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{cfg}}
	if _, err := hipRunGemma4Q4PrefillForwardBatch(context.Background(), driver, forwardCfg, []int32{0}, 1, 1e-6, rocmKVCacheModeKQ8VQ4, nil, nil, nil); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillForwardBatch succeeded with nonzero start position")
	}
	prior := &rocmDeviceKVCache{driver: driver, mode: rocmKVCacheModeKQ8VQ4, blockSize: defaultROCmKVBlockSize, tokenCount: 2}
	if _, err := hipRunGemma4Q4PrefillForwardBatchWithPrior(context.Background(), driver, forwardCfg, []int32{0}, 0, 1e-6, rocmKVCacheModeKQ8VQ4, []*rocmDeviceKVCache{prior}, nil, nil, nil); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillForwardBatchWithPrior succeeded with prior at start position 0")
	}
	if _, err := hipRunGemma4Q4PrefillForwardBatchWithPrior(context.Background(), driver, forwardCfg, []int32{0}, 1, 1e-6, rocmKVCacheModeKQ8VQ4, []*rocmDeviceKVCache{prior}, nil, nil, nil); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillForwardBatchWithPrior succeeded with mismatched prior token count")
	}
	if _, err := hipRunGemma4Q4PrefillForwardBatch(context.Background(), driver, forwardCfg, []int32{0}, 0, 1e-6, rocmKVCacheModeKQ8VQ4, nil, nil, nil); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillForwardBatch succeeded without required per-layer inputs")
	}
	if _, err := hipRunGemma4Q4PrefillForwardBatch(context.Background(), driver, forwardCfg, []int32{0}, 0, 1e-6, rocmKVCacheModeKQ8VQ4, []*hipDeviceByteBuffer{badPerLayer}, []bool{true, false}, nil); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillForwardBatch succeeded with mismatched output mask")
	}
	core.AssertEqual(t, start, len(driver.launches))
}

func TestHIPGemma4Q4PrefillLayerKVBatch_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()

	inputValues := make([]float32, cfg.HiddenSize)
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill layer KV bad input fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()
	start := len(driver.launches)

	if _, err := hipRunGemma4Q4PrefillLayerKVBatch(context.Background(), driver, cfg, input, 0, 0, 1e-6, rocmKVCacheModeKQ8VQ4); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillLayerKVBatch succeeded with zero token count")
	}
	if _, err := hipRunGemma4Q4PrefillLayerKVBatch(context.Background(), driver, cfg, input, 2, 0, 1e-6, rocmKVCacheModeKQ8VQ4); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillLayerKVBatch succeeded with mismatched input shape")
	}
	if _, err := hipRunGemma4Q4PrefillLayerKVBatch(context.Background(), driver, cfg, input, 1, -1, 1e-6, rocmKVCacheModeKQ8VQ4); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillLayerKVBatch succeeded with negative start position")
	}
	prior := &rocmDeviceKVCache{driver: driver, mode: rocmKVCacheModeKQ8VQ4, blockSize: defaultROCmKVBlockSize, tokenCount: 2}
	if _, err := hipRunGemma4Q4PrefillLayerKVBatchWithPrior(context.Background(), driver, cfg, input, prior, 1, 1, 1e-6, rocmKVCacheModeKQ8VQ4); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillLayerKVBatchWithPrior succeeded with mismatched prior token count")
	}
	core.AssertEqual(t, start, len(driver.launches))
}

func TestHIPGemma4Q4PrefillMLPBatch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup()

	tokenCount := 2
	inputValues := make([]float32, tokenCount*cfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(index%cfg.HiddenSize + 1)
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill MLP fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()

	start := len(driver.launches)
	output, err := hipRunGemma4Q4PrefillMLPBatch(context.Background(), driver, cfg, input, tokenCount)
	core.RequireNoError(t, err)
	defer output.Close()

	core.AssertEqual(t, tokenCount*cfg.DownProjection.Rows, output.Count())
	launches := driver.launches[start:]
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameMLXQ4GELUTanhMulBatch))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameMLXQ4ProjBatch))
	for _, launch := range launches {
		core.AssertEqual(t, uint32((tokenCount+hipMLXQ4ProjectionBatchTokensPerBlock-1)/hipMLXQ4ProjectionBatchTokensPerBlock), launch.GridY)
	}
}

func TestHIPGemma4Q4PrefillMLPBatch_BatchOneUsesSingleGELUMultiply_Good(t *testing.T) {
	setHIPMLXQ4GELUTanhMLPPersistentRouteForTest(t, false)
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup()

	tokenCount := 1
	inputValues := make([]float32, cfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(index + 1)
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill MLP batch-one fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	start := len(driver.launches)
	output, err := hipRunGemma4Q4PrefillMLPBatchWorkspace(context.Background(), driver, cfg, input, tokenCount, workspace)
	core.RequireNoError(t, err)
	defer output.Close()

	core.AssertEqual(t, cfg.DownProjection.Rows, output.Count())
	launches := driver.launches[start:]
	singleGELULaunches := countLaunchName(launches, hipKernelNameMLXQ4GELUTanhMul) +
		countLaunchName(launches, hipKernelNameMLXQ4GELUTanhMulQ4G32Cols1536Row16) +
		countLaunchName(launches, hipKernelNameMLXQ4GELUTanhMulQ6Cols1536) +
		countLaunchName(launches, hipKernelNameMLXQ4GELUTanhMulQ6Cols1536Row32) +
		countLaunchName(launches, hipKernelNameMLXQ4GELUTanhMulQ6Cols1536Row64)
	core.AssertEqual(t, 1, singleGELULaunches)
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameMLXQ4GELUTanhMulBatch))
	singleProjectionLaunches := countLaunchName(launches, hipKernelNameMLXQ4Proj) +
		countLaunchName(launches, hipKernelNameMLXQ4ProjCols256) +
		countLaunchName(launches, hipKernelNameMLXQ4ProjQ6G16Row16) +
		countLaunchName(launches, hipKernelNameMLXQ4ProjQ6Row16) +
		countLaunchName(launches, hipKernelNameMLXQ4ProjQ6Row32) +
		countLaunchName(launches, hipKernelNameMLXQ4ProjQ6Row64)
	core.AssertEqual(t, 1, singleProjectionLaunches)
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameMLXQ4ProjBatch))
}

func TestHIPGemma4Q4PrefillMLPBatch_BatchOneUsesPersistentQ4G32Cols1536Kernel_Good(t *testing.T) {
	setHIPMLXQ4GELUTanhMLPPersistentRouteForTest(t, true)
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4ProductionMLPFixtureConfig(t, driver, 1536, 6144, 32)
	defer cleanup()

	inputValues := make([]float32, cfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(index%17) / 17
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill persistent MLP fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	start := len(driver.launches)
	output, err := hipRunGemma4Q4PrefillMLPBatchWorkspace(context.Background(), driver, cfg, input, 1, workspace)
	core.RequireNoError(t, err)
	defer output.Close()

	const persistentKernel = "rocm_mlx_q4_gelu_tanh_mlp_q4_g32_cols1536_persistent"
	core.AssertEqual(t, cfg.DownProjection.Rows, output.Count())
	launches := driver.launches[start:]
	core.AssertEqual(t, 1, countLaunchName(launches, persistentKernel))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameMLXQ4GELUTanhMul))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameMLXQ4GELUTanhMulQ4G32Cols1536Row16))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameMLXQ4Proj))
	for _, launch := range launches {
		if launch.Name == persistentKernel {
			core.AssertEqual(t, hipMLXQ4GELUTanhMLPPersistentBlocks, launch.GridX)
			core.AssertEqual(t, hipMLXQ4ProjectionBlockSize, launch.BlockX)
		}
	}
}

func TestHIPGemma4Q4DeviceGELUTanhMLP_UsesPersistentQ4G32Cols1536Kernel_Good(t *testing.T) {
	setHIPMLXQ4GELUTanhMLPPersistentRouteForTest(t, true)
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4ProductionMLPFixtureConfig(t, driver, 1536, 6144, 32)
	defer cleanup()

	inputValues := make([]float32, cfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32((index%23)-11) / 23
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "decode persistent MLP input fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()
	output, err := hipAllocateByteBuffer(driver, hipGemma4Q4Layer0Operation, "decode persistent MLP output fixture", uint64(cfg.DownProjection.Rows*4), cfg.DownProjection.Rows)
	core.RequireNoError(t, err)
	defer output.Close()
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	start := len(driver.launches)
	err = hipRunGemma4Q4DeviceGELUTanhMLPWithDeviceInputOutput(context.Background(), driver, input, cfg.GateProjection, cfg.UpProjection, cfg.DownProjection, output, workspace)
	core.RequireNoError(t, err)

	launches := driver.launches[start:]
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameMLXQ4GELUTanhMLPQ4G32Cols1536Persistent))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameMLXQ4GELUTanhMul))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameMLXQ4GELUTanhMulQ4G32Cols1536Row16))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameMLXQ4Proj))
	for _, launch := range launches {
		if launch.Name == hipKernelNameMLXQ4GELUTanhMLPQ4G32Cols1536Persistent {
			core.AssertEqual(t, hipMLXQ4GELUTanhMLPPersistentBlocks, launch.GridX)
			core.AssertEqual(t, hipMLXQ4ProjectionBlockSize, launch.BlockX)
		}
	}
}

func TestHIPGemma4Q4PrefillPerLayerInputProjectionBatch_BatchOneUsesSingleGELUProjection_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()

	tokenCount := 1
	inputValues := make([]float32, cfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(index + 1)
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill per-layer batch-one hidden fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()

	perLayerValues := make([]float32, cfg.PerLayerInput.InputSize)
	for index := range perLayerValues {
		perLayerValues[index] = float32(index + 1)
	}
	perLayerPayload, err := hipFloat32Payload(perLayerValues)
	core.RequireNoError(t, err)
	perLayerInput, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill per-layer batch-one multiplier fixture", perLayerPayload, len(perLayerValues))
	core.RequireNoError(t, err)
	defer perLayerInput.Close()
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	start := len(driver.launches)
	output, err := hipRunGemma4Q4PrefillPerLayerInputProjectionBatchWorkspace(context.Background(), driver, cfg, input, perLayerInput, tokenCount, workspace)
	core.RequireNoError(t, err)
	defer output.Close()

	core.AssertEqual(t, cfg.PerLayerInput.Projection.Rows, output.Count())
	launches := driver.launches[start:]
	singleGELUProjectionLaunches := countLaunchName(launches, hipKernelNameMLXQ4GELUTanhProj) +
		countLaunchName(launches, hipKernelNameMLXQ4GELUTanhProjQ6Row16)
	core.AssertEqual(t, 1, singleGELUProjectionLaunches)
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameMLXQ4GELUTanhProjBatch))
	singleProjectionLaunches := countLaunchName(launches, hipKernelNameMLXQ4Proj) +
		countLaunchName(launches, hipKernelNameMLXQ4ProjCols256) +
		countLaunchName(launches, hipKernelNameMLXQ4ProjQ6G16Row16) +
		countLaunchName(launches, hipKernelNameMLXQ4ProjQ6Row16) +
		countLaunchName(launches, hipKernelNameMLXQ4ProjQ6Row32) +
		countLaunchName(launches, hipKernelNameMLXQ4ProjQ6Row64)
	core.AssertEqual(t, 1, singleProjectionLaunches)
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameMLXQ4ProjBatch))
}

func TestHIPGemma4Q4PrefillMLPBatchWorkspaceReused_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup()

	tokenCount := 2
	inputValues := make([]float32, tokenCount*cfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(index%cfg.HiddenSize + 1)
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill MLP workspace fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()

	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	allocStart := len(driver.allocations)
	output, err := hipRunGemma4Q4PrefillMLPBatchWorkspace(context.Background(), driver, cfg, input, tokenCount, workspace)
	core.RequireNoError(t, err)
	defer output.Close()
	core.AssertEqual(t, tokenCount*cfg.DownProjection.Rows, output.Count())
	core.AssertEqual(t, allocStart+2, len(driver.allocations))

	output, err = hipRunGemma4Q4PrefillMLPBatchWorkspace(context.Background(), driver, cfg, input, tokenCount, workspace)
	core.RequireNoError(t, err)
	defer output.Close()
	core.AssertEqual(t, tokenCount*cfg.DownProjection.Rows, output.Count())
	core.AssertEqual(t, allocStart+2, len(driver.allocations))
}

func TestHIPGemma4Q4PrefillMLPBatch_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup()

	inputValues := make([]float32, cfg.HiddenSize)
	for index := range inputValues {
		inputValues[index] = float32(index + 1)
	}
	payload, err := hipFloat32Payload(inputValues)
	core.RequireNoError(t, err)
	input, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "prefill MLP bad fixture", payload, len(inputValues))
	core.RequireNoError(t, err)
	defer input.Close()
	start := len(driver.launches)

	if _, err := hipRunGemma4Q4PrefillMLPBatch(context.Background(), driver, cfg, input, 0); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillMLPBatch succeeded with zero token count")
	}
	if _, err := hipRunGemma4Q4PrefillMLPBatch(context.Background(), driver, cfg, input, 2); err == nil {
		t.Fatalf("hipRunGemma4Q4PrefillMLPBatch succeeded with mismatched token count")
	}
	core.AssertEqual(t, start, len(driver.launches))
}

func TestHIPGemma4Q4GenerateTokenSeq_BadPrefillUBatchConfig(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup()

	engineConfig := defaultHIPGemma4Q4EngineConfig()
	engineConfig.PrefillUBatchTokens = 0
	stream, streamErr := hipGemma4Q4GenerateTokenSeqWithEngineConfig(context.Background(), &hipLoadedModel{driver: driver}, hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{cfg}}, []int32{1}, inference.GenerateConfig{MaxTokens: 1}, engineConfig)
	for range stream {
		t.Fatalf("hipGemma4Q4GenerateTokenSeq yielded token, want prefill ubatch config error")
	}
	err := streamErr()
	if err == nil {
		t.Fatalf("hipGemma4Q4GenerateTokenSeq succeeded, want prefill ubatch config error")
	}
	core.AssertContains(t, err.Error(), "prefill ubatch tokens")
	core.AssertEqual(t, 0, len(driver.launches))
}

func TestHIPGemma4Q4PerLayerInputPrecompute_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup0()
	layer1, cleanup1 := hipGemma4Q4FixtureConfig(t, driver, 1, 8, 1, 8)
	defer cleanup1()
	layers, cleanupPerLayer := hipGemma4Q4GlobalPerLayerInputFixture(t, driver, []hipGemma4Q4Layer0Config{layer0, layer1})
	defer cleanupPerLayer()

	start := len(driver.launches)
	forward, err := hipRunGemma4Q4SingleTokenForward(context.Background(), driver, hipGemma4Q4ForwardConfig{Layers: layers}, hipGemma4Q4ForwardRequest{
		TokenID:  1,
		Position: 0,
		Epsilon:  1e-6,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, hipKernelStatusLinked, forward.Labels["gemma4_per_layer_inputs"])
	core.AssertEqual(t, "8", forward.Labels["gemma4_per_layer_input_size"])
	core.AssertContains(t, forward.Labels["decode_primitives"], "gemma4_per_layer_input")

	embeddingLaunches := 0
	projectionLaunches := 0
	for _, launch := range driver.launches[start:] {
		switch launch.Name {
		case hipKernelNameEmbedLookup:
			embeddingLaunches++
		case hipKernelNameProjection:
			projectionLaunches++
		}
	}
	core.AssertEqual(t, 2, embeddingLaunches)
	core.AssertEqual(t, 1, projectionLaunches)
}

func TestHIPGemma4Q4PerLayerInputConfigScalesCached_Good(t *testing.T) {
	layerCfg := hipGemma4Q4Layer0Config{HiddenSize: 2048}
	layerCfg.finalizeScales()
	wantLayerEmbedding := hipBFloat16ToFloat32(hipFloat32ToBFloat16(float32(math.Sqrt(float64(layerCfg.HiddenSize)))))
	if layerCfg.EmbeddingScale != wantLayerEmbedding || layerCfg.embeddingScale() != wantLayerEmbedding {
		t.Fatalf("layer embedding scale cached=%v helper=%v want=%v", layerCfg.EmbeddingScale, layerCfg.embeddingScale(), wantLayerEmbedding)
	}
	layerCfg.HiddenSize = 0
	layerCfg.finalizeScales()
	core.AssertEqual(t, float32(0), layerCfg.EmbeddingScale)

	cases := []struct {
		inputSize int
		hidden    int
	}{
		{inputSize: 2, hidden: 2},
		{inputSize: 256, hidden: 2048},
		{inputSize: 384, hidden: 3072},
	}
	for _, tc := range cases {
		cfg := hipGemma4Q4PerLayerInputConfig{
			InputSize: tc.inputSize,
			ModelProjection: hipBF16DeviceWeightConfig{
				Cols: tc.hidden,
			},
		}
		cfg.finalizeScales()
		wantEmbedding := hipBFloat16ToFloat32(hipFloat32ToBFloat16(float32(math.Sqrt(float64(tc.inputSize)))))
		if cfg.EmbeddingScale != wantEmbedding || cfg.embeddingScale() != wantEmbedding {
			t.Fatalf("embedding scale input=%d cached=%v helper=%v want=%v", tc.inputSize, cfg.EmbeddingScale, cfg.embeddingScale(), wantEmbedding)
		}
		wantProjection := float32(math.Pow(float64(tc.hidden), -0.5))
		if cfg.ModelProjectionScale != wantProjection || cfg.modelProjectionScale() != wantProjection {
			t.Fatalf("projection scale hidden=%d cached=%v helper=%v want=%v", tc.hidden, cfg.ModelProjectionScale, cfg.modelProjectionScale(), wantProjection)
		}
	}
	wantCombine := float32(math.Pow(2, -0.5))
	if hipGemma4Q4PerLayerCombineScale != wantCombine {
		t.Fatalf("per-layer combine scale = %v, want %v", hipGemma4Q4PerLayerCombineScale, wantCombine)
	}
	cfg := hipGemma4Q4PerLayerInputConfig{
		InputSize: 128,
		ModelProjection: hipBF16DeviceWeightConfig{
			Cols: 1024,
		},
	}
	cfg.finalizeScales()
	cfg.InputSize = 0
	cfg.ModelProjection.Cols = 0
	cfg.finalizeScales()
	core.AssertEqual(t, float32(0), cfg.EmbeddingScale)
	core.AssertEqual(t, float32(0), cfg.ModelProjectionScale)
}

func TestHIPGemma4Q4PerLayerInputQuantizedModelProjection_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup0()
	layer1, cleanup1 := hipGemma4Q4FixtureConfig(t, driver, 1, 4, 2, 8)
	defer cleanup1()
	layers, cleanupPerLayer := hipGemma4Q4GlobalPerLayerInputFixture(t, driver, []hipGemma4Q4Layer0Config{layer0, layer1})
	defer cleanupPerLayer()

	rows := layer0.HiddenSize * len(layers)
	cols := layer0.HiddenSize
	groupSize := layer0.GroupSize
	uploadU32 := func(label string, count int) *hipDeviceByteBuffer {
		payload, err := hipUint32Payload(make([]uint32, count))
		core.RequireNoError(t, err)
		buffer, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, label, payload, count)
		core.RequireNoError(t, err)
		return buffer
	}
	uploadU16 := func(label string, count int) *hipDeviceByteBuffer {
		payload, err := hipUint16Payload(make([]uint16, count))
		core.RequireNoError(t, err)
		buffer, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, label, payload, count)
		core.RequireNoError(t, err)
		return buffer
	}
	weights := uploadU32("quantized per-layer model projection weights", rows*(cols/8))
	defer weights.Close()
	scales := uploadU16("quantized per-layer model projection scales", rows*(cols/groupSize))
	defer scales.Close()
	biases := uploadU16("quantized per-layer model projection biases", rows*(cols/groupSize))
	defer biases.Close()
	projection := hipMLXQ4DeviceWeightConfig{
		WeightPointer: weights.Pointer(),
		ScalePointer:  scales.Pointer(),
		BiasPointer:   biases.Pointer(),
		WeightBytes:   weights.SizeBytes(),
		ScaleBytes:    scales.SizeBytes(),
		BiasBytes:     biases.SizeBytes(),
		Rows:          rows,
		Cols:          cols,
		GroupSize:     groupSize,
		Bits:          4,
	}
	for index := range layers {
		layers[index].PerLayerInput.ModelProjection = hipBF16DeviceWeightConfig{}
		layers[index].PerLayerInput.ModelProjectionQ4 = projection
		layers[index].finalizeScales()
		core.RequireNoError(t, layers[index].validatePerLayerInput())
	}

	assertQuantizedProjectionLaunch := func(start int) {
		t.Helper()
		quantized := 0
		dense := 0
		for _, launch := range driver.launches[start:] {
			switch launch.Name {
			case hipKernelNameMLXQ4Proj:
				quantized++
			case hipKernelNameProjection:
				dense++
			}
		}
		core.AssertEqual(t, 1, quantized)
		core.AssertEqual(t, 0, dense)
	}

	start := len(driver.launches)
	hostSet, err := hipRunGemma4Q4PerLayerInputSet(context.Background(), driver, layers[0].PerLayerInput, 0, make([]float32, cols), 1e-6)
	core.RequireNoError(t, err)
	core.AssertEqual(t, len(layers), len(hostSet))
	assertQuantizedProjectionLaunch(start)

	hiddenPayload, err := hipFloat32Payload(make([]float32, cols))
	core.RequireNoError(t, err)
	hidden, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "quantized per-layer hidden", hiddenPayload, cols)
	core.RequireNoError(t, err)
	defer hidden.Close()
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	start = len(driver.launches)
	deviceSet, err := hipRunGemma4Q4PerLayerInputConfigDeviceSet(context.Background(), driver, layers[0].PerLayerInput, 0, nil, hidden, 1e-6, workspace)
	core.RequireNoError(t, err)
	defer deviceSet.Close()
	core.AssertEqual(t, len(layers), deviceSet.LayerCount())
	assertQuantizedProjectionLaunch(start)

	tokens := []int32{0, 1}
	hiddenBatchPayload, err := hipFloat32Payload(make([]float32, cols*len(tokens)))
	core.RequireNoError(t, err)
	hiddenBatch, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "quantized per-layer hidden batch", hiddenBatchPayload, cols*len(tokens))
	core.RequireNoError(t, err)
	defer hiddenBatch.Close()
	start = len(driver.launches)
	prefillSet, err := hipRunGemma4Q4PrefillPerLayerInputDeviceSetBatchWorkspace(context.Background(), driver, hipGemma4Q4ForwardConfig{Layers: layers}, tokens, hiddenBatch, 1e-6, workspace)
	core.RequireNoError(t, err)
	defer prefillSet.Close()
	core.AssertEqual(t, len(layers), prefillSet.LayerCount())
	quantizedBatch := 0
	denseBatch := 0
	for _, launch := range driver.launches[start:] {
		switch launch.Name {
		case hipKernelNameMLXQ4ProjBatch:
			quantizedBatch++
		case hipKernelNameProjectionBatch:
			denseBatch++
		}
	}
	core.AssertEqual(t, 1, quantizedBatch)
	core.AssertEqual(t, 0, denseBatch)
}

func TestHIPGemma4Q4PerLayerInputConfigDeviceSetWorkspaceScalesProjectionInPlace_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup0()
	layer1, cleanup1 := hipGemma4Q4FixtureConfig(t, driver, 1, 4, 2, 8)
	defer cleanup1()
	layers, cleanupPerLayer := hipGemma4Q4GlobalPerLayerInputFixture(t, driver, []hipGemma4Q4Layer0Config{layer0, layer1})
	defer cleanupPerLayer()

	hiddenValues := make([]float32, layer0.HiddenSize)
	payload, err := hipFloat32Payload(hiddenValues)
	core.RequireNoError(t, err)
	hidden, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, "per-layer single-token hidden", payload, len(hiddenValues))
	core.RequireNoError(t, err)
	defer hidden.Close()

	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	allocStart := len(driver.allocations)
	set, err := hipRunGemma4Q4PerLayerInputConfigDeviceSet(context.Background(), driver, layers[0].PerLayerInput, 0, nil, hidden, 1e-6, workspace)
	core.RequireNoError(t, err)
	defer set.Close()

	core.AssertEqual(t, len(layers), set.LayerCount())
	core.AssertEqual(t, layer0.HiddenSize, set.Layer(0).Count())
	if workspace.PerLayerProjScaled != nil {
		t.Fatalf("per-layer projected scaled workspace allocated: %+v", workspace.PerLayerProjScaled)
	}
	if workspace.PerLayerNorm != nil {
		t.Fatalf("per-layer norm workspace allocated: %+v", workspace.PerLayerNorm)
	}
	if len(workspace.PerLayerOutput) > 0 {
		t.Fatalf("per-layer output workspace allocated: %+v", workspace.PerLayerOutput)
	}
	core.AssertEqual(t, 3, len(driver.allocations)-allocStart)
}

func TestHIPGemma4Q4SharedKV_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 8, 1, 8)
	defer cleanup0()
	layer1, cleanup1 := hipGemma4Q4FixtureConfig(t, driver, 1, 4, 2, 8)
	defer cleanup1()
	layer2, cleanup2 := hipGemma4Q4FixtureConfig(t, driver, 2, 8, 1, 8)
	defer cleanup2()
	layer3, cleanup3 := hipGemma4Q4FixtureConfig(t, driver, 3, 4, 2, 8)
	defer cleanup3()
	layer0.LayerType = "sliding_attention"
	layer1.LayerType = "full_attention"
	layer2.LayerType = "sliding_attention"
	layer3.LayerType = "full_attention"
	cfg := hipGemma4Q4ForwardConfig{
		Layers:         []hipGemma4Q4Layer0Config{layer0, layer1, layer2, layer3},
		KVSharedLayers: 2,
	}
	sources := hipGemma4Q4SharedKVSourceByLayer(cfg)
	core.AssertEqual(t, []int{0, 1, 0, 1}, sources)

	start := len(driver.launches)
	forward, err := hipRunGemma4Q4SingleTokenForward(context.Background(), driver, cfg, hipGemma4Q4ForwardRequest{
		TokenID:  1,
		Position: 0,
		Epsilon:  1e-6,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, "2", forward.Labels["gemma4_q4_kv_shared_layers"])
	core.AssertEqual(t, "2", forward.Labels["gemma4_q4_kv_shared_runtime_layers"])
	core.AssertContains(t, forward.Labels["decode_primitives"], "gemma4_shared_kv")
	core.AssertEqual(t, layer0.HeadDim, len(forward.LayerResults[2].UpdatedKeys))
	core.AssertEqual(t, layer1.HeadDim, len(forward.LayerResults[3].UpdatedKeys))

	q4Ops := 0
	tripleQ4Launches := 0
	for _, launch := range driver.launches[start:] {
		switch launch.Name {
		case hipKernelNameMLXQ4Proj:
			q4Ops++
		case hipKernelNameMLXQ4TripleProj:
			q4Ops += 3
			tripleQ4Launches++
		}
	}
	core.AssertEqual(t, 17, q4Ops)
	core.AssertEqual(t, 2, tripleQ4Launches)
}

func TestHIPGemma4Q4LoadedTextConfigOverridesHeadDimHeuristics_Good(t *testing.T) {
	model := &hipLoadedModel{
		contextSize: 2048,
		gemma4TextConfig: nativeGemma4TextConfig{
			LayerTypes:        []string{"sliding_attention", "full_attention"},
			KVSharedLayers:    18,
			KVSharedLayersSet: true,
			SlidingWindow:     1024,
			RoPEParameters: map[string]nativeGemma4RoPEParameters{
				"sliding_attention": {RopeTheta: 10000, RopeType: "default"},
				"full_attention":    {PartialRotaryFactor: 0.25, RopeTheta: 1000000, RopeType: "proportional", Factor: 8},
			},
		},
	}

	core.AssertEqual(t, "sliding_attention", model.loadedGemma4Q4LayerType(0, 512))
	slidingBase, slidingRotaryDim, slidingFrequencyScale := model.loadedGemma4Q4LayerRoPE("sliding_attention", 512)
	core.AssertEqual(t, float32(10000), slidingBase)
	core.AssertEqual(t, 512, slidingRotaryDim)
	core.AssertEqual(t, float32(1), slidingFrequencyScale)
	core.AssertEqual(t, 1024, model.loadedGemma4Q4EffectiveSlidingWindow("sliding_attention", 512))

	core.AssertEqual(t, "full_attention", model.loadedGemma4Q4LayerType(1, 1024))
	fullBase, fullRotaryDim, fullFrequencyScale := model.loadedGemma4Q4LayerRoPE("full_attention", 1024)
	core.AssertEqual(t, float32(1000000), fullBase)
	core.AssertEqual(t, 256, fullRotaryDim)
	core.AssertEqual(t, float32(0.125), fullFrequencyScale)
	core.AssertEqual(t, 0, model.loadedGemma4Q4EffectiveSlidingWindow("full_attention", 1024))
	core.AssertEqual(t, 18, model.loadedGemma4Q4KVSharedLayers(42))
}

func TestHIPGemma4Q4LoadedTextConfigKEqVOnlyFullAttention_Good(t *testing.T) {
	model := &hipLoadedModel{gemma4TextConfig: nativeGemma4TextConfig{AttentionKEqV: true}}

	core.AssertEqual(t, false, model.loadedGemma4Q4AttentionKEqV("sliding_attention"))
	core.AssertEqual(t, true, model.loadedGemma4Q4AttentionKEqV("full_attention"))

	cfg, cleanup := hipGemma4Q4Layer0FixtureConfig(t, &fakeHIPDriver{available: true})
	defer cleanup()
	cfg.LayerType = "sliding_attention"
	cfg.AttentionKEqV = true
	err := cfg.validate()
	if err == nil {
		t.Fatal("validate K=V sliding layer error = nil")
	}
	core.AssertContains(t, err.Error(), "K=V attention is only valid for full-attention layers")
}

func TestHIPGemma4Q4LoadedGGUFFullAttentionInfersKEqVFromMissingV_Good(t *testing.T) {
	const prefix = "language_model.model.layers.5"
	model := &hipLoadedModel{
		modelLabels: map[string]string{
			"format":               "gguf",
			"gemma4_source_format": "gguf",
		},
		tensors: map[string]hipTensor{
			"blk.5.attn_k.weight": {pointer: 1},
		},
	}
	core.AssertEqual(t, true, model.loadedGemma4Q4LayerAttentionKEqV(prefix, "full_attention"))
	core.AssertEqual(t, false, model.loadedGemma4Q4LayerAttentionKEqV(prefix, "sliding_attention"))
	model.tensors["blk.5.attn_v.weight"] = hipTensor{pointer: 2}
	core.AssertEqual(t, false, model.loadedGemma4Q4LayerAttentionKEqV(prefix, "full_attention"))
}

func TestHIPGemma4Q4LoadedSafetensorsFullAttentionInfersKEqVFromMissingV_Good(t *testing.T) {
	const prefix = "language_model.model.layers.5"
	model := &hipLoadedModel{tensors: map[string]hipTensor{
		prefix + ".self_attn.k_proj.weight": {pointer: 1},
	}}

	core.AssertEqual(t, true, model.loadedGemma4Q4LayerAttentionKEqV(prefix, "full_attention"))
	core.AssertEqual(t, false, model.loadedGemma4Q4LayerAttentionKEqV(prefix, "sliding_attention"))
	model.tensors[prefix+".self_attn.v_proj.weight"] = hipTensor{pointer: 2}
	core.AssertEqual(t, false, model.loadedGemma4Q4LayerAttentionKEqV(prefix, "full_attention"))
}

func TestHIPGemma4Q4LoadedNormConfigPreservesF32_Good(t *testing.T) {
	model := &hipLoadedModel{tensors: map[string]hipTensor{
		"language_model.model.norm.weight": {
			info: nativeTensorInfo{
				Name:       "language_model.model.norm.weight",
				Type:       0,
				TypeName:   "F32",
				Dimensions: []uint64{4},
				ByteSize:   16,
			},
			pointer: 0x1000,
		},
	}}

	cfg, err := model.loadedGemma4NormConfig("language_model.model.norm.weight", "final_norm", 4)
	core.RequireNoError(t, err)
	core.AssertEqual(t, hipRMSNormWeightEncodingF32, cfg.WeightEncoding)
	core.AssertEqual(t, uint64(16), cfg.WeightBytes)
	core.AssertEqual(t, 4, cfg.Count)
}

func TestHIPGemma4Q4LoadedLayerScalarPreservesF32_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	payload := make([]byte, 4)
	binary.LittleEndian.PutUint32(payload, math.Float32bits(0.125))
	pointer, err := driver.Malloc(uint64(len(payload)))
	core.RequireNoError(t, err)
	core.RequireNoError(t, driver.CopyHostToDevice(pointer, payload))
	model := &hipLoadedModel{
		driver: driver,
		tensors: map[string]hipTensor{
			"language_model.model.layers.0.layer_scalar": {
				info: nativeTensorInfo{
					Name:       "language_model.model.layers.0.layer_scalar",
					Type:       0,
					TypeName:   "F32",
					Dimensions: []uint64{1},
					ByteSize:   4,
				},
				pointer: pointer,
			},
		},
	}

	scalar, err := model.loadedGemma4Q4LayerScalar(0)
	core.RequireNoError(t, err)
	assertFloat32Near(t, 0.125, scalar)
}

func TestHIPAttachedDrafterAssistantNormPlanPreservesF32_Good(t *testing.T) {
	cfg, err := hipAttachedDrafterAssistantNormPlan("model.norm.weight", hipTensor{
		info: nativeTensorInfo{
			Name:       "model.norm.weight",
			Type:       0,
			TypeName:   "F32",
			Dimensions: []uint64{4},
			ByteSize:   16,
		},
		pointer: 0x2000,
	}, 4)
	core.RequireNoError(t, err)
	core.AssertEqual(t, hipRMSNormWeightEncodingF32, cfg.WeightEncoding)
	core.AssertEqual(t, uint64(16), cfg.WeightBytes)
}

func TestHIPAttachedDrafterAssistantLayerScalarPreservesF32_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	payload := make([]byte, 4)
	binary.LittleEndian.PutUint32(payload, math.Float32bits(0.25))
	pointer, err := driver.Malloc(uint64(len(payload)))
	core.RequireNoError(t, err)
	core.RequireNoError(t, driver.CopyHostToDevice(pointer, payload))

	scalar, err := hipAttachedDrafterAssistantLayerScalar(&hipLoadedModel{driver: driver}, hipTensor{
		info: nativeTensorInfo{
			Name:       "language_model.model.layers.0.layer_scalar",
			Type:       0,
			TypeName:   "F32",
			Dimensions: []uint64{1},
			ByteSize:   4,
		},
		pointer: pointer,
	})
	core.RequireNoError(t, err)
	assertFloat32Near(t, 0.25, scalar)
}

func TestHIPGemma4Q4LoadedConfigAttentionKEqVSkipsVProjection_Good(t *testing.T) {
	const (
		hidden    = 8
		vocab     = 2
		groupSize = 8
	)
	model := &hipLoadedModel{
		driver: &fakeHIPDriver{available: true},
		modelInfo: inference.ModelInfo{
			Architecture: "gemma4_text",
			VocabSize:    vocab,
			HiddenSize:   hidden,
			NumLayers:    1,
			QuantBits:    4,
			QuantGroup:   groupSize,
		},
		modelLabels: linkedGemma4TestLabels("E2B", "q4"),
		gemma4TextConfig: nativeGemma4TextConfig{
			AttentionKEqV: true,
			LayerTypes:    []string{"full_attention"},
		},
		tensors: map[string]hipTensor{},
	}
	nextPointer := nativeDevicePointer(0x1000)
	addTensor := func(name, typeName string, dims []uint64, bytes uint64) {
		t.Helper()
		model.tensors[name] = hipTensor{
			info: nativeTensorInfo{
				Name:       name,
				TypeName:   typeName,
				Dimensions: dims,
				ByteSize:   bytes,
			},
			pointer: nextPointer,
		}
		nextPointer += nativeDevicePointer(bytes) + 0x100
	}
	addQ4Projection := func(baseName string, rows, cols int) {
		t.Helper()
		groups := cols / groupSize
		addTensor(baseName+".weight", "U32", []uint64{uint64(rows), uint64(cols / 8)}, uint64(rows*(cols/8)*4))
		addTensor(baseName+".scales", "BF16", []uint64{uint64(rows), uint64(groups)}, uint64(rows*groups*2))
		addTensor(baseName+".biases", "BF16", []uint64{uint64(rows), uint64(groups)}, uint64(rows*groups*2))
	}
	addNorm := func(name string, count int) {
		t.Helper()
		addTensor(name, "BF16", []uint64{uint64(count)}, uint64(count*2))
	}

	addQ4Projection("language_model.model.embed_tokens", vocab, hidden)
	prefix := "language_model.model.layers.0"
	addNorm(prefix+".input_layernorm.weight", hidden)
	addNorm(prefix+".self_attn.q_norm.weight", hidden)
	addNorm(prefix+".self_attn.k_norm.weight", hidden)
	addNorm(prefix+".post_attention_layernorm.weight", hidden)
	addNorm(prefix+".pre_feedforward_layernorm.weight", hidden)
	addNorm(prefix+".post_feedforward_layernorm.weight", hidden)
	addNorm("language_model.model.norm.weight", hidden)
	addQ4Projection(prefix+".self_attn.q_proj", hidden, hidden)
	addQ4Projection(prefix+".self_attn.k_proj", hidden, hidden)
	addQ4Projection(prefix+".self_attn.o_proj", hidden, hidden)
	addQ4Projection(prefix+".mlp.gate_proj", hidden*2, hidden)
	addQ4Projection(prefix+".mlp.up_proj", hidden*2, hidden)
	addQ4Projection(prefix+".mlp.down_proj", hidden, hidden*2)

	cfg, err := model.loadedGemma4Q4LayerConfig(0)
	core.RequireNoError(t, err)
	core.AssertEqual(t, true, cfg.AttentionKEqV)
	core.AssertEqual(t, cfg.KeyProjection.WeightPointer, cfg.ValueProjection.WeightPointer)
	core.AssertEqual(t, cfg.KeyProjection.ScalePointer, cfg.ValueProjection.ScalePointer)
	core.AssertEqual(t, cfg.KeyProjection.BiasPointer, cfg.ValueProjection.BiasPointer)
	core.AssertEqual(t, false, model.hasHIPTensor(prefix+".self_attn.v_proj.weight"))
}

func TestHIPGemma4Q4LoadedConfigSharedKVOmitsRedundantTensors_Good(t *testing.T) {
	const (
		hidden    = 8
		vocab     = 2
		groupSize = 8
	)
	model := &hipLoadedModel{
		driver: &fakeHIPDriver{available: true},
		modelInfo: inference.ModelInfo{
			Architecture: "gemma4_text",
			VocabSize:    vocab,
			HiddenSize:   hidden,
			NumLayers:    2,
			QuantBits:    4,
			QuantGroup:   groupSize,
		},
		modelLabels: linkedGemma4TestLabels("E4B", "q4"),
		gemma4TextConfig: nativeGemma4TextConfig{
			HeadDim:           hidden,
			GlobalHeadDim:     hidden,
			KVSharedLayers:    1,
			KVSharedLayersSet: true,
			LayerTypes:        []string{"full_attention", "full_attention"},
		},
		tensors: map[string]hipTensor{},
	}
	nextPointer := nativeDevicePointer(0x1000)
	addTensor := func(name, typeName string, dims []uint64, bytes uint64) {
		t.Helper()
		model.tensors[name] = hipTensor{
			info: nativeTensorInfo{
				Name:       name,
				TypeName:   typeName,
				Dimensions: dims,
				ByteSize:   bytes,
			},
			pointer: nextPointer,
		}
		nextPointer += nativeDevicePointer(bytes) + 0x100
	}
	addQ4Projection := func(baseName string, rows, cols int) {
		t.Helper()
		groups := cols / groupSize
		addTensor(baseName+".weight", "U32", []uint64{uint64(rows), uint64(cols / 8)}, uint64(rows*(cols/8)*4))
		addTensor(baseName+".scales", "BF16", []uint64{uint64(rows), uint64(groups)}, uint64(rows*groups*2))
		addTensor(baseName+".biases", "BF16", []uint64{uint64(rows), uint64(groups)}, uint64(rows*groups*2))
	}
	addNorm := func(name string, count int) {
		t.Helper()
		addTensor(name, "BF16", []uint64{uint64(count)}, uint64(count*2))
	}

	addQ4Projection("language_model.model.embed_tokens", vocab, hidden)
	addNorm("language_model.model.norm.weight", hidden)
	for layer := 0; layer < 2; layer++ {
		prefix := core.Sprintf("language_model.model.layers.%d", layer)
		addNorm(prefix+".input_layernorm.weight", hidden)
		addNorm(prefix+".self_attn.q_norm.weight", hidden)
		addNorm(prefix+".post_attention_layernorm.weight", hidden)
		addNorm(prefix+".pre_feedforward_layernorm.weight", hidden)
		addNorm(prefix+".post_feedforward_layernorm.weight", hidden)
		addQ4Projection(prefix+".self_attn.q_proj", hidden, hidden)
		addQ4Projection(prefix+".self_attn.o_proj", hidden, hidden)
		addQ4Projection(prefix+".mlp.gate_proj", hidden*2, hidden)
		addQ4Projection(prefix+".mlp.up_proj", hidden*2, hidden)
		addQ4Projection(prefix+".mlp.down_proj", hidden, hidden*2)
		if layer == 0 {
			addNorm(prefix+".self_attn.k_norm.weight", hidden)
			addQ4Projection(prefix+".self_attn.k_proj", hidden, hidden)
			addQ4Projection(prefix+".self_attn.v_proj", hidden, hidden)
		}
	}

	cfg, err := model.loadedGemma4Q4ForwardConfig(2)
	core.RequireNoError(t, err)
	core.AssertEqual(t, []int{0, 0}, cfg.SharedKVSources)
	core.AssertEqual(t, cfg.Layers[0].KeyProjection.WeightPointer, cfg.Layers[1].KeyProjection.WeightPointer)
	core.AssertEqual(t, cfg.Layers[0].ValueProjection.WeightPointer, cfg.Layers[1].ValueProjection.WeightPointer)
	core.AssertEqual(t, cfg.Layers[0].KeyNorm.WeightPointer, cfg.Layers[1].KeyNorm.WeightPointer)
}

func TestHIPGemma4Q4RequiredTensorResolvesGGUFAliases_Good(t *testing.T) {
	model := &hipLoadedModel{
		modelLabels: map[string]string{
			"format":               "gguf",
			"gemma4_source_format": "gguf",
		},
		tensors: map[string]hipTensor{},
	}
	addTensor := func(name string, pointer nativeDevicePointer) {
		t.Helper()
		model.tensors[name] = hipTensor{
			info: nativeTensorInfo{
				Name:       name,
				TypeName:   "BF16",
				Dimensions: []uint64{8},
				ByteSize:   16,
			},
			pointer: pointer,
		}
	}
	addTensor("token_embd.weight", 0x1000)
	addTensor("output_norm.weight", 0x2000)
	addTensor("per_layer_token_embd.weight", 0x3000)
	addTensor("per_layer_model_proj.weight", 0x4000)
	addTensor("per_layer_proj_norm.weight", 0x5000)
	addTensor("blk.0.attn_q.weight", 0x6000)
	addTensor("blk.0.attn_k.weight", 0x7000)
	addTensor("blk.0.attn_v.weight", 0x8000)
	addTensor("blk.0.attn_output.weight", 0x9000)
	addTensor("blk.0.attn_norm.weight", 0xa000)
	addTensor("blk.0.attn_q_norm.weight", 0xb000)
	addTensor("blk.0.attn_k_norm.weight", 0xc000)
	addTensor("blk.0.ffn_gate.weight", 0xd000)
	addTensor("blk.0.ffn_up.weight", 0xe000)
	addTensor("blk.0.ffn_down.weight", 0xf000)
	addTensor("blk.0.ffn_norm.weight", 0x11000)
	addTensor("blk.0.post_attention_norm.weight", 0x12000)
	addTensor("blk.0.post_ffw_norm.weight", 0x13000)
	addTensor("blk.0.inp_gate.weight", 0x14000)
	addTensor("blk.0.proj.weight", 0x15000)
	addTensor("blk.0.post_norm.weight", 0x16000)

	for _, tc := range []struct {
		name    string
		pointer nativeDevicePointer
	}{
		{name: "language_model.model.embed_tokens.weight", pointer: 0x1000},
		{name: "language_model.model.norm.weight", pointer: 0x2000},
		{name: "language_model.model.embed_tokens_per_layer.weight", pointer: 0x3000},
		{name: "language_model.model.per_layer_model_projection.weight", pointer: 0x4000},
		{name: "language_model.model.per_layer_projection_norm.weight", pointer: 0x5000},
		{name: "language_model.model.layers.0.self_attn.q_proj.weight", pointer: 0x6000},
		{name: "language_model.model.layers.0.self_attn.k_proj.weight", pointer: 0x7000},
		{name: "language_model.model.layers.0.self_attn.v_proj.weight", pointer: 0x8000},
		{name: "language_model.model.layers.0.self_attn.o_proj.weight", pointer: 0x9000},
		{name: "language_model.model.layers.0.input_layernorm.weight", pointer: 0xa000},
		{name: "language_model.model.layers.0.self_attn.q_norm.weight", pointer: 0xb000},
		{name: "language_model.model.layers.0.self_attn.k_norm.weight", pointer: 0xc000},
		{name: "language_model.model.layers.0.mlp.gate_proj.weight", pointer: 0xd000},
		{name: "language_model.model.layers.0.mlp.up_proj.weight", pointer: 0xe000},
		{name: "language_model.model.layers.0.mlp.down_proj.weight", pointer: 0xf000},
		{name: "language_model.model.layers.0.pre_feedforward_layernorm.weight", pointer: 0x11000},
		{name: "language_model.model.layers.0.post_attention_layernorm.weight", pointer: 0x12000},
		{name: "language_model.model.layers.0.post_feedforward_layernorm.weight", pointer: 0x13000},
		{name: "language_model.model.layers.0.per_layer_input_gate.weight", pointer: 0x14000},
		{name: "language_model.model.layers.0.per_layer_projection.weight", pointer: 0x15000},
		{name: "language_model.model.layers.0.post_per_layer_input_norm.weight", pointer: 0x16000},
	} {
		if !model.hasHIPTensor(tc.name) {
			t.Fatalf("hasHIPTensor(%q) = false, want GGUF alias", tc.name)
		}
		tensor, err := model.requiredHIPTensor(tc.name, tc.name)
		core.RequireNoError(t, err)
		core.AssertEqual(t, tc.pointer, tensor.pointer)
	}
}

func TestHIPGemma4Q4LoadedTextConfigFinalLogitSoftcap_Good(t *testing.T) {
	core.AssertEqual(t, float32(30), (*hipLoadedModel)(nil).loadedGemma4Q4FinalLogitSoftcap())
	model := &hipLoadedModel{gemma4TextConfig: nativeGemma4TextConfig{FinalLogitSoftcap: 42}}
	core.AssertEqual(t, float32(42), model.loadedGemma4Q4FinalLogitSoftcap())
	model.gemma4TextConfig.FinalLogitSoftcap = -1
	core.AssertEqual(t, float32(30), model.loadedGemma4Q4FinalLogitSoftcap())
	model.gemma4TextConfig.FinalLogitSoftcap = math.Inf(1)
	core.AssertEqual(t, float32(30), model.loadedGemma4Q4FinalLogitSoftcap())
}

func TestHIPGemma4Q4LMHeadProjectionPrefersUntiedHead_Good(t *testing.T) {
	const groupSize = 64
	model := &hipLoadedModel{tensors: map[string]hipTensor{}}
	addQ4ProjectionTensors := func(baseName string, pointer nativeDevicePointer) {
		model.tensors[baseName+".weight"] = hipTensor{
			info:    nativeTensorInfo{TypeName: "U32", Dimensions: []uint64{8, 8}, ByteSize: 256},
			pointer: pointer,
		}
		model.tensors[baseName+".scales"] = hipTensor{
			info:    nativeTensorInfo{TypeName: "BF16", Dimensions: []uint64{8, 1}, ByteSize: 16},
			pointer: pointer + 1,
		}
		model.tensors[baseName+".biases"] = hipTensor{
			info:    nativeTensorInfo{TypeName: "BF16", Dimensions: []uint64{8, 1}, ByteSize: 16},
			pointer: pointer + 2,
		}
	}
	addQ4ProjectionTensors("language_model.model.embed_tokens", 100)
	addQ4ProjectionTensors("language_model.lm_head", 200)

	cfg, rows, cols, err := model.loadedGemma4Q4LMHeadProjectionConfig(groupSize)
	core.RequireNoError(t, err)
	core.AssertEqual(t, nativeDevicePointer(200), cfg.WeightPointer)
	core.AssertEqual(t, 8, rows)
	core.AssertEqual(t, 64, cols)

	delete(model.tensors, "language_model.lm_head.weight")
	delete(model.tensors, "language_model.lm_head.scales")
	delete(model.tensors, "language_model.lm_head.biases")
	cfg, _, _, err = model.loadedGemma4Q4LMHeadProjectionConfig(groupSize)
	core.RequireNoError(t, err)
	core.AssertEqual(t, nativeDevicePointer(100), cfg.WeightPointer)
}

func TestHIPGemma4Q4E4BSharedKVLayoutUsesLayerTypes_Good(t *testing.T) {
	const layerCount = 42
	layers := make([]hipGemma4Q4Layer0Config, layerCount)
	for index := range layers {
		layerType := "full_attention"
		if (index+1)%6 != 0 {
			layerType = "sliding_attention"
		}
		if index == layerCount-1 {
			layerType = "full_attention"
		}
		layers[index] = hipGemma4Q4Layer0Config{Layer: index, LayerType: layerType, HeadDim: 512}
	}

	sources := hipGemma4Q4BuildSharedKVSourceByLayer(hipGemma4Q4ForwardConfig{Layers: layers, KVSharedLayers: 18})

	ownerCount := 0
	for index, source := range sources {
		if source == index {
			ownerCount++
		}
	}
	core.AssertEqual(t, 24, ownerCount)
	core.AssertEqual(t, 22, sources[24])
	core.AssertEqual(t, 23, sources[29])
	core.AssertEqual(t, 23, sources[41])
}

func TestHIPGemma4Q4E2BSharedKVLayoutUsesLayerTypes_Good(t *testing.T) {
	const layerCount = 35
	layers := make([]hipGemma4Q4Layer0Config, layerCount)
	slidingLayers := 0
	fullLayers := 0
	for index := range layers {
		layerType := "sliding_attention"
		headDim := 256
		if (index+1)%5 == 0 {
			layerType = "full_attention"
			headDim = 512
		}
		layers[index] = hipGemma4Q4Layer0Config{Layer: index, LayerType: layerType, HeadDim: headDim}
		switch layerType {
		case "sliding_attention":
			slidingLayers++
		case "full_attention":
			fullLayers++
		}
	}

	sources := hipGemma4Q4BuildSharedKVSourceByLayer(hipGemma4Q4ForwardConfig{Layers: layers, KVSharedLayers: 20})

	ownerCount := 0
	for index, source := range sources {
		if source == index {
			ownerCount++
		}
	}
	core.AssertEqual(t, 28, slidingLayers)
	core.AssertEqual(t, 7, fullLayers)
	core.AssertEqual(t, 15, ownerCount)
	core.AssertEqual(t, 13, sources[15])
	core.AssertEqual(t, 14, sources[19])
	core.AssertEqual(t, 14, sources[34])
}

func TestHIPGemma4Q4SharedDeviceKV_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 8, 1, 8)
	defer cleanup0()
	layer1, cleanup1 := hipGemma4Q4FixtureConfig(t, driver, 1, 4, 2, 8)
	defer cleanup1()
	layer2, cleanup2 := hipGemma4Q4FixtureConfig(t, driver, 2, 8, 1, 8)
	defer cleanup2()
	layer3, cleanup3 := hipGemma4Q4FixtureConfig(t, driver, 3, 4, 2, 8)
	defer cleanup3()
	layer0.LayerType = "sliding_attention"
	layer0.SlidingWindow = 1
	layer1.LayerType = "full_attention"
	layer1.SlidingWindow = 0
	layer2.LayerType = "sliding_attention"
	layer2.SlidingWindow = 1
	layer3.LayerType = "full_attention"
	layer3.SlidingWindow = 0
	cfg := hipGemma4Q4ForwardConfig{
		Layers:         []hipGemma4Q4Layer0Config{layer0, layer1, layer2, layer3},
		KVSharedLayers: 2,
	}

	launchStart := len(driver.launches)
	first, firstState, err := hipRunGemma4Q4SingleTokenForwardWithStateInternal(context.Background(), driver, cfg, hipGemma4Q4DecodeState{}, hipGemma4Q4ForwardRequest{
		TokenID:           1,
		Position:          0,
		Epsilon:           1e-6,
		DeviceKVAttention: true,
		DeviceKVMode:      rocmKVCacheModeKQ8VQ4,
		ReturnDeviceState: true,
		DeviceFinalSample: true,
		OmitDebugTensors:  true,
	}, false)
	core.RequireNoError(t, err)
	if first.DeviceState == nil {
		t.Fatal("first forward device state is nil")
	}
	firstLaunches := driver.launches[launchStart:]
	core.AssertEqual(t, []int{1, 1, 1, 1}, first.DeviceState.LayerTokenCounts())
	core.AssertEqual(t, "0", first.Labels["attention_kv_remirror_layers"])
	core.AssertEqual(t, "2", first.Labels["attention_kv_shared_device_layers"])
	core.AssertEqual(t, "2", first.Labels["gemma4_q4_device_kv_shared_layers"])
	core.AssertEqual(t, 2, countKVEncodeTokenLaunches(firstLaunches))
	core.AssertEqual(t, 2, countLaunchName(firstLaunches, hipKernelNameMLXQ4TripleProj))
	for index, layer := range firstState.Layers {
		if len(layer.Keys) != 0 || len(layer.Values) != 0 {
			t.Fatalf("first host state layer %d retained host KV in device-only generation path", index)
		}
	}

	priorDeviceState := first.DeviceState
	first.DeviceState = nil
	secondLaunchStart := len(driver.launches)
	second, secondState, err := hipRunGemma4Q4SingleTokenForwardWithStateInternal(context.Background(), driver, cfg, firstState, hipGemma4Q4ForwardRequest{
		TokenID:           int32(first.Greedy.TokenID),
		Position:          1,
		Epsilon:           1e-6,
		DeviceKVAttention: true,
		DeviceKVMode:      rocmKVCacheModeKQ8VQ4,
		PriorDeviceState:  priorDeviceState,
		ReturnDeviceState: true,
		DeviceFinalSample: true,
		OmitDebugTensors:  true,
	}, false)
	if err != nil {
		_ = priorDeviceState.Close()
	}
	core.RequireNoError(t, err)
	defer second.DeviceState.Close()
	core.AssertEqual(t, true, priorDeviceState.closed)
	core.AssertEqual(t, []int{1, 2, 1, 2}, second.DeviceState.LayerTokenCounts())
	core.AssertEqual(t, "2", second.Labels["attention_kv_append_layers"])
	core.AssertEqual(t, "0", second.Labels["attention_kv_remirror_layers"])
	core.AssertEqual(t, "2", second.Labels["attention_kv_shared_device_layers"])
	core.AssertEqual(t, "2", second.Labels["gemma4_q4_device_kv_shared_layers"])
	secondLaunches := driver.launches[secondLaunchStart:]
	core.AssertEqual(t, 2, countLaunchName(secondLaunches, hipKernelNameMLXQ4TripleProj))
	for index, layer := range secondState.Layers {
		if len(layer.Keys) != 0 || len(layer.Values) != 0 {
			t.Fatalf("second host state layer %d retained host KV in device-only generation path", index)
		}
	}
	if countDeviceAttentionLaunches(driver.launches[launchStart:]) == 0 {
		t.Fatalf("Gemma4 q4 shared-device forward launched no descriptor-backed attention kernels")
	}
}

func TestHIPGemma4Q4DecoderLayerAttentionKEqVUsesPairProjection_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()
	cfg.LayerType = "full_attention"
	cfg.AttentionKEqV = true
	cfg.SlidingWindow = 0
	cfg.ValueProjection = cfg.KeyProjection
	cfg.PerLayerInput = hipGemma4Q4PerLayerInputConfig{}
	input, err := hipUploadGemma4Q4Float32Input(driver, "Gemma4 q4 K=V decoder route input", []float32{1, 2, 3, 4, 5, 6, 7, 8})
	core.RequireNoError(t, err)
	defer input.Close()
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	launchStart := len(driver.launches)
	layer, err := hipRunGemma4Q4DecoderLayerInternalWithDeviceInput(context.Background(), driver, cfg, nil, input, hipGemma4Q4DecoderLayerRequest{
		Position:           0,
		Epsilon:            1e-6,
		DeviceKVAttention:  true,
		DeviceKVMode:       rocmKVCacheModeKQ8VQ4,
		KeepDeviceKV:       true,
		OmitHostKV:         true,
		OmitDebugTensors:   true,
		ReturnDeviceHidden: true,
		AttentionWorkspace: workspace,
	}, true)
	core.RequireNoError(t, err)
	if layer.DeviceLayerValid {
		defer layer.DeviceLayer.Close()
	}
	if layer.DeviceFinalHidden != nil && !layer.DeviceFinalHiddenBorrowed {
		defer layer.DeviceFinalHidden.Close()
	}
	if layer.DeviceNextLayerInput != nil && !layer.DeviceNextLayerInputBorrowed {
		defer layer.DeviceNextLayerInput.Close()
	}

	launches := driver.launches[launchStart:]
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameMLXQ4PairProj))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameMLXQ4TripleProj))
	core.AssertEqual(t, 2, countLaunchName(launches, hipKernelNameMLXQ4Proj))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameRMSNormRoPEHeadsPair))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameRMSNormRoPEHeads))
	core.AssertEqual(t, 1, countKVEncodeTokenLaunches(launches))
	queryRoPE := &workspace.RMSRoPEOutputView
	if queryRoPE.Pointer() == 0 || queryRoPE.Count() != cfg.QueryProjection.Rows {
		t.Fatalf("query RMS RoPE workspace = %#v, want %d rows", queryRoPE, cfg.QueryProjection.Rows)
	}
	keyValueNormPair := &workspace.KeyValueNormFixed
	if keyValueNormPair.Pointer() == 0 || keyValueNormPair.Count() != cfg.HeadDim*2 {
		t.Fatalf("key/value norm workspace = %#v, want %d rows", keyValueNormPair, cfg.HeadDim*2)
	}
	if queryRoPE.Pointer() == keyValueNormPair.Pointer() {
		t.Fatalf("query and key/value norm workspaces aliased at %x", queryRoPE.Pointer())
	}
	if _, ok := workspace.RMSRoPEOutputs[cfg.HeadDim]; ok {
		t.Fatalf("decode key RoPE used shared query RMS RoPE workspace")
	}
	if countDeviceAttentionLaunches(launches) == 0 {
		t.Fatalf("Gemma4 q4 K=V decoder layer launched no descriptor-backed attention kernels")
	}
}

func TestHIPGemma4Q4DecoderLayerPairsQKRoPE_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()
	cfg.LayerType = "full_attention"
	cfg.SlidingWindow = 0
	cfg.RoPERotaryDim = cfg.HeadDim / 2
	cfg.PerLayerInput = hipGemma4Q4PerLayerInputConfig{}
	input, err := hipUploadGemma4Q4Float32Input(driver, "Gemma4 q4 paired Q/K RoPE decoder input", []float32{1, 2, 3, 4, 5, 6, 7, 8})
	core.RequireNoError(t, err)
	defer input.Close()
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	launchStart := len(driver.launches)
	layer, err := hipRunGemma4Q4DecoderLayerInternalWithDeviceInput(context.Background(), driver, cfg, nil, input, hipGemma4Q4DecoderLayerRequest{
		Position:           3,
		Epsilon:            1e-6,
		DeviceKVAttention:  true,
		DeviceKVMode:       rocmKVCacheModeKQ8VQ4,
		KeepDeviceKV:       true,
		OmitHostKV:         true,
		OmitDebugTensors:   true,
		ReturnDeviceHidden: true,
		AttentionWorkspace: workspace,
	}, true)
	core.RequireNoError(t, err)
	if layer.DeviceLayerValid {
		defer layer.DeviceLayer.Close()
	}
	if layer.DeviceFinalHidden != nil && !layer.DeviceFinalHiddenBorrowed {
		defer layer.DeviceFinalHidden.Close()
	}
	if layer.DeviceNextLayerInput != nil && !layer.DeviceNextLayerInputBorrowed {
		defer layer.DeviceNextLayerInput.Close()
	}

	launches := driver.launches[launchStart:]
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameMLXQ4TripleProj))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameRMSNormRoPEHeadsPair))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameRMSNormRoPEHeads))
	var pairLaunch *hipKernelLaunchConfig
	for index := range launches {
		if launches[index].Name == hipKernelNameRMSNormRoPEHeadsPair {
			pairLaunch = &launches[index]
			break
		}
	}
	if pairLaunch == nil {
		t.Fatal("Gemma4 q4 decoder layer launched no paired Q/K RMSNorm RoPE kernel")
	}
	core.AssertEqual(t, uint32(cfg.QueryHeads+firstPositiveInt(cfg.KeyHeads, 1)), pairLaunch.GridX)
	core.AssertEqual(t, uint32(cfg.HeadDim), binary.LittleEndian.Uint32(pairLaunch.Args[56:]))
	core.AssertEqual(t, uint32(cfg.HeadDim), binary.LittleEndian.Uint32(pairLaunch.Args[124:]))
	core.AssertEqual(t, uint32(cfg.RoPERotaryDim), binary.LittleEndian.Uint32(pairLaunch.Args[128:]))
}

func TestHIPGemma4Q4DecoderLayerOmitHostKVRejectsHostSharedFallback_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()
	cfg.LayerType = "full_attention"
	cfg.SlidingWindow = 0
	input, err := hipUploadGemma4Q4Float32Input(driver, "Gemma4 q4 no host KV fallback input", []float32{1, 2, 3, 4, 5, 6, 7, 8})
	core.RequireNoError(t, err)
	defer input.Close()
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	_, err = hipRunGemma4Q4DecoderLayerInternalWithDeviceInput(context.Background(), driver, cfg, nil, input, hipGemma4Q4DecoderLayerRequest{
		Position:           0,
		Epsilon:            1e-6,
		SharedKeys:         []float32{0.1, 0.2, 0.3, 0.4},
		SharedValues:       []float32{0.5, 0.6, 0.7, 0.8},
		DeviceKVAttention:  true,
		DeviceKVMode:       rocmKVCacheModeKQ8VQ4,
		KeepDeviceKV:       true,
		OmitHostKV:         true,
		OmitDebugTensors:   true,
		ReturnDeviceHidden: true,
		AttentionWorkspace: workspace,
	}, true)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "device-only KV path requires device token buffers or shared device KV")
}

func TestHIPGemma4Q4PackagePrefillDecode_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup0()
	layer1, cleanup1 := hipGemma4Q4FixtureConfig(t, driver, 1, 4, 2, 16)
	defer cleanup1()
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0, layer1}}
	model := &hipLoadedModel{driver: driver, modelLabels: linkedGemma4TestLabels("E2B", "q4")}

	launchStart := len(driver.launches)
	prefill, err := hipRunGemma4Q4PackagePrefill(context.Background(), model, cfg, hipPrefillRequest{
		TokenIDs: []int32{1, 0},
	})
	core.RequireNoError(t, err)
	defer prefill.Gemma4Q4DeviceState.Close()

	core.AssertEqual(t, 2, prefill.PromptTokens)
	core.AssertEqual(t, layer0.VocabSize, len(prefill.Logits))
	core.AssertEqual(t, 2, len(prefill.Gemma4Q4State.Layers))
	core.AssertEqual(t, []int{2, 2}, prefill.Gemma4Q4DeviceState.LayerTokenCounts())
	core.AssertEqual(t, "loaded_gemma4_q4_experimental_prefill", prefill.Labels["kernel_scope"])
	core.AssertEqual(t, hipKernelStatusLinked, prefill.Labels["gemma4_q4_prefill_kernel"])
	core.AssertEqual(t, hipKernelStatusNotLinked, prefill.Labels["prefill_kernel"])
	core.AssertEqual(t, hipKernelStatusNotLinked, prefill.Labels["production_prefill"])
	core.AssertEqual(t, hipKernelStatusNotLinked, prefill.Labels["production_decode"])
	core.AssertEqual(t, hipKernelStatusNotLinked, prefill.Labels["production_kv_cache_backing"])
	core.AssertEqual(t, "hip_device_descriptor", prefill.Labels["attention_kv_backing"])
	core.AssertEqual(t, rocmKVCacheModeKQ8VQ4, prefill.Labels["attention_kv_mode"])
	core.AssertEqual(t, "forward_returned_device_state", prefill.Labels["gemma4_q4_device_kv_state"])
	core.AssertEqual(t, "2", prefill.Labels["prefill_prompt_tokens"])
	core.AssertEqual(t, "1", prefill.Labels["decode_position"])
	if countDeviceAttentionLaunches(driver.launches[launchStart:]) == 0 {
		t.Fatalf("Gemma4 q4 package prefill launched no descriptor-backed attention kernels")
	}

	prefillDeviceState := prefill.Gemma4Q4DeviceState
	launchStart = len(driver.launches)
	decode, err := hipRunGemma4Q4PackageDecode(context.Background(), model, cfg, hipDecodeRequest{
		TokenID:             int32(0),
		Gemma4Q4State:       prefill.Gemma4Q4State,
		Gemma4Q4DeviceState: prefillDeviceState,
	})
	core.RequireNoError(t, err)
	defer decode.Gemma4Q4DeviceState.Close()

	core.AssertEqual(t, int32(0), decode.Token.ID)
	core.AssertEqual(t, "<token:0>", decode.Token.Text)
	core.AssertEqual(t, layer0.VocabSize, len(decode.Logits))
	core.AssertEqual(t, 2, len(decode.Gemma4Q4State.Layers))
	core.AssertEqual(t, []int{3, 3}, decode.Gemma4Q4DeviceState.LayerTokenCounts())
	core.AssertEqual(t, true, prefillDeviceState.closed)
	core.AssertEqual(t, "loaded_gemma4_q4_experimental_decode", decode.Labels["kernel_scope"])
	core.AssertEqual(t, hipKernelStatusLinked, decode.Labels["gemma4_q4_decode_kernel"])
	core.AssertEqual(t, hipKernelStatusNotLinked, decode.Labels["decode_kernel"])
	core.AssertEqual(t, hipKernelStatusNotLinked, decode.Labels["production_prefill"])
	core.AssertEqual(t, hipKernelStatusNotLinked, decode.Labels["production_decode"])
	core.AssertEqual(t, hipKernelStatusNotLinked, decode.Labels["production_kv_cache_backing"])
	core.AssertEqual(t, "hip_device_descriptor", decode.Labels["attention_kv_backing"])
	core.AssertEqual(t, rocmKVCacheModeKQ8VQ4, decode.Labels["attention_kv_mode"])
	core.AssertEqual(t, "forward_returned_device_state", decode.Labels["gemma4_q4_device_kv_state"])
	core.AssertEqual(t, "3", decode.Labels["decode_state_tokens"])
	core.AssertEqual(t, "2", decode.Labels["decode_position"])
	core.AssertEqual(t, "2", decode.Labels["attention_kv_append_layers"])
	core.AssertEqual(t, "0", decode.Labels["attention_kv_remirror_layers"])
	if countDeviceAttentionLaunches(driver.launches[launchStart:]) == 0 {
		t.Fatalf("Gemma4 q4 package decode launched no descriptor-backed attention kernels")
	}

	launchStart = len(driver.launches)
	batch := hipGemma4Q4BatchGenerate(context.Background(), model, cfg, []string{"tokens:1,0", "plain"}, inference.GenerateConfig{MaxTokens: 1})
	core.AssertEqual(t, 2, len(batch))
	core.AssertEqual(t, 1, len(batch[0].Tokens))
	core.AssertEqual(t, int32(0), batch[0].Tokens[0].ID)
	core.RequireNoError(t, batch[0].Err)
	core.AssertError(t, batch[1].Err)
	core.AssertContains(t, batch[1].Err.Error(), "native decode kernels are not linked yet")
	if countDeviceAttentionLaunches(driver.launches[launchStart:]) == 0 {
		t.Fatalf("Gemma4 q4 batch generate launched no descriptor-backed attention kernels")
	}

	model.modelInfo = inference.ModelInfo{Architecture: "gemma4", QuantBits: 4, VocabSize: layer0.VocabSize}
	model.modelLabels = linkedGemma4TestLabels("E2B", "q4")
	model.tokenText = &hipTokenTextDecoder{
		vocab: map[string]int32{
			"a": 1,
			"b": 0,
		},
		pieces: map[int32]string{
			0: "b",
			1: "a",
		},
	}
	launchStart = len(driver.launches)
	textBatch := hipGemma4Q4BatchGenerate(context.Background(), model, cfg, []string{"text:a", "a"}, inference.GenerateConfig{MaxTokens: 1})
	core.AssertEqual(t, 2, len(textBatch))
	for index, result := range textBatch {
		core.RequireNoError(t, result.Err)
		if len(result.Tokens) != 1 || result.Tokens[0].ID < 0 || int(result.Tokens[0].ID) >= layer0.VocabSize {
			t.Fatalf("Gemma4 q4 text batch result[%d] = %+v, want one in-vocab token", index, result)
		}
		core.AssertEqual(t, "b", result.Tokens[0].Text)
	}
	if countDeviceAttentionLaunches(driver.launches[launchStart:]) == 0 {
		t.Fatalf("Gemma4 q4 text batch generate launched no descriptor-backed attention kernels")
	}

	launchStart = len(driver.launches)
	classify, err := hipGemma4Q4Classify(context.Background(), model, cfg, []string{"tokens:1,0"}, inference.GenerateConfig{ReturnLogits: true})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, len(classify))
	core.AssertEqual(t, int32(0), classify[0].Token.ID)
	core.AssertEqual(t, layer0.VocabSize, len(classify[0].Logits))
	if countDeviceAttentionLaunches(driver.launches[launchStart:]) == 0 {
		t.Fatalf("Gemma4 q4 classify launched no descriptor-backed attention kernels")
	}
}

func TestHIPGemma4Q4PackageDecodePositionUsesGlobalOwnerState_Good(t *testing.T) {
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{
		{LayerType: "sliding_attention", HeadDim: 4, SlidingWindow: 2},
		{LayerType: "full_attention", HeadDim: 8},
	}}
	state := hipGemma4Q4DecodeState{Layers: []hipGemma4Q4LayerKVState{
		{Keys: make([]float32, 2*4), Values: make([]float32, 2*4)},
		{Keys: make([]float32, 9*8), Values: make([]float32, 9*8)},
	}}

	position, err := hipGemma4Q4PackageDecodePosition(cfg, hipDecodeRequest{Gemma4Q4State: state})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 9, position)
	core.AssertEqual(t, 9, state.tokenCountForConfig(cfg))

	labels := hipGemma4Q4PackageDecodeLabels(cfg, rocmKVCacheModeKQ8VQ4, state, nil, nil)
	core.AssertEqual(t, "9", labels["decode_state_tokens"])

	deviceState := &hipGemma4Q4DeviceDecodeState{layers: []hipGemma4Q4DeviceLayerKVState{
		{cache: &rocmDeviceKVCache{tokenCount: 11}},
	}}
	position, err = hipGemma4Q4PackageDecodePosition(cfg, hipDecodeRequest{
		Gemma4Q4State:       state,
		Gemma4Q4DeviceState: deviceState,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 11, position)

	position, err = hipGemma4Q4PackageDecodePosition(cfg, hipDecodeRequest{
		Position:            7,
		Gemma4Q4State:       state,
		Gemma4Q4DeviceState: deviceState,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 7, position)
}

func assertFloat32SlicesNearRelative(t *testing.T, want, got []float32, absoluteTolerance, relativeTolerance float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("slice len = %d, want %d: %+v", len(got), len(want), got)
	}
	for i := range want {
		tolerance := absoluteTolerance
		if scaled := float32(math.Abs(float64(want[i]))) * relativeTolerance; scaled > tolerance {
			tolerance = scaled
		}
		if math.Abs(float64(want[i]-got[i])) > float64(tolerance) {
			t.Fatalf("slice[%d] = %f, want %f within abs=%f rel=%f; got %+v", i, got[i], want[i], absoluteTolerance, relativeTolerance, got)
		}
	}
}

func TestHIPGemma4Q4SkipFinalSample_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 8, 1, 8)
	defer cleanup0()
	layer1, cleanup1 := hipGemma4Q4FixtureConfig(t, driver, 1, 4, 2, 8)
	defer cleanup1()
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0, layer1}}

	launchStart := len(driver.launches)
	forward, state, err := hipRunGemma4Q4SingleTokenForwardWithStateInternal(context.Background(), driver, cfg, hipGemma4Q4DecodeState{}, hipGemma4Q4ForwardRequest{
		TokenID:           1,
		Position:          0,
		Epsilon:           1e-6,
		DeviceKVAttention: true,
		DeviceKVMode:      rocmKVCacheModeKQ8VQ4,
		ReturnDeviceState: true,
		SkipFinalSample:   true,
		OmitDebugTensors:  true,
	}, false)
	core.RequireNoError(t, err)
	defer forward.DeviceState.Close()

	core.AssertEqual(t, 0, len(forward.Logits))
	core.AssertEqual(t, 0, forward.Greedy.TokenID)
	assertFloat32Near(t, 0, forward.Greedy.Score)
	core.AssertEqual(t, "skipped", forward.Labels["gemma4_q4_final_sample"])
	core.AssertEqual(t, []int{1, 1}, forward.DeviceState.LayerTokenCounts())
	for index, layer := range state.Layers {
		if len(layer.Keys) != 0 || len(layer.Values) != 0 {
			t.Fatalf("skip-final forward host state layer %d retained host KV in device-only path", index)
		}
	}
	launches := driver.launches[launchStart:]
	if countLaunchName(launches, hipKernelNameMLXQ4ProjGreedy) != 0 {
		t.Fatalf("skip-final forward launched fused LM-head greedy projection")
	}
	if countLaunchName(launches, hipKernelNameGreedy) != 0 {
		t.Fatalf("skip-final forward launched host greedy sampling")
	}
	if countLaunchName(launches, hipKernelNameMLXQ4Proj) == 0 {
		t.Fatalf("skip-final forward did not run decoder q4 projections")
	}
}

func TestHIPGemma4Q4ForwardReturnsDeviceFinalHiddenForMTP_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 8, 1, 8)
	defer cleanup0()
	layer1, cleanup1 := hipGemma4Q4FixtureConfig(t, driver, 1, 4, 2, 8)
	defer cleanup1()
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0, layer1}}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	forward, state, err := hipRunGemma4Q4SingleTokenForwardWithStateInternal(context.Background(), driver, cfg, hipGemma4Q4DecodeState{}, hipGemma4Q4ForwardRequest{
		TokenID:                 1,
		Position:                0,
		Epsilon:                 1e-6,
		DeviceKVAttention:       true,
		DeviceKVMode:            rocmKVCacheModeKQ8VQ4,
		ReturnDeviceState:       true,
		ReturnDeviceFinalHidden: true,
		SkipFinalSample:         true,
		OmitDebugTensors:        true,
		AttentionWorkspace:      workspace,
	}, false)
	core.RequireNoError(t, err)
	defer forward.DeviceState.Close()

	if forward.DeviceFinalHidden == nil || forward.DeviceFinalHidden.Pointer() == 0 {
		t.Fatalf("device final hidden = %#v, want workspace-backed target hidden for MTP handoff", forward.DeviceFinalHidden)
	}
	core.AssertEqual(t, layer0.HiddenSize, forward.DeviceFinalHidden.Count())
	core.AssertEqual(t, uint64(layer0.HiddenSize*4), forward.DeviceFinalHidden.SizeBytes())
	core.AssertEqual(t, true, forward.DeviceFinalHiddenBorrowed)
	core.AssertEqual(t, workspace.FinalHiddenOutputViews[1].Pointer(), forward.DeviceFinalHidden.Pointer())
	core.AssertEqual(t, "returned", forward.Labels["gemma4_q4_device_final_hidden"])
	core.AssertEqual(t, "true", forward.Labels["gemma4_q4_device_final_hidden_borrowed"])
	core.AssertEqual(t, "skipped", forward.Labels["gemma4_q4_final_sample"])
	core.AssertEqual(t, 0, len(forward.FinalHidden))
	core.AssertEqual(t, 0, len(forward.Logits))
	core.AssertEqual(t, []int{1, 1}, forward.DeviceState.LayerTokenCounts())
	for index, layer := range state.Layers {
		if len(layer.Keys) != 0 || len(layer.Values) != 0 {
			t.Fatalf("state layer %d retained host KV in device-hidden MTP handoff path", index)
		}
	}
}

func TestHIPAttachedDrafterAssistantDraftStepInputBridge_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 8, 1, 8)
	defer cleanup0()
	layer1, cleanup1 := hipGemma4Q4FixtureConfig(t, driver, 1, 4, 2, 8)
	defer cleanup1()
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0, layer1}}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	forward, _, err := hipRunGemma4Q4SingleTokenForwardWithStateInternal(context.Background(), driver, cfg, hipGemma4Q4DecodeState{}, hipGemma4Q4ForwardRequest{
		TokenID:                 1,
		Position:                0,
		Epsilon:                 1e-6,
		DeviceKVAttention:       true,
		DeviceKVMode:            rocmKVCacheModeKQ8VQ4,
		ReturnDeviceState:       true,
		ReturnDeviceFinalHidden: true,
		SkipFinalSample:         true,
		OmitDebugTensors:        true,
		AttentionWorkspace:      workspace,
	}, false)
	core.RequireNoError(t, err)
	defer forward.DeviceState.Close()

	preProjectionPayload, err := hipUint16Payload(make([]uint16, layer0.HiddenSize*layer0.HiddenSize*2))
	core.RequireNoError(t, err)
	preProjection, err := hipUploadByteBuffer(driver, "rocm.hip.AttachedDrafterAssistantDraftStep", "assistant pre-projection fixture", preProjectionPayload, layer0.HiddenSize*layer0.HiddenSize*2)
	core.RequireNoError(t, err)
	defer preProjection.Close()

	plan := hipAttachedDrafterAssistantDraftStepInputPlan{
		Status:               attachedDrafterAssistantDraftStepInputLinked,
		HiddenSize:           layer0.HiddenSize,
		VocabSize:            layer0.VocabSize,
		TargetHiddenSize:     layer0.HiddenSize,
		CombinedInputSize:    layer0.HiddenSize * 2,
		ProjectionEncoding:   "bf16",
		TargetEmbedding:      layer0.Embedding,
		TargetEmbeddingScale: layer0.embeddingScale(),
		PreProjection: hipAttachedDrafterAssistantProjectionPlan{
			Encoding: "bf16",
			Rows:     layer0.HiddenSize,
			Cols:     layer0.HiddenSize * 2,
			BF16: hipBF16DeviceWeightConfig{
				WeightPointer: preProjection.Pointer(),
				WeightBytes:   preProjection.SizeBytes(),
				Rows:          layer0.HiddenSize,
				Cols:          layer0.HiddenSize * 2,
			},
		},
		KernelFamilies: []string{hipKernelNameEmbedLookup, hipKernelNameVectorScale, hipKernelNameProjection},
	}

	launchStart := len(driver.launches)
	result, err := hipRunAttachedDrafterAssistantDraftStepInputBridge(context.Background(), driver, hipAttachedDrafterAssistantDraftStepInputRequest{
		LastToken:         1,
		TargetHidden:      forward.DeviceFinalHidden,
		TargetDeviceState: forward.DeviceState,
		Plan:              plan,
		Workspace:         workspace,
	})
	core.RequireNoError(t, err)
	defer result.Close()

	if result.Hidden == nil || result.Hidden.Pointer() == 0 {
		t.Fatalf("assistant draft-step hidden = %#v, want device pre-projection output", result.Hidden)
	}
	core.AssertEqual(t, layer0.HiddenSize, result.Hidden.Count())
	core.AssertEqual(t, uint64(layer0.HiddenSize*4), result.Hidden.SizeBytes())
	core.AssertEqual(t, attachedDrafterAssistantDraftStepInputLinked, result.Labels["attached_drafter_assistant_draft_step_input_bridge"])
	core.AssertEqual(t, "device", result.Labels["attached_drafter_assistant_draft_step_target_hidden_source"])
	core.AssertEqual(t, "device_combined_token_hidden", result.Labels["attached_drafter_assistant_draft_step_input_buffer"])
	core.AssertEqual(t, "workspace", result.Labels["attached_drafter_assistant_draft_step_input_buffer_reuse"])
	core.AssertEqual(t, "bf16", result.Labels["attached_drafter_assistant_draft_step_pre_projection_encoding"])

	launches := driver.launches[launchStart:]
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameEmbedLookup))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameVectorScale))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameProjection))
	core.RequireNoError(t, result.Close())

	allocationCount := len(driver.allocations)
	second, err := hipRunAttachedDrafterAssistantDraftStepInputBridge(context.Background(), driver, hipAttachedDrafterAssistantDraftStepInputRequest{
		LastToken:         1,
		TargetHidden:      forward.DeviceFinalHidden,
		TargetDeviceState: forward.DeviceState,
		Plan:              plan,
		Workspace:         workspace,
	})
	core.RequireNoError(t, err)
	defer second.Close()
	core.AssertEqual(t, allocationCount, len(driver.allocations))
}

func TestHIPAttachedDrafterAssistantDraftStepInputPlanForModelUsesScaledTargetEmbedding_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 8, 1, 8)
	defer cleanup0()
	target := &hipLoadedModel{
		modelInfo: inference.ModelInfo{
			Architecture: "gemma4",
			NumLayers:    1,
			HiddenSize:   layer0.HiddenSize,
			VocabSize:    layer0.VocabSize,
		},
		q4Config: hipGemma4Q4ForwardConfig{
			Layers: []hipGemma4Q4Layer0Config{layer0},
		},
		q4Layers:   1,
		q4ConfigOK: true,
	}
	assistantPlan := hipAttachedDrafterAssistantVerifierPlan{
		Status:     attachedDrafterAssistantVerifierPlanTensorBound,
		HiddenSize: layer0.HiddenSize,
		PreProjection: hipAttachedDrafterAssistantProjectionPlan{
			Encoding: "bf16",
			Rows:     layer0.HiddenSize,
			Cols:     layer0.HiddenSize * 2,
			BF16: hipBF16DeviceWeightConfig{
				WeightPointer: 1,
				WeightBytes:   uint64(layer0.HiddenSize * layer0.HiddenSize * 2 * 2),
				Rows:          layer0.HiddenSize,
				Cols:          layer0.HiddenSize * 2,
			},
		},
	}

	plan := hipAttachedDrafterAssistantDraftStepInputPlanForModel(target, assistantPlan)

	core.AssertEqual(t, attachedDrafterAssistantDraftStepInputLinked, plan.Status)
	core.AssertEqual(t, layer0.embeddingScale(), plan.TargetEmbeddingScale)
}

func TestHIPAttachedDrafterAssistantDraftStepInputPlanAcceptsAsymmetricAssistantHidden_Good(t *testing.T) {
	plan := hipAttachedDrafterAssistantDraftStepInputPlan{
		Status:             attachedDrafterAssistantDraftStepInputLinked,
		HiddenSize:         256,
		VocabSize:          262144,
		TargetHiddenSize:   1536,
		CombinedInputSize:  3072,
		ProjectionEncoding: "bf16",
		TargetEmbedding: hipDeviceEmbeddingLookupConfig{
			EmbeddingPointer: 1,
			EmbeddingBytes:   uint64(262144 * 1536 * 2),
			TableEncoding:    hipEmbeddingTableEncodingBF16,
			VocabSize:        262144,
			HiddenSize:       1536,
		},
		TargetEmbeddingScale: 1,
		PreProjection: hipAttachedDrafterAssistantProjectionPlan{
			Encoding: "bf16",
			Rows:     256,
			Cols:     3072,
			BF16: hipBF16DeviceWeightConfig{
				WeightPointer: 2,
				WeightBytes:   uint64(256 * 3072 * 2),
				Rows:          256,
				Cols:          3072,
			},
		},
	}
	if reason := hipAttachedDrafterAssistantDraftStepInputPlanInvalidReason(plan); reason != "" {
		t.Fatalf("asymmetric assistant input plan invalid: %s", reason)
	}
	labels := plan.Labels()
	core.AssertEqual(t, "256", labels["attached_drafter_assistant_draft_step_hidden_size"])
	core.AssertEqual(t, "1536", labels["attached_drafter_assistant_draft_step_target_hidden_size"])
	core.AssertEqual(t, "1536", labels["attached_drafter_assistant_draft_step_target_embedding_hidden_size"])
	core.AssertEqual(t, "3072", labels["attached_drafter_assistant_draft_step_combined_input_size"])

	wrong := plan
	wrong.CombinedInputSize = wrong.TargetHiddenSize + wrong.HiddenSize
	if reason := hipAttachedDrafterAssistantDraftStepInputPlanInvalidReason(wrong); reason != "combined input size must equal target token embedding plus target hidden" {
		t.Fatalf("wrong asymmetric assistant input plan reason = %q", reason)
	}
}

func TestHIPAttachedDrafterAssistantDraftStepInputBridge_MLXAffineQAT_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 8, 1, 8)
	defer cleanup0()
	layer1, cleanup1 := hipGemma4Q4FixtureConfig(t, driver, 1, 4, 2, 8)
	defer cleanup1()
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0, layer1}}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	forward, _, err := hipRunGemma4Q4SingleTokenForwardWithStateInternal(context.Background(), driver, cfg, hipGemma4Q4DecodeState{}, hipGemma4Q4ForwardRequest{
		TokenID:                 1,
		Position:                0,
		Epsilon:                 1e-6,
		DeviceKVAttention:       true,
		DeviceKVMode:            rocmKVCacheModeKQ8VQ4,
		ReturnDeviceState:       true,
		ReturnDeviceFinalHidden: true,
		SkipFinalSample:         true,
		OmitDebugTensors:        true,
		AttentionWorkspace:      workspace,
	}, false)
	core.RequireNoError(t, err)
	defer forward.DeviceState.Close()

	combinedInput := layer0.HiddenSize * 2
	packedCols, err := hipMLXAffinePackedCols(combinedInput, 4)
	core.RequireNoError(t, err)
	weightsPayload, err := hipUint32Payload(make([]uint32, layer0.HiddenSize*packedCols))
	core.RequireNoError(t, err)
	weights, err := hipUploadByteBuffer(driver, "rocm.hip.AttachedDrafterAssistantDraftStep", "assistant q4 pre-projection weights", weightsPayload, layer0.HiddenSize*packedCols)
	core.RequireNoError(t, err)
	defer weights.Close()
	scaleCount := layer0.HiddenSize * (combinedInput / layer0.GroupSize)
	scalesPayload, err := hipUint16Payload(make([]uint16, scaleCount))
	core.RequireNoError(t, err)
	scales, err := hipUploadByteBuffer(driver, "rocm.hip.AttachedDrafterAssistantDraftStep", "assistant q4 pre-projection scales", scalesPayload, scaleCount)
	core.RequireNoError(t, err)
	defer scales.Close()
	biasesPayload, err := hipUint16Payload(make([]uint16, scaleCount))
	core.RequireNoError(t, err)
	biases, err := hipUploadByteBuffer(driver, "rocm.hip.AttachedDrafterAssistantDraftStep", "assistant q4 pre-projection biases", biasesPayload, scaleCount)
	core.RequireNoError(t, err)
	defer biases.Close()

	plan := hipAttachedDrafterAssistantDraftStepInputPlan{
		Status:               attachedDrafterAssistantDraftStepInputLinked,
		HiddenSize:           layer0.HiddenSize,
		VocabSize:            layer0.VocabSize,
		TargetHiddenSize:     layer0.HiddenSize,
		CombinedInputSize:    combinedInput,
		ProjectionEncoding:   "mlx_affine",
		TargetEmbedding:      layer0.Embedding,
		TargetEmbeddingScale: layer0.embeddingScale(),
		PreProjection: hipAttachedDrafterAssistantProjectionPlan{
			Encoding: "mlx_affine",
			Rows:     layer0.HiddenSize,
			Cols:     combinedInput,
			MLXAffine: hipMLXQ4DeviceWeightConfig{
				WeightPointer: weights.Pointer(),
				ScalePointer:  scales.Pointer(),
				BiasPointer:   biases.Pointer(),
				WeightBytes:   weights.SizeBytes(),
				ScaleBytes:    scales.SizeBytes(),
				BiasBytes:     biases.SizeBytes(),
				Rows:          layer0.HiddenSize,
				Cols:          combinedInput,
				GroupSize:     layer0.GroupSize,
				Bits:          4,
			},
		},
		KernelFamilies: []string{hipKernelNameEmbedLookup, hipKernelNameVectorScale, hipKernelNameMLXQ4Proj},
	}

	launchStart := len(driver.launches)
	result, err := hipRunAttachedDrafterAssistantDraftStepInputBridge(context.Background(), driver, hipAttachedDrafterAssistantDraftStepInputRequest{
		LastToken:         1,
		TargetHidden:      forward.DeviceFinalHidden,
		TargetDeviceState: forward.DeviceState,
		Plan:              plan,
		Workspace:         workspace,
	})
	core.RequireNoError(t, err)
	defer result.Close()

	core.AssertEqual(t, layer0.HiddenSize, result.Hidden.Count())
	core.AssertEqual(t, "mlx_affine", result.Labels["attached_drafter_assistant_draft_step_pre_projection_encoding"])
	core.AssertEqual(t, "workspace", result.Labels["attached_drafter_assistant_draft_step_input_buffer_reuse"])
	launches := driver.launches[launchStart:]
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameEmbedLookup))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameVectorScale))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameMLXQ4Proj))
}

func TestHIPAttachedDrafterAssistantLayerUsesTargetDeviceKV_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 8, 1, 8)
	defer cleanup0()
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0}}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	forward, _, err := hipRunGemma4Q4SingleTokenForwardWithStateInternal(context.Background(), driver, cfg, hipGemma4Q4DecodeState{}, hipGemma4Q4ForwardRequest{
		TokenID:                 1,
		Position:                0,
		Epsilon:                 1e-6,
		DeviceKVAttention:       true,
		DeviceKVMode:            rocmKVCacheModeKQ8VQ4,
		ReturnDeviceState:       true,
		ReturnDeviceFinalHidden: true,
		SkipFinalSample:         true,
		OmitDebugTensors:        true,
		AttentionWorkspace:      workspace,
	}, false)
	core.RequireNoError(t, err)
	defer forward.DeviceState.Close()

	targetLayer, targetLayerConfig, targetLayerIndex, err := hipAttachedDrafterAssistantTargetLayerFor(layer0.LayerType, cfg, forward.DeviceState)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, targetLayerIndex)

	plan := hipAttachedDrafterAssistantVerifierLayerPlan{
		Layer:              0,
		LayerType:          layer0.LayerType,
		HiddenSize:         layer0.HiddenSize,
		HeadDim:            layer0.HeadDim,
		QueryHeads:         layer0.QueryHeads,
		RoPEBase:           layer0.RoPEBase,
		RoPERotaryDim:      layer0.RoPERotaryDim,
		RoPEFrequencyScale: layer0.RoPEFrequencyScale,
		SlidingWindow:      layer0.SlidingWindow,
		LayerScalar:        layer0.effectiveLayerScalar(),
		InputNorm:          layer0.InputNorm,
		PostAttentionNorm:  layer0.PostAttentionNorm,
		PreFeedforward:     layer0.PreFeedForwardNorm,
		PostFeedforward:    layer0.PostFeedForwardNorm,
		QueryNorm:          layer0.QueryNorm,
		QueryProjection: hipAttachedDrafterAssistantProjectionPlan{
			Encoding:  "mlx_affine",
			Rows:      layer0.QueryProjection.Rows,
			Cols:      layer0.QueryProjection.Cols,
			MLXAffine: layer0.QueryProjection,
		},
		OutputProjection: hipAttachedDrafterAssistantProjectionPlan{
			Encoding:  "mlx_affine",
			Rows:      layer0.OutputProjection.Rows,
			Cols:      layer0.OutputProjection.Cols,
			MLXAffine: layer0.OutputProjection,
		},
		GateProjection: hipAttachedDrafterAssistantProjectionPlan{
			Encoding:  "mlx_affine",
			Rows:      layer0.GateProjection.Rows,
			Cols:      layer0.GateProjection.Cols,
			MLXAffine: layer0.GateProjection,
		},
		UpProjection: hipAttachedDrafterAssistantProjectionPlan{
			Encoding:  "mlx_affine",
			Rows:      layer0.UpProjection.Rows,
			Cols:      layer0.UpProjection.Cols,
			MLXAffine: layer0.UpProjection,
		},
		DownProjection: hipAttachedDrafterAssistantProjectionPlan{
			Encoding:  "mlx_affine",
			Rows:      layer0.DownProjection.Rows,
			Cols:      layer0.DownProjection.Cols,
			MLXAffine: layer0.DownProjection,
		},
	}

	launchStart := len(driver.launches)
	result, err := hipRunAttachedDrafterAssistantLayer(context.Background(), driver, hipAttachedDrafterAssistantLayerRequest{
		Hidden:            forward.DeviceFinalHidden,
		TargetLayer:       targetLayer,
		TargetLayerConfig: targetLayerConfig,
		Plan:              plan,
		Position:          0,
		Epsilon:           1e-6,
		Workspace:         workspace,
	})
	core.RequireNoError(t, err)
	defer result.Close()

	if result.Hidden == nil || result.Hidden.Pointer() == 0 {
		t.Fatalf("assistant layer hidden = %#v, want device output", result.Hidden)
	}
	core.AssertEqual(t, layer0.HiddenSize, result.Hidden.Count())
	core.AssertEqual(t, attachedDrafterAssistantLayerRuntimeLinked, result.Labels["attached_drafter_assistant_layer_runtime"])
	core.AssertEqual(t, "device", result.Labels["attached_drafter_assistant_layer_target_kv"])
	core.AssertEqual(t, "1", result.Labels["attached_drafter_assistant_layer_target_key_heads"])
	core.AssertEqual(t, "8", result.Labels["attached_drafter_assistant_layer_target_kv_width"])
	core.AssertEqual(t, "mlx_affine", result.Labels["attached_drafter_assistant_layer_projection_mode"])

	launches := driver.launches[launchStart:]
	core.AssertEqual(t, 1, countDeviceAttentionLaunches(launches))
	core.AssertEqual(t, 3, countLaunchName(launches, hipKernelNameMLXQ4Proj))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameMLXQ4GELUTanhMul))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameVectorAdd))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameVectorAddScaled))
}

func TestHIPAttachedDrafterAssistantLayerScalesWholeResidualOutput_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 8, 1, 8)
	defer cleanup0()
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0}}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	forward, _, err := hipRunGemma4Q4SingleTokenForwardWithStateInternal(context.Background(), driver, cfg, hipGemma4Q4DecodeState{}, hipGemma4Q4ForwardRequest{
		TokenID:            1,
		Position:           0,
		Epsilon:            1e-6,
		DeviceKVAttention:  true,
		DeviceKVMode:       rocmKVCacheModeKQ8VQ4,
		ReturnDeviceState:  true,
		SkipFinalSample:    true,
		OmitDebugTensors:   true,
		AttentionWorkspace: workspace,
	}, false)
	core.RequireNoError(t, err)
	defer forward.DeviceState.Close()

	targetLayer, targetLayerConfig, _, err := hipAttachedDrafterAssistantTargetLayerFor(layer0.LayerType, cfg, forward.DeviceState)
	core.RequireNoError(t, err)
	hiddenPayload, err := hipFloat32Payload([]float32{1, 2, 3, 4, 5, 6, 7, 8})
	core.RequireNoError(t, err)
	hidden, err := hipUploadByteBuffer(driver, "rocm.hip.AttachedDrafterAssistantLayer", "assistant scalar hidden", hiddenPayload, layer0.HiddenSize)
	core.RequireNoError(t, err)
	defer hidden.Close()

	plan := hipAttachedDrafterAssistantVerifierLayerPlan{
		Layer:              0,
		LayerType:          layer0.LayerType,
		HiddenSize:         layer0.HiddenSize,
		HeadDim:            layer0.HeadDim,
		QueryHeads:         layer0.QueryHeads,
		RoPEBase:           layer0.RoPEBase,
		RoPERotaryDim:      layer0.RoPERotaryDim,
		RoPEFrequencyScale: layer0.RoPEFrequencyScale,
		SlidingWindow:      layer0.SlidingWindow,
		LayerScalar:        0.5,
		InputNorm:          layer0.InputNorm,
		PostAttentionNorm:  layer0.PostAttentionNorm,
		PreFeedforward:     layer0.PreFeedForwardNorm,
		PostFeedforward:    layer0.PostFeedForwardNorm,
		QueryNorm:          layer0.QueryNorm,
		QueryProjection: hipAttachedDrafterAssistantProjectionPlan{
			Encoding:  "mlx_affine",
			Rows:      layer0.QueryProjection.Rows,
			Cols:      layer0.QueryProjection.Cols,
			MLXAffine: layer0.QueryProjection,
		},
		OutputProjection: hipAttachedDrafterAssistantProjectionPlan{
			Encoding:  "mlx_affine",
			Rows:      layer0.OutputProjection.Rows,
			Cols:      layer0.OutputProjection.Cols,
			MLXAffine: layer0.OutputProjection,
		},
		GateProjection: hipAttachedDrafterAssistantProjectionPlan{
			Encoding:  "mlx_affine",
			Rows:      layer0.GateProjection.Rows,
			Cols:      layer0.GateProjection.Cols,
			MLXAffine: layer0.GateProjection,
		},
		UpProjection: hipAttachedDrafterAssistantProjectionPlan{
			Encoding:  "mlx_affine",
			Rows:      layer0.UpProjection.Rows,
			Cols:      layer0.UpProjection.Cols,
			MLXAffine: layer0.UpProjection,
		},
		DownProjection: hipAttachedDrafterAssistantProjectionPlan{
			Encoding:  "mlx_affine",
			Rows:      layer0.DownProjection.Rows,
			Cols:      layer0.DownProjection.Cols,
			MLXAffine: layer0.DownProjection,
		},
	}

	result, err := hipRunAttachedDrafterAssistantLayer(context.Background(), driver, hipAttachedDrafterAssistantLayerRequest{
		Hidden:            hidden,
		TargetLayer:       targetLayer,
		TargetLayerConfig: targetLayerConfig,
		Plan:              plan,
		Position:          0,
		Epsilon:           1e-6,
		Workspace:         workspace,
	})
	core.RequireNoError(t, err)
	defer result.Close()
	got, err := hipReadFloat32DeviceOutput(result.Hidden, "rocm.hip.AttachedDrafterAssistantLayer", "assistant scalar output", layer0.HiddenSize)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4}, got, 0.0001)
}

func TestHIPAttachedDrafterAssistantTargetAttentionGeometrySupportsGQA_Good(t *testing.T) {
	target := hipGemma4Q4Layer0Config{
		HeadDim:    256,
		QueryHeads: 16,
		KeyHeads:   8,
	}
	plan := hipAttachedDrafterAssistantVerifierLayerPlan{
		HeadDim:    256,
		QueryHeads: 16,
	}

	keyHeads, kvWidth, err := hipAttachedDrafterAssistantTargetAttentionGeometry(target, plan)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 8, keyHeads)
	core.AssertEqual(t, 2048, kvWidth)
}

func TestHIPAttachedDrafterAssistantDraftStepHiddenRunsLayerChain_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layers := make([]hipGemma4Q4Layer0Config, 0, 4)
	cleanups := make([]func(), 0, 4)
	for index := 0; index < 4; index++ {
		layer, cleanup := hipGemma4Q4FixtureConfig(t, driver, index, 8, 1, 8)
		layers = append(layers, layer)
		cleanups = append(cleanups, cleanup)
	}
	defer func() {
		for index := len(cleanups) - 1; index >= 0; index-- {
			cleanups[index]()
		}
	}()
	cfg := hipGemma4Q4ForwardConfig{Layers: layers}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	forward, _, err := hipRunGemma4Q4SingleTokenForwardWithStateInternal(context.Background(), driver, cfg, hipGemma4Q4DecodeState{}, hipGemma4Q4ForwardRequest{
		TokenID:                 1,
		Position:                0,
		Epsilon:                 1e-6,
		DeviceKVAttention:       true,
		DeviceKVMode:            rocmKVCacheModeKQ8VQ4,
		ReturnDeviceState:       true,
		ReturnDeviceFinalHidden: true,
		SkipFinalSample:         true,
		OmitDebugTensors:        true,
		AttentionWorkspace:      workspace,
	}, false)
	core.RequireNoError(t, err)
	defer forward.DeviceState.Close()

	preProjection, closePre := hipAssistantBF16ProjectionPlanFixture(t, driver, "assistant chain pre_projection", layers[0].HiddenSize, layers[0].HiddenSize*2)
	defer closePre()
	postProjection, closePost := hipAssistantBF16ProjectionPlanFixture(t, driver, "assistant chain post_projection", layers[0].HiddenSize, layers[0].HiddenSize)
	defer closePost()
	assistantLayers := make([]hipAttachedDrafterAssistantVerifierLayerPlan, 0, len(layers))
	for _, layer := range layers {
		assistantLayers = append(assistantLayers, hipAssistantLayerPlanFromGemma4Q4Fixture(layer))
	}
	inputPlan := hipAttachedDrafterAssistantDraftStepInputPlan{
		Status:               attachedDrafterAssistantDraftStepInputLinked,
		HiddenSize:           layers[0].HiddenSize,
		VocabSize:            layers[0].VocabSize,
		TargetHiddenSize:     layers[0].HiddenSize,
		CombinedInputSize:    layers[0].HiddenSize * 2,
		ProjectionEncoding:   "bf16",
		TargetEmbedding:      layers[0].Embedding,
		TargetEmbeddingScale: layers[0].embeddingScale(),
		PreProjection:        preProjection,
		KernelFamilies:       []string{hipKernelNameEmbedLookup, hipKernelNameVectorScale, hipKernelNameProjection},
	}
	plan := hipAttachedDrafterAssistantVerifierPlan{
		Status:             attachedDrafterAssistantVerifierPlanTensorBound,
		HiddenSize:         layers[0].HiddenSize,
		VocabSize:          layers[0].VocabSize,
		LayerCount:         len(assistantLayers),
		ProjectionEncoding: "mlx_affine",
		Norm:               layers[0].FinalNorm,
		PreProjection:      preProjection,
		PostProjection:     postProjection,
		Layers:             assistantLayers,
	}

	launchStart := len(driver.launches)
	result, err := hipRunAttachedDrafterAssistantDraftStepHidden(context.Background(), driver, hipAttachedDrafterAssistantDraftStepHiddenRequest{
		LastToken:         1,
		TargetHidden:      forward.DeviceFinalHidden,
		TargetForward:     cfg,
		TargetDeviceState: forward.DeviceState,
		Plan:              plan,
		InputPlan:         inputPlan,
		Position:          0,
		Epsilon:           1e-6,
		Workspace:         workspace,
	})
	core.RequireNoError(t, err)
	defer result.Close()

	if result.Normed == nil || result.Normed.Pointer() == 0 {
		t.Fatalf("assistant normed = %#v, want device final assistant norm", result.Normed)
	}
	if result.Hidden == nil || result.Hidden.Pointer() == 0 {
		t.Fatalf("assistant hidden = %#v, want device post-projection target hidden", result.Hidden)
	}
	core.AssertEqual(t, layers[0].HiddenSize, result.Normed.Count())
	core.AssertEqual(t, layers[0].HiddenSize, result.Hidden.Count())
	core.AssertEqual(t, attachedDrafterAssistantLayerRuntimeLinked, result.Labels["attached_drafter_assistant_draft_step_hidden_runtime"])
	core.AssertEqual(t, "4", result.Labels["attached_drafter_assistant_draft_step_hidden_layers_executed"])
	core.AssertEqual(t, "bf16", result.Labels["attached_drafter_assistant_draft_step_post_projection_encoding"])
	core.AssertEqual(t, "assistant_post_projection", result.Labels["attached_drafter_assistant_draft_step_hidden_source"])

	launches := driver.launches[launchStart:]
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameEmbedLookup))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameVectorScale))
	core.AssertEqual(t, 2, countLaunchName(launches, hipKernelNameProjection))
	core.AssertEqual(t, 4, countDeviceAttentionLaunches(launches))
	core.AssertEqual(t, 12, countLaunchName(launches, hipKernelNameMLXQ4Proj))
	core.AssertEqual(t, 4, countLaunchName(launches, hipKernelNameMLXQ4GELUTanhMul))
	core.AssertEqual(t, 4, countLaunchName(launches, hipKernelNameVectorAdd))
	core.AssertEqual(t, 4, countLaunchName(launches, hipKernelNameVectorAddScaled))
}

func TestHIPAttachedDrafterAssistantDraftStepProposalBF16DenseLogits_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 8, 1, 8)
	defer cleanup0()
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0}}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	forward, _, err := hipRunGemma4Q4SingleTokenForwardWithStateInternal(context.Background(), driver, cfg, hipGemma4Q4DecodeState{}, hipGemma4Q4ForwardRequest{
		TokenID:                 1,
		Position:                0,
		Epsilon:                 1e-6,
		DeviceKVAttention:       true,
		DeviceKVMode:            rocmKVCacheModeKQ8VQ4,
		ReturnDeviceState:       true,
		ReturnDeviceFinalHidden: true,
		SkipFinalSample:         true,
		OmitDebugTensors:        true,
		AttentionWorkspace:      workspace,
	}, false)
	core.RequireNoError(t, err)
	defer forward.DeviceState.Close()

	preProjection, closePre := hipAssistantBF16ProjectionPlanFixture(t, driver, "assistant proposal bf16 pre_projection", layer0.HiddenSize, layer0.HiddenSize*2)
	defer closePre()
	postProjection, closePost := hipAssistantBF16ProjectionPlanFixture(t, driver, "assistant proposal bf16 post_projection", layer0.HiddenSize, layer0.HiddenSize)
	defer closePost()
	embeddingPayload, err := hipUint16Payload(make([]uint16, layer0.VocabSize*layer0.HiddenSize))
	core.RequireNoError(t, err)
	embedding, err := hipUploadByteBuffer(driver, "rocm.hip.AttachedDrafterAssistantDraftStepProposal", "assistant proposal bf16 embedding", embeddingPayload, layer0.VocabSize*layer0.HiddenSize)
	core.RequireNoError(t, err)
	defer embedding.Close()

	inputPlan := hipAttachedDrafterAssistantDraftStepInputPlan{
		Status:               attachedDrafterAssistantDraftStepInputLinked,
		HiddenSize:           layer0.HiddenSize,
		VocabSize:            layer0.VocabSize,
		TargetHiddenSize:     layer0.HiddenSize,
		CombinedInputSize:    layer0.HiddenSize * 2,
		ProjectionEncoding:   "bf16",
		TargetEmbedding:      layer0.Embedding,
		TargetEmbeddingScale: layer0.embeddingScale(),
		PreProjection:        preProjection,
		KernelFamilies:       []string{hipKernelNameEmbedLookup, hipKernelNameVectorScale, hipKernelNameProjection},
	}
	plan := hipAttachedDrafterAssistantVerifierPlan{
		Status:             attachedDrafterAssistantVerifierPlanTensorBound,
		HiddenSize:         layer0.HiddenSize,
		VocabSize:          layer0.VocabSize,
		LayerCount:         1,
		ProjectionEncoding: "bf16",
		Embedding: hipDeviceEmbeddingLookupConfig{
			EmbeddingPointer: embedding.Pointer(),
			EmbeddingBytes:   embedding.SizeBytes(),
			TableEncoding:    hipEmbeddingTableEncodingBF16,
			VocabSize:        layer0.VocabSize,
			HiddenSize:       layer0.HiddenSize,
		},
		Norm:           layer0.FinalNorm,
		PreProjection:  preProjection,
		PostProjection: postProjection,
		Layers:         []hipAttachedDrafterAssistantVerifierLayerPlan{hipAssistantLayerPlanFromGemma4Q4Fixture(layer0)},
	}

	launchStart := len(driver.launches)
	result, err := hipRunAttachedDrafterAssistantDraftStepProposal(context.Background(), driver, hipAttachedDrafterAssistantDraftStepProposalRequest{
		LastToken:         1,
		TargetHidden:      forward.DeviceFinalHidden,
		TargetForward:     cfg,
		TargetDeviceState: forward.DeviceState,
		Plan:              plan,
		InputPlan:         inputPlan,
		Position:          0,
		Epsilon:           1e-6,
		Workspace:         workspace,
	})
	core.RequireNoError(t, err)
	defer result.Close()

	core.AssertEqual(t, 0, result.Token.TokenID)
	assertFloat32Near(t, 0, result.Token.Score)
	if result.Logits == nil || result.Logits.Pointer() == 0 {
		t.Fatalf("assistant proposal logits = %#v, want retained dense logits for BF16 path", result.Logits)
	}
	if result.Hidden == nil || result.Hidden.Pointer() == 0 {
		t.Fatalf("assistant proposal hidden = %#v, want next target hidden", result.Hidden)
	}
	core.AssertEqual(t, layer0.VocabSize, result.Logits.Count())
	core.AssertEqual(t, layer0.HiddenSize, result.Hidden.Count())
	core.AssertEqual(t, attachedDrafterAssistantDraftStepProposalLinked, result.Labels["attached_drafter_assistant_draft_step_proposal_runtime"])
	core.AssertEqual(t, "bf16", result.Labels["attached_drafter_assistant_draft_step_proposal_embedding_encoding"])
	core.AssertEqual(t, "dense_retained", result.Labels["attached_drafter_assistant_draft_step_logits"])
	core.AssertEqual(t, "dense_logits_greedy", result.Labels["attached_drafter_assistant_draft_step_token_source"])
	core.AssertEqual(t, "0", result.Labels["attached_drafter_assistant_draft_step_token_id"])

	launches := driver.launches[launchStart:]
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameEmbedLookup))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameVectorScale))
	core.AssertEqual(t, 3, countLaunchName(launches, hipKernelNameProjection))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameGreedy))
	core.AssertEqual(t, 1, countDeviceAttentionLaunches(launches))
	core.AssertEqual(t, 3, countLaunchName(launches, hipKernelNameMLXQ4Proj))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameMLXQ4GELUTanhMul))
}

func TestHIPAttachedDrafterAssistantDraftStepProposalMLXAffineQATGreedy_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 8, 1, 8)
	defer cleanup0()
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0}}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	forward, _, err := hipRunGemma4Q4SingleTokenForwardWithStateInternal(context.Background(), driver, cfg, hipGemma4Q4DecodeState{}, hipGemma4Q4ForwardRequest{
		TokenID:                 1,
		Position:                0,
		Epsilon:                 1e-6,
		DeviceKVAttention:       true,
		DeviceKVMode:            rocmKVCacheModeKQ8VQ4,
		ReturnDeviceState:       true,
		ReturnDeviceFinalHidden: true,
		SkipFinalSample:         true,
		OmitDebugTensors:        true,
		AttentionWorkspace:      workspace,
	}, false)
	core.RequireNoError(t, err)
	defer forward.DeviceState.Close()

	preProjection, closePre := hipAssistantBF16ProjectionPlanFixture(t, driver, "assistant proposal q4 pre_projection", layer0.HiddenSize, layer0.HiddenSize*2)
	defer closePre()
	postProjection, closePost := hipAssistantBF16ProjectionPlanFixture(t, driver, "assistant proposal q4 post_projection", layer0.HiddenSize, layer0.HiddenSize)
	defer closePost()
	inputPlan := hipAttachedDrafterAssistantDraftStepInputPlan{
		Status:               attachedDrafterAssistantDraftStepInputLinked,
		HiddenSize:           layer0.HiddenSize,
		VocabSize:            layer0.VocabSize,
		TargetHiddenSize:     layer0.HiddenSize,
		CombinedInputSize:    layer0.HiddenSize * 2,
		ProjectionEncoding:   "bf16",
		TargetEmbedding:      layer0.Embedding,
		TargetEmbeddingScale: layer0.embeddingScale(),
		PreProjection:        preProjection,
		KernelFamilies:       []string{hipKernelNameEmbedLookup, hipKernelNameVectorScale, hipKernelNameProjection},
	}
	plan := hipAttachedDrafterAssistantVerifierPlan{
		Status:             attachedDrafterAssistantVerifierPlanTensorBound,
		HiddenSize:         layer0.HiddenSize,
		VocabSize:          layer0.VocabSize,
		LayerCount:         1,
		ProjectionEncoding: "mlx_affine",
		Embedding:          layer0.Embedding,
		Norm:               layer0.FinalNorm,
		PreProjection:      preProjection,
		PostProjection:     postProjection,
		Layers:             []hipAttachedDrafterAssistantVerifierLayerPlan{hipAssistantLayerPlanFromGemma4Q4Fixture(layer0)},
	}

	launchStart := len(driver.launches)
	copyStart := len(driver.copies)
	result, err := hipRunAttachedDrafterAssistantDraftStepProposal(context.Background(), driver, hipAttachedDrafterAssistantDraftStepProposalRequest{
		LastToken:         1,
		TargetHidden:      forward.DeviceFinalHidden,
		TargetForward:     cfg,
		TargetDeviceState: forward.DeviceState,
		Plan:              plan,
		InputPlan:         inputPlan,
		Position:          0,
		Epsilon:           1e-6,
		Softcap:           30,
		Workspace:         workspace,
	})
	core.RequireNoError(t, err)
	defer result.Close()

	core.AssertEqual(t, 0, result.Token.TokenID)
	if result.Logits != nil {
		t.Fatalf("assistant proposal q4 logits = %#v, want fused projection-greedy without dense logits", result.Logits)
	}
	if result.Hidden == nil || result.Hidden.Pointer() == 0 {
		t.Fatalf("assistant proposal q4 hidden = %#v, want next target hidden", result.Hidden)
	}
	core.AssertEqual(t, layer0.HiddenSize, result.Hidden.Count())
	core.AssertEqual(t, attachedDrafterAssistantDraftStepProposalLinked, result.Labels["attached_drafter_assistant_draft_step_proposal_runtime"])
	core.AssertEqual(t, "mlx_affine", result.Labels["attached_drafter_assistant_draft_step_proposal_embedding_encoding"])
	core.AssertEqual(t, "30", result.Labels["attached_drafter_assistant_draft_step_proposal_softcap"])
	core.AssertEqual(t, "not_retained", result.Labels["attached_drafter_assistant_draft_step_logits"])
	core.AssertEqual(t, "projection_greedy", result.Labels["attached_drafter_assistant_draft_step_token_source"])

	launches := driver.launches[launchStart:]
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameEmbedLookup))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameVectorScale))
	core.AssertEqual(t, 2, countLaunchName(launches, hipKernelNameProjection))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameGreedy))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameMLXQ4ProjGreedy))
	core.AssertEqual(t, 3, countLaunchName(launches, hipKernelNameMLXQ4Proj))
	core.AssertEqual(t, 1, countDeviceAttentionLaunches(launches))
	if len(driver.copies) <= copyStart {
		t.Fatalf("assistant proposal q4 did not read the device greedy token")
	}
	core.AssertEqual(t, uint64(4), driver.copies[len(driver.copies)-1])
}

func TestHIPAttachedDrafterAssistantDraftBlockUsesSeenTokenPosition_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 8, 1, 8)
	defer cleanup0()
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0}}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	forward, _, err := hipRunGemma4Q4SingleTokenForwardWithStateInternal(context.Background(), driver, cfg, hipGemma4Q4DecodeState{}, hipGemma4Q4ForwardRequest{
		TokenID:                 1,
		Position:                0,
		Epsilon:                 1e-6,
		DeviceKVAttention:       true,
		DeviceKVMode:            rocmKVCacheModeKQ8VQ4,
		ReturnDeviceState:       true,
		ReturnDeviceFinalHidden: true,
		SkipFinalSample:         true,
		OmitDebugTensors:        true,
		AttentionWorkspace:      workspace,
	}, false)
	core.RequireNoError(t, err)
	defer forward.DeviceState.Close()

	preProjection, closePre := hipAssistantBF16ProjectionPlanFixture(t, driver, "assistant block q4 pre_projection", layer0.HiddenSize, layer0.HiddenSize*2)
	defer closePre()
	postProjection, closePost := hipAssistantBF16ProjectionPlanFixture(t, driver, "assistant block q4 post_projection", layer0.HiddenSize, layer0.HiddenSize)
	defer closePost()
	inputPlan := hipAttachedDrafterAssistantDraftStepInputPlan{
		Status:               attachedDrafterAssistantDraftStepInputLinked,
		HiddenSize:           layer0.HiddenSize,
		VocabSize:            layer0.VocabSize,
		TargetHiddenSize:     layer0.HiddenSize,
		CombinedInputSize:    layer0.HiddenSize * 2,
		ProjectionEncoding:   "bf16",
		TargetEmbedding:      layer0.Embedding,
		TargetEmbeddingScale: layer0.embeddingScale(),
		PreProjection:        preProjection,
		KernelFamilies:       []string{hipKernelNameEmbedLookup, hipKernelNameVectorScale, hipKernelNameProjection},
	}
	plan := hipAttachedDrafterAssistantVerifierPlan{
		Status:             attachedDrafterAssistantVerifierPlanTensorBound,
		HiddenSize:         layer0.HiddenSize,
		VocabSize:          layer0.VocabSize,
		LayerCount:         1,
		ProjectionEncoding: "mlx_affine",
		Embedding:          layer0.Embedding,
		Norm:               layer0.FinalNorm,
		PreProjection:      preProjection,
		PostProjection:     postProjection,
		Layers:             []hipAttachedDrafterAssistantVerifierLayerPlan{hipAssistantLayerPlanFromGemma4Q4Fixture(layer0)},
	}

	launchStart := len(driver.launches)
	copyStart := len(driver.copies)
	result, err := hipRunAttachedDrafterAssistantDraftBlock(context.Background(), driver, hipAttachedDrafterAssistantDraftBlockRequest{
		LastToken:         1,
		TargetHidden:      forward.DeviceFinalHidden,
		TargetForward:     cfg,
		TargetDeviceState: forward.DeviceState,
		Plan:              plan,
		InputPlan:         inputPlan,
		Position:          3,
		Epsilon:           1e-6,
		Softcap:           30,
		MaxDraftTokens:    2,
		Workspace:         workspace,
	})
	core.RequireNoError(t, err)
	defer result.Close()

	core.AssertEqual(t, []int32{0, 0}, result.Tokens)
	if result.Hidden == nil || result.Hidden.Pointer() == 0 {
		t.Fatalf("assistant draft block hidden = %#v, want final draft hidden", result.Hidden)
	}
	core.AssertEqual(t, layer0.HiddenSize, result.Hidden.Count())

	launches := driver.launches[launchStart:]
	if len(launches) == 0 {
		t.Fatalf("assistant draft block launched no kernels")
	}
	core.AssertEqual(t, hipKernelNameRMSNorm, launches[0].Name)
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameEmbedLookup))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameEmbedLookupGreedyToken))
	core.AssertEqual(t, 2, countLaunchName(launches, hipKernelNameVectorScale))
	core.AssertEqual(t, 4, countLaunchName(launches, hipKernelNameProjection))
	core.AssertEqual(t, 2, countLaunchName(launches, hipKernelNameMLXQ4ProjGreedy))
	core.AssertEqual(t, 2, countDeviceAttentionLaunches(launches))
	core.AssertEqual(t, []int{2, 3}, rmsNormRoPEHeadPositions(launches))
	core.AssertEqual(t, []uint64{uint64(2 * hipMLXQ4ProjectionBestBytes)}, driver.copies[copyStart:])
}

func TestHIPAttachedDrafterTargetVerifyBlockLeadAcceptTruncatesBatchForward_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 8, 1, 8)
	defer cleanup0()
	hipGemma4Q4InstallNonzeroEmbeddingFixture(t, driver, &layer0, "attached verify partial")
	layer0.PerLayerInput = hipGemma4Q4PerLayerInputConfig{}
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0}}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	initialForward, _, err := hipRunGemma4Q4SingleTokenForwardWithStateInternal(context.Background(), driver, cfg, hipGemma4Q4DecodeState{}, hipGemma4Q4ForwardRequest{
		TokenID:            1,
		Position:           0,
		Epsilon:            1e-6,
		DeviceKVAttention:  true,
		DeviceKVMode:       rocmKVCacheModeKQ8VQ4,
		ReturnDeviceState:  true,
		SkipFinalSample:    true,
		OmitDebugTensors:   true,
		AttentionWorkspace: workspace,
	}, false)
	core.RequireNoError(t, err)
	initialState := initialForward.DeviceState
	initialForward.DeviceState = nil
	defer initialState.Close()

	workspace.EnsureProjectionGreedyBestCapacity(2)
	greedyBuffer, err := workspace.BorrowProjectionGreedyBest(driver)
	core.RequireNoError(t, err)

	launchStart := len(driver.launches)
	result, err := hipRunAttachedDrafterTargetVerifyBlock(context.Background(), driver, hipAttachedDrafterTargetVerifyBlockRequest{
		TargetForward:     cfg,
		DeviceKVMode:      rocmKVCacheModeKQ8VQ4,
		EngineConfig:      defaultHIPGemma4Q4EngineConfig(),
		TargetDeviceState: initialState,
		CurrentGreedy:     hipGreedySampleResult{TokenID: 0},
		DraftTokens:       []int32{0, 1},
		Position:          1,
		Epsilon:           1e-6,
		GreedyBuffer:      greedyBuffer,
		Workspace:         workspace,
	})
	core.RequireNoError(t, err)
	defer result.Close()
	launches := driver.launches[launchStart:]

	core.AssertEqual(t, 1, result.AcceptedCount)
	core.AssertEqual(t, 1, result.RejectedCount)
	core.AssertEqual(t, false, result.AllAccepted)
	core.AssertEqual(t, 1, result.TargetCalls)
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameMLXQ4Proj))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameMLXQ4ProjGreedyBatch))
	if batchProjectionLaunches := countLaunchName(launches, hipKernelNameMLXQ4ProjBatch); batchProjectionLaunches == 0 {
		t.Fatalf("partial verify launched projection_batch=%d, want batched projection", batchProjectionLaunches)
	}
	if batchAttentionLaunches := countLaunchName(launches, hipKernelNameAttentionHeadsBatchCausal); batchAttentionLaunches == 0 {
		t.Fatalf("partial verify launched attention_batch=%d, want batched attention", batchAttentionLaunches)
	}
	core.AssertEqual(t, 0, result.Replacement.TokenID)
	core.AssertEqual(t, false, result.PriorDeviceStateFinalized)
	if result.DeviceHidden == nil || result.DeviceHidden.Pointer() == 0 {
		t.Fatalf("partial verify hidden = %#v, want accepted-prefix hidden", result.DeviceHidden)
	}
	if result.DeviceState == nil || result.DeviceState.closed {
		t.Fatalf("partial verify state = %#v, want live accepted-prefix device state", result.DeviceState)
	}
	core.AssertEqual(t, []int{2}, result.DeviceState.LayerTokenCounts())
}

func TestHIPAttachedDrafterTargetVerifyBlockSingleAcceptUsesBatchOneForward_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 8, 1, 8)
	defer cleanup0()
	hipGemma4Q4InstallNonzeroEmbeddingFixture(t, driver, &layer0, "attached verify single accept")
	layer0.PerLayerInput = hipGemma4Q4PerLayerInputConfig{}
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0}}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	initialForward, _, err := hipRunGemma4Q4SingleTokenForwardWithStateInternal(context.Background(), driver, cfg, hipGemma4Q4DecodeState{}, hipGemma4Q4ForwardRequest{
		TokenID:            1,
		Position:           0,
		Epsilon:            1e-6,
		DeviceKVAttention:  true,
		DeviceKVMode:       rocmKVCacheModeKQ8VQ4,
		ReturnDeviceState:  true,
		SkipFinalSample:    true,
		OmitDebugTensors:   true,
		AttentionWorkspace: workspace,
	}, false)
	core.RequireNoError(t, err)
	initialState := initialForward.DeviceState
	initialForward.DeviceState = nil
	defer initialState.Close()

	workspace.EnsureProjectionGreedyBestCapacity(1)
	greedyBuffer, err := workspace.BorrowProjectionGreedyBest(driver)
	core.RequireNoError(t, err)

	launchStart := len(driver.launches)
	result, err := hipRunAttachedDrafterTargetVerifyBlock(context.Background(), driver, hipAttachedDrafterTargetVerifyBlockRequest{
		TargetForward:     cfg,
		DeviceKVMode:      rocmKVCacheModeKQ8VQ4,
		EngineConfig:      defaultHIPGemma4Q4EngineConfig(),
		TargetDeviceState: initialState,
		CurrentGreedy:     hipGreedySampleResult{TokenID: 0},
		DraftTokens:       []int32{0},
		Position:          1,
		Epsilon:           1e-6,
		GreedyBuffer:      greedyBuffer,
		Workspace:         workspace,
	})
	core.RequireNoError(t, err)
	defer result.Close()
	launches := driver.launches[launchStart:]

	core.AssertEqual(t, 1, result.AcceptedCount)
	core.AssertEqual(t, 0, result.RejectedCount)
	core.AssertEqual(t, true, result.AllAccepted)
	core.AssertEqual(t, 1, result.TargetCalls)
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameMLXQ4Proj))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameMLXQ4ProjGreedyBatch))
	if batchProjectionLaunches := countLaunchName(launches, hipKernelNameMLXQ4ProjBatch); batchProjectionLaunches == 0 {
		t.Fatalf("single accept verify launched projection_batch=%d, want batched projection", batchProjectionLaunches)
	}
	if batchAttentionLaunches := countLaunchName(launches, hipKernelNameAttentionHeadsBatchCausal); batchAttentionLaunches == 0 {
		t.Fatalf("single accept verify launched attention_batch=%d, want batched attention", batchAttentionLaunches)
	}
	core.AssertEqual(t, true, result.PriorDeviceStateFinalized)
	if result.DeviceHidden == nil || result.DeviceHidden.Pointer() == 0 {
		t.Fatalf("single accept verify hidden = %#v, want compact accepted-token hidden", result.DeviceHidden)
	}
	if result.DeviceState == nil || result.DeviceState.closed {
		t.Fatalf("single accept verify state = %#v, want live compact device state", result.DeviceState)
	}
	core.AssertEqual(t, []int{2}, result.DeviceState.LayerTokenCounts())
}

func TestHIPAttachedDrafterTargetVerifyBlockAcceptedPrefixBatchesSuffix_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 8, 1, 8)
	defer cleanup0()
	hipGemma4Q4InstallNonzeroEmbeddingFixture(t, driver, &layer0, "attached verify prefix batch")
	layer0.PerLayerInput = hipGemma4Q4PerLayerInputConfig{}
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0}}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	initialForward, _, err := hipRunGemma4Q4SingleTokenForwardWithStateInternal(context.Background(), driver, cfg, hipGemma4Q4DecodeState{}, hipGemma4Q4ForwardRequest{
		TokenID:            1,
		Position:           0,
		Epsilon:            1e-6,
		DeviceKVAttention:  true,
		DeviceKVMode:       rocmKVCacheModeKQ8VQ4,
		ReturnDeviceState:  true,
		SkipFinalSample:    true,
		OmitDebugTensors:   true,
		AttentionWorkspace: workspace,
	}, false)
	core.RequireNoError(t, err)
	initialState := initialForward.DeviceState
	initialForward.DeviceState = nil
	defer initialState.Close()

	workspace.EnsureProjectionGreedyBestCapacity(3)
	greedyBuffer, err := workspace.BorrowProjectionGreedyBest(driver)
	core.RequireNoError(t, err)

	launchStart := len(driver.launches)
	result, err := hipRunAttachedDrafterTargetVerifyBlock(context.Background(), driver, hipAttachedDrafterTargetVerifyBlockRequest{
		TargetForward:     cfg,
		DeviceKVMode:      rocmKVCacheModeKQ8VQ4,
		EngineConfig:      defaultHIPGemma4Q4EngineConfig(),
		TargetDeviceState: initialState,
		CurrentGreedy:     hipGreedySampleResult{TokenID: 0},
		DraftTokens:       []int32{0, 0, 1},
		Position:          1,
		Epsilon:           1e-6,
		GreedyBuffer:      greedyBuffer,
		Workspace:         workspace,
	})
	core.RequireNoError(t, err)
	defer result.Close()
	launches := driver.launches[launchStart:]

	core.AssertEqual(t, 2, result.AcceptedCount)
	core.AssertEqual(t, 1, result.RejectedCount)
	core.AssertEqual(t, false, result.AllAccepted)
	core.AssertEqual(t, 1, result.TargetCalls)
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameMLXQ4ProjGreedy))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameMLXQ4ProjGreedyBatch))
	core.AssertEqual(t, 0, result.Replacement.TokenID)
	if result.DeviceHidden == nil || result.DeviceHidden.Pointer() == 0 {
		t.Fatalf("prefix-batch verify hidden = %#v, want accepted-prefix hidden", result.DeviceHidden)
	}
	if result.DeviceState == nil || result.DeviceState.closed {
		t.Fatalf("prefix-batch verify state = %#v, want live accepted-prefix device state", result.DeviceState)
	}
	core.AssertEqual(t, []int{3}, result.DeviceState.LayerTokenCounts())
}

func TestHIPAttachedDrafterTargetVerifyBlockFirstMismatchSkipsGreedyBatch_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 8, 1, 8)
	defer cleanup0()
	hipGemma4Q4InstallNonzeroEmbeddingFixture(t, driver, &layer0, "attached verify first mismatch")
	layer0.PerLayerInput = hipGemma4Q4PerLayerInputConfig{}
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0}}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	initialForward, _, err := hipRunGemma4Q4SingleTokenForwardWithStateInternal(context.Background(), driver, cfg, hipGemma4Q4DecodeState{}, hipGemma4Q4ForwardRequest{
		TokenID:            1,
		Position:           0,
		Epsilon:            1e-6,
		DeviceKVAttention:  true,
		DeviceKVMode:       rocmKVCacheModeKQ8VQ4,
		ReturnDeviceState:  true,
		SkipFinalSample:    true,
		OmitDebugTensors:   true,
		AttentionWorkspace: workspace,
	}, false)
	core.RequireNoError(t, err)
	initialState := initialForward.DeviceState
	initialForward.DeviceState = nil
	defer initialState.Close()

	launchStart := len(driver.launches)
	result, err := hipRunAttachedDrafterTargetVerifyBlock(context.Background(), driver, hipAttachedDrafterTargetVerifyBlockRequest{
		TargetForward:     cfg,
		DeviceKVMode:      rocmKVCacheModeKQ8VQ4,
		EngineConfig:      defaultHIPGemma4Q4EngineConfig(),
		TargetDeviceState: initialState,
		CurrentGreedy:     hipGreedySampleResult{TokenID: 1},
		DraftTokens:       []int32{0, 1},
		Position:          1,
		Epsilon:           1e-6,
		Workspace:         workspace,
	})
	core.RequireNoError(t, err)
	defer result.Close()
	launches := driver.launches[launchStart:]

	core.AssertEqual(t, 0, result.AcceptedCount)
	core.AssertEqual(t, 2, result.RejectedCount)
	core.AssertEqual(t, false, result.AllAccepted)
	core.AssertEqual(t, 0, result.TargetCalls)
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameMLXQ4ProjGreedy))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameMLXQ4ProjGreedyBatch))
	core.AssertEqual(t, 1, result.Replacement.TokenID)
	if result.DeviceState != nil {
		t.Fatalf("first-mismatch verify state = %#v, want no accepted-prefix state", result.DeviceState)
	}
}

func TestHIPAttachedDrafterGenerateFromStateRetainsGeneratedToken_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 8, 1, 8)
	defer cleanup0()
	hipGemma4Q4InstallNonzeroEmbeddingFixture(t, driver, &layer0, "attached retained generate")
	layer0.PerLayerInput = hipGemma4Q4PerLayerInputConfig{}
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0}}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	initialForward, _, err := hipRunGemma4Q4SingleTokenForwardWithStateInternal(context.Background(), driver, cfg, hipGemma4Q4DecodeState{}, hipGemma4Q4ForwardRequest{
		TokenID:            1,
		Position:           0,
		Epsilon:            1e-6,
		DeviceKVAttention:  true,
		DeviceKVMode:       rocmKVCacheModeKQ8VQ4,
		ReturnDeviceState:  true,
		SkipFinalSample:    true,
		OmitDebugTensors:   true,
		AttentionWorkspace: workspace,
	}, false)
	core.RequireNoError(t, err)
	initialState := initialForward.DeviceState
	initialForward.DeviceState = nil
	session := newStateSessionWithRuntime(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil, initialState)
	defer session.Close()

	preProjection, closePre := hipAssistantBF16ProjectionPlanFixture(t, driver, "assistant retained generate pre_projection", layer0.HiddenSize, layer0.HiddenSize*2)
	defer closePre()
	postProjection, closePost := hipAssistantBF16ProjectionPlanFixture(t, driver, "assistant retained generate post_projection", layer0.HiddenSize, layer0.HiddenSize)
	defer closePost()
	embeddingPayload, err := hipUint16Payload(make([]uint16, layer0.VocabSize*layer0.HiddenSize))
	core.RequireNoError(t, err)
	embedding, err := hipUploadByteBuffer(driver, "rocm.hip.AttachedDrafterGenerate", "assistant retained generate embedding", embeddingPayload, layer0.VocabSize*layer0.HiddenSize)
	core.RequireNoError(t, err)
	defer embedding.Close()

	target := &hipLoadedModel{
		driver: driver,
		modelInfo: inference.ModelInfo{
			Architecture: "gemma4_text",
			NumLayers:    1,
			HiddenSize:   layer0.HiddenSize,
			VocabSize:    layer0.VocabSize,
			QuantBits:    4,
			QuantGroup:   64,
		},
		q4Config:   cfg,
		q4Layers:   1,
		q4ConfigOK: true,
	}
	inputPlan := hipAttachedDrafterAssistantDraftStepInputPlan{
		Status:               attachedDrafterAssistantDraftStepInputLinked,
		HiddenSize:           layer0.HiddenSize,
		VocabSize:            layer0.VocabSize,
		TargetHiddenSize:     layer0.HiddenSize,
		CombinedInputSize:    layer0.HiddenSize * 2,
		ProjectionEncoding:   "bf16",
		TargetEmbedding:      layer0.Embedding,
		TargetEmbeddingScale: layer0.embeddingScale(),
		PreProjection:        preProjection,
		KernelFamilies:       []string{hipKernelNameEmbedLookup, hipKernelNameVectorScale, hipKernelNameProjection},
	}
	assistantPlan := hipAttachedDrafterAssistantVerifierPlan{
		Status:             attachedDrafterAssistantVerifierPlanTensorBound,
		HiddenSize:         layer0.HiddenSize,
		VocabSize:          layer0.VocabSize,
		LayerCount:         1,
		ProjectionEncoding: "bf16",
		Embedding: hipDeviceEmbeddingLookupConfig{
			EmbeddingPointer: embedding.Pointer(),
			EmbeddingBytes:   embedding.SizeBytes(),
			TableEncoding:    hipEmbeddingTableEncodingBF16,
			VocabSize:        layer0.VocabSize,
			HiddenSize:       layer0.HiddenSize,
		},
		Norm:           layer0.FinalNorm,
		PreProjection:  preProjection,
		PostProjection: postProjection,
		Layers:         []hipAttachedDrafterAssistantVerifierLayerPlan{hipAssistantLayerPlanFromGemma4Q4Fixture(layer0)},
	}
	target.storeAttachedDrafterRuntime(&hipAttachedDrafterRuntime{
		attachment: AttachedDrafterAttachment{
			NativeAttachment: hipKernelStatusLinked,
			Labels: map[string]string{
				"attached_drafter_native_attachment": hipKernelStatusLinked,
			},
		},
		assistantPlan: assistantPlan,
		inputPlan:     inputPlan,
	})

	launchStart := len(driver.launches)
	result, err := target.GenerateAttachedDrafterFromState(context.Background(), AttachedDrafterAttachment{NativeAttachment: hipKernelStatusLinked}, AttachedDrafterStateGenerateRequest{
		State:       session,
		Input:       "tokens:0",
		MaxTokens:   2,
		DraftTokens: 2,
	})
	core.RequireNoError(t, err)

	core.AssertEqual(t, inferdecode.ModeSpeculative, result.Mode)
	core.AssertEqual(t, "tokens:0", result.Prompt)
	core.AssertEqual(t, 2, len(result.Tokens))
	core.AssertEqual(t, 2, result.Metrics.EmittedTokens)
	core.AssertEqual(t, 2, result.Metrics.DraftTokens)
	core.AssertEqual(t, 1, result.Metrics.DraftCalls)
	core.AssertEqual(t, 2, result.Metrics.TargetCalls)
	core.AssertEqual(t, 2, result.Metrics.AcceptedTokens)
	core.AssertEqual(t, 0, result.Metrics.RejectedTokens)
	retained, ok := session.runtime.(*hipGemma4Q4DeviceDecodeState)
	if !ok || retained == nil || retained.closed {
		t.Fatalf("retained runtime = %#v ok=%v, want live Gemma4 q4 device state", session.runtime, ok)
	}
	core.AssertEqual(t, []int{4}, retained.LayerTokenCounts())
	if countDeviceAttentionLaunches(driver.launches[launchStart:]) == 0 {
		t.Fatalf("attached retained generate launched no descriptor-backed attention kernels")
	}
}

func TestHIPAttachedDrafterCarryAdvanceReturnsHiddenForNextDraft_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 8, 1, 8)
	defer cleanup0()
	hipGemma4Q4InstallNonzeroEmbeddingFixture(t, driver, &layer0, "attached carry advance")
	layer0.PerLayerInput = hipGemma4Q4PerLayerInputConfig{}
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0}}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	greedyBuffer, err := workspace.BorrowProjectionGreedyBest(driver)
	core.RequireNoError(t, err)

	initial, _, err := hipRunGemma4Q4SingleTokenForwardWithStateInternal(context.Background(), driver, cfg, hipGemma4Q4DecodeState{}, hipGemma4Q4ForwardRequest{
		TokenID:            1,
		Position:           0,
		Epsilon:            1e-6,
		DeviceKVAttention:  true,
		DeviceKVMode:       rocmKVCacheModeKQ8VQ4,
		ReturnDeviceState:  true,
		SkipFinalSample:    true,
		OmitDebugTensors:   true,
		AttentionWorkspace: workspace,
	}, false)
	core.RequireNoError(t, err)
	defer initial.DeviceState.Close()

	launchStart := len(driver.launches)
	advanced, err := hipAdvanceAttachedDrafterCarryLead(context.Background(), driver, hipAttachedDrafterCarryAdvanceRequest{
		TargetForward:    cfg,
		DeviceKVMode:     rocmKVCacheModeKQ8VQ4,
		EngineConfig:     defaultHIPGemma4Q4EngineConfig(),
		State:            hipGemma4Q4DecodeState{},
		PriorDeviceState: initial.DeviceState,
		TokenID:          0,
		Position:         1,
		Epsilon:          1e-6,
		GreedyBuffer:     greedyBuffer,
		Workspace:        workspace,
	})
	core.RequireNoError(t, err)
	defer advanced.Close()

	core.AssertEqual(t, 2, advanced.Position)
	if advanced.Current.DeviceFinalHidden == nil || advanced.Current.DeviceFinalHidden.Pointer() == 0 {
		t.Fatalf("carry advance hidden = %#v, want hidden for next assistant draft", advanced.Current.DeviceFinalHidden)
	}
	if advanced.DeviceState == nil || advanced.DeviceState.closed {
		t.Fatalf("carry advance device state = %#v, want live state", advanced.DeviceState)
	}
	core.AssertEqual(t, []int{2}, advanced.DeviceState.LayerTokenCounts())
	core.AssertEqual(t, true, advanced.Current.Greedy.TokenID >= 0)
	launches := driver.launches[launchStart:]
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameEmbedLookupGreedyToken))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameMLXQ4Proj))
	if batchProjectionLaunches := countLaunchName(launches, hipKernelNameMLXQ4ProjBatch); batchProjectionLaunches == 0 {
		t.Fatalf("carry advance launched projection_batch=%d, want batched projection", batchProjectionLaunches)
	}
	if batchAttentionLaunches := countLaunchName(launches, hipKernelNameAttentionHeadsBatchCausal); batchAttentionLaunches == 0 {
		t.Fatalf("carry advance launched attention_batch=%d, want batched attention", batchAttentionLaunches)
	}
	if finalGreedyLaunches := countLaunchName(launches, hipKernelNameMLXQ4ProjGreedy); finalGreedyLaunches == 0 {
		t.Fatalf("carry advance launched final_greedy=%d, want final greedy projection", finalGreedyLaunches)
	}
}

func hipAssistantBF16ProjectionPlanFixture(t *testing.T, driver nativeHIPDriver, label string, rows, cols int) (hipAttachedDrafterAssistantProjectionPlan, func()) {
	t.Helper()
	payload, err := hipUint16Payload(make([]uint16, rows*cols))
	core.RequireNoError(t, err)
	buffer, err := hipUploadByteBuffer(driver, "rocm.hip.AttachedDrafterAssistantDraftStep", label, payload, rows*cols)
	core.RequireNoError(t, err)
	plan := hipAttachedDrafterAssistantProjectionPlan{
		Encoding: "bf16",
		Rows:     rows,
		Cols:     cols,
		BF16: hipBF16DeviceWeightConfig{
			WeightPointer: buffer.Pointer(),
			WeightBytes:   buffer.SizeBytes(),
			Rows:          rows,
			Cols:          cols,
		},
	}
	return plan, func() { _ = buffer.Close() }
}

func hipAssistantLayerPlanFromGemma4Q4Fixture(layer hipGemma4Q4Layer0Config) hipAttachedDrafterAssistantVerifierLayerPlan {
	q4 := func(cfg hipMLXQ4DeviceWeightConfig) hipAttachedDrafterAssistantProjectionPlan {
		return hipAttachedDrafterAssistantProjectionPlan{
			Encoding:  "mlx_affine",
			Rows:      cfg.Rows,
			Cols:      cfg.Cols,
			MLXAffine: cfg,
		}
	}
	return hipAttachedDrafterAssistantVerifierLayerPlan{
		Layer:              layer.Layer,
		LayerType:          layer.LayerType,
		HiddenSize:         layer.HiddenSize,
		HeadDim:            layer.HeadDim,
		QueryHeads:         layer.QueryHeads,
		RoPEBase:           layer.RoPEBase,
		RoPERotaryDim:      layer.RoPERotaryDim,
		RoPEFrequencyScale: layer.RoPEFrequencyScale,
		SlidingWindow:      layer.SlidingWindow,
		LayerScalar:        layer.effectiveLayerScalar(),
		InputNorm:          layer.InputNorm,
		PostAttentionNorm:  layer.PostAttentionNorm,
		PreFeedforward:     layer.PreFeedForwardNorm,
		PostFeedforward:    layer.PostFeedForwardNorm,
		QueryNorm:          layer.QueryNorm,
		QueryProjection:    q4(layer.QueryProjection),
		OutputProjection:   q4(layer.OutputProjection),
		GateProjection:     q4(layer.GateProjection),
		UpProjection:       q4(layer.UpProjection),
		DownProjection:     q4(layer.DownProjection),
	}
}

func countDeviceAttentionLaunches(launches []hipKernelLaunchConfig) int {
	var count int
	for _, launch := range launches {
		if launch.Name == hipKernelNameAttention &&
			len(launch.Args) >= hipAttentionLaunchArgsBytes &&
			binary.LittleEndian.Uint32(launch.Args[76:]) == hipAttentionKVSourceDevice {
			count++
		}
		if launch.Name == hipKernelNameAttentionHeads &&
			len(launch.Args) >= hipAttentionHeadsLaunchArgsBytes &&
			binary.LittleEndian.Uint32(launch.Args[80:]) == hipAttentionKVSourceDevice {
			count++
		}
	}
	return count
}

func countLaunchName(launches []hipKernelLaunchConfig, name string) int {
	var count int
	for _, launch := range launches {
		if launch.Name == name {
			count++
		}
	}
	return count
}

func setHIPMLXQ4GELUTanhMLPPersistentRouteForTest(t *testing.T, enabled bool) {
	t.Helper()
	previous := hipMLXQ4GELUTanhMLPPersistentRouteEnabled
	hipMLXQ4GELUTanhMLPPersistentRouteEnabled = enabled
	t.Cleanup(func() {
		hipMLXQ4GELUTanhMLPPersistentRouteEnabled = previous
	})
}

func rmsNormRoPEHeadPositions(launches []hipKernelLaunchConfig) []int {
	positions := []int{}
	for _, launch := range launches {
		if launch.Name != hipKernelNameRMSNormRoPEHeads || len(launch.Args) < hipRMSNormRoPEHeadsLaunchArgsBytes {
			continue
		}
		positions = append(positions, int(binary.LittleEndian.Uint32(launch.Args[64:])))
	}
	return positions
}

func countUint64Value(values []uint64, want uint64) int {
	var count int
	for _, value := range values {
		if value == want {
			count++
		}
	}
	return count
}

func gemma4Q4DeviceKVPagesForTokens(tokens int) int {
	if tokens <= 0 {
		return 0
	}
	blockSize := hipGemma4Q4DeviceKVBlockSize()
	if blockSize <= 0 {
		return tokens
	}
	return (tokens + blockSize - 1) / blockSize
}

func countKVEncodeTokenLaunches(launches []hipKernelLaunchConfig) int {
	var count int
	for _, launch := range launches {
		if launch.Name == hipKernelNameKVEncodeToken {
			count++
		}
	}
	return count
}

func repeatUint16(value uint16, count int) []uint16 {
	values := make([]uint16, count)
	for index := range values {
		values[index] = value
	}
	return values
}

func TestHIPAttentionHeadsBlockSize_Good(t *testing.T) {
	core.AssertEqual(t, uint32(256), hipAttentionHeadsBlockSize(1))
	core.AssertEqual(t, uint32(256), hipAttentionHeadsBlockSize(15))
	core.AssertEqual(t, uint32(512), hipAttentionHeadsBlockSize(16))
	core.AssertEqual(t, uint32(512), hipAttentionHeadsBlockSize(511))
	core.AssertEqual(t, uint32(512), hipAttentionHeadsBlockSize(512))
	core.AssertEqual(t, uint32(512), hipAttentionHeadsBlockSize(1023))
	core.AssertEqual(t, uint32(512), hipAttentionHeadsBlockSize(1024))
	core.AssertEqual(t, uint32(512), hipAttentionHeadsBlockSize(2000))
}

func TestHIPAttentionHeadsSharedMemBytes_Good(t *testing.T) {
	plain, err := hipAttentionHeadsSharedMemBytes(2000, false)
	core.RequireNoError(t, err)
	core.AssertEqual(t, uint32(8000), plain)

	shortDevice, err := hipAttentionHeadsSharedMemBytes(511, true)
	core.RequireNoError(t, err)
	core.AssertEqual(t, uint32(8180), shortDevice)

	longDevice, err := hipAttentionHeadsSharedMemBytes(2000, true)
	core.RequireNoError(t, err)
	core.AssertEqual(t, uint32(32000), longDevice)
}

func TestHIPAttentionHeadsChunkedSharedMemBytes_Good(t *testing.T) {
	dim256, err := hipAttentionHeadsChunkedSharedMemBytes(128, 256)
	core.RequireNoError(t, err)
	core.AssertEqual(t, uint32(3072), dim256)

	dim512, err := hipAttentionHeadsChunkedSharedMemBytes(128, 512)
	core.RequireNoError(t, err)
	core.AssertEqual(t, uint32(4096), dim512)

	_, err = hipAttentionHeadsChunkedSharedMemBytes(0, 512)
	core.AssertNotEqual(t, nil, err)
}

func TestHIPAttentionHeadsChunkedEligible_BlockPagesGood(t *testing.T) {
	pages := make([]rocmDeviceKVPage, 0, 20)
	for index := 0; index < 20; index++ {
		pages = append(pages, rocmDeviceKVPage{
			tokenStart: index * 16,
			tokenCount: 16,
			keyWidth:   256,
			valueWidth: 256,
			key:        rocmDeviceKVTensor{pointer: nativeDevicePointer(1000 + index), encoding: rocmKVEncodingQ8Rows, sizeBytes: 16*256 + 16*4},
			value:      rocmDeviceKVTensor{pointer: nativeDevicePointer(2000 + index), encoding: rocmKVEncodingQ4Rows, sizeBytes: (16*256)/2 + 16*4},
		})
	}
	req := hipAttentionRequest{
		DeviceKV: &rocmDeviceKVCache{
			mode:       rocmKVCacheModeKQ8VQ4,
			blockSize:  16,
			pages:      pages,
			tokenCount: 320,
		},
		DescriptorTable: &rocmDeviceKVDescriptorTable{},
	}
	core.AssertEqual(t, true, hipAttentionHeadsChunkedEligible(req, 2, 256, 320))
	req.WindowSize = 512
	core.AssertEqual(t, false, hipAttentionHeadsChunkedEligible(req, 2, 256, 320))
	req.WindowSize = 0

	req.DeviceKV.mode = rocmKVCacheModeQ8
	core.AssertEqual(t, false, hipAttentionHeadsChunkedEligible(req, 2, 256, 320))
}

func TestHIPAttentionHeadsChunkedEligible_Gemma4HeadDim512_Good(t *testing.T) {
	cache := &rocmDeviceKVCache{
		mode:       rocmKVCacheModeKQ8VQ4,
		blockSize:  1,
		pages:      []rocmDeviceKVPage{{tokenStart: 0, tokenCount: 513, keyWidth: 512, valueWidth: 512}},
		tokenCount: 513,
	}
	descriptor := &rocmDeviceKVDescriptorTable{}
	core.AssertEqual(t, true, hipAttentionHeadsChunkedEligible(hipAttentionRequest{
		DeviceKV:        cache,
		DescriptorTable: descriptor,
	}, 4, 512, 513))
	core.AssertEqual(t, false, hipAttentionHeadsChunkedEligible(hipAttentionRequest{
		DeviceKV:        cache,
		DescriptorTable: descriptor,
	}, 4, 513, 513))

	multiHeadCache := &rocmDeviceKVCache{
		mode:       rocmKVCacheModeKQ8VQ4,
		blockSize:  1,
		pages:      []rocmDeviceKVPage{{tokenStart: 0, tokenCount: 513, keyWidth: 2048, valueWidth: 2048}},
		tokenCount: 513,
	}
	multiHeadReq := hipAttentionRequest{
		KeyHeads:        4,
		DeviceKV:        multiHeadCache,
		DescriptorTable: descriptor,
	}
	core.AssertEqual(t, true, hipAttentionHeadsChunkedEligible(multiHeadReq, 8, 512, 513))
	core.AssertEqual(t, false, hipAttentionHeadsChunkedEligible(multiHeadReq, 7, 512, 513))
	core.AssertEqual(t, false, hipAttentionHeadsChunkedEligible(multiHeadReq, 8, 511, 513))
	multiHeadReq.KeyHeads = 9
	core.AssertEqual(t, false, hipAttentionHeadsChunkedEligible(multiHeadReq, 8, 512, 513))

	workspace := &hipAttentionHeadsChunkedWorkspace{}
	core.AssertEqual(t, false, hipAttentionHeadsBatchChunkedEligible(hipAttentionHeadsBatchCausalDeviceRequest{
		DeviceKV:        cache,
		DescriptorTable: descriptor,
		Dim:             512,
		TokenCount:      512,
		HeadCount:       1,
		QueryCount:      1,
	}, workspace))
	core.AssertEqual(t, true, hipAttentionHeadsBatchChunkedEligible(hipAttentionHeadsBatchCausalDeviceRequest{
		DeviceKV:        cache,
		DescriptorTable: descriptor,
		Dim:             512,
		TokenCount:      513,
		HeadCount:       1,
		QueryCount:      1,
	}, workspace))
	core.AssertEqual(t, false, hipAttentionHeadsBatchChunkedEligible(hipAttentionHeadsBatchCausalDeviceRequest{
		DeviceKV:        cache,
		DescriptorTable: descriptor,
		Dim:             513,
		TokenCount:      513,
		HeadCount:       1,
		QueryCount:      1,
	}, workspace))
	core.AssertEqual(t, true, hipAttentionHeadsBatchChunkedEligible(hipAttentionHeadsBatchCausalDeviceRequest{
		DeviceKV:        multiHeadCache,
		DescriptorTable: descriptor,
		Dim:             512,
		TokenCount:      513,
		HeadCount:       8,
		KeyHeads:        4,
		QueryCount:      1,
	}, workspace))
	multiHeadCache.tokenCount = 512
	multiHeadCache.pages[0].tokenCount = 512
	core.AssertEqual(t, false, hipAttentionHeadsBatchChunkedEligible(hipAttentionHeadsBatchCausalDeviceRequest{
		DeviceKV:        multiHeadCache,
		DescriptorTable: descriptor,
		Dim:             512,
		TokenCount:      512,
		HeadCount:       8,
		KeyHeads:        4,
		QueryCount:      1,
	}, workspace))
	cache.tokenCount = 513
	cache.pages[0].tokenCount = 513
	core.AssertEqual(t, true, hipAttentionHeadsBatchChunkedEligible(hipAttentionHeadsBatchCausalDeviceRequest{
		DeviceKV:        cache,
		DescriptorTable: descriptor,
		Dim:             512,
		TokenCount:      513,
		HeadCount:       8,
		KeyHeads:        1,
		QueryCount:      1,
	}, workspace))
	cache.tokenCount = 512
	cache.pages[0].tokenCount = 512
	core.AssertEqual(t, false, hipAttentionHeadsBatchChunkedEligible(hipAttentionHeadsBatchCausalDeviceRequest{
		DeviceKV:        cache,
		DescriptorTable: descriptor,
		Dim:             512,
		TokenCount:      512,
		HeadCount:       8,
		KeyHeads:        1,
		QueryCount:      1,
	}, workspace))
}

func TestHIPAttentionHeadsBatchChunkedEligibilityReason_Good(t *testing.T) {
	cache := &rocmDeviceKVCache{
		mode:       rocmKVCacheModeKQ8VQ4,
		blockSize:  1,
		pages:      []rocmDeviceKVPage{{tokenStart: 0, tokenCount: 513, keyWidth: 256, valueWidth: 256}},
		tokenCount: 513,
	}
	descriptor := &rocmDeviceKVDescriptorTable{}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	req := hipAttentionHeadsBatchCausalDeviceRequest{
		DeviceKV:        cache,
		DescriptorTable: descriptor,
		Dim:             256,
		TokenCount:      513,
		HeadCount:       8,
		KeyHeads:        1,
		QueryCount:      1,
		QueryStartToken: 512,
		WindowSize:      512,
	}

	core.AssertEqual(t, hipAttentionHeadsBatchChunkedEligibilityEligible, hipAttentionHeadsBatchChunkedEligibilityReasonFor(req, workspace))
	req.KeyHeads = 3
	core.AssertEqual(t, hipAttentionHeadsBatchChunkedEligibilityInvalidHeads, hipAttentionHeadsBatchChunkedEligibilityReasonFor(req, workspace))
	req.KeyHeads = 1
	core.AssertEqual(t, hipAttentionHeadsBatchChunkedEligibilityMissingWorkspace, hipAttentionHeadsBatchChunkedEligibilityReasonFor(req, nil))
	req.KeyHeads = 2
	req.Dim = 255
	core.AssertEqual(t, hipAttentionHeadsBatchChunkedEligibilityInvalidDimension, hipAttentionHeadsBatchChunkedEligibilityReasonFor(req, workspace))
	req.KeyHeads = 1
	req.Dim = 256
	req.DeviceKV = nil
	core.AssertEqual(t, hipAttentionHeadsBatchChunkedEligibilityMissingDeviceKV, hipAttentionHeadsBatchChunkedEligibilityReasonFor(req, workspace))
	req.DeviceKV = cache
	req.DescriptorTable = nil
	core.AssertEqual(t, hipAttentionHeadsBatchChunkedEligibilityMissingDescriptor, hipAttentionHeadsBatchChunkedEligibilityReasonFor(req, workspace))
	req.DescriptorTable = descriptor
	req.Dim = hipAttentionHeadsChunkedBlockSize + 1
	core.AssertEqual(t, hipAttentionHeadsBatchChunkedEligibilityInvalidDimension, hipAttentionHeadsBatchChunkedEligibilityReasonFor(req, workspace))
	req.Dim = 256
	req.WindowSize = 0
	req.TokenCount = hipAttentionHeadsSharedMaxTokens
	core.AssertEqual(t, hipAttentionHeadsBatchChunkedEligibilityBelowTokenThreshold, hipAttentionHeadsBatchChunkedEligibilityReasonFor(req, workspace))
	req.TokenCount = hipAttentionHeadsSharedMaxTokens + 1
	cache.tokenCount = req.TokenCount
	cache.pages[0].tokenCount = req.TokenCount
	core.AssertEqual(t, hipAttentionHeadsBatchChunkedEligibilityEligible, hipAttentionHeadsBatchChunkedEligibilityReasonFor(req, workspace))
	cache.mode = rocmKVCacheModeQ8
	core.AssertEqual(t, hipAttentionHeadsBatchChunkedEligibilityUnsupportedKVMode, hipAttentionHeadsBatchChunkedEligibilityReasonFor(req, workspace))
	cache.mode = rocmKVCacheModeKQ8VQ4
	cache.tokenCount++
	core.AssertEqual(t, hipAttentionHeadsBatchChunkedEligibilityTokenCountMismatch, hipAttentionHeadsBatchChunkedEligibilityReasonFor(req, workspace))
}

func TestHIPAttentionHeadsBatchChunkedEligibilityReason_SlidingWindowCrossover_Good(t *testing.T) {
	cache := &rocmDeviceKVCache{
		mode:       rocmKVCacheModeKQ8VQ4,
		blockSize:  1,
		pages:      []rocmDeviceKVPage{{tokenStart: 0, tokenCount: 129, keyWidth: 256, valueWidth: 256}},
		tokenCount: 129,
	}
	req := hipAttentionHeadsBatchCausalDeviceRequest{
		DeviceKV:        cache,
		DescriptorTable: &rocmDeviceKVDescriptorTable{},
		Dim:             256,
		TokenCount:      129,
		HeadCount:       8,
		KeyHeads:        1,
		QueryCount:      1,
		QueryStartToken: 128,
		WindowSize:      512,
	}

	core.AssertEqual(t, hipAttentionHeadsBatchChunkedEligibilityEligible, hipAttentionHeadsBatchChunkedEligibilityReasonFor(req, &hipAttentionHeadsChunkedWorkspace{}))
	req.TokenCount = 128
	req.QueryStartToken = 127
	cache.tokenCount = 128
	cache.pages[0].tokenCount = 128
	core.AssertEqual(t, hipAttentionHeadsBatchChunkedEligibilityBelowTokenThreshold, hipAttentionHeadsBatchChunkedEligibilityReasonFor(req, &hipAttentionHeadsChunkedWorkspace{}))
}

func TestHIPAttentionHeadsBatchChunkedEligibilityReason_HeadDim512Crossover_Good(t *testing.T) {
	cache := &rocmDeviceKVCache{
		mode:       rocmKVCacheModeKQ8VQ4,
		blockSize:  1,
		pages:      []rocmDeviceKVPage{{tokenStart: 0, tokenCount: 129, keyWidth: 512, valueWidth: 512}},
		tokenCount: 129,
	}
	req := hipAttentionHeadsBatchCausalDeviceRequest{
		DeviceKV:        cache,
		DescriptorTable: &rocmDeviceKVDescriptorTable{},
		Dim:             512,
		TokenCount:      129,
		HeadCount:       8,
		KeyHeads:        1,
		QueryCount:      1,
		QueryStartToken: 128,
	}

	core.AssertEqual(t, hipAttentionHeadsBatchChunkedEligibilityEligible, hipAttentionHeadsBatchChunkedEligibilityReasonFor(req, &hipAttentionHeadsChunkedWorkspace{}))
}

func BenchmarkHIPDeviceByteBufferPool_ReusedSize(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	const sizeBytes uint64 = 4096
	const pointer nativeDevicePointer = 42
	hipDeviceByteBufferPool.Lock()
	hipDeviceByteBufferPool.single = [hipDeviceByteBufferPoolSingleSlots]hipDeviceByteBufferPoolSingleSlot{}
	hipDeviceByteBufferPool.entries = make(map[uint64][]hipDeviceByteBufferPoolEntry)
	hipDeviceByteBufferPool.bytes = 0
	hipDeviceByteBufferPool.Unlock()
	if !hipDeviceByteBufferPoolPut(driver, pointer, sizeBytes) {
		b.Fatal("seed device buffer pool")
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		got, ok := hipDeviceByteBufferPoolTake(driver, sizeBytes)
		if !ok || got != pointer {
			b.Fatalf("take = %d, %v; want %d, true", got, ok, pointer)
		}
		if !hipDeviceByteBufferPoolPut(driver, got, sizeBytes) {
			b.Fatal("return device buffer to pool")
		}
	}
}

func TestHIPDeviceByteBufferPool_Good_PrewarmSeedsExactSize(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	const sizeBytes uint64 = hipMLXQ4ProjectionBestBytes
	hipDeviceByteBufferPool.Lock()
	hipDeviceByteBufferPool.single = [hipDeviceByteBufferPoolSingleSlots]hipDeviceByteBufferPoolSingleSlot{}
	hipDeviceByteBufferPool.entries = make(map[uint64][]hipDeviceByteBufferPoolEntry)
	hipDeviceByteBufferPool.bytes = 0
	hipDeviceByteBufferPool.Unlock()

	hipPrewarmDeviceByteBufferPool(driver, sizeBytes, 2)
	core.AssertEqual(t, 2, len(driver.allocations))
	core.AssertEqual(t, sizeBytes, driver.allocations[0])
	core.AssertEqual(t, sizeBytes, driver.allocations[1])

	before := len(driver.allocations)
	first, err := hipAllocateByteBuffer(driver, "rocm.hip.Test", "first prewarmed greedy result", sizeBytes, 1)
	if err != nil {
		t.Fatalf("allocate first prewarmed buffer: %v", err)
	}
	second, err := hipAllocateByteBuffer(driver, "rocm.hip.Test", "second prewarmed greedy result", sizeBytes, 1)
	if err != nil {
		t.Fatalf("allocate second prewarmed buffer: %v", err)
	}
	core.AssertEqual(t, before, len(driver.allocations))
	core.AssertTrue(t, first.Pointer() != 0, "first pointer should be non-zero")
	core.AssertTrue(t, second.Pointer() != 0, "second pointer should be non-zero")
	if first.Pointer() == second.Pointer() {
		t.Fatal("prewarmed buffers should be distinct while both are borrowed")
	}
	if err := first.Close(); err != nil {
		t.Fatalf("close first: %v", err)
	}
	if err := second.Close(); err != nil {
		t.Fatalf("close second: %v", err)
	}
}

func TestHIPDeviceByteBufferPool_Good_SingleSlotAvoidsSliceEntry(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	const sizeBytes uint64 = 4096
	pointers := [...]nativeDevicePointer{42, 43, 44}
	hipDeviceByteBufferPool.Lock()
	hipDeviceByteBufferPool.single = [hipDeviceByteBufferPoolSingleSlots]hipDeviceByteBufferPoolSingleSlot{}
	hipDeviceByteBufferPool.entries = make(map[uint64][]hipDeviceByteBufferPoolEntry)
	hipDeviceByteBufferPool.bytes = 0
	hipDeviceByteBufferPool.Unlock()

	for _, pointer := range pointers {
		if !hipDeviceByteBufferPoolPut(driver, pointer, sizeBytes) {
			t.Fatalf("put single-slot pointer %d", pointer)
		}
	}
	hipDeviceByteBufferPool.Lock()
	entries := len(hipDeviceByteBufferPool.entries[sizeBytes])
	hipDeviceByteBufferPool.Unlock()
	core.AssertEqual(t, 0, entries)

	for range pointers {
		got, ok := hipDeviceByteBufferPoolTake(driver, sizeBytes)
		if !ok {
			t.Fatal("take = false; want true")
		}
		found := false
		for _, pointer := range pointers {
			if got == pointer {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("take = %d; want one of %v", got, pointers)
		}
	}
}

func BenchmarkHIPLaunchPacketPool_ReusedSize(b *testing.B) {
	hipLaunchPacketPools.Range(func(key, _ any) bool {
		hipLaunchPacketPools.Delete(key)
		return true
	})
	packet := hipBorrowLaunchPacket(hipMLXQ4TripleProjLaunchArgsBytes)
	hipReleaseLaunchPacket(packet)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		packet = hipBorrowLaunchPacket(hipMLXQ4TripleProjLaunchArgsBytes)
		if len(packet) != hipMLXQ4TripleProjLaunchArgsBytes {
			b.Fatalf("packet len = %d, want %d", len(packet), hipMLXQ4TripleProjLaunchArgsBytes)
		}
		hipReleaseLaunchPacket(packet)
	}
}

func BenchmarkHIPKernelLaunchConfigValidate_Hot(b *testing.B) {
	config := hipKernelLaunchConfig{
		Name:   hipKernelNameMLXQ4Proj,
		Args:   []byte{1},
		GridX:  1,
		GridY:  1,
		GridZ:  1,
		BlockX: hipMLXQ4ProjectionBlockSize,
		BlockY: 1,
		BlockZ: 1,
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := config.Validate(); err != nil {
			b.Fatal(err)
		}
	}
}

type fakeHIPUint64Reader struct {
	fakeHIPDriver
	value uint64
}

func (driver *fakeHIPUint64Reader) CopyDeviceToHostUint64(nativeDevicePointer) (uint64, error) {
	return driver.value, nil
}

func BenchmarkHIPReadDeviceUint64_DirectReader(b *testing.B) {
	driver := &fakeHIPUint64Reader{
		fakeHIPDriver: fakeHIPDriver{available: true},
		value:         0x400921fb54442d18,
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		value, err := hipReadDeviceUint64(driver, 42)
		if err != nil {
			b.Fatal(err)
		}
		if value != driver.value {
			b.Fatalf("value = %#x, want %#x", value, driver.value)
		}
	}
}

func BenchmarkHIPGemma4Q4DeviceDecodeStatePool_Reused(b *testing.B) {
	state := hipNewGemma4Q4DeviceDecodeState(rocmKVCacheModeKQ8VQ4, 35)
	hipReleaseGemma4Q4DeviceLayerStates(state.layers)
	state.layers = nil
	state.closed = true
	hipReleaseClosedGemma4Q4DeviceDecodeState(state)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		state = hipNewGemma4Q4DeviceDecodeState(rocmKVCacheModeKQ8VQ4, 35)
		if state == nil || state.mode != rocmKVCacheModeKQ8VQ4 || len(state.layers) != 0 || cap(state.layers) < 35 {
			b.Fatal("decode state pool returned invalid state")
		}
		hipReleaseGemma4Q4DeviceLayerStates(state.layers)
		state.layers = nil
		state.closed = true
		hipReleaseClosedGemma4Q4DeviceDecodeState(state)
	}
}

func BenchmarkHIPAttentionHeadsChunkedWorkspace_AttentionOutputReused(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	output, err := workspace.EnsureAttentionOutput(driver, 8, 256)
	if err != nil {
		b.Fatal(err)
	}
	if output.Count() != 2048 || output.SizeBytes() != 8192 {
		b.Fatalf("attention output shape = %d/%d, want 2048/8192", output.Count(), output.SizeBytes())
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		output, err = workspace.EnsureAttentionOutput(driver, 8, 256)
		if err != nil {
			b.Fatal(err)
		}
		if output.Count() != 2048 || output.SizeBytes() != 8192 {
			b.Fatalf("attention output shape = %d/%d, want 2048/8192", output.Count(), output.SizeBytes())
		}
	}
}

func TestHIPAttentionHeadsChunkedWorkspace_AttentionOutputReusesLarger_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	large, err := workspace.EnsureBatchAttentionOutput(driver, 8192)
	core.RequireNoError(t, err)
	largePointer := large.Pointer()
	smallBatch, err := workspace.EnsureBatchAttentionOutput(driver, 4096)
	core.RequireNoError(t, err)
	smallBatchPointer := smallBatch.Pointer()
	smallBatchCount := smallBatch.Count()
	smallConcat, err := workspace.EnsureAttentionOutput(driver, 8, 256)
	core.RequireNoError(t, err)

	if smallBatchPointer != largePointer || smallBatchCount != 4096 {
		t.Fatalf("small batch attention view = %x/%d, want borrowed view of %x", smallBatchPointer, smallBatchCount, largePointer)
	}
	if smallConcat.Pointer() != largePointer || smallConcat.Count() != 2048 {
		t.Fatalf("small concat attention view = %#v, want borrowed view of %x", smallConcat, largePointer)
	}
	core.AssertEqual(t, 1, len(driver.allocations))
	if _, ok := workspace.AttentionOutputs[4096]; ok {
		t.Fatalf("smaller batch attention output got its own allocation")
	}
	if _, ok := workspace.AttentionOutputs[2048]; ok {
		t.Fatalf("smaller concat attention output got its own allocation")
	}
}

func TestHIPAttentionHeadsChunkedWorkspace_EnsureCapacityRoundsChunkCounts_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	core.RequireNoError(t, workspace.Ensure(driver, 8, 512, 1793, hipAttentionHeadsChunkSize))
	core.AssertEqual(t, 2, len(driver.allocations))
	core.AssertEqual(t, 131072, workspace.partialCap)
	core.AssertEqual(t, 512, workspace.statsCap)
	core.AssertEqual(t, uint64(131072*4), workspace.Partial.SizeBytes())
	core.AssertEqual(t, uint64(512*4), workspace.Stats.SizeBytes())

	allocationCount := len(driver.allocations)
	core.RequireNoError(t, workspace.Ensure(driver, 8, 512, 2048, hipAttentionHeadsChunkSize))
	core.AssertEqual(t, allocationCount, len(driver.allocations))
	core.AssertEqual(t, 131072, workspace.partialCap)
	core.AssertEqual(t, 512, workspace.statsCap)

	core.RequireNoError(t, workspace.Ensure(driver, 8, 512, 2049, hipAttentionHeadsChunkSize))
	core.AssertEqual(t, allocationCount+2, len(driver.allocations))
	core.AssertEqual(t, 262144, workspace.partialCap)
	core.AssertEqual(t, 1024, workspace.statsCap)
}

func TestHIPGemma4Q4AttentionWorkspaceDecodeCapacity_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	cfg := hipGemma4Q4ForwardConfig{
		Layers: []hipGemma4Q4Layer0Config{
			{QueryHeads: 4, HeadDim: 256},
			{QueryHeads: 8, HeadDim: hipAttentionHeadsChunkedBlockSize},
		},
	}

	core.RequireNoError(t, hipGemma4Q4EnsureAttentionWorkspaceDecodeCapacity(driver, workspace, cfg, 2050))
	core.AssertEqual(t, 2, len(driver.allocations))
	core.AssertEqual(t, 262144, workspace.partialCap)
	core.AssertEqual(t, 1024, workspace.statsCap)

	allocationCount := len(driver.allocations)
	core.RequireNoError(t, workspace.Ensure(driver, 8, hipAttentionHeadsChunkedBlockSize, 2048, hipAttentionHeadsChunkSize))
	core.AssertEqual(t, allocationCount, len(driver.allocations))
}

func BenchmarkHIPAttentionHeadsChunkedWorkspace_EnsureCapacityReuse(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	if err := workspace.Ensure(driver, 8, 512, 2049, hipAttentionHeadsChunkSize); err != nil {
		b.Fatal(err)
	}
	allocationCount := len(driver.allocations)

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := workspace.Ensure(driver, 8, 512, 2048, hipAttentionHeadsChunkSize); err != nil {
			b.Fatal(err)
		}
		if len(driver.allocations) != allocationCount {
			b.Fatalf("workspace ensure allocated after warmup: got %d allocations, want %d", len(driver.allocations), allocationCount)
		}
	}
}

func TestHIPAttentionHeadsBatchCausalWorkspaceCap_Good(t *testing.T) {
	t.Setenv("GO_ROCM_DISABLE_DEVICE_BUFFER_POOL", "1")
	const (
		dim        = 1
		tokenCount = hipAttentionHeadsSharedMaxTokens + 1
		queryCount = 1
	)
	keyValues := make([]float32, tokenCount*dim)
	valueValues := make([]float32, tokenCount*dim)
	for index := range keyValues {
		keyValues[index] = 1
		valueValues[index] = float32(index + 1)
	}
	keyPayload, err := hipFloat32Payload(keyValues)
	core.RequireNoError(t, err)
	valuePayload, err := hipFloat32Payload(valueValues)
	core.RequireNoError(t, err)

	for _, tc := range []struct {
		name          string
		headCount     int
		wantWorkspace bool
	}{
		{name: "under_cap", headCount: 1, wantWorkspace: true},
		{name: "over_cap", headCount: 33, wantWorkspace: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			driver := &fakeHIPDriver{available: true}
			workspace := &hipAttentionHeadsChunkedWorkspace{}
			defer workspace.Close()
			queryValues := make([]float32, queryCount*tc.headCount*dim)
			for index := range queryValues {
				queryValues[index] = 1
			}
			queryPayload, err := hipFloat32Payload(queryValues)
			core.RequireNoError(t, err)
			query, err := hipUploadByteBuffer(driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "attention batch query", queryPayload, len(queryValues))
			core.RequireNoError(t, err)
			defer query.Close()
			keys, err := hipUploadByteBuffer(driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "attention batch keys", keyPayload, len(keyValues))
			core.RequireNoError(t, err)
			defer keys.Close()
			values, err := hipUploadByteBuffer(driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "attention batch values", valuePayload, len(valueValues))
			core.RequireNoError(t, err)
			defer values.Close()
			output, err := hipAllocateByteBuffer(driver, "rocm.hip.AttentionHeadsBatchCausalLaunch", "attention batch output", uint64(len(queryValues)*4), len(queryValues))
			core.RequireNoError(t, err)
			defer output.Close()

			err = hipRunAttentionHeadsBatchCausalOutputFromDeviceQueryToDeviceKernelWorkspace(context.Background(), driver, hipAttentionHeadsBatchCausalDeviceRequest{
				Key:             keys,
				Value:           values,
				Dim:             dim,
				TokenCount:      tokenCount,
				HeadCount:       tc.headCount,
				QueryCount:      queryCount,
				QueryStartToken: tokenCount - 1,
				Scale:           1,
			}, query, output, workspace)
			core.RequireNoError(t, err)
			launch := driver.launches[len(driver.launches)-1]
			weightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(launch.Args[40:]))
			core.AssertEqual(t, uint32(queryCount*tc.headCount*tokenCount*4), binary.LittleEndian.Uint32(launch.Args[84:]))
			if tc.wantWorkspace {
				if workspace.BatchAttentionWeight == nil || workspace.BatchAttentionWeight.Pointer() != weightPointer {
					t.Fatalf("workspace weight pointer = %#v, launch pointer %x", workspace.BatchAttentionWeight, weightPointer)
				}
				core.AssertEqual(t, 0, len(driver.frees))
				return
			}
			if workspace.BatchAttentionWeight != nil {
				t.Fatalf("workspace retained over-cap attention weights")
			}
			foundFree := false
			for _, freed := range driver.frees {
				if freed == weightPointer {
					foundFree = true
					break
				}
			}
			if !foundFree {
				t.Fatalf("over-cap attention weights %x were not released", weightPointer)
			}
		})
	}
}

func TestHIPAttentionHeadsChunkedWorkspace_BatchAttentionWeightsReused_Good(t *testing.T) {
	t.Setenv("GO_ROCM_DISABLE_DEVICE_BUFFER_POOL", "1")
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	first, err := workspace.EnsureBatchAttentionWeights(driver, 4096)
	core.RequireNoError(t, err)
	if first == nil || first.Count() != 4096 || first.SizeBytes() != 16384 {
		t.Fatalf("batch attention weights = %#v, want 4096/16384", first)
	}
	firstPointer := first.Pointer()
	core.AssertEqual(t, 1, len(driver.allocations))

	smaller, err := workspace.EnsureBatchAttentionWeights(driver, 2048)
	core.RequireNoError(t, err)
	if smaller.Pointer() != firstPointer || smaller.Count() != 4096 {
		t.Fatalf("smaller weights reused pointer/count = %#v, want pointer %x count 4096", smaller, firstPointer)
	}
	core.AssertEqual(t, 1, len(driver.allocations))
	core.AssertEqual(t, 0, len(driver.frees))

	larger, err := workspace.EnsureBatchAttentionWeights(driver, 8192)
	core.RequireNoError(t, err)
	if larger == nil || larger.Pointer() == firstPointer || larger.Count() != 8192 || larger.SizeBytes() != 32768 {
		t.Fatalf("larger weights = %#v, want fresh 8192/32768 buffer", larger)
	}
	core.AssertEqual(t, 2, len(driver.allocations))
	core.AssertEqual(t, 1, len(driver.frees))
	core.AssertEqual(t, firstPointer, driver.frees[0])
}

func BenchmarkHIPAttentionHeadsChunkedWorkspace_BatchAttentionWeightsReused(b *testing.B) {
	b.Setenv("GO_ROCM_DISABLE_DEVICE_BUFFER_POOL", "1")
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	output, err := workspace.EnsureBatchAttentionWeights(driver, 4096)
	if err != nil {
		b.Fatal(err)
	}
	if output.Count() != 4096 || output.SizeBytes() != 16384 {
		b.Fatalf("batch attention weight shape = %d/%d, want 4096/16384", output.Count(), output.SizeBytes())
	}
	pointer := output.Pointer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		output, err = workspace.EnsureBatchAttentionWeights(driver, 2048)
		if err != nil {
			b.Fatal(err)
		}
		if output.Pointer() != pointer || output.Count() != 4096 || output.SizeBytes() != 16384 {
			b.Fatalf("batch attention weight shape = %x %d/%d, want %x 4096/16384", output.Pointer(), output.Count(), output.SizeBytes(), pointer)
		}
	}
}

func BenchmarkHIPAttentionHeadsBatchChunkedLaunchArgs_FullWindow(b *testing.B) {
	args := benchmarkHIPAttentionHeadsBatchChunkedLaunchArgs()
	packet, err := args.Binary()
	if err != nil {
		b.Fatal(err)
	}
	hipReleaseLaunchPacket(packet)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		packet, err = args.Binary()
		if err != nil {
			b.Fatal(err)
		}
		hipReleaseLaunchPacket(packet)
	}
}

func BenchmarkHIPAttentionHeadsBatchCausalLaunchArgsBinaryInto_FullWindow(b *testing.B) {
	args := benchmarkHIPAttentionHeadsBatchCausalLaunchArgs()
	var scratch [hipAttentionHeadsBatchCausalLaunchArgsBytes]byte
	packet, err := args.BinaryInto(scratch[:])
	if err != nil {
		b.Fatal(err)
	}
	if len(packet) != hipAttentionHeadsBatchCausalLaunchArgsBytes {
		b.Fatalf("packet len = %d, want %d", len(packet), hipAttentionHeadsBatchCausalLaunchArgsBytes)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		packet, err = args.BinaryInto(scratch[:])
		if err != nil {
			b.Fatal(err)
		}
		if len(packet) != hipAttentionHeadsBatchCausalLaunchArgsBytes {
			b.Fatalf("packet len = %d, want %d", len(packet), hipAttentionHeadsBatchCausalLaunchArgsBytes)
		}
	}
}

func BenchmarkHIPAttentionHeadsChunkedLaunchArgsBinaryInto_FullWindow(b *testing.B) {
	args := benchmarkHIPAttentionHeadsChunkedLaunchArgs()
	var scratch [hipAttentionHeadsChunkedLaunchArgsBytes]byte
	packet, err := args.BinaryInto(scratch[:])
	if err != nil {
		b.Fatal(err)
	}
	if len(packet) != hipAttentionHeadsChunkedLaunchArgsBytes {
		b.Fatalf("packet len = %d, want %d", len(packet), hipAttentionHeadsChunkedLaunchArgsBytes)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		packet, err = args.BinaryInto(scratch[:])
		if err != nil {
			b.Fatal(err)
		}
		if len(packet) != hipAttentionHeadsChunkedLaunchArgsBytes {
			b.Fatalf("packet len = %d, want %d", len(packet), hipAttentionHeadsChunkedLaunchArgsBytes)
		}
	}
}

func BenchmarkHIPAttentionHeadsBatchChunkedLaunchArgsBinaryInto_FullWindow(b *testing.B) {
	args := benchmarkHIPAttentionHeadsBatchChunkedLaunchArgs()
	var scratch [hipAttentionHeadsBatchChunkedLaunchArgsBytes]byte
	packet, err := args.BinaryInto(scratch[:])
	if err != nil {
		b.Fatal(err)
	}
	if len(packet) != hipAttentionHeadsBatchChunkedLaunchArgsBytes {
		b.Fatalf("packet len = %d, want %d", len(packet), hipAttentionHeadsBatchChunkedLaunchArgsBytes)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		packet, err = args.BinaryInto(scratch[:])
		if err != nil {
			b.Fatal(err)
		}
		if len(packet) != hipAttentionHeadsBatchChunkedLaunchArgsBytes {
			b.Fatalf("packet len = %d, want %d", len(packet), hipAttentionHeadsBatchChunkedLaunchArgsBytes)
		}
	}
}

func benchmarkHIPAttentionHeadsBatchCausalLaunchArgs() hipAttentionHeadsBatchCausalLaunchArgs {
	const (
		dim        = 256
		tokenCount = 4096
		headCount  = 8
		queryCount = 16
	)
	queryElements := dim * headCount * queryCount
	return hipAttentionHeadsBatchCausalLaunchArgs{
		QueryPointer:      1,
		DescriptorPointer: 2,
		OutputPointer:     3,
		WeightPointer:     4,
		Dim:               dim,
		TokenCount:        tokenCount,
		HeadCount:         headCount,
		QueryCount:        queryCount,
		QueryStartToken:   tokenCount - queryCount,
		QueryBytes:        uint64(queryElements * 4),
		DescriptorBytes:   uint64(rocmDeviceKVDescriptorHeaderBytes + tokenCount*rocmDeviceKVDescriptorPageBytes),
		OutputBytes:       uint64(queryElements * 4),
		WeightBytes:       uint64(queryCount * headCount * tokenCount * 4),
		KVSource:          hipAttentionKVSourceDevice,
		Scale:             1,
	}
}

func benchmarkHIPAttentionHeadsChunkedLaunchArgs() hipAttentionHeadsChunkedLaunchArgs {
	const (
		dim        = 256
		tokenCount = 4096
		headCount  = 8
		chunkSize  = hipAttentionHeadsChunkSize
	)
	chunkCount := (tokenCount + chunkSize - 1) / chunkSize
	queryElements := dim * headCount
	return hipAttentionHeadsChunkedLaunchArgs{
		QueryPointer:      1,
		DescriptorPointer: 2,
		PartialPointer:    3,
		StatsPointer:      4,
		OutputPointer:     5,
		Dim:               dim,
		TokenCount:        tokenCount,
		HeadCount:         headCount,
		ChunkSize:         chunkSize,
		ChunkCount:        chunkCount,
		QueryBytes:        uint64(queryElements * 4),
		DescriptorBytes:   uint64(rocmDeviceKVDescriptorHeaderBytes + tokenCount*rocmDeviceKVDescriptorPageBytes),
		PartialBytes:      uint64(queryElements * chunkCount * 4),
		StatsBytes:        uint64(headCount * chunkCount * 2 * 4),
		OutputBytes:       uint64(queryElements * 4),
		Scale:             1,
	}
}

func benchmarkHIPAttentionHeadsBatchChunkedLaunchArgs() hipAttentionHeadsBatchChunkedLaunchArgs {
	const (
		dim        = 256
		tokenCount = 4096
		headCount  = 8
		queryCount = 16
		chunkSize  = hipAttentionHeadsChunkSize
	)
	chunkCount := (tokenCount + chunkSize - 1) / chunkSize
	queryElements := dim * headCount * queryCount
	return hipAttentionHeadsBatchChunkedLaunchArgs{
		QueryPointer:      1,
		DescriptorPointer: 2,
		PartialPointer:    3,
		StatsPointer:      4,
		OutputPointer:     5,
		Dim:               dim,
		TokenCount:        tokenCount,
		HeadCount:         headCount,
		QueryCount:        queryCount,
		QueryStartToken:   tokenCount - queryCount,
		ChunkSize:         chunkSize,
		ChunkCount:        chunkCount,
		QueryBytes:        uint64(queryElements * 4),
		DescriptorBytes:   uint64(rocmDeviceKVDescriptorHeaderBytes + tokenCount*rocmDeviceKVDescriptorPageBytes),
		PartialBytes:      uint64(queryElements * chunkCount * 4),
		StatsBytes:        uint64(queryCount * headCount * chunkCount * 2 * 4),
		OutputBytes:       uint64(queryElements * 4),
		Scale:             1,
	}
}

func BenchmarkHIPAttentionHeadsChunkedWorkspace_ProjectionOutputReused(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	output, err := workspace.EnsureProjectionOutput(driver, 2304)
	if err != nil {
		b.Fatal(err)
	}
	if output.Count() != 2304 || output.SizeBytes() != 9216 {
		b.Fatalf("projection output shape = %d/%d, want 2304/9216", output.Count(), output.SizeBytes())
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		output, err = workspace.EnsureProjectionOutput(driver, 2304)
		if err != nil {
			b.Fatal(err)
		}
		if output.Count() != 2304 || output.SizeBytes() != 9216 {
			b.Fatalf("projection output shape = %d/%d, want 2304/9216", output.Count(), output.SizeBytes())
		}
	}
}

func TestHIPAttentionHeadsChunkedWorkspace_ProjectionOutputReusesLarger_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	large, err := workspace.EnsureProjectionOutput(driver, 8192)
	core.RequireNoError(t, err)
	largePointer := large.Pointer()
	largeCount := large.Count()
	largeSize := large.SizeBytes()
	small, err := workspace.EnsureProjectionOutput(driver, 4096)
	core.RequireNoError(t, err)

	if large == nil || largeCount != 8192 || largeSize != uint64(8192*4) {
		t.Fatalf("large projection output shape = %#v, want 8192 floats", large)
	}
	if small == nil || small.Count() != 4096 || small.SizeBytes() != uint64(4096*4) {
		t.Fatalf("small projection output shape = %#v, want exact borrowed view", small)
	}
	if small.Pointer() != largePointer {
		t.Fatalf("small projection output pointer = %x, want reuse of %x", small.Pointer(), largePointer)
	}
	core.AssertEqual(t, 1, len(driver.allocations))
	if _, ok := workspace.ProjectionOutputs[4096]; ok {
		t.Fatalf("smaller projection output got its own allocation")
	}
}

func TestHIPAttentionHeadsChunkedWorkspace_ProjectionScoreOutputReused_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	first, err := workspace.EnsureProjectionScoreOutput(driver, 262144)
	core.RequireNoError(t, err)
	if first == nil || first.Count() != 262144 || first.SizeBytes() != uint64(262144*hipMLXQ4ProjectionBestBytes) {
		t.Fatalf("projection score output = %#v, want vocab-sized packed score buffer", first)
	}
	firstPointer := first.Pointer()
	core.AssertEqual(t, 1, len(driver.allocations))

	second, err := workspace.EnsureProjectionScoreOutput(driver, 262144)
	core.RequireNoError(t, err)
	if second.Pointer() != firstPointer {
		t.Fatalf("projection score pointer = %x, want reused %x", second.Pointer(), firstPointer)
	}
	core.AssertEqual(t, 1, len(driver.allocations))
	core.AssertEqual(t, 0, len(driver.frees))
}

func TestHIPAttentionHeadsChunkedWorkspace_ProjectionGreedyBestSlots_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	first, err := workspace.BorrowProjectionGreedyBest(driver)
	core.RequireNoError(t, err)
	firstPointer := first.Pointer()
	second, err := workspace.BorrowProjectionGreedyBest(driver)
	core.RequireNoError(t, err)
	secondPointer := second.Pointer()
	third, err := workspace.BorrowProjectionGreedyBest(driver)
	core.RequireNoError(t, err)
	if first == nil || first.Count() != 1 || first.SizeBytes() != hipMLXQ4ProjectionBestBytes {
		t.Fatalf("first greedy best slot = %#v, want one packed result", first)
	}
	if second == nil || second.Count() != 1 || second.SizeBytes() != hipMLXQ4ProjectionBestBytes {
		t.Fatalf("second greedy best slot = %#v, want one packed result", second)
	}
	if third == nil || third.Count() != 1 || third.SizeBytes() != hipMLXQ4ProjectionBestBytes {
		t.Fatalf("third greedy best slot = %#v, want one packed result", third)
	}
	if secondPointer != firstPointer+nativeDevicePointer(hipMLXQ4ProjectionBestBytes) {
		t.Fatalf("second greedy best slot pointer = %x, want first+%d", secondPointer, hipMLXQ4ProjectionBestBytes)
	}
	if third.Pointer() != secondPointer+nativeDevicePointer(hipMLXQ4ProjectionBestBytes) {
		t.Fatalf("third greedy best slot pointer = %x, want second+%d", third.Pointer(), hipMLXQ4ProjectionBestBytes)
	}
	core.AssertEqual(t, 1, len(driver.allocations))
	core.AssertEqual(t, []uint64{uint64(hipProjectionGreedyBestWorkspaceSlots * hipMLXQ4ProjectionBestBytes)}, driver.memsets)
}

func TestHIPAttentionHeadsChunkedWorkspace_ProjectionGreedyBestBatchSlots_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	batch, err := workspace.BorrowProjectionGreedyBestBatch(driver, 3)
	core.RequireNoError(t, err)
	batchView := hipCloneDeviceByteBufferView(batch)
	next, err := workspace.BorrowProjectionGreedyBest(driver)
	core.RequireNoError(t, err)

	if batchView.Count() != 3 || batchView.SizeBytes() != uint64(3*hipMLXQ4ProjectionBestBytes) {
		t.Fatalf("batch greedy best slots = %#v, want three packed results", batchView)
	}
	if next.Pointer() != batchView.Pointer()+nativeDevicePointer(3*hipMLXQ4ProjectionBestBytes) {
		t.Fatalf("next greedy best slot pointer = %x, want batch+%d", next.Pointer(), 3*hipMLXQ4ProjectionBestBytes)
	}
	core.AssertEqual(t, 1, len(driver.allocations))
	core.AssertEqual(t, []uint64{uint64(hipProjectionGreedyBestWorkspaceSlots * hipMLXQ4ProjectionBestBytes)}, driver.memsets)
}

func TestHIPAttentionHeadsChunkedWorkspace_ProjectionGreedyBestClonePreservesBorrowedSlot_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	first, err := workspace.BorrowProjectionGreedyBest(driver)
	core.RequireNoError(t, err)
	firstPointer := first.Pointer()
	firstClone := hipCloneDeviceByteBufferView(first)
	second, err := workspace.BorrowProjectionGreedyBest(driver)
	core.RequireNoError(t, err)

	if first.Pointer() != second.Pointer() {
		t.Fatalf("borrowed greedy view pointer = %x, want reused latest view %x", first.Pointer(), second.Pointer())
	}
	if firstClone.Pointer() != firstPointer {
		t.Fatalf("cloned greedy view pointer = %x, want original slot %x", firstClone.Pointer(), firstPointer)
	}
	if firstClone.Pointer() == second.Pointer() {
		t.Fatalf("cloned greedy view aliased second borrow pointer %x", second.Pointer())
	}
}

func TestHIPAttentionHeadsChunkedWorkspace_GreedyBestTokenBufferView_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	greedy := hipBorrowDeviceByteBuffer(driver, "greedy result", 0x2000, hipMLXQ4ProjectionBestBytes, 1)

	tokens, err := hipGreedyBestTokenBufferView(greedy)
	core.RequireNoError(t, err)

	if tokens.Pointer() != greedy.Pointer() {
		t.Fatalf("token buffer pointer = %x, want greedy pointer %x", tokens.Pointer(), greedy.Pointer())
	}
	core.AssertEqual(t, 1, tokens.Count())
	core.AssertEqual(t, uint64(hipMLXQ4ProjectionBestBytes), tokens.SizeBytes())
	core.AssertEqual(t, true, tokens.borrowed)
	core.RequireNoError(t, tokens.Close())
	if greedy.Pointer() == 0 {
		t.Fatalf("closing borrowed token view closed greedy buffer")
	}
}

func TestHIPAttentionHeadsChunkedWorkspace_GreedyBestTokenBufferView_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	greedy := hipBorrowDeviceByteBuffer(driver, "greedy result", 0x2000, 4, 1)

	_, err := hipGreedyBestTokenBufferView(greedy)
	if err == nil {
		t.Fatalf("hipGreedyBestTokenBufferView accepted malformed greedy result")
	}
}

func TestHIPAttentionHeadsChunkedWorkspace_BatchDeviceGreedyUnrollEnabled_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	current := hipGemma4Q4ForwardResult{
		GreedyDevice: hipBorrowDeviceByteBuffer(driver, "greedy result", 0x2000, hipMLXQ4ProjectionBestBytes, 1),
	}

	enabled := hipGemma4Q4BatchDeviceGreedyUnrollEnabled(inference.GenerateConfig{MaxTokens: 8}, false, false, false, workspace, current)

	core.AssertEqual(t, true, enabled)
}

func TestHIPAttentionHeadsChunkedWorkspace_BatchDeviceGreedyUnrollEnabled_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	current := hipGemma4Q4ForwardResult{
		GreedyDevice: hipBorrowDeviceByteBuffer(driver, "greedy result", 0x2000, hipMLXQ4ProjectionBestBytes, 1),
	}

	core.AssertEqual(t, false, hipGemma4Q4BatchDeviceGreedyUnrollEnabled(inference.GenerateConfig{MaxTokens: 1}, false, false, false, workspace, current))
	core.AssertEqual(t, false, hipGemma4Q4BatchDeviceGreedyUnrollEnabled(inference.GenerateConfig{MaxTokens: 8, StopTokens: []int32{1}}, false, false, false, workspace, current))
	core.AssertEqual(t, false, hipGemma4Q4BatchDeviceGreedyUnrollEnabled(inference.GenerateConfig{MaxTokens: 8}, true, false, false, workspace, current))
	core.AssertEqual(t, false, hipGemma4Q4BatchDeviceGreedyUnrollEnabled(inference.GenerateConfig{MaxTokens: 8}, false, true, false, workspace, current))
	core.AssertEqual(t, false, hipGemma4Q4BatchDeviceGreedyUnrollEnabled(inference.GenerateConfig{MaxTokens: 8}, false, false, true, workspace, current))
	core.AssertEqual(t, false, hipGemma4Q4BatchDeviceGreedyUnrollEnabled(inference.GenerateConfig{MaxTokens: 8}, false, false, false, nil, current))
	core.AssertEqual(t, false, hipGemma4Q4BatchDeviceGreedyUnrollEnabled(inference.GenerateConfig{MaxTokens: 8}, false, false, false, workspace, hipGemma4Q4ForwardResult{}))
}

func TestHIPAttentionHeadsChunkedWorkspace_ProjectionGreedyBestUsesFullLaterSlabs_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	var lastFirstSlab *hipDeviceByteBuffer
	for index := 0; index < hipProjectionGreedyBestWorkspaceUseSlots; index++ {
		var err error
		lastFirstSlab, err = workspace.BorrowProjectionGreedyBest(driver)
		core.RequireNoError(t, err)
	}
	firstSlab := workspace.ProjectionGreedyBest[0]
	core.AssertEqual(t, firstSlab.Pointer()+nativeDevicePointer((hipProjectionGreedyBestWorkspaceUseSlots-1)*hipMLXQ4ProjectionBestBytes), lastFirstSlab.Pointer())
	if lastFirstSlab.Pointer()+nativeDevicePointer(lastFirstSlab.SizeBytes()) > firstSlab.Pointer()+nativeDevicePointer(hipProjectionGreedyPrefillReserveOffsetBytes) {
		t.Fatalf("first slab greedy slots overlapped reserve tail")
	}

	next, err := workspace.BorrowProjectionGreedyBest(driver)
	core.RequireNoError(t, err)
	if len(workspace.ProjectionGreedyBest) != 2 {
		t.Fatalf("greedy slabs = %d, want second slab after first reserved region fills", len(workspace.ProjectionGreedyBest))
	}
	core.AssertEqual(t, workspace.ProjectionGreedyBest[1].Pointer(), next.Pointer())
	core.AssertEqual(t, 2, len(driver.allocations))
}

func TestHIPAttentionHeadsChunkedWorkspace_ProjectionGreedyBestDynamicFirstSlab_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	workspace.EnsureProjectionGreedyBestCapacity(2049)
	wantSlots := hipProjectionGreedyRoundFirstSlabSlots(hipProjectionGreedyReserveSlots + 2049)
	wantUseSlots := wantSlots - hipProjectionGreedyReserveSlots

	var lastFirstSlab *hipDeviceByteBuffer
	for index := 0; index < wantUseSlots; index++ {
		var err error
		lastFirstSlab, err = workspace.BorrowProjectionGreedyBest(driver)
		core.RequireNoError(t, err)
	}
	firstSlab := workspace.ProjectionGreedyBest[0]
	core.AssertEqual(t, uint64(wantSlots*hipMLXQ4ProjectionBestBytes), driver.allocations[0])
	core.AssertEqual(t, firstSlab.Pointer()+nativeDevicePointer((wantUseSlots-1)*hipMLXQ4ProjectionBestBytes), lastFirstSlab.Pointer())
	if lastFirstSlab.Pointer()+nativeDevicePointer(lastFirstSlab.SizeBytes()) > firstSlab.Pointer()+nativeDevicePointer(workspace.projectionGreedyPrefillReserveOffsetBytes()) {
		t.Fatalf("dynamic first slab greedy slots overlapped reserve tail")
	}

	next, err := workspace.BorrowProjectionGreedyBest(driver)
	core.RequireNoError(t, err)
	if len(workspace.ProjectionGreedyBest) != 2 {
		t.Fatalf("greedy slabs = %d, want second slab after dynamic first slab fills", len(workspace.ProjectionGreedyBest))
	}
	core.AssertEqual(t, workspace.ProjectionGreedyBest[1].Pointer(), next.Pointer())
	core.AssertEqual(t, uint64(hipProjectionGreedyBestWorkspaceSlots*hipMLXQ4ProjectionBestBytes), driver.allocations[1])
}

func TestHIPAttentionHeadsChunkedWorkspace_ProjectionGreedyBestReusedDynamicFirstSlabKeepsReserveOffsets_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	workspace.EnsureProjectionGreedyBestCapacity(18)
	_, err := workspace.BorrowProjectionGreedyBest(driver)
	core.RequireNoError(t, err)
	firstSlab := workspace.ProjectionGreedyBest[0]
	firstSlabSlots := int(firstSlab.SizeBytes() / hipMLXQ4ProjectionBestBytes)
	if firstSlabSlots >= hipProjectionGreedyBestWorkspaceSlots {
		t.Fatalf("first slab slots = %d, want dynamic slab smaller than default %d", firstSlabSlots, hipProjectionGreedyBestWorkspaceSlots)
	}

	workspace.GreedyFirstSlabSlots = 0
	workspace.ProjectionGreedyView = hipDeviceByteBuffer{}
	workspace.ProjectionGreedyNext = 0
	suppress, err := workspace.EnsureSuppressTokenBuffer(driver, []int32{0, 2, 105, 106})
	core.RequireNoError(t, err)
	wantSuppress := firstSlab.Pointer() + nativeDevicePointer((firstSlabSlots-hipProjectionGreedySuppressReserveSlots)*hipMLXQ4ProjectionBestBytes)
	core.AssertEqual(t, wantSuppress, suppress.Pointer())
	if _, _, ok := driver.memoryForPointer(suppress.Pointer(), int(suppress.SizeBytes())); !ok {
		t.Fatalf("reused dynamic slab suppress pointer %x/%d fell outside first slab %x/%d", suppress.Pointer(), suppress.SizeBytes(), firstSlab.Pointer(), firstSlab.SizeBytes())
	}

	prefill, err := workspace.EnsurePrefillTokenBuffer(driver, []int32{11, 12, 13, 14})
	core.RequireNoError(t, err)
	wantPrefill := firstSlab.Pointer() + nativeDevicePointer((firstSlabSlots-hipProjectionGreedyReserveSlots)*hipMLXQ4ProjectionBestBytes)
	core.AssertEqual(t, wantPrefill, prefill.Pointer())
	if prefill.Pointer()+nativeDevicePointer(prefill.SizeBytes()) > suppress.Pointer() {
		t.Fatalf("reused dynamic slab prefill reserve overlapped suppress reserve: prefill=%x/%d suppress=%x", prefill.Pointer(), prefill.SizeBytes(), suppress.Pointer())
	}
}

func BenchmarkHIPAttentionHeadsChunkedWorkspace_ProjectionScoreOutputReused(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	output, err := workspace.EnsureProjectionScoreOutput(driver, 262144)
	if err != nil {
		b.Fatal(err)
	}
	if output.Count() != 262144 || output.SizeBytes() != uint64(262144*hipMLXQ4ProjectionBestBytes) {
		b.Fatalf("projection score output shape = %d/%d, want 262144/%d", output.Count(), output.SizeBytes(), 262144*hipMLXQ4ProjectionBestBytes)
	}
	pointer := output.Pointer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		output, err = workspace.EnsureProjectionScoreOutput(driver, 262144)
		if err != nil {
			b.Fatal(err)
		}
		if output.Pointer() != pointer || output.Count() != 262144 || output.SizeBytes() != uint64(262144*hipMLXQ4ProjectionBestBytes) {
			b.Fatalf("projection score output shape = %x %d/%d, want %x 262144/%d", output.Pointer(), output.Count(), output.SizeBytes(), pointer, 262144*hipMLXQ4ProjectionBestBytes)
		}
	}
}

func BenchmarkHIPAttentionHeadsChunkedWorkspace_ActivationOutputReused(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	output, err := workspace.EnsureActivationOutput(driver, 9216)
	if err != nil {
		b.Fatal(err)
	}
	if output.Count() != 9216 || output.SizeBytes() != 36864 {
		b.Fatalf("activation output shape = %d/%d, want 9216/36864", output.Count(), output.SizeBytes())
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		output, err = workspace.EnsureActivationOutput(driver, 9216)
		if err != nil {
			b.Fatal(err)
		}
		if output.Count() != 9216 || output.SizeBytes() != 36864 {
			b.Fatalf("activation output shape = %d/%d, want 9216/36864", output.Count(), output.SizeBytes())
		}
	}
}

func BenchmarkHIPAttentionHeadsChunkedWorkspace_RMSOutputsReused(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	residualOutput, err := workspace.EnsureRMSResidualOutput(driver, 2304)
	if err != nil {
		b.Fatal(err)
	}
	normOutput, err := workspace.EnsureRMSNormOutput(driver, 2304)
	if err != nil {
		b.Fatal(err)
	}
	if residualOutput.Count() != 2304 || residualOutput.SizeBytes() != 9216 {
		b.Fatalf("RMS residual output shape = %d/%d, want 2304/9216", residualOutput.Count(), residualOutput.SizeBytes())
	}
	if normOutput.Count() != 2304 || normOutput.SizeBytes() != 9216 {
		b.Fatalf("RMS norm output shape = %d/%d, want 2304/9216", normOutput.Count(), normOutput.SizeBytes())
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		residualOutput, err = workspace.EnsureRMSResidualOutput(driver, 2304)
		if err != nil {
			b.Fatal(err)
		}
		normOutput, err = workspace.EnsureRMSNormOutput(driver, 2304)
		if err != nil {
			b.Fatal(err)
		}
		if residualOutput.Count() != 2304 || residualOutput.SizeBytes() != 9216 {
			b.Fatalf("RMS residual output shape = %d/%d, want 2304/9216", residualOutput.Count(), residualOutput.SizeBytes())
		}
		if normOutput.Count() != 2304 || normOutput.SizeBytes() != 9216 {
			b.Fatalf("RMS norm output shape = %d/%d, want 2304/9216", normOutput.Count(), normOutput.SizeBytes())
		}
	}
}

func BenchmarkHIPAttentionHeadsChunkedWorkspace_RMSRoPEOutputsReused(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	queryOutput, err := workspace.EnsureRMSRoPEOutput(driver, 2048)
	if err != nil {
		b.Fatal(err)
	}
	keyOutput, err := workspace.EnsureKeyRMSRoPEOutput(driver, 256)
	if err != nil {
		b.Fatal(err)
	}
	noScaleOutput, err := workspace.EnsureRMSNoScaleOutput(driver, 256)
	if err != nil {
		b.Fatal(err)
	}
	if queryOutput.Count() != 2048 || queryOutput.SizeBytes() != 8192 {
		b.Fatalf("RMS RoPE query output shape = %d/%d, want 2048/8192", queryOutput.Count(), queryOutput.SizeBytes())
	}
	if keyOutput.Count() != 256 || keyOutput.SizeBytes() != 1024 {
		b.Fatalf("RMS RoPE key output shape = %d/%d, want 256/1024", keyOutput.Count(), keyOutput.SizeBytes())
	}
	if noScaleOutput.Count() != 256 || noScaleOutput.SizeBytes() != 1024 {
		b.Fatalf("RMS no-scale output shape = %d/%d, want 256/1024", noScaleOutput.Count(), noScaleOutput.SizeBytes())
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		queryOutput, err = workspace.EnsureRMSRoPEOutput(driver, 2048)
		if err != nil {
			b.Fatal(err)
		}
		keyOutput, err = workspace.EnsureKeyRMSRoPEOutput(driver, 256)
		if err != nil {
			b.Fatal(err)
		}
		noScaleOutput, err = workspace.EnsureRMSNoScaleOutput(driver, 256)
		if err != nil {
			b.Fatal(err)
		}
		if queryOutput.Count() != 2048 || queryOutput.SizeBytes() != 8192 {
			b.Fatalf("RMS RoPE query output shape = %d/%d, want 2048/8192", queryOutput.Count(), queryOutput.SizeBytes())
		}
		if keyOutput.Count() != 256 || keyOutput.SizeBytes() != 1024 {
			b.Fatalf("RMS RoPE key output shape = %d/%d, want 256/1024", keyOutput.Count(), keyOutput.SizeBytes())
		}
		if noScaleOutput.Count() != 256 || noScaleOutput.SizeBytes() != 1024 {
			b.Fatalf("RMS no-scale output shape = %d/%d, want 256/1024", noScaleOutput.Count(), noScaleOutput.SizeBytes())
		}
	}
}

func TestHIPAttentionHeadsChunkedWorkspace_RMSRoPEOutputReusesLarger_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	large, err := workspace.EnsureRMSRoPEOutput(driver, 2048)
	core.RequireNoError(t, err)
	small, err := workspace.EnsureRMSRoPEOutput(driver, 1024)
	core.RequireNoError(t, err)

	if small.Pointer() != large.Pointer() || small.Count() != 1024 {
		t.Fatalf("small RMS RoPE view = %#v, want borrowed view of %x", small, large.Pointer())
	}
	core.AssertEqual(t, 1, len(driver.allocations))
	if _, ok := workspace.RMSRoPEOutputs[1024]; ok {
		t.Fatalf("smaller RMS RoPE output got its own allocation")
	}
}

func BenchmarkHIPAttentionHeadsChunkedWorkspace_IntermediateOutputReused(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	output, err := workspace.EnsureIntermediateOutput(driver, 2304)
	if err != nil {
		b.Fatal(err)
	}
	if output.Count() != 2304 || output.SizeBytes() != 9216 {
		b.Fatalf("intermediate output shape = %d/%d, want 2304/9216", output.Count(), output.SizeBytes())
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		output, err = workspace.EnsureIntermediateOutput(driver, 2304)
		if err != nil {
			b.Fatal(err)
		}
		if output.Count() != 2304 || output.SizeBytes() != 9216 {
			b.Fatalf("intermediate output shape = %d/%d, want 2304/9216", output.Count(), output.SizeBytes())
		}
	}
}

func TestHIPAttentionHeadsChunkedWorkspace_IntermediateOutputReusesLarger_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	large, err := workspace.EnsureIntermediateOutput(driver, 3072)
	core.RequireNoError(t, err)
	small, err := workspace.EnsureIntermediateOutput(driver, 1536)
	core.RequireNoError(t, err)

	if small.Pointer() != large.Pointer() || small.Count() != 1536 || small.SizeBytes() != uint64(1536*4) {
		t.Fatalf("small intermediate view = %#v, want borrowed view of %x", small, large.Pointer())
	}
	core.AssertEqual(t, 1, len(driver.allocations))
	if _, ok := workspace.IntermediateOutputs[1536]; ok {
		t.Fatalf("smaller intermediate output got its own allocation")
	}
}

func BenchmarkHIPAttentionHeadsChunkedWorkspace_QKVOutputReused(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	output, err := workspace.EnsureQKVOutput(driver, 2560)
	if err != nil {
		b.Fatal(err)
	}
	if output.Count() != 2560 || output.SizeBytes() != 10240 {
		b.Fatalf("QKV output shape = %d/%d, want 2560/10240", output.Count(), output.SizeBytes())
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		output, err = workspace.EnsureQKVOutput(driver, 2560)
		if err != nil {
			b.Fatal(err)
		}
		if output.Count() != 2560 || output.SizeBytes() != 10240 {
			b.Fatalf("QKV output shape = %d/%d, want 2560/10240", output.Count(), output.SizeBytes())
		}
	}
}

func TestHIPAttentionHeadsChunkedWorkspace_QKVOutputReusesLarger_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	large, err := workspace.EnsureQKVOutput(driver, 5120)
	core.RequireNoError(t, err)
	largePointer := large.Pointer()
	largeCount := large.Count()
	largeSize := large.SizeBytes()
	small, err := workspace.EnsureQKVOutput(driver, 2560)
	core.RequireNoError(t, err)

	if large == nil || largeCount != 5120 || largeSize != uint64(5120*4) {
		t.Fatalf("large QKV output shape = %#v, want 5120 floats", large)
	}
	if small == nil || small.Count() != 2560 || small.SizeBytes() != uint64(2560*4) {
		t.Fatalf("small QKV output shape = %#v, want exact borrowed view", small)
	}
	if small.Pointer() != largePointer {
		t.Fatalf("small QKV output pointer = %x, want reuse of %x", small.Pointer(), largePointer)
	}
	core.AssertEqual(t, 1, len(driver.allocations))
	if _, ok := workspace.QKVOutputs[2560]; ok {
		t.Fatalf("smaller QKV output got its own allocation")
	}
}

func TestHIPAttentionHeadsChunkedWorkspace_ActivationOutputReusesLarger_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	large, err := workspace.EnsureActivationOutput(driver, 24576)
	core.RequireNoError(t, err)
	largePointer := large.Pointer()
	largeCount := large.Count()
	largeSize := large.SizeBytes()
	small, err := workspace.EnsureActivationOutput(driver, 12288)
	core.RequireNoError(t, err)

	if large == nil || largeCount != 24576 || largeSize != uint64(24576*4) {
		t.Fatalf("large activation output shape = %#v, want 24576 floats", large)
	}
	if small == nil || small.Count() != 12288 || small.SizeBytes() != uint64(12288*4) {
		t.Fatalf("small activation output shape = %#v, want exact borrowed view", small)
	}
	if small.Pointer() != largePointer {
		t.Fatalf("small activation output pointer = %x, want reuse of %x", small.Pointer(), largePointer)
	}
	core.AssertEqual(t, 1, len(driver.allocations))
	if _, ok := workspace.ActivationOutputs[12288]; ok {
		t.Fatalf("smaller activation output got its own allocation")
	}
}

func TestHIPAttentionHeadsChunkedWorkspace_RMSOutputsReuseLarger_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	largeResidual, err := workspace.EnsureRMSResidualOutput(driver, 3072)
	core.RequireNoError(t, err)
	largeResidualPointer := largeResidual.Pointer()
	smallResidual, err := workspace.EnsureRMSResidualOutput(driver, 1536)
	core.RequireNoError(t, err)
	largeNorm, err := workspace.EnsureRMSNormOutput(driver, 3072)
	core.RequireNoError(t, err)
	largeNormPointer := largeNorm.Pointer()
	smallNorm, err := workspace.EnsureRMSNormOutput(driver, 1536)
	core.RequireNoError(t, err)

	if smallResidual.Pointer() != largeResidualPointer || smallResidual.Count() != 1536 {
		t.Fatalf("small RMS residual view = %#v, want borrowed view of %x", smallResidual, largeResidualPointer)
	}
	if smallNorm.Pointer() != largeNormPointer || smallNorm.Count() != 1536 {
		t.Fatalf("small RMS norm view = %#v, want borrowed view of %x", smallNorm, largeNormPointer)
	}
	rmsCapCount := 4096
	if largeNormPointer != largeResidualPointer+nativeDevicePointer(rmsCapCount*4) {
		t.Fatalf("RMS norm pointer = %x, want residual+%d", largeNormPointer, rmsCapCount*4)
	}
	core.AssertEqual(t, 1, len(driver.allocations))
	if _, ok := workspace.RMSResidualOutputs[1536]; ok {
		t.Fatalf("smaller RMS residual output got its own allocation")
	}
	if _, ok := workspace.RMSNormOutputs[1536]; ok {
		t.Fatalf("smaller RMS norm output got its own allocation")
	}
	if output := &workspace.RMSResidualNormFixed; output.Pointer() == 0 || output.Count() != rmsCapCount*2 {
		t.Fatalf("RMS residual/norm pair output = %#v, want %d rows", output, rmsCapCount*2)
	}
}

func TestHIPAttentionHeadsChunkedWorkspacePool_ReusesWorkspace_Good(t *testing.T) {
	workspace := hipBorrowAttentionHeadsChunkedWorkspace()
	if workspace == nil {
		t.Fatalf("borrowed workspace = %#v, want workspace", workspace)
	}
	workspacePointer := uintptr(unsafe.Pointer(workspace))
	hipReleaseAttentionHeadsChunkedWorkspace(workspace)

	reused := hipBorrowAttentionHeadsChunkedWorkspace()
	defer hipReleaseAttentionHeadsChunkedWorkspace(reused)
	if reused == nil || uintptr(unsafe.Pointer(reused)) != workspacePointer {
		t.Fatalf("reused workspace = %#v, want original workspace pointer %x", reused, workspacePointer)
	}
}

func TestHIPAttentionHeadsChunkedWorkspacePool_RecycleKeepsGreedySlab_Good(t *testing.T) {
	t.Setenv("GO_ROCM_DISABLE_DEVICE_BUFFER_POOL", "1")
	driver := &fakeHIPDriver{available: true}
	hipDrainAttentionHeadsChunkedWorkspacePoolForTest(t)

	workspace := hipBorrowAttentionHeadsChunkedWorkspace()
	first, err := workspace.BorrowProjectionGreedyBest(driver)
	core.RequireNoError(t, err)
	firstPointer := first.Pointer()
	core.AssertEqual(t, 1, len(driver.allocations))

	core.RequireNoError(t, hipRecycleAttentionHeadsChunkedWorkspace(workspace))
	core.AssertEqual(t, 0, len(driver.frees))

	reused := hipBorrowAttentionHeadsChunkedWorkspace()
	if reused == nil {
		t.Fatalf("borrowed nil workspace")
	}
	next, err := reused.BorrowProjectionGreedyBest(driver)
	core.RequireNoError(t, err)
	core.AssertEqual(t, firstPointer, next.Pointer())
	core.AssertEqual(t, 1, len(driver.allocations))
	core.AssertEqual(t, 0, len(driver.frees))

	core.RequireNoError(t, reused.Close())
	core.AssertEqual(t, 1, len(driver.frees))
	core.AssertEqual(t, firstPointer, driver.frees[0])
}

func TestHIPAttentionHeadsChunkedWorkspacePool_Good_PrewarmDeviceBuffers(t *testing.T) {
	t.Setenv("GO_ROCM_DISABLE_DEVICE_BUFFER_POOL", "1")
	driver := &fakeHIPDriver{available: true}
	hipDrainAttentionHeadsChunkedWorkspacePoolForTest(t)
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 8, 512, 512)
	defer cleanup0()
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0}}

	core.RequireNoError(t, hipPrewarmGemma4Q4AttentionWorkspaceDeviceBuffers(driver, cfg, 128))

	workspace := hipBorrowAttentionHeadsChunkedWorkspace()
	defer func() {
		core.RequireNoError(t, hipRecycleAttentionHeadsChunkedWorkspace(workspace))
	}()
	if workspace.Partial == nil || workspace.Partial.Pointer() == 0 {
		t.Fatal("prewarmed workspace partial buffer is missing")
	}
	if workspace.Stats == nil || workspace.Stats.Pointer() == 0 {
		t.Fatal("prewarmed workspace stats buffer is missing")
	}
	if len(workspace.ProjectionGreedyBest) == 0 || workspace.ProjectionGreedyBest[0] == nil || workspace.ProjectionGreedyBest[0].Pointer() == 0 {
		t.Fatal("prewarmed workspace greedy slab is missing")
	}
	if workspace.ProjectionGreedyNext != 0 || workspace.ProjectionGreedyView.Pointer() != 0 {
		t.Fatalf("prewarmed workspace retained borrowed greedy view state: next=%d view=%x", workspace.ProjectionGreedyNext, workspace.ProjectionGreedyView.Pointer())
	}
	if workspace.ActivationOutputFixed.Pointer() == 0 {
		t.Fatal("prewarmed workspace activation buffer is missing")
	}
	if workspace.TokenID == nil || workspace.TokenID.Pointer() == 0 {
		t.Fatal("prewarmed workspace token ID buffer is missing")
	}
	if workspace.ProjectionTopK == nil || workspace.ProjectionTopK.Pointer() == 0 {
		t.Fatal("prewarmed workspace projection top-k buffer is missing")
	}
	wantTopKCap := hipPackedTopKOutputCount(layer0.VocabSize, hipGemma4Q4AttentionWorkspacePrewarmTopK)
	wantTopKWorkCap := 0
	if wantTopKCap > hipGemma4Q4AttentionWorkspacePrewarmTopK {
		wantTopKWorkCap = hipPackedTopKOutputCount(wantTopKCap, hipGemma4Q4AttentionWorkspacePrewarmTopK)
	}
	if wantTopKWorkCap > 0 && (workspace.ProjectionTopKWork == nil || workspace.ProjectionTopKWork.Pointer() == 0) {
		t.Fatal("prewarmed workspace projection top-k work buffer is missing")
	}
	if workspace.ActivationOutputView.Pointer() != 0 ||
		workspace.ScaledEmbeddingView.Pointer() != 0 ||
		workspace.PrefillInputNormView.Pointer() != 0 ||
		workspace.IntermediateOutputView.Pointer() != 0 ||
		workspace.FinalHiddenOutputViews[0].Pointer() != 0 ||
		workspace.NextInputOutputViews[0].Pointer() != 0 {
		t.Fatal("prewarmed workspace retained borrowed fixed-buffer views")
	}
	if workspace.ScaledEmbeddingFixed.Pointer() == 0 ||
		workspace.PrefillInputNormFixed.Pointer() == 0 ||
		workspace.IntermediateFixed.Pointer() == 0 ||
		workspace.FinalHiddenPairFixed.Pointer() == 0 ||
		workspace.NextInputPairFixed.Pointer() == 0 {
		t.Fatal("prewarmed workspace decode hot buffers are missing")
	}
	if layer0.PerLayerInput.hasGlobalPrecompute() && workspace.PerLayerScaledFixed.Pointer() == 0 {
		t.Fatal("prewarmed workspace per-layer scaled buffer is missing")
	}
	chunkCount := (hipGemma4Q4AttentionWorkspacePrewarmTokenCount(128) + hipAttentionHeadsChunkSize - 1) / hipAttentionHeadsChunkSize
	minPartialCap := layer0.QueryHeads * chunkCount * layer0.HeadDim
	minStatsCap := layer0.QueryHeads * chunkCount * 2
	core.AssertTrue(t, workspace.partialCap >= minPartialCap, "partial cap should cover decode prewarm floor")
	core.AssertTrue(t, workspace.statsCap >= minStatsCap, "stats cap should cover decode prewarm floor")
	core.AssertEqual(t, uint64(24704), workspace.ProjectionGreedyBest[0].SizeBytes())
	core.AssertEqual(t, wantTopKCap, workspace.ProjectionTopKCap)
	core.AssertEqual(t, wantTopKWorkCap, workspace.ProjectionTopKWorkCap)
	core.AssertEqual(t, hipGemma4Q4PrefillDefaultUBatchTokens*layer0.GateProjection.Rows, workspace.ActivationOutputFixed.Count())
	core.AssertEqual(t, hipGemma4Q4PrefillDefaultUBatchTokens*layer0.HiddenSize, workspace.ScaledEmbeddingFixed.Count())
	core.AssertEqual(t, hipGemma4Q4PrefillDefaultUBatchTokens*layer0.HiddenSize*2, workspace.FinalHiddenPairFixed.Count())
	if layer0.PerLayerInput.hasGlobalPrecompute() {
		core.AssertEqual(t, layer0.PerLayerInput.ModelProjection.Rows, workspace.PerLayerScaledFixed.Count())
	}
	allocationsAfterPrewarm := len(driver.allocations)
	core.RequireNoError(t, hipGemma4Q4EnsureAttentionWorkspaceDecodeHotCapacity(driver, workspace, cfg))
	core.RequireNoError(t, hipGemma4Q4EnsureAttentionWorkspaceSamplingCapacity(driver, workspace, cfg, hipGemma4Q4AttentionWorkspacePrewarmTopK))
	core.AssertEqual(t, allocationsAfterPrewarm, len(driver.allocations))
}

func TestHIPAttentionHeadsChunkedWorkspacePool_Good_PrewarmRetainedPrefillContext(t *testing.T) {
	t.Setenv("GO_ROCM_DISABLE_DEVICE_BUFFER_POOL", "1")
	driver := &fakeHIPDriver{available: true}
	hipDrainAttentionHeadsChunkedWorkspacePoolForTest(t)
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 8, 512, 512)
	defer cleanup0()
	layer0.SlidingWindow = 0
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0}}

	const contextSize = 4096
	core.RequireNoError(t, hipPrewarmGemma4Q4AttentionWorkspaceDeviceBuffers(driver, cfg, contextSize))

	workspace := hipBorrowAttentionHeadsChunkedWorkspace()
	defer func() {
		core.RequireNoError(t, hipRecycleAttentionHeadsChunkedWorkspace(workspace))
	}()
	queryTokens := hipGemma4Q4PrefillDefaultUBatchTokens
	if attentionQueryTokens := hipGemma4Q4PrefillAttentionQueryChunkTokens(); attentionQueryTokens > 0 && queryTokens > attentionQueryTokens {
		queryTokens = attentionQueryTokens
	}
	chunkCount := (contextSize + hipGemma4Q4PrefillDefaultUBatchTokens + hipAttentionHeadsChunkSize - 1) / hipAttentionHeadsChunkSize
	minPartialCap := layer0.QueryHeads * queryTokens * chunkCount * layer0.HeadDim
	minStatsCap := layer0.QueryHeads * queryTokens * chunkCount * 2
	core.AssertTrue(t, workspace.partialCap >= minPartialCap, "partial cap should cover retained prefill context")
	core.AssertTrue(t, workspace.statsCap >= minStatsCap, "stats cap should cover retained prefill context")
}

func TestHIPAttentionHeadsChunkedWorkspacePool_Good_PrewarmDecodeHotUsesNormWidth(t *testing.T) {
	t.Setenv("GO_ROCM_DISABLE_DEVICE_BUFFER_POOL", "1")
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 8, 512, 512)
	defer cleanup0()
	layer0.PostFeedForwardNorm.Count = 16
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0}}
	workspace := hipBorrowAttentionHeadsChunkedWorkspace()
	defer func() {
		core.RequireNoError(t, hipRecycleAttentionHeadsChunkedWorkspace(workspace))
	}()

	core.RequireNoError(t, hipGemma4Q4EnsureAttentionWorkspaceDecodeHotCapacity(driver, workspace, cfg))

	core.AssertEqual(t, 32, workspace.ScaledEmbeddingFixed.Count())
	core.AssertEqual(t, 32, workspace.PrefillInputNormFixed.Count())
	core.AssertEqual(t, 32, workspace.IntermediateFixed.Count())
	core.AssertEqual(t, 64, workspace.FinalHiddenPairFixed.Count())
	core.AssertEqual(t, 64, workspace.NextInputPairFixed.Count())
}

func TestHIPAttentionHeadsChunkedWorkspacePool_Good_PrewarmModelHiddenBuffers(t *testing.T) {
	t.Setenv("GO_ROCM_DISABLE_DEVICE_BUFFER_POOL", "1")
	driver := &fakeHIPDriver{available: true}
	hipDrainAttentionHeadsChunkedWorkspacePoolForTest(t)

	core.RequireNoError(t, hipPrewarmGemma4Q4AttentionWorkspaceModelHiddenBuffers(driver, 16))

	workspace := hipBorrowAttentionHeadsChunkedWorkspace()
	defer func() {
		core.RequireNoError(t, hipRecycleAttentionHeadsChunkedWorkspace(workspace))
	}()
	core.AssertEqual(t, 16, workspace.ScaledEmbeddingFixed.Count())
	core.AssertEqual(t, 16, workspace.PrefillInputNormFixed.Count())
	core.AssertEqual(t, 16, workspace.IntermediateFixed.Count())
	core.AssertEqual(t, 32, workspace.FinalHiddenPairFixed.Count())
	core.AssertEqual(t, 32, workspace.NextInputPairFixed.Count())
	allocationsAfterPrewarm := len(driver.allocations)
	_, err := workspace.EnsureScaledEmbedding(driver, 16)
	core.RequireNoError(t, err)
	_, err = workspace.EnsurePrefillInputNormOutput(driver, 16)
	core.RequireNoError(t, err)
	_, err = workspace.EnsureIntermediateOutput(driver, 16)
	core.RequireNoError(t, err)
	_, err = workspace.EnsureFinalHiddenOutput(driver, 16, 0)
	core.RequireNoError(t, err)
	_, err = workspace.EnsureNextInputOutput(driver, 16, 0)
	core.RequireNoError(t, err)
	core.AssertEqual(t, allocationsAfterPrewarm, len(driver.allocations))
}

func TestHIPAttentionHeadsChunkedWorkspacePool_Good_PrewarmDefaultSuppressBuffer(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	hipDrainAttentionHeadsChunkedWorkspacePoolForTest(t)
	model := &hipLoadedModel{
		driver: driver,
		modelInfo: inference.ModelInfo{
			Architecture: "gemma4",
			QuantBits:    4,
		},
		modelLabels: linkedGemma4TestLabels("E2B", "q4"),
		tokenText: &hipTokenTextDecoder{
			specialText: map[string]int32{
				"<pad>":   0,
				"<bos>":   2,
				"<|turn>": 105,
				"<turn|>": 106,
			},
		},
	}
	tokens := hipGemma4Q4GenerationSuppressTokenIDs(model, nil)
	core.RequireTrue(t, len(tokens) > 0, "default suppress tokens should be present")

	hipPrewarmGemma4Q4DefaultSuppressTokenBufferForModel(model)
	copiesAfterPrewarm := len(driver.copies)
	core.RequireTrue(t, copiesAfterPrewarm > 0, "suppress prewarm should copy token IDs once")

	workspace := hipBorrowAttentionHeadsChunkedWorkspace()
	defer func() {
		core.RequireNoError(t, hipRecycleAttentionHeadsChunkedWorkspace(workspace))
	}()
	buffer, err := workspace.EnsureSuppressTokenBuffer(driver, tokens)
	core.RequireNoError(t, err)

	if buffer == nil || buffer.Pointer() == 0 {
		t.Fatalf("prewarmed suppress token buffer = %#v, want device buffer", buffer)
	}
	core.AssertEqual(t, copiesAfterPrewarm, len(driver.copies))
}

func TestHIPAttentionHeadsChunkedWorkspace_FixedReusableOutputCapacity_Good(t *testing.T) {
	t.Setenv("GO_ROCM_DISABLE_DEVICE_BUFFER_POOL", "1")
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	first, err := workspace.EnsureScaledEmbedding(driver, 2049)
	core.RequireNoError(t, err)
	firstPointer := first.Pointer()
	if first == nil || first.Count() != 2049 || first.SizeBytes() != 2049*4 {
		t.Fatalf("scaled embedding view = %#v, want 2049 float32 values", first)
	}
	core.AssertEqual(t, 4096, workspace.ScaledEmbeddingFixedCap)
	core.AssertEqual(t, 1, len(driver.allocations))

	smaller, err := workspace.EnsureScaledEmbedding(driver, 1536)
	core.RequireNoError(t, err)
	core.AssertEqual(t, firstPointer, smaller.Pointer())
	core.AssertEqual(t, 1536, smaller.Count())
	core.AssertEqual(t, 1, len(driver.allocations))
	core.AssertEqual(t, 0, len(driver.frees))

	core.RequireNoError(t, workspace.Close())
	core.AssertEqual(t, 1, len(driver.frees))
	core.AssertEqual(t, firstPointer, driver.frees[0])
}

func TestHIPAttentionHeadsChunkedWorkspace_FixedReusableOutputDriverMismatch_Good(t *testing.T) {
	t.Setenv("GO_ROCM_DISABLE_DEVICE_BUFFER_POOL", "1")
	firstDriver := &fakeHIPDriver{available: true}
	nextDriver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	first, err := workspace.EnsureScaledEmbedding(firstDriver, 8)
	core.RequireNoError(t, err)
	firstPointer := first.Pointer()
	core.AssertEqual(t, 1, len(firstDriver.allocations))

	next, err := workspace.EnsureScaledEmbedding(nextDriver, 8)
	core.RequireNoError(t, err)
	core.AssertTrue(t, next.driver == nextDriver, "driver mismatch must allocate a fresh fixed buffer owned by the next driver")
	core.AssertEqual(t, 1, len(firstDriver.frees))
	core.AssertEqual(t, firstPointer, firstDriver.frees[0])
	core.AssertEqual(t, 1, len(nextDriver.allocations))
}

func hipDrainAttentionHeadsChunkedWorkspacePoolForTest(t *testing.T) {
	t.Helper()
	hipAttentionHeadsChunkedWorkspacePool.Lock()
	pooled := append([]*hipAttentionHeadsChunkedWorkspace(nil), hipAttentionHeadsChunkedWorkspacePool.workspaces...)
	clear(hipAttentionHeadsChunkedWorkspacePool.workspaces)
	hipAttentionHeadsChunkedWorkspacePool.workspaces = hipAttentionHeadsChunkedWorkspacePool.workspaces[:0]
	hipAttentionHeadsChunkedWorkspacePool.Unlock()
	for _, workspace := range pooled {
		core.RequireNoError(t, workspace.Close())
	}
}

func TestHIPGemma4Q4PrefillForwardLayerBatchPool_ReusesCapacity_Good(t *testing.T) {
	hipPrewarmGemma4Q4PrefillForwardLayerBatchPool(35, 2)
	batch := hipBorrowGemma4Q4PrefillForwardBatch(35)
	if batch == nil || len(batch.Layers) != 0 || cap(batch.Layers) < 35 || len(batch.Greedy) != 0 || cap(batch.Greedy) != 1 {
		t.Fatalf("borrowed prefill forward batch = %#v, want empty layers with >=35 capacity and one greedy slot", batch)
	}
	batchPointer := uintptr(unsafe.Pointer(batch))
	batch.Greedy = append(batch.Greedy, hipGemma4Q4PrefillGreedyBatchOutput{Row: 7})
	core.RequireNoError(t, batch.Close())

	reusedBatch := hipBorrowGemma4Q4PrefillForwardBatch(35)
	defer reusedBatch.Close()
	if reusedBatch == nil || uintptr(unsafe.Pointer(reusedBatch)) != batchPointer {
		t.Fatalf("reused prefill forward batch = %#v, want original batch pointer %x", reusedBatch, batchPointer)
	}
	if len(reusedBatch.Layers) != 0 || cap(reusedBatch.Layers) < 35 || len(reusedBatch.Greedy) != 0 || cap(reusedBatch.Greedy) != 1 || reusedBatch.closed {
		t.Fatalf("reused prefill forward batch = %#v, want cleared open batch", reusedBatch)
	}

	layers := hipBorrowGemma4Q4PrefillForwardLayerBatches(35)
	if len(layers) != 0 || cap(layers) < 35 {
		t.Fatalf("borrowed prefill forward layer slice len/cap = %d/%d, want empty slice with capacity >= 35", len(layers), cap(layers))
	}
	layers = append(layers, hipGemma4Q4PrefillForwardLayerBatch{
		KV:   &hipGemma4Q4PrefillLayerKVBatch{},
		Body: &hipGemma4Q4PrefillLayerBodyBatch{},
	})
	hipReleaseGemma4Q4PrefillForwardLayerBatches(layers)

	reused := hipBorrowGemma4Q4PrefillForwardLayerBatches(35)
	defer hipReleaseGemma4Q4PrefillForwardLayerBatches(reused)
	if len(reused) != 0 || cap(reused) < 35 {
		t.Fatalf("reused prefill forward layer slice len/cap = %d/%d, want empty slice with capacity >= 35", len(reused), cap(reused))
	}
	reused = append(reused, hipGemma4Q4PrefillForwardLayerBatch{})
	if reused[0].KV != nil || reused[0].Body != nil {
		t.Fatalf("reused prefill forward layer slice kept stale pointers: %#v", reused[0])
	}
}

func TestHIPGemma4Q4EnsureAttentionWorkspacePrefillCapacity_HotPathOutputsGood(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup0()
	layer1, cleanup1 := hipGemma4Q4FixtureConfig(t, driver, 1, 4, 2, 8)
	defer cleanup1()
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer0, layer1}}
	plan := hipGemma4Q4PrefillPlan{
		Batches: []hipGemma4Q4PrefillUBatch{
			{Tokens: []int32{0}},
			{Tokens: []int32{0, 1}},
		},
	}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	allocStart := len(driver.allocations)
	core.RequireNoError(t, hipGemma4Q4EnsureAttentionWorkspacePrefillCapacity(driver, workspace, cfg, plan, true))
	activationRows := 2 * layer0.GateProjection.Rows
	large := &workspace.ActivationOutputFixed
	if large.Pointer() == 0 || large.Count() < activationRows {
		t.Fatalf("prefill activation workspace = %#v, want largest prefill batch", large)
	}
	hiddenRows := 2 * layer0.HiddenSize
	rmsCap := hiddenRows
	rmsOutput := &workspace.RMSResidualNormFixed
	if rmsOutput.Pointer() == 0 || rmsOutput.Count() != rmsCap*2 {
		t.Fatalf("RMS residual/norm workspace = %#v, want %d rows", rmsOutput, rmsCap*2)
	}
	projectionRows := 2 * hipGemma4Q4ProjectionWorkspaceRows(layer0)
	projectionOutput := &workspace.ProjectionOutputFixed
	if projectionOutput.Pointer() == 0 || projectionOutput.Count() != projectionRows {
		t.Fatalf("projection workspace = %#v, want %d rows", projectionOutput, projectionRows)
	}
	keyRows := 2 * layer0.KeyProjection.Rows
	kvProjectionCap := keyRows
	kvProjectionPair := &workspace.KVProjectionPairFixed
	if kvProjectionPair.Pointer() == 0 || kvProjectionPair.Count() != kvProjectionCap*2 {
		t.Fatalf("KV projection pair workspace = %#v, want %d rows", kvProjectionPair, kvProjectionCap*2)
	}
	valueRows := 2 * layer0.ValueProjection.Rows
	if valueRows != keyRows {
		t.Fatalf("KV projection fixture rows differ key=%d value=%d", keyRows, valueRows)
	}
	keyOutputPointer := kvProjectionPair.Pointer()
	valueOutputPointer := kvProjectionPair.Pointer() + nativeDevicePointer(kvProjectionCap*4)
	headRows := 2 * layer0.HeadDim
	keyValueNormOutput := &workspace.KeyValueNormFixed
	if keyValueNormOutput.Pointer() == 0 || keyValueNormOutput.Count() != headRows*2 {
		t.Fatalf("key/value norm workspace = %#v, want %d rows", keyValueNormOutput, headRows*2)
	}
	queryRoPERows := 2 * layer0.QueryProjection.Rows
	queryRoPEOutput := &workspace.RMSRoPEFixed
	if queryRoPEOutput.Pointer() == 0 || queryRoPEOutput.Count() != queryRoPERows {
		t.Fatalf("query RMS RoPE workspace = %#v, want %d rows", queryRoPEOutput, queryRoPERows)
	}
	attentionOutput := &workspace.AttentionOutputFixed
	if attentionOutput.Pointer() == 0 || attentionOutput.Count() < queryRoPERows {
		t.Fatalf("attention output workspace = %#v, want at least %d rows", attentionOutput, queryRoPERows)
	}
	qkvRows := hipGemma4Q4FusedDecodeQKVOutputRows(layer0)
	qkvOutput := &workspace.QKVOutputFixed
	if qkvOutput.Pointer() == 0 || qkvOutput.Count() != qkvRows {
		t.Fatalf("decode QKV workspace = %#v, want %d rows", qkvOutput, qkvRows)
	}
	core.AssertEqual(t, allocStart+13, len(driver.allocations))
	small, err := workspace.EnsureActivationOutput(driver, layer0.GateProjection.Rows)
	core.RequireNoError(t, err)
	if small.Pointer() != large.Pointer() || small.Count() != layer0.GateProjection.Rows {
		t.Fatalf("small activation view = %#v, want borrowed view of large workspace %x", small, large.Pointer())
	}
	smallResidual, err := workspace.EnsureRMSResidualOutput(driver, layer0.HiddenSize)
	core.RequireNoError(t, err)
	if smallResidual.Pointer() != rmsOutput.Pointer() || smallResidual.Count() != layer0.HiddenSize {
		t.Fatalf("small RMS residual view = %#v, want borrowed view of large workspace %x", smallResidual, rmsOutput.Pointer())
	}
	smallNorm, err := workspace.EnsureRMSNormOutput(driver, layer0.HiddenSize)
	core.RequireNoError(t, err)
	rmsNormPointer := rmsOutput.Pointer() + nativeDevicePointer(rmsCap*4)
	if smallNorm.Pointer() != rmsNormPointer || smallNorm.Count() != layer0.HiddenSize {
		t.Fatalf("small RMS norm view = %#v, want borrowed view of large workspace %x", smallNorm, rmsNormPointer)
	}
	smallProjection, err := workspace.EnsureProjectionOutput(driver, projectionRows/2)
	core.RequireNoError(t, err)
	if smallProjection.Pointer() != projectionOutput.Pointer() || smallProjection.Count() != projectionRows/2 {
		t.Fatalf("small projection view = %#v, want borrowed view of large workspace %x", smallProjection, projectionOutput.Pointer())
	}
	smallKey, err := workspace.EnsureKVProjectionOutput(driver, layer0.KeyProjection.Rows, 0)
	core.RequireNoError(t, err)
	if smallKey.Pointer() != keyOutputPointer || smallKey.Count() != layer0.KeyProjection.Rows {
		t.Fatalf("small KV key projection view = %#v, want borrowed view of large workspace %x", smallKey, keyOutputPointer)
	}
	smallValue, err := workspace.EnsureKVProjectionOutput(driver, layer0.ValueProjection.Rows, 1)
	core.RequireNoError(t, err)
	if smallValue.Pointer() != valueOutputPointer || smallValue.Count() != layer0.ValueProjection.Rows {
		t.Fatalf("small KV value projection view = %#v, want borrowed view of large workspace %x", smallValue, valueOutputPointer)
	}
	smallKeyRoPE, err := workspace.EnsureKeyRMSRoPEOutput(driver, layer0.HeadDim)
	core.RequireNoError(t, err)
	if smallKeyRoPE.Pointer() != keyValueNormOutput.Pointer() || smallKeyRoPE.Count() != layer0.HeadDim {
		t.Fatalf("small key RMS RoPE view = %#v, want borrowed view of large workspace %x", smallKeyRoPE, keyValueNormOutput.Pointer())
	}
	smallValueNorm, err := workspace.EnsureRMSNoScaleOutput(driver, layer0.HeadDim)
	core.RequireNoError(t, err)
	valueNormPointer := keyValueNormOutput.Pointer() + nativeDevicePointer(headRows*4)
	if smallValueNorm.Pointer() != valueNormPointer || smallValueNorm.Count() != layer0.HeadDim {
		t.Fatalf("small RMS no-scale view = %#v, want borrowed view of large workspace %x", smallValueNorm, valueNormPointer)
	}
	smallQueryRoPE, err := workspace.EnsureRMSRoPEOutput(driver, layer0.QueryProjection.Rows)
	core.RequireNoError(t, err)
	if smallQueryRoPE.Pointer() != queryRoPEOutput.Pointer() || smallQueryRoPE.Count() != layer0.QueryProjection.Rows {
		t.Fatalf("small query RMS RoPE view = %#v, want borrowed view of large workspace %x", smallQueryRoPE, queryRoPEOutput.Pointer())
	}
	smallQKV, err := workspace.EnsureQKVOutput(driver, qkvRows/2)
	core.RequireNoError(t, err)
	if smallQKV.Pointer() != qkvOutput.Pointer() || smallQKV.Count() != qkvRows/2 {
		t.Fatalf("small QKV view = %#v, want borrowed view of large workspace %x", smallQKV, qkvOutput.Pointer())
	}
	core.AssertEqual(t, allocStart+13, len(driver.allocations))
}

func TestHIPAttentionHeadsChunkedWorkspace_PerLayerProjectedReusesLarger_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	large, err := workspace.EnsurePerLayerProjected(driver, 24576)
	core.RequireNoError(t, err)
	largePointer := large.Pointer()
	small, err := workspace.EnsurePerLayerProjected(driver, 12288)
	core.RequireNoError(t, err)

	if small.Pointer() != largePointer || small.Count() != 12288 {
		t.Fatalf("small per-layer projected view = %#v, want borrowed view of large workspace %x", small, largePointer)
	}
	core.AssertEqual(t, 1, len(driver.allocations))
	if _, ok := workspace.PerLayerProjected[hipAttentionHeadsChunkedWorkspaceCapacityCount(12288)]; ok && hipAttentionHeadsChunkedWorkspaceCapacityCount(12288) != hipAttentionHeadsChunkedWorkspaceCapacityCount(24576) {
		t.Fatalf("smaller per-layer projected output got its own allocation")
	}
}

func BenchmarkHIPAttentionHeadsChunkedWorkspace_FinalHiddenOutputReused(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	output, err := workspace.EnsureFinalHiddenOutput(driver, 2304, 0)
	if err != nil {
		b.Fatal(err)
	}
	if output.Count() != 2304 || output.SizeBytes() != 9216 {
		b.Fatalf("final hidden output shape = %d/%d, want 2304/9216", output.Count(), output.SizeBytes())
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		output, err = workspace.EnsureFinalHiddenOutput(driver, 2304, i&1)
		if err != nil {
			b.Fatal(err)
		}
		if output.Count() != 2304 || output.SizeBytes() != 9216 {
			b.Fatalf("final hidden output shape = %d/%d, want 2304/9216", output.Count(), output.SizeBytes())
		}
	}
}

func TestHIPAttentionHeadsChunkedWorkspace_FinalHiddenOutputReusesLargerSlot_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	large0, err := workspace.EnsureFinalHiddenOutput(driver, 3072, 0)
	core.RequireNoError(t, err)
	small0, err := workspace.EnsureFinalHiddenOutput(driver, 1536, 0)
	core.RequireNoError(t, err)
	large1, err := workspace.EnsureFinalHiddenOutput(driver, 3072, 1)
	core.RequireNoError(t, err)
	small1, err := workspace.EnsureFinalHiddenOutput(driver, 1536, 1)
	core.RequireNoError(t, err)

	if small0.Pointer() != large0.Pointer() || small0.Count() != 1536 {
		t.Fatalf("small final-hidden slot 0 view = %#v, want borrowed view of %x", small0, large0.Pointer())
	}
	if small1.Pointer() != large1.Pointer() || small1.Count() != 1536 {
		t.Fatalf("small final-hidden slot 1 view = %#v, want borrowed view of %x", small1, large1.Pointer())
	}
	pairCapCount := 4096
	if large1.Pointer() != large0.Pointer()+nativeDevicePointer(pairCapCount*4) {
		t.Fatalf("final-hidden slot 1 pointer = %x, want slot 0+%d", large1.Pointer(), pairCapCount*4)
	}
	core.AssertEqual(t, 1, len(driver.allocations))
	if _, ok := workspace.FinalHiddenOutputs[0][1536]; ok {
		t.Fatalf("smaller final-hidden slot 0 output got its own allocation")
	}
	if _, ok := workspace.FinalHiddenOutputs[1][1536]; ok {
		t.Fatalf("smaller final-hidden slot 1 output got its own allocation")
	}
	if output := &workspace.FinalHiddenPairFixed; output.Pointer() == 0 || output.Count() != pairCapCount*2 {
		t.Fatalf("final-hidden pair output = %#v, want %d rows", output, pairCapCount*2)
	}
}

func TestHIPAttentionHeadsChunkedWorkspace_KVProjectionOutputReusesLargerSlot_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	largeKey, err := workspace.EnsureKVProjectionOutput(driver, 1024, 0)
	core.RequireNoError(t, err)
	largeKeyPointer := largeKey.Pointer()
	smallKey, err := workspace.EnsureKVProjectionOutput(driver, 512, 0)
	core.RequireNoError(t, err)
	largeValue, err := workspace.EnsureKVProjectionOutput(driver, 1024, 1)
	core.RequireNoError(t, err)
	largeValuePointer := largeValue.Pointer()
	smallValue, err := workspace.EnsureKVProjectionOutput(driver, 512, 1)
	core.RequireNoError(t, err)

	if smallKey.Pointer() != largeKeyPointer || smallKey.Count() != 512 {
		t.Fatalf("small KV key slot view = %#v, want borrowed view of %x", smallKey, largeKeyPointer)
	}
	if smallValue.Pointer() != largeValuePointer || smallValue.Count() != 512 {
		t.Fatalf("small KV value slot view = %#v, want borrowed view of %x", smallValue, largeValuePointer)
	}
	if largeKeyPointer == largeValuePointer {
		t.Fatalf("KV projection slots share backing pointer %x", largeKeyPointer)
	}
	if largeValuePointer != largeKeyPointer+nativeDevicePointer(1024*4) {
		t.Fatalf("KV value slot pointer = %x, want key+%d", largeValuePointer, 1024*4)
	}
	core.AssertEqual(t, 1, len(driver.allocations))
	if _, ok := workspace.KVProjectionOutputs[0][512]; ok {
		t.Fatalf("smaller KV key slot output got its own allocation")
	}
	if _, ok := workspace.KVProjectionOutputs[1][512]; ok {
		t.Fatalf("smaller KV value slot output got its own allocation")
	}
	if output := &workspace.KVProjectionPairFixed; output.Pointer() == 0 || output.Count() != 2048 {
		t.Fatalf("KV projection pair output = %#v, want 2048 rows", output)
	}
}

func TestHIPAttentionHeadsChunkedWorkspace_KeyValueNormOutputsReuseLarger_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	largeKey, err := workspace.EnsureKeyRMSRoPEOutput(driver, 1024)
	core.RequireNoError(t, err)
	largeKeyPointer := largeKey.Pointer()
	smallKey, err := workspace.EnsureKeyRMSRoPEOutput(driver, 512)
	core.RequireNoError(t, err)
	largeValue, err := workspace.EnsureRMSNoScaleOutput(driver, 1024)
	core.RequireNoError(t, err)
	largeValuePointer := largeValue.Pointer()
	smallValue, err := workspace.EnsureRMSNoScaleOutput(driver, 512)
	core.RequireNoError(t, err)

	if smallKey.Pointer() != largeKeyPointer || smallKey.Count() != 512 {
		t.Fatalf("small key RMS RoPE view = %#v, want borrowed view of %x", smallKey, largeKeyPointer)
	}
	if smallValue.Pointer() != largeValuePointer || smallValue.Count() != 512 {
		t.Fatalf("small RMS no-scale view = %#v, want borrowed view of %x", smallValue, largeValuePointer)
	}
	if largeValuePointer != largeKeyPointer+nativeDevicePointer(1024*4) {
		t.Fatalf("RMS no-scale pointer = %x, want key+%d", largeValuePointer, 1024*4)
	}
	core.AssertEqual(t, 1, len(driver.allocations))
	if _, ok := workspace.KeyRMSRoPEOutputs[512]; ok {
		t.Fatalf("smaller key RMS RoPE output got its own allocation")
	}
	if _, ok := workspace.RMSNoScaleOutputs[512]; ok {
		t.Fatalf("smaller RMS no-scale output got its own allocation")
	}
	if output := &workspace.KeyValueNormFixed; output.Pointer() == 0 || output.Count() != 2048 {
		t.Fatalf("key/value norm pair output = %#v, want 2048 rows", output)
	}
}

func BenchmarkHIPAttentionHeadsChunkedWorkspace_NextInputOutputReused(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	output, err := workspace.EnsureNextInputOutput(driver, 2304, 0)
	if err != nil {
		b.Fatal(err)
	}
	if output.Count() != 2304 || output.SizeBytes() != 9216 {
		b.Fatalf("next input output shape = %d/%d, want 2304/9216", output.Count(), output.SizeBytes())
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		output, err = workspace.EnsureNextInputOutput(driver, 2304, i&1)
		if err != nil {
			b.Fatal(err)
		}
		if output.Count() != 2304 || output.SizeBytes() != 9216 {
			b.Fatalf("next input output shape = %d/%d, want 2304/9216", output.Count(), output.SizeBytes())
		}
	}
}

func TestHIPAttentionHeadsChunkedWorkspace_NextInputOutputReusesLargerSlot_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()

	large0, err := workspace.EnsureNextInputOutput(driver, 3072, 0)
	core.RequireNoError(t, err)
	small0, err := workspace.EnsureNextInputOutput(driver, 1536, 0)
	core.RequireNoError(t, err)
	large1, err := workspace.EnsureNextInputOutput(driver, 3072, 1)
	core.RequireNoError(t, err)
	small1, err := workspace.EnsureNextInputOutput(driver, 1536, 1)
	core.RequireNoError(t, err)

	if small0.Pointer() != large0.Pointer() || small0.Count() != 1536 {
		t.Fatalf("small next-input slot 0 view = %#v, want borrowed view of %x", small0, large0.Pointer())
	}
	if small1.Pointer() != large1.Pointer() || small1.Count() != 1536 {
		t.Fatalf("small next-input slot 1 view = %#v, want borrowed view of %x", small1, large1.Pointer())
	}
	pairCapCount := 4096
	if large1.Pointer() != large0.Pointer()+nativeDevicePointer(pairCapCount*4) {
		t.Fatalf("next-input slot 1 pointer = %x, want slot 0+%d", large1.Pointer(), pairCapCount*4)
	}
	core.AssertEqual(t, 1, len(driver.allocations))
	if _, ok := workspace.NextInputOutputs[0][1536]; ok {
		t.Fatalf("smaller next-input slot 0 output got its own allocation")
	}
	if _, ok := workspace.NextInputOutputs[1][1536]; ok {
		t.Fatalf("smaller next-input slot 1 output got its own allocation")
	}
	if output := &workspace.NextInputPairFixed; output.Pointer() == 0 || output.Count() != pairCapCount*2 {
		t.Fatalf("next-input pair output = %#v, want %d rows", output, pairCapCount*2)
	}
}

func BenchmarkHIPGemma4Q4PerLayerInputDeviceSetLayer_View(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	const (
		layerCount = 32
		inputSize  = 2304
	)
	set := &hipGemma4Q4PerLayerInputDeviceSet{
		driver:           driver,
		layerCount:       layerCount,
		layerStrideBytes: uint64(inputSize * 4),
		layerValueCount:  inputSize,
		viewLabel:        "per-layer input slice",
		Backing: []*hipDeviceByteBuffer{{
			driver:    driver,
			pointer:   0x100000,
			count:     layerCount * inputSize,
			sizeBytes: uint64(layerCount * inputSize * 4),
		}},
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		layer := set.Layer(i % layerCount)
		if layer == nil || layer.Pointer() == 0 || layer.Count() != inputSize {
			b.Fatalf("layer view = %#v", layer)
		}
	}
}

func BenchmarkHIPAttentionHeadsChunkedWorkspace_PerLayerInputDeviceSetReused(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	const (
		layerCount = 32
		inputSize  = 2304
	)
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	backing := &hipDeviceByteBuffer{
		driver:    driver,
		pointer:   0x100000,
		count:     layerCount * inputSize,
		sizeBytes: uint64(layerCount * inputSize * 4),
	}
	set, err := workspace.BorrowPerLayerInputDeviceSet(driver, layerCount, inputSize, backing)
	if err != nil {
		b.Fatal(err)
	}
	if set.LayerCount() != layerCount || set.Layer(0) == nil {
		b.Fatalf("per-layer input set = %#v, want reusable device views", set)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		set, err = workspace.BorrowPerLayerInputDeviceSet(driver, layerCount, inputSize, backing)
		if err != nil {
			b.Fatal(err)
		}
		layer := set.Layer(i % layerCount)
		if layer == nil || layer.Pointer() == 0 || layer.Count() != inputSize {
			b.Fatalf("layer view = %#v", layer)
		}
	}
}

func TestHIPAttentionHeadsChunkedWorkspace_PerLayerInputDeviceSetBatchReused_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	const (
		layerCount      = 32
		layerValueCount = 4608
	)
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	backing := &hipDeviceByteBuffer{
		driver:    driver,
		pointer:   0x100000,
		count:     layerCount * layerValueCount,
		sizeBytes: uint64(layerCount * layerValueCount * 4),
	}
	set, err := workspace.BorrowPerLayerInputDeviceSetBatch(driver, layerCount, layerValueCount, backing, "test batch layer")
	core.RequireNoError(t, err)
	if set != &workspace.PerLayerInputSet {
		t.Fatalf("set pointer = %#v, want workspace-owned set %#v", set, &workspace.PerLayerInputSet)
	}
	first := set.Layer(0)
	firstPointer := first.Pointer()
	second := set.Layer(1)
	secondPointer := second.Pointer()
	if first == nil || second == nil {
		t.Fatalf("layer views = %#v %#v, want two borrowed views", first, second)
	}
	core.AssertEqual(t, backing.Pointer(), firstPointer)
	core.AssertEqual(t, backing.Pointer()+nativeDevicePointer(layerValueCount*4), secondPointer)
	core.AssertEqual(t, layerValueCount, second.Count())

	next, err := workspace.BorrowPerLayerInputDeviceSetBatch(driver, layerCount, layerValueCount, backing, "test batch layer")
	core.RequireNoError(t, err)
	if next != set {
		t.Fatalf("next set pointer = %#v, want reused %#v", next, set)
	}
}

func BenchmarkHIPGemma4Q4PerLayerInputConfigScales_Cached(b *testing.B) {
	cfg := hipGemma4Q4PerLayerInputConfig{
		InputSize: 256,
		ModelProjection: hipBF16DeviceWeightConfig{
			Cols: 2048,
		},
	}
	cfg.finalizeScales()
	var sink float32
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sink += cfg.embeddingScale()
		sink += cfg.modelProjectionScale()
		sink += hipGemma4Q4PerLayerCombineScale
	}
	if sink == 0 {
		b.Fatal("unexpected zero scale sum")
	}
}

func BenchmarkHIPGemma4Q4LayerConfigEmbeddingScale_Cached(b *testing.B) {
	cfg := hipGemma4Q4Layer0Config{HiddenSize: 2048}
	cfg.finalizeScales()
	var sink float32
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sink += cfg.embeddingScale()
	}
	if sink == 0 {
		b.Fatal("unexpected zero scale sum")
	}
}

func TestHIPAttentionHeadsChunkedWorkspace_SuppressTokenBufferUsesGreedyReserve_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	tokens := []int32{0, 2, 105, 106, 107, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218}

	buffer, err := workspace.EnsureSuppressTokenBuffer(driver, tokens)
	core.RequireNoError(t, err)
	if buffer == nil || buffer.Count() != len(tokens) || buffer.SizeBytes() != uint64(len(tokens)*4) {
		t.Fatalf("suppress token buffer = %#v, want %d tokens", buffer, len(tokens))
	}
	if len(workspace.ProjectionGreedyBest) != 1 {
		t.Fatalf("greedy slabs = %d, want one shared suppress-token slab", len(workspace.ProjectionGreedyBest))
	}
	slab := workspace.ProjectionGreedyBest[0]
	wantPointer := slab.Pointer() + nativeDevicePointer(hipProjectionGreedySuppressReserveOffsetBytes)
	core.AssertEqual(t, wantPointer, buffer.Pointer())
	core.AssertEqual(t, true, buffer.borrowed)
	core.AssertEqual(t, hipProjectionGreedySuppressReserveBytes, cap(workspace.SuppressTokenPayload))
	core.AssertEqual(t, hipProjectionGreedySuppressReserveBytes/4, cap(workspace.SuppressTokenIDs))
	core.AssertEqual(t, []uint64{uint64(hipProjectionGreedyBestWorkspaceSlots * hipMLXQ4ProjectionBestBytes)}, driver.allocations)
	core.AssertEqual(t, []uint64{uint64(hipProjectionGreedyBestWorkspaceSlots * hipMLXQ4ProjectionBestBytes)}, driver.memsets)
	core.AssertEqual(t, []uint64{uint64(len(tokens) * 4)}, driver.copies)

	payload, offset, ok := driver.memoryForPointer(buffer.Pointer(), int(buffer.SizeBytes()))
	core.RequireTrue(t, ok)
	for index, token := range tokens {
		got := int32(binary.LittleEndian.Uint32(payload[offset+index*4:]))
		core.AssertEqual(t, token, got)
	}

	greedy, err := workspace.BorrowProjectionGreedyBest(driver)
	core.RequireNoError(t, err)
	core.AssertEqual(t, slab.Pointer(), greedy.Pointer())
	if greedy.Pointer() == buffer.Pointer() {
		t.Fatalf("greedy slot overlapped suppress-token reserve at %x", greedy.Pointer())
	}
}

func TestHIPAttentionHeadsChunkedWorkspace_PrefillTokenBufferUsesGreedyReserve_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	tokens := make([]int32, 512)
	for index := range tokens {
		tokens[index] = int32(index + 11)
	}

	prefill, err := workspace.EnsurePrefillTokenBuffer(driver, tokens)
	core.RequireNoError(t, err)
	if prefill == nil || prefill.Count() != len(tokens) || prefill.SizeBytes() != uint64(len(tokens)*4) {
		t.Fatalf("prefill token buffer = %#v, want %d tokens", prefill, len(tokens))
	}
	if len(workspace.ProjectionGreedyBest) != 1 {
		t.Fatalf("greedy slabs = %d, want one shared prefill-token slab", len(workspace.ProjectionGreedyBest))
	}
	slab := workspace.ProjectionGreedyBest[0]
	core.AssertEqual(t, slab.Pointer()+nativeDevicePointer(hipProjectionGreedyPrefillReserveOffsetBytes), prefill.Pointer())
	core.AssertEqual(t, true, prefill.borrowed)
	core.AssertEqual(t, []uint64{uint64(hipProjectionGreedyBestWorkspaceSlots * hipMLXQ4ProjectionBestBytes)}, driver.allocations)
	core.AssertEqual(t, []uint64{uint64(hipProjectionGreedyBestWorkspaceSlots * hipMLXQ4ProjectionBestBytes)}, driver.memsets)
	core.AssertEqual(t, []uint64{uint64(len(tokens) * 4)}, driver.copies)

	payload, offset, ok := driver.memoryForPointer(prefill.Pointer(), int(prefill.SizeBytes()))
	core.RequireTrue(t, ok)
	for index, token := range tokens {
		got := int32(binary.LittleEndian.Uint32(payload[offset+index*4:]))
		core.AssertEqual(t, token, got)
	}

	suppress, err := workspace.EnsureSuppressTokenBuffer(driver, []int32{0, 2, 105, 106})
	core.RequireNoError(t, err)
	core.AssertEqual(t, slab.Pointer()+nativeDevicePointer(hipProjectionGreedySuppressReserveOffsetBytes), suppress.Pointer())
	if prefill.Pointer()+nativeDevicePointer(prefill.SizeBytes()) > suppress.Pointer() {
		t.Fatalf("prefill token reserve overlapped suppress reserve: prefill=%x/%d suppress=%x", prefill.Pointer(), prefill.SizeBytes(), suppress.Pointer())
	}
	greedy, err := workspace.BorrowProjectionGreedyBest(driver)
	core.RequireNoError(t, err)
	core.AssertEqual(t, slab.Pointer(), greedy.Pointer())
	if greedy.Pointer()+nativeDevicePointer(greedy.SizeBytes()) > prefill.Pointer() {
		t.Fatalf("greedy slot overlapped prefill reserve: greedy=%x/%d prefill=%x", greedy.Pointer(), greedy.SizeBytes(), prefill.Pointer())
	}
}

func BenchmarkHIPAttentionHeadsChunkedWorkspace_SuppressTokenBufferReused(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	tokens := []int32{0, 2, 105, 106, 107, 200}
	buffer, err := workspace.EnsureSuppressTokenBuffer(driver, tokens)
	if err != nil {
		b.Fatal(err)
	}
	if buffer == nil || buffer.Count() != len(tokens) {
		b.Fatalf("suppress token buffer = %#v, want %d tokens", buffer, len(tokens))
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		buffer, err = workspace.EnsureSuppressTokenBuffer(driver, tokens)
		if err != nil {
			b.Fatal(err)
		}
		if buffer == nil || buffer.Count() != len(tokens) {
			b.Fatalf("suppress token buffer = %#v, want %d tokens", buffer, len(tokens))
		}
	}
}

func BenchmarkHIPAttentionHeadsChunkedWorkspace_PrefillTokenBufferBorrowed(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	tokens := make([]int32, 512)
	for index := range tokens {
		tokens[index] = int32(index + 11)
	}
	buffer, err := workspace.EnsurePrefillTokenBuffer(driver, tokens)
	if err != nil {
		b.Fatal(err)
	}
	if buffer == nil || buffer.Count() != len(tokens) {
		b.Fatalf("prefill token buffer = %#v, want %d tokens", buffer, len(tokens))
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		buffer, err = workspace.EnsurePrefillTokenBuffer(driver, tokens)
		if err != nil {
			b.Fatal(err)
		}
		if buffer == nil || buffer.Count() != len(tokens) {
			b.Fatalf("prefill token buffer = %#v, want %d tokens", buffer, len(tokens))
		}
	}
}

func TestHIPAttentionHeadsChunkedWorkspace_TokenIDValueCached_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	buffer, err := workspace.EnsureTokenIDValue(driver, 42, 128)
	core.RequireNoError(t, err)
	if buffer == nil || buffer.Count() != 1 {
		t.Fatalf("token buffer = %#v, want one token", buffer)
	}
	copiesAfterFirst := len(driver.copies)
	_, err = workspace.EnsureTokenIDValue(driver, 42, 128)
	core.RequireNoError(t, err)
	core.AssertEqual(t, copiesAfterFirst, len(driver.copies))
	_, err = workspace.EnsureTokenIDValue(driver, 43, 128)
	core.RequireNoError(t, err)
	core.AssertEqual(t, copiesAfterFirst+1, len(driver.copies))
	_, err = workspace.EnsureTokenIDValue(driver, 128, 128)
	core.AssertError(t, err)
	core.AssertEqual(t, copiesAfterFirst+1, len(driver.copies))
}

func BenchmarkHIPAttentionHeadsChunkedWorkspace_TokenIDValueCached(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	buffer, err := workspace.EnsureTokenIDValue(driver, 42, 128)
	if err != nil {
		b.Fatal(err)
	}
	if buffer == nil || buffer.Count() != 1 {
		b.Fatalf("token buffer = %#v, want one token", buffer)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		buffer, err = workspace.EnsureTokenIDValue(driver, 42, 128)
		if err != nil {
			b.Fatal(err)
		}
		if buffer == nil || buffer.Count() != 1 {
			b.Fatalf("token buffer = %#v, want one token", buffer)
		}
	}
}

func BenchmarkHIPGemma4Q4SharedKVSourceByLayer_Cached(b *testing.B) {
	const layerCount = 32
	layers := make([]hipGemma4Q4Layer0Config, layerCount)
	for index := range layers {
		if index%6 == 5 {
			layers[index].LayerType = "full_attention"
			layers[index].HeadDim = 256
		} else {
			layers[index].LayerType = "sliding_attention"
			layers[index].HeadDim = 256
		}
	}
	cfg := hipGemma4Q4ForwardConfig{
		Layers:         layers,
		KVSharedLayers: hipGemma4Q4DefaultKVSharedLayers(layerCount),
	}
	cfg.SharedKVSources = hipGemma4Q4BuildSharedKVSourceByLayer(cfg)
	if got := len(hipGemma4Q4SharedKVSourceByLayer(cfg)); got != layerCount {
		b.Fatalf("shared KV source count = %d, want %d", got, layerCount)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sources := hipGemma4Q4SharedKVSourceByLayer(cfg)
		if len(sources) != layerCount || sources[layerCount-1] < 0 {
			b.Fatalf("shared KV sources = %#v", sources)
		}
	}
}

func BenchmarkHIPGemma4Q4DecoderLayerRequest_NextInputNormValue(b *testing.B) {
	req := hipGemma4Q4DecoderLayerRequest{
		NextInputNormValue: hipRMSNormDeviceWeightConfig{
			WeightPointer:  0x1000,
			WeightBytes:    4608,
			Count:          2304,
			WeightEncoding: hipRMSNormWeightEncodingBF16,
			Flags:          hipRMSNormLaunchFlagAddUnitWeight,
			Epsilon:        1e-6,
		},
		HasNextInputNorm: true,
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		cfg, ok := req.nextInputNormConfig()
		if !ok || cfg.Count != 2304 || cfg.WeightPointer == 0 {
			b.Fatalf("next input norm = %#v, %v", cfg, ok)
		}
	}
}

func BenchmarkHIPGemma4Q4DeviceLayerKVStateValueHandoff(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	cache := &rocmDeviceKVCache{
		driver:     driver,
		mode:       rocmKVCacheModeKQ8VQ4,
		blockSize:  1,
		tokenCount: 1,
		pages: []rocmDeviceKVPage{{
			tokenStart: 0,
			tokenCount: 1,
			keyWidth:   4,
			valueWidth: 4,
			key:        rocmDeviceKVTensor{pointer: 0x1000, sizeBytes: 4, encoding: rocmKVEncodingQ8},
			value:      rocmDeviceKVTensor{pointer: 0x2000, sizeBytes: 2, encoding: rocmKVEncodingQ4},
			owned:      true,
		}},
	}
	table := &rocmDeviceKVDescriptorTable{
		driver:    driver,
		pointer:   0x3000,
		sizeBytes: rocmDeviceKVDescriptorHeaderBytes + rocmDeviceKVDescriptorPageBytes,
		version:   rocmDeviceKVDescriptorVersion,
		pageCount: 1,
	}
	next := &hipGemma4Q4DeviceDecodeState{layers: make([]hipGemma4Q4DeviceLayerKVState, 0, 1)}
	result := hipGemma4Q4DecoderLayerResult{
		DeviceLayer: hipGemma4Q4DeviceLayerKVState{
			cache:                   cache,
			descriptorTable:         table,
			borrowedCache:           true,
			borrowedDescriptorTable: true,
		},
		DeviceLayerValid: true,
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		next.layers = next.layers[:0]
		if !result.DeviceLayerValid {
			b.Fatal("device layer was not returned")
		}
		next.layers = append(next.layers, result.DeviceLayer)
		if len(next.layers) != 1 || next.layers[0].cache != cache {
			b.Fatalf("handoff layers = %#v", next.layers)
		}
	}
}

func BenchmarkHIPLoadedSmallDecodePriorKVRestoreInto_Reused(b *testing.B) {
	smoke := hipSmallDecodeFixture("qwen3")
	cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, defaultROCmKVBlockSize)
	if err != nil {
		b.Fatalf("create KV cache: %v", err)
	}
	if err := cache.AppendVectors(0, smoke.HiddenSize, smoke.HiddenSize, smoke.PriorKeys, smoke.PriorValues); err != nil {
		b.Fatalf("append KV cache vectors: %v", err)
	}
	model := &hipLoadedModel{}
	req := hipDecodeRequest{TokenID: 2, KV: cache}
	keyWidth, valueWidth, err := req.kvVectorWidths()
	if err != nil {
		b.Fatalf("KV widths: %v", err)
	}
	if _, _, err := model.restoreLoadedSmallDecodePriorKV(req, keyWidth, valueWidth); err != nil {
		b.Fatalf("warm restore prior KV: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		keys, values, err := model.restoreLoadedSmallDecodePriorKV(req, keyWidth, valueWidth)
		if err != nil {
			b.Fatalf("restore prior KV: %v", err)
		}
		if len(keys) != len(smoke.PriorKeys) || len(values) != len(smoke.PriorValues) {
			b.Fatalf("restored KV lengths = %d/%d, want %d/%d", len(keys), len(values), len(smoke.PriorKeys), len(smoke.PriorValues))
		}
	}
}

func TestHIPGemma4Q4DeviceDecodeStatePoolPrewarm_Good(t *testing.T) {
	hipPrewarmGemma4Q4DeviceDecodeStatePool(35, 2)
	state := hipNewGemma4Q4DeviceDecodeState(rocmKVCacheModeKQ8VQ4, 35)
	if state == nil || state.mode != rocmKVCacheModeKQ8VQ4 || len(state.layers) != 0 || cap(state.layers) < 35 {
		t.Fatalf("prewarmed decode state = %#v len/cap=%d/%d, want empty state with >=35 layer capacity", state, len(state.layers), cap(state.layers))
	}
	statePointer := uintptr(unsafe.Pointer(state))
	state.appendLayers = 3
	hipReleaseGemma4Q4DeviceLayerStates(state.layers)
	state.layers = nil
	state.closed = true
	hipReleaseClosedGemma4Q4DeviceDecodeState(state)

	reused := hipNewGemma4Q4DeviceDecodeState(rocmKVCacheModeFP16, 35)
	defer func() {
		hipReleaseGemma4Q4DeviceLayerStates(reused.layers)
		reused.layers = nil
		reused.closed = true
		hipReleaseClosedGemma4Q4DeviceDecodeState(reused)
	}()
	if reused == nil || uintptr(unsafe.Pointer(reused)) != statePointer {
		t.Fatalf("reused decode state = %#v, want original state pointer %x", reused, statePointer)
	}
	if reused.mode != rocmKVCacheModeFP16 || reused.appendLayers != 0 || len(reused.layers) != 0 || cap(reused.layers) < 35 {
		t.Fatalf("reused decode state = %#v len/cap=%d/%d, want cleared state with FP16 mode and >=35 layer capacity", reused, len(reused.layers), cap(reused.layers))
	}
}

func BenchmarkHIPGemma4Q4DeviceLayerStatePool_Reused(b *testing.B) {
	layers := hipBorrowGemma4Q4DeviceLayerStates(32)
	hipReleaseGemma4Q4DeviceLayerStates(layers)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		layers = hipBorrowGemma4Q4DeviceLayerStates(32)
		if len(layers) != 0 || cap(layers) < 32 {
			b.Fatalf("layer state slice len/cap = %d/%d, want 0/>=32", len(layers), cap(layers))
		}
		hipReleaseGemma4Q4DeviceLayerStates(layers)
	}
}

func TestHIPGemma4Q4DeviceOwnershipActionPool_ReusesClearedSlice_Good(t *testing.T) {
	actions := hipBorrowGemma4Q4DeviceOwnershipActions(8)
	if len(actions) != 0 || cap(actions) < 8 {
		t.Fatalf("borrowed ownership actions len/cap = %d/%d, want 0/>=8", len(actions), cap(actions))
	}
	layer := &hipGemma4Q4DeviceLayerKVState{}
	cache := &rocmDeviceKVCache{}
	actions = append(actions, hipGemma4Q4DeviceOwnershipAction{oldLayer: layer, newCache: cache, append: true})
	hipReleaseGemma4Q4DeviceOwnershipActions(actions)

	reused := hipBorrowGemma4Q4DeviceOwnershipActions(8)
	defer hipReleaseGemma4Q4DeviceOwnershipActions(reused)
	if len(reused) != 0 || cap(reused) < 8 {
		t.Fatalf("reused ownership actions len/cap = %d/%d, want 0/>=8", len(reused), cap(reused))
	}
	full := reused[:cap(reused)]
	for index, action := range full {
		if action.oldLayer != nil || action.newCache != nil || action.append {
			t.Fatalf("reused ownership action %d = %+v, want cleared", index, action)
		}
	}
}

func BenchmarkHIPGemma4Q4DeviceOwnershipActionPool_Reused(b *testing.B) {
	actions := hipBorrowGemma4Q4DeviceOwnershipActions(32)
	hipReleaseGemma4Q4DeviceOwnershipActions(actions)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		actions = hipBorrowGemma4Q4DeviceOwnershipActions(32)
		if len(actions) != 0 || cap(actions) < 32 {
			b.Fatalf("ownership action slice len/cap = %d/%d, want 0/>=32", len(actions), cap(actions))
		}
		hipReleaseGemma4Q4DeviceOwnershipActions(actions)
	}
}

func BenchmarkHIPGemma4Q4DeviceLayerKVStateClose_Borrowed(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	cache := &rocmDeviceKVCache{
		driver:     driver,
		mode:       rocmKVCacheModeKQ8VQ4,
		blockSize:  1,
		tokenCount: 1,
		pages: []rocmDeviceKVPage{{
			tokenStart: 0,
			tokenCount: 1,
			keyWidth:   4,
			valueWidth: 4,
			key:        rocmDeviceKVTensor{pointer: 0x1000, sizeBytes: 4, encoding: rocmKVEncodingQ8},
			value:      rocmDeviceKVTensor{pointer: 0x2000, sizeBytes: 2, encoding: rocmKVEncodingQ4},
			owned:      true,
		}},
	}
	table := &rocmDeviceKVDescriptorTable{
		driver:    driver,
		pointer:   0x3000,
		sizeBytes: rocmDeviceKVDescriptorHeaderBytes + rocmDeviceKVDescriptorPageBytes,
		version:   rocmDeviceKVDescriptorVersion,
		pageCount: 1,
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		layer := hipGemma4Q4DeviceLayerKVState{
			cache:                   cache,
			descriptorTable:         table,
			borrowedCache:           true,
			borrowedDescriptorTable: true,
		}
		if err := layer.Close(); err != nil {
			b.Fatal(err)
		}
		if cache.closed || table.closed {
			b.Fatal("borrowed layer close closed source owner")
		}
	}
}

func BenchmarkROCmDeviceKVCacheBorrowRelease_Hot(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	cache := rocmBorrowDeviceKVCache(driver, rocmKVCacheModeKQ8VQ4, 1, 0, nil, false)
	rocmReleaseDeviceKVCache(cache)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		cache = rocmBorrowDeviceKVCache(driver, rocmKVCacheModeKQ8VQ4, 1, 128, nil, false)
		if cache.driver != driver || cache.mode != rocmKVCacheModeKQ8VQ4 || cache.TokenCount() != 128 {
			b.Fatalf("cache = %#v", cache)
		}
		rocmReleaseDeviceKVCache(cache)
	}
}

func BenchmarkHIPMLXQ4DeviceWeightConfigValidateInputCount_Hot(b *testing.B) {
	cfg := hipMLXQ4DeviceWeightConfig{
		WeightPointer: 0x1000,
		ScalePointer:  0x2000,
		BiasPointer:   0x3000,
		WeightBytes:   2304 * 288 * 4,
		ScaleBytes:    2304 * 36 * 2,
		BiasBytes:     2304 * 36 * 2,
		Rows:          2304,
		Cols:          2304,
		GroupSize:     64,
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := cfg.validateInputCount(2304); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkROCmModelEmitTokenProbe_NoSink(b *testing.B) {
	model := &rocmModel{}
	token := inference.Token{ID: 42, Text: "hello"}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		model.emitTokenProbe(token, 2, i+1)
	}
}

func BenchmarkHIPMLXQ4TripleProjLaunchArgsBinary_Hot(b *testing.B) {
	args := hipMLXQ4TripleProjLaunchArgs{
		InputPointer:        0x1000,
		OutputPointer:       0x2000,
		FirstWeightPointer:  0x3000,
		FirstScalePointer:   0x4000,
		FirstBiasPointer:    0x5000,
		SecondWeightPointer: 0x6000,
		SecondScalePointer:  0x7000,
		SecondBiasPointer:   0x8000,
		ThirdWeightPointer:  0x9000,
		ThirdScalePointer:   0xa000,
		ThirdBiasPointer:    0xb000,
		FirstRows:           16,
		SecondRows:          4,
		ThirdRows:           4,
		Cols:                16,
		GroupSize:           8,
		Bits:                hipMLXQ4ProjectionBits,
		InputBytes:          64,
		OutputBytes:         96,
		FirstWeightBytes:    128,
		FirstScaleBytes:     64,
		FirstBiasBytes:      64,
		SecondWeightBytes:   32,
		SecondScaleBytes:    16,
		SecondBiasBytes:     16,
		ThirdWeightBytes:    32,
		ThirdScaleBytes:     16,
		ThirdBiasBytes:      16,
	}
	packet, err := args.Binary()
	if err != nil {
		b.Fatal(err)
	}
	hipReleaseLaunchPacket(packet)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		packet, err = args.Binary()
		if err != nil {
			b.Fatal(err)
		}
		if len(packet) != hipMLXQ4TripleProjLaunchArgsBytes {
			b.Fatalf("packet len = %d, want %d", len(packet), hipMLXQ4TripleProjLaunchArgsBytes)
		}
		hipReleaseLaunchPacket(packet)
	}
}

func BenchmarkHIPMLXQ4TripleProjLaunchArgsBinaryInto_Hot(b *testing.B) {
	args := hipMLXQ4TripleProjLaunchArgs{
		InputPointer:        0x1000,
		OutputPointer:       0x2000,
		FirstWeightPointer:  0x3000,
		FirstScalePointer:   0x4000,
		FirstBiasPointer:    0x5000,
		SecondWeightPointer: 0x6000,
		SecondScalePointer:  0x7000,
		SecondBiasPointer:   0x8000,
		ThirdWeightPointer:  0x9000,
		ThirdScalePointer:   0xa000,
		ThirdBiasPointer:    0xb000,
		FirstRows:           16,
		SecondRows:          4,
		ThirdRows:           4,
		Cols:                16,
		GroupSize:           8,
		Bits:                hipMLXQ4ProjectionBits,
		InputBytes:          64,
		OutputBytes:         96,
		FirstWeightBytes:    128,
		FirstScaleBytes:     64,
		FirstBiasBytes:      64,
		SecondWeightBytes:   32,
		SecondScaleBytes:    16,
		SecondBiasBytes:     16,
		ThirdWeightBytes:    32,
		ThirdScaleBytes:     16,
		ThirdBiasBytes:      16,
	}
	var scratch [hipMLXQ4TripleProjLaunchArgsBytes]byte
	packet, err := args.BinaryInto(scratch[:])
	if err != nil {
		b.Fatal(err)
	}
	if len(packet) != hipMLXQ4TripleProjLaunchArgsBytes {
		b.Fatalf("packet len = %d, want %d", len(packet), hipMLXQ4TripleProjLaunchArgsBytes)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		packet, err = args.BinaryInto(scratch[:])
		if err != nil {
			b.Fatal(err)
		}
		if len(packet) != hipMLXQ4TripleProjLaunchArgsBytes {
			b.Fatalf("packet len = %d, want %d", len(packet), hipMLXQ4TripleProjLaunchArgsBytes)
		}
	}
}

type hipLaunchPacketReleasingStubDriver struct {
	inferenceBenchmarkHIPKernelCountingStubDriver
}

func (hipLaunchPacketReleasingStubDriver) LaunchKernel(config hipKernelLaunchConfig) error {
	hipReleaseLaunchPacket(config.Args)
	return nil
}

type hipPackedTopKSampleStubDriver struct {
	hipLaunchPacketReleasingStubDriver
}

func (hipPackedTopKSampleStubDriver) CopyDeviceToHost(_ nativeDevicePointer, payload []byte) error {
	if len(payload) >= 8 {
		binary.LittleEndian.PutUint64(payload, hipPackGreedyBest(1, 1))
	}
	return nil
}

func (hipPackedTopKSampleStubDriver) CopyDeviceToHostUint64(nativeDevicePointer) (uint64, error) {
	return hipPackGreedyBest(1, 1), nil
}

func BenchmarkHIPMLXQ4TripleProjectionKernelWithDeviceInputViewsOutput_Hot(b *testing.B) {
	driver := hipLaunchPacketReleasingStubDriver{}
	input := &hipDeviceByteBuffer{
		driver:    driver,
		pointer:   0x1000,
		count:     16,
		sizeBytes: 64,
		borrowed:  true,
		label:     "benchmark q4 triple input",
	}
	output := &hipDeviceByteBuffer{
		driver:    driver,
		pointer:   0x2000,
		count:     24,
		sizeBytes: 96,
		borrowed:  true,
		label:     "benchmark q4 triple output",
	}
	firstCfg := hipMLXQ4DeviceWeightConfig{
		WeightPointer: 0x3000,
		ScalePointer:  0x4000,
		BiasPointer:   0x5000,
		WeightBytes:   128,
		ScaleBytes:    64,
		BiasBytes:     64,
		Rows:          16,
		Cols:          16,
		GroupSize:     8,
	}
	secondCfg := hipMLXQ4DeviceWeightConfig{
		WeightPointer: 0x6000,
		ScalePointer:  0x7000,
		BiasPointer:   0x8000,
		WeightBytes:   32,
		ScaleBytes:    16,
		BiasBytes:     16,
		Rows:          4,
		Cols:          16,
		GroupSize:     8,
	}
	thirdCfg := hipMLXQ4DeviceWeightConfig{
		WeightPointer: 0x9000,
		ScalePointer:  0xa000,
		BiasPointer:   0xb000,
		WeightBytes:   32,
		ScaleBytes:    16,
		BiasBytes:     16,
		Rows:          4,
		Cols:          16,
		GroupSize:     8,
	}
	first, second, third, err := hipRunMLXQ4TripleProjectionKernelWithDeviceInputViewsOutput(context.Background(), driver, input, firstCfg, secondCfg, thirdCfg, output)
	if err != nil {
		b.Fatal(err)
	}
	if first.Pointer() != output.Pointer() ||
		second.Pointer() != output.Pointer()+nativeDevicePointer(firstCfg.Rows*4) ||
		third.Pointer() != output.Pointer()+nativeDevicePointer((firstCfg.Rows+secondCfg.Rows)*4) {
		b.Fatalf("bad borrowed output views")
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		first, second, third, err = hipRunMLXQ4TripleProjectionKernelWithDeviceInputViewsOutput(context.Background(), driver, input, firstCfg, secondCfg, thirdCfg, output)
		if err != nil {
			b.Fatal(err)
		}
		if first.Pointer() != output.Pointer() ||
			second.Pointer() != output.Pointer()+nativeDevicePointer(firstCfg.Rows*4) ||
			third.Pointer() != output.Pointer()+nativeDevicePointer((firstCfg.Rows+secondCfg.Rows)*4) {
			b.Fatalf("bad borrowed output views")
		}
	}
}

func BenchmarkHIPMLXQ4ProjectionKernelWithDeviceInputOutputWithWorkspace_Hot(b *testing.B) {
	driver := hipLaunchPacketReleasingStubDriver{}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	input := &hipDeviceByteBuffer{driver: driver, pointer: 0x1000, count: 16, sizeBytes: 64, borrowed: true, label: "benchmark q4 projection input"}
	output := &hipDeviceByteBuffer{driver: driver, pointer: 0x2000, count: 16, sizeBytes: 64, borrowed: true, label: "benchmark q4 projection output"}
	cfg := hipMLXQ4DeviceWeightConfig{WeightPointer: 0x3000, ScalePointer: 0x4000, BiasPointer: 0x5000, WeightBytes: 128, ScaleBytes: 64, BiasBytes: 64, Rows: 16, Cols: 16, GroupSize: 8}
	if err := hipRunMLXQ4ProjectionKernelWithDeviceInputOutputWithWorkspace(context.Background(), driver, input, cfg, output, workspace); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := hipRunMLXQ4ProjectionKernelWithDeviceInputOutputWithWorkspace(context.Background(), driver, input, cfg, output, workspace); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHIPMLXQ4TripleProjectionKernelWithDeviceInputViewsOutputWithWorkspace_Hot(b *testing.B) {
	driver := hipLaunchPacketReleasingStubDriver{}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	input := &hipDeviceByteBuffer{driver: driver, pointer: 0x1000, count: 16, sizeBytes: 64, borrowed: true, label: "benchmark q4 triple input"}
	output := &hipDeviceByteBuffer{driver: driver, pointer: 0x2000, count: 24, sizeBytes: 96, borrowed: true, label: "benchmark q4 triple output"}
	firstCfg := hipMLXQ4DeviceWeightConfig{WeightPointer: 0x3000, ScalePointer: 0x4000, BiasPointer: 0x5000, WeightBytes: 128, ScaleBytes: 64, BiasBytes: 64, Rows: 16, Cols: 16, GroupSize: 8}
	secondCfg := hipMLXQ4DeviceWeightConfig{WeightPointer: 0x6000, ScalePointer: 0x7000, BiasPointer: 0x8000, WeightBytes: 32, ScaleBytes: 16, BiasBytes: 16, Rows: 4, Cols: 16, GroupSize: 8}
	thirdCfg := hipMLXQ4DeviceWeightConfig{WeightPointer: 0x9000, ScalePointer: 0xa000, BiasPointer: 0xb000, WeightBytes: 32, ScaleBytes: 16, BiasBytes: 16, Rows: 4, Cols: 16, GroupSize: 8}
	first, second, third, err := hipRunMLXQ4TripleProjectionKernelWithDeviceInputViewsOutputWithWorkspace(context.Background(), driver, input, firstCfg, secondCfg, thirdCfg, output, workspace)
	if err != nil {
		b.Fatal(err)
	}
	if first.Pointer() != output.Pointer() || second.Pointer() != output.Pointer()+nativeDevicePointer(firstCfg.Rows*4) || third.Pointer() != output.Pointer()+nativeDevicePointer((firstCfg.Rows+secondCfg.Rows)*4) {
		b.Fatalf("bad borrowed output views")
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		first, second, third, err = hipRunMLXQ4TripleProjectionKernelWithDeviceInputViewsOutputWithWorkspace(context.Background(), driver, input, firstCfg, secondCfg, thirdCfg, output, workspace)
		if err != nil {
			b.Fatal(err)
		}
		if first.Pointer() != output.Pointer() || second.Pointer() != output.Pointer()+nativeDevicePointer(firstCfg.Rows*4) || third.Pointer() != output.Pointer()+nativeDevicePointer((firstCfg.Rows+secondCfg.Rows)*4) {
			b.Fatalf("bad borrowed output views")
		}
	}
}

func BenchmarkHIPMLXQ4PairProjectionKernelWithDeviceInputViewsOutputWithWorkspace_Hot(b *testing.B) {
	driver := hipLaunchPacketReleasingStubDriver{}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	input := &hipDeviceByteBuffer{driver: driver, pointer: 0x1000, count: 16, sizeBytes: 64, borrowed: true, label: "benchmark q4 pair input"}
	output := &hipDeviceByteBuffer{driver: driver, pointer: 0x2000, count: 20, sizeBytes: 80, borrowed: true, label: "benchmark q4 pair output"}
	firstCfg := hipMLXQ4DeviceWeightConfig{WeightPointer: 0x3000, ScalePointer: 0x4000, BiasPointer: 0x5000, WeightBytes: 128, ScaleBytes: 64, BiasBytes: 64, Rows: 16, Cols: 16, GroupSize: 8}
	secondCfg := hipMLXQ4DeviceWeightConfig{WeightPointer: 0x6000, ScalePointer: 0x7000, BiasPointer: 0x8000, WeightBytes: 32, ScaleBytes: 16, BiasBytes: 16, Rows: 4, Cols: 16, GroupSize: 8}
	first, second, err := hipRunMLXQ4PairProjectionKernelWithDeviceInputViewsOutputWithWorkspace(context.Background(), driver, input, firstCfg, secondCfg, output, workspace)
	if err != nil {
		b.Fatal(err)
	}
	if first.Pointer() != output.Pointer() || second.Pointer() != output.Pointer()+nativeDevicePointer(firstCfg.Rows*4) {
		b.Fatalf("bad borrowed output views")
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		first, second, err = hipRunMLXQ4PairProjectionKernelWithDeviceInputViewsOutputWithWorkspace(context.Background(), driver, input, firstCfg, secondCfg, output, workspace)
		if err != nil {
			b.Fatal(err)
		}
		if first.Pointer() != output.Pointer() || second.Pointer() != output.Pointer()+nativeDevicePointer(firstCfg.Rows*4) {
			b.Fatalf("bad borrowed output views")
		}
	}
}

func BenchmarkHIPMLXQ4GELUTanhMultiplyKernelWithDeviceInputOutputWithWorkspace_Hot(b *testing.B) {
	driver := hipLaunchPacketReleasingStubDriver{}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	input := &hipDeviceByteBuffer{driver: driver, pointer: 0x1000, count: 16, sizeBytes: 64, borrowed: true, label: "benchmark q4 gelu multiply input"}
	output := &hipDeviceByteBuffer{driver: driver, pointer: 0x2000, count: 16, sizeBytes: 64, borrowed: true, label: "benchmark q4 gelu multiply output"}
	gateCfg := hipMLXQ4DeviceWeightConfig{WeightPointer: 0x3000, ScalePointer: 0x4000, BiasPointer: 0x5000, WeightBytes: 128, ScaleBytes: 64, BiasBytes: 64, Rows: 16, Cols: 16, GroupSize: 8}
	upCfg := hipMLXQ4DeviceWeightConfig{WeightPointer: 0x6000, ScalePointer: 0x7000, BiasPointer: 0x8000, WeightBytes: 128, ScaleBytes: 64, BiasBytes: 64, Rows: 16, Cols: 16, GroupSize: 8}
	if err := hipRunMLXQ4GELUTanhMultiplyKernelWithDeviceInputOutputWithWorkspace(context.Background(), driver, input, gateCfg, upCfg, output, workspace); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := hipRunMLXQ4GELUTanhMultiplyKernelWithDeviceInputOutputWithWorkspace(context.Background(), driver, input, gateCfg, upCfg, output, workspace); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHIPMLXQ4GELUTanhProjectionKernelWithDeviceMultiplierOutputWithWorkspace_Hot(b *testing.B) {
	driver := hipLaunchPacketReleasingStubDriver{}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	input := &hipDeviceByteBuffer{driver: driver, pointer: 0x1000, count: 16, sizeBytes: 64, borrowed: true, label: "benchmark q4 gelu projection input"}
	multiplier := &hipDeviceByteBuffer{driver: driver, pointer: 0x2000, count: 16, sizeBytes: 64, borrowed: true, label: "benchmark q4 gelu projection multiplier"}
	output := &hipDeviceByteBuffer{driver: driver, pointer: 0x3000, count: 16, sizeBytes: 64, borrowed: true, label: "benchmark q4 gelu projection output"}
	cfg := hipMLXQ4DeviceWeightConfig{WeightPointer: 0x4000, ScalePointer: 0x5000, BiasPointer: 0x6000, WeightBytes: 128, ScaleBytes: 64, BiasBytes: 64, Rows: 16, Cols: 16, GroupSize: 8}
	if err := hipRunMLXQ4GELUTanhProjectionKernelWithDeviceMultiplierOutputWithWorkspace(context.Background(), driver, input, multiplier, cfg, output, workspace); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := hipRunMLXQ4GELUTanhProjectionKernelWithDeviceMultiplierOutputWithWorkspace(context.Background(), driver, input, multiplier, cfg, output, workspace); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHIPEmbeddingLookupTokenBufferScaledOutputWithWorkspace_Hot(b *testing.B) {
	driver := hipLaunchPacketReleasingStubDriver{}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	cfg := hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: 0x3000,
		EmbeddingBytes:   256 * 16 * 2,
		VocabSize:        256,
		HiddenSize:       16,
		TableEncoding:    hipEmbeddingTableEncodingBF16,
	}
	token := &hipDeviceByteBuffer{driver: driver, pointer: 0x1000, count: 1, sizeBytes: 4, borrowed: true, label: "benchmark embedding token"}
	output := &hipDeviceByteBuffer{driver: driver, pointer: 0x2000, count: 16, sizeBytes: 16 * 4, borrowed: true, label: "benchmark embedding output"}
	if err := hipRunEmbeddingLookupKernelWithDeviceTableTokenBufferScaledOutputWithWorkspace(context.Background(), driver, cfg, token, output, 0.5, workspace); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := hipRunEmbeddingLookupKernelWithDeviceTableTokenBufferScaledOutputWithWorkspace(context.Background(), driver, cfg, token, output, 0.5, workspace); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHIPEmbeddingLookupGreedyTokenScaledOutputWithWorkspace_Hot(b *testing.B) {
	driver := hipLaunchPacketReleasingStubDriver{}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	cfg := hipDeviceEmbeddingLookupConfig{
		EmbeddingPointer: 0x3000,
		EmbeddingBytes:   256 * 16 * 2,
		VocabSize:        256,
		HiddenSize:       16,
		TableEncoding:    hipEmbeddingTableEncodingBF16,
	}
	token := &hipDeviceByteBuffer{driver: driver, pointer: 0x1000, count: 1, sizeBytes: hipMLXQ4ProjectionBestBytes, borrowed: true, label: "benchmark greedy embedding token"}
	output := &hipDeviceByteBuffer{driver: driver, pointer: 0x2000, count: 16, sizeBytes: 16 * 4, borrowed: true, label: "benchmark greedy embedding output"}
	if err := hipRunEmbeddingLookupKernelWithDeviceTableGreedyTokenScaledOutputWithWorkspace(context.Background(), driver, cfg, token, output, 0.5, workspace); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := hipRunEmbeddingLookupKernelWithDeviceTableGreedyTokenScaledOutputWithWorkspace(context.Background(), driver, cfg, token, output, 0.5, workspace); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHIPPackedTopKLaunchArgsBinary_Hot(b *testing.B) {
	inputCount := 256000
	topK := 64
	chunkCount := (inputCount + hipPackedTopKChunkSize - 1) / hipPackedTopKChunkSize
	outputCount := chunkCount * topK
	args := hipPackedTopKLaunchArgs{
		InputPointer:  0x1000,
		OutputPointer: 0x2000,
		InputCount:    inputCount,
		OutputCount:   outputCount,
		TopK:          topK,
		ChunkSize:     hipPackedTopKChunkSize,
		InputBytes:    uint64(inputCount * hipMLXQ4ProjectionBestBytes),
		OutputBytes:   uint64(outputCount * hipMLXQ4ProjectionBestBytes),
	}
	packet, err := args.Binary()
	if err != nil {
		b.Fatal(err)
	}
	hipReleaseLaunchPacket(packet)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		packet, err = args.Binary()
		if err != nil {
			b.Fatal(err)
		}
		if len(packet) != hipPackedTopKLaunchArgsBytes {
			b.Fatalf("packet len = %d, want %d", len(packet), hipPackedTopKLaunchArgsBytes)
		}
		hipReleaseLaunchPacket(packet)
	}
}

func BenchmarkHIPPackedTopKLaunchArgsBinaryInto_Hot(b *testing.B) {
	inputCount := 256000
	topK := 64
	chunkCount := (inputCount + hipPackedTopKChunkSize - 1) / hipPackedTopKChunkSize
	outputCount := chunkCount * topK
	args := hipPackedTopKLaunchArgs{
		InputPointer:  0x1000,
		OutputPointer: 0x2000,
		InputCount:    inputCount,
		OutputCount:   outputCount,
		TopK:          topK,
		ChunkSize:     hipPackedTopKChunkSize,
		InputBytes:    uint64(inputCount * hipMLXQ4ProjectionBestBytes),
		OutputBytes:   uint64(outputCount * hipMLXQ4ProjectionBestBytes),
	}
	var scratch [hipPackedTopKLaunchArgsBytes]byte
	packet, err := args.BinaryInto(scratch[:])
	if err != nil {
		b.Fatal(err)
	}
	if len(packet) != hipPackedTopKLaunchArgsBytes {
		b.Fatalf("packet len = %d, want %d", len(packet), hipPackedTopKLaunchArgsBytes)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		packet, err = args.BinaryInto(scratch[:])
		if err != nil {
			b.Fatal(err)
		}
		if len(packet) != hipPackedTopKLaunchArgsBytes {
			b.Fatalf("packet len = %d, want %d", len(packet), hipPackedTopKLaunchArgsBytes)
		}
	}
}

func BenchmarkHIPPackedTopKSampleLaunchArgsBinaryInto_Hot(b *testing.B) {
	args := hipPackedTopKSampleLaunchArgs{
		InputPointer:  0x1000,
		OutputPointer: 0x2000,
		InputCount:    64,
		TopK:          64,
		InputBytes:    64 * hipMLXQ4ProjectionBestBytes,
		OutputBytes:   hipMLXQ4ProjectionBestBytes,
		Temperature:   0.7,
		TopP:          0.95,
		Draw:          0.25,
	}
	var scratch [hipPackedTopKSampleLaunchArgsBytes]byte
	packet, err := args.BinaryInto(scratch[:])
	if err != nil {
		b.Fatal(err)
	}
	if len(packet) != hipPackedTopKSampleLaunchArgsBytes {
		b.Fatalf("packet len = %d, want %d", len(packet), hipPackedTopKSampleLaunchArgsBytes)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		packet, err = args.BinaryInto(scratch[:])
		if err != nil {
			b.Fatal(err)
		}
		if len(packet) != hipPackedTopKSampleLaunchArgsBytes {
			b.Fatalf("packet len = %d, want %d", len(packet), hipPackedTopKSampleLaunchArgsBytes)
		}
	}
}

func BenchmarkHIPVectorScaleDeviceKernelOutputWithWorkspace_Hot(b *testing.B) {
	driver := hipLaunchPacketReleasingStubDriver{}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	input := &hipDeviceByteBuffer{
		driver:    driver,
		pointer:   0x1000,
		count:     2048,
		sizeBytes: 2048 * 4,
		borrowed:  true,
		label:     "benchmark vector scale input",
	}
	output := &hipDeviceByteBuffer{
		driver:    driver,
		pointer:   0x2000,
		count:     2048,
		sizeBytes: 2048 * 4,
		borrowed:  true,
		label:     "benchmark vector scale output",
	}
	if err := hipRunVectorScaleDeviceKernelOutputWithWorkspace(context.Background(), driver, input, 0.5, output, workspace); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := hipRunVectorScaleDeviceKernelOutputWithWorkspace(context.Background(), driver, input, 0.5, output, workspace); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHIPVectorAddScaledDeviceKernelOutputWithWorkspace_Hot(b *testing.B) {
	driver := hipLaunchPacketReleasingStubDriver{}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	left := &hipDeviceByteBuffer{
		driver:    driver,
		pointer:   0x1000,
		count:     2048,
		sizeBytes: 2048 * 4,
		borrowed:  true,
		label:     "benchmark vector add-scaled left",
	}
	right := &hipDeviceByteBuffer{
		driver:    driver,
		pointer:   0x2000,
		count:     2048,
		sizeBytes: 2048 * 4,
		borrowed:  true,
		label:     "benchmark vector add-scaled right",
	}
	output := &hipDeviceByteBuffer{
		driver:    driver,
		pointer:   0x3000,
		count:     2048,
		sizeBytes: 2048 * 4,
		borrowed:  true,
		label:     "benchmark vector add-scaled output",
	}
	if err := hipRunVectorAddScaledDeviceKernelOutputWithWorkspace(context.Background(), driver, left, right, 0.5, output, workspace); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := hipRunVectorAddScaledDeviceKernelOutputWithWorkspace(context.Background(), driver, left, right, 0.5, output, workspace); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHIPRMSNormDeviceToDeviceKernelWithWorkspace_Hot(b *testing.B) {
	driver := hipLaunchPacketReleasingStubDriver{}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	cfg := hipRMSNormDeviceWeightConfig{
		WeightPointer:  0x3000,
		WeightBytes:    2048 * 2,
		Count:          2048,
		Epsilon:        1e-6,
		WeightEncoding: hipRMSNormWeightEncodingBF16,
	}
	if err := hipRunRMSNormDeviceToDeviceKernelWithWorkspace(context.Background(), driver, 0x1000, 2048*4, 0x2000, 2048*4, cfg, workspace); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := hipRunRMSNormDeviceToDeviceKernelWithWorkspace(context.Background(), driver, 0x1000, 2048*4, 0x2000, 2048*4, cfg, workspace); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHIPGemma4Q4RMSNormNoScaleDeviceKernelOutputWithWorkspace_Hot(b *testing.B) {
	driver := hipLaunchPacketReleasingStubDriver{}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	input := &hipDeviceByteBuffer{driver: driver, pointer: 0x1000, count: 512, sizeBytes: 512 * 4, borrowed: true, label: "benchmark rms no-scale input"}
	output := &hipDeviceByteBuffer{driver: driver, pointer: 0x2000, count: 512, sizeBytes: 512 * 4, borrowed: true, label: "benchmark rms no-scale output"}
	if err := hipRunGemma4Q4RMSNormNoScaleDeviceKernelOutputWithWorkspace(context.Background(), driver, input, output, 1e-6, workspace); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := hipRunGemma4Q4RMSNormNoScaleDeviceKernelOutputWithWorkspace(context.Background(), driver, input, output, 1e-6, workspace); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHIPRMSNormResidualAddScaledKernelWithWorkspace_Hot(b *testing.B) {
	driver := hipLaunchPacketReleasingStubDriver{}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	input := &hipDeviceByteBuffer{driver: driver, pointer: 0x1000, count: 2048, sizeBytes: 2048 * 4, borrowed: true, label: "benchmark rms residual input"}
	residual := &hipDeviceByteBuffer{driver: driver, pointer: 0x2000, count: 2048, sizeBytes: 2048 * 4, borrowed: true, label: "benchmark rms residual residual"}
	output := &hipDeviceByteBuffer{driver: driver, pointer: 0x3000, count: 2048, sizeBytes: 2048 * 4, borrowed: true, label: "benchmark rms residual output"}
	cfg := hipRMSNormDeviceWeightConfig{
		WeightPointer:  0x4000,
		WeightBytes:    2048 * 2,
		Count:          2048,
		Epsilon:        1e-6,
		WeightEncoding: hipRMSNormWeightEncodingBF16,
	}
	if err := hipRunRMSNormResidualAddScaledKernelWithDeviceInputWeightConfigOutputWithWorkspace(context.Background(), driver, input, residual, cfg, output, 1, workspace); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := hipRunRMSNormResidualAddScaledKernelWithDeviceInputWeightConfigOutputWithWorkspace(context.Background(), driver, input, residual, cfg, output, 1, workspace); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHIPRMSNormResidualAddNormScaledKernelWithWorkspace_Hot(b *testing.B) {
	driver := hipLaunchPacketReleasingStubDriver{}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	input := &hipDeviceByteBuffer{driver: driver, pointer: 0x1000, count: 2048, sizeBytes: 2048 * 4, borrowed: true, label: "benchmark rms residual-norm input"}
	residual := &hipDeviceByteBuffer{driver: driver, pointer: 0x2000, count: 2048, sizeBytes: 2048 * 4, borrowed: true, label: "benchmark rms residual-norm residual"}
	residualOutput := &hipDeviceByteBuffer{driver: driver, pointer: 0x3000, count: 2048, sizeBytes: 2048 * 4, borrowed: true, label: "benchmark rms residual-norm residual output"}
	normOutput := &hipDeviceByteBuffer{driver: driver, pointer: 0x4000, count: 2048, sizeBytes: 2048 * 4, borrowed: true, label: "benchmark rms residual-norm norm output"}
	residualCfg := hipRMSNormDeviceWeightConfig{WeightPointer: 0x5000, WeightBytes: 2048 * 2, Count: 2048, Epsilon: 1e-6, WeightEncoding: hipRMSNormWeightEncodingBF16}
	normCfg := hipRMSNormDeviceWeightConfig{WeightPointer: 0x6000, WeightBytes: 2048 * 2, Count: 2048, Epsilon: 1e-6, WeightEncoding: hipRMSNormWeightEncodingBF16}
	if err := hipRunRMSNormResidualAddNormScaledKernelWithDeviceInputWeightConfigOutputWithWorkspace(context.Background(), driver, input, residual, residualCfg, normCfg, residualOutput, normOutput, 1, workspace); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := hipRunRMSNormResidualAddNormScaledKernelWithDeviceInputWeightConfigOutputWithWorkspace(context.Background(), driver, input, residual, residualCfg, normCfg, residualOutput, normOutput, 1, workspace); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHIPRMSNormHeadsKernelWithWorkspace_Hot(b *testing.B) {
	driver := hipLaunchPacketReleasingStubDriver{}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	input := &hipDeviceByteBuffer{driver: driver, pointer: 0x1000, count: 2048, sizeBytes: 2048 * 4, borrowed: true, label: "benchmark rms heads input"}
	output := &hipDeviceByteBuffer{driver: driver, pointer: 0x2000, count: 2048, sizeBytes: 2048 * 4, borrowed: true, label: "benchmark rms heads output"}
	cfg := hipRMSNormDeviceWeightConfig{WeightPointer: 0x3000, WeightBytes: 512 * 2, Count: 512, Epsilon: 1e-6, WeightEncoding: hipRMSNormWeightEncodingBF16}
	if err := hipRunRMSNormHeadsKernelWithDeviceInputWeightConfigOutputWithWorkspace(context.Background(), driver, input, cfg, 4, output, workspace); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := hipRunRMSNormHeadsKernelWithDeviceInputWeightConfigOutputWithWorkspace(context.Background(), driver, input, cfg, 4, output, workspace); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHIPRMSNormRoPEHeadsKernelWithWorkspace_Hot(b *testing.B) {
	driver := hipLaunchPacketReleasingStubDriver{}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	input := &hipDeviceByteBuffer{driver: driver, pointer: 0x1000, count: 2048, sizeBytes: 2048 * 4, borrowed: true, label: "benchmark rms rope heads input"}
	output := &hipDeviceByteBuffer{driver: driver, pointer: 0x2000, count: 2048, sizeBytes: 2048 * 4, borrowed: true, label: "benchmark rms rope heads output"}
	cfg := hipRMSNormDeviceWeightConfig{WeightPointer: 0x3000, WeightBytes: 512 * 2, Count: 512, Epsilon: 1e-6, WeightEncoding: hipRMSNormWeightEncodingBF16}
	if err := hipRunRMSNormRoPEHeadsKernelWithDeviceInputWeightConfigOutputFrequencyScaleWithWorkspace(context.Background(), driver, input, cfg, 4, 17, 10000, 512, 512, 1, output, workspace); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := hipRunRMSNormRoPEHeadsKernelWithDeviceInputWeightConfigOutputFrequencyScaleWithWorkspace(context.Background(), driver, input, cfg, 4, 17, 10000, 512, 512, 1, output, workspace); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHIPPackedTopKKernelWithWorkspaceOutput_Hot(b *testing.B) {
	driver := hipLaunchPacketReleasingStubDriver{}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	inputCount := 256000
	topK := 64
	input := &hipDeviceByteBuffer{
		driver:    driver,
		pointer:   0x1000,
		count:     inputCount,
		sizeBytes: uint64(inputCount * hipMLXQ4ProjectionBestBytes),
		borrowed:  true,
		label:     "benchmark packed top-k input",
	}
	output, outputCount, err := hipRunPackedTopKKernelWithWorkspaceOutput(context.Background(), driver, input, inputCount, topK, workspace, false)
	if err != nil {
		b.Fatal(err)
	}
	wantOutputCount := hipPackedTopKOutputCount(inputCount, topK)
	if outputCount != wantOutputCount || output.Count() != wantOutputCount || output.SizeBytes() != uint64(wantOutputCount*hipMLXQ4ProjectionBestBytes) {
		b.Fatalf("packed top-k output = %d %d/%d, want %d/%d", outputCount, output.Count(), output.SizeBytes(), wantOutputCount, wantOutputCount*hipMLXQ4ProjectionBestBytes)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		output, outputCount, err = hipRunPackedTopKKernelWithWorkspaceOutput(context.Background(), driver, input, inputCount, topK, workspace, false)
		if err != nil {
			b.Fatal(err)
		}
		if outputCount != wantOutputCount || output.Count() != wantOutputCount || output.SizeBytes() != uint64(wantOutputCount*hipMLXQ4ProjectionBestBytes) {
			b.Fatalf("packed top-k output = %d %d/%d, want %d/%d", outputCount, output.Count(), output.SizeBytes(), wantOutputCount, wantOutputCount*hipMLXQ4ProjectionBestBytes)
		}
	}
}

func BenchmarkHIPPackedTopKSampleKernelWithWorkspace_Hot(b *testing.B) {
	driver := hipPackedTopKSampleStubDriver{}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	inputCount := 64
	topK := 64
	input := &hipDeviceByteBuffer{
		driver:    driver,
		pointer:   0x1000,
		count:     inputCount,
		sizeBytes: uint64(inputCount * hipMLXQ4ProjectionBestBytes),
		borrowed:  true,
		label:     "benchmark packed top-k sample input",
	}
	output := &hipDeviceByteBuffer{
		driver:    driver,
		pointer:   0x2000,
		count:     1,
		sizeBytes: hipMLXQ4ProjectionBestBytes,
		borrowed:  true,
		label:     "benchmark packed top-k sample output",
	}
	_, _, err := hipRunPackedTopKSampleKernel(context.Background(), driver, input, inputCount, topK, 0.7, 0.95, 0, 0.25, output, workspace)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _, err = hipRunPackedTopKSampleKernel(context.Background(), driver, input, inputCount, topK, 0.7, 0.95, 0, 0.25, output, workspace)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHIPMLXQ4ProjectionSoftcapSampleKernelWithWorkspace_Hot(b *testing.B) {
	driver := hipPackedTopKSampleStubDriver{}
	workspace := &hipAttentionHeadsChunkedWorkspace{}
	defer workspace.Close()
	rows := 256000
	cols := 16
	groupSize := 8
	groupsPerRow := cols / groupSize
	packedPerRow, err := hipMLXAffinePackedCols(cols, hipMLXQ4ProjectionBits)
	if err != nil {
		b.Fatal(err)
	}
	input := &hipDeviceByteBuffer{
		driver:    driver,
		pointer:   0x1000,
		count:     cols,
		sizeBytes: uint64(cols * 4),
		borrowed:  true,
		label:     "benchmark q4 sampled projection input",
	}
	best := &hipDeviceByteBuffer{
		driver:    driver,
		pointer:   0x2000,
		count:     1,
		sizeBytes: hipMLXQ4ProjectionBestBytes,
		borrowed:  true,
		label:     "benchmark q4 sampled projection best",
	}
	cfg := hipMLXQ4DeviceWeightConfig{
		WeightPointer: 0x3000,
		ScalePointer:  0x4000,
		BiasPointer:   0x5000,
		WeightBytes:   uint64(rows * packedPerRow * 4),
		ScaleBytes:    uint64(rows * groupsPerRow * 2),
		BiasBytes:     uint64(rows * groupsPerRow * 2),
		Rows:          rows,
		Cols:          cols,
		GroupSize:     groupSize,
	}
	_, _, err = hipRunMLXQ4ProjectionSoftcapSampleKernelWithDeviceInputBufferSuppress(context.Background(), driver, input, cfg, 30, 64, 0.7, 0.95, 0.25, best, nil, workspace)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _, err = hipRunMLXQ4ProjectionSoftcapSampleKernelWithDeviceInputBufferSuppress(context.Background(), driver, input, cfg, 30, 64, 0.7, 0.95, 0.25, best, nil, workspace)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHIPMLXQ4GELUTanhMultiplyLaunchArgsBinary_Hot(b *testing.B) {
	args := hipMLXQ4GELUTanhMulLaunchArgs{
		InputPointer:      0x1000,
		GateWeightPointer: 0x2000,
		GateScalePointer:  0x3000,
		GateBiasPointer:   0x4000,
		UpWeightPointer:   0x5000,
		UpScalePointer:    0x6000,
		UpBiasPointer:     0x7000,
		OutputPointer:     0x8000,
		Rows:              32,
		Cols:              16,
		GroupSize:         8,
		Bits:              hipMLXQ4ProjectionBits,
		InputBytes:        64,
		GateWeightBytes:   256,
		GateScaleBytes:    128,
		GateBiasBytes:     128,
		UpWeightBytes:     256,
		UpScaleBytes:      128,
		UpBiasBytes:       128,
		OutputBytes:       128,
	}
	packet, err := args.Binary()
	if err != nil {
		b.Fatal(err)
	}
	hipReleaseLaunchPacket(packet)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		packet, err = args.Binary()
		if err != nil {
			b.Fatal(err)
		}
		if len(packet) != hipMLXQ4GELUTanhMulLaunchArgsBytes {
			b.Fatalf("packet len = %d, want %d", len(packet), hipMLXQ4GELUTanhMulLaunchArgsBytes)
		}
		hipReleaseLaunchPacket(packet)
	}
}

func BenchmarkHIPMLXQ4GELUTanhMultiplyLaunchArgsBinaryInto_Hot(b *testing.B) {
	args := hipMLXQ4GELUTanhMulLaunchArgs{
		InputPointer:      0x1000,
		GateWeightPointer: 0x2000,
		GateScalePointer:  0x3000,
		GateBiasPointer:   0x4000,
		UpWeightPointer:   0x5000,
		UpScalePointer:    0x6000,
		UpBiasPointer:     0x7000,
		OutputPointer:     0x8000,
		Rows:              32,
		Cols:              16,
		GroupSize:         8,
		Bits:              hipMLXQ4ProjectionBits,
		InputBytes:        64,
		GateWeightBytes:   256,
		GateScaleBytes:    128,
		GateBiasBytes:     128,
		UpWeightBytes:     256,
		UpScaleBytes:      128,
		UpBiasBytes:       128,
		OutputBytes:       128,
	}
	var scratch [hipMLXQ4GELUTanhMulLaunchArgsBytes]byte
	packet, err := args.BinaryInto(scratch[:])
	if err != nil {
		b.Fatal(err)
	}
	if len(packet) != hipMLXQ4GELUTanhMulLaunchArgsBytes {
		b.Fatalf("packet len = %d, want %d", len(packet), hipMLXQ4GELUTanhMulLaunchArgsBytes)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		packet, err = args.BinaryInto(scratch[:])
		if err != nil {
			b.Fatal(err)
		}
		if len(packet) != hipMLXQ4GELUTanhMulLaunchArgsBytes {
			b.Fatalf("packet len = %d, want %d", len(packet), hipMLXQ4GELUTanhMulLaunchArgsBytes)
		}
	}
}

func BenchmarkHIPMLXQ4GELUTanhMultiplyBatchLaunchArgsBinaryInto_Hot(b *testing.B) {
	args := hipMLXQ4GELUTanhMulBatchLaunchArgs{
		InputPointer:      0x1000,
		GateWeightPointer: 0x2000,
		GateScalePointer:  0x3000,
		GateBiasPointer:   0x4000,
		UpWeightPointer:   0x5000,
		UpScalePointer:    0x6000,
		UpBiasPointer:     0x7000,
		OutputPointer:     0x8000,
		Rows:              32,
		Cols:              16,
		Batch:             8,
		GroupSize:         8,
		Bits:              hipMLXQ4ProjectionBits,
		InputBytes:        8 * 64,
		GateWeightBytes:   256,
		GateScaleBytes:    128,
		GateBiasBytes:     128,
		UpWeightBytes:     256,
		UpScaleBytes:      128,
		UpBiasBytes:       128,
		OutputBytes:       8 * 128,
	}
	var scratch [hipMLXQ4GELUTanhMulBatchLaunchArgsBytes]byte
	packet, err := args.BinaryInto(scratch[:])
	if err != nil {
		b.Fatal(err)
	}
	if len(packet) != hipMLXQ4GELUTanhMulBatchLaunchArgsBytes {
		b.Fatalf("packet len = %d, want %d", len(packet), hipMLXQ4GELUTanhMulBatchLaunchArgsBytes)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		packet, err = args.BinaryInto(scratch[:])
		if err != nil {
			b.Fatal(err)
		}
		if len(packet) != hipMLXQ4GELUTanhMulBatchLaunchArgsBytes {
			b.Fatalf("packet len = %d, want %d", len(packet), hipMLXQ4GELUTanhMulBatchLaunchArgsBytes)
		}
	}
}

func BenchmarkHIPMLXQ4GELUTanhProjectionLaunchArgsBinaryInto_Hot(b *testing.B) {
	args := hipMLXQ4GELUTanhProjLaunchArgs{
		InputPointer:      0x1000,
		WeightPointer:     0x2000,
		ScalePointer:      0x3000,
		BiasPointer:       0x4000,
		MultiplierPointer: 0x5000,
		OutputPointer:     0x6000,
		Rows:              32,
		Cols:              16,
		GroupSize:         8,
		Bits:              hipMLXQ4ProjectionBits,
		InputBytes:        64,
		WeightBytes:       256,
		ScaleBytes:        128,
		BiasBytes:         128,
		MultiplierBytes:   128,
		OutputBytes:       128,
	}
	var scratch [hipMLXQ4GELUTanhProjLaunchArgsBytes]byte
	packet, err := args.BinaryInto(scratch[:])
	if err != nil {
		b.Fatal(err)
	}
	if len(packet) != hipMLXQ4GELUTanhProjLaunchArgsBytes {
		b.Fatalf("packet len = %d, want %d", len(packet), hipMLXQ4GELUTanhProjLaunchArgsBytes)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		packet, err = args.BinaryInto(scratch[:])
		if err != nil {
			b.Fatal(err)
		}
		if len(packet) != hipMLXQ4GELUTanhProjLaunchArgsBytes {
			b.Fatalf("packet len = %d, want %d", len(packet), hipMLXQ4GELUTanhProjLaunchArgsBytes)
		}
	}
}

func BenchmarkHIPMLXQ4GELUTanhProjectionBatchLaunchArgsBinaryInto_Hot(b *testing.B) {
	args := hipMLXQ4GELUTanhProjBatchLaunchArgs{
		InputPointer:      0x1000,
		WeightPointer:     0x2000,
		ScalePointer:      0x3000,
		BiasPointer:       0x4000,
		MultiplierPointer: 0x5000,
		OutputPointer:     0x6000,
		Rows:              32,
		Cols:              16,
		Batch:             8,
		GroupSize:         8,
		Bits:              hipMLXQ4ProjectionBits,
		InputBytes:        8 * 64,
		WeightBytes:       256,
		ScaleBytes:        128,
		BiasBytes:         128,
		MultiplierBytes:   8 * 128,
		OutputBytes:       8 * 128,
	}
	var scratch [hipMLXQ4GELUTanhProjBatchLaunchArgsBytes]byte
	packet, err := args.BinaryInto(scratch[:])
	if err != nil {
		b.Fatal(err)
	}
	if len(packet) != hipMLXQ4GELUTanhProjBatchLaunchArgsBytes {
		b.Fatalf("packet len = %d, want %d", len(packet), hipMLXQ4GELUTanhProjBatchLaunchArgsBytes)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		packet, err = args.BinaryInto(scratch[:])
		if err != nil {
			b.Fatal(err)
		}
		if len(packet) != hipMLXQ4GELUTanhProjBatchLaunchArgsBytes {
			b.Fatalf("packet len = %d, want %d", len(packet), hipMLXQ4GELUTanhProjBatchLaunchArgsBytes)
		}
	}
}

func BenchmarkROCmDeviceKVPageSlicePool_ReusedCapacity(b *testing.B) {
	rocmDeviceKVPageSlicePools.Range(func(key, _ any) bool {
		rocmDeviceKVPageSlicePools.Delete(key)
		return true
	})
	pages := rocmDeviceKVBorrowPageSlice(32, rocmDeviceKVHotPageCapacity)
	rocmDeviceKVReleasePageSlice(pages)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pages = rocmDeviceKVBorrowPageSlice(32, rocmDeviceKVHotPageCapacity)
		if len(pages) != 32 || cap(pages) != rocmDeviceKVHotPageCapacity {
			b.Fatalf("page slice len/cap = %d/%d, want 32/%d", len(pages), cap(pages), rocmDeviceKVHotPageCapacity)
		}
		rocmDeviceKVReleasePageSlice(pages)
	}
}

func BenchmarkROCmDeviceKVPageSlicePool_SmallReusedCapacity(b *testing.B) {
	rocmDeviceKVPageSlicePools.Range(func(key, _ any) bool {
		rocmDeviceKVPageSlicePools.Delete(key)
		return true
	})
	pages := rocmDeviceKVBorrowPageSlice(1, 1)
	rocmDeviceKVReleasePageSlice(pages)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pages = rocmDeviceKVBorrowPageSlice(1, 1)
		if len(pages) != 1 || cap(pages) != rocmDeviceKVPagePoolMinCapacity {
			b.Fatalf("page slice len/cap = %d/%d, want 1/%d", len(pages), cap(pages), rocmDeviceKVPagePoolMinCapacity)
		}
		rocmDeviceKVReleasePageSlice(pages)
	}
}

func BenchmarkROCmDeviceKVTransferSharedPages_TrimmedSuffix(b *testing.B) {
	driver := &fakeHIPDriver{available: true}
	sourcePages := make([]rocmDeviceKVPage, rocmDeviceKVHotPageCapacity+1, rocmDeviceKVPagePoolMaxCapacity+1)
	targetPages := make([]rocmDeviceKVPage, rocmDeviceKVHotPageCapacity, rocmDeviceKVPagePoolMaxCapacity+1)
	for index := range sourcePages {
		pointerBase := nativeDevicePointer(0x100000 + index*0x100)
		sourcePages[index] = rocmDeviceKVPage{
			tokenStart: index,
			tokenCount: 1,
			keyWidth:   256,
			valueWidth: 256,
			key:        rocmDeviceKVTensor{pointer: pointerBase + 1, sizeBytes: 260, encoding: rocmKVEncodingQ8},
			value:      rocmDeviceKVTensor{pointer: pointerBase + 2, sizeBytes: 132, encoding: rocmKVEncodingQ4},
		}
	}
	for index := range targetPages {
		targetPages[index] = sourcePages[index+1]
		targetPages[index].tokenStart = index
	}
	var source rocmDeviceKVCache
	var target rocmDeviceKVCache
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		source = rocmDeviceKVCache{driver: driver, mode: rocmKVCacheModeKQ8VQ4, blockSize: 1, tokenCount: len(sourcePages), pages: sourcePages}
		target = rocmDeviceKVCache{driver: driver, mode: rocmKVCacheModeKQ8VQ4, blockSize: 1, tokenCount: len(targetPages), pages: targetPages}
		if err := source.transferSharedPagesTo(&target); err != nil {
			b.Fatal(err)
		}
	}
}

func assertGemma4Q4DeviceStateMatchesQuantizedHost(t *testing.T, cfg hipGemma4Q4ForwardConfig, hostState, restoredState hipGemma4Q4DecodeState, deviceState *hipGemma4Q4DeviceDecodeState, mode string) {
	t.Helper()
	core.AssertEqual(t, len(hostState.Layers), len(restoredState.Layers))
	if deviceState == nil {
		t.Fatalf("device state is nil")
	}
	core.AssertEqual(t, len(hostState.Layers), len(deviceState.layers))
	for index := range hostState.Layers {
		cache, err := newROCmKVCache(mode, defaultROCmKVBlockSize)
		core.RequireNoError(t, err)
		layerCfg := cfg.Layers[index]
		for _, page := range deviceState.layers[index].cache.pages {
			keyStart := page.tokenStart * layerCfg.HeadDim
			keyEnd := keyStart + page.tokenCount*layerCfg.HeadDim
			if keyStart < 0 || keyEnd > len(hostState.Layers[index].Keys) {
				t.Fatalf("device layer %d page token range [%d,%d) exceeds host key length %d", index, keyStart, keyEnd, len(hostState.Layers[index].Keys))
			}
			valueStart := page.tokenStart * layerCfg.HeadDim
			valueEnd := valueStart + page.tokenCount*layerCfg.HeadDim
			if valueStart < 0 || valueEnd > len(hostState.Layers[index].Values) {
				t.Fatalf("device layer %d page token range [%d,%d) exceeds host value length %d", index, valueStart, valueEnd, len(hostState.Layers[index].Values))
			}
			core.RequireNoError(t, cache.AppendVectors(page.tokenStart, layerCfg.HeadDim, layerCfg.HeadDim, hostState.Layers[index].Keys[keyStart:keyEnd], hostState.Layers[index].Values[valueStart:valueEnd]))
		}
		wantKeys, wantValues, err := cache.Restore(0, cache.TokenCount())
		core.RequireNoError(t, err)
		assertFloat32SlicesNearRelative(t, wantKeys, restoredState.Layers[index].Keys, 0.0001, 0.0001)
		assertFloat32SlicesNearRelative(t, wantValues, restoredState.Layers[index].Values, 0.0001, 0.0001)
	}
}

func TestHIPGemma4Q4Layer0_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup()

	_, err := hipRunGemma4Q4Layer0(context.Background(), driver, cfg, hipGemma4Q4Layer0Request{
		TokenID:  int32(cfg.VocabSize),
		Position: 1,
		RoPEBase: 10000,
		Epsilon:  1e-6,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "outside vocabulary")

	_, err = hipRunGemma4Q4Layer0(context.Background(), driver, cfg, hipGemma4Q4Layer0Request{
		TokenID:  1,
		Position: 1,
		RoPEBase: float32(math.NaN()),
		Epsilon:  1e-6,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	badCfg := cfg
	badCfg.QueryProjection.WeightPointer = 0
	_, err = hipRunGemma4Q4Layer0(context.Background(), driver, badCfg, hipGemma4Q4Layer0Request{
		TokenID:  1,
		Position: 1,
		RoPEBase: 10000,
		Epsilon:  1e-6,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "q_proj config")

	badCfg = cfg
	badCfg.Layer = -1
	_, err = hipRunGemma4Q4Layer0(context.Background(), driver, badCfg, hipGemma4Q4Layer0Request{
		TokenID:  1,
		Position: 1,
		RoPEBase: 10000,
		Epsilon:  1e-6,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "layer index")

	decodeCfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{cfg}}
	validState := hipGemma4Q4DecodeState{Layers: []hipGemma4Q4LayerKVState{{
		Keys:   make([]float32, cfg.HeadDim),
		Values: make([]float32, cfg.HeadDim),
	}}}
	_, err = hipMirrorGemma4Q4DecodeState(nil, decodeCfg, validState, "")
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "HIP driver is nil")

	_, err = hipMirrorGemma4Q4DecodeState(&fakeHIPDriver{available: false}, decodeCfg, validState, "")
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "HIP driver is not available")

	_, err = hipMirrorGemma4Q4DecodeState(driver, decodeCfg, hipGemma4Q4DecodeState{}, "")
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "decode state has no layers")

	_, err = hipMirrorGemma4Q4DecodeState(driver, decodeCfg, hipGemma4Q4DecodeState{Layers: []hipGemma4Q4LayerKVState{{}}}, "")
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "no KV tokens")

	_, err = hipMirrorGemma4Q4DecodeState(driver, decodeCfg, validState, "bad")
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "unsupported cache mode")

	deviceState, err := hipMirrorGemma4Q4DecodeState(driver, decodeCfg, validState, "")
	core.RequireNoError(t, err)
	defer deviceState.Close()
	_, err = hipRunGemma4Q4DecoderLayer(context.Background(), driver, cfg, make([]float32, cfg.HiddenSize), hipGemma4Q4DecoderLayerRequest{
		Position:          1,
		Epsilon:           1e-6,
		PriorKeys:         validState.Layers[0].Keys,
		PriorValues:       validState.Layers[0].Values,
		DeviceKVAttention: true,
		DeviceKVMode:      rocmKVCacheModeQ8,
		PriorDeviceKV:     deviceState.layerCache(0),
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "prior device KV mode mismatch")

	_, err = hipRunGemma4Q4GreedyDecode(context.Background(), driver, decodeCfg, hipGemma4Q4GreedyDecodeRequest{
		PromptTokenIDs: []int32{1},
		MaxNewTokens:   1,
		MirrorDeviceKV: true,
		DeviceKVMode:   "bad",
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "unsupported device KV cache mode")

	_, _, err = hipRunGemma4Q4SingleTokenForwardWithState(context.Background(), driver, decodeCfg, validState, hipGemma4Q4ForwardRequest{
		TokenID:          1,
		Position:         1,
		Epsilon:          1e-6,
		PriorDeviceState: &hipGemma4Q4DeviceDecodeState{},
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "prior device state requires device KV attention")

	_, _, err = hipRunGemma4Q4SingleTokenForwardWithState(context.Background(), driver, decodeCfg, validState, hipGemma4Q4ForwardRequest{
		TokenID:           1,
		Position:          1,
		Epsilon:           1e-6,
		ReturnDeviceState: true,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "returning device state requires device KV attention")

	badCfg = cfg
	badCfg.RoPEBase = -1
	_, err = hipRunGemma4Q4Layer0(context.Background(), driver, badCfg, hipGemma4Q4Layer0Request{
		TokenID:  1,
		Position: 1,
		Epsilon:  1e-6,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "layer RoPE base")

	badCfg = cfg
	badCfg.RoPERotaryDim = 3
	_, err = hipRunGemma4Q4Layer0(context.Background(), driver, badCfg, hipGemma4Q4Layer0Request{
		TokenID:  1,
		Position: 1,
		Epsilon:  1e-6,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "rotary dimension")

	badCfg = cfg
	badCfg.FinalLogitSoftcap = float32(math.Inf(1))
	_, err = hipRunGemma4Q4Layer0(context.Background(), driver, badCfg, hipGemma4Q4Layer0Request{
		TokenID:  1,
		Position: 1,
		Epsilon:  1e-6,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "softcap")

	_, err = hipRunGemma4Q4DecoderLayer(context.Background(), driver, cfg, []float32{1}, hipGemma4Q4DecoderLayerRequest{
		Position: 1,
		Epsilon:  1e-6,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "input length")

	_, err = hipRunGemma4Q4DecoderLayer(context.Background(), driver, cfg, make([]float32, cfg.HiddenSize), hipGemma4Q4DecoderLayerRequest{
		Position:    1,
		Epsilon:     1e-6,
		PriorKeys:   []float32{1},
		PriorValues: []float32{1},
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "prior key/value")

	_, err = hipRunGemma4Q4SingleTokenForward(context.Background(), driver, hipGemma4Q4ForwardConfig{}, hipGemma4Q4ForwardRequest{
		TokenID:  1,
		Position: 1,
		RoPEBase: 10000,
		Epsilon:  1e-6,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "at least one")

	_, err = hipRunGemma4Q4GreedyDecode(context.Background(), driver, hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{cfg}}, hipGemma4Q4GreedyDecodeRequest{
		MaxNewTokens: 1,
		Epsilon:      1e-6,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "prompt token")

	_, err = hipRunGemma4Q4GreedyDecode(context.Background(), driver, hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{cfg}}, hipGemma4Q4GreedyDecodeRequest{
		PromptTokenIDs: []int32{1},
		Epsilon:        1e-6,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "max new tokens")

	_, tokenPrompt, err := hipGemma4Q4TokenPromptIDs("tokens:", cfg.VocabSize)
	core.AssertEqual(t, true, tokenPrompt)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "at least one")

	_, tokenPrompt, err = hipGemma4Q4TokenPromptIDs("tokens:999", cfg.VocabSize)
	core.AssertEqual(t, true, tokenPrompt)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "outside vocabulary")

	_, textPrompt, err := hipGemma4Q4TextPromptIDs("text:", &hipLoadedModel{})
	core.AssertEqual(t, true, textPrompt)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "prompt text")
}

func TestHIPGemma4Q4PackagePrefillDecode_Bad(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	cfg, cleanup := hipGemma4Q4Layer0FixtureConfig(t, driver)
	defer cleanup()
	forwardCfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{cfg}}
	model := &hipLoadedModel{driver: driver}

	_, err := hipRunGemma4Q4PackagePrefill(context.Background(), model, forwardCfg, hipPrefillRequest{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "prompt or token IDs are required")

	_, err = hipRunGemma4Q4PackagePrefill(context.Background(), model, forwardCfg, hipPrefillRequest{
		TokenIDs:  []int32{1},
		CacheMode: "bad",
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "unsupported cache mode")

	_, err = hipRunGemma4Q4PackagePrefill(context.Background(), model, forwardCfg, hipPrefillRequest{
		TokenIDs: []int32{1},
		KeyWidth: cfg.HeadDim + 1,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "KV widths")

	_, err = hipRunGemma4Q4PackageDecode(context.Background(), model, forwardCfg, hipDecodeRequest{TokenID: 1})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "Gemma4 q4 decode state is required")

	validPrefill, err := hipRunGemma4Q4PackagePrefill(context.Background(), model, forwardCfg, hipPrefillRequest{TokenIDs: []int32{1}})
	core.RequireNoError(t, err)
	defer validPrefill.Gemma4Q4DeviceState.Close()

	_, err = hipRunGemma4Q4PackageDecode(context.Background(), model, forwardCfg, hipDecodeRequest{
		TokenID:       1,
		DeviceKVMode:  "bad",
		Gemma4Q4State: validPrefill.Gemma4Q4State,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "unsupported device KV cache mode")

	_, err = hipRunGemma4Q4PackageDecode(context.Background(), model, forwardCfg, hipDecodeRequest{
		TokenID:       1,
		Position:      -1,
		Gemma4Q4State: validPrefill.Gemma4Q4State,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "decode position")

	core.RequireNoError(t, validPrefill.Gemma4Q4DeviceState.Close())
	_, err = hipRunGemma4Q4PackageDecode(context.Background(), model, forwardCfg, hipDecodeRequest{
		TokenID:             1,
		Gemma4Q4State:       validPrefill.Gemma4Q4State,
		Gemma4Q4DeviceState: validPrefill.Gemma4Q4DeviceState,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "device decode state is closed")
}

func TestHIPSmallDecode_Bad(t *testing.T) {
	_, err := hipReferenceSmallDecode(hipSmallDecodeFixture("llama"))
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "Qwen, Gemma, or dense route")

	req := hipSmallDecodeFixture("qwen3")
	req.Position = 99
	_, err = hipReferenceSmallDecode(req)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "decode position")

	req = hipSmallDecodeFixture("qwen3")
	req.Epsilon = float32(math.NaN())
	_, err = hipReferenceSmallDecode(req)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	req = hipSmallDecodeFixture("qwen3")
	req.RoPEBase = float32(math.Inf(1))
	_, err = hipReferenceSmallDecode(req)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	req = hipSmallDecodeFixture("qwen3")
	req.QueryFP16 = req.QueryFP16[:1]
	_, err = hipRunSmallDecode(context.Background(), &fakeHIPDriver{available: true}, req)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "query projection weight length")

	_, err = hipRunSmallDecode(context.Background(), &fakeHIPDriver{}, hipSmallDecodeFixture("qwen3"))
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "HIP driver is not available")
}

func TestHIPSmallDecode_DenseQuickWinArchitectures_Good(t *testing.T) {
	for _, architecture := range []string{"mistral", "phi", "glm", "glm4", "hermes", "granite"} {
		req := hipSmallDecodeFixture(architecture)
		reference, err := hipReferenceSmallDecode(req)
		core.RequireNoError(t, err)
		core.AssertEqual(t, architecture, reference.Labels["decode_architecture"])
		core.AssertEqual(t, "dense_route", reference.Labels["decode_family"])

		driver := &fakeHIPDriver{available: true}
		got, err := hipRunSmallDecode(context.Background(), driver, req)
		core.RequireNoError(t, err)
		core.AssertEqual(t, architecture, got.Labels["decode_architecture"])
		core.AssertEqual(t, "dense_route", got.Labels["decode_family"])
		assertFloat32SlicesNear(t, reference.Logits, got.Logits, 0.0001)

		loaded, _ := hipLoadedSmallDecodeFixture(t, architecture)
		cfg, err := loaded.loadedSmallDecodeConfig()
		core.RequireNoError(t, err)
		core.AssertEqual(t, architecture, normalizeROCmArchitecture(cfg.Architecture))
		core.RequireNoError(t, loaded.Close())
	}
}

func TestHIPRuntime_LoadedSmallDecodeRequestFiniteValidation_Bad(t *testing.T) {
	loaded, _ := hipLoadedSmallDecodeFixture(t, "qwen3")
	defer loaded.Close()
	cfg, err := loaded.loadedSmallDecodeConfig()
	core.RequireNoError(t, err)
	smoke := hipSmallDecodeFixture("qwen3")

	_, err = hipRunLoadedSmallDecode(context.Background(), loaded.driver, cfg, hipLoadedSmallDecodeRequest{
		Input:       smoke.Input,
		PriorKeys:   smoke.PriorKeys,
		PriorValues: smoke.PriorValues,
		Position:    smoke.Position,
		RoPEBase:    smoke.RoPEBase,
		Epsilon:     float32(math.NaN()),
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")

	_, err = hipRunLoadedSmallDecode(context.Background(), loaded.driver, cfg, hipLoadedSmallDecodeRequest{
		Input:       smoke.Input,
		PriorKeys:   smoke.PriorKeys,
		PriorValues: smoke.PriorValues,
		Position:    smoke.Position,
		RoPEBase:    float32(math.Inf(1)),
		Epsilon:     smoke.Epsilon,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "finite")
}

func TestHIPRuntime_LoadedSmallDecodeEmbeddingReadFiniteValidation_Bad(t *testing.T) {
	loaded, driver := hipLoadedSmallDecodeFixture(t, "qwen3")
	defer loaded.Close()
	cfg, err := loaded.loadedSmallDecodeConfig()
	core.RequireNoError(t, err)

	payload, err := hipFloat32Payload([]float32{1, float32(math.NaN())})
	core.RequireNoError(t, err)
	core.RequireNoError(t, driver.CopyHostToDevice(cfg.EmbeddingPointer, payload))

	_, err = hipReadLoadedSmallEmbedding(context.Background(), driver, cfg, 0)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "embedding row values must be finite")
}

func TestHIPRuntime_LoadedEmbeddingTableFiniteValidation_Bad(t *testing.T) {
	loaded, driver := hipLoadedSmallDecodeFixture(t, "qwen3")
	defer loaded.Close()
	cfg, err := loaded.loadedEmbeddingConfig()
	core.RequireNoError(t, err)

	payload, err := hipFloat32Payload([]float32{1, float32(math.Inf(1))})
	core.RequireNoError(t, err)
	core.RequireNoError(t, driver.CopyHostToDevice(cfg.EmbeddingPointer, payload))

	_, err = loaded.loadedEmbeddingTable(cfg)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "embedding table values must be finite")
}

func TestHIPRuntime_LoadModelRunsSmallDecodeSmokeWhenHSACOConfigured_Good(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-small-decode.hsaco")
	loaded, driver := hipLoadedSmallDecodeFixture(t, "qwen3")
	defer loaded.Close()

	cfg, err := loaded.loadedSmallDecodeConfig()
	core.RequireNoError(t, err)
	smoke := hipSmallDecodeFixture("qwen3")
	want, err := hipReferenceSmallDecode(smoke)
	core.RequireNoError(t, err)
	got, err := hipRunLoadedSmallDecode(context.Background(), loaded.driver, cfg, hipLoadedSmallDecodeRequest{
		Input:       smoke.Input,
		PriorKeys:   smoke.PriorKeys,
		PriorValues: smoke.PriorValues,
		Position:    smoke.Position,
		RoPEBase:    smoke.RoPEBase,
		Epsilon:     smoke.Epsilon,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, want.TokenID, got.TokenID)
	assertFloat32Near(t, want.Score, got.Score)
	assertFloat32SlicesNear(t, want.Logits, got.Logits, 0.0001)
	assertFloat32SlicesNear(t, want.Attention, got.Attention, 0.0001)
	assertFloat32SlicesNear(t, want.UpdatedKeys, got.UpdatedKeys, 0.0001)
	assertFloat32SlicesNear(t, want.UpdatedValues, got.UpdatedValues, 0.0001)
	core.AssertEqual(t, "loaded_device", got.Labels["decode_tensor_backing"])

	cache, err := newROCmKVCache(rocmKVCacheModeFP16, defaultROCmKVBlockSize)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, smoke.HiddenSize, smoke.HiddenSize, smoke.PriorKeys, smoke.PriorValues))
	decoded, err := loaded.DecodeToken(context.Background(), hipDecodeRequest{TokenID: 2, KV: cache})
	core.RequireNoError(t, err)
	core.AssertEqual(t, int32(want.TokenID), decoded.Token.ID)
	core.AssertEqual(t, 3, decoded.KV.TokenCount())
	if decoded.KV != cache {
		t.Fatalf("decoded KV cache = %p, want original cache %p", decoded.KV, cache)
	}
	decodedKeys, decodedValues, err := decoded.KV.Restore(0, decoded.KV.TokenCount())
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, want.Logits, decoded.Logits, 0.0001)
	assertFloat32SlicesNear(t, want.UpdatedKeys, decodedKeys, 0.0005)
	assertFloat32SlicesNear(t, want.UpdatedValues, decodedValues, 0.0005)
	core.AssertEqual(t, "loaded_device", decoded.Labels["decode_tensor_backing"])
	core.AssertEqual(t, "2", decoded.Labels["decode_launch_token"])

	deviceCache, err := newROCmKVCache(rocmKVCacheModeFP16, defaultROCmKVBlockSize)
	core.RequireNoError(t, err)
	core.RequireNoError(t, deviceCache.AppendVectors(0, smoke.HiddenSize, smoke.HiddenSize, smoke.PriorKeys, smoke.PriorValues))
	deviceKV, table, err := hipMirrorTinyKV(driver, deviceCache, map[string]string{})
	core.RequireNoError(t, err)
	defer deviceKV.Close()
	defer table.Close()
	decodedWithDevice, err := loaded.DecodeToken(context.Background(), hipDecodeRequest{
		TokenID:         2,
		KV:              deviceCache,
		DeviceKV:        deviceKV,
		DescriptorTable: table,
	})
	core.RequireNoError(t, err)
	defer decodedWithDevice.DeviceKV.Close()
	defer decodedWithDevice.DescriptorTable.Close()
	if decodedWithDevice.KV == deviceCache {
		t.Fatalf("decoded device KV cache = original cache %p, want cloned host cache", deviceCache)
	}
	core.AssertEqual(t, 2, deviceCache.TokenCount())
	core.AssertEqual(t, 3, decodedWithDevice.KV.TokenCount())
	core.AssertEqual(t, 3, decodedWithDevice.DeviceKV.TokenCount())
	if !deviceKV.closed || !table.closed {
		t.Fatalf("original device resources should be closed after successful small decode device append")
	}
	deviceDecodedKeys, deviceDecodedValues, err := decodedWithDevice.KV.Restore(0, decodedWithDevice.KV.TokenCount())
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, want.Logits, decodedWithDevice.Logits, 0.0001)
	assertFloat32SlicesNear(t, want.UpdatedKeys, deviceDecodedKeys, 0.0005)
	assertFloat32SlicesNear(t, want.UpdatedValues, deviceDecodedValues, 0.0005)
	core.AssertEqual(t, "loaded_device", decodedWithDevice.Labels["decode_tensor_backing"])
	core.AssertEqual(t, "hip_device", decodedWithDevice.Labels["kv_descriptor_table"])
	core.AssertEqual(t, "append_token", decodedWithDevice.Labels["kv_device_update"])
	core.AssertEqual(t, "1", decodedWithDevice.Labels["kv_device_update_pages"])
	core.AssertEqual(t, "1", decodedWithDevice.Labels["kv_device_update_from_pages"])
	core.AssertEqual(t, "2", decodedWithDevice.Labels["kv_device_update_from_tokens"])
	core.AssertEqual(t, "2", decodedWithDevice.Labels["kv_device_update_to_pages"])
	core.AssertEqual(t, "3", decodedWithDevice.Labels["kv_device_update_to_tokens"])
	core.AssertEqual(t, "success", decodedWithDevice.Labels["kv_device_update_descriptor_refresh"])
	core.AssertEqual(t, "3", decodedWithDevice.Labels["kv_tokens"])

	for _, tt := range []struct {
		mode           string
		keyTolerance   float32
		valueTolerance float32
	}{
		{mode: rocmKVCacheModeQ8, keyTolerance: 0.01, valueTolerance: 0.03},
		{mode: rocmKVCacheModeKQ8VQ4, keyTolerance: 0.01, valueTolerance: 0.15},
	} {
		t.Run("typed-"+tt.mode, func(t *testing.T) {
			modeCache, err := newROCmKVCache(tt.mode, defaultROCmKVBlockSize)
			core.RequireNoError(t, err)
			core.RequireNoError(t, modeCache.AppendVectors(0, smoke.HiddenSize, smoke.HiddenSize, smoke.PriorKeys, smoke.PriorValues))
			modeDecoded, err := loaded.DecodeToken(context.Background(), hipDecodeRequest{TokenID: 2, KV: modeCache})
			core.RequireNoError(t, err)
			core.AssertEqual(t, int32(want.TokenID), modeDecoded.Token.ID)
			core.AssertEqual(t, 3, modeDecoded.KV.TokenCount())
			core.AssertEqual(t, tt.mode, modeDecoded.KV.Stats().CacheMode)
			modeKeys, modeValues, err := modeDecoded.KV.Restore(0, modeDecoded.KV.TokenCount())
			core.RequireNoError(t, err)
			assertFloat32SlicesNear(t, want.Logits, modeDecoded.Logits, 0.0001)
			assertFloat32SlicesNear(t, want.UpdatedKeys, modeKeys, tt.keyTolerance)
			assertFloat32SlicesNear(t, want.UpdatedValues, modeValues, tt.valueTolerance)

			modeDeviceCache, err := newROCmKVCache(tt.mode, defaultROCmKVBlockSize)
			core.RequireNoError(t, err)
			core.RequireNoError(t, modeDeviceCache.AppendVectors(0, smoke.HiddenSize, smoke.HiddenSize, smoke.PriorKeys, smoke.PriorValues))
			modeDeviceKV, modeTable, err := hipMirrorTinyKV(driver, modeDeviceCache, map[string]string{})
			core.RequireNoError(t, err)
			defer modeDeviceKV.Close()
			defer modeTable.Close()
			modeDecodedWithDevice, err := loaded.DecodeToken(context.Background(), hipDecodeRequest{
				TokenID:         2,
				KV:              modeDeviceCache,
				DeviceKV:        modeDeviceKV,
				DescriptorTable: modeTable,
			})
			core.RequireNoError(t, err)
			defer modeDecodedWithDevice.DeviceKV.Close()
			defer modeDecodedWithDevice.DescriptorTable.Close()
			core.AssertEqual(t, int32(want.TokenID), modeDecodedWithDevice.Token.ID)
			core.AssertEqual(t, 2, modeDeviceCache.TokenCount())
			core.AssertEqual(t, 3, modeDecodedWithDevice.KV.TokenCount())
			core.AssertEqual(t, 3, modeDecodedWithDevice.DeviceKV.TokenCount())
			core.AssertEqual(t, tt.mode, modeDecodedWithDevice.KV.Stats().CacheMode)
			core.AssertEqual(t, tt.mode, modeDecodedWithDevice.DeviceKV.Stats().CacheMode)
			if !modeDeviceKV.closed || !modeTable.closed {
				t.Fatalf("original %s device resources should be closed after successful small decode device append", tt.mode)
			}
			modeDeviceKeys, modeDeviceValues, err := modeDecodedWithDevice.KV.Restore(0, modeDecodedWithDevice.KV.TokenCount())
			core.RequireNoError(t, err)
			assertFloat32SlicesNear(t, want.Logits, modeDecodedWithDevice.Logits, 0.0001)
			assertFloat32SlicesNear(t, want.UpdatedKeys, modeDeviceKeys, tt.keyTolerance)
			assertFloat32SlicesNear(t, want.UpdatedValues, modeDeviceValues, tt.valueTolerance)
			core.AssertEqual(t, "hip_device", modeDecodedWithDevice.Labels["kv_descriptor_table"])
			core.AssertEqual(t, "append_token", modeDecodedWithDevice.Labels["kv_device_update"])
			core.AssertEqual(t, "2", modeDecodedWithDevice.Labels["kv_device_update_to_pages"])
			core.AssertEqual(t, "3", modeDecodedWithDevice.Labels["kv_device_update_to_tokens"])
			core.AssertEqual(t, "success", modeDecodedWithDevice.Labels["kv_device_update_descriptor_refresh"])
			core.AssertEqual(t, "3", modeDecodedWithDevice.Labels["kv_tokens"])
		})
	}

	rmsPointer := loaded.tensors["model.layers.0.input_layernorm.weight"].pointer
	queryPointer := loaded.tensors["model.layers.0.self_attn.q_proj.weight"].pointer
	lmHeadPointer := loaded.tensors["output.weight"].pointer
	var sawRMSWeight, sawQueryWeight, sawLMHead bool
	for _, launch := range driver.launches {
		switch launch.Name {
		case hipKernelNameRMSNorm:
			if nativeDevicePointer(binary.LittleEndian.Uint64(launch.Args[16:])) == rmsPointer {
				sawRMSWeight = true
			}
		case hipKernelNameProjection:
			weightPointer := nativeDevicePointer(binary.LittleEndian.Uint64(launch.Args[24:]))
			if weightPointer == queryPointer {
				sawQueryWeight = true
			}
			if weightPointer == lmHeadPointer {
				sawLMHead = true
			}
		}
	}
	core.AssertTrue(t, sawRMSWeight)
	core.AssertTrue(t, sawQueryWeight)
	core.AssertTrue(t, sawLMHead)
}

func TestHIPRuntime_LoadModelRunsSmallDecodeLoRAAdapterWhenHSACOConfigured_Good(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-small-decode-lora.hsaco")
	loaded, driver := hipLoadedSmallDecodeFixture(t, "qwen3")
	defer loaded.Close()

	status := loaded.KernelStatus()
	core.AssertEqual(t, hipKernelStatusLinked, status.LoRA)
	adapterPath := core.PathJoin(t.TempDir(), "rocm_lm_head_lora.json")
	writeTinyLoRAAdapterFile(t, adapterPath, `{
		"format":"rocm-small-lm-head-lora",
		"name":"boost-zero",
		"target":"lm_head.weight",
		"rank":1,
		"alpha":1,
		"hidden_size":2,
		"vocab_size":3,
		"lora_a":[0,0],
		"lora_b":[0,0,0],
		"bias":[10,0,0]
	}`)
	identity, err := loaded.LoadAdapter(adapterPath)
	core.RequireNoError(t, err)
	core.AssertEqual(t, rocmSmallLoRAFormat, identity.Format)
	core.AssertEqual(t, "hip_small_lm_head", identity.Labels["adapter_runtime"])
	core.AssertEqual(t, hipKernelNameLoRA, identity.Labels["lora_kernel_name"])

	smoke := hipSmallDecodeFixture("qwen3")
	want, err := hipReferenceSmallDecode(smoke)
	core.RequireNoError(t, err)
	want.Logits[0] += 10
	cache, err := newROCmKVCache(rocmKVCacheModeFP16, defaultROCmKVBlockSize)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, smoke.HiddenSize, smoke.HiddenSize, smoke.PriorKeys, smoke.PriorValues))
	decoded, err := loaded.DecodeToken(context.Background(), hipDecodeRequest{TokenID: 2, KV: cache})
	core.RequireNoError(t, err)
	core.AssertEqual(t, int32(0), decoded.Token.ID)
	core.AssertEqual(t, identity.Hash, decoded.Labels["adapter_hash"])
	core.AssertEqual(t, "hip_small_lm_head", decoded.Labels["adapter_runtime"])
	core.AssertEqual(t, hipKernelNameLoRA, decoded.Labels["lora_kernel_name"])
	core.AssertEqual(t, "experimental_qwen_gemma_small_decode", decoded.Labels["lora_model_status"])
	assertFloat32SlicesNear(t, want.Logits, decoded.Logits, 0.0001)

	var sawLoRA bool
	for _, launch := range driver.launches {
		if launch.Name == hipKernelNameLoRA {
			sawLoRA = true
		}
	}
	core.AssertTrue(t, sawLoRA)
}

func TestHIPRuntime_LoadedSmallDecodeConfig_Bad(t *testing.T) {
	t.Setenv("GO_ROCM_KERNEL_HSACO", "fake-small-decode.hsaco")

	loaded, _ := hipLoadedSmallDecodeFixture(t, "llama")
	defer loaded.Close()
	_, err := loaded.loadedSmallDecodeConfig()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "Qwen, Gemma, or dense route")

	loaded, _ = hipLoadedSmallDecodeFixture(t, "qwen3")
	defer loaded.Close()
	tensor := loaded.tensors["model.layers.0.self_attn.q_proj.weight"]
	tensor.info.Type = 0
	tensor.info.TypeName = "f32"
	loaded.tensors["model.layers.0.self_attn.q_proj.weight"] = tensor
	_, err = loaded.loadedSmallDecodeConfig()
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "query weight must be f16")

	typedLoaded, _ := hipLoadedSmallDecodeFixture(t, "qwen3")
	defer typedLoaded.Close()
	cache, err := newROCmKVCache(rocmKVCacheModeFP16, defaultROCmKVBlockSize)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 1, 1, []float32{1, 0}, []float32{1, 0}))
	_, err = typedLoaded.DecodeToken(context.Background(), hipDecodeRequest{TokenID: 2, KV: cache})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "KV widths must match hidden size")

	failingLoaded, failingDriver := hipLoadedSmallDecodeFixture(t, "qwen3")
	defer failingLoaded.Close()
	smoke := hipSmallDecodeFixture("qwen3")
	deviceCache, err := newROCmKVCache(rocmKVCacheModeFP16, defaultROCmKVBlockSize)
	core.RequireNoError(t, err)
	core.RequireNoError(t, deviceCache.AppendVectors(0, smoke.HiddenSize, smoke.HiddenSize, smoke.PriorKeys, smoke.PriorValues))
	deviceKV, table, err := hipMirrorTinyKV(failingDriver, deviceCache, map[string]string{})
	core.RequireNoError(t, err)
	defer deviceKV.Close()
	defer table.Close()
	failingDriver.copyErr = core.NewError("append copy failed")
	const smallDecodePrimitiveLaunches = 10
	failingDriver.copyHostErrAfterLaunches = len(failingDriver.launches) + smallDecodePrimitiveLaunches
	decoded, err := failingLoaded.DecodeToken(context.Background(), hipDecodeRequest{
		TokenID:         2,
		KV:              deviceCache,
		DeviceKV:        deviceKV,
		DescriptorTable: table,
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "append copy failed")
	core.AssertNil(t, decoded.KV)
	core.AssertEqual(t, 2, deviceCache.TokenCount())
	core.AssertEqual(t, 2, deviceKV.TokenCount())
	if deviceKV.closed || table.closed {
		t.Fatalf("original device resources were closed after failed small decode device append")
	}
}

func hipSmallDecodeFixture(architecture string) hipSmallDecodeRequest {
	identity := []uint16{
		0x3c00, 0,
		0, 0x3c00,
	}
	lmHead := []uint16{
		0x3c00, 0,
		0, 0x3c00,
		0x3c00, 0x3c00,
	}
	return hipSmallDecodeRequest{
		Architecture: architecture,
		Input:        []float32{1, 1},
		RMSWeight:    []float32{1, 1},
		Epsilon:      0,
		QueryFP16:    append([]uint16(nil), identity...),
		KeyFP16:      append([]uint16(nil), identity...),
		ValueFP16:    append([]uint16(nil), identity...),
		OutputFP16:   append([]uint16(nil), identity...),
		LMHeadFP16:   lmHead,
		PriorKeys: []float32{
			1, 0,
			0, 1,
		},
		PriorValues: []float32{
			1, 0,
			0, 1,
		},
		Position:   2,
		RoPEBase:   10000,
		VocabSize:  3,
		HiddenSize: 2,
	}
}

func hipLoadedSmallDecodeFixture(t *testing.T, architecture string) (*hipLoadedModel, *fakeHIPDriver) {
	t.Helper()
	payload, tensors := hipSmallDecodeModelPayload(t, architecture)
	modelPath := core.PathJoin(t.TempDir(), "small-decode.bin")
	write := core.WriteFile(modelPath, payload, 0o644)
	core.RequireTrue(t, write.OK)
	driver := &fakeHIPDriver{available: true}
	model, err := newHIPRuntime(driver).LoadModel(modelPath, nativeLoadConfig{
		ModelInfo: inference.ModelInfo{Architecture: architecture, VocabSize: 3, HiddenSize: 2, NumLayers: 1, QuantBits: 16},
		Tensors:   tensors,
	})
	core.RequireNoError(t, err)
	loaded, ok := model.(*hipLoadedModel)
	core.RequireTrue(t, ok)
	return loaded, driver
}

func hipSmallDecodeModelPayload(t *testing.T, architecture string) ([]byte, []nativeTensorInfo) {
	t.Helper()
	smoke := hipSmallDecodeFixture(architecture)
	embeddingPayload, err := hipFloat32Payload(hipReferenceTinyLMFixture().EmbeddingTable)
	core.RequireNoError(t, err)
	rmsPayload, err := hipFloat32Payload(smoke.RMSWeight)
	core.RequireNoError(t, err)
	queryPayload, err := hipUint16Payload(smoke.QueryFP16)
	core.RequireNoError(t, err)
	keyPayload, err := hipUint16Payload(smoke.KeyFP16)
	core.RequireNoError(t, err)
	valuePayload, err := hipUint16Payload(smoke.ValueFP16)
	core.RequireNoError(t, err)
	outputPayload, err := hipUint16Payload(smoke.OutputFP16)
	core.RequireNoError(t, err)
	lmHeadPayload, err := hipUint16Payload(smoke.LMHeadFP16)
	core.RequireNoError(t, err)

	var payload []byte
	var tensors []nativeTensorInfo
	appendTensor := func(name string, tensorType uint32, dimensions []uint64, tensorPayload []byte) {
		tensors = append(tensors, nativeTensorInfo{
			Name:       name,
			Type:       tensorType,
			Dimensions: dimensions,
			Offset:     uint64(len(payload)),
			ByteSize:   uint64(len(tensorPayload)),
		})
		payload = append(payload, tensorPayload...)
	}
	appendTensor("tok_embeddings.weight", 0, []uint64{3, 2}, embeddingPayload)
	appendTensor("model.layers.0.input_layernorm.weight", 0, []uint64{2}, rmsPayload)
	appendTensor("model.layers.0.self_attn.q_proj.weight", 1, []uint64{2, 2}, queryPayload)
	appendTensor("model.layers.0.self_attn.k_proj.weight", 1, []uint64{2, 2}, keyPayload)
	appendTensor("model.layers.0.self_attn.v_proj.weight", 1, []uint64{2, 2}, valuePayload)
	appendTensor("model.layers.0.self_attn.o_proj.weight", 1, []uint64{2, 2}, outputPayload)
	appendTensor("output.weight", 1, []uint64{3, 2}, lmHeadPayload)
	return payload, tensors
}

func hipGemma4Q4Layer0FixtureConfig(t *testing.T, driver nativeHIPDriver) (hipGemma4Q4Layer0Config, func()) {
	t.Helper()
	return hipGemma4Q4FixtureConfig(t, driver, 0, 8, 1, 8)
}

func hipGemma4Q4ProductionMLPFixtureConfig(t *testing.T, driver nativeHIPDriver, hidden, intermediate, groupSize int) (hipGemma4Q4Layer0Config, func()) {
	t.Helper()
	var buffers []*hipDeviceByteBuffer
	uploadU16 := func(label string, count int) *hipDeviceByteBuffer {
		t.Helper()
		payload, err := hipUint16Payload(make([]uint16, count))
		core.RequireNoError(t, err)
		buffer, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, label, payload, count)
		core.RequireNoError(t, err)
		buffers = append(buffers, buffer)
		return buffer
	}
	uploadU32 := func(label string, count int) *hipDeviceByteBuffer {
		t.Helper()
		payload, err := hipUint32Payload(make([]uint32, count))
		core.RequireNoError(t, err)
		buffer, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, label, payload, count)
		core.RequireNoError(t, err)
		buffers = append(buffers, buffer)
		return buffer
	}
	q4Projection := func(label string, rows, cols int) hipMLXQ4DeviceWeightConfig {
		t.Helper()
		weights := uploadU32(label+" weights", rows*(cols/8))
		scales := uploadU16(label+" scales", rows*(cols/groupSize))
		biases := uploadU16(label+" biases", rows*(cols/groupSize))
		return hipMLXQ4DeviceWeightConfig{
			WeightPointer: weights.Pointer(),
			ScalePointer:  scales.Pointer(),
			BiasPointer:   biases.Pointer(),
			WeightBytes:   weights.SizeBytes(),
			ScaleBytes:    scales.SizeBytes(),
			BiasBytes:     biases.SizeBytes(),
			Rows:          rows,
			Cols:          cols,
			GroupSize:     groupSize,
			Bits:          hipMLXQ4ProjectionBits,
		}
	}
	cleanup := func() {
		for index := len(buffers) - 1; index >= 0; index-- {
			_ = buffers[index].Close()
		}
	}
	cfg := hipGemma4Q4Layer0Config{
		HiddenSize:       hidden,
		IntermediateSize: intermediate,
		GroupSize:        groupSize,
		GateProjection:   q4Projection("mlp.gate_proj", intermediate, hidden),
		UpProjection:     q4Projection("mlp.up_proj", intermediate, hidden),
		DownProjection:   q4Projection("mlp.down_proj", hidden, intermediate),
	}
	cfg.finalizeScales()
	return cfg, cleanup
}

func hipGemma4Q4GlobalPerLayerInputFixture(t *testing.T, driver nativeHIPDriver, layers []hipGemma4Q4Layer0Config) ([]hipGemma4Q4Layer0Config, func()) {
	t.Helper()
	if len(layers) == 0 {
		t.Fatalf("per-layer input fixture requires layers")
	}
	hidden := layers[0].HiddenSize
	vocab := layers[0].VocabSize
	groupSize := layers[0].GroupSize
	totalHidden := hidden * len(layers)
	if hidden <= 0 || vocab <= 0 || groupSize <= 0 || totalHidden%8 != 0 || totalHidden%groupSize != 0 {
		t.Fatalf("invalid per-layer input fixture geometry hidden=%d vocab=%d group=%d layers=%d", hidden, vocab, groupSize, len(layers))
	}
	var buffers []*hipDeviceByteBuffer
	uploadU16 := func(label string, count int) *hipDeviceByteBuffer {
		t.Helper()
		payload, err := hipUint16Payload(make([]uint16, count))
		core.RequireNoError(t, err)
		buffer, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, label, payload, count)
		core.RequireNoError(t, err)
		buffers = append(buffers, buffer)
		return buffer
	}
	uploadU32 := func(label string, count int) *hipDeviceByteBuffer {
		t.Helper()
		payload, err := hipUint32Payload(make([]uint32, count))
		core.RequireNoError(t, err)
		buffer, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, label, payload, count)
		core.RequireNoError(t, err)
		buffers = append(buffers, buffer)
		return buffer
	}
	norm := func(label string, count int) hipRMSNormDeviceWeightConfig {
		t.Helper()
		buffer := uploadU16(label, count)
		return hipRMSNormDeviceWeightConfig{
			WeightPointer:  buffer.Pointer(),
			WeightBytes:    buffer.SizeBytes(),
			Count:          count,
			WeightEncoding: hipRMSNormWeightEncodingBF16,
		}
	}

	embeddingWeights := uploadU32("embed_tokens_per_layer weights", vocab*(totalHidden/8))
	embeddingScales := uploadU16("embed_tokens_per_layer scales", vocab*(totalHidden/groupSize))
	embeddingBiases := uploadU16("embed_tokens_per_layer biases", vocab*(totalHidden/groupSize))
	modelProjectionWeights := uploadU16("per_layer_model_projection weights", totalHidden*hidden)
	projectionNorm := norm("per_layer_projection_norm", hidden)
	output := append([]hipGemma4Q4Layer0Config(nil), layers...)
	for index := range output {
		perLayer := output[index].PerLayerInput
		perLayer.InputSize = hidden
		perLayer.Embedding = hipDeviceEmbeddingLookupConfig{
			EmbeddingPointer: embeddingWeights.Pointer(),
			EmbeddingBytes:   embeddingWeights.SizeBytes(),
			TableEncoding:    hipEmbeddingTableEncodingMLXQ4,
			VocabSize:        vocab,
			HiddenSize:       totalHidden,
			GroupSize:        groupSize,
			ScalePointer:     embeddingScales.Pointer(),
			BiasPointer:      embeddingBiases.Pointer(),
			ScaleBytes:       embeddingScales.SizeBytes(),
			BiasBytes:        embeddingBiases.SizeBytes(),
		}
		perLayer.ModelProjection = hipBF16DeviceWeightConfig{
			WeightPointer: modelProjectionWeights.Pointer(),
			WeightBytes:   modelProjectionWeights.SizeBytes(),
			Rows:          totalHidden,
			Cols:          hidden,
		}
		perLayer.ProjectionNorm = projectionNorm
		output[index].PerLayerInput = perLayer
		output[index].finalizeScales()
	}
	cleanup := func() {
		for index := len(buffers) - 1; index >= 0; index-- {
			_ = buffers[index].Close()
		}
	}
	return output, cleanup
}

func hipGemma4Q4FixtureConfig(t *testing.T, driver nativeHIPDriver, layer, headDim, queryHeads, intermediate int) (hipGemma4Q4Layer0Config, func()) {
	t.Helper()
	const (
		hidden    = 8
		vocab     = 2
		groupSize = 8
	)
	var buffers []*hipDeviceByteBuffer
	uploadU16 := func(label string, count int) *hipDeviceByteBuffer {
		t.Helper()
		payload, err := hipUint16Payload(make([]uint16, count))
		core.RequireNoError(t, err)
		buffer, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, label, payload, count)
		core.RequireNoError(t, err)
		buffers = append(buffers, buffer)
		return buffer
	}
	uploadU32 := func(label string, count int) *hipDeviceByteBuffer {
		t.Helper()
		payload, err := hipUint32Payload(make([]uint32, count))
		core.RequireNoError(t, err)
		buffer, err := hipUploadByteBuffer(driver, hipGemma4Q4Layer0Operation, label, payload, count)
		core.RequireNoError(t, err)
		buffers = append(buffers, buffer)
		return buffer
	}
	norm := func(label string, count int) hipRMSNormDeviceWeightConfig {
		buffer := uploadU16(label, count)
		return hipRMSNormDeviceWeightConfig{
			WeightPointer:  buffer.Pointer(),
			WeightBytes:    buffer.SizeBytes(),
			Count:          count,
			WeightEncoding: hipRMSNormWeightEncodingBF16,
		}
	}
	q4Projection := func(label string, rows, cols int) hipMLXQ4DeviceWeightConfig {
		t.Helper()
		weights := uploadU32(label+" weights", rows*(cols/8))
		scales := uploadU16(label+" scales", rows*(cols/groupSize))
		biases := uploadU16(label+" biases", rows*(cols/groupSize))
		return hipMLXQ4DeviceWeightConfig{
			WeightPointer: weights.Pointer(),
			ScalePointer:  scales.Pointer(),
			BiasPointer:   biases.Pointer(),
			WeightBytes:   weights.SizeBytes(),
			ScaleBytes:    scales.SizeBytes(),
			BiasBytes:     biases.SizeBytes(),
			Rows:          rows,
			Cols:          cols,
			GroupSize:     groupSize,
		}
	}

	embeddingWeights := uploadU32("embed_tokens weights", vocab*(hidden/8))
	embeddingScales := uploadU16("embed_tokens scales", vocab*(hidden/groupSize))
	embeddingBiases := uploadU16("embed_tokens biases", vocab*(hidden/groupSize))
	cleanup := func() {
		for index := len(buffers) - 1; index >= 0; index-- {
			_ = buffers[index].Close()
		}
	}
	cfg := hipGemma4Q4Layer0Config{
		Layer:     layer,
		LayerType: hipGemma4Q4LayerTypeFromHeadDim(headDim),
		Embedding: hipDeviceEmbeddingLookupConfig{
			EmbeddingPointer: embeddingWeights.Pointer(),
			EmbeddingBytes:   embeddingWeights.SizeBytes(),
			TableEncoding:    hipEmbeddingTableEncodingMLXQ4,
			VocabSize:        vocab,
			HiddenSize:       hidden,
			GroupSize:        groupSize,
			ScalePointer:     embeddingScales.Pointer(),
			BiasPointer:      embeddingBiases.Pointer(),
			ScaleBytes:       embeddingScales.SizeBytes(),
			BiasBytes:        embeddingBiases.SizeBytes(),
		},
		HiddenSize:        hidden,
		VocabSize:         vocab,
		GroupSize:         groupSize,
		HeadDim:           headDim,
		QueryHeads:        queryHeads,
		IntermediateSize:  intermediate,
		RoPEBase:          10000,
		RoPERotaryDim:     headDim,
		SlidingWindow:     512,
		FinalLogitSoftcap: 30,
		LayerScalar:       1,
		PerLayerInput: hipGemma4Q4PerLayerInputConfig{
			InputSize:     hidden,
			InputGate:     q4Projection("per_layer_input_gate", hidden, hidden),
			Projection:    q4Projection("per_layer_projection", hidden, hidden),
			PostInputNorm: norm("post_per_layer_input_norm", hidden),
		},
		InputNorm:           norm("input_layernorm", hidden),
		QueryNorm:           norm("q_norm", headDim),
		KeyNorm:             norm("k_norm", headDim),
		PostAttentionNorm:   norm("post_attention_layernorm", hidden),
		PreFeedForwardNorm:  norm("pre_feedforward_layernorm", hidden),
		PostFeedForwardNorm: norm("post_feedforward_layernorm", hidden),
		FinalNorm:           norm("final_norm", hidden),
		QueryProjection:     q4Projection("q_proj", queryHeads*headDim, hidden),
		KeyProjection:       q4Projection("k_proj", headDim, hidden),
		ValueProjection:     q4Projection("v_proj", headDim, hidden),
		OutputProjection:    q4Projection("o_proj", hidden, queryHeads*headDim),
		GateProjection:      q4Projection("mlp.gate_proj", intermediate, hidden),
		UpProjection:        q4Projection("mlp.up_proj", intermediate, hidden),
		DownProjection:      q4Projection("mlp.down_proj", hidden, intermediate),
		LMHeadProjection:    q4Projection("embed_tokens_lm_head", vocab, hidden),
	}
	cfg.finalizeScales()
	return cfg, cleanup
}

func TestHIPAttachedDrafterAssistantLayerPlanFor_Good_UsesQueryNormHeadDim(t *testing.T) {
	const (
		hidden       = 256
		headDim      = 512
		queryRows    = 2048
		intermediate = 2048
	)
	bf16Tensor := func(name string, dims ...uint64) hipTensor {
		count := uint64(1)
		for _, dim := range dims {
			count *= dim
		}
		return hipTensor{
			info: nativeTensorInfo{
				Name:       name,
				Dimensions: dims,
				Type:       hipNativeTensorTypeBF16,
				TypeName:   "BF16",
				ByteSize:   count * 2,
			},
			pointer: 1,
		}
	}
	bf16Linear := func(name string, rows, cols uint64) hipAttachedDrafterAssistantLinearBinding {
		return hipAttachedDrafterAssistantLinearBinding{Weight: bf16Tensor(name+".weight", rows, cols)}
	}
	layer := hipAttachedDrafterAssistantVerifierLayerBinding{
		Layer:             3,
		InputNorm:         bf16Tensor("input_layernorm", hidden),
		PostAttentionNorm: bf16Tensor("post_attention_layernorm", hidden),
		PreFeedforward:    bf16Tensor("pre_feedforward_layernorm", hidden),
		PostFeedforward:   bf16Tensor("post_feedforward_layernorm", hidden),
		QueryNorm:         bf16Tensor("q_norm", headDim),
		QueryProjection:   bf16Linear("q_proj", queryRows, hidden),
		OutputProjection:  bf16Linear("o_proj", hidden, queryRows),
		GateProjection:    bf16Linear("mlp.gate_proj", intermediate, hidden),
		UpProjection:      bf16Linear("mlp.up_proj", intermediate, hidden),
		DownProjection:    bf16Linear("mlp.down_proj", hidden, intermediate),
	}
	draft := &hipLoadedModel{gemma4TextConfig: nativeGemma4TextConfig{
		HeadDim:    hidden,
		LayerTypes: []string{"sliding_attention", "sliding_attention", "sliding_attention", "full_attention"},
	}}

	plan, err := hipAttachedDrafterAssistantLayerPlanFor(draft, layer, hidden, 0, 16)
	core.RequireNoError(t, err)

	core.AssertEqual(t, headDim, plan.HeadDim)
	core.AssertEqual(t, queryRows/headDim, plan.QueryHeads)
	core.AssertEqual(t, headDim, plan.QueryNorm.Count)
}
