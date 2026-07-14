// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"math"
	"os"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/kv"
)

func TestHIPGemma4Q4LaneForwardMatchesSerial_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer, cleanup := hipGemma4Q4FixtureConfig(t, driver, 0, 4, 2, 8)
	defer cleanup()
	hipGemma4Q4InstallNonzeroEmbeddingFixture(t, driver, &layer, "lane forward")
	layer.LayerType = "full_attention"
	layer.SlidingWindow = 0
	layer.PerLayerInput = hipGemma4Q4PerLayerInputConfig{}
	cfg := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer}}
	engineConfig := defaultHIPGemma4Q4EngineConfig()

	serial0 := hipGemma4Q4LaneForwardTestState(t, driver, cfg, []int32{0})
	serial1 := hipGemma4Q4LaneForwardTestState(t, driver, cfg, []int32{0, 1})
	want0, serial0 := hipGemma4Q4LaneForwardTestSerialAdvance(t, driver, cfg, serial0, 1, 1, engineConfig)
	want1, serial1 := hipGemma4Q4LaneForwardTestSerialAdvance(t, driver, cfg, serial1, 0, 2, engineConfig)
	defer serial0.Close()
	defer serial1.Close()

	lane0 := hipGemma4Q4LaneForwardTestState(t, driver, cfg, []int32{0})
	lane1 := hipGemma4Q4LaneForwardTestState(t, driver, cfg, []int32{0, 1})
	defer lane0.Close()
	defer lane1.Close()
	workspace := hipNewAttentionHeadsChunkedWorkspace()
	defer workspace.Close()

	start := len(driver.launches)
	got, err := hipRunGemma4Q4LaneForward(context.Background(), driver, cfg, hipGemma4Q4LaneForwardRequest{
		Tokens:       []int32{1, 0},
		Positions:    []int{1, 2},
		DeviceStates: []*hipGemma4Q4DeviceDecodeState{lane0, lane1},
		Epsilon:      1e-6,
		Mode:         rocmKVCacheModeKQ8VQ4,
		Workspace:    workspace,
		EngineConfig: engineConfig,
	})
	core.RequireNoError(t, err)
	if got == nil {
		t.Fatal("hipRunGemma4Q4LaneForward returned nil")
	}
	if len(got.Greedy) != 2 || len(got.DeviceStates) != 2 {
		t.Fatalf("lane forward result lengths greedy=%d states=%d", len(got.Greedy), len(got.DeviceStates))
	}
	defer got.DeviceStates[0].Close()
	defer got.DeviceStates[1].Close()
	core.AssertEqual(t, want0.TokenID, got.Greedy[0].TokenID)
	core.AssertEqual(t, want1.TokenID, got.Greedy[1].TokenID)
	core.AssertEqual(t, []int{2}, got.DeviceStates[0].LayerTokenCounts())
	core.AssertEqual(t, []int{3}, got.DeviceStates[1].LayerTokenCounts())
	core.AssertEqual(t, true, lane0.closed)
	core.AssertEqual(t, true, lane1.closed)

	launches := driver.launches[start:]
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameRMSNormRoPEHeadsPairLaneBatch))
	core.AssertEqual(t, 1, countLaunchName(launches, hipKernelNameAttentionHeadsLaneBatch))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameAttentionHeadsBatchCausal))
	core.AssertEqual(t, 3, countLaunchName(launches, hipKernelNameRMSNormHeads))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameRMSNorm))
	if count := countLaunchName(launches, hipKernelNameMLXQ4ProjBatch); count == 0 {
		t.Fatal("lane forward did not use batched Q4 projections")
	}
}

func TestHIPGemma4Q4LaneForwardSharedKVPLEMatchesSerial_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	layer0, cleanup0 := hipGemma4Q4FixtureConfig(t, driver, 0, 8, 1, 8)
	defer cleanup0()
	layer1, cleanup1 := hipGemma4Q4FixtureConfig(t, driver, 1, 4, 2, 8)
	defer cleanup1()
	layer2, cleanup2 := hipGemma4Q4FixtureConfig(t, driver, 2, 8, 1, 8)
	defer cleanup2()
	layer3, cleanup3 := hipGemma4Q4FixtureConfig(t, driver, 3, 4, 2, 8)
	defer cleanup3()
	hipGemma4Q4InstallNonzeroEmbeddingFixture(t, driver, &layer0, "lane forward shared PLE")
	layer0.LayerType, layer0.SlidingWindow = "sliding_attention", 1
	layer1.LayerType, layer1.SlidingWindow = "full_attention", 0
	layer2.LayerType, layer2.SlidingWindow = "sliding_attention", 1
	layer3.LayerType, layer3.SlidingWindow = "full_attention", 0
	layers, cleanupPLE := hipGemma4Q4GlobalPerLayerInputFixture(t, driver, []hipGemma4Q4Layer0Config{layer0, layer1, layer2, layer3})
	defer cleanupPLE()
	cfg := hipGemma4Q4ForwardConfig{Layers: layers, KVSharedLayers: 2}
	engineConfig := defaultHIPGemma4Q4EngineConfig()

	serial0 := hipGemma4Q4LaneForwardTestState(t, driver, cfg, []int32{0})
	serial1 := hipGemma4Q4LaneForwardTestState(t, driver, cfg, []int32{0, 1})
	want0, serial0 := hipGemma4Q4LaneForwardTestSerialAdvance(t, driver, cfg, serial0, 1, 1, engineConfig)
	want1, serial1 := hipGemma4Q4LaneForwardTestSerialAdvance(t, driver, cfg, serial1, 0, 2, engineConfig)
	defer serial0.Close()
	defer serial1.Close()

	lane0 := hipGemma4Q4LaneForwardTestState(t, driver, cfg, []int32{0})
	lane1 := hipGemma4Q4LaneForwardTestState(t, driver, cfg, []int32{0, 1})
	defer lane0.Close()
	defer lane1.Close()
	workspace := hipNewAttentionHeadsChunkedWorkspace()
	defer workspace.Close()
	start := len(driver.launches)
	got, err := hipRunGemma4Q4LaneForward(context.Background(), driver, cfg, hipGemma4Q4LaneForwardRequest{
		Tokens:       []int32{1, 0},
		Positions:    []int{1, 2},
		DeviceStates: []*hipGemma4Q4DeviceDecodeState{lane0, lane1},
		Epsilon:      1e-6,
		Mode:         rocmKVCacheModeKQ8VQ4,
		Workspace:    workspace,
		EngineConfig: engineConfig,
	})
	core.RequireNoError(t, err)
	defer got.DeviceStates[0].Close()
	defer got.DeviceStates[1].Close()
	core.AssertEqual(t, want0.TokenID, got.Greedy[0].TokenID)
	core.AssertEqual(t, want1.TokenID, got.Greedy[1].TokenID)
	core.AssertEqual(t, []int{1, 2, 1, 2}, got.DeviceStates[0].LayerTokenCounts())
	core.AssertEqual(t, []int{1, 3, 1, 3}, got.DeviceStates[1].LayerTokenCounts())
	for lane, state := range got.DeviceStates {
		core.AssertEqual(t, true, state.layers[2].borrowedCache)
		core.AssertEqual(t, true, state.layers[3].borrowedCache)
		core.AssertEqual(t, state.layers[0].cache, state.layers[2].cache)
		core.AssertEqual(t, state.layers[1].cache, state.layers[3].cache)
		if state.layers[2].descriptorTable != state.layers[0].descriptorTable || state.layers[3].descriptorTable != state.layers[1].descriptorTable {
			t.Fatalf("lane %d shared descriptor tables do not alias their owners", lane)
		}
	}

	launches := driver.launches[start:]
	core.AssertEqual(t, 2, countLaunchName(launches, hipKernelNameRMSNormRoPEHeadsPairLaneBatch))
	core.AssertEqual(t, 4, countLaunchName(launches, hipKernelNameAttentionHeadsLaneBatch))
	core.AssertEqual(t, 4, countLaunchName(launches, hipKernelNameRMSNormRoPEHeads))
	core.AssertEqual(t, 0, countLaunchName(launches, hipKernelNameAttentionHeadsBatchCausal))
}

func TestHIPGemma4Q4LaneForwardE2BHardware_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_MODEL_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_MODEL_TESTS=1 to run the HIP Gemma-4 lane-forward receipt")
	}
	modelPath := strings.TrimSpace(os.Getenv("GO_ROCM_MODEL_PATH"))
	if modelPath == "" {
		t.Skip("set GO_ROCM_MODEL_PATH to a linked Gemma-4 Q4 GGUF")
	}
	if strings.TrimSpace(os.Getenv("GO_ROCM_KERNEL_HSACO")) == "" {
		t.Skip("set GO_ROCM_KERNEL_HSACO to the linked ROCm kernels HSACO")
	}
	if !ROCmAvailable() {
		t.Skip("ROCm runtime is not available on this host")
	}

	loadedResult := (&rocmBackend{}).LoadModel(modelPath, inference.WithContextLen(256))
	if !loadedResult.OK {
		t.Fatalf("production ROCm LoadModel(%q): %v", modelPath, loadedResult.Value)
	}
	model, ok := loadedResult.Value.(*rocmModel)
	if !ok {
		t.Fatalf("production ROCm LoadModel returned %T, want *rocmModel", loadedResult.Value)
	}
	defer func() {
		if result := model.Close(); !result.OK {
			t.Errorf("Close model: %v", result.Value)
		}
	}()
	loaded, ok := model.native.(*hipLoadedModel)
	if !ok {
		t.Fatalf("production ROCm native model = %T, want *hipLoadedModel", model.native)
	}
	cfg, err := loaded.cachedGemma4Q4ForwardConfig(loaded.modelInfo.NumLayers)
	core.RequireNoError(t, err)
	engineConfig := loaded.gemma4Q4EngineConfig()
	mode, err := engineConfig.deviceKVMode()
	core.RequireNoError(t, err)

	prompt0, err := model.Tokenize("Hello")
	core.RequireNoError(t, err)
	prompt1, err := model.Tokenize("Hello, briefly explain attention.")
	core.RequireNoError(t, err)
	if len(prompt0) == 0 || len(prompt1) == 0 || len(prompt0) == len(prompt1) {
		t.Fatalf("hardware prompts must have distinct positive token counts: %d %d", len(prompt0), len(prompt1))
	}
	seed0, token0 := hipGemma4Q4LaneForwardHardwareSeed(t, loaded, cfg, prompt0, engineConfig, mode)
	seed1, token1 := hipGemma4Q4LaneForwardHardwareSeed(t, loaded, cfg, prompt1, engineConfig, mode)
	serial0, lane0 := hipGemma4Q4LaneForwardHardwareClone(t, loaded, cfg, seed0, prompt0, engineConfig, mode)
	serial1, lane1 := hipGemma4Q4LaneForwardHardwareClone(t, loaded, cfg, seed1, prompt1, engineConfig, mode)
	core.RequireNoError(t, seed0.Close())
	core.RequireNoError(t, seed1.Close())

	want0, serial0 := hipGemma4Q4LaneForwardTestSerialAdvance(t, loaded.driver, cfg, serial0, token0, len(prompt0), engineConfig)
	want1, serial1 := hipGemma4Q4LaneForwardTestSerialAdvance(t, loaded.driver, cfg, serial1, token1, len(prompt1), engineConfig)
	defer serial0.Close()
	defer serial1.Close()
	workspace := hipNewAttentionHeadsChunkedWorkspace()
	defer workspace.Close()
	got, err := hipRunGemma4Q4LaneForward(context.Background(), loaded.driver, cfg, hipGemma4Q4LaneForwardRequest{
		Tokens:       []int32{token0, token1},
		Positions:    []int{len(prompt0), len(prompt1)},
		DeviceStates: []*hipGemma4Q4DeviceDecodeState{lane0, lane1},
		Epsilon:      1e-6,
		Mode:         mode,
		Workspace:    workspace,
		EngineConfig: engineConfig,
	})
	core.RequireNoError(t, err)
	defer got.DeviceStates[0].Close()
	defer got.DeviceStates[1].Close()
	core.AssertEqual(t, want0.TokenID, got.Greedy[0].TokenID)
	core.AssertEqual(t, want1.TokenID, got.Greedy[1].TokenID)
	assertHIPGemma4Q4LaneForwardStateClose(t, serial0, got.DeviceStates[0], cfg)
	assertHIPGemma4Q4LaneForwardStateClose(t, serial1, got.DeviceStates[1], cfg)

	serialSnapshot0 := hipGemma4Q4LaneForwardHardwareSnapshot(t, serial0, cfg, append(append([]int32(nil), prompt0...), token0))
	laneSnapshot0 := hipGemma4Q4LaneForwardHardwareSnapshot(t, got.DeviceStates[0], cfg, append(append([]int32(nil), prompt0...), token0))
	assertHIPGemma4Q4LaneForwardSnapshotClose(t, serialSnapshot0, laneSnapshot0)
	serialSnapshot1 := hipGemma4Q4LaneForwardHardwareSnapshot(t, serial1, cfg, append(append([]int32(nil), prompt1...), token1))
	laneSnapshot1 := hipGemma4Q4LaneForwardHardwareSnapshot(t, got.DeviceStates[1], cfg, append(append([]int32(nil), prompt1...), token1))
	assertHIPGemma4Q4LaneForwardSnapshotClose(t, serialSnapshot1, laneSnapshot1)
}

func hipGemma4Q4LaneForwardTestState(t *testing.T, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, prompt []int32) *hipGemma4Q4DeviceDecodeState {
	t.Helper()
	forward, err := hipRunGemma4Q4PrefillForwardBatch(context.Background(), driver, cfg, prompt, 0, 1e-6, rocmKVCacheModeKQ8VQ4, nil, nil, nil)
	core.RequireNoError(t, err)
	state, err := hipGemma4Q4DeviceDecodeStateFromPrefillForward(forward, rocmKVCacheModeKQ8VQ4)
	core.RequireNoError(t, err)
	core.RequireNoError(t, forward.Close())
	return state
}

func hipGemma4Q4LaneForwardTestSerialAdvance(t *testing.T, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, previous *hipGemma4Q4DeviceDecodeState, token int32, position int, engineConfig hipGemma4Q4EngineConfig) (hipGreedySampleResult, *hipGemma4Q4DeviceDecodeState) {
	t.Helper()
	workspace := hipNewAttentionHeadsChunkedWorkspace()
	defer workspace.Close()
	priorKV := hipGemma4Q4DeviceLayerCaches(previous, nil, len(cfg.Layers))
	priorDescriptors := hipGemma4Q4DeviceLayerDescriptorTables(previous, nil, len(cfg.Layers))
	forward, err := hipRunGemma4Q4PrefillForwardBatchWithPriorDescriptorWorkspaceOutputRowWithEngineConfig(
		context.Background(), driver, cfg, []int32{token}, position, 1e-6, rocmKVCacheModeKQ8VQ4,
		priorKV, priorDescriptors, nil, nil, 0, nil, workspace, engineConfig,
	)
	core.RequireNoError(t, err)
	if len(forward.Greedy) != 1 {
		t.Fatalf("serial forward greedy outputs=%d, want 1", len(forward.Greedy))
	}
	greedy := forward.Greedy[0].Greedy
	next, err := hipGemma4Q4DeviceDecodeStateFromPrefillForward(forward, rocmKVCacheModeKQ8VQ4)
	core.RequireNoError(t, err)
	core.RequireNoError(t, forward.Close())
	core.RequireNoError(t, hipFinalizeGemma4Q4ForwardDeviceState(previous, next))
	hipReleaseClosedGemma4Q4DeviceDecodeState(previous)
	return greedy, next
}

func hipGemma4Q4LaneForwardHardwareSeed(t *testing.T, loaded *hipLoadedModel, cfg hipGemma4Q4ForwardConfig, prompt []int32, engineConfig hipGemma4Q4EngineConfig, mode string) (*hipGemma4Q4DeviceDecodeState, int32) {
	t.Helper()
	workspace := hipNewAttentionHeadsChunkedWorkspace()
	defer workspace.Close()
	forward, err := hipRunGemma4Q4PrefillForwardBatchWithPriorDescriptorWorkspaceOutputRowWithEngineConfig(
		context.Background(), loaded.driver, cfg, prompt, 0, 1e-6, mode,
		nil, nil, nil, nil, len(prompt)-1, nil, workspace, engineConfig,
	)
	core.RequireNoError(t, err)
	if len(forward.Greedy) != 1 {
		t.Fatalf("hardware seed greedy outputs=%d, want 1", len(forward.Greedy))
	}
	token := int32(forward.Greedy[0].Greedy.TokenID)
	state, err := hipGemma4Q4DeviceDecodeStateFromPrefillForward(forward, mode)
	core.RequireNoError(t, err)
	core.RequireNoError(t, forward.Close())
	return state, token
}

func hipGemma4Q4LaneForwardHardwareClone(t *testing.T, loaded *hipLoadedModel, cfg hipGemma4Q4ForwardConfig, state *hipGemma4Q4DeviceDecodeState, tokens []int32, engineConfig hipGemma4Q4EngineConfig, mode string) (*hipGemma4Q4DeviceDecodeState, *hipGemma4Q4DeviceDecodeState) {
	t.Helper()
	host, err := state.HostState()
	core.RequireNoError(t, err)
	snapshot, err := hipDecodeStateToSnapshot(host, cfg, tokens, nil, kv.CaptureOptions{})
	core.RequireNoError(t, err)
	core.RequireNoError(t, hipAttachDeviceKVPayloads(snapshot, state))
	serial, err := hipRestoreGemma4Q4DeviceDecodeState(snapshot, loaded.driver, cfg, engineConfig, mode, host)
	core.RequireNoError(t, err)
	lane, err := hipRestoreGemma4Q4DeviceDecodeState(snapshot, loaded.driver, cfg, engineConfig, mode, host)
	if err != nil {
		_ = serial.Close()
		core.RequireNoError(t, err)
	}
	return serial, lane
}

func hipGemma4Q4LaneForwardHardwareSnapshot(t *testing.T, state *hipGemma4Q4DeviceDecodeState, cfg hipGemma4Q4ForwardConfig, tokens []int32) *kv.Snapshot {
	t.Helper()
	host, err := state.HostState()
	core.RequireNoError(t, err)
	snapshot, err := hipDecodeStateToSnapshot(host, cfg, tokens, nil, kv.CaptureOptions{})
	core.RequireNoError(t, err)
	core.RequireNoError(t, hipAttachDeviceKVPayloads(snapshot, state))
	return snapshot
}

func assertHIPGemma4Q4LaneForwardStateClose(t *testing.T, wantState, gotState *hipGemma4Q4DeviceDecodeState, cfg hipGemma4Q4ForwardConfig) {
	t.Helper()
	want, err := wantState.HostState()
	core.RequireNoError(t, err)
	got, err := gotState.HostState()
	core.RequireNoError(t, err)
	if len(want.Layers) != len(got.Layers) || len(want.Layers) != len(cfg.Layers) {
		t.Fatalf("lane host state layers=%d, want %d", len(got.Layers), len(want.Layers))
	}
	var maximum float32
	for layer := range want.Layers {
		width := cfg.Layers[layer].keyValueDim()
		if len(want.Layers[layer].Keys) != len(got.Layers[layer].Keys) || len(want.Layers[layer].Values) != len(got.Layers[layer].Values) || len(want.Layers[layer].Keys) < width || len(want.Layers[layer].Values) < width {
			t.Fatalf("lane host state layer %d shape mismatch", layer)
		}
		wantKey := want.Layers[layer].Keys[len(want.Layers[layer].Keys)-width:]
		gotKey := got.Layers[layer].Keys[len(got.Layers[layer].Keys)-width:]
		wantValue := want.Layers[layer].Values[len(want.Layers[layer].Values)-width:]
		gotValue := got.Layers[layer].Values[len(got.Layers[layer].Values)-width:]
		keyMax := hipGemma4Q4LaneForwardVectorDelta(wantKey, gotKey)
		valueMax := hipGemma4Q4LaneForwardVectorDelta(wantValue, gotValue)
		keyTolerance := hipGemma4Q4LaneForwardQuantizedTolerance(wantKey, gotKey, 127)
		valueTolerance := hipGemma4Q4LaneForwardQuantizedTolerance(wantValue, gotValue, 7)
		keyCosine := hipGemma4Q4LaneForwardVectorCosine(wantKey, gotKey)
		valueCosine := hipGemma4Q4LaneForwardVectorCosine(wantValue, gotValue)
		if keyMax > maximum {
			maximum = keyMax
		}
		if valueMax > maximum {
			maximum = valueMax
		}
		if keyMax > keyTolerance || valueMax > valueTolerance {
			t.Logf("lane host state layer %d quantized drift: key=%g/%g cosine=%g value=%g/%g cosine=%g", layer, keyMax, keyTolerance, keyCosine, valueMax, valueTolerance, valueCosine)
		}
		if math.IsNaN(keyCosine) || math.IsInf(keyCosine, 0) || math.IsNaN(valueCosine) || math.IsInf(valueCosine, 0) {
			t.Fatalf("lane host state layer %d contains non-finite vector similarity", layer)
		}
	}
	t.Logf("lane host state maximum appended-row delta=%g", maximum)
}

func hipGemma4Q4LaneForwardVectorCosine(want, got []float32) float64 {
	var dot, leftSquared, rightSquared float64
	for index := range want {
		left := float64(want[index])
		right := float64(got[index])
		dot += left * right
		leftSquared += left * left
		rightSquared += right * right
	}
	if leftSquared == 0 || rightSquared == 0 {
		if leftSquared == rightSquared {
			return 1
		}
		return 0
	}
	return dot / math.Sqrt(leftSquared*rightSquared)
}

func hipGemma4Q4LaneForwardQuantizedTolerance(want, got []float32, levels float32) float32 {
	maximum := float32(0)
	for index := range want {
		left := want[index]
		if left < 0 {
			left = -left
		}
		right := got[index]
		if right < 0 {
			right = -right
		}
		if left > maximum {
			maximum = left
		}
		if right > maximum {
			maximum = right
		}
	}
	return 2*maximum/levels + 1e-4
}

func hipGemma4Q4LaneForwardVectorDelta(want, got []float32) float32 {
	var maximum float32
	for index := range want {
		delta := want[index] - got[index]
		if delta < 0 {
			delta = -delta
		}
		if delta > maximum {
			maximum = delta
		}
	}
	return maximum
}

func assertHIPGemma4Q4LaneForwardSnapshotClose(t *testing.T, want, got *kv.Snapshot) {
	t.Helper()
	if got == nil || want.Architecture != got.Architecture || want.SeqLen != got.SeqLen || want.NumLayers != got.NumLayers || len(want.Layers) != len(got.Layers) {
		t.Fatalf("lane snapshot geometry mismatch")
	}
	if len(want.Tokens) != len(got.Tokens) {
		t.Fatalf("lane snapshot token count=%d, want %d", len(got.Tokens), len(want.Tokens))
	}
	for index := range want.Tokens {
		if want.Tokens[index] != got.Tokens[index] {
			t.Fatalf("lane snapshot token %d=%d, want %d", index, got.Tokens[index], want.Tokens[index])
		}
	}
	totalChanged := 0
	for layer := range want.Layers {
		if want.Layers[layer].CacheIndex != got.Layers[layer].CacheIndex || want.Layers[layer].CacheMode != got.Layers[layer].CacheMode || len(want.Layers[layer].TurboQuantPayloads) != len(got.Layers[layer].TurboQuantPayloads) {
			t.Fatalf("lane snapshot layer %d page geometry mismatch", layer)
		}
		for page := range want.Layers[layer].TurboQuantPayloads {
			left := want.Layers[layer].TurboQuantPayloads[page]
			right := got.Layers[layer].TurboQuantPayloads[page]
			if len(left) != len(right) {
				t.Fatalf("lane snapshot layer %d page %d bytes=%d, want %d", layer, page, len(right), len(left))
			}
			changed := 0
			for index := range left {
				if left[index] != right[index] {
					changed++
				}
			}
			if page+1 < len(want.Layers[layer].TurboQuantPayloads) && changed != 0 {
				t.Fatalf("lane snapshot inherited layer %d page %d changed by %d bytes", layer, page, changed)
			}
			totalChanged += changed
		}
	}
	t.Logf("lane snapshot appended pages changed by %d encoded bytes", totalChanged)
}
