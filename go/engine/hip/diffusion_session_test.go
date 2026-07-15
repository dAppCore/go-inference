// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/inference/model"
)

type rocmDiffusionSessionTestDouble struct {
	cacheOffset int
	prefills    [][]int32
	requests    []ROCmDiffusionDenoiseRequest
	truncates   []int
	commits     [][]int32
}

func (s *rocmDiffusionSessionTestDouble) PrefillTokens(ids []int32) (int, error) {
	s.prefills = append(s.prefills, append([]int32(nil), ids...))
	return len(ids), nil
}

func (s *rocmDiffusionSessionTestDouble) CacheOffset() int {
	if s.cacheOffset > 0 {
		return s.cacheOffset
	}
	return 1
}

func (s *rocmDiffusionSessionTestDouble) Denoise(_ context.Context, req ROCmDiffusionDenoiseRequest) (ROCmDiffusionStepResult, error) {
	s.requests = append(s.requests, req)
	return ROCmDiffusionStepResult{
		Canvas:      []int32{7, 9},
		Greedy:      []int32{7, 9},
		MeanEntropy: 0,
	}, nil
}

func (s *rocmDiffusionSessionTestDouble) TruncateTo(pos int) error {
	s.truncates = append(s.truncates, pos)
	return nil
}

func (s *rocmDiffusionSessionTestDouble) CommitTokens(ids []int32) error {
	s.commits = append(s.commits, append([]int32(nil), ids...))
	return nil
}

func TestRunROCmDiffusionGenerate_Good(t *testing.T) {
	session := &rocmDiffusionSessionTestDouble{}
	var yielded []int32
	metrics, err := RunROCmDiffusionGenerate(context.Background(), ROCmDiffusionGenerateConfig{
		CanvasLength:        2,
		MaxSteps:            2,
		StabilityThreshold:  1,
		ConfidenceThreshold: 0.005,
		MaxCanvases:         2,
		TextVocabSize:       16,
		StopTokens:          []int32{9},
		Seed:                7,
	}, session, []int32{1}, func(id int32) bool {
		yielded = append(yielded, id)
		return true
	})
	if err != nil {
		t.Fatalf("RunROCmDiffusionGenerate: %v", err)
	}
	if metrics.Canvases != 1 || metrics.TotalSteps != 2 || metrics.EmittedTokens != 1 || !metrics.StoppedOnToken {
		t.Fatalf("metrics = %+v, want one stopped canvas with two denoise steps and one emitted token", metrics)
	}
	if len(session.prefills) != 1 || len(session.prefills[0]) != 1 || session.prefills[0][0] != 1 {
		t.Fatalf("prefills = %#v, want prompt [1]", session.prefills)
	}
	if len(session.requests) != 2 {
		t.Fatalf("denoise requests = %d, want 2", len(session.requests))
	}
	for _, req := range session.requests {
		if req.Prefix != 1 || len(req.Canvas) != 2 || len(req.GlobalMask) != 6 || len(req.LocalMask) != 6 {
			t.Fatalf("denoise request = %+v, want prefix-1 canvas and 2x3 masks", req)
		}
	}
	if len(session.truncates) != 2 || session.truncates[0] != 1 || session.truncates[1] != 1 {
		t.Fatalf("truncates = %v, want [1 1]", session.truncates)
	}
	if len(session.commits) != 1 || len(session.commits[0]) != 1 || session.commits[0][0] != 7 {
		t.Fatalf("commits = %#v, want [[7]]", session.commits)
	}
	if len(yielded) != 1 || yielded[0] != 7 {
		t.Fatalf("yielded = %v, want [7]", yielded)
	}
}

func TestRunROCmDiffusionGenerate_Bad(t *testing.T) {
	_, err := RunROCmDiffusionGenerate(context.Background(), ROCmDiffusionGenerateConfig{TextVocabSize: 16}, nil, []int32{1}, nil)
	if err == nil {
		t.Fatal("RunROCmDiffusionGenerate(nil session) error = nil")
	}
}

func TestRunROCmDiffusionGenerate_Ugly(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	session := &rocmDiffusionSessionTestDouble{}
	_, err := RunROCmDiffusionGenerate(ctx, ROCmDiffusionGenerateConfig{TextVocabSize: 16}, session, []int32{1}, nil)
	if err == nil {
		t.Fatal("RunROCmDiffusionGenerate(cancelled context) error = nil")
	}
	if len(session.prefills) != 0 {
		t.Fatalf("prefills = %#v, want no work after cancellation", session.prefills)
	}
}

func TestRunROCmDiffusionGenerate_UsesPerCanvasSeed_Good(t *testing.T) {
	session := &rocmDiffusionSessionTestDouble{}
	_, err := RunROCmDiffusionGenerate(context.Background(), ROCmDiffusionGenerateConfig{
		CanvasLength:  2,
		MaxTokens:     4,
		MaxSteps:      1,
		MaxCanvases:   2,
		TextVocabSize: 16,
		Seed:          41,
	}, session, []int32{1}, nil)
	if err != nil {
		t.Fatalf("RunROCmDiffusionGenerate: %v", err)
	}
	if len(session.requests) != 2 {
		t.Fatalf("denoise requests = %d, want 2", len(session.requests))
	}
	if session.requests[0].StepConfig.Seed != 41 || session.requests[1].StepConfig.Seed != 41+rocmDiffusionCanvasSeedStride {
		t.Fatalf("canvas seeds = [%d %d], want deterministic per-canvas stride", session.requests[0].StepConfig.Seed, session.requests[1].StepConfig.Seed)
	}
}

func TestROCmDiffusionInitialCanvas_MatchesSharedSampler_Good(t *testing.T) {
	const (
		seed        = uint64(73)
		canvasIndex = 2
		vocab       = int32(31)
		length      = 8
	)
	sampler := model.NewSampler(seed ^ (uint64(canvasIndex+1) << 32))
	want := make([]int32, length)
	for index := range want {
		want[index] = int32(sampler.Draw() * float32(vocab))
		if want[index] >= vocab {
			want[index] = vocab - 1
		}
	}
	got := rocmDiffusionInitialCanvas(length, vocab, seed, canvasIndex)
	for index := range want {
		if got[index] != want[index] {
			t.Fatalf("canvas[%d] = %d, want %d", index, got[index], want[index])
		}
	}
}

func TestROCmDiffusionSampleDenoiseStep_Good(t *testing.T) {
	encoded := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	result, err := rocmDiffusionSampleDenoiseStep(
		[]float32{2, 1, 0, 0, 1, 2},
		[]int32{0, 2},
		3,
		2,
		1,
		0.5,
		ROCmDiffusionStepConfig{
			EntropyBound:   0.3,
			MaxTemperature: 0.8,
			MinTemperature: 0.4,
			Exponent:       1,
			TextVocabSize:  3,
			Seed:           11,
		},
		func(probabilities []float32) ([]byte, error) {
			if len(probabilities) != 6 {
				t.Fatalf("probabilities = %d, want 6", len(probabilities))
			}
			for row := 0; row < 2; row++ {
				var sum float32
				for _, probability := range probabilities[row*3 : (row+1)*3] {
					sum += probability
				}
				if math.Abs(float64(sum-1)) > 1e-5 {
					t.Fatalf("probability row %d sum = %g, want 1", row, sum)
				}
			}
			return encoded, nil
		},
	)
	if err != nil {
		t.Fatalf("rocmDiffusionSampleDenoiseStep: %v", err)
	}
	if len(result.Canvas) != 2 || len(result.Greedy) != 2 || result.Greedy[0] != 0 || result.Greedy[1] != 2 {
		t.Fatalf("sample result = %+v, want two rows with greedy [0 2]", result)
	}
	if string(result.SCEmb) != string(encoded) || result.Accepted <= 0 || result.Accepted > 2 || result.MeanEntropy <= 0 {
		t.Fatalf("sample result = %+v, want encoded self-conditioning and bounded acceptance", result)
	}
}

func TestROCmDiffusionSampleDenoiseStep_Bad(t *testing.T) {
	_, err := rocmDiffusionSampleDenoiseStep([]float32{1}, []int32{0}, 2, 1, 0, 0.5, ROCmDiffusionStepConfig{TextVocabSize: 2}, func([]float32) ([]byte, error) {
		return nil, nil
	})
	if err == nil {
		t.Fatal("rocmDiffusionSampleDenoiseStep(mismatched logits) error = nil")
	}
}

func TestROCmDiffusionSampleDenoiseStep_Ugly(t *testing.T) {
	result, err := rocmDiffusionSampleDenoiseStep(nil, nil, 2, 1, 0, 0.5, ROCmDiffusionStepConfig{TextVocabSize: 2}, func([]float32) ([]byte, error) {
		t.Fatal("empty canvas invoked encoder")
		return nil, nil
	})
	if err != nil || len(result.Canvas) != 0 || len(result.Greedy) != 0 || len(result.SCEmb) != 0 {
		t.Fatalf("empty sample result = %+v, err=%v", result, err)
	}
}

func TestLoadedDiffusionGemmaEncoderLayerScalar_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	pointer, err := driver.Malloc(2)
	if err != nil {
		t.Fatalf("Allocate: %v", err)
	}
	defer func() { _ = driver.Free(pointer) }()
	payload := make([]byte, 2)
	binary.LittleEndian.PutUint16(payload, hipFloat32ToBFloat16(0.75))
	if err := driver.CopyHostToDevice(pointer, payload); err != nil {
		t.Fatalf("CopyHostToDevice: %v", err)
	}
	loaded := &hipLoadedModel{driver: driver, tensors: map[string]hipTensor{
		"model.encoder.language_model.layers.0.layer_scalar": {
			pointer: pointer,
			info: nativeTensorInfo{
				Name:       "model.encoder.language_model.layers.0.layer_scalar",
				Dimensions: []uint64{1},
				TypeName:   "BF16",
				ByteSize:   2,
			},
		},
	}}
	scalar, err := loaded.loadedDiffusionGemmaEncoderLayerScalar(0)
	if err != nil {
		t.Fatalf("loadedDiffusionGemmaEncoderLayerScalar: %v", err)
	}
	if scalar != hipBFloat16ToFloat32(hipFloat32ToBFloat16(0.75)) {
		t.Fatalf("scalar = %g, want 0.75 BF16", scalar)
	}
}

func TestHIPDiffusionDenoiseForwardConfig_Good(t *testing.T) {
	base := hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{
		{Layer: 0, SlidingWindow: 128, LayerScalar: 1},
		{Layer: 1, SlidingWindow: 0, LayerScalar: 1},
	}}
	got, err := hipDiffusionDenoiseForwardConfig(base, []float32{0.25, 0.75}, 64)
	if err != nil {
		t.Fatalf("hipDiffusionDenoiseForwardConfig: %v", err)
	}
	if got.Layers[0].SlidingWindow != 192 || got.Layers[1].SlidingWindow != 0 ||
		got.Layers[0].LayerScalar != 0.25 || got.Layers[1].LayerScalar != 0.75 {
		t.Fatalf("denoise config = %+v, want widened local window and encoder scalars", got.Layers)
	}
	if base.Layers[0].SlidingWindow != 128 || base.Layers[0].LayerScalar != 1 {
		t.Fatalf("base config was mutated: %+v", base.Layers[0])
	}
}

func TestHIPDiffusionDenoiseForwardConfig_Bad(t *testing.T) {
	_, err := hipDiffusionDenoiseForwardConfig(hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{{}}}, nil, 64)
	if err == nil {
		t.Fatal("hipDiffusionDenoiseForwardConfig(missing scalar) error = nil")
	}
}

func TestHIPDiffusionDenoiseForwardConfig_Ugly(t *testing.T) {
	_, err := hipDiffusionDenoiseForwardConfig(hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{{}}}, []float32{1}, 0)
	if err == nil {
		t.Fatal("hipDiffusionDenoiseForwardConfig(zero canvas) error = nil")
	}
}

type rocmDiffusionKernelTestDouble struct {
	hipKernelStub
	session *rocmDiffusionSessionTestDouble
}

func (k rocmDiffusionKernelTestDouble) OpenROCmDiffusionSession(context.Context, *hipLoadedModel) (ROCmDiffusionSession, error) {
	return k.session, nil
}

func TestROCmModelGenerateBlockDiffusionTokens_Good(t *testing.T) {
	session := &rocmDiffusionSessionTestDouble{}
	loaded := &hipLoadedModel{
		kernels: rocmDiffusionKernelTestDouble{session: session},
		modelInfo: inference.ModelInfo{
			Architecture: "diffusion_gemma",
			VocabSize:    16,
		},
	}
	model := &rocmModel{native: loaded, modelInfo: loaded.modelInfo, modelLabels: map[string]string{
		"diffusion_canvas_length":       "2",
		"diffusion_default_max_steps":   "2",
		"diffusion_stability_threshold": "1",
	}}
	metrics, err := model.GenerateBlockDiffusionTokens(context.Background(), []int32{1}, ROCmBlockDiffusionOptions{MaxTokens: 1, StopTokens: []int32{9}}, nil)
	if err != nil {
		t.Fatalf("GenerateBlockDiffusionTokens: %v", err)
	}
	if metrics.EmittedTokens != 1 || len(session.prefills) != 1 || len(session.commits) != 1 {
		t.Fatalf("dispatch metrics/session = %+v/%#v, want reachable loaded-model diffusion path", metrics, session)
	}
}

func TestROCmModelGenerateBlockDiffusionTokens_DefaultCanvas_Ugly(t *testing.T) {
	session := &rocmDiffusionSessionTestDouble{}
	loaded := &hipLoadedModel{
		kernels: rocmDiffusionKernelTestDouble{session: session},
		modelInfo: inference.ModelInfo{
			Architecture: "diffusion_gemma",
			VocabSize:    256,
		},
	}
	model := &rocmModel{native: loaded, modelInfo: loaded.modelInfo}
	_, err := model.GenerateBlockDiffusionTokens(context.Background(), []int32{1}, ROCmBlockDiffusionOptions{MaxTokens: 128, StopTokens: []int32{9}}, nil)
	if err != nil {
		t.Fatalf("GenerateBlockDiffusionTokens: %v", err)
	}
	if len(session.requests) == 0 || len(session.requests[0].Canvas) != rocmDiffusionDefaultCanvasLength {
		t.Fatalf("default canvas length = %d, want %d", len(session.requests[0].Canvas), rocmDiffusionDefaultCanvasLength)
	}
}

func TestROCmModelGenerateBlockDiffusionTokens_RespectsTokenBudget_Ugly(t *testing.T) {
	session := &rocmDiffusionSessionTestDouble{}
	loaded := &hipLoadedModel{
		kernels: rocmDiffusionKernelTestDouble{session: session},
		modelInfo: inference.ModelInfo{
			Architecture: "diffusion_gemma",
			VocabSize:    16,
		},
	}
	model := &rocmModel{native: loaded, modelInfo: loaded.modelInfo, modelLabels: map[string]string{
		"diffusion_canvas_length":       "2",
		"diffusion_default_max_steps":   "2",
		"diffusion_stability_threshold": "1",
	}}
	metrics, err := model.GenerateBlockDiffusionTokens(context.Background(), []int32{1}, ROCmBlockDiffusionOptions{MaxTokens: 1}, nil)
	if err != nil {
		t.Fatalf("GenerateBlockDiffusionTokens: %v", err)
	}
	if metrics.EmittedTokens != 1 || len(session.commits) != 1 || len(session.commits[0]) != 1 {
		t.Fatalf("budgeted metrics/commits = %+v/%#v, want exactly one committed token", metrics, session.commits)
	}
}

func TestROCmModelGenerateBlockDiffusionTokens_UsesAttentionWindow_Good(t *testing.T) {
	session := &rocmDiffusionSessionTestDouble{cacheOffset: 3}
	loaded := &hipLoadedModel{
		kernels: rocmDiffusionKernelTestDouble{session: session},
		modelInfo: inference.ModelInfo{
			Architecture: "diffusion_gemma",
			VocabSize:    16,
		},
	}
	model := &rocmModel{native: loaded, modelInfo: loaded.modelInfo, modelLabels: map[string]string{
		"diffusion_canvas_length":       "2",
		"diffusion_default_max_steps":   "2",
		"diffusion_stability_threshold": "1",
		"attention_sliding_window":      "1",
	}}
	_, err := model.GenerateBlockDiffusionTokens(context.Background(), []int32{1}, ROCmBlockDiffusionOptions{MaxTokens: 1, StopTokens: []int32{9}}, nil)
	if err != nil {
		t.Fatalf("GenerateBlockDiffusionTokens: %v", err)
	}
	if len(session.requests) == 0 || !math.IsInf(float64(session.requests[0].LocalMask[0]), -1) || !math.IsInf(float64(session.requests[0].LocalMask[1]), -1) {
		t.Fatalf("local mask = %v, want prefix rows outside the one-token window blocked", session.requests[0].LocalMask)
	}
}

func TestHIPNativeProjectionKernelSet_DiffusionProvider_Good(t *testing.T) {
	provider, ok := any(hipNativeProjectionKernelSet{}).(hipROCmDiffusionSessionProvider)
	if !ok {
		t.Fatal("production HIP kernel set does not provide DiffusionGemma sessions")
	}
	_, err := provider.OpenROCmDiffusionSession(context.Background(), &hipLoadedModel{
		modelInfo: inference.ModelInfo{Architecture: "diffusion_gemma"},
	})
	if err == nil {
		t.Fatal("OpenROCmDiffusionSession(incomplete model) error = nil")
	}
}

func TestNewHIPROCmNativeDiffusionSession_MoE_Good(t *testing.T) {
	driver := &fakeHIPDriver{available: true}
	hidden := 2
	tensors := map[string]hipTensor{}
	addTensor := func(name, typeName string, dimensions []uint64, payload []byte) {
		t.Helper()
		pointer, err := driver.Malloc(uint64(len(payload)))
		if err != nil {
			t.Fatalf("Malloc(%s): %v", name, err)
		}
		if err := driver.CopyHostToDevice(pointer, payload); err != nil {
			t.Fatalf("CopyHostToDevice(%s): %v", name, err)
		}
		tensors[name] = hipTensor{
			pointer: pointer,
			info: nativeTensorInfo{
				Name:       name,
				TypeName:   typeName,
				Dimensions: append([]uint64(nil), dimensions...),
				ByteSize:   uint64(len(payload)),
			},
		}
	}
	scalar := make([]byte, 4)
	binary.LittleEndian.PutUint32(scalar, math.Float32bits(0.75))
	addTensor("model.encoder.language_model.layers.0.layer_scalar", "F32", []uint64{1}, scalar)
	addTensor("self_conditioning.pre_norm.weight", "F32", []uint64{uint64(hidden)}, make([]byte, hidden*4))
	addTensor("self_conditioning.gate_proj.weight", "BF16", []uint64{3, uint64(hidden)}, make([]byte, 3*hidden*2))
	addTensor("self_conditioning.up_proj.weight", "BF16", []uint64{3, uint64(hidden)}, make([]byte, 3*hidden*2))
	addTensor("self_conditioning.down_proj.weight", "BF16", []uint64{uint64(hidden), 3}, make([]byte, 3*hidden*2))

	loaded := &hipLoadedModel{
		driver: driver,
		modelInfo: inference.ModelInfo{
			Architecture: "diffusion_gemma",
			NumLayers:    1,
			HiddenSize:   hidden,
			VocabSize:    8,
			QuantGroup:   64,
		},
		tensors:    tensors,
		q4Config:   hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{{MoE: &hipGemma4MoELayerConfig{}}}},
		q4Layers:   1,
		q4ConfigOK: true,
	}
	session, err := newHIPROCmNativeDiffusionSession(loaded)
	if err != nil {
		t.Fatalf("newHIPROCmNativeDiffusionSession(MoE): %v", err)
	}
	if session.causalConfig.Layers[0].MoE == nil {
		t.Fatal("native diffusion session dropped the MoE decoder configuration")
	}
	if err := session.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

func TestHIPROCmNativeDiffusionSession_ValidateDenoiseUsesSlidingLayer_Good(t *testing.T) {
	const (
		prefix = 3
		length = 2
		window = 1
	)
	globalMask, globalShape := rocmDiffusionGlobalCanvasMask(length, prefix+length)
	localMask, localShape := rocmDiffusionLocalCanvasMask(length, prefix+length, prefix, window)
	session := &hipROCmNativeDiffusionSession{
		loaded:       &hipLoadedModel{modelInfo: inference.ModelInfo{HiddenSize: 2}, contextSize: 32},
		causalConfig: hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{{SlidingWindow: 0}, {SlidingWindow: window}}},
		device:       &hipGemma4Q4DeviceDecodeState{},
		position:     prefix,
	}
	err := session.validateDenoiseRequestLocked(ROCmDiffusionDenoiseRequest{
		Canvas:          []int32{4, 5},
		Prefix:          prefix,
		GlobalMask:      globalMask,
		GlobalMaskShape: globalShape,
		LocalMask:       localMask,
		LocalMaskShape:  localShape,
	})
	if err != nil {
		t.Fatalf("validateDenoiseRequestLocked(first sliding layer): %v", err)
	}
}
