// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/engine"
	rocmprofile "dappco.re/go/inference/engine/hip/profile"
	"dappco.re/go/inference/kv"
	sharedmodel "dappco.re/go/inference/model"
	"dappco.re/go/inference/model/arch/Qwen/qwen3"
	"dappco.re/go/inference/model/composed"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

func hipComposedTestValues(n, seed int) []float32 {
	out := make([]float32, n)
	for index := range out {
		out[index] = float32((index*seed+7)%101-50) * 0.02
	}
	return out
}

func hipComposedTestTokenModel(layers int) *composed.ComposedTokenModel {
	const hidden, vocab, intermediate = 8, 32, 16
	cfg := qwen3.GatedDeltaConfig{KeyHeads: 2, ValueHeads: 4, HeadDim: 8, ConvKernel: 4, Eps: 1e-5}
	valueDim := cfg.ValueHeads * cfg.HeadDim
	convDim := 2*cfg.KeyHeads*cfg.HeadDim + valueDim
	blocks := make([]composed.Layer, layers)
	for layer := range blocks {
		seed := layer*13 + 20
		weights := &qwen3.GatedDeltaWeights{
			InProjQKV:  hipComposedTestValues(convDim*hidden, seed+1),
			ConvWeight: hipComposedTestValues(convDim*cfg.ConvKernel, seed+2),
			ConvBias:   hipComposedTestValues(convDim, seed+3),
			InProjA:    hipComposedTestValues(cfg.ValueHeads*hidden, seed+4),
			ALog:       hipComposedTestValues(cfg.ValueHeads, seed+5),
			DtBias:     hipComposedTestValues(cfg.ValueHeads, seed+6),
			InProjB:    hipComposedTestValues(cfg.ValueHeads*hidden, seed+7),
			InProjZ:    hipComposedTestValues(valueDim*hidden, seed+8),
			Norm:       hipComposedTestValues(cfg.HeadDim, seed+9),
			OutProj:    hipComposedTestValues(hidden*valueDim, seed+10),
		}
		blocks[layer] = composed.Layer{
			InputNorm:    hipComposedTestValues(hidden, layer*13+1),
			Mixer:        composed.NewGatedDeltaMixer(weights, cfg),
			PostAttnNorm: hipComposedTestValues(hidden, layer*13+2),
			MLP: &composed.MLP{
				Gate: hipComposedTestValues(intermediate*hidden, layer*13+3),
				Up:   hipComposedTestValues(intermediate*hidden, layer*13+4),
				Down: hipComposedTestValues(hidden*intermediate, layer*13+5),
				FF:   intermediate,
			},
		}
	}
	return composed.NewTokenModel(&composed.ComposedModel{
		Embed:  hipComposedTestValues(vocab*hidden, 100),
		Layers: blocks,
		NormF:  hipComposedTestValues(hidden, 101),
		D:      hidden,
		Vocab:  vocab,
		Eps:    1e-5,
	})
}

func TestHIPComposedEngineSession_StateIncludesGeneratedTokens_Good(t *testing.T) {
	const layers = 2
	session := &hipComposedEngineSession{model: hipComposedTestTokenModel(layers), architecture: "qwen3_6", numLayers: layers}
	core.RequireNoError(t, session.PrefillTokens([]int32{1, 5, 9, 2}))
	generated, err := session.GenerateFromCacheEach(4, -1, func(int32) bool { return true })
	core.RequireNoError(t, err)
	core.AssertEqual(t, 8, session.Pos())

	snapshot, err := session.CaptureKVWithOptions(kv.CaptureOptions{})
	core.RequireNoError(t, err)
	core.AssertEqual(t, append([]int32{1, 5, 9, 2}, generated...), snapshot.Tokens)

	resumed := &hipComposedEngineSession{model: hipComposedTestTokenModel(layers), architecture: "qwen3_6", numLayers: layers}
	core.RequireNoError(t, resumed.RestoreFromKV(context.Background(), snapshot))
	want, err := session.GenerateFromCacheEach(3, -1, func(int32) bool { return true })
	core.RequireNoError(t, err)
	got, err := resumed.GenerateFromCacheEach(3, -1, func(int32) bool { return true })
	core.RequireNoError(t, err)
	core.AssertEqual(t, want, got)
}

func TestHIPComposedEngineSession_CaptureAndRangeRejectInvalidState_Bad(t *testing.T) {
	session := &hipComposedEngineSession{model: hipComposedTestTokenModel(1), architecture: "qwen3_6", numLayers: 1}
	_, err := session.CaptureKVWithOptions(kv.CaptureOptions{})
	core.AssertError(t, err)
	core.RequireNoError(t, session.PrefillTokens([]int32{1, 2, 3}))
	err = session.RangeKVBlocks(2, kv.CaptureOptions{BlockStartToken: -1}, func(kv.Block) (bool, error) { return true, nil })
	core.AssertError(t, err)
}

func TestHIPComposedEngineSession_RangeHonorsTrustedPrefix_Ugly(t *testing.T) {
	session := &hipComposedEngineSession{model: hipComposedTestTokenModel(1), architecture: "qwen3_6", numLayers: 1}
	core.RequireNoError(t, session.PrefillTokens([]int32{1, 2, 3, 4, 5, 6, 7}))
	var blocks []kv.Block
	err := session.RangeKVBlocks(3, kv.CaptureOptions{BlockStartToken: 3}, func(block kv.Block) (bool, error) {
		blocks = append(blocks, block)
		return true, nil
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, []int{1, 2}, []int{blocks[0].Index, blocks[1].Index})
	core.AssertEqual(t, []int32{4, 5, 6}, blocks[0].Snapshot.Tokens)
	core.AssertEqual(t, []int32{7}, blocks[1].Snapshot.Tokens)
}

func TestHIPComposedTextModel_DeclaresQwenChatML_Good(t *testing.T) {
	source := &hipComposedTextModel{model: hipComposedTestTokenModel(1), modelType: "qwen3_6", numLayers: 1}
	var memory engine.MemoryReporter = source
	template, ok := source.DeclaredChatTemplate()
	core.RequireTrue(t, ok)
	core.AssertEqual(t, "<|im_start|>", template.Open)
	core.AssertEqual(t, "assistant", template.AssistantRole)
	core.AssertEqual(t, []string{"<|im_end|>"}, template.Stops)
	core.AssertEqual(t, uint64(0), memory.ActiveMemoryBytes())
	core.AssertEqual(t, uint64(0), memory.PeakMemoryBytes())
}

type hipComposedRouteRuntime struct{ calls int }

func (runtime *hipComposedRouteRuntime) Available() bool {
	runtime.calls++
	return false
}

func (*hipComposedRouteRuntime) DeviceInfo() nativeDeviceInfo { return nativeDeviceInfo{} }

func (runtime *hipComposedRouteRuntime) LoadModel(string, nativeLoadConfig) (nativeModel, error) {
	runtime.calls++
	return nil, core.NewError("native runtime must not receive a composed checkpoint")
}

func TestROCmBackend_LoadsRegisteredComposedModelBeforeNativeRuntime_Good(t *testing.T) {
	const modelType = "qwen_hip_composed_test"
	sharedmodel.RegisterArch(sharedmodel.ArchSpec{
		ModelTypes: []string{modelType},
		Composed: func(map[string]safetensors.Tensor, []byte) (sharedmodel.TokenModel, error) {
			return hipComposedTestTokenModel(2), nil
		},
	})
	dir := t.TempDir()
	core.RequireNoError(t, coreio.Local.Write(core.PathJoin(dir, "config.json"), `{"model_type":"`+modelType+`","max_position_embeddings":8192}`))
	encoded, err := safetensors.Encode(map[string]safetensors.Tensor{
		"unused": {Dtype: "F32", Shape: []int{1}, Data: []byte{0, 0, 0, 0}},
	})
	core.RequireNoError(t, err)
	core.RequireNoError(t, coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(encoded)))
	core.RequireNoError(t, coreio.Local.Write(core.PathJoin(dir, "tokenizer.json"), `{
  "model":{"type":"BPE","vocab":{"H":0,"i":1},"merges":[]},
  "added_tokens":[
    {"id":2,"content":"<eos>","special":true},
    {"id":3,"content":"<|im_start|>","special":true},
    {"id":4,"content":"<|im_end|>","special":true}
  ]
}`))

	runtime := &hipComposedRouteRuntime{}
	loaded, err := newROCmBackendWithRuntime(runtime).loadModelWithROCmConfigMode(
		dir,
		inference.LoadConfig{},
		ROCmLoadConfig{},
		false,
	)
	core.RequireNoError(t, err)
	defer loaded.Close()
	core.AssertEqual(t, 0, runtime.calls)
	core.AssertEqual(t, modelType, loaded.ModelType())
	core.AssertEqual(t, 8192, loaded.(*engine.TextModel).MaxLen())
	core.AssertEqual(t, inference.ModelInfo{Architecture: modelType, VocabSize: 32, NumLayers: 2, HiddenSize: 8}, loaded.Info())

	thinking := true
	prompt := loaded.(*engine.TextModel).FormatChatPromptWithThinking(
		[]inference.Message{{Role: "system", Content: "brief"}, {Role: "user", Content: "Hi"}},
		&thinking,
	)
	core.AssertEqual(t,
		"<|im_start|>system\nbrief<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
		prompt,
	)
}

func TestReactiveSequenceMixerReport_ComposedRunnerLinked_Good(t *testing.T) {
	report := baseReactiveSequenceMixerReport("/models/qwen", nil)
	core.AssertTrue(t, report.RunnerReady)
	core.AssertEqual(t, "portable_composed_session_model", report.Labels["sequence_mixer_runner_status"])
	core.AssertEqual(t,
		[]string{"portable composed incremental runner is linked; projection hooks remain available for HIP++ acceleration"},
		rocmprofile.ArchitectureProfileNotes("qwen3_6"),
	)
	for _, architecture := range []string{"composed", "hybrid", "qwen3_6", "qwen3_6_moe"} {
		profile, ok := rocmprofile.LookupArchitectureProfile(architecture)
		core.RequireTrue(t, ok)
		core.RequireTrue(t, profile.NativeRuntime)
		core.RequireTrue(t, profile.Generation)
		core.RequireTrue(t, profile.Chat)
	}
}
