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
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

func hipMamba2TestTensor(shape ...int) safetensors.Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	values := make([]float32, size)
	for index := range values {
		values[index] = float32(index%17-8) * 0.01
	}
	return safetensors.Tensor{Dtype: "F32", Shape: shape, Data: safetensors.EncodeFloat32(values)}
}

func writeHIPMamba2TestModel(t *testing.T) string {
	t.Helper()
	const heads, headDim, stateDim, kernel = 2, 4, 4, 3
	const hidden, vocab = 8, 16
	inner := heads * headDim
	conv := inner + 2*stateDim
	projection := 2*inner + 2*stateDim + heads
	tensors := map[string]safetensors.Tensor{
		"backbone.embeddings.weight":              hipMamba2TestTensor(vocab, hidden),
		"backbone.norm_f.weight":                  hipMamba2TestTensor(hidden),
		"backbone.layers.0.norm.weight":           hipMamba2TestTensor(hidden),
		"backbone.layers.0.mixer.in_proj.weight":  hipMamba2TestTensor(projection, hidden),
		"backbone.layers.0.mixer.conv1d.weight":   hipMamba2TestTensor(conv, 1, kernel),
		"backbone.layers.0.mixer.conv1d.bias":     hipMamba2TestTensor(conv),
		"backbone.layers.0.mixer.A_log":           hipMamba2TestTensor(heads),
		"backbone.layers.0.mixer.D":               hipMamba2TestTensor(heads),
		"backbone.layers.0.mixer.dt_bias":         hipMamba2TestTensor(heads),
		"backbone.layers.0.mixer.norm.weight":     hipMamba2TestTensor(inner),
		"backbone.layers.0.mixer.out_proj.weight": hipMamba2TestTensor(hidden, inner),
	}
	encoded, err := safetensors.Encode(tensors)
	core.RequireNoError(t, err)
	dir := t.TempDir()
	core.RequireNoError(t, coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(encoded)))
	core.RequireNoError(t, coreio.Local.Write(core.PathJoin(dir, "config.json"), `{
  "model_type":"mamba2",
  "max_position_embeddings":32768,
  "rms_norm_eps":0.00001
}`))
	core.RequireNoError(t, coreio.Local.Write(core.PathJoin(dir, "tokenizer.json"), `{
  "model":{"type":"BPE","vocab":{"H":0,"i":1},"merges":[]},
  "added_tokens":[{"id":2,"content":"<eos>","special":true}]
}`))
	return dir
}

func TestROCmBackend_LoadsMamba2BeforeNativeRuntime_Good(t *testing.T) {
	dir := writeHIPMamba2TestModel(t)
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
	core.AssertEqual(t, "mamba2", loaded.ModelType())
	core.AssertEqual(t, 32768, loaded.(*engine.TextModel).MaxLen())
	core.AssertEqual(t, inference.ModelInfo{Architecture: "mamba2", VocabSize: 16, NumLayers: 1, HiddenSize: 8}, loaded.Info())
	var generated []int32
	for token := range loaded.Generate(context.Background(), "H", inference.WithMaxTokens(2)) {
		generated = append(generated, token.ID)
	}
	core.RequireTrue(t, loaded.(*engine.TextModel).Err().OK)
	core.AssertEqual(t, 2, len(generated))
	profile, ok := rocmprofile.LookupArchitectureProfile("mamba2")
	core.RequireTrue(t, ok)
	core.RequireTrue(t, profile.NativeRuntime)
	core.RequireTrue(t, profile.Generation)
	core.RequireTrue(t, profile.Chat)
}

func TestHIPMamba2Epsilon_NestedAndDefault_Ugly(t *testing.T) {
	core.AssertEqual(t, float32(2e-5), hipMamba2Epsilon([]byte(`{"text_config":{"rms_norm_eps":0.00002}}`)))
	core.AssertEqual(t, float32(1e-5), hipMamba2Epsilon([]byte(`{"rms_norm_eps":0}`)))
}
