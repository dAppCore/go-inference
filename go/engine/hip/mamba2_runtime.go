// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/tokenizer"
	"dappco.re/go/inference/engine"
	sharedmodel "dappco.re/go/inference/model"
	"dappco.re/go/inference/model/mamba2"
	"dappco.re/go/inference/model/safetensors"
)

// loadHIPMamba2TextModel routes the standalone recurrent Mamba2 family before
// the transformer loader. Mamba2 owns fixed recurrent state rather than a
// transformer KV cache, so its shared SessionModel is served through the same
// token-prefix state bridge used by composed hybrid models.
func loadHIPMamba2TextModel(path string, cfg inference.LoadConfig) (inference.TextModel, bool, error) {
	stat := core.Stat(path)
	if !stat.OK {
		return nil, false, nil
	}
	info, ok := stat.Value.(core.FsFileInfo)
	if !ok || !info.IsDir() {
		return nil, false, nil
	}

	modelType, configJSON, err := sharedmodel.ProbeDirArch(path)
	if err != nil || modelType != "mamba2" {
		return nil, false, nil
	}
	if cfg.AdapterPath != "" {
		return nil, true, core.NewError("rocm.LoadModel: adapters are not supported by Mamba2 models")
	}
	mapping, err := safetensors.LoadDirMmap(path)
	if err != nil {
		return nil, true, core.E("rocm.LoadModel", "load Mamba2 tensors", err)
	}
	defer func() { _ = mapping.Close() }()
	mambaModel, err := mamba2.LoadMambaModel(mapping.Tensors, hipMamba2Epsilon(configJSON))
	if err != nil {
		return nil, true, core.E("rocm.LoadModel", "assemble Mamba2 model", err)
	}
	tokenModel := mamba2.NewTokenModel(mambaModel)
	tok, err := tokenizer.LoadTokenizer(core.PathJoin(path, "tokenizer.json"))
	if err != nil {
		return nil, true, core.E("rocm.LoadModel", "load Mamba2 tokenizer", err)
	}
	maxLen := cfg.ContextLen
	if maxLen <= 0 {
		maxLen = sharedmodel.ProbeDirContextWindow(path)
		if maxLen <= 0 {
			maxLen = defaultContextLengthCap
		}
	}
	source := &hipComposedTextModel{
		model:            tokenModel,
		tokenizer:        tok,
		modelType:        modelType,
		numLayers:        len(mambaModel.Layers),
		declaredStops:    loadGenerationConfigStops(path),
		declaredSampling: loadGenerationConfigSamplingDefaults(path),
	}
	modelInfo := inference.ModelInfo{
		Architecture: modelType,
		VocabSize:    mambaModel.Vocab,
		NumLayers:    len(mambaModel.Layers),
		HiddenSize:   mambaModel.D,
	}
	return engine.NewTextModel(source, tok, modelType, modelInfo, maxLen), true, nil
}

func hipMamba2Epsilon(configJSON []byte) float32 {
	var probe struct {
		Epsilon    float32 `json:"rms_norm_eps"`
		TextConfig struct {
			Epsilon float32 `json:"rms_norm_eps"`
		} `json:"text_config"`
	}
	_ = core.JSONUnmarshal(configJSON, &probe)
	if probe.Epsilon > 0 {
		return probe.Epsilon
	}
	if probe.TextConfig.Epsilon > 0 {
		return probe.TextConfig.Epsilon
	}
	return 1e-5
}
