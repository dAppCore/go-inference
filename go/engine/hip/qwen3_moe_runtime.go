// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/tokenizer"
	"dappco.re/go/inference/engine"
	sharedmodel "dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// loadHIPQwen3MoETextModel routes the qwen3_moe sparse-expert family through its own
// native ROCm forward pass (hip_qwen3_moe_*.go) — the first SiLU-gated MoE family
// engine/hip serves natively; every other MoE arch remains a named decline
// (rocmRetiredComposedArchDecline). Wired exactly the way loadHIPMamba2TextModel wires
// Mamba2: a self-contained model.TokenModel bridged into the shared engine.Session
// surface via hipComposedTextModel, tried ahead of loadHIPComposedTextModel's decline
// check in the load chain (native.go).
func loadHIPQwen3MoETextModel(path string, cfg inference.LoadConfig) (inference.TextModel, bool, error) {
	stat := core.Stat(path)
	if !stat.OK {
		return nil, false, nil
	}
	info, ok := stat.Value.(core.FsFileInfo)
	if !ok || !info.IsDir() {
		return nil, false, nil
	}

	modelType, configJSON, err := sharedmodel.ProbeDirArch(path)
	if err != nil {
		return nil, false, nil
	}
	architecture := normalizeROCmArchitecture(modelType)
	if architecture != "qwen3_moe" {
		return nil, false, nil
	}
	if cfg.AdapterPath != "" {
		return nil, true, core.NewError("rocm.LoadModel: adapters are not supported by qwen3_moe models yet")
	}

	denseCfg, err := ParseDenseConfig(configJSON)
	if err != nil {
		return nil, true, core.E("rocm.LoadModel", "parse qwen3_moe config", err)
	}
	geometry, err := resolveHIPQwen3MoEConfig(denseCfg)
	if err != nil {
		return nil, true, core.E("rocm.LoadModel", "resolve qwen3_moe geometry", err)
	}

	mapping, err := safetensors.LoadDirMmap(path)
	if err != nil {
		return nil, true, core.E("rocm.LoadModel", "load qwen3_moe tensors", err)
	}
	defer func() { _ = mapping.Close() }()

	driver := newSystemHIPDriver()
	if driver == nil || !driver.Available() {
		return nil, true, core.NewError("rocm.LoadModel: HIP driver is not available")
	}
	weights, err := loadHIPQwen3MoEWeights(driver, mapping.Tensors, geometry)
	if err != nil {
		return nil, true, core.E("rocm.LoadModel", "load qwen3_moe weights", err)
	}
	tokenModel := &hipQwen3MoEModel{driver: driver, cfg: geometry, weights: weights}

	tok, err := tokenizer.LoadTokenizer(core.PathJoin(path, "tokenizer.json"))
	if err != nil {
		weights.Close()
		return nil, true, core.E("rocm.LoadModel", "load qwen3_moe tokenizer", err)
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
		modelType:        architecture,
		numLayers:        geometry.NumLayers,
		declaredStops:    loadGenerationConfigStops(path),
		declaredSampling: loadGenerationConfigSamplingDefaults(path),
	}
	modelInfo := inference.ModelInfo{
		Architecture: architecture,
		VocabSize:    geometry.VocabSize,
		NumLayers:    geometry.NumLayers,
		HiddenSize:   geometry.HiddenSize,
	}
	return engine.NewTextModel(source, tok, architecture, modelInfo, maxLen), true, nil
}
