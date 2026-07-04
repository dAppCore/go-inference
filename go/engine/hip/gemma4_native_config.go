// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import modelgemma4 "dappco.re/go/inference/engine/hip/model/gemma4"

type nativeGemma4TextConfig struct {
	NumLayers               int
	LayerTypes              []string
	KVSharedLayers          int
	KVSharedLayersSet       bool
	SlidingWindow           int
	SlidingWindowPattern    int
	HeadDim                 int
	GlobalHeadDim           int
	HiddenSizePerLayerInput int
	VocabSizePerLayerInput  int
	AttentionKEqV           bool
	FinalLogitSoftcap       float64
	UseDoubleWideMLP        bool
	EnableMoEBlock          bool
	NumExperts              int
	TopKExperts             int
	MoEIntermediateSize     int
	Vision                  bool
	Audio                   bool
	RoPEParameters          map[string]nativeGemma4RoPEParameters
}

type nativeGemma4RoPEParameters struct {
	PartialRotaryFactor float64
	RopeTheta           float64
	RopeType            string
	Factor              float64
}

func cloneNativeGemma4TextConfig(cfg nativeGemma4TextConfig) nativeGemma4TextConfig {
	cfg.LayerTypes = append([]string(nil), cfg.LayerTypes...)
	if len(cfg.RoPEParameters) > 0 {
		params := make(map[string]nativeGemma4RoPEParameters, len(cfg.RoPEParameters))
		for key, value := range cfg.RoPEParameters {
			params[key] = value
		}
		cfg.RoPEParameters = params
	}
	return cfg
}

func rocmNativeGemma4TextConfig(path string) nativeGemma4TextConfig {
	root, err := rocmModelPackRoot(path)
	if err != nil {
		return nativeGemma4TextConfig{}
	}
	cfg, err := readROCmModelConfig(root)
	if err != nil || cfg == nil {
		return nativeGemma4TextConfig{}
	}
	return rocmNativeGemma4TextConfigFromProbe(*cfg)
}

func rocmNativeGemma4TextConfigFromProbe(cfg rocmModelPackConfigProbe) nativeGemma4TextConfig {
	layerTypes := rocmConfigLayerTypes(cfg)
	numLayers := firstPositiveInt(cfg.NumHiddenLayers, cfg.NumLayers, cfg.TextConfig.NumHiddenLayers, cfg.TextConfig.NumLayers)
	if numLayers > 0 && len(layerTypes) >= numLayers {
		layerTypes = append([]string(nil), layerTypes[:numLayers]...)
	} else {
		layerTypes = nil
	}
	kvShared, kvSharedSet := rocmConfigKVSharedLayers(cfg)
	return nativeGemma4TextConfig{
		NumLayers:               numLayers,
		LayerTypes:              layerTypes,
		KVSharedLayers:          kvShared,
		KVSharedLayersSet:       kvSharedSet,
		SlidingWindow:           firstPositiveInt(cfg.SlidingWindow, cfg.TextConfig.SlidingWindow),
		SlidingWindowPattern:    firstPositiveInt(cfg.SlidingWindowPattern, cfg.TextConfig.SlidingWindowPattern),
		HeadDim:                 firstPositiveInt(cfg.HeadDim, cfg.TextConfig.HeadDim),
		GlobalHeadDim:           firstPositiveInt(cfg.GlobalHeadDim, cfg.TextConfig.GlobalHeadDim),
		HiddenSizePerLayerInput: firstPositiveInt(cfg.HiddenSizePerLayer, cfg.TextConfig.HiddenSizePerLayer),
		VocabSizePerLayerInput:  firstPositiveInt(cfg.VocabSizePerLayer, cfg.TextConfig.VocabSizePerLayer),
		AttentionKEqV:           cfg.AttentionKEqV || cfg.TextConfig.AttentionKEqV,
		FinalLogitSoftcap:       firstPositiveFloat(cfg.FinalLogitSoftcap, cfg.TextConfig.FinalLogitSoftcap),
		UseDoubleWideMLP:        cfg.UseDoubleWideMLP || cfg.TextConfig.UseDoubleWideMLP,
		EnableMoEBlock:          cfg.EnableMoEBlock || cfg.TextConfig.EnableMoEBlock,
		NumExperts:              firstPositiveInt(cfg.NumExperts, cfg.TextConfig.NumExperts),
		TopKExperts:             firstPositiveInt(cfg.TopKExperts, cfg.NumExpertsPerTok, cfg.TextConfig.TopKExperts, cfg.TextConfig.NumExpertsPerTok),
		MoEIntermediateSize:     firstPositiveInt(cfg.MoEIntermediateSize, cfg.ExpertIntermediateSize, cfg.TextConfig.MoEIntermediateSize, cfg.TextConfig.ExpertIntermediateSize),
		Vision:                  rocmModelPackConfigHasVision(cfg),
		Audio:                   rocmModelPackConfigHasAudio(cfg),
		RoPEParameters:          rocmNativeGemma4RoPEParameters(cfg),
	}
}

func rocmModelPackConfigHasVision(cfg rocmModelPackConfigProbe) bool {
	return rocmGemma4ConfigHasVision(cfg)
}

func rocmModelPackConfigHasAudio(cfg rocmModelPackConfigProbe) bool {
	return rocmGemma4ConfigHasAudio(cfg)
}

func rocmNativeGemma4RoPEParameters(cfg rocmModelPackConfigProbe) map[string]nativeGemma4RoPEParameters {
	params := rocmGemma4RoPEParametersFromProbe(cfg)
	if isROCmGemma4Architecture(rocmConfigArchitecture(cfg)) {
		policy := modelgemma4.RoPEPolicyOf(modelgemma4.TextConfig{
			GlobalPartialRotaryFactor: firstPositiveFloat(cfg.GlobalPartialRotary, cfg.TextConfig.GlobalPartialRotary),
			RoPEParameters:            params,
		})
		return rocmNativeGemma4RoPEParametersFromModel(policy.Parameters)
	}
	for layerType, rope := range params {
		if rope.RopeType == "proportional" && rope.Factor <= 0 {
			rope.Factor = 1
			params[layerType] = rope
		}
	}
	return rocmNativeGemma4RoPEParametersFromModel(params)
}

func rocmNativeGemma4RoPEParametersFromModel(src map[string]modelgemma4.RoPEParameters) map[string]nativeGemma4RoPEParameters {
	if len(src) == 0 {
		return nil
	}
	out := make(map[string]nativeGemma4RoPEParameters, len(src))
	for layerType, params := range src {
		if layerType == "" {
			continue
		}
		out[layerType] = nativeGemma4RoPEParameters{
			PartialRotaryFactor: params.PartialRotaryFactor,
			RopeTheta:           params.RopeTheta,
			RopeType:            params.RopeType,
			Factor:              params.Factor,
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}
