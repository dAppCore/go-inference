// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"dappco.re/go/inference"
	modelgemma4 "dappco.re/go/inference/engine/hip/model/gemma4"
)

func rocmGemma4DeclaredFeaturesForModel(identity inference.ModelIdentity) Gemma4DeclaredFeatures {
	return rocmGemma4DeclaredFeaturesFromModel(modelgemma4.FeaturesOfIdentity(identity))
}

func rocmGemma4DeclaredFeaturesFromModel(features modelgemma4.Features) Gemma4DeclaredFeatures {
	return Gemma4DeclaredFeatures{
		Mixture:     features.Mixture,
		NumExperts:  features.NumExperts,
		TopKExperts: features.TopKExperts,
		Vision:      features.Vision,
		Audio:       features.Audio,
		Attention: Gemma4AttentionClass{
			SlidingWindow:  features.Attention.SlidingWindow,
			SlidingPattern: features.Attention.SlidingPattern,
			SharedKVLayers: features.Attention.SharedKVLayers,
		},
	}
}

func rocmGemma4ModelFeaturesFromDeclared(features Gemma4DeclaredFeatures) modelgemma4.Features {
	return modelgemma4.Features{
		Mixture:     features.Mixture,
		NumExperts:  features.NumExperts,
		TopKExperts: features.TopKExperts,
		Vision:      features.Vision,
		Audio:       features.Audio,
		Attention: modelgemma4.AttentionClass{
			SlidingWindow:  features.Attention.SlidingWindow,
			SlidingPattern: features.Attention.SlidingPattern,
			SharedKVLayers: features.Attention.SharedKVLayers,
		},
	}
}

func rocmGemma4TextConfigFromProbe(cfg rocmModelPackConfigProbe) modelgemma4.TextConfig {
	kvShared, kvSharedSet := rocmConfigKVSharedLayers(cfg)
	return modelgemma4.TextConfig{
		NumLayers:                 firstPositiveInt(cfg.NumHiddenLayers, cfg.NumLayers, cfg.TextConfig.NumHiddenLayers, cfg.TextConfig.NumLayers),
		LayerTypes:                rocmConfigLayerTypes(cfg),
		EnableMoEBlock:            cfg.EnableMoEBlock || cfg.TextConfig.EnableMoEBlock,
		NumExperts:                firstPositiveInt(cfg.NumExperts, cfg.TextConfig.NumExperts),
		TopKExperts:               firstPositiveInt(cfg.TopKExperts, cfg.NumExpertsPerTok, cfg.TextConfig.TopKExperts, cfg.TextConfig.NumExpertsPerTok),
		Vision:                    rocmGemma4ConfigHasVision(cfg),
		VisionConfig:              rocmGemma4VisionConfigFromProbe(cfg),
		Audio:                     rocmGemma4ConfigHasAudio(cfg),
		AudioConfig:               rocmGemma4AudioConfigFromProbe(cfg),
		SlidingWindow:             firstPositiveInt(cfg.SlidingWindow, cfg.TextConfig.SlidingWindow),
		SlidingWindowPattern:      firstPositiveInt(cfg.SlidingWindowPattern, cfg.TextConfig.SlidingWindowPattern),
		KVSharedLayers:            kvShared,
		KVSharedLayersSet:         kvSharedSet,
		GlobalPartialRotaryFactor: firstPositiveFloat(cfg.GlobalPartialRotary, cfg.TextConfig.GlobalPartialRotary),
		RoPEParameters:            rocmGemma4RoPEParametersFromProbe(cfg),
		HiddenSizePerLayer:        firstPositiveInt(cfg.HiddenSizePerLayer, cfg.TextConfig.HiddenSizePerLayer),
		VocabSizePerLayer:         firstPositiveInt(cfg.VocabSizePerLayer, cfg.TextConfig.VocabSizePerLayer),
		UseDoubleWideMLP:          cfg.UseDoubleWideMLP || cfg.TextConfig.UseDoubleWideMLP,
		MoEIntermediateSize:       firstPositiveInt(cfg.MoEIntermediateSize, cfg.ExpertIntermediateSize, cfg.TextConfig.MoEIntermediateSize, cfg.TextConfig.ExpertIntermediateSize),
	}
}

func rocmGemma4DiffusionPolicyFromProbe(cfg rocmModelPackConfigProbe) modelgemma4.DiffusionGeneratePolicy {
	return modelgemma4.DiffusionGeneratePolicyOf(modelgemma4.DiffusionPolicyConfig{
		ReferenceCanvasLength: firstPositiveInt(cfg.CanvasLength, cfg.TextConfig.CanvasLength),
		TextVocabSize:         firstPositiveInt(cfg.TextConfig.VocabSize, cfg.VocabSize),
		VocabSize:             firstPositiveInt(cfg.VocabSize, cfg.TextConfig.VocabSize),
	})
}

func rocmGemma4RoPEParametersFromProbe(cfg rocmModelPackConfigProbe) map[string]modelgemma4.RoPEParameters {
	params := modelgemma4.OverlayRoPEParameters(nil, rocmGemma4RoPEParametersFromProbeMap(cfg.TextConfig.RoPEParameters))
	params = modelgemma4.OverlayRoPEParameters(params, rocmGemma4RoPEParametersFromProbeMap(cfg.RoPEParameters))
	return params
}

func rocmGemma4RoPEParametersFromProbeMap(src map[string]rocmRoPEProbe) map[string]modelgemma4.RoPEParameters {
	if len(src) == 0 {
		return nil
	}
	params := make(map[string]modelgemma4.RoPEParameters, len(src))
	for attentionType, value := range src {
		if attentionType == "" {
			continue
		}
		params[attentionType] = modelgemma4.RoPEParameters{
			PartialRotaryFactor: value.PartialRotaryFactor,
			RopeTheta:           value.RopeTheta,
			RopeType:            value.RopeType,
			Factor:              value.Factor,
		}
	}
	if len(params) == 0 {
		return nil
	}
	return params
}

func rocmGemma4ConfigHasVision(cfg rocmModelPackConfigProbe) bool {
	architecture := rocmConfigArchitecture(cfg)
	if !isROCmGemma4Architecture(architecture) || isROCmGemma4AssistantArchitecture(architecture) {
		return false
	}
	return rocmGemma4VisionConfigFromProbe(cfg).Present()
}

func rocmGemma4ConfigHasAudio(cfg rocmModelPackConfigProbe) bool {
	if isROCmGemma4AssistantArchitecture(rocmConfigArchitecture(cfg)) {
		return false
	}
	return rocmGemma4AudioConfigFromProbe(cfg).Present()
}

func rocmGemma4VisionConfigFromProbe(cfg rocmModelPackConfigProbe) modelgemma4.VisionConfig {
	return modelgemma4.VisionConfig{
		ImageTokenID:          cfg.ImageTokenID,
		ImageTokenIndex:       cfg.ImageTokenIndex,
		VideoTokenID:          cfg.VideoTokenID,
		BOITokenID:            cfg.BOITokenID,
		BOITokenIndex:         cfg.BOITokenIndex,
		EOITokenID:            cfg.EOITokenID,
		EOITokenIndex:         cfg.EOITokenIndex,
		SoftTokensPerImage:    cfg.VisionSoftTokensPerImage,
		MMTokensPerImage:      cfg.MMTokensPerImage,
		ModelType:             cfg.VisionConfig.ModelType,
		DType:                 cfg.VisionConfig.DType,
		ImageSize:             cfg.VisionConfig.ImageSize,
		PatchSize:             cfg.VisionConfig.PatchSize,
		NumChannels:           cfg.VisionConfig.NumChannels,
		HiddenSize:            cfg.VisionConfig.HiddenSize,
		IntermediateSize:      cfg.VisionConfig.IntermediateSize,
		NumHiddenLayers:       cfg.VisionConfig.NumHiddenLayers,
		NumAttentionHeads:     cfg.VisionConfig.NumAttentionHeads,
		NumKeyValueHeads:      cfg.VisionConfig.NumKeyValueHeads,
		HeadDim:               cfg.VisionConfig.HeadDim,
		GlobalHeadDim:         cfg.VisionConfig.GlobalHeadDim,
		PoolingKernelSize:     cfg.VisionConfig.PoolingKernelSize,
		PositionEmbeddingSize: cfg.VisionConfig.PositionEmbeddingSize,
		DefaultOutputLength:   cfg.VisionConfig.DefaultOutputLength,
		HiddenActivation:      cfg.VisionConfig.HiddenActivation,
		RMSNormEps:            firstPositiveFloat(cfg.VisionConfig.RMSNormEps, cfg.VisionConfig.LayerNormEps),
		RoPEParameters: modelgemma4.RoPEParameters{
			PartialRotaryFactor: cfg.VisionConfig.RoPEParameters.PartialRotaryFactor,
			RopeTheta:           cfg.VisionConfig.RoPEParameters.RopeTheta,
			RopeType:            cfg.VisionConfig.RoPEParameters.RopeType,
			Factor:              cfg.VisionConfig.RoPEParameters.Factor,
		},
		Standardize:       cfg.VisionConfig.Standardize,
		UseClippedLinears: cfg.VisionConfig.UseClippedLinears,
	}
}

func rocmGemma4AudioConfigFromProbe(cfg rocmModelPackConfigProbe) modelgemma4.AudioConfig {
	return modelgemma4.AudioConfig{
		AudioTokenID:                cfg.AudioTokenID,
		AudioTokenIndex:             cfg.AudioTokenIndex,
		BOATokenID:                  cfg.BOATokenID,
		BOATokenIndex:               cfg.BOATokenIndex,
		EOATokenID:                  cfg.EOATokenID,
		EOATokenIndex:               cfg.EOATokenIndex,
		ModelType:                   cfg.AudioConfig.ModelType,
		HiddenSize:                  cfg.AudioConfig.HiddenSize,
		AudioEmbedDim:               cfg.AudioConfig.AudioEmbedDim,
		AudioSamplesPerToken:        cfg.AudioConfig.AudioSamplesPerToken,
		NumHiddenLayers:             cfg.AudioConfig.NumHiddenLayers,
		NumAttentionHeads:           cfg.AudioConfig.NumAttentionHeads,
		AttentionChunkSize:          cfg.AudioConfig.AttentionChunkSize,
		AttentionContextLeft:        cfg.AudioConfig.AttentionContextLeft,
		AttentionContextRight:       cfg.AudioConfig.AttentionContextRight,
		AttentionLogitCap:           cfg.AudioConfig.AttentionLogitCap,
		AttentionInvalidLogitsValue: cfg.AudioConfig.AttentionInvalidLogitsValue,
		ConvKernelSize:              cfg.AudioConfig.ConvKernelSize,
		OutputProjDims:              cfg.AudioConfig.OutputProjDims,
		RMSNormEps:                  cfg.AudioConfig.RMSNormEps,
		GradientClipping:            cfg.AudioConfig.GradientClipping,
		ResidualWeight:              cfg.AudioConfig.ResidualWeight,
		HiddenAct:                   cfg.AudioConfig.HiddenAct,
		UseClippedLinears:           cfg.AudioConfig.UseClippedLinears,
	}
}

func rocmGemma4EngineFeaturesForModel(identity inference.ModelIdentity) Gemma4EngineFeatures {
	return rocmGemma4EngineFeaturesFromModel(modelgemma4.EngineFeaturesOfIdentity(identity))
}

func rocmGemma4EngineFeaturesFromModel(features modelgemma4.EngineFeatures) Gemma4EngineFeatures {
	return Gemma4EngineFeatures{
		DirectGreedyToken:           features.DirectGreedyToken,
		NativeMLPMatVec:             features.NativeMLPMatVec,
		NativeLinearMatVec:          features.NativeLinearMatVec,
		NativeQ6BitstreamMatVec:     features.NativeQ6BitstreamMatVec,
		NativeAttentionOMatVec:      features.NativeAttentionOMatVec,
		NativeFixedSlidingAttention: features.NativeFixedSlidingAttention,
		GenerationStream:            features.GenerationStream,
		AsyncDecodePrefetch:         features.AsyncDecodePrefetch,
		ModelContextWindow:          features.ModelContextWindow,
		FixedSlidingCache:           features.FixedSlidingCache,
		FixedSlidingCacheBound:      features.FixedSlidingCacheBound,
		CompiledLayerDecode:         features.CompiledLayerDecode,
		PipelinedDecode:             features.PipelinedDecode,
	}
}

func rocmGemma4ModelEngineFeatures(features Gemma4EngineFeatures) modelgemma4.EngineFeatures {
	return modelgemma4.EngineFeatures{
		DirectGreedyToken:           features.DirectGreedyToken,
		NativeMLPMatVec:             features.NativeMLPMatVec,
		NativeLinearMatVec:          features.NativeLinearMatVec,
		NativeQ6BitstreamMatVec:     features.NativeQ6BitstreamMatVec,
		NativeAttentionOMatVec:      features.NativeAttentionOMatVec,
		NativeFixedSlidingAttention: features.NativeFixedSlidingAttention,
		GenerationStream:            features.GenerationStream,
		AsyncDecodePrefetch:         features.AsyncDecodePrefetch,
		ModelContextWindow:          features.ModelContextWindow,
		FixedSlidingCache:           features.FixedSlidingCache,
		FixedSlidingCacheBound:      features.FixedSlidingCacheBound,
		CompiledLayerDecode:         features.CompiledLayerDecode,
		PipelinedDecode:             features.PipelinedDecode,
	}
}

func rocmGemma4LinkedGenerationEngineFeatures(features Gemma4EngineFeatures) Gemma4EngineFeatures {
	linked := rocmGemma4EngineFeaturesFromModel(modelgemma4.LinkedGenerationEngineFeatures(rocmGemma4ModelEngineFeatures(features)))
	features.DirectGreedyToken = linked.DirectGreedyToken
	features.NativeMLPMatVec = linked.NativeMLPMatVec
	features.NativeLinearMatVec = linked.NativeLinearMatVec
	features.NativeQ6BitstreamMatVec = linked.NativeQ6BitstreamMatVec
	features.NativeAttentionOMatVec = linked.NativeAttentionOMatVec
	features.NativeFixedSlidingAttention = linked.NativeFixedSlidingAttention
	features.GenerationStream = linked.GenerationStream
	features.AsyncDecodePrefetch = linked.AsyncDecodePrefetch
	features.CompiledLayerDecode = linked.CompiledLayerDecode
	features.PipelinedDecode = linked.PipelinedDecode
	return features
}

func rocmApplyGemma4ConfigFeatureLabels(labels map[string]string, features Gemma4DeclaredFeatures) map[string]string {
	return modelgemma4.ApplyConfigFeatureLabels(labels, rocmGemma4ModelFeaturesFromDeclared(features))
}

func rocmApplyGemma4ConfigLabels(labels map[string]string, cfg modelgemma4.TextConfig) map[string]string {
	return modelgemma4.ApplyConfigLabels(labels, cfg)
}

func rocmApplyGemma4DeclaredFeatureLabels(labels map[string]string, features Gemma4DeclaredFeatures) map[string]string {
	return modelgemma4.ApplyDeclaredFeatureLabels(labels, rocmGemma4ModelFeaturesFromDeclared(features))
}
