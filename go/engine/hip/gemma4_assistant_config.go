// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"dappco.re/go/inference"
	modelgemma4 "dappco.re/go/inference/engine/hip/model/gemma4"
)

func applyROCmGemma4AssistantConfigLabels(inspection *inference.ModelPackInspection, cfg rocmModelPackConfigProbe) {
	if inspection == nil || !isROCmGemma4AssistantArchitecture(rocmConfigArchitecture(cfg)) {
		return
	}
	cfgProbe := rocmGemma4AssistantConfigProbe(cfg)
	if ordered, ok := rocmConfigUseOrderedEmbeddings(cfg); ok {
		cfgProbe.UseOrderedEmbeddings = ordered
		cfgProbe.UseOrderedEmbeddingsSet = true
	}
	var contradictsOfficial bool
	inspection.Labels, contradictsOfficial = modelgemma4.ApplyAssistantConfigLabels(inspection.Labels, cfgProbe)
	if contradictsOfficial {
		inspection.Labels["attached_drafter_official_pair_verified"] = "false"
		inspection.Labels["attached_drafter_gemma4_family_pair_verified"] = "false"
		inspection.Notes = append(inspection.Notes, "Gemma4 assistant config does not match the locked official E2B assistant layout; production MTP promotion must not use static official-pair evidence")
	}
}

func rocmGemma4AssistantConfigProbe(cfg rocmModelPackConfigProbe) modelgemma4.AssistantConfig {
	return modelgemma4.AssistantConfig{
		BackboneHiddenSize:       firstPositiveInt(cfg.BackboneHiddenSize, cfg.TextConfig.BackboneHiddenSize),
		NumCentroids:             firstPositiveInt(cfg.NumCentroids, cfg.TextConfig.NumCentroids),
		CentroidIntermediateTopK: firstPositiveInt(cfg.CentroidIntermediateTopK, cfg.TextConfig.CentroidIntermediateTopK),
		NumLayers:                firstPositiveInt(cfg.NumHiddenLayers, cfg.NumLayers, cfg.TextConfig.NumHiddenLayers, cfg.TextConfig.NumLayers),
		VocabSize:                firstPositiveInt(cfg.VocabSize, cfg.TextConfig.VocabSize),
	}
}

func rocmConfigUseOrderedEmbeddings(cfg rocmModelPackConfigProbe) (bool, bool) {
	switch {
	case cfg.UseOrderedEmbeddings != nil:
		return *cfg.UseOrderedEmbeddings, true
	case cfg.TextConfig.UseOrderedEmbeddings != nil:
		return *cfg.TextConfig.UseOrderedEmbeddings, true
	default:
		return false, false
	}
}
