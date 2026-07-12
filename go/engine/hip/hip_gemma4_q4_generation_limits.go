// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

func hipGemma4Q4ResolveGenerateContext(model *hipLoadedModel, promptTokens []int32, generate inference.GenerateConfig) (inference.GenerateConfig, error) {
	if model == nil {
		return generate, core.E(hipGemma4Q4Layer0Operation, "loaded model is required", nil)
	}
	contextSize := model.contextSize
	if contextSize <= 0 {
		contextSize = defaultContextLengthCap
	}
	remaining := contextSize - len(promptTokens)
	if remaining <= 0 {
		return generate, core.E(hipGemma4Q4Layer0Operation, "prompt reaches model context window", nil)
	}
	if generate.MaxTokens > remaining {
		return generate, core.E(hipGemma4Q4Layer0Operation, "requested max tokens exceed remaining model context window", nil)
	}
	if generate.MaxTokens > 0 {
		return generate, nil
	}
	generate.MaxTokens = remaining
	return generate, nil
}

func hipLoadedGemma4Q4GenerateLinked(model *hipLoadedModel) bool {
	if model == nil {
		return false
	}
	if model.gemma4TextConfig.EnableMoEBlock {
		return hipLoadedGemma4MoERuntimeReady(model)
	}
	if rocmGemma4LabelsVetoGenerateLinked(model.modelLabels) {
		return false
	}
	identity := model.modelIdentity()
	if rocmGemma4LabelsVetoGenerateLinked(identity.Labels) {
		return false
	}
	if model.engineProfile.Matched() && model.engineProfile.Family == "gemma4" {
		return model.engineProfile.Gemma4EngineFeatures.GenerateLinked()
	}
	if isROCmGemma4Architecture(identity.Architecture) {
		identity.QuantType = model.modelLabels["gemma4_quant_mode"]
	}
	return Gemma4EngineFeaturesForIdentity(identity).GenerateLinked()
}
