// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"dappco.re/go/inference"
	modelgemma4 "dappco.re/go/inference/engine/hip/model/gemma4"
)

type gemma4ChatTemplateConfig struct {
	EnableThinking     bool
	LargeVariant       bool
	NoGenerationPrompt bool
	Continuation       bool
}

func formatGemma4ChatTemplate(messages []inference.Message) string {
	return formatGemma4ChatTemplateWithConfig(messages, gemma4ChatTemplateConfig{})
}

func formatGemma4ChatTemplateWithConfig(messages []inference.Message, cfg gemma4ChatTemplateConfig) string {
	return modelgemma4.FormatChatTemplateWithConfig(messages, modelgemma4.ChatTemplateConfig{
		EnableThinking:     cfg.EnableThinking,
		LargeVariant:       cfg.LargeVariant,
		NoGenerationPrompt: cfg.NoGenerationPrompt,
		Continuation:       cfg.Continuation,
	})
}

func gemma4ChatTemplateConfigForIdentity(model inference.ModelIdentity, cfg inference.GenerateConfig, continuation bool) gemma4ChatTemplateConfig {
	enableThinking := ROCmDefaultThinkingEnabled(model.Architecture)
	if cfg.EnableThinking != nil {
		enableThinking = *cfg.EnableThinking
	}
	return gemma4ChatTemplateConfig{
		EnableThinking: enableThinking,
		LargeVariant:   rocmGemma4NeedsThoughtChannelSuppressor(model),
		Continuation:   continuation,
	}
}

func (m *rocmModel) gemma4ChatTemplateConfig(cfg inference.GenerateConfig, continuation bool) gemma4ChatTemplateConfig {
	if m == nil {
		return gemma4ChatTemplateConfig{}
	}
	return gemma4ChatTemplateConfigForIdentity(m.modelIdentity(), cfg, continuation)
}

func (model *hipLoadedModel) gemma4ChatTemplateConfig(cfg inference.GenerateConfig, continuation bool) gemma4ChatTemplateConfig {
	if model == nil {
		return gemma4ChatTemplateConfig{}
	}
	return gemma4ChatTemplateConfigForIdentity(inference.ModelIdentity{
		Architecture: model.modelInfo.Architecture,
		VocabSize:    model.modelInfo.VocabSize,
		NumLayers:    model.modelInfo.NumLayers,
		HiddenSize:   model.modelInfo.HiddenSize,
		QuantBits:    model.modelInfo.QuantBits,
		QuantGroup:   model.modelInfo.QuantGroup,
		Labels:       cloneStringMap(model.modelLabels),
	}, cfg, continuation)
}

func rocmGemma4SizeNeedsThoughtChannelSuppressor(size string) bool {
	return modelgemma4.SizeNeedsThoughtChannelSuppressor(size)
}

func rocmGemma4NeedsThoughtChannelSuppressor(model inference.ModelIdentity) bool {
	if needs, ok := modelgemma4.NeedsThoughtChannelSuppressorForIdentity(model); ok {
		return needs
	}
	return rocmGemma4SizeNeedsThoughtChannelSuppressor(firstNonEmptyString(model.Labels["gemma4_size"], model.Labels["production_quant_size"], rocmGemma4ModelPackSize(model, model.Path)))
}

func initialGemma4SystemRole(messages []inference.Message) bool {
	return len(messages) > 0 && gemma4MessageRole(messages[0].Role) == "system"
}

func gemma4MessageRole(role string) string {
	return modelgemma4.MessageRole(role)
}

func gemma4NormalizedRole(role string) string {
	return modelgemma4.NormalizedRole(role)
}

func stripGemma4ThinkingChannels(text string) string {
	return modelgemma4.StripThinkingChannels(text)
}
