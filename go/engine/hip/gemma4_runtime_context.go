// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

func (m *rocmModel) applyGenerateOpts(opts []inference.GenerateOption) inference.GenerateConfig {
	cfg := cloneGenerateConfig(inference.ApplyGenerateOpts(opts))
	if m == nil || !isROCmGemma4Architecture(m.modelIdentity().Architecture) {
		return cfg
	}
	explicit := inference.GenerateConfig{}
	for _, opt := range opts {
		if opt != nil {
			opt(&explicit)
		}
	}
	if explicit.MaxTokens == 0 {
		cfg.MaxTokens = 0
	}
	return cfg
}

func (m *rocmModel) resolveGenerateGemma4Context(prompt string, cfg *inference.GenerateConfig, operation string) (int, error) {
	promptTokens := m.promptTokenCount(prompt)
	return promptTokens, m.resolveGemma4ContextTokens(promptTokens, cfg, operation)
}

func (m *rocmModel) resolveChatGemma4Context(messages []inference.Message, cfg *inference.GenerateConfig) (int, error) {
	promptTokens := m.chatPromptTokenCount(messages)
	return promptTokens, m.resolveGemma4ContextTokens(promptTokens, cfg, "rocm.Chat")
}

func (m *rocmModel) resolveChatGemma4ContextWithTemplateConfig(messages []inference.Message, cfg *inference.GenerateConfig, template gemma4ChatTemplateConfig) (int, error) {
	promptTokens := m.chatPromptTokenCountWithTemplateConfig(messages, template)
	return promptTokens, m.resolveGemma4ContextTokens(promptTokens, cfg, "rocm.Chat")
}

func (m *rocmModel) resolveBatchGenerateGemma4Context(prompts []string, cfg *inference.GenerateConfig) error {
	if m == nil || !isROCmGemma4Architecture(m.modelIdentity().Architecture) {
		return nil
	}
	remaining := 0
	for _, prompt := range prompts {
		promptTokens := m.promptTokenCount(prompt)
		maxTokens, err := m.gemma4MaxTokensForPromptTokens(promptTokens, 0, "rocm.BatchGenerate")
		if err != nil {
			return err
		}
		if remaining == 0 || maxTokens < remaining {
			remaining = maxTokens
		}
	}
	if cfg != nil && cfg.MaxTokens <= 0 {
		cfg.MaxTokens = remaining
	}
	if cfg != nil && cfg.MaxTokens > remaining {
		return core.E("rocm.BatchGenerate", "max tokens exceed remaining model context window", nil)
	}
	return nil
}

func (m *rocmModel) resolveGemma4ContextTokens(promptTokens int, cfg *inference.GenerateConfig, operation string) error {
	if m == nil || !isROCmGemma4Architecture(m.modelIdentity().Architecture) {
		return nil
	}
	requested := 0
	if cfg != nil {
		requested = cfg.MaxTokens
	}
	maxTokens, err := m.gemma4MaxTokensForPromptTokens(promptTokens, requested, operation)
	if err != nil {
		return err
	}
	if cfg != nil && cfg.MaxTokens <= 0 {
		cfg.MaxTokens = maxTokens
	}
	return nil
}

func (m *rocmModel) gemma4MaxTokensForPromptTokens(promptTokens, requested int, operation string) (int, error) {
	contextLength := defaultContextLengthCap
	if m != nil {
		if identityContext := m.modelIdentity().ContextLength; identityContext > 0 {
			contextLength = identityContext
		}
	}
	remaining := contextLength - promptTokens
	if remaining <= 0 {
		return 0, core.E(operation, "prompt reaches model context window", nil)
	}
	if requested > 0 {
		if requested > remaining {
			return 0, core.E(operation, "max tokens exceed remaining model context window", nil)
		}
		return requested, nil
	}
	return remaining, nil
}

func (m *rocmModel) benchmarkMaxTokens(prompts []string, requested int) (int, error) {
	if m == nil || !isROCmGemma4Architecture(m.modelIdentity().Architecture) {
		if requested > 0 {
			return requested, nil
		}
		return 32, nil
	}
	contextLength := m.modelIdentity().ContextLength
	if contextLength <= 0 {
		contextLength = defaultContextLengthCap
	}
	remaining := contextLength
	for _, prompt := range prompts {
		promptTokens := m.promptTokenCount(prompt)
		if promptTokens >= contextLength {
			return 0, core.E("rocm.Benchmark", "prompt reaches model context window", nil)
		}
		if current := contextLength - promptTokens; current < remaining {
			remaining = current
		}
	}
	if remaining <= 0 {
		return 0, core.E("rocm.Benchmark", "prompt reaches model context window", nil)
	}
	if requested > 0 {
		if requested > remaining {
			return 0, core.E("rocm.Benchmark", "max tokens exceed remaining model context window", nil)
		}
		return requested, nil
	}
	return remaining, nil
}

func (m *rocmModel) qualityProbeMaxTokens(probes []inference.QualityProbe, requested int) (int, error) {
	if m == nil || !isROCmGemma4Architecture(m.modelIdentity().Architecture) {
		if requested > 0 {
			return requested, nil
		}
		return 32, nil
	}
	contextLength := m.modelIdentity().ContextLength
	if contextLength <= 0 {
		contextLength = defaultContextLengthCap
	}
	remaining := contextLength
	for _, probe := range probes {
		prompt := m.generatedPrompt(firstNonEmptyString(probe.Prompt, probe.Name))
		promptTokens := m.promptTokenCount(prompt)
		if promptTokens >= contextLength {
			return 0, core.E("rocm.Evaluate", "quality probe reaches model context window", nil)
		}
		if current := contextLength - promptTokens; current < remaining {
			remaining = current
		}
	}
	if remaining <= 0 {
		return 0, core.E("rocm.Evaluate", "quality probe reaches model context window", nil)
	}
	if requested > 0 {
		if requested > remaining {
			return 0, core.E("rocm.Evaluate", "max tokens exceed remaining model context window", nil)
		}
		return requested, nil
	}
	return remaining, nil
}

func rocmDecodeHelperStatusLabel(status hipKernelStatus, gemma4Q4GenerateLinked bool) string {
	if gemma4Q4GenerateLinked {
		return "experimental"
	}
	if normalizeHIPKernelStatus(status).Decode == hipKernelStatusLinked {
		return "experimental"
	}
	return "planned"
}

func rocmReportKernelStatusForModel(status hipKernelStatus, model inference.ModelIdentity) hipKernelStatus {
	status = normalizeHIPKernelStatus(status)
	if !isROCmGemma4Architecture(model.Architecture) || Gemma4EngineFeaturesForIdentity(model).GenerateLinked() {
		return status
	}
	status.Decode = hipKernelStatusNotLinked
	status.Prefill = hipKernelStatusNotLinked
	return status
}
