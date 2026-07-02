// SPDX-Licence-Identifier: EUPL-1.2

package inference

import (
	"slices"

	"dappco.re/go/inference/state"
)

type ModelIdentity = state.ModelIdentity
type TokenizerIdentity = state.TokenizerIdentity
type AdapterIdentity = state.AdapterIdentity
type RuntimeIdentity = state.RuntimeIdentity
type SamplerConfig = state.SamplerConfig
type StateRef = state.StateRef
type StateBundle = state.Bundle
type ProjectSeedMode = state.ProjectSeedMode
type ProjectSeedOptions = state.ProjectSeedOptions
type ProjectSeed = state.ProjectSeed
type ProjectSeedWakeOptions = state.ProjectSeedWakeOptions
type ProjectSeedContinuationOptions = state.ProjectSeedContinuationOptions
type ProjectSeedContinuationPlan = state.ProjectSeedContinuationPlan
type WakeCompatibilityReport = state.WakeCompatibilityReport

const (
	ProjectSeedStateCheckpoint = state.ProjectSeedStateCheckpoint
	ProjectSeedReuseCurrent    = state.ProjectSeedReuseCurrent
	ProjectSeedSummaryWindow   = state.ProjectSeedSummaryWindow
	ProjectSeedHybrid          = state.ProjectSeedHybrid
)

var (
	NewProjectSeed         = state.NewProjectSeed
	CheckWakeCompatibility = state.CheckWakeCompatibility
)

// SamplerConfigFromGenerateConfig converts generation options to portable
// sampler metadata while preserving slice ownership.
func SamplerConfigFromGenerateConfig(cfg GenerateConfig) SamplerConfig {
	return SamplerConfig{
		MaxTokens:     cfg.MaxTokens,
		Temperature:   cfg.Temperature,
		TopK:          cfg.TopK,
		TopP:          cfg.TopP,
		MinP:          cfg.MinP,
		RepeatPenalty: cfg.RepeatPenalty,
		StopTokens:    slices.Clone(cfg.StopTokens),
		ReturnLogits:  cfg.ReturnLogits,
	}
}

// GenerateConfigFromSamplerConfig converts portable sampler metadata back into
// generation options while preserving slice ownership.
func GenerateConfigFromSamplerConfig(cfg SamplerConfig) GenerateConfig {
	return GenerateConfig{
		MaxTokens:     cfg.MaxTokens,
		Temperature:   cfg.Temperature,
		TopK:          cfg.TopK,
		TopP:          cfg.TopP,
		MinP:          cfg.MinP,
		StopTokens:    slices.Clone(cfg.StopTokens),
		RepeatPenalty: cfg.RepeatPenalty,
		ReturnLogits:  cfg.ReturnLogits,
	}
}
