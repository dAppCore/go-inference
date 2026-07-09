// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// init registers gemma4's ArchSpec for the model_type ids the family declares, so the engine's reactive
// loader (model.Load) parses + assembles it with no central switch — adding an arch is a config + this
// init(). gemma4_unified is the multimodal wrapper; its nested text_config.model_type is gemma4_text —
// both registered here. Parse is the faithful config parser; Weights is the HF/gemma weight layout
// (the superset, which gemma4 is); InferFromWeights + Arch() are Gemma4TextConfig's own methods.
func init() {
	parse := func(data []byte) (model.ArchConfig, error) {
		cfg, err := parseGemma4Config(data)
		if err != nil {
			return nil, err
		}
		return cfg, nil
	}
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{"gemma4", "gemma4_text", "gemma4_unified"},
		Parse:      parse,
		Weights:    model.StandardWeightNames(),
		Normalize: func(tensors map[string]safetensors.Tensor) map[string]safetensors.Tensor {
			return canonicalTextWeights("gemma4", tensors)
		},
		Vision: func(tensors map[string]safetensors.Tensor, cfg model.ArchConfig) (*model.LoadedVision, error) {
			textCfg, ok := cfg.(*Gemma4TextConfig)
			if !ok {
				return nil, nil
			}
			return AssembleVision(SanitizeVisionWeights(tensors), textCfg)
		},
		UnifiedVision: func(tensors map[string]safetensors.Tensor, cfg model.ArchConfig) (*model.LoadedUnifiedVision, error) {
			textCfg, ok := cfg.(*Gemma4TextConfig)
			if !ok {
				return nil, nil
			}
			return AssembleUnifiedVision(tensors, textCfg)
		},
		Audio: func(tensors map[string]safetensors.Tensor, cfg model.ArchConfig) (*model.LoadedAudio, error) {
			textCfg, ok := cfg.(*Gemma4TextConfig)
			if !ok {
				return nil, nil
			}
			return AssembleAudio(SanitizeAudioWeights(tensors), textCfg)
		},
	})
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{"diffusion_gemma"},
		Parse:      parse,
		Weights:    model.StandardWeightNames(),
		Normalize: func(tensors map[string]safetensors.Tensor) map[string]safetensors.Tensor {
			return canonicalTextWeights("diffusion_gemma", tensors)
		},
		Diffusion: func(tensors map[string]safetensors.Tensor, cfg model.ArchConfig) (*model.LoadedDiffusion, error) {
			textCfg, ok := cfg.(*Gemma4TextConfig)
			if !ok {
				return nil, nil
			}
			return AssembleDiffusion(tensors, textCfg)
		},
	})
}
