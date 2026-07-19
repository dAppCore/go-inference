// SPDX-Licence-Identifier: EUPL-1.2

package gptoss

import "dappco.re/go/inference/model"

// init registers GPT-OSS's model_type, its Parse, AND its Weights mapping (weights.go) — the #18 factory
// pattern's data half. model.Assemble would react to this mapping correctly (LoadLinear resolves every
// named tensor including the auto-probed additive biases), but Config.Arch still refuses (see config.go),
// so model.Load never reaches Assemble for gpt_oss yet — Weights is carried here anyway, same as
// qwen35.ArchSpec, so the registration is complete and the boundary is purely Arch's, not a missing
// mapping.
func init() {
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{"gpt_oss"},
		Parse: func(data []byte) (model.ArchConfig, error) {
			return ParseConfig(data)
		},
		Weights: WeightNames(),
	})
}
