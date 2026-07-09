// SPDX-Licence-Identifier: EUPL-1.2

package mistral

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// init registers mistral's ArchSpec for the Mistral model_type ids, so the engine's reactive loader
// (model.Load) parses + assembles it with no central switch. Mistral3ForConditionalGeneration declares
// "mistral3" / "ministral3"; the bare text variants declare "mistral" / "ministral". Mistral is a gemma4
// SUBSET, so its Weights are the standard layout with two overrides: the pre-MLP norm is
// post_attention_layernorm (Mistral's name for it), and there is no gemma-style post-attention norm.
func init() {
	w := model.StandardWeightNames()
	w.MLPNorm = ".post_attention_layernorm.weight"
	w.PostAttnNorm = ""
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{"mistral3", "ministral3", "mistral", "ministral"},
		Parse: func(data []byte) (model.ArchConfig, error) {
			var cfg Config
			if r := core.JSONUnmarshal(data, &cfg); !r.OK {
				return nil, core.NewError("mistral.Parse: config.json parse failed")
			}
			return &cfg, nil
		},
		Weights: w,
	})
}
