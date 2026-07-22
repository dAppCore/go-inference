// SPDX-Licence-Identifier: EUPL-1.2

package llama

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	_ "dappco.re/go/inference/model/arch/meta-llama/llama/gguf"
)

// init registers dense Hugging Face Llama checkpoints with the reactive loader.
func init() {
	w := model.StandardWeightNames()
	w.MLPNorm = ".post_attention_layernorm.weight"
	w.PostAttnNorm = ""
	w.PostFFNorm = ""
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{"llama"},
		Parse: func(data []byte) (model.ArchConfig, error) {
			var cfg Config
			if r := core.JSONUnmarshal(data, &cfg); !r.OK {
				return nil, core.NewError("llama.Parse: config.json parse failed")
			}
			return &cfg, nil
		},
		Weights: w,
	})
}
