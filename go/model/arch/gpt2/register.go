// SPDX-Licence-Identifier: EUPL-1.2

package gpt2

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

func init() {
	w := model.WeightNames{
		Embed: "transformer.wte", PositionEmbed: "transformer.wpe", FinalNorm: "transformer.ln_f.weight",
		LayerPrefix: "transformer.h.%d", AttnNorm: ".ln_1.weight", Q: ".attn.q_proj", K: ".attn.k_proj", V: ".attn.v_proj", O: ".attn.c_proj",
		MLPNorm: ".ln_2.weight", Gate: ".mlp.c_fc", Up: ".mlp.c_fc", Down: ".mlp.c_proj",
	}
	model.RegisterArch(model.ArchSpec{ModelTypes: []string{"gpt2", "gpt_bigcode", "starcoder"}, Weights: w,
		Parse: func(data []byte) (model.ArchConfig, error) {
			var c Config
			if r := core.JSONUnmarshal(data, &c); !r.OK {
				return nil, core.NewError("gpt2.Parse: config.json parse failed")
			}
			return &c, nil
		},
		Normalize: NormalizeWeights,
	})
}
