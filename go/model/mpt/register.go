// SPDX-Licence-Identifier: EUPL-1.2

package mpt

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/attn"
	"dappco.re/go/inference/model/safetensors"
)

func init() {
	w := model.WeightNames{Embed: "transformer.wte", PositionEmbed: "transformer.wpe", LMHead: "lm_head", FinalNorm: "transformer.norm_f.weight", LayerPrefix: "transformer.blocks.%d", AttnNorm: ".norm_1.weight", MLPNorm: ".norm_2.weight", Q: ".attn.q_proj", K: ".attn.k_proj", V: ".attn.v_proj", O: ".attn.out_proj", Gate: ".ffn.up_proj", Up: ".ffn.up_proj", Down: ".ffn.down_proj"}
	model.RegisterArch(model.ArchSpec{ModelTypes: []string{"mpt"}, Weights: w, Parse: func(data []byte) (model.ArchConfig, error) {
		var c Config
		if r := core.JSONUnmarshal(data, &c); !r.OK {
			return nil, core.E("mpt.Parse", "config.json parse failed", nil)
		}
		return &c, nil
	}, NormalizeConfig: func(t map[string]safetensors.Tensor, ac model.ArchConfig) map[string]safetensors.Tensor {
		c := ac.(*Config)
		rows := c.DModel
		for i := 0; i < c.NLayers; i++ {
			p := core.Sprintf("transformer.blocks.%d.attn.", i)
			t = attn.SplitContiguousQKV(t, p+"Wqkv.weight", p+"q_proj.weight", p+"k_proj.weight", p+"v_proj.weight", rows, rows)
		}
		return t
	}})
}
