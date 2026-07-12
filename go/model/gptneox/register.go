// SPDX-Licence-Identifier: EUPL-1.2

package gptneox

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// WeightNames returns tensor roles verified against each checkpoint's canonical safetensors keys.
func WeightNames(modelType string) model.WeightNames {
	w := model.StandardWeightNames()
	switch modelType {
	case "gpt_neox":
		w.Embed, w.LMHead, w.FinalNorm = "gpt_neox.embed_in", "embed_out", "gpt_neox.final_layer_norm.weight"
		w.LayerPrefix = "gpt_neox.layers.%d"
		w.AttnNorm, w.PostAttnNorm = ".input_layernorm.weight", ".post_attention_layernorm.weight"
		w.Q, w.K, w.V, w.O = ".attention.query_key_value", "", "", ".attention.dense"
		w.Gate, w.Up, w.Down = "", ".mlp.dense_h_to_4h", ".mlp.dense_4h_to_h"
	case "gptj":
		w.Embed, w.LMHead, w.FinalNorm = "transformer.wte", "lm_head", "transformer.ln_f.weight"
		w.LayerPrefix, w.AttnNorm, w.PostAttnNorm = "transformer.h.%d", ".ln_1.weight", ""
		w.Q, w.K, w.V, w.O = ".attn.q_proj", ".attn.k_proj", ".attn.v_proj", ".attn.out_proj"
		w.Gate, w.Up, w.Down = "", ".mlp.fc_in", ".mlp.fc_out"
	case "gpt_neo":
		w.Embed, w.LMHead, w.FinalNorm = "transformer.wte", "lm_head", "transformer.ln_f.weight"
		w.LayerPrefix, w.AttnNorm, w.PostAttnNorm = "transformer.h.%d", ".ln_1.weight", ".ln_2.weight"
		w.Q, w.K, w.V, w.O = ".attn.attention.q_proj", ".attn.attention.k_proj", ".attn.attention.v_proj", ".attn.attention.out_proj"
		w.Gate, w.Up, w.Down = "", ".mlp.c_fc", ".mlp.c_proj"
	}
	return w
}

func init() {
	parse := func(data []byte) (model.ArchConfig, error) {
		var cfg Config
		if r := core.JSONUnmarshal(data, &cfg); !r.OK {
			return nil, core.NewError("gptneox.Parse: config.json parse failed")
		}
		return &cfg, nil
	}
	for _, modelType := range []string{"gpt_neox", "gptj", "gpt_neo"} {
		model.RegisterArch(model.ArchSpec{ModelTypes: []string{modelType}, Parse: parse, Weights: WeightNames(modelType)})
	}
}
