// SPDX-Licence-Identifier: EUPL-1.2

package bloom

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/attn"
	"dappco.re/go/inference/model/safetensors"
)

func init() {
	w := model.WeightNames{Embed: "word_embeddings", EmbedNorm: "word_embeddings_layernorm.weight", LMHead: "lm_head", FinalNorm: "ln_f.weight", LayerPrefix: "h.%d", AttnNorm: ".input_layernorm.weight", MLPNorm: ".post_attention_layernorm.weight", Q: ".self_attention.query", K: ".self_attention.key", V: ".self_attention.value", O: ".self_attention.dense", Gate: ".mlp.dense_h_to_4h", Down: ".mlp.dense_4h_to_h"}
	model.RegisterArch(model.ArchSpec{ModelTypes: []string{"bloom"}, Parse: func(data []byte) (model.ArchConfig, error) {
		var cfg Config
		if r := core.JSONUnmarshal(data, &cfg); !r.OK {
			return nil, core.E("bloom.Parse", "config.json parse failed", nil)
		}
		return &cfg, nil
	}, Weights: w, NormalizeConfig: func(tensors map[string]safetensors.Tensor, ac model.ArchConfig) map[string]safetensors.Tensor {
		cfg := ac.(*Config)
		for i := 0; ; i++ {
			p := core.Sprintf("h.%d.self_attention.", i)
			_, ok := tensors[p+"query_key_value.weight"]
			if !ok {
				break
			}
			headDim := cfg.HiddenSize / cfg.NumAttentionHeads
			tensors = attn.SplitInterleavedQKV(tensors, p+"query_key_value", p+"query", p+"key", p+"value", cfg.NumAttentionHeads, headDim)
		}
		return tensors
	}})
}
