// SPDX-Licence-Identifier: EUPL-1.2

package falcon

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

func init() {
	w := model.WeightNames{Embed: "transformer.word_embeddings", LMHead: "lm_head", FinalNorm: "transformer.ln_f.weight", LayerPrefix: "transformer.h.%d", AttnNorm: ".input_layernorm.weight", MLPNorm: ".post_attention_layernorm.weight", Q: ".self_attention.query", K: ".self_attention.key", V: ".self_attention.value", O: ".self_attention.dense", Gate: ".mlp.dense_h_to_4h", Down: ".mlp.dense_4h_to_h"}
	model.RegisterArch(model.ArchSpec{ModelTypes: []string{"falcon"}, Parse: func(data []byte) (model.ArchConfig, error) {
		var cfg Config
		if r := core.JSONUnmarshal(data, &cfg); !r.OK {
			return nil, core.E("falcon.Parse", "config.json parse failed", nil)
		}
		return &cfg, nil
	}, Weights: w, NormalizeConfig: func(tensors map[string]safetensors.Tensor, ac model.ArchConfig) map[string]safetensors.Tensor {
		cfg := ac.(*Config)
		// Falcon-RW/old decoder checkpoints interleave Q,K,V per head.
		for i := 0; ; i++ {
			p := core.Sprintf("transformer.h.%d.self_attention.", i)
			_, ok := tensors[p+"query_key_value.weight"]
			if !ok {
				break
			}
			headDim := cfg.HiddenSize / cfg.NumAttentionHeads
			if cfg.NewDecoderArchitecture && cfg.NumKVHeads > 0 {
				tensors = model.SplitGroupedQKV(tensors, p+"query_key_value", p+"query", p+"key", p+"value", cfg.NumAttentionHeads, cfg.NumKVHeads, headDim)
			} else if cfg.MultiQuery {
				tensors = model.SplitContiguousQKV(tensors, p+"query_key_value", p+"query", p+"key", p+"value", cfg.NumAttentionHeads*headDim, headDim)
			} else {
				tensors = model.SplitInterleavedQKV(tensors, p+"query_key_value", p+"query", p+"key", p+"value", cfg.NumAttentionHeads, headDim)
			}
		}
		return tensors
	}})
}
