// SPDX-Licence-Identifier: EUPL-1.2

package qwen3

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// init registers dense qwen3 with the reactive loader. The weight layout is the llama/mistral 2-norm
// shape — the MLP norm is post_attention_layernorm and there is no gemma post-attention norm — while the
// QK-norm names (q_norm/k_norm) are kept from the standard set so they bind automatically from a qwen3
// checkpoint. RMSNorm is plain (NormBiasOne stays false; qwen is not gemma).
func init() {
	w := model.StandardWeightNames()
	w.MLPNorm = ".post_attention_layernorm.weight" // qwen/llama: the MLP norm IS post_attention_layernorm
	w.PostAttnNorm = ""                            // no gemma-style post-attention norm
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{"qwen3"},
		Parse: func(data []byte) (model.ArchConfig, error) {
			var cfg Config
			if r := core.JSONUnmarshal(data, &cfg); !r.OK {
				return nil, core.NewError("qwen3.Parse: config.json parse failed")
			}
			return &cfg, nil
		},
		Weights: w,
	})
}
