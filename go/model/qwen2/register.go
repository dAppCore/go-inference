// SPDX-Licence-Identifier: EUPL-1.2

package qwen2

import "dappco.re/go/inference/model"

// init registers Qwen2ForCausalLM. The Qwen2.5 safetensors index proves the
// Llama/Mistral norm and projection names. QKV biases need no family mapping:
// neutral model.LoadLinear loads an adjacent .bias whenever it is present.
func init() {
	weights := model.StandardWeightNames()
	weights.MLPNorm = ".post_attention_layernorm.weight"
	weights.PostAttnNorm = ""
	weights.PostFFNorm = ""
	weights.QNorm = ""
	weights.KNorm = ""
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{"qwen2"},
		Parse: func(data []byte) (model.ArchConfig, error) {
			return ParseConfig(data)
		},
		Weights: weights,
	})
}
