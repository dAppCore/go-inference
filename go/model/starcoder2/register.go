// SPDX-Licence-Identifier: EUPL-1.2

package starcoder2

import "dappco.re/go/inference/model"

// init registers StarCoder2's indexed Hugging Face tensor layout.
func init() {
	weights := model.StandardWeightNames()
	weights.MLPNorm = ".post_attention_layernorm.weight"
	weights.PostAttnNorm = ""
	weights.QNorm = ""
	weights.KNorm = ""
	weights.Gate = ".mlp.c_fc"
	weights.Up = ".mlp.c_fc"
	weights.Down = ".mlp.c_proj"
	weights.PostFFNorm = ""
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{"starcoder2"},
		Parse: func(data []byte) (model.ArchConfig, error) {
			return ParseConfig(data)
		},
		Weights: weights,
	})
}
