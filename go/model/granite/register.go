// SPDX-Licence-Identifier: EUPL-1.2

package granite

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

func init() {
	weights := model.StandardWeightNames()
	weights.MLPNorm = ".post_attention_layernorm.weight"
	weights.PostAttnNorm = ""
	weights.PostFFNorm = ""
	weights.QNorm = ""
	weights.KNorm = ""
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{"granite"},
		Parse: func(data []byte) (model.ArchConfig, error) {
			r := ParseConfig(data)
			if !r.OK {
				return nil, core.NewError(r.Error())
			}
			return r.Value.(*Config), nil
		},
		Weights: weights,
	})
}
