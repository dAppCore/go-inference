// SPDX-Licence-Identifier: EUPL-1.2

package cohere

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

func init() {
	w := model.StandardWeightNames()
	w.MLPNorm = ""
	w.PostAttnNorm = ""
	w.PostFFNorm = ""
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{"cohere", "cohere2"},
		Parse: func(data []byte) (model.ArchConfig, error) {
			var cfg Config
			if r := core.JSONUnmarshal(data, &cfg); !r.OK {
				return nil, core.NewError("cohere.Parse: config.json parse failed")
			}
			return &cfg, nil
		},
		Weights: w,
	})
}
