// SPDX-Licence-Identifier: EUPL-1.2

package opt

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

func init() {
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{"opt"},
		Weights:    WeightNames(),
		Parse: func(data []byte) (model.ArchConfig, error) {
			var config Config
			if result := core.JSONUnmarshal(data, &config); !result.OK {
				return nil, core.NewError("opt.Parse: config.json parse failed")
			}
			return &config, nil
		},
	})
}
