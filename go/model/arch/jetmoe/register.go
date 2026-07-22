// SPDX-Licence-Identifier: EUPL-1.2

package jetmoe

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

func init() {
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{"jetmoe"},
		Parse: func(data []byte) (model.ArchConfig, error) {
			var cfg Config
			if r := core.JSONUnmarshal(data, &cfg); !r.OK {
				return nil, core.NewError("jetmoe.Parse: config.json parse failed")
			}
			return &cfg, nil
		},
	})
}
