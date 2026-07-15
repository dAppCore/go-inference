// SPDX-Licence-Identifier: EUPL-1.2

package smollm3

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

func init() {
	model.RegisterArch(model.ArchSpec{ModelTypes: []string{"smollm3"}, Weights: model.StandardWeightNames(), Parse: func(data []byte) (model.ArchConfig, error) {
		var c Config
		if r := core.JSONUnmarshal(data, &c); !r.OK {
			return nil, core.E("smollm3.Parse", "config.json parse failed", nil)
		}
		return &c, nil
	}})
}
