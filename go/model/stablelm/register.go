// SPDX-Licence-Identifier: EUPL-1.2

package stablelm

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

func init() {
	w := model.StandardWeightNames()
	model.RegisterArch(model.ArchSpec{ModelTypes: []string{"stablelm"}, Weights: w, Parse: func(data []byte) (model.ArchConfig, error) {
		var c Config
		if r := core.JSONUnmarshal(data, &c); !r.OK {
			return nil, core.E("stablelm.Parse", "config.json parse failed", nil)
		}
		return &c, nil
	}})
}
