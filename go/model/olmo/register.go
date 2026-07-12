// SPDX-Licence-Identifier: EUPL-1.2

package olmo

import (
	"dappco.re/go/inference/model"
)

func weightNames(modelType string) model.WeightNames {
	w := model.StandardWeightNames()
	w.AttnNorm, w.MLPNorm = "", ""
	if modelType == "olmo" {
		w.FinalNorm, w.PostAttnNorm, w.PostFFNorm = "", "", ""
		w.QNorm, w.KNorm = "", ""
	}
	return w
}

func init() {
	parse := func(data []byte) (model.ArchConfig, error) {
		return ParseConfig(data)
	}
	for _, modelType := range []string{"olmo", "olmo2"} {
		model.RegisterArch(model.ArchSpec{
			ModelTypes: []string{modelType},
			Parse:      parse,
			Weights:    weightNames(modelType),
		})
	}
}
