// SPDX-Licence-Identifier: EUPL-1.2

package gemma3

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	_ "dappco.re/go/inference/model/arch/google/gemma3/gguf" // registers the gemma-3 GGUF export lane (model/gguf.RegisterQuantizeLane)
)

// init registers gemma3's ArchSpec for the gemma3 model_type ids, so the reactive loader (model.Load)
// parses + assembles a gemma3 checkpoint with no central switch — adding the arch is this init() + the
// Config in gemma3.go. gemma3 uses the standard gemma weight layout (4 norms + QK-norm) with the gemma
// "(1 + weight)" RMSNorm convention folded at load (NormBiasOne); Parse is the gemma3 config parser,
// Arch()/InferFromWeights are the Config's own methods.
func init() {
	w := model.StandardWeightNames()
	w.NormBiasOne = true // gemma (1+w) RMSNorm, folded into every norm weight at load
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{"gemma3", "gemma3_text"},
		Parse: func(data []byte) (model.ArchConfig, error) {
			var cfg Config
			if r := core.JSONUnmarshal(data, &cfg); !r.OK {
				return nil, core.NewError("gemma3.Parse: config.json parse failed")
			}
			return &cfg, nil
		},
		Weights: w,
	})
}
