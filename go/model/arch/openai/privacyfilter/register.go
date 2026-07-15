// SPDX-Licence-Identifier: EUPL-1.2

package privacyfilter

import "dappco.re/go/inference/model"

// init registers openai/privacy-filter's model_type so the reactive loader recognises it. There is no
// Weights layout to declare: Config.Arch always refuses (see config.go) before model.Load would ever reach
// model.Assemble, so no weight-name convention has been confirmed against a real safetensors index.
func init() {
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{"openai_privacy_filter"},
		Parse: func(data []byte) (model.ArchConfig, error) {
			return ParseConfig(data)
		},
	})
}
