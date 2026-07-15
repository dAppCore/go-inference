// SPDX-Licence-Identifier: EUPL-1.2

package gptoss

import "dappco.re/go/inference/model"

// init registers GPT-OSS's model_type so the reactive loader recognises it. There is no Weights layout to
// declare yet: Config.Arch refuses (see config.go) before model.Load would reach model.Assemble — the MoE
// expert tensor layout and mxfp4 quantisation have not been confirmed against a real GPT-OSS safetensors
// index, so no weight-name convention is claimed here. That mapping is the follow-up serving lane.
func init() {
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{"gpt_oss"},
		Parse: func(data []byte) (model.ArchConfig, error) {
			return ParseConfig(data)
		},
	})
}
