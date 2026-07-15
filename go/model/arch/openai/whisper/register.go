// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import "dappco.re/go/inference/model"

// init registers Whisper's model_type so the reactive loader recognises it. There is no Weights layout to
// declare: Config.Arch always refuses (see config.go) before model.Load would ever reach model.Assemble —
// Whisper's encoder/cross-attention-decoder tensors do not fit the Assemble weight-role convention anyway.
func init() {
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{"whisper"},
		Parse: func(data []byte) (model.ArchConfig, error) {
			return ParseConfig(data)
		},
	})
}
