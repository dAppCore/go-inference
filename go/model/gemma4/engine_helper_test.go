// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/safetensors"
)

// gemma4Assemble runs the engine's generic assembler with gemma4's weight layout — gemma4 no longer
// owns an Assemble (model.Assemble does), so the tests that exercise the gemma4-flavoured assembly path
// go through this thin wrapper.
func gemma4Assemble(ts map[string]safetensors.Tensor, arch model.Arch) (*model.LoadedModel, error) {
	return model.Assemble(ts, arch, model.StandardWeightNames())
}
