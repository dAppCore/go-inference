// SPDX-Licence-Identifier: EUPL-1.2

// Package builtin registers the model-profile factories the root ROCm package
// enables by default.
package builtin

import (
	"dappco.re/go/inference/engine/hip/model"
	"dappco.re/go/inference/engine/hip/model/architecture"
	_ "dappco.re/go/inference/engine/hip/model/gemma4" // registers Gemma-4 before the generic fallback
)

func init() {
	model.RegisterProfileFactory(architecture.ProfileFactory{})
}
