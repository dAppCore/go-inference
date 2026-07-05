// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"dappco.re/go/inference/model/safetensors"
	"dappco.re/go/inference/profile"
)

func canonicalTextWeights(architecture string, raw map[string]safetensors.Tensor) map[string]safetensors.Tensor {
	if len(raw) == 0 {
		return raw
	}
	out := make(map[string]safetensors.Tensor, len(raw)*2)
	for name, tensor := range raw {
		out[name] = tensor
	}
	for name, tensor := range raw {
		canonical, ok := profile.CanonicalWeightName(architecture, name)
		if ok && canonical != "" {
			out[canonical] = tensor
		}
	}
	return out
}
