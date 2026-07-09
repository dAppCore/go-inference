// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"maps"

	"dappco.re/go/inference/eval/profile"
	"dappco.re/go/inference/model/safetensors"
)

func canonicalTextWeights(architecture string, raw map[string]safetensors.Tensor) map[string]safetensors.Tensor {
	if len(raw) == 0 {
		return raw
	}
	out := make(map[string]safetensors.Tensor, len(raw)*2)
	maps.Copy(out, raw)
	for name, tensor := range raw {
		canonical, ok := profile.CanonicalWeightName(architecture, name)
		if ok && canonical != "" {
			out[canonical] = tensor
		}
	}
	return out
}
