// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	core "dappco.re/go"
	"dappco.re/go/inference/safetensors"
)

// NormalizeWrapperNames makes every "language_model."-prefixed tensor (the multimodal wrapper layout
// some packs use) ALSO addressable by its stripped "model.…" name, so an assembler's bare lookups
// work whether or not the checkpoint nests the text model under the wrapper.
// Returns the input unchanged when there is no such prefix (flat text-only packs). Shared by every
// arch's assembler — the one place the wrapper convention is known.
func NormalizeWrapperNames(t map[string]safetensors.Tensor) map[string]safetensors.Tensor {
	const pfx = "language_model."
	has := false
	for k := range t {
		if core.HasPrefix(k, pfx) {
			has = true
			break
		}
	}
	if !has {
		return t
	}
	out := make(map[string]safetensors.Tensor, len(t))
	for k, v := range t {
		out[k] = v
		if core.HasPrefix(k, pfx) {
			out[k[len(pfx):]] = v
		}
	}
	return out
}
