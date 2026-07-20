// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// NormalizeWrapperNames makes every "language_model."-prefixed tensor (the multimodal wrapper layout
// some packs use) ALSO addressable by its stripped "model.…" name, so an assembler's bare lookups
// work whether or not the checkpoint nests the text model under the wrapper.
// Returns the input unchanged when there is nothing to alias — no such prefix (flat text-only
// packs), or every wrapped tensor's stripped alias is ALREADY present (a map this function has
// aliased before). That same-map return is load-bearing for the owned-tensor adoption (#60):
// model.Load aliases dm.Tensors ONCE, so Assemble's own call passes the SAME map through and
// LoadLinear's b1→b2 repack writeback lands where the post-Assemble adoption sweep (and the
// zero-copy binder behind it) can see it — a fresh transient map here silently swallowed the
// writeback for every wrapped checkpoint (Bonsai) while flat packs worked. A pre-existing
// stripped name is never overwritten by an alias (the checkpoint's own tensor stays
// authoritative; the old fresh-map build left that collision to map iteration order).
func NormalizeWrapperNames(t map[string]safetensors.Tensor) map[string]safetensors.Tensor {
	const pfx = "language_model."
	needsAlias := false
	for k := range t {
		if core.HasPrefix(k, pfx) {
			if _, ok := t[k[len(pfx):]]; !ok {
				needsAlias = true
				break
			}
		}
	}
	if !needsAlias {
		return t
	}
	out := make(map[string]safetensors.Tensor, len(t))
	for k, v := range t {
		out[k] = v
	}
	for k, v := range t {
		if core.HasPrefix(k, pfx) {
			if _, ok := out[k[len(pfx):]]; !ok {
				out[k[len(pfx):]] = v
			}
		}
	}
	return out
}
