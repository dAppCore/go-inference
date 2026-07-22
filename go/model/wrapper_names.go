// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// wrapperStripped returns k's bare-text-model alias for the two multimodal wrapper layouts HF
// packs ship — "language_model.…" (the classic wrapper nesting: language_model.model.norm.weight
// → model.norm.weight) and "model.language_model.…" (the current transformers layout Qwen3.5
// snapshots use: model.language_model.embed_tokens.weight → model.embed_tokens.weight) — or
// ("", false) when k carries neither prefix. The two prefixes are mutually exclusive (one starts
// "language_model.", the other "model."), so the match order cannot mistake one form for the
// other; non-text wrapper siblings (model.visual.…, mtp.…) match neither and pass through.
func wrapperStripped(k string) (string, bool) {
	if rest, ok := core.CutPrefix(k, "language_model."); ok {
		return rest, true
	}
	if rest, ok := core.CutPrefix(k, "model.language_model."); ok {
		return "model." + rest, true
	}
	return "", false
}

// NormalizeWrapperNames makes every wrapper-prefixed tensor (either multimodal wrapper layout —
// see wrapperStripped) ALSO addressable by its stripped "model.…" name, so an assembler's bare
// lookups work whether or not the checkpoint nests the text model under a wrapper.
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
	needsAlias := false
	for k := range t {
		if alias, ok := wrapperStripped(k); ok {
			if _, present := t[alias]; !present {
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
		if alias, ok := wrapperStripped(k); ok {
			if _, present := out[alias]; !present {
				out[alias] = v
			}
		}
	}
	return out
}
