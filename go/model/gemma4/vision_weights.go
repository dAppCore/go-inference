// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/safetensors"
	"dappco.re/go/inference/profile"
)

// vision_weights.go is the neutral front of the gemma4 vision-tower loader: it canonicalises the vision
// weight names off a checkpoint's tensors (strip the multimodal wrapper prefixes, then the
// vision_tower./vision_model. tower prefix) and detects whether a pack carries a full SigLIP tower or
// only the projector. The DEVICE build of the tower from these weights lives backend-side (pkg/native),
// the same split as the text path (neutral assemble → native device-build). Lifted from
// pkg/metal/model/gemma4/vision_load.go, reusing profile.TrimWeightWrapperPrefix + model.WeightAny
// rather than re-porting them.

// canonicalGemma4VisionWeightName strips the wrapper prefixes then the vision-tower prefix, returning the
// canonical vision-weight name and whether the tensor is a vision weight at all. multi_modal_projector.
// and embed_vision. weights keep their prefix — the projector reads them by full name.
func canonicalGemma4VisionWeightName(name string) (string, bool) {
	trimmed := name
	for {
		next, changed := profile.TrimWeightWrapperPrefix("gemma4", trimmed)
		if !changed {
			break
		}
		trimmed = next
	}
	for _, prefix := range []string{"vision_tower.", "vision_model."} {
		if core.HasPrefix(trimmed, prefix) {
			return core.TrimPrefix(trimmed, prefix), true
		}
	}
	for _, prefix := range []string{"multi_modal_projector.", "embed_vision."} {
		if core.HasPrefix(trimmed, prefix) {
			return trimmed, true
		}
	}
	return "", false
}

// SanitizeVisionWeights returns the vision weights from a checkpoint's tensors, keyed by their canonical
// names — the input the device-side vision build consumes.
func SanitizeVisionWeights(raw map[string]safetensors.Tensor) map[string]safetensors.Tensor {
	vision := make(map[string]safetensors.Tensor)
	for name, t := range raw {
		if canonical, ok := canonicalGemma4VisionWeightName(name); ok {
			vision[canonical] = t
		}
	}
	return vision
}

// HasVisionTowerWeights reports whether the pack carries a full SigLIP vision tower (a patch embedder),
// vs only the multimodal projector.
func HasVisionTowerWeights(weights map[string]safetensors.Tensor) bool {
	_, ok := model.WeightAny(weights,
		"patch_embedder.input_proj.weight",
		"patch_embedder.input_proj.linear.weight",
		"embeddings.patch_embedding.weight",
		"patch_embedding.weight",
	)
	return ok
}

// HasVisionProjectionWeights reports whether the pack carries the vision-to-text multimodal projector.
func HasVisionProjectionWeights(weights map[string]safetensors.Tensor) bool {
	_, ok := model.WeightAny(weights,
		"embed_vision.embedding_projection.weight",
		"embed_vision.embedding_projection.linear.weight",
		"multi_modal_projector.embedding_projection.weight",
		"multi_modal_projector.embedding_projection.linear.weight",
		"multi_modal_projector.proj.weight",
		"multi_modal_projector.weight",
	)
	return ok
}
