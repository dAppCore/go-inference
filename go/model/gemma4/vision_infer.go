// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/safetensors"
)

// vision_infer.go completes the neutral vision-loader front: decide whether a pack carries a SigLIP
// encoder tower (vs the unified text-only / projector-only variants) and infer the vision config's
// per-model dims from the weight SHAPES (the don't-guess rule) — the vision-side parallel of infer.go.
// Lifted from pkg/metal/model/gemma4/vision_load.go; the device build of the tower from these weights is
// backend-side (pkg/native), the same split as the text path.

// gemma4VisionShouldBuildEncoderTower reports whether a pack carries a full SigLIP encoder tower. The
// unified text / unified-vision variants declare no encoder.
func gemma4VisionShouldBuildEncoderTower(cfg *Gemma4TextConfig) bool {
	if cfg == nil {
		return true
	}
	if cfg.ModelType == "gemma4_unified" || cfg.ModelType == "gemma4_unified_text" {
		return false
	}
	if cfg.VisionConfig != nil && cfg.VisionConfig.ModelType == "gemma4_unified_vision" {
		return false
	}
	return true
}

// inferGemma4VisionConfig fills the vision config's per-model dims from the weight shapes: hidden_size +
// patch_size from the patch-embedding weight, head_dim from hidden/heads, kv-heads default, and the layer
// count by walking encoder.layers.N until a q_proj is absent. Read from the tensors, never guessed.
func inferGemma4VisionConfig(weights map[string]safetensors.Tensor, cfg *Gemma4VisionConfig) *Gemma4VisionConfig {
	if cfg == nil {
		cfg = &Gemma4VisionConfig{}
	}
	if w, ok := model.WeightAny(weights,
		"patch_embedder.input_proj.weight",
		"patch_embedder.input_proj.linear.weight",
		"embeddings.patch_embedding.weight",
		"patch_embedding.weight",
	); ok {
		shape := w.Shape
		if len(shape) > 0 && shape[0] > 0 {
			cfg.HiddenSize = int32(shape[0])
		}
		patchDim := 0
		switch len(shape) {
		case 2:
			patchDim = shape[1]
		case 4:
			patchDim = shape[1] * shape[2] * shape[3]
		}
		channels := int(cfg.NumChannels)
		if channels <= 0 {
			channels = 3
		}
		if patchDim > 0 && patchDim%channels == 0 {
			patch := int(math.Round(math.Sqrt(float64(patchDim / channels))))
			if patch > 0 && channels*patch*patch == patchDim {
				cfg.PatchSize = int32(patch)
			}
		}
	}
	if cfg.HiddenSize > 0 && cfg.NumAttentionHeads > 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	}
	if cfg.NumKeyValueHeads == 0 {
		cfg.NumKeyValueHeads = cfg.NumAttentionHeads
	}
	for i := 0; ; i++ {
		prefix := core.Sprintf("encoder.layers.%d", i)
		if _, ok := model.WeightAny(weights,
			prefix+".self_attn.q_proj.weight",
			prefix+".self_attn.q_proj.linear.weight",
			prefix+".attention.q_proj.weight",
			prefix+".attention.q_proj.linear.weight",
		); !ok {
			if i > 0 {
				cfg.NumHiddenLayers = int32(i)
			}
			break
		}
	}
	return normalizeGemma4VisionConfig(cfg)
}
