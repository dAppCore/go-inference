// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/arch/Qwen/qwen35"
	"dappco.re/go/inference/model/safetensors"
)

// qwen_vision.go is the serve glue between the qwen35 factory vision tower (payload assembled by
// model/arch/Qwen/qwen35, forward in qwen_vision_encoder.go) and the NativeTokenModel's
// engine.VisionTokenModel surface. The qwen35 hybrids serve as *NativeTokenModel (the #18 factory
// route), so the whole multimodal chain past ProjectImage — the generic placeholder splice
// (TokenEmbeddingsWithFeatures), ArchSession.PrefillTokenEmbeddings, chatMultimodal's placeholder-run
// verification — is the SAME machinery gemma4 vision rides, untouched. This file supplies only the
// qwen-specific pieces: attaching the tower at load and projecting one image through it.

// loadQwenVisionTower probes dir for a qwen-family vision tower and assembles it from the checkpoint's
// mapped tensors. Returns (nil, nil) for a non-qwen arch or a text-only qwen checkpoint (no
// vision_tower.* tensors — every text-only hybrid keeps loading exactly as before); a PRESENT but
// malformed tower fails loudly rather than silently serving text-only (the load.go caller propagates,
// so a corrupt multimodal checkpoint is a named load error, not a surprise refusal at the first image
// turn). The returned tower owns f32 copies of every weight (qwen35.LoadVisionTower widens/dequantises
// at assembly), so it has no lifetime tie to the shard mmap.
func loadQwenVisionTower(dir string, dm *safetensors.DirMapping) (*qwen35.VisionTower, error) {
	modelType, cfgJSON, err := model.ProbeDirArch(dir)
	if err != nil || !qwen35.HybridModelType(modelType) {
		return nil, nil // not this family's checkpoint — never an error
	}
	tower, terr := qwen35.LoadVisionTower(dm.Tensors, cfgJSON)
	if terr != nil {
		return nil, core.E("native.loadQwenVisionTower", "assemble qwen vision tower", terr)
	}
	return tower, nil
}

// projectQwenImage preprocesses one raw PNG/JPEG image through the qwen patchifier (top-left crop to
// a patch·merge-aligned grid — the tower's own policy, NOT the gemma feature config's resize/budget)
// and runs the tower forward, returning the soft-token feature rows as bf16 bytes ready for
// TokenEmbeddingsWithFeatures and the soft-token count (the placeholder run length for this image).
func projectQwenImage(tower *qwen35.VisionTower, image []byte) ([]byte, int, error) {
	patches, gridH, gridW, err := qwen35.ImageToPatchGrid(image, tower.Cfg)
	if err != nil {
		return nil, 0, core.E("native.projectQwenImage", "patchify", err)
	}
	features, softTokens, err := QwenVisionTowerForward(patches, gridH, gridW, tower)
	if err != nil {
		return nil, 0, core.E("native.projectQwenImage", "tower forward", err)
	}
	return f32ToBf16Slice(features), softTokens, nil
}
