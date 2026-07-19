// SPDX-Licence-Identifier: EUPL-1.2

package glmocr

import core "dappco.re/go"

// ExampleVisionForward runs the vision tower over a tiny synthetic checkpoint's geometry (the
// same toy dimensions weights_test.go's syntheticCheckpoint builds) and a deterministic patch
// grid — 16 patches (a 4x4 pre-merge grid at spatial_merge_size 2) collapsing to 4 merged
// image-token embeddings.
func ExampleVisionForward() {
	tensors, cfg := syntheticCheckpoint()
	w, err := LoadWeights(tensors, cfg)
	if err != nil {
		core.Println("load:", err)
		return
	}
	patchDim := cfg.VisionConfig.InChannels * cfg.VisionConfig.TemporalPatchSize * cfg.VisionConfig.PatchSize * cfg.VisionConfig.PatchSize
	patches := make([]float32, 16*patchDim)
	for i := range patches {
		patches[i] = float32(i%7) * 0.1
	}
	grid := &PatchGrid{Patches: patches, GridT: 1, GridH: 4, GridW: 4, PatchDim: patchDim}

	embeds, numMerged, err := VisionForward(grid, &w.Vision, cfg.VisionConfig)
	if err != nil {
		core.Println("vision forward:", err)
		return
	}
	core.Println(numMerged, len(embeds) == numMerged*cfg.VisionConfig.OutHiddenSize)
	// Output: 4 true
}
