// SPDX-Licence-Identifier: EUPL-1.2

package glmocr

import (
	"os"

	core "dappco.re/go"
)

func ExampleLoadImagePreprocessorConfig() {
	dir := "testdata" // no preprocessor_config.json here — fails predictably, no checkpoint needed
	pc, err := LoadImagePreprocessorConfig(dir)
	core.Println(pc == nil, err != nil)
	// Output: true true
}

// ExampleDecodeAndPatchify patchifies this package's own deterministic 112x112 fixture image
// (testdata/fixture.png — see testdata/gen_fixture.py) using the REAL zai-org/GLM-OCR
// preprocessing geometry (patch_size 14, temporal_patch_size 2, merge_size 2): an 8x8 pre-merge
// patch grid.
func ExampleDecodeAndPatchify() {
	pc := &ImagePreprocessorConfig{
		PatchSize: 14, TemporalPatchSize: 2, MergeSize: 2,
		ImageMean: []float32{0.48145466, 0.4578275, 0.40821073},
		ImageStd:  []float32{0.26862954, 0.26130258, 0.27577711},
	}
	pc.Size.ShortestEdge = 112 * 112
	pc.Size.LongestEdge = 14 * 14 * 2 * 2 * 2 * 6144
	vc := &VisionConfig{InChannels: 3}

	imgBytes, err := os.ReadFile("testdata/fixture.png")
	if err != nil {
		core.Println("read fixture:", err)
		return
	}
	grid, err := DecodeAndPatchify(imgBytes, pc, vc)
	if err != nil {
		core.Println("decode:", err)
		return
	}
	core.Println(grid.GridT, grid.GridH, grid.GridW, grid.PatchDim)
	// Output: 1 8 8 1176
}
