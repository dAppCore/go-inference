// SPDX-Licence-Identifier: EUPL-1.2

package glmocr

import (
	"os"
	"path/filepath"
	"testing"
)

func realImagePreprocessorConfig() *ImagePreprocessorConfig {
	pc := &ImagePreprocessorConfig{
		PatchSize: 14, TemporalPatchSize: 2, MergeSize: 2,
		ImageMean: []float32{0.48145466, 0.4578275, 0.40821073},
		ImageStd:  []float32{0.26862954, 0.26130258, 0.27577711},
	}
	pc.Size.ShortestEdge = 112 * 112
	pc.Size.LongestEdge = 14 * 14 * 2 * 2 * 2 * 6144
	return pc
}

func TestSmartResizeTarget_Good(t *testing.T) {
	// 112x112 is this package's own fixture.png size, chosen BECAUSE it is smart_resize-stable
	// (112 = 4×28, and its pixel count already sits inside [min,max]) — confirmed empirically
	// against the REAL Glm46VImageProcessor (max abs diff 2.4e-7 between its output and a plain
	// rescale/normalise of the untouched pixels — see image.go's file doc comment).
	hBar, wBar, err := smartResizeTarget(112, 112, 28, 112*112, 14*14*2*2*2*6144, 2)
	if err != nil {
		t.Fatalf("smartResizeTarget: %v", err)
	}
	if hBar != 112 || wBar != 112 {
		t.Fatalf("smartResizeTarget(112,112) = (%d,%d), want (112,112) — self-mapping", hBar, wBar)
	}
}

func TestSmartResizeTarget_Bad(t *testing.T) {
	// a tiny image (below min_pixels) must be scaled UP to at least the minimum, in multiples
	// of factor
	// Cross-checked against the REAL smart_resize (image_processing_glm46v.py) for several
	// height/width pairs spanning the grow (<min_pixels), shrink (>max_pixels), and untouched
	// branches — pins the exact arithmetic, not just "some multiple of 28".
	cases := []struct{ h, w, wantH, wantW int }{
		{10, 10, 84, 84},         // grow branch (was the fixture that caught #37's (h*beta*factor) parenthesis bug)
		{112, 112, 112, 112},     // already stable
		{50, 200, 56, 196},       // grow branch, non-square
		{1000, 1000, 1008, 1008}, // shrink branch would trigger far higher; this stays in "round only"
		{30, 29, 84, 84},         // grow branch
		{300, 50, 308, 56},       // grow branch, non-square
	}
	for _, c := range cases {
		hBar, wBar, err := smartResizeTarget(c.h, c.w, 28, 112*112, 14*14*2*2*2*6144, 2)
		if err != nil {
			t.Fatalf("smartResizeTarget(%d,%d): %v", c.h, c.w, err)
		}
		if hBar != c.wantH || wBar != c.wantW {
			t.Fatalf("smartResizeTarget(%d,%d) = (%d,%d), want (%d,%d)", c.h, c.w, hBar, wBar, c.wantH, c.wantW)
		}
	}
}

func TestSmartResizeTarget_Ugly(t *testing.T) {
	// an absurd aspect ratio (>200:1) refuses rather than producing a degenerate 0-width grid
	if _, _, err := smartResizeTarget(10000, 10, 28, 112*112, 14*14*2*2*2*6144, 2); err == nil {
		t.Fatal("smartResizeTarget accepted a >200:1 aspect ratio")
	}
}

func TestLoadImagePreprocessorConfig_Good(t *testing.T) {
	dir := t.TempDir()
	writeTestFile(t, dir, "preprocessor_config.json", `{"size":{"shortest_edge":12544,"longest_edge":9633792},"patch_size":14,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.48145466,0.4578275,0.40821073],"image_std":[0.26862954,0.26130258,0.27577711]}`)
	pc, err := LoadImagePreprocessorConfig(dir)
	if err != nil {
		t.Fatalf("LoadImagePreprocessorConfig: %v", err)
	}
	if pc.Size.ShortestEdge != 12544 || pc.Size.LongestEdge != 9633792 || pc.PatchSize != 14 || pc.MergeSize != 2 {
		t.Fatalf("LoadImagePreprocessorConfig parsed = %+v", pc)
	}
}

func TestLoadImagePreprocessorConfig_Bad(t *testing.T) {
	dir := t.TempDir()
	if _, err := LoadImagePreprocessorConfig(dir); err == nil {
		t.Fatal("LoadImagePreprocessorConfig accepted a directory with no preprocessor_config.json")
	}
}

func TestLoadImagePreprocessorConfig_Ugly(t *testing.T) {
	dir := t.TempDir()
	// well-formed JSON, but missing every field this package needs
	writeTestFile(t, dir, "preprocessor_config.json", `{}`)
	if _, err := LoadImagePreprocessorConfig(dir); err == nil {
		t.Fatal("LoadImagePreprocessorConfig accepted an empty document")
	}
}

func writeTestFile(t *testing.T, dir, name, content string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o600); err != nil {
		t.Fatalf("write %s/%s: %v", dir, name, err)
	}
}

func TestDecodeAndPatchify_Good(t *testing.T) {
	imgBytes := readTestdata(t, "fixture.png")
	pc := realImagePreprocessorConfig()
	vc := &VisionConfig{InChannels: 3}
	got, err := DecodeAndPatchify(imgBytes, pc, vc)
	if err != nil {
		t.Fatalf("DecodeAndPatchify: %v", err)
	}
	if got.GridT != 1 || got.GridH != 8 || got.GridW != 8 {
		t.Fatalf("DecodeAndPatchify grid = (%d,%d,%d), want (1,8,8)", got.GridT, got.GridH, got.GridW)
	}
	if got.PatchDim != 3*2*14*14 {
		t.Fatalf("DecodeAndPatchify PatchDim = %d, want %d", got.PatchDim, 3*2*14*14)
	}
	// Spot-check patch 37 (the highest-variance patch — real text pixels, not flat background)
	// against values captured from the REAL Glm46VImageProcessor on this exact fixture.png.
	p := got.Patches[37*got.PatchDim : 38*got.PatchDim]
	wantR := []float32{-0.872562, -0.872562, -0.872562, 1.098226, 1.930336}
	wantG := []float32{-0.806608, -0.806608, -0.806608, 1.219441, 2.074883}
	wantB := []float32{-0.584356, -0.584356, -0.584356, 1.335353, 2.145897}
	if d := maxAbsDiff32(t, p[0:5], wantR); d > 2e-5 {
		t.Fatalf("DecodeAndPatchify patch37 R channel maxAbsDiff = %v, want < 2e-5", d)
	}
	if d := maxAbsDiff32(t, p[392:397], wantG); d > 2e-5 {
		t.Fatalf("DecodeAndPatchify patch37 G channel maxAbsDiff = %v, want < 2e-5", d)
	}
	if d := maxAbsDiff32(t, p[784:789], wantB); d > 2e-5 {
		t.Fatalf("DecodeAndPatchify patch37 B channel maxAbsDiff = %v, want < 2e-5", d)
	}
	// Both temporal copies of a static image are identical (see the file doc comment).
	if d := maxAbsDiff32(t, p[0:196], p[196:392]); d > 0 {
		t.Fatalf("DecodeAndPatchify's two temporal copies diverged: maxAbsDiff = %v", d)
	}
}

func TestDecodeAndPatchify_Bad(t *testing.T) {
	imgBytes := readTestdata(t, "fixture.png") // 112x112, NOT a multiple of 28*3=84 nor otherwise re-sizeable to itself at a different factor
	pc := realImagePreprocessorConfig()
	pc.PatchSize = 15 // factor = 15*2 = 30; 112 is not a multiple of 30 -> must refuse, not silently resample
	vc := &VisionConfig{InChannels: 3}
	_, err := DecodeAndPatchify(imgBytes, pc, vc)
	if err == nil {
		t.Fatal("DecodeAndPatchify accepted an image whose size is not a smart_resize-stable target")
	}
}

func TestDecodeAndPatchify_Ugly(t *testing.T) {
	imgBytes := readTestdata(t, "fixture.png")
	pc := realImagePreprocessorConfig()
	vc := &VisionConfig{InChannels: 4} // unsupported channel count
	if _, err := DecodeAndPatchify(imgBytes, pc, vc); err == nil {
		t.Fatal("DecodeAndPatchify accepted in_channels=4")
	}
}
