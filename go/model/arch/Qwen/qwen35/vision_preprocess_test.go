// SPDX-Licence-Identifier: EUPL-1.2

package qwen35

import (
	"bytes"
	"image"
	"os"
	"path/filepath"
	"testing"

	core "dappco.re/go"
)

// vision_preprocess_test.go proves the #59 normalisation follow-up's own pieces:
// qwenSmartResizeTarget's dimension arithmetic, LoadVisionPreprocessConfig's file handling, and
// resizeBicubicRGB's interpolation. Every table row and oracle constant below was cross-checked
// against the REAL reference (transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize /
// Qwen2VLImageProcessor, fed through a local transformers+torch+torchvision install — not a repo
// dependency, so that comparison is reproducible but not re-run by `go test`; see
// docs/design-qwen-vision-factory.md's preprocessing-decision addendum for the session that produced
// these numbers). End-to-end normalisation-through-patchify is vision_test.go's
// TestImageToPatchGrid_Normalise_Good.

func TestQwenSmartResizeTarget_Good(t *testing.T) {
	cases := []struct {
		name                   string
		h, w, factor, min, max int
		wantH, wantW           int
	}{
		// mlx-community/Qwen3.6-27B-4bit's REAL declared bounds (factor 32 = patch16·merge2,
		// min/max 65536/16777216): a tiny image is upscaled to the pixel-count floor.
		{"below-min-upscale", 64, 64, 32, 65536, 16777216, 256, 256},
		{"already-at-bound-noop", 256, 256, 32, 65536, 16777216, 256, 256},
		{"above-max-downscale", 8192, 8192, 32, 65536, 16777216, 4096, 4096},
		// round-half-to-EVEN (Python's builtin round(), which math.RoundToEven matches and
		// math.Round — round-half-away-from-zero — does not): 2/4=0.5 rounds DOWN to the even 0,
		// collapsing the initial estimate to zero; the min_pixels branch then re-derives from the
		// ORIGINAL height/width via beta, not from the collapsed estimate.
		{"round-half-to-even-collapse", 2, 2, 4, 1, 1_000_000_000, 4, 4},
		// unbounded (min=max<=0, VisionPreprocessConfig's zero-value degenerate reading): only the
		// round-to-nearest-multiple step applies, no forced up/downscale — 6/4=1.5 and 10/4=2.5 both
		// round to the nearest EVEN result (2 and 2), landing both axes on 8.
		{"unbounded-round-only", 6, 10, 4, 0, 0, 8, 8},
		{"unbounded-exact-multiple-noop", 4, 4, 2, 0, 0, 4, 4},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			gotH, gotW, err := qwenSmartResizeTarget(c.h, c.w, c.factor, c.min, c.max)
			if err != nil {
				t.Fatalf("qwenSmartResizeTarget: %v", err)
			}
			if gotH != c.wantH || gotW != c.wantW {
				t.Fatalf("qwenSmartResizeTarget(h=%d,w=%d,factor=%d,min=%d,max=%d) = %dx%d, want %dx%d",
					c.h, c.w, c.factor, c.min, c.max, gotH, gotW, c.wantH, c.wantW)
			}
		})
	}
}

func TestQwenSmartResizeTarget_AspectRatio_Bad(t *testing.T) {
	// ratio 300:1 (>200) refuses; ratio exactly 200:1 is the confirmed boundary (accepted, not
	// refused — the reference check is a strict `>`).
	if _, _, err := qwenSmartResizeTarget(1, 300, 2, 1, 1_000_000_000); err == nil {
		t.Fatal("qwenSmartResizeTarget accepted a >200:1 aspect ratio")
	}
	if _, _, err := qwenSmartResizeTarget(1, 200, 2, 1, 1_000_000_000); err != nil {
		t.Fatalf("qwenSmartResizeTarget rejected the exact 200:1 boundary: %v", err)
	}
}

func TestQwenSmartResizeTarget_InvalidInput_Bad(t *testing.T) {
	for _, c := range []struct{ h, w, factor int }{
		{0, 10, 4}, {10, 0, 4}, {10, 10, 0}, {-1, 10, 4},
	} {
		if _, _, err := qwenSmartResizeTarget(c.h, c.w, c.factor, 1, 1_000_000_000); err == nil {
			t.Fatalf("qwenSmartResizeTarget(%d,%d,factor=%d) accepted a non-positive dimension", c.h, c.w, c.factor)
		}
	}
}

// realPreprocessorConfigJSON is mlx-community/Qwen3.6-27B-4bit's ACTUAL shipped
// preprocessor_config.json (the snapshot this port's live receipt loads) — see
// docs/design-qwen-vision-factory.md.
const realPreprocessorConfigJSON = `{
    "size": {"longest_edge": 16777216, "shortest_edge": 65536},
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "processor_class": "Qwen3VLProcessor",
    "image_processor_type": "Qwen2VLImageProcessorFast"
}`

func writeVisionPreprocessTestFile(t *testing.T, dir, name, content string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o600); err != nil {
		t.Fatalf("write %s/%s: %v", dir, name, err)
	}
}

func TestLoadVisionPreprocessConfig_Good(t *testing.T) {
	dir := t.TempDir()
	writeVisionPreprocessTestFile(t, dir, "preprocessor_config.json", realPreprocessorConfigJSON)
	cfg, ok, err := LoadVisionPreprocessConfig(dir)
	if err != nil {
		t.Fatalf("LoadVisionPreprocessConfig: %v", err)
	}
	if !ok {
		t.Fatal("LoadVisionPreprocessConfig ok=false with a present, well-formed file")
	}
	want := VisionPreprocessConfig{
		ImageMean: [3]float32{0.5, 0.5, 0.5}, ImageStd: [3]float32{0.5, 0.5, 0.5},
		MinPixels: 65536, MaxPixels: 16777216,
	}
	if cfg != want {
		t.Fatalf("LoadVisionPreprocessConfig = %+v, want %+v", cfg, want)
	}
}

func TestLoadVisionPreprocessConfig_Missing_Good(t *testing.T) {
	dir := t.TempDir() // no preprocessor_config.json written
	cfg, ok, err := LoadVisionPreprocessConfig(dir)
	if err != nil {
		t.Fatalf("a missing preprocessor_config.json must not be an error: %v", err)
	}
	if ok {
		t.Fatal("ok=true with no preprocessor_config.json present")
	}
	want := defaultVisionPreprocessConfig()
	if cfg != want {
		t.Fatalf("LoadVisionPreprocessConfig (missing file) = %+v, want the HF class defaults %+v", cfg, want)
	}
	if cfg.ImageMean != openAIClipMean || cfg.ImageStd != openAIClipStd {
		t.Fatalf("missing-file fallback mean/std = %v/%v, want the OpenAI CLIP constants %v/%v",
			cfg.ImageMean, cfg.ImageStd, openAIClipMean, openAIClipStd)
	}
}

func TestLoadVisionPreprocessConfig_Malformed_Bad(t *testing.T) {
	dir := t.TempDir()
	writeVisionPreprocessTestFile(t, dir, "preprocessor_config.json", `{not valid json`)
	if _, _, err := LoadVisionPreprocessConfig(dir); err == nil {
		t.Fatal("LoadVisionPreprocessConfig accepted malformed JSON")
	}
}

func TestLoadVisionPreprocessConfig_PartialFields_Good(t *testing.T) {
	dir := t.TempDir()
	// declares only the resize bounds — no image_mean/image_std — so the mean/std fields must fall
	// back to the class default while the declared bounds are honoured.
	writeVisionPreprocessTestFile(t, dir, "preprocessor_config.json",
		`{"size":{"shortest_edge":4096,"longest_edge":2097152}}`)
	cfg, ok, err := LoadVisionPreprocessConfig(dir)
	if err != nil {
		t.Fatalf("LoadVisionPreprocessConfig: %v", err)
	}
	if !ok {
		t.Fatal("ok=false with a present file")
	}
	if cfg.MinPixels != 4096 || cfg.MaxPixels != 2097152 {
		t.Fatalf("declared bounds not honoured: min=%d max=%d, want 4096/2097152", cfg.MinPixels, cfg.MaxPixels)
	}
	if cfg.ImageMean != openAIClipMean || cfg.ImageStd != openAIClipStd {
		t.Fatalf("mean/std with no declared image_mean/image_std = %v/%v, want the class default %v/%v",
			cfg.ImageMean, cfg.ImageStd, openAIClipMean, openAIClipStd)
	}
}

// TestResizeBicubicRGB_Identity_Good confirms a same-size call is the identity transform bar float
// noise: cubicKernel(0)=1 and cubicKernel(±1)=cubicKernel(±2)=0, so at scale=1 every output sample's
// weight vector is an exact one-hot at its own source pixel (the same property GLM-OCR's own resize
// boundary comment leans on for its own bicubic reference: "at scale=1 ... a real bicubic resize is
// the identity transform bar float noise").
func TestResizeBicubicRGB_Identity_Good(t *testing.T) {
	src := []float64{
		0, 0, 0, 255, 255, 255,
		10, 20, 30, 200, 150, 100,
	}
	out := resizeBicubicRGB(src, 2, 2, 2, 2)
	for i := range src {
		if core.Abs(out[i]-src[i]) > 1e-6 {
			t.Fatalf("resizeBicubicRGB same-size[%d] = %v, want %v (identity)", i, out[i], src[i])
		}
	}
}

// TestResizeBicubicRGB_Upscale_Good pins resizeBicubicRGB against the REAL reference at points that
// are genuinely BLENDED (inside the background/foreground edge transition, not a uniform region — so
// this is a receipt on the interpolation maths, not just pass-through) for the exact 64x64->256x256
// upscale engine/metal's live receipt and vision_test.go's TestImageToPatchGrid_Normalise_Good both
// exercise. (128,64) and (128,67) are two of the sample points confirmed to round to the IDENTICAL
// byte value as the real transformers+torchvision oracle (most sample points in this scan matched
// exactly; a minority differ by 1/255 — the file doc comment's "max abs diff 1/255" — these two were
// picked because they do not, so the assertion below can be an exact rounded-byte check).
func TestResizeBicubicRGB_Upscale_Good(t *testing.T) {
	img, _, err := image.Decode(bytes.NewReader(qwenSampleImagePNG(t, 64, 64)))
	if err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	src := decodeImageRGB255(img, img.Bounds())
	out := resizeBicubicRGB(src, 64, 64, 256, 256)

	cases := []struct {
		y, x int
		want [3]float64
	}{
		{0, 0, [3]float64{16, 16, 24}},     // deep background, untouched
		{128, 128, [3]float64{230, 220, 40}}, // deep foreground, untouched
		{70, 70, [3]float64{230, 220, 40}},  // just inside the foreground edge, still exact
		{128, 64, [3]float64{156, 149, 34}}, // genuinely blended — edge transition row
		{128, 67, [3]float64{246, 235, 41}}, // genuinely blended — edge transition row
	}
	for _, c := range cases {
		base := (c.y*256 + c.x) * 3
		for ch := range 3 {
			got := roundClampByte(out[base+ch])
			if core.Abs(got-c.want[ch]) > 1e-6 {
				t.Fatalf("resizeBicubicRGB(64x64->256x256)[y=%d,x=%d,ch=%d] rounded = %v, want %v (oracle: real transformers+torchvision Qwen2VLImageProcessor)",
					c.y, c.x, ch, got, c.want[ch])
			}
		}
	}
}
