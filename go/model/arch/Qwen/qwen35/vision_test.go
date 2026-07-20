// SPDX-Licence-Identifier: EUPL-1.2

package qwen35

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"testing"

	core "dappco.re/go"
)

// vision_test.go proves the host preprocessing that ships with the payload: ImageToPatchGrid's
// crop/patchify/temporal-repeat semantics and InterpolatePosEmbed's reference-convention bilinear
// resample. The tower forward itself runs in engine/metal (qwen_vision_encoder_test.go).

// testPNG encodes a w×h RGBA image whose pixel (x,y) has R = x, G = y, B = 7 — position-decodable
// values for asserting crop and patch layout.
func testPNG(t *testing.T, w, h int) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			img.Set(x, y, color.RGBA{R: uint8(x), G: uint8(y), B: 7, A: 255})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode png: %v", err)
	}
	return buf.Bytes()
}

// noopPreprocess disables both the smart-resize pixel bounds (so an already patch·merge-aligned
// image passes through with NO resample — pure crop-era behaviour) and normalisation (mean 0, std 1
// via normalized() — plain /255 rescale), isolating the patch/temporal LAYOUT this test targets from
// the resize/normalise concerns vision_preprocess_test.go and TestImageToPatchGrid_Normalise_Good own.
var noopPreprocess = VisionPreprocessConfig{}

func TestImageToPatchGrid_Good(t *testing.T) {
	cfg := VisionTowerConfig{PatchSize: 2, InChannels: 3, TemporalPatchSize: 2, MergeSize: 2}
	// 8×4 image — already an exact patch·merge (unit=4) multiple on both axes, so with resize bounds
	// disabled (noopPreprocess) smart_resize's round-to-nearest-multiple step is a no-op and the
	// grid is a pure crop-free reshape: grid 2 rows × 4 cols, source pixels unchanged.
	patches, gridH, gridW, err := ImageToPatchGrid(testPNG(t, 8, 4), cfg, noopPreprocess)
	if err != nil {
		t.Fatalf("ImageToPatchGrid: %v", err)
	}
	if gridH != 2 || gridW != 4 {
		t.Fatalf("grid = %dx%d, want 2x4 (8x4 image, already patch·merge aligned)", gridH, gridW)
	}
	patchPixels := 3 * 2 * 2
	if len(patches) != gridH*gridW*patchPixels*2 {
		t.Fatalf("patch buffer len %d, want %d (grid · patchPixels · temporal)", len(patches), gridH*gridW*patchPixels*2)
	}
	// Patch (row 1, col 2) first pixel is image (x=4, y=2): R=4/255, G=2/255, B=7/255 — channel-last,
	// no resample (aligned input) and no normalisation (noopPreprocess) to blur the source values.
	base := (1*gridW + 2) * patchPixels * 2
	if got, want := patches[base], float32(4)/255; got != want {
		t.Fatalf("patch(1,2) R = %v, want %v", got, want)
	}
	if got, want := patches[base+1], float32(2)/255; got != want {
		t.Fatalf("patch(1,2) G = %v, want %v", got, want)
	}
	if got, want := patches[base+2], float32(7)/255; got != want {
		t.Fatalf("patch(1,2) B = %v, want %v", got, want)
	}
	// The temporal repeat: frame 1 duplicates frame 0 exactly.
	for i := range patchPixels {
		if patches[base+i] != patches[base+patchPixels+i] {
			t.Fatalf("temporal frame 1 diverges from frame 0 at %d — a still image must repeat identically", i)
		}
	}
}

func TestImageToPatchGrid_TooSmall_Bad(t *testing.T) {
	// PatchSize=16, MergeSize=2 → unit=32; an 8x8 image at unbounded resize (noopPreprocess) rounds
	// DOWN to a 0x0 target (round-half-to-even(8/32)=round(0.25)=0) — smart_resize's floor, not a
	// pre-#59 crop, but the same degenerate "too small" case still fails loudly.
	cfg := VisionTowerConfig{PatchSize: 16, InChannels: 3, TemporalPatchSize: 2, MergeSize: 2}
	if _, _, _, err := ImageToPatchGrid(testPNG(t, 8, 8), cfg, noopPreprocess); err == nil {
		t.Fatal("an image resolving to a 0x0 resize target must fail loudly")
	}
}

func TestImageToPatchGrid_NotAnImage_Bad(t *testing.T) {
	cfg := VisionTowerConfig{PatchSize: 2, InChannels: 3, TemporalPatchSize: 1, MergeSize: 1}
	if _, _, _, err := ImageToPatchGrid([]byte("not an image"), cfg, noopPreprocess); err == nil {
		t.Fatal("undecodable bytes must fail loudly")
	}
}

// qwenSampleImagePNG encodes the SAME visual fixture engine/metal's live-receipt test uses
// (qwenVisionTestPNG): a dark-navy background with a bright-yellow square over the centre half of
// each axis — chosen so specific patch positions land ENTIRELY inside one colour or the other (no
// resample blending at the sample point), letting normalisation be checked against hand-computed
// values even once smart_resize's bicubic resample is in the pipeline.
func qwenSampleImagePNG(t *testing.T, w, h int) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			c := color.RGBA{R: 16, G: 16, B: 24, A: 255}
			if x >= w/4 && x < 3*w/4 && y >= h/4 && y < 3*h/4 {
				c = color.RGBA{R: 230, G: 220, B: 40, A: 255}
			}
			img.Set(x, y, c)
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode png: %v", err)
	}
	return buf.Bytes()
}

// TestImageToPatchGrid_Normalise_Good pins ImageToPatchGrid's normalisation + smart-resize output
// against HAND-COMPUTED values for the real mlx-community/Qwen3.6-27B-4bit checkpoint's declared
// geometry (patch 16, merge 2) and preprocessor_config.json policy (mean/std 0.5/0.5/0.5 per channel,
// min/max pixels 65536/16777216) — the exact scenario engine/metal's
// TestQwenVisionImageTurn_RealCheckpoint live receipt exercises at 64x64. Independently confirmed
// against the REAL transformers Qwen2VLImageProcessor (a local install, not a repo dependency): a
// 64x64 image under these bounds smart-resizes to 256x256 (image_grid_thw [1,16,16] in the oracle
// run), and the two sample pixels below are deep inside uniform regions (no resample blending), so
// the oracle's resized pixel values equal the SOURCE values exactly — the normalise formula is then
// the only unverified step, and it is hand-computable.
func TestImageToPatchGrid_Normalise_Good(t *testing.T) {
	cfg := VisionTowerConfig{PatchSize: 16, InChannels: 3, TemporalPatchSize: 2, MergeSize: 2}
	pp := VisionPreprocessConfig{
		ImageMean: [3]float32{0.5, 0.5, 0.5}, ImageStd: [3]float32{0.5, 0.5, 0.5},
		MinPixels: 65536, MaxPixels: 16777216,
	}
	patches, gridH, gridW, err := ImageToPatchGrid(qwenSampleImagePNG(t, 64, 64), cfg, pp)
	if err != nil {
		t.Fatalf("ImageToPatchGrid: %v", err)
	}
	if gridH != 16 || gridW != 16 {
		t.Fatalf("grid = %dx%d, want 16x16 (64x64 upscaled to the 65536px floor = 256x256, /16 patch)", gridH, gridW)
	}
	patchPixels := 3 * 16 * 16

	// Patch (0,0): resized pixels [0,16)x[0,16) — entirely background (16,16,24), well outside the
	// square's resized bounds [64,192)x[64,192). want = (v/255-0.5)/0.5.
	bg := [3]float32{16, 16, 24}
	base00 := 0
	for c := range 3 {
		want := (bg[c]/255 - 0.5) / 0.5
		if got := patches[base00+c]; core.Abs(got-want) > 1e-6 {
			t.Fatalf("patch(0,0) channel %d = %v, want %v ((%v/255-0.5)/0.5, background)", c, got, want, bg[c])
		}
	}

	// Patch (8,8): resized pixels [128,144)x[128,144) — entirely the yellow square (230,220,40),
	// well inside its resized bounds. want = (v/255-0.5)/0.5.
	fg := [3]float32{230, 220, 40}
	base88 := (8*gridW + 8) * patchPixels * 2
	for c := range 3 {
		want := (fg[c]/255 - 0.5) / 0.5
		if got := patches[base88+c]; core.Abs(got-want) > 1e-6 {
			t.Fatalf("patch(8,8) channel %d = %v, want %v ((%v/255-0.5)/0.5, foreground)", c, got, want, fg[c])
		}
	}
}

// TestImageToPatchGrid_DefaultPreprocess_Good confirms an unset VisionPreprocessConfig{} (a caller
// with no resize/normalise policy to declare) degenerates to the pre-#59 behaviour rather than
// inventing one: an already patch·merge-aligned image passes through unresized, and normalised()'s
// std-1 fallback makes the "normalisation" a plain /255 rescale (mean 0).
func TestImageToPatchGrid_DefaultPreprocess_Good(t *testing.T) {
	cfg := VisionTowerConfig{PatchSize: 2, InChannels: 3, TemporalPatchSize: 1, MergeSize: 1}
	patches, gridH, gridW, err := ImageToPatchGrid(testPNG(t, 4, 4), cfg, VisionPreprocessConfig{})
	if err != nil {
		t.Fatalf("ImageToPatchGrid: %v", err)
	}
	if gridH != 2 || gridW != 2 {
		t.Fatalf("grid = %dx%d, want 2x2 (4x4 image, patch 2, unbounded resize is a no-op)", gridH, gridW)
	}
	// Patch(0,0) first pixel is image (0,0): R=0,G=0,B=7 -> plain /255, no mean/std shift.
	if got, want := patches[2], float32(7)/255; got != want {
		t.Fatalf("patch(0,0) B = %v, want %v (plain /255, zero-value preprocess config)", got, want)
	}
}

func TestInterpolatePosEmbed_Good(t *testing.T) {
	// A 2×2 table with hidden 1: corners 0, 1, 2, 3. Resampled onto 3×3 with linspace(0,1,3) per
	// axis, the centre lands exactly between all four corners: (0+1+2+3)/4 = 1.5; edges are the
	// two-corner midpoints.
	table := []float32{0, 1, 2, 3}
	out, err := InterpolatePosEmbed(table, 1, 3, 3)
	if err != nil {
		t.Fatalf("InterpolatePosEmbed: %v", err)
	}
	want := []float32{0, 0.5, 1, 1, 1.5, 2, 2, 2.5, 3}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("resampled[%d] = %v, want %v (reference align-corners bilinear)", i, out[i], want[i])
		}
	}
	// A grid-exact request reproduces the table verbatim.
	same, err := InterpolatePosEmbed(table, 1, 2, 2)
	if err != nil {
		t.Fatalf("InterpolatePosEmbed (exact grid): %v", err)
	}
	for i := range table {
		if same[i] != table[i] {
			t.Fatalf("grid-exact resample[%d] = %v, want the table value %v", i, same[i], table[i])
		}
	}
}

func TestInterpolatePosEmbed_NotSquare_Bad(t *testing.T) {
	if _, err := InterpolatePosEmbed(make([]float32, 3), 1, 2, 2); err == nil {
		t.Fatal("a non-square position table must fail loudly")
	}
}
