// SPDX-Licence-Identifier: EUPL-1.2

package qwen35

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"testing"
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

func TestImageToPatchGrid_Good(t *testing.T) {
	cfg := VisionTowerConfig{PatchSize: 2, InChannels: 3, TemporalPatchSize: 2, MergeSize: 2}
	// 10×6 image, crop unit = patch·merge = 4 → crop to 8×4 → grid 2 rows × 4 cols.
	patches, gridH, gridW, err := ImageToPatchGrid(testPNG(t, 10, 6), cfg)
	if err != nil {
		t.Fatalf("ImageToPatchGrid: %v", err)
	}
	if gridH != 2 || gridW != 4 {
		t.Fatalf("grid = %dx%d, want 2x4 (top-left crop of 10x6 to 8x4 at patch 2)", gridH, gridW)
	}
	patchPixels := 3 * 2 * 2
	if len(patches) != gridH*gridW*patchPixels*2 {
		t.Fatalf("patch buffer len %d, want %d (grid · patchPixels · temporal)", len(patches), gridH*gridW*patchPixels*2)
	}
	// Patch (row 1, col 2) first pixel is image (x=4, y=2): R=4/255, G=2/255, B=7/255 — channel-last.
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
	cfg := VisionTowerConfig{PatchSize: 16, InChannels: 3, TemporalPatchSize: 2, MergeSize: 2}
	if _, _, _, err := ImageToPatchGrid(testPNG(t, 8, 8), cfg); err == nil {
		t.Fatal("an image smaller than one patch·merge block must fail loudly")
	}
}

func TestImageToPatchGrid_NotAnImage_Bad(t *testing.T) {
	cfg := VisionTowerConfig{PatchSize: 2, InChannels: 3, TemporalPatchSize: 1, MergeSize: 1}
	if _, _, _, err := ImageToPatchGrid([]byte("not an image"), cfg); err == nil {
		t.Fatal("undecodable bytes must fail loudly")
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
