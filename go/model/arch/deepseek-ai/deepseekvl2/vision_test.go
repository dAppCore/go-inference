// SPDX-Licence-Identifier: EUPL-1.2

package deepseekvl2

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"testing"
)

// vision_test.go covers DecodeAndNormaliseImage (the PNG/JPEG decode + [-1,1] normalise step) and
// VisionForward's top-level shape contract. Block-level tower correctness lives in
// vision_sam_test.go/vision_clip_test.go (golden-gated); the live E2E gate (live_test.go) proves
// the whole VisionForward pipeline against the real checkpoint.

func encodeTestPNG(t *testing.T, w, h int, fill color.RGBA) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			img.SetRGBA(x, y, fill)
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode test PNG: %v", err)
	}
	return buf.Bytes()
}

// TestDecodeAndNormaliseImage_Good pins the exact normalisation formula (pixel/127.5 - 1 —
// BasicImageTransform's ToTensor()+Normalize(mean=0.5,std=0.5)) against three known input values:
// 255->1.0, 0->-1.0, 128->~0.0039 (128/127.5-1).
func TestDecodeAndNormaliseImage_Good(t *testing.T) {
	data := encodeTestPNG(t, samImgSize, samImgSize, color.RGBA{R: 255, G: 0, B: 128, A: 255})
	pixels, err := DecodeAndNormaliseImage(data)
	if err != nil {
		t.Fatalf("DecodeAndNormaliseImage: %v", err)
	}
	if len(pixels) != samImgSize*samImgSize*3 {
		t.Fatalf("pixel buffer len = %d, want %d", len(pixels), samImgSize*samImgSize*3)
	}
	r, g, b := pixels[0], pixels[1], pixels[2]
	if diff := absF32(r - 1.0); diff > 1e-3 {
		t.Fatalf("R (255) normalised = %g, want ~1.0", r)
	}
	if diff := absF32(g - (-1.0)); diff > 1e-3 {
		t.Fatalf("G (0) normalised = %g, want ~-1.0", g)
	}
	wantB := float32(128)/127.5 - 1
	if diff := absF32(b - wantB); diff > 1e-3 {
		t.Fatalf("B (128) normalised = %g, want ~%g", b, wantB)
	}
}

// TestDecodeAndNormaliseImage_Bad proves an image whose size is not exactly 1024x1024 is refused
// by name (the v1 "Base" resolution mode boundary — vision.go's doc comment), not silently
// resized.
func TestDecodeAndNormaliseImage_Bad(t *testing.T) {
	data := encodeTestPNG(t, 512, 512, color.RGBA{R: 1, G: 1, B: 1, A: 255})
	if _, err := DecodeAndNormaliseImage(data); err == nil {
		t.Fatal("DecodeAndNormaliseImage accepted a 512x512 image, want a named refusal (only exactly 1024x1024 is implemented)")
	}
}

// TestDecodeAndNormaliseImage_Ugly proves non-image bytes fail at decode, not deeper with a
// confusing shape error.
func TestDecodeAndNormaliseImage_Ugly(t *testing.T) {
	if _, err := DecodeAndNormaliseImage([]byte("not an image")); err == nil {
		t.Fatal("DecodeAndNormaliseImage accepted non-image bytes")
	}
}

// TestVisionForward_Projector_Golden_Good pins the projector's Linear(2048,1280)-shaped
// projection (toy input_dim=16/n_embed=10 here) against the real deepencoder.MlpProjector class —
// VisionForward's own concat+project step, isolated from the two towers it sits between (which
// vision_sam_test.go/vision_clip_test.go gate separately).
func TestVisionForward_Projector_Golden_Good(t *testing.T) {
	g := readVisionToyGolden(t).Projector
	got := linear(g.Input, g.Weight, g.InputDim, g.NEmbed, g.Bias)
	if d := maxAbsDiff32(t, got, g.Output); d > 1e-4 {
		t.Fatalf("projector output max abs diff %g, want <=1e-4", d)
	}
}

// TestVisionForward_Bad proves a pixel buffer that fails SAMForward's own shape guard propagates
// as a clean VisionForward error (never a panic) — the same guard TestSAMForward_Bad pins
// directly, exercised here through the top-level entry point.
func TestVisionForward_Bad(t *testing.T) {
	_, err := VisionForward(make([]float32, 10), &Weights{})
	if err == nil {
		t.Fatal("VisionForward accepted a pixel buffer far shorter than 1024x1024x3")
	}
}
