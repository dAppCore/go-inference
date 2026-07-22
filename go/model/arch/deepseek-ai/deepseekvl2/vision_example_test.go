// SPDX-Licence-Identifier: EUPL-1.2

package deepseekvl2

import (
	"bytes"
	"image"
	"image/color"
	"image/png"

	core "dappco.re/go"
)

// ExampleDecodeAndNormaliseImage documents the exact [-1,1] normalisation
// (pixel/127.5 - 1) DeepSeek-OCR's BasicImageTransform applies, and the v1 fixed-1024x1024-canvas
// boundary (see vision.go's doc comment) — a fully executable, dependency-free example (no real
// checkpoint needed, unlike Load/Model.OCR's Examples).
func ExampleDecodeAndNormaliseImage() {
	img := image.NewRGBA(image.Rect(0, 0, samImgSize, samImgSize))
	white := color.RGBA{R: 255, G: 255, B: 255, A: 255}
	for y := range samImgSize {
		for x := range samImgSize {
			img.SetRGBA(x, y, white)
		}
	}
	var buf bytes.Buffer
	_ = png.Encode(&buf, img)

	pixels, err := DecodeAndNormaliseImage(buf.Bytes())
	core.Println(err == nil, pixels[0]) // white (255) normalises to 1.0
	// Output: true 1
}
