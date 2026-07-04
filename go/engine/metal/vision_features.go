// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"
	"math"

	core "dappco.re/go"
)

// VisionImageFeatureConfig mirrors the Gemma 4 image_processor section needed
// to turn encoded image bytes into pre-patchified native vision rows.
type VisionImageFeatureConfig struct {
	PatchSize         int32
	MaxSoftTokens     int32
	PoolingKernelSize int32
	RescaleFactor     float64
	DoResize          bool
	DoConvertRGB      bool
}

func normalizeVisionImageFeatureConfig(cfg *VisionImageFeatureConfig) *VisionImageFeatureConfig {
	if cfg == nil {
		return nil
	}
	out := *cfg
	if out.PatchSize <= 0 {
		out.PatchSize = 16
	}
	if out.MaxSoftTokens <= 0 {
		out.MaxSoftTokens = 280
	}
	if out.PoolingKernelSize <= 0 {
		out.PoolingKernelSize = 3
	}
	if out.RescaleFactor <= 0 {
		out.RescaleFactor = 1.0 / 255.0
	}
	return &out
}

// VisionImagePatches decodes PNG/JPEG bytes, applies the Gemma 4 image sizing
// rule, rescales to [0,1], and returns pre-patchified BF16 rows
// [numPatches, patchSize*patchSize*3] for VisionTower.
func VisionImagePatches(data []byte, cfg *VisionImageFeatureConfig) ([]byte, int, error) {
	pixels, h, w, softTokens, err := VisionImagePixels(data, cfg)
	if err != nil {
		return nil, 0, err
	}
	return patchifyVisionPixelsBF16(pixels, h, w, normalizeVisionImageFeatureConfig(cfg).PatchSize), softTokens, nil
}

// VisionImagePixels decodes PNG/JPEG bytes, applies the Gemma 4 image sizing
// rule, and returns raw NHWC float32 pixels in [0,1] plus the soft-token count.
// This is the native sibling of metal's Gemma4ImagePixels.
func VisionImagePixels(data []byte, cfg *VisionImageFeatureConfig) ([]float32, int32, int32, int, error) {
	cfg = normalizeVisionImageFeatureConfig(cfg)
	if cfg == nil {
		return nil, 0, 0, 0, core.NewError("native.VisionImagePixels: image feature config is nil")
	}
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, 0, 0, 0, core.E("native.VisionImagePixels", "decode image", err)
	}
	bounds := img.Bounds()
	h, w := int32(bounds.Dy()), int32(bounds.Dx())
	if h <= 0 || w <= 0 {
		return nil, 0, 0, 0, core.NewError("native.VisionImagePixels: image has empty bounds")
	}

	src := visionImageRGBFloat64(img, bounds)

	maxPatches := cfg.MaxSoftTokens * cfg.PoolingKernelSize * cfg.PoolingKernelSize
	th, tw := h, w
	if cfg.DoResize || th%(cfg.PatchSize*cfg.PoolingKernelSize) != 0 || tw%(cfg.PatchSize*cfg.PoolingKernelSize) != 0 {
		th, tw, err = visionAspectPreservingSize(h, w, cfg.PatchSize, maxPatches, cfg.PoolingKernelSize)
		if err != nil {
			return nil, 0, 0, 0, err
		}
	}
	resized := src
	if th != h || tw != w {
		resized = visionResizeBicubicAA(src, h, w, th, tw)
	}

	pixels := make([]float32, len(resized))
	for i, v := range resized {
		u := math.RoundToEven(v)
		if u < 0 {
			u = 0
		} else if u > 255 {
			u = 255
		}
		pixels[i] = float32(u * cfg.RescaleFactor)
	}

	grid := (th / cfg.PatchSize) * (tw / cfg.PatchSize)
	softTokens := int(grid / (cfg.PoolingKernelSize * cfg.PoolingKernelSize))
	return pixels, th, tw, softTokens, nil
}

func visionImageRGBFloat64(img image.Image, bounds image.Rectangle) []float64 {
	switch src := img.(type) {
	case *image.NRGBA:
		return visionNRGBAToRGBFloat64(src, bounds)
	case *image.RGBA:
		return visionRGBAToRGBFloat64(src, bounds)
	case *image.YCbCr:
		return visionYCbCrToRGBFloat64(src, bounds)
	default:
		return visionGenericRGBFloat64(img, bounds)
	}
}

func visionNRGBAToRGBFloat64(img *image.NRGBA, bounds image.Rectangle) []float64 {
	h, w := bounds.Dy(), bounds.Dx()
	out := make([]float64, h*w*3)
	dst := 0
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		src := img.PixOffset(bounds.Min.X, y)
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			a := img.Pix[src+3]
			if a == 0xff {
				out[dst] = float64(img.Pix[src])
				out[dst+1] = float64(img.Pix[src+1])
				out[dst+2] = float64(img.Pix[src+2])
			} else {
				out[dst] = float64(visionNRGBAPremul8(img.Pix[src], a))
				out[dst+1] = float64(visionNRGBAPremul8(img.Pix[src+1], a))
				out[dst+2] = float64(visionNRGBAPremul8(img.Pix[src+2], a))
			}
			dst += 3
			src += 4
		}
	}
	return out
}

func visionNRGBAPremul8(v, a byte) byte {
	x := uint32(v)
	x |= x << 8
	x *= uint32(a)
	x /= 0xff
	return byte(x >> 8)
}

func visionRGBAToRGBFloat64(img *image.RGBA, bounds image.Rectangle) []float64 {
	h, w := bounds.Dy(), bounds.Dx()
	out := make([]float64, h*w*3)
	dst := 0
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		src := img.PixOffset(bounds.Min.X, y)
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			out[dst] = float64(img.Pix[src])
			out[dst+1] = float64(img.Pix[src+1])
			out[dst+2] = float64(img.Pix[src+2])
			dst += 3
			src += 4
		}
	}
	return out
}

func visionYCbCrToRGBFloat64(img *image.YCbCr, bounds image.Rectangle) []float64 {
	h, w := bounds.Dy(), bounds.Dx()
	out := make([]float64, h*w*3)
	dst := 0
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			yi := img.YOffset(x, y)
			ci := img.COffset(x, y)
			r, g, b := color.YCbCrToRGB(img.Y[yi], img.Cb[ci], img.Cr[ci])
			out[dst] = float64(r)
			out[dst+1] = float64(g)
			out[dst+2] = float64(b)
			dst += 3
		}
	}
	return out
}

func visionGenericRGBFloat64(img image.Image, bounds image.Rectangle) []float64 {
	h, w := bounds.Dy(), bounds.Dx()
	out := make([]float64, h*w*3)
	idx := 0
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			out[idx] = float64(r >> 8)
			out[idx+1] = float64(g >> 8)
			out[idx+2] = float64(b >> 8)
			idx += 3
		}
	}
	return out
}

func visionAspectPreservingSize(height, width, patchSize, maxPatches, pool int32) (int32, int32, error) {
	if height <= 0 || width <= 0 {
		return 0, 0, core.E("native.VisionImagePatches", core.Sprintf("invalid image size %dx%d", height, width), nil)
	}
	targetPx := float64(maxPatches) * float64(patchSize) * float64(patchSize)
	factor := math.Sqrt(targetPx / (float64(height) * float64(width)))
	sideMult := pool * patchSize

	th := int32(math.Floor(factor*float64(height)/float64(sideMult))) * sideMult
	tw := int32(math.Floor(factor*float64(width)/float64(sideMult))) * sideMult
	if th == 0 && tw == 0 {
		return 0, 0, core.E("native.VisionImagePatches", "image degenerates to 0x0 under the patch budget", nil)
	}
	maxSide := (maxPatches / (pool * pool)) * sideMult
	if th == 0 {
		th = sideMult
		tw = min(int32(math.Floor(float64(width)/float64(height)))*sideMult, maxSide)
	} else if tw == 0 {
		tw = sideMult
		th = min(int32(math.Floor(float64(height)/float64(width)))*sideMult, maxSide)
	}
	if int64(th)*int64(tw) > int64(targetPx) {
		return 0, 0, core.E("native.VisionImagePatches", core.Sprintf("target %dx%d exceeds the %d-patch budget", th, tw, maxPatches), nil)
	}
	return th, tw, nil
}

func patchifyVisionPixelsBF16(pixels []float32, h, w, patch int32) []byte {
	gridH, gridW := int(h/patch), int(w/patch)
	p := int(patch)
	patchDim := p * p * 3
	out := make([]byte, gridH*gridW*patchDim*bf16Size)
	row := 0
	for gy := 0; gy < gridH; gy++ {
		for gx := 0; gx < gridW; gx++ {
			col := 0
			for py := 0; py < p; py++ {
				y := gy*p + py
				for px := 0; px < p; px++ {
					x := gx*p + px
					src := (y*int(w) + x) * 3
					for c := 0; c < 3; c++ {
						hh := f32ToBF16(pixels[src+c])
						dst := (row*patchDim + col) * bf16Size
						out[dst], out[dst+1] = byte(hh), byte(hh>>8)
						col++
					}
				}
			}
			row++
		}
	}
	return out
}

func visionResizeBicubicAA(src []float64, h, w, th, tw int32) []float64 {
	horiz := make([]float64, int(h)*int(tw)*3)
	visionResamplePass(src, horiz, int(w), int(tw), int(h), 3, true)
	out := make([]float64, int(th)*int(tw)*3)
	visionResamplePass(horiz, out, int(h), int(th), int(tw), 3, false)
	return out
}

func visionCubicFilter(x float64) float64 {
	const a = -0.5
	if x < 0 {
		x = -x
	}
	switch {
	case x < 1:
		return ((a+2)*x-(a+3))*x*x + 1
	case x < 2:
		return (((x-5)*x+8)*x - 4) * a
	default:
		return 0
	}
}

func visionResamplePass(src, dst []float64, inLen, outLen, lines, channels int, horizontal bool) {
	scale := float64(inLen) / float64(outLen)
	filterScale := scale
	if filterScale < 1 {
		filterScale = 1
	}
	support := 2.0 * filterScale
	weights := make([]float64, 0, int(support)*2+3)
	for out := 0; out < outLen; out++ {
		center := (float64(out) + 0.5) * scale
		xmin := int(center - support + 0.5)
		if xmin < 0 {
			xmin = 0
		}
		xmax := int(center + support + 0.5)
		if xmax > inLen {
			xmax = inLen
		}
		weights = weights[:0]
		sum := 0.0
		for x := xmin; x < xmax; x++ {
			wgt := visionCubicFilter((float64(x) - center + 0.5) / filterScale)
			weights = append(weights, wgt)
			sum += wgt
		}
		if sum != 0 {
			for i := range weights {
				weights[i] /= sum
			}
		}
		for line := 0; line < lines; line++ {
			for c := 0; c < channels; c++ {
				acc := 0.0
				for k, wgt := range weights {
					var at int
					if horizontal {
						at = (line*inLen + xmin + k) * channels
					} else {
						at = ((xmin+k)*lines + line) * channels
					}
					acc += src[at+c] * wgt
				}
				var to int
				if horizontal {
					to = (line*outLen + out) * channels
				} else {
					to = (out*lines + line) * channels
				}
				dst[to+c] = acc
			}
		}
	}
}
