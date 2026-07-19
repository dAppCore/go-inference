// SPDX-Licence-Identifier: EUPL-1.2

package dotsocr

import (
	"bytes"
	"image"
	_ "image/jpeg" // registers the JPEG decoder with image.Decode
	_ "image/png"  // registers the PNG decoder with image.Decode
	"math"

	core "dappco.re/go"
)

// image.go turns raw image bytes into the [nPatches, patchDim] pixel_values + grid_thw EncodeImage
// consumes, replicating transformers' Qwen2VLImageProcessorPil (the torchvision-free "slow"/PIL
// backend DOTS-OCR's own preprocessor_config.json selects — image_processor_type
// "Qwen2VLImageProcessor" — see image_processing_pil_qwen2_vl.py): smart_resize the image to the
// nearest patch_size·merge_size-aligned box within [min_pixels,max_pixels], rescale to [0,1],
// normalise (CLIP mean/std), then patchify into spatial_merge_size²-grouped patch vectors.
//
// PRECISION BOUNDARY (named per this lane's "truthful partial" contract): smart_resize's
// DIMENSION arithmetic and the normalise/patchify maths are bit-exact vs the real processor (see
// image_preproc_golden.json, captured from the real Qwen2VLImageProcessorPil on the committed
// fixture.png). The PIXEL RESAMPLING kernel is NOT — resizeBilinear below is a standard half-
// pixel-centre bilinear resize, while the reference uses PIL's BICUBIC (a windowed cubic-
// convolution filter with its own antialiasing pass when downscaling); matching that bit-for-bit
// would mean re-implementing PIL's C resampler, out of scope for this lane. This boundary is
// INVISIBLE to the committed golden and the E2E test: fixture.png's dimensions (280×84) are an
// exact multiple of patch_size·merge_size=28 on both axes, so smart_resize leaves it untouched
// and resizeBilinear is never invoked on that path — see gen_fixture.py's doc comment. Any image
// whose dimensions are ALREADY smart_resize-aligned gets the same exact treatment; only images
// needing an actual resample take the approximate path.

// smartResize ports transformers' smart_resize verbatim (image_processing_pil_qwen2_vl.py):
// round both dimensions to the nearest multiple of factor, then grow/shrink (preserving aspect
// ratio) until the pixel count lands in [minPixels,maxPixels]. Refuses (matching the reference's
// ValueError) when the aspect ratio exceeds 200:1.
func smartResize(height, width, factor, minPixels, maxPixels int) (int, int, error) {
	if height <= 0 || width <= 0 || factor <= 0 {
		return 0, 0, core.NewError("dotsocr.smartResize: height/width/factor must be positive")
	}
	hf, wf := float64(height), float64(width)
	maxDim, minDim := math.Max(hf, wf), math.Min(hf, wf)
	if maxDim/minDim > 200 {
		return 0, 0, core.NewError(core.Sprintf("dotsocr.smartResize: absolute aspect ratio must be smaller than 200, got %v", maxDim/minDim))
	}
	hBar := pythonRound(hf/float64(factor)) * factor
	wBar := pythonRound(wf/float64(factor)) * factor
	switch {
	case hBar*wBar > maxPixels:
		beta := math.Sqrt(hf * wf / float64(maxPixels))
		hBar = max(factor, int(math.Floor(hf/beta/float64(factor)))*factor)
		wBar = max(factor, int(math.Floor(wf/beta/float64(factor)))*factor)
	case hBar*wBar < minPixels:
		beta := math.Sqrt(float64(minPixels) / (hf * wf))
		hBar = int(math.Ceil(hf*beta/float64(factor))) * factor
		wBar = int(math.Ceil(wf*beta/float64(factor))) * factor
	}
	return hBar, wBar, nil
}

// pythonRound replicates Python 3's round-half-to-even for a non-negative float (smart_resize
// never rounds a negative value) — Go's math.Round is round-half-AWAY-from-zero, which silently
// disagrees with Python at exact .5 boundaries (common in practice: e.g. height/factor==1.5
// whenever height==42 and factor==28, an exactly-representable binary fraction, not a rare edge
// case).
func pythonRound(x float64) int {
	floor := math.Floor(x)
	diff := x - floor
	i := int(floor)
	switch {
	case diff < 0.5:
		return i
	case diff > 0.5:
		return i + 1
	default:
		if i%2 == 0 {
			return i
		}
		return i + 1
	}
}

// decodeImage decodes PNG or JPEG bytes (the two formats image.Decode's blank-imported decoders
// above register) into a Go image.Image.
func decodeImage(data []byte) (image.Image, error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, core.E("dotsocr.decodeImage", "decode image bytes", err)
	}
	return img, nil
}

// pixelRGB reads one pixel as [0,1]-normalised, alpha-unpremultiplied RGB — fully transparent
// pixels flatten to white (1,1,1), the sane default for a document/OCR image compositing over a
// page background; fully opaque pixels (every published test fixture, and the overwhelming
// majority of real photos/scans) take the direct path.
func pixelRGB(img image.Image, x, y int) (r, g, b float64) {
	rr, gg, bb, aa := img.At(x, y).RGBA()
	if aa == 0 {
		return 1, 1, 1
	}
	if aa == 0xffff {
		return float64(rr) / 0xffff, float64(gg) / 0xffff, float64(bb) / 0xffff
	}
	return float64(rr) / float64(aa), float64(gg) / float64(aa), float64(bb) / float64(aa)
}

// resizeBilinear resamples img to newW×newH using standard half-pixel-centre bilinear
// interpolation — the approximate path named in this file's doc comment; returns
// [newH][newW][3]float64 in [0,1] RGB.
func resizeBilinear(img image.Image, newW, newH int) [][][3]float64 {
	b := img.Bounds()
	oldW, oldH := b.Dx(), b.Dy()
	// Cache the source as a flat [0,1] RGB buffer once — pixelRGB re-decoding the same source
	// pixel up to newW·newH/(oldW·oldH) times each would otherwise dominate for large upscales.
	src := make([][3]float64, oldW*oldH)
	for y := range oldH {
		for x := range oldW {
			r, g, bl := pixelRGB(img, b.Min.X+x, b.Min.Y+y)
			src[y*oldW+x] = [3]float64{r, g, bl}
		}
	}
	at := func(x, y int) [3]float64 {
		if x < 0 {
			x = 0
		}
		if x >= oldW {
			x = oldW - 1
		}
		if y < 0 {
			y = 0
		}
		if y >= oldH {
			y = oldH - 1
		}
		return src[y*oldW+x]
	}

	scaleX := float64(oldW) / float64(newW)
	scaleY := float64(oldH) / float64(newH)
	out := make([][][3]float64, newH)
	for oy := range newH {
		srcY := (float64(oy)+0.5)*scaleY - 0.5
		y0 := int(math.Floor(srcY))
		fy := srcY - float64(y0)
		row := make([][3]float64, newW)
		for ox := range newW {
			srcX := (float64(ox)+0.5)*scaleX - 0.5
			x0 := int(math.Floor(srcX))
			fx := srcX - float64(x0)
			p00, p10 := at(x0, y0), at(x0+1, y0)
			p01, p11 := at(x0, y0+1), at(x0+1, y0+1)
			var px [3]float64
			for c := range 3 {
				top := p00[c]*(1-fx) + p10[c]*fx
				bot := p01[c]*(1-fx) + p11[c]*fx
				px[c] = top*(1-fy) + bot*fy
			}
			row[ox] = px
		}
		out[oy] = row
	}
	return out
}

// PixelValues is one preprocessed image, ready for EncodeImage: Values is the flat
// [GridT·GridH·GridW, patchDim] patch matrix, patchDim = NumChannels·TemporalPatchSize·PatchSize².
type PixelValues struct {
	Values              []float32
	GridT, GridH, GridW int
}

// clipMean/clipStd are the OPENAI_CLIP_MEAN/STD constants DOTS-OCR's preprocessor_config.json
// pins (image_mean/image_std) — confirmed against the real checkpoint's shipped file.
var clipMean = [3]float64{0.48145466, 0.4578275, 0.40821073}
var clipStd = [3]float64{0.26862954, 0.26130258, 0.27577711}

// Patchify decodes imageBytes and produces PixelValues per DOTS-OCR's vision_config: smart_resize
// (factor = PatchSize·SpatialMergeSize, bounds [minPixels,maxPixels]), rescale+normalise, then
// group into spatial_merge_size²-blocks in the (block_h,block_w,i_h,i_w) order EncodeImage's
// rotary table (vision.go) and PatchMerger both assume — see visionRotaryTable's doc comment.
// Only TemporalPatchSize==1 (every published DOTS-OCR checkpoint) is supported: a >1 value would
// need the reference's last-frame-repeat padding, unverified against any golden in this lane, so
// this refuses cleanly rather than guessing.
func Patchify(imageBytes []byte, vc *VisionConfig, minPixels, maxPixels int) (PixelValues, error) {
	if vc == nil {
		return PixelValues{}, core.NewError("dotsocr.Patchify: nil vision_config")
	}
	if vc.TemporalPatchSize != 1 {
		return PixelValues{}, core.NewError(core.Sprintf("dotsocr.Patchify: temporal_patch_size=%d is not supported (only 1, every published DOTS-OCR checkpoint)", vc.TemporalPatchSize))
	}
	if vc.NumChannels != 3 {
		// pixelRGB always decodes to 3 channels (RGB) — every published DOTS-OCR checkpoint's
		// num_channels is 3; a hypothetical variant declaring otherwise would need a different
		// decode path, not silently mismatched channel counts below.
		return PixelValues{}, core.NewError(core.Sprintf("dotsocr.Patchify: num_channels=%d is not supported (only 3, RGB)", vc.NumChannels))
	}
	img, err := decodeImage(imageBytes)
	if err != nil {
		return PixelValues{}, err
	}
	b := img.Bounds()
	width, height := b.Dx(), b.Dy()
	factor := vc.PatchSize * vc.SpatialMergeSize
	resizedH, resizedW, err := smartResize(height, width, factor, minPixels, maxPixels)
	if err != nil {
		return PixelValues{}, err
	}

	var pixels [][][3]float64 // [resizedH][resizedW][3], [0,1] RGB
	if resizedH == height && resizedW == width {
		pixels = make([][][3]float64, height)
		for y := range height {
			row := make([][3]float64, width)
			for x := range width {
				r, g, bl := pixelRGB(img, b.Min.X+x, b.Min.Y+y)
				row[x] = [3]float64{r, g, bl}
			}
			pixels[y] = row
		}
	} else {
		pixels = resizeBilinear(img, resizedW, resizedH) // NAMED APPROXIMATION — see file doc comment
	}

	gridH, gridW := resizedH/vc.PatchSize, resizedW/vc.PatchSize
	merge := vc.SpatialMergeSize
	patchDim := vc.NumChannels * vc.TemporalPatchSize * vc.PatchSize * vc.PatchSize
	nPatches := gridH * gridW
	values := make([]float32, nPatches*patchDim)

	idx := 0
	for blockH := 0; blockH < gridH/merge; blockH++ {
		for blockW := 0; blockW < gridW/merge; blockW++ {
			for ih := range merge {
				for iw := range merge {
					py0 := (blockH*merge + ih) * vc.PatchSize
					px0 := (blockW*merge + iw) * vc.PatchSize
					dst := values[idx*patchDim : (idx+1)*patchDim]
					// (channel, patch_h, patch_w) order — matches the real Conv2d weight's
					// [outC,inC,kh,kw] row-major flatten (weights.go's patch_embed doc comment).
					for c := range 3 {
						base := c * vc.PatchSize * vc.PatchSize
						for py := range vc.PatchSize {
							srcRow := pixels[py0+py][px0 : px0+vc.PatchSize]
							for px := range vc.PatchSize {
								v := (srcRow[px][c] - clipMean[c]) / clipStd[c]
								dst[base+py*vc.PatchSize+px] = float32(v)
							}
						}
					}
					idx++
				}
			}
		}
	}
	// The reference's do_rescale (divide by 255) is already folded into pixelRGB's [0,1] output:
	// color.Color.RGBA() widens an 8-bit channel value to 16-bit as value·257, so
	// value·257/65535 == value/255 exactly — a second explicit /255 here would double-rescale.

	return PixelValues{Values: values, GridT: 1, GridH: gridH, GridW: gridW}, nil
}

// PreprocessorConfig is the architecture-relevant subset of a DOTS-OCR preprocessor_config.json —
// the min/max pixel bounds and patch geometry Patchify's smart_resize/grid maths need. Field
// names/values confirmed against the real checkpoint's shipped file (image_processor_type
// "Qwen2VLImageProcessor").
type PreprocessorConfig struct {
	MinPixels         int `json:"min_pixels"`
	MaxPixels         int `json:"max_pixels"`
	PatchSize         int `json:"patch_size"`
	TemporalPatchSize int `json:"temporal_patch_size"`
	MergeSize         int `json:"merge_size"`
}

// ParsePreprocessorConfig parses a DOTS-OCR preprocessor_config.json.
func ParsePreprocessorConfig(data []byte) (*PreprocessorConfig, error) {
	var pc PreprocessorConfig
	if r := core.JSONUnmarshal(data, &pc); !r.OK {
		return nil, core.NewError("dotsocr.ParsePreprocessorConfig: preprocessor_config.json parse failed")
	}
	return &pc, nil
}
