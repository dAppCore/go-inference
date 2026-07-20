// SPDX-Licence-Identifier: EUPL-1.2

package qwen35

import (
	"image"
	"math"

	core "dappco.re/go"
)

// vision_preprocess.go is the #59 normalisation follow-up (docs/design-qwen-vision-factory.md §4
// named it: "aligning preprocessing with Qwen2VLImageProcessorFast is a separate, measured
// follow-up" — this file is that follow-up). It supplies the piece ImageToPatchGrid was missing: the
// HF processor's declared per-channel pixel normalisation and smart_resize pixel-count-budget resize
// policy, sourced from a checkpoint's preprocessor_config.json (VisionPreprocessConfig,
// LoadVisionPreprocessConfig), plus the two host algorithms that implement it — qwenSmartResizeTarget
// (the dimension arithmetic, a verbatim port of
// transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize) and resizeBicubicRGB (the
// pixel resample, an antialiased separable bicubic filter matching Pillow/torchvision's convention).
// Both were verified against the REAL reference (mlx-community/Qwen3.6-27B-4bit's shipped
// preprocessor_config.json fed through a local transformers+torchvision install) — see this
// package's vision_preprocess_test.go and the design doc addendum for the receipt; no golden fixture
// is committed (no repo dependency on a Python environment), so the numbers there are reproducible,
// not re-checked by `go test`.

// VisionPreprocessConfig is the HF Qwen2VLImageProcessor's declared pixel-normalisation and
// resize-bound policy for one checkpoint — sourced from preprocessor_config.json, engine-neutral (no
// engine import; engine/metal's qwen_vision.go loads it at the same load seam it loads the tower
// itself, mirroring how the gemma tower attaches its own processor config).
type VisionPreprocessConfig struct {
	// ImageMean/ImageStd are the per-channel (R,G,B) normalisation constants the reference applies
	// AFTER [0,1] rescale: (pixel/255 - mean[c]) / std[c]. A zero ImageStd[c] is treated as 1 (a
	// no-op divisor) by normalized() — the safe degenerate reading of an unset config, not a real
	// checkpoint value (every real image_std channel HF ships is > 0).
	ImageMean, ImageStd [3]float32
	// MinPixels/MaxPixels are smart_resize's pixel-count budget (HF's size.shortest_edge/
	// longest_edge) — the resized image's total pixel count is clamped into [MinPixels,MaxPixels]
	// before the patch grid is cut. <=0 disables that side's bound (no forced up/downscale) — the
	// safe degenerate reading for a caller with no resize policy to declare (e.g. a unit test
	// exercising the tower's shape maths on a pre-aligned fixture); every real checkpoint's
	// preprocessor_config.json this package has seen declares positive bounds.
	MinPixels, MaxPixels int
}

// openAIClipMean/openAIClipStd are transformers' Qwen2VLImageProcessor CLASS defaults
// (transformers.image_utils.OPENAI_CLIP_MEAN/STD — confirmed against a local transformers install,
// 5.6.0.dev) — the values that apply when a checkpoint's preprocessor_config.json omits image_mean/
// image_std, or ships no preprocessor_config.json at all. The REAL checkpoint this port targets
// (mlx-community/Qwen3.6-27B-4bit) overrides both to [0.5,0.5,0.5] explicitly — the class default is
// the fallback path, not the common case, but it is what "HF-standard defaults when absent" means.
var (
	openAIClipMean = [3]float32{0.48145466, 0.4578275, 0.40821073}
	openAIClipStd  = [3]float32{0.26862954, 0.26130258, 0.27577711}
)

// qwen2VLDefaultMinPixels/qwen2VLDefaultMaxPixels are Qwen2VLImageProcessor's class-level size
// default (size = {"shortest_edge": 56*56, "longest_edge": 28*28*1280}) — confirmed against the same
// local transformers install. Like the CLIP mean/std above, this is the fallback a checkpoint's own
// preprocessor_config.json normally overrides (the 27B declares 65536/16777216).
const (
	qwen2VLDefaultMinPixels = 56 * 56        // 3136
	qwen2VLDefaultMaxPixels = 28 * 28 * 1280 // 1003520
)

// defaultVisionPreprocessConfig is the HF-standard fallback LoadVisionPreprocessConfig returns when
// a checkpoint ships no preprocessor_config.json.
func defaultVisionPreprocessConfig() VisionPreprocessConfig {
	return VisionPreprocessConfig{
		ImageMean: openAIClipMean,
		ImageStd:  openAIClipStd,
		MinPixels: qwen2VLDefaultMinPixels,
		MaxPixels: qwen2VLDefaultMaxPixels,
	}
}

// normalized returns pp with degenerate fields made safe: a zero ImageStd[c] becomes 1 so
// normalisePixel never divides by zero. MinPixels/MaxPixels are left untouched — qwenSmartResizeTarget
// already treats <=0 as "no bound on this side", so a genuinely zero-value VisionPreprocessConfig{}
// degenerates to the pre-#59 behaviour (round-to-nearest-patch·merge-multiple, plain /255 rescale, no
// normalisation) rather than this package inventing a resize/normalisation policy nobody declared.
func (pp VisionPreprocessConfig) normalized() VisionPreprocessConfig {
	out := pp
	for c := range 3 {
		if out.ImageStd[c] == 0 {
			out.ImageStd[c] = 1
		}
	}
	return out
}

// visionPreprocessorConfigJSON is the subset of a Qwen-VL-family checkpoint's preprocessor_config.json
// this package reads — https://huggingface.co/mlx-community/Qwen3.6-27B-4bit/resolve/main/
// preprocessor_config.json is the confirmed real shape. patch_size/merge_size/temporal_patch_size are
// deliberately NOT read here: VisionTowerConfig already carries them (weight/vision_config-derived,
// per vision_loader.go), and re-reading a second, potentially-conflicting source of truth for the
// same numbers is out of this file's job.
type visionPreprocessorConfigJSON struct {
	Size struct {
		ShortestEdge int `json:"shortest_edge"`
		LongestEdge  int `json:"longest_edge"`
	} `json:"size"`
	ImageMean []float32 `json:"image_mean"`
	ImageStd  []float32 `json:"image_std"`
}

// LoadVisionPreprocessConfig reads preprocessor_config.json from a qwen35 vision-towered checkpoint
// directory — the HF Qwen2VLImageProcessor's declared per-channel normalisation and smart_resize
// pixel bounds. Returns the HF class defaults (openAIClipMean/Std, 56²/28²·1280 px — see
// defaultVisionPreprocessConfig) and ok=false when the checkpoint ships no preprocessor_config.json:
// normal for an older/bare checkpoint, never an error — mirrors the gemma tower's own processor-config
// fallback (LoadGemma4ImageFeatureConfigs returns nil,nil on a missing file rather than failing the
// load). A PRESENT but malformed file (bad JSON) fails loud, the same present-but-broken-fails-loud
// contract LoadVisionTower applies to the tower weights themselves. A present file's individual
// missing fields (e.g. no image_mean array) fall back to the class default for that field only —
// partial real-world preprocessor_config.json files do exist.
func LoadVisionPreprocessConfig(dir string) (cfg VisionPreprocessConfig, ok bool, err error) {
	def := defaultVisionPreprocessConfig()
	path := core.PathJoin(dir, "preprocessor_config.json")
	read := core.ReadFile(path)
	if !read.OK {
		return def, false, nil
	}
	var pc visionPreprocessorConfigJSON
	if r := core.JSONUnmarshal(read.Bytes(), &pc); !r.OK {
		return VisionPreprocessConfig{}, false, core.NewError("qwen35.LoadVisionPreprocessConfig: parse " + path)
	}
	out := def
	if len(pc.ImageMean) == 3 {
		out.ImageMean = [3]float32{pc.ImageMean[0], pc.ImageMean[1], pc.ImageMean[2]}
	}
	if len(pc.ImageStd) == 3 {
		out.ImageStd = [3]float32{pc.ImageStd[0], pc.ImageStd[1], pc.ImageStd[2]}
	}
	if pc.Size.ShortestEdge > 0 {
		out.MinPixels = pc.Size.ShortestEdge
	}
	if pc.Size.LongestEdge > 0 {
		out.MaxPixels = pc.Size.LongestEdge
	}
	return out, true, nil
}

// qwenSmartResizeTarget ports Qwen2VLImageProcessor's smart_resize (image_processing_qwen2_vl.py)
// dimension arithmetic: both output axes are multiples of factor (=PatchSize·MergeSize), and the
// resulting pixel count is clamped into [minPixels,maxPixels] when a bound is positive (<=0 disables
// that side — see VisionPreprocessConfig's doc comment). The rounding step uses round-half-to-even
// (math.RoundToEven), matching Python's builtin round() — NOT math.Round, which is round-half-away-
// -from-zero and disagrees with Python at exact .5 boundaries (confirmed against a local transformers
// install: smart_resize(2,2,factor=4,...) rounds 2/4=0.5 DOWN to the even 0, not up to 1).
func qwenSmartResizeTarget(height, width, factor, minPixels, maxPixels int) (hBar, wBar int, err error) {
	if height <= 0 || width <= 0 || factor <= 0 {
		return 0, 0, core.NewError("qwen35.qwenSmartResizeTarget: height, width and factor must be positive")
	}
	lo, hi := height, width
	if lo > hi {
		lo, hi = hi, lo
	}
	if float64(hi)/float64(lo) > 200 {
		return 0, 0, core.NewError(core.Sprintf(
			"qwen35.qwenSmartResizeTarget: absolute aspect ratio must be smaller than 200, got %dx%d", height, width))
	}
	hBarF := math.RoundToEven(float64(height)/float64(factor)) * float64(factor)
	wBarF := math.RoundToEven(float64(width)/float64(factor)) * float64(factor)
	switch area := hBarF * wBarF; {
	case maxPixels > 0 && area > float64(maxPixels):
		beta := math.Sqrt(float64(height*width) / float64(maxPixels))
		hBarF = math.Max(float64(factor), math.Floor(float64(height)/beta/float64(factor))*float64(factor))
		wBarF = math.Max(float64(factor), math.Floor(float64(width)/beta/float64(factor))*float64(factor))
	case minPixels > 0 && area < float64(minPixels):
		beta := math.Sqrt(float64(minPixels) / float64(height*width))
		hBarF = math.Ceil(float64(height)*beta/float64(factor)) * float64(factor)
		wBarF = math.Ceil(float64(width)*beta/float64(factor)) * float64(factor)
	}
	return int(hBarF), int(wBarF), nil
}

// decodeImageRGB255 decodes an image.Image into an interleaved [h,w,3] float64 RGB buffer in the
// 0..255 domain — the resize/normalise stage's working precision. img.At(x,y).RGBA() is the standard-
// library generic path (16-bit, alpha-premultiplied per image.Color's contract; for an opaque source
// — every fixture and real photo this package has seen — premultiplication is a no-op), the same
// per-pixel read ImageToPatchGrid used inline before this file gave it a resize stage to feed.
func decodeImageRGB255(img image.Image, bounds image.Rectangle) []float64 {
	h, w := bounds.Dy(), bounds.Dx()
	out := make([]float64, h*w*3)
	i := 0
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			out[i] = float64(r >> 8)
			out[i+1] = float64(g >> 8)
			out[i+2] = float64(b >> 8)
			i += 3
		}
	}
	return out
}

// roundClampByte rounds to the nearest uint8 value (round-half-to-even) and clamps to [0,255] — the
// reference resizes in the 0..255 domain and lands back on integer pixel values before rescale/
// normalise (confirmed against the real processor: its resized tensor dtype is uint8, not float).
func roundClampByte(v float64) float64 {
	v = math.RoundToEven(v)
	switch {
	case v < 0:
		return 0
	case v > 255:
		return 255
	default:
		return v
	}
}

// resizeBicubicRGB resamples an interleaved [h,w,3] float64 RGB buffer (0..255 domain) to [th,tw,3]
// with a separable two-pass antialiased bicubic filter (a=-0.5 cubic convolution kernel, support
// widened by the downsample factor when shrinking) — the Pillow-compatible convention torchvision's
// resize(..., antialias=True) implements (torchvision added antialias specifically to match Pillow),
// which is what transformers' Qwen2VLImageProcessor.resize calls. Independently validated against the
// real processor (mlx-community/Qwen3.6-27B-4bit's preprocessor_config.json fed through a local
// transformers+torchvision install) on both an upscale (64x64->256x256) and a downscale
// (4001x5003->3648x4576) case: max abs diff 1/255, 0.02%-0.09% of pixels differing by that single LSB
// — quantisation-noise-level agreement with the reference, not a byte-identical port (no Python
// dependency ships in this repo, so that comparison is not a committed golden — see
// vision_preprocess_test.go for the reproducible fixture and the design doc addendum for the run).
func resizeBicubicRGB(src []float64, h, w, th, tw int) []float64 {
	horiz := make([]float64, h*tw*3)
	resampleAxis(src, horiz, w, tw, h, true)
	out := make([]float64, th*tw*3)
	resampleAxis(horiz, out, h, th, tw, false)
	return out
}

// cubicKernel is the a=-0.5 cubic convolution kernel (Catmull-Rom family; Pillow's own BICUBIC
// filter uses the same constant).
func cubicKernel(x float64) float64 {
	const a = -0.5
	if x < 0 {
		x = -x
	}
	switch {
	case x < 1:
		return ((a+2)*x-(a+3))*x*x + 1
	case x < 2:
		return (((x-5)*x+8)*x-4)*a
	default:
		return 0
	}
}

// resampleAxis runs one separable pass of the bicubic resample over 3-channel rows: horizontal
// (inLen/outLen index the width axis, lines=height, row-major [lines,inLen,3] -> [lines,outLen,3]) or
// vertical (inLen/outLen index the height axis, lines=width, column-major [inLen,lines,3] ->
// [outLen,lines,3] — the horizontal pass's own output layout, so the two passes compose without a
// transpose).
func resampleAxis(src, dst []float64, inLen, outLen, lines int, horizontal bool) {
	scale := float64(inLen) / float64(outLen)
	filterScale := max(scale, 1)
	support := 2.0 * filterScale
	weights := make([]float64, 0, int(support)*2+3)
	for out := range outLen {
		center := (float64(out) + 0.5) * scale
		xmin := max(int(center-support+0.5), 0)
		xmax := min(int(center+support+0.5), inLen)
		weights = weights[:0]
		sum := 0.0
		for x := xmin; x < xmax; x++ {
			wgt := cubicKernel((float64(x) - center + 0.5) / filterScale)
			weights = append(weights, wgt)
			sum += wgt
		}
		if sum != 0 {
			for i := range weights {
				weights[i] /= sum
			}
		}
		for line := range lines {
			var acc [3]float64
			for k, wgt := range weights {
				var at int
				if horizontal {
					at = (line*inLen + xmin + k) * 3
				} else {
					at = ((xmin+k)*lines + line) * 3
				}
				acc[0] += src[at] * wgt
				acc[1] += src[at+1] * wgt
				acc[2] += src[at+2] * wgt
			}
			var to int
			if horizontal {
				to = (line*outLen + out) * 3
			} else {
				to = (out*lines + line) * 3
			}
			dst[to], dst[to+1], dst[to+2] = acc[0], acc[1], acc[2]
		}
	}
}
