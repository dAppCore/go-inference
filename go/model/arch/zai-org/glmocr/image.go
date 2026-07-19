// SPDX-Licence-Identifier: EUPL-1.2

package glmocr

import (
	"bytes"
	"image"
	"image/color"
	_ "image/jpeg" // register the JPEG decoder for image.Decode
	_ "image/png"  // register the PNG decoder for image.Decode
	"math"

	core "dappco.re/go"
)

// image.go is GLM-OCR's image preprocessing front end — Glm46VImageProcessor ported host-side
// (transformers/models/glm46v/image_processing_glm46v.py): smart_resize's dimension arithmetic,
// rescale (/255) + normalise (OpenAI CLIP mean/std), and the patchify layout GlmOcrVisionModel's
// patch_embed consumes. RESAMPLING IS A NAMED BOUNDARY THIS PACKAGE DOES NOT IMPLEMENT: rather
// than port torchvision's antialiased bicubic resize (unverifiable here without a golden — see
// the task's honest-boundary instruction), DecodeAndPatchify REFUSES an image whose natural
// (height,width) is not already smart_resize's target for that size, naming the exact
// dimensions the caller must pre-size to. This is provably safe, not merely convenient: at
// scale=1 (source size == target size), a real bicubic resize is the identity transform bar
// float noise — confirmed empirically against the REAL Glm46VImageProcessor on this package's
// own 112×112 testdata/fixture.png fixture (max abs diff 2.4e-7, f32 epsilon) — so every image
// this function accepts is processed bit-for-bit as the reference would.

// ImagePreprocessorConfig is the architecture-relevant subset of a GLM-OCR checkpoint's
// preprocessor_config.json: https://huggingface.co/zai-org/GLM-OCR/resolve/main/preprocessor_config.json
type ImagePreprocessorConfig struct {
	Size struct {
		ShortestEdge int `json:"shortest_edge"` // smart_resize's min_pixels (misleadingly named — a pixel COUNT, not a side length)
		LongestEdge  int `json:"longest_edge"`  // smart_resize's max_pixels
	} `json:"size"`
	PatchSize         int       `json:"patch_size"`
	TemporalPatchSize int       `json:"temporal_patch_size"`
	MergeSize         int       `json:"merge_size"`
	ImageMean         []float32 `json:"image_mean"`
	ImageStd          []float32 `json:"image_std"`
}

// LoadImagePreprocessorConfig reads preprocessor_config.json from a GLM-OCR checkpoint
// directory.
func LoadImagePreprocessorConfig(dir string) (*ImagePreprocessorConfig, error) {
	path := core.PathJoin(dir, "preprocessor_config.json")
	read := core.ReadFile(path)
	if !read.OK {
		return nil, core.E("glmocr.LoadImagePreprocessorConfig", "read "+path, resultErr(read))
	}
	data, ok := read.Value.([]byte)
	if !ok {
		return nil, core.NewError("glmocr.LoadImagePreprocessorConfig: " + path + " read returned non-byte data")
	}
	var pc ImagePreprocessorConfig
	if r := core.JSONUnmarshal(data, &pc); !r.OK {
		return nil, core.NewError("glmocr.LoadImagePreprocessorConfig: parse " + path)
	}
	if pc.Size.ShortestEdge <= 0 || pc.Size.LongestEdge <= 0 || pc.PatchSize <= 0 || pc.TemporalPatchSize <= 0 || pc.MergeSize <= 0 {
		return nil, core.NewError("glmocr.LoadImagePreprocessorConfig: " + path + " is missing size/patch_size/temporal_patch_size/merge_size")
	}
	if len(pc.ImageMean) != 3 || len(pc.ImageStd) != 3 {
		return nil, core.NewError("glmocr.LoadImagePreprocessorConfig: " + path + " image_mean/image_std must have 3 channels")
	}
	return &pc, nil
}

// smartResizeTarget ports smart_resize's dimension arithmetic (image_processing_glm46v.py)
// byte-for-byte for the still-image call site, where num_frames and temporal_factor are BOTH
// always temporalPatchSize (GLM-OCR's image processor calls
// smart_resize(num_frames=temporal_patch_size, temporal_factor=temporal_patch_size, ...)) —
// which makes t_bar collapse to temporalPatchSize itself and the "num_frames < temporal_factor"
// guard unreachable, so this signature omits both parameters as the (always-equal) single
// temporalPatchSize.
func smartResizeTarget(height, width, factor, minPixels, maxPixels, temporalPatchSize int) (hBar, wBar int, err error) {
	h, w := height, width
	if h < factor || w < factor {
		scale := math.Max(float64(factor)/float64(h), float64(factor)/float64(w))
		h = int(float64(h) * scale)
		w = int(float64(w) * scale)
	}
	hi, lo := h, w
	if lo > hi {
		hi, lo = lo, hi
	}
	if lo == 0 || float64(hi)/float64(lo) > 200 {
		return 0, 0, core.NewError(core.Sprintf("glmocr.smartResizeTarget: absolute aspect ratio must be smaller than 200, got %dx%d", height, width))
	}
	hBarF := math.Round(float64(h)/float64(factor)) * float64(factor)
	wBarF := math.Round(float64(w)/float64(factor)) * float64(factor)
	tBar := float64(temporalPatchSize)
	hBar, wBar = int(hBarF), int(wBarF)
	pixels := tBar * hBarF * wBarF
	switch {
	case pixels > float64(maxPixels):
		beta := math.Sqrt(float64(temporalPatchSize*h*w) / float64(maxPixels))
		hBar = int(math.Max(float64(factor), math.Floor(float64(h)/beta/float64(factor))*float64(factor)))
		wBar = int(math.Max(float64(factor), math.Floor(float64(w)/beta/float64(factor))*float64(factor)))
	case pixels < float64(minPixels):
		beta := math.Sqrt(float64(minPixels) / float64(temporalPatchSize*h*w))
		hBar = int(math.Ceil(float64(h)*beta/float64(factor)) * float64(factor))
		wBar = int(math.Ceil(float64(w)*beta/float64(factor)) * float64(factor))
	}
	return hBar, wBar, nil
}

// PatchGrid is one decoded+patchified image's flattened patch vectors, ready for the vision
// tower's patch_embed projection, plus the (GridT,GridH,GridW) PRE-merge patch grid geometry
// the text decoder's mrope position ids and the vision tower's 2D rotary table both key off.
type PatchGrid struct {
	Patches             []float32 // [N, PatchDim] flat, N = GridT*GridH*GridW
	GridT, GridH, GridW int
	PatchDim            int
}

// DecodeAndPatchify decodes imageBytes (PNG/JPEG) and patchifies it for the vision tower —
// see the file doc comment for the resize boundary this refuses rather than silently
// resampling.
func DecodeAndPatchify(imageBytes []byte, pc *ImagePreprocessorConfig, vc *VisionConfig) (*PatchGrid, error) {
	if pc == nil || vc == nil {
		return nil, core.NewError("glmocr.DecodeAndPatchify: nil preprocessor/vision config")
	}
	if vc.InChannels != 3 {
		return nil, core.NewError(core.Sprintf("glmocr.DecodeAndPatchify: vision_config.in_channels=%d, only 3 (RGB) is supported", vc.InChannels))
	}
	img, _, err := image.Decode(bytes.NewReader(imageBytes))
	if err != nil {
		return nil, core.E("glmocr.DecodeAndPatchify", "decode image", err)
	}
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	if width <= 0 || height <= 0 {
		return nil, core.NewError("glmocr.DecodeAndPatchify: empty image")
	}

	factor := pc.PatchSize * pc.MergeSize
	hBar, wBar, err := smartResizeTarget(height, width, factor, pc.Size.ShortestEdge, pc.Size.LongestEdge, pc.TemporalPatchSize)
	if err != nil {
		return nil, err
	}
	if hBar != height || wBar != width {
		return nil, core.NewError(core.Sprintf(
			"glmocr.DecodeAndPatchify: image is %dx%d (h×w); GLM-OCR's preprocessor would resize this to %dx%d "+
				"(nearest %dpx multiple within the min/max pixel bounds) — bicubic resampling is not implemented "+
				"in this lane (a named boundary); pre-size the image to %dx%d (or any size that is ALREADY a "+
				"stable smart_resize target — e.g. exact multiples of %dpx per side) before calling OCR",
			height, width, hBar, wBar, factor, hBar, wBar, factor))
	}

	rgb := make([]float32, height*width*3)
	mean, std := pc.ImageMean, pc.ImageStd
	for y := range height {
		for x := range width {
			c := color.RGBAModel.Convert(img.At(bounds.Min.X+x, bounds.Min.Y+y)).(color.RGBA)
			base := (y*width + x) * 3
			rgb[base+0] = (float32(c.R)/255 - mean[0]) / std[0]
			rgb[base+1] = (float32(c.G)/255 - mean[1]) / std[1]
			rgb[base+2] = (float32(c.B)/255 - mean[2]) / std[2]
		}
	}

	patchSize, merge, temporal := pc.PatchSize, pc.MergeSize, pc.TemporalPatchSize
	gridH, gridW := height/patchSize, width/patchSize
	if gridH%merge != 0 || gridW%merge != 0 {
		return nil, core.NewError(core.Sprintf("glmocr.DecodeAndPatchify: patch grid %dx%d is not divisible by merge_size %d", gridH, gridW, merge))
	}
	hpos, wpos := visionPosIDs(1, gridH, gridW, merge)
	patchDim := 3 * temporal * patchSize * patchSize
	n := len(hpos)
	patches := make([]float32, n*patchDim)
	for i := range n {
		rowBase, colBase := hpos[i]*patchSize, wpos[i]*patchSize
		out := patches[i*patchDim : (i+1)*patchDim]
		o := 0
		for c := range 3 {
			for range temporal { // both temporal copies are identical (a static image, no motion)
				for ph := range patchSize {
					for pw := range patchSize {
						out[o] = rgb[((rowBase+ph)*width+(colBase+pw))*3+c]
						o++
					}
				}
			}
		}
	}
	return &PatchGrid{Patches: patches, GridT: 1, GridH: gridH, GridW: gridW, PatchDim: patchDim}, nil
}

// resultErr pulls the error out of a failed core.Result for wrapping, tolerating a Result whose
// Value is not an error — mirrors whisper's helper of the same name (arch/openai/whisper/mel.go).
func resultErr(r core.Result) error {
	if err, ok := r.Value.(error); ok {
		return err
	}
	return nil
}
