// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"bytes"
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model/gemma4"
)

func hipNormalizeVisionImageFeatureConfig(cfg *gemma4.Gemma4ImageFeatureConfig) *gemma4.Gemma4ImageFeatureConfig {
	if cfg == nil {
		cfg = &gemma4.Gemma4ImageFeatureConfig{}
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

// hipUnifiedVisionImagePatches decodes an image and emits model-patch rows in
// the exact HWC/kernel-grouped order consumed by Gemma 4 unified vision.
func hipUnifiedVisionImagePatches(data []byte, cfg *gemma4.Gemma4ImageFeatureConfig) ([]float32, []int32, int, error) {
	pixels, height, width, softTokens, err := hipVisionImagePixels(data, cfg)
	if err != nil {
		return nil, nil, 0, err
	}
	normalized := hipNormalizeVisionImageFeatureConfig(cfg)
	patchSize := int(normalized.PatchSize)
	pool := int(normalized.PoolingKernelSize)
	modelPatch := patchSize * pool
	gridHeight := int(height) / modelPatch
	gridWidth := int(width) / modelPatch
	rows := gridHeight * gridWidth
	if rows <= 0 || rows != softTokens {
		return nil, nil, 0, core.E("hip.UnifiedVisionImagePatches", "image produced inconsistent model-patch geometry", nil)
	}
	patchDim := modelPatch * modelPatch * 3
	patches := make([]float32, rows*patchDim)
	positions := make([]int32, rows*2)
	row := 0
	for gridRow := range gridHeight {
		for gridCol := range gridWidth {
			positions[row*2] = int32(gridCol)
			positions[row*2+1] = int32(gridRow)
			col := 0
			for kernelRow := range pool {
				for kernelCol := range pool {
					for patchRow := range patchSize {
						y := (gridRow*pool+kernelRow)*patchSize + patchRow
						for patchCol := range patchSize {
							x := (gridCol*pool+kernelCol)*patchSize + patchCol
							source := (y*int(width) + x) * 3
							copy(patches[row*patchDim+col:row*patchDim+col+3], pixels[source:source+3])
							col += 3
						}
					}
				}
			}
			row++
		}
	}
	return patches, positions, rows, nil
}

func hipVisionImagePixels(data []byte, cfg *gemma4.Gemma4ImageFeatureConfig) ([]float32, int32, int32, int, error) {
	normalized := hipNormalizeVisionImageFeatureConfig(cfg)
	decoded, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, 0, 0, 0, core.E("hip.VisionImagePixels", "decode image", err)
	}
	bounds := decoded.Bounds()
	height, width := int32(bounds.Dy()), int32(bounds.Dx())
	if height <= 0 || width <= 0 {
		return nil, 0, 0, 0, core.NewError("hip.VisionImagePixels: image has empty bounds")
	}
	source := hipVisionImageRGBFloat64(decoded, bounds)
	maxPatches := normalized.MaxSoftTokens * normalized.PoolingKernelSize * normalized.PoolingKernelSize
	targetHeight, targetWidth := height, width
	sideMultiple := normalized.PatchSize * normalized.PoolingKernelSize
	if normalized.DoResize || targetHeight%sideMultiple != 0 || targetWidth%sideMultiple != 0 {
		targetHeight, targetWidth, err = hipVisionAspectPreservingSize(height, width, normalized.PatchSize, maxPatches, normalized.PoolingKernelSize)
		if err != nil {
			return nil, 0, 0, 0, err
		}
	}
	resized := source
	if targetHeight != height || targetWidth != width {
		resized = hipVisionResizeBicubicAA(source, height, width, targetHeight, targetWidth)
	}
	pixels := make([]float32, len(resized))
	for index, value := range resized {
		value = math.RoundToEven(value)
		value = max(0, min(255, value))
		pixels[index] = float32(value * normalized.RescaleFactor)
	}
	grid := (targetHeight / normalized.PatchSize) * (targetWidth / normalized.PatchSize)
	softTokens := int(grid / (normalized.PoolingKernelSize * normalized.PoolingKernelSize))
	return pixels, targetHeight, targetWidth, softTokens, nil
}

func hipVisionImageRGBFloat64(input image.Image, bounds image.Rectangle) []float64 {
	switch source := input.(type) {
	case *image.NRGBA:
		return hipVisionNRGBAToRGBFloat64(source, bounds)
	case *image.RGBA:
		return hipVisionRGBAToRGBFloat64(source, bounds)
	case *image.YCbCr:
		return hipVisionYCbCrToRGBFloat64(source, bounds)
	default:
		return hipVisionGenericRGBFloat64(input, bounds)
	}
}

func hipVisionNRGBAToRGBFloat64(input *image.NRGBA, bounds image.Rectangle) []float64 {
	out := make([]float64, bounds.Dx()*bounds.Dy()*3)
	destination := 0
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		source := input.PixOffset(bounds.Min.X, y)
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			alpha := input.Pix[source+3]
			for channel := range 3 {
				value := input.Pix[source+channel]
				if alpha != 0xff {
					expanded := uint32(value)
					expanded |= expanded << 8
					expanded *= uint32(alpha)
					expanded /= 0xff
					value = byte(expanded >> 8)
				}
				out[destination+channel] = float64(value)
			}
			destination += 3
			source += 4
		}
	}
	return out
}

func hipVisionRGBAToRGBFloat64(input *image.RGBA, bounds image.Rectangle) []float64 {
	out := make([]float64, bounds.Dx()*bounds.Dy()*3)
	destination := 0
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		source := input.PixOffset(bounds.Min.X, y)
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			out[destination] = float64(input.Pix[source])
			out[destination+1] = float64(input.Pix[source+1])
			out[destination+2] = float64(input.Pix[source+2])
			destination += 3
			source += 4
		}
	}
	return out
}

func hipVisionYCbCrToRGBFloat64(input *image.YCbCr, bounds image.Rectangle) []float64 {
	out := make([]float64, bounds.Dx()*bounds.Dy()*3)
	destination := 0
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			yIndex := input.YOffset(x, y)
			chromaIndex := input.COffset(x, y)
			red, green, blue := color.YCbCrToRGB(input.Y[yIndex], input.Cb[chromaIndex], input.Cr[chromaIndex])
			out[destination] = float64(red)
			out[destination+1] = float64(green)
			out[destination+2] = float64(blue)
			destination += 3
		}
	}
	return out
}

func hipVisionGenericRGBFloat64(input image.Image, bounds image.Rectangle) []float64 {
	out := make([]float64, bounds.Dx()*bounds.Dy()*3)
	destination := 0
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			red, green, blue, _ := input.At(x, y).RGBA()
			out[destination] = float64(red >> 8)
			out[destination+1] = float64(green >> 8)
			out[destination+2] = float64(blue >> 8)
			destination += 3
		}
	}
	return out
}

func hipVisionAspectPreservingSize(height, width, patchSize, maxPatches, pool int32) (int32, int32, error) {
	if height <= 0 || width <= 0 {
		return 0, 0, core.E("hip.VisionImagePixels", core.Sprintf("invalid image size %dx%d", height, width), nil)
	}
	targetPixels := float64(maxPatches) * float64(patchSize) * float64(patchSize)
	factor := math.Sqrt(targetPixels / (float64(height) * float64(width)))
	sideMultiple := pool * patchSize
	targetHeight := int32(math.Floor(factor*float64(height)/float64(sideMultiple))) * sideMultiple
	targetWidth := int32(math.Floor(factor*float64(width)/float64(sideMultiple))) * sideMultiple
	if targetHeight == 0 && targetWidth == 0 {
		return 0, 0, core.NewError("hip.VisionImagePixels: image degenerates to 0x0 under the patch budget")
	}
	maxSide := (maxPatches / (pool * pool)) * sideMultiple
	if targetHeight == 0 {
		targetHeight = sideMultiple
		targetWidth = min(int32(math.Floor(float64(width)/float64(height)))*sideMultiple, maxSide)
	} else if targetWidth == 0 {
		targetWidth = sideMultiple
		targetHeight = min(int32(math.Floor(float64(height)/float64(width)))*sideMultiple, maxSide)
	}
	if int64(targetHeight)*int64(targetWidth) > int64(targetPixels) {
		return 0, 0, core.E("hip.VisionImagePixels", core.Sprintf("target %dx%d exceeds the %d-patch budget", targetHeight, targetWidth, maxPatches), nil)
	}
	return targetHeight, targetWidth, nil
}

func hipVisionResizeBicubicAA(source []float64, height, width, targetHeight, targetWidth int32) []float64 {
	horizontal := make([]float64, int(height)*int(targetWidth)*3)
	hipVisionResamplePass(source, horizontal, int(width), int(targetWidth), int(height), 3, true)
	out := make([]float64, int(targetHeight)*int(targetWidth)*3)
	hipVisionResamplePass(horizontal, out, int(height), int(targetHeight), int(targetWidth), 3, false)
	return out
}

func hipVisionCubicFilter(value float64) float64 {
	const coefficient = -0.5
	value = math.Abs(value)
	switch {
	case value < 1:
		return ((coefficient+2)*value-(coefficient+3))*value*value + 1
	case value < 2:
		return (((value-5)*value+8)*value - 4) * coefficient
	default:
		return 0
	}
}

func hipVisionResamplePass(source, destination []float64, inputLength, outputLength, lines, channels int, horizontal bool) {
	scale := float64(inputLength) / float64(outputLength)
	filterScale := max(scale, 1)
	support := 2 * filterScale
	weights := make([]float64, 0, int(support)*2+3)
	for output := range outputLength {
		center := (float64(output) + 0.5) * scale
		minimum := max(int(center-support+0.5), 0)
		maximum := min(int(center+support+0.5), inputLength)
		weights = weights[:0]
		var total float64
		for at := minimum; at < maximum; at++ {
			weight := hipVisionCubicFilter((float64(at) - center + 0.5) / filterScale)
			weights = append(weights, weight)
			total += weight
		}
		if total != 0 {
			for index := range weights {
				weights[index] /= total
			}
		}
		for line := range lines {
			for channel := range channels {
				var value float64
				for offset, weight := range weights {
					index := 0
					if horizontal {
						index = (line*inputLength + minimum + offset) * channels
					} else {
						index = ((minimum+offset)*lines + line) * channels
					}
					value += source[index+channel] * weight
				}
				index := 0
				if horizontal {
					index = (line*outputLength + output) * channels
				} else {
					index = (output*lines + line) * channels
				}
				destination[index+channel] = value
			}
		}
	}
}
