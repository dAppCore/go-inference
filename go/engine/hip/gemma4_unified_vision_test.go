// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"bytes"
	"encoding/binary"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/gemma4"
	"dappco.re/go/inference/model/quant/mlxaffine"
	"dappco.re/go/inference/model/vision"
)

func TestUnifiedVisionTowerProjectPatches_Good(t *testing.T) {
	const rows, patchDim, embedDim, hidden, posemb = 2, 12, 4, 3, 2
	patches := hipUnifiedVisionTestValues(rows*patchDim, 0.07, -0.4)
	ln1w := hipUnifiedVisionTestValues(patchDim, 0.02, 0.8)
	ln1b := hipUnifiedVisionTestValues(patchDim, 0.01, -0.05)
	denseW := hipUnifiedVisionTestValues(embedDim*patchDim, 0.015, -0.2)
	denseB := hipUnifiedVisionTestValues(embedDim, 0.03, -0.04)
	ln2w := hipUnifiedVisionTestValues(embedDim, 0.04, 0.9)
	ln2b := hipUnifiedVisionTestValues(embedDim, 0.02, -0.03)
	positions := []int32{0, 0, 1, 0}
	pos := hipUnifiedVisionTestValues(posemb*2*embedDim, 0.01, -0.08)
	posw := hipUnifiedVisionTestValues(embedDim, 0.03, 0.85)
	posb := hipUnifiedVisionTestValues(embedDim, 0.015, -0.02)
	projectionW := hipUnifiedVisionTestValues(hidden*embedDim, 0.025, -0.15)

	uv := &vision.Unified{
		PatchLN1W: hipUnifiedVisionTestBF16(ln1w), PatchLN1B: hipUnifiedVisionTestBF16(ln1b),
		PatchDense: vision.Linear{
			Weight: hipUnifiedVisionTestBF16(denseW), Bias: hipUnifiedVisionTestBF16(denseB),
			OutDim: embedDim, InDim: patchDim,
		},
		PatchLN2W: hipUnifiedVisionTestBF16(ln2w), PatchLN2B: hipUnifiedVisionTestBF16(ln2b),
		PosEmbedding: hipUnifiedVisionTestBF16(pos),
		PosNormW:     hipUnifiedVisionTestBF16(posw), PosNormB: hipUnifiedVisionTestBF16(posb),
		Projection: vision.Linear{
			Weight: hipUnifiedVisionTestBF16(projectionW), OutDim: hidden, InDim: embedDim,
		},
		Cfg: vision.UnifiedConfig{
			MMEmbedDim: embedDim, TextHidden: hidden, PosembSize: posemb,
			PatchSize: 1, ModelPatchSize: 2, PoolKernel: 2,
			LayerNormEps: 1e-5, RMSNormEps: 1e-6,
		},
	}
	tower, err := newUnifiedVisionTowerFromLoaded(uv, nil, nil, nil)
	core.RequireNoError(t, err)

	quantizedPatches := hipUnifiedVisionTestDecodeBF16(hipUnifiedVisionTestBF16(patches))
	got, err := tower.ProjectPatches(quantizedPatches, positions, rows)
	core.RequireNoError(t, err)

	want := append([]float32(nil), quantizedPatches...)
	hipUnifiedVisionTestLayerNorm(want, hipUnifiedVisionTestDecodeBF16(uv.PatchLN1W), hipUnifiedVisionTestDecodeBF16(uv.PatchLN1B), rows, patchDim, uv.Cfg.LayerNormEps)
	want = hipUnifiedVisionTestMatMul(want, hipUnifiedVisionTestDecodeBF16(uv.PatchDense.Weight), rows, patchDim, embedDim, hipUnifiedVisionTestDecodeBF16(uv.PatchDense.Bias))
	hipUnifiedVisionTestLayerNorm(want, hipUnifiedVisionTestDecodeBF16(uv.PatchLN2W), hipUnifiedVisionTestDecodeBF16(uv.PatchLN2B), rows, embedDim, uv.Cfg.LayerNormEps)
	posValues := hipUnifiedVisionTestDecodeBF16(uv.PosEmbedding)
	for row := range rows {
		for axis := range 2 {
			at := int(positions[row*2+axis])
			for col := range embedDim {
				want[row*embedDim+col] += posValues[(at*2+axis)*embedDim+col]
			}
		}
	}
	hipUnifiedVisionTestLayerNorm(want, hipUnifiedVisionTestDecodeBF16(uv.PosNormW), hipUnifiedVisionTestDecodeBF16(uv.PosNormB), rows, embedDim, uv.Cfg.LayerNormEps)
	hipUnifiedVisionTestRMS(want, rows, embedDim, uv.Cfg.RMSNormEps)
	want = hipUnifiedVisionTestMatMul(want, hipUnifiedVisionTestDecodeBF16(uv.Projection.Weight), rows, embedDim, hidden, nil)
	assertFloat32SlicesNear(t, want, got, 0.0001)
}

func TestUnifiedVisionTowerQuantizedLinear_Good(t *testing.T) {
	const outDim, inDim, bits, groupSize = 3, 8, 4, 8
	values := hipUnifiedVisionTestValues(outDim*inDim, 0.125, -1)
	packed, scales, biases, err := mlxaffine.QuantizeTensor(values, outDim, inDim, bits, groupSize)
	core.RequireNoError(t, err)
	linear := vision.Linear{
		Weight: packed, Scales: scales, Biases: biases,
		OutDim: outDim, InDim: inDim, Bits: bits, GroupSize: groupSize, Kind: mlxaffine.Mode,
	}
	got, err := hipUnifiedVisionLinearWeights(linear)
	core.RequireNoError(t, err)
	want, err := mlxaffine.DequantizeTensor(packed, scales, biases, outDim, inDim, bits, groupSize)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, want, got, 0)
}

func TestHIPUnifiedVisionImagePatches_Good(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 96, 96))
	for y := range 96 {
		for x := range 96 {
			img.SetRGBA(x, y, color.RGBA{R: 200, G: 100, B: 50, A: 255})
		}
	}
	var encoded bytes.Buffer
	core.RequireNoError(t, png.Encode(&encoded, img))
	patches, positions, rows, err := hipUnifiedVisionImagePatches(encoded.Bytes(), &gemma4.Gemma4ImageFeatureConfig{})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 4, rows)
	core.AssertEqual(t, 4*48*48*3, len(patches))
	core.AssertEqual(t, []int32{0, 0, 1, 0, 0, 1, 1, 1}, positions)
	for _, value := range patches {
		if value < 0 || value > 1 || math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("patch value %v is not finite in [0,1]", value)
		}
	}
}

func TestHIPHardwareUnifiedVisionProjectImage_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_HIP_VISION_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_HIP_VISION_TESTS=1 to run the Gemma 4 unified vision smoke")
	}
	path := strings.TrimSpace(os.Getenv("GO_ROCM_VISION_MODEL_PATH"))
	if path == "" {
		t.Fatal("GO_ROCM_VISION_MODEL_PATH is required")
	}
	gemm := newSystemHIPAudioGEMM()
	if gemm == nil {
		t.Fatal("HIP rocBLAS GEMM is unavailable")
	}
	tower, err := loadUnifiedVisionTowerWithGEMM(path, gemm)
	core.RequireNoError(t, err)
	if tower == nil {
		t.Fatal("model has no Gemma 4 unified vision payload")
	}
	defer tower.Close()

	img := image.NewRGBA(image.Rect(0, 0, 96, 96))
	for y := range 96 {
		for x := range 96 {
			img.SetRGBA(x, y, color.RGBA{R: uint8(x * 2), G: uint8(y * 2), B: 50, A: 255})
		}
	}
	var encoded bytes.Buffer
	core.RequireNoError(t, png.Encode(&encoded, img))
	features, softTokens, err := tower.ProjectImage(encoded.Bytes())
	core.RequireNoError(t, err)
	if softTokens <= 0 || len(features) != softTokens*tower.loaded.Cfg.TextHidden {
		t.Fatalf("projected image = %d tokens, %d values, hidden %d", softTokens, len(features), tower.loaded.Cfg.TextHidden)
	}
	for _, value := range features {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("projected image contains non-finite value %v", value)
		}
	}
}

func hipUnifiedVisionTestValues(count int, step, offset float32) []float32 {
	out := make([]float32, count)
	for index := range out {
		out[index] = offset + float32(index%17)*step
	}
	return out
}

func hipUnifiedVisionTestBF16(values []float32) []byte {
	out := make([]byte, len(values)*2)
	for index, value := range values {
		binary.LittleEndian.PutUint16(out[index*2:], hipFloat32ToBFloat16(value))
	}
	return out
}

func hipUnifiedVisionTestDecodeBF16(payload []byte) []float32 {
	out := make([]float32, len(payload)/2)
	for index := range out {
		out[index] = hipBFloat16ToFloat32(binary.LittleEndian.Uint16(payload[index*2:]))
	}
	return out
}

func hipUnifiedVisionTestLayerNorm(values, weight, bias []float32, rows, dim int, epsilon float32) {
	for row := range rows {
		current := values[row*dim : (row+1)*dim]
		var mean float64
		for _, value := range current {
			mean += float64(value)
		}
		mean /= float64(dim)
		var variance float64
		for _, value := range current {
			delta := float64(value) - mean
			variance += delta * delta
		}
		inverse := 1 / math.Sqrt(variance/float64(dim)+float64(epsilon))
		for col := range current {
			current[col] = float32((float64(current[col])-mean)*inverse)*weight[col] + bias[col]
		}
	}
}

func hipUnifiedVisionTestMatMul(input, weight []float32, rows, inDim, outDim int, bias []float32) []float32 {
	out := make([]float32, rows*outDim)
	for row := range rows {
		for output := range outDim {
			var sum float32
			for col := range inDim {
				sum += input[row*inDim+col] * weight[output*inDim+col]
			}
			if len(bias) > 0 {
				sum += bias[output]
			}
			out[row*outDim+output] = sum
		}
	}
	return out
}

func hipUnifiedVisionTestRMS(values []float32, rows, dim int, epsilon float32) {
	for row := range rows {
		current := values[row*dim : (row+1)*dim]
		var squares float64
		for _, value := range current {
			squares += float64(value) * float64(value)
		}
		inverse := 1 / math.Sqrt(squares/float64(dim)+float64(epsilon))
		for col := range current {
			current[col] = float32(float64(current[col]) * inverse)
		}
	}
}
