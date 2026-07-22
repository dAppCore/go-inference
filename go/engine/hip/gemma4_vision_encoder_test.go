// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/quant/mlxaffine"
	"dappco.re/go/inference/model/vision"
)

func TestHIPVisionSDPA_Good(t *testing.T) {
	const tokens, heads, headDim = 3, 2, 2
	q := []float32{
		1, 0, 0, 1, 1, 1,
		0.5, -0.5, 1, 0, 0, 1,
	}
	k := []float32{
		1, 0, 0, 1, 1, 1,
		1, -1, 0.5, 0.5, -0.5, 1,
	}
	v := []float32{
		2, 0, 0, 4, 8, 8,
		1, 3, 5, 7, 9, 11,
	}
	got, err := hipVisionSDPA(nil, q, k, v, tokens, heads, heads, headDim, 1)
	core.RequireNoError(t, err)
	want := hipVisionTestAttention(q, k, v, tokens, heads, heads, headDim, 1)
	assertFloat32SlicesNear(t, want, got, 0.00001)
}

func TestHIPVision2DRoPEHeadMajor_Good(t *testing.T) {
	const tokens, heads, headDim, gridHeight, gridWidth = 4, 1, 4, 2, 2
	input := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}
	got := hipVision2DRoPEHeadMajor(input, tokens, heads, headDim, gridHeight, gridWidth, 100)
	want := hipVisionTestRoPE2D(input, tokens, heads, headDim, gridWidth, 100)
	assertFloat32SlicesNear(t, want, got, 0.00001)
}

func TestHIPVisionEncoderTowerProjectPatches_Good(t *testing.T) {
	const patchDim, hidden, heads, headDim, ffDim, textHidden = 3, 4, 1, 4, 8, 3
	ones := hipUnifiedVisionTestBF16([]float32{1, 1, 1, 1})
	zeros := func(count int) []byte { return hipUnifiedVisionTestBF16(make([]float32, count)) }
	patchWeight := []float32{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
		1, 1, 1,
	}
	projection := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
	}
	loaded := &vision.Loaded{
		PatchEmbedding: hipUnifiedVisionTestBF16(patchWeight),
		PostLayernorm:  ones,
		Layers: []vision.Layer{{
			InputNorm: ones, PostAttnNorm: ones, PreFFNorm: ones, PostFFNorm: ones,
			Q:     vision.Linear{Weight: zeros(hidden * hidden), OutDim: hidden, InDim: hidden},
			K:     vision.Linear{Weight: zeros(hidden * hidden), OutDim: hidden, InDim: hidden},
			V:     vision.Linear{Weight: zeros(hidden * hidden), OutDim: hidden, InDim: hidden},
			O:     vision.Linear{Weight: zeros(hidden * hidden), OutDim: hidden, InDim: hidden},
			QNorm: hipUnifiedVisionTestBF16([]float32{1, 1, 1, 1}),
			KNorm: hipUnifiedVisionTestBF16([]float32{1, 1, 1, 1}),
			Gate:  vision.Linear{Weight: zeros(ffDim * hidden)},
			Up:    vision.Linear{Weight: zeros(ffDim * hidden)},
			Down:  vision.Linear{Weight: zeros(hidden * ffDim)},
		}},
		Projector: vision.Projector{Projection: vision.Linear{
			Weight: hipUnifiedVisionTestBF16(projection), OutDim: textHidden, InDim: hidden,
		}},
		Cfg: vision.Config{
			Hidden: hidden, PatchDim: patchDim, NumLayers: 1,
			NumHeads: heads, NumKVHeads: heads, HeadDim: headDim,
			RMSNormEps: 1e-6, PoolKernel: 2, EmbeddingScale: 2,
		},
	}
	tower, err := newHIPVisionEncoderTowerFromLoaded(loaded, nil, nil, nil)
	core.RequireNoError(t, err)
	patches := []float32{
		0.1, 0.2, 0.3,
		0.4, 0.5, 0.6,
		0.7, 0.8, 0.9,
		1.0, 0.9, 0.8,
	}
	got, err := tower.ProjectPatches(patches, 2, 2)
	core.RequireNoError(t, err)

	scaled := make([]float32, len(patches))
	for index, value := range patches {
		scaled[index] = (value - 0.5) * 2
	}
	hiddenRows := hipUnifiedVisionTestMatMul(scaled, patchWeight, 4, patchDim, hidden, nil)
	hipVisionTestRMSRows(hiddenRows, []float32{1, 1, 1, 1}, 4, hidden, 1e-6)
	pooled := make([]float32, hidden)
	for row := range 4 {
		for col := range hidden {
			pooled[col] += hiddenRows[row*hidden+col] / 2
		}
	}
	hipVisionTestRMSRows(pooled, nil, 1, hidden, 1e-6)
	want := hipUnifiedVisionTestMatMul(pooled, projection, 1, hidden, textHidden, nil)
	assertFloat32SlicesNear(t, want, got, 0.0001)
}

func TestHIPVisionEncoderTowerQuantizedLinears_Good(t *testing.T) {
	const patchDim, hidden, ffDim, textHidden, groupSize, bits = 8, 8, 8, 4, 8, 4
	quantized := func(values []float32, outDim, inDim int) vision.Linear {
		packed, scales, biases, err := mlxaffine.QuantizeTensor(values, outDim, inDim, bits, groupSize)
		core.RequireNoError(t, err)
		return vision.Linear{
			Weight: packed, Scales: scales, Biases: biases,
			OutDim: outDim, InDim: inDim, GroupSize: groupSize, Bits: bits, Kind: mlxaffine.Mode,
		}
	}
	ones := hipUnifiedVisionTestBF16(hipUnifiedVisionTestValues(hidden, 0, 1))
	patchValues := hipUnifiedVisionTestValues(hidden*patchDim, 0.025, -0.5)
	hiddenValues := hipUnifiedVisionTestValues(hidden*hidden, 0.02, -0.4)
	ffValues := hipUnifiedVisionTestValues(ffDim*hidden, 0.015, -0.3)
	projectionValues := hipUnifiedVisionTestValues(textHidden*hidden, 0.03, -0.4)
	loaded := &vision.Loaded{
		PatchProjection: quantized(patchValues, hidden, patchDim),
		Layers: []vision.Layer{{
			InputNorm: ones, PostAttnNorm: ones, PreFFNorm: ones, PostFFNorm: ones,
			Q: quantized(hiddenValues, hidden, hidden), K: quantized(hiddenValues, hidden, hidden),
			V: quantized(hiddenValues, hidden, hidden), O: quantized(hiddenValues, hidden, hidden),
			QNorm: ones, KNorm: ones,
			Gate: quantized(ffValues, ffDim, hidden), Up: quantized(ffValues, ffDim, hidden),
			Down: quantized(hiddenValues, hidden, ffDim),
		}},
		Projector: vision.Projector{Projection: quantized(projectionValues, textHidden, hidden)},
		Cfg: vision.Config{
			Hidden: hidden, PatchDim: patchDim, NumLayers: 1,
			NumHeads: 1, NumKVHeads: 1, HeadDim: hidden,
			RMSNormEps: 1e-6, PoolKernel: 1, EmbeddingScale: 1,
		},
	}

	tower, err := newHIPVisionEncoderTowerFromLoaded(loaded, nil, nil, nil)
	core.RequireNoError(t, err)
	if len(tower.patchWeight) != hidden*patchDim {
		t.Fatalf("decoded patch values = %d, want %d", len(tower.patchWeight), hidden*patchDim)
	}
	features, err := tower.ProjectPatches(hipUnifiedVisionTestValues(patchDim, 0.05, 0.1), 1, 1)
	core.RequireNoError(t, err)
	if len(features) != textHidden {
		t.Fatalf("projected values = %d, want %d", len(features), textHidden)
	}
	for _, value := range features {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("quantized vision output contains non-finite value %v", value)
		}
	}
}

func TestHIPHardwareVisionEncoderProjectImage_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_HIP_VISION_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_HIP_VISION_TESTS=1 to run the Gemma 4 vision encoder smoke")
	}
	path := strings.TrimSpace(os.Getenv("GO_ROCM_ENCODER_VISION_MODEL_PATH"))
	if path == "" {
		t.Skip("GO_ROCM_ENCODER_VISION_MODEL_PATH is not set")
	}
	gemm := newSystemHIPAudioGEMM()
	if gemm == nil {
		t.Fatal("HIP rocBLAS GEMM is unavailable")
	}
	tower, err := loadHIPVisionEncoderTowerWithGEMM(path, gemm)
	core.RequireNoError(t, err)
	if tower == nil {
		t.Fatal("model has no Gemma 4 encoder vision payload")
	}
	defer tower.Close()
	config := *tower.imageConfig
	config.DoResize = false
	config.MaxSoftTokens = 1
	tower.imageConfig = &config

	img := image.NewRGBA(image.Rect(0, 0, 48, 48))
	for y := range 48 {
		for x := range 48 {
			img.SetRGBA(x, y, color.RGBA{R: uint8(x * 4), G: uint8(y * 4), B: 80, A: 255})
		}
	}
	var encoded bytes.Buffer
	core.RequireNoError(t, png.Encode(&encoded, img))
	features, softTokens, err := tower.ProjectImage(encoded.Bytes())
	core.RequireNoError(t, err)
	if softTokens != 1 || len(features) != tower.outputDim() {
		t.Fatalf("projected image = %d tokens, %d values, output dim %d", softTokens, len(features), tower.outputDim())
	}
	for _, value := range features {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("projected image contains non-finite value %v", value)
		}
	}
}

func hipVisionTestAttention(q, k, v []float32, tokens, heads, kvHeads, headDim int, scale float32) []float32 {
	out := make([]float32, heads*tokens*headDim)
	group := heads / kvHeads
	for head := range heads {
		kvHead := head / group
		for row := range tokens {
			scores := make([]float64, tokens)
			maxScore := math.Inf(-1)
			for col := range tokens {
				var score float64
				for dim := range headDim {
					score += float64(q[(head*tokens+row)*headDim+dim] * k[(kvHead*tokens+col)*headDim+dim])
				}
				scores[col] = score * float64(scale)
				maxScore = math.Max(maxScore, scores[col])
			}
			var denominator float64
			for col := range scores {
				scores[col] = math.Exp(scores[col] - maxScore)
				denominator += scores[col]
			}
			for dim := range headDim {
				var value float64
				for col := range tokens {
					value += scores[col] / denominator * float64(v[(kvHead*tokens+col)*headDim+dim])
				}
				out[(head*tokens+row)*headDim+dim] = float32(value)
			}
		}
	}
	return out
}

func hipVisionTestRoPE2D(input []float32, tokens, heads, headDim, gridWidth int, base float32) []float32 {
	out := make([]float32, len(input))
	part := 2 * (headDim / 4)
	half := part / 2
	for position := range tokens {
		coordinates := [2]float64{float64(position % gridWidth), float64(position / gridWidth)}
		for head := range heads {
			source := input[(position*heads+head)*headDim : (position*heads+head+1)*headDim]
			destination := out[(head*tokens+position)*headDim : (head*tokens+position+1)*headDim]
			for axis := range 2 {
				for dim := range part {
					angle := coordinates[axis] / math.Pow(float64(base), float64(2*(dim%half))/float64(part))
					rotated := source[axis*part+(dim+half)%part]
					if dim < half {
						rotated = -rotated
					}
					destination[axis*part+dim] = source[axis*part+dim]*float32(math.Cos(angle)) + rotated*float32(math.Sin(angle))
				}
			}
			copy(destination[2*part:], source[2*part:])
		}
	}
	return out
}

func hipVisionTestRMSRows(values, weight []float32, rows, dim int, epsilon float32) {
	for row := range rows {
		current := values[row*dim : (row+1)*dim]
		var squares float64
		for _, value := range current {
			squares += float64(value) * float64(value)
		}
		inverse := 1 / math.Sqrt(squares/float64(dim)+float64(epsilon))
		for col := range dim {
			current[col] = float32(float64(current[col]) * inverse)
			if len(weight) == dim {
				current[col] *= weight[col]
			}
		}
	}
}
