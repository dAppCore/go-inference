// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"encoding/binary"
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/gemma4"
	"dappco.re/go/inference/model/gemma4/audio"
	"dappco.re/go/inference/model/quant/mlxaffine"
	"dappco.re/go/inference/model/safetensors"
)

// UnifiedVisionTower owns Gemma 4's encoder-free image/video projection
// payload. Quantized linears are expanded once into system RAM; each image's
// two matrix products use the portable rocBLAS seam when available.
type UnifiedVisionTower struct {
	loaded      *model.LoadedUnifiedVision
	imageConfig *gemma4.Gemma4ImageFeatureConfig
	mapping     *safetensors.DirMapping
	gemm        audio.GEMM

	patchLN1Weight  []float32
	patchLN1Bias    []float32
	patchDense      []float32
	patchDenseBias  []float32
	patchLN2Weight  []float32
	patchLN2Bias    []float32
	positions       []float32
	posNormWeight   []float32
	posNormBias     []float32
	projection      []float32
	audioProjection []float32
}

// LoadUnifiedVisionTower loads a safetensors checkpoint carrying the
// gemma4_unified encoder-free payload. A non-unified checkpoint returns nil.
func LoadUnifiedVisionTower(dir string) (*UnifiedVisionTower, error) {
	return loadUnifiedVisionTowerWithGEMM(dir, newSystemHIPAudioGEMM())
}

func loadUnifiedVisionTowerWithGEMM(dir string, gemm audio.GEMM) (*UnifiedVisionTower, error) {
	loaded, mapping, err := model.Load(dir)
	if err != nil {
		return nil, core.E("hip.LoadUnifiedVisionTower", "load model", err)
	}
	if loaded == nil || loaded.UnifiedVision == nil {
		if mapping != nil {
			_ = mapping.Close()
		}
		return nil, nil
	}
	imageConfig, _, err := gemma4.LoadGemma4ImageFeatureConfigs(dir)
	if err != nil {
		if mapping != nil {
			_ = mapping.Close()
		}
		return nil, core.E("hip.LoadUnifiedVisionTower", "load image processor config", err)
	}
	if imageConfig == nil {
		cfg := loaded.UnifiedVision.Cfg
		imageConfig = &gemma4.Gemma4ImageFeatureConfig{
			PatchSize:         int32(cfg.PatchSize),
			MaxSoftTokens:     int32(cfg.MaxSoftTokens),
			PoolingKernelSize: int32(cfg.PoolKernel),
			RescaleFactor:     1.0 / 255.0,
		}
	}
	tower, err := newUnifiedVisionTowerFromLoaded(loaded.UnifiedVision, mapping, gemm, imageConfig)
	if err != nil {
		if mapping != nil {
			_ = mapping.Close()
		}
		return nil, err
	}
	return tower, nil
}

func newUnifiedVisionTowerFromLoaded(loaded *model.LoadedUnifiedVision, mapping *safetensors.DirMapping, gemm audio.GEMM, imageConfig *gemma4.Gemma4ImageFeatureConfig) (*UnifiedVisionTower, error) {
	if loaded == nil {
		return nil, core.NewError("hip.UnifiedVisionTower: loaded payload is nil")
	}
	cfg := loaded.Cfg
	patchDim := cfg.ModelPatchSize * cfg.ModelPatchSize * 3
	if patchDim <= 0 || cfg.MMEmbedDim <= 0 || cfg.TextHidden <= 0 || cfg.PosembSize <= 0 {
		return nil, core.NewError("hip.UnifiedVisionTower: invalid unified vision geometry")
	}
	tower := &UnifiedVisionTower{loaded: loaded, mapping: mapping, gemm: gemm, imageConfig: hipNormalizeVisionImageFeatureConfig(imageConfig)}
	var err error
	if tower.patchLN1Weight, err = hipUnifiedVisionBF16Vector(loaded.PatchLN1W, patchDim, "patch_ln1 weight"); err != nil {
		return nil, err
	}
	if tower.patchLN1Bias, err = hipUnifiedVisionBF16Vector(loaded.PatchLN1B, patchDim, "patch_ln1 bias"); err != nil {
		return nil, err
	}
	if tower.patchDense, err = hipUnifiedVisionLinearWeights(loaded.PatchDense); err != nil {
		return nil, core.E("hip.UnifiedVisionTower", "load patch_dense", err)
	}
	if tower.patchDenseBias, err = hipUnifiedVisionOptionalBF16Vector(loaded.PatchDense.Bias, cfg.MMEmbedDim, "patch_dense bias"); err != nil {
		return nil, err
	}
	if tower.patchLN2Weight, err = hipUnifiedVisionBF16Vector(loaded.PatchLN2W, cfg.MMEmbedDim, "patch_ln2 weight"); err != nil {
		return nil, err
	}
	if tower.patchLN2Bias, err = hipUnifiedVisionBF16Vector(loaded.PatchLN2B, cfg.MMEmbedDim, "patch_ln2 bias"); err != nil {
		return nil, err
	}
	if tower.positions, err = hipUnifiedVisionBF16Vector(loaded.PosEmbedding, cfg.PosembSize*2*cfg.MMEmbedDim, "position embedding"); err != nil {
		return nil, err
	}
	if tower.posNormWeight, err = hipUnifiedVisionBF16Vector(loaded.PosNormW, cfg.MMEmbedDim, "position norm weight"); err != nil {
		return nil, err
	}
	if tower.posNormBias, err = hipUnifiedVisionBF16Vector(loaded.PosNormB, cfg.MMEmbedDim, "position norm bias"); err != nil {
		return nil, err
	}
	if tower.projection, err = hipUnifiedVisionLinearWeights(loaded.Projection); err != nil {
		return nil, core.E("hip.UnifiedVisionTower", "load embedding projection", err)
	}
	if loaded.AudioProjection.Weight != nil {
		tower.audioProjection, err = hipUnifiedVisionLinearWeights(loaded.AudioProjection)
		if err != nil {
			return nil, core.E("hip.UnifiedVisionTower", "load audio projection", err)
		}
	}
	return tower, nil
}

func hipUnifiedVisionLinearWeights(linear model.LoadedVisionLinear) ([]float32, error) {
	if linear.OutDim <= 0 || linear.InDim <= 0 || len(linear.Weight) == 0 {
		return nil, core.NewError("hip.UnifiedVisionLinear: invalid linear geometry")
	}
	if len(linear.Scales) == 0 && len(linear.Biases) == 0 {
		return hipUnifiedVisionBF16Vector(linear.Weight, linear.OutDim*linear.InDim, "linear weight")
	}
	if linear.Kind != mlxaffine.Mode || linear.Bits <= 0 || linear.GroupSize <= 0 || len(linear.Scales) == 0 || len(linear.Biases) == 0 {
		return nil, core.NewError("hip.UnifiedVisionLinear: complete MLX affine metadata is required")
	}
	weights, err := mlxaffine.DequantizeTensor(linear.Weight, linear.Scales, linear.Biases, linear.OutDim, linear.InDim, linear.Bits, linear.GroupSize)
	if err != nil {
		return nil, core.E("hip.UnifiedVisionLinear", "dequantise weight", err)
	}
	return weights, nil
}

func hipUnifiedVisionBF16Vector(payload []byte, count int, label string) ([]float32, error) {
	if count <= 0 || len(payload) != count*2 {
		return nil, core.E("hip.UnifiedVisionTower", label+" has invalid BF16 geometry", nil)
	}
	out := make([]float32, count)
	for index := range out {
		out[index] = hipBFloat16ToFloat32(binary.LittleEndian.Uint16(payload[index*2:]))
	}
	return out, nil
}

func hipUnifiedVisionOptionalBF16Vector(payload []byte, count int, label string) ([]float32, error) {
	if len(payload) == 0 {
		return nil, nil
	}
	return hipUnifiedVisionBF16Vector(payload, count, label)
}

// ProjectImage preprocesses and projects one PNG/JPEG into text-width float32
// soft-token rows.
func (tower *UnifiedVisionTower) ProjectImage(payload []byte) ([]float32, int, error) {
	if tower == nil || tower.loaded == nil {
		return nil, 0, core.NewError("hip.UnifiedVisionTower.ProjectImage: tower is not loaded")
	}
	patches, positions, rows, err := hipUnifiedVisionImagePatches(payload, tower.imageConfig)
	if err != nil {
		return nil, 0, core.E("hip.UnifiedVisionTower.ProjectImage", "preprocess image", err)
	}
	features, err := tower.ProjectPatches(patches, positions, rows)
	if err != nil {
		return nil, 0, err
	}
	return features, rows, nil
}

// ProjectPatches executes LayerNorm -> dense -> LayerNorm -> position add ->
// LayerNorm -> scale-free RMSNorm -> text projection.
func (tower *UnifiedVisionTower) ProjectPatches(patches []float32, positions []int32, rows int) ([]float32, error) {
	if tower == nil || tower.loaded == nil {
		return nil, core.NewError("hip.UnifiedVisionTower.ProjectPatches: tower is not loaded")
	}
	cfg := tower.loaded.Cfg
	patchDim := cfg.ModelPatchSize * cfg.ModelPatchSize * 3
	if rows <= 0 || len(patches) != rows*patchDim || len(positions) < rows*2 {
		return nil, core.NewError("hip.UnifiedVisionTower.ProjectPatches: invalid patch geometry")
	}
	hidden := append([]float32(nil), patches...)
	if err := hipUnifiedVisionLayerNorm(hidden, tower.patchLN1Weight, tower.patchLN1Bias, rows, patchDim, cfg.LayerNormEps); err != nil {
		return nil, core.E("hip.UnifiedVisionTower.ProjectPatches", "patch_ln1", err)
	}
	hidden = hipUnifiedVisionMatMulNT(tower.gemm, hidden, tower.patchDense, rows, patchDim, cfg.MMEmbedDim)
	hipUnifiedVisionAddBias(hidden, tower.patchDenseBias, rows, cfg.MMEmbedDim)
	if err := hipUnifiedVisionLayerNorm(hidden, tower.patchLN2Weight, tower.patchLN2Bias, rows, cfg.MMEmbedDim, cfg.LayerNormEps); err != nil {
		return nil, core.E("hip.UnifiedVisionTower.ProjectPatches", "patch_ln2", err)
	}
	if err := tower.addPositions(hidden, positions, rows); err != nil {
		return nil, err
	}
	if err := hipUnifiedVisionLayerNorm(hidden, tower.posNormWeight, tower.posNormBias, rows, cfg.MMEmbedDim, cfg.LayerNormEps); err != nil {
		return nil, core.E("hip.UnifiedVisionTower.ProjectPatches", "position norm", err)
	}
	if err := hipUnifiedVisionRMSNoScale(hidden, rows, cfg.MMEmbedDim, cfg.RMSNormEps); err != nil {
		return nil, err
	}
	return hipUnifiedVisionMatMulNT(tower.gemm, hidden, tower.projection, rows, cfg.MMEmbedDim, cfg.TextHidden), nil
}

func (tower *UnifiedVisionTower) addPositions(hidden []float32, positions []int32, rows int) error {
	cfg := tower.loaded.Cfg
	if len(hidden) != rows*cfg.MMEmbedDim || len(tower.positions) != cfg.PosembSize*2*cfg.MMEmbedDim {
		return core.NewError("hip.UnifiedVisionTower.ProjectPatches: invalid position geometry")
	}
	for row := range rows {
		for axis := range 2 {
			index := int(positions[row*2+axis])
			if index < 0 {
				continue
			}
			index = min(index, cfg.PosembSize-1)
			table := tower.positions[(index*2+axis)*cfg.MMEmbedDim:]
			for col := range cfg.MMEmbedDim {
				hidden[row*cfg.MMEmbedDim+col] += table[col]
			}
		}
	}
	return nil
}

func hipUnifiedVisionLayerNorm(values, weight, bias []float32, rows, dim int, epsilon float32) error {
	if rows <= 0 || dim <= 0 || len(values) != rows*dim || len(weight) != dim || len(bias) != dim || epsilon < 0 || math.IsNaN(float64(epsilon)) || math.IsInf(float64(epsilon), 0) {
		return core.NewError("hip.UnifiedVisionLayerNorm: invalid geometry or epsilon")
	}
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
	return nil
}

func hipUnifiedVisionRMSNoScale(values []float32, rows, dim int, epsilon float32) error {
	if rows <= 0 || dim <= 0 || len(values) != rows*dim || epsilon < 0 || math.IsNaN(float64(epsilon)) || math.IsInf(float64(epsilon), 0) {
		return core.NewError("hip.UnifiedVisionRMS: invalid geometry or epsilon")
	}
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
	return nil
}

func hipUnifiedVisionMatMulNT(gemm audio.GEMM, input, weight []float32, rows, inDim, outDim int) []float32 {
	if gemm != nil {
		if output, ok := gemm.MatMul(input, weight, rows, inDim, outDim, true); ok && len(output) == rows*outDim {
			return output
		}
	}
	output := make([]float32, rows*outDim)
	for row := range rows {
		for out := range outDim {
			var sum float32
			for col := range inDim {
				sum += input[row*inDim+col] * weight[out*inDim+col]
			}
			output[row*outDim+out] = sum
		}
	}
	return output
}

func hipUnifiedVisionAddBias(values, bias []float32, rows, dim int) {
	if len(bias) != dim {
		return
	}
	for row := range rows {
		for col := range dim {
			values[row*dim+col] += bias[col]
		}
	}
}

func (tower *UnifiedVisionTower) AcceptsAudio() bool {
	return tower != nil && tower.loaded != nil && tower.loaded.Cfg.AudioSamplesPerToken > 0 && len(tower.audioProjection) > 0
}

func (tower *UnifiedVisionTower) ProjectAudioSamples(samples []float32) ([]float32, int, error) {
	if !tower.AcceptsAudio() || len(samples) == 0 {
		return nil, 0, core.NewError("hip.UnifiedVisionTower.ProjectAudio: unified audio head is unavailable or input is empty")
	}
	cfg := tower.loaded.Cfg
	samplesPerToken := cfg.AudioSamplesPerToken
	rows := (len(samples) + samplesPerToken - 1) / samplesPerToken
	input := make([]float32, rows*samplesPerToken)
	copy(input, samples)
	if err := hipUnifiedVisionRMSNoScale(input, rows, samplesPerToken, cfg.RMSNormEps); err != nil {
		return nil, 0, err
	}
	return hipUnifiedVisionMatMulNT(tower.gemm, input, tower.audioProjection, rows, samplesPerToken, cfg.TextHidden), rows, nil
}

func (tower *UnifiedVisionTower) Close() error {
	if tower == nil || tower.mapping == nil {
		return nil
	}
	mapping := tower.mapping
	tower.mapping = nil
	return mapping.Close()
}
