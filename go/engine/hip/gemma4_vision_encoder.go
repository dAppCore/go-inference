// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/gemma4"
	"dappco.re/go/inference/model/gemma4/audio"
	"dappco.re/go/inference/model/safetensors"
)

type hipVisionLinear struct {
	weight []float32
	bias   []float32
	inDim  int
	outDim int
}

type hipVisionEncoderLayer struct {
	inputNorm    []float32
	postAttnNorm []float32
	preFFNorm    []float32
	postFFNorm   []float32
	q            hipVisionLinear
	k            hipVisionLinear
	v            hipVisionLinear
	o            hipVisionLinear
	qNorm        []float32
	kNorm        []float32
	gate         hipVisionLinear
	up           hipVisionLinear
	down         hipVisionLinear
}

// HIPVisionEncoderTower owns Gemma 4's encoder-based SigLIP payload. Dense and
// MLX-affine weights are expanded once into system RAM; matrix products use the
// portable HIP rocBLAS seam when present and the same float32 layout on CPU.
type HIPVisionEncoderTower struct {
	loaded      *model.LoadedVision
	imageConfig *gemma4.Gemma4ImageFeatureConfig
	mapping     *safetensors.DirMapping
	gemm        audio.GEMM

	patchWeight []float32
	patchBias   []float32
	positions   []float32
	postNorm    []float32
	stdBias     []float32
	stdScale    []float32
	layers      []hipVisionEncoderLayer
	projection  hipVisionLinear
	linear1     hipVisionLinear
	linear2     hipVisionLinear
}

func LoadHIPVisionEncoderTower(dir string) (*HIPVisionEncoderTower, error) {
	return loadHIPVisionEncoderTowerWithGEMM(dir, newSystemHIPAudioGEMM())
}

func loadHIPVisionEncoderTowerWithGEMM(dir string, gemm audio.GEMM) (*HIPVisionEncoderTower, error) {
	loaded, mapping, err := model.Load(dir)
	if err != nil {
		return nil, core.E("hip.LoadVisionEncoderTower", "load model", err)
	}
	if loaded == nil || loaded.Vision == nil {
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
		return nil, core.E("hip.LoadVisionEncoderTower", "load image processor config", err)
	}
	if imageConfig == nil {
		cfg := loaded.Vision.Cfg
		imageConfig = &gemma4.Gemma4ImageFeatureConfig{
			PatchSize:         int32(cfg.PatchSize),
			MaxSoftTokens:     280,
			PoolingKernelSize: int32(cfg.PoolKernel),
			RescaleFactor:     1.0 / 255.0,
		}
	}
	tower, err := newHIPVisionEncoderTowerFromLoaded(loaded.Vision, mapping, gemm, imageConfig)
	if err != nil {
		if mapping != nil {
			_ = mapping.Close()
		}
		return nil, err
	}
	return tower, nil
}

func newHIPVisionEncoderTowerFromLoaded(loaded *model.LoadedVision, mapping *safetensors.DirMapping, gemm audio.GEMM, imageConfig *gemma4.Gemma4ImageFeatureConfig) (*HIPVisionEncoderTower, error) {
	if loaded == nil {
		return nil, core.NewError("hip.VisionEncoderTower: loaded payload is nil")
	}
	cfg := loaded.Cfg
	if cfg.Hidden <= 0 || cfg.PatchDim <= 0 || cfg.NumHeads <= 0 || cfg.NumKVHeads <= 0 ||
		cfg.HeadDim <= 0 || cfg.NumHeads%cfg.NumKVHeads != 0 || cfg.NumHeads*cfg.HeadDim <= 0 ||
		len(loaded.Layers) != cfg.NumLayers {
		return nil, core.NewError("hip.VisionEncoderTower: invalid encoder geometry")
	}
	tower := &HIPVisionEncoderTower{
		loaded: loaded, mapping: mapping, gemm: gemm,
		imageConfig: hipNormalizeVisionImageFeatureConfig(imageConfig),
		layers:      make([]hipVisionEncoderLayer, len(loaded.Layers)),
	}
	var err error
	patch := loaded.PatchProjection
	if len(patch.Weight) == 0 {
		payload := loaded.PatchConvWeight
		if len(payload) == 0 {
			payload = loaded.PatchEmbedding
		}
		patch = model.LoadedVisionLinear{Weight: payload, OutDim: cfg.Hidden, InDim: cfg.PatchDim}
	}
	decodedPatch, err := hipVisionDecodeLinear(patch, cfg.Hidden, cfg.PatchDim, "vision patch projection")
	if err != nil {
		return nil, err
	}
	tower.patchWeight = decodedPatch.weight
	tower.patchBias = decodedPatch.bias
	if tower.positions, err = hipVisionOptionalBF16Values(loaded.PositionEmbeddings, "vision positions"); err != nil {
		return nil, err
	}
	if tower.postNorm, err = hipVisionOptionalBF16Vector(loaded.PostLayernorm, cfg.Hidden, "vision post norm"); err != nil {
		return nil, err
	}
	if tower.stdBias, err = hipVisionOptionalBF16Vector(loaded.StdBias, cfg.Hidden, "vision standardize bias"); err != nil {
		return nil, err
	}
	if tower.stdScale, err = hipVisionOptionalBF16Vector(loaded.StdScale, cfg.Hidden, "vision standardize scale"); err != nil {
		return nil, err
	}
	qDim := cfg.NumHeads * cfg.HeadDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	for index := range loaded.Layers {
		source := loaded.Layers[index]
		layer := &tower.layers[index]
		if layer.inputNorm, err = hipUnifiedVisionBF16Vector(source.InputNorm, cfg.Hidden, "vision input norm"); err != nil {
			return nil, err
		}
		if layer.postAttnNorm, err = hipUnifiedVisionBF16Vector(source.PostAttnNorm, cfg.Hidden, "vision post-attention norm"); err != nil {
			return nil, err
		}
		if layer.preFFNorm, err = hipUnifiedVisionBF16Vector(source.PreFFNorm, cfg.Hidden, "vision pre-ff norm"); err != nil {
			return nil, err
		}
		if layer.postFFNorm, err = hipUnifiedVisionBF16Vector(source.PostFFNorm, cfg.Hidden, "vision post-ff norm"); err != nil {
			return nil, err
		}
		if layer.q, err = hipVisionDecodeLinear(source.Q, qDim, cfg.Hidden, "q projection"); err != nil {
			return nil, err
		}
		if layer.k, err = hipVisionDecodeLinear(source.K, kvDim, cfg.Hidden, "k projection"); err != nil {
			return nil, err
		}
		if layer.v, err = hipVisionDecodeLinear(source.V, kvDim, cfg.Hidden, "v projection"); err != nil {
			return nil, err
		}
		if layer.o, err = hipVisionDecodeLinear(source.O, cfg.Hidden, qDim, "o projection"); err != nil {
			return nil, err
		}
		if layer.qNorm, err = hipUnifiedVisionBF16Vector(source.QNorm, cfg.HeadDim, "vision q norm"); err != nil {
			return nil, err
		}
		if layer.kNorm, err = hipUnifiedVisionBF16Vector(source.KNorm, cfg.HeadDim, "vision k norm"); err != nil {
			return nil, err
		}
		if layer.gate, err = hipVisionDecodeLinear(source.Gate, source.Gate.OutDim, cfg.Hidden, "gate projection"); err != nil {
			return nil, err
		}
		if layer.up, err = hipVisionDecodeLinear(source.Up, layer.gate.outDim, cfg.Hidden, "up projection"); err != nil {
			return nil, err
		}
		if layer.down, err = hipVisionDecodeLinear(source.Down, cfg.Hidden, layer.gate.outDim, "down projection"); err != nil {
			return nil, err
		}
	}
	if len(loaded.Projector.Projection.Weight) > 0 {
		if tower.projection, err = hipVisionDecodeLinear(loaded.Projector.Projection, loaded.Projector.Projection.OutDim, cfg.Hidden, "vision projection"); err != nil {
			return nil, err
		}
	} else if len(loaded.Projector.Linear1.Weight) > 0 && len(loaded.Projector.Linear2.Weight) > 0 {
		if tower.linear1, err = hipVisionDecodeLinear(loaded.Projector.Linear1, loaded.Projector.Linear1.OutDim, cfg.Hidden, "vision linear1"); err != nil {
			return nil, err
		}
		if tower.linear2, err = hipVisionDecodeLinear(loaded.Projector.Linear2, loaded.Projector.Linear2.OutDim, tower.linear1.outDim, "vision linear2"); err != nil {
			return nil, err
		}
	} else {
		return nil, core.NewError("hip.VisionEncoderTower: projector is missing")
	}
	return tower, nil
}

func hipVisionDecodeLinear(linear model.LoadedVisionLinear, fallbackOut, fallbackIn int, label string) (hipVisionLinear, error) {
	if linear.InDim <= 0 {
		linear.InDim = fallbackIn
	}
	if linear.OutDim <= 0 {
		linear.OutDim = fallbackOut
	}
	if linear.OutDim <= 0 && linear.InDim > 0 && len(linear.Scales) == 0 && len(linear.Weight)%(linear.InDim*2) == 0 {
		linear.OutDim = len(linear.Weight) / (linear.InDim * 2)
	}
	if linear.OutDim <= 0 || linear.InDim <= 0 {
		return hipVisionLinear{}, core.E("hip.VisionEncoderTower", label+" has invalid geometry", nil)
	}
	weight, err := hipUnifiedVisionLinearWeights(linear)
	if err != nil {
		return hipVisionLinear{}, core.E("hip.VisionEncoderTower", "load "+label, err)
	}
	bias, err := hipUnifiedVisionOptionalBF16Vector(linear.Bias, linear.OutDim, label+" bias")
	if err != nil {
		return hipVisionLinear{}, err
	}
	return hipVisionLinear{weight: weight, bias: bias, inDim: linear.InDim, outDim: linear.OutDim}, nil
}

func hipVisionOptionalBF16Vector(payload []byte, count int, label string) ([]float32, error) {
	if len(payload) == 0 {
		return nil, nil
	}
	return hipUnifiedVisionBF16Vector(payload, count, label)
}

func hipVisionOptionalBF16Values(payload []byte, label string) ([]float32, error) {
	if len(payload) == 0 {
		return nil, nil
	}
	if len(payload)%2 != 0 {
		return nil, core.E("hip.VisionEncoderTower", label+" has invalid BF16 geometry", nil)
	}
	return hipUnifiedVisionBF16Vector(payload, len(payload)/2, label)
}

func (tower *HIPVisionEncoderTower) ProjectImage(payload []byte) ([]float32, int, error) {
	if tower == nil || tower.loaded == nil {
		return nil, 0, core.NewError("hip.VisionEncoderTower.ProjectImage: tower is not loaded")
	}
	pixels, height, width, softTokens, err := hipVisionImagePixels(payload, tower.imageConfig)
	if err != nil {
		return nil, 0, core.E("hip.VisionEncoderTower.ProjectImage", "preprocess image", err)
	}
	patches, gridHeight, gridWidth, err := hipVisionPatchifyPixels(pixels, int(height), int(width), tower.loaded.Cfg.PatchSize, tower.loaded.Cfg.NumChannels)
	if err != nil {
		return nil, 0, err
	}
	features, err := tower.ProjectPatches(patches, gridHeight, gridWidth)
	if err != nil {
		return nil, 0, err
	}
	outputDim := tower.outputDim()
	if outputDim <= 0 || len(features)%outputDim != 0 || len(features)/outputDim != softTokens {
		return nil, 0, core.NewError("hip.VisionEncoderTower.ProjectImage: projected soft-token geometry mismatch")
	}
	return features, softTokens, nil
}

func hipVisionPatchifyPixels(pixels []float32, height, width, patch, channels int) ([]float32, int, int, error) {
	if channels <= 0 {
		channels = 3
	}
	if height <= 0 || width <= 0 || patch <= 0 || height%patch != 0 || width%patch != 0 || len(pixels) != height*width*channels {
		return nil, 0, 0, core.NewError("hip.VisionEncoderTower: invalid image patch geometry")
	}
	gridHeight, gridWidth := height/patch, width/patch
	patchDim := patch * patch * channels
	patches := make([]float32, gridHeight*gridWidth*patchDim)
	row := 0
	for gridRow := range gridHeight {
		for gridCol := range gridWidth {
			column := 0
			for patchRow := range patch {
				for patchCol := range patch {
					source := (((gridRow*patch + patchRow) * width) + (gridCol*patch + patchCol)) * channels
					copy(patches[row*patchDim+column:row*patchDim+column+channels], pixels[source:source+channels])
					column += channels
				}
			}
			row++
		}
	}
	return patches, gridHeight, gridWidth, nil
}

func (tower *HIPVisionEncoderTower) ProjectPatches(patches []float32, gridHeight, gridWidth int) ([]float32, error) {
	if tower == nil || tower.loaded == nil {
		return nil, core.NewError("hip.VisionEncoderTower.ProjectPatches: tower is not loaded")
	}
	cfg := tower.loaded.Cfg
	rows := gridHeight * gridWidth
	if rows <= 0 || len(patches) != rows*cfg.PatchDim {
		return nil, core.NewError("hip.VisionEncoderTower.ProjectPatches: invalid patch geometry")
	}
	scaled := make([]float32, len(patches))
	for index, value := range patches {
		scaled[index] = (value - 0.5) * 2
	}
	hidden := hipVisionMatMul(tower.gemm, scaled, tower.patchWeight, rows, cfg.PatchDim, cfg.Hidden, true)
	hipUnifiedVisionAddBias(hidden, tower.patchBias, rows, cfg.Hidden)
	if err := tower.addPositions(hidden, rows, gridHeight, gridWidth); err != nil {
		return nil, err
	}
	var err error
	for index := range tower.layers {
		hidden, err = tower.forwardLayer(hidden, &tower.layers[index], rows, gridHeight, gridWidth)
		if err != nil {
			return nil, core.E("hip.VisionEncoderTower.ProjectPatches", core.Sprintf("encoder layer %d", index), err)
		}
	}
	if len(tower.postNorm) > 0 {
		if err := hipVisionRMSRows(hidden, tower.postNorm, rows, cfg.Hidden, cfg.RMSNormEps); err != nil {
			return nil, err
		}
	}
	hidden, rows = hipVisionPool(hidden, rows, gridHeight, gridWidth, cfg.Hidden, cfg.PoolKernel, cfg.EmbeddingScale)
	if len(tower.stdBias) > 0 && len(tower.stdScale) > 0 {
		for row := range rows {
			for col := range cfg.Hidden {
				hidden[row*cfg.Hidden+col] = (hidden[row*cfg.Hidden+col] - tower.stdBias[col]) * tower.stdScale[col]
			}
		}
	}
	if err := hipVisionRMSRows(hidden, nil, rows, cfg.Hidden, cfg.RMSNormEps); err != nil {
		return nil, err
	}
	if tower.projection.outDim > 0 {
		return tower.projection.forward(tower.gemm, hidden, rows), nil
	}
	hidden = tower.linear1.forward(tower.gemm, hidden, rows)
	for index := range hidden {
		hidden[index] = hipVisionGELUTanh(hidden[index])
	}
	return tower.linear2.forward(tower.gemm, hidden, rows), nil
}

func (tower *HIPVisionEncoderTower) addPositions(hidden []float32, rows, gridHeight, gridWidth int) error {
	if len(tower.positions) == 0 {
		return nil
	}
	cfg := tower.loaded.Cfg
	if len(hidden) != rows*cfg.Hidden {
		return core.NewError("hip.VisionEncoderTower: invalid hidden geometry")
	}
	slots := cfg.PositionEmbeddingSize
	if slots > 0 && len(tower.positions) == 2*slots*cfg.Hidden {
		if gridHeight > slots || gridWidth > slots {
			return core.NewError("hip.VisionEncoderTower: split position table is shorter than image grid")
		}
		for y := range gridHeight {
			for x := range gridWidth {
				row := y*gridWidth + x
				for col := range cfg.Hidden {
					hidden[row*cfg.Hidden+col] += tower.positions[x*cfg.Hidden+col] + tower.positions[(slots+y)*cfg.Hidden+col]
				}
			}
		}
		return nil
	}
	if len(tower.positions) < rows*cfg.Hidden {
		return core.NewError("hip.VisionEncoderTower: position table is shorter than image grid")
	}
	for index := range rows * cfg.Hidden {
		hidden[index] += tower.positions[index]
	}
	return nil
}

func (tower *HIPVisionEncoderTower) forwardLayer(hidden []float32, layer *hipVisionEncoderLayer, rows, gridHeight, gridWidth int) ([]float32, error) {
	cfg := tower.loaded.Cfg
	normed := append([]float32(nil), hidden...)
	if err := hipVisionRMSRows(normed, layer.inputNorm, rows, cfg.Hidden, cfg.RMSNormEps); err != nil {
		return nil, err
	}
	q := layer.q.forward(tower.gemm, normed, rows)
	k := layer.k.forward(tower.gemm, normed, rows)
	v := layer.v.forward(tower.gemm, normed, rows)
	q = hipVisionQKHeadMajor(q, layer.qNorm, rows, cfg.NumHeads, cfg.HeadDim, gridHeight, gridWidth, cfg.RopeBase, cfg.RMSNormEps)
	k = hipVisionQKHeadMajor(k, layer.kNorm, rows, cfg.NumKVHeads, cfg.HeadDim, gridHeight, gridWidth, cfg.RopeBase, cfg.RMSNormEps)
	v = hipVisionValueHeadMajor(v, rows, cfg.NumKVHeads, cfg.HeadDim, cfg.RMSNormEps)
	attention, err := hipVisionSDPA(tower.gemm, q, k, v, rows, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim, 1)
	if err != nil {
		return nil, err
	}
	tokenMajor := hipVisionHeadToTokenMajor(attention, rows, cfg.NumHeads, cfg.HeadDim)
	attention = layer.o.forward(tower.gemm, tokenMajor, rows)
	if err := hipVisionRMSRows(attention, layer.postAttnNorm, rows, cfg.Hidden, cfg.RMSNormEps); err != nil {
		return nil, err
	}
	for index := range hidden {
		hidden[index] += attention[index]
	}
	ffInput := append([]float32(nil), hidden...)
	if err := hipVisionRMSRows(ffInput, layer.preFFNorm, rows, cfg.Hidden, cfg.RMSNormEps); err != nil {
		return nil, err
	}
	gate := layer.gate.forward(tower.gemm, ffInput, rows)
	up := layer.up.forward(tower.gemm, ffInput, rows)
	for index := range gate {
		gate[index] = hipVisionGELUTanh(gate[index]) * up[index]
	}
	ff := layer.down.forward(tower.gemm, gate, rows)
	if err := hipVisionRMSRows(ff, layer.postFFNorm, rows, cfg.Hidden, cfg.RMSNormEps); err != nil {
		return nil, err
	}
	for index := range hidden {
		hidden[index] += ff[index]
	}
	return hidden, nil
}

func (linear hipVisionLinear) forward(gemm audio.GEMM, input []float32, rows int) []float32 {
	output := hipVisionMatMul(gemm, input, linear.weight, rows, linear.inDim, linear.outDim, true)
	hipUnifiedVisionAddBias(output, linear.bias, rows, linear.outDim)
	return output
}

func hipVisionMatMul(gemm audio.GEMM, left, right []float32, rows, inner, columns int, transposeRight bool) []float32 {
	if gemm != nil {
		if output, ok := gemm.MatMul(left, right, rows, inner, columns, transposeRight); ok && len(output) == rows*columns {
			return output
		}
	}
	output := make([]float32, rows*columns)
	for row := range rows {
		for column := range columns {
			var sum float32
			for index := range inner {
				rightIndex := index*columns + column
				if transposeRight {
					rightIndex = column*inner + index
				}
				sum += left[row*inner+index] * right[rightIndex]
			}
			output[row*columns+column] = sum
		}
	}
	return output
}

func hipVisionRMSRows(values, weight []float32, rows, dim int, epsilon float32) error {
	if rows <= 0 || dim <= 0 || len(values) != rows*dim || (len(weight) != 0 && len(weight) != dim) || epsilon < 0 {
		return core.NewError("hip.VisionEncoderTower: invalid RMSNorm geometry")
	}
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
	return nil
}

func hipVisionQKHeadMajor(values, weight []float32, rows, heads, headDim, gridHeight, gridWidth int, base, epsilon float32) []float32 {
	for index := range rows * heads {
		_ = hipVisionRMSRows(values[index*headDim:(index+1)*headDim], weight, 1, headDim, epsilon)
	}
	return hipVision2DRoPEHeadMajor(values, rows, heads, headDim, gridHeight, gridWidth, base)
}

func hipVisionValueHeadMajor(values []float32, rows, heads, headDim int, epsilon float32) []float32 {
	for index := range rows * heads {
		_ = hipVisionRMSRows(values[index*headDim:(index+1)*headDim], nil, 1, headDim, epsilon)
	}
	out := make([]float32, len(values))
	for position := range rows {
		for head := range heads {
			copy(out[(head*rows+position)*headDim:(head*rows+position+1)*headDim], values[(position*heads+head)*headDim:(position*heads+head+1)*headDim])
		}
	}
	return out
}

func hipVision2DRoPEHeadMajor(values []float32, rows, heads, headDim, gridHeight, gridWidth int, base float32) []float32 {
	out := make([]float32, len(values))
	part := 2 * (headDim / 4)
	doRoPE := base != 0 && part >= 2 && gridHeight*gridWidth == rows
	for position := range rows {
		coordinates := [2]float64{float64(position % gridWidth), float64(position / gridWidth)}
		for head := range heads {
			source := values[(position*heads+head)*headDim : (position*heads+head+1)*headDim]
			destination := out[(head*rows+position)*headDim : (head*rows+position+1)*headDim]
			if !doRoPE {
				copy(destination, source)
				continue
			}
			half := part / 2
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

func hipVisionSDPA(gemm audio.GEMM, q, k, v []float32, rows, heads, kvHeads, headDim int, scale float32) ([]float32, error) {
	if rows <= 0 || kvHeads <= 0 || heads%kvHeads != 0 || len(q) != heads*rows*headDim || len(k) != kvHeads*rows*headDim || len(v) != len(k) {
		return nil, core.NewError("hip.VisionSDPA: invalid attention geometry")
	}
	group := heads / kvHeads
	output := make([]float32, len(q))
	for head := range heads {
		kvHead := head / group
		query := q[head*rows*headDim : (head+1)*rows*headDim]
		key := k[kvHead*rows*headDim : (kvHead+1)*rows*headDim]
		value := v[kvHead*rows*headDim : (kvHead+1)*rows*headDim]
		scores := hipVisionMatMul(gemm, query, key, rows, headDim, rows, true)
		for row := range rows {
			current := scores[row*rows : (row+1)*rows]
			maxValue := float32(-math.MaxFloat32)
			for index := range current {
				current[index] *= scale
				maxValue = max(maxValue, current[index])
			}
			var denominator float64
			for index := range current {
				current[index] = float32(math.Exp(float64(current[index] - maxValue)))
				denominator += float64(current[index])
			}
			for index := range current {
				current[index] /= float32(denominator)
			}
		}
		result := hipVisionMatMul(gemm, scores, value, rows, rows, headDim, false)
		copy(output[head*rows*headDim:(head+1)*rows*headDim], result)
	}
	return output, nil
}

func hipVisionHeadToTokenMajor(values []float32, rows, heads, headDim int) []float32 {
	out := make([]float32, len(values))
	for head := range heads {
		for position := range rows {
			copy(out[(position*heads+head)*headDim:(position*heads+head+1)*headDim], values[(head*rows+position)*headDim:(head*rows+position+1)*headDim])
		}
	}
	return out
}

func hipVisionPool(values []float32, rows, gridHeight, gridWidth, hidden, kernel int, scale float32) ([]float32, int) {
	if scale == 0 {
		scale = float32(math.Sqrt(float64(hidden)))
	}
	if kernel > 1 && gridHeight%kernel == 0 && gridWidth%kernel == 0 && rows == gridHeight*gridWidth {
		outHeight, outWidth := gridHeight/kernel, gridWidth/kernel
		out := make([]float32, outHeight*outWidth*hidden)
		for y := range outHeight {
			for x := range outWidth {
				for col := range hidden {
					var sum float32
					for dy := range kernel {
						for dx := range kernel {
							sum += values[((y*kernel+dy)*gridWidth+(x*kernel+dx))*hidden+col]
						}
					}
					out[(y*outWidth+x)*hidden+col] = sum / float32(kernel*kernel) * scale
				}
			}
		}
		return out, outHeight * outWidth
	}
	if kernel > 1 && rows%(kernel*kernel) == 0 {
		outRows := rows / (kernel * kernel)
		out := make([]float32, outRows*hidden)
		for row := range outRows {
			for col := range hidden {
				for group := range kernel * kernel {
					out[row*hidden+col] += values[(row*kernel*kernel+group)*hidden+col]
				}
				out[row*hidden+col] = out[row*hidden+col] / float32(kernel*kernel) * scale
			}
		}
		return out, outRows
	}
	out := append([]float32(nil), values...)
	for index := range out {
		out[index] *= scale
	}
	return out, rows
}

func hipVisionGELUTanh(value float32) float32 {
	return 0.5 * value * (1 + float32(math.Tanh(float64(0.7978845608028654*(value+0.044715*value*value*value)))))
}

func (tower *HIPVisionEncoderTower) outputDim() int {
	if tower == nil {
		return 0
	}
	if tower.projection.outDim > 0 {
		return tower.projection.outDim
	}
	return tower.linear2.outDim
}

func (tower *HIPVisionEncoderTower) Close() error {
	if tower == nil || tower.mapping == nil {
		return nil
	}
	mapping := tower.mapping
	tower.mapping = nil
	return mapping.Close()
}
