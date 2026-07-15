// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// vision_assemble.go is the gemma4 vision tower's loader output: it gathers the SigLIP tower's weights
// off a checkpoint's tensors into a LoadedVision — byte views by role, the vision parallel of
// model.LoadedModel. A backend uploads these views to its device at encode time (native splits the
// loader (byte views) from the forward (device upload), unlike metal which couples the upload into the
// build), so this stays pure Go with no driver type. Lifted from buildGemma4VisionModel in
// pkg/metal/model/gemma4/vision_load.go, reusing the canonicalise + infer front + model.WeightAny.

// LoadedVisionLinear is one vision linear's weight + optional bias byte views (nil bias = none).
type LoadedVisionLinear = model.LoadedVisionLinear

// LoadedVisionLayer is one SigLIP encoder layer's weights: pre/post norms, QK-normed attention, gated MLP.
type LoadedVisionLayer = model.LoadedVisionLayer

// LoadedVisionProjector is the vision-to-text projector's weights (a single projection, or fc1+fc2).
type LoadedVisionProjector = model.LoadedVisionProjector

// LoadedVision is the whole SigLIP tower + projector as byte views — the loader output a backend uploads.
type LoadedVision = model.LoadedVision

const (
	Gemma4BOIToken   = "<|image>"
	Gemma4ImageToken = "<|image|>"
	Gemma4EOIToken   = "<image|>"
	Gemma4VideoToken = "<|video|>"
)

func visionWeight(weights map[string]safetensors.Tensor, names ...string) []byte {
	if t, ok := model.WeightAny(weights, names...); ok {
		return t.Data
	}
	return nil
}

func visionPatchProjection(weights map[string]safetensors.Tensor, cfg *Gemma4VisionConfig) (LoadedVisionLinear, []byte) {
	channels := 3
	patchSize := 0
	if cfg != nil {
		channels = int(cfg.NumChannels)
		patchSize = int(cfg.PatchSize)
	}
	if channels <= 0 {
		channels = 3
	}
	patchDim := channels * patchSize * patchSize
	for _, prefix := range []string{"patch_embedder.input_proj", "embeddings.patch_embedding", "patch_embedding"} {
		for _, candidate := range []string{prefix, prefix + ".linear"} {
			tensor, ok := weights[candidate+".weight"]
			if !ok {
				continue
			}
			linear := model.LoadLinear(weights, candidate, patchDim, "affine")
			if linear == nil {
				continue
			}
			loaded := loadedVisionLinear(linear)
			shape := tensor.Shape
			if len(shape) != 4 {
				if len(shape) == 2 {
					return loaded, loaded.Weight
				}
				return loaded, nil
			}
			if shape[3] == channels {
				return loaded, loaded.Weight
			}
			if shape[1] == channels {
				if out := transposeVisionPatchConvChannelsFirst(tensor); out != nil {
					loaded.Weight = out
					return loaded, out
				}
			}
			return loaded, loaded.Weight
		}
	}
	return LoadedVisionLinear{}, nil
}

func transposeVisionPatchConvChannelsFirst(t safetensors.Tensor) []byte {
	shape := t.Shape
	if len(shape) != 4 {
		return nil
	}
	elem := visionTensorElemBytes(t)
	if elem <= 0 {
		return nil
	}
	hidden, channels, patchH, patchW := shape[0], shape[1], shape[2], shape[3]
	out := make([]byte, len(t.Data))
	for h := range hidden {
		for y := range patchH {
			for x := range patchW {
				for c := range channels {
					src := (((h*channels+c)*patchH+y)*patchW + x) * elem
					dst := (((h*patchH+y)*patchW+x)*channels + c) * elem
					copy(out[dst:dst+elem], t.Data[src:src+elem])
				}
			}
		}
	}
	return out
}

func visionTensorElemBytes(t safetensors.Tensor) int {
	n := 1
	for _, d := range t.Shape {
		if d <= 0 {
			return 0
		}
		n *= d
	}
	if n <= 0 || len(t.Data)%n != 0 {
		return 0
	}
	return len(t.Data) / n
}

func visionPositionEmbeddingTable(weights map[string]safetensors.Tensor) ([]byte, int) {
	t, ok := model.WeightAny(weights, "patch_embedder.position_embedding_table", "embeddings.position_embedding.weight")
	if !ok {
		return nil, 0
	}
	slots := 0
	switch shape := t.Shape; {
	case len(shape) >= 3 && shape[0] >= 2:
		slots = shape[1]
	case len(shape) >= 2:
		slots = shape[0]
	}
	return t.Data, slots
}

func visionLinearWithInputDim(weights map[string]safetensors.Tensor, inDim int, prefixes ...string) LoadedVisionLinear {
	for _, p := range prefixes {
		for _, candidate := range []string{p, p + ".linear"} {
			lin := model.LoadLinear(weights, candidate, inDim, "affine")
			if lin == nil {
				continue
			}
			return loadedVisionLinear(lin)
		}
	}
	return LoadedVisionLinear{}
}

func loadedVisionLinear(linear *model.Linear) LoadedVisionLinear {
	if linear == nil {
		return LoadedVisionLinear{}
	}
	return LoadedVisionLinear{
		Weight:    linear.Weight,
		Scales:    linear.Scales,
		Biases:    linear.Biases,
		Bias:      linear.Bias,
		OutDim:    linear.OutDim,
		InDim:     linear.InDim,
		GroupSize: linear.GroupSize,
		Bits:      linear.Bits,
		Kind:      linear.Kind,
	}
}

// AssembleVision gathers the gemma4 vision tower (when the pack carries one) into a LoadedVision, with the
// config inferred from the weight shapes. Returns (nil, nil) when the pack is text-only / projector-only.
func AssembleVision(weights map[string]safetensors.Tensor, textCfg *Gemma4TextConfig) (*LoadedVision, error) {
	if !gemma4VisionShouldBuildEncoderTower(textCfg) || !HasVisionTowerWeights(weights) {
		return nil, nil
	}
	visionCfg := textCfg.VisionConfig
	if visionCfg == nil {
		visionCfg = &Gemma4VisionConfig{}
	}
	visionCfg = inferGemma4VisionConfig(weights, normalizeGemma4VisionConfig(visionCfg))

	patch, patchConv := visionPatchProjection(weights, visionCfg)
	if len(patch.Weight) == 0 {
		return nil, core.E("gemma4.AssembleVision", "missing patch embedding weight", nil)
	}
	positionTable, positionSlots := visionPositionEmbeddingTable(weights)

	v := &LoadedVision{
		PatchProjection:    patch,
		PatchEmbedding:     patch.Weight,
		PatchConvWeight:    patchConv,
		PositionEmbeddings: positionTable,
		PostLayernorm:      visionWeight(weights, "post_layernorm.weight", "post_layer_norm.weight", "encoder.post_layernorm.weight", "vision_model.post_layernorm.weight"),
		StdBias:            visionWeight(weights, "std_bias"),
		StdScale:           visionWeight(weights, "std_scale"),
		Layers:             make([]LoadedVisionLayer, int(visionCfg.NumHiddenLayers)),
		Cfg:                loadedVisionConfig(visionCfg, textCfg),
	}
	if v.Cfg.PositionEmbeddingSize == 0 {
		v.Cfg.PositionEmbeddingSize = positionSlots
	}
	visionHidden := int(visionCfg.HiddenSize)
	qDim := int(visionCfg.NumAttentionHeads * visionCfg.HeadDim)
	if qDim <= 0 {
		qDim = visionHidden
	}
	for i := range v.Layers {
		p := core.Sprintf("encoder.layers.%d", i)
		L := &v.Layers[i]
		L.InputNorm = visionWeight(weights, p+".input_layernorm.weight", p+".layer_norm1.weight")
		L.PostAttnNorm = visionWeight(weights, p+".post_attention_layernorm.weight", p+".post_attention_layernorm.linear.weight")
		L.PreFFNorm = visionWeight(weights, p+".pre_feedforward_layernorm.weight", p+".layer_norm2.weight")
		L.PostFFNorm = visionWeight(weights, p+".post_feedforward_layernorm.weight", p+".post_feedforward_layernorm.linear.weight")
		L.Q = visionLinearWithInputDim(weights, visionHidden, p+".self_attn.q_proj", p+".attention.q_proj")
		L.K = visionLinearWithInputDim(weights, visionHidden, p+".self_attn.k_proj", p+".attention.k_proj")
		L.V = visionLinearWithInputDim(weights, visionHidden, p+".self_attn.v_proj", p+".attention.v_proj")
		L.O = visionLinearWithInputDim(weights, qDim, p+".self_attn.o_proj", p+".attention.out_proj", p+".attention.o_proj")
		L.QNorm = visionWeight(weights, p+".self_attn.q_norm.weight")
		L.KNorm = visionWeight(weights, p+".self_attn.k_norm.weight")
		L.Gate = visionLinearWithInputDim(weights, visionHidden, p+".mlp.gate_proj", p+".mlp.fc1")
		L.Up = visionLinearWithInputDim(weights, visionHidden, p+".mlp.up_proj")
		ffDim := L.Gate.OutDim
		if ffDim <= 0 {
			ffDim = int(visionCfg.IntermediateSize)
		}
		L.Down = visionLinearWithInputDim(weights, ffDim, p+".mlp.down_proj", p+".mlp.fc2")
		if err := validateLoadedVisionLayer(L, i); err != nil {
			return nil, err
		}
	}
	v.Projector.Projection = visionLinearWithInputDim(weights, visionHidden, "embed_vision.embedding_projection", "multi_modal_projector.embedding_projection", "multi_modal_projector.proj", "multi_modal_projector")
	v.Projector.Linear1 = visionLinearWithInputDim(weights, visionHidden, "multi_modal_projector.linear_1", "multi_modal_projector.fc1")
	linear2In := v.Projector.Linear1.OutDim
	if linear2In == 0 {
		linear2In = visionHidden
	}
	v.Projector.Linear2 = visionLinearWithInputDim(weights, linear2In, "multi_modal_projector.linear_2", "multi_modal_projector.fc2")
	return v, nil
}

func loadedVisionConfig(cfg *Gemma4VisionConfig, textCfg *Gemma4TextConfig) model.LoadedVisionConfig {
	if cfg == nil {
		return model.LoadedVisionConfig{}
	}
	hidden := int(cfg.HiddenSize)
	patch := int(cfg.PatchSize)
	channels := int(cfg.NumChannels)
	out := model.LoadedVisionConfig{
		Hidden:                hidden,
		PatchDim:              channels * patch * patch,
		NumLayers:             int(cfg.NumHiddenLayers),
		NumHeads:              int(cfg.NumAttentionHeads),
		NumKVHeads:            int(cfg.NumKeyValueHeads),
		HeadDim:               int(cfg.HeadDim),
		PatchSize:             int(cfg.PatchSize),
		NumChannels:           int(cfg.NumChannels),
		PositionEmbeddingSize: int(cfg.PositionEmbeddingSize),
		RopeBase:              cfg.RopeParameters.RopeTheta,
		RMSNormEps:            cfg.RMSNormEps,
		PoolKernel:            int(cfg.PoolingKernelSize),
		Standardize:           cfg.Standardize,
		EmbeddingScale:        float32(math.Sqrt(float64(hidden))),
	}
	if textCfg != nil {
		out.ImageTokenID = textCfg.ImageTokenID
		out.ImageBeginToken = Gemma4BOIToken
		out.ImageToken = Gemma4ImageToken
		out.ImageEndToken = Gemma4EOIToken
		out.VideoTokenID = textCfg.VideoTokenID
		out.VideoToken = Gemma4VideoToken
	}
	return out
}

// validateLoadedVisionLayer fails loud on a missing required weight in an encoder layer — a malformed
// vision pack, surfaced at load rather than as a nil view deep in the encode.
func validateLoadedVisionLayer(L *LoadedVisionLayer, idx int) error {
	for _, c := range []struct {
		b    []byte
		name string
	}{
		{L.InputNorm, "input norm"}, {L.PostAttnNorm, "post-attn norm"}, {L.PreFFNorm, "pre-ff norm"}, {L.PostFFNorm, "post-ff norm"},
		{L.Q.Weight, "q proj"}, {L.K.Weight, "k proj"}, {L.V.Weight, "v proj"}, {L.O.Weight, "o proj"},
		{L.QNorm, "q norm"}, {L.KNorm, "k norm"},
		{L.Gate.Weight, "gate proj"}, {L.Up.Weight, "up proj"}, {L.Down.Weight, "down proj"},
	} {
		if len(c.b) == 0 {
			return core.E("gemma4.AssembleVision", core.Sprintf("encoder layer %d missing %s", idx, c.name), nil)
		}
	}
	return nil
}
