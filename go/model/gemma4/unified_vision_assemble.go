// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
	"dappco.re/go/inference/model/vision"
)

// unified_vision_assemble.go gathers the ENCODER-FREE vision embedder the
// gemma4_unified packs ship (12B): no SigLIP tower — raw 48px patches run
// LayerNorm → patch dense (+bias) → LayerNorm → factorised per-axis position
// add → LayerNorm → scale-free RMSNorm → projection, straight into the
// backbone hidden. AssembleVision deliberately declines these packs
// (gemma4VisionShouldBuildEncoderTower == false); this is their assembler.

// AssembleUnifiedVision gathers the gemma4_unified vision embedder into a
// vision.Unified. Returns (nil, nil) when the pack carries no unified
// embedder (tower packs populate LoadedModel.Vision instead; text-only packs
// carry neither).
func AssembleUnifiedVision(weights map[string]safetensors.Tensor, textCfg *Gemma4TextConfig) (*vision.Unified, error) {
	if gemma4VisionShouldBuildEncoderTower(textCfg) {
		return nil, nil
	}
	if _, ok := weights["vision_embedder.patch_dense.weight"]; !ok {
		return nil, nil
	}
	visionCfg := textCfg.VisionConfig
	if visionCfg == nil {
		visionCfg = &Gemma4VisionConfig{}
	}
	visionCfg = normalizeGemma4VisionConfig(visionCfg)

	modelPatch := int(visionCfg.ModelPatchSize)
	if modelPatch <= 0 {
		modelPatch = int(visionCfg.PatchSize) * int(visionCfg.PoolingKernelSize)
	}
	if modelPatch <= 0 {
		return nil, core.E("gemma4.AssembleUnifiedVision", "model_patch_size is undeclared", nil)
	}
	patchDim := modelPatch * modelPatch * int(visionCfg.NumChannels)

	dense := visionLinearWithInputDim(weights, patchDim, "vision_embedder.patch_dense")
	if dense.Weight == nil {
		return nil, core.E("gemma4.AssembleUnifiedVision", "missing vision_embedder.patch_dense", nil)
	}
	mmEmbed := int(visionCfg.MMEmbedDim)
	if mmEmbed <= 0 {
		mmEmbed = dense.OutDim
	}
	proj := visionLinearWithInputDim(weights, mmEmbed, "embed_vision.embedding_projection")
	if proj.Weight == nil {
		return nil, core.E("gemma4.AssembleUnifiedVision", "missing embed_vision.embedding_projection", nil)
	}
	pos, ok := weights["vision_embedder.pos_embedding"]
	if !ok || len(pos.Data) == 0 || len(pos.Shape) != 3 || pos.Shape[1] != 2 {
		return nil, core.E("gemma4.AssembleUnifiedVision", "missing or malformed vision_embedder.pos_embedding (want [positions, 2, mm_embed_dim])", nil)
	}

	uv := &vision.Unified{
		PatchLN1W:    visionWeight(weights, "vision_embedder.patch_ln1.weight"),
		PatchLN1B:    visionWeight(weights, "vision_embedder.patch_ln1.bias"),
		PatchDense:   dense,
		PatchLN2W:    visionWeight(weights, "vision_embedder.patch_ln2.weight"),
		PatchLN2B:    visionWeight(weights, "vision_embedder.patch_ln2.bias"),
		PosEmbedding: pos.Data,
		PosNormW:     visionWeight(weights, "vision_embedder.pos_norm.weight"),
		PosNormB:     visionWeight(weights, "vision_embedder.pos_norm.bias"),
		Projection:   proj,
	}
	for _, c := range []struct {
		b    []byte
		name string
	}{
		{uv.PatchLN1W, "patch_ln1.weight"}, {uv.PatchLN1B, "patch_ln1.bias"},
		{uv.PatchLN2W, "patch_ln2.weight"}, {uv.PatchLN2B, "patch_ln2.bias"},
		{uv.PosNormW, "pos_norm.weight"}, {uv.PosNormB, "pos_norm.bias"},
	} {
		if len(c.b) == 0 {
			return nil, core.E("gemma4.AssembleUnifiedVision", "missing vision_embedder."+c.name, nil)
		}
	}

	textHidden := 0
	if textCfg != nil {
		textHidden = int(textCfg.HiddenSize)
	}
	if textHidden <= 0 {
		textHidden = proj.OutDim
	}
	posemb := int(visionCfg.MMPosembSize)
	if posemb <= 0 {
		posemb = pos.Shape[0]
	}
	uv.Cfg = vision.UnifiedConfig{
		MMEmbedDim:     mmEmbed,
		TextHidden:     textHidden,
		PosembSize:     posemb,
		PatchSize:      int(visionCfg.PatchSize),
		ModelPatchSize: modelPatch,
		PoolKernel:     int(visionCfg.PoolingKernelSize),
		MaxSoftTokens:  int(visionCfg.NumSoftTokens),
		// The upstream patch/pos norms are nn.LayerNorm at the PyTorch default
		// epsilon; the config's layer_norm_eps is cross-filled from
		// rms_norm_eps (1e-6) and does NOT describe them.
		LayerNormEps: 1e-5,
		RMSNormEps:   visionCfg.RMSNormEps,
	}
	if textCfg != nil {
		uv.Cfg.ImageTokenID = textCfg.ImageTokenID
		uv.Cfg.ImageBeginToken = Gemma4BOIToken
		uv.Cfg.ImageToken = Gemma4ImageToken
		uv.Cfg.ImageEndToken = Gemma4EOIToken
		uv.Cfg.VideoTokenID = textCfg.VideoTokenID
		uv.Cfg.VideoToken = Gemma4VideoToken
		uv.Cfg.BidirectionalImages = textCfg.UseBidirectionalAttention == "vision"
	}
	// The unified audio head: one projection over raw-waveform tokens
	// (AudioSamplesPerToken 16 kHz samples each — no mel, no Conformer).
	if textCfg != nil && textCfg.AudioConfig != nil && textCfg.AudioConfig.AudioSamplesPerToken > 0 {
		spt := int(textCfg.AudioConfig.AudioSamplesPerToken)
		if audio := visionLinearWithInputDim(weights, spt, "embed_audio.embedding_projection"); audio.Weight != nil {
			uv.AudioProjection = audio
			uv.Cfg.AudioSamplesPerToken = spt
			uv.Cfg.AudioTokenID = textCfg.AudioTokenID
			uv.Cfg.AudioBeginToken = Gemma4BOAToken
			uv.Cfg.AudioToken = Gemma4AudioToken
			uv.Cfg.AudioEndToken = Gemma4EOAToken
		}
	}
	return uv, nil
}
