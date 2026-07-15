// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"testing"

	"dappco.re/go/inference/model/safetensors"
)

// unifiedVisionConfig builds the text config an encoder-free gemma4_unified pack declares: the
// ModelType that makes gemma4VisionShouldBuildEncoderTower decline the SigLIP tower, a 4px model
// patch, an 8-wide mm embed, a 16-wide text hidden, and the "vision" bidirectional flag.
func unifiedVisionConfig() *Gemma4TextConfig {
	cfg := &Gemma4TextConfig{
		ImageTokenID: 42, VideoTokenID: 43,
		UseBidirectionalAttention: "vision",
		VisionConfig:              &Gemma4VisionConfig{ModelPatchSize: 4, MMEmbedDim: 8},
	}
	cfg.ModelType = "gemma4_unified"
	cfg.HiddenSize = 16
	return cfg
}

// unifiedVisionWeights builds a complete encoder-free vision-embedder tensor set matching
// unifiedVisionConfig: patch_dense [8, patchDim=4*4*3], projection [16, 8], a [positions,2,8]
// factorised position table, and the six patch/pos LayerNorm weight+bias tensors.
func unifiedVisionWeights() map[string]safetensors.Tensor {
	const modelPatch, channels, hidden, mmEmbed, textHidden, positions = 4, 3, 8, 8, 16, 5
	patchDim := modelPatch * modelPatch * channels // 48
	return map[string]safetensors.Tensor{
		"vision_embedder.patch_dense.weight":       audioBF16Tensor(hidden, patchDim),
		"embed_vision.embedding_projection.weight": audioBF16Tensor(textHidden, mmEmbed),
		"vision_embedder.pos_embedding":            audioBF16Tensor(positions, 2, mmEmbed),
		"vision_embedder.patch_ln1.weight":         audioBF16Tensor(patchDim),
		"vision_embedder.patch_ln1.bias":           audioBF16Tensor(patchDim),
		"vision_embedder.patch_ln2.weight":         audioBF16Tensor(hidden),
		"vision_embedder.patch_ln2.bias":           audioBF16Tensor(hidden),
		"vision_embedder.pos_norm.weight":          audioBF16Tensor(hidden),
		"vision_embedder.pos_norm.bias":            audioBF16Tensor(hidden),
	}
}

// TestAssembleUnifiedVision_Good assembles a complete encoder-free unified embedder plus the
// optional raw-waveform audio head, and asserts the derived config (mm embed, text hidden,
// position count from the table) and the image/video prompt metadata all flow through.
func TestAssembleUnifiedVision_Good(t *testing.T) {
	cfg := unifiedVisionConfig()
	cfg.AudioConfig = &Gemma4AudioConfig{AudioSamplesPerToken: 640}
	cfg.AudioTokenID = 44
	w := unifiedVisionWeights()
	w["embed_audio.embedding_projection.weight"] = audioBF16Tensor(4, 4) // raw-waveform audio head

	uv, err := AssembleUnifiedVision(w, cfg)
	if err != nil {
		t.Fatalf("AssembleUnifiedVision: %v", err)
	}
	if uv == nil {
		t.Fatal("AssembleUnifiedVision returned nil for a complete unified pack")
	}
	if uv.PatchDense.Weight == nil || uv.Projection.Weight == nil || len(uv.PosEmbedding) == 0 {
		t.Fatalf("unified embedder payload incomplete: %+v", uv)
	}
	if uv.Cfg.MMEmbedDim != 8 || uv.Cfg.TextHidden != 16 || uv.Cfg.PosembSize != 5 || uv.Cfg.ModelPatchSize != 4 {
		t.Fatalf("derived config = %+v, want mm8/text16/pos5/patch4", uv.Cfg)
	}
	if uv.Cfg.ImageTokenID != 42 || uv.Cfg.VideoTokenID != 43 || !uv.Cfg.BidirectionalImages {
		t.Fatalf("prompt metadata = %+v, want image42/video43/bidirectional", uv.Cfg)
	}
	if uv.AudioProjection.Weight == nil || uv.Cfg.AudioSamplesPerToken != 640 || uv.Cfg.AudioTokenID != 44 {
		t.Fatalf("audio head = %+v (samples=%d), want attached with 640 samples", uv.AudioProjection, uv.Cfg.AudioSamplesPerToken)
	}
	t.Logf("AssembleUnifiedVision: complete unified pack + audio head assembled, config derived from shapes")
}

// TestAssembleUnifiedVisionDeclines covers the two (nil, nil) opt-out paths: an encoder-tower pack
// (ModelType leaves gemma4VisionShouldBuildEncoderTower true) and a pack carrying no patch_dense
// weight — both mean "this isn't a unified embedder", handled by the caller, not an error.
func TestAssembleUnifiedVisionDeclines(t *testing.T) {
	// Encoder-tower pack: default model_type keeps the SigLIP tower, so the unified assembler declines.
	tower := &Gemma4TextConfig{}
	tower.ModelType = "gemma4"
	if uv, err := AssembleUnifiedVision(unifiedVisionWeights(), tower); uv != nil || err != nil {
		t.Fatalf("encoder-tower pack should yield (nil,nil), got (%v,%v)", uv, err)
	}

	// Unified model_type but no patch_dense weight → still (nil,nil): text-only / projector-only pack.
	w := unifiedVisionWeights()
	delete(w, "vision_embedder.patch_dense.weight")
	if uv, err := AssembleUnifiedVision(w, unifiedVisionConfig()); uv != nil || err != nil {
		t.Fatalf("pack without patch_dense should yield (nil,nil), got (%v,%v)", uv, err)
	}
	t.Logf("AssembleUnifiedVision: encoder-tower pack and patch_dense-less pack both decline with (nil,nil)")
}

// TestAssembleUnifiedVisionErrors drives each malformed-input error branch: an undeclared model
// patch size, a projection weight absent, a pos_embedding with the wrong middle dimension, a
// pos_embedding absent, and a missing patch/pos LayerNorm.
func TestAssembleUnifiedVisionErrors(t *testing.T) {
	t.Run("model_patch_size undeclared", func(t *testing.T) {
		cfg := unifiedVisionConfig()
		cfg.VisionConfig = &Gemma4VisionConfig{} // no ModelPatchSize, no PatchSize → 0*pooling = 0
		if _, err := AssembleUnifiedVision(unifiedVisionWeights(), cfg); err == nil {
			t.Fatal("expected an error when model_patch_size resolves to 0")
		}
	})

	t.Run("missing projection", func(t *testing.T) {
		w := unifiedVisionWeights()
		delete(w, "embed_vision.embedding_projection.weight")
		if _, err := AssembleUnifiedVision(w, unifiedVisionConfig()); err == nil {
			t.Fatal("expected an error when embed_vision.embedding_projection is absent")
		}
	})

	t.Run("pos_embedding wrong middle dim", func(t *testing.T) {
		w := unifiedVisionWeights()
		w["vision_embedder.pos_embedding"] = audioBF16Tensor(5, 3, 8) // want [positions, 2, mm]
		if _, err := AssembleUnifiedVision(w, unifiedVisionConfig()); err == nil {
			t.Fatal("expected an error when pos_embedding middle dim != 2")
		}
	})

	t.Run("pos_embedding absent", func(t *testing.T) {
		w := unifiedVisionWeights()
		delete(w, "vision_embedder.pos_embedding")
		if _, err := AssembleUnifiedVision(w, unifiedVisionConfig()); err == nil {
			t.Fatal("expected an error when pos_embedding is absent")
		}
	})

	t.Run("missing pos_norm", func(t *testing.T) {
		w := unifiedVisionWeights()
		delete(w, "vision_embedder.pos_norm.weight")
		if _, err := AssembleUnifiedVision(w, unifiedVisionConfig()); err == nil {
			t.Fatal("expected an error when a patch/pos LayerNorm is absent")
		}
	})
	t.Logf("AssembleUnifiedVision: undeclared patch size / missing projection / malformed+absent pos / missing norm all rejected")
}
