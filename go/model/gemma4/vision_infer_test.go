// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"testing"

	"dappco.re/go/inference/safetensors"
)

// TestInferGemma4VisionConfig pins the shape-derived dims: hidden_size + patch_size from the
// patch-embedding weight (588 = 3·14·14 → patch 14), and the encoder-layer count by walking the q_projs.
func TestInferGemma4VisionConfig(t *testing.T) {
	weights := map[string]safetensors.Tensor{
		"patch_embedding.weight":                   {Shape: []int{1152, 588}},
		"encoder.layers.0.self_attn.q_proj.weight": {Shape: []int{1152, 1152}},
		"encoder.layers.1.self_attn.q_proj.weight": {Shape: []int{1152, 1152}},
	}
	cfg := &Gemma4VisionConfig{}
	cfg.NumAttentionHeads = 16 // promoted from the embedded neutral core
	got := inferGemma4VisionConfig(weights, cfg)
	if got.HiddenSize != 1152 {
		t.Fatalf("HiddenSize = %d, want 1152", got.HiddenSize)
	}
	if got.PatchSize != 14 {
		t.Fatalf("PatchSize = %d, want 14 (round(sqrt(588/3)))", got.PatchSize)
	}
	if got.NumHiddenLayers != 2 {
		t.Fatalf("NumHiddenLayers = %d, want 2", got.NumHiddenLayers)
	}
}

// TestGemma4VisionShouldBuildEncoderTower pins the tower/no-tower decision by model_type.
func TestGemma4VisionShouldBuildEncoderTower(t *testing.T) {
	unified := &Gemma4TextConfig{}
	unified.ModelType = "gemma4_unified"
	if gemma4VisionShouldBuildEncoderTower(unified) {
		t.Fatal("gemma4_unified declares no encoder tower")
	}
	dense := &Gemma4TextConfig{}
	dense.ModelType = "gemma4"
	if !gemma4VisionShouldBuildEncoderTower(dense) {
		t.Fatal("gemma4 should build an encoder tower")
	}
}
