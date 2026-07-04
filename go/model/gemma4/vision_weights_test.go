// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"testing"

	"dappco.re/go/inference/safetensors"
)

// TestCanonicalGemma4VisionWeightName pins the canonicalisation: the vision_tower./vision_model. prefix
// is stripped, the projector prefixes are kept, and a text weight is rejected.
func TestCanonicalGemma4VisionWeightName(t *testing.T) {
	cases := []struct {
		in, want string
		ok       bool
	}{
		{"vision_tower.encoder.layers.0.self_attn.q_proj.weight", "encoder.layers.0.self_attn.q_proj.weight", true},
		{"vision_model.embeddings.patch_embedding.weight", "embeddings.patch_embedding.weight", true},
		{"multi_modal_projector.proj.weight", "multi_modal_projector.proj.weight", true},
		{"embed_vision.embedding_projection.weight", "embed_vision.embedding_projection.weight", true},
		{"model.layers.0.self_attn.q_proj.weight", "", false},
	}
	for _, c := range cases {
		got, ok := canonicalGemma4VisionWeightName(c.in)
		if got != c.want || ok != c.ok {
			t.Fatalf("canonicalGemma4VisionWeightName(%q) = (%q,%v), want (%q,%v)", c.in, got, ok, c.want, c.ok)
		}
	}
}

// TestSanitizeAndDetectVisionWeights pins the gather + tower/projector detection over a tiny tensor set.
func TestSanitizeAndDetectVisionWeights(t *testing.T) {
	raw := map[string]safetensors.Tensor{
		"vision_tower.embeddings.patch_embedding.weight": {Dtype: "BF16", Shape: []int{8, 4}},
		"multi_modal_projector.proj.weight":              {Dtype: "BF16", Shape: []int{4, 8}},
		"model.layers.0.self_attn.q_proj.weight":         {Dtype: "BF16", Shape: []int{4, 4}}, // text — dropped
	}
	vision := SanitizeVisionWeights(raw)
	if _, ok := vision["embeddings.patch_embedding.weight"]; !ok {
		t.Fatal("tower weight should be present under its canonical name")
	}
	if _, ok := vision["multi_modal_projector.proj.weight"]; !ok {
		t.Fatal("projector weight should keep its prefix")
	}
	if len(vision) != 2 {
		t.Fatalf("the text weight should be dropped: got %d vision weights", len(vision))
	}
	if !HasVisionTowerWeights(vision) {
		t.Fatal("should detect a full vision tower")
	}
	if !HasVisionProjectionWeights(vision) {
		t.Fatal("should detect the multimodal projector")
	}
}
