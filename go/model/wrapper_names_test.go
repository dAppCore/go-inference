// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"testing"

	"dappco.re/go/inference/model/safetensors"
)

// TestWrapperNames_NormalizeWrapperNames_Good covers the multimodal wrapper case: every
// "language_model."-prefixed tensor is ALSO addressable by its stripped "model.…" name, and
// the original prefixed entry is kept too (an assembler's bare lookups work regardless of
// nesting).
func TestWrapperNames_NormalizeWrapperNames_Good(t *testing.T) {
	in := map[string]safetensors.Tensor{
		"language_model.model.embed_tokens.weight": {Shape: []int{4, 4}},
		"vision_tower.patch_embed.weight":          {Shape: []int{4, 4}},
	}
	out := NormalizeWrapperNames(in)
	if _, ok := out["language_model.model.embed_tokens.weight"]; !ok {
		t.Fatal("the original prefixed entry must be kept")
	}
	if _, ok := out["model.embed_tokens.weight"]; !ok {
		t.Fatal("the stripped, unprefixed entry must be added")
	}
	if _, ok := out["vision_tower.patch_embed.weight"]; !ok {
		t.Fatal("a tensor without the wrapper prefix must be untouched")
	}
	if len(out) != 3 {
		t.Fatalf("len(out) = %d, want 3 (2 in + 1 stripped alias)", len(out))
	}
}

// TestWrapperNames_NormalizeWrapperNames_Bad covers the flat text-only pack: no tensor
// carries the "language_model." prefix, so the map is returned UNCHANGED (same length, same
// keys) rather than gaining spurious aliases.
func TestWrapperNames_NormalizeWrapperNames_Bad(t *testing.T) {
	in := map[string]safetensors.Tensor{
		"model.embed_tokens.weight": {Shape: []int{4, 4}},
		"model.norm.weight":         {Shape: []int{4}},
	}
	out := NormalizeWrapperNames(in)
	if len(out) != len(in) {
		t.Fatalf("len(out) = %d, want %d (unchanged, no wrapper prefix present)", len(out), len(in))
	}
	for k := range in {
		if _, ok := out[k]; !ok {
			t.Fatalf("out is missing input key %q", k)
		}
	}
}

// TestWrapperNames_NormalizeWrapperNames_Ugly covers the degenerate empty tensor set: no
// keys to scan, so the function must return cleanly without indexing anything.
func TestWrapperNames_NormalizeWrapperNames_Ugly(t *testing.T) {
	out := NormalizeWrapperNames(map[string]safetensors.Tensor{})
	if len(out) != 0 {
		t.Fatalf("NormalizeWrapperNames(empty) = %v, want empty", out)
	}
}
