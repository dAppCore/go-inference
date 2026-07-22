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

// TestWrapperNames_NormalizeWrapperNames_SameMapWhenAliased covers the load-bearing #60
// contract: once every wrapped tensor's stripped alias exists, the function returns the SAME
// map (no fresh copy) — so model.Load can alias dm.Tensors once and Assemble's own call passes
// that map through, keeping LoadLinear's repack writeback visible to the owned-tensor adoption.
// A pre-existing stripped name is never clobbered by an alias.
func TestWrapperNames_NormalizeWrapperNames_SameMapWhenAliased(t *testing.T) {
	in := map[string]safetensors.Tensor{
		"language_model.model.embed_tokens.weight": {Shape: []int{4, 4}},
		"vision_tower.patch_embed.weight":          {Shape: []int{4, 4}},
	}
	first := NormalizeWrapperNames(in)
	if len(first) != 3 {
		t.Fatalf("first pass len = %d, want 3", len(first))
	}
	// Second pass: everything already aliased → the SAME map comes back, so a writeback into it
	// is visible to every holder of the map.
	second := NormalizeWrapperNames(first)
	marker := safetensors.Tensor{Shape: []int{1}, Data: []byte{1, 2}}
	second["model.embed_tokens.weight"] = marker
	if got := first["model.embed_tokens.weight"]; len(got.Data) != 2 {
		t.Fatal("second pass returned a fresh map — writebacks would be swallowed (the Bonsai wrapper miss)")
	}
	// No-clobber: a checkpoint that genuinely ships BOTH the wrapped and the bare name keeps its
	// own bare tensor authoritative.
	both := map[string]safetensors.Tensor{
		"language_model.model.norm.weight": {Shape: []int{8}},
		"model.norm.weight":                {Shape: []int{4}},
	}
	out := NormalizeWrapperNames(both)
	if got := out["model.norm.weight"]; len(got.Shape) != 1 || got.Shape[0] != 4 {
		t.Fatalf("pre-existing bare tensor was clobbered by the alias: %+v", got)
	}
}

// TestWrapperNames_WrapperStripped_Good covers both wrapper layouts: the classic
// "language_model." nesting strips to the bare name, and the current transformers
// "model.language_model." nesting (Qwen3.5 snapshots) strips its language_model segment.
func TestWrapperNames_WrapperStripped_Good(t *testing.T) {
	cases := []struct{ in, want string }{
		{"language_model.model.embed_tokens.weight", "model.embed_tokens.weight"},
		{"language_model.model.norm.weight", "model.norm.weight"},
		{"language_model.lm_head.weight", "lm_head.weight"},
		{"model.language_model.embed_tokens.weight", "model.embed_tokens.weight"},
		{"model.language_model.layers.0.linear_attn.in_proj_qkv.weight", "model.layers.0.linear_attn.in_proj_qkv.weight"},
		{"model.language_model.norm.weight", "model.norm.weight"},
	}
	for _, tc := range cases {
		got, ok := wrapperStripped(tc.in)
		if !ok || got != tc.want {
			t.Errorf("wrapperStripped(%q) = %q, %v; want %q, true", tc.in, got, ok, tc.want)
		}
	}
}

// TestWrapperNames_WrapperStripped_Bad covers names that carry NEITHER wrapper prefix — bare
// text-model names and the wrapper's non-text siblings (the vision tower, the MTP head) — which
// must pass through unaliased.
func TestWrapperNames_WrapperStripped_Bad(t *testing.T) {
	for _, in := range []string{
		"model.embed_tokens.weight",
		"model.norm.weight",
		"model.visual.merger.linear_fc1.weight",
		"mtp.layers.0.self_attn.q_proj.weight",
		"vision_tower.patch_embed.weight",
		"lm_head.weight",
	} {
		if got, ok := wrapperStripped(in); ok {
			t.Errorf("wrapperStripped(%q) = %q, true; want no alias", in, got)
		}
	}
}

// TestWrapperNames_NormalizeWrapperNames_NestedWrapperLayout covers the transformers
// "model.language_model." layout Qwen3.5 snapshots ship: every nested text tensor gains its bare
// "model.…" alias (so model.Assemble's model.embed_tokens lookup resolves), while the wrapper's
// non-text siblings (model.visual.…, mtp.…) stay untouched, and a second pass returns the SAME
// map (the #60 writeback contract holds for this layout too).
func TestWrapperNames_NormalizeWrapperNames_NestedWrapperLayout(t *testing.T) {
	in := map[string]safetensors.Tensor{
		"model.language_model.embed_tokens.weight":                     {Shape: []int{4, 4}},
		"model.language_model.layers.0.linear_attn.in_proj_qkv.weight": {Shape: []int{4, 4}},
		"model.visual.merger.linear_fc1.weight":                        {Shape: []int{4, 4}},
		"mtp.layers.0.self_attn.q_proj.weight":                         {Shape: []int{4, 4}},
	}
	out := NormalizeWrapperNames(in)
	for _, want := range []string{
		"model.embed_tokens.weight",
		"model.layers.0.linear_attn.in_proj_qkv.weight",
		"model.language_model.embed_tokens.weight", // originals kept
		"model.visual.merger.linear_fc1.weight",    // non-text siblings untouched
		"mtp.layers.0.self_attn.q_proj.weight",
	} {
		if _, ok := out[want]; !ok {
			t.Fatalf("out is missing %q", want)
		}
	}
	if len(out) != 6 {
		t.Fatalf("len(out) = %d, want 6 (4 in + 2 stripped aliases)", len(out))
	}
	// Second pass: already aliased → the SAME map (writebacks stay visible, see #60).
	second := NormalizeWrapperNames(out)
	marker := safetensors.Tensor{Shape: []int{1}, Data: []byte{1, 2}}
	second["model.embed_tokens.weight"] = marker
	if got := out["model.embed_tokens.weight"]; len(got.Data) != 2 {
		t.Fatal("second pass returned a fresh map — writebacks would be swallowed")
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
