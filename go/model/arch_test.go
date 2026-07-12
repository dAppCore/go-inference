// SPDX-Licence-Identifier: EUPL-1.2

package model

import "testing"

func TestArch_QKNormalization_Good(t *testing.T) {
	if got := (Arch{QKNormalization: QKLayerNorm}).QKNormalization; got != QKLayerNorm {
		t.Fatalf("QKNormalization = %q, want %q", got, QKLayerNorm)
	}
}

func TestArch_QKNormalization_Bad(t *testing.T) {
	if got := (Arch{}).QKNormalization; got != QKNone {
		t.Fatalf("zero-value QKNormalization = %q, want none", got)
	}
}

func TestArch_QKNormalization_Ugly(t *testing.T) {
	const unknown QKNormalization = "unknown"
	if got := (Arch{QKNormalization: unknown}).QKNormalization; got != unknown {
		t.Fatalf("declared QKNormalization = %q, want preserved %q", got, unknown)
	}
}

func TestArch_NormPlacement_Good(t *testing.T) {
	if NormPlacementPre != NormPlacement("pre") {
		t.Fatalf("NormPlacementPre = %q", NormPlacementPre)
	}
}

func TestArch_NormPlacement_Bad(t *testing.T) {
	if NormPlacementPost != NormPlacement("post") {
		t.Fatalf("NormPlacementPost = %q", NormPlacementPost)
	}
}

func TestArch_NormPlacement_Ugly(t *testing.T) {
	var placement NormPlacement
	if placement != NormPlacementUnspecified {
		t.Fatalf("zero NormPlacement = %q", placement)
	}
}

// TestResolveMoEGating covers the gating default: an unset gating resolves to
// MoEGatingSoftmax (the only router variant the metal engine ships, and gemma4's
// method), while an explicitly-declared gating passes through unchanged — the path
// gemma4's always-softmax config never exercises.
func TestResolveMoEGating(t *testing.T) {
	if got := resolveMoEGating(""); got != MoEGatingSoftmax {
		t.Fatalf(`resolveMoEGating("") = %q, want the default %q`, got, MoEGatingSoftmax)
	}
	if got := resolveMoEGating(MoEGatingSoftmax); got != MoEGatingSoftmax {
		t.Fatalf("resolveMoEGating(MoEGatingSoftmax) = %q, want it unchanged", got)
	}
	if got := resolveMoEGating(MoEGating("sigmoid")); got != MoEGating("sigmoid") {
		t.Fatalf(`resolveMoEGating("sigmoid") = %q, want a declared gating to pass through`, got)
	}
}

// TestArch_LayerSpec_OwnsCache_Good covers an owning layer (CacheIndex >= 0): it holds
// its own KV cache.
func TestArch_LayerSpec_OwnsCache_Good(t *testing.T) {
	if l := (LayerSpec{CacheIndex: 0}); !l.OwnsCache() {
		t.Fatal("CacheIndex=0 should own its cache")
	}
	if l := (LayerSpec{CacheIndex: 3}); !l.OwnsCache() {
		t.Fatal("CacheIndex=3 should own its cache")
	}
}

// TestArch_LayerSpec_OwnsCache_Bad covers a sharing layer (CacheIndex < 0): it reads
// another layer's cache rather than owning one.
func TestArch_LayerSpec_OwnsCache_Bad(t *testing.T) {
	if l := (LayerSpec{CacheIndex: -1}); l.OwnsCache() {
		t.Fatal("CacheIndex=-1 should NOT own its cache (shares another layer's)")
	}
}

// TestArch_LayerSpec_OwnsCache_Ugly covers the zero-value LayerSpec: CacheIndex defaults
// to 0, which OwnsCache treats as an owner — the zero value is "owns", not "shares",
// so a caller that forgets to set CacheIndex explicitly does not silently share a cache
// that does not exist.
func TestArch_LayerSpec_OwnsCache_Ugly(t *testing.T) {
	var l LayerSpec
	if !l.OwnsCache() {
		t.Fatal("the zero-value LayerSpec should own its cache (CacheIndex zero-value is 0, not -1)")
	}
}

// TestArch_LayerSpec_TypeName_Good covers the sliding-attention spelling — the config
// vocabulary DeriveLayers reads back.
func TestArch_LayerSpec_TypeName_Good(t *testing.T) {
	if got := (LayerSpec{Attention: SlidingAttention}).TypeName(); got != "sliding_attention" {
		t.Fatalf("TypeName(sliding) = %q, want %q", got, "sliding_attention")
	}
}

// TestArch_LayerSpec_TypeName_Bad covers the global-attention spelling — the other half
// of the config vocabulary.
func TestArch_LayerSpec_TypeName_Bad(t *testing.T) {
	if got := (LayerSpec{Attention: GlobalAttention}).TypeName(); got != "full_attention" {
		t.Fatalf("TypeName(global) = %q, want %q", got, "full_attention")
	}
}

// TestArch_LayerSpec_TypeName_Ugly covers an out-of-range AttentionType value (neither
// declared constant): TypeName must fall back to "full_attention" rather than panicking
// or returning a garbage string.
func TestArch_LayerSpec_TypeName_Ugly(t *testing.T) {
	if got := (LayerSpec{Attention: AttentionType(99)}).TypeName(); got != "full_attention" {
		t.Fatalf("TypeName(undeclared value) = %q, want the full_attention fallback", got)
	}
}

// TestArch_Arch_HasMoE_Good covers an arch with at least one MoE layer among others:
// HasMoE reports true even when it is not layer 0.
func TestArch_Arch_HasMoE_Good(t *testing.T) {
	a := Arch{Layer: []LayerSpec{{MoE: false}, {MoE: true}}}
	if !a.HasMoE() {
		t.Fatal("HasMoE() = false, want true (layer 1 is MoE)")
	}
}

// TestArch_Arch_HasMoE_Bad covers an arch with every layer dense: HasMoE reports false.
func TestArch_Arch_HasMoE_Bad(t *testing.T) {
	a := Arch{Layer: []LayerSpec{{MoE: false}, {MoE: false}}}
	if a.HasMoE() {
		t.Fatal("HasMoE() = true, want false (no layer is MoE)")
	}
}

// TestArch_Arch_HasMoE_Ugly covers the zero-layer arch: no layers to scan, so HasMoE
// must return false rather than panic on an empty slice.
func TestArch_Arch_HasMoE_Ugly(t *testing.T) {
	if (Arch{}).HasMoE() {
		t.Fatal("HasMoE() on a zero-layer Arch = true, want false")
	}
}

// TestArch_Arch_MaxHeadDim_Good covers the ordinary case: full_attention layers use a
// LARGER head_dim than sliding — MaxHeadDim picks the global (larger) value.
func TestArch_Arch_MaxHeadDim_Good(t *testing.T) {
	a := Arch{HeadDim: 256, GlobalHeadDim: 512}
	if got := a.MaxHeadDim(); got != 512 {
		t.Fatalf("MaxHeadDim() = %d, want the larger GlobalHeadDim 512", got)
	}
}

// TestArch_Arch_MaxHeadDim_Bad covers the uniform case (no sliding/full distinction):
// GlobalHeadDim left at its zero value must NOT win over the real HeadDim.
func TestArch_Arch_MaxHeadDim_Bad(t *testing.T) {
	a := Arch{HeadDim: 128}
	if got := a.MaxHeadDim(); got != 128 {
		t.Fatalf("MaxHeadDim() = %d, want HeadDim 128 (GlobalHeadDim undeclared)", got)
	}
}

// TestArch_Arch_MaxHeadDim_Ugly covers GlobalHeadDim declared SMALLER than HeadDim (an
// unusual but not impossible config): the sliding value must still win, since MaxHeadDim
// promises the LARGER of the two, whichever field that is.
func TestArch_Arch_MaxHeadDim_Ugly(t *testing.T) {
	a := Arch{HeadDim: 256, GlobalHeadDim: 64}
	if got := a.MaxHeadDim(); got != 256 {
		t.Fatalf("MaxHeadDim() = %d, want HeadDim 256 (GlobalHeadDim is smaller)", got)
	}
}

// TestArch_Arch_MaxKVHeads_Good mirrors MaxHeadDim for the KV-head count: the full
// (global) layers' larger KV-head count wins.
func TestArch_Arch_MaxKVHeads_Good(t *testing.T) {
	a := Arch{KVHeads: 2, GlobalKVHeads: 8}
	if got := a.MaxKVHeads(); got != 8 {
		t.Fatalf("MaxKVHeads() = %d, want the larger GlobalKVHeads 8", got)
	}
}

// TestArch_Arch_MaxKVHeads_Bad covers the undeclared-distinction case: GlobalKVHeads at
// its zero value must not win over the real KVHeads.
func TestArch_Arch_MaxKVHeads_Bad(t *testing.T) {
	a := Arch{KVHeads: 4}
	if got := a.MaxKVHeads(); got != 4 {
		t.Fatalf("MaxKVHeads() = %d, want KVHeads 4 (GlobalKVHeads undeclared)", got)
	}
}

// TestArch_Arch_MaxKVHeads_Ugly covers GlobalKVHeads declared smaller than KVHeads: the
// sliding value must still win (the larger of the two, whichever field).
func TestArch_Arch_MaxKVHeads_Ugly(t *testing.T) {
	a := Arch{KVHeads: 8, GlobalKVHeads: 2}
	if got := a.MaxKVHeads(); got != 8 {
		t.Fatalf("MaxKVHeads() = %d, want KVHeads 8 (GlobalKVHeads is smaller)", got)
	}
}

// TestArch_DeriveLayers_Good covers the ordinary sliding/global interleave with KV
// sharing: the first (n-numKVShared) layers own their cache, and a later layer of a
// type that already has an owner shares that owner's cache.
func TestArch_DeriveLayers_Good(t *testing.T) {
	layerTypes := []string{"sliding_attention", "sliding_attention", "full_attention", "sliding_attention"}
	specs := DeriveLayers(layerTypes, 2) // last 2 layers are in the shared region
	if len(specs) != 4 {
		t.Fatalf("len(specs) = %d, want 4", len(specs))
	}
	// layers 0,1 own their cache (outside the shared region).
	if !specs[0].OwnsCache() || specs[0].CacheIndex != 0 {
		t.Fatalf("layer 0 = %+v, want owner at cache slot 0", specs[0])
	}
	if !specs[1].OwnsCache() || specs[1].CacheIndex != 1 {
		t.Fatalf("layer 1 = %+v, want owner at cache slot 1", specs[1])
	}
	// layer 2 (full_attention) is in the shared region but the FIRST of its type →
	// promoted to owner.
	if !specs[2].OwnsCache() {
		t.Fatalf("layer 2 = %+v, want promoted to owner (first full_attention layer)", specs[2])
	}
	// layer 3 (sliding_attention) is in the shared region and sliding already has an
	// owner (layer 1) → shares it.
	if specs[3].OwnsCache() {
		t.Fatalf("layer 3 = %+v, want sharing, not owning", specs[3])
	}
	if specs[3].KVShareFrom != 1 {
		t.Fatalf("layer 3 KVShareFrom = %d, want 1 (the latest sliding owner)", specs[3].KVShareFrom)
	}
}

// TestArch_DeriveLayers_Bad covers numKVShared out of range: a negative or
// larger-than-n value is clamped rather than producing a negative/overflowing
// firstShared boundary.
func TestArch_DeriveLayers_Bad(t *testing.T) {
	// numKVShared > n clamps firstShared to 0: EVERY layer lands in the shared
	// region, so with two DIFFERENT types each is still the first of its own type
	// there and gets promoted to owner (no owner to share from yet).
	distinct := []string{"full_attention", "sliding_attention"}
	specs := DeriveLayers(distinct, 99)
	for i, s := range specs {
		if !s.OwnsCache() {
			t.Fatalf("layer %d = %+v, want owner (first of its type, even in a fully-shared region)", i, s)
		}
	}
	// the SAME type repeated shows the shared region actually shares: only the
	// first occurrence is promoted, the second shares it.
	repeated := []string{"full_attention", "full_attention"}
	specs = DeriveLayers(repeated, 99)
	if !specs[0].OwnsCache() {
		t.Fatalf("layer 0 = %+v, want owner (first full_attention layer)", specs[0])
	}
	if specs[1].OwnsCache() || specs[1].KVShareFrom != 0 {
		t.Fatalf("layer 1 = %+v, want sharing layer 0's cache", specs[1])
	}
	// numKVShared < 0 clamps firstShared to n: no shared region at all, every
	// layer owns regardless of repeated types.
	specs = DeriveLayers(repeated, -5)
	for i, s := range specs {
		if !s.OwnsCache() {
			t.Fatalf("layer %d = %+v, want owner (no shared region, numKVShared<0 clamped to n)", i, s)
		}
	}
}

// TestArch_DeriveLayers_Ugly covers the zero-layer input: an empty layerTypes must
// return an empty (non-nil-panicking) spec slice, not error.
func TestArch_DeriveLayers_Ugly(t *testing.T) {
	specs := DeriveLayers(nil, 0)
	if len(specs) != 0 {
		t.Fatalf("DeriveLayers(nil) = %v, want empty", specs)
	}
}
