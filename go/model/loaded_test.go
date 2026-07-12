// SPDX-Licence-Identifier: EUPL-1.2

package model

import "testing"

// TestLoaded_LoadedModel_Tied_Good covers the ordinary separate-lm_head checkpoint: a
// non-nil LMHead means Tied() reports false.
func TestLoaded_LoadedModel_Tied_Good(t *testing.T) {
	m := &LoadedModel{LMHead: &Linear{OutDim: 8}}
	if m.Tied() {
		t.Fatal("Tied() = true with a separate LMHead, want false")
	}
}

// TestLoaded_LoadedModel_Tied_Bad covers the tied case: a nil LMHead (the checkpoint
// carries no separate lm_head weight) means Tied() reports true — the LM head reuses
// the token embedding.
func TestLoaded_LoadedModel_Tied_Bad(t *testing.T) {
	m := &LoadedModel{LMHead: nil}
	if !m.Tied() {
		t.Fatal("Tied() = false with a nil LMHead, want true")
	}
}

// TestLoaded_LoadedModel_Tied_Ugly covers the zero-value LoadedModel: Tied() must not
// panic on a model with no fields set at all, and must report tied (no LMHead).
func TestLoaded_LoadedModel_Tied_Ugly(t *testing.T) {
	var m LoadedModel
	if !m.Tied() {
		t.Fatal("Tied() on the zero-value LoadedModel = false, want true")
	}
}

// validLoadedModel builds a minimal, fully-valid single-layer dense LoadedModel — every
// weight ValidateRequired demands, and nothing else.
func validLoadedModel() (*LoadedModel, Arch) {
	arch := Arch{Layer: []LayerSpec{{CacheIndex: 0}}}
	m := &LoadedModel{
		Embed: &Linear{OutDim: 4}, FinalNorm: []byte{1, 2},
		Layers: []LoadedLayer{{
			AttnNorm: []byte{1, 2}, Q: &Linear{}, K: &Linear{}, O: &Linear{},
			MLPNorm: []byte{1, 2}, Gate: &Linear{}, Up: &Linear{}, Down: &Linear{},
		}},
	}
	return m, arch
}

// TestLoaded_LoadedModel_ValidateRequired_Good covers a well-formed checkpoint (every
// always-present weight populated, the cache owner carries K): ValidateRequired passes.
func TestLoaded_LoadedModel_ValidateRequired_Good(t *testing.T) {
	m, arch := validLoadedModel()
	if err := m.ValidateRequired(arch); err != nil {
		t.Fatalf("ValidateRequired on a well-formed model: %v", err)
	}
}

// TestLoaded_LoadedModel_ValidateRequired_Bad covers a missing model-level weight
// (Embed absent): a malformed checkpoint is rejected before any per-layer check runs.
func TestLoaded_LoadedModel_ValidateRequired_Bad(t *testing.T) {
	m, arch := validLoadedModel()
	m.Embed = nil
	if err := m.ValidateRequired(arch); err == nil {
		t.Fatal("ValidateRequired with a nil Embed: expected an error")
	}
}

// TestLoaded_LoadedModel_ValidateRequired_Ugly covers the OPTIONAL-weight rules that
// distinguish a genuinely-incomplete checkpoint from a well-formed one of a different
// shape: a KV-shared layer (OwnsCache()==false) is valid with NO k_proj at all, while the
// SAME layer as a cache owner without k_proj is rejected — the one bit that decides
// whether K is required.
func TestLoaded_LoadedModel_ValidateRequired_Ugly(t *testing.T) {
	m, _ := validLoadedModel()
	m.Layers[0].K = nil
	sharedArch := Arch{Layer: []LayerSpec{{CacheIndex: -1, KVShareFrom: 0}}} // shares another layer's cache
	if err := m.ValidateRequired(sharedArch); err != nil {
		t.Fatalf("ValidateRequired: a KV-shared layer must not require its own k_proj: %v", err)
	}
	ownerArch := Arch{Layer: []LayerSpec{{CacheIndex: 0}}} // owns its cache
	if err := m.ValidateRequired(ownerArch); err == nil {
		t.Fatal("ValidateRequired: a cache-owning layer with no k_proj: expected an error")
	}
}

func TestLoaded_LoadedModel_ValidateRequired_OLMoNormPlacement(t *testing.T) {
	linear := &Linear{Weight: []byte{1}}
	baseLayer := LoadedLayer{Q: linear, K: linear, V: linear, O: linear, Gate: linear, Up: linear, Down: linear}
	baseArch := Arch{NormPlacement: NormPlacementPre, NonParametricLayerNorm: true, Layer: []LayerSpec{{CacheIndex: 0}}}
	if err := (&LoadedModel{Embed: linear, Layers: []LoadedLayer{baseLayer}}).ValidateRequired(baseArch); err != nil {
		t.Fatalf("OLMo non-parametric norms rejected: %v", err)
	}
	postLayer := baseLayer
	postLayer.PostAttnNorm, postLayer.PostFFNorm = []byte{1}, []byte{1}
	postArch := Arch{NormPlacement: NormPlacementPost, Layer: []LayerSpec{{CacheIndex: 0}}}
	if err := (&LoadedModel{Embed: linear, FinalNorm: []byte{1}, Layers: []LoadedLayer{postLayer}}).ValidateRequired(postArch); err != nil {
		t.Fatalf("OLMo2 post norms rejected: %v", err)
	}
}
