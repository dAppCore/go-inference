// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"testing"

	"dappco.re/go/inference/model"
)

// i32p returns a pointer to an int32 literal — the shape NumExperts/TopKExperts take on
// Gemma4TextConfig (a nil pointer means "the config didn't declare it").
func i32p(v int32) *int32 { return &v }

// TestFeaturesOf covers the single "what is this model" reader in both directions of every
// feature it decodes: the nil-config zero surface, a dense text-only build, a MoE build with
// and without the expert-count pointers, and the vision / audio / hybrid-attention flags. The
// engine reacts to this surface rather than poking config fields, so each branch is load-bearing.
func TestFeaturesOf(t *testing.T) {
	// Nil config → the zero surface (dense, text-only, no attention topology).
	if got := FeaturesOf(nil); got != (Features{}) {
		t.Fatalf("FeaturesOf(nil) = %+v, want the zero surface", got)
	}

	dense := FeaturesOf(&Gemma4TextConfig{
		TransformerConfig: model.TransformerConfig{HiddenSize: 64},
		SlidingWindow:     0,
	})
	if dense.Mixture || dense.Vision || dense.Audio || dense.Attention.Hybrid() {
		t.Fatalf("dense text-only build reported features %+v", dense)
	}
	if dense.NumExperts != 0 || dense.TopKExperts != 0 {
		t.Fatalf("dense build should carry no experts, got %+v", dense)
	}

	// MoE with the expert-count pointers declared → they populate; without them → Mixture true
	// but the counts stay zero (the engine still routes; it reads the counts from the weights).
	moe := FeaturesOf(&Gemma4TextConfig{
		EnableMoEBlock: true, NumExperts: i32p(8), TopKExperts: i32p(2),
	})
	if !moe.Mixture || moe.NumExperts != 8 || moe.TopKExperts != 2 {
		t.Fatalf("MoE build features = %+v, want mixture with 8/2 experts", moe)
	}
	moeNoCounts := FeaturesOf(&Gemma4TextConfig{EnableMoEBlock: true})
	if !moeNoCounts.Mixture || moeNoCounts.NumExperts != 0 || moeNoCounts.TopKExperts != 0 {
		t.Fatalf("MoE build without expert pointers = %+v, want mixture with zero counts", moeNoCounts)
	}

	// Vision / audio presence is read off the sub-config pointers, not a name.
	vis := FeaturesOf(&Gemma4TextConfig{VisionConfig: &Gemma4VisionConfig{}})
	if !vis.Vision || vis.Audio {
		t.Fatalf("vision build features = %+v, want vision only", vis)
	}
	aud := FeaturesOf(&Gemma4TextConfig{AudioConfig: &Gemma4AudioConfig{}})
	if aud.Vision || !aud.Audio {
		t.Fatalf("audio build features = %+v, want audio only", aud)
	}

	// Hybrid attention: the sliding window / pattern / shared-KV counts flow straight through
	// so the engine can select the fixed-sliding KV cache.
	hybrid := FeaturesOf(&Gemma4TextConfig{
		SlidingWindow: 512, SlidingWindowPattern: 6, NumKVSharedLayers: 4,
	})
	if !hybrid.Attention.Hybrid() {
		t.Fatalf("a sliding-window build should report Hybrid(), got %+v", hybrid.Attention)
	}
	if hybrid.Attention.SlidingWindow != 512 || hybrid.Attention.SlidingPattern != 6 || hybrid.Attention.SharedKVLayers != 4 {
		t.Fatalf("hybrid attention class = %+v, want 512/6/4", hybrid.Attention)
	}
	t.Logf("FeaturesOf: nil→zero, dense→text-only, MoE±pointers, vision/audio flags, hybrid attention all decoded")
}

// TestAttentionClassHybrid covers the Hybrid predicate in both directions: a sliding window > 0
// is hybrid (alternating local/global), a zero window is dense (full attention on every layer).
func TestAttentionClassHybrid(t *testing.T) {
	if !(AttentionClass{SlidingWindow: 1}).Hybrid() {
		t.Fatal("SlidingWindow>0 should report Hybrid() = true")
	}
	if (AttentionClass{SlidingWindow: 0, SlidingPattern: 6}).Hybrid() {
		t.Fatal("SlidingWindow==0 should report Hybrid() = false even with a pattern set")
	}
	t.Logf("AttentionClass.Hybrid: window>0 → true, window==0 → false")
}
