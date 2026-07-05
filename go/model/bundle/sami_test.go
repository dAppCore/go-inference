// SPDX-Licence-Identifier: EUPL-1.2

package bundle

import (
	"math"
	"testing"

	"dappco.re/go/inference/kv"
)

// TestSami_SAMIFromKV_Good converts a real snapshot into the SAMI
// visualisation schema and confirms the per-layer coherence and
// cross-alignment arrays are built to the snapshot's layer count.
func TestSami_SAMIFromKV_Good(t *testing.T) {
	snapshot := bundleTestSnapshot()
	sami := SAMIFromKV(snapshot, nil, SAMIOptions{Model: "m", Prompt: "p"})
	if sami.Architecture != "gemma4_text" || sami.NumLayers != 1 {
		t.Fatalf("SAMI = %+v", sami)
	}
	if len(sami.LayerCoherence) != 1 || len(sami.LayerCrossAlignment) != 1 {
		t.Fatalf("SAMI layer arrays = coherence:%d cross:%d", len(sami.LayerCoherence), len(sami.LayerCrossAlignment))
	}
	if sami.Model != "m" || sami.Prompt != "p" {
		t.Fatalf("SAMI provenance = %q/%q", sami.Model, sami.Prompt)
	}
}

// TestSami_SAMIFromKV_Bad feeds analysis carrying NaN/Inf coherence and an
// over-count of joint collapses: SAMIFromKV must clamp every metric into
// the documented [0,1] (and composite [0,100]) range and cap the collapse
// count at the layer count rather than leak the poisoned values through.
func TestSami_SAMIFromKV_Bad(t *testing.T) {
	snapshot := bundleTestSnapshot()
	analysis := &kv.Analysis{
		MeanKeyCoherence:    math.NaN(),
		MeanValueCoherence:  math.Inf(1),
		MeanCrossAlignment:  2.0,
		MeanHeadEntropy:     -1.0,
		PhaseLockScore:      math.Inf(-1),
		JointCollapseCount:  99,
		LayerKeyCoherence:   []float64{math.NaN()},
		LayerValueCoherence: []float64{2.0},
		LayerCrossAlignment: []float64{-5.0},
	}
	sami := SAMIFromKV(snapshot, analysis, SAMIOptions{Model: "m"})
	for _, v := range sami.LayerCoherence {
		if v < 0 || v > 1 || math.IsNaN(v) {
			t.Fatalf("LayerCoherence out of range: %v", v)
		}
	}
	for _, v := range sami.LayerCrossAlignment {
		if v < 0 || v > 1 || math.IsNaN(v) {
			t.Fatalf("LayerCrossAlignment out of range: %v", v)
		}
	}
	if sami.MeanCoherence < 0 || sami.MeanCoherence > 1 {
		t.Fatalf("MeanCoherence not clamped: %v", sami.MeanCoherence)
	}
	if sami.MeanHeadEntropy < 0 || sami.PhaseLockScore < 0 {
		t.Fatalf("entropy/phase not clamped: %v/%v", sami.MeanHeadEntropy, sami.PhaseLockScore)
	}
	if sami.Composite < 0 || sami.Composite > 100 {
		t.Fatalf("Composite out of range: %v", sami.Composite)
	}
	if sami.JointCollapseCount > sami.NumLayers {
		t.Fatalf("JointCollapseCount = %d exceeds NumLayers %d", sami.JointCollapseCount, sami.NumLayers)
	}
}

// TestSami_SAMIFromKV_Ugly drives the nil-snapshot boundary: SAMIFromKV
// returns a zero-valued result rather than panicking or allocating layer
// arrays.
func TestSami_SAMIFromKV_Ugly(t *testing.T) {
	got := SAMIFromKV(nil, nil, SAMIOptions{})
	if got.Architecture != "" || got.NumLayers != 0 || len(got.LayerCoherence) != 0 || len(got.LayerCrossAlignment) != 0 {
		t.Fatalf("SAMIFromKV(nil) = %+v, want zero", got)
	}
}
