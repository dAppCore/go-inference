// SPDX-Licence-Identifier: EUPL-1.2

package bundle

import (
	"math"
	"testing"

	"dappco.re/go/inference/kv"
)

// TestSami_SAMIFromKV_LayerCountFromLayers covers the NumLayers<=0 fallback
// in SAMIFromKV: when the snapshot leaves NumLayers unset, the layer count
// is derived from len(snapshot.Layers) instead of the header field.
func TestSami_SAMIFromKV_LayerCountFromLayers(t *testing.T) {
	snapshot := &kv.Snapshot{
		Version:      kv.SnapshotVersion,
		Architecture: "gemma4_text",
		NumLayers:    0, // force the len(Layers) fallback
		Layers: []kv.LayerSnapshot{
			{Layer: 0, Heads: []kv.HeadSnapshot{{Key: []float32{1, 0}, Value: []float32{0, 1}}}},
			{Layer: 1, Heads: []kv.HeadSnapshot{{Key: []float32{0, 1}, Value: []float32{1, 0}}}},
		},
	}
	result := SAMIFromKV(snapshot, nil, SAMIOptions{Model: "m"})
	if result.NumLayers != 2 {
		t.Fatalf("NumLayers = %d, want 2 (derived from len(Layers))", result.NumLayers)
	}
	if len(result.LayerCoherence) != 2 || len(result.LayerCrossAlignment) != 2 {
		t.Fatalf("layer arrays = %d/%d, want 2/2", len(result.LayerCoherence), len(result.LayerCrossAlignment))
	}
}

// TestSami_SAMIFromKV_ShortValueCoherence drives the inBounds shrink branch
// where LayerValueCoherence is shorter than the other per-layer slices, so
// the in-bounds prefix is clamped to the value-coherence length and the
// remaining layers fall through the fallback tail.
func TestSami_SAMIFromKV_ShortValueCoherence(t *testing.T) {
	snapshot := &kv.Snapshot{
		Version:      kv.SnapshotVersion,
		Architecture: "gemma4_text",
		NumLayers:    3,
	}
	// Analysis with a deliberately short LayerValueCoherence (len 1) but full
	// LayerKeyCoherence / LayerCrossAlignment (len 3). inBounds collapses to
	// 1, so layers 1 and 2 take the fallback path for value coherence.
	analysis := &kv.Analysis{
		MeanKeyCoherence:    0.8,
		MeanValueCoherence:  0.4,
		MeanCrossAlignment:  0.6,
		LayerKeyCoherence:   []float64{0.9, 0.8, 0.7},
		LayerValueCoherence: []float64{0.5}, // shorter than numLayers and the others
		LayerCrossAlignment: []float64{0.6, 0.5, 0.4},
	}
	result := SAMIFromKV(snapshot, analysis, SAMIOptions{Model: "m"})
	if result.NumLayers != 3 {
		t.Fatalf("NumLayers = %d, want 3", result.NumLayers)
	}
	if len(result.LayerCoherence) != 3 {
		t.Fatalf("LayerCoherence len = %d, want 3", len(result.LayerCoherence))
	}
	// Layer 0 is in-bounds: (clamp(0.9)+clamp(0.5))/2 = 0.7.
	if math.Abs(result.LayerCoherence[0]-0.7) > 1e-9 {
		t.Fatalf("LayerCoherence[0] = %v, want 0.7", result.LayerCoherence[0])
	}
	// Layer 1 takes value-coherence fallback (clampedFallbackValue=clamp(0.4)):
	// (clamp(0.8)+0.4)/2 = 0.6.
	if math.Abs(result.LayerCoherence[1]-0.6) > 1e-9 {
		t.Fatalf("LayerCoherence[1] = %v, want 0.6 (value fallback)", result.LayerCoherence[1])
	}
}

// TestSami_SAMIFromKV_ShortKeyAndAlign exercises the fallback-tail branches
// for key coherence and cross alignment independently: LayerKeyCoherence and
// LayerCrossAlignment are short while LayerValueCoherence is full, so the
// tail loop hits the "layer >= keyLen" and "layer >= alignLen" else-arms.
func TestSami_SAMIFromKV_ShortKeyAndAlign(t *testing.T) {
	snapshot := &kv.Snapshot{
		Version:      kv.SnapshotVersion,
		Architecture: "gemma4_text",
		NumLayers:    3,
	}
	analysis := &kv.Analysis{
		MeanKeyCoherence:    0.2,
		MeanValueCoherence:  0.6,
		MeanCrossAlignment:  0.3,
		LayerKeyCoherence:   []float64{0.9}, // short → key fallback on layers 1,2
		LayerValueCoherence: []float64{0.5, 0.4, 0.3},
		LayerCrossAlignment: []float64{0.8}, // short → align fallback on layers 1,2
	}
	result := SAMIFromKV(snapshot, analysis, SAMIOptions{Model: "m"})
	if len(result.LayerCrossAlignment) != 3 {
		t.Fatalf("LayerCrossAlignment len = %d, want 3", len(result.LayerCrossAlignment))
	}
	// Layer 1 cross alignment uses clampedFallbackAlign = clamp(0.3).
	if math.Abs(result.LayerCrossAlignment[1]-0.3) > 1e-9 {
		t.Fatalf("LayerCrossAlignment[1] = %v, want 0.3 (align fallback)", result.LayerCrossAlignment[1])
	}
	// Layer 1 coherence: key fallback clamp(0.2), value in-bounds clamp(0.4):
	// (0.2+0.4)/2 = 0.3.
	if math.Abs(result.LayerCoherence[1]-0.3) > 1e-9 {
		t.Fatalf("LayerCoherence[1] = %v, want 0.3 (key fallback)", result.LayerCoherence[1])
	}
}

// TestSami_layerMetric covers the standalone layerMetric helper directly:
// an in-range index clamps the indexed value, an out-of-range index (both
// negative and past-end) clamps the fallback instead.
func TestSami_layerMetric(t *testing.T) {
	values := []float64{0.5, 2.0, -1.0}
	// In-range, already in [0,1].
	if got := layerMetric(values, 0, 0.9); got != 0.5 {
		t.Fatalf("layerMetric(in-range 0) = %v, want 0.5", got)
	}
	// In-range but >1 → clamped to 1.
	if got := layerMetric(values, 1, 0.9); got != 1.0 {
		t.Fatalf("layerMetric(in-range 1, >1) = %v, want 1.0", got)
	}
	// Past-end index → fallback clamped.
	if got := layerMetric(values, 5, 0.3); got != 0.3 {
		t.Fatalf("layerMetric(past-end) = %v, want 0.3 (fallback)", got)
	}
	// Negative index → fallback clamped; fallback >1 clamps to 1.
	if got := layerMetric(values, -1, 1.5); got != 1.0 {
		t.Fatalf("layerMetric(negative, fallback>1) = %v, want 1.0", got)
	}
}
