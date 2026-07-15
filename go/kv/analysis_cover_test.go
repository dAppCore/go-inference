// SPDX-Licence-Identifier: EUPL-1.2

package kv

import "testing"

// makeKVAnalysisZeroSnapshot builds a snapshot whose head vectors are all
// zero, driving the degenerate-norm guards in the analysis math (zero sums,
// zero entropy, zero anchor components).
func makeKVAnalysisZeroSnapshot(layers, heads, seqLen, headDim int) *Snapshot {
	snapshot := &Snapshot{
		Version:      SnapshotVersion,
		Architecture: "test",
		Tokens:       make([]int32, seqLen),
		NumLayers:    layers,
		NumHeads:     heads,
		SeqLen:       seqLen,
		HeadDim:      headDim,
		Layers:       make([]LayerSnapshot, layers),
	}
	for layer := range layers {
		snapshot.Layers[layer] = LayerSnapshot{Layer: layer, CacheIndex: layer, Heads: make([]HeadSnapshot, heads)}
		for h := range heads {
			snapshot.Layers[layer].Heads[h] = HeadSnapshot{
				Key:   make([]float32, seqLen*headDim),
				Value: make([]float32, seqLen*headDim),
			}
		}
	}
	return snapshot
}

// TestAnalysisCover_ZeroVectors_MultiHead drives the all-zero degenerate
// branches of the multi-head analysis path: zero norms, zero entropy, zero
// position sums. Analyze must not panic and returns a populated result.
func TestAnalysisCover_ZeroVectors_MultiHead(t *testing.T) {
	result := Analyze(makeKVAnalysisZeroSnapshot(4, 8, 4, 4))
	if result == nil {
		t.Fatal("Analyze(zero vectors) = nil")
	}
}

// TestAnalysisCover_ZeroVectors_GQA drives the GQA analysis path (single KV
// head) over all-zero vectors, covering the GQA-side degenerate branches.
func TestAnalysisCover_ZeroVectors_GQA(t *testing.T) {
	result := Analyze(makeKVAnalysisZeroSnapshot(4, 1, 4, 4))
	if result == nil {
		t.Fatal("Analyze(zero GQA) = nil")
	}
	if !result.GQA {
		t.Fatal("GQA = false, want true for single KV head")
	}
}

// TestAnalysisCover_SinglePosition drives the count==0 / pairs==0 guards: a
// single-token snapshot has no position pairs to differentiate, so the
// per-position coherence accumulators stay empty.
func TestAnalysisCover_SinglePosition(t *testing.T) {
	if result := Analyze(makeKVAnalysisCoherentSnapshot(2, 8, 1, 4)); result == nil {
		t.Fatal("Analyze(single position, multi-head) = nil")
	}
	if result := Analyze(makeKVAnalysisCoherentSnapshot(2, 1, 1, 4)); result == nil {
		t.Fatal("Analyze(single position, GQA) = nil")
	}
}

// TestAnalysisCover_NonAlignedHeadDim drives the scalar remainder loops of the
// cosine/coherence kernels: a head vector length (seqLen × headDim) that is not
// a multiple of 4 leaves a tail the unrolled-by-4 loop cannot consume. seqLen 3
// × headDim 2 = 6 → a 2-element remainder.
func TestAnalysisCover_NonAlignedHeadDim(t *testing.T) {
	if result := Analyze(makeKVAnalysisCoherentSnapshot(2, 8, 3, 2)); result == nil {
		t.Fatal("Analyze(len 6, multi-head) = nil")
	}
	if result := Analyze(makeKVAnalysisCoherentSnapshot(2, 1, 3, 2)); result == nil {
		t.Fatal("Analyze(len 6, GQA) = nil")
	}
	// seqLen 1 makes the per-head vector length == headDim; headDim 6 → a
	// 2-element remainder, and a single position drives the count/pairs guards.
	if result := Analyze(makeKVAnalysisCoherentSnapshot(2, 8, 1, 6)); result == nil {
		t.Fatal("Analyze(single position, len 6) = nil")
	}
}

// TestAnalysisCover_CrossLayerCollapse drives the JointCollapseCount increment
// in the GQA cross-layer alignment (single KV head, ≤4 heads → GQA path): layer
// 0 has identical positions (high coherence) and layer 1 has sign-alternating
// positions (low/negative coherence), so the layer-to-layer coherence delta
// exceeds 1 and the cross-layer smoothness falls below the collapse threshold.
func TestAnalysisCover_CrossLayerCollapse(t *testing.T) {
	snapshot := &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "test",
		Tokens:        make([]int32, 4),
		NumLayers:     2,
		NumHeads:      1,
		SeqLen:        4,
		HeadDim:       1,
		NumQueryHeads: 4,
		Layers: []LayerSnapshot{
			// Layer 0: identical positions → high position coherence.
			{Layer: 0, CacheIndex: 0, Heads: []HeadSnapshot{{
				Key:   []float32{1, 1, 1, 1},
				Value: []float32{1, 1, 1, 1},
			}}},
			// Layer 1: sign-alternating positions → strongly anti-correlated.
			{Layer: 1, CacheIndex: 1, Heads: []HeadSnapshot{{
				Key:   []float32{1, -1, 1, -1},
				Value: []float32{1, -1, 1, -1},
			}}},
		},
	}
	result := Analyze(snapshot)
	if result == nil {
		t.Fatal("Analyze(cross-layer collapse) = nil")
	}
	if !result.GQA {
		t.Fatal("GQA = false, want true for single KV head")
	}
}

// TestAnalysisCover_GQAHeadDimOne drives the GQA position-differentiation path
// with headDim 1, including the ai == 0 zero-component shortcut when a position
// vector is zero.
func TestAnalysisCover_GQAHeadDimOne(t *testing.T) {
	// Single KV head, headDim 1, with one zero position so ai == 0 fires.
	snapshot := &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "test",
		Tokens:        make([]int32, 4),
		NumLayers:     2,
		NumHeads:      1,
		SeqLen:        4,
		HeadDim:       1,
		NumQueryHeads: 4,
		Layers:        make([]LayerSnapshot, 2),
	}
	for layer := range 2 {
		snapshot.Layers[layer] = LayerSnapshot{Layer: layer, CacheIndex: layer, Heads: []HeadSnapshot{{
			Key:   []float32{1, 0, 1, 1}, // position 1 is zero → ai == 0
			Value: []float32{1, 1, 0, 1},
		}}}
	}
	if result := Analyze(snapshot); result == nil {
		t.Fatal("Analyze(GQA headDim 1, zero position) = nil")
	}
}

// TestAnalysisCover_SingleSeqLenEntropy drives the maxEntropy == 0 guard: a
// snapshot with seqLen 1 has log2(1) == 0 max entropy.
func TestAnalysisCover_SingleSeqLenEntropy(t *testing.T) {
	snapshot := &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "test",
		Tokens:        make([]int32, 1),
		NumLayers:     2,
		NumHeads:      1,
		SeqLen:        1,
		HeadDim:       4,
		NumQueryHeads: 1,
		Layers: []LayerSnapshot{
			{Layer: 0, CacheIndex: 0, Heads: []HeadSnapshot{{Key: []float32{1, 2, 3, 4}, Value: []float32{5, 6, 7, 8}}}},
			{Layer: 1, CacheIndex: 1, Heads: []HeadSnapshot{{Key: []float32{1, 2, 3, 4}, Value: []float32{5, 6, 7, 8}}}},
		},
	}
	if result := Analyze(snapshot); result == nil {
		t.Fatal("Analyze(seqLen 1) = nil")
	}
}

// TestAnalysisCover_DivergentHeadShapes drives the count == 0 / size == 0 guards
// of kvAnalysisLayerState and kvAnalysisLayerCoupling: a layer whose heads carry
// mismatched or empty Key/Value lengths so no head contributes a mean vector.
func TestAnalysisCover_DivergentHeadShapes(t *testing.T) {
	// First head sets the size; the rest diverge so count stays 0 after it,
	// and an all-empty layer drives the size == 0 / count == 0 arms.
	empty := &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "test",
		Tokens:        make([]int32, 2),
		NumLayers:     1,
		NumHeads:      2,
		SeqLen:        2,
		HeadDim:       2,
		NumQueryHeads: 2,
		Layers: []LayerSnapshot{{Layer: 0, CacheIndex: 0, Heads: []HeadSnapshot{
			{Key: nil, Value: nil}, // empty
			{Key: nil, Value: nil}, // empty → size stays 0
		}}},
	}
	if result := Analyze(empty); result == nil {
		t.Fatal("Analyze(empty heads) = nil")
	}
}

// TestAnalysisCover_ShortHeadVectors drives the start >= len(head) guard in the
// entropy walk: a head whose backing slice is shorter than seqLen*headDim, so a
// later position's window starts past the end of the data.
func TestAnalysisCover_ShortHeadVectors(t *testing.T) {
	snapshot := &Snapshot{
		Version:      SnapshotVersion,
		Architecture: "test",
		Tokens:       make([]int32, 4),
		NumLayers:    1,
		NumHeads:     1,
		SeqLen:       4,
		HeadDim:      4,
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []HeadSnapshot{{
				// Only two positions' worth of data for a seqLen-4 head → the
				// entropy/position walk runs off the end and the guard fires.
				Key:   make([]float32, 2*4),
				Value: make([]float32, 2*4),
			}},
		}},
	}
	if result := Analyze(snapshot); result == nil {
		t.Fatal("Analyze(short head vectors) = nil")
	}
}
