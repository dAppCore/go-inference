// SPDX-Licence-Identifier: EUPL-1.2

package kv

import core "dappco.re/go"

// ExampleAnalyze_multiHead runs the multi-head coherence path (numHeads > 4)
// over a synthetic snapshot whose every head carries the same one-hot vector.
// Identical heads are perfectly coherent, so the non-GQA Composite saturates at
// the 10000 ceiling. The GQA flag stays false on the multi-head branch.
func ExampleAnalyze_multiHead() {
	snapshot := makeKVAnalysisCoherentSnapshot(3, 8, 4, 4)

	result := Analyze(snapshot)

	core.Println("gqa:", result.GQA)
	core.Println("composite:", result.Composite())
	// Output:
	// gqa: false
	// composite: 10000
}

// ExampleAnalyze_gqa runs the grouped-query path (numHeads <= 4) over the same
// coherent shape. The GQA branch reports GQA=true and a positive Composite, and
// Features always returns the 7-D model-state vector regardless of branch.
func ExampleAnalyze_gqa() {
	snapshot := makeKVAnalysisCoherentSnapshot(3, 4, 4, 4)

	result := Analyze(snapshot)

	core.Println("gqa:", result.GQA)
	core.Println("composite > 0:", result.Composite() > 0)
	core.Println("features:", len(Features(result)))
	// Output:
	// gqa: true
	// composite > 0: true
	// features: 7
}

// ExampleAnalyze_orthogonalHeads contrasts coherent heads with orthogonal ones:
// when every head points a different way the pairwise coherence collapses, so
// the orthogonal Composite scores strictly below the coherent one over the same
// shape. Demonstrates the score responding to cache posture.
func ExampleAnalyze_orthogonalHeads() {
	coherent := Analyze(makeKVAnalysisCoherentSnapshot(4, 8, 4, 4)).Composite()
	orthogonal := Analyze(makeKVAnalysisOrthogonalSnapshot(4, 8, 4, 4)).Composite()

	core.Println("orthogonal below coherent:", orthogonal < coherent)
	// Output:
	// orthogonal below coherent: true
}

// ExampleAnalyze_sharedCacheGroups shows the shared-cache grouping: two layers
// pointing at the same CacheIndex are bucketed together, while a uniquely
// indexed layer is dropped (groups only keep buckets of size >= 2). This is the
// sliding-window / global-attention layer-sharing signal.
func ExampleAnalyze_sharedCacheGroups() {
	head := []float32{1, 0, 0, 1}
	mk := func(layer, cacheIndex int) LayerSnapshot {
		return LayerSnapshot{
			Layer:      layer,
			CacheIndex: cacheIndex,
			Heads:      []HeadSnapshot{{Key: head, Value: head}},
		}
	}
	snapshot := &Snapshot{
		Version: SnapshotVersion, Architecture: "test",
		NumLayers: 3, NumHeads: 1, SeqLen: 2, HeadDim: 2,
		Tokens: []int32{1, 2},
		Layers: []LayerSnapshot{
			mk(0, 0), // shares cache 0 with layer 2
			mk(1, 1), // unique → dropped
			mk(2, 0), // shares cache 0 with layer 0
		},
	}

	result := Analyze(snapshot)

	core.Println("shared groups:", len(result.SharedCacheLayerGroups))
	core.Println("cache 0 members:", len(result.SharedCacheLayerGroups[0]))
	// Output:
	// shared groups: 1
	// cache 0 members: 2
}

// ExampleAnalyze_emptySnapshot covers the nil/empty guard: an analysis of a
// snapshot with no layers returns a zeroed Analysis. Its Composite is not zero
// — the joint-stability term has a baseline of 1.0 with no observed collapses,
// contributing its 0.05 weight (= 500) even when every coherence metric is 0.
// The feature vector is still the canonical 7-D vector.
func ExampleAnalyze_emptySnapshot() {
	result := Analyze(&Snapshot{})

	core.Println("composite:", result.Composite())
	core.Println("features:", len(Features(result)))
	core.Println("labels:", len(FeatureLabels()))
	// Output:
	// composite: 500
	// features: 7
	// labels: 7
}

// ExampleAnalysis_Composite shows Composite scoring a hand-built Analysis
// directly: the GQA weighting differs from the dense weighting, and a nil
// receiver scores 0. The values are clamped into the 0-10000 range.
func ExampleAnalysis_Composite() {
	dense := &Analysis{
		MeanKeyCoherence:   1,
		MeanValueCoherence: 1,
		MeanCrossAlignment: 1,
		PhaseLockScore:     1,
		MeanKVCoupling:     1,
		MeanHeadEntropy:    1,
	}
	gqa := *dense
	gqa.GQA = true

	core.Println("dense:", dense.Composite())
	core.Println("gqa:", gqa.Composite())
	core.Println("nil:", (*Analysis)(nil).Composite())
	// Output:
	// dense: 10000
	// gqa: 10000
	// nil: 0
}

// ExampleFeatures flattens an Analysis into the fixed 7-dimensional model-state
// vector used as a downstream feature input. A nil Analysis yields all zeros.
func ExampleFeatures() {
	result := &Analysis{MeanKeyCoherence: 0.1, MeanKVCoupling: 0.6}

	features := Features(result)
	core.Println("dimensions:", len(features))
	core.Println("key coherence:", features[0])
	// Output:
	// dimensions: 7
	// key coherence: 0.1
}

// ExampleFeatureLabels returns the stable axis names matching the Features
// vector order, so a feature index can be named.
func ExampleFeatureLabels() {
	labels := FeatureLabels()
	core.Println("labels:", len(labels))
	core.Println("first:", labels[0])
	// Output:
	// labels: 7
	// first: key_coherence
}
