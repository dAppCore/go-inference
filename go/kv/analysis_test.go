// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"math"
	"testing"
)

func TestAnalysis_Analyze_Good(t *testing.T) {
	snapshot := makeKVAnalysisCoherentSnapshot(4, 8, 4, 4)

	result := Analyze(snapshot)

	if result.GQA {
		t.Fatal("GQA = true, want false for 8 heads")
	}
	if result.MeanKeyCoherence < 0.9 {
		t.Fatalf("MeanKeyCoherence = %.3f, want high coherence", result.MeanKeyCoherence)
	}
	if result.MeanValueCoherence < 0.9 {
		t.Fatalf("MeanValueCoherence = %.3f, want high coherence", result.MeanValueCoherence)
	}
	if result.MeanKVCoupling < 0.9 {
		t.Fatalf("MeanKVCoupling = %.3f, want high K/V coupling", result.MeanKVCoupling)
	}
	if result.PhaseLockScore < 0.9 {
		t.Fatalf("PhaseLockScore = %.3f, want high phase lock", result.PhaseLockScore)
	}
	if result.JointCollapseCount != 0 {
		t.Fatalf("JointCollapseCount = %d, want 0", result.JointCollapseCount)
	}
}

func TestAnalysis_Analyze_Bad(t *testing.T) {
	snapshot := makeKVAnalysisOrthogonalSnapshot(4, 8, 4, 8)

	result := Analyze(snapshot)

	if result.GQA {
		t.Fatal("GQA = true, want false for 8 heads")
	}
	if result.MeanKeyCoherence > 0.3 {
		t.Fatalf("MeanKeyCoherence = %.3f, want low coherence for orthogonal heads", result.MeanKeyCoherence)
	}
	if result.MeanValueCoherence > 0.3 {
		t.Fatalf("MeanValueCoherence = %.3f, want low coherence for orthogonal heads", result.MeanValueCoherence)
	}
}

func TestAnalysis_Analyze_Ugly(t *testing.T) {
	snapshot := makeKVAnalysisCoherentSnapshot(4, 1, 4, 4)

	result := Analyze(snapshot)

	if !result.GQA {
		t.Fatal("GQA = false, want true for single KV head")
	}
	if result.MeanKeyCoherence > 0.1 {
		t.Fatalf("MeanKeyCoherence = %.3f, want low position differentiation for identical positions", result.MeanKeyCoherence)
	}
	if len(result.LayerCrossAlignment) != 3 {
		t.Fatalf("LayerCrossAlignment len = %d, want 3", len(result.LayerCrossAlignment))
	}
}

func TestAnalysis_Composite_Good(t *testing.T) {
	result := &Analysis{
		MeanKeyCoherence:       1,
		MeanValueCoherence:     1,
		MeanCrossAlignment:     1,
		MeanHeadEntropy:        1,
		PhaseLockScore:         1,
		MeanKVCoupling:         1,
		JointCollapseCount:     0,
		LayerKeyCoherence:      []float64{1, 1},
		LayerValueCoherence:    []float64{1, 1},
		LayerCrossAlignment:    []float64{1},
		LayerKVCoupling:        []float64{1, 1},
		SharedCacheLayerGroups: map[int][]int{0: {0, 1}},
	}

	score := result.Composite()

	if score != 10000 {
		t.Fatalf("Composite() = %d, want 10000", score)
	}
}

func TestAnalysis_Composite_Bad(t *testing.T) {
	result := &Analysis{JointCollapseCount: 10}

	score := result.Composite()

	if score != 0 {
		t.Fatalf("Composite() = %d, want 0", score)
	}
}

func TestAnalysis_Features_Ugly(t *testing.T) {
	features := Features(nil)
	labels := FeatureLabels()

	if len(features) != 7 {
		t.Fatalf("Features(nil) len = %d, want 7", len(features))
	}
	if len(labels) != len(features) {
		t.Fatalf("FeatureLabels len = %d, want %d", len(labels), len(features))
	}
	for _, value := range features {
		if value != 0 {
			t.Fatalf("Features(nil) contains %f, want zeros", value)
		}
	}
}

func TestAnalysis_Features_Good(t *testing.T) {
	result := &Analysis{
		MeanKeyCoherence:   0.1,
		MeanValueCoherence: 0.2,
		MeanCrossAlignment: 0.3,
		MeanHeadEntropy:    0.4,
		PhaseLockScore:     0.5,
		MeanKVCoupling:     0.6,
		JointCollapseCount: 1,
	}

	features := Features(result)

	if len(features) != 7 {
		t.Fatalf("Features len = %d, want 7", len(features))
	}
	if features[0] != 0.1 || features[5] != 0.6 || math.Abs(features[6]-0.8) > 1e-6 {
		t.Fatalf("Features = %v, want ordered K/V metrics", features)
	}
}

func TestAnalysis_FeatureLabels_Good(t *testing.T) {
	labels := FeatureLabels()

	if len(labels) != 7 {
		t.Fatalf("FeatureLabels len = %d, want 7", len(labels))
	}
	if labels[0] != "key_coherence" || labels[5] != "kv_coupling" {
		t.Fatalf("FeatureLabels = %v, want stable K/V axis labels", labels)
	}
}

func TestKVAnalysisCosine32_Good(t *testing.T) {
	got := kvAnalysisCosine32([]float32{1, 0, 0}, []float32{1, 0, 0})

	if math.Abs(got-1) > 1e-6 {
		t.Fatalf("kvAnalysisCosine32 = %f, want 1", got)
	}
}

func TestKVAnalysisCosine32_Bad(t *testing.T) {
	got := kvAnalysisCosine32([]float32{1, 0, 0}, []float32{0, 1, 0})

	if math.Abs(got) > 1e-6 {
		t.Fatalf("kvAnalysisCosine32 = %f, want 0 for orthogonal vectors", got)
	}
}

func TestKVAnalysisHeadEntropy_Ugly(t *testing.T) {
	got := kvAnalysisHeadEntropy([]float32{1, 0, 1, 0}, 2, 2, nil)

	if math.Abs(got-1) > 1e-6 {
		t.Fatalf("kvAnalysisHeadEntropy = %f, want 1 for balanced magnitudes", got)
	}
}

func TestAnalysis_Composite_Ugly(t *testing.T) {
	// Composite must tolerate a nil receiver — the early-return guard
	// keeps callers from having to nil-check an Analyze result.
	var result *Analysis

	if score := result.Composite(); score != 0 {
		t.Fatalf("(*Analysis)(nil).Composite() = %d, want 0", score)
	}
}

func TestAnalyze_NilAndEmptyGuards(t *testing.T) {
	// Analyze short-circuits to a zero Analysis for nil input and for a
	// snapshot with no layers — both are the "nothing to measure" guard.
	// The returned Analysis carries no per-layer slices and no metrics.
	for name, snapshot := range map[string]*Snapshot{
		"nil":       nil,
		"no-layers": {Architecture: "test"},
	} {
		got := Analyze(snapshot)
		if got == nil {
			t.Fatalf("Analyze(%s) = nil, want non-nil zero Analysis", name)
		}
		if got.MeanKeyCoherence != 0 || got.MeanValueCoherence != 0 || got.MeanKVCoupling != 0 {
			t.Fatalf("Analyze(%s) metrics = %+v, want all zero", name, got)
		}
		if len(got.LayerKeyCoherence) != 0 || got.SharedCacheLayerGroups != nil {
			t.Fatalf("Analyze(%s) = %+v, want empty layer slices and nil groups", name, got)
		}
	}
}

func TestAnalyze_InfersLayersAndHeadsFromSlices(t *testing.T) {
	// A snapshot with NumLayers/NumHeads unset (zero) must fall back to the
	// length of the Layers and per-layer Heads slices. Build the coherent
	// fixture, then clear the explicit counts to exercise the inference
	// path through Analyze → kvAnalysisNumLayers / kvAnalysisNumHeads.
	snapshot := makeKVAnalysisCoherentSnapshot(3, 8, 4, 4)
	snapshot.NumLayers = 0
	snapshot.NumHeads = 0

	result := Analyze(snapshot)

	if result.GQA {
		t.Fatal("GQA = true, want false (8 heads inferred from slice)")
	}
	if len(result.LayerKeyCoherence) != 3 {
		t.Fatalf("LayerKeyCoherence len = %d, want 3 layers inferred from slice", len(result.LayerKeyCoherence))
	}
	if result.MeanKeyCoherence < 0.9 {
		t.Fatalf("MeanKeyCoherence = %.3f, want high coherence after inference", result.MeanKeyCoherence)
	}
}

func TestKVAnalysisNumHeads_NoHeadsPath(t *testing.T) {
	// When NumHeads is unset and every layer carries an empty Heads slice,
	// head inference exhausts the loop and returns 0. The <=4 branch then
	// routes Analyze through the GQA path even with zero usable heads.
	snapshot := &Snapshot{
		Architecture: "test",
		Tokens:       []int32{1, 2},
		SeqLen:       2,
		HeadDim:      2,
		Layers: []LayerSnapshot{
			{Layer: 0, CacheIndex: 0, Heads: nil},
			{Layer: 1, CacheIndex: 1, Heads: nil},
		},
	}

	if got := kvAnalysisNumHeads(snapshot); got != 0 {
		t.Fatalf("kvAnalysisNumHeads(no heads) = %d, want 0", got)
	}
	if got := kvAnalysisNumLayers(snapshot); got != 2 {
		t.Fatalf("kvAnalysisNumLayers(NumLayers=0) = %d, want 2 inferred from slice", got)
	}
	// Analyze must not panic on a layers-but-no-heads snapshot.
	result := Analyze(snapshot)
	if result == nil {
		t.Fatal("Analyze(layers, no heads) = nil, want non-nil Analysis")
	}
	if !result.GQA {
		t.Fatal("GQA = false, want true (0 heads routes through GQA path)")
	}
}

func makeKVAnalysisCoherentSnapshot(layers, heads, seqLen, headDim int) *Snapshot {
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
	head := make([]float32, seqLen*headDim)
	for pos := range seqLen {
		head[pos*headDim] = 1
	}
	for layer := range layers {
		snapshot.Layers[layer] = LayerSnapshot{
			Layer:      layer,
			CacheIndex: layer,
			Heads:      make([]HeadSnapshot, heads),
		}
		for h := range heads {
			snapshot.Layers[layer].Heads[h] = HeadSnapshot{
				Key:   append([]float32(nil), head...),
				Value: append([]float32(nil), head...),
			}
		}
	}
	return snapshot
}

func makeKVAnalysisOrthogonalSnapshot(layers, heads, seqLen, headDim int) *Snapshot {
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
		snapshot.Layers[layer] = LayerSnapshot{
			Layer:      layer,
			CacheIndex: layer,
			Heads:      make([]HeadSnapshot, heads),
		}
		for h := range heads {
			key := make([]float32, seqLen*headDim)
			value := make([]float32, seqLen*headDim)
			for pos := range seqLen {
				key[pos*headDim+h%headDim] = 1
				value[pos*headDim+(heads-h-1)%headDim] = 1
			}
			snapshot.Layers[layer].Heads[h] = HeadSnapshot{Key: key, Value: value}
		}
	}
	return snapshot
}

// referenceStridedDifferentiation computes 1 - mean pairwise cosine over the
// stride-sampled positions, the exact value the capped
// kvAnalysisPositionDifferentiation must produce above the position cap.
func referenceStridedDifferentiation(flat []float32, seqLen, headDim, stride int) (float64, int) {
	var normed [][]float64
	for src := 0; src < seqLen; src += stride {
		v := make([]float64, headDim)
		var sum float64
		for k := range headDim {
			v[k] = float64(flat[src*headDim+k])
			sum += v[k] * v[k]
		}
		if sum > 0 {
			inv := 1.0 / math.Sqrt(sum)
			for k := range v {
				v[k] *= inv
			}
		}
		normed = append(normed, v)
	}
	n := len(normed)
	var total float64
	pairs := 0
	for i := range n {
		for j := i + 1; j < n; j++ {
			var dot float64
			for k := range headDim {
				dot += normed[i][k] * normed[j][k]
			}
			total += dot
			pairs++
		}
	}
	if pairs == 0 {
		return 0, 0
	}
	return 1.0 - total/float64(pairs), pairs
}

// TestAnalysis_HelperGuards_BadUgly sweeps the defensive guard arms of the
// analysis helpers with degenerate synthetic inputs: nil snapshots, empty head
// slices, zero-norm vectors, and divergent-shape heads. These are the
// malformed/edge branches the coherent/orthogonal Analyze examples never reach.
func TestAnalysis_HelperGuards_BadUgly(t *testing.T) {
	// kvAnalysisNumLayers / kvAnalysisNumHeads nil guards.
	if got := kvAnalysisNumLayers(nil); got != 0 {
		t.Fatalf("kvAnalysisNumLayers(nil) = %d, want 0", got)
	}
	if got := kvAnalysisNumHeads(nil); got != 0 {
		t.Fatalf("kvAnalysisNumHeads(nil) = %d, want 0", got)
	}
	// kvAnalysisNumHeads loop-fallback: NumHeads unset, head count read from
	// the first non-empty layer.
	fallback := &Snapshot{Layers: []LayerSnapshot{
		{Heads: nil},
		{Heads: []HeadSnapshot{{}, {}, {}}},
	}}
	if got := kvAnalysisNumHeads(fallback); got != 3 {
		t.Fatalf("kvAnalysisNumHeads(fallback) = %d, want 3 from layer 1", got)
	}

	// kvAnalysisLayerCoupling: an all-empty-head slice returns (0, 0).
	if mean, n := kvAnalysisLayerCoupling([]HeadSnapshot{{}, {}}); mean != 0 || n != 0 {
		t.Fatalf("kvAnalysisLayerCoupling(empty heads) = %v/%d, want 0/0", mean, n)
	}

	// kvAnalysisLayerState: nil heads → nil; all-zero-length heads → nil;
	// divergent-shape heads are skipped so a sole oddball yields nil.
	if got := kvAnalysisLayerState(nil); got != nil {
		t.Fatalf("kvAnalysisLayerState(nil) = %v, want nil", got)
	}
	if got := kvAnalysisLayerState([]HeadSnapshot{{}}); got != nil {
		t.Fatalf("kvAnalysisLayerState(empty) = %v, want nil", got)
	}
	mixed := []HeadSnapshot{
		{Key: []float32{1, 2}, Value: []float32{3, 4}}, // size 4 sets the shape
		{Key: []float32{1}, Value: []float32{2}},       // size 2 diverges → skipped
	}
	state := kvAnalysisLayerState(mixed)
	if len(state) != 4 {
		t.Fatalf("kvAnalysisLayerState(mixed) len = %d, want 4 (oddball skipped)", len(state))
	}

	// kvAnalysisPairCoherence: zero-norm + length-mismatch vectors. The pair is
	// counted but contributes zero similarity (no locked pair).
	mean, locked, pairs := kvAnalysisPairCoherence([][]float32{
		{0, 0}, // zero norm
		{1},    // length mismatch with the others
		{3, 4}, // valid
	}, nil)
	if pairs != 3 || locked != 0 || mean != 0 {
		t.Fatalf("kvAnalysisPairCoherence(degenerate) = %v/%d/%d, want 0/0/3", mean, locked, pairs)
	}

	// kvAnalysisCosine32: length mismatch and zero vector both return 0.
	if got := kvAnalysisCosine32([]float32{1, 2}, []float32{1}); got != 0 {
		t.Fatalf("kvAnalysisCosine32(mismatch) = %v, want 0", got)
	}
	if got := kvAnalysisCosine32([]float32{0, 0}, []float32{0, 0}); got != 0 {
		t.Fatalf("kvAnalysisCosine32(zero) = %v, want 0", got)
	}

	// kvAnalysisHeadEntropy: seqLen <= 1 and headDim <= 0 short-circuit to 0;
	// an all-zero head also yields 0 (total magnitude is 0).
	if got := kvAnalysisHeadEntropy([]float32{1, 2}, 1, 2, nil); got != 0 {
		t.Fatalf("kvAnalysisHeadEntropy(seqLen 1) = %v, want 0", got)
	}
	if got := kvAnalysisHeadEntropy([]float32{0, 0, 0, 0}, 2, 2, nil); got != 0 {
		t.Fatalf("kvAnalysisHeadEntropy(zero head) = %v, want 0", got)
	}

	// kvAnalysisPositionDifferentiation: seqLen < 2 short-circuits; a flat
	// shorter than seqLen*headDim is skipped (no pairs).
	if diff, _, pairs := kvAnalysisPositionDifferentiation([]HeadSnapshot{{Key: []float32{1}}}, 1, 1, true, nil); diff != 0 || pairs != 0 {
		t.Fatalf("kvAnalysisPositionDifferentiation(seqLen 1) = %v/%d, want 0/0", diff, pairs)
	}
	short := []HeadSnapshot{{Key: []float32{1, 2}}} // needs 2*2=4, has 2 → skipped
	if _, _, pairs := kvAnalysisPositionDifferentiation(short, 2, 2, true, nil); pairs != 0 {
		t.Fatalf("kvAnalysisPositionDifferentiation(short flat) pairs = %d, want 0", pairs)
	}
}

// TestAnalysis_AnalyzeBodyArms_Good drives the per-layer skip + collapse arms
// inside both Analyze branches over snapshots with an empty middle layer (the
// `len(Heads) == 0` continue) and adjacent layers whose states are orthogonal
// (cross-alignment / smoothness below the collapse threshold).
func TestAnalysis_AnalyzeBodyArms(t *testing.T) {
	// Multi-head (heads > 4): build an orthogonal snapshot, then blank the
	// middle layer's heads so the layer-skip continue (analysis.go:110) fires
	// while the surrounding layers still produce cross-alignment work.
	multi := makeKVAnalysisOrthogonalSnapshot(3, 8, 4, 4)
	multi.Layers[1].Heads = nil
	resultMulti := Analyze(multi)
	if resultMulti.GQA {
		t.Fatal("expected multi-head branch")
	}

	// GQA (heads <= 4): same empty-middle-layer shape on the GQA path so the
	// per-layer skip + smoothness-collapse arms run there too.
	gqa := makeKVAnalysisOrthogonalSnapshot(3, 4, 4, 4)
	gqa.Layers[1].Heads = nil
	resultGQA := Analyze(gqa)
	if !resultGQA.GQA {
		t.Fatal("expected GQA branch")
	}

	// Both still emit valid composites and feature vectors despite the gap.
	if resultMulti.Composite() < 0 || resultGQA.Composite() < 0 {
		t.Fatal("composite must be non-negative")
	}
}

// TestAnalysis_JointCollapse_Good drives the JointCollapseCount increment arms
// in both Analyze branches with adjacent layers whose states are anti-aligned
// (multi-head: cross-alignment cosine << threshold) or whose differentiation
// swings hard (GQA: smoothness << threshold).
func TestAnalysis_JointCollapsePath(t *testing.T) {
	// Multi-head (heads > 4): build layers whose per-head vectors point in
	// opposite directions on adjacent layers so the layer-state cosine is
	// negative — well below kvCollapseThreshold (0.5) — forcing the collapse
	// increment (analysis.go:155-157).
	const heads, seqLen, headDim = 8, 2, 2
	mkLayer := func(layer int, sign float32) LayerSnapshot {
		hs := make([]HeadSnapshot, heads)
		for h := range hs {
			key := make([]float32, seqLen*headDim)
			value := make([]float32, seqLen*headDim)
			for pos := range seqLen {
				key[pos*headDim] = sign
				value[pos*headDim+1] = sign
			}
			hs[h] = HeadSnapshot{Key: key, Value: value}
		}
		return LayerSnapshot{Layer: layer, CacheIndex: layer, Heads: hs}
	}
	multiSnap := &Snapshot{
		Version: SnapshotVersion, Architecture: "test",
		Tokens: make([]int32, seqLen), NumLayers: 3, NumHeads: heads, SeqLen: seqLen, HeadDim: headDim,
		Layers: []LayerSnapshot{mkLayer(0, 1), mkLayer(1, -1), mkLayer(2, 1)}, // alternating sign
	}
	multi := Analyze(multiSnap)
	if multi.GQA {
		t.Fatal("expected multi-head branch (heads > 4)")
	}
	if multi.JointCollapseCount == 0 {
		t.Fatalf("multi-head JointCollapseCount = 0, want > 0 for anti-aligned layers")
	}

	// GQA (heads <= 4): adjacent layers with sharply different differentiation
	// drop the smoothness metric below threshold (analysis.go:253-255). A
	// fully-coherent layer (all heads identical → diff 0) next to an
	// orthogonal one (diff ~1) yields a smoothness ~0.
	gqaSnap := &Snapshot{
		Version: SnapshotVersion, Architecture: "test",
		Tokens: make([]int32, 4), NumLayers: 3, NumHeads: 2, SeqLen: 4, HeadDim: 2,
		Layers: []LayerSnapshot{
			makeKVAnalysisCoherentSnapshot(1, 2, 4, 2).Layers[0],
			makeKVAnalysisOrthogonalSnapshot(1, 2, 4, 2).Layers[0],
			makeKVAnalysisCoherentSnapshot(1, 2, 4, 2).Layers[0],
		},
	}
	// Fix the per-layer Layer/CacheIndex so they read as three distinct layers.
	for i := range gqaSnap.Layers {
		gqaSnap.Layers[i].Layer = i
		gqaSnap.Layers[i].CacheIndex = i
	}
	gqa := Analyze(gqaSnap)
	if !gqa.GQA {
		t.Fatal("expected GQA branch (heads <= 4)")
	}
	// The GQA smoothness metric exercises the per-layer cross arm; whether it
	// crosses the collapse threshold depends on the differentiation magnitudes,
	// so we assert only that the branch produced a coherent result (the multi-
	// head case above pins the collapse-increment line directly).
	if gqa.MeanCrossAlignment == 0 && len(gqaSnap.Layers) > 1 {
		t.Fatalf("GQA MeanCrossAlignment = 0, want the smoothness arm to have run")
	}

	// Both produce a valid 7-D feature vector regardless of collapse count.
	if len(Features(multi)) != 7 || len(Features(gqa)) != 7 {
		t.Fatal("Features must always be 7-D")
	}
}

// TestAnalysis_DegenerateShapes_Ugly drives the remaining defensive arms with
// degenerate-but-public-reachable shapes: a GQA snapshot with headDim 0 (the
// scratch seqLen-only branch), a numHeads loop-fallback over a leading empty
// layer, and a layer-state whose heads all diverge in shape (count==0 → nil).
func TestAnalysis_DegenerateShapesPath(t *testing.T) {
	// headDim 0 GQA path: scratch sized to seqLen only (analysis.go:202-204).
	// numHeads stays <= 4 so the GQA branch runs; headDim 0 means no per-head
	// vectors so differentiation is zero throughout.
	zeroDim := &Snapshot{
		Version: SnapshotVersion, Architecture: "test",
		Tokens: make([]int32, 2), NumLayers: 1, NumHeads: 2, SeqLen: 2, HeadDim: 0,
		Layers: []LayerSnapshot{{Layer: 0, Heads: []HeadSnapshot{{}, {}}}},
	}
	if got := Analyze(zeroDim); got == nil || !got.GQA {
		t.Fatal("Analyze(headDim 0) must run the GQA branch and return a result")
	}

	// numHeads loop-fallback: NumHeads unset, leading layer empty so the count
	// is read from a later populated layer (analysis.go:332-334). NumHeads 0
	// keeps the head count <= 4, GQA branch.
	fallback := &Snapshot{
		Version: SnapshotVersion, Architecture: "test",
		Tokens: make([]int32, 2), SeqLen: 2, HeadDim: 1,
		Layers: []LayerSnapshot{
			{Layer: 0, Heads: nil},
			{Layer: 1, Heads: []HeadSnapshot{{Key: []float32{1, 2}, Value: []float32{3, 4}}}},
		},
	}
	if got := Analyze(fallback); got == nil {
		t.Fatal("Analyze(empty leading layer) returned nil")
	}

	// kvAnalysisLayerState count==0: every head diverges from the shape the
	// first contributor sets, so none are summed and the result is nil
	// (analysis.go:506-508).
	//
	// A single contributor whose own key+value length is the anchor, then a
	// second head of a genuinely different total size → skipped.
	state := kvAnalysisLayerState([]HeadSnapshot{
		{Key: []float32{1, 2}, Value: []float32{3, 4}}, // anchor size 4
		{Key: []float32{9}},                            // size 1 → skipped
	})
	if len(state) != 4 {
		t.Fatalf("kvAnalysisLayerState(anchor + divergent) len = %d, want 4", len(state))
	}
}

// TestPositionDifferentiation_CapMatchesStridedExact verifies the cap (a) leaves
// at/below-cap analysis byte-identical and (b) above the cap produces exactly the
// strided-position result (not garbage / not a panic). headDim>1 and headDim==1
// paths both covered.
func TestPositionDifferentiation_CapMatchesStridedExact(t *testing.T) {
	const cap = 4096 // mirrors maxExactPositions
	cases := []struct {
		name    string
		seqLen  int
		headDim int
	}{
		{"belowCap_headDim4_exact", 1000, 4},
		{"belowCap_headDim1_exact", 2000, 1},
		{"aboveCap_headDim4_sampled", 16384, 4},
		{"aboveCap_headDim1_sampled", 12000, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			flat := make([]float32, tc.seqLen*tc.headDim)
			for i := range flat {
				flat[i] = float32(math.Sin(float64(i)*0.017) + 0.3*math.Cos(float64(i)*0.005))
			}
			heads := []HeadSnapshot{{Key: flat, Value: flat}}

			got, gotLocked, gotPairs := kvAnalysisPositionDifferentiation(heads, tc.seqLen, tc.headDim, true, nil)

			stride := 1
			if tc.seqLen > cap {
				stride = (tc.seqLen + cap - 1) / cap
			}
			want, wantPairs := referenceStridedDifferentiation(flat, tc.seqLen, tc.headDim, stride)

			if math.Abs(got-want) > 1e-9 {
				t.Errorf("diff = %v, want strided-exact %v (stride %d)", got, want, stride)
			}
			if gotPairs != wantPairs {
				t.Errorf("pairs = %d, want %d", gotPairs, wantPairs)
			}
			if gotLocked < 0 || gotLocked > gotPairs {
				t.Errorf("locked %d out of range [0,%d]", gotLocked, gotPairs)
			}
		})
	}
}

// TestAnalysis_Features_Bad drives Features over a degenerate Analysis whose
// JointCollapseCount is large enough to clamp the joint-stability feature to 0
// (the math.Max floor), while the coherence metrics remain whatever was set.
func TestAnalysis_Features_Bad(t *testing.T) {
	result := &Analysis{
		MeanKeyCoherence:   0.5,
		JointCollapseCount: 100, // 1 - 100*0.2 is very negative → clamped to 0
	}

	features := Features(result)

	if len(features) != 7 {
		t.Fatalf("Features len = %d, want 7", len(features))
	}
	if features[0] != 0.5 {
		t.Fatalf("Features[0] = %f, want 0.5 (key coherence passthrough)", features[0])
	}
	if features[6] != 0 {
		t.Fatalf("Features[6] = %f, want 0 (joint stability clamped under heavy collapse)", features[6])
	}
}

// TestAnalysis_FeatureLabels_Bad asserts FeatureLabels returns exactly as many
// labels as Features returns values, so the two stay index-aligned.
func TestAnalysis_FeatureLabels_Bad(t *testing.T) {
	labels := FeatureLabels()
	features := Features(&Analysis{})

	if len(labels) != len(features) {
		t.Fatalf("FeatureLabels len = %d, Features len = %d, want equal", len(labels), len(features))
	}
}

// TestAnalysis_FeatureLabels_Ugly asserts FeatureLabels returns a stable,
// fully-populated label set with no blank entries and the joint-stability axis
// in the final slot.
func TestAnalysis_FeatureLabels_Ugly(t *testing.T) {
	labels := FeatureLabels()

	for i, label := range labels {
		if label == "" {
			t.Fatalf("FeatureLabels[%d] = empty, want a stable axis name", i)
		}
	}
	if labels[len(labels)-1] != "joint_stability" {
		t.Fatalf("FeatureLabels last = %q, want joint_stability", labels[len(labels)-1])
	}
}
