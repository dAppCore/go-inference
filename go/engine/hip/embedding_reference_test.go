// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestEmbeddingReferenceMeanPool_Good(t *testing.T) {
	vector, err := rocmReferenceMeanPoolEmbedding([][]float32{{1, 3}, {3, 5}}, false)

	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{2, 4}, vector, 0)
}

func TestEmbeddingReferenceMeanPool_Good_Normalizes(t *testing.T) {
	vector, err := rocmReferenceMeanPoolEmbedding([][]float32{{3, 4}}, true)

	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{0.6, 0.8}, vector, 0.0001)
}

func TestEmbeddingReferenceMeanPool_Bad_RejectsEmptyTokens(t *testing.T) {
	_, err := rocmReferenceMeanPoolEmbedding(nil, false)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "required")
}

func TestEmbeddingReferenceMeanPool_Bad_RejectsEmptyDimension(t *testing.T) {
	_, err := rocmReferenceMeanPoolEmbedding([][]float32{{}}, false)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "dimension")
}

func TestEmbeddingReferenceMeanPool_Bad_RejectsMismatchedDimensions(t *testing.T) {
	_, err := rocmReferenceMeanPoolEmbedding([][]float32{{1, 2}, {3}}, false)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "dimension")
}

func TestEmbeddingReferenceMeanPool_Bad_RejectsZeroVectorNormalization(t *testing.T) {
	_, err := rocmReferenceMeanPoolEmbedding([][]float32{{0, 0}}, true)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "zero vector")
}

func TestEmbeddingReferenceL2Normalize_Bad_RejectsZeroVector(t *testing.T) {
	_, err := rocmReferenceL2Normalize([]float32{0, 0})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "zero vector")
}

func TestRerankReferenceCosine_Good(t *testing.T) {
	score, err := rocmReferenceCosineSimilarity([]float32{1, 1}, []float32{1, 0})

	core.RequireNoError(t, err)
	assertFloat64Near(t, 0.7071, score, 0.0001)
}

func TestRerankReference_Good_CosineTopN(t *testing.T) {
	results, err := rocmReferenceRerank(
		[]float32{1, 0},
		[][]float32{{0, 1}, {1, 1}, {1, 0}},
		[]string{"orthogonal", "mixed", "exact"},
		2,
	)

	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, len(results))
	core.AssertEqual(t, 2, results[0].Index)
	core.AssertEqual(t, "exact", results[0].Text)
	assertFloat32Near(t, 1, float32(results[0].Score))
	core.AssertEqual(t, 1, results[1].Index)
}

func TestRerankReference_Good_TieBreaksByOriginalIndex(t *testing.T) {
	results, err := rocmReferenceRerank(
		[]float32{1, 0},
		[][]float32{{1, 0}, {1, 0}},
		nil,
		0,
	)

	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, results[0].Index)
	core.AssertEqual(t, 1, results[1].Index)
}

func TestRerankReference_Good_NegativeTopNReturnsAll(t *testing.T) {
	results, err := rocmReferenceRerank(
		[]float32{1, 0},
		[][]float32{{1, 0}, {0, 1}},
		nil,
		-1,
	)

	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, len(results))
}

func TestRerankReference_Bad_RejectsEmptyDocuments(t *testing.T) {
	_, err := rocmReferenceRerank([]float32{1}, nil, nil, 0)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "documents")
}

func TestRerankReference_Bad_RejectsMismatchedDocumentTexts(t *testing.T) {
	_, err := rocmReferenceRerank([]float32{1}, [][]float32{{1}}, []string{"one", "two"}, 0)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "text count")
}

func TestRerankReference_Bad_RejectsEmptyVectors(t *testing.T) {
	_, err := rocmReferenceRerank([]float32{1}, [][]float32{{}}, nil, 0)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "vectors")
}

func TestRerankReference_Bad_RejectsMismatchedVectorWidths(t *testing.T) {
	_, err := rocmReferenceRerank([]float32{1, 0}, [][]float32{{1}}, nil, 0)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "vectors")
}

func TestRerankReference_Bad_RejectsZeroVectors(t *testing.T) {
	_, err := rocmReferenceRerank([]float32{1, 0}, [][]float32{{0, 0}}, nil, 0)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "zero vector")
}

func TestHIPLoadedModelIdentity_Good_UsesEngineProfileLabelsAndContext(t *testing.T) {
	model := &hipLoadedModel{
		modelInfo: inference.ModelInfo{
			Architecture: "gemma4_text",
			VocabSize:    ProductionMTPAssistantTokenOrderingVocabSize,
			NumLayers:    productionLaneGemma4E2BLayers,
			HiddenSize:   productionLaneGemma4E2BHiddenSize,
			QuantBits:    6,
			QuantGroup:   64,
		},
		contextSize: 8192,
		modelLabels: map[string]string{
			"gemma4_size":       "E2B",
			"gemma4_quant_mode": "q6",
			"runtime_label":     "loaded",
		},
		engineProfile: ROCmModelProfile{
			Model: inference.ModelIdentity{
				Path:   "/models/lmstudio-community-gemma-4-e2b-it-6bit",
				Labels: map[string]string{"profile_label": "kept", "runtime_label": "profile"},
			},
		},
	}

	identity := hipLoadedModelIdentity(model)
	if identity.Path != "/models/lmstudio-community-gemma-4-e2b-it-6bit" ||
		identity.Architecture != "gemma4_text" ||
		identity.ContextLength != 8192 ||
		identity.QuantBits != 6 ||
		identity.QuantGroup != 64 ||
		identity.QuantType != "q6" ||
		identity.Labels["profile_label"] != "kept" ||
		identity.Labels["runtime_label"] != "loaded" ||
		identity.Labels["gemma4_size"] != "E2B" ||
		identity.Labels["gemma4_quant_mode"] != "q6" ||
		identity.Labels["gemma4_generate_status"] == "" {
		t.Fatalf("hipLoadedModelIdentity = %+v, want loaded Gemma4 profile identity with context and labels", identity)
	}
	identity.Labels["runtime_label"] = "mutated"
	if next := hipLoadedModelIdentity(model); next.Labels["runtime_label"] == "mutated" {
		t.Fatalf("hipLoadedModelIdentity returned aliased labels: %+v", next.Labels)
	}
}
