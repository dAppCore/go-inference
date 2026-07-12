// SPDX-Licence-Identifier: EUPL-1.2

package bert_test

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/inference/model/bert"
)

// referenceFixture is the sentence-transformers gold dumped by
// testdata/dump_reference.py — the normalised CLS embeddings the host encoder
// must reproduce within cosine 0.999 per vector.
type referenceFixture struct {
	Model      string `json:"model"`
	Pooling    string `json:"pooling"`
	Normalize  bool   `json:"normalize"`
	HiddenSize int    `json:"hidden_size"`
	Records    []struct {
		Text      string    `json:"text"`
		InputIDs  []int32   `json:"input_ids"`
		Embedding []float32 `json:"embedding"`
	} `json:"records"`
}

// TestModel_Embed_BGESmallParity proves the host BERT forward matches
// sentence-transformers' BAAI/bge-small-en-v1.5 within cosine 0.999 per vector.
// It is opt-in: point BERT_PARITY_SNAPSHOT at the snapshot dir, or let it resolve
// the default Hugging Face cache location. Regenerate the fixture with
// testdata/dump_reference.py after any tokeniser or forward change.
func TestModel_Embed_BGESmallParity(t *testing.T) {
	snapshot := bgeSmallSnapshot(t)
	if snapshot == "" {
		t.Skip("bge-small-en-v1.5 snapshot not found; set BERT_PARITY_SNAPSHOT to run the parity receipt")
	}

	fixture := loadReferenceFixture(t)
	model, err := bert.Load(snapshot)
	if err != nil {
		t.Fatalf("bert.Load(%q): %v", snapshot, err)
	}
	if got := model.Config().HiddenSize; got != fixture.HiddenSize {
		t.Fatalf("hidden size = %d, want %d", got, fixture.HiddenSize)
	}
	if model.Pooling() != bert.PoolingCLS {
		t.Fatalf("pooling = %q, want cls (bge-small uses CLS pooling)", model.Pooling())
	}
	if !model.Normalises() {
		t.Fatal("model should normalise (bge-small ships a 2_Normalize module)")
	}

	const parityFloor = 0.999
	worst := 1.0
	for _, record := range fixture.Records {
		result, err := model.Embed(context.Background(), inference.EmbeddingRequest{Input: []string{record.Text}})
		if err != nil {
			t.Fatalf("Embed(%q): %v", record.Text, err)
		}
		if len(result.Vectors) != 1 {
			t.Fatalf("Embed(%q) returned %d vectors, want 1", record.Text, len(result.Vectors))
		}
		got := result.Vectors[0]
		if len(got) != len(record.Embedding) {
			t.Fatalf("Embed(%q) dim = %d, want %d", record.Text, len(got), len(record.Embedding))
		}
		sim := cosineSimilarity(got, record.Embedding)
		if sim < worst {
			worst = sim
		}
		t.Logf("cosine=%.6f  %q", sim, record.Text)
		if sim < parityFloor {
			t.Errorf("Embed(%q) cosine %.6f below parity floor %.3f", record.Text, sim, parityFloor)
		}
	}
	t.Logf("worst-case cosine across %d sentences: %.6f (floor %.3f)", len(fixture.Records), worst, parityFloor)
}

// TestModel_Rerank_BGESmallOrdering proves rerank orders documents by relevance
// to the query — the closest sentence to a question ranks first.
func TestModel_Rerank_BGESmallOrdering(t *testing.T) {
	snapshot := bgeSmallSnapshot(t)
	if snapshot == "" {
		t.Skip("bge-small-en-v1.5 snapshot not found; set BERT_PARITY_SNAPSHOT to run the rerank receipt")
	}
	model, err := bert.Load(snapshot)
	if err != nil {
		t.Fatalf("bert.Load: %v", err)
	}
	query := "How do I reset my password?"
	docs := []string{
		"The quick brown fox jumps over the lazy dog.",
		"To change your password, open account settings and choose reset password.",
		"Vector search retrieves semantically similar documents.",
	}
	result, err := model.Rerank(context.Background(), inference.RerankRequest{Query: query, Documents: docs})
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if len(result.Results) != len(docs) {
		t.Fatalf("Rerank returned %d results, want %d", len(result.Results), len(docs))
	}
	if result.Results[0].Index != 1 {
		t.Errorf("top document index = %d, want 1 (the password-reset sentence); scores=%v", result.Results[0].Index, scoreList(result.Results))
	}
	for i := 1; i < len(result.Results); i++ {
		if result.Results[i-1].Score < result.Results[i].Score {
			t.Errorf("results not sorted descending at %d: %v", i, scoreList(result.Results))
		}
	}
}

// TestTokenizer_Encode_BGEIDs checks the Go WordPiece IDs against the reference
// fixture's transformers-tokenised input_ids, pinning tokeniser drift
// independently of the forward pass.
func TestTokenizer_Encode_BGEIDs(t *testing.T) {
	snapshot := bgeSmallSnapshot(t)
	if snapshot == "" {
		t.Skip("bge-small-en-v1.5 snapshot not found; set BERT_PARITY_SNAPSHOT for the tokeniser ID check")
	}
	vocab, err := os.ReadFile(filepath.Join(snapshot, "vocab.txt"))
	if err != nil {
		t.Fatalf("read vocab.txt: %v", err)
	}
	tk, err := bert.NewTokenizer(vocab, true)
	if err != nil {
		t.Fatalf("NewTokenizer: %v", err)
	}
	fixture := loadReferenceFixture(t)
	for _, record := range fixture.Records {
		got := tk.Encode(record.Text)
		if len(got) != len(record.InputIDs) {
			t.Fatalf("Encode(%q) produced %d ids, want %d: %v vs %v", record.Text, len(got), len(record.InputIDs), got, record.InputIDs)
		}
		for i := range record.InputIDs {
			if got[i] != record.InputIDs[i] {
				t.Fatalf("Encode(%q) id[%d] = %d, want %d", record.Text, i, got[i], record.InputIDs[i])
			}
		}
	}
}

func scoreList(results []inference.RerankScore) []float64 {
	out := make([]float64, len(results))
	for i, r := range results {
		out[i] = r.Score
	}
	return out
}

func cosineSimilarity(a, b []float32) float64 {
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func loadReferenceFixture(t *testing.T) referenceFixture {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", "bge_small_reference.json"))
	if err != nil {
		t.Fatalf("read reference fixture: %v", err)
	}
	var fixture referenceFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatalf("decode reference fixture: %v", err)
	}
	if len(fixture.Records) == 0 {
		t.Fatal("reference fixture has no records")
	}
	return fixture
}

// bgeSmallSnapshot resolves the snapshot directory from BERT_PARITY_SNAPSHOT or
// the default Hugging Face cache; returns "" when neither is present. Takes
// testing.TB (not *testing.T) so both the parity tests above and the
// tokens/sec benchmark in model_bench_test.go can share one resolver.
func bgeSmallSnapshot(t testing.TB) string {
	t.Helper()
	if dir := os.Getenv("BERT_PARITY_SNAPSHOT"); dir != "" {
		return dir
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	root := filepath.Join(home, ".cache", "huggingface", "hub", "models--BAAI--bge-small-en-v1.5", "snapshots")
	entries, err := os.ReadDir(root)
	if err != nil {
		return ""
	}
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		candidate := filepath.Join(root, entry.Name())
		if _, err := os.Stat(filepath.Join(candidate, "model.safetensors")); err == nil {
			return candidate
		}
	}
	return ""
}
