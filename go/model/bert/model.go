// SPDX-Licence-Identifier: EUPL-1.2

package bert

import (
	"context"
	"math"
	"sort"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/model/safetensors"
)

// Model is a loaded host BERT encoder: config, weights, tokeniser, and the
// pooling/normalise choices read from the sentence-transformers snapshot. It
// satisfies inference.EmbeddingModel and inference.RerankModel.
//
//	m, err := bert.Load(snapshotDir)
//	res, err := m.Embed(ctx, inference.EmbeddingRequest{Input: []string{"hello world"}})
type Model struct {
	cfg       Config
	weights   *Weights
	tokenizer *Tokenizer
	pooling   Pooling
	normalise bool
}

// Config returns the parsed BertModel configuration.
func (m *Model) Config() Config { return m.cfg }

// Pooling returns the pooling mode resolved from the snapshot (CLS for bge-class).
func (m *Model) Pooling() Pooling { return m.pooling }

// Normalises reports whether the model L2-normalises its output vectors (bge does).
func (m *Model) Normalises() bool { return m.normalise }

// Load reads a BertModel HF snapshot directory (config.json, vocab.txt,
// model.safetensors, and the optional sentence-transformers 1_Pooling/modules
// files) into a Model. The snapshot must be a plain float32 BertModel — quantised
// or sharded encoder checkpoints are out of scope for the round-1 host path.
//
//	m, err := bert.Load("/cache/models--BAAI--bge-small-en-v1.5/snapshots/<rev>")
func Load(dir string) (*Model, error) {
	cfgResult := core.ReadFile(core.PathJoin(dir, "config.json"))
	if !cfgResult.OK {
		return nil, core.E("bert.Load", "read config.json", cfgResult.Err())
	}
	cfg, err := ParseConfig(cfgResult.Value.([]byte))
	if err != nil {
		return nil, err
	}

	vocabResult := core.ReadFile(core.PathJoin(dir, "vocab.txt"))
	if !vocabResult.OK {
		return nil, core.E("bert.Load", "read vocab.txt", vocabResult.Err())
	}
	tokenizer, err := NewTokenizer(vocabResult.Value.([]byte), lowerCaseFromSnapshot(dir))
	if err != nil {
		return nil, err
	}

	tensors, err := safetensors.Load(core.PathJoin(dir, "model.safetensors"))
	if err != nil {
		return nil, core.E("bert.Load", "read model.safetensors", err)
	}
	weights, err := bindWeights(cfg, tensors)
	if err != nil {
		return nil, err
	}

	pooling, normalise := poolingFromSnapshot(dir)
	return &Model{
		cfg:       cfg,
		weights:   weights,
		tokenizer: tokenizer,
		pooling:   pooling,
		normalise: normalise,
	}, nil
}

// Embed runs each input sentence through the encoder and pools it to one vector.
// The model normalises when its snapshot declares a Normalize module; the
// request's Normalize flag forces normalisation on top for callers that want
// unit vectors regardless of the model's default.
//
//	res, err := m.Embed(ctx, inference.EmbeddingRequest{Input: texts, Normalize: true})
func (m *Model) Embed(ctx context.Context, req inference.EmbeddingRequest) (*inference.EmbeddingResult, error) {
	if m == nil || m.weights == nil {
		return nil, core.E("bert.Embed", "model is not loaded", nil)
	}
	if err := ctxErr(ctx); err != nil {
		return nil, err
	}
	if len(req.Input) == 0 {
		return nil, core.E("bert.Embed", "input text is required", nil)
	}
	vectors := make([][]float32, 0, len(req.Input))
	promptTokens := 0
	for index, text := range req.Input {
		if core.Trim(text) == "" {
			return nil, core.E("bert.Embed", core.Sprintf("input %d is empty", index), nil)
		}
		ids := m.tokenizer.Encode(text)
		promptTokens += len(ids)
		hidden, err := m.weights.forward(m.cfg, ids)
		if err != nil {
			return nil, err
		}
		vector, err := pool(m.pooling, hidden)
		if err != nil {
			return nil, err
		}
		if m.normalise || req.Normalize {
			l2Normalise(vector)
		}
		vectors = append(vectors, vector)
	}
	return &inference.EmbeddingResult{
		Model:   m.identity(req.Model),
		Vectors: vectors,
		Usage:   inference.EmbeddingUsage{PromptTokens: promptTokens, TotalTokens: promptTokens},
		Labels: map[string]string{
			"backend":           "host_f32",
			"embedding_pooling": string(m.pooling),
			"embedding_source":  "bert_encoder_forward",
		},
	}, nil
}

// Rerank scores each document against the query by cosine similarity of their
// embeddings (higher is more relevant), sorts descending, and truncates to
// TopN when positive. It is the embedding-cosine (bi-encoder) rerank — a
// cross-encoder head is a later addition, mirroring the device path's two
// rerank modes.
//
// Evidenced gap (item B of #50): a cross-encoder pack needs (1) config.go's
// Config to carry a classifier width (HF's num_labels; absent today), (2)
// weights.go's bindWeights to read a classifier head (classifier.{weight,bias}
// [1,hidden]/[1], and usually a pooler.dense.{weight,bias} ahead of it —
// neither is read; bindWeights only binds embeddings.* and encoder.layer.N.*),
// (3) tokenizer.go's Encode(text string) to grow a paired form (query [SEP]
// passage [SEP] in one sequence — no such method exists), and (4) encoder.go's
// forward to accept per-token segment ids (today typeRow is hardcoded to
// token_type_ids=0 for every position — see forward below — so even a paired
// encoding would embed both segments identically). No cross-encoder snapshot
// or parity fixture is available locally to verify any of this against (only
// bge-small-en-v1.5, a bi-encoder, is cached; testdata/ carries only
// bge_small_reference.json) — shipping the maths unverified risks a silently
// wrong score with no test to catch it, so this slice documents the gap
// rather than guessing at it.
//
//	res, err := m.Rerank(ctx, inference.RerankRequest{Query: q, Documents: docs, TopN: 3})
func (m *Model) Rerank(ctx context.Context, req inference.RerankRequest) (*inference.RerankResult, error) {
	if m == nil || m.weights == nil {
		return nil, core.E("bert.Rerank", "model is not loaded", nil)
	}
	if core.Trim(req.Query) == "" {
		return nil, core.E("bert.Rerank", "query is required", nil)
	}
	if len(req.Documents) == 0 {
		return nil, core.E("bert.Rerank", "documents are required", nil)
	}
	inputs := make([]string, 0, len(req.Documents)+1)
	inputs = append(inputs, req.Query)
	inputs = append(inputs, req.Documents...)
	embedded, err := m.Embed(ctx, inference.EmbeddingRequest{Model: req.Model, Input: inputs, Normalize: true})
	if err != nil {
		return nil, err
	}
	query := embedded.Vectors[0]
	results := make([]inference.RerankScore, len(req.Documents))
	for i, docVec := range embedded.Vectors[1:] {
		results[i] = inference.RerankScore{
			Index: i,
			Score: cosine(query, docVec),
			Text:  req.Documents[i],
		}
	}
	sort.SliceStable(results, func(a, b int) bool {
		if results[a].Score == results[b].Score {
			return results[a].Index < results[b].Index
		}
		return results[a].Score > results[b].Score
	})
	if req.TopN > 0 && req.TopN < len(results) {
		results = results[:req.TopN]
	}
	return &inference.RerankResult{
		Model:   m.identity(req.Model),
		Results: results,
		Labels: map[string]string{
			"backend":             "host_f32",
			"rerank_score_source": "embedding_cosine",
		},
	}, nil
}

func (m *Model) identity(name string) inference.ModelIdentity {
	return inference.ModelIdentity{
		ID:           name,
		Architecture: m.cfg.ModelType,
		VocabSize:    m.cfg.VocabSize,
		NumLayers:    m.cfg.NumHiddenLayers,
		HiddenSize:   m.cfg.HiddenSize,
	}
}

// cosine is the cosine similarity of two equal-length vectors, guarding against
// length mismatch and zero magnitude by returning 0.
func cosine(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		av, bv := float64(a[i]), float64(b[i])
		dot += av * bv
		normA += av * av
		normB += bv * bv
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func ctxErr(ctx context.Context) error {
	if ctx == nil {
		return nil
	}
	select {
	case <-ctx.Done():
		return core.E("bert", "context cancelled", ctx.Err())
	default:
		return nil
	}
}
