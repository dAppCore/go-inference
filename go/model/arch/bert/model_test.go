// SPDX-Licence-Identifier: EUPL-1.2

package bert_test

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/inference/model/arch/bert"
	"dappco.re/go/inference/model/safetensors"
)

// syntheticSnapshotCfg is the tiny geometry every synthetic on-disk snapshot in this file
// shares: hidden 8, 1 layer, 2 heads (head_dim 4), intermediate 16, a 10-word vocab (5
// specials + 5 real words), 16 max positions, 2 token types.
const (
	synHidden       = 8
	synLayers       = 1
	synHeads        = 2
	synIntermediate = 16
	synVocab        = 10
	synMaxPos       = 16
	synTypeVocab    = 2
)

// f32Bytes little-endian-encodes a deterministic seeded fill of n float32 values — the same
// small-signed-fraction generator used across the arch packages' synthetic-weight tests
// (e.g. mamba2/gpt2's syn/tensor helpers), sized for an F32 safetensors payload.
func f32Bytes(n, seed int) []byte {
	out := make([]byte, n*4)
	for i := range n {
		v := float32((i*seed+7)%101-50) * 0.02
		bits := math.Float32bits(v)
		out[4*i], out[4*i+1], out[4*i+2], out[4*i+3] = byte(bits), byte(bits>>8), byte(bits>>16), byte(bits>>24)
	}
	return out
}

// buildSyntheticSnapshot writes a real, on-disk BertModel HF snapshot (config.json,
// vocab.txt, model.safetensors) into dir at the tiny synHidden/synLayers/... geometry —
// small enough to construct every required tensor by hand, but a REAL forward pass (not a
// stub): bert.Load reads it back exactly as it would a real bge-small snapshot. seed varies
// the weight fill so two snapshots built by this helper are never bit-identical. Takes a
// plain dir + error return (no *testing.T) so both the _test.go triplets and the
// *testing.T-free _example_test.go Examples can share one builder.
func buildSyntheticSnapshot(dir string, seed int) error {
	configJSON := `{
		"model_type": "bert",
		"hidden_size": ` + itoa(synHidden) + `,
		"num_hidden_layers": ` + itoa(synLayers) + `,
		"num_attention_heads": ` + itoa(synHeads) + `,
		"intermediate_size": ` + itoa(synIntermediate) + `,
		"vocab_size": ` + itoa(synVocab) + `,
		"max_position_embeddings": ` + itoa(synMaxPos) + `,
		"type_vocab_size": ` + itoa(synTypeVocab) + `,
		"layer_norm_eps": 1e-12
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(configJSON), 0o644); err != nil {
		return err
	}

	vocab := "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nthe\nquick\nfox\nreset\npassword\n"
	if err := os.WriteFile(filepath.Join(dir, "vocab.txt"), []byte(vocab), 0o644); err != nil {
		return err
	}

	tensors := map[string]safetensors.SafetensorsTensorInfo{}
	data := map[string][]byte{}
	add := func(name string, shape []int) {
		n := 1
		for _, d := range shape {
			n *= d
		}
		seed++
		data[name] = f32Bytes(n, seed)
		tensors[name] = safetensors.SafetensorsTensorInfo{Dtype: "F32", Shape: shape}
	}
	add("embeddings.word_embeddings.weight", []int{synVocab, synHidden})
	add("embeddings.position_embeddings.weight", []int{synMaxPos, synHidden})
	add("embeddings.token_type_embeddings.weight", []int{synTypeVocab, synHidden})
	add("embeddings.LayerNorm.weight", []int{synHidden})
	add("embeddings.LayerNorm.bias", []int{synHidden})
	for l := 0; l < synLayers; l++ {
		p := "encoder.layer." + itoa(l) + "."
		add(p+"attention.self.query.weight", []int{synHidden, synHidden})
		add(p+"attention.self.query.bias", []int{synHidden})
		add(p+"attention.self.key.weight", []int{synHidden, synHidden})
		add(p+"attention.self.key.bias", []int{synHidden})
		add(p+"attention.self.value.weight", []int{synHidden, synHidden})
		add(p+"attention.self.value.bias", []int{synHidden})
		add(p+"attention.output.dense.weight", []int{synHidden, synHidden})
		add(p+"attention.output.dense.bias", []int{synHidden})
		add(p+"attention.output.LayerNorm.weight", []int{synHidden})
		add(p+"attention.output.LayerNorm.bias", []int{synHidden})
		add(p+"intermediate.dense.weight", []int{synIntermediate, synHidden})
		add(p+"intermediate.dense.bias", []int{synIntermediate})
		add(p+"output.dense.weight", []int{synHidden, synIntermediate})
		add(p+"output.dense.bias", []int{synHidden})
		add(p+"output.LayerNorm.weight", []int{synHidden})
		add(p+"output.LayerNorm.bias", []int{synHidden})
	}
	if r := safetensors.WriteSafetensors(filepath.Join(dir, "model.safetensors"), tensors, data); !r.OK {
		return r.Err()
	}
	return nil
}

// writeSyntheticSnapshot is buildSyntheticSnapshot wrapped for *testing.T callers: a fresh
// t.TempDir() (cleaned up automatically) and a t.Fatalf on any build error.
func writeSyntheticSnapshot(t *testing.T, seed int) string {
	t.Helper()
	dir := t.TempDir()
	if err := buildSyntheticSnapshot(dir, seed); err != nil {
		t.Fatalf("buildSyntheticSnapshot: %v", err)
	}
	return dir
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b [20]byte
	p := len(b)
	for i > 0 {
		p--
		b[p] = byte('0' + i%10)
		i /= 10
	}
	return string(b[p:])
}

// TestModel_Load_Good loads a real (if tiny) on-disk BertModel snapshot end-to-end: config,
// vocab and every layer tensor bind, and the returned Model reports the snapshot's own dims.
func TestModel_Load_Good(t *testing.T) {
	dir := writeSyntheticSnapshot(t, 1)
	m, err := bert.Load(dir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if m.Config().HiddenSize != synHidden || m.Config().NumHiddenLayers != synLayers {
		t.Fatalf("Config = %+v, want hidden %d layers %d", m.Config(), synHidden, synLayers)
	}
	// no 1_Pooling/modules.json in this synthetic snapshot: Load's documented
	// bare-BertModel default is mean pooling with normalisation.
	if m.Pooling() != bert.PoolingMean || !m.Normalises() {
		t.Fatalf("pooling = %q normalise %v, want mean/true (bare-BertModel default)", m.Pooling(), m.Normalises())
	}
}

func TestModel_Load_Bad(t *testing.T) {
	if _, err := bert.Load(filepath.Join(t.TempDir(), "does-not-exist")); err == nil {
		t.Fatal("Load accepted a directory with no config.json")
	}
}

// TestModel_Load_Ugly proves a snapshot with a truncated model.safetensors (missing a
// required layer tensor) is rejected at Load — distinct from _Bad's missing-directory case,
// this is a well-formed config+vocab paired with an incomplete checkpoint.
func TestModel_Load_Ugly(t *testing.T) {
	dir := writeSyntheticSnapshot(t, 2)
	// Overwrite model.safetensors with a version missing the intermediate.dense weight —
	// bindWeights must fail on the missing tensor name, not panic or silently zero-fill.
	tensors := map[string]safetensors.SafetensorsTensorInfo{
		"embeddings.word_embeddings.weight": {Dtype: "F32", Shape: []int{synVocab, synHidden}},
	}
	data := map[string][]byte{
		"embeddings.word_embeddings.weight": f32Bytes(synVocab*synHidden, 99),
	}
	if r := safetensors.WriteSafetensors(filepath.Join(dir, "model.safetensors"), tensors, data); !r.OK {
		t.Fatalf("WriteSafetensors: %v", r.Error())
	}
	if _, err := bert.Load(dir); err == nil {
		t.Fatal("Load accepted a truncated checkpoint missing required tensors")
	}
}

func TestModel_Config_Good(t *testing.T) {
	m, err := bert.Load(writeSyntheticSnapshot(t, 3))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got := m.Config().HiddenSize; got != synHidden {
		t.Fatalf("Config().HiddenSize = %d, want %d", got, synHidden)
	}
}

// TestModel_Config_Bad proves absence is reported honestly: a plain embedder snapshot (no
// num_labels, no id2label) reports NumLabels 0 rather than a fabricated classifier width.
func TestModel_Config_Bad(t *testing.T) {
	m, err := bert.Load(writeSyntheticSnapshot(t, 15))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got := m.Config().NumLabels; got != 0 {
		t.Fatalf("Config().NumLabels = %d, want 0 (plain embedder snapshot declares no classifier head)", got)
	}
}

// TestModel_Config_Ugly proves Config() returns the FULL parsed config (not just the
// dimensions Embed/Rerank consume) — e.g. ModelType survives the round trip through Load.
func TestModel_Config_Ugly(t *testing.T) {
	m, err := bert.Load(writeSyntheticSnapshot(t, 4))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got := m.Config().ModelType; got != "bert" {
		t.Fatalf("Config().ModelType = %q, want %q (full config preserved, not just dims)", got, "bert")
	}
}

func TestModel_Pooling_Good(t *testing.T) {
	m, err := bert.Load(writeSyntheticSnapshot(t, 5))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got := m.Pooling(); got != bert.PoolingMean {
		t.Fatalf("Pooling() = %q, want %q (bare-BertModel default)", got, bert.PoolingMean)
	}
}

// TestModel_Pooling_Bad proves Load always resolves a CONCRETE pooling mode — Pooling()
// is never the empty string for any successfully loaded model.
func TestModel_Pooling_Bad(t *testing.T) {
	m, err := bert.Load(writeSyntheticSnapshot(t, 16))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got := m.Pooling(); got != bert.PoolingCLS && got != bert.PoolingMean {
		t.Fatalf("Pooling() = %q, want one of cls/mean (Load must always resolve a concrete mode)", got)
	}
}

// TestModel_Pooling_Ugly proves Pooling() reflects a snapshot's OWN 1_Pooling/config.json
// declaration (CLS here) rather than always returning the bare-BertModel mean default —
// distinct from _Good's default-path case.
func TestModel_Pooling_Ugly(t *testing.T) {
	dir := writeSyntheticSnapshot(t, 6)
	if err := os.MkdirAll(filepath.Join(dir, "1_Pooling"), 0o755); err != nil {
		t.Fatalf("mkdir 1_Pooling: %v", err)
	}
	poolingJSON := `{"pooling_mode_cls_token": true, "pooling_mode_mean_tokens": false}`
	if err := os.WriteFile(filepath.Join(dir, "1_Pooling", "config.json"), []byte(poolingJSON), 0o644); err != nil {
		t.Fatalf("write 1_Pooling/config.json: %v", err)
	}
	m, err := bert.Load(dir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got := m.Pooling(); got != bert.PoolingCLS {
		t.Fatalf("Pooling() = %q, want %q (snapshot declares CLS pooling)", got, bert.PoolingCLS)
	}
}

func TestModel_Normalises_Good(t *testing.T) {
	// no modules.json: normaliseFromModules defaults to true.
	m, err := bert.Load(writeSyntheticSnapshot(t, 7))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !m.Normalises() {
		t.Fatal("Normalises() = false, want true (no modules.json ⇒ default normalise)")
	}
}

// TestModel_Normalises_Bad proves a MALFORMED modules.json degrades to the safe default
// (true) rather than erroring the whole Load — normaliseFromModules' JSON-decode-failure
// branch.
func TestModel_Normalises_Bad(t *testing.T) {
	dir := writeSyntheticSnapshot(t, 17)
	if err := os.WriteFile(filepath.Join(dir, "modules.json"), []byte("{not json"), 0o644); err != nil {
		t.Fatalf("write modules.json: %v", err)
	}
	m, err := bert.Load(dir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !m.Normalises() {
		t.Fatal("Normalises() = false, want true (malformed modules.json must default to true)")
	}
}

// TestModel_Normalises_Ugly proves a snapshot whose modules.json declares NO Normalize
// module reports false — distinct from _Good's no-modules.json-at-all default.
func TestModel_Normalises_Ugly(t *testing.T) {
	dir := writeSyntheticSnapshot(t, 8)
	modulesJSON := `[{"type": "sentence_transformers.models.Transformer"}, {"type": "sentence_transformers.models.Pooling"}]`
	if err := os.WriteFile(filepath.Join(dir, "modules.json"), []byte(modulesJSON), 0o644); err != nil {
		t.Fatalf("write modules.json: %v", err)
	}
	m, err := bert.Load(dir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if m.Normalises() {
		t.Fatal("Normalises() = true, want false (modules.json present without a Normalize module)")
	}
}

func TestModel_Embed_Good(t *testing.T) {
	m, err := bert.Load(writeSyntheticSnapshot(t, 9))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	res, err := m.Embed(context.Background(), inference.EmbeddingRequest{Input: []string{"the quick fox"}})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(res.Vectors) != 1 || len(res.Vectors[0]) != synHidden {
		t.Fatalf("Embed vectors = %d of dim %d, want 1 of dim %d", len(res.Vectors), len(res.Vectors[0]), synHidden)
	}
	if res.Usage.PromptTokens == 0 {
		t.Fatal("Embed reported zero prompt tokens for a non-empty input")
	}
}

func TestModel_Embed_Bad(t *testing.T) {
	m, err := bert.Load(writeSyntheticSnapshot(t, 10))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if _, err := m.Embed(context.Background(), inference.EmbeddingRequest{Input: nil}); err == nil {
		t.Fatal("Embed accepted an empty input list")
	}
}

// TestModel_Embed_Ugly proves a WHITESPACE-ONLY input string (non-empty len, but blank after
// Trim) is rejected — distinct from _Bad's zero-length input list.
func TestModel_Embed_Ugly(t *testing.T) {
	m, err := bert.Load(writeSyntheticSnapshot(t, 11))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if _, err := m.Embed(context.Background(), inference.EmbeddingRequest{Input: []string{"   "}}); err == nil {
		t.Fatal("Embed accepted a whitespace-only input string")
	}
}

func TestModel_Rerank_Good(t *testing.T) {
	m, err := bert.Load(writeSyntheticSnapshot(t, 12))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	res, err := m.Rerank(context.Background(), inference.RerankRequest{
		Query:     "reset password",
		Documents: []string{"the quick fox", "reset password"},
	})
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if len(res.Results) != 2 {
		t.Fatalf("Rerank returned %d results, want 2", len(res.Results))
	}
	for i := 1; i < len(res.Results); i++ {
		if res.Results[i-1].Score < res.Results[i].Score {
			t.Fatalf("Rerank results not sorted descending: %+v", res.Results)
		}
	}
}

func TestModel_Rerank_Bad(t *testing.T) {
	m, err := bert.Load(writeSyntheticSnapshot(t, 13))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if _, err := m.Rerank(context.Background(), inference.RerankRequest{Query: "", Documents: []string{"x"}}); err == nil {
		t.Fatal("Rerank accepted an empty query")
	}
}

// TestModel_Rerank_Ugly proves TopN truncation: with TopN=1 over two documents, Rerank
// returns exactly the single top-scoring result — distinct from _Bad's empty-query rejection.
func TestModel_Rerank_Ugly(t *testing.T) {
	m, err := bert.Load(writeSyntheticSnapshot(t, 14))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	res, err := m.Rerank(context.Background(), inference.RerankRequest{
		Query:     "reset password",
		Documents: []string{"the quick fox", "reset password"},
		TopN:      1,
	})
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if len(res.Results) != 1 {
		t.Fatalf("Rerank with TopN=1 returned %d results, want 1", len(res.Results))
	}
}
