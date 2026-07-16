// SPDX-Licence-Identifier: EUPL-1.2

package bert_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/model/arch/bert"
	"dappco.re/go/inference/serving/compat"
	"dappco.re/go/inference/serving/provider/openai"
)

// newTestServeModel loads the same tiny synthetic on-disk snapshot model_test.go's triplets
// exercise and wraps it as a ServeModel — a real (if minimal) embedding-only encoder, not a
// mock, so Generate/Chat/Classify/BatchGenerate exercise the actual stub bodies and
// ModelType/Info reflect a real loaded config.
func newTestServeModel(t *testing.T, seed int) *bert.ServeModel {
	t.Helper()
	m, err := bert.Load(writeSyntheticSnapshot(t, seed))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	return bert.NewServeModel(m)
}

func TestServeModel_NewServeModel_Good(t *testing.T) {
	m, err := bert.Load(writeSyntheticSnapshot(t, 100))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	sm := bert.NewServeModel(m)
	if sm.Model != m {
		t.Fatal("NewServeModel did not retain the wrapped Model")
	}
}

// TestServeModel_NewServeModel_Bad proves wrapping a nil Model does not itself panic —
// construction is a plain struct literal, no dereference.
func TestServeModel_NewServeModel_Bad(t *testing.T) {
	sm := bert.NewServeModel(nil)
	if sm.Model != nil {
		t.Fatalf("NewServeModel(nil).Model = %v, want nil", sm.Model)
	}
}

// TestServeModel_NewServeModel_Ugly proves two independent wrappers over the SAME Model
// report identical metadata — deterministic wrapping, no hidden per-wrapper state.
func TestServeModel_NewServeModel_Ugly(t *testing.T) {
	m, err := bert.Load(writeSyntheticSnapshot(t, 101))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	sm1, sm2 := bert.NewServeModel(m), bert.NewServeModel(m)
	if sm1.ModelType() != sm2.ModelType() || sm1.Info() != sm2.Info() {
		t.Fatalf("two wrappers over the same Model diverge: %+v vs %+v", sm1.Info(), sm2.Info())
	}
}

func TestServeModel_Generate_Good(t *testing.T) {
	sm := newTestServeModel(t, 102)
	n := 0
	for range sm.Generate(context.Background(), "hello world") {
		n++
	}
	if n != 0 {
		t.Fatalf("Generate yielded %d tokens, want 0 (encoder has no generative path)", n)
	}
}

func TestServeModel_Generate_Bad(t *testing.T) {
	sm := newTestServeModel(t, 103)
	n := 0
	for range sm.Generate(context.Background(), "") {
		n++
	}
	if n != 0 {
		t.Fatalf(`Generate("") yielded %d tokens, want 0`, n)
	}
}

// TestServeModel_Generate_Ugly proves Generate never consults its context — a nil ctx (which
// Embed's ctxErr guard would reject if reached) still yields the same empty stream, no panic.
func TestServeModel_Generate_Ugly(t *testing.T) {
	sm := newTestServeModel(t, 104)
	n := 0
	for range sm.Generate(nil, "hello") { //nolint // deliberate nil ctx: Generate never touches it
		n++
	}
	if n != 0 {
		t.Fatalf("Generate with a nil context yielded %d tokens, want 0", n)
	}
}

func TestServeModel_Chat_Good(t *testing.T) {
	sm := newTestServeModel(t, 105)
	n := 0
	for range sm.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}) {
		n++
	}
	if n != 0 {
		t.Fatalf("Chat yielded %d tokens, want 0 (encoder has no generative path)", n)
	}
}

func TestServeModel_Chat_Bad(t *testing.T) {
	sm := newTestServeModel(t, 106)
	n := 0
	for range sm.Chat(context.Background(), nil) {
		n++
	}
	if n != 0 {
		t.Fatalf("Chat(nil) yielded %d tokens, want 0", n)
	}
}

// TestServeModel_Chat_Ugly proves Chat never consults its context, mirroring Generate_Ugly —
// distinct from _Bad's nil-message-history case.
func TestServeModel_Chat_Ugly(t *testing.T) {
	sm := newTestServeModel(t, 107)
	n := 0
	for range sm.Chat(nil, []inference.Message{{Role: "user", Content: "hi"}}) { //nolint // deliberate nil ctx
		n++
	}
	if n != 0 {
		t.Fatalf("Chat with a nil context yielded %d tokens, want 0", n)
	}
}

func TestServeModel_Classify_Good(t *testing.T) {
	sm := newTestServeModel(t, 108)
	r := sm.Classify(context.Background(), []string{"hello world"})
	if r.OK {
		t.Fatal("Classify succeeded on an embedding-only model")
	}
	if !core.Contains(r.Error(), "embedding-only") {
		t.Fatalf("Classify error = %q, want it to explain the encoder is embedding-only", r.Error())
	}
}

func TestServeModel_Classify_Bad(t *testing.T) {
	sm := newTestServeModel(t, 109)
	r := sm.Classify(context.Background(), nil)
	if r.OK {
		t.Fatal("Classify(nil) succeeded on an embedding-only model")
	}
}

// TestServeModel_Classify_Ugly proves the refusal is a FIXED message independent of the
// input shape — one document vs three documents refuse identically, distinct from an arch
// like gpt-oss's Arch() whose refusal echoes the input.
func TestServeModel_Classify_Ugly(t *testing.T) {
	sm := newTestServeModel(t, 110)
	r1 := sm.Classify(context.Background(), []string{"a"})
	r2 := sm.Classify(context.Background(), []string{"a", "b", "c"})
	if r1.Error() != r2.Error() {
		t.Fatalf("Classify refusal varies with input shape: %q vs %q, want identical", r1.Error(), r2.Error())
	}
}

func TestServeModel_BatchGenerate_Good(t *testing.T) {
	sm := newTestServeModel(t, 111)
	r := sm.BatchGenerate(context.Background(), []string{"hello world"})
	if r.OK {
		t.Fatal("BatchGenerate succeeded on an embedding-only model")
	}
	if !core.Contains(r.Error(), "embedding-only") {
		t.Fatalf("BatchGenerate error = %q, want it to explain the encoder is embedding-only", r.Error())
	}
}

func TestServeModel_BatchGenerate_Bad(t *testing.T) {
	sm := newTestServeModel(t, 112)
	r := sm.BatchGenerate(context.Background(), nil)
	if r.OK {
		t.Fatal("BatchGenerate(nil) succeeded on an embedding-only model")
	}
}

// TestServeModel_BatchGenerate_Ugly proves the refusal is a FIXED message independent of
// input shape, mirroring Classify_Ugly — distinct from _Bad's nil-input case.
func TestServeModel_BatchGenerate_Ugly(t *testing.T) {
	sm := newTestServeModel(t, 113)
	r1 := sm.BatchGenerate(context.Background(), []string{"a"})
	r2 := sm.BatchGenerate(context.Background(), []string{"a", "b", "c"})
	if r1.Error() != r2.Error() {
		t.Fatalf("BatchGenerate refusal varies with input shape: %q vs %q, want identical", r1.Error(), r2.Error())
	}
}

func TestServeModel_ModelType_Good(t *testing.T) {
	sm := newTestServeModel(t, 114)
	if got := sm.ModelType(); got != "bert" {
		t.Fatalf("ModelType() = %q, want %q", got, "bert")
	}
}

// TestServeModel_ModelType_Bad proves absence is reported honestly: a config.json that omits
// model_type reports an empty string, not a fabricated "bert".
func TestServeModel_ModelType_Bad(t *testing.T) {
	dir := writeSyntheticSnapshot(t, 115)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,"intermediate_size":16,"vocab_size":10,"max_position_embeddings":16,"type_vocab_size":2}`), 0o644); err != nil {
		t.Fatalf("overwrite config.json: %v", err)
	}
	m, err := bert.Load(dir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got := bert.NewServeModel(m).ModelType(); got != "" {
		t.Fatalf("ModelType() = %q, want empty (model_type omitted from the snapshot)", got)
	}
}

// TestServeModel_ModelType_Ugly proves ModelType() reflects an UNUSUAL but valid model_type
// (e.g. a GTE/E5-style checkpoint that still reports "bert" architecturally is the norm; a
// checkpoint declaring something else must round-trip verbatim, not be normalised away).
func TestServeModel_ModelType_Ugly(t *testing.T) {
	dir := writeSyntheticSnapshot(t, 116)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type":"nomic_bert","hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,"intermediate_size":16,"vocab_size":10,"max_position_embeddings":16,"type_vocab_size":2}`), 0o644); err != nil {
		t.Fatalf("overwrite config.json: %v", err)
	}
	m, err := bert.Load(dir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got := bert.NewServeModel(m).ModelType(); got != "nomic_bert" {
		t.Fatalf("ModelType() = %q, want the verbatim %q (not normalised to bert)", got, "nomic_bert")
	}
}

func TestServeModel_Info_Good(t *testing.T) {
	sm := newTestServeModel(t, 117)
	info := sm.Info()
	if info.Architecture != "bert" || info.HiddenSize != synHidden || info.NumLayers != synLayers || info.VocabSize != synVocab {
		t.Fatalf("Info() = %+v, want arch bert, hidden %d, layers %d, vocab %d", info, synHidden, synLayers, synVocab)
	}
}

// TestServeModel_Info_Bad proves Info() on a degenerate (zero-value) wrapped Model reports
// honest zeros rather than panicking or fabricating dimensions.
func TestServeModel_Info_Bad(t *testing.T) {
	sm := bert.NewServeModel(&bert.Model{})
	info := sm.Info()
	if info.HiddenSize != 0 || info.NumLayers != 0 || info.VocabSize != 0 {
		t.Fatalf("Info() on a zero-value Model = %+v, want all-zero", info)
	}
}

// TestServeModel_Info_Ugly proves Info() and ModelType() agree on Architecture — the same
// underlying config field surfaced two ways must never diverge.
func TestServeModel_Info_Ugly(t *testing.T) {
	sm := newTestServeModel(t, 118)
	if sm.Info().Architecture != sm.ModelType() {
		t.Fatalf("Info().Architecture = %q != ModelType() = %q, want them to agree", sm.Info().Architecture, sm.ModelType())
	}
}

func TestServeModel_Metrics_Good(t *testing.T) {
	sm := newTestServeModel(t, 119)
	m := sm.Metrics()
	if m != (inference.GenerateMetrics{}) {
		t.Fatalf("Metrics() = %+v, want the zero value (the encoder reports usage via EmbeddingResult, not here)", m)
	}
}

// TestServeModel_Metrics_Bad proves Metrics() is safe on a degenerate (zero-value) wrapped
// Model — still the zero value, no panic.
func TestServeModel_Metrics_Bad(t *testing.T) {
	sm := bert.NewServeModel(&bert.Model{})
	if m := sm.Metrics(); m != (inference.GenerateMetrics{}) {
		t.Fatalf("Metrics() on a zero-value Model = %+v, want the zero value", m)
	}
}

// TestServeModel_Metrics_Ugly proves Metrics() stays the zero value even AFTER a real Embed
// call — it never accumulates the embedding usage, distinct from _Good's before-any-call case.
func TestServeModel_Metrics_Ugly(t *testing.T) {
	sm := newTestServeModel(t, 120)
	if _, err := sm.Embed(context.Background(), inference.EmbeddingRequest{Input: []string{"hello"}}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if m := sm.Metrics(); m != (inference.GenerateMetrics{}) {
		t.Fatalf("Metrics() after Embed = %+v, want still the zero value", m)
	}
}

func TestServeModel_Err_Good(t *testing.T) {
	sm := newTestServeModel(t, 121)
	r := sm.Err()
	if r.OK {
		t.Fatal("Err() reported OK on an embedding-only model")
	}
	if !core.Contains(r.Error(), "embedding-only") {
		t.Fatalf("Err() = %q, want it to explain the /v1/embeddings or /v1/rerank routes", r.Error())
	}
}

// TestServeModel_Err_Bad proves Err() is safe on a degenerate (zero-value) wrapped Model —
// still the same fixed refusal, no panic.
func TestServeModel_Err_Bad(t *testing.T) {
	sm := bert.NewServeModel(&bert.Model{})
	if r := sm.Err(); r.OK {
		t.Fatal("Err() on a zero-value Model reported OK")
	}
}

// TestServeModel_Err_Ugly proves Err() returns the SAME fixed message regardless of which
// generative call (Generate vs Chat) preceded it — Err doesn't track per-call state.
func TestServeModel_Err_Ugly(t *testing.T) {
	sm := newTestServeModel(t, 122)
	for range sm.Generate(context.Background(), "x") {
	}
	afterGenerate := sm.Err().Error()
	for range sm.Chat(context.Background(), []inference.Message{{Role: "user", Content: "x"}}) {
	}
	afterChat := sm.Err().Error()
	if afterGenerate != afterChat {
		t.Fatalf("Err() after Generate (%q) != Err() after Chat (%q), want identical", afterGenerate, afterChat)
	}
}

func TestServeModel_Close_Good(t *testing.T) {
	sm := newTestServeModel(t, 123)
	if r := sm.Close(); !r.OK {
		t.Fatalf("Close() = %v, want OK (the host encoder holds only heap weights)", r.Error())
	}
}

// TestServeModel_Close_Bad proves Close() is safe on a degenerate (zero-value) wrapped Model
// — still succeeds, no panic (there is no file handle or device resource to fail releasing).
func TestServeModel_Close_Bad(t *testing.T) {
	sm := bert.NewServeModel(&bert.Model{})
	if r := sm.Close(); !r.OK {
		t.Fatalf("Close() on a zero-value Model = %v, want OK", r.Error())
	}
}

// TestServeModel_Close_Ugly proves Close() is safe to call MORE THAN ONCE on the same
// wrapper — idempotent, since it releases nothing — distinct from _Bad's degenerate-Model case.
func TestServeModel_Close_Ugly(t *testing.T) {
	sm := newTestServeModel(t, 124)
	if r := sm.Close(); !r.OK {
		t.Fatalf("first Close() = %v, want OK", r.Error())
	}
	if r := sm.Close(); !r.OK {
		t.Fatalf("second Close() = %v, want OK (idempotent)", r.Error())
	}
}

// TestServeModel_MuxEmbeddingsAndRerank drives the host encoder end-to-end
// through the assembled OpenAI-compatible mux — the same path a live `curl` to
// /v1/embeddings and /v1/rerank exercises. Opt-in on the bge-small snapshot.
func TestServeModel_MuxEmbeddingsAndRerank(t *testing.T) {
	snapshot := bgeSmallSnapshot(t)
	if snapshot == "" {
		t.Skip("bge-small-en-v1.5 snapshot not found; set BERT_PARITY_SNAPSHOT for the serve receipt")
	}
	model, err := bert.Load(snapshot)
	if err != nil {
		t.Fatalf("bert.Load: %v", err)
	}
	resolver := openai.NewStaticResolver(map[string]inference.TextModel{
		"bge-small-en-v1.5": bert.NewServeModel(model),
	})
	server := httptest.NewServer(compat.NewMux(resolver))
	t.Cleanup(server.Close)

	t.Run("embeddings_float", func(t *testing.T) {
		body := post(t, server.URL+openai.DefaultEmbeddingsPath,
			`{"model":"bge-small-en-v1.5","input":["hello world","password reset"]}`)
		var resp struct {
			Data []struct {
				Embedding []float32 `json:"embedding"`
			} `json:"data"`
			Usage inference.EmbeddingUsage `json:"usage"`
		}
		if err := json.Unmarshal(body, &resp); err != nil {
			t.Fatalf("decode: %v (body=%s)", err, body)
		}
		if len(resp.Data) != 2 {
			t.Fatalf("data len = %d, want 2", len(resp.Data))
		}
		if len(resp.Data[0].Embedding) != model.Config().HiddenSize {
			t.Fatalf("vector dim = %d, want %d", len(resp.Data[0].Embedding), model.Config().HiddenSize)
		}
		t.Logf("embeddings: 2 vectors of dim %d, prompt_tokens=%d", len(resp.Data[0].Embedding), resp.Usage.PromptTokens)
	})

	t.Run("embeddings_base64", func(t *testing.T) {
		body := post(t, server.URL+openai.DefaultEmbeddingsPath,
			`{"model":"bge-small-en-v1.5","input":"hello world","encoding_format":"base64"}`)
		var resp struct {
			Data []struct {
				Embedding string `json:"embedding"`
			} `json:"data"`
		}
		if err := json.Unmarshal(body, &resp); err != nil {
			t.Fatalf("decode: %v (body=%s)", err, body)
		}
		if len(resp.Data) != 1 || resp.Data[0].Embedding == "" {
			t.Fatalf("expected one base64 embedding string, got %s", body)
		}
	})

	t.Run("rerank", func(t *testing.T) {
		body := post(t, server.URL+openai.DefaultRerankPath, `{
			"model":"bge-small-en-v1.5",
			"query":"How do I reset my password?",
			"documents":[
				"The quick brown fox jumps over the lazy dog.",
				"Open account settings and choose reset password.",
				"Vector search retrieves semantically similar documents."
			],
			"top_n":2
		}`)
		var resp struct {
			Results []inference.RerankScore `json:"results"`
		}
		if err := json.Unmarshal(body, &resp); err != nil {
			t.Fatalf("decode: %v (body=%s)", err, body)
		}
		if len(resp.Results) != 2 {
			t.Fatalf("results len = %d, want 2 (top_n)", len(resp.Results))
		}
		if resp.Results[0].Index != 1 {
			t.Errorf("top result index = %d, want 1 (the reset-password doc)", resp.Results[0].Index)
		}
		t.Logf("rerank top: index=%d score=%.4f", resp.Results[0].Index, resp.Results[0].Score)
	})
}

func post(t *testing.T, url, body string) []byte {
	t.Helper()
	resp, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST %s: %v", url, err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("POST %s status = %d, body=%s", url, resp.StatusCode, data)
	}
	return data
}
