// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	openai "dappco.re/go/inference/serving/provider/openai"
)

// bgeSmallSnapshot resolves a local bge-small-en-v1.5 snapshot directory from
// BERT_PARITY_SNAPSHOT or the default Hugging Face cache location — the same
// opt-in convention model/bert's parity tests use — so these tests prove real
// vectors flow end to end wherever that convention already gives them a
// model, and skip cleanly everywhere else instead of faking infrastructure
// that isn't there.
func bgeSmallSnapshot(t *testing.T) string {
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

// TestLoadEmbedModel_Bad_MissingSnapshot fails closed on a directory that
// isn't a bert snapshot — the shape a mistyped -embed-model produces.
func TestLoadEmbedModel_Bad_MissingSnapshot(t *testing.T) {
	_, _, err := loadEmbedModel(t.TempDir(), "")
	if err == nil {
		t.Fatal("loadEmbedModel(empty dir) = nil error, want a load failure")
	}
}

// TestLoadEmbedModel_Good_RealSnapshot proves loadEmbedModel produces a
// TextModel that actually satisfies EmbeddingModel and RerankModel, and that
// an empty id defaults to the pack's basename — the real -embed-model path,
// not a fake standing in for it.
func TestLoadEmbedModel_Good_RealSnapshot(t *testing.T) {
	snapshot := bgeSmallSnapshot(t)
	if snapshot == "" {
		t.Skip("bge-small-en-v1.5 snapshot not found; set BERT_PARITY_SNAPSHOT to run this receipt")
	}
	model, id, err := loadEmbedModel(snapshot, "")
	if err != nil {
		t.Fatalf("loadEmbedModel(%q, \"\"): %v", snapshot, err)
	}
	if id != core.PathBase(snapshot) {
		t.Fatalf("id = %q, want the snapshot basename %q", id, core.PathBase(snapshot))
	}
	if _, ok := model.(inference.EmbeddingModel); !ok {
		t.Fatal("loaded embed model does not satisfy inference.EmbeddingModel")
	}
	if _, ok := model.(inference.RerankModel); !ok {
		t.Fatal("loaded embed model does not satisfy inference.RerankModel")
	}
}

// TestLoadEmbedModel_Good_ExplicitID proves an explicit id passes through
// unchanged rather than being overridden by the basename default.
func TestLoadEmbedModel_Good_ExplicitID(t *testing.T) {
	snapshot := bgeSmallSnapshot(t)
	if snapshot == "" {
		t.Skip("bge-small-en-v1.5 snapshot not found; set BERT_PARITY_SNAPSHOT to run this receipt")
	}
	_, id, err := loadEmbedModel(snapshot, "my-embedder")
	if err != nil {
		t.Fatalf("loadEmbedModel: %v", err)
	}
	if id != "my-embedder" {
		t.Fatalf("id = %q, want the explicit \"my-embedder\"", id)
	}
}

// embedFake is a minimal inference.TextModel used to test wrapEmbedResolver
// without paying for a real encoder load.
type embedFake struct{ inference.TextModel }

// TestWrapEmbedResolver_Good_RoutesByID proves a request naming the embed
// model's id resolves to it without ever reaching inner.
func TestWrapEmbedResolver_Good_RoutesByID(t *testing.T) {
	embed := &embedFake{}
	inner := openai.ResolverFunc(func(context.Context, string) (inference.TextModel, error) {
		t.Fatal("inner resolver reached for a name that should have matched the embed model")
		return nil, nil
	})
	resolver := wrapEmbedResolver(inner, "bge-small", embed)

	got, err := resolver.ResolveModel(context.Background(), "bge-small")
	if err != nil {
		t.Fatalf("ResolveModel(bge-small): %v", err)
	}
	if got != inference.TextModel(embed) {
		t.Fatalf("ResolveModel(bge-small) = %v, want the embed model", got)
	}
}

// TestWrapEmbedResolver_Good_CaseInsensitiveTrimmed matches
// openai.StaticResolver's own comparison — -embed-model-id "BGE-Small" and a
// request for " bge-small " both route to it.
func TestWrapEmbedResolver_Good_CaseInsensitiveTrimmed(t *testing.T) {
	embed := &embedFake{}
	inner := openai.ResolverFunc(func(context.Context, string) (inference.TextModel, error) {
		t.Fatal("inner resolver reached for a name that should have matched the embed model")
		return nil, nil
	})
	resolver := wrapEmbedResolver(inner, "BGE-Small", embed)

	got, err := resolver.ResolveModel(context.Background(), " bge-small ")
	if err != nil {
		t.Fatalf("ResolveModel: %v", err)
	}
	if got != inference.TextModel(embed) {
		t.Fatalf("ResolveModel(case/space variant) = %v, want the embed model", got)
	}
}

// TestWrapEmbedResolver_Good_FallsThroughOtherNames proves any OTHER name
// still reaches inner unchanged — the wrap adds a name, it never removes one.
func TestWrapEmbedResolver_Good_FallsThroughOtherNames(t *testing.T) {
	embed := &embedFake{}
	chatModel := &mockTextModel{modelType: "chat"}
	inner := openai.ResolverFunc(func(_ context.Context, name string) (inference.TextModel, error) {
		if name != "chat-model" {
			t.Fatalf("inner resolver saw unexpected name %q", name)
		}
		return chatModel, nil
	})
	resolver := wrapEmbedResolver(inner, "bge-small", embed)

	got, err := resolver.ResolveModel(context.Background(), "chat-model")
	if err != nil {
		t.Fatalf("ResolveModel(chat-model): %v", err)
	}
	if got != inference.TextModel(chatModel) {
		t.Fatalf("ResolveModel(chat-model) = %v, want the chat model (fall-through)", got)
	}
}

// TestRunServe_EmbedModel_Bad_LoadFailureFailsClosed proves a bad
// -embed-model fails RunServe's boot rather than silently starting without
// embeddings — no listener ever binds, mirroring the outbound-policy and
// admin-token fail-closed tests.
func TestRunServe_EmbedModel_Bad_LoadFailureFailsClosed(t *testing.T) {
	addr := freeListenAddr(t)
	loader := func(path string, _ ...inference.LoadOption) (inference.TextModel, error) {
		return &mockTextModel{modelType: core.PathBase(path)}, nil
	}
	err := RunServe(context.Background(), ServeConfig{
		Addr:           addr,
		Log:            core.NewBuffer(),
		Loader:         loader,
		ModelPath:      "/m/chat",
		EmbedModelPath: t.TempDir(), // a real dir, but not a bert snapshot
	})
	if err == nil {
		t.Fatal("RunServe with a bad -embed-model returned nil, want a fail-closed error")
	}
}

// TestRunServe_EmbedModel_Good_NoEmbedModelReturns4xx proves the existing
// capability-honest behaviour is unchanged when -embed-model is unset: a chat
// model that does not implement EmbeddingModel gets a clean 400, never a
// silent wrong answer.
func TestRunServe_EmbedModel_Good_NoEmbedModelReturns4xx(t *testing.T) {
	addr := freeListenAddr(t)
	loader := func(path string, _ ...inference.LoadOption) (inference.TextModel, error) {
		return &mockTextModel{modelType: core.PathBase(path)}, nil
	}
	ctx, cancel := context.WithCancel(context.Background())
	errCh := make(chan error, 1)
	go func() {
		errCh <- RunServe(ctx, ServeConfig{
			Addr:      addr,
			Log:       core.NewBuffer(),
			Loader:    loader,
			ModelPath: "/m/chat",
		})
	}()
	resp := waitForHTTPUp(t, "http://"+addr+"/v1/health")
	resp.Body.Close()

	body := `{"model":"chat","input":"hello"}`
	req, err := http.NewRequest(http.MethodPost, "http://"+addr+openai.DefaultEmbeddingsPath, bytes.NewReader([]byte(body)))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	embedResp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST /v1/embeddings: %v", err)
	}
	embedBody, _ := io.ReadAll(embedResp.Body)
	embedResp.Body.Close()
	if embedResp.StatusCode < 400 || embedResp.StatusCode >= 500 {
		t.Fatalf("/v1/embeddings with no embed model = %d, want a 4xx (body: %s)", embedResp.StatusCode, embedBody)
	}

	cancel()
	select {
	case err := <-errCh:
		if err != nil {
			t.Fatalf("RunServe returned %v after cancel, want nil", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("RunServe did not shut down within 3s of cancel")
	}
}

// TestRunServe_EmbedModel_Good_ServesRealVectors is the full-stack proof for
// item A: -embed-model loads a real bert snapshot, /v1/models and /v1/health
// both name it, and /v1/embeddings + /v1/rerank serve REAL vectors/scores
// through it — with no chat model loaded at all (ModelPath ""), proving
// -embed-model works INSTEAD OF a chat model, not only alongside one.
func TestRunServe_EmbedModel_Good_ServesRealVectors(t *testing.T) {
	snapshot := bgeSmallSnapshot(t)
	if snapshot == "" {
		t.Skip("bge-small-en-v1.5 snapshot not found; set BERT_PARITY_SNAPSHOT to run this receipt")
	}
	addr := freeListenAddr(t)
	log := core.NewBuffer()
	ctx, cancel := context.WithCancel(context.Background())
	errCh := make(chan error, 1)
	go func() {
		errCh <- RunServe(ctx, ServeConfig{
			Addr:           addr,
			Log:            log,
			EmbedModelPath: snapshot,
			EmbedModelID:   "bge-small",
		})
	}()
	resp := waitForHTTPUp(t, "http://"+addr+"/v1/health")
	resp.Body.Close()

	modelsBody := httpGetBody(t, "http://"+addr+"/v1/models", "")
	if !core.Contains(modelsBody, `"bge-small"`) {
		t.Fatalf("/v1/models missing the embed model id:\n%s", modelsBody)
	}
	if !core.Contains(log.String(), "bge-small") {
		t.Fatalf("boot log = %q, want a notice naming the embed model", log.String())
	}

	// /v1/embeddings — a real vector, not a stub shape.
	embedReqBody := `{"model":"bge-small","input":"hello world"}`
	embedReq, err := http.NewRequest(http.MethodPost, "http://"+addr+openai.DefaultEmbeddingsPath, bytes.NewReader([]byte(embedReqBody)))
	if err != nil {
		t.Fatalf("new embeddings request: %v", err)
	}
	embedReq.Header.Set("Content-Type", "application/json")
	embedResp, err := http.DefaultClient.Do(embedReq)
	if err != nil {
		t.Fatalf("POST /v1/embeddings: %v", err)
	}
	embedBodyBytes, _ := io.ReadAll(embedResp.Body)
	embedResp.Body.Close()
	if embedResp.StatusCode != http.StatusOK {
		t.Fatalf("POST /v1/embeddings = %d, want 200 (body: %s)", embedResp.StatusCode, embedBodyBytes)
	}
	var embedParsed struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(embedBodyBytes, &embedParsed); err != nil {
		t.Fatalf("decode /v1/embeddings response: %v (body: %s)", err, embedBodyBytes)
	}
	if len(embedParsed.Data) != 1 || len(embedParsed.Data[0].Embedding) == 0 {
		t.Fatalf("/v1/embeddings returned no vector: %s", embedBodyBytes)
	}

	// /v1/rerank — real relevance ordering, mirroring model/bert's own
	// TestModel_Rerank_BGESmallOrdering fixture.
	rerankReqBody := `{"model":"bge-small","query":"How do I reset my password?","documents":["The quick brown fox jumps over the lazy dog.","To change your password, open account settings and choose reset password."]}`
	rerankReq, err := http.NewRequest(http.MethodPost, "http://"+addr+openai.DefaultRerankPath, bytes.NewReader([]byte(rerankReqBody)))
	if err != nil {
		t.Fatalf("new rerank request: %v", err)
	}
	rerankReq.Header.Set("Content-Type", "application/json")
	rerankResp, err := http.DefaultClient.Do(rerankReq)
	if err != nil {
		t.Fatalf("POST /v1/rerank: %v", err)
	}
	rerankBodyBytes, _ := io.ReadAll(rerankResp.Body)
	rerankResp.Body.Close()
	if rerankResp.StatusCode != http.StatusOK {
		t.Fatalf("POST /v1/rerank = %d, want 200 (body: %s)", rerankResp.StatusCode, rerankBodyBytes)
	}
	var rerankParsed struct {
		Results []inference.RerankScore `json:"results"`
	}
	if err := json.Unmarshal(rerankBodyBytes, &rerankParsed); err != nil {
		t.Fatalf("decode /v1/rerank response: %v (body: %s)", err, rerankBodyBytes)
	}
	if len(rerankParsed.Results) != 2 {
		t.Fatalf("/v1/rerank returned %d results, want 2", len(rerankParsed.Results))
	}
	if rerankParsed.Results[0].Index != 1 {
		t.Fatalf("/v1/rerank top result index = %d, want 1 (the password-reset sentence): %v", rerankParsed.Results[0].Index, rerankParsed.Results)
	}

	cancel()
	select {
	case err := <-errCh:
		if err != nil {
			t.Fatalf("RunServe returned %v after cancel, want nil", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("RunServe did not shut down within 3s of cancel")
	}
}
