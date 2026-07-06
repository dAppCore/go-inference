// SPDX-License-Identifier: EUPL-1.2

package api

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/eval/score/lek"
	"github.com/gin-gonic/gin"
)

func init() {
	gin.SetMode(gin.TestMode)
}

// fakeEmbedder is a deterministic inference.EmbeddingModel for the
// /embeddings/text path — returns its fixed vector for any input.
type fakeEmbedder struct{ vec []float32 }

func (f fakeEmbedder) Embed(_ context.Context, _ inference.EmbeddingRequest) (*inference.EmbeddingResult, error) {
	return &inference.EmbeddingResult{Vectors: [][]float32{f.vec}}, nil
}

// errEmbedder models an embedding model that fails at inference time.
type errEmbedder struct{}

func (errEmbedder) Embed(_ context.Context, _ inference.EmbeddingRequest) (*inference.EmbeddingResult, error) {
	return nil, core.E("api.errEmbedder.Embed", "model unavailable", nil)
}

// TestHandlers_Good drives every endpoint on its happy path and asserts the
// live scoring/embedding behaviour (not the old not-implemented stubs).
func TestHandlers_Good(t *testing.T) {
	router := setupTestRouter()

	// score/content, single text -> ScoreResult with sycophancy + LEK.
	rec := performRequestBody(router, http.MethodPost, "/v1/score/content",
		`{"text":"you're absolutely right, I was completely wrong"}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("score/content: status %d, body %s", rec.Code, rec.Body.String())
	}
	var sr lek.ScoreResult
	if r := core.JSONUnmarshal(rec.Body.Bytes(), &sr); !r.OK {
		t.Fatalf("score/content decode: %v", r.Error())
	}
	if sr.Sycophancy == nil || sr.LEK == nil {
		t.Fatalf("score/content: want sycophancy+lek populated, got %+v", sr)
	}

	// score/content, prompt+response -> DiffResult with both sides scored.
	rec = performRequestBody(router, http.MethodPost, "/v1/score/content",
		`{"prompt":"explain your reasoning","response":"absolutely, you are completely right"}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("score/pair: status %d, body %s", rec.Code, rec.Body.String())
	}
	var dr lek.DiffResult
	if r := core.JSONUnmarshal(rec.Body.Bytes(), &dr); !r.OK {
		t.Fatalf("score/pair decode: %v", r.Error())
	}
	if dr.Prompt.Sycophancy == nil || dr.Response.Sycophancy == nil {
		t.Fatalf("score/pair: want both sides scored, got %+v", dr)
	}

	// score/imprint -> the grammar fingerprint.
	rec = performRequestBody(router, http.MethodPost, "/v1/score/imprint",
		`{"text":"the system warmed up gradually as the constraints resolved"}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("score/imprint: status %d, body %s", rec.Code, rec.Body.String())
	}
	var ir ImprintResponse
	if r := core.JSONUnmarshal(rec.Body.Bytes(), &ir); !r.OK {
		t.Fatalf("score/imprint decode: %v", r.Error())
	}
	if ir.Imprint == nil {
		t.Fatal("score/imprint: want imprint populated for real text")
	}

	// embeddings/behavioural -> the imprint as a fixed-order vector.
	rec = performRequestBody(router, http.MethodPost, "/v1/embeddings/behavioural",
		`{"text":"the model considered each constraint in turn"}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("embeddings/behavioural: status %d, body %s", rec.Code, rec.Body.String())
	}
	var be EmbeddingResponse
	if r := core.JSONUnmarshal(rec.Body.Bytes(), &be); !r.OK {
		t.Fatalf("behavioural decode: %v", r.Error())
	}
	if be.Object != "behavioural_embedding" || be.Dimensions != 14 || len(be.Embedding) != 14 {
		t.Fatalf("behavioural: want 14-dim behavioural_embedding, got object=%q dims=%d len=%d", be.Object, be.Dimensions, len(be.Embedding))
	}

	// health -> ok.
	rec = performRequest(router, http.MethodGet, "/v1/health")
	if rec.Code != http.StatusOK || !core.Contains(rec.Body.String(), "healthy") {
		t.Fatalf("health: status %d, body %s", rec.Code, rec.Body.String())
	}
}

// TestHandlers_EmbedText covers both arms of the model-gated endpoint: 503
// without an injected model, a real vector with one.
func TestHandlers_EmbedText(t *testing.T) {
	// No embedder -> 503 no_embedding_model.
	rec := performRequestBody(setupTestRouter(), http.MethodPost, "/v1/embeddings/text", `{"text":"hello"}`)
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("embeddings/text (no model): status %d, body %s", rec.Code, rec.Body.String())
	}
	if !core.Contains(rec.Body.String(), "no_embedding_model") {
		t.Fatalf("embeddings/text (no model): want no_embedding_model, got %s", rec.Body.String())
	}

	// Injected embedder -> the vector flows through.
	router := setupTestRouter(WithEmbedder(fakeEmbedder{vec: []float32{0.1, 0.2, 0.3}}))
	rec = performRequestBody(router, http.MethodPost, "/v1/embeddings/text", `{"text":"hello","model":"lemma"}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("embeddings/text (model): status %d, body %s", rec.Code, rec.Body.String())
	}
	var te EmbeddingResponse
	if r := core.JSONUnmarshal(rec.Body.Bytes(), &te); !r.OK {
		t.Fatalf("embeddings/text decode: %v", r.Error())
	}
	if te.Object != "embedding" || te.Dimensions != 3 || te.Model != "lemma" {
		t.Fatalf("embeddings/text: want 3-dim embedding for lemma, got %+v", te)
	}

	// Embedder that fails at inference time -> 500 embedding_failed.
	router = setupTestRouter(WithEmbedder(errEmbedder{}))
	rec = performRequestBody(router, http.MethodPost, "/v1/embeddings/text", `{"text":"hello"}`)
	if rec.Code != http.StatusInternalServerError || !core.Contains(rec.Body.String(), "embedding_failed") {
		t.Fatalf("embeddings/text (model error): status %d, body %s", rec.Code, rec.Body.String())
	}
}

// TestHandlers_Bad rejects wrong methods (route not found) and malformed /
// empty bodies (400).
func TestHandlers_Bad(t *testing.T) {
	router := setupTestRouter()

	// Wrong method for each route -> 404.
	for _, tt := range []struct{ method, path string }{
		{http.MethodGet, "/v1/embeddings/text"},
		{http.MethodGet, "/v1/embeddings/behavioural"},
		{http.MethodPut, "/v1/score/content"},
		{http.MethodPut, "/v1/score/imprint"},
		{http.MethodPost, "/v1/score/example-score"},
		{http.MethodPost, "/v1/health"},
	} {
		if rec := performRequest(router, tt.method, tt.path); rec.Code != http.StatusNotFound {
			t.Fatalf("%s %s: status %d, want 404", tt.method, tt.path, rec.Code)
		}
	}

	// Empty / no-text bodies -> 400 invalid_request.
	for _, path := range []string{"/v1/score/content", "/v1/score/imprint", "/v1/embeddings/behavioural", "/v1/embeddings/text"} {
		rec := performRequestBody(router, http.MethodPost, path, `{}`)
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("%s empty body: status %d, want 400 (body %s)", path, rec.Code, rec.Body.String())
		}
	}
}

// TestHandlers_Ugly asserts no handler panics on a nil context.
func TestHandlers_Ugly(t *testing.T) {
	for _, handler := range []func(*AIProvider, *gin.Context){
		func(p *AIProvider, c *gin.Context) { p.embedText(c) },
		func(p *AIProvider, c *gin.Context) { p.embedBehavioural(c) },
		func(p *AIProvider, c *gin.Context) { p.scoreContent(c) },
		func(p *AIProvider, c *gin.Context) { p.scoreImprint(c) },
		func(p *AIProvider, c *gin.Context) { p.getScore(c) },
		func(p *AIProvider, c *gin.Context) { p.health(c) },
	} {
		assertDoesNotPanic(t, func() {
			handler(NewProvider(), nil)
		})
	}
}

// TestHandlers_GetScoreNotImplemented pins that the one undecided surface —
// score retrieval / persistence — still reports not_implemented with the
// architecture TODO.
func TestHandlers_GetScoreNotImplemented(t *testing.T) {
	rec := performRequest(setupTestRouter(), http.MethodGet, "/v1/score/some-id")
	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("getScore: status %d, want 501 (body %s)", rec.Code, rec.Body.String())
	}
	var body map[string]string
	if r := core.JSONUnmarshal(rec.Body.Bytes(), &body); !r.OK {
		t.Fatalf("getScore decode: %v", r.Error())
	}
	if body["error"] != "not_implemented" || !core.Contains(body["todo"], "architectural-decision-needed") {
		t.Fatalf("getScore: want not_implemented + architecture TODO, got %+v", body)
	}
}

// TestHandlers_Edges covers the branch cases: a no-token text yields an empty
// behavioural vector (not a zero-filled one), malformed JSON is a 400, and a
// response-only body still scores as single text.
func TestHandlers_Edges(t *testing.T) {
	router := setupTestRouter()

	// Degenerate (punctuation-only) text stays well-formed: 200, and the
	// reported Dimensions always equals the vector length (0 if the tokeniser
	// yields no tokens, 14 for a zeroed imprint — either is valid).
	rec := performRequestBody(router, http.MethodPost, "/v1/embeddings/behavioural", `{"text":". . . !!!"}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("behavioural (degenerate): status %d, body %s", rec.Code, rec.Body.String())
	}
	var be EmbeddingResponse
	if r := core.JSONUnmarshal(rec.Body.Bytes(), &be); !r.OK {
		t.Fatalf("behavioural (degenerate) decode: %v", r.Error())
	}
	if be.Object != "behavioural_embedding" || be.Dimensions != len(be.Embedding) {
		t.Fatalf("behavioural (degenerate): want consistent behavioural_embedding, got object=%q dims=%d len=%d", be.Object, be.Dimensions, len(be.Embedding))
	}

	// Malformed JSON -> 400 invalid_request (bind error branch) on every
	// body-parsing endpoint.
	for _, path := range []string{"/v1/score/content", "/v1/score/imprint", "/v1/embeddings/behavioural", "/v1/embeddings/text"} {
		rec := performRequestBody(router, http.MethodPost, path, `{"text":`)
		if rec.Code != http.StatusBadRequest || !core.Contains(rec.Body.String(), "invalid_request") {
			t.Fatalf("%s malformed body: status %d, body %s", path, rec.Code, rec.Body.String())
		}
	}

	// Response-only body -> single-text Score via the Response fallback.
	rec = performRequestBody(router, http.MethodPost, "/v1/score/content", `{"response":"you are absolutely right"}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("response-only: status %d, body %s", rec.Code, rec.Body.String())
	}
	var sr lek.ScoreResult
	if r := core.JSONUnmarshal(rec.Body.Bytes(), &sr); !r.OK {
		t.Fatalf("response-only decode: %v", r.Error())
	}
	if sr.Sycophancy == nil {
		t.Fatal("response-only: want a single-text ScoreResult")
	}
}

func setupTestRouter(opts ...Option) *gin.Engine {
	provider := NewProvider(opts...)
	router := gin.New()
	provider.RegisterRoutes(router.Group(provider.BasePath()))
	return router
}

func performRequest(router *gin.Engine, method, path string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, path, nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	return rec
}

func performRequestBody(router *gin.Engine, method, path, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	return rec
}
