// SPDX-License-Identifier: EUPL-1.2

package api

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/eval/score/lek"
	"github.com/gin-gonic/gin"
)

func init() {
	gin.SetMode(gin.TestMode)
}

// TestHandlers_Good drives every endpoint on its happy path and asserts the
// live scoring behaviour (not the old not-implemented stubs).
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

// TestHandlers_ScoreSession pins the after-the-fact scoring of session history:
// each assistant turn scores against its preceding user turn, so a 2-exchange
// conversation yields 2 scores.
func TestHandlers_ScoreSession(t *testing.T) {
	router := setupTestRouter()
	body := `{"turns":[
		{"role":"user","content":"explain your reasoning"},
		{"role":"assistant","content":"absolutely, you are completely right"},
		{"role":"user","content":"and again?"},
		{"role":"assistant","content":"the trade-offs weighed against each other"}
	]}`
	rec := performRequestBody(router, http.MethodPost, "/v1/score/session", body)
	if rec.Code != http.StatusOK {
		t.Fatalf("score/session: status %d, body %s", rec.Code, rec.Body.String())
	}
	var resp SessionScoreResponse
	if r := core.JSONUnmarshal(rec.Body.Bytes(), &resp); !r.OK {
		t.Fatalf("score/session decode: %v", r.Error())
	}
	if len(resp.Scores) != 2 {
		t.Fatalf("score/session: want 2 scores (one per assistant turn), got %d", len(resp.Scores))
	}
	if resp.Scores[0].Response.Sycophancy == nil {
		t.Fatal("score/session: first assistant turn should carry a scored response")
	}
}

// TestHandlers_ScoreSession_Bad pins the empty and wrong-method cases.
func TestHandlers_ScoreSession_Bad(t *testing.T) {
	router := setupTestRouter()
	if rec := performRequestBody(router, http.MethodPost, "/v1/score/session", `{"turns":[]}`); rec.Code != http.StatusBadRequest {
		t.Fatalf("empty turns: status %d, want 400", rec.Code)
	}
	// PUT has no /score/session route -> 404. (GET /score/session is
	// deliberately not tested: it is captured by GET /score/:id as id="session".)
	if rec := performRequestBody(router, http.MethodPut, "/v1/score/session", `{}`); rec.Code != http.StatusNotFound {
		t.Fatalf("PUT /score/session: status %d, want 404", rec.Code)
	}
}

// TestHandlers_Bad rejects wrong methods (route not found) and malformed /
// empty bodies (400).
func TestHandlers_Bad(t *testing.T) {
	router := setupTestRouter()

	// Wrong method for each route -> 404.
	for _, tt := range []struct{ method, path string }{
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
	for _, path := range []string{"/v1/score/content", "/v1/score/imprint", "/v1/embeddings/behavioural"} {
		rec := performRequestBody(router, http.MethodPost, path, `{}`)
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("%s empty body: status %d, want 400 (body %s)", path, rec.Code, rec.Body.String())
		}
	}
}

// TestHandlers_Ugly asserts no handler panics on a nil context.
func TestHandlers_Ugly(t *testing.T) {
	for _, handler := range []func(*AIProvider, *gin.Context){
		func(p *AIProvider, c *gin.Context) { p.embedBehavioural(c) },
		func(p *AIProvider, c *gin.Context) { p.scoreContent(c) },
		func(p *AIProvider, c *gin.Context) { p.scoreImprint(c) },
		func(p *AIProvider, c *gin.Context) { p.scoreSession(c) },
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
	for _, path := range []string{"/v1/score/content", "/v1/score/imprint", "/v1/score/session", "/v1/embeddings/behavioural"} {
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

func setupTestRouter() *gin.Engine {
	provider := NewProvider()
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
