// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

type serviceModel struct {
	*stubModel
	cancelled string
	cleared   bool
	warmed    inference.CacheWarmRequest
	// err, when set, makes every service method below fail instead of
	// succeeding — covers each handler's "operation returned an error"
	// response branch.
	err error
}

func (m *serviceModel) Embed(_ context.Context, req inference.EmbeddingRequest) (*inference.EmbeddingResult, error) {
	if m.err != nil {
		return nil, m.err
	}
	return &inference.EmbeddingResult{
		Vectors: [][]float32{{float32(len(req.Input)), 0.5}},
		Usage:   inference.EmbeddingUsage{PromptTokens: len(req.Input), TotalTokens: len(req.Input)},
	}, nil
}

func (m *serviceModel) Rerank(_ context.Context, req inference.RerankRequest) (*inference.RerankResult, error) {
	if m.err != nil {
		return nil, m.err
	}
	return &inference.RerankResult{
		Results: []inference.RerankScore{{Index: 1, Score: 0.95, Text: req.Documents[1]}},
	}, nil
}

func (m *serviceModel) CacheStats(context.Context) (inference.CacheStats, error) {
	if m.err != nil {
		return inference.CacheStats{}, m.err
	}
	return inference.CacheStats{Blocks: 7, Hits: 9, Misses: 1, HitRate: 0.9, CacheMode: "block-q8"}, nil
}

func (m *serviceModel) WarmCache(_ context.Context, req inference.CacheWarmRequest) (inference.CacheWarmResult, error) {
	if m.err != nil {
		return inference.CacheWarmResult{}, m.err
	}
	m.warmed = req
	return inference.CacheWarmResult{Blocks: []inference.CacheBlockRef{{ID: "blk", TokenCount: len(req.Tokens)}}}, nil
}

func (m *serviceModel) ClearCache(context.Context, map[string]string) (inference.CacheStats, error) {
	if m.err != nil {
		return inference.CacheStats{}, m.err
	}
	m.cleared = true
	return inference.CacheStats{CacheMode: "block-q8"}, nil
}

func (m *serviceModel) CancelRequest(_ context.Context, id string) (inference.RequestCancelResult, error) {
	if m.err != nil {
		return inference.RequestCancelResult{}, m.err
	}
	m.cancelled = id
	return inference.RequestCancelResult{ID: id, Cancelled: id != ""}, nil
}

// capabilityStubModel is a stubModel that also reports capabilities —
// used to exercise CapabilityHandler's inference.CapabilityReporter
// fast path (as opposed to the inference.TextModelCapabilities
// fallback the plain stubModel takes).
type capabilityStubModel struct {
	stubModel
}

func (m *capabilityStubModel) Capabilities() inference.CapabilityReport {
	return inference.CapabilityReport{
		Runtime:   inference.RuntimeIdentity{Backend: "capability-stub"},
		Available: true,
	}
}

// erroringReadCloser fails every Read — used to exercise
// decodeServiceRequest's io.ReadAll error branch without any real
// network I/O.
type erroringReadCloser struct{}

func (erroringReadCloser) Read([]byte) (int, error) { return 0, errRead }
func (erroringReadCloser) Close() error             { return nil }

var errRead = core.E("test.erroringReadCloser", "synthetic read failure", nil)

func TestOpenAI_EmbeddingsHandler_Good_UsesEmbeddingModel(t *testing.T) {
	model := &serviceModel{stubModel: &stubModel{}}
	handler := NewEmbeddingsHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	req := httptest.NewRequest(http.MethodPost, DefaultEmbeddingsPath, strings.NewReader(`{"model":"qwen","input":["one","two"]}`))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"object":"list"`) || !strings.Contains(rec.Body.String(), `"embedding":[2,0.5]`) {
		t.Fatalf("embedding response = %s", rec.Body.String())
	}
}

func TestOpenAI_RerankHandler_Good_UsesRerankModel(t *testing.T) {
	model := &serviceModel{stubModel: &stubModel{}}
	handler := NewRerankHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	req := httptest.NewRequest(http.MethodPost, DefaultRerankPath, strings.NewReader(`{"model":"qwen","query":"core","documents":["a","b"]}`))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"index":1`) || !strings.Contains(rec.Body.String(), `"score":0.95`) {
		t.Fatalf("rerank response = %s", rec.Body.String())
	}
}

func TestOpenAI_CapabilityHandler_Good_ReportsModelCapabilities(t *testing.T) {
	model := &serviceModel{stubModel: &stubModel{}}
	handler := NewCapabilityHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	req := httptest.NewRequest(http.MethodGet, DefaultCapabilitiesPath+"?model=qwen", nil)
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"embeddings"`) || !strings.Contains(rec.Body.String(), `"request.cancel"`) {
		t.Fatalf("capability response = %s", rec.Body.String())
	}
}

func TestOpenAI_CacheHandlers_Good_StatsWarmClear(t *testing.T) {
	model := &serviceModel{stubModel: &stubModel{}}
	resolver := NewStaticResolver(map[string]inference.TextModel{"qwen": model})

	statsReq := httptest.NewRequest(http.MethodGet, DefaultCacheStatsPath+"?model=qwen", nil)
	statsRec := httptest.NewRecorder()
	NewCacheStatsHandler(resolver).ServeHTTP(statsRec, statsReq)
	if statsRec.Code != http.StatusOK || !strings.Contains(statsRec.Body.String(), `"hit_rate":0.9`) {
		t.Fatalf("cache stats = %d %s", statsRec.Code, statsRec.Body.String())
	}

	warmReq := httptest.NewRequest(http.MethodPost, DefaultCacheWarmPath, strings.NewReader(`{"model":"qwen","tokens":[1,2,3]}`))
	warmRec := httptest.NewRecorder()
	NewCacheWarmHandler(resolver).ServeHTTP(warmRec, warmReq)
	if warmRec.Code != http.StatusOK || model.warmed.Model.ID != "qwen" || len(model.warmed.Tokens) != 3 {
		t.Fatalf("cache warm = %d %s warmed=%+v", warmRec.Code, warmRec.Body.String(), model.warmed)
	}

	clearReq := httptest.NewRequest(http.MethodPost, DefaultCacheClearPath, strings.NewReader(`{"model":"qwen","labels":{"adapter":"none"}}`))
	clearRec := httptest.NewRecorder()
	NewCacheClearHandler(resolver).ServeHTTP(clearRec, clearReq)
	if clearRec.Code != http.StatusOK || !model.cleared {
		t.Fatalf("cache clear = %d %s cleared=%v", clearRec.Code, clearRec.Body.String(), model.cleared)
	}
}

func TestOpenAI_CancelHandler_Good_UsesCancellableModel(t *testing.T) {
	model := &serviceModel{stubModel: &stubModel{}}
	handler := NewCancelHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	req := httptest.NewRequest(http.MethodPost, DefaultCancelPath, strings.NewReader(`{"model":"qwen","id":"req_1"}`))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if model.cancelled != "req_1" || !strings.Contains(rec.Body.String(), `"cancelled":true`) {
		t.Fatalf("cancel response = %s cancelled=%q", rec.Body.String(), model.cancelled)
	}
}

func TestOpenAI_ServiceHandlers_Bad_UnsupportedInterface(t *testing.T) {
	handler := NewEmbeddingsHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": &stubModel{}}))
	req := httptest.NewRequest(http.MethodPost, DefaultEmbeddingsPath, strings.NewReader(`{"model":"qwen","input":"hello"}`))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d body=%s, want bad request (model cannot embed)", rec.Code, rec.Body.String())
	}
}

// TestOpenAI_ServiceHandlers_Bad_WrongMethod drives every service
// handler's requireServiceMethod call site with the wrong HTTP verb —
// each is a separate uninstrumented call site even though
// requireServiceMethod itself is shared.
func TestOpenAI_ServiceHandlers_Bad_WrongMethod(t *testing.T) {
	resolver := NewStaticResolver(map[string]inference.TextModel{"qwen": &serviceModel{stubModel: &stubModel{}}})
	cases := []struct {
		name    string
		handler http.Handler
		method  string
		path    string
	}{
		{"embeddings", NewEmbeddingsHandler(resolver), http.MethodGet, DefaultEmbeddingsPath},
		{"rerank", NewRerankHandler(resolver), http.MethodGet, DefaultRerankPath},
		{"capability", NewCapabilityHandler(resolver), http.MethodPost, DefaultCapabilitiesPath},
		{"cache-stats", NewCacheStatsHandler(resolver), http.MethodPost, DefaultCacheStatsPath},
		{"cache-warm", NewCacheWarmHandler(resolver), http.MethodGet, DefaultCacheWarmPath},
		{"cache-clear", NewCacheClearHandler(resolver), http.MethodGet, DefaultCacheClearPath},
		{"cancel", NewCancelHandler(resolver), http.MethodGet, DefaultCancelPath},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			tc.handler.ServeHTTP(rec, httptest.NewRequest(tc.method, tc.path, nil))
			if rec.Code != http.StatusMethodNotAllowed {
				t.Fatalf("status = %d body=%s, want 405", rec.Code, rec.Body.String())
			}
			if got := rec.Header().Get("Allow"); got == "" {
				t.Fatalf("Allow header not set")
			}
		})
	}
}

// TestOpenAI_ServiceHandlers_Bad_MalformedBody drives every body-
// decoding handler's decodeServiceRequest call site with unparsable
// JSON.
func TestOpenAI_ServiceHandlers_Bad_MalformedBody(t *testing.T) {
	resolver := NewStaticResolver(map[string]inference.TextModel{"qwen": &serviceModel{stubModel: &stubModel{}}})
	cases := []struct {
		name    string
		handler http.Handler
		path    string
	}{
		{"embeddings", NewEmbeddingsHandler(resolver), DefaultEmbeddingsPath},
		{"rerank", NewRerankHandler(resolver), DefaultRerankPath},
		{"cache-warm", NewCacheWarmHandler(resolver), DefaultCacheWarmPath},
		{"cache-clear", NewCacheClearHandler(resolver), DefaultCacheClearPath},
		{"cancel", NewCancelHandler(resolver), DefaultCancelPath},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			tc.handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, tc.path, strings.NewReader(`{`)))
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status = %d body=%s, want 400", rec.Code, rec.Body.String())
			}
		})
	}
}

// TestOpenAI_EmbeddingsHandler_Bad_Validation covers the model-
// required and input-required rejections.
func TestOpenAI_EmbeddingsHandler_Bad_Validation(t *testing.T) {
	resolver := NewStaticResolver(map[string]inference.TextModel{"qwen": &serviceModel{stubModel: &stubModel{}}})
	handler := NewEmbeddingsHandler(resolver)
	cases := map[string]string{
		"model-empty": `{"input":"hi"}`,
		"input-empty": `{"model":"qwen","input":[]}`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultEmbeddingsPath, strings.NewReader(body)))
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status = %d body=%s, want 400", rec.Code, rec.Body.String())
			}
		})
	}
}

// TestOpenAI_EmbeddingsHandler_Bad_ModelError covers the Embed()
// error-propagation branch.
func TestOpenAI_EmbeddingsHandler_Bad_ModelError(t *testing.T) {
	model := &serviceModel{stubModel: &stubModel{}, err: core.E("test", "embed failed", nil)}
	handler := NewEmbeddingsHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultEmbeddingsPath, strings.NewReader(`{"model":"qwen","input":"hi"}`)))

	if rec.Code != http.StatusInternalServerError || !strings.Contains(rec.Body.String(), "embed failed") {
		t.Fatalf("status = %d body = %s, want 500 embed failed", rec.Code, rec.Body.String())
	}
}

// TestOpenAI_RerankHandler_Bad_Validation covers model/query/documents
// required rejections, the resolve failure, and the unsupported-
// interface rejection.
func TestOpenAI_RerankHandler_Bad_Validation(t *testing.T) {
	resolver := NewStaticResolver(map[string]inference.TextModel{"qwen": &serviceModel{stubModel: &stubModel{}}})
	handler := NewRerankHandler(resolver)
	cases := map[string]struct {
		body string
		want int
	}{
		"model-empty":     {`{"query":"q","documents":["a"]}`, http.StatusBadRequest},
		"query-empty":     {`{"model":"qwen","documents":["a"]}`, http.StatusBadRequest},
		"documents-empty": {`{"model":"qwen","query":"q","documents":[]}`, http.StatusBadRequest},
		"model-not-found": {`{"model":"missing","query":"q","documents":["a"]}`, http.StatusNotFound},
		"unsupported":     {`{"model":"plain","query":"q","documents":["a"]}`, http.StatusBadRequest},
	}
	resolverWithPlain := NewStaticResolver(map[string]inference.TextModel{
		"qwen": &serviceModel{stubModel: &stubModel{}}, "plain": &stubModel{},
	})
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			h := handler
			if name == "unsupported" {
				h = NewRerankHandler(resolverWithPlain)
			}
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultRerankPath, strings.NewReader(tc.body)))
			if rec.Code != tc.want {
				t.Fatalf("status = %d body=%s, want %d", rec.Code, rec.Body.String(), tc.want)
			}
		})
	}
}

// TestOpenAI_RerankHandler_Bad_ModelError covers the Rerank()
// error-propagation branch.
func TestOpenAI_RerankHandler_Bad_ModelError(t *testing.T) {
	model := &serviceModel{stubModel: &stubModel{}, err: core.E("test", "rerank failed", nil)}
	handler := NewRerankHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultRerankPath, strings.NewReader(`{"model":"qwen","query":"q","documents":["a"]}`)))

	if rec.Code != http.StatusInternalServerError || !strings.Contains(rec.Body.String(), "rerank failed") {
		t.Fatalf("status = %d body = %s, want 500 rerank failed", rec.Code, rec.Body.String())
	}
}

// TestOpenAI_CapabilityHandler_Bad covers the missing-model-parameter
// and resolve-failure rejections.
func TestOpenAI_CapabilityHandler_Bad(t *testing.T) {
	handler := NewCapabilityHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": &stubModel{}}))

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultCapabilitiesPath, nil))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("missing model: status = %d body=%s, want 400", rec.Code, rec.Body.String())
	}

	rec = httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultCapabilitiesPath+"?model=missing", nil))
	if rec.Code != http.StatusNotFound {
		t.Fatalf("unknown model: status = %d body=%s, want 404", rec.Code, rec.Body.String())
	}
}

// TestOpenAI_CapabilityHandler_Good_ReporterFastPath covers the
// inference.CapabilityReporter branch — as opposed to the
// TextModelCapabilities fallback the other Capability test takes.
func TestOpenAI_CapabilityHandler_Good_ReporterFastPath(t *testing.T) {
	model := &capabilityStubModel{}
	handler := NewCapabilityHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultCapabilitiesPath+"?model=qwen", nil))

	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"capability-stub"`) {
		t.Fatalf("status = %d body=%s, want reporter-sourced capabilities", rec.Code, rec.Body.String())
	}
}

// TestOpenAI_CapabilityHandler_Good_ToolParseForGemma4 pins #37's capability
// honesty: a Gemma 4 architecture (no CapabilityReporter of its own, so this
// drives the TextModelCapabilities fallback branch withServingCapabilities
// layers onto) reports tool.parse supported.
func TestOpenAI_CapabilityHandler_Good_ToolParseForGemma4(t *testing.T) {
	handler := NewCapabilityHandler(NewStaticResolver(map[string]inference.TextModel{"gemma": &recordingModel{arch: "gemma4_text"}}))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultCapabilitiesPath+"?model=gemma", nil))

	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"tool.parse"`) {
		t.Fatalf("status = %d body=%s, want tool.parse reported for a gemma4 architecture", rec.Code, rec.Body.String())
	}
}

func TestOpenAI_CapabilityHandler_Good_ToolParseForLlama3(t *testing.T) {
	handler := NewCapabilityHandler(NewStaticResolver(map[string]inference.TextModel{"llama": &recordingModel{arch: "llama3_1"}}))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultCapabilitiesPath+"?model=llama", nil))

	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"tool.parse"`) {
		t.Fatalf("status = %d body=%s, want tool.parse reported for a llama3 architecture", rec.Code, rec.Body.String())
	}
}

// TestOpenAI_CapabilityHandler_Bad_ToolParseAbsentForNonGemma pins the honest
// negative: an architecture with no Gemma 4 tool syntax never gets a tool.parse
// entry — omission, not a misleading "supported", matching the same
// present-only-when-true convention TextModelCapabilities already uses for
// every other optional capability.
func TestOpenAI_CapabilityHandler_Bad_ToolParseAbsentForNonGemma(t *testing.T) {
	handler := NewCapabilityHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": &recordingModel{arch: "qwen3"}}))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultCapabilitiesPath+"?model=qwen", nil))

	if rec.Code != http.StatusOK || strings.Contains(rec.Body.String(), `"tool.parse"`) {
		t.Fatalf("status = %d body=%s, want no tool.parse entry for a non-gemma4 architecture", rec.Code, rec.Body.String())
	}
}

// TestOpenAI_CapabilityHandler_Good_StructuredOutputAlwaysReported pins that
// structured.output is reported regardless of architecture — it works from
// the model's plain visible text alone (serving/structured), needing no
// native support.
func TestOpenAI_CapabilityHandler_Good_StructuredOutputAlwaysReported(t *testing.T) {
	handler := NewCapabilityHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": &stubModel{}}))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultCapabilitiesPath+"?model=qwen", nil))

	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"structured.output"`) {
		t.Fatalf("status = %d body=%s, want structured.output reported regardless of architecture", rec.Code, rec.Body.String())
	}
}

// TestOpenAI_CacheHandlers_Bad_MissingModel exercises resolve's own
// modelName=="" rejection — reachable only through the cache handlers,
// none of which pre-check model emptiness before calling
// resolveCacheService (unlike Embeddings/Rerank/Cancel).
func TestOpenAI_CacheHandlers_Bad_MissingModel(t *testing.T) {
	resolver := NewStaticResolver(map[string]inference.TextModel{"qwen": &serviceModel{stubModel: &stubModel{}}})
	rec := httptest.NewRecorder()
	NewCacheStatsHandler(resolver).ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultCacheStatsPath, nil))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d body=%s, want 400", rec.Code, rec.Body.String())
	}
}

// TestOpenAI_CacheHandlers_Bad_UnsupportedInterface exercises
// resolveCacheService's cache,ok type-assert failure across all three
// cache handlers.
func TestOpenAI_CacheHandlers_Bad_UnsupportedInterface(t *testing.T) {
	resolver := NewStaticResolver(map[string]inference.TextModel{"qwen": &stubModel{}})
	cases := []struct {
		name    string
		handler http.Handler
		req     *http.Request
	}{
		{"stats", NewCacheStatsHandler(resolver), httptest.NewRequest(http.MethodGet, DefaultCacheStatsPath+"?model=qwen", nil)},
		{"warm", NewCacheWarmHandler(resolver), httptest.NewRequest(http.MethodPost, DefaultCacheWarmPath, strings.NewReader(`{"model":"qwen"}`))},
		{"clear", NewCacheClearHandler(resolver), httptest.NewRequest(http.MethodPost, DefaultCacheClearPath, strings.NewReader(`{"model":"qwen"}`))},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			tc.handler.ServeHTTP(rec, tc.req)
			if rec.Code != http.StatusNotImplemented {
				t.Fatalf("status = %d body=%s, want 501", rec.Code, rec.Body.String())
			}
		})
	}
}

// TestOpenAI_CacheHandlers_Bad_OperationError covers the
// CacheStats/WarmCache/ClearCache error-propagation branches.
func TestOpenAI_CacheHandlers_Bad_OperationError(t *testing.T) {
	model := &serviceModel{stubModel: &stubModel{}, err: core.E("test", "cache op failed", nil)}
	resolver := NewStaticResolver(map[string]inference.TextModel{"qwen": model})
	cases := []struct {
		name    string
		handler http.Handler
		req     *http.Request
	}{
		{"stats", NewCacheStatsHandler(resolver), httptest.NewRequest(http.MethodGet, DefaultCacheStatsPath+"?model=qwen", nil)},
		{"warm", NewCacheWarmHandler(resolver), httptest.NewRequest(http.MethodPost, DefaultCacheWarmPath, strings.NewReader(`{"model":"qwen"}`))},
		{"clear", NewCacheClearHandler(resolver), httptest.NewRequest(http.MethodPost, DefaultCacheClearPath, strings.NewReader(`{"model":"qwen"}`))},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			tc.handler.ServeHTTP(rec, tc.req)
			if rec.Code != http.StatusInternalServerError || !strings.Contains(rec.Body.String(), "cache op failed") {
				t.Fatalf("status = %d body=%s, want 500 cache op failed", rec.Code, rec.Body.String())
			}
		})
	}
}

// TestOpenAI_CancelHandler_Bad covers id-required, resolve failure,
// unsupported-interface, and the CancelRequest() error-propagation
// branches.
func TestOpenAI_CancelHandler_Bad(t *testing.T) {
	okResolver := NewStaticResolver(map[string]inference.TextModel{
		"qwen": &serviceModel{stubModel: &stubModel{}}, "plain": &stubModel{},
	})
	cases := []struct {
		name    string
		handler http.Handler
		body    string
		want    int
	}{
		{"id-empty", NewCancelHandler(okResolver), `{"model":"qwen"}`, http.StatusBadRequest},
		{"model-not-found", NewCancelHandler(okResolver), `{"model":"missing","id":"r1"}`, http.StatusNotFound},
		{"unsupported", NewCancelHandler(okResolver), `{"model":"plain","id":"r1"}`, http.StatusNotImplemented},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			tc.handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultCancelPath, strings.NewReader(tc.body)))
			if rec.Code != tc.want {
				t.Fatalf("status = %d body=%s, want %d", rec.Code, rec.Body.String(), tc.want)
			}
		})
	}

	errModel := &serviceModel{stubModel: &stubModel{}, err: core.E("test", "cancel failed", nil)}
	handler := NewCancelHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": errModel}))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultCancelPath, strings.NewReader(`{"model":"qwen","id":"r1"}`)))
	if rec.Code != http.StatusInternalServerError || !strings.Contains(rec.Body.String(), "cancel failed") {
		t.Fatalf("status = %d body=%s, want 500 cancel failed", rec.Code, rec.Body.String())
	}
}

// TestOpenAI_ServiceHandlers_Bad_NilResolver drives resolve's
// h.resolver==nil branch across every handler family.
func TestOpenAI_ServiceHandlers_Bad_NilResolver(t *testing.T) {
	cases := []struct {
		name    string
		handler http.Handler
		req     *http.Request
	}{
		{"embeddings", NewEmbeddingsHandler(nil), httptest.NewRequest(http.MethodPost, DefaultEmbeddingsPath, strings.NewReader(`{"model":"m","input":"hi"}`))},
		{"rerank", NewRerankHandler(nil), httptest.NewRequest(http.MethodPost, DefaultRerankPath, strings.NewReader(`{"model":"m","query":"q","documents":["a"]}`))},
		{"capability", NewCapabilityHandler(nil), httptest.NewRequest(http.MethodGet, DefaultCapabilitiesPath+"?model=m", nil)},
		{"cache-stats", NewCacheStatsHandler(nil), httptest.NewRequest(http.MethodGet, DefaultCacheStatsPath+"?model=m", nil)},
		{"cancel", NewCancelHandler(nil), httptest.NewRequest(http.MethodPost, DefaultCancelPath, strings.NewReader(`{"model":"m","id":"r1"}`))},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			tc.handler.ServeHTTP(rec, tc.req)
			if rec.Code != http.StatusServiceUnavailable {
				t.Fatalf("status = %d body=%s, want 503", rec.Code, rec.Body.String())
			}
		})
	}
}

// TestOpenAI_DecodeServiceRequest_Bad drives decodeServiceRequest's
// own branches directly — nil request, nil body, a body that errors
// on Read, and a body that fails JSON decode with a non-empty error
// message.
func TestOpenAI_DecodeServiceRequest_Bad(t *testing.T) {
	var into EmbeddingRequest

	rec := httptest.NewRecorder()
	if decodeServiceRequest(rec, nil, &into, "test") {
		t.Fatal("decodeServiceRequest(nil request) = true, want false")
	}
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("nil request: status = %d, want 400", rec.Code)
	}

	rec = httptest.NewRecorder()
	if decodeServiceRequest(rec, &http.Request{Body: nil}, &into, "test") {
		t.Fatal("decodeServiceRequest(nil body) = true, want false")
	}
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("nil body: status = %d, want 400", rec.Code)
	}

	rec = httptest.NewRecorder()
	if decodeServiceRequest(rec, &http.Request{Body: erroringReadCloser{}}, &into, "test") {
		t.Fatal("decodeServiceRequest(read error) = true, want false")
	}
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("read error: status = %d, want 400", rec.Code)
	}

	rec = httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(`{`))
	if decodeServiceRequest(rec, req, &into, "test.scope") {
		t.Fatal("decodeServiceRequest(malformed json) = true, want false")
	}
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "test.scope") {
		t.Fatalf("malformed json: status = %d body=%s, want 400 scoped message", rec.Code, rec.Body.String())
	}
}

// TestOpenAI_RequireServiceMethod_Bad drives requireServiceMethod's
// own nil-request branch directly.
func TestOpenAI_RequireServiceMethod_Bad(t *testing.T) {
	rec := httptest.NewRecorder()
	if requireServiceMethod(rec, nil, http.MethodGet) {
		t.Fatal("requireServiceMethod(nil request) = true, want false")
	}
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", rec.Code)
	}
}

// TestOpenAI_QueryModel_Bad drives queryModel's own nil-request and
// nil-URL branches directly.
func TestOpenAI_QueryModel_Bad(t *testing.T) {
	if got := queryModel(nil); got != "" {
		t.Fatalf("queryModel(nil) = %q, want empty", got)
	}
	if got := queryModel(&http.Request{URL: nil}); got != "" {
		t.Fatalf("queryModel(nil URL) = %q, want empty", got)
	}
}

// --- EmbeddingInput.UnmarshalJSON ---

func TestServices_EmbeddingInput_UnmarshalJSON_Good(t *testing.T) {
	var single EmbeddingInput
	if err := single.UnmarshalJSON([]byte(`"hello"`)); err != nil || !reflect.DeepEqual(single, EmbeddingInput{"hello"}) {
		t.Fatalf("UnmarshalJSON(single) = %v, err = %v", single, err)
	}

	var many EmbeddingInput
	if err := many.UnmarshalJSON([]byte(`["a","b"]`)); err != nil || !reflect.DeepEqual(many, EmbeddingInput{"a", "b"}) {
		t.Fatalf("UnmarshalJSON(array) = %v, err = %v", many, err)
	}
}

// TestServices_EmbeddingInput_UnmarshalJSON_Bad covers rejection of a
// bare JSON number — EmbeddingInput only accepts a string or an array
// of strings.
func TestServices_EmbeddingInput_UnmarshalJSON_Bad(t *testing.T) {
	var bad EmbeddingInput
	if err := bad.UnmarshalJSON([]byte(`42`)); err == nil {
		t.Fatal("UnmarshalJSON(42) returned nil error, want rejection of a bare number")
	}
}

// TestServices_EmbeddingInput_UnmarshalJSON_Ugly covers the JSON null
// literal — it decodes to a nil EmbeddingInput without an error,
// distinct from an empty array.
func TestServices_EmbeddingInput_UnmarshalJSON_Ugly(t *testing.T) {
	input := EmbeddingInput{"stale"}
	if err := input.UnmarshalJSON([]byte(`null`)); err != nil || input != nil {
		t.Fatalf("UnmarshalJSON(null) = %v, err = %v, want nil input and no error", input, err)
	}
}

// --- NewEmbeddingsHandler / EmbeddingsHandler.ServeHTTP ---

func TestServices_NewEmbeddingsHandler_Good(t *testing.T) {
	resolver := NewStaticResolver(map[string]inference.TextModel{"qwen": &serviceModel{stubModel: &stubModel{}}})
	rec := httptest.NewRecorder()

	NewEmbeddingsHandler(resolver).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultEmbeddingsPath, strings.NewReader(`{"model":"qwen","input":"hi"}`)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s, want 200 through the wired resolver", rec.Code, rec.Body.String())
	}
}

// TestServices_NewEmbeddingsHandler_Bad covers a nil resolver — the
// constructor must not panic, and the resulting handler must reject
// every request with 503 (proving the nil actually reached the
// handler rather than being swapped for a default).
func TestServices_NewEmbeddingsHandler_Bad(t *testing.T) {
	rec := httptest.NewRecorder()

	NewEmbeddingsHandler(nil).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultEmbeddingsPath, strings.NewReader(`{"model":"qwen","input":"hi"}`)))

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503 for a nil resolver", rec.Code)
	}
}

// TestServices_NewEmbeddingsHandler_Ugly covers construction with a
// functional ResolverFunc adapter rather than *StaticResolver.
func TestServices_NewEmbeddingsHandler_Ugly(t *testing.T) {
	model := &serviceModel{stubModel: &stubModel{}}
	resolver := ResolverFunc(func(context.Context, string) (inference.TextModel, error) { return model, nil })
	rec := httptest.NewRecorder()

	NewEmbeddingsHandler(resolver).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultEmbeddingsPath, strings.NewReader(`{"model":"qwen","input":"hi"}`)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s, want 200 through a functional Resolver", rec.Code, rec.Body.String())
	}
}

func TestServices_EmbeddingsHandler_ServeHTTP_Good(t *testing.T) {
	model := &serviceModel{stubModel: &stubModel{}}
	handler := NewEmbeddingsHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultEmbeddingsPath, strings.NewReader(`{"model":"qwen","input":["one","two"]}`)))

	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"object":"list"`) {
		t.Fatalf("status = %d body=%s, want 200 embedding list", rec.Code, rec.Body.String())
	}
}

// TestServices_EmbeddingsHandler_ServeHTTP_Bad covers the method-
// rejection branch.
func TestServices_EmbeddingsHandler_ServeHTTP_Bad(t *testing.T) {
	handler := NewEmbeddingsHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": &serviceModel{stubModel: &stubModel{}}}))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultEmbeddingsPath, nil))

	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", rec.Code)
	}
}

// TestServices_EmbeddingsHandler_ServeHTTP_Ugly covers the empty-input
// rejection — a present but empty input array.
func TestServices_EmbeddingsHandler_ServeHTTP_Ugly(t *testing.T) {
	handler := NewEmbeddingsHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": &serviceModel{stubModel: &stubModel{}}}))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultEmbeddingsPath, strings.NewReader(`{"model":"qwen","input":[]}`)))

	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), `"param":"input"`) {
		t.Fatalf("status = %d body=%s, want 400 param=input", rec.Code, rec.Body.String())
	}
}

// --- NewRerankHandler / RerankHandler.ServeHTTP ---

func TestServices_NewRerankHandler_Good(t *testing.T) {
	resolver := NewStaticResolver(map[string]inference.TextModel{"qwen": &serviceModel{stubModel: &stubModel{}}})
	rec := httptest.NewRecorder()

	NewRerankHandler(resolver).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultRerankPath, strings.NewReader(`{"model":"qwen","query":"q","documents":["a","b"]}`)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s, want 200 through the wired resolver", rec.Code, rec.Body.String())
	}
}

// TestServices_NewRerankHandler_Bad covers a nil resolver.
func TestServices_NewRerankHandler_Bad(t *testing.T) {
	rec := httptest.NewRecorder()

	NewRerankHandler(nil).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultRerankPath, strings.NewReader(`{"model":"qwen","query":"q","documents":["a"]}`)))

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503 for a nil resolver", rec.Code)
	}
}

// TestServices_NewRerankHandler_Ugly covers construction with a
// functional ResolverFunc adapter.
func TestServices_NewRerankHandler_Ugly(t *testing.T) {
	model := &serviceModel{stubModel: &stubModel{}}
	resolver := ResolverFunc(func(context.Context, string) (inference.TextModel, error) { return model, nil })
	rec := httptest.NewRecorder()

	NewRerankHandler(resolver).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultRerankPath, strings.NewReader(`{"model":"qwen","query":"q","documents":["a","b"]}`)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s, want 200 through a functional Resolver", rec.Code, rec.Body.String())
	}
}

func TestServices_RerankHandler_ServeHTTP_Good(t *testing.T) {
	model := &serviceModel{stubModel: &stubModel{}}
	handler := NewRerankHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultRerankPath, strings.NewReader(`{"model":"qwen","query":"core","documents":["a","b"]}`)))

	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"index":1`) {
		t.Fatalf("status = %d body=%s, want 200 rerank results", rec.Code, rec.Body.String())
	}
}

// TestServices_RerankHandler_ServeHTTP_Bad covers the method-rejection
// branch.
func TestServices_RerankHandler_ServeHTTP_Bad(t *testing.T) {
	handler := NewRerankHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": &serviceModel{stubModel: &stubModel{}}}))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultRerankPath, nil))

	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", rec.Code)
	}
}

// TestServices_RerankHandler_ServeHTTP_Ugly covers the empty-documents
// rejection.
func TestServices_RerankHandler_ServeHTTP_Ugly(t *testing.T) {
	handler := NewRerankHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": &serviceModel{stubModel: &stubModel{}}}))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultRerankPath, strings.NewReader(`{"model":"qwen","query":"q","documents":[]}`)))

	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), `"param":"documents"`) {
		t.Fatalf("status = %d body=%s, want 400 param=documents", rec.Code, rec.Body.String())
	}
}

// --- NewCapabilityHandler / CapabilityHandler.ServeHTTP ---

func TestServices_NewCapabilityHandler_Good(t *testing.T) {
	resolver := NewStaticResolver(map[string]inference.TextModel{"qwen": &serviceModel{stubModel: &stubModel{}}})
	rec := httptest.NewRecorder()

	NewCapabilityHandler(resolver).ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultCapabilitiesPath+"?model=qwen", nil))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s, want 200 through the wired resolver", rec.Code, rec.Body.String())
	}
}

// TestServices_NewCapabilityHandler_Bad covers a nil resolver.
func TestServices_NewCapabilityHandler_Bad(t *testing.T) {
	rec := httptest.NewRecorder()

	NewCapabilityHandler(nil).ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultCapabilitiesPath+"?model=qwen", nil))

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503 for a nil resolver", rec.Code)
	}
}

// TestServices_NewCapabilityHandler_Ugly covers construction with a
// functional ResolverFunc adapter.
func TestServices_NewCapabilityHandler_Ugly(t *testing.T) {
	model := &serviceModel{stubModel: &stubModel{}}
	resolver := ResolverFunc(func(context.Context, string) (inference.TextModel, error) { return model, nil })
	rec := httptest.NewRecorder()

	NewCapabilityHandler(resolver).ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultCapabilitiesPath+"?model=qwen", nil))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s, want 200 through a functional Resolver", rec.Code, rec.Body.String())
	}
}

func TestServices_CapabilityHandler_ServeHTTP_Good(t *testing.T) {
	model := &serviceModel{stubModel: &stubModel{}}
	handler := NewCapabilityHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultCapabilitiesPath+"?model=qwen", nil))

	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"embeddings"`) {
		t.Fatalf("status = %d body=%s, want 200 fallback capability report", rec.Code, rec.Body.String())
	}
}

// TestServices_CapabilityHandler_ServeHTTP_Bad covers the missing-
// model-parameter rejection.
func TestServices_CapabilityHandler_ServeHTTP_Bad(t *testing.T) {
	handler := NewCapabilityHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": &serviceModel{stubModel: &stubModel{}}}))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultCapabilitiesPath, nil))

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", rec.Code)
	}
}

// TestServices_CapabilityHandler_ServeHTTP_Ugly covers the
// inference.CapabilityReporter fast path, distinct from Good's
// TextModelCapabilities fallback.
func TestServices_CapabilityHandler_ServeHTTP_Ugly(t *testing.T) {
	handler := NewCapabilityHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": &capabilityStubModel{}}))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultCapabilitiesPath+"?model=qwen", nil))

	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"capability-stub"`) {
		t.Fatalf("status = %d body=%s, want reporter-sourced capabilities", rec.Code, rec.Body.String())
	}
}

// --- NewCacheStatsHandler / CacheStatsHandler.ServeHTTP ---

func TestServices_NewCacheStatsHandler_Good(t *testing.T) {
	resolver := NewStaticResolver(map[string]inference.TextModel{"qwen": &serviceModel{stubModel: &stubModel{}}})
	rec := httptest.NewRecorder()

	NewCacheStatsHandler(resolver).ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultCacheStatsPath+"?model=qwen", nil))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s, want 200 through the wired resolver", rec.Code, rec.Body.String())
	}
}

// TestServices_NewCacheStatsHandler_Bad covers a nil resolver.
func TestServices_NewCacheStatsHandler_Bad(t *testing.T) {
	rec := httptest.NewRecorder()

	NewCacheStatsHandler(nil).ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultCacheStatsPath+"?model=qwen", nil))

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503 for a nil resolver", rec.Code)
	}
}

// TestServices_NewCacheStatsHandler_Ugly covers construction with a
// functional ResolverFunc adapter.
func TestServices_NewCacheStatsHandler_Ugly(t *testing.T) {
	model := &serviceModel{stubModel: &stubModel{}}
	resolver := ResolverFunc(func(context.Context, string) (inference.TextModel, error) { return model, nil })
	rec := httptest.NewRecorder()

	NewCacheStatsHandler(resolver).ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultCacheStatsPath+"?model=qwen", nil))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s, want 200 through a functional Resolver", rec.Code, rec.Body.String())
	}
}

func TestServices_CacheStatsHandler_ServeHTTP_Good(t *testing.T) {
	model := &serviceModel{stubModel: &stubModel{}}
	handler := NewCacheStatsHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultCacheStatsPath+"?model=qwen", nil))

	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"hit_rate":0.9`) {
		t.Fatalf("status = %d body=%s, want 200 cache stats", rec.Code, rec.Body.String())
	}
}

// TestServices_CacheStatsHandler_ServeHTTP_Bad covers the method-
// rejection branch.
func TestServices_CacheStatsHandler_ServeHTTP_Bad(t *testing.T) {
	handler := NewCacheStatsHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": &serviceModel{stubModel: &stubModel{}}}))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultCacheStatsPath, nil))

	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", rec.Code)
	}
}

// TestServices_CacheStatsHandler_ServeHTTP_Ugly covers the
// unsupported-interface rejection — a model that isn't a
// inference.CacheService.
func TestServices_CacheStatsHandler_ServeHTTP_Ugly(t *testing.T) {
	handler := NewCacheStatsHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": &stubModel{}}))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultCacheStatsPath+"?model=qwen", nil))

	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("status = %d, want 501", rec.Code)
	}
}

// --- NewCacheWarmHandler / CacheWarmHandler.ServeHTTP ---

func TestServices_NewCacheWarmHandler_Good(t *testing.T) {
	resolver := NewStaticResolver(map[string]inference.TextModel{"qwen": &serviceModel{stubModel: &stubModel{}}})
	rec := httptest.NewRecorder()

	NewCacheWarmHandler(resolver).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultCacheWarmPath, strings.NewReader(`{"model":"qwen","tokens":[1,2,3]}`)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s, want 200 through the wired resolver", rec.Code, rec.Body.String())
	}
}

// TestServices_NewCacheWarmHandler_Bad covers a nil resolver.
func TestServices_NewCacheWarmHandler_Bad(t *testing.T) {
	rec := httptest.NewRecorder()

	NewCacheWarmHandler(nil).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultCacheWarmPath, strings.NewReader(`{"model":"qwen"}`)))

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503 for a nil resolver", rec.Code)
	}
}

// TestServices_NewCacheWarmHandler_Ugly covers construction with a
// functional ResolverFunc adapter.
func TestServices_NewCacheWarmHandler_Ugly(t *testing.T) {
	model := &serviceModel{stubModel: &stubModel{}}
	resolver := ResolverFunc(func(context.Context, string) (inference.TextModel, error) { return model, nil })
	rec := httptest.NewRecorder()

	NewCacheWarmHandler(resolver).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultCacheWarmPath, strings.NewReader(`{"model":"qwen","tokens":[1,2,3]}`)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s, want 200 through a functional Resolver", rec.Code, rec.Body.String())
	}
}

func TestServices_CacheWarmHandler_ServeHTTP_Good(t *testing.T) {
	model := &serviceModel{stubModel: &stubModel{}}
	handler := NewCacheWarmHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultCacheWarmPath, strings.NewReader(`{"model":"qwen","tokens":[1,2,3]}`)))

	if rec.Code != http.StatusOK || model.warmed.Model.ID != "qwen" || len(model.warmed.Tokens) != 3 {
		t.Fatalf("status = %d body=%s warmed=%+v, want 200 with the warmed request recorded", rec.Code, rec.Body.String(), model.warmed)
	}
}

// TestServices_CacheWarmHandler_ServeHTTP_Bad covers the method-
// rejection branch.
func TestServices_CacheWarmHandler_ServeHTTP_Bad(t *testing.T) {
	handler := NewCacheWarmHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": &serviceModel{stubModel: &stubModel{}}}))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultCacheWarmPath, nil))

	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", rec.Code)
	}
}

// TestServices_CacheWarmHandler_ServeHTTP_Ugly covers the malformed-
// body rejection.
func TestServices_CacheWarmHandler_ServeHTTP_Ugly(t *testing.T) {
	handler := NewCacheWarmHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": &serviceModel{stubModel: &stubModel{}}}))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultCacheWarmPath, strings.NewReader(`{`)))

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", rec.Code)
	}
}

// --- NewCacheClearHandler / CacheClearHandler.ServeHTTP ---

func TestServices_NewCacheClearHandler_Good(t *testing.T) {
	resolver := NewStaticResolver(map[string]inference.TextModel{"qwen": &serviceModel{stubModel: &stubModel{}}})
	rec := httptest.NewRecorder()

	NewCacheClearHandler(resolver).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultCacheClearPath, strings.NewReader(`{"model":"qwen"}`)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s, want 200 through the wired resolver", rec.Code, rec.Body.String())
	}
}

// TestServices_NewCacheClearHandler_Bad covers a nil resolver.
func TestServices_NewCacheClearHandler_Bad(t *testing.T) {
	rec := httptest.NewRecorder()

	NewCacheClearHandler(nil).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultCacheClearPath, strings.NewReader(`{"model":"qwen"}`)))

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503 for a nil resolver", rec.Code)
	}
}

// TestServices_NewCacheClearHandler_Ugly covers construction with a
// functional ResolverFunc adapter.
func TestServices_NewCacheClearHandler_Ugly(t *testing.T) {
	model := &serviceModel{stubModel: &stubModel{}}
	resolver := ResolverFunc(func(context.Context, string) (inference.TextModel, error) { return model, nil })
	rec := httptest.NewRecorder()

	NewCacheClearHandler(resolver).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultCacheClearPath, strings.NewReader(`{"model":"qwen"}`)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s, want 200 through a functional Resolver", rec.Code, rec.Body.String())
	}
}

func TestServices_CacheClearHandler_ServeHTTP_Good(t *testing.T) {
	model := &serviceModel{stubModel: &stubModel{}}
	handler := NewCacheClearHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultCacheClearPath, strings.NewReader(`{"model":"qwen","labels":{"adapter":"none"}}`)))

	if rec.Code != http.StatusOK || !model.cleared {
		t.Fatalf("status = %d body=%s cleared=%v, want 200 with cache cleared", rec.Code, rec.Body.String(), model.cleared)
	}
}

// TestServices_CacheClearHandler_ServeHTTP_Bad covers the method-
// rejection branch.
func TestServices_CacheClearHandler_ServeHTTP_Bad(t *testing.T) {
	handler := NewCacheClearHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": &serviceModel{stubModel: &stubModel{}}}))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultCacheClearPath, nil))

	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", rec.Code)
	}
}

// TestServices_CacheClearHandler_ServeHTTP_Ugly covers the
// ClearCache()-error-propagation branch.
func TestServices_CacheClearHandler_ServeHTTP_Ugly(t *testing.T) {
	model := &serviceModel{stubModel: &stubModel{}, err: core.E("test", "clear failed", nil)}
	handler := NewCacheClearHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultCacheClearPath, strings.NewReader(`{"model":"qwen"}`)))

	if rec.Code != http.StatusInternalServerError || !strings.Contains(rec.Body.String(), "clear failed") {
		t.Fatalf("status = %d body=%s, want 500 clear failed", rec.Code, rec.Body.String())
	}
}

// --- NewCancelHandler / CancelHandler.ServeHTTP ---

func TestServices_NewCancelHandler_Good(t *testing.T) {
	resolver := NewStaticResolver(map[string]inference.TextModel{"qwen": &serviceModel{stubModel: &stubModel{}}})
	rec := httptest.NewRecorder()

	NewCancelHandler(resolver).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultCancelPath, strings.NewReader(`{"model":"qwen","id":"r1"}`)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s, want 200 through the wired resolver", rec.Code, rec.Body.String())
	}
}

// TestServices_NewCancelHandler_Bad covers a nil resolver.
func TestServices_NewCancelHandler_Bad(t *testing.T) {
	rec := httptest.NewRecorder()

	NewCancelHandler(nil).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultCancelPath, strings.NewReader(`{"model":"qwen","id":"r1"}`)))

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503 for a nil resolver", rec.Code)
	}
}

// TestServices_NewCancelHandler_Ugly covers construction with a
// functional ResolverFunc adapter.
func TestServices_NewCancelHandler_Ugly(t *testing.T) {
	model := &serviceModel{stubModel: &stubModel{}}
	resolver := ResolverFunc(func(context.Context, string) (inference.TextModel, error) { return model, nil })
	rec := httptest.NewRecorder()

	NewCancelHandler(resolver).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultCancelPath, strings.NewReader(`{"model":"qwen","id":"r1"}`)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s, want 200 through a functional Resolver", rec.Code, rec.Body.String())
	}
}

func TestServices_CancelHandler_ServeHTTP_Good(t *testing.T) {
	model := &serviceModel{stubModel: &stubModel{}}
	handler := NewCancelHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultCancelPath, strings.NewReader(`{"model":"qwen","id":"req_1"}`)))

	if rec.Code != http.StatusOK || model.cancelled != "req_1" {
		t.Fatalf("status = %d body=%s cancelled=%q, want 200 with the request cancelled", rec.Code, rec.Body.String(), model.cancelled)
	}
}

// TestServices_CancelHandler_ServeHTTP_Bad covers the method-rejection
// branch.
func TestServices_CancelHandler_ServeHTTP_Bad(t *testing.T) {
	handler := NewCancelHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": &serviceModel{stubModel: &stubModel{}}}))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultCancelPath, nil))

	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", rec.Code)
	}
}

// TestServices_CancelHandler_ServeHTTP_Ugly covers the id-required
// rejection.
func TestServices_CancelHandler_ServeHTTP_Ugly(t *testing.T) {
	handler := NewCancelHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": &serviceModel{stubModel: &stubModel{}}}))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultCancelPath, strings.NewReader(`{"model":"qwen"}`)))

	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), `"param":"id"`) {
		t.Fatalf("status = %d body=%s, want 400 param=id", rec.Code, rec.Body.String())
	}
}
