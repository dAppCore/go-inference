// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"dappco.re/go/inference"
)

type serviceModel struct {
	*stubModel
	cancelled string
	cleared   bool
	warmed    inference.CacheWarmRequest
}

func (m *serviceModel) Embed(_ context.Context, req inference.EmbeddingRequest) (*inference.EmbeddingResult, error) {
	return &inference.EmbeddingResult{
		Vectors: [][]float32{{float32(len(req.Input)), 0.5}},
		Usage:   inference.EmbeddingUsage{PromptTokens: len(req.Input), TotalTokens: len(req.Input)},
	}, nil
}

func (m *serviceModel) Rerank(_ context.Context, req inference.RerankRequest) (*inference.RerankResult, error) {
	return &inference.RerankResult{
		Results: []inference.RerankScore{{Index: 1, Score: 0.95, Text: req.Documents[1]}},
	}, nil
}

func (m *serviceModel) CacheStats(context.Context) (inference.CacheStats, error) {
	return inference.CacheStats{Blocks: 7, Hits: 9, Misses: 1, HitRate: 0.9, CacheMode: "block-q8"}, nil
}

func (m *serviceModel) WarmCache(_ context.Context, req inference.CacheWarmRequest) (inference.CacheWarmResult, error) {
	m.warmed = req
	return inference.CacheWarmResult{Blocks: []inference.CacheBlockRef{{ID: "blk", TokenCount: len(req.Tokens)}}}, nil
}

func (m *serviceModel) ClearCache(context.Context, map[string]string) (inference.CacheStats, error) {
	m.cleared = true
	return inference.CacheStats{CacheMode: "block-q8"}, nil
}

func (m *serviceModel) CancelRequest(_ context.Context, id string) (inference.RequestCancelResult, error) {
	m.cancelled = id
	return inference.RequestCancelResult{ID: id, Cancelled: id != ""}, nil
}

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

	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("status = %d body=%s, want not implemented", rec.Code, rec.Body.String())
	}
}
