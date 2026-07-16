// SPDX-Licence-Identifier: EUPL-1.2

package compat

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	openaicompat "dappco.re/go/inference/serving/provider/openai"
)

// fakeCacheLister is a fakeTextModel that additionally implements
// CacheEntryLister but NOT inference.CacheService — pins the "lister without
// a stats service" shape of the /v1/cache/entries response (Stats omitted).
type fakeCacheLister struct {
	*fakeTextModel
	entries   []inference.CacheBlockRef
	err       error
	gotLabels map[string]string
}

func (m *fakeCacheLister) CacheEntries(_ context.Context, labels map[string]string) ([]inference.CacheBlockRef, error) {
	m.gotLabels = labels
	if m.err != nil {
		return nil, m.err
	}
	return m.entries, nil
}

// fakeCacheService layers CacheStats/WarmCache/ClearCache onto a
// fakeCacheLister, so it satisfies both CacheEntryLister and
// inference.CacheService — the /v1/cache/entries "stats" block.
type fakeCacheService struct {
	*fakeCacheLister
	stats    inference.CacheStats
	statsErr error
}

func (m *fakeCacheService) CacheStats(context.Context) (inference.CacheStats, error) {
	if m.statsErr != nil {
		return inference.CacheStats{}, m.statsErr
	}
	return m.stats, nil
}
func (m *fakeCacheService) WarmCache(context.Context, inference.CacheWarmRequest) (inference.CacheWarmResult, error) {
	return inference.CacheWarmResult{}, nil
}
func (m *fakeCacheService) ClearCache(context.Context, map[string]string) (inference.CacheStats, error) {
	return inference.CacheStats{}, nil
}

// TestMountAdminHandlers_NilMux_Bad proves the nil-mux guard returns without
// panicking (NewMuxWithAdmin never passes nil, but the guard is the package's
// own defence for any future caller that does).
func TestMountAdminHandlers_NilMux_Bad(t *testing.T) {
	mountAdminHandlers(nil, openaicompat.NewStaticResolver(nil), AdminConfig{})
}

// TestAdminHealthHandler_MethodRejection_Bad proves a non-GET /v1/health is
// rejected before any health callback runs.
func TestAdminHealthHandler_MethodRejection_StatusMethodNotAllowed_Bad(t *testing.T) {
	rec := do(t, http.MethodPost, DefaultHealthPath, "")
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("POST /v1/health = %d, want 405", rec.Code)
	}
}

// TestAdminHealthHandler_DefaultBody_Good proves GET /v1/health with no
// AdminConfig.Health callback reports the built-in status, naming the
// resolver's known models.
func TestAdminHealthHandler_DefaultBody_DefaultHealthPath_Good(t *testing.T) {
	rec := do(t, http.MethodGet, DefaultHealthPath, "")
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /v1/health = %d, want 200", rec.Code)
	}
	body := rec.Body.String()
	if !core.Contains(body, `"status":"ok"`) || !core.Contains(body, "go-inference") {
		t.Fatalf("health body = %s, want the default status+runtime", body)
	}
}

// TestAdminHealthHandler_CustomCallback_Good proves a host-supplied Health
// callback's zero-valued fields (Status/Runtime/Time) are filled with the
// same defaults the built-in path uses, rather than serialising as blank.
func TestAdminHealthHandler_CustomCallback_NewMuxWithAdmin_Good(t *testing.T) {
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"test-model": newFakeTextModel()})
	mux := NewMuxWithAdmin(resolver, AdminConfig{
		Health: func(context.Context) (Health, error) {
			return Health{}, nil
		},
	})
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultHealthPath, nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /v1/health = %d, want 200", rec.Code)
	}
	body := rec.Body.String()
	if !core.Contains(body, `"status":"ok"`) || !core.Contains(body, "go-inference") {
		t.Fatalf("health body = %s, want defaults filled in for a blank custom Health", body)
	}
}

// TestAdminHealthHandler_CustomCallbackError_Bad proves a failing Health
// callback surfaces as a 500 rather than a 200 with an empty body.
func TestAdminHealthHandler_CustomCallbackError_NewMuxWithAdmin_Bad(t *testing.T) {
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"test-model": newFakeTextModel()})
	mux := NewMuxWithAdmin(resolver, AdminConfig{
		Health: func(context.Context) (Health, error) {
			return Health{}, core.NewError("health check boom")
		},
	})
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultHealthPath, nil))
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("GET /v1/health (failing callback) = %d, want 500", rec.Code)
	}
	if !core.Contains(rec.Body.String(), "health check boom") {
		t.Fatalf("body = %s, want the callback error", rec.Body.String())
	}
}

// TestAdminActionHandler_MethodRejection_Bad proves a non-POST /v1/runtime/wake
// is rejected before any callback runs.
func TestAdminActionHandler_MethodRejection_StatusMethodNotAllowed_Bad(t *testing.T) {
	rec := do(t, http.MethodGet, DefaultAdminWakePath, "")
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("GET /v1/runtime/wake = %d, want 405", rec.Code)
	}
}

// TestAdminActionHandler_NilCallback_Good proves wake/sleep with no host
// callback wired still answers 200 "ok" — the routes are always mounted even
// when the host has nothing to hook them to.
func TestAdminActionHandler_NilCallback_DefaultAdminWakePath_Good(t *testing.T) {
	for _, path := range []string{DefaultAdminWakePath, DefaultAdminSleepPath} {
		rec := do(t, http.MethodPost, path, "")
		if rec.Code != http.StatusOK {
			t.Fatalf("POST %s = %d, want 200", path, rec.Code)
		}
		if !core.Contains(rec.Body.String(), `"status":"ok"`) {
			t.Fatalf("POST %s body = %s, want status ok", path, rec.Body.String())
		}
	}
}

// TestAdminActionHandler_CallbackError_Bad proves a failing wake/sleep
// callback is surfaced as a 500, not swallowed into a 200.
func TestAdminActionHandler_CallbackError_NewMuxWithAdmin_Bad(t *testing.T) {
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"test-model": newFakeTextModel()})
	mux := NewMuxWithAdmin(resolver, AdminConfig{
		Wake: func(context.Context) error { return core.NewError("wake boom") },
	})
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultAdminWakePath, nil))
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("POST /v1/runtime/wake (failing callback) = %d, want 500", rec.Code)
	}
	if !core.Contains(rec.Body.String(), "wake boom") {
		t.Fatalf("body = %s, want the callback error", rec.Body.String())
	}
}

// TestAdminActionHandler_DefaultAction_Good proves a zero-valued handler
// (action == "") reports the "runtime" fallback name rather than an empty
// action field — a direct-construction edge NewMux never produces (it always
// sets "wake"/"sleep") but the handler defends against on its own.
func TestAdminActionHandler_DefaultAction_Good(t *testing.T) {
	h := &adminActionHandler{}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/whatever", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if !core.Contains(rec.Body.String(), `"action":"runtime"`) {
		t.Fatalf("body = %s, want the runtime fallback action name", rec.Body.String())
	}
}

// TestAdminCacheEntriesHandler_MethodRejection_Bad proves a non-GET
// /v1/cache/entries is rejected before any model resolve.
func TestAdminCacheEntriesHandler_MethodRejection_StatusMethodNotAllowed_Bad(t *testing.T) {
	rec := do(t, http.MethodPost, DefaultAdminCacheEntriesPath, "")
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("POST /v1/cache/entries = %d, want 405", rec.Code)
	}
}

// TestAdminCacheEntriesHandler_MissingModel_Bad proves an absent ?model= query
// parameter is refused with 400 before any resolve attempt.
func TestAdminCacheEntriesHandler_MissingModel_StatusBadRequest_Bad(t *testing.T) {
	rec := do(t, http.MethodGet, DefaultAdminCacheEntriesPath, "")
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("GET /v1/cache/entries (no model) = %d, want 400", rec.Code)
	}
}

// TestAdminCacheEntriesHandler_UnknownModel_Bad proves an unresolvable model
// name surfaces as a 404, not a 200 or a panic.
func TestAdminCacheEntriesHandler_UnknownModel_StatusNotFound_Bad(t *testing.T) {
	rec := do(t, http.MethodGet, DefaultAdminCacheEntriesPath+"?model=absent", "")
	if rec.Code != http.StatusNotFound {
		t.Fatalf("GET /v1/cache/entries (unknown model) = %d, want 404", rec.Code)
	}
}

// TestAdminCacheEntriesHandler_NotLister_Bad proves a resolved model that
// does not implement CacheEntryLister answers 501, naming the reason, rather
// than a bare panic on the failed type assertion.
func TestAdminCacheEntriesHandler_NotLister_StatusNotImplemented_Bad(t *testing.T) {
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"plain": newFakeTextModel()})
	mux := NewMux(resolver)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultAdminCacheEntriesPath+"?model=plain", nil))
	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("GET /v1/cache/entries (non-lister model) = %d, want 501", rec.Code)
	}
	if !core.Contains(rec.Body.String(), "does not support cache entry listing") {
		t.Fatalf("body = %s, want the not-a-lister reason", rec.Body.String())
	}
}

// TestAdminCacheEntriesHandler_ListerError_Bad proves a CacheEntries failure
// surfaces as a 500.
func TestAdminCacheEntriesHandler_ListerError_FakeCacheLister_Bad(t *testing.T) {
	model := &fakeCacheLister{fakeTextModel: newFakeTextModel(), err: core.NewError("cache read boom")}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"cached": model})
	mux := NewMux(resolver)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultAdminCacheEntriesPath+"?model=cached", nil))
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("GET /v1/cache/entries (lister error) = %d, want 500", rec.Code)
	}
	if !core.Contains(rec.Body.String(), "cache read boom") {
		t.Fatalf("body = %s, want the underlying error", rec.Body.String())
	}
}

// TestAdminCacheEntriesHandler_NoStatsService_Good proves a model that
// implements CacheEntryLister but not inference.CacheService returns its
// entries with the stats block omitted, rather than erroring.
func TestAdminCacheEntriesHandler_NoStatsService_FakeCacheLister_Good(t *testing.T) {
	model := &fakeCacheLister{
		fakeTextModel: newFakeTextModel(),
		entries:       []inference.CacheBlockRef{{ID: "blk-1", TokenCount: 128}},
	}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"cached": model})
	mux := NewMux(resolver)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultAdminCacheEntriesPath+"?model=cached&region=eu", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /v1/cache/entries = %d, want 200 (body: %s)", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	if !core.Contains(body, "blk-1") {
		t.Fatalf("body = %s, want the entry", body)
	}
	if core.Contains(body, `"stats"`) {
		t.Fatalf("body = %s, want no stats block for a non-CacheService model", body)
	}
	if model.gotLabels["region"] != "eu" || model.gotLabels["model"] != "" {
		t.Fatalf("gotLabels = %v, want region=eu and no model key", model.gotLabels)
	}
}

// TestAdminCacheEntriesHandler_StatsError_Bad proves a model implementing
// both CacheEntryLister and inference.CacheService, whose CacheStats call
// fails, surfaces a 500 (the entries themselves already succeeded).
func TestAdminCacheEntriesHandler_StatsError_FakeCacheService_Bad(t *testing.T) {
	model := &fakeCacheService{
		fakeCacheLister: &fakeCacheLister{fakeTextModel: newFakeTextModel()},
		statsErr:        core.NewError("stats boom"),
	}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"cached": model})
	mux := NewMux(resolver)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultAdminCacheEntriesPath+"?model=cached", nil))
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("GET /v1/cache/entries (stats error) = %d, want 500", rec.Code)
	}
	if !core.Contains(rec.Body.String(), "stats boom") {
		t.Fatalf("body = %s, want the underlying stats error", rec.Body.String())
	}
}

// TestAdminCacheEntriesHandler_HappyPath_Good proves a model implementing
// both interfaces returns entries plus the stats block.
func TestAdminCacheEntriesHandler_HappyPath_FakeCacheService_Good(t *testing.T) {
	model := &fakeCacheService{
		fakeCacheLister: &fakeCacheLister{
			fakeTextModel: newFakeTextModel(),
			entries:       []inference.CacheBlockRef{{ID: "blk-1"}},
		},
		stats: inference.CacheStats{Blocks: 1, Hits: 4},
	}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"cached": model})
	mux := NewMux(resolver)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultAdminCacheEntriesPath+"?model=cached", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /v1/cache/entries = %d, want 200 (body: %s)", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	if !core.Contains(body, "blk-1") || !core.Contains(body, `"hits":4`) {
		t.Fatalf("body = %s, want entries + stats", body)
	}
}

// TestAdminCacheEntryLabels_NilRequest_Bad proves a nil *http.Request yields
// an empty label map rather than panicking.
func TestAdminCacheEntryLabels_NilRequest_Bad(t *testing.T) {
	if got := adminCacheEntryLabels(nil); len(got) != 0 {
		t.Fatalf("adminCacheEntryLabels(nil) = %v, want empty", got)
	}
}

// TestAdminCacheEntryLabels_NilURL_Bad proves a request with a nil URL (a
// zero-valued http.Request) is handled the same way.
func TestAdminCacheEntryLabels_NilURL_Bad(t *testing.T) {
	if got := adminCacheEntryLabels(&http.Request{}); len(got) != 0 {
		t.Fatalf("adminCacheEntryLabels(nil URL) = %v, want empty", got)
	}
}

// TestAdminCacheEntryLabels_Good proves the model query key is excluded and
// the remaining query params become the label filter.
func TestAdminCacheEntryLabels_Good(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/v1/cache/entries?model=x&region=eu", nil)
	got := adminCacheEntryLabels(req)
	if got["region"] != "eu" {
		t.Fatalf("labels = %v, want region=eu", got)
	}
	if _, has := got["model"]; has {
		t.Fatalf("labels = %v, want the model key excluded", got)
	}
}

// TestCacheEntryLabelsFrom_Good pins the label-derivation rules: "model" is
// always excluded, blank values are dropped, a multi-value key uses only its
// first value, and a key with a zero-length value slice is skipped outright.
func TestCacheEntryLabelsFrom_Good(t *testing.T) {
	query := core.URLValues{
		"model":  {"ignored"},
		"region": {"eu"},
		"empty":  {""},
		"multi":  {"a", "b"},
		"weird":  {},
	}
	got := cacheEntryLabelsFrom(query)
	want := map[string]string{"region": "eu", "multi": "a"}
	if len(got) != len(want) {
		t.Fatalf("cacheEntryLabelsFrom = %v, want %v", got, want)
	}
	for k, v := range want {
		if got[k] != v {
			t.Fatalf("cacheEntryLabelsFrom[%q] = %q, want %q", k, got[k], v)
		}
	}
}
