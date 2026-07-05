// SPDX-Licence-Identifier: EUPL-1.2

package compat

import (
	"context"
	"net/http"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	openaicompat "dappco.re/go/inference/serving/provider/openai"
)

const (
	DefaultHealthPath            = "/v1/health"
	DefaultAdminWakePath         = "/v1/runtime/wake"
	DefaultAdminSleepPath        = "/v1/runtime/sleep"
	DefaultAdminCacheEntriesPath = "/v1/cache/entries"
)

// AdminConfig supplies host-owned runtime callbacks for the compatibility mux.
type AdminConfig struct {
	Health func(context.Context) (Health, error)
	Wake   func(context.Context) error
	Sleep  func(context.Context) error
}

// Health is the small health payload served by the local compatibility mux.
type Health struct {
	Status  string            `json:"status"`
	Runtime string            `json:"runtime,omitempty"`
	Models  []string          `json:"models,omitempty"`
	Time    int64             `json:"time,omitempty"`
	Labels  map[string]string `json:"labels,omitempty"`
}

// ActionResponse records a runtime wake/sleep callback result.
type ActionResponse struct {
	Action string            `json:"action"`
	Status string            `json:"status"`
	Labels map[string]string `json:"labels,omitempty"`
}

// CacheEntryLister exposes cache block refs without expanding CacheService.
type CacheEntryLister interface {
	CacheEntries(ctx context.Context, labels map[string]string) ([]inference.CacheBlockRef, error)
}

type adminCacheEntriesResponse struct {
	Object  string                    `json:"object"`
	Model   string                    `json:"model,omitempty"`
	Entries []inference.CacheBlockRef `json:"entries"`
	Stats   *inference.CacheStats     `json:"stats,omitempty"`
}

func mountAdminHandlers(mux *http.ServeMux, resolver openaicompat.Resolver, cfg AdminConfig) {
	if mux == nil {
		return
	}
	mux.Handle(DefaultHealthPath, &adminHealthHandler{resolver: resolver, cfg: cfg})
	mux.Handle(DefaultAdminWakePath, &adminActionHandler{action: "wake", callback: cfg.Wake})
	mux.Handle(DefaultAdminSleepPath, &adminActionHandler{action: "sleep", callback: cfg.Sleep})
	mux.Handle(DefaultAdminCacheEntriesPath, &adminCacheEntriesHandler{resolver: resolver})
}

type adminHealthHandler struct {
	resolver openaicompat.Resolver
	cfg      AdminConfig
}

func (h *adminHealthHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !requireCompatMethod(w, r, http.MethodGet) {
		return
	}
	health := Health{
		Status:  "ok",
		Runtime: "go-inference",
		Models:  resolverModelNames(h.resolver),
		Time:    time.Now().Unix(),
	}
	if h != nil && h.cfg.Health != nil {
		custom, err := h.cfg.Health(r.Context())
		if err != nil {
			writeOpenAIError(w, http.StatusInternalServerError, err.Error(), "health")
			return
		}
		health = custom
		if health.Status == "" {
			health.Status = "ok"
		}
		if health.Runtime == "" {
			health.Runtime = "go-inference"
		}
		if health.Time == 0 {
			health.Time = time.Now().Unix()
		}
	}
	writeOpenAIJSON(w, http.StatusOK, health)
}

type adminActionHandler struct {
	action   string
	callback func(context.Context) error
}

func (h *adminActionHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !requireCompatMethod(w, r, http.MethodPost) {
		return
	}
	action := "runtime"
	if h != nil && h.action != "" {
		action = h.action
	}
	if h != nil && h.callback != nil {
		if err := h.callback(r.Context()); err != nil {
			writeOpenAIError(w, http.StatusInternalServerError, err.Error(), action)
			return
		}
	}
	writeOpenAIJSON(w, http.StatusOK, ActionResponse{Action: action, Status: "ok"})
}

type adminCacheEntriesHandler struct {
	resolver openaicompat.Resolver
}

func (h *adminCacheEntriesHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !requireCompatMethod(w, r, http.MethodGet) {
		return
	}
	// Parse the query string once and derive both the model name and the
	// label filters from it. The handler previously called r.URL.Query()
	// twice — once for Get("model") and again inside adminCacheEntryLabels
	// — re-parsing the whole RawQuery (and rebuilding its url.Values map)
	// on every cache-entries request.
	query := r.URL.Query()
	modelName := core.Trim(query.Get("model"))
	model, ok := resolveCompatModel(w, r.Context(), h.resolver, modelName)
	if !ok {
		return
	}
	lister, ok := model.(CacheEntryLister)
	if !ok {
		writeOpenAIError(w, http.StatusNotImplemented, "model does not support cache entry listing", "model")
		return
	}
	labels := cacheEntryLabelsFrom(query)
	entries, err := lister.CacheEntries(r.Context(), labels)
	if err != nil {
		writeOpenAIError(w, http.StatusInternalServerError, err.Error(), "cache")
		return
	}
	response := adminCacheEntriesResponse{
		Object:  "list",
		Model:   modelName,
		Entries: entries,
	}
	if service, ok := model.(inference.CacheService); ok {
		stats, err := service.CacheStats(r.Context())
		if err != nil {
			writeOpenAIError(w, http.StatusInternalServerError, err.Error(), "cache")
			return
		}
		response.Stats = &stats
	}
	writeOpenAIJSON(w, http.StatusOK, response)
}

func adminCacheEntryLabels(r *http.Request) map[string]string {
	if r == nil || r.URL == nil {
		return map[string]string{}
	}
	return cacheEntryLabelsFrom(r.URL.Query())
}

// cacheEntryLabelsFrom builds the label-filter map from an already-parsed
// query. Splitting it out of adminCacheEntryLabels lets the cache-entries
// handler parse r.URL.Query() once and reuse the result for both the model
// name and the labels, rather than re-parsing RawQuery twice per request.
func cacheEntryLabelsFrom(query core.URLValues) map[string]string {
	labels := map[string]string{}
	for key, values := range query {
		if key == "model" || len(values) == 0 {
			continue
		}
		value := core.Trim(values[0])
		if value != "" {
			labels[key] = value
		}
	}
	return labels
}
