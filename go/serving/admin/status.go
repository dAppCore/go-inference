// SPDX-Licence-Identifier: EUPL-1.2

package admin

import "net/http"

// ServeStatus is the response shape for GET /v1/admin/serve/status — the
// boot-time snapshot of what serve was configured with. Field names stay
// backend-neutral so the same JSON works across the lthn-{mlx,cuda,amd} family;
// Runtime tells the caller which backend produced it.
type ServeStatus struct {
	ModelPath    string            `json:"model_path"`
	Runtime      string            `json:"runtime"`
	LoadedAtUnix int64             `json:"loaded_at_unix"`
	Config       ServeStatusConfig `json:"config"`
	Memory       ServeStatusMemory `json:"memory"`
}

// ServeStatusConfig mirrors the cross-backend load config fields every GPU
// runtime carries. Only the fields serve currently plumbs are rendered.
type ServeStatusConfig struct {
	ContextLength int    `json:"context_length,omitempty"`
	CacheMode     string `json:"cache_mode,omitempty"`
	AdapterPath   string `json:"adapter_path,omitempty"`
}

// ServeStatusMemory is the live GPU memory snapshot. go-inference exposes no
// engine-neutral live-memory read surface yet (lthn-mlx read it from the metal
// allocator), so Available is false and the byte fields are zero until that
// engine seam lands.
type ServeStatusMemory struct {
	Available   bool   `json:"available"`
	ActiveBytes uint64 `json:"active_bytes"`
	CacheBytes  uint64 `json:"cache_bytes"`
	PeakBytes   uint64 `json:"peak_bytes"`
}

// statusHandler returns the boot snapshot of what serve was configured with,
// with ModelPath refreshed to the live active model so a hot-swap reload is
// reflected (the Config block stays the boot snapshot — the effective load
// config after profile + --context resolution). Read-only, GET only, behind
// Bearer auth. currentPath is nil when no resolver is wired.
func statusHandler(snapshot ServeStatus, currentPath func() string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if currentPath != nil {
			if p := currentPath(); p != "" {
				snapshot.ModelPath = p
			}
		}
		writeJSON(w, http.StatusOK, snapshot)
	}
}
