// SPDX-License-Identifier: EUPL-1.2

package ai

import (
	"net/http"

	core "dappco.re/go"
)

// NewBookStateDemoHandler exposes a small JSON API for the book-state demo.
//
// Endpoints:
//   - GET /health
//   - GET /state
//   - POST /ask with BookStateAskRequest
func NewBookStateDemoHandler(demo *BookStateDemo) http.Handler {
	return bookStateDemoHandler{demo: demo}
}

type bookStateDemoHandler struct {
	demo *BookStateDemo
}

func (h bookStateDemoHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/health":
		h.serveHealth(w, r)
	case "/state":
		h.serveState(w, r)
	case "/ask":
		h.serveAsk(w, r)
	default:
		writeBookStateError(w, http.StatusNotFound, "not found")
	}
}

func (h bookStateDemoHandler) serveHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeBookStateError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	writeBookStateJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (h bookStateDemoHandler) serveState(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeBookStateError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if h.demo == nil {
		writeBookStateError(w, http.StatusInternalServerError, "demo is nil")
		return
	}
	writeBookStateJSON(w, http.StatusOK, h.demo.State())
}

func (h bookStateDemoHandler) serveAsk(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeBookStateError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if h.demo == nil {
		writeBookStateError(w, http.StatusInternalServerError, "demo is nil")
		return
	}
	dataResult := core.ReadAll(r.Body)
	if !dataResult.OK {
		writeBookStateError(w, http.StatusBadRequest, "read request body")
		return
	}
	var request BookStateAskRequest
	if result := core.JSONUnmarshalString(dataResult.Value.(string), &request); !result.OK {
		writeBookStateError(w, http.StatusBadRequest, "invalid JSON")
		return
	}
	result := h.demo.Ask(r.Context(), request)
	if !result.OK {
		writeBookStateError(w, http.StatusBadRequest, result.Error())
		return
	}
	writeBookStateJSON(w, http.StatusOK, result.Value)
}

func writeBookStateJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write([]byte(core.JSONMarshalString(payload)))
}

func writeBookStateError(w http.ResponseWriter, status int, message string) {
	writeBookStateJSON(w, status, map[string]string{"error": message})
}
