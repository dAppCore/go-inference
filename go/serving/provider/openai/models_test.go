// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestModels_NewModelsHandler_Good(t *testing.T) {
	handler := NewModelsHandler(func() []string { return []string{"gemma-4-e2b-it-4bit"} })

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultModelsPath, nil))

	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"id":"gemma-4-e2b-it-4bit"`) {
		t.Fatalf("status = %d body=%s, want 200 with the configured model", rec.Code, rec.Body.String())
	}
}

// TestModels_NewModelsHandler_Bad covers a nil models callback — the
// handler must still construct and serve an empty list rather than
// panic when nothing has been wired up yet.
func TestModels_NewModelsHandler_Bad(t *testing.T) {
	handler := NewModelsHandler(nil)

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultModelsPath, nil))

	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"data":[]`) {
		t.Fatalf("status = %d body=%s, want 200 empty list", rec.Code, rec.Body.String())
	}
}

// TestModels_NewModelsHandler_Ugly covers that models is invoked fresh
// on every request rather than snapshotted at construction time — the
// callback's answer legitimately changes between requests as a serve
// swaps its loaded --model.
func TestModels_NewModelsHandler_Ugly(t *testing.T) {
	current := "first"
	handler := NewModelsHandler(func() []string { return []string{current} })

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultModelsPath, nil))
	if !strings.Contains(rec.Body.String(), `"id":"first"`) {
		t.Fatalf("first call body=%s, want first", rec.Body.String())
	}

	current = "second"
	rec = httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultModelsPath, nil))
	if !strings.Contains(rec.Body.String(), `"id":"second"`) {
		t.Fatalf("second call body=%s, want second", rec.Body.String())
	}
}

func TestModels_ModelsHandler_ServeHTTP_Good(t *testing.T) {
	handler := NewModelsHandler(func() []string { return []string{"a", "b"} })

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultModelsPath, nil))

	body := rec.Body.String()
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if !strings.Contains(body, `"object":"list"`) || !strings.Contains(body, `"id":"a"`) ||
		!strings.Contains(body, `"id":"b"`) || !strings.Contains(body, `"owned_by":"lethean"`) {
		t.Fatalf("body = %s, want list of both models", body)
	}
}

// TestModels_ModelsHandler_ServeHTTP_Bad covers the method-rejection
// branch — only GET is served.
func TestModels_ModelsHandler_ServeHTTP_Bad(t *testing.T) {
	handler := NewModelsHandler(func() []string { return []string{"a"} })

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultModelsPath, nil))

	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", rec.Code)
	}
	if got := rec.Header().Get("Allow"); got != http.MethodGet {
		t.Fatalf("Allow = %q, want GET", got)
	}
}

// TestModels_ModelsHandler_ServeHTTP_Ugly covers two edge shapes: a nil
// *ModelsHandler receiver (must not panic — still 200s an empty list),
// and a models callback that returns blank entries, which must be
// filtered out of the response rather than surfacing an empty id.
func TestModels_ModelsHandler_ServeHTTP_Ugly(t *testing.T) {
	var nilHandler *ModelsHandler
	rec := httptest.NewRecorder()
	nilHandler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultModelsPath, nil))
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"data":[]`) {
		t.Fatalf("nil receiver: status = %d body=%s, want 200 empty list", rec.Code, rec.Body.String())
	}

	handler := NewModelsHandler(func() []string { return []string{"", "real", ""} })
	rec = httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultModelsPath, nil))
	if strings.Contains(rec.Body.String(), `"id":""`) {
		t.Fatalf("blank entries leaked into response: %s", rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"id":"real"`) {
		t.Fatalf("body=%s, want the one real model", rec.Body.String())
	}
}
