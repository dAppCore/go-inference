// SPDX-Licence-Identifier: EUPL-1.2

package admin

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestStatusHandler_MethodRejection_Bad proves a non-GET
// /v1/admin/serve/status is rejected before any snapshot work.
func TestStatusHandler_MethodRejection_StatusMethodNotAllowed_Bad(t *testing.T) {
	mux := NewMux(Config{ServeStatus: ServeStatus{ModelPath: "/models/boot"}})
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, PathServeStatus, nil))
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("POST /v1/admin/serve/status = %d, want 405", rec.Code)
	}
}

// TestStatusHandler_NoReloader_Good proves that with no Reloader wired
// (currentPath nil), GET reports the boot-snapshot ModelPath unchanged.
func TestStatusHandler_NoReloader_ServeStatus_Good(t *testing.T) {
	mux := NewMux(Config{ServeStatus: ServeStatus{ModelPath: "/models/boot", Runtime: "go-inference"}})
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, PathServeStatus, nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("GET status = %d, want 200", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), `"model_path":"/models/boot"`) {
		t.Fatalf("body = %s, want the boot-snapshot model_path", rec.Body.String())
	}
}

// TestStatusHandler_CurrentPathOverride_Good proves a live Reloader's
// CurrentPath refreshes ModelPath, reflecting a completed hot-swap reload
// while the rest of the boot Config snapshot stays put.
func TestStatusHandler_CurrentPathOverride_Good(t *testing.T) {
	rl := &fakeReloader{current: "/models/swapped"}
	mux := NewMux(Config{
		Reloader:    rl,
		ServeStatus: ServeStatus{ModelPath: "/models/boot", Config: ServeStatusConfig{ContextLength: 8192}},
	})
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, PathServeStatus, nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("GET status = %d, want 200", rec.Code)
	}
	body := rec.Body.String()
	if !strings.Contains(body, `"model_path":"/models/swapped"`) {
		t.Fatalf("body = %s, want the live CurrentPath, not the boot snapshot", body)
	}
	if !strings.Contains(body, `"context_length":8192`) {
		t.Fatalf("body = %s, want the boot Config block preserved", body)
	}
}

// TestStatusHandler_CurrentPathEmpty_Good proves a Reloader that reports an
// empty CurrentPath (pre-first-load) leaves the boot-snapshot ModelPath
// alone rather than blanking it.
func TestStatusHandler_CurrentPathEmpty_FakeReloader_Good(t *testing.T) {
	rl := &fakeReloader{current: ""}
	mux := NewMux(Config{Reloader: rl, ServeStatus: ServeStatus{ModelPath: "/models/boot"}})
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, PathServeStatus, nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("GET status = %d, want 200", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), `"model_path":"/models/boot"`) {
		t.Fatalf("body = %s, want the boot-snapshot model_path preserved", rec.Body.String())
	}
}
