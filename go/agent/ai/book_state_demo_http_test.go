// SPDX-License-Identifier: EUPL-1.2

package ai

import (
	"net/http"
	"net/http/httptest"
	"testing"

	core "dappco.re/go"
)

func TestBookStateDemoHttp_NewBookStateDemoHandler_Good(t *testing.T) {
	demo := mustBookStateDemo(t, BookStateDemoConfig{
		State:         BookState{Title: "Meditations", Excerpt: "gentleness"},
		TeacherRoutes: []ProviderRoute{{Name: "teacher", ModelID: "teacher", Model: &routerFakeModel{modelType: "teacher", output: "gentleness"}}},
	})
	handler := NewBookStateDemoHandler(demo)
	body := core.JSONMarshalString(BookStateAskRequest{Question: "What lesson?", MaxTokens: 8})
	req := httptest.NewRequest(http.MethodPost, "/ask", core.NewReader(body))
	rr := httptest.NewRecorder()

	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s, want 200", rr.Code, rr.Body.String())
	}
	var response BookStateAskResponse
	if result := core.JSONUnmarshalString(rr.Body.String(), &response); !result.OK {
		t.Fatalf("decode response = %s", result.Error())
	}
	if response.TeacherAnswer != "gentleness" || response.State.Title != "Meditations" {
		t.Fatalf("response = %+v, want teacher answer and state", response)
	}
}

func TestBookStateDemoHTTP_NewBookStateDemoHandler_Good_ReturnsState(t *testing.T) {
	demo := mustBookStateDemo(t, BookStateDemoConfig{
		State:         BookState{Title: "Meditations", EntryURI: "memvid://book"},
		TeacherRoutes: []ProviderRoute{{Name: "teacher", ModelID: "teacher", Model: &routerFakeModel{modelType: "teacher", output: "ok"}}},
	})
	handler := NewBookStateDemoHandler(demo)
	req := httptest.NewRequest(http.MethodGet, "/state", nil)
	rr := httptest.NewRecorder()

	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s, want 200", rr.Code, rr.Body.String())
	}
	var state BookState
	if result := core.JSONUnmarshalString(rr.Body.String(), &state); !result.OK {
		t.Fatalf("decode state = %s", result.Error())
	}
	if state.EntryURI != "memvid://book" {
		t.Fatalf("state = %+v, want configured state", state)
	}
}

func TestBookStateDemoHttp_NewBookStateDemoHandler_Bad(t *testing.T) {
	demo := mustBookStateDemo(t, BookStateDemoConfig{
		State:         BookState{Title: "Meditations"},
		TeacherRoutes: []ProviderRoute{{Name: "teacher", ModelID: "teacher", Model: &routerFakeModel{modelType: "teacher", output: "ok"}}},
	})
	handler := NewBookStateDemoHandler(demo)
	req := httptest.NewRequest(http.MethodPost, "/ask", core.NewReader("{"))
	rr := httptest.NewRecorder()

	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", rr.Code)
	}
	if !core.Contains(rr.Body.String(), "invalid JSON") {
		t.Fatalf("body = %s, want invalid JSON error", rr.Body.String())
	}
}

func TestBookStateDemoHttp_NewBookStateDemoHandler_Ugly(t *testing.T) {
	demo := mustBookStateDemo(t, BookStateDemoConfig{
		State:         BookState{Title: "Meditations"},
		TeacherRoutes: []ProviderRoute{{Name: "teacher", ModelID: "teacher", Model: &routerFakeModel{modelType: "teacher", output: "ok"}}},
	})
	handler := NewBookStateDemoHandler(demo)
	req := httptest.NewRequest(http.MethodGet, "/ask", nil)
	rr := httptest.NewRecorder()

	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", rr.Code)
	}
}
