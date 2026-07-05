// SPDX-License-Identifier: EUPL-1.2

package ai

import (
	"net/http"
	"net/http/httptest"

	core "dappco.re/go"
)

func ExampleNewBookStateDemoHandler() {
	demo := core.MustCast[*BookStateDemo](NewBookStateDemo(BookStateDemoConfig{
		State:         BookState{Title: "Meditations"},
		TeacherRoutes: []ProviderRoute{{Name: "teacher", ModelID: "teacher", Model: &routerFakeModel{output: "answer"}}},
	}))
	handler := NewBookStateDemoHandler(demo)
	req := httptest.NewRequest(http.MethodGet, "/state", nil)
	rr := httptest.NewRecorder()

	handler.ServeHTTP(rr, req)

	core.Println(rr.Code)
	core.Println(core.Contains(rr.Body.String(), "Meditations"))
	// Output:
	// 200
	// true
}
