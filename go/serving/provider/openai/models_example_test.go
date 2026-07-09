// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"net/http"
	"net/http/httptest"
	"strings"

	core "dappco.re/go"
)

func ExampleNewModelsHandler() {
	handler := NewModelsHandler(func() []string {
		return []string{"gemma-4-e2b-it-4bit"}
	})

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultModelsPath, nil))

	core.Println(rec.Code)
	core.Println(strings.Contains(rec.Body.String(), `"id":"gemma-4-e2b-it-4bit"`))
	// Output:
	// 200
	// true
}

func ExampleModelsHandler_ServeHTTP() {
	handler := NewModelsHandler(func() []string { return []string{"gpt-test"} })

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultModelsPath, nil))

	core.Println(rec.Code)
	core.Println(strings.Contains(rec.Body.String(), `"object":"list"`))
	// Output:
	// 200
	// true
}
