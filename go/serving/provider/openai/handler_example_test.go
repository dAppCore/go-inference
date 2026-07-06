// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"net/http"
	"net/http/httptest"
	"strings"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func ExampleNewHandler() {
	resolver := NewStaticResolver(map[string]inference.TextModel{
		"gemma": &stubModel{tokens: []inference.Token{{Text: "hi there"}}},
	})
	handler := NewHandler(resolver)

	rec := httptest.NewRecorder()
	body := `{"model":"gemma","messages":[{"role":"user","content":"hello"}]}`
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultChatCompletionsPath, strings.NewReader(body)))

	core.Println(rec.Code)
	core.Println(strings.Contains(rec.Body.String(), `"content":"hi there"`))
	// Output:
	// 200
	// true
}

func ExampleHandler_ServeHTTP() {
	model := &stubModel{tokens: []inference.Token{{Text: "hello"}}}
	handler := NewHandler(NewStaticResolver(map[string]inference.TextModel{"gemma": model}))

	rec := httptest.NewRecorder()
	body := `{"model":"gemma","messages":[{"role":"user","content":"hi"}]}`
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultChatCompletionsPath, strings.NewReader(body)))

	core.Println(rec.Code)
	core.Println(strings.Contains(rec.Body.String(), `"finish_reason":"stop"`))
	// Output:
	// 200
	// true
}
