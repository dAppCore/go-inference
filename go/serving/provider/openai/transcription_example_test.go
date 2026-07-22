// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"net/http"
	"net/http/httptest"
	"strings"

	core "dappco.re/go"
)

func ExampleNewTranscriptionHandler() {
	handler := NewTranscriptionHandler(&stubTranscriber{text: "hello world", language: "en"})

	rec := httptest.NewRecorder()
	body := `{"audio":"` + audioDataURL("RIFF...") + `"}`
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultTranscriptionsPath, strings.NewReader(body)))

	core.Println(rec.Code)
	// Output:
	// 200
}

func ExampleTranscriptionHandler_ServeHTTP() {
	handler := NewTranscriptionHandler(&stubTranscriber{text: "the quick brown fox", language: "en"})

	rec := httptest.NewRecorder()
	body := `{"audio":"` + audioDataURL("RIFF...") + `","language":"en"}`
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultTranscriptionsPath, strings.NewReader(body)))

	core.Println(rec.Code)
	core.Println(strings.Contains(rec.Body.String(), `"text":"the quick brown fox"`))
	// Output:
	// 200
	// true
}
