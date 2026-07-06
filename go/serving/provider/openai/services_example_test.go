// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"net/http"
	"net/http/httptest"
	"strings"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func ExampleEmbeddingInput_UnmarshalJSON() {
	var input EmbeddingInput
	if err := input.UnmarshalJSON([]byte(`["one","two"]`)); err != nil {
		core.Println(err)
		return
	}

	core.Println(len(input))
	core.Println(input[0], input[1])
	// Output:
	// 2
	// one two
}

func exampleServiceResolver() Resolver {
	return NewStaticResolver(map[string]inference.TextModel{
		"qwen": &serviceModel{stubModel: &stubModel{}},
	})
}

func ExampleNewEmbeddingsHandler() {
	handler := NewEmbeddingsHandler(exampleServiceResolver())

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultEmbeddingsPath, strings.NewReader(`{"model":"qwen","input":"hi"}`)))

	core.Println(rec.Code)
	// Output:
	// 200
}

func ExampleEmbeddingsHandler_ServeHTTP() {
	handler := NewEmbeddingsHandler(exampleServiceResolver())

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultEmbeddingsPath, strings.NewReader(`{"model":"qwen","input":["one","two"]}`)))

	core.Println(rec.Code)
	core.Println(strings.Contains(rec.Body.String(), `"object":"list"`))
	// Output:
	// 200
	// true
}

func ExampleNewRerankHandler() {
	handler := NewRerankHandler(exampleServiceResolver())

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultRerankPath, strings.NewReader(`{"model":"qwen","query":"core","documents":["a","b"]}`)))

	core.Println(rec.Code)
	// Output:
	// 200
}

func ExampleRerankHandler_ServeHTTP() {
	handler := NewRerankHandler(exampleServiceResolver())

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultRerankPath, strings.NewReader(`{"model":"qwen","query":"core","documents":["a","b"]}`)))

	core.Println(rec.Code)
	core.Println(strings.Contains(rec.Body.String(), `"index":1`))
	// Output:
	// 200
	// true
}

func ExampleNewCapabilityHandler() {
	handler := NewCapabilityHandler(exampleServiceResolver())

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultCapabilitiesPath+"?model=qwen", nil))

	core.Println(rec.Code)
	// Output:
	// 200
}

func ExampleCapabilityHandler_ServeHTTP() {
	handler := NewCapabilityHandler(exampleServiceResolver())

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultCapabilitiesPath+"?model=qwen", nil))

	core.Println(rec.Code)
	core.Println(strings.Contains(rec.Body.String(), `"embeddings"`))
	// Output:
	// 200
	// true
}

func ExampleNewCacheStatsHandler() {
	handler := NewCacheStatsHandler(exampleServiceResolver())

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultCacheStatsPath+"?model=qwen", nil))

	core.Println(rec.Code)
	// Output:
	// 200
}

func ExampleCacheStatsHandler_ServeHTTP() {
	handler := NewCacheStatsHandler(exampleServiceResolver())

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultCacheStatsPath+"?model=qwen", nil))

	core.Println(rec.Code)
	core.Println(strings.Contains(rec.Body.String(), `"hit_rate"`))
	// Output:
	// 200
	// true
}

func ExampleNewCacheWarmHandler() {
	handler := NewCacheWarmHandler(exampleServiceResolver())

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultCacheWarmPath, strings.NewReader(`{"model":"qwen","tokens":[1,2,3]}`)))

	core.Println(rec.Code)
	// Output:
	// 200
}

func ExampleCacheWarmHandler_ServeHTTP() {
	handler := NewCacheWarmHandler(exampleServiceResolver())

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultCacheWarmPath, strings.NewReader(`{"model":"qwen","tokens":[1,2,3]}`)))

	core.Println(rec.Code)
	// Output:
	// 200
}

func ExampleNewCacheClearHandler() {
	handler := NewCacheClearHandler(exampleServiceResolver())

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultCacheClearPath, strings.NewReader(`{"model":"qwen"}`)))

	core.Println(rec.Code)
	// Output:
	// 200
}

func ExampleCacheClearHandler_ServeHTTP() {
	handler := NewCacheClearHandler(exampleServiceResolver())

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultCacheClearPath, strings.NewReader(`{"model":"qwen"}`)))

	core.Println(rec.Code)
	// Output:
	// 200
}

func ExampleNewCancelHandler() {
	handler := NewCancelHandler(exampleServiceResolver())

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultCancelPath, strings.NewReader(`{"model":"qwen","id":"req_1"}`)))

	core.Println(rec.Code)
	// Output:
	// 200
}

func ExampleCancelHandler_ServeHTTP() {
	handler := NewCancelHandler(exampleServiceResolver())

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultCancelPath, strings.NewReader(`{"model":"qwen","id":"req_1"}`)))

	core.Println(rec.Code)
	core.Println(strings.Contains(rec.Body.String(), `"cancelled":true`))
	// Output:
	// 200
	// true
}
