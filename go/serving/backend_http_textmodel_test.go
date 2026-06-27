// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"context"
	"net/http"
	"net/http/httptest"

	core "dappco.re/go"

	"dappco.re/go/inference"
	"dappco.re/go/inference/provider/openai"
)

// newTestServer creates an httptest.Server that responds with the given content.
func newTestServer(t *core.T, content string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatChoice{{Message: openai.ChatMessage{Role: "assistant", Content: content}}},
		}
		mustWriteJSONResponse(t, w, resp)
	}))
}

// newTestServerMulti creates an httptest.Server that responds based on the prompt.
func newTestServerMulti(t *core.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req openai.ChatCompletionRequest
		mustReadJSONRequest(t, r, &req)
		// Echo back the last message content with a prefix.
		lastContent := ""
		if len(req.Messages) > 0 {
			lastContent = req.Messages[len(req.Messages)-1].Content
		}
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatChoice{{Message: openai.ChatMessage{Role: "assistant", Content: "reply:" + lastContent}}},
		}
		mustWriteJSONResponse(t, w, resp)
	}))
}

func TestHTTPTextModel_Generate_Good(t *core.T) {
	srv := newTestServer(t, "Hello from HTTP")
	defer srv.Close()

	backend := NewHTTPBackend(srv.URL, "test-model")
	model := NewHTTPTextModel(backend)

	var collected []inference.Token
	for tok := range model.Generate(context.Background(), "test prompt") {
		collected = append(collected, tok)
	}

	core.AssertLen(t, collected, 1)
	core.AssertEqual(t, "Hello from HTTP", collected[0].Text)
	assertResultOK(t, model.Err())
}

func TestHTTPTextModel_Generate_WithOpts_Good(t *core.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req openai.ChatCompletionRequest
		mustReadJSONRequest(t, r, &req)

		// Verify that options are passed through.
		core.AssertInDelta(t, 0.8, float64(*req.Temperature), 0.01)
		core.AssertEqual(t, 100, *req.MaxTokens)

		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatChoice{{Message: openai.ChatMessage{Role: "assistant", Content: "configured"}}},
		}
		mustWriteJSONResponse(t, w, resp)
	}))
	defer srv.Close()

	backend := NewHTTPBackend(srv.URL, "test-model")
	model := NewHTTPTextModel(backend)

	var result string
	for tok := range model.Generate(context.Background(), "prompt",
		inference.WithTemperature(0.8),
		inference.WithMaxTokens(100),
	) {
		result = tok.Text
	}
	core.AssertEqual(t, "configured", result)
	assertResultOK(t, model.Err())
}

func TestHTTPTextModel_Chat_Good(t *core.T) {
	srv := newTestServer(t, "chat response")
	defer srv.Close()

	backend := NewHTTPBackend(srv.URL, "test-model")
	model := NewHTTPTextModel(backend)

	messages := []inference.Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "Hello"},
	}

	var collected []inference.Token
	for tok := range model.Chat(context.Background(), messages) {
		collected = append(collected, tok)
	}

	core.AssertLen(t, collected, 1)
	core.AssertEqual(t, "chat response", collected[0].Text)
	assertResultOK(t, model.Err())
}

func TestHTTPTextModel_Generate_Error_Bad(t *core.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte("invalid request"))
	}))
	defer srv.Close()

	backend := NewHTTPBackend(srv.URL, "test-model")
	model := NewHTTPTextModel(backend)

	var collected []inference.Token
	for tok := range model.Generate(context.Background(), "bad prompt") {
		collected = append(collected, tok)
	}

	core.AssertEmpty(t, collected, "no tokens should be yielded on error")
	core.AssertError(t, model.Err())
	core.AssertContains(t, model.Err().Error(), "400")
}

func TestHTTPTextModel_Chat_Error_Bad(t *core.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte("bad chat"))
	}))
	defer srv.Close()

	backend := NewHTTPBackend(srv.URL, "test-model")
	model := NewHTTPTextModel(backend)

	messages := []inference.Message{{Role: "user", Content: "test"}}
	var collected []inference.Token
	for tok := range model.Chat(context.Background(), messages) {
		collected = append(collected, tok)
	}

	core.AssertEmpty(t, collected)
	core.AssertError(t, model.Err())
}

func TestHTTPTextModel_Classify_Bad(t *core.T) {
	backend := NewHTTPBackend("http://localhost", "test-model")
	model := NewHTTPTextModel(backend)

	r := model.Classify(context.Background(), []string{"test"})
	core.AssertFalse(t, r.OK)
	err := r.Value.(error)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "classify not supported")
}

func TestHTTPTextModel_BatchGenerate_Good(t *core.T) {
	srv := newTestServerMulti(t)
	defer srv.Close()

	backend := NewHTTPBackend(srv.URL, "test-model")
	model := NewHTTPTextModel(backend)

	prompts := []string{"alpha", "beta", "gamma"}
	r := model.BatchGenerate(context.Background(), prompts)
	requireResultOK(t, r)
	results := r.Value.([]inference.BatchResult)
	core.AssertLen(t, results, 3)

	core.AssertEqual(t, "reply:alpha", results[0].Tokens[0].Text)
	core.AssertNoError(t, results[0].Err)

	core.AssertEqual(t, "reply:beta", results[1].Tokens[0].Text)
	core.AssertNoError(t, results[1].Err)

	core.AssertEqual(t, "reply:gamma", results[2].Tokens[0].Text)
	core.AssertNoError(t, results[2].Err)
}

func TestHTTPTextModel_BatchGenerate_PartialError_Bad(t *core.T) {
	callCount := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount == 2 {
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte("error on second"))
			return
		}
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatChoice{{Message: openai.ChatMessage{Role: "assistant", Content: "ok"}}},
		}
		mustWriteJSONResponse(t, w, resp)
	}))
	defer srv.Close()

	backend := NewHTTPBackend(srv.URL, "test-model")
	model := NewHTTPTextModel(backend)

	r := model.BatchGenerate(context.Background(), []string{"a", "b", "c"})
	requireResultOK(t, r) // BatchGenerate itself doesn't fail.
	results := r.Value.([]inference.BatchResult)
	core.AssertLen(t, results, 3)

	core.AssertLen(t, results[0].Tokens, 1)
	core.AssertNoError(t, results[0].Err)

	core.AssertEmpty(t, results[1].Tokens)
	core.AssertError(t, results[1].Err)

	core.AssertLen(t, results[2].Tokens, 1)
	core.AssertNoError(t, results[2].Err)
}

func TestHTTPTextModel_ModelType_Good(t *core.T) {
	backend := NewHTTPBackend("http://localhost", "gpt-4o")
	model := NewHTTPTextModel(backend)
	core.AssertEqual(t, "gpt-4o", model.ModelType())
}

func TestHTTPTextModel_ModelType_Empty_Good(t *core.T) {
	backend := NewHTTPBackend("http://localhost", "")
	model := NewHTTPTextModel(backend)
	core.AssertEqual(t, "http", model.ModelType())
}

func TestHTTPTextModel_Info_Good(t *core.T) {
	backend := NewHTTPBackend("http://localhost", "test")
	model := NewHTTPTextModel(backend)
	info := model.Info()
	core.AssertEqual(t, "http", info.Architecture)
}

func TestHTTPTextModel_Metrics_Good(t *core.T) {
	backend := NewHTTPBackend("http://localhost", "test")
	model := NewHTTPTextModel(backend)
	metrics := model.Metrics()
	core.AssertEqual(t, inference.GenerateMetrics{}, metrics)
}

func TestHTTPTextModel_Err_ClearedOnSuccess_Good(t *core.T) {
	// First request fails.
	callCount := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount == 1 {
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte("fail"))
			return
		}
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatChoice{{Message: openai.ChatMessage{Role: "assistant", Content: "ok"}}},
		}
		mustWriteJSONResponse(t, w, resp)
	}))
	defer srv.Close()

	backend := NewHTTPBackend(srv.URL, "test-model")
	model := NewHTTPTextModel(backend)

	// First call: error.
	for range model.Generate(context.Background(), "fail") {
	}
	core.AssertError(t, model.Err())

	// Second call: success — error should be cleared.
	for range model.Generate(context.Background(), "ok") {
	}
	assertResultOK(t, model.Err())
}

func TestHTTPTextModel_Close_Good(t *core.T) {
	backend := NewHTTPBackend("http://localhost", "test")
	model := NewHTTPTextModel(backend)
	assertResultOK(t, model.Close())
}

func TestLlamaTextModel_ModelType_Good(t *core.T) {
	// LlamaBackend requires a process.Service but we only test ModelType here.
	llama := &LlamaBackend{
		http: NewHTTPBackend("http://127.0.0.1:18090", ""),
	}
	model := NewLlamaTextModel(llama)
	core.AssertEqual(t, "llama", model.ModelType())
}

func TestLlamaTextModel_Close_Good(t *core.T) {
	// LlamaBackend with no procID — Stop() is a no-op.
	llama := &LlamaBackend{
		http: NewHTTPBackend("http://127.0.0.1:18090", ""),
	}
	model := NewLlamaTextModel(llama)
	assertResultOK(t, model.Close())
}

// --- v0.9.0 shape triplets ---

func TestBackendHttpTextmodel_NewHTTPTextModel_Good(t *core.T) {
	backend := NewHTTPBackend("http://127.0.0.1", "model")
	model := NewHTTPTextModel(backend)
	core.AssertNotNil(t, model)
	core.AssertEqual(t, "model", model.ModelType())
}

func TestBackendHttpTextmodel_NewHTTPTextModel_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	model := NewHTTPTextModel(NewHTTPBackend("", ""))
	core.AssertEqual(t, "http", model.ModelType())
}

func TestBackendHttpTextmodel_NewHTTPTextModel_Ugly(t *core.T) {
	backend := NewHTTPBackend("http://127.0.0.1", "edge")
	model := NewHTTPTextModel(backend)
	assertResultOK(t, model.Close())
}

func TestBackendHttpTextmodel_HTTPTextModel_Generate_Good(t *core.T) {
	srv := newTestServer(t, "generated")
	defer srv.Close()
	model := NewHTTPTextModel(NewHTTPBackend(srv.URL, "model"))
	var got []inference.Token
	for tok := range model.Generate(context.Background(), "prompt") {
		got = append(got, tok)
	}
	core.AssertLen(t, got, 1)
	core.AssertEqual(t, "generated", got[0].Text)
}

func TestBackendHttpTextmodel_HTTPTextModel_Generate_Bad(t *core.T) {
	model := NewHTTPTextModel(NewHTTPBackend("http://127.0.0.1:1", "model"))
	var got []inference.Token
	for tok := range model.Generate(context.Background(), "prompt") {
		got = append(got, tok)
	}
	core.AssertEmpty(t, got)
	core.AssertError(t, model.Err())
}

func TestBackendHttpTextmodel_HTTPTextModel_Generate_Ugly(t *core.T) {
	srv := newTestServer(t, "configured")
	defer srv.Close()
	model := NewHTTPTextModel(NewHTTPBackend(srv.URL, "model"))
	var got string
	for tok := range model.Generate(context.Background(), "prompt", inference.WithMaxTokens(2)) {
		got = tok.Text
	}
	core.AssertEqual(t, "configured", got)
}

func TestBackendHttpTextmodel_HTTPTextModel_Chat_Good(t *core.T) {
	srv := newTestServer(t, "chat")
	defer srv.Close()
	model := NewHTTPTextModel(NewHTTPBackend(srv.URL, "model"))
	var got []inference.Token
	for tok := range model.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}) {
		got = append(got, tok)
	}
	core.AssertLen(t, got, 1)
	core.AssertEqual(t, "chat", got[0].Text)
}

func TestBackendHttpTextmodel_HTTPTextModel_Chat_Bad(t *core.T) {
	model := NewHTTPTextModel(NewHTTPBackend("http://127.0.0.1:1", "model"))
	for range model.Chat(context.Background(), nil) {
	}
	core.AssertError(t, model.Err())
}

func TestBackendHttpTextmodel_HTTPTextModel_Chat_Ugly(t *core.T) {
	srv := newTestServer(t, "empty")
	defer srv.Close()
	model := NewHTTPTextModel(NewHTTPBackend(srv.URL, "model"))
	var got string
	for tok := range model.Chat(context.Background(), nil) {
		got = tok.Text
	}
	core.AssertEqual(t, "empty", got)
}

func TestBackendHttpTextmodel_HTTPTextModel_Classify_Good(t *core.T) {
	model := NewHTTPTextModel(NewHTTPBackend("http://127.0.0.1", "model"))
	r := model.Classify(context.Background(), []string{"sample"})
	assertResultError(t, r, "not supported")
}

func TestBackendHttpTextmodel_HTTPTextModel_Classify_Bad(t *core.T) {
	model := NewHTTPTextModel(NewHTTPBackend("", ""))
	assertResultError(t, model.Classify(context.Background(), nil))
}

func TestBackendHttpTextmodel_HTTPTextModel_Classify_Ugly(t *core.T) {
	model := NewHTTPTextModel(NewHTTPBackend("http://127.0.0.1", "model"))
	assertResultError(t, model.Classify(context.Background(), []string{}), "classify")
}

func TestBackendHttpTextmodel_HTTPTextModel_BatchGenerate_Good(t *core.T) {
	srv := newTestServerMulti(t)
	defer srv.Close()
	model := NewHTTPTextModel(NewHTTPBackend(srv.URL, "model"))
	r := model.BatchGenerate(context.Background(), []string{"a", "b"})
	requireResultOK(t, r)
	results := r.Value.([]inference.BatchResult)
	core.AssertLen(t, results, 2)
	core.AssertEqual(t, "reply:a", results[0].Tokens[0].Text)
}

func TestBackendHttpTextmodel_HTTPTextModel_BatchGenerate_Bad(t *core.T) {
	model := NewHTTPTextModel(NewHTTPBackend("http://127.0.0.1:1", "model"))
	r := model.BatchGenerate(context.Background(), []string{"a"})
	requireResultOK(t, r)
	results := r.Value.([]inference.BatchResult)
	core.AssertLen(t, results, 1)
	core.AssertError(t, results[0].Err)
}

func TestBackendHttpTextmodel_HTTPTextModel_BatchGenerate_Ugly(t *core.T) {
	model := NewHTTPTextModel(NewHTTPBackend("http://127.0.0.1", "model"))
	r := model.BatchGenerate(context.Background(), nil)
	requireResultOK(t, r)
	results := r.Value.([]inference.BatchResult)
	core.AssertEmpty(t, results)
}

func TestBackendHttpTextmodel_HTTPTextModel_ModelType_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	model := NewHTTPTextModel(NewHTTPBackend("http://127.0.0.1", "named"))
	core.AssertEqual(t, "named", model.ModelType())
}

func TestBackendHttpTextmodel_HTTPTextModel_ModelType_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	model := NewHTTPTextModel(NewHTTPBackend("http://127.0.0.1", ""))
	core.AssertEqual(t, "http", model.ModelType())
}

func TestBackendHttpTextmodel_HTTPTextModel_ModelType_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	model := NewHTTPTextModel(NewHTTPBackend("", "edge:model"))
	core.AssertEqual(t, "edge:model", model.ModelType())
}

func TestBackendHttpTextmodel_HTTPTextModel_Info_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	model := NewHTTPTextModel(NewHTTPBackend("http://127.0.0.1", "model"))
	core.AssertEqual(t, "http", model.Info().Architecture)
}

func TestBackendHttpTextmodel_HTTPTextModel_Info_Bad(t *core.T) {
	model := NewHTTPTextModel(NewHTTPBackend("", ""))
	info := model.Info()
	core.AssertEqual(t, 0, info.NumLayers)
}

func TestBackendHttpTextmodel_HTTPTextModel_Info_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	model := NewHTTPTextModel(NewHTTPBackend("http://127.0.0.1", "edge"))
	core.AssertEqual(t, "http", model.Info().Architecture)
}

func TestBackendHttpTextmodel_HTTPTextModel_Metrics_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	model := NewHTTPTextModel(NewHTTPBackend("http://127.0.0.1", "model"))
	core.AssertEqual(t, inference.GenerateMetrics{}, model.Metrics())
}

func TestBackendHttpTextmodel_HTTPTextModel_Metrics_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	model := NewHTTPTextModel(NewHTTPBackend("", ""))
	core.AssertEqual(t, uint64(0), model.Metrics().PeakMemoryBytes)
}

func TestBackendHttpTextmodel_HTTPTextModel_Metrics_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	model := NewHTTPTextModel(NewHTTPBackend("http://127.0.0.1", "edge"))
	core.AssertEqual(t, 0.0, model.Metrics().DecodeTokensPerSec)
}

func TestBackendHttpTextmodel_HTTPTextModel_Err_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	model := NewHTTPTextModel(NewHTTPBackend("http://127.0.0.1", "model"))
	assertResultOK(t, model.Err())
}

func TestBackendHttpTextmodel_HTTPTextModel_Err_Bad(t *core.T) {
	model := NewHTTPTextModel(NewHTTPBackend("http://127.0.0.1:1", "model"))
	for range model.Generate(context.Background(), "prompt") {
	}
	core.AssertError(t, model.Err())
}

func TestBackendHttpTextmodel_HTTPTextModel_Err_Ugly(t *core.T) {
	srv := newTestServer(t, "ok")
	defer srv.Close()
	model := NewHTTPTextModel(NewHTTPBackend("http://127.0.0.1:1", "model"))
	for range model.Generate(context.Background(), "prompt") {
	}
	model.http = NewHTTPBackend(srv.URL, "model")
	for range model.Generate(context.Background(), "prompt") {
	}
	assertResultOK(t, model.Err())
}

func TestBackendHttpTextmodel_HTTPTextModel_Close_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	model := NewHTTPTextModel(NewHTTPBackend("http://127.0.0.1", "model"))
	assertResultOK(t, model.Close())
}

func TestBackendHttpTextmodel_HTTPTextModel_Close_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	model := NewHTTPTextModel(NewHTTPBackend("", ""))
	assertResultOK(t, model.Close())
}

func TestBackendHttpTextmodel_HTTPTextModel_Close_Ugly(t *core.T) {
	model := NewHTTPTextModel(NewHTTPBackend("http://127.0.0.1", "model"))
	assertResultOK(t, model.Close())
	assertResultOK(t, model.Close())
}

func TestBackendHttpTextmodel_NewLlamaTextModel_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	model := NewLlamaTextModel(NewLlamaBackend())
	core.AssertEqual(t, "llama", model.ModelType())
}

func TestBackendHttpTextmodel_NewLlamaTextModel_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	model := NewLlamaTextModel(NewLlamaBackend(LlamaOpts{Port: 19090}))
	core.AssertNotNil(t, model.HTTPTextModel)
}

func TestBackendHttpTextmodel_NewLlamaTextModel_Ugly(t *core.T) {
	backend := NewLlamaBackend("model.gguf")
	model := NewLlamaTextModel(backend)
	core.AssertEqual(t, backend, model.llama)
}

func TestBackendHttpTextmodel_LlamaTextModel_ModelType_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	model := NewLlamaTextModel(NewLlamaBackend())
	core.AssertEqual(t, "llama", model.ModelType())
}

func TestBackendHttpTextmodel_LlamaTextModel_ModelType_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	model := NewLlamaTextModel(NewLlamaBackend(nil))
	core.AssertEqual(t, "llama", model.ModelType())
}

func TestBackendHttpTextmodel_LlamaTextModel_ModelType_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	model := NewLlamaTextModel(NewLlamaBackend(LlamaOpts{LoraPath: "adapter.gguf"}))
	core.AssertEqual(t, "llama", model.ModelType())
}

func TestBackendHttpTextmodel_LlamaTextModel_Close_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	model := NewLlamaTextModel(NewLlamaBackend())
	assertResultOK(t, model.Close())
}

func TestBackendHttpTextmodel_LlamaTextModel_Close_Bad(t *core.T) {
	backend := NewLlamaBackend()
	backend.procID = "proc"
	model := NewLlamaTextModel(backend)
	core.AssertError(t, model.Close(), "process service")
}

func TestBackendHttpTextmodel_LlamaTextModel_Close_Ugly(t *core.T) {
	model := NewLlamaTextModel(NewLlamaBackend())
	assertResultOK(t, model.Close())
	assertResultOK(t, model.Close())
}
