package serving

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/serving/provider/openai"
)

func TestHTTPBackend_Generate_Good(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}

		var req openai.ChatCompletionRequest
		mustReadJSONRequest(t, r, &req)

		if len(req.Messages) != 1 || req.Messages[0].Content != "hello" {
			t.Errorf("unexpected messages: %+v", req.Messages)
		}

		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatChoice{{Message: openai.ChatMessage{Role: "assistant", Content: "world"}}},
		}
		mustWriteJSONResponse(t, w, resp)
	}))
	defer srv.Close()

	b := NewHTTPBackend(srv.URL, "test-model")
	rGen := b.Generate(context.Background(), "hello", DefaultGenOpts())
	if !rGen.OK {
		t.Fatalf("Generate: %s", rGen.Error())
	}
	result := rGen.Value.(Result)
	if result.Text != "world" {
		t.Errorf("got %q, want %q", result.Text, "world")
	}
	if result.Metrics != nil {
		t.Error("HTTP backend should return nil Metrics")
	}
}

func TestHTTPBackend_Generate_Bad(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte("bad request"))
	}))
	defer srv.Close()

	b := NewHTTPBackend(srv.URL, "test-model")
	if rGen := b.Generate(context.Background(), "hello", DefaultGenOpts()); rGen.OK {
		t.Fatal("expected error for 400 response")
	}
}

func TestHTTPBackend_StopSequences_Good(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatChoice{{Message: openai.ChatMessage{Role: "assistant", Content: "hello STOP world"}}},
		}
		mustWriteJSONResponse(t, w, resp)
	}))
	defer srv.Close()

	b := NewHTTPBackend(srv.URL, "test-model")
	rGen := b.Generate(context.Background(), "hello", GenOpts{StopSequences: []string{"STOP"}})
	if !rGen.OK {
		t.Fatalf("Generate: %s", rGen.Error())
	}
	result := rGen.Value.(Result)
	if result.Text != "hello " {
		t.Errorf("got %q, want %q", result.Text, "hello ")
	}
}

func TestHTTPBackendRetryUglyScenario(t *testing.T) {
	attempts := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts < 3 {
			w.WriteHeader(http.StatusInternalServerError)
			w.Write([]byte("internal error"))
			return
		}
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatChoice{{Message: openai.ChatMessage{Role: "assistant", Content: "recovered"}}},
		}
		mustWriteJSONResponse(t, w, resp)
	}))
	defer srv.Close()

	b := NewHTTPBackend(srv.URL, "test-model")
	rGen := b.Generate(context.Background(), "test", DefaultGenOpts())
	if !rGen.OK {
		t.Fatalf("Generate after retry: %s", rGen.Error())
	}
	result := rGen.Value.(Result)
	if result.Text != "recovered" {
		t.Errorf("got %q, want %q", result.Text, "recovered")
	}
	if attempts != 3 {
		t.Errorf("expected 3 attempts, got %d", attempts)
	}
}

func TestHTTPBackend_Name_Good(t *testing.T) {
	b := NewHTTPBackend("http://localhost", "model")
	if b.Name() != "http" {
		t.Errorf("Name() = %q, want %q", b.Name(), "http")
	}
}

func TestHTTPBackend_Available_Good(t *testing.T) {
	b := NewHTTPBackend("http://localhost", "model")
	if !b.Available() {
		t.Error("Available() should be true when baseURL is set")
	}

	b2 := NewHTTPBackend("", "model")
	if b2.Available() {
		t.Error("Available() should be false when baseURL is empty")
	}
}

func TestHTTPBackend_WithMedium_Good(t *testing.T) {
	// Spec §10 — io.Medium supplied at construction is retained.
	// We pass nil to verify the option is accepted and the getter returns
	// the stored value (nil) rather than panicking.
	b := NewHTTPBackend("http://localhost", "model", WithMedium(nil))
	if b.Medium() != nil {
		t.Errorf("Medium() = %v, want nil", b.Medium())
	}
}

func TestHTTPBackend_WithHTTPMaxTokens_Good(t *testing.T) {
	b := NewHTTPBackend("http://localhost", "model", WithHTTPMaxTokens(512))
	if b.maxTokens != 512 {
		t.Errorf("maxTokens = %d, want 512", b.maxTokens)
	}
}

func TestHTTPBackend_WithHTTPClient_Ugly(t *testing.T) {
	// Nil HTTP client must be ignored (option is a no-op rather than breaking
	// the default 300s client).
	b := NewHTTPBackend("http://localhost", "model", WithHTTPClient(nil))
	if b.httpClient == nil {
		t.Error("nil HTTP client must not overwrite default")
	}
}

// --- v0.9.0 shape triplets ---

func TestBackendHttp_WithHTTPClient_Good(t *core.T) {
	client := &http.Client{}
	b := NewHTTPBackend("http://127.0.0.1", "model", WithHTTPClient(client))
	core.AssertEqual(t, client, b.httpClient)
}

func TestBackendHttp_WithHTTPClient_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	b := NewHTTPBackend("http://127.0.0.1", "model", WithHTTPClient(nil))
	core.AssertNotNil(t, b.httpClient)
}

func TestBackendHttp_WithHTTPClient_Ugly(t *core.T) {
	client := &http.Client{}
	opt := WithHTTPClient(client)
	b := NewHTTPBackend("", "", opt, WithHTTPClient(nil))
	core.AssertEqual(t, client, b.httpClient)
}

func TestBackendHttp_WithMedium_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	b := NewHTTPBackend("http://127.0.0.1", "model", WithMedium(nil))
	core.AssertNil(t, b.Medium())
}

func TestBackendHttp_WithMedium_Bad(t *core.T) {
	b := NewHTTPBackend("http://127.0.0.1", "model")
	WithMedium(nil)(b)
	core.AssertNil(t, b.Medium())
}

func TestBackendHttp_WithMedium_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	b := NewHTTPBackend("", "", WithMedium(nil))
	core.AssertNil(t, b.Medium())
}

func TestBackendHttp_WithHTTPMaxTokens_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	b := NewHTTPBackend("http://127.0.0.1", "model", WithHTTPMaxTokens(77))
	core.AssertEqual(t, 77, b.maxTokens)
}

func TestBackendHttp_WithHTTPMaxTokens_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	b := NewHTTPBackend("http://127.0.0.1", "model", WithHTTPMaxTokens(-1))
	core.AssertEqual(t, -1, b.maxTokens)
}

func TestBackendHttp_WithHTTPMaxTokens_Ugly(t *core.T) {
	b := NewHTTPBackend("http://127.0.0.1", "model")
	WithHTTPMaxTokens(0)(b)
	core.AssertEqual(t, 0, b.maxTokens)
}

func TestBackendHttp_Error_Error_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	err := &retryableError{core.NewError("temporary")}
	core.AssertEqual(t, "temporary", err.Error())
}

func TestBackendHttp_Error_Error_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	err := &retryableError{core.NewError("")}
	core.AssertEqual(t, "", err.Error())
}

func TestBackendHttp_Error_Error_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	err := &retryableError{core.NewError("edge")}
	core.AssertContains(t, err.Error(), "edge")
}

func TestBackendHttp_Error_Unwrap_Good(t *core.T) {
	cause := core.NewError("cause")
	err := &retryableError{cause}
	core.AssertErrorIs(t, err.Unwrap(), cause)
}

func TestBackendHttp_Error_Unwrap_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	err := &retryableError{core.NewError("bad")}
	core.AssertError(t, err.Unwrap())
}

func TestBackendHttp_Error_Unwrap_Ugly(t *core.T) {
	cause := core.NewError("edge")
	err := &retryableError{cause}
	core.AssertEqual(t, cause, err.Unwrap())
}

func TestBackendHttp_NewHTTPBackend_Good(t *core.T) {
	b := NewHTTPBackend("http://127.0.0.1", "model")
	core.AssertEqual(t, "http://127.0.0.1", b.BaseURL())
	core.AssertEqual(t, "model", b.Model())
}

func TestBackendHttp_NewHTTPBackend_Bad(t *core.T) {
	b := NewHTTPBackend("", "")
	core.AssertFalse(t, b.Available())
	core.AssertEqual(t, "", b.Model())
}

func TestBackendHttp_NewHTTPBackend_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	b := NewHTTPBackend("http://127.0.0.1", "model", WithHTTPMaxTokens(1))
	core.AssertEqual(t, 1, b.maxTokens)
}

func TestBackendHttp_HTTPBackend_Medium_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	b := NewHTTPBackend("http://127.0.0.1", "model", WithMedium(nil))
	core.AssertNil(t, b.Medium())
}

func TestBackendHttp_HTTPBackend_Medium_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	var b HTTPBackend
	core.AssertNil(t, b.Medium())
}

func TestBackendHttp_HTTPBackend_Medium_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	b := NewHTTPBackend("", "")
	core.AssertNil(t, b.Medium())
}

func TestBackendHttp_HTTPBackend_Name_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	b := NewHTTPBackend("http://127.0.0.1", "model")
	core.AssertEqual(t, "http", b.Name())
}

func TestBackendHttp_HTTPBackend_Name_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	var b HTTPBackend
	core.AssertEqual(t, "http", b.Name())
}

func TestBackendHttp_HTTPBackend_Name_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	b := NewHTTPBackend("", "")
	core.AssertEqual(t, "http", b.Name())
}

func TestBackendHttp_HTTPBackend_Available_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	b := NewHTTPBackend("http://127.0.0.1", "model")
	core.AssertTrue(t, b.Available())
}

func TestBackendHttp_HTTPBackend_Available_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	b := NewHTTPBackend("", "model")
	core.AssertFalse(t, b.Available())
}

func TestBackendHttp_HTTPBackend_Available_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	b := NewHTTPBackend(" ", "model")
	core.AssertTrue(t, b.Available())
}

func TestBackendHttp_HTTPBackend_Model_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	b := NewHTTPBackend("http://127.0.0.1", "model")
	core.AssertEqual(t, "model", b.Model())
}

func TestBackendHttp_HTTPBackend_Model_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	b := NewHTTPBackend("http://127.0.0.1", "")
	core.AssertEqual(t, "", b.Model())
}

func TestBackendHttp_HTTPBackend_Model_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	b := NewHTTPBackend("http://127.0.0.1", "model:tag")
	core.AssertEqual(t, "model:tag", b.Model())
}

func TestBackendHttp_HTTPBackend_BaseURL_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	b := NewHTTPBackend("http://127.0.0.1", "model")
	core.AssertEqual(t, "http://127.0.0.1", b.BaseURL())
}

func TestBackendHttp_HTTPBackend_BaseURL_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	b := NewHTTPBackend("", "model")
	core.AssertEqual(t, "", b.BaseURL())
}

func TestBackendHttp_HTTPBackend_BaseURL_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	b := NewHTTPBackend("http://127.0.0.1/", "model")
	core.AssertEqual(t, "http://127.0.0.1/", b.BaseURL())
}

func TestBackendHttp_HTTPBackend_SetMaxTokens_Good(t *core.T) {
	b := NewHTTPBackend("http://127.0.0.1", "model")
	b.SetMaxTokens(128)
	core.AssertEqual(t, 128, b.maxTokens)
}

func TestBackendHttp_HTTPBackend_SetMaxTokens_Bad(t *core.T) {
	b := NewHTTPBackend("http://127.0.0.1", "model")
	b.SetMaxTokens(-2)
	core.AssertEqual(t, -2, b.maxTokens)
}

func TestBackendHttp_HTTPBackend_SetMaxTokens_Ugly(t *core.T) {
	var b HTTPBackend
	b.SetMaxTokens(9)
	core.AssertEqual(t, 9, b.maxTokens)
}

func TestBackendHttp_HTTPBackend_LoadModel_Good(t *core.T) {
	b := NewHTTPBackend("http://127.0.0.1", "model")
	r := b.LoadModel("ignored")
	requireResultOK(t, r)
	model := r.Value.(*HTTPTextModel)
	core.AssertNotNil(t, model)
	core.AssertEqual(t, "model", model.ModelType())
}

func TestBackendHttp_HTTPBackend_LoadModel_Bad(t *core.T) {
	b := NewHTTPBackend("", "")
	r := b.LoadModel("")
	requireResultOK(t, r)
	model := r.Value.(*HTTPTextModel)
	core.AssertEqual(t, "http", model.ModelType())
}

func TestBackendHttp_HTTPBackend_LoadModel_Ugly(t *core.T) {
	b := NewHTTPBackend("http://127.0.0.1", "edge")
	r := b.LoadModel("unused")
	requireResultOK(t, r)
	model := r.Value.(*HTTPTextModel)
	assertResultOK(t, model.Close())
}

func TestBackendHttp_HTTPBackend_Generate_Good(t *core.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mustWriteJSONResponse(t, w, openai.ChatCompletionResponse{Choices: []openai.ChatChoice{{Message: openai.ChatMessage{Role: "assistant", Content: "ok"}}}})
	}))
	defer srv.Close()
	b := NewHTTPBackend(srv.URL, "model")
	rGen := b.Generate(context.Background(), "hello", GenOpts{})
	requireResultOK(t, rGen)
	result := rGen.Value.(Result)
	core.AssertEqual(t, "ok", result.Text)
}

func TestBackendHttp_HTTPBackend_Generate_Bad(t *core.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusBadRequest) }))
	defer srv.Close()
	b := NewHTTPBackend(srv.URL, "model")
	rGen := b.Generate(context.Background(), "hello", GenOpts{})
	assertResultError(t, rGen)
}

func TestBackendHttp_HTTPBackend_Generate_Ugly(t *core.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mustWriteJSONResponse(t, w, openai.ChatCompletionResponse{Choices: []openai.ChatChoice{{Message: openai.ChatMessage{Role: "assistant", Content: "cut STOP rest"}}}})
	}))
	defer srv.Close()
	b := NewHTTPBackend(srv.URL, "model")
	rGen := b.Generate(context.Background(), "hello", GenOpts{StopSequences: []string{"STOP"}})
	requireResultOK(t, rGen)
	result := rGen.Value.(Result)
	core.AssertEqual(t, "cut ", result.Text)
}

func TestBackendHttp_HTTPBackend_Chat_Good(t *core.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mustWriteJSONResponse(t, w, openai.ChatCompletionResponse{Choices: []openai.ChatChoice{{Message: openai.ChatMessage{Role: "assistant", Content: "chat"}}}})
	}))
	defer srv.Close()
	b := NewHTTPBackend(srv.URL, "model")
	rChat := b.Chat(context.Background(), []Message{{Role: "user", Content: "hello"}}, GenOpts{})
	requireResultOK(t, rChat)
	result := rChat.Value.(Result)
	core.AssertEqual(t, "chat", result.Text)
}

func TestBackendHttp_HTTPBackend_Chat_Bad(t *core.T) {
	b := NewHTTPBackend("http://127.0.0.1:1", "model")
	rChat := b.Chat(context.Background(), nil, GenOpts{})
	assertResultError(t, rChat)
}

func TestBackendHttp_HTTPBackend_Chat_Ugly(t *core.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mustWriteJSONResponse(t, w, openai.ChatCompletionResponse{})
	}))
	defer srv.Close()
	b := NewHTTPBackend(srv.URL, "model")
	rChat := b.Chat(context.Background(), []Message{{Role: "user", Content: "hello"}}, GenOpts{})
	assertResultError(t, rChat, "no choices")
}
