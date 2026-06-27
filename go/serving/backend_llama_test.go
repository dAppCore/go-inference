// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"context"
	"dappco.re/go"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
)

// ---------------------------------------------------------------------------
// LlamaBackend unit tests — no subprocess, HTTP mocked via httptest
// ---------------------------------------------------------------------------

// newMockLlamaServer creates an httptest.Server that responds to both
// /health and /v1/chat/completions.  Returns a fixed content string for chat
// and 200 OK for health.
func newMockLlamaServer(t *core.T, chatContent string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			w.WriteHeader(http.StatusOK)
		case "/v1/chat/completions":
			resp := chatResponse{
				Choices: []chatChoice{
					{Message: Message{Role: "assistant", Content: chatContent}},
				},
			}
			w.Header().Set("Content-Type", "application/json")
			mustWriteJSONResponse(t, w, resp)
		default:
			http.NotFound(w, r)
		}
	}))
}

// newLlamaBackendWithServer wires up a LlamaBackend pointing at the given
// test server.  The procID is set so Available() attempts the health check.
func newLlamaBackendWithServer(srv *httptest.Server) *LlamaBackend {
	return &LlamaBackend{
		procID: "test-proc",
		port:   serverPort(srv),
		http:   NewHTTPBackend(srv.URL, ""),
	}
}

// serverPort extracts the port number from an httptest.Server.
func serverPort(srv *httptest.Server) int {
	u, _ := url.Parse(srv.URL)
	p, _ := strconv.Atoi(u.Port())
	return p
}

// --- Name ---

func TestLlamaBackend_Name_Good(t *core.T) {
	lb := &LlamaBackend{}
	name := lb.Name()
	core.AssertEqual(t, "llama", name)
	core.AssertFalse(t, lb.Available())
}

// --- Available ---

func TestLlamaBackend_Available_NoProcID_Bad(t *core.T) {
	lb := &LlamaBackend{} // procID is ""
	available := lb.Available()
	core.AssertFalse(t, available, "Available should return false when procID is empty")
	core.AssertEqual(t, "", lb.procID)
}

func TestLlamaBackend_Available_HealthyServer_Good(t *core.T) {
	srv := newMockLlamaServer(t, "unused")
	defer srv.Close()

	lb := &LlamaBackend{
		procID: "test-proc",
		port:   serverPort(srv),
	}

	core.AssertTrue(t, lb.Available())
}

func TestLlamaBackend_Available_UnreachableServer_Bad(t *core.T) {
	lb := &LlamaBackend{
		procID: "test-proc",
		port:   19999, // nothing listening here
	}
	core.AssertFalse(t, lb.Available())
}

func TestLlamaBackend_Available_UnhealthyServer_Bad(t *core.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	lb := &LlamaBackend{
		procID: "test-proc",
		port:   serverPort(srv),
	}
	core.AssertFalse(t, lb.Available())
}

// --- Generate ---

func TestLlamaBackend_Generate_Good(t *core.T) {
	srv := newMockLlamaServer(t, "generated response")
	defer srv.Close()

	lb := newLlamaBackendWithServer(srv)

	r := lb.Generate(context.Background(), "test prompt", DefaultGenOpts())
	requireResultOK(t, r)
	result := r.Value.(Result)
	core.AssertEqual(t, "generated response", result.Text)
	core.AssertNil(t, result.Metrics)
}

func TestLlamaBackend_Generate_NotAvailable_Bad(t *core.T) {
	lb := &LlamaBackend{
		procID: "",
		http:   NewHTTPBackend("http://127.0.0.1:19999", ""),
	}

	r := lb.Generate(context.Background(), "test", DefaultGenOpts())
	assertResultError(t, r)
	core.AssertContains(t, r.Error(), "not available")
}

func TestLlamaBackend_Generate_ServerError_Bad(t *core.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			w.WriteHeader(http.StatusOK)
		case "/v1/chat/completions":
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte("bad request"))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	lb := newLlamaBackendWithServer(srv)

	r := lb.Generate(context.Background(), "test", DefaultGenOpts())
	assertResultError(t, r)
}

// --- Chat ---

func TestLlamaBackend_Chat_Good(t *core.T) {
	srv := newMockLlamaServer(t, "chat reply")
	defer srv.Close()

	lb := newLlamaBackendWithServer(srv)
	messages := []Message{
		{Role: "user", Content: "hello"},
	}

	r := lb.Chat(context.Background(), messages, DefaultGenOpts())
	requireResultOK(t, r)
	result := r.Value.(Result)
	core.AssertEqual(t, "chat reply", result.Text)
}

func TestLlamaBackend_Chat_MultiTurn_Good(t *core.T) {
	srv := newMockLlamaServer(t, "multi-turn reply")
	defer srv.Close()

	lb := newLlamaBackendWithServer(srv)
	messages := []Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "Hi there"},
		{Role: "assistant", Content: "Hello!"},
		{Role: "user", Content: "How are you?"},
	}

	r := lb.Chat(context.Background(), messages, DefaultGenOpts())
	requireResultOK(t, r)
	result := r.Value.(Result)
	core.AssertEqual(t, "multi-turn reply", result.Text)
}

func TestLlamaBackend_Chat_NotAvailable_Bad(t *core.T) {
	lb := &LlamaBackend{
		procID: "",
		http:   NewHTTPBackend("http://127.0.0.1:19999", ""),
	}

	messages := []Message{{Role: "user", Content: "hello"}}
	r := lb.Chat(context.Background(), messages, DefaultGenOpts())
	assertResultError(t, r)
	core.AssertContains(t, r.Error(), "not available")
}

// --- Stop ---

func TestLlamaBackend_Stop_NoProcID_Good(t *core.T) {
	lb := &LlamaBackend{} // procID is ""
	err := lb.Stop()
	assertResultOK(t, err)
}

// --- NewLlamaBackend constructor ---

func TestNewLlamaBackendDefaultPortGoodScenario(t *core.T) {
	lb := NewLlamaBackend(nil, LlamaOpts{ModelPath: "/tmp/model.gguf"})

	core.AssertEqual(t, 18090, lb.port)
	core.AssertEqual(t, "/tmp/model.gguf", lb.modelPath)
	core.AssertEqual(t, "llama-server", lb.llamaPath)
	core.AssertNotNil(t, lb.http)
}

func TestNewLlamaBackendCustomPortGoodScenario(t *core.T) {
	lb := NewLlamaBackend(nil, LlamaOpts{
		ModelPath: "/tmp/model.gguf",
		Port:      9999,
		LlamaPath: "/usr/local/bin/llama-server",
	})

	core.AssertEqual(t, 9999, lb.port)
	core.AssertEqual(t, "/usr/local/bin/llama-server", lb.llamaPath)
}

func TestNewLlamaBackendWithLoRAGoodScenario(t *core.T) {
	lb := NewLlamaBackend(nil, LlamaOpts{
		ModelPath: "/tmp/model.gguf",
		LoraPath:  "/tmp/lora.gguf",
	})

	core.AssertEqual(t, "/tmp/lora.gguf", lb.loraPath)
}

func TestNewLlamaBackendDefaultLlamaPathGoodScenario(t *core.T) {
	lb := NewLlamaBackend(nil, LlamaOpts{
		ModelPath: "/tmp/model.gguf",
		LlamaPath: "", // should default
	})
	core.AssertEqual(t, "llama-server", lb.llamaPath)
}

// --- Context cancellation ---

func TestLlamaBackend_Generate_ContextCancelled_Bad(t *core.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			w.WriteHeader(http.StatusOK)
		case "/v1/chat/completions":
			// Block until client disconnects.
			<-r.Context().Done()
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	lb := newLlamaBackendWithServer(srv)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	r := lb.Generate(ctx, "test", DefaultGenOpts())
	assertResultError(t, r)
}

// --- Empty choices edge case ---

func TestLlamaBackend_Generate_EmptyChoices_Ugly(t *core.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			w.WriteHeader(http.StatusOK)
		case "/v1/chat/completions":
			resp := chatResponse{Choices: []chatChoice{}}
			mustWriteJSONResponse(t, w, resp)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	lb := newLlamaBackendWithServer(srv)

	r := lb.Generate(context.Background(), "test", DefaultGenOpts())
	assertResultError(t, r)
	core.AssertContains(t, r.Error(), "no choices")
}

// --- GenOpts forwarding ---

func TestLlamaBackend_Generate_OptsForwarded_Good(t *core.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			w.WriteHeader(http.StatusOK)
		case "/v1/chat/completions":
			var req chatRequest
			mustReadJSONRequest(t, r, &req)
			// Verify opts were forwarded.
			core.AssertInDelta(t, 0.7, req.Temperature, 0.01)
			core.AssertEqual(t, 256, req.MaxTokens)

			resp := chatResponse{
				Choices: []chatChoice{{Message: Message{Role: "assistant", Content: "ok"}}},
			}
			mustWriteJSONResponse(t, w, resp)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	lb := newLlamaBackendWithServer(srv)

	opts := GenOpts{Temperature: 0.7, MaxTokens: 256}
	r := lb.Generate(context.Background(), "test", opts)
	requireResultOK(t, r)
	result := r.Value.(Result)
	core.AssertEqual(t, "ok", result.Text)
}

// --- Verify Backend interface compliance ---

func TestLlamaBackendInterfaceComplianceGoodScenario(t *core.T) {
	var backend Backend = &LlamaBackend{}
	core.AssertNotNil(t, backend)
	core.AssertEqual(t, "llama", backend.Name())
}

// TestLlamaBackend_SetMaxTokens_Good — spec §2.4: SetMaxTokens forwards to
// the internal HTTP client so subsequent generate calls carry max_tokens.
//
//	backend := ml.NewLlamaBackend(svc, opts)
//	backend.SetMaxTokens(2048)
func TestLlamaBackend_SetMaxTokens_Good(t *core.T) {
	lb := &LlamaBackend{
		http: NewHTTPBackend("http://localhost", ""),
	}
	lb.SetMaxTokens(2048)
	core.AssertEqual(t, 2048, lb.http.maxTokens)
}

// --- v0.9.0 shape triplets ---

func TestBackendLlama_NewLlamaBackend_Good(t *core.T) {
	b := NewLlamaBackend(LlamaOpts{ModelPath: "model.gguf", Port: 19001})
	core.AssertEqual(t, "llama", b.Name())
	core.AssertEqual(t, 19001, b.port)
	core.AssertEqual(t, "model.gguf", b.modelPath)
}

func TestBackendLlama_NewLlamaBackend_Bad(t *core.T) {
	b := NewLlamaBackend(nil)
	core.AssertEqual(t, 18090, b.port)
	core.AssertEqual(t, "llama-server", b.llamaPath)
}

func TestBackendLlama_NewLlamaBackend_Ugly(t *core.T) {
	b := NewLlamaBackend("from-string.gguf", LlamaOpts{Port: 19002})
	core.AssertEqual(t, "from-string.gguf", b.modelPath)
	core.AssertEqual(t, 19002, b.port)
}

func TestBackendLlama_LlamaBackend_Name_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	b := NewLlamaBackend()
	core.AssertEqual(t, "llama", b.Name())
}

func TestBackendLlama_LlamaBackend_Name_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	var b LlamaBackend
	core.AssertEqual(t, "llama", b.Name())
}

func TestBackendLlama_LlamaBackend_Name_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	b := NewLlamaBackend(LlamaOpts{LoraPath: "adapter.gguf"})
	core.AssertEqual(t, "llama", b.Name())
}

func TestBackendLlama_LlamaBackend_SetMaxTokens_Good(t *core.T) {
	b := NewLlamaBackend()
	b.SetMaxTokens(64)
	core.AssertEqual(t, 64, b.http.maxTokens)
}

func TestBackendLlama_LlamaBackend_SetMaxTokens_Bad(t *core.T) {
	var b LlamaBackend
	b.SetMaxTokens(64)
	core.AssertNil(t, b.http)
}

func TestBackendLlama_LlamaBackend_SetMaxTokens_Ugly(t *core.T) {
	b := NewLlamaBackend()
	b.SetMaxTokens(0)
	core.AssertEqual(t, 0, b.http.maxTokens)
}

func TestBackendLlama_LlamaBackend_LoadModel_Good(t *core.T) {
	b := NewLlamaBackend()
	r := b.LoadModel("ignored")
	requireResultOK(t, r)
	model := r.Value.(*LlamaTextModel)
	core.AssertEqual(t, "llama", model.ModelType())
}

func TestBackendLlama_LlamaBackend_LoadModel_Bad(t *core.T) {
	var b LlamaBackend
	assertResultError(t, b.LoadModel(""), "HTTP shim")
}

func TestBackendLlama_LlamaBackend_LoadModel_Ugly(t *core.T) {
	b := NewLlamaBackend(LlamaOpts{Port: 19003})
	r := b.LoadModel("unused")
	requireResultOK(t, r)
	model := r.Value.(*LlamaTextModel)
	assertResultOK(t, model.Close())
}

func TestBackendLlama_LlamaBackend_Available_Good(t *core.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer srv.Close()
	u, _ := url.Parse(srv.URL)
	port, _ := strconv.Atoi(u.Port())
	b := NewLlamaBackend(LlamaOpts{Port: port})
	b.procID = "proc"
	core.AssertTrue(t, b.Available())
}

func TestBackendLlama_LlamaBackend_Available_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	b := NewLlamaBackend()
	core.AssertFalse(t, b.Available())
}

func TestBackendLlama_LlamaBackend_Available_Ugly(t *core.T) {
	b := NewLlamaBackend(LlamaOpts{Port: 1})
	b.procID = "proc"
	core.AssertFalse(t, b.Available())
}

func TestBackendLlama_LlamaBackend_Start_Good(t *core.T) {
	b := NewLlamaBackend()
	err := b.Start(context.Background())
	core.AssertError(t, err, "process service")
}

func TestBackendLlama_LlamaBackend_Start_Bad(t *core.T) {
	var b LlamaBackend
	err := b.Start(context.Background())
	core.AssertError(t, err, "process service")
}

func TestBackendLlama_LlamaBackend_Start_Ugly(t *core.T) {
	b := NewLlamaBackend(LlamaOpts{ModelPath: "missing.gguf"})
	err := b.Start(context.Background())
	core.AssertError(t, err)
}

func TestBackendLlama_LlamaBackend_Stop_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	b := NewLlamaBackend()
	assertResultOK(t, b.Stop())
}

func TestBackendLlama_LlamaBackend_Stop_Bad(t *core.T) {
	b := NewLlamaBackend()
	b.procID = "proc"
	assertResultError(t, b.Stop(), "process service")
}

func TestBackendLlama_LlamaBackend_Stop_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	var b LlamaBackend
	assertResultOK(t, b.Stop())
}

func TestBackendLlama_LlamaBackend_Generate_Good(t *core.T) {
	b := NewLlamaBackend()
	r := b.Generate(context.Background(), "prompt", GenOpts{})
	assertResultError(t, r, "not available")
}

func TestBackendLlama_LlamaBackend_Generate_Bad(t *core.T) {
	var b LlamaBackend
	r := b.Generate(context.Background(), "prompt", GenOpts{})
	assertResultError(t, r, "not available")
}

func TestBackendLlama_LlamaBackend_Generate_Ugly(t *core.T) {
	b := NewLlamaBackend(LlamaOpts{Port: 1})
	b.procID = "proc"
	r := b.Generate(context.Background(), "prompt", GenOpts{})
	assertResultError(t, r, "not available")
}

func TestBackendLlama_LlamaBackend_Chat_Good(t *core.T) {
	b := NewLlamaBackend()
	r := b.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}, GenOpts{})
	assertResultError(t, r, "not available")
}

func TestBackendLlama_LlamaBackend_Chat_Bad(t *core.T) {
	var b LlamaBackend
	r := b.Chat(context.Background(), nil, GenOpts{})
	assertResultError(t, r, "not available")
}

func TestBackendLlama_LlamaBackend_Chat_Ugly(t *core.T) {
	b := NewLlamaBackend(LlamaOpts{Port: 1})
	b.procID = "proc"
	r := b.Chat(context.Background(), nil, GenOpts{})
	assertResultError(t, r, "not available")
}
