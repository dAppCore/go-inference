// SPDX-License-Identifier: EUPL-1.2

package openai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/ratelimit"
)

func TestOpenAI_Chat_Good_PostsRequestAndRecordsUsage(t *testing.T) {
	var waited atomic.Bool
	var recorded atomic.Bool

	limiter, err := ratelimit.NewWithConfig(ratelimit.Config{
		FilePath:  core.JoinPath(t.TempDir(), "ratelimits.yaml"),
		Providers: []ratelimit.Provider{ratelimit.ProviderOpenAI},
	})
	if err != nil {
		t.Fatalf("NewWithConfig() error = %v", err)
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !waited.Load() {
			t.Fatal("provider called HTTP before waiting for rate-limit capacity")
		}
		if r.Method != http.MethodPost {
			t.Fatalf("method = %s, want POST", r.Method)
		}
		if r.URL.Path != DefaultChatCompletionsPath {
			t.Fatalf("path = %s, want %s", r.URL.Path, DefaultChatCompletionsPath)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer sk-test" {
			t.Fatalf("Authorization = %q, want bearer token", got)
		}

		req, err := DecodeRequest(r.Body)
		if err != nil {
			t.Fatalf("DecodeRequest() error = %v", err)
		}
		if req.Model != "gpt-test" {
			t.Fatalf("model = %q, want gpt-test", req.Model)
		}
		if len(req.Messages) != 1 || req.Messages[0].Role != "user" || req.Messages[0].Content != "hello" {
			t.Fatalf("messages = %+v, want single user prompt", req.Messages)
		}
		if req.MaxTokens == nil || *req.MaxTokens != 8 {
			t.Fatalf("max_tokens = %v, want 8", req.MaxTokens)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(core.JSONMarshalString(ChatCompletionResponse{
			ID:      "chatcmpl-test",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   "gpt-test",
			Choices: []ChatChoice{{
				Index:        0,
				Message:      ChatMessage{Role: "assistant", Content: "hello back"},
				FinishReason: "stop",
			}},
			Usage: ChatUsage{
				PromptTokens:     5,
				CompletionTokens: 2,
				TotalTokens:      7,
			},
		})))
	}))
	defer server.Close()

	backend := NewBackend(Config{
		Name:         "openai-test",
		BaseURL:      server.URL,
		APIKey:       "sk-test",
		DefaultModel: "gpt-test",
		HTTPClient:   server.Client(),
		Limiter: waitRecordLimiter{
			inner:    limiter,
			waited:   &waited,
			recorded: &recorded,
		},
	})

	modelResult := backend.LoadModel("", inference.WithBackend("ignored"))
	if !modelResult.OK {
		t.Fatalf("LoadModel() error = %s", modelResult.Error())
	}
	model := modelResult.Value.(inference.TextModel)
	defer model.Close()

	var got string
	for token := range model.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hello"}}, inference.WithMaxTokens(8)) {
		got += token.Text
	}
	if errResult := model.Err(); !errResult.OK {
		t.Fatalf("Chat() Err() = %s", errResult.Error())
	}
	if got != "hello back" {
		t.Fatalf("Chat() = %q, want hello back", got)
	}
	if !recorded.Load() {
		t.Fatal("provider did not record usage after successful response")
	}
	metrics := model.Metrics()
	if metrics.PromptTokens != 5 || metrics.GeneratedTokens != 2 {
		t.Fatalf("Metrics() = %+v, want prompt=5 generated=2", metrics)
	}
}

func TestOpenAI_Chat_Good_PrependsContextAssemblerOutput(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		req, err := DecodeRequest(r.Body)
		if err != nil {
			t.Fatalf("DecodeRequest() error = %v", err)
		}
		if len(req.Messages) != 2 {
			t.Fatalf("messages len = %d, want context + user", len(req.Messages))
		}
		if req.Messages[0].Role != "system" || !core.Contains(req.Messages[0].Content, "retrieved context") {
			t.Fatalf("context message = %+v, want system context", req.Messages[0])
		}
		_, _ = w.Write([]byte(core.JSONMarshalString(ChatCompletionResponse{
			Model: "gpt-test",
			Choices: []ChatChoice{{
				Message: ChatMessage{Role: "assistant", Content: "context answer"},
			}},
		})))
	}))
	defer server.Close()

	backend := NewBackend(Config{
		Name:         "openai-test",
		BaseURL:      server.URL,
		DefaultModel: "gpt-test",
		HTTPClient:   server.Client(),
		ContextAssembler: ContextAssemblerFunc(func(context.Context, []inference.Message) core.Result {
			return core.Ok("retrieved context")
		}),
	})
	modelResult := backend.LoadModel("")
	if !modelResult.OK {
		t.Fatalf("LoadModel() error = %s", modelResult.Error())
	}
	model := modelResult.Value.(inference.TextModel)

	var got string
	for token := range model.Chat(context.Background(), []inference.Message{{Role: "user", Content: "question"}}) {
		got += token.Text
	}
	if errResult := model.Err(); !errResult.OK {
		t.Fatalf("Chat() Err() = %s", errResult.Error())
	}
	if got != "context answer" {
		t.Fatalf("Chat() = %q, want context answer", got)
	}
}

func TestOpenAI_Chat_Bad_ProviderErrorDoesNotRecordUsage(t *testing.T) {
	var recorded atomic.Bool

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(core.JSONMarshalString(ErrorResponse{
			Error: ErrorObject{
				Message: "rate limited",
				Type:    "rate_limit_error",
				Code:    "rate_limit_error",
			},
		})))
	}))
	defer server.Close()

	backend := NewBackend(Config{
		Name:         "openai-test",
		BaseURL:      server.URL,
		DefaultModel: "gpt-test",
		HTTPClient:   server.Client(),
		Limiter: waitRecordLimiter{
			recorded: &recorded,
		},
	})
	modelResult := backend.LoadModel("")
	if !modelResult.OK {
		t.Fatalf("LoadModel() error = %s", modelResult.Error())
	}
	model := modelResult.Value.(inference.TextModel)

	for range model.Generate(context.Background(), "hello") {
	}
	if model.Err().OK {
		t.Fatal("Generate() Err() = nil, want provider error")
	}
	if recorded.Load() {
		t.Fatal("provider recorded usage for failed response")
	}
}

func TestOpenAI_Capabilities_Good_ReportProviderIdentity(t *testing.T) {
	backend := NewBackend(Config{
		Name:         "openai-test",
		BaseURL:      "https://api.example.test",
		DefaultModel: "gpt-test",
	})
	if backend.Name() != "openai-test" {
		t.Fatalf("Name() = %q, want openai-test", backend.Name())
	}
	if !backend.Available() {
		t.Fatal("Available() = false, want true for configured provider")
	}
	backendReport := backend.Capabilities()
	if !backendReport.Supports(inference.CapabilityGenerate) || !backendReport.Supports(inference.CapabilityChat) {
		t.Fatalf("Backend Capabilities() = %+v, want generate and chat", backendReport.Capabilities)
	}

	modelResult := backend.LoadModel("")
	if !modelResult.OK {
		t.Fatalf("LoadModel() error = %s", modelResult.Error())
	}
	model := modelResult.Value.(inference.TextModel)
	report := model.(inference.CapabilityReporter).Capabilities()
	if report.Runtime.Backend != "openai-test" {
		t.Fatalf("Runtime.Backend = %q, want openai-test", report.Runtime.Backend)
	}
	if report.Runtime.NativeRuntime {
		t.Fatal("Runtime.NativeRuntime = true, want external provider")
	}
	if report.Model.ID != "gpt-test" {
		t.Fatalf("Model.ID = %q, want gpt-test", report.Model.ID)
	}
	if !report.Supports(inference.CapabilityGenerate) || !report.Supports(inference.CapabilityChat) {
		t.Fatalf("Capabilities() = %+v, want generate and chat", report.Capabilities)
	}
}

func TestOpenAI_Register_Good_AddsBackendToInferenceRegistry(t *testing.T) {
	name := "openai-register-" + t.Name()
	backend := Register(Config{
		Name:         name,
		BaseURL:      "https://api.example.test",
		DefaultModel: "gpt-test",
	})
	if backend == nil {
		t.Fatal("Register() returned nil")
	}

	got, ok := inference.Get(name)
	if !ok {
		t.Fatalf("inference.Get(%q) not found", name)
	}
	if got != backend {
		t.Fatalf("inference.Get(%q) = %T, want registered backend", name, got)
	}
}

func TestOpenai_ContextAssemblerFunc_AssembleContext_Good(t *testing.T) {
	assembler := ContextAssemblerFunc(func(context.Context, []inference.Message) core.Result {
		return core.Ok("retrieved context")
	})
	result := assembler.AssembleContext(context.Background(), nil)

	if !result.OK || result.Value.(string) != "retrieved context" {
		t.Fatalf("ContextAssemblerFunc.AssembleContext() = %#v, want context text", result)
	}
}

func TestOpenai_ContextAssemblerFunc_AssembleContext_Bad(t *testing.T) {
	var assembler ContextAssemblerFunc
	result := assembler.AssembleContext(context.Background(), nil)

	if !result.OK || result.Value.(string) != "" {
		t.Fatalf("ContextAssemblerFunc.AssembleContext() = %#v, want empty context", result)
	}
}

func TestOpenai_ContextAssemblerFunc_AssembleContext_Ugly(t *testing.T) {
	assembler := ContextAssemblerFunc(func(context.Context, []inference.Message) core.Result {
		return core.Fail(core.E("test.assembler", "failed", nil))
	})
	result := assembler.AssembleContext(context.Background(), nil)

	if result.OK || !core.Contains(result.Error(), "failed") {
		t.Fatalf("ContextAssemblerFunc.AssembleContext() = %#v, want failure", result)
	}
}

func TestOpenai_NewBackend_Good(t *testing.T) {
	backend := NewBackend(Config{Name: "provider", BaseURL: "https://api.example.test/", DefaultModel: "gpt"})

	if backend == nil || backend.Name() != "provider" {
		t.Fatalf("NewBackend() = %#v, want named backend", backend)
	}
	if backend.cfg.BaseURL != "https://api.example.test" {
		t.Fatalf("NewBackend() BaseURL = %q, want trimmed URL", backend.cfg.BaseURL)
	}
}

func TestOpenai_NewBackend_Bad(t *testing.T) {
	backend := NewBackend(Config{})

	if backend == nil || backend.Name() != defaultProviderName {
		t.Fatalf("NewBackend() = %#v, want default provider name", backend)
	}
	if backend.Available() {
		t.Fatal("NewBackend() Available() = true, want unavailable without URL/model")
	}
}

func TestOpenai_NewBackend_Ugly(t *testing.T) {
	backend := NewBackend(Config{Name: "  ", BaseURL: "https://api.example.test///", DefaultModel: "gpt"})

	if backend.Name() != defaultProviderName {
		t.Fatalf("NewBackend() Name() = %q, want default", backend.Name())
	}
	if backend.cfg.BaseURL != "https://api.example.test" {
		t.Fatalf("NewBackend() BaseURL = %q, want all trailing slashes removed", backend.cfg.BaseURL)
	}
}

func TestOpenai_Register_Good(t *testing.T) {
	name := "openai-register-good-" + t.Name()
	backend := Register(Config{Name: name, BaseURL: "https://api.example.test", DefaultModel: "gpt"})
	got, ok := inference.Get(name)

	if backend == nil || !ok || got != backend {
		t.Fatalf("Register() backend=%#v ok=%v got=%#v, want registered backend", backend, ok, got)
	}
}

func TestOpenai_Register_Bad(t *testing.T) {
	name := "openai-register-bad-" + t.Name()
	backend := Register(Config{Name: name})

	if backend == nil {
		t.Fatal("Register() returned nil")
	}
	if backend.Available() {
		t.Fatal("Register() backend Available() = true, want unavailable without static config")
	}
}

func TestOpenai_Register_Ugly(t *testing.T) {
	name := "openai-register-ugly-" + t.Name()
	first := Register(Config{Name: name, BaseURL: "https://first.example", DefaultModel: "first"})
	second := Register(Config{Name: name, BaseURL: "https://second.example", DefaultModel: "second"})
	got, ok := inference.Get(name)

	if first == nil || second == nil || !ok || got != second {
		t.Fatalf("Register() overwrite got=%#v ok=%v, want second backend", got, ok)
	}
}

func TestOpenai_Backend_Name_Good(t *testing.T) {
	backend := NewBackend(Config{Name: "openai-test"})

	if got := backend.Name(); got != "openai-test" {
		t.Fatalf("Backend.Name() = %q, want custom name", got)
	}
}

func TestOpenai_Backend_Name_Bad(t *testing.T) {
	var backend *Backend

	if got := backend.Name(); got != defaultProviderName {
		t.Fatalf("Backend.Name() = %q, want default for nil backend", got)
	}
}

func TestOpenai_Backend_Name_Ugly(t *testing.T) {
	backend := NewBackend(Config{Name: ""})

	if got := backend.Name(); got != defaultProviderName {
		t.Fatalf("Backend.Name() = %q, want default for blank name", got)
	}
}

func TestOpenai_Backend_Available_Good(t *testing.T) {
	backend := NewBackend(Config{BaseURL: "https://api.example.test", DefaultModel: "gpt"})

	if !backend.Available() {
		t.Fatal("Backend.Available() = false, want true for configured provider")
	}
}

func TestOpenai_Backend_Available_Bad(t *testing.T) {
	backend := NewBackend(Config{BaseURL: "https://api.example.test"})

	if backend.Available() {
		t.Fatal("Backend.Available() = true, want false without model")
	}
}

func TestOpenai_Backend_Available_Ugly(t *testing.T) {
	var backend *Backend

	if backend.Available() {
		t.Fatal("Backend.Available() = true, want false for nil backend")
	}
}

func TestOpenai_Backend_LoadModel_Good(t *testing.T) {
	backend := NewBackend(Config{BaseURL: "https://api.example.test", DefaultModel: "gpt"})
	result := backend.LoadModel("")

	if !result.OK {
		t.Fatalf("Backend.LoadModel() error = %s", result.Error())
	}
	if model := result.Value.(*Model); model.modelID != "gpt" {
		t.Fatalf("Backend.LoadModel() modelID = %q, want default model", model.modelID)
	}
}

func TestOpenai_Backend_LoadModel_Bad(t *testing.T) {
	var backend *Backend
	result := backend.LoadModel("gpt")

	if result.OK || !core.Contains(result.Error(), "backend is nil") {
		t.Fatalf("Backend.LoadModel() = %#v, want nil backend failure", result)
	}
}

func TestOpenai_Backend_LoadModel_Ugly(t *testing.T) {
	backend := NewBackend(Config{BaseURL: "https://api.example.test", DefaultModel: "fallback"})
	result := backend.LoadModel("override")

	if !result.OK {
		t.Fatalf("Backend.LoadModel() error = %s", result.Error())
	}
	if model := result.Value.(*Model); model.modelID != "override" {
		t.Fatalf("Backend.LoadModel() modelID = %q, want explicit path", model.modelID)
	}
}

func TestOpenai_Backend_Capabilities_Good(t *testing.T) {
	backend := NewBackend(Config{Name: "cap", BaseURL: "https://api.example.test", DefaultModel: "gpt"})
	report := backend.Capabilities()

	if !report.Available || !report.Supports(inference.CapabilityGenerate) || !report.Supports(inference.CapabilityChat) {
		t.Fatalf("Backend.Capabilities() = %+v, want available generate/chat report", report)
	}
}

func TestOpenai_Backend_Capabilities_Bad(t *testing.T) {
	var backend *Backend
	report := backend.Capabilities()

	if report.Available || report.Runtime.Backend != defaultProviderName {
		t.Fatalf("Backend.Capabilities() = %+v, want unavailable default report", report)
	}
}

func TestOpenai_Backend_Capabilities_Ugly(t *testing.T) {
	backend := NewBackend(Config{Name: "labels", BaseURL: "https://api.example.test/", DefaultModel: "gpt"})
	report := backend.Capabilities()

	if report.Runtime.Labels["base_url"] != "https://api.example.test" {
		t.Fatalf("Backend.Capabilities() labels = %+v, want trimmed base_url", report.Runtime.Labels)
	}
}

func TestOpenai_Model_Generate_Good(t *testing.T) {
	model, cleanup := newTestModel(t, "generated text", http.StatusOK)
	defer cleanup()

	var got string
	for token := range model.Generate(context.Background(), "hello", inference.WithMaxTokens(8)) {
		got += token.Text
	}

	if got != "generated text" {
		t.Fatalf("Model.Generate() = %q, want generated text", got)
	}
	if errResult := model.Err(); !errResult.OK {
		t.Fatalf("Model.Generate() Err() = %s", errResult.Error())
	}
}

func TestOpenai_Model_Generate_Bad(t *testing.T) {
	model, cleanup := newTestModel(t, "rate limited", http.StatusTooManyRequests)
	defer cleanup()

	for range model.Generate(context.Background(), "hello") {
		t.Fatal("Model.Generate() yielded token for provider error")
	}

	if errResult := model.Err(); errResult.OK || !core.Contains(errResult.Error(), "HTTP") {
		t.Fatalf("Model.Generate() Err() = %#v, want provider failure", errResult)
	}
}

func TestOpenai_Model_Generate_Ugly(t *testing.T) {
	model, cleanup := newTestModel(t, "", http.StatusOK)
	defer cleanup()

	count := 0
	for range model.Generate(context.Background(), "hello") {
		count++
	}

	if count != 0 {
		t.Fatalf("Model.Generate() yielded %d tokens, want none for empty content", count)
	}
}

func TestOpenai_Model_Chat_Good(t *testing.T) {
	model, cleanup := newTestModel(t, "chat text", http.StatusOK)
	defer cleanup()

	var got string
	for token := range model.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}) {
		got += token.Text
	}

	if got != "chat text" {
		t.Fatalf("Model.Chat() = %q, want chat text", got)
	}
}

func TestOpenai_Model_Chat_Bad(t *testing.T) {
	model, cleanup := newTestModel(t, "bad", http.StatusInternalServerError)
	defer cleanup()

	for range model.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}) {
		t.Fatal("Model.Chat() yielded token for failed provider")
	}
	if errResult := model.Err(); errResult.OK {
		t.Fatal("Model.Chat() Err() OK = true, want failure")
	}
}

func TestOpenai_Model_Chat_Ugly(t *testing.T) {
	model, cleanup := newTestModel(t, "context chat", http.StatusOK)
	defer cleanup()
	model.backend.cfg.ContextAssembler = ContextAssemblerFunc(func(context.Context, []inference.Message) core.Result {
		return core.Ok("context")
	})

	for range model.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}) {
	}

	if errResult := model.Err(); !errResult.OK {
		t.Fatalf("Model.Chat() Err() = %s, want context-injected success", errResult.Error())
	}
}

func TestOpenai_Model_Classify_Good(t *testing.T) {
	model := &Model{}
	result := model.Classify(context.Background(), []string{"prompt"})

	if result.OK || !core.Contains(result.Error(), "not supported") {
		t.Fatalf("Model.Classify() = %#v, want unsupported failure", result)
	}
}

func TestOpenai_Model_Classify_Bad(t *testing.T) {
	var model *Model
	result := model.Classify(context.Background(), nil)

	if result.OK {
		t.Fatal("Model.Classify() OK = true, want unsupported failure")
	}
}

func TestOpenai_Model_Classify_Ugly(t *testing.T) {
	model := &Model{}
	result := model.Classify(context.Background(), []string{"a", "b"}, inference.WithMaxTokens(1))

	if !core.Contains(result.Error(), "classification") {
		t.Fatalf("Model.Classify() error = %q, want classification context", result.Error())
	}
}

func TestOpenai_Model_BatchGenerate_Good(t *testing.T) {
	model, cleanup := newTestModel(t, "batch", http.StatusOK)
	defer cleanup()
	result := model.BatchGenerate(context.Background(), []string{"a", "b"})

	if !result.OK {
		t.Fatalf("Model.BatchGenerate() error = %s", result.Error())
	}
	if batches := result.Value.([]inference.BatchResult); len(batches) != 2 || len(batches[0].Tokens) != 1 {
		t.Fatalf("Model.BatchGenerate() = %+v, want two token batches", batches)
	}
}

func TestOpenai_Model_BatchGenerate_Bad(t *testing.T) {
	model, cleanup := newTestModel(t, "bad", http.StatusBadGateway)
	defer cleanup()
	result := model.BatchGenerate(context.Background(), []string{"a"})

	if !result.OK {
		t.Fatalf("Model.BatchGenerate() outer error = %s, want per-prompt error", result.Error())
	}
	if batches := result.Value.([]inference.BatchResult); len(batches) != 1 || batches[0].Err == nil {
		t.Fatalf("Model.BatchGenerate() = %+v, want per-prompt error", batches)
	}
}

func TestOpenai_Model_BatchGenerate_Ugly(t *testing.T) {
	model, cleanup := newTestModel(t, "unused", http.StatusOK)
	defer cleanup()
	result := model.BatchGenerate(context.Background(), nil)

	if !result.OK || len(result.Value.([]inference.BatchResult)) != 0 {
		t.Fatalf("Model.BatchGenerate() = %#v, want empty batch success", result)
	}
}

func TestOpenai_Model_ModelType_Good(t *testing.T) {
	model := &Model{}

	if got := model.ModelType(); got != "openai-compatible" {
		t.Fatalf("Model.ModelType() = %q, want openai-compatible", got)
	}
}

func TestOpenai_Model_ModelType_Bad(t *testing.T) {
	var model *Model

	if got := model.ModelType(); got == "" {
		t.Fatal("Model.ModelType() = empty, want stable type even for nil receiver")
	}
}

func TestOpenai_Model_ModelType_Ugly(t *testing.T) {
	model := &Model{modelID: "custom"}

	if got := model.ModelType(); !core.Contains(got, "openai") {
		t.Fatalf("Model.ModelType() = %q, want provider family", got)
	}
}

func TestOpenai_Model_Info_Good(t *testing.T) {
	model := &Model{}
	info := model.Info()

	if info.Architecture != "openai-compatible" {
		t.Fatalf("Model.Info() = %+v, want openai-compatible architecture", info)
	}
}

func TestOpenai_Model_Info_Bad(t *testing.T) {
	var model *Model
	info := model.Info()

	if info.Architecture == "" {
		t.Fatalf("Model.Info() = %+v, want architecture for nil receiver", info)
	}
}

func TestOpenai_Model_Info_Ugly(t *testing.T) {
	model := &Model{modelID: "gpt-test"}
	info := model.Info()

	if info.QuantBits != 0 || info.NumLayers != 0 {
		t.Fatalf("Model.Info() = %+v, want external provider metadata only", info)
	}
}

func TestOpenai_Model_Metrics_Good(t *testing.T) {
	model := &Model{metrics: inference.GenerateMetrics{PromptTokens: 3, GeneratedTokens: 2}}
	metrics := model.Metrics()

	if metrics.PromptTokens != 3 || metrics.GeneratedTokens != 2 {
		t.Fatalf("Model.Metrics() = %+v, want stored metrics", metrics)
	}
}

func TestOpenai_Model_Metrics_Bad(t *testing.T) {
	model := &Model{}
	metrics := model.Metrics()

	if metrics.PromptTokens != 0 || metrics.GeneratedTokens != 0 {
		t.Fatalf("Model.Metrics() = %+v, want zero metrics before request", metrics)
	}
}

func TestOpenai_Model_Metrics_Ugly(t *testing.T) {
	model := &Model{}
	model.setResult(inference.GenerateMetrics{GeneratedTokens: 7}, core.Ok(nil))
	metrics := model.Metrics()

	if metrics.GeneratedTokens != 7 {
		t.Fatalf("Model.Metrics() = %+v, want setResult metrics", metrics)
	}
}

func TestOpenai_Model_Err_Good(t *testing.T) {
	model := &Model{}
	result := model.Err()

	if !result.OK {
		t.Fatalf("Model.Err() = %#v, want OK before failure", result)
	}
}

func TestOpenai_Model_Err_Bad(t *testing.T) {
	model := &Model{lastErr: core.E("test", "failed", nil)}
	result := model.Err()

	if result.OK || !core.Contains(result.Error(), "failed") {
		t.Fatalf("Model.Err() = %#v, want stored error", result)
	}
}

func TestOpenai_Model_Err_Ugly(t *testing.T) {
	model := &Model{}
	model.setResult(inference.GenerateMetrics{}, core.Fail(core.E("test", "set failure", nil)))
	result := model.Err()

	if result.OK || !core.Contains(result.Error(), "set failure") {
		t.Fatalf("Model.Err() = %#v, want setResult failure", result)
	}
}

func TestOpenai_Model_Close_Good(t *testing.T) {
	model := &Model{}
	result := model.Close()

	if !result.OK {
		t.Fatalf("Model.Close() = %#v, want OK", result)
	}
}

func TestOpenai_Model_Close_Bad(t *testing.T) {
	var model *Model
	result := model.Close()

	if !result.OK {
		t.Fatalf("Model.Close() = %#v, want nil receiver close OK", result)
	}
}

func TestOpenai_Model_Close_Ugly(t *testing.T) {
	model := &Model{lastErr: core.AnError}
	result := model.Close()

	if !result.OK || model.lastErr == nil {
		t.Fatalf("Model.Close() = %#v lastErr=%v, want close without clearing generation error", result, model.lastErr)
	}
}

func TestOpenai_Model_Capabilities_Good(t *testing.T) {
	backend := NewBackend(Config{Name: "cap", BaseURL: "https://api.example.test", DefaultModel: "gpt"})
	model := &Model{backend: backend, modelID: "gpt"}
	report := model.Capabilities()

	if report.Model.ID != "gpt" || !report.Supports(inference.CapabilityGenerate) {
		t.Fatalf("Model.Capabilities() = %+v, want model capability report", report)
	}
}

func TestOpenai_Model_Capabilities_Bad(t *testing.T) {
	var model *Model
	report := model.Capabilities()

	if report.Runtime.Backend != defaultProviderName || report.Model.ID != "" {
		t.Fatalf("Model.Capabilities() = %+v, want default nil model report", report)
	}
}

func TestOpenai_Model_Capabilities_Ugly(t *testing.T) {
	backend := NewBackend(Config{Name: "cap", BaseURL: "https://api.example.test/", DefaultModel: "gpt"})
	model := &Model{backend: backend, modelID: "gpt"}
	report := model.Capabilities()

	if report.Runtime.Labels["base_url"] != "https://api.example.test" || report.Model.Labels["provider"] == "" {
		t.Fatalf("Model.Capabilities() labels = runtime:%+v model:%+v", report.Runtime.Labels, report.Model.Labels)
	}
}

func newTestModel(t *testing.T, content string, status int) (*Model, func()) {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if status != http.StatusOK {
			w.WriteHeader(status)
			_, _ = w.Write([]byte(core.JSONMarshalString(ErrorResponse{
				Error: ErrorObject{Message: content},
			})))
			return
		}
		_, _ = w.Write([]byte(core.JSONMarshalString(ChatCompletionResponse{
			Model: "gpt-test",
			Choices: []ChatChoice{{
				Message: ChatMessage{Role: "assistant", Content: content},
			}},
			Usage: ChatUsage{PromptTokens: 1, CompletionTokens: 1},
		})))
	}))
	backend := NewBackend(Config{
		Name:         "test",
		BaseURL:      server.URL,
		DefaultModel: "gpt-test",
		HTTPClient:   server.Client(),
	})
	result := backend.LoadModel("")
	if !result.OK {
		t.Fatalf("LoadModel() error = %s", result.Error())
	}
	return result.Value.(*Model), server.Close
}

type waitRecordLimiter struct {
	inner interface {
		WaitForCapacity(context.Context, string, int) error
		RecordUsage(string, int, int)
	}
	waited   *atomic.Bool
	recorded *atomic.Bool
}

func (l waitRecordLimiter) WaitForCapacity(ctx context.Context, model string, tokens int) error {
	if l.waited != nil {
		l.waited.Store(true)
	}
	if l.inner != nil {
		return l.inner.WaitForCapacity(ctx, model, tokens)
	}
	return nil
}

func (l waitRecordLimiter) RecordUsage(model string, promptTokens, outputTokens int) {
	if l.recorded != nil {
		l.recorded.Store(true)
	}
	if l.inner != nil {
		l.inner.RecordUsage(model, promptTokens, outputTokens)
	}
}

// stubRoundTripper injects a network-free HTTP failure — either
// RoundTrip itself erroring (client.Do failing) or a canned Response
// whose Body errors on Read. Both paths in doRequest are otherwise
// unreachable without a real flaky socket.
type stubRoundTripper struct {
	err  error
	resp *http.Response
}

func (rt stubRoundTripper) RoundTrip(*http.Request) (*http.Response, error) {
	if rt.err != nil {
		return nil, rt.err
	}
	return rt.resp, nil
}

// TestOpenai_Backend_LoadModel_Bad_NoDefaultModel covers the
// modelID=="" branch — no explicit path AND no configured
// DefaultModel.
func TestOpenai_Backend_LoadModel_Bad_NoDefaultModel(t *testing.T) {
	backend := NewBackend(Config{BaseURL: "https://api.example.test"})
	result := backend.LoadModel("")

	if result.OK || !core.Contains(result.Error(), "model id is required") {
		t.Fatalf("LoadModel() = %#v, want model-id-required failure", result)
	}
}

// TestOpenai_Backend_LoadModel_Bad_NoBaseURL covers the
// BaseURL=="" branch — a resolved model id but no configured base URL.
func TestOpenai_Backend_LoadModel_Bad_NoBaseURL(t *testing.T) {
	backend := NewBackend(Config{DefaultModel: "gpt-test"})
	result := backend.LoadModel("")

	if result.OK || !core.Contains(result.Error(), "base URL is required") {
		t.Fatalf("LoadModel() = %#v, want base-URL-required failure", result)
	}
}

// TestOpenai_Model_Complete_Bad_NilBackend covers complete()'s own
// nil-model/nil-backend guard, reached through the public Chat entry
// point on a zero-value Model.
func TestOpenai_Model_Complete_Bad_NilBackend(t *testing.T) {
	model := &Model{}
	for range model.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}) {
		t.Fatal("Chat() yielded a token for a backend-less model")
	}
	if errResult := model.Err(); errResult.OK || !core.Contains(errResult.Error(), "model is nil") {
		t.Fatalf("Chat() Err() = %#v, want model-is-nil failure", errResult)
	}
}

// TestOpenai_Model_Complete_Bad_ContextAssemblerError covers
// contextMessages' own failure-propagation branch and complete()'s
// pass-through of it.
func TestOpenai_Model_Complete_Bad_ContextAssemblerError(t *testing.T) {
	model, cleanup := newTestModel(t, "unused", http.StatusOK)
	defer cleanup()
	model.backend.cfg.ContextAssembler = ContextAssemblerFunc(func(context.Context, []inference.Message) core.Result {
		return core.Fail(core.E("test", "assembler failed", nil))
	})

	for range model.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}) {
		t.Fatal("Chat() yielded a token despite a failing context assembler")
	}
	if errResult := model.Err(); errResult.OK || !core.Contains(errResult.Error(), "assembler failed") {
		t.Fatalf("Chat() Err() = %#v, want assembler failure", errResult)
	}
}

// TestOpenai_Model_ContextMessages_Ugly_EmptyAssembledText covers the
// contextText=="" branch — the assembler succeeds but returns
// whitespace-only text, which must not prepend an empty context
// message.
func TestOpenai_Model_ContextMessages_Ugly_EmptyAssembledText(t *testing.T) {
	var gotMessages []inference.Message
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		req, err := DecodeRequest(r.Body)
		if err != nil {
			t.Fatalf("DecodeRequest() error = %v", err)
		}
		gotMessages = openaiMessages2Inference(req.Messages)
		_, _ = w.Write([]byte(core.JSONMarshalString(ChatCompletionResponse{
			Model:   "gpt-test",
			Choices: []ChatChoice{{Message: ChatMessage{Role: "assistant", Content: "ok"}}},
		})))
	}))
	defer server.Close()

	backend := NewBackend(Config{
		BaseURL: server.URL, DefaultModel: "gpt-test", HTTPClient: server.Client(),
		ContextAssembler: ContextAssemblerFunc(func(context.Context, []inference.Message) core.Result {
			return core.Ok("   ")
		}),
	})
	modelResult := backend.LoadModel("")
	if !modelResult.OK {
		t.Fatalf("LoadModel() error = %s", modelResult.Error())
	}
	model := modelResult.Value.(inference.TextModel)

	for range model.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}) {
	}
	if errResult := model.Err(); !errResult.OK {
		t.Fatalf("Chat() Err() = %s", errResult.Error())
	}
	if len(gotMessages) != 1 || gotMessages[0].Role != "user" {
		t.Fatalf("provider received %+v, want no synthesised context message", gotMessages)
	}
}

// TestOpenai_Model_Complete_Bad_LimiterWaitError covers the
// limiter.WaitForCapacity error-propagation branch.
func TestOpenai_Model_Complete_Bad_LimiterWaitError(t *testing.T) {
	model, cleanup := newTestModel(t, "unused", http.StatusOK)
	defer cleanup()
	model.backend.cfg.Limiter = waitErrLimiter{err: core.E("test", "quota exhausted", nil)}

	for range model.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}) {
		t.Fatal("Chat() yielded a token despite a limiter rejection")
	}
	if errResult := model.Err(); errResult.OK || !core.Contains(errResult.Error(), "quota exhausted") {
		t.Fatalf("Chat() Err() = %#v, want quota-exhausted failure", errResult)
	}
}

type waitErrLimiter struct{ err error }

func (l waitErrLimiter) WaitForCapacity(context.Context, string, int) error { return l.err }
func (l waitErrLimiter) RecordUsage(string, int, int)                       {}

// TestOpenai_Model_Complete_Ugly_TopPTopKThreaded covers the
// cfg.TopP>0 / cfg.TopK>0 branches — both must thread onto the
// outbound request when the caller supplies them.
func TestOpenai_Model_Complete_Ugly_TopPTopKThreaded(t *testing.T) {
	var gotTopP float32
	var gotTopK int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		req, err := DecodeRequest(r.Body)
		if err != nil {
			t.Fatalf("DecodeRequest() error = %v", err)
		}
		if req.TopP != nil {
			gotTopP = *req.TopP
		}
		if req.TopK != nil {
			gotTopK = *req.TopK
		}
		_, _ = w.Write([]byte(core.JSONMarshalString(ChatCompletionResponse{
			Model:   "gpt-test",
			Choices: []ChatChoice{{Message: ChatMessage{Role: "assistant", Content: "ok"}}},
		})))
	}))
	defer server.Close()

	backend := NewBackend(Config{BaseURL: server.URL, DefaultModel: "gpt-test", HTTPClient: server.Client()})
	modelResult := backend.LoadModel("")
	if !modelResult.OK {
		t.Fatalf("LoadModel() error = %s", modelResult.Error())
	}
	model := modelResult.Value.(inference.TextModel)

	for range model.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}, inference.WithTopP(0.9), inference.WithTopK(40)) {
	}
	if errResult := model.Err(); !errResult.OK {
		t.Fatalf("Chat() Err() = %s", errResult.Error())
	}
	if gotTopP != 0.9 || gotTopK != 40 {
		t.Fatalf("provider received top_p=%v top_k=%v, want 0.9/40", gotTopP, gotTopK)
	}
}

// TestOpenai_Model_Complete_Bad_NoChoices covers the
// len(response.Choices)==0 rejection.
func TestOpenai_Model_Complete_Bad_NoChoices(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(core.JSONMarshalString(ChatCompletionResponse{Model: "gpt-test"})))
	}))
	defer server.Close()

	backend := NewBackend(Config{BaseURL: server.URL, DefaultModel: "gpt-test", HTTPClient: server.Client()})
	modelResult := backend.LoadModel("")
	if !modelResult.OK {
		t.Fatalf("LoadModel() error = %s", modelResult.Error())
	}
	model := modelResult.Value.(inference.TextModel)

	for range model.Generate(context.Background(), "hi") {
		t.Fatal("Generate() yielded a token for a choice-less response")
	}
	if errResult := model.Err(); errResult.OK || !core.Contains(errResult.Error(), "no choices") {
		t.Fatalf("Generate() Err() = %#v, want no-choices failure", errResult)
	}
}

// TestOpenai_Model_DoRequest_Bad_MalformedURL covers
// http.NewRequestWithContext's own construction failure — a base URL
// containing a raw control character.
func TestOpenai_Model_DoRequest_Bad_MalformedURL(t *testing.T) {
	backend := NewBackend(Config{BaseURL: "http://example.test/\x7f", DefaultModel: "gpt-test"})
	modelResult := backend.LoadModel("")
	if !modelResult.OK {
		t.Fatalf("LoadModel() error = %s", modelResult.Error())
	}
	model := modelResult.Value.(inference.TextModel)

	for range model.Generate(context.Background(), "hi") {
		t.Fatal("Generate() yielded a token for a malformed URL")
	}
	if errResult := model.Err(); errResult.OK {
		t.Fatal("Generate() Err() OK = true, want request-construction failure")
	}
}

// TestOpenai_Model_DoRequest_Good_OrganisationAndProjectHeaders
// covers the Organisation/Project header branches.
func TestOpenai_Model_DoRequest_Good_OrganisationAndProjectHeaders(t *testing.T) {
	var gotOrg, gotProject string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotOrg = r.Header.Get("OpenAI-Organization")
		gotProject = r.Header.Get("OpenAI-Project")
		_, _ = w.Write([]byte(core.JSONMarshalString(ChatCompletionResponse{
			Model:   "gpt-test",
			Choices: []ChatChoice{{Message: ChatMessage{Role: "assistant", Content: "ok"}}},
		})))
	}))
	defer server.Close()

	backend := NewBackend(Config{
		BaseURL: server.URL, DefaultModel: "gpt-test", HTTPClient: server.Client(),
		Organisation: "org-1", Project: "proj-1",
	})
	modelResult := backend.LoadModel("")
	if !modelResult.OK {
		t.Fatalf("LoadModel() error = %s", modelResult.Error())
	}
	model := modelResult.Value.(inference.TextModel)

	for range model.Generate(context.Background(), "hi") {
	}
	if errResult := model.Err(); !errResult.OK {
		t.Fatalf("Generate() Err() = %s", errResult.Error())
	}
	if gotOrg != "org-1" || gotProject != "proj-1" {
		t.Fatalf("headers org=%q project=%q, want org-1/proj-1", gotOrg, gotProject)
	}
}

// TestOpenai_Model_DoRequest_Bad_ClientDoFails covers the
// m.client.Do error branch via a RoundTripper that fails outright —
// hermetic (no real socket involved).
func TestOpenai_Model_DoRequest_Bad_ClientDoFails(t *testing.T) {
	backend := NewBackend(Config{
		BaseURL: "http://unused.test", DefaultModel: "gpt-test",
		HTTPClient: &http.Client{Transport: stubRoundTripper{err: core.E("test", "connection refused", nil)}},
	})
	modelResult := backend.LoadModel("")
	if !modelResult.OK {
		t.Fatalf("LoadModel() error = %s", modelResult.Error())
	}
	model := modelResult.Value.(inference.TextModel)

	for range model.Generate(context.Background(), "hi") {
		t.Fatal("Generate() yielded a token despite a transport failure")
	}
	if errResult := model.Err(); errResult.OK || !core.Contains(errResult.Error(), "connection refused") {
		t.Fatalf("Generate() Err() = %#v, want transport failure", errResult)
	}
}

// TestOpenai_Model_DoRequest_Bad_ResponseBodyReadFails covers the
// io.ReadAll(resp.Body) error branch via a RoundTripper returning a
// canned 200 response whose Body errors on Read.
func TestOpenai_Model_DoRequest_Bad_ResponseBodyReadFails(t *testing.T) {
	backend := NewBackend(Config{
		BaseURL: "http://unused.test", DefaultModel: "gpt-test",
		HTTPClient: &http.Client{Transport: stubRoundTripper{resp: &http.Response{
			StatusCode: http.StatusOK,
			Body:       erroringReadCloser{},
			Header:     make(http.Header),
		}}},
	})
	modelResult := backend.LoadModel("")
	if !modelResult.OK {
		t.Fatalf("LoadModel() error = %s", modelResult.Error())
	}
	model := modelResult.Value.(inference.TextModel)

	for range model.Generate(context.Background(), "hi") {
		t.Fatal("Generate() yielded a token despite an unreadable response body")
	}
	if errResult := model.Err(); errResult.OK {
		t.Fatal("Generate() Err() OK = true, want response-read failure")
	}
}

// TestOpenai_Model_DoRequest_Bad_MalformedResponseBody covers the
// core.JSONUnmarshalString decode-failure branch on a 200 response
// whose body is not valid JSON.
func TestOpenai_Model_DoRequest_Bad_MalformedResponseBody(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`not json`))
	}))
	defer server.Close()

	backend := NewBackend(Config{BaseURL: server.URL, DefaultModel: "gpt-test", HTTPClient: server.Client()})
	modelResult := backend.LoadModel("")
	if !modelResult.OK {
		t.Fatalf("LoadModel() error = %s", modelResult.Error())
	}
	model := modelResult.Value.(inference.TextModel)

	for range model.Generate(context.Background(), "hi") {
		t.Fatal("Generate() yielded a token for a malformed response body")
	}
	if errResult := model.Err(); errResult.OK || !core.Contains(errResult.Error(), "decode provider response") {
		t.Fatalf("Generate() Err() = %#v, want decode failure", errResult)
	}
}

// TestOpenai_Model_EstimateTokens_Good_CustomEstimator covers the
// Config.EstimateTokens override branch.
func TestOpenai_Model_EstimateTokens_Good_CustomEstimator(t *testing.T) {
	var called bool
	model := &Model{backend: NewBackend(Config{
		EstimateTokens: func([]inference.Message, inference.GenerateConfig) int {
			called = true
			return 99
		},
	})}

	if got := model.estimateTokens(nil, inference.GenerateConfig{}); got != 99 {
		t.Fatalf("estimateTokens() = %d, want 99 from the custom estimator", got)
	}
	if !called {
		t.Fatal("custom EstimateTokens was not invoked")
	}
}

// TestOpenai_Model_EstimateTokens_Bad_FloorsAtOne covers the
// estimate<1 floor for very short (or empty) content with no
// MaxTokens contribution.
func TestOpenai_Model_EstimateTokens_Bad_FloorsAtOne(t *testing.T) {
	model := &Model{backend: NewBackend(Config{})}

	if got := model.estimateTokens([]inference.Message{{Content: "hi"}}, inference.GenerateConfig{}); got != 1 {
		t.Fatalf("estimateTokens() = %d, want floor of 1", got)
	}
}

// TestOpenai_Model_SetResult_Ugly_NonErrorFailureValue drives
// setResult directly with a hand-constructed core.Result whose Value
// is not an error — production call sites never build one this way
// (Model.Err() always stores a genuine error), but setResult must not
// panic on it.
func TestOpenai_Model_SetResult_Ugly_NonErrorFailureValue(t *testing.T) {
	model := &Model{}
	model.setResult(inference.GenerateMetrics{GeneratedTokens: 3}, core.Result{OK: false, Value: "not an error"})

	if model.lastErr == nil {
		t.Fatal("setResult(non-error Value) left lastErr nil, want the unexpected-value fallback error")
	}
}

// TestOpenai_ProviderError_Ugly_NonJSONBody covers the non-JSON-body
// fallback branch — a 5xx body that doesn't start with '{'.
func TestOpenai_ProviderError_Ugly_NonJSONBody(t *testing.T) {
	result := providerError(http.StatusBadGateway, "<html>Bad Gateway</html>")

	if result.OK || !core.Contains(result.Error(), "Bad Gateway") {
		t.Fatalf("providerError(non-JSON body) = %#v, want the raw body in the message", result)
	}
}

// TestOpenai_ProviderError_Ugly_JSONBodyMissingMessage covers the
// case where the body starts with '{' and parses as an ErrorResponse,
// but Error.Message is empty — falls through to the raw-body message
// rather than surfacing an empty reason.
func TestOpenai_ProviderError_Ugly_JSONBodyMissingMessage(t *testing.T) {
	result := providerError(http.StatusBadGateway, `{"error":{}}`)

	if result.OK || !core.Contains(result.Error(), `{"error":{}}`) {
		t.Fatalf("providerError(empty message) = %#v, want raw body fallback", result)
	}
}

// openaiMessages2Inference is a tiny test-local inverse of
// openaiMessages — converts decoded wire ChatMessages back into
// inference.Message for assertions in this file.
func openaiMessages2Inference(messages []ChatMessage) []inference.Message {
	out := make([]inference.Message, 0, len(messages))
	for _, msg := range messages {
		out = append(out, inference.Message{Role: msg.Role, Content: msg.Content})
	}
	return out
}
