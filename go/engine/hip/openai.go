// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"encoding/json"
	"net/http"
	"strings"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	openaicompat "dappco.re/go/inference/serving/provider/openai"
)

// NewOpenAIResolver returns a resolver that lazily loads modelPath through the
// ROCm backend registered by this package.
func NewOpenAIResolver(modelPath string, opts ...inference.LoadOption) *openaicompat.BackendResolver {
	return openaicompat.NewBackendResolver("rocm", modelPath, opts...)
}

// NewOpenAIHandler exposes modelPath through the shared OpenAI-compatible chat
// completions handler.
func NewOpenAIHandler(modelPath string, opts ...inference.LoadOption) http.Handler {
	return openaicompat.NewHandler(NewOpenAIResolver(modelPath, opts...))
}

// NewOpenAIResponsesHandler exposes the OpenAI-compatible Responses endpoint
// over a caller-provided resolver.
func NewOpenAIResponsesHandler(resolver openaicompat.Resolver) http.Handler {
	return &openAIResponsesHandler{resolver: resolver}
}

// NewOpenAIResponsesHandlerForModel exposes modelPath through the
// OpenAI-compatible Responses endpoint.
func NewOpenAIResponsesHandlerForModel(modelPath string, opts ...inference.LoadOption) http.Handler {
	return NewOpenAIResponsesHandler(NewOpenAIResolver(modelPath, opts...))
}

// NewOpenAIServiceMux returns a mux with chat completions, responses, and the
// shared capability/cache/cancel service endpoints mounted.
func NewOpenAIServiceMux(resolver openaicompat.Resolver) *http.ServeMux {
	mux := http.NewServeMux()
	mux.Handle(openaicompat.DefaultChatCompletionsPath, openaicompat.NewHandler(resolver))
	mux.Handle(openaicompat.DefaultResponsesPath, NewOpenAIResponsesHandler(resolver))
	mux.Handle(openaicompat.DefaultCapabilitiesPath, openaicompat.NewCapabilityHandler(resolver))
	mux.Handle(openaicompat.DefaultCacheStatsPath, openaicompat.NewCacheStatsHandler(resolver))
	mux.Handle(openaicompat.DefaultCacheWarmPath, openaicompat.NewCacheWarmHandler(resolver))
	mux.Handle(openaicompat.DefaultCacheClearPath, openaicompat.NewCacheClearHandler(resolver))
	mux.Handle(openaicompat.DefaultCancelPath, openaicompat.NewCancelHandler(resolver))
	mux.Handle(openaicompat.DefaultEmbeddingsPath, openaicompat.NewEmbeddingsHandler(resolver))
	mux.Handle(openaicompat.DefaultRerankPath, openaicompat.NewRerankHandler(resolver))
	return mux
}

// NewOpenAIServiceMuxForModel exposes modelPath through the OpenAI-compatible
// chat, responses, capability, cache, cancel, embeddings, and rerank endpoints.
func NewOpenAIServiceMuxForModel(modelPath string, opts ...inference.LoadOption) *http.ServeMux {
	return NewOpenAIServiceMux(NewOpenAIResolver(modelPath, opts...))
}

type openAIResponsesHandler struct {
	resolver openaicompat.Resolver
}

func (handler *openAIResponsesHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if handler == nil || handler.resolver == nil {
		writeROCmOpenAIError(w, http.StatusServiceUnavailable, "responses handler is not configured", "model")
		return
	}
	if r == nil || r.Body == nil {
		writeROCmOpenAIError(w, http.StatusBadRequest, "request body is nil", "body")
		return
	}
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		writeROCmOpenAIError(w, http.StatusMethodNotAllowed, "method not allowed", "method")
		return
	}
	var req openaicompat.ResponseRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeROCmOpenAIError(w, http.StatusBadRequest, "invalid request body", "body")
		return
	}
	if core.Trim(req.Model) == "" {
		writeROCmOpenAIError(w, http.StatusBadRequest, "model is required", "model")
		return
	}
	messages := openaicompat.ResponseMessages(req)
	if !hasROCmWireMessages(messages) {
		writeROCmOpenAIError(w, http.StatusBadRequest, "input or instructions are required", "input")
		return
	}
	opts, err := openaicompat.ResponseGenerateOptions(req)
	if err != nil {
		writeROCmOpenAIError(w, http.StatusBadRequest, err.Error(), "request")
		return
	}
	model, err := handler.resolver.ResolveModel(r.Context(), req.Model)
	if err != nil {
		writeROCmOpenAIError(w, http.StatusNotFound, err.Error(), "model")
		return
	}
	if req.Stream {
		serveROCmOpenAIResponseStream(w, r, model, req, messages, opts...)
		return
	}
	text := collectROCmWireTokenText(model.Chat(r.Context(), messages, opts...))
	if r := model.Err(); !r.OK {
		writeROCmOpenAIError(w, http.StatusInternalServerError, r.Value.(error).Error(), "model")
		return
	}
	writeROCmOpenAIJSON(w, http.StatusOK, openaicompat.NewTextResponse("resp_rocm", req.Model, text, model.Metrics()))
}

func serveROCmOpenAIResponseStream(w http.ResponseWriter, r *http.Request, model inference.TextModel, req openaicompat.ResponseRequest, messages []inference.Message, opts ...inference.GenerateOption) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)
	writeEvent := func(event openaicompat.ResponseStreamEvent) {
		writeROCmOpenAISSEData(w, core.JSONMarshalString(event))
		if flusher != nil {
			flusher.Flush()
		}
	}

	const id = "resp_rocm"
	writeEvent(openaicompat.ResponseStreamEvent{
		Type: "response.created",
		Response: &openaicompat.Response{
			ID:      id,
			Object:  "response",
			Created: time.Now().Unix(),
			Model:   req.Model,
		},
	})

	var text strings.Builder
	for token := range model.Chat(r.Context(), messages, opts...) {
		text.WriteString(token.Text)
		writeEvent(openaicompat.ResponseStreamEvent{Type: "response.output_text.delta", Delta: token.Text})
	}
	if r := model.Err(); !r.OK {
		writeEvent(openaicompat.ResponseStreamEvent{Type: "response.error", Delta: r.Value.(error).Error()})
		writeROCmOpenAISSEDone(w)
		if flusher != nil {
			flusher.Flush()
		}
		return
	}
	response := openaicompat.NewTextResponse(id, req.Model, text.String(), model.Metrics())
	writeEvent(openaicompat.ResponseStreamEvent{Type: "response.completed", Response: &response})
	writeROCmOpenAISSEDone(w)
	if flusher != nil {
		flusher.Flush()
	}
}

func writeROCmOpenAISSEData(w http.ResponseWriter, payload string) {
	_, _ = w.Write([]byte("data: "))
	_, _ = w.Write([]byte(payload))
	_, _ = w.Write([]byte("\n\n"))
}

func writeROCmOpenAISSEDone(w http.ResponseWriter) {
	_, _ = w.Write([]byte("data: [DONE]\n\n"))
}

func writeROCmOpenAIJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func writeROCmOpenAIError(w http.ResponseWriter, status int, message, param string) {
	writeROCmOpenAIJSON(w, status, map[string]any{
		"error": map[string]string{
			"message": message,
			"type":    "invalid_request_error",
			"param":   param,
		},
	})
}
