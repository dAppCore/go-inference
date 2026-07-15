// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"context"
	"encoding/json"
	"iter"
	"net/http"
	"path/filepath"
	"sort"
	"strings"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/serving/provider/anthropic"
	"dappco.re/go/inference/serving/provider/ollama"
	openaicompat "dappco.re/go/inference/serving/provider/openai"
)

// NewAnthropicMessagesHandler exposes Anthropic Messages over a caller-provided
// resolver, including the SSE streaming shape when request.stream is true.
func NewAnthropicMessagesHandler(resolver openaicompat.Resolver) http.Handler {
	return &anthropicMessagesHandler{resolver: resolver}
}

// NewOllamaHandler exposes Ollama chat and generate endpoints over a
// caller-provided resolver, including NDJSON streaming when request.stream is
// true.
func NewOllamaHandler(resolver openaicompat.Resolver) *http.ServeMux {
	mux := http.NewServeMux()
	handler := &ollamaCompatHandler{resolver: resolver}
	mux.Handle(ollama.DefaultChatPath, http.HandlerFunc(handler.chat))
	mux.Handle(ollama.DefaultGeneratePath, http.HandlerFunc(handler.generate))
	mux.Handle(ollama.DefaultTagsPath, http.HandlerFunc(handler.tags))
	mux.Handle(ollama.DefaultShowPath, http.HandlerFunc(handler.show))
	return mux
}

type anthropicMessagesHandler struct {
	resolver openaicompat.Resolver
}

func (handler *anthropicMessagesHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if handler == nil || handler.resolver == nil {
		writeROCmOpenAIError(w, http.StatusServiceUnavailable, "anthropic messages handler is not configured", "model")
		return
	}
	if !requireROCmWireMethod(w, r, http.MethodPost) {
		return
	}
	var req anthropic.MessageRequest
	if !decodeROCmWireRequest(w, r, &req) {
		return
	}
	if core.Trim(req.Model) == "" {
		writeROCmOpenAIError(w, http.StatusBadRequest, "model is required", "model")
		return
	}
	messages := anthropic.InferenceMessages(req)
	if !hasROCmWireMessages(messages) {
		writeROCmOpenAIError(w, http.StatusBadRequest, "messages or system are required", "messages")
		return
	}
	if req.Stream {
		handler.serveStreaming(w, r, req, messages)
		return
	}
	model, ok := resolveROCmWireModel(w, r, handler.resolver, req.Model)
	if !ok {
		return
	}
	text, ok := runROCmWireChat(w, r, model, messages, anthropic.GenerateOptions(req)...)
	if !ok {
		return
	}
	writeROCmOpenAIJSON(w, http.StatusOK, anthropic.NewTextResponse("msg_rocm", req.Model, text, model.Metrics()))
}

func (handler *anthropicMessagesHandler) serveStreaming(w http.ResponseWriter, r *http.Request, req anthropic.MessageRequest, messages []inference.Message) {
	model, ok := resolveROCmWireModel(w, r, handler.resolver, req.Model)
	if !ok {
		return
	}
	header := w.Header()
	header.Set("Content-Type", "text/event-stream")
	header.Set("Cache-Control", "no-cache")
	header.Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	metrics := model.Metrics()
	writeROCmAnthropicSSEEvent(w, "message_start", anthropic.AppendMessageStartEvent(nil, anthropic.MessageResponse{
		ID:      "msg_rocm",
		Type:    "message",
		Role:    "assistant",
		Model:   req.Model,
		Content: []anthropic.ContentBlock{},
		Usage:   anthropic.Usage{InputTokens: metrics.PromptTokens},
	}))
	writeROCmAnthropicSSEEvent(w, "content_block_start", anthropic.AppendContentBlockStartEvent(nil, 0))

	generated := 0
	eventBuf := make([]byte, 0, 256)
	for token := range model.Chat(r.Context(), messages, anthropic.GenerateOptions(req)...) {
		generated++
		eventBuf = anthropic.AppendContentBlockDeltaEvent(eventBuf[:0], 0, token.Text)
		writeROCmAnthropicSSEEvent(w, "content_block_delta", eventBuf)
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
	}
	if r := model.Err(); !r.OK {
		writeROCmAnthropicSSEEvent(w, "error", []byte(core.Sprintf(`{"type":"error","error":{"type":"api_error","message":%q}}`, r.Value.(error).Error())))
		return
	}
	if got := model.Metrics().GeneratedTokens; got > 0 {
		generated = got
	}
	writeROCmAnthropicSSEEvent(w, "content_block_stop", anthropic.AppendContentBlockStopEvent(nil, 0))
	writeROCmAnthropicSSEEvent(w, "message_delta", anthropic.AppendMessageDeltaEvent(nil, "end_turn", "", generated))
	writeROCmAnthropicSSEEvent(w, "message_stop", []byte(anthropic.MessageStopPayload))
}

func writeROCmAnthropicSSEEvent(w http.ResponseWriter, event string, payload []byte) {
	_, _ = w.Write(rocmAnthropicSSEEventPrefix)
	_, _ = w.Write(rocmAnthropicSSEEventBytes(event))
	_, _ = w.Write(rocmAnthropicSSEDataPrefix)
	_, _ = w.Write(payload)
	_, _ = w.Write(rocmAnthropicSSEFrameEnd)
}

func rocmAnthropicSSEEventBytes(event string) []byte {
	switch event {
	case "message_start":
		return rocmAnthropicSSEMessageStart
	case "content_block_start":
		return rocmAnthropicSSEContentBlockStart
	case "content_block_delta":
		return rocmAnthropicSSEContentBlockDelta
	case "content_block_stop":
		return rocmAnthropicSSEContentBlockStop
	case "message_delta":
		return rocmAnthropicSSEMessageDelta
	case "message_stop":
		return rocmAnthropicSSEMessageStop
	case "error":
		return rocmAnthropicSSEError
	default:
		return rocmAnthropicSSEFallbackEventBytes
	}
}

type ollamaCompatHandler struct {
	resolver openaicompat.Resolver
}

type ollamaModelNameResolver interface {
	OllamaModelNames(ctx context.Context) ([]string, error)
}

const rocmWireTokenTextInitialBytes = 4096

var (
	rocmAnthropicSSEEventPrefix = []byte("event: ")
	rocmAnthropicSSEDataPrefix  = []byte("\ndata: ")
	rocmAnthropicSSEFrameEnd    = []byte("\n\n")

	rocmAnthropicSSEMessageStart       = []byte("message_start")
	rocmAnthropicSSEContentBlockStart  = []byte("content_block_start")
	rocmAnthropicSSEContentBlockDelta  = []byte("content_block_delta")
	rocmAnthropicSSEContentBlockStop   = []byte("content_block_stop")
	rocmAnthropicSSEMessageDelta       = []byte("message_delta")
	rocmAnthropicSSEMessageStop        = []byte("message_stop")
	rocmAnthropicSSEError              = []byte("error")
	rocmAnthropicSSEFallbackEventBytes = []byte("message_delta")
)

func (handler *ollamaCompatHandler) chat(w http.ResponseWriter, r *http.Request) {
	if handler == nil || handler.resolver == nil {
		writeROCmOpenAIError(w, http.StatusServiceUnavailable, "ollama handler is not configured", "model")
		return
	}
	if !requireROCmWireMethod(w, r, http.MethodPost) {
		return
	}
	var req ollama.ChatRequest
	if !decodeROCmWireRequest(w, r, &req) {
		return
	}
	if core.Trim(req.Model) == "" {
		writeROCmOpenAIError(w, http.StatusBadRequest, "model is required", "model")
		return
	}
	messages := ollama.InferenceMessages(req.Messages)
	if !hasROCmWireMessages(messages) {
		writeROCmOpenAIError(w, http.StatusBadRequest, "messages are required", "messages")
		return
	}
	model, ok := resolveROCmWireModel(w, r, handler.resolver, req.Model)
	if !ok {
		return
	}
	if req.Stream {
		serveROCmOllamaChatStream(w, r, model, req, messages)
		return
	}
	text, ok := runROCmWireChat(w, r, model, messages, ollama.GenerateOptions(req.Options)...)
	if !ok {
		return
	}
	writeROCmOpenAIJSON(w, http.StatusOK, ollama.NewChatResponse(req.Model, text, model.Metrics()))
}

func (handler *ollamaCompatHandler) generate(w http.ResponseWriter, r *http.Request) {
	if handler == nil || handler.resolver == nil {
		writeROCmOpenAIError(w, http.StatusServiceUnavailable, "ollama handler is not configured", "model")
		return
	}
	if !requireROCmWireMethod(w, r, http.MethodPost) {
		return
	}
	var req ollama.GenerateRequest
	if !decodeROCmWireRequest(w, r, &req) {
		return
	}
	if core.Trim(req.Model) == "" {
		writeROCmOpenAIError(w, http.StatusBadRequest, "model is required", "model")
		return
	}
	if core.Trim(req.Prompt) == "" {
		writeROCmOpenAIError(w, http.StatusBadRequest, "prompt is required", "prompt")
		return
	}
	model, ok := resolveROCmWireModel(w, r, handler.resolver, req.Model)
	if !ok {
		return
	}
	if req.Stream {
		serveROCmOllamaGenerateStream(w, r, model, req)
		return
	}
	text := collectROCmWireTokenText(model.Generate(r.Context(), req.Prompt, ollama.GenerateOptions(req.Options)...))
	if r := model.Err(); !r.OK {
		writeROCmOpenAIError(w, http.StatusInternalServerError, r.Value.(error).Error(), "model")
		return
	}
	writeROCmOpenAIJSON(w, http.StatusOK, ollama.NewGenerateResponse(req.Model, text, model.Metrics()))
}

func serveROCmOllamaChatStream(w http.ResponseWriter, r *http.Request, model inference.TextModel, req ollama.ChatRequest, messages []inference.Message) {
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)
	for token := range model.Chat(r.Context(), messages, ollama.GenerateOptions(req.Options)...) {
		writeROCmOllamaNDJSON(w, ollama.ChatResponse{Model: req.Model, Message: ollama.Message{Role: "assistant", Content: token.Text}})
		if flusher != nil {
			flusher.Flush()
		}
	}
	writeROCmOllamaNDJSON(w, ollama.NewChatResponse(req.Model, "", model.Metrics()))
	if flusher != nil {
		flusher.Flush()
	}
}

func serveROCmOllamaGenerateStream(w http.ResponseWriter, r *http.Request, model inference.TextModel, req ollama.GenerateRequest) {
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)
	for token := range model.Generate(r.Context(), req.Prompt, ollama.GenerateOptions(req.Options)...) {
		writeROCmOllamaNDJSON(w, ollama.GenerateResponse{Model: req.Model, Response: token.Text})
		if flusher != nil {
			flusher.Flush()
		}
	}
	writeROCmOllamaNDJSON(w, ollama.NewGenerateResponse(req.Model, "", model.Metrics()))
	if flusher != nil {
		flusher.Flush()
	}
}

func writeROCmOllamaNDJSON(w http.ResponseWriter, payload any) {
	_, _ = w.Write([]byte(core.JSONMarshalString(payload)))
	_, _ = w.Write([]byte("\n"))
}

func (handler *ollamaCompatHandler) tags(w http.ResponseWriter, r *http.Request) {
	if handler == nil || handler.resolver == nil {
		writeROCmOpenAIError(w, http.StatusServiceUnavailable, "ollama handler is not configured", "model")
		return
	}
	if !requireROCmWireMethod(w, r, http.MethodGet) {
		return
	}
	tags, err := handler.ollamaModelTags(r.Context())
	if err != nil {
		writeROCmOpenAIError(w, http.StatusInternalServerError, err.Error(), "model")
		return
	}
	writeROCmOpenAIJSON(w, http.StatusOK, ollama.TagsResponse{Models: tags})
}

func (handler *ollamaCompatHandler) show(w http.ResponseWriter, r *http.Request) {
	if handler == nil || handler.resolver == nil {
		writeROCmOpenAIError(w, http.StatusServiceUnavailable, "ollama handler is not configured", "model")
		return
	}
	if !requireROCmWireMethod(w, r, http.MethodPost) {
		return
	}
	var req ollama.ShowRequest
	if !decodeROCmWireRequest(w, r, &req) {
		return
	}
	if core.Trim(req.Model) == "" {
		writeROCmOpenAIError(w, http.StatusBadRequest, "model is required", "model")
		return
	}
	model, ok := resolveROCmWireModel(w, r, handler.resolver, req.Model)
	if !ok {
		return
	}
	writeROCmOpenAIJSON(w, http.StatusOK, rocmOllamaShowResponse(req.Model, model))
}

func (handler *ollamaCompatHandler) ollamaModelTags(ctx context.Context) ([]ollama.ModelTag, error) {
	names := []string(nil)
	if named, ok := handler.resolver.(ollamaModelNameResolver); ok {
		resolved, err := named.OllamaModelNames(ctx)
		if err != nil {
			return nil, err
		}
		names = append(names, resolved...)
	} else if backend, ok := handler.resolver.(*openaicompat.BackendResolver); ok && backend != nil {
		names = append(names, rocmOllamaModelNameFromPath(backend.ModelPath))
	}
	names = compactSortedOllamaModelNames(names)
	tags := make([]ollama.ModelTag, 0, len(names))
	for _, name := range names {
		tags = append(tags, ollama.ModelTag{Name: name, Model: name})
	}
	return tags, nil
}

func decodeROCmWireRequest(w http.ResponseWriter, r *http.Request, into any) bool {
	if r == nil || r.Body == nil {
		writeROCmOpenAIError(w, http.StatusBadRequest, "request body is nil", "body")
		return false
	}
	if err := json.NewDecoder(r.Body).Decode(into); err != nil {
		writeROCmOpenAIError(w, http.StatusBadRequest, "invalid request body", "body")
		return false
	}
	return true
}

func requireROCmWireMethod(w http.ResponseWriter, r *http.Request, method string) bool {
	if r == nil {
		writeROCmOpenAIError(w, http.StatusBadRequest, "request is nil", "request")
		return false
	}
	if r.Method != method {
		w.Header().Set("Allow", method)
		writeROCmOpenAIError(w, http.StatusMethodNotAllowed, "method not allowed", "method")
		return false
	}
	return true
}

func resolveROCmWireModel(w http.ResponseWriter, r *http.Request, resolver openaicompat.Resolver, name string) (inference.TextModel, bool) {
	model, err := resolver.ResolveModel(r.Context(), name)
	if err != nil {
		writeROCmOpenAIError(w, http.StatusNotFound, err.Error(), "model")
		return nil, false
	}
	return model, true
}

func runROCmWireChat(w http.ResponseWriter, r *http.Request, model inference.TextModel, messages []inference.Message, opts ...inference.GenerateOption) (string, bool) {
	text := collectROCmWireTokenText(model.Chat(r.Context(), messages, opts...))
	if r := model.Err(); !r.OK {
		writeROCmOpenAIError(w, http.StatusInternalServerError, r.Value.(error).Error(), "model")
		return "", false
	}
	return text, true
}

func hasROCmWireMessages(messages []inference.Message) bool {
	for _, message := range messages {
		if core.Trim(message.Content) != "" {
			return true
		}
	}
	return false
}

func collectROCmWireTokenText(tokens iter.Seq[inference.Token]) string {
	var text strings.Builder
	text.Grow(rocmWireTokenTextInitialBytes)
	for token := range tokens {
		text.WriteString(token.Text)
	}
	return text.String()
}

func rocmOllamaShowResponse(name string, model inference.TextModel) ollama.ShowResponse {
	info := model.Info()
	identity := rocmOllamaShowModelIdentity(model, info)
	profile, hasProfile := ResolveROCmModelProfileForModel(model)
	details := map[string]string{
		"architecture": firstNonEmptyString(identity.Architecture, info.Architecture, model.ModelType()),
		"backend":      "rocm",
		"family":       firstNonEmptyString(profile.Family, model.ModelType(), info.Architecture),
	}
	if identity.Path != "" {
		details["model_path"] = identity.Path
	}
	if identity.VocabSize > 0 {
		details["vocab_size"] = core.Sprintf("%d", identity.VocabSize)
	}
	if identity.HiddenSize > 0 {
		details["hidden_size"] = core.Sprintf("%d", identity.HiddenSize)
	}
	if identity.NumLayers > 0 {
		details["num_layers"] = core.Sprintf("%d", identity.NumLayers)
	}
	if identity.ContextLength > 0 {
		details["context_length"] = core.Sprintf("%d", identity.ContextLength)
	}
	if identity.QuantBits > 0 {
		details["quantization"] = core.Sprintf("%d-bit", identity.QuantBits)
	}
	if identity.QuantType != "" {
		details["quant_type"] = identity.QuantType
	}
	if identity.QuantGroup > 0 {
		details["quant_group"] = core.Sprintf("%d", identity.QuantGroup)
	}
	for _, key := range []string{"gemma4_size", "gemma4_quant_mode", "gemma4_source_format", "gemma4_generate_status"} {
		if value := identity.Labels[key]; value != "" {
			details[key] = value
		}
	}
	if hasProfile && profile.Matched() {
		details["engine_profile"] = profile.Name
		if profile.Registry != "" {
			details["engine_registry"] = profile.Registry
		}
		if profile.Family != "" {
			details["engine_profile_family"] = profile.Family
		}
		if profile.Architecture != "" {
			details["engine_profile_architecture"] = profile.Architecture
		}
		if profile.LoadStatus.Status != "" {
			details["engine_load_status"] = string(profile.LoadStatus.Status)
		}
		if template := firstNonEmptyString(profile.EngineFeatures.ChatTemplateID, profile.TokenizerRoute.ChatTemplateID); template != "" {
			details["chat_template"] = template
		}
		if parser := profile.EngineFeatures.ReasoningParserID; parser != "" {
			details["reasoning_parser"] = parser
		}
		if parser := profile.EngineFeatures.ToolParserID; parser != "" {
			details["tool_parser"] = parser
		}
		if capabilities := profile.EngineFeatures.EnabledCapabilities(); len(capabilities) > 0 {
			details["engine_feature_capabilities"] = rocmCapabilityIDsCSV(capabilities)
		}
	}
	if capabilityReport, ok := inference.CapabilitiesOf(model); ok {
		if capabilities := capabilityReport.SupportedCapabilityIDs(); len(capabilities) > 0 {
			details["capabilities"] = rocmCapabilityIDsCSV(capabilities)
			details["capability_count"] = core.Sprintf("%d", len(capabilities))
		}
	}
	return ollama.ShowResponse{
		Modelfile:  "FROM " + core.Trim(name),
		Parameters: rocmOllamaShowParameters(identity),
		Details:    details,
	}
}

func rocmOllamaShowModelIdentity(model inference.TextModel, info inference.ModelInfo) inference.ModelIdentity {
	if reporter, ok := model.(ROCmModelIdentityReporter); ok {
		identity := reporter.ModelIdentity()
		if !rocmModelIdentityIsZero(identity) {
			return rocmGemma4ModelWithInferredPathQuant(rocmCloneModelIdentity(identity))
		}
	}
	if info.Architecture == "" {
		info.Architecture = model.ModelType()
	}
	return rocmGemma4ModelWithInferredPathQuant(inference.ModelIdentity{
		Architecture: normalizeROCmArchitecture(info.Architecture),
		VocabSize:    info.VocabSize,
		NumLayers:    info.NumLayers,
		HiddenSize:   info.HiddenSize,
		QuantBits:    info.QuantBits,
		QuantGroup:   info.QuantGroup,
	})
}

func rocmOllamaShowParameters(identity inference.ModelIdentity) string {
	var parameters []string
	if identity.QuantBits > 0 {
		parameters = append(parameters, "quant_bits "+core.Sprintf("%d", identity.QuantBits))
	}
	if identity.QuantType != "" {
		parameters = append(parameters, "quant_type "+identity.QuantType)
	}
	if identity.QuantGroup > 0 {
		parameters = append(parameters, "quant_group "+core.Sprintf("%d", identity.QuantGroup))
	}
	if identity.ContextLength > 0 {
		parameters = append(parameters, "context_length "+core.Sprintf("%d", identity.ContextLength))
	}
	return strings.Join(parameters, "\n")
}

func rocmOllamaModelNameFromPath(path string) string {
	name := core.Trim(filepath.Base(path))
	if name == "" || name == "." || name == string(filepath.Separator) {
		return "rocm"
	}
	ext := filepath.Ext(name)
	if ext != "" {
		name = strings.TrimSuffix(name, ext)
	}
	return firstNonEmptyString(name, "rocm")
}

func compactSortedOllamaModelNames(names []string) []string {
	if len(names) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(names))
	out := make([]string, 0, len(names))
	for _, name := range names {
		name = core.Trim(name)
		if name == "" {
			continue
		}
		key := core.Lower(name)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}
