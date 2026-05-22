// SPDX-Licence-Identifier: EUPL-1.2

// Package openai adapts inference.TextModel implementations to the
// OpenAI-compatible chat completions wire format.
package openai

import (
	"context"
	"io"
	"net/http"
	"strconv"
	"sync"
	"time"
	"unicode"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

const DefaultChatCompletionsPath = "/v1/chat/completions"

const (
	DefaultTemperature = 1.0
	DefaultTopP        = 0.95
	DefaultTopK        = 64
	DefaultMaxTokens   = 2048
)

const channelMarker = "<|channel>"

// ChatCompletionRequest is the OpenAI-compatible request body.
type ChatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	Temperature *float32      `json:"temperature,omitempty"`
	TopP        *float32      `json:"top_p,omitempty"`
	TopK        *int          `json:"top_k,omitempty"`
	MaxTokens   *int          `json:"max_tokens,omitempty"`
	Stream      bool          `json:"stream,omitempty"`
	Stop        StopList      `json:"stop,omitempty"`
	User        string        `json:"user,omitempty"`
}

// StopList accepts OpenAI stop sequences as either a JSON string or string
// array.
type StopList []string

func (s *StopList) UnmarshalJSON(data []byte) error {
	// Hot path: this is called per OpenAI chat-completion request.
	// parseJSONStringList walks the variant string-or-array shape in
	// a single pass — drops the recursive core.JSONUnmarshal that
	// re-paid encoder-state + per-element string allocs on every
	// call. Same wire contract: null -> nil, "X" -> []string{"X"},
	// ["X","Y"] -> []string{"X","Y"}.
	values, err := parseJSONStringList(data)
	if err != nil {
		return err
	}
	*s = values
	return nil
}

// ChatMessage is a single chat turn.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatCompletionResponse is the non-streaming OpenAI-compatible response body.
type ChatCompletionResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []ChatChoice `json:"choices"`
	Usage   ChatUsage    `json:"usage"`
	Thought *string      `json:"thought,omitempty"`
}

type ChatChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

type ChatUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatCompletionChunk is one Server-Sent Event payload for streaming requests.
type ChatCompletionChunk struct {
	ID      string            `json:"id"`
	Object  string            `json:"object"`
	Created int64             `json:"created"`
	Model   string            `json:"model"`
	Choices []ChatChunkChoice `json:"choices"`
	Thought *string           `json:"thought,omitempty"`
}

type ChatChunkChoice struct {
	Index        int              `json:"index"`
	Delta        ChatMessageDelta `json:"delta"`
	FinishReason *string          `json:"finish_reason"`
}

type ChatMessageDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

// MarshalJSON hand-rolls the OpenAI ChatMessageDelta shape into a
// single caller-owned buffer. Fires per streamed SSE delta — the
// reflect path through encoding/json + the intermediate *string
// envelope structs together cost 4-5 allocs per call (encoder state,
// grow-doubled output, two pointer-string copies, JSONMarshalString
// AsString wrap). Hand-roll lands at 1 alloc for the typical
// content-only case and the role-priming case.
//
// Wire-compatible cases (matches the previous behaviour):
//   - Role == "" && Content == ""    -> {}
//   - Role set                       -> {"role":"X","content":"Y"}  (priming emits both)
//   - Content only                   -> {"content":"Y"}
//
// Empty case routes to the package-level emptyDeltaBytes — no alloc.
func (d ChatMessageDelta) MarshalJSON() ([]byte, error) {
	if d.Role == "" && d.Content == "" {
		return emptyDeltaBytes, nil
	}
	// Tight upper bound — both branches emit two ASCII keys plus the
	// quoted value bodies. Worst-case doubling on escape-heavy content
	// lets append grow once.
	size := 2 // braces
	if d.Role != "" {
		size += 9 + len(d.Role)      // "role":"...",
		size += 11 + len(d.Content)  // "content":"..."
	} else {
		size += 11 + len(d.Content) // "content":"..."
	}
	buf := make([]byte, 0, size)
	buf = append(buf, '{')
	if d.Role != "" {
		buf = appendStringField(buf, "role", d.Role, false)
		buf = appendStringField(buf, "content", d.Content, true)
	} else {
		buf = appendStringField(buf, "content", d.Content, false)
	}
	return append(buf, '}'), nil
}

// emptyDeltaBytes is the canonical "{}" slice returned for the
// no-fields case — shared across every priming/closing chunk that
// would otherwise allocate a fresh two-byte slice per call.
var emptyDeltaBytes = []byte("{}")

type ErrorResponse struct {
	Error ErrorObject `json:"error"`
}

type ErrorObject struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Param   string `json:"param,omitempty"`
	Code    string `json:"code"`
}

// DecodeRequest decodes an OpenAI-compatible chat completion request.
func DecodeRequest(body io.Reader) (ChatCompletionRequest, error) {
	if body == nil {
		return ChatCompletionRequest{}, core.E("openai.DecodeRequest", "request body is nil", nil)
	}
	data, err := io.ReadAll(body)
	if err != nil {
		return ChatCompletionRequest{}, core.E("openai.DecodeRequest", "read request body", err)
	}
	var req ChatCompletionRequest
	// Direct []byte path — skips the redundant []byte→string→[]byte
	// round-trip that JSONUnmarshalString(string(data), ...) would do.
	result := core.JSONUnmarshal(data, &req)
	if !result.OK {
		return ChatCompletionRequest{}, resultError(result)
	}
	return req, nil
}

// ValidateRequest validates the subset of the OpenAI request shape supported by
// this adapter.
func ValidateRequest(req ChatCompletionRequest) error {
	if core.Trim(req.Model) == "" {
		return requestError("model is required", "model")
	}
	if len(req.Messages) == 0 {
		return requestError("messages must be a non-empty array", "messages")
	}
	for i, msg := range req.Messages {
		role := core.Lower(core.Trim(msg.Role))
		switch role {
		case "system", "developer", "user", "assistant", "tool":
		default:
			return requestError(core.Sprintf("messages[%d].role must be system, developer, user, assistant, or tool", i), core.Sprintf("messages[%d].role", i))
		}
	}
	if req.Temperature != nil && (*req.Temperature < 0 || *req.Temperature > 2) {
		return requestError("temperature must be in [0, 2]", "temperature")
	}
	if req.TopP != nil && (*req.TopP < 0 || *req.TopP > 1) {
		return requestError("top_p must be in [0, 1]", "top_p")
	}
	if req.TopK != nil && *req.TopK < 0 {
		return requestError("top_k must be >= 0", "top_k")
	}
	if req.MaxTokens != nil && *req.MaxTokens < 0 {
		return requestError("max_tokens must be >= 0", "max_tokens")
	}
	return nil
}

// GenerateOptions converts request sampling fields into inference options.
func GenerateOptions(req ChatCompletionRequest) ([]inference.GenerateOption, error) {
	if err := ValidateRequest(req); err != nil {
		return nil, err
	}
	return []inference.GenerateOption{
		inference.WithTemperature(resolvedFloat(req.Temperature, DefaultTemperature)),
		inference.WithTopP(resolvedFloat(req.TopP, DefaultTopP)),
		inference.WithTopK(resolvedInt(req.TopK, DefaultTopK)),
		inference.WithMaxTokens(resolvedInt(req.MaxTokens, DefaultMaxTokens)),
	}, nil
}

func resolvedFloat(value *float32, fallback float32) float32 {
	if value == nil {
		return fallback
	}
	return *value
}

func resolvedInt(value *int, fallback int) int {
	if value == nil {
		return fallback
	}
	return *value
}

// NormalizeStopSequences trims and validates request stop strings.
func NormalizeStopSequences(stops StopList) ([]string, error) {
	if len(stops) == 0 {
		return nil, nil
	}
	out := make([]string, 0, len(stops))
	for _, stop := range stops {
		trimmed := core.Trim(stop)
		if trimmed == "" {
			return nil, requestError("stop sequences must not be empty", "stop")
		}
		out = append(out, trimmed)
	}
	return out, nil
}

// Resolver maps request model names to loaded inference models.
type Resolver interface {
	ResolveModel(ctx context.Context, name string) (inference.TextModel, error)
}

type ResolverFunc func(context.Context, string) (inference.TextModel, error)

func (fn ResolverFunc) ResolveModel(ctx context.Context, name string) (inference.TextModel, error) {
	if fn == nil {
		return nil, core.E("openai.ResolverFunc", "resolver is nil", nil)
	}
	return fn(ctx, name)
}

type StaticResolver struct {
	models map[string]inference.TextModel
}

func NewStaticResolver(models map[string]inference.TextModel) *StaticResolver {
	resolver := &StaticResolver{models: make(map[string]inference.TextModel, len(models))}
	for name, model := range models {
		resolver.models[core.Lower(core.Trim(name))] = model
	}
	return resolver
}

func (r *StaticResolver) ResolveModel(ctx context.Context, name string) (inference.TextModel, error) {
	if ctx != nil {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
	}
	if r == nil {
		return nil, core.E("openai.StaticResolver", "resolver is nil", nil)
	}
	model, ok := r.models[core.Lower(core.Trim(name))]
	if !ok || model == nil {
		return nil, core.E("openai.StaticResolver", core.Sprintf("model %q not found", name), nil)
	}
	return model, nil
}

// BackendResolver lazily loads one model through the inference backend registry.
type BackendResolver struct {
	BackendName string
	ModelPath   string
	LoadOptions []inference.LoadOption

	mu    sync.Mutex
	model inference.TextModel
}

func NewBackendResolver(backendName, modelPath string, opts ...inference.LoadOption) *BackendResolver {
	return &BackendResolver{
		BackendName: core.Trim(backendName),
		ModelPath:   core.Trim(modelPath),
		LoadOptions: append([]inference.LoadOption(nil), opts...),
	}
}

func (r *BackendResolver) ResolveModel(ctx context.Context, _ string) (inference.TextModel, error) {
	if ctx != nil {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
	}
	if r == nil {
		return nil, core.E("openai.BackendResolver", "resolver is nil", nil)
	}
	if r.ModelPath == "" {
		return nil, core.E("openai.BackendResolver", "model path is required", nil)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.model != nil {
		return r.model, nil
	}
	opts := append([]inference.LoadOption(nil), r.LoadOptions...)
	if r.BackendName != "" {
		opts = append(opts, inference.WithBackend(r.BackendName))
	}
	result := inference.LoadModel(r.ModelPath, opts...)
	if !result.OK {
		return nil, resultError(result)
	}
	model, ok := result.Value.(inference.TextModel)
	if !ok || model == nil {
		return nil, core.E("openai.BackendResolver", "loaded value is not an inference.TextModel", nil)
	}
	r.model = model
	return model, nil
}

// Handler serves OpenAI-compatible chat completion requests.
type Handler struct {
	resolver Resolver
}

func NewHandler(resolver Resolver) *Handler {
	return &Handler{resolver: resolver}
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if h == nil || h.resolver == nil {
		writeError(w, http.StatusServiceUnavailable, "chat handler is not configured", "model")
		return
	}
	if r == nil {
		writeError(w, http.StatusBadRequest, "request is nil", "request")
		return
	}
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		writeError(w, http.StatusMethodNotAllowed, "method not allowed", "method")
		return
	}
	req, err := DecodeRequest(r.Body)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body", "body")
		return
	}
	if err := ValidateRequest(req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), errorParam(err))
		return
	}
	stops, err := NormalizeStopSequences(req.Stop)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "stop")
		return
	}
	opts, err := GenerateOptions(ChatCompletionRequest{
		Model:       req.Model,
		Messages:    req.Messages,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		TopK:        req.TopK,
		MaxTokens:   req.MaxTokens,
	})
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), errorParam(err))
		return
	}
	model, err := h.resolver.ResolveModel(r.Context(), req.Model)
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error(), "model")
		return
	}
	messages := requestMessages(req.Messages)
	if req.Stream {
		h.serveStreaming(w, r, model, req, messages, stops, opts...)
		return
	}
	h.serveNonStreaming(w, r, model, req, messages, stops, opts...)
}

func (h *Handler) serveNonStreaming(w http.ResponseWriter, r *http.Request, model inference.TextModel, req ChatCompletionRequest, messages []inference.Message, stops []string, opts ...inference.GenerateOption) {
	created := time.Now().Unix()
	completionID := completionID()
	extractor := NewThinkingExtractor()
	for token := range model.Chat(r.Context(), messages, opts...) {
		extractor.Process(token)
	}
	visibleTail, thoughtTail := extractor.Flush()
	_ = visibleTail
	_ = thoughtTail
	if err := model.Err(); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error(), "model")
		return
	}
	metrics := model.Metrics()
	content := TruncateAtStopSequence(extractor.Content(), stops)
	finishReason := "stop"
	if isTokenLengthCapReached(req.MaxTokens, metrics.GeneratedTokens) {
		finishReason = "length"
	}
	response := ChatCompletionResponse{
		ID:      completionID,
		Object:  "chat.completion",
		Created: created,
		Model:   req.Model,
		Choices: []ChatChoice{{
			Index:        0,
			Message:      ChatMessage{Role: "assistant", Content: content},
			FinishReason: finishReason,
		}},
		Usage: ChatUsage{
			PromptTokens:     metrics.PromptTokens,
			CompletionTokens: metrics.GeneratedTokens,
			TotalTokens:      metrics.PromptTokens + metrics.GeneratedTokens,
		},
	}
	if thought := extractor.Thinking(); thought != "" {
		response.Thought = &thought
	}
	writeJSON(w, http.StatusOK, response)
}

func (h *Handler) serveStreaming(w http.ResponseWriter, r *http.Request, model inference.TextModel, req ChatCompletionRequest, messages []inference.Message, stops []string, opts ...inference.GenerateOption) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	created := time.Now().Unix()
	completionID := completionID()
	flusher, _ := w.(http.Flusher)
	writeChunk := func(chunk ChatCompletionChunk) {
		// Single-buffer SSE frame — the previous shape did
		// JSONMarshalString (reflect path + grow-doubled scratch
		// buffer) then Concat to wrap with "data: " / "\n\n" then
		// []byte conversion. appendChatCompletionChunkSSE walks the
		// chunk directly into a pre-sized buffer that already carries
		// the SSE framing.
		frame := appendChatCompletionChunkSSE(make([]byte, 0, chunkSSEFrameSize(chunk)), chunk)
		_, _ = w.Write(frame)
		if flusher != nil {
			flusher.Flush()
		}
	}
	writeChunk(ChatCompletionChunk{
		ID:      completionID,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   req.Model,
		Choices: []ChatChunkChoice{{
			Index: 0,
			Delta: ChatMessageDelta{Role: "assistant"},
		}},
	})

	extractor := NewThinkingExtractor()
	emittedContent := ""
	finishReason := "stop"
	for token := range model.Chat(r.Context(), messages, opts...) {
		contentDelta, thoughtDelta := extractor.Process(token)
		candidate := emittedContent + contentDelta
		stopCut, stopHit := firstStopSequenceCut(candidate, stops)
		if stopHit {
			if stopCut <= len(emittedContent) {
				contentDelta = ""
			} else {
				contentDelta = candidate[len(emittedContent):stopCut]
			}
		}
		if contentDelta != "" || thoughtDelta != "" {
			chunk := ChatCompletionChunk{
				ID:      completionID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   req.Model,
				Choices: []ChatChunkChoice{{
					Index: 0,
					Delta: ChatMessageDelta{Content: contentDelta},
				}},
			}
			if thoughtDelta != "" {
				chunk.Thought = &thoughtDelta
			}
			writeChunk(chunk)
		}
		if stopHit {
			emittedContent = candidate[:stopCut]
			break
		}
		emittedContent = candidate
	}
	if visibleTail, thoughtTail := extractor.Flush(); visibleTail != "" || thoughtTail != "" {
		chunk := ChatCompletionChunk{
			ID:      completionID,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   req.Model,
			Choices: []ChatChunkChoice{{
				Index: 0,
				Delta: ChatMessageDelta{Content: visibleTail},
			}},
		}
		if thoughtTail != "" {
			chunk.Thought = &thoughtTail
		}
		writeChunk(chunk)
	}
	if err := model.Err(); err != nil {
		finishReason = "error"
	}
	if finishReason != "error" && isTokenLengthCapReached(req.MaxTokens, model.Metrics().GeneratedTokens) {
		finishReason = "length"
	}
	writeChunk(ChatCompletionChunk{
		ID:      completionID,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   req.Model,
		Choices: []ChatChunkChoice{{
			Index:        0,
			Delta:        ChatMessageDelta{},
			FinishReason: &finishReason,
		}},
	})
	_, _ = w.Write([]byte("data: [DONE]\n\n"))
	if flusher != nil {
		flusher.Flush()
	}
}

func requestMessages(messages []ChatMessage) []inference.Message {
	out := make([]inference.Message, 0, len(messages))
	for _, msg := range messages {
		out = append(out, inference.Message{Role: msg.Role, Content: msg.Content})
	}
	return out
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	// Hand-rolled fast path for the canonical non-streaming
	// ChatCompletionResponse — fires once per served request and
	// previously paid 2 allocs / 432 B through the reflect path.
	// Encoding directly into a pre-sized buffer skips
	// JSONMarshalString + the []byte(string) conversion.
	if p, ok := payload.(ChatCompletionResponse); ok {
		buf := appendChatCompletionResponse(make([]byte, 0, chatCompletionResponseSize(p)), p)
		_, _ = w.Write(buf)
		return
	}
	if p, ok := payload.(EmbeddingResponse); ok {
		// Embedding responses scale with vector dimensionality —
		// a 20-input × 1024-dim response is ~190 KB. The reflect
		// path pays a per-element float32 marshal cost; the hand-
		// rolled walk emits directly via strconv.AppendFloat.
		buf := appendEmbeddingResponse(make([]byte, 0, embeddingResponseSize(p)), p)
		_, _ = w.Write(buf)
		return
	}
	if p, ok := payload.(Response); ok {
		// Responses API non-streaming body — fires per served
		// /v1/responses request. Same shape as ChatCompletionResponse
		// (id/object/created/model/output/usage/thought) but with
		// the Responses output-message envelope.
		buf := appendResponse(make([]byte, 0, responseSize(p)), p)
		_, _ = w.Write(buf)
		return
	}
	if p, ok := payload.(RerankResponse); ok {
		// Rerank results scale with the documents slice — walking
		// inference.RerankScore inline skips the per-element reflect
		// cost. Labels field is rarely set in practice; encoder
		// handles both shapes.
		buf := appendRerankResponse(make([]byte, 0, rerankResponseSize(p)), p)
		_, _ = w.Write(buf)
		return
	}
	result := core.JSONMarshal(payload)
	if !result.OK {
		_, _ = w.Write([]byte(`{}`))
		return
	}
	_, _ = w.Write(result.Value.([]byte))
}

func writeError(w http.ResponseWriter, status int, message, param string) {
	writeJSON(w, status, ErrorResponse{Error: ErrorObject{
		Message: message,
		Type:    "invalid_request_error",
		Param:   param,
		Code:    "invalid_request_error",
	}})
}

type requestValidationError struct {
	message string
	param   string
}

func (e *requestValidationError) Error() string {
	if e == nil {
		return ""
	}
	return e.message
}

func requestError(message, param string) error {
	return &requestValidationError{message: message, param: param}
}

func errorParam(err error) string {
	if validation, ok := err.(*requestValidationError); ok {
		return validation.param
	}
	return ""
}

func resultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return core.E("openai.result", "unexpected failed result value", nil)
}

func completionID() string {
	// Fires once per chat-completion response. core.Sprintf was 2 allocs
	// (fmt formatter scratch + result string); the append-into-prefix
	// path is a single alloc backing the returned string via AsString.
	buf := make([]byte, 0, 32) // "chatcmpl-" (9) + max int64 (20) + slack
	buf = append(buf, "chatcmpl-"...)
	buf = strconv.AppendInt(buf, time.Now().UnixNano(), 10)
	return core.AsString(buf)
}

func isTokenLengthCapReached(maxTokens *int, generated int) bool {
	return maxTokens != nil && *maxTokens > 0 && generated >= *maxTokens
}

// TruncateAtStopSequence removes the first matching stop sequence and anything
// after it.
func TruncateAtStopSequence(content string, stops []string) string {
	cut, ok := firstStopSequenceCut(content, stops)
	if !ok {
		return content
	}
	return content[:cut]
}

func firstStopSequenceCut(content string, stops []string) (int, bool) {
	if content == "" || len(stops) == 0 {
		return 0, false
	}
	best := -1
	for _, stop := range stops {
		idx := indexString(content, stop)
		if idx < 0 {
			continue
		}
		if best < 0 || idx < best {
			best = idx
		}
	}
	if best < 0 {
		return 0, false
	}
	return best, true
}

// indexString delegates to core.Index (strings.Index — Rabin-Karp +
// SIMD byte search). The earlier hand-rolled loop was O(N×M) per call
// and fired multiple times per chat-completion (stop-sequence cut +
// thinking-extractor per streaming chunk + channel-marker detection
// on every delta).
//
// Returns -1 on empty needle to preserve the caller contract — the
// stop-sequence + extractor paths treat empty as "no match" rather
// than the strings.Index "match at 0" semantics.
func indexString(s, needle string) int {
	if needle == "" {
		return -1
	}
	return core.Index(s, needle)
}

type pairedMarker struct {
	start string
	end   string
}

var reasoningMarkers = []pairedMarker{
	{start: "<think>", end: "</think>"},
	{start: "<thinking>", end: "</thinking>"},
	{start: "<thought>", end: "</thought>"},
	{start: "<reasoning>", end: "</reasoning>"},
}

// ThinkingExtractor separates model-internal reasoning text from assistant
// content.
type ThinkingExtractor struct {
	pending        string
	content        string
	thinking       string
	inPaired       bool
	pairedEnd      string
	currentChannel string
}

func NewThinkingExtractor() *ThinkingExtractor {
	return &ThinkingExtractor{currentChannel: "assistant"}
}

func (e *ThinkingExtractor) Process(token inference.Token) (contentDelta, thoughtDelta string) {
	if e == nil {
		return "", ""
	}
	e.pending += token.Text
	return e.drain(false)
}

func (e *ThinkingExtractor) Flush() (contentDelta, thoughtDelta string) {
	if e == nil {
		return "", ""
	}
	contentDelta, thoughtDelta = e.drain(true)
	if e.pending == "" {
		return contentDelta, thoughtDelta
	}
	if e.inPaired || e.currentChannel == "thought" || e.currentChannel == "thinking" || e.currentChannel == "reasoning" {
		thoughtDelta += e.pending
		e.thinking += e.pending
	} else {
		contentDelta += e.pending
		e.content += e.pending
	}
	e.pending = ""
	e.inPaired = false
	return contentDelta, thoughtDelta
}

func (e *ThinkingExtractor) Content() string {
	if e == nil {
		return ""
	}
	return e.content
}

func (e *ThinkingExtractor) Thinking() string {
	if e == nil {
		return ""
	}
	return e.thinking
}

func (e *ThinkingExtractor) drain(final bool) (string, string) {
	contentDelta := core.NewBuilder()
	thoughtDelta := core.NewBuilder()
	for e.pending != "" {
		if e.inPaired {
			idx := indexString(e.pending, e.pairedEnd)
			if idx >= 0 {
				writeThought(e, thoughtDelta, e.pending[:idx])
				e.pending = e.pending[idx+len(e.pairedEnd):]
				e.inPaired = false
				e.pairedEnd = ""
				continue
			}
			emit, keep := splitSafeSuffix(e.pending, []string{e.pairedEnd}, final)
			writeThought(e, thoughtDelta, emit)
			e.pending = keep
			if keep != "" && !final {
				break
			}
			continue
		}

		if ok := e.consumeMarkerAtStart(); ok {
			continue
		}

		if e.currentChannel == "thought" || e.currentChannel == "thinking" || e.currentChannel == "reasoning" {
			idx := indexString(e.pending, channelMarker)
			if idx >= 0 {
				writeThought(e, thoughtDelta, e.pending[:idx])
				e.pending = e.pending[idx:]
				if e.consumeMarkerAtStart() {
					continue
				}
				if !final {
					break
				}
				writeThought(e, thoughtDelta, channelMarker)
				e.pending = e.pending[len(channelMarker):]
				continue
			}
			emit, keep := splitSafeSuffix(e.pending, []string{channelMarker}, final)
			writeThought(e, thoughtDelta, emit)
			e.pending = keep
			if keep != "" && !final {
				break
			}
			continue
		}

		start, idx := earliestReasoningStart(e.pending)
		channelIdx := indexString(e.pending, channelMarker)
		if channelIdx >= 0 && (idx < 0 || channelIdx < idx) {
			idx = channelIdx
			start = channelMarker
		}
		if idx >= 0 {
			writeContent(e, contentDelta, e.pending[:idx])
			e.pending = e.pending[idx:]
			if start == channelMarker {
				if e.consumeMarkerAtStart() {
					continue
				}
				if !final {
					break
				}
				writeContent(e, contentDelta, channelMarker)
				e.pending = e.pending[len(channelMarker):]
				continue
			}
			e.inPaired = true
			e.pairedEnd = pairedEndFor(start)
			e.pending = e.pending[len(start):]
			continue
		}
		emit, keep := splitSafeSuffix(e.pending, markerStarts(), final)
		writeContent(e, contentDelta, emit)
		e.pending = keep
		if keep != "" && !final {
			break
		}
	}
	return contentDelta.String(), thoughtDelta.String()
}

func (e *ThinkingExtractor) consumeMarkerAtStart() bool {
	if !core.HasPrefix(e.pending, channelMarker) {
		for _, marker := range reasoningMarkers {
			if core.HasPrefix(e.pending, marker.start) {
				e.inPaired = true
				e.pairedEnd = marker.end
				e.pending = e.pending[len(marker.start):]
				return true
			}
		}
		return false
	}
	remaining := e.pending[len(channelMarker):]
	consumedSpace := 0
	for consumedSpace < len(remaining) {
		r, size := rune(remaining[consumedSpace]), 1
		if r >= 0x80 {
			r, size = utf8Rune(remaining[consumedSpace:])
		}
		if !unicode.IsSpace(r) {
			break
		}
		consumedSpace += size
	}
	nameLen := 0
	for consumedSpace+nameLen < len(remaining) {
		c := remaining[consumedSpace+nameLen]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' {
			nameLen++
			continue
		}
		break
	}
	if nameLen == 0 {
		return false
	}
	e.currentChannel = core.Lower(remaining[consumedSpace : consumedSpace+nameLen])
	e.pending = remaining[consumedSpace+nameLen:]
	return true
}

func utf8Rune(s string) (rune, int) {
	for _, r := range s {
		return r, len(string(r))
	}
	return 0, 0
}

func writeContent(e *ThinkingExtractor, builder interface{ WriteString(string) (int, error) }, text string) {
	if text == "" {
		return
	}
	builder.WriteString(text)
	e.content += text
}

func writeThought(e *ThinkingExtractor, builder interface{ WriteString(string) (int, error) }, text string) {
	if text == "" {
		return
	}
	builder.WriteString(text)
	e.thinking += text
}

func earliestReasoningStart(s string) (string, int) {
	best := -1
	bestStart := ""
	for _, marker := range reasoningMarkers {
		idx := indexString(s, marker.start)
		if idx < 0 {
			continue
		}
		if best < 0 || idx < best {
			best = idx
			bestStart = marker.start
		}
	}
	return bestStart, best
}

func pairedEndFor(start string) string {
	for _, marker := range reasoningMarkers {
		if marker.start == start {
			return marker.end
		}
	}
	return ""
}

func markerStarts() []string {
	out := make([]string, 0, len(reasoningMarkers)+1)
	out = append(out, channelMarker)
	for _, marker := range reasoningMarkers {
		out = append(out, marker.start)
	}
	return out
}

func splitSafeSuffix(s string, markers []string, final bool) (emit, keep string) {
	if final {
		return s, ""
	}
	keepLen := 0
	for _, marker := range markers {
		max := min(len(s), len(marker)-1)
		for n := 1; n <= max; n++ {
			if s[len(s)-n:] == marker[:n] && n > keepLen {
				keepLen = n
			}
		}
	}
	if keepLen == 0 {
		return s, ""
	}
	return s[:len(s)-keepLen], s[len(s)-keepLen:]
}
