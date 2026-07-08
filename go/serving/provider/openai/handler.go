// SPDX-Licence-Identifier: EUPL-1.2

// Handler serves the net/http chat-completions route (streaming and
// non-streaming) plus its JSON response and error helpers.
package openai

import (
	"net/http"
	"strconv"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/parser"
)

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
		// Surface the parse detail — multimodal content errors (bad data:
		// URL, oversized image, unsupported part type) are actionable for
		// the caller, and a local engine's JSON errors carry no secrets.
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error(), "body")
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
	opts, err := GenerateOptions(req)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), errorParam(err))
		return
	}
	model, err := h.resolver.ResolveModel(r.Context(), req.Model)
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error(), "model")
		return
	}
	messages := requestMessages(req.Messages, req.Tools)
	if messagesCarryImages(messages) {
		vision, ok := model.(inference.VisionModel)
		if !ok || !vision.AcceptsImages() {
			writeError(w, http.StatusBadRequest, "model does not accept image input", "messages")
			return
		}
	}
	if messagesCarryAudios(messages) {
		audio, ok := model.(inference.AudioModel)
		if !ok || !audio.AcceptsAudio() {
			writeError(w, http.StatusBadRequest, "model does not accept audio input", "messages")
			return
		}
	}
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
	if r := model.Err(); !r.OK {
		writeError(w, http.StatusInternalServerError, r.Error(), "model")
		return
	}
	metrics := model.Metrics()
	content := TruncateAtStopSequence(extractor.Content(), stops)
	finishReason := "stop"
	if isTokenLengthCapReached(req.MaxTokens, metrics.GeneratedTokens) {
		finishReason = "length"
	}
	// A turn that emitted <|tool_call> spans returns tool_calls +
	// finish_reason:"tool_calls" — the shape Codex/OpenAI clients read to run the
	// tools and reply with a role:"tool" message. Any leading prose stays as
	// content.
	message := ChatMessage{Role: "assistant", Content: content}
	if calls, clean := parser.ParseGemmaToolCalls(content); len(calls) > 0 {
		message.Content = core.Trim(clean)
		message.ToolCalls = make([]ToolCall, len(calls))
		for i, c := range calls {
			message.ToolCalls[i] = ToolCall{ID: toolCallID(), Type: "function", Function: ToolCallFunction{Name: c.Name, Arguments: c.ArgumentsJSON}}
		}
		finishReason = "tool_calls"
	}
	response := ChatCompletionResponse{
		ID:      completionID,
		Object:  "chat.completion",
		Created: created,
		Model:   req.Model,
		Choices: []ChatChoice{{
			Index:        0,
			Message:      message,
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
	// writeChunkReflect frames a chunk via the reflect encoder (which honours
	// ChatMessageDelta.MarshalJSON's tool_calls path). Only the rare tool_calls
	// chunks use it; the hot content path stays on the hand-rolled writeChunk.
	writeChunkReflect := func(chunk ChatCompletionChunk) {
		frame := append([]byte("data: "), core.AsBytes(core.JSONMarshalString(chunk))...)
		frame = append(frame, '\n', '\n')
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
	inTool := false
	finishReason := "stop"
	for token := range model.Chat(r.Context(), messages, opts...) {
		contentDelta, thoughtDelta := extractor.Process(token)
		candidate := emittedContent + contentDelta
		if inTool {
			emittedContent = candidate // buffering the tool-call span, lifted below
			continue
		}
		if idx := core.Index(candidate, parser.ToolCallOpenMarker); idx >= 0 {
			inTool = true
			if idx > len(emittedContent) {
				writeChunk(ChatCompletionChunk{
					ID:      completionID,
					Object:  "chat.completion.chunk",
					Created: created,
					Model:   req.Model,
					Choices: []ChatChunkChoice{{Index: 0, Delta: ChatMessageDelta{Content: candidate[len(emittedContent):idx]}}},
				})
			}
			emittedContent = candidate
			continue
		}
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
	visibleTail, thoughtTail := extractor.Flush()
	if visibleTail != "" {
		emittedContent += visibleTail
	}
	if !inTool && (visibleTail != "" || thoughtTail != "") {
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
	if r := model.Err(); !r.OK {
		finishReason = "error"
	}
	if finishReason != "error" && isTokenLengthCapReached(req.MaxTokens, model.Metrics().GeneratedTokens) {
		finishReason = "length"
	}
	// Lift any buffered <|tool_call> span into streamed tool_calls deltas — per
	// call an opening chunk (id + name) then an arguments chunk, each indexed so
	// the client assembles them. finish_reason flips to tool_calls. These rare
	// chunks go through the reflect writer (the hand-rolled path is text-only).
	if finishReason != "error" {
		if calls, _ := parser.ParseGemmaToolCalls(emittedContent); len(calls) > 0 {
			finishReason = "tool_calls"
			for i, c := range calls {
				id := toolCallID()
				writeChunkReflect(ChatCompletionChunk{
					ID:      completionID,
					Object:  "chat.completion.chunk",
					Created: created,
					Model:   req.Model,
					Choices: []ChatChunkChoice{{Index: 0, Delta: ChatMessageDelta{ToolCalls: []ToolCallDelta{{
						Index: i, ID: id, Type: "function", Function: &ToolCallFunctionDelta{Name: c.Name},
					}}}}},
				})
				writeChunkReflect(ChatCompletionChunk{
					ID:      completionID,
					Object:  "chat.completion.chunk",
					Created: created,
					Model:   req.Model,
					Choices: []ChatChunkChoice{{Index: 0, Delta: ChatMessageDelta{ToolCalls: []ToolCallDelta{{
						Index: i, Function: &ToolCallFunctionDelta{Arguments: c.ArgumentsJSON},
					}}}}},
				})
			}
		}
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

func requestMessages(messages []ChatMessage, tools []Tool) []inference.Message {
	out := make([]inference.Message, 0, len(messages)+1)
	if decl := renderOpenAITools(tools); decl != "" {
		out = append(out, inference.Message{Role: "system", Content: decl})
	}
	for _, msg := range messages {
		out = append(out, inference.Message{Role: msg.Role, Content: openAIMessageContent(msg), Images: msg.Images, Audios: msg.Audios})
	}
	return out
}

// renderOpenAITools converts OpenAI function declarations to the neutral shape
// and renders them into Gemma 4's tool syntax via the shared renderer.
func renderOpenAITools(tools []Tool) string {
	if len(tools) == 0 {
		return ""
	}
	decls := make([]parser.ToolDecl, len(tools))
	for i, t := range tools {
		props := make(map[string]parser.ToolParam, len(t.Function.Parameters.Properties))
		for name, p := range t.Function.Parameters.Properties {
			props[name] = parser.ToolParam{Type: p.Type, Description: p.Description}
		}
		decls[i] = parser.ToolDecl{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			Properties:  props,
			Required:    t.Function.Parameters.Required,
		}
	}
	return parser.RenderGemmaToolDeclarations(decls)
}

// openAIMessageContent renders one message's content. A role:"tool" message (a
// tool result) becomes a <|tool_response> span the model reads to continue — the
// model holds the original call in its retained KV state, so no history is
// re-rendered.
func openAIMessageContent(msg ChatMessage) string {
	if msg.Role == "tool" {
		return "<|tool_response>" + msg.Content + "<tool_response|>"
	}
	// A prior assistant turn that made tool calls: re-render each call into its
	// <|tool_call> span so a STATELESS client replaying full history keeps the
	// call context a following tool result answers. Under KV continuity the
	// client sends minimal history, so old turns don't reach here.
	if len(msg.ToolCalls) > 0 {
		b := core.NewBuilder()
		b.WriteString(msg.Content)
		for _, tc := range msg.ToolCalls {
			b.WriteString(parser.RenderGemmaToolCall(tc.Function.Name, tc.Function.Arguments))
		}
		return b.String()
	}
	return msg.Content
}

// chatResponseHasToolCalls reports whether the response carries tool_calls, so
// writeJSON routes it through reflect (the hand-rolled fast path models only the
// text shape; agentic responses are rare enough to afford reflect).
func chatResponseHasToolCalls(r ChatCompletionResponse) bool {
	return len(r.Choices) > 0 && len(r.Choices[0].Message.ToolCalls) > 0
}

// toolCallID mints an OpenAI-style tool-call id (call_<nanos>).
func toolCallID() string {
	buf := make([]byte, 0, 25) // "call_" (5) + max int64 (20)
	buf = append(buf, "call_"...)
	buf = strconv.AppendInt(buf, time.Now().UnixNano(), 10)
	return string(buf)
}

func messagesCarryImages(messages []inference.Message) bool {
	for i := range messages {
		if len(messages[i].Images) > 0 {
			return true
		}
	}
	return false
}

func messagesCarryAudios(messages []inference.Message) bool {
	for i := range messages {
		if len(messages[i].Audios) > 0 {
			return true
		}
	}
	return false
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	// Hand-rolled fast path for the canonical non-streaming
	// ChatCompletionResponse — fires once per served request and
	// previously paid 2 allocs / 432 B through the reflect path.
	// Encoding directly into a pre-sized buffer skips
	// JSONMarshalString + the []byte(string) conversion.
	if p, ok := payload.(ChatCompletionResponse); ok && !chatResponseHasToolCalls(p) {
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
