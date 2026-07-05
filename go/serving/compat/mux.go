// SPDX-Licence-Identifier: EUPL-1.2

// Package compat is the assembled OpenAI / Anthropic / Ollama compatibility
// server: it mounts the per-protocol handlers from go-inference's
// provider/{openai,anthropic,ollama} packages over a single resolver so a host
// (cmd/lem serve, tests, an embedding app) gets one drop-in HTTP surface.
//
// It is the go-inference home of the assembled mux that lived in go-mlx's
// dappco.re/go/mlx/openai package — lifted here so the serving library owns the
// composed server and cmd/lem compiles from go-inference alone (no go-mlx
// import). The package depends only on go-inference provider packages, so it
// carries no engine coupling.
//
//	handler := compat.NewModelMux("/path/to/model", inference.WithContextLen(8192))
//	http.ListenAndServe(":8080", handler)
package compat

import (
	"context"
	"io"
	"net/http"
	"strconv"
	"time"
	"unicode/utf8"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/parser"
	anthropiccompat "dappco.re/go/inference/provider/anthropic"
	ollamacompat "dappco.re/go/inference/provider/ollama"
	openaicompat "dappco.re/go/inference/provider/openai"
)

// NewResolver returns a resolver that lazily loads modelPath through the
// native Metal backend registered by go-mlx.
//
//	resolver := openai.NewResolver(modelPath)
func NewResolver(modelPath string, opts ...inference.LoadOption) *openaicompat.BackendResolver {
	return openaicompat.NewBackendResolver("metal", modelPath, opts...)
}

// NewHandler exposes modelPath through the shared OpenAI-compatible chat
// completions handler.
//
//	handler := openai.NewHandler(modelPath)
func NewHandler(modelPath string, opts ...inference.LoadOption) http.Handler {
	return openaicompat.NewHandler(NewResolver(modelPath, opts...))
}

// NewModelMux exposes a local MLX model through the package-first
// OpenAI-compatible route set. It lazily loads modelPath through the registered
// native Metal inference backend.
//
//	handler := openai.NewModelMux(modelPath)
func NewModelMux(modelPath string, opts ...inference.LoadOption) http.Handler {
	return NewMux(NewResolver(modelPath, opts...))
}

// NewMux mounts the shared local-inference endpoints over resolver. The
// handler is deliberately package-first: callers can host it from core/api,
// go-ai, a standalone server, or tests without making go-mlx depend on any of
// those layers.
//
//	handler := openai.NewMux(resolver)
func NewMux(resolver openaicompat.Resolver) http.Handler {
	return NewMuxWithAdmin(resolver, AdminConfig{})
}

// NewMuxWithAdmin mounts the same compatibility routes as NewMux plus
// package-first admin callbacks supplied by the host application.
//
//	handler := openai.NewMuxWithAdmin(resolver, openai.AdminConfig{Health: hostHealth})
func NewMuxWithAdmin(resolver openaicompat.Resolver, admin AdminConfig) http.Handler {
	mux := http.NewServeMux()
	mux.Handle(openaicompat.DefaultChatCompletionsPath, openaicompat.NewHandler(resolver))
	mux.Handle(openaicompat.DefaultResponsesPath, newOpenAIResponsesHandler(resolver))
	mux.Handle(openaicompat.DefaultEmbeddingsPath, openaicompat.NewEmbeddingsHandler(resolver))
	mux.Handle(openaicompat.DefaultRerankPath, openaicompat.NewRerankHandler(resolver))
	mux.Handle(openaicompat.DefaultCapabilitiesPath, openaicompat.NewCapabilityHandler(resolver))
	mux.Handle(openaicompat.DefaultCacheStatsPath, openaicompat.NewCacheStatsHandler(resolver))
	mux.Handle(openaicompat.DefaultCacheWarmPath, openaicompat.NewCacheWarmHandler(resolver))
	mux.Handle(openaicompat.DefaultCacheClearPath, openaicompat.NewCacheClearHandler(resolver))
	mux.Handle(openaicompat.DefaultCancelPath, openaicompat.NewCancelHandler(resolver))
	mux.Handle(anthropiccompat.DefaultMessagesPath, newAnthropicMessagesHandler(resolver))
	mux.Handle(ollamacompat.DefaultChatPath, newOllamaChatHandler(resolver))
	mux.Handle(ollamacompat.DefaultGeneratePath, newOllamaGenerateHandler(resolver))
	mux.Handle(ollamacompat.DefaultTagsPath, newOllamaTagsHandler(resolver))
	mux.Handle(ollamacompat.DefaultShowPath, newOllamaShowHandler(resolver))
	mountAdminHandlers(mux, resolver, admin)
	return mux
}

type openAIResponsesHandler struct {
	resolver openaicompat.Resolver
}

func newOpenAIResponsesHandler(resolver openaicompat.Resolver) http.Handler {
	return &openAIResponsesHandler{resolver: resolver}
}

func (h *openAIResponsesHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if h == nil || h.resolver == nil {
		writeOpenAIError(w, http.StatusServiceUnavailable, "responses handler is not configured", "model")
		return
	}
	if r == nil {
		writeOpenAIError(w, http.StatusBadRequest, "request is nil", "request")
		return
	}
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		writeOpenAIError(w, http.StatusMethodNotAllowed, "method not allowed", "method")
		return
	}
	req, err := decodeOpenAIResponseRequest(r.Body, r.ContentLength)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err.Error(), "body")
		return
	}
	if core.Trim(req.Model) == "" {
		writeOpenAIError(w, http.StatusBadRequest, "model is required", "model")
		return
	}
	opts, err := openaicompat.ResponseGenerateOptions(req)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err.Error(), "request")
		return
	}
	stops, err := openaicompat.NormalizeStopSequences(req.Stop)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err.Error(), "stop")
		return
	}
	model, err := h.resolver.ResolveModel(r.Context(), req.Model)
	if err != nil {
		writeOpenAIError(w, http.StatusNotFound, err.Error(), "model")
		return
	}
	messages := openaicompat.ResponseMessages(req)
	if req.Stream {
		serveOpenAIResponseStream(w, r.Context(), model, req, messages, stops, opts...)
		return
	}
	serveOpenAIResponse(w, r.Context(), model, req, messages, stops, opts...)
}

func decodeOpenAIResponseRequest(body io.Reader, contentLength int64) (openaicompat.ResponseRequest, error) {
	var req openaicompat.ResponseRequest
	if err := decodeWireJSONSized(body, contentLength, &req, "mlx.openai.responses"); err != nil {
		return openaicompat.ResponseRequest{}, err
	}
	return req, nil
}

func serveOpenAIResponse(w http.ResponseWriter, ctx context.Context, model inference.TextModel, req openaicompat.ResponseRequest, messages []inference.Message, stops []string, opts ...inference.GenerateOption) {
	id := openAIResponseID()
	tokens, err := collectOpenAIResponseTokens(ctx, model, id, req.Model, messages, opts...)
	if err != nil {
		writeOpenAIError(w, http.StatusInternalServerError, err.Error(), "model")
		return
	}
	if result := model.Err(); !result.OK {
		writeOpenAIError(w, http.StatusInternalServerError, result.Error(), "model")
		return
	}
	visible, thought := parseOpenAIModelOutput(model, tokens, openAITokensText(tokens))
	response := openaicompat.NewTextResponse(id, req.Model, openaicompat.TruncateAtStopSequence(visible, stops), model.Metrics())
	if thought != "" {
		response.Thought = &thought
	}
	writeOpenAIJSON(w, http.StatusOK, response)
}

// SSE frame fragments — package-level []byte so the streaming hot path
// writes them by reference instead of allocating a fresh slice per token.
// SSE frames are ASCII-fixed: "data: " <payload> "\n\n" (OpenAI/Ollama)
// and "event: " <name> "\n" "data: " <payload> "\n\n" (Anthropic).
var (
	sseDataPrefix  = []byte("data: ")
	sseEventPrefix = []byte("event: ")
	sseLF          = []byte("\n")
	sseFrameEnd    = []byte("\n\n")
	sseDoneFrame   = []byte("data: [DONE]\n\n")
)

// Fixed bytes wrapping a "response.output_text.delta" SSE frame —
// "data: " <prefix> <escaped-delta> <suffix>. The Type field is the
// fixed literal "response.output_text.delta" (no escapable bytes), so
// the only variable part is the JSON-escaped delta string. Holding the
// invariant punctuation as package []byte lets serveOpenAIResponseStream
// build each per-token frame into one reused buffer with zero allocation,
// matching the chat-completions (appendChatCompletionChunkSSE) and
// Anthropic (writeSSEEventBytes) streaming paths. Byte-identical to
// `core.JSONMarshalString(ResponseStreamEvent{Type: …, Delta: d})` framed
// by writeSSEData — proven by the FuzzResponseDeltaFrame golden and the
// FuzzAppendJSONStringHTML escaper fuzz. Requires a non-empty delta: the
// Delta field is `omitempty`, so the marshalled event drops it when empty
// and this frame would diverge ("delta":"" vs absent).
var (
	sseResponseDeltaFramePrefix = []byte(`data: {"type":"response.output_text.delta","delta":`)
	sseResponseDeltaFrameSuffix = []byte("}\n\n")
)

// Fixed bytes wrapping the Ollama per-token NDJSON frames. Ollama streams
// one ChatResponse / GenerateResponse JSON object per generated token; on
// the per-token path every count/duration field is zero (omitempty drops
// them) and "done" is always false, so the only variable parts are the
// model name and the content/response text — both JSON-escaped. Holding the
// invariant punctuation as package []byte lets serveOllamaStream build each
// frame into one reused buffer with zero per-token allocation, matching the
// Responses (writeResponseDeltaFrame) and chat-completions paths. The
// builders below are byte-identical to
// `core.JSONMarshalString(ollamacompat.ChatResponse{…})` / GenerateResponse
// terminated with "\n" — proven by FuzzOllamaChatFrame / FuzzOllamaGenerateFrame.
// The "role" is fixed "assistant" on the per-token path (loop + flush tail),
// so it lives in the fixed fragment.
var (
	ndjsonChatFrameModelPrefix   = []byte(`{"model":`)
	ndjsonChatFrameMessagePrefix = []byte(`,"message":{"role":"assistant","content":`)
	ndjsonChatFrameSuffix        = []byte("},\"done\":false}\n")
	ndjsonGenFrameModelPrefix    = []byte(`{"model":`)
	ndjsonGenFrameResponsePrefix = []byte(`,"response":`)
	ndjsonGenFrameSuffix         = []byte(",\"done\":false}\n")
)

// writeSSEData writes one "data: <payload>\n\n" SSE frame to w. payload is
// viewed zero-copy via core.AsBytes — w.Write does not retain its argument,
// so the only allocation the caller incurs is building payload itself. This
// replaces []byte(core.Concat("data: ", payload, "\n\n")), which cost two
// extra allocations every call (the concat result + the []byte conversion).
// Fires per delta token on the streaming path; net.http buffers the three
// small writes, so the wire output is identical to the single-write form.
func writeSSEData(w io.Writer, payload string) {
	_, _ = w.Write(sseDataPrefix)
	_, _ = w.Write(core.AsBytes(payload))
	_, _ = w.Write(sseFrameEnd)
}

// writeSSEEvent writes one "event: <name>\ndata: <payload>\n\n" SSE frame
// (the Anthropic streaming shape). Same zero-copy rationale as writeSSEData.
func writeSSEEvent(w io.Writer, name, payload string) {
	_, _ = w.Write(sseEventPrefix)
	_, _ = w.Write(core.AsBytes(name))
	_, _ = w.Write(sseLF)
	_, _ = w.Write(sseDataPrefix)
	_, _ = w.Write(core.AsBytes(payload))
	_, _ = w.Write(sseFrameEnd)
}

// writeSSEEventBytes is writeSSEEvent for a payload already held as bytes —
// it writes the slice directly instead of round-tripping it through a string.
// The Anthropic Append*Event builders hand back a fresh []byte; wrapping that
// in string(...) copied it, only for writeSSEEvent to view it back as bytes.
// Writing the slice as-is skips that per-event copy and lets the caller reuse
// one scratch buffer across tokens. w.Write does not retain its argument.
func writeSSEEventBytes(w io.Writer, name string, payload []byte) {
	_, _ = w.Write(sseEventPrefix)
	_, _ = w.Write(core.AsBytes(name))
	_, _ = w.Write(sseLF)
	_, _ = w.Write(sseDataPrefix)
	_, _ = w.Write(payload)
	_, _ = w.Write(sseFrameEnd)
}

// writeNDJSONLine writes one newline-delimited-JSON record ("<payload>\n") —
// the Ollama streaming wire shape. Same zero-copy rationale as writeSSEData:
// payload is viewed via core.AsBytes and the terminator reuses the package
// sseLF slice, so the only allocation is building payload. Replaces
// []byte(core.Concat(payload, "\n")), which cost two extra allocations per
// delta token (the concat result + the []byte conversion).
func writeNDJSONLine(w io.Writer, payload string) {
	_, _ = w.Write(core.AsBytes(payload))
	_, _ = w.Write(sseLF)
}

// writeResponseDeltaFrame writes one "response.output_text.delta" SSE frame
// for delta into w, building it in buf (reused across tokens, returned grown).
// This is the /v1/responses streaming hot path: it fires per generated text
// delta. The previous shape marshalled a fresh ResponseStreamEvent struct
// through core.JSONMarshalString every token — the encoding/json reflect path
// (boxing the struct into `any` + a grow-doubled scratch buffer + the result
// copy) cost two allocations and ~120 B per token. Walking the fixed frame
// punctuation plus the escaped delta into a single reused buffer drops that to
// zero per-token allocations. delta must be non-empty (the caller guards on
// processor output) — see sseResponseDeltaFramePrefix for the omitempty
// equivalence note. w.Write does not retain buf.
func writeResponseDeltaFrame(w io.Writer, buf []byte, delta string) []byte {
	buf = buf[:0]
	buf = append(buf, sseResponseDeltaFramePrefix...)
	buf = appendJSONStringHTML(buf, delta)
	buf = append(buf, sseResponseDeltaFrameSuffix...)
	_, _ = w.Write(buf)
	return buf
}

// writeOllamaChatFrame writes one /api/chat NDJSON token frame for content
// (model fixed for the stream) into w, building it in buf (reused across
// tokens, returned grown). The previous shape marshalled a fresh
// ollamacompat.ChatResponse through core.JSONMarshalString every token — the
// reflect path's box + scratch buffer + result copy cost two allocations per
// token. Walking the fixed punctuation plus the escaped model/content into a
// reused buffer drops that to zero per-token allocations. content must be
// non-empty (the caller guards on processor output) — Message.Content has no
// omitempty so an empty content would still emit "content":"" and stay
// byte-identical, but the guard means it never arises. w.Write does not retain
// buf.
func writeOllamaChatFrame(w io.Writer, buf []byte, model, content string) []byte {
	buf = buf[:0]
	buf = append(buf, ndjsonChatFrameModelPrefix...)
	buf = appendJSONStringHTML(buf, model)
	buf = append(buf, ndjsonChatFrameMessagePrefix...)
	buf = appendJSONStringHTML(buf, content)
	buf = append(buf, ndjsonChatFrameSuffix...)
	_, _ = w.Write(buf)
	return buf
}

// writeOllamaGenerateFrame is writeOllamaChatFrame for the /api/generate
// shape — {"model":M,"response":C,"done":false} — same reused-buffer,
// zero-per-token-alloc rationale.
func writeOllamaGenerateFrame(w io.Writer, buf []byte, model, response string) []byte {
	buf = buf[:0]
	buf = append(buf, ndjsonGenFrameModelPrefix...)
	buf = appendJSONStringHTML(buf, model)
	buf = append(buf, ndjsonGenFrameResponsePrefix...)
	buf = appendJSONStringHTML(buf, response)
	buf = append(buf, ndjsonGenFrameSuffix...)
	_, _ = w.Write(buf)
	return buf
}

// appendJSONStringHTML appends s to buf as a JSON string literal — opening
// quote, escaped body, closing quote — byte-identical to encoding/json's
// default Marshal of a Go string. That means the same HTML-safety escaping
// (`<` `>` `&` → < > &), the same control-byte handling (the
// \b \f \n \r \t mnemonics, \u00XX for the rest), the same   /   for
// the Unicode line/paragraph separators, and � for invalid UTF-8. It
// exists so the streaming wire encoders can stay off the reflect path while
// keeping the exact bytes encoding/json would have produced (a streamed delta
// routinely carries `<`, `>` and `&` — code, comparisons, markup — so matching
// the HTML escaping is contract, not cosmetic). jsonenc.AppendJSONString is
// deliberately *not* used here: it omits the `<` `>` `&` escaping by design.
//
// Fast path: scan for the first byte that needs escaping and bulk-copy the safe
// run, so the common all-safe delta is a single append. Equivalence is locked
// by FuzzAppendJSONStringHTML against core.JSONMarshalString.
func appendJSONStringHTML(buf []byte, s string) []byte {
	buf = append(buf, '"')
	start := 0
	for i := 0; i < len(s); {
		if b := s[i]; b < utf8.RuneSelf {
			if jsonHTMLSafe(b) {
				i++
				continue
			}
			if start < i {
				buf = append(buf, s[start:i]...)
			}
			switch b {
			case '\\', '"':
				buf = append(buf, '\\', b)
			case '\n':
				buf = append(buf, '\\', 'n')
			case '\r':
				buf = append(buf, '\\', 'r')
			case '\t':
				buf = append(buf, '\\', 't')
			case '\b':
				buf = append(buf, '\\', 'b')
			case '\f':
				buf = append(buf, '\\', 'f')
			default:
				buf = append(buf, '\\', 'u', '0', '0', hexNibble(b>>4), hexNibble(b&0xF))
			}
			i++
			start = i
			continue
		}
		c, size := utf8.DecodeRuneInString(s[i:])
		if c == utf8.RuneError && size == 1 {
			// Invalid UTF-8 byte — encoding/json emits the escaped
			// replacement char, not the raw � rune bytes.
			if start < i {
				buf = append(buf, s[start:i]...)
			}
			buf = append(buf, '\\', 'u', 'f', 'f', 'f', 'd')
			i += size
			start = i
			continue
		}
		// U+2028 LINE SEPARATOR / U+2029 PARAGRAPH SEPARATOR are valid JSON
		// but break embedded-in-HTML <script> parsing, so encoding/json's
		// HTML-safe default escapes them.
		if c == ' ' || c == ' ' {
			if start < i {
				buf = append(buf, s[start:i]...)
			}
			buf = append(buf, '\\', 'u', '2', '0', '2', hexNibble(byte(c&0xF)))
			i += size
			start = i
			continue
		}
		i += size
	}
	if start < len(s) {
		buf = append(buf, s[start:]...)
	}
	return append(buf, '"')
}

// jsonHTMLSafe reports whether ASCII byte b passes through a JSON string body
// unescaped under encoding/json's HTML-safe default: printable, and not one of
// the quote / backslash / HTML-meta (`<` `>` `&`) bytes.
func jsonHTMLSafe(b byte) bool {
	if b < 0x20 {
		return false
	}
	switch b {
	case '"', '\\', '<', '>', '&':
		return false
	}
	return true
}

// hexNibble returns the lowercase ASCII hex digit for the low nibble of v —
// the \u00XX / \u202X escape branches of appendJSONStringHTML.
func hexNibble(v byte) byte {
	const hex = "0123456789abcdef"
	return hex[v&0xF]
}

func serveOpenAIResponseStream(w http.ResponseWriter, ctx context.Context, model inference.TextModel, req openaicompat.ResponseRequest, messages []inference.Message, stops []string, opts ...inference.GenerateOption) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)
	// writeEvent serves the once-per-request events (created / completed /
	// error). Those embed a full Response (or carry an error string) and fire
	// at most a few times per stream, so the reflect marshal is not a hot-path
	// cost and not worth a hand-rolled encoder. The per-token text deltas —
	// which is where the allocations multiply — go through writeDelta below.
	writeEvent := func(event openaicompat.ResponseStreamEvent) {
		writeSSEData(w, core.JSONMarshalString(event))
		if flusher != nil {
			flusher.Flush()
		}
	}
	// deltaBuf is the reused per-token frame buffer — writeResponseDeltaFrame
	// rebuilds each "response.output_text.delta" frame into it, so the whole
	// stream's worth of deltas costs one amortised buffer grow rather than a
	// marshal + box per token.
	var deltaBuf []byte
	writeDelta := func(delta string) {
		deltaBuf = writeResponseDeltaFrame(w, deltaBuf, delta)
		if flusher != nil {
			flusher.Flush()
		}
	}

	id := openAIResponseID()
	writeEvent(openaicompat.ResponseStreamEvent{
		Type: "response.created",
		Response: &openaicompat.Response{
			ID:      id,
			Object:  "response",
			Created: time.Now().Unix(),
			Model:   req.Model,
		},
	})

	processor := parser.NewProcessor(parser.Config{Mode: parser.Capture}, parser.HintFromInference(model.Info()))
	tokens := []inference.Token{}
	// No parallel `raw` accumulator: every token is already retained in `tokens`,
	// so the full raw text is openAITokensText(tokens) — one Grow-sized allocation
	// at end-of-stream — rather than a strings.Builder that grow-doubled per token.
	// Byte-identical: both walk token.Text in append order over the same token set
	// (the slice append below precedes the empty-delta skip, exactly as the old
	// raw.WriteString did).
	visibleBuilder := core.NewBuilder()
	err := forEachOpenAIResponseToken(ctx, model, id, req.Model, messages, opts, func(token inference.Token) bool {
		tokens = append(tokens, token)
		contentDelta := processor.Process(token.Text)
		// The empty-delta skip is also load-bearing for wire equivalence:
		// ResponseStreamEvent.Delta is `omitempty`, so writeDelta's frame
		// (which always emits "delta":<value>) only matches the marshalled
		// event when the delta is non-empty.
		if contentDelta == "" {
			return true
		}
		visibleBuilder.WriteString(contentDelta)
		writeDelta(contentDelta)
		return true
	})
	if contentTail := processor.Flush(); contentTail != "" {
		visibleBuilder.WriteString(contentTail)
		writeDelta(contentTail)
	}

	if err != nil {
		writeEvent(openaicompat.ResponseStreamEvent{Type: "response.error", Delta: err.Error()})
		_, _ = w.Write(sseDoneFrame)
		if flusher != nil {
			flusher.Flush()
		}
		return
	}
	visible, thought := parseOpenAIModelOutput(model, tokens, openAITokensText(tokens))
	if visible == "" && visibleBuilder.String() != "" {
		visible = visibleBuilder.String()
	}
	response := openaicompat.NewTextResponse(id, req.Model, openaicompat.TruncateAtStopSequence(visible, stops), model.Metrics())
	if thought == "" {
		thought = processor.Reasoning()
	}
	if thought != "" {
		response.Thought = &thought
	}
	writeEvent(openaicompat.ResponseStreamEvent{Type: "response.completed", Response: &response})
	_, _ = w.Write(sseDoneFrame)
	if flusher != nil {
		flusher.Flush()
	}
}

func writeOpenAIJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	// AsBytes views the freshly-marshalled JSON string zero-copy — w.Write
	// does not retain it, so this drops the []byte conversion alloc the
	// explicit []byte(...) form paid on every non-streaming response.
	_, _ = w.Write(core.AsBytes(core.JSONMarshalString(payload)))
}

func writeOpenAIError(w http.ResponseWriter, status int, message, param string) {
	writeOpenAIJSON(w, status, openaicompat.ErrorResponse{Error: openaicompat.ErrorObject{
		Message: message,
		Type:    "invalid_request_error",
		Param:   param,
		Code:    "invalid_request_error",
	}})
}

func openAIResponseID() string {
	return idWithPrefix("resp_")
}

// idWithPrefix builds "<prefix><nanos>" in a single allocation. The
// previous prefix + core.FormatInt(...) form allocated twice — once for
// FormatInt's result string and once for the concatenation. strconv
// appends the decimal digits straight into a prefix-seeded buffer sized
// for the longest int64 (19 digits), so the request-ID helpers (one per
// request on every wire protocol) drop from two allocs to one. AsString
// views the single-owner buffer without a further copy.
func idWithPrefix(prefix string) string {
	buf := make([]byte, 0, len(prefix)+20)
	buf = append(buf, prefix...)
	buf = strconv.AppendInt(buf, time.Now().UnixNano(), 10)
	return core.AsString(buf)
}

func collectOpenAIResponseTokens(ctx context.Context, model inference.TextModel, requestID, modelName string, messages []inference.Message, opts ...inference.GenerateOption) ([]inference.Token, error) {
	return collectCompatTokens(ctx, model, requestID, modelName, "", messages, opts...)
}

func collectCompatTokens(ctx context.Context, model inference.TextModel, requestID, modelName, prompt string, messages []inference.Message, opts ...inference.GenerateOption) ([]inference.Token, error) {
	tokens := []inference.Token{}
	err := forEachCompatToken(ctx, model, requestID, modelName, prompt, messages, opts, func(token inference.Token) bool {
		tokens = append(tokens, token)
		return true
	})
	return tokens, err
}

func forEachOpenAIResponseToken(ctx context.Context, model inference.TextModel, requestID, modelName string, messages []inference.Message, opts []inference.GenerateOption, yield func(inference.Token) bool) error {
	return forEachCompatToken(ctx, model, requestID, modelName, "", messages, opts, yield)
}

func forEachCompatToken(ctx context.Context, model inference.TextModel, requestID, modelName, prompt string, messages []inference.Message, opts []inference.GenerateOption, yield func(inference.Token) bool) error {
	if scheduler, ok := model.(inference.SchedulerModel); ok {
		handle, stream, err := scheduler.Schedule(ctx, inference.ScheduledRequest{
			ID:       requestID,
			Model:    modelName,
			Prompt:   prompt,
			Messages: append([]inference.Message(nil), messages...),
			Sampler:  inference.SamplerConfigFromGenerateConfig(inference.ApplyGenerateOpts(opts)),
		})
		if err != nil {
			return err
		}
		for scheduled := range stream {
			if !yield(scheduled.Token) {
				if cancellable, ok := model.(inference.CancellableModel); ok {
					_, _ = cancellable.CancelRequest(ctx, handle.ID)
				}
				return nil
			}
		}
		return nil
	}
	var stream func(func(inference.Token) bool)
	if len(messages) > 0 {
		stream = model.Chat(ctx, messages, opts...)
	} else {
		stream = model.Generate(ctx, prompt, opts...)
	}
	for token := range stream {
		if !yield(token) {
			return nil
		}
	}
	return nil
}

type anthropicMessagesHandler struct {
	resolver openaicompat.Resolver
}

func newAnthropicMessagesHandler(resolver openaicompat.Resolver) http.Handler {
	return &anthropicMessagesHandler{resolver: resolver}
}

func (h *anthropicMessagesHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if h == nil || h.resolver == nil {
		writeOpenAIError(w, http.StatusServiceUnavailable, "anthropic messages handler is not configured", "model")
		return
	}
	if r == nil {
		writeOpenAIError(w, http.StatusBadRequest, "request is nil", "request")
		return
	}
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		writeOpenAIError(w, http.StatusMethodNotAllowed, "method not allowed", "method")
		return
	}
	var req anthropiccompat.MessageRequest
	if err := decodeWireJSONSized(r.Body, r.ContentLength, &req, "mlx.anthropic.messages"); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err.Error(), "body")
		return
	}
	if core.Trim(req.Model) == "" {
		writeOpenAIError(w, http.StatusBadRequest, "model is required", "model")
		return
	}
	stops, err := normalizeAnthropicStopSequences(req.StopSequences)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err.Error(), "stop_sequences")
		return
	}
	model, err := h.resolver.ResolveModel(r.Context(), req.Model)
	if err != nil {
		writeOpenAIError(w, http.StatusNotFound, err.Error(), "model")
		return
	}
	messages := anthropiccompat.InferenceMessages(req)
	opts := anthropiccompat.GenerateOptions(req)
	if req.Stream {
		serveAnthropicMessageStream(w, r.Context(), model, req, messages, stops, opts...)
		return
	}
	tokens, err := collectCompatTokens(r.Context(), model, anthropicMessageID(), req.Model, "", messages, opts...)
	if err != nil {
		writeOpenAIError(w, http.StatusInternalServerError, err.Error(), "model")
		return
	}
	if result := model.Err(); !result.OK {
		writeOpenAIError(w, http.StatusInternalServerError, result.Error(), "model")
		return
	}
	visible, _ := parseOpenAIModelOutput(model, tokens, openAITokensText(tokens))
	response := anthropiccompat.NewTextResponse(anthropicMessageID(), req.Model, openaicompat.TruncateAtStopSequence(visible, stops), model.Metrics())
	writeOpenAIJSON(w, http.StatusOK, response)
}

func serveAnthropicMessageStream(w http.ResponseWriter, ctx context.Context, model inference.TextModel, req anthropiccompat.MessageRequest, messages []inference.Message, stops []string, opts ...inference.GenerateOption) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)
	messageID := anthropicMessageID()
	writeEvent := func(event, payload string) {
		writeSSEEvent(w, event, payload)
		if flusher != nil {
			flusher.Flush()
		}
	}
	// Every event payload is built into one reused scratch buffer and written
	// as bytes, so the only per-request payload allocation is this buffer's
	// first grow — amortised across the whole stream. The terminal events
	// (message_start/content_block_start/stop/message_delta) and the per-token
	// content_block_delta never overlap, so they share one buffer safely.
	// The previous string(AppendXxxEvent(nil, …)) form paid two allocations
	// per event — a fresh buffer inside Append plus the string(...) copy —
	// which for the four fixed terminal events alone cost 24 allocs/1536 B per
	// request; AppendContentBlockDeltaEvent(nil, …) added one per token on top.
	var eventBuf []byte
	writeEventBytes := func(event string, build func([]byte) []byte) {
		eventBuf = build(eventBuf[:0])
		writeSSEEventBytes(w, event, eventBuf)
		if flusher != nil {
			flusher.Flush()
		}
	}
	writeDelta := func(text string) {
		writeEventBytes("content_block_delta", func(b []byte) []byte {
			return anthropiccompat.AppendContentBlockDeltaEvent(b, 0, text)
		})
	}
	// Full Anthropic streaming sequence — Claude Code's parser requires it:
	// message_start (wrapped) → content_block_start → content_block_delta* →
	// content_block_stop → message_delta (usage) → message_stop. Text block is
	// index 0; input_tokens is unknown until generation finishes, so
	// message_start opens at 0 and the cumulative output lands in message_delta.
	writeEventBytes("message_start", func(b []byte) []byte {
		return anthropiccompat.AppendMessageStartEvent(b, anthropiccompat.MessageResponse{ID: messageID, Type: "message", Role: "assistant", Model: req.Model})
	})
	writeEventBytes("content_block_start", func(b []byte) []byte {
		return anthropiccompat.AppendContentBlockStartEvent(b, 0)
	})
	processor := parser.NewProcessor(parser.Config{Mode: parser.Capture}, parser.HintFromInference(model.Info()))
	stopReason := "end_turn"
	// emittedBuf accumulates the cumulative output only so the stop-sequence
	// scan can see across the token boundary (a stop string split between two
	// tokens). When the request carries no stop sequences (the common case)
	// that accumulation is dead work, so we skip it and emit each delta as-is.
	//
	// With stops, the accumulation must hold every prior delta — but the old
	// shape recomputed it as `emitted + delta`, allocating a fresh, growing
	// string every token: O(n) bytes per token, O(n²) over the stream. A
	// strings.Builder appends delta in amortised O(1) and hands back a
	// zero-copy String() view for the per-token scan. That view aliases the
	// builder's buffer, but writeDelta copies the bytes it needs into eventBuf
	// before the next WriteString, and a stop-hit returns immediately — so no
	// live string is ever observed across an append. Byte-identical to the
	// concat: candidate holds exactly the same accumulated text each token.
	hasStops := len(stops) > 0
	emittedBuf := core.NewBuilder()
	streamErr := forEachCompatToken(ctx, model, messageID, req.Model, "", messages, opts, func(token inference.Token) bool {
		delta := processor.Process(token.Text)
		if !hasStops {
			if delta != "" {
				writeDelta(delta)
			}
			return true
		}
		prevLen := emittedBuf.Len()
		emittedBuf.WriteString(delta)
		candidate := emittedBuf.String()
		stopCut, stopHit := firstStopSequenceCut(candidate, stops)
		if stopHit {
			if stopCut <= prevLen {
				delta = ""
			} else {
				delta = candidate[prevLen:stopCut]
			}
		}
		if delta != "" {
			writeDelta(delta)
		}
		if stopHit {
			stopReason = "stop_sequence"
			return false
		}
		return true
	})
	// Headers are already flushed by the time generation runs, so a token-stream
	// error cannot change the HTTP response — surface it to the operator log and
	// finish the stream cleanly (the SDK parser still gets a well-formed close).
	if streamErr != nil {
		core.Warn("openai anthropic stream: generation error", "err", streamErr)
	}
	if tail := processor.Flush(); tail != "" {
		writeDelta(tail)
	}
	writeEventBytes("content_block_stop", func(b []byte) []byte {
		return anthropiccompat.AppendContentBlockStopEvent(b, 0)
	})
	generatedTokens := model.Metrics().GeneratedTokens
	writeEventBytes("message_delta", func(b []byte) []byte {
		return anthropiccompat.AppendMessageDeltaEvent(b, stopReason, "", generatedTokens)
	})
	writeEvent("message_stop", anthropiccompat.MessageStopPayload)
}

type ollamaChatHandler struct{ resolver openaicompat.Resolver }
type ollamaGenerateHandler struct{ resolver openaicompat.Resolver }
type ollamaTagsHandler struct{ resolver openaicompat.Resolver }
type ollamaShowHandler struct{ resolver openaicompat.Resolver }

func newOllamaChatHandler(resolver openaicompat.Resolver) http.Handler {
	return &ollamaChatHandler{resolver: resolver}
}

func newOllamaGenerateHandler(resolver openaicompat.Resolver) http.Handler {
	return &ollamaGenerateHandler{resolver: resolver}
}

func newOllamaTagsHandler(resolver openaicompat.Resolver) http.Handler {
	return &ollamaTagsHandler{resolver: resolver}
}

func newOllamaShowHandler(resolver openaicompat.Resolver) http.Handler {
	return &ollamaShowHandler{resolver: resolver}
}

func (h *ollamaChatHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !requireCompatMethod(w, r, http.MethodPost) {
		return
	}
	var req ollamacompat.ChatRequest
	if err := decodeWireJSONSized(r.Body, r.ContentLength, &req, "mlx.ollama.chat"); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err.Error(), "body")
		return
	}
	model, ok := resolveCompatModel(w, r.Context(), h.resolver, req.Model)
	if !ok {
		return
	}
	messages := ollamacompat.InferenceMessages(req.Messages)
	opts := ollamacompat.GenerateOptions(req.Options)
	if req.Stream {
		serveOllamaChatStream(w, r.Context(), model, req, messages, opts...)
		return
	}
	tokens, err := collectCompatTokens(r.Context(), model, ollamaRequestID(), req.Model, "", messages, opts...)
	if err != nil {
		writeOpenAIError(w, http.StatusInternalServerError, err.Error(), "model")
		return
	}
	if result := model.Err(); !result.OK {
		writeOpenAIError(w, http.StatusInternalServerError, result.Error(), "model")
		return
	}
	visible, _ := parseOpenAIModelOutput(model, tokens, openAITokensText(tokens))
	writeOpenAIJSON(w, http.StatusOK, ollamacompat.NewChatResponse(req.Model, visible, model.Metrics()))
}

func (h *ollamaGenerateHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !requireCompatMethod(w, r, http.MethodPost) {
		return
	}
	var req ollamacompat.GenerateRequest
	if err := decodeWireJSONSized(r.Body, r.ContentLength, &req, "mlx.ollama.generate"); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err.Error(), "body")
		return
	}
	model, ok := resolveCompatModel(w, r.Context(), h.resolver, req.Model)
	if !ok {
		return
	}
	opts := ollamacompat.GenerateOptions(req.Options)
	if req.Stream {
		serveOllamaGenerateStream(w, r.Context(), model, req, opts...)
		return
	}
	tokens, err := collectCompatTokens(r.Context(), model, ollamaRequestID(), req.Model, req.Prompt, nil, opts...)
	if err != nil {
		writeOpenAIError(w, http.StatusInternalServerError, err.Error(), "model")
		return
	}
	if result := model.Err(); !result.OK {
		writeOpenAIError(w, http.StatusInternalServerError, result.Error(), "model")
		return
	}
	visible, _ := parseOpenAIModelOutput(model, tokens, openAITokensText(tokens))
	writeOpenAIJSON(w, http.StatusOK, ollamacompat.NewGenerateResponse(req.Model, visible, model.Metrics()))
}

func (h *ollamaTagsHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !requireCompatMethod(w, r, http.MethodGet) {
		return
	}
	tags := []ollamacompat.ModelTag{}
	for _, name := range resolverModelNames(h.resolver) {
		tags = append(tags, ollamacompat.ModelTag{Name: name, Model: name})
	}
	writeOpenAIJSON(w, http.StatusOK, ollamacompat.TagsResponse{Models: tags})
}

func (h *ollamaShowHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !requireCompatMethod(w, r, http.MethodPost) {
		return
	}
	var req ollamacompat.ShowRequest
	if err := decodeWireJSONSized(r.Body, r.ContentLength, &req, "mlx.ollama.show"); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err.Error(), "body")
		return
	}
	model, ok := resolveCompatModel(w, r.Context(), h.resolver, req.Model)
	if !ok {
		return
	}
	info := model.Info()
	details := map[string]string{
		"architecture": info.Architecture,
		"model_type":   model.ModelType(),
	}
	if info.QuantBits > 0 {
		details["quantization"] = core.Sprintf("q%d", info.QuantBits)
	}
	writeOpenAIJSON(w, http.StatusOK, ollamacompat.ShowResponse{Details: details})
}

func serveOllamaChatStream(w http.ResponseWriter, ctx context.Context, model inference.TextModel, req ollamacompat.ChatRequest, messages []inference.Message, opts ...inference.GenerateOption) {
	serveOllamaStream(w, ctx, model, req.Model, "", messages, true, opts...)
}

func serveOllamaGenerateStream(w http.ResponseWriter, ctx context.Context, model inference.TextModel, req ollamacompat.GenerateRequest, opts ...inference.GenerateOption) {
	serveOllamaStream(w, ctx, model, req.Model, req.Prompt, nil, false, opts...)
}

func serveOllamaStream(w http.ResponseWriter, ctx context.Context, model inference.TextModel, modelName, prompt string, messages []inference.Message, chat bool, opts ...inference.GenerateOption) {
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)
	processor := parser.NewProcessor(parser.Config{Mode: parser.Capture}, parser.HintFromInference(model.Info()))
	// writeLine serves the once-per-request summary frame (the terminal
	// NewChatResponse/NewGenerateResponse carrying the metrics) — not a hot-path
	// cost, so it stays on the reflect marshal. The per-token text frames go
	// through writeDelta below, which is where the allocations multiply.
	writeLine := func(payload any) {
		writeNDJSONLine(w, core.JSONMarshalString(payload))
		if flusher != nil {
			flusher.Flush()
		}
	}
	// frameBuf is the reused per-token frame buffer — writeOllamaChatFrame /
	// writeOllamaGenerateFrame rebuild each NDJSON line into it, so the whole
	// stream's deltas cost one amortised buffer grow rather than a marshal + box
	// per token.
	var frameBuf []byte
	writeDelta := func(delta string) {
		if chat {
			frameBuf = writeOllamaChatFrame(w, frameBuf, modelName, delta)
		} else {
			frameBuf = writeOllamaGenerateFrame(w, frameBuf, modelName, delta)
		}
		if flusher != nil {
			flusher.Flush()
		}
	}
	streamErr := forEachCompatToken(ctx, model, ollamaRequestID(), modelName, prompt, messages, opts, func(token inference.Token) bool {
		delta := processor.Process(token.Text)
		// The empty-delta skip is also wire-equivalence load-bearing for the
		// generate shape: GenerateResponse.Response is omitempty, so the
		// marshalled summary drops it when empty — the per-token frame only
		// matches a marshalled response with a non-empty delta. (Chat content
		// has no omitempty, but the guard keeps both paths uniform.)
		if delta == "" {
			return true
		}
		writeDelta(delta)
		return true
	})
	// NDJSON headers are already on the wire; a generation error can't alter the
	// HTTP status, so log it for the operator and still emit the final summary
	// frame below to close the stream the way the Ollama client expects.
	if streamErr != nil {
		core.Warn("openai ollama stream: generation error", "err", streamErr)
	}
	if tail := processor.Flush(); tail != "" {
		writeDelta(tail)
	}
	if chat {
		writeLine(ollamacompat.NewChatResponse(modelName, "", model.Metrics()))
	} else {
		writeLine(ollamacompat.NewGenerateResponse(modelName, "", model.Metrics()))
	}
}

func decodeWireJSON(body io.Reader, into any, scope string) error {
	return decodeWireJSONSized(body, -1, into, scope)
}

// decodeWireJSONSized decodes a JSON request body, seeding the read buffer
// from contentLength when the caller knows it (an HTTP handler passes
// r.ContentLength; -1 means unknown). io.ReadAll starts from a 512-byte
// buffer and grows-and-copies as it reads, discarding each intermediate —
// a 12 KB body (Claude Code routinely sends multi-KB system prompts) churns
// through 512→1K→2K→4K→8K→16K, ~6 allocations and ~2× the body's bytes in
// dead intermediate buffers. Seeding capacity at the known length collapses
// that to a single right-sized allocation. The result is byte-identical to
// io.ReadAll — readBodySized reads to EOF, so a Content-Length that
// under- or over-states the body still yields exactly the bytes net/http
// delivers.
func decodeWireJSONSized(body io.Reader, contentLength int64, into any, scope string) error {
	if body == nil {
		return core.E(scope, "request body is nil", nil)
	}
	data, err := readBodySized(body, contentLength)
	if err != nil {
		return core.E(scope, "read request body", err)
	}
	// data is a freshly-read, single-owner buffer — decode it directly
	// rather than copying it into a string first. The previous
	// string(data) round-trip allocated a full copy of the body on every
	// request (all three wire protocols decode through here), and
	// JSONUnmarshalString immediately viewed it back to bytes via AsBytes
	// anyway, so the copy was pure waste.
	result := core.JSONUnmarshal(data, into)
	if !result.OK {
		if err, ok := result.Value.(error); ok {
			return err
		}
		return core.E(scope, "invalid request body", nil)
	}
	return nil
}

// readBodySized reads all of body into a single buffer, seeding its capacity
// from sizeHint when positive. It mirrors io.ReadAll's behaviour exactly —
// reads until EOF, returns whatever bytes arrived, treats io.EOF as success
// — but skips the grow-and-copy churn when the length is known up front.
// A wrong sizeHint only costs a normal append-grow for the overflow (body
// longer than declared) or wastes a little capacity (body shorter); the
// returned bytes are always the body's true contents.
func readBodySized(body io.Reader, sizeHint int64) ([]byte, error) {
	if sizeHint <= 0 || sizeHint > maxPresizedBody {
		return io.ReadAll(body)
	}
	// Seed one extra byte of capacity so the EOF that a Reader signals on the
	// read *after* it has handed back the last byte (the shape strings.Reader
	// and http.Body both use) lands in spare capacity instead of forcing a
	// grow — the common case where the body is exactly sizeHint bytes then
	// stays a single allocation.
	buf := make([]byte, 0, sizeHint+1)
	for {
		if len(buf) == cap(buf) {
			// Body ran past the seeded capacity (Content-Length understated
			// the body) — grow and keep reading so the result still matches
			// io.ReadAll.
			buf = append(buf, 0)[:len(buf)]
		}
		n, err := body.Read(buf[len(buf):cap(buf)])
		buf = buf[:len(buf)+n]
		if err != nil {
			if err == io.EOF {
				return buf, nil
			}
			return buf, err
		}
	}
}

// maxPresizedBody caps how large a Content-Length we trust for up-front
// allocation, so a hostile or mistaken header can't make us reserve an
// enormous buffer before a single byte is read. Bodies above the cap fall
// back to io.ReadAll's incremental growth.
const maxPresizedBody = 8 << 20 // 8 MiB

func requireCompatMethod(w http.ResponseWriter, r *http.Request, method string) bool {
	if r == nil {
		writeOpenAIError(w, http.StatusBadRequest, "request is nil", "request")
		return false
	}
	if r.Method != method {
		w.Header().Set("Allow", method)
		writeOpenAIError(w, http.StatusMethodNotAllowed, "method not allowed", "method")
		return false
	}
	return true
}

func resolveCompatModel(w http.ResponseWriter, ctx context.Context, resolver openaicompat.Resolver, modelName string) (inference.TextModel, bool) {
	if resolver == nil {
		writeOpenAIError(w, http.StatusServiceUnavailable, "handler is not configured", "model")
		return nil, false
	}
	if core.Trim(modelName) == "" {
		writeOpenAIError(w, http.StatusBadRequest, "model is required", "model")
		return nil, false
	}
	model, err := resolver.ResolveModel(ctx, modelName)
	if err != nil {
		writeOpenAIError(w, http.StatusNotFound, err.Error(), "model")
		return nil, false
	}
	return model, true
}

type resolverModelNameLister interface {
	ModelNames() []string
}

func resolverModelNames(resolver openaicompat.Resolver) []string {
	if lister, ok := resolver.(resolverModelNameLister); ok {
		return lister.ModelNames()
	}
	if backend, ok := resolver.(*openaicompat.BackendResolver); ok && backend != nil && backend.ModelPath != "" {
		return []string{core.PathBase(backend.ModelPath)}
	}
	return nil
}

func firstStopSequenceCut(content string, stops []string) (int, bool) {
	if content == "" || len(stops) == 0 {
		return 0, false
	}
	best := -1
	for _, stop := range stops {
		if stop == "" {
			continue
		}
		idx := indexString(content, stop)
		if idx >= 0 && (best < 0 || idx < best) {
			best = idx
		}
	}
	if best < 0 {
		return 0, false
	}
	return best, true
}

func normalizeAnthropicStopSequences(stops []string) ([]string, error) {
	if len(stops) == 0 {
		return nil, nil
	}
	out := make([]string, 0, len(stops))
	for _, stop := range stops {
		if stop == "" {
			return nil, core.E("mlx.anthropic.messages", "stop_sequences must not contain empty strings", nil)
		}
		out = append(out, stop)
	}
	return out, nil
}

func anthropicMessageID() string {
	return idWithPrefix("msg_")
}

func ollamaRequestID() string {
	return idWithPrefix("ollama_")
}

func parseOpenAIModelOutput(model inference.TextModel, tokens []inference.Token, text string) (string, string) {
	var (
		result inference.ReasoningParseResult
		err    error
	)
	if p, ok := model.(inference.ReasoningParser); ok {
		result, err = p.ParseReasoning(tokens, text)
	} else if model != nil {
		result, err = parser.ForHint(parser.HintFromInference(model.Info())).ParseReasoning(tokens, text)
	} else {
		result, err = parser.ForHint(parser.Hint{}).ParseReasoning(tokens, text)
	}
	if err != nil {
		return cleanChannelMarkers(text), ""
	}
	visible := result.VisibleText
	if visible == "" && text != "" {
		// Gemma 4 31B/26B open a <|channel>thought channel without reliably
		// emitting the <channel|> close, so the parser classifies the whole
		// unterminated span — answer included — as reasoning and leaves nothing
		// visible. We don't replay thoughts, so display the output rather than
		// dropping it: fall back to the full text (markers cleaned below). Never
		// return an empty reply when the model actually generated tokens.
		visible = text
	}
	return cleanChannelMarkers(visible), reasoningText(result.Reasoning)
}

// cleanChannelMarkers strips Gemma 4 / gpt-oss reasoning-channel control tokens
// (the <|channel><name> header, its <channel|> close, and bare residue) from
// text while keeping the readable reasoning and answer, so the display path
// shows the model's output inline instead of raw control scaffolding. No-op on
// text the parser already cleaned.
func cleanChannelMarkers(text string) string {
	for _, m := range []string{
		"<|channel>thought\n", "<|channel>thinking\n", "<|channel>reasoning\n",
		"<|channel>analysis\n", "<|channel>final\n",
		"<|channel>thought", "<|channel>thinking", "<|channel>reasoning",
		"<|channel>analysis", "<|channel>final",
		"<channel|>", "<|channel>",
	} {
		text = core.Replace(text, m, "")
	}
	return core.Trim(text)
}

// indexString locates substr inside s, returning its index or -1.
func indexString(s, substr string) int {
	if substr == "" {
		return 0
	}
	if len(substr) > len(s) {
		return -1
	}
	for i := range len(s) - len(substr) + 1 {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}

func openAITokensText(tokens []inference.Token) string {
	builder := core.NewBuilder()
	builder.Grow(openAITokensTextLen(tokens))
	for _, token := range tokens {
		builder.WriteString(token.Text)
	}
	return builder.String()
}

func reasoningText(segments []inference.ReasoningSegment) string {
	if len(segments) == 0 {
		return ""
	}
	builder := core.NewBuilder()
	total := 0
	for _, segment := range segments {
		total += len(segment.Text)
	}
	builder.Grow(total)
	for _, segment := range segments {
		builder.WriteString(segment.Text)
	}
	return builder.String()
}

func openAITokensTextLen(tokens []inference.Token) int {
	total := 0
	for _, token := range tokens {
		total += len(token.Text)
	}
	return total
}
