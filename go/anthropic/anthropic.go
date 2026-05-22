// SPDX-Licence-Identifier: EUPL-1.2

// Package anthropic provides Anthropic Messages wire primitives over the
// shared inference contracts.
package anthropic

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

// DefaultMessagesPath is the Anthropic-compatible Messages endpoint.
const DefaultMessagesPath = "/v1/messages"

// ContentBlock is the text block shape used by Anthropic Messages.
type ContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

// Message is one Anthropic chat turn.
type Message struct {
	Role    string         `json:"role"`
	Content []ContentBlock `json:"content"`
}

// MessageRequest is the minimal Anthropic-compatible request shape.
type MessageRequest struct {
	Model         string    `json:"model"`
	System        string    `json:"system,omitempty"`
	Messages      []Message `json:"messages"`
	MaxTokens     int       `json:"max_tokens"`
	Temperature   *float32  `json:"temperature,omitempty"`
	TopP          *float32  `json:"top_p,omitempty"`
	TopK          *int      `json:"top_k,omitempty"`
	Stream        bool      `json:"stream,omitempty"`
	StopSequences []string  `json:"stop_sequences,omitempty"`
}

// Usage records Anthropic-style token accounting.
type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// MessageResponse is the non-streaming Anthropic-compatible response body.
type MessageResponse struct {
	ID           string         `json:"id"`
	Type         string         `json:"type"`
	Role         string         `json:"role"`
	Model        string         `json:"model"`
	Content      []ContentBlock `json:"content"`
	StopReason   string         `json:"stop_reason,omitempty"`
	StopSequence string         `json:"stop_sequence,omitempty"`
	Usage        Usage          `json:"usage"`
}

// AppendMessageResponse walks an Anthropic MessageResponse into the
// caller-owned buf and returns the extended slice. Fires at the HTTP-
// response-emit boundary on every non-streaming completion — callers
// bypass the encoding/json reflect path (encoder state + grow-doubled
// output buffer + per-nested-struct allocations) and pre-size the
// buffer once via MessageResponseSize. Same caller-passes-buf shape
// as state/filestore's encodeRecordMeta (W8-D) and openai's
// appendChatCompletionResponse (W9-D).
//
// MarshalJSON is deliberately NOT implemented on MessageResponse: the
// bench for core.JSONMarshalString shows that wrapping a flat struct
// in a MarshalJSON method REGRESSES json.Marshal — encoding/json then
// calls MarshalJSON, validates (compact) the returned bytes, then
// copies them into its own grow-buffer. The hand-roll wins only when
// the call site bypasses json.Marshal and calls this helper directly.
//
// Wire-compatible with json.Marshal across every branch:
//   - Always emits id, type, role, model, content, usage.
//   - stop_reason / stop_sequence: omitempty (string).
//   - content: each ContentBlock emits type always, text only when
//     non-empty (matches ContentBlock's `text,omitempty` tag).
//   - usage: always emits input_tokens + output_tokens (no
//     omitempty).
//
// Output round-trips through core.JSONUnmarshal back into a
// MessageResponse — verified by the round-trip pinning test.
//
//	buf := AppendMessageResponse(make([]byte, 0, MessageResponseSize(resp)), resp)
//	w.Write(buf)  // typical HTTP-emit shape.
func AppendMessageResponse(buf []byte, r MessageResponse) []byte {
	buf = append(buf, '{')
	buf = appendStringField(buf, "id", r.ID, false)
	buf = appendStringField(buf, "type", r.Type, true)
	buf = appendStringField(buf, "role", r.Role, true)
	buf = appendStringField(buf, "model", r.Model, true)
	buf = append(buf, ',', '"', 'c', 'o', 'n', 't', 'e', 'n', 't', '"', ':', '[')
	for i, b := range r.Content {
		if i > 0 {
			buf = append(buf, ',')
		}
		buf = appendContentBlock(buf, b)
	}
	buf = append(buf, ']')
	if r.StopReason != "" {
		buf = appendStringField(buf, "stop_reason", r.StopReason, true)
	}
	if r.StopSequence != "" {
		buf = appendStringField(buf, "stop_sequence", r.StopSequence, true)
	}
	// Usage object — always emitted (no omitempty on the field).
	buf = append(buf, ',', '"', 'u', 's', 'a', 'g', 'e', '"', ':', '{')
	buf = appendIntField(buf, "input_tokens", r.Usage.InputTokens, false)
	buf = appendIntField(buf, "output_tokens", r.Usage.OutputTokens, true)
	return append(buf, '}', '}')
}

// MessageResponseSize estimates the backing-buffer size for one
// MessageResponse so the caller's make([]byte, 0, ...) lands on a
// memory class that fits the encoded body in a single allocation.
// Returns a tight upper bound — ASCII key bytes plus the string-
// value bodies. Worst-case escape doubling on text fields lets
// append grow once at most.
func MessageResponseSize(r MessageResponse) int {
	// Per-field cost: ,"key":"value"
	//   leading-comma (1) + "key" (len(key)+2) + : (1) + "value" (len(value)+2)
	//   = 6 + len(key) + len(value)
	// First field omits leading comma: 5 + len(key) + len(value).
	size := 2 // outer braces
	size += 5 + 2 + len(r.ID)     // "id":"…"
	size += 6 + 4 + len(r.Type)   // ,"type":"…"
	size += 6 + 4 + len(r.Role)   // ,"role":"…"
	size += 6 + 5 + len(r.Model)  // ,"model":"…"
	size += 6 + 7                 // ,"content":[]
	for i, b := range r.Content {
		size += 5 + 2 + 4 + len(b.Type) // {"type":"X"}
		if b.Text != "" {
			size += 6 + 4 + len(b.Text) // ,"text":"X"
		}
		size += 1 // closing brace }
		if i > 0 {
			size += 1 // , separator between blocks
		}
	}
	if r.StopReason != "" {
		size += 6 + 11 + len(r.StopReason) // ,"stop_reason":"X"
	}
	if r.StopSequence != "" {
		size += 6 + 13 + len(r.StopSequence) // ,"stop_sequence":"X"
	}
	// ,"usage":{"input_tokens":N,"output_tokens":N}
	// 9 ("usage":) + 2 (object braces) + 5+2+10+1+11+11+10+1+11 ≈ 60
	size += 6 + 5 + 2 + 26 + 28
	return size
}

// appendContentBlock encodes a single ContentBlock as JSON onto buf.
// type is always emitted; text is omitted when empty (matches the
// `text,omitempty` tag on the struct). Lifted out so
// AppendMessageResponse / AppendMessageRequest and future content-array
// shapes share it.
func appendContentBlock(buf []byte, b ContentBlock) []byte {
	buf = append(buf, '{')
	buf = appendStringField(buf, "type", b.Type, false)
	if b.Text != "" {
		buf = appendStringField(buf, "text", b.Text, true)
	}
	return append(buf, '}')
}

// appendMessage encodes a single chat-turn Message as JSON onto buf.
// role + content always emitted; content is an array of ContentBlocks.
func appendMessage(buf []byte, m Message) []byte {
	buf = append(buf, '{')
	buf = appendStringField(buf, "role", m.Role, false)
	buf = append(buf, ',', '"', 'c', 'o', 'n', 't', 'e', 'n', 't', '"', ':', '[')
	for i, b := range m.Content {
		if i > 0 {
			buf = append(buf, ',')
		}
		buf = appendContentBlock(buf, b)
	}
	return append(buf, ']', '}')
}

// AppendMessageRequest walks an Anthropic MessageRequest into the
// caller-owned buf and returns the extended slice. Fires at the
// client-side request-encode boundary — proxies and SDK clients pay
// 2 allocs / 480-3500 B through json.Marshal's reflect path even
// before per-field pointer-allocation cost. The hand-rolled encoder
// lands at a single buffer allocation regardless of pointer-field
// count and slice depth.
//
// Wire-compatible with json.Marshal across every branch:
//   - model + messages + max_tokens always emitted (no omitempty).
//   - system: omitempty (string).
//   - temperature / top_p / top_k: omitempty (pointer); emitted as
//     number only when non-nil.
//   - stream: omitempty (bool); emitted as true only when true.
//   - stop_sequences: omitempty (slice); emitted as JSON array of
//     strings when len > 0.
//
//	buf := AppendMessageRequest(make([]byte, 0, MessageRequestSize(req)), req)
//	httpClient.Post(url, "application/json", bytes.NewReader(buf))
func AppendMessageRequest(buf []byte, r MessageRequest) []byte {
	buf = append(buf, '{')
	buf = appendStringField(buf, "model", r.Model, false)
	if r.System != "" {
		buf = appendStringField(buf, "system", r.System, true)
	}
	buf = append(buf, ',', '"', 'm', 'e', 's', 's', 'a', 'g', 'e', 's', '"', ':', '[')
	for i, m := range r.Messages {
		if i > 0 {
			buf = append(buf, ',')
		}
		buf = appendMessage(buf, m)
	}
	buf = append(buf, ']')
	buf = appendIntField(buf, "max_tokens", r.MaxTokens, true)
	if r.Temperature != nil {
		buf = appendFloat32Field(buf, "temperature", *r.Temperature, true)
	}
	if r.TopP != nil {
		buf = appendFloat32Field(buf, "top_p", *r.TopP, true)
	}
	if r.TopK != nil {
		buf = appendIntField(buf, "top_k", *r.TopK, true)
	}
	if r.Stream {
		buf = appendBoolField(buf, "stream", true, true)
	}
	if len(r.StopSequences) > 0 {
		buf = append(buf, ',', '"', 's', 't', 'o', 'p', '_', 's', 'e', 'q', 'u', 'e', 'n', 'c', 'e', 's', '"', ':', '[')
		for i, s := range r.StopSequences {
			if i > 0 {
				buf = append(buf, ',')
			}
			buf = appendJSONString(buf, s)
		}
		buf = append(buf, ']')
	}
	return append(buf, '}')
}

// MessageRequestSize estimates a tight upper bound for the backing
// buffer one MessageRequest needs so the caller's make([]byte, 0,
// MessageRequestSize(req)) lands on a memory class that fits the
// encoded body in a single allocation.
//
// Per-field overhead = ,"key":<value-framing> as documented in
// MessageResponseSize. Pointer/bool/slice fields fold in only when
// they would emit under the omitempty contract.
func MessageRequestSize(r MessageRequest) int {
	size := 2 // outer braces
	size += 5 + 5 + len(r.Model) // "model":"…"
	if r.System != "" {
		size += 6 + 6 + len(r.System) // ,"system":"…"
	}
	size += 6 + 8 // ,"messages":[]
	for i, m := range r.Messages {
		// {"role":"…","content":[…]}
		size += 5 + 4 + len(m.Role)
		size += 6 + 7 // ,"content":[]
		for j, b := range m.Content {
			size += 5 + 2 + 4 + len(b.Type) // {"type":"X"}
			if b.Text != "" {
				size += 6 + 4 + len(b.Text) // ,"text":"X"
			}
			size += 1 // }
			if j > 0 {
				size += 1 // ,
			}
		}
		size += 1 // }
		if i > 0 {
			size += 1 // ,
		}
	}
	size += 6 + 10 + 20 // ,"max_tokens":N   (20-digit int)
	if r.Temperature != nil {
		size += 6 + 11 + 24 // ,"temperature":F  (24-byte float)
	}
	if r.TopP != nil {
		size += 6 + 5 + 24 // ,"top_p":F
	}
	if r.TopK != nil {
		size += 6 + 5 + 20 // ,"top_k":N
	}
	if r.Stream {
		size += 6 + 6 + 4 // ,"stream":true
	}
	if len(r.StopSequences) > 0 {
		size += 6 + 14 // ,"stop_sequences":[]
		for i, s := range r.StopSequences {
			size += 2 + len(s) // "X"
			if i > 0 {
				size += 1 // ,
			}
		}
	}
	return size
}

// InferenceMessages converts Anthropic messages into shared inference messages.
func InferenceMessages(req MessageRequest) []inference.Message {
	out := make([]inference.Message, 0, len(req.Messages)+1)
	if req.System != "" {
		out = append(out, inference.Message{Role: "system", Content: req.System})
	}
	for _, msg := range req.Messages {
		out = append(out, inference.Message{Role: msg.Role, Content: blockText(msg.Content)})
	}
	return out
}

// GenerateOptions converts Anthropic sampling fields into inference options.
func GenerateOptions(req MessageRequest) []inference.GenerateOption {
	opts := make([]inference.GenerateOption, 0, 4)
	if req.MaxTokens > 0 {
		opts = append(opts, inference.WithMaxTokens(req.MaxTokens))
	}
	if req.Temperature != nil {
		opts = append(opts, inference.WithTemperature(*req.Temperature))
	}
	if req.TopP != nil {
		opts = append(opts, inference.WithTopP(*req.TopP))
	}
	if req.TopK != nil {
		opts = append(opts, inference.WithTopK(*req.TopK))
	}
	return opts
}

// NewTextResponse builds a text response from shared inference metrics.
func NewTextResponse(id, model, text string, metrics inference.GenerateMetrics) MessageResponse {
	return MessageResponse{
		ID:         id,
		Type:       "message",
		Role:       "assistant",
		Model:      model,
		Content:    []ContentBlock{{Type: "text", Text: text}},
		StopReason: "end_turn",
		Usage: Usage{
			InputTokens:  metrics.PromptTokens,
			OutputTokens: metrics.GeneratedTokens,
		},
	}
}

func blockText(blocks []ContentBlock) string {
	// Fast paths — common cases produce 0 or 1 string without
	// touching the builder. Per-message hot path; InferenceMessages
	// calls this once per Anthropic content array on every request.
	if len(blocks) == 0 {
		return ""
	}
	if len(blocks) == 1 {
		b := blocks[0]
		if b.Type == "" || b.Type == "text" {
			return b.Text
		}
		return ""
	}
	// Multi-block: pre-sum then Grow the builder once. Previous shape
	// (out += block.Text) was O(N²) — each += reallocated and copied
	// the entire prefix.
	total := 0
	for _, block := range blocks {
		if block.Type == "" || block.Type == "text" {
			total += len(block.Text)
		}
	}
	if total == 0 {
		return ""
	}
	builder := core.NewBuilder()
	builder.Grow(total)
	for _, block := range blocks {
		if block.Type == "" || block.Type == "text" {
			builder.WriteString(block.Text)
		}
	}
	return builder.String()
}
