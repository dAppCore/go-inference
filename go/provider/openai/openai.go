// SPDX-Licence-Identifier: EUPL-1.2

// Package openai adapts inference.TextModel implementations to the
// OpenAI-compatible chat completions wire format.
package openai

import "dappco.re/go/inference/jsonenc"

const DefaultChatCompletionsPath = "/v1/chat/completions"

const (
	DefaultTemperature = 1.0
	DefaultTopP        = 0.95
	DefaultTopK        = 64
	DefaultMaxTokens   = 2048
)

// ChatCompletionRequest is the OpenAI-compatible request body.
type ChatCompletionRequest struct {
	Model              string              `json:"model"`
	Messages           []ChatMessage       `json:"messages"`
	Temperature        *float32            `json:"temperature,omitempty"`
	TopP               *float32            `json:"top_p,omitempty"`
	TopK               *int                `json:"top_k,omitempty"`
	MaxTokens          *int                `json:"max_tokens,omitempty"`
	Stream             bool                `json:"stream,omitempty"`
	Stop               StopList            `json:"stop,omitempty"`
	User               string              `json:"user,omitempty"`
	ReasoningEffort    string              `json:"reasoning_effort,omitempty"`
	ChatTemplateKwargs *ChatTemplateKwargs `json:"chat_template_kwargs,omitempty"`
}

// ChatTemplateKwargs carries chat-template parameters (the vLLM/SGLang
// convention). Only fields the runtime acts on are modelled; unknown keys in
// the object are skipped by the decoder.
type ChatTemplateKwargs struct {
	EnableThinking *bool `json:"enable_thinking,omitempty"`
	// ThinkingBudget caps thought-channel tokens; the backend forces the
	// channel close on overrun. 0/absent = unlimited.
	ThinkingBudget *int `json:"thinking_budget,omitempty"`
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

// ChatMessage is a single chat turn. Content accepts both the plain-string
// form and the OpenAI multimodal content-part array (text + image_url parts;
// see UnmarshalJSON in content.go) — decoded images land in Images and never
// round-trip into responses.
type ChatMessage struct {
	Role    string   `json:"role"`
	Content string   `json:"content"`
	Images  [][]byte `json:"-"`
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
	// Exact upper bound on the no-escape path — both branches emit the
	// fixed key envelope plus the raw value bytes. AppendJSONString may
	// double the value size when escapes fire; that's a one-time append
	// grow on the escape-heavy path, not the streaming hot path.
	//
	// "role":"X" envelope         = 9 chars + len(value)
	// "content":"X" envelope      = 12 chars + len(value)
	// leading comma adds          = 1 char
	size := 2 // braces
	if d.Role != "" {
		size += 9 + len(d.Role)         // "role":"X"
		size += 1 + 12 + len(d.Content) // ,"content":"X"
	} else {
		size += 12 + len(d.Content) // "content":"X"
	}
	buf := make([]byte, 0, size)
	buf = append(buf, '{')
	if d.Role != "" {
		buf = jsonenc.AppendStringField(buf, "role", d.Role, false)
		buf = jsonenc.AppendStringField(buf, "content", d.Content, true)
	} else {
		buf = jsonenc.AppendStringField(buf, "content", d.Content, false)
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
