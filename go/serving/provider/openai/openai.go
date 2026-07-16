// SPDX-Licence-Identifier: EUPL-1.2

// Package openai adapts inference.TextModel implementations to the
// OpenAI-compatible chat completions wire format.
package openai

import (
	core "dappco.re/go"
	"dappco.re/go/inference/jsonenc"
)

const DefaultChatCompletionsPath = "/v1/chat/completions"

const (
	// DefaultMaxTokens applies when a request omits max_tokens. gemma4 thinks by default and
	// regularly spends >2k tokens inside the thought channel before the visible answer — at
	// the old 2048 default the most common client failure was an EMPTY reply with
	// finish=length (the budget died mid-thought). 8192 is the family's documented working
	// range; clients that want tighter budgets say so explicitly.
	DefaultMaxTokens = 8192
)

// ChatCompletionRequest is the OpenAI-compatible request body.
type ChatCompletionRequest struct {
	Model              string              `json:"model"`
	Messages           []ChatMessage       `json:"messages"`
	Temperature        *float32            `json:"temperature,omitempty"`
	TopP               *float32            `json:"top_p,omitempty"`
	MinP               *float32            `json:"min_p,omitempty"`
	TopK               *int                `json:"top_k,omitempty"`
	MaxTokens          *int                `json:"max_tokens,omitempty"`
	Stream             bool                `json:"stream,omitempty"`
	Stop               StopList            `json:"stop,omitempty"`
	User               string              `json:"user,omitempty"`
	ReasoningEffort    string              `json:"reasoning_effort,omitempty"`
	ChatTemplateKwargs *ChatTemplateKwargs `json:"chat_template_kwargs,omitempty"`
	// MMProcessorKwargs carries multimodal image-processor parameters — a
	// separate top-level object from ChatTemplateKwargs (the vLLM convention:
	// chat_template_kwargs and mm_processor_kwargs are distinct request
	// fields). See MMProcessorKwargs.
	MMProcessorKwargs *MMProcessorKwargs `json:"mm_processor_kwargs,omitempty"`
	Tools             []Tool             `json:"tools,omitempty"`
	// ToolChoice controls whether/which of Tools the model is offered this turn
	// (RFC §6.4) — see toolchoice.go.
	ToolChoice *ToolChoice `json:"tool_choice,omitempty"`
	// ResponseFormat selects structured output (RFC §6.15) — see responseformat.go.
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
}

// Tool is one OpenAI function-calling tool declaration.
type Tool struct {
	Type     string       `json:"type"` // "function"
	Function ToolFunction `json:"function"`
}

// ToolFunction is a tool's name, description, and JSON-schema parameters.
type ToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  ToolParameters `json:"parameters"`
}

// ToolParameters is the JSON-schema object describing a function's arguments.
type ToolParameters struct {
	Type       string                  `json:"type"`
	Properties map[string]ToolProperty `json:"properties,omitempty"`
	Required   []string                `json:"required,omitempty"`
}

// ToolProperty is one parameter's schema (type + description).
type ToolProperty struct {
	Type        string `json:"type"`
	Description string `json:"description,omitempty"`
}

// ToolCall is one call the model emitted, in the OpenAI response shape.
type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"` // "function"
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction is a call's name + JSON-string arguments.
type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ChatTemplateKwargs carries chat-template parameters (the vLLM/SGLang
// convention). Only fields the runtime acts on are modelled; unknown keys in
// the object are skipped by the decoder.
type ChatTemplateKwargs struct {
	EnableThinking *bool `json:"enable_thinking,omitempty"`
	// ThinkingBudget caps thought-channel tokens; the backend forces the
	// channel close on overrun. 0/absent = unlimited.
	ThinkingBudget *int `json:"thinking_budget,omitempty"`
	// PreserveThinking extends reasoning preservation to tool-calling
	// assistant turns ANYWHERE in history, not just after the last user
	// message — the gemma4 canonical template's preserve_thinking flag.
	PreserveThinking *bool `json:"preserve_thinking,omitempty"`
}

// MMProcessorKwargs carries multimodal image-processor parameters (the vLLM
// convention). Only fields the runtime acts on are modelled; unknown keys in
// the object are skipped by the decoder.
type MMProcessorKwargs struct {
	// MaxSoftTokens overrides the vision soft-token budget for this request
	// (inference.GenerateConfig.VisionBudget). gemma4's model card declares
	// 70/140/280/560/1120 as its supported set, but this adapter does not
	// hard-code that set — engines/families clamp an out-of-range value to
	// what they actually support. nil = no override (image_url.detail or the
	// model's configured default applies instead).
	MaxSoftTokens *int `json:"max_soft_tokens,omitempty"`
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
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	Images    [][]byte   `json:"-"`
	Audios    [][]byte   `json:"-"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"` // assistant response: the model's function calls
	// ImageDetail carries the OpenAI-native image_url.detail hint ("low" or
	// "high") off this message's image content parts — request.go's
	// visionBudgetOverride maps it onto inference.GenerateConfig.VisionBudget
	// ("low"->70, "high"->1120) when mm_processor_kwargs.max_soft_tokens is
	// absent. The last part carrying an explicit "low"/"high" wins; "auto" or
	// an absent detail never overwrites a prior explicit value. Empty = no
	// hint (this message's images carry no detail override).
	ImageDetail string `json:"-"`
	// Reasoning / ReasoningContent carry an assistant turn's thought channel
	// echoed back by a stateless client replaying history (reasoning is the
	// canonical spelling, reasoning_content the vLLM/DeepSeek one). The
	// handler re-frames them into the native thought span for turns after the
	// last user message — the gemma4 canonical-template (2026-07-09)
	// reasoning-preservation rule — so agentic tool loops keep their chain of
	// thought across stateless replays.
	Reasoning        string `json:"reasoning,omitempty"`
	ReasoningContent string `json:"reasoning_content,omitempty"`
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
	Role      string          `json:"role,omitempty"`
	Content   string          `json:"content,omitempty"`
	ToolCalls []ToolCallDelta `json:"tool_calls,omitempty"`
}

// ToolCallDelta is one streamed tool_call fragment in a chat.completion.chunk
// delta — the OpenAI shape carries an index so the client assembles calls across
// chunks (name + id arrive first, arguments stream after).
type ToolCallDelta struct {
	Index    int                    `json:"index"`
	ID       string                 `json:"id,omitempty"`
	Type     string                 `json:"type,omitempty"`
	Function *ToolCallFunctionDelta `json:"function,omitempty"`
}

// ToolCallFunctionDelta streams a call's name and/or an arguments fragment.
type ToolCallFunctionDelta struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
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
	if d.Role == "" && d.Content == "" && len(d.ToolCalls) == 0 {
		return emptyDeltaBytes, nil
	}
	if len(d.ToolCalls) > 0 {
		// tool_calls deltas are the rare agentic path — reflect-encode via an
		// alias (no MarshalJSON) so role/content/tool_calls render from tags,
		// keeping the hand-rolled fast path below text-only.
		type alias ChatMessageDelta
		res := core.JSONMarshal(alias(d))
		if !res.OK {
			return nil, res.Err()
		}
		return res.Value.([]byte), nil
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
