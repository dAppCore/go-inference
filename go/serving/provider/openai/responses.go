// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"time"

	"dappco.re/go/inference"
)

// DefaultResponsesPath is the OpenAI-compatible Responses endpoint.
const DefaultResponsesPath = "/v1/responses"

// ResponseInputMessage is the message form accepted by the Responses adapter.
type ResponseInputMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ResponseRequest is the minimal OpenAI-compatible Responses request shape
// shared by local runtimes and provider clients.
type ResponseRequest struct {
	Model           string                 `json:"model"`
	Input           []ResponseInputMessage `json:"input,omitempty"`
	Instructions    string                 `json:"instructions,omitempty"`
	Temperature     *float32               `json:"temperature,omitempty"`
	TopP            *float32               `json:"top_p,omitempty"`
	MinP            *float32               `json:"min_p,omitempty"`
	TopK            *int                   `json:"top_k,omitempty"`
	MaxOutputTokens *int                   `json:"max_output_tokens,omitempty"`
	Stream          bool                   `json:"stream,omitempty"`
	Stop            StopList               `json:"stop,omitempty"`
	User            string                 `json:"user,omitempty"`
}

// ResponseOutputText is one visible text item in a Responses output message.
type ResponseOutputText struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// ResponseOutputMessage is the assistant message emitted by a response.
type ResponseOutputMessage struct {
	ID      string               `json:"id,omitempty"`
	Type    string               `json:"type"`
	Role    string               `json:"role"`
	Content []ResponseOutputText `json:"content"`
}

// ResponseUsage records token accounting for a Responses call.
type ResponseUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// Response is the non-streaming OpenAI-compatible Responses body.
type Response struct {
	ID      string                  `json:"id"`
	Object  string                  `json:"object"`
	Created int64                   `json:"created"`
	Model   string                  `json:"model"`
	Output  []ResponseOutputMessage `json:"output"`
	Usage   ResponseUsage           `json:"usage"`
	Thought *string                 `json:"thought,omitempty"`
}

// ResponseStreamEvent is a compact SSE event payload for Responses streaming.
type ResponseStreamEvent struct {
	Type     string    `json:"type"`
	Response *Response `json:"response,omitempty"`
	Delta    string    `json:"delta,omitempty"`
	Thought  *string   `json:"thought,omitempty"`
}

// ResponseMessages converts a Responses request into inference messages.
func ResponseMessages(req ResponseRequest) []inference.Message {
	out := make([]inference.Message, 0, len(req.Input)+1)
	if req.Instructions != "" {
		out = append(out, inference.Message{Role: "system", Content: req.Instructions})
	}
	for _, msg := range req.Input {
		out = append(out, inference.Message{Role: msg.Role, Content: msg.Content})
	}
	return out
}

// ResponseGenerateOptions converts Responses sampling fields into inference
// options.
func ResponseGenerateOptions(req ResponseRequest) ([]inference.GenerateOption, error) {
	chatReq := ChatCompletionRequest{
		Model:       req.Model,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		MinP:        req.MinP,
		TopK:        req.TopK,
		MaxTokens:   req.MaxOutputTokens,
		// Pre-size — saves the append-grow cascade on every Responses
		// API call. Twenty-turn requests previously paid ~4 grow allocs
		// before reaching their final size.
		Messages: make([]ChatMessage, 0, len(req.Input)),
	}
	for _, msg := range req.Input {
		chatReq.Messages = append(chatReq.Messages, ChatMessage{Role: msg.Role, Content: msg.Content})
	}
	if len(chatReq.Messages) == 0 && req.Instructions != "" {
		chatReq.Messages = []ChatMessage{{Role: "system", Content: req.Instructions}}
	}
	return GenerateOptions(chatReq)
}

// NewTextResponse builds a Responses body from visible text and metrics.
func NewTextResponse(id, model, text string, metrics inference.GenerateMetrics) Response {
	return Response{
		ID:      id,
		Object:  "response",
		Created: time.Now().Unix(),
		Model:   model,
		Output: []ResponseOutputMessage{{
			Type: "message",
			Role: "assistant",
			Content: []ResponseOutputText{{
				Type: "output_text",
				Text: text,
			}},
		}},
		Usage: ResponseUsage{
			InputTokens:  metrics.PromptTokens,
			OutputTokens: metrics.GeneratedTokens,
			TotalTokens:  metrics.PromptTokens + metrics.GeneratedTokens,
		},
	}
}
