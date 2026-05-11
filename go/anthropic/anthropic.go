// SPDX-Licence-Identifier: EUPL-1.2

// Package anthropic provides Anthropic Messages wire primitives over the
// shared inference contracts.
package anthropic

import "dappco.re/go/inference"

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
	out := ""
	for _, block := range blocks {
		if block.Type == "" || block.Type == "text" {
			out += block.Text
		}
	}
	return out
}
