// SPDX-Licence-Identifier: EUPL-1.2

// Package ollama provides Ollama-compatible wire primitives over the shared
// inference contracts.
package ollama

import "dappco.re/go/inference"

const (
	DefaultChatPath     = "/api/chat"
	DefaultGeneratePath = "/api/generate"
	DefaultTagsPath     = "/api/tags"
	DefaultShowPath     = "/api/show"
)

// Message is one Ollama chat turn.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Options carries Ollama generation options that map cleanly to inference.
type Options struct {
	Temperature float32 `json:"temperature,omitempty"`
	TopK        int     `json:"top_k,omitempty"`
	TopP        float32 `json:"top_p,omitempty"`
	NumPredict  int     `json:"num_predict,omitempty"`
}

// ChatRequest is the Ollama chat request shape.
type ChatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream,omitempty"`
	Options  Options   `json:"options,omitempty"`
}

// GenerateRequest is the Ollama prompt-generation request shape.
type GenerateRequest struct {
	Model   string  `json:"model"`
	Prompt  string  `json:"prompt"`
	Stream  bool    `json:"stream,omitempty"`
	Options Options `json:"options,omitempty"`
}

// ChatResponse is the Ollama chat response shape.
type ChatResponse struct {
	Model              string  `json:"model"`
	Message            Message `json:"message"`
	Done               bool    `json:"done"`
	PromptEvalCount    int     `json:"prompt_eval_count,omitempty"`
	EvalCount          int     `json:"eval_count,omitempty"`
	TotalDuration      int64   `json:"total_duration,omitempty"`
	LoadDuration       int64   `json:"load_duration,omitempty"`
	PromptEvalDuration int64   `json:"prompt_eval_duration,omitempty"`
	EvalDuration       int64   `json:"eval_duration,omitempty"`
}

// GenerateResponse is the Ollama generate response shape.
type GenerateResponse struct {
	Model              string `json:"model"`
	Response           string `json:"response"`
	Done               bool   `json:"done"`
	PromptEvalCount    int    `json:"prompt_eval_count,omitempty"`
	EvalCount          int    `json:"eval_count,omitempty"`
	TotalDuration      int64  `json:"total_duration,omitempty"`
	LoadDuration       int64  `json:"load_duration,omitempty"`
	PromptEvalDuration int64  `json:"prompt_eval_duration,omitempty"`
	EvalDuration       int64  `json:"eval_duration,omitempty"`
}

// ModelTag is one entry in /api/tags.
type ModelTag struct {
	Name       string `json:"name"`
	Model      string `json:"model,omitempty"`
	ModifiedAt string `json:"modified_at,omitempty"`
	Size       int64  `json:"size,omitempty"`
}

// TagsResponse is the /api/tags response shape.
type TagsResponse struct {
	Models []ModelTag `json:"models"`
}

// ShowRequest is the /api/show request shape.
type ShowRequest struct {
	Model string `json:"model"`
}

// ShowResponse is the /api/show response shape.
type ShowResponse struct {
	License    string            `json:"license,omitempty"`
	Modelfile  string            `json:"modelfile,omitempty"`
	Parameters string            `json:"parameters,omitempty"`
	Template   string            `json:"template,omitempty"`
	Details    map[string]string `json:"details,omitempty"`
}

// InferenceMessages converts Ollama messages into shared inference messages.
func InferenceMessages(messages []Message) []inference.Message {
	out := make([]inference.Message, 0, len(messages))
	for _, msg := range messages {
		out = append(out, inference.Message{Role: msg.Role, Content: msg.Content})
	}
	return out
}

// GenerateOptions converts Ollama options into inference options.
func GenerateOptions(options Options) []inference.GenerateOption {
	opts := make([]inference.GenerateOption, 0, 4)
	if options.NumPredict > 0 {
		opts = append(opts, inference.WithMaxTokens(options.NumPredict))
	}
	if options.Temperature != 0 {
		opts = append(opts, inference.WithTemperature(options.Temperature))
	}
	if options.TopK > 0 {
		opts = append(opts, inference.WithTopK(options.TopK))
	}
	if options.TopP > 0 {
		opts = append(opts, inference.WithTopP(options.TopP))
	}
	return opts
}

// NewChatResponse builds an Ollama chat response from metrics.
func NewChatResponse(model, text string, metrics inference.GenerateMetrics) ChatResponse {
	return ChatResponse{
		Model:           model,
		Message:         Message{Role: "assistant", Content: text},
		Done:            true,
		PromptEvalCount: metrics.PromptTokens,
		EvalCount:       metrics.GeneratedTokens,
	}
}

// NewGenerateResponse builds an Ollama generate response from metrics.
func NewGenerateResponse(model, text string, metrics inference.GenerateMetrics) GenerateResponse {
	return GenerateResponse{
		Model:           model,
		Response:        text,
		Done:            true,
		PromptEvalCount: metrics.PromptTokens,
		EvalCount:       metrics.GeneratedTokens,
	}
}
