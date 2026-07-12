// SPDX-Licence-Identifier: EUPL-1.2

// Request decoding, validation, and generate-option resolution for the
// OpenAI chat-completions wire shape.
package openai

import (
	"io"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

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
		return ChatCompletionRequest{}, result.Err()
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
	if req.MinP != nil && (*req.MinP < 0 || *req.MinP > 1) {
		return requestError("min_p must be in [0, 1]", "min_p")
	}
	if req.TopK != nil && *req.TopK < 0 {
		return requestError("top_k must be >= 0", "top_k")
	}
	if req.MaxTokens != nil && *req.MaxTokens < 0 {
		return requestError("max_tokens must be >= 0", "max_tokens")
	}
	if err := validateResponseFormat(req.ResponseFormat, req.Stream); err != nil {
		return err
	}
	return nil
}

// validateResponseFormat checks response_format's shape ahead of generation: a
// nil format (the field omitted) or Type "" / "text" is always valid (no
// schema required); "json_object" needs no schema either; "json_schema" must
// carry a non-empty JSONSchema.Schema (structured.ValidateWithRepair has
// nothing to validate against otherwise). Any other Type is rejected outright.
//
// stream=true is rejected for a validating format (json_object / json_schema):
// validateStructuredOutput needs the full response materialised before it can
// check — let alone repair — the shape, which is fundamentally at odds with
// emitting content deltas as they arrive. Rather than half-build streaming
// structured output (silently skipping validation, or buffering and replaying
// as a single fake "stream"), this is a clean, documented 4xx — see the gap
// note in handler.go.
func validateResponseFormat(format *ResponseFormat, stream bool) error {
	if format == nil {
		return nil
	}
	switch format.Type {
	case "", "text", "json_object":
		// no schema required
	case "json_schema":
		if format.JSONSchema == nil || len(format.JSONSchema.Schema) == 0 {
			return requestError("response_format.json_schema.schema is required", "response_format")
		}
	default:
		return requestError("response_format.type must be \"text\", \"json_object\", or \"json_schema\"", "response_format")
	}
	if stream && format.needsValidation() {
		return requestError("response_format json_object/json_schema requires stream=false", "response_format")
	}
	return nil
}

// GenerateOptions converts request sampling fields into inference options.
func GenerateOptions(req ChatCompletionRequest) ([]inference.GenerateOption, error) {
	if err := ValidateRequest(req); err != nil {
		return nil, err
	}
	opts := []inference.GenerateOption{
		inference.WithTemperature(resolvedFloat(req.Temperature, DefaultTemperature)),
		inference.WithTopP(resolvedFloat(req.TopP, DefaultTopP)),
		inference.WithMinP(resolvedFloat(req.MinP, 0)),
		inference.WithTopK(resolvedInt(req.TopK, DefaultTopK)),
		inference.WithMaxTokens(resolvedInt(req.MaxTokens, DefaultMaxTokens)),
	}
	if et := req.thinkingOverride(); et != nil {
		opts = append(opts, inference.WithEnableThinking(et))
	}
	if req.ChatTemplateKwargs != nil && req.ChatTemplateKwargs.ThinkingBudget != nil && *req.ChatTemplateKwargs.ThinkingBudget > 0 {
		opts = append(opts, inference.WithThinkingBudget(*req.ChatTemplateKwargs.ThinkingBudget))
	}
	return opts, nil
}

// thinkingOverride resolves an explicit reasoning toggle from the request:
// chat_template_kwargs.enable_thinking (vLLM/SGLang convention) wins; otherwise
// reasoning_effort=="none" disables thinking. nil = no override (model default).
func (req ChatCompletionRequest) thinkingOverride() *bool {
	if req.ChatTemplateKwargs != nil && req.ChatTemplateKwargs.EnableThinking != nil {
		return req.ChatTemplateKwargs.EnableThinking
	}
	if core.Lower(core.Trim(req.ReasoningEffort)) == "none" {
		off := false
		return &off
	}
	return nil
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
