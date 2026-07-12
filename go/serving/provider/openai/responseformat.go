// SPDX-Licence-Identifier: EUPL-1.2

// response_format (RFC §6.15): validating a chat completion's visible text
// against the shape the caller asked for, and repairing it by re-prompting the
// model on a shape mismatch. The actual validate-and-repair loop is
// serving/structured.ValidateWithRepair; this file owns the OpenAI wire DTO,
// the JSON-schema -> structured.Schema projection, and the model-backed
// Reprompter that closes the loop.
package openai

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/serving/structured"
)

// ResponseFormat selects structured output for a chat-completion request
// (RFC §6.15): "" / "text" is unconstrained prose (the default — never pays
// the validation path below); "json_object" requires the visible text to be a
// JSON object; "json_schema" additionally requires JSONSchema's declared
// fields, repairing a mismatch by re-prompting the model up to
// DefaultStructuredRepairAttempts total tries.
type ResponseFormat struct {
	Type       string              `json:"type"`
	JSONSchema *ResponseJSONSchema `json:"json_schema,omitempty"`
}

// ResponseJSONSchema is the OpenAI json_schema response_format payload: an
// arbitrary JSON-schema document plus its name and strictness flag. Only
// Schema's top-level "required" list and each named field's "type" are used
// (jsonSchemaToStructuredSchema) — serving/structured.Schema is deliberately
// shallow (the plain-text validation fallback, RFC §6.15), not a full
// recursive JSON-schema evaluator.
type ResponseJSONSchema struct {
	Name   string         `json:"name,omitempty"`
	Schema map[string]any `json:"schema,omitempty"`
	Strict *bool          `json:"strict,omitempty"`
}

// DefaultStructuredRepairAttempts bounds the model re-prompt loop
// validateStructuredOutput runs on a shape mismatch: one initial attempt plus
// up to two repair re-prompts.
const DefaultStructuredRepairAttempts = 3

// needsValidation reports whether f asks for a JSON shape serving/structured
// must check the model's output against. A nil f (response_format omitted
// from the request) or Type "" / "text" is unconstrained prose — the common
// case — and never reaches the validation path.
func (f *ResponseFormat) needsValidation() bool {
	return f != nil && (f.Type == "json_object" || f.Type == "json_schema")
}

// schema derives the serving/structured.Schema f's contract validates
// against: "json_object" has no required fields — an empty Schema — so
// Validate's own contract ("the root must be a JSON object") is the whole
// check; "json_schema" additionally requires every name in JSONSchema.Schema's
// top-level "required" list, typed from its matching "properties" entry.
func (f *ResponseFormat) schema() structured.Schema {
	if f == nil || f.Type != "json_schema" || f.JSONSchema == nil {
		return structured.Schema{}
	}
	return jsonSchemaToStructuredSchema(f.JSONSchema.Schema)
}

// jsonSchemaToStructuredSchema converts a top-level JSON-schema object's
// "required" list into a shallow structured.Schema — one Kind per the
// matching "properties" entry's "type" (KindString when a required name has
// no matching property or an unrecognised/absent type, so a schema quirk never
// panics — it just falls back to the most permissive scalar kind).
func jsonSchemaToStructuredSchema(schema map[string]any) structured.Schema {
	required, _ := schema["required"].([]any)
	if len(required) == 0 {
		return structured.Schema{}
	}
	properties, _ := schema["properties"].(map[string]any)
	fields := make(map[string]structured.Kind, len(required))
	for _, entry := range required {
		name, ok := entry.(string)
		if !ok || name == "" {
			continue
		}
		fields[name] = propertyKind(properties, name)
	}
	return structured.Schema{Fields: fields}
}

// propertyKind resolves one required field's structured.Kind from its
// JSON-schema "properties" entry, defaulting to KindString when the property
// is missing or its "type" is absent/unrecognised.
func propertyKind(properties map[string]any, name string) structured.Kind {
	prop, _ := properties[name].(map[string]any)
	if prop == nil {
		return structured.KindString
	}
	kind, _ := prop["type"].(string)
	switch kind {
	case "integer", "number":
		return structured.KindNumber
	case "boolean":
		return structured.KindBool
	case "object":
		return structured.KindObject
	case "array":
		return structured.KindArray
	default: // "string" and anything unrecognised
		return structured.KindString
	}
}

// modelReprompter re-invokes model with the validation error appended as a
// fresh turn, so structured.ValidateWithRepair can ask a live model to correct
// a shape mismatch. It strips reasoning the same way the main generation path
// does (NewThinkingExtractor), so a re-prompted <think>/<|channel>thought
// block never leaks into the text being validated.
type modelReprompter struct {
	ctx      context.Context
	model    inference.TextModel
	messages []inference.Message
	opts     []inference.GenerateOption
}

// Reprompt implements structured.Reprompter.
func (rp *modelReprompter) Reprompt(prevRaw string, validateErr error) (string, error) {
	repairMessages := append(append(make([]inference.Message, 0, len(rp.messages)+2), rp.messages...),
		inference.Message{Role: "assistant", Content: prevRaw},
		inference.Message{Role: "user", Content: "Your previous response did not match the requested format: " +
			validateErr.Error() + ". Reply again with ONLY the corrected JSON — no other text."},
	)
	extractor := NewThinkingExtractor()
	for token := range rp.model.Chat(rp.ctx, repairMessages, rp.opts...) {
		extractor.Process(token)
	}
	extractor.Flush()
	if result := rp.model.Err(); !result.OK {
		return "", result.Err()
	}
	return core.Trim(extractor.Content()), nil
}

// validateStructuredOutput checks content against format's contract,
// repairing via fresh model turns (up to DefaultStructuredRepairAttempts total
// tries, built from messages/opts — the same request context the original
// generation used) when the shape doesn't match. It returns the (possibly
// repaired) text, or the last validation error once attempts are exhausted.
// format is assumed to satisfy needsValidation() — callers guard on that
// before paying this path.
func validateStructuredOutput(ctx context.Context, model inference.TextModel, messages []inference.Message, opts []inference.GenerateOption, content string, format *ResponseFormat) (string, error) {
	reprompter := &modelReprompter{ctx: ctx, model: model, messages: messages, opts: opts}
	return structured.ValidateWithRepair(content, format.schema(), reprompter, DefaultStructuredRepairAttempts)
}
