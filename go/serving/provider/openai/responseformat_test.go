// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"context"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/inference/serving/structured"
)

// --- ResponseFormat.needsValidation -----------------------------------------

// TestResponseFormat_NeedsValidation_Good pins the two validating types.
func TestResponseFormat_NeedsValidation_Good(t *testing.T) {
	if !(&ResponseFormat{Type: "json_object"}).needsValidation() {
		t.Fatal(`needsValidation("json_object") = false, want true`)
	}
	if !(&ResponseFormat{Type: "json_schema"}).needsValidation() {
		t.Fatal(`needsValidation("json_schema") = false, want true`)
	}
}

// TestResponseFormat_NeedsValidation_Bad pins the non-validating cases: a nil
// format (response_format omitted) and the "text"/"" types.
func TestResponseFormat_NeedsValidation_Bad(t *testing.T) {
	var nilFormat *ResponseFormat
	if nilFormat.needsValidation() {
		t.Fatal("nil ResponseFormat.needsValidation() = true, want false")
	}
	if (&ResponseFormat{Type: "text"}).needsValidation() {
		t.Fatal(`needsValidation("text") = true, want false`)
	}
	if (&ResponseFormat{}).needsValidation() {
		t.Fatal(`needsValidation("") = true, want false`)
	}
}

// TestResponseFormat_NeedsValidation_Ugly pins that an unrecognised type is
// treated as non-validating (ValidateRequest rejects it earlier as a 400 —
// needsValidation itself never panics or false-positives on it).
func TestResponseFormat_NeedsValidation_Ugly(t *testing.T) {
	if (&ResponseFormat{Type: "yaml"}).needsValidation() {
		t.Fatal(`needsValidation("yaml") = true, want false`)
	}
}

// --- ResponseFormat.schema / jsonSchemaToStructuredSchema / propertyKind ---

// TestResponseFormat_Schema_Good pins the full conversion: every JSON-schema
// basic type maps to its matching structured.Kind for a required field.
func TestResponseFormat_Schema_Good(t *testing.T) {
	format := &ResponseFormat{Type: "json_schema", JSONSchema: &ResponseJSONSchema{Schema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name":   map[string]any{"type": "string"},
			"age":    map[string]any{"type": "integer"},
			"score":  map[string]any{"type": "number"},
			"vip":    map[string]any{"type": "boolean"},
			"meta":   map[string]any{"type": "object"},
			"tags":   map[string]any{"type": "array"},
			"unused": map[string]any{"type": "string"}, // not required — must not appear
		},
		"required": []any{"name", "age", "score", "vip", "meta", "tags"},
	}}}
	schema := format.schema()
	want := map[string]structured.Kind{
		"name": structured.KindString, "age": structured.KindNumber, "score": structured.KindNumber,
		"vip": structured.KindBool, "meta": structured.KindObject, "tags": structured.KindArray,
	}
	if len(schema.Fields) != len(want) {
		t.Fatalf("schema.Fields = %+v, want exactly the required fields %+v", schema.Fields, want)
	}
	for name, kind := range want {
		if schema.Fields[name] != kind {
			t.Fatalf("schema.Fields[%q] = %q, want %q", name, schema.Fields[name], kind)
		}
	}
}

// TestResponseFormat_Schema_Bad pins the non-json_schema paths: a nil format,
// a "json_object" format (no schema to derive from), and json_schema with a
// nil JSONSchema all yield the empty Schema — Validate's own "must be a JSON
// object" contract is the whole check.
func TestResponseFormat_Schema_Bad(t *testing.T) {
	var nilFormat *ResponseFormat
	if fields := nilFormat.schema().Fields; len(fields) != 0 {
		t.Fatalf("nil ResponseFormat.schema() = %+v, want empty", fields)
	}
	if fields := (&ResponseFormat{Type: "json_object"}).schema().Fields; len(fields) != 0 {
		t.Fatalf(`json_object ResponseFormat.schema() = %+v, want empty`, fields)
	}
	if fields := (&ResponseFormat{Type: "json_schema"}).schema().Fields; len(fields) != 0 {
		t.Fatalf("json_schema with nil JSONSchema .schema() = %+v, want empty", fields)
	}
}

// TestResponseFormat_Schema_Ugly pins the defensive fallbacks: a required name
// with no matching "properties" entry, and one whose "type" is missing or
// unrecognised, both default to KindString rather than panicking or dropping
// the field.
func TestResponseFormat_Schema_Ugly(t *testing.T) {
	format := &ResponseFormat{Type: "json_schema", JSONSchema: &ResponseJSONSchema{Schema: map[string]any{
		"required": []any{"missing_property", "no_type", "weird_type", 42 /* non-string entry, skipped */},
		"properties": map[string]any{
			"no_type":    map[string]any{},
			"weird_type": map[string]any{"type": "frobnicate"},
		},
	}}}
	schema := format.schema()
	if len(schema.Fields) != 3 {
		t.Fatalf("schema.Fields = %+v, want 3 entries (non-string required entry skipped)", schema.Fields)
	}
	for _, name := range []string{"missing_property", "no_type", "weird_type"} {
		if schema.Fields[name] != structured.KindString {
			t.Fatalf("schema.Fields[%q] = %q, want KindString fallback", name, schema.Fields[name])
		}
	}
}

// --- modelReprompter ---------------------------------------------------------

// TestModelReprompter_Reprompt_Good pins the happy path: Reprompt appends the
// failed attempt + a repair instruction as fresh turns, re-invokes the model,
// and returns its stripped visible text.
func TestModelReprompter_Reprompt_Good(t *testing.T) {
	model := &recordingModel{stubModel: stubModel{tokens: []inference.Token{{Text: `{"name":"Ada"}`}}}}
	rp := &modelReprompter{ctx: context.Background(), model: model, messages: []inference.Message{{Role: "user", Content: "give me json"}}}
	got, err := rp.Reprompt(`{"name":`, structured.Validate(`{"name":`, structured.Schema{}))
	if err != nil {
		t.Fatalf("Reprompt: unexpected error: %v", err)
	}
	if got != `{"name":"Ada"}` {
		t.Fatalf("Reprompt content = %q, want the model's fresh output", got)
	}
	if len(model.received) != 3 {
		t.Fatalf("Reprompt sent %d messages, want the original 1 plus assistant+user repair turns", len(model.received))
	}
	if model.received[0].Content != "give me json" {
		t.Fatalf("Reprompt messages[0] = %+v, want the original turn preserved", model.received[0])
	}
	if model.received[1].Role != "assistant" || model.received[1].Content != `{"name":` {
		t.Fatalf("Reprompt messages[1] = %+v, want the failed attempt as an assistant turn", model.received[1])
	}
	if model.received[2].Role != "user" {
		t.Fatalf("Reprompt messages[2].Role = %q, want user", model.received[2].Role)
	}
}

// TestModelReprompter_Reprompt_Bad pins that a model-side generation error
// surfaces as the Reprompt error rather than an empty success.
func TestModelReprompter_Reprompt_Bad(t *testing.T) {
	model := &recordingModel{stubModel: stubModel{err: errRepromptFailed}}
	rp := &modelReprompter{ctx: context.Background(), model: model, messages: nil}
	if _, err := rp.Reprompt("bad", errRepromptFailed); err == nil {
		t.Fatal("Reprompt with a failing model: expected error, got nil")
	}
}

// --- validateStructuredOutput ------------------------------------------------

// TestValidateStructuredOutput_Good pins the no-repair-needed path: valid
// content passes straight through with no model re-call.
func TestValidateStructuredOutput_Good(t *testing.T) {
	model := &recordingModel{}
	format := &ResponseFormat{Type: "json_object"}
	got, err := validateStructuredOutput(context.Background(), model, nil, nil, `{"ok":true}`, format)
	if err != nil {
		t.Fatalf("validateStructuredOutput: unexpected error: %v", err)
	}
	if got != `{"ok":true}` {
		t.Fatalf("validateStructuredOutput = %q, want the original content unchanged", got)
	}
	if model.calls != 0 {
		t.Fatalf("validateStructuredOutput on valid content called the model %d times, want 0", model.calls)
	}
}

// TestValidateStructuredOutput_Bad pins the exhausted-repair path: a model
// that always returns invalid JSON surfaces the last validation error.
func TestValidateStructuredOutput_Bad(t *testing.T) {
	model := &recordingModel{sequenced: [][]inference.Token{
		{{Text: "still not json"}},
		{{Text: "also not json"}},
	}}
	format := &ResponseFormat{Type: "json_object"}
	_, err := validateStructuredOutput(context.Background(), model, nil, nil, "not json at all", format)
	if err == nil {
		t.Fatal("validateStructuredOutput exhausted repair: expected error, got nil")
	}
}

// TestValidateStructuredOutput_Ugly pins the recovery path end-to-end: the
// first (already-generated) content is invalid, the model's second turn
// (driven through the real modelReprompter, not a fake Reprompter) returns
// valid JSON, and the repaired text comes back with no error.
func TestValidateStructuredOutput_Ugly(t *testing.T) {
	model := &recordingModel{sequenced: [][]inference.Token{
		{{Text: `{"name":"Ada","age":36}`}}, // served on the repair re-call
	}}
	format := &ResponseFormat{Type: "json_schema", JSONSchema: &ResponseJSONSchema{Schema: map[string]any{
		"required":   []any{"name", "age"},
		"properties": map[string]any{"name": map[string]any{"type": "string"}, "age": map[string]any{"type": "integer"}},
	}}}
	got, err := validateStructuredOutput(context.Background(), model, []inference.Message{{Role: "user", Content: "describe Ada"}}, nil, `{"name":"Ada"}`, format)
	if err != nil {
		t.Fatalf("validateStructuredOutput recovery: unexpected error: %v", err)
	}
	if got != `{"name":"Ada","age":36}` {
		t.Fatalf("validateStructuredOutput recovery = %q, want the repaired payload", got)
	}
	if model.calls != 1 {
		t.Fatalf("validateStructuredOutput recovery called the model %d times, want exactly 1 repair call", model.calls)
	}
}

var errRepromptFailed = &requestValidationError{message: "generation failed"}
