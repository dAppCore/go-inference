// SPDX-Licence-Identifier: EUPL-1.2

// Package structured closes the response_format loop (RFC §6.15): it
// validates a model's returned text against the shape the caller asked for,
// coerces it into a typed value, and on a validation failure repairs it by
// re-prompting the model with the parser's error — up to a bounded number of
// attempts.
//
// It is the parse-and-repair fallback strategy from §6.15. Where a backend
// supports native json-schema / grammar or tool-call extraction, the serving
// path enforces shape at generation time; this package handles the plain-text
// case the same way for every target (typed struct, or the schema descriptor
// for the no-Go-type case).
//
//	// Typed target:
//	var out Plan
//	err := structured.ParseWithRepair(raw, &out, model, 3)
//
//	// No Go type — a minimal required-fields/kinds check:
//	schema := structured.Schema{Fields: map[string]structured.Kind{
//	    "title": structured.KindString,
//	    "score": structured.KindNumber,
//	}}
//	err := structured.Validate(raw, schema)
package structured

import core "dappco.re/go"

// Parse unmarshals raw into target using core.JSONUnmarshalString, returning a
// typed error (scope "structured") on invalid JSON or a shape mismatch — a
// field whose JSON type can't coerce into the Go field, e.g. a string into an
// int. target must be a non-nil pointer.
//
//	var p Person
//	if err := structured.Parse(raw, &p); err != nil { return err }
func Parse(raw string, target any) error {
	r := core.JSONUnmarshalString(raw, target)
	if !r.OK {
		// r.Value carries the underlying *json error; surface it as the cause
		// so a Reprompter can show the model exactly what failed.
		return core.E("structured", "parse: "+r.Error(), nil)
	}
	return nil
}

// Kind is a basic JSON value type for the schema descriptor — the minimal
// vocabulary the no-Go-type validation path checks against.
type Kind string

// The JSON value kinds Validate understands. They map onto how
// encoding/json decodes an untyped value: numbers → float64, booleans → bool,
// strings → string, objects → map[string]any, arrays → []any.
const (
	KindString Kind = "string"
	KindNumber Kind = "number"
	KindBool   Kind = "bool"
	KindObject Kind = "object"
	KindArray  Kind = "array"
)

// Schema is the minimal shape descriptor for the no-Go-type case: a set of
// required field names, each with its expected basic Kind. It is deliberately
// shallow — required top-level keys and their kinds — for the plain-text
// fallback where there is no struct to coerce into. Unlisted fields are
// ignored, so a model returning extra keys still validates.
//
//	schema := structured.Schema{Fields: map[string]structured.Kind{
//	    "name": structured.KindString,
//	    "age":  structured.KindNumber,
//	}}
type Schema struct {
	Fields map[string]Kind `json:"fields"`
}

// Validate checks raw against schema: raw must be a JSON object containing
// every field in schema.Fields, each of the declared Kind. Returns a typed
// error (scope "structured") on malformed JSON, a non-object root, a missing
// required field, or a field of the wrong kind. An empty schema validates any
// JSON object.
//
//	if err := structured.Validate(raw, schema); err != nil { return err }
func Validate(raw string, schema Schema) error {
	var obj map[string]any
	r := core.JSONUnmarshalString(raw, &obj)
	if !r.OK {
		return core.E("structured", "validate: "+r.Error(), nil)
	}
	if obj == nil {
		// Valid JSON, but not an object (e.g. a JSON array or null decoded into
		// a nil map) — the schema describes object fields, so this is a miss.
		return core.E("structured", "validate: root is not a JSON object", nil)
	}
	for name, want := range schema.Fields {
		val, present := obj[name]
		if !present {
			return core.E("structured", core.Sprintf("validate: missing required field %q", name), nil)
		}
		if !kindMatches(val, want) {
			return core.E("structured", core.Sprintf("validate: field %q wrong kind, want %s", name, string(want)), nil)
		}
	}
	return nil
}

// kindMatches reports whether a decoded JSON value matches the expected Kind.
// encoding/json decodes untyped JSON to: float64 (number), bool, string,
// map[string]any (object), []any (array).
func kindMatches(val any, want Kind) bool {
	switch want {
	case KindString:
		_, ok := val.(string)
		return ok
	case KindNumber:
		_, ok := val.(float64)
		return ok
	case KindBool:
		_, ok := val.(bool)
		return ok
	case KindObject:
		_, ok := val.(map[string]any)
		return ok
	case KindArray:
		_, ok := val.([]any)
		return ok
	default:
		// Unknown kind in the schema — treat as unsatisfiable rather than
		// silently passing, so a typo in the descriptor surfaces.
		return false
	}
}

// Reprompter abstracts the model re-call used to repair a malformed response.
// Reprompt receives the raw text that failed to parse and the parse error, and
// returns a fresh attempt from the model (or an error if the model couldn't be
// reached). It is the seam between this pure parsing package and the provider
// router (§6.2) — the router-backed implementation lives at the host surface.
//
//	type routerReprompt struct{ ... }
//	func (r routerReprompt) Reprompt(prev string, perr error) (string, error) {
//	    return r.router.Chat(repairPrompt(prev, perr))
//	}
type Reprompter interface {
	Reprompt(prevRaw string, parseErr error) (string, error)
}

// ParseWithRepair tries to Parse raw into target; on failure it asks reprompt
// for a fresh response (passing the parse error so the model can correct
// itself) and retries, up to maxAttempts total tries. It returns nil on the
// first success, or the last error when attempts are exhausted.
//
// A nil reprompt (or maxAttempts <= 0) means a single attempt with no repair —
// behaving exactly like Parse. maxAttempts counts every Parse, so maxAttempts
// of 3 is one initial parse plus up to two repair re-prompts.
//
//	var out Plan
//	err := structured.ParseWithRepair(raw, &out, model, 3)
func ParseWithRepair(raw string, target any, reprompt Reprompter, maxAttempts int) error {
	if maxAttempts < 1 {
		maxAttempts = 1
	}

	err := Parse(raw, target)
	if err == nil {
		return nil
	}
	// No repair channel — one shot only.
	if reprompt == nil {
		return err
	}

	current := raw
	// One attempt already spent above; loop the remaining budget.
	for attempt := 1; attempt < maxAttempts; attempt++ {
		next, rErr := reprompt.Reprompt(current, err)
		if rErr != nil {
			// Couldn't reach the model — surface the reprompt failure as the
			// final error, with the last parse error as its cause for context.
			return core.E("structured", "repair: reprompt failed: "+rErr.Error(), err)
		}
		current = next
		err = Parse(current, target)
		if err == nil {
			return nil
		}
	}
	return err
}

// ValidateWithRepair is Validate with the same bounded reprompt-and-retry
// contract as ParseWithRepair, for the schema-descriptor (no Go type) case.
// There is no target out-parameter to coerce into here, so — unlike
// ParseWithRepair, which reports only success/failure — the (possibly
// repaired) raw text travels back to the caller alongside the error: on
// success it is the first attempt that validated; on exhaustion it is the
// last attempt tried, paired with that attempt's error.
//
// A nil reprompt (or maxAttempts <= 0) means a single attempt with no
// repair — behaving exactly like Validate, with raw echoed back unchanged.
// maxAttempts counts every Validate, so maxAttempts of 3 is one initial
// validate plus up to two repair re-prompts.
//
//	fixed, err := structured.ValidateWithRepair(raw, schema, model, 3)
//	if err != nil { return err }
//	// fixed is raw's text once it satisfied schema.
func ValidateWithRepair(raw string, schema Schema, reprompt Reprompter, maxAttempts int) (string, error) {
	if maxAttempts < 1 {
		maxAttempts = 1
	}

	err := Validate(raw, schema)
	if err == nil {
		return raw, nil
	}
	// No repair channel — one shot only.
	if reprompt == nil {
		return raw, err
	}

	current := raw
	// One attempt already spent above; loop the remaining budget.
	for attempt := 1; attempt < maxAttempts; attempt++ {
		next, rErr := reprompt.Reprompt(current, err)
		if rErr != nil {
			// Couldn't reach the model — surface the reprompt failure as the
			// final error, with the last validation error as its cause for context.
			return current, core.E("structured", "repair: reprompt failed: "+rErr.Error(), err)
		}
		current = next
		err = Validate(current, schema)
		if err == nil {
			return current, nil
		}
	}
	return current, err
}
