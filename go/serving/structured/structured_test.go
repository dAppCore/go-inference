// SPDX-Licence-Identifier: EUPL-1.2

package structured

import (
	"testing"

	core "dappco.re/go"
)

// --- fixtures ------------------------------------------------------------

// person is the typed target the parser coerces JSON into.
type person struct {
	Name string `json:"name"`
	Age  int    `json:"age"`
}

// fakeReprompter is a test double for Reprompter: it serves a queued list of
// payloads, one per Reprompt call, recording how many times it was asked. A
// nil/empty queue ⇒ Reprompt returns an error (model couldn't be reached).
type fakeReprompter struct {
	payloads []string
	calls    int
}

func (f *fakeReprompter) Reprompt(prevRaw string, parseErr error) (string, error) {
	idx := f.calls
	f.calls++
	if idx >= len(f.payloads) {
		return "", core.E("structuredtest", "no more payloads", nil)
	}
	return f.payloads[idx], nil
}

// --- Parse ---------------------------------------------------------------

func TestStructured_Parse_Good(t *testing.T) {
	var p person
	if err := Parse(`{"name":"Ada","age":36}`, &p); err != nil {
		t.Fatalf("Parse valid JSON: unexpected error: %v", err)
	}
	if p.Name != "Ada" || p.Age != 36 {
		t.Fatalf("Parse coercion: got %+v, want {Ada 36}", p)
	}
}

func TestStructured_Parse_Bad(t *testing.T) {
	// Malformed JSON — unterminated object — must error, not panic.
	var p person
	if err := Parse(`{"name":"Ada", `, &p); err == nil {
		t.Fatal("Parse malformed JSON: expected error, got nil")
	}
}

func TestStructured_Parse_Ugly(t *testing.T) {
	// Wrong type for a field (age as string) — shape mismatch must error.
	var p person
	if err := Parse(`{"name":"Ada","age":"old"}`, &p); err == nil {
		t.Fatal("Parse type mismatch: expected error, got nil")
	}
	// Empty input is not valid JSON for a struct target.
	if err := Parse(``, &p); err == nil {
		t.Fatal("Parse empty input: expected error, got nil")
	}
}

// --- Validate ------------------------------------------------------------

func TestStructured_Validate_Good(t *testing.T) {
	schema := Schema{Fields: map[string]Kind{
		"name": KindString,
		"age":  KindNumber,
		"vip":  KindBool,
	}}
	raw := `{"name":"Ada","age":36,"vip":true,"extra":"ignored"}`
	if err := Validate(raw, schema); err != nil {
		t.Fatalf("Validate matching shape: unexpected error: %v", err)
	}
}

func TestStructured_Validate_Bad(t *testing.T) {
	// Required field "age" missing entirely.
	schema := Schema{Fields: map[string]Kind{
		"name": KindString,
		"age":  KindNumber,
	}}
	if err := Validate(`{"name":"Ada"}`, schema); err == nil {
		t.Fatal("Validate missing required field: expected error, got nil")
	}
}

func TestStructured_Validate_Ugly(t *testing.T) {
	schema := Schema{Fields: map[string]Kind{
		"name": KindString,
		"age":  KindNumber,
	}}
	// Field present but wrong type (age is a string).
	if err := Validate(`{"name":"Ada","age":"old"}`, schema); err == nil {
		t.Fatal("Validate wrong field type: expected error, got nil")
	}
	// Not an object at all.
	if err := Validate(`["Ada",36]`, schema); err == nil {
		t.Fatal("Validate non-object root: expected error, got nil")
	}
	// Malformed JSON.
	if err := Validate(`{"name":`, schema); err == nil {
		t.Fatal("Validate malformed JSON: expected error, got nil")
	}
}

// TestStructured_Validate_Kinds_Good — every Kind in the vocabulary is
// satisfied by its matching JSON value: object and array (the two not exercised
// by the simpler shape test) decode to map[string]any and []any respectively.
func TestStructured_Validate_Kinds_Good(t *testing.T) {
	schema := Schema{Fields: map[string]Kind{
		"who":   KindObject,
		"tags":  KindArray,
		"vip":   KindBool,
		"score": KindNumber,
		"name":  KindString,
	}}
	raw := `{"who":{"id":1},"tags":["a","b"],"vip":false,"score":1.5,"name":"x"}`
	if err := Validate(raw, schema); err != nil {
		t.Fatalf("Validate every kind: unexpected error: %v", err)
	}
}

// TestStructured_Validate_NullRoot_Ugly — valid JSON that decodes to a nil map
// (a literal null) is not an object, so the schema (which describes object
// fields) rejects it rather than treating absent fields as satisfied.
func TestStructured_Validate_NullRoot_Ugly(t *testing.T) {
	schema := Schema{Fields: map[string]Kind{"name": KindString}}
	// JSON null unmarshals successfully into a nil map[string]any — the distinct
	// "valid JSON, not an object" path (separate from a malformed-JSON failure).
	if err := Validate(`null`, schema); err == nil {
		t.Fatal("Validate null root: expected error, got nil")
	}
}

// TestStructured_Validate_UnknownKind_Bad — a kind not in the vocabulary (a typo
// in the descriptor) is unsatisfiable, so even a present field of any JSON type
// fails rather than silently passing.
func TestStructured_Validate_UnknownKind_Bad(t *testing.T) {
	schema := Schema{Fields: map[string]Kind{"name": Kind("stringg")}}
	if err := Validate(`{"name":"Ada"}`, schema); err == nil {
		t.Fatal("Validate unknown kind: expected error, got nil")
	}
}

// --- ParseWithRepair -----------------------------------------------------

func TestStructured_Repair_Good(t *testing.T) {
	// First payload is the raw arg (bad); the reprompter serves a good one on
	// the 2nd attempt. ParseWithRepair must succeed and the struct coerced.
	rp := &fakeReprompter{payloads: []string{`{"name":"Ada","age":36}`}}
	var p person
	err := ParseWithRepair(`not json at all`, &p, rp, 3)
	if err != nil {
		t.Fatalf("ParseWithRepair recovery: unexpected error: %v", err)
	}
	if p.Name != "Ada" || p.Age != 36 {
		t.Fatalf("ParseWithRepair coercion: got %+v, want {Ada 36}", p)
	}
	if rp.calls != 1 {
		t.Fatalf("ParseWithRepair: expected 1 reprompt call, got %d", rp.calls)
	}
}

func TestStructured_Repair_Bad(t *testing.T) {
	// The reprompter keeps serving junk; attempts exhausted ⇒ last error.
	rp := &fakeReprompter{payloads: []string{`still bad`, `also bad`}}
	var p person
	err := ParseWithRepair(`bad`, &p, rp, 3)
	if err == nil {
		t.Fatal("ParseWithRepair exhausted: expected error, got nil")
	}
	// 3 attempts total: 1 initial Parse + 2 repair re-parses ⇒ 2 reprompts.
	if rp.calls != 2 {
		t.Fatalf("ParseWithRepair exhausted: expected 2 reprompt calls, got %d", rp.calls)
	}
}

func TestStructured_Repair_RepromptError_Bad(t *testing.T) {
	// The model can't be reached: the reprompter errors on the first repair call.
	// ParseWithRepair surfaces that reprompt failure immediately (not the parse
	// error), wrapping the last parse error as its cause for context.
	rp := &fakeReprompter{} // empty queue ⇒ first Reprompt returns an error
	var p person
	err := ParseWithRepair(`bad`, &p, rp, 3)
	if err == nil {
		t.Fatal("ParseWithRepair reprompt failure: expected error, got nil")
	}
	// Exactly one reprompt was attempted before the failure short-circuited.
	if rp.calls != 1 {
		t.Fatalf("ParseWithRepair reprompt failure: expected 1 reprompt call, got %d", rp.calls)
	}
}

func TestStructured_Repair_Ugly(t *testing.T) {
	// nil reprompter ⇒ single attempt, no repair. Bad input ⇒ error.
	var p person
	if err := ParseWithRepair(`bad`, &p, nil, 5); err == nil {
		t.Fatal("ParseWithRepair nil reprompter on bad input: expected error, got nil")
	}
	// nil reprompter with good input ⇒ success on the single attempt.
	var p2 person
	if err := ParseWithRepair(`{"name":"Linus","age":54}`, &p2, nil, 5); err != nil {
		t.Fatalf("ParseWithRepair nil reprompter on good input: unexpected error: %v", err)
	}
	if p2.Name != "Linus" {
		t.Fatalf("ParseWithRepair nil reprompter: coercion failed, got %+v", p2)
	}
	// maxAttempts <= 0 with a reprompter ⇒ still a single attempt (bounded).
	rp := &fakeReprompter{payloads: []string{`{"name":"X","age":1}`}}
	var p3 person
	if err := ParseWithRepair(`bad`, &p3, rp, 0); err == nil {
		t.Fatal("ParseWithRepair maxAttempts<=0 on bad input: expected error, got nil")
	}
	if rp.calls != 0 {
		t.Fatalf("ParseWithRepair maxAttempts<=0: expected 0 reprompt calls, got %d", rp.calls)
	}
}
