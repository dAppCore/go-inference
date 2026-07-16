// SPDX-Licence-Identifier: EUPL-1.2

package structured

import (
	core "dappco.re/go"
)

// ExampleParse demonstrates coercing a model's raw JSON response into a
// typed value.
func ExampleParse() {
	var p person
	if err := Parse(`{"name":"Ada","age":36}`, &p); err != nil {
		core.Println(err)
		return
	}
	core.Println(p.Name)
	core.Println(p.Age)
	// Output:
	// Ada
	// 36
}

// ExampleValidate demonstrates the no-Go-type schema check: raw must be a
// JSON object carrying every declared field at its declared Kind.
func ExampleValidate() {
	schema := Schema{Fields: map[string]Kind{
		"title": KindString,
		"score": KindNumber,
	}}
	err := Validate(`{"title":"report","score":9.5}`, schema)
	core.Println(err == nil)
	// Output:
	// true
}

// ExampleParseWithRepair demonstrates the bounded reprompt-and-retry loop: a
// malformed first response is repaired by re-asking the model, which is
// given the parse error and returns a fixed payload.
func ExampleParseWithRepair() {
	rp := &fakeReprompter{payloads: []string{`{"name":"Ada","age":36}`}}
	var p person
	err := ParseWithRepair(`not json at all`, &p, rp, 3)
	if err != nil {
		core.Println(err)
		return
	}
	core.Println(p.Name)
	core.Println(rp.calls)
	// Output:
	// Ada
	// 1
}

// ExampleValidateWithRepair demonstrates the schema-descriptor sibling of
// ParseWithRepair: on success it returns the (possibly repaired) raw text
// that satisfied the schema.
func ExampleValidateWithRepair() {
	schema := Schema{Fields: map[string]Kind{"title": KindString}}
	rp := &fakeReprompter{payloads: []string{`{"title":"fixed"}`}}

	fixed, err := ValidateWithRepair(`not json`, schema, rp, 3)
	if err != nil {
		core.Println(err)
		return
	}
	core.Println(fixed)
	// Output:
	// {"title":"fixed"}
}
