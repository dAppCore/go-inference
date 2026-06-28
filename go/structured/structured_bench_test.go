// SPDX-Licence-Identifier: EUPL-1.2

package structured_test

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/structured"
)

// --- fixtures ------------------------------------------------------------

// benchPlan is a realistic typed target for the parse path: a handful of
// scalar fields of mixed kind — the shape a response_format struct target
// takes when the caller has a Go type to coerce into.
type benchPlan struct {
	Title   string  `json:"title"`
	Score   float64 `json:"score"`
	Count   int     `json:"count"`
	Enabled bool    `json:"enabled"`
}

const (
	benchGoodJSON = `{"title":"quarterly plan","score":0.87,"count":12,"enabled":true}`
	benchBadJSON  = `{"title":"quarterly plan","score":` // unterminated object

	// benchNestedJSON exercises the recursive map[string]any decode in
	// Validate: a nested object plus an array that the schema lists by Kind, so
	// the decoder must fully box every level — the realistic JSON-schema shape.
	benchNestedJSON = `{"title":"q","score":1.5,"who":{"id":1,"name":"x"},"tags":["a","b","c"],"enabled":true}`
)

// benchSchema is a realistic descriptor: five required top-level fields across
// every Kind in the vocabulary.
var benchSchema = structured.Schema{Fields: map[string]structured.Kind{
	"title":   structured.KindString,
	"score":   structured.KindNumber,
	"who":     structured.KindObject,
	"tags":    structured.KindArray,
	"enabled": structured.KindBool,
}}

// queueReprompter serves canned payloads, one per Reprompt call — a stand-in
// for the router-backed repair channel, with no per-call allocation of its own
// so the benchmark measures ParseWithRepair, not the double.
type queueReprompter struct {
	payloads []string
	calls    int
}

func (q *queueReprompter) Reprompt(prevRaw string, parseErr error) (string, error) {
	i := q.calls
	q.calls++
	if i >= len(q.payloads) {
		return "", core.E("structuredbench", "exhausted", nil)
	}
	return q.payloads[i], nil
}

// errSink keeps the returned error live so the compiler can't elide the call.
var errSink error

// --- Parse ---------------------------------------------------------------

func BenchmarkParse_Good(b *testing.B) {
	b.ReportAllocs()
	var p benchPlan
	for i := 0; i < b.N; i++ {
		errSink = structured.Parse(benchGoodJSON, &p)
	}
}

func BenchmarkParse_Bad(b *testing.B) {
	b.ReportAllocs()
	var p benchPlan
	for i := 0; i < b.N; i++ {
		errSink = structured.Parse(benchBadJSON, &p)
	}
}

// --- Validate ------------------------------------------------------------

func BenchmarkValidate_Good(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		errSink = structured.Validate(benchNestedJSON, benchSchema)
	}
}

func BenchmarkValidate_MissingField(b *testing.B) {
	b.ReportAllocs()
	// benchGoodJSON lacks the schema's "who"/"tags" — the missing-field branch.
	for i := 0; i < b.N; i++ {
		errSink = structured.Validate(benchGoodJSON, benchSchema)
	}
}

// --- ParseWithRepair -----------------------------------------------------

func BenchmarkParseWithRepair_Success(b *testing.B) {
	b.ReportAllocs()
	var p benchPlan
	for i := 0; i < b.N; i++ {
		errSink = structured.ParseWithRepair(benchGoodJSON, &p, nil, 3)
	}
}

func BenchmarkParseWithRepair_Repair(b *testing.B) {
	b.ReportAllocs()
	var p benchPlan
	// One reprompter reused across iterations (calls reset each loop) so its
	// construction stays out of the per-op allocation count.
	rp := &queueReprompter{payloads: []string{benchGoodJSON}}
	for i := 0; i < b.N; i++ {
		rp.calls = 0
		errSink = structured.ParseWithRepair(benchBadJSON, &p, rp, 3)
	}
}
