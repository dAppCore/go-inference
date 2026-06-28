// SPDX-Licence-Identifier: EUPL-1.2

package score

import "testing"

// Per-LLM-call judge helpers not covered by benchmark_test.go. extractJSON
// is already covered; firstJSONObject and normalizeBenchmarkName run on
// every ingest row.

func BenchmarkFirstJSONObject_RawJSON(b *testing.B) {
	input := `{"sovereignty": 8, "ethical_depth": 7, "creative_expression": 6}`
	b.ReportAllocs()
	for b.Loop() {
		firstJSONObject(input)
	}
}

func BenchmarkFirstJSONObject_WithPreamble(b *testing.B) {
	input := `Some preamble text here {"a": 1, "b": {"c": 2}} trailing notes that get ignored.`
	b.ReportAllocs()
	for b.Loop() {
		firstJSONObject(input)
	}
}

func BenchmarkFirstJSONObject_Nested(b *testing.B) {
	input := `{"outer": {"middle": {"inner": {"deep": 1}}}, "extra": [1,2,3]}`
	b.ReportAllocs()
	for b.Loop() {
		firstJSONObject(input)
	}
}

func BenchmarkFirstJSONObject_NoJSON(b *testing.B) {
	input := "No JSON here, just plain prose explaining the scoring rationale."
	b.ReportAllocs()
	for b.Loop() {
		firstJSONObject(input)
	}
}

func BenchmarkNormalizeBenchmarkName_Simple(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		sinkString = normalizeBenchmarkName("truthfulqa")
	}
}

func BenchmarkNormalizeBenchmarkName_Messy(b *testing.B) {
	// Mixed-case + spaces + underscores + hyphens — the real ingest shape.
	b.ReportAllocs()
	for b.Loop() {
		sinkString = normalizeBenchmarkName(" Truthful_QA-V2 ")
	}
}

// extractJSON runs on every judge LLM response. Both paths are benched:
// the markdown-code-block path (the regex branch) and the plain-text path
// that falls through to firstJSONObject.

func BenchmarkExtractJSON_CodeBlock(b *testing.B) {
	input := "Here are the scores:\n```json\n" +
		`{"sovereignty": 8, "ethical_depth": 7, "creative_expression": 6}` +
		"\n```\nThat's my assessment."
	b.ReportAllocs()
	for b.Loop() {
		sinkString = extractJSON(input)
	}
}

func BenchmarkExtractJSON_Plain(b *testing.B) {
	input := `Some preamble {"sovereignty": 8, "ethical_depth": 7} and trailing prose.`
	b.ReportAllocs()
	for b.Loop() {
		sinkString = extractJSON(input)
	}
}
