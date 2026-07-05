// SPDX-Licence-Identifier: EUPL-1.2

package datapipe

import "testing"

// Package-level sinks keep the compiler from eliding benchmarked work.
var (
	benchStmt   LQLStatement
	benchStmts  []LQLStatement
	benchErr    error
	benchTokens []string
	benchStr    string
	benchInt    int
)

const benchLQLScript = `
# research batch
USE "base.vindex";
WALK "same; token in quote" LIMIT 2;
-- compare after walk
DIFF base "base" tuned "fine";
TRACE INFER "why did this fine tune prefer the operator name?";
SELECT layers WHERE kind = attention LIMIT 5;
`

func BenchmarkParseLQL_Use(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchStmt, benchErr = ParseLQL(`USE "models/gemma4-ft.vindex"`)
	}
}

func BenchmarkParseLQL_Walk(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchStmt, benchErr = ParseLQL(`WALK "operator project context" LIMIT 12`)
	}
}

func BenchmarkParseLQL_Diff(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchStmt, benchErr = ParseLQL(`DIFF "base/gemma4" WITH "fine-tunes/project-gemma4" PATCH "findings.patch" LIMIT 8`)
	}
}

func BenchmarkParseLQL_Trace(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchStmt, benchErr = ParseLQL(`TRACE INFER "why did this fine tune prefer the operator name?"`)
	}
}

func BenchmarkParseLQLScript(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchStmts, benchErr = ParseLQLScript(benchLQLScript)
	}
}

func BenchmarkLexLQL(b *testing.B) {
	const q = `DIFF "base/gemma4" WITH "fine-tunes/project-gemma4" PATCH "findings.patch" LIMIT 8`
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchTokens, benchErr = lexLQL(q)
	}
}

func BenchmarkSplitLQLScript(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		var parts []string
		parts, benchErr = splitLQLScript(benchLQLScript)
		benchTokens = parts
	}
}

func BenchmarkLQLLimit(b *testing.B) {
	tokens := []string{"walk", "operator project context", "limit", "12"}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchInt = lqlLimit(tokens)
	}
}

func BenchmarkLQLRest(b *testing.B) {
	tokens := []string{"select", "layers", "where", "kind", "=", "attention"}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchStr = lqlRest(tokens, 1)
	}
}
