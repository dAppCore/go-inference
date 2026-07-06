// SPDX-Licence-Identifier: EUPL-1.2

// Deeper TokensText + token-surface benchmarks. The existing bench
// suite covers all-Text streams; this file adds mixed Text+Value
// (the tokenizer-emitting-both case some drivers see), all-Value
// (when the tokenizer can't render UTF-8 but can emit byte
// sequences), tokens-with-whitespace-only (hasNonSpace tight loop),
// and tokens-with-Unicode-whitespace (the multi-byte core.Trim
// fallback path).
//
// Per AX-11 — TokensText runs once per Speculative + PromptLookup
// call but iterates the whole stream twice (pre-grow walk + write
// walk). The hot loop is tokenSurface → hasNonSpace, which has a
// fast ASCII path and a slower multi-byte path. Coverage on those
// two paths is the difference between knowing the cost and guessing.
//
// Run:    go test -bench='BenchmarkDecode_TokensTextDeep' -benchmem -run='^$' ./go/decode

package decode

import (
	"testing"
)

// buildDecodeTokensMixedTextValue mints n Tokens where half carry
// Text and half carry only Value — the tokenSurface fallback path
// triggers on every Value-only token. The existing all-Text and
// all-Value benches cover the pure paths; this one stresses the
// branch density and shows whether the fallback adds measurable
// per-token cost.
func buildDecodeTokensMixedTextValue(n int) []Token {
	tokens := make([]Token, n)
	for i := range n {
		if i%2 == 0 {
			tokens[i] = Token{ID: int32(i + 1), Text: "tok"}
		} else {
			tokens[i] = Token{ID: int32(i + 1), Value: "tok"}
		}
	}
	return tokens
}

// buildDecodeTokensAllValueOnly mints n Tokens where Text is empty
// and only Value is populated — the path some byte-sequence-only
// tokenizers (raw BPE, some classification heads) take. Stresses
// the tokenSurface Text-empty fallthrough.
func buildDecodeTokensAllValueOnly(n int) []Token {
	tokens := make([]Token, n)
	for i := range n {
		tokens[i] = Token{ID: int32(i + 1), Value: "tok"}
	}
	return tokens
}

// buildDecodeTokensWhitespaceOnly mints n Tokens whose Text is a
// pure-whitespace ASCII string — exercises the hasNonSpace inner
// loop where every byte is the "skip" case, forcing the longest
// straight-line read. Sentinel pattern for stride-of-whitespace
// content (markdown, structured output).
func buildDecodeTokensWhitespaceOnly(n int) []Token {
	tokens := make([]Token, n)
	for i := range n {
		tokens[i] = Token{ID: int32(i + 1), Text: "   \t\n"}
	}
	return tokens
}

// buildDecodeTokensUnicodeWhitespace mints n Tokens whose Text is
// a non-breaking-space character (U+00A0, multi-byte UTF-8). Forces
// hasNonSpace into the core.Trim fallback on every token — the only
// reliable way to see that path's cost in isolation.
func buildDecodeTokensUnicodeWhitespace(n int) []Token {
	tokens := make([]Token, n)
	for i := range n {
		tokens[i] = Token{ID: int32(i + 1), Text: "  "}
	}
	return tokens
}

// buildDecodeTokensVariableLength mints n Tokens whose Text varies
// in length (1, 4, 16, 64 bytes cycled). Real token streams vary
// by ~2 orders of magnitude — bench against that, not against the
// constant-3-byte happy path.
func buildDecodeTokensVariableLength(n int) []Token {
	lengths := []int{1, 4, 16, 64}
	tokens := make([]Token, n)
	for i := range n {
		size := lengths[i%len(lengths)]
		buf := make([]byte, size)
		for j := range size {
			buf[j] = byte('a' + (i % 26))
		}
		tokens[i] = Token{ID: int32(i + 1), Text: string(buf)}
	}
	return tokens
}

// --- TokensText over mixed / Value-only / whitespace / Unicode ---

func BenchmarkDecode_TokensTextDeep_MixedTextValue_256(b *testing.B) {
	tokens := buildDecodeTokensMixedTextValue(256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkText = TokensText(tokens)
	}
}

func BenchmarkDecode_TokensTextDeep_MixedTextValue_2048(b *testing.B) {
	tokens := buildDecodeTokensMixedTextValue(2048)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkText = TokensText(tokens)
	}
}

func BenchmarkDecode_TokensTextDeep_AllValueOnly_256(b *testing.B) {
	tokens := buildDecodeTokensAllValueOnly(256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkText = TokensText(tokens)
	}
}

func BenchmarkDecode_TokensTextDeep_VariableLength_256(b *testing.B) {
	tokens := buildDecodeTokensVariableLength(256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkText = TokensText(tokens)
	}
}

// --- TokenEqual surface-form edges ---

// BothValueOnlyEqual — tokens carry only Value, the same Value;
// TokenEqual must agree but takes the Value-side branch.
func BenchmarkDecode_TokensTextDeep_TokenEqual_BothValueOnly(b *testing.B) {
	a := Token{ID: 1, Value: "abcdef"}
	c := Token{ID: 1, Value: "abcdef"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkBool = TokenEqual(a, c)
	}
}

// TextMismatch — IDs agree but Text strings differ. Forces the full
// string compare to reach the not-equal verdict. The existing benches
// cover the equal and ID-mismatch cases; this is the
// always-runs-the-compare path.
func BenchmarkDecode_TokensTextDeep_TokenEqual_TextMismatch(b *testing.B) {
	a := Token{ID: 1, Text: "abcdef"}
	c := Token{ID: 1, Text: "abcxyz"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkBool = TokenEqual(a, c)
	}
}

// LongTextEqual — typical chat token is ~3 bytes, but punctuation
// runs and code-block tokens can hit 32+. Tests the strcmp path
// at a length closer to worst-case.
func BenchmarkDecode_TokensTextDeep_TokenEqual_LongTextEqual(b *testing.B) {
	a := Token{ID: 1, Text: "abcdefghijklmnopqrstuvwxyz0123456"}
	c := Token{ID: 1, Text: "abcdefghijklmnopqrstuvwxyz0123456"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkBool = TokenEqual(a, c)
	}
}

// WhitespaceOnlyTextSkipsCompare — text is whitespace-only on
// both sides; tokenSurface treats them as "empty" via hasNonSpace
// and the compare short-circuits to true. The skip-compare branch
// at non-empty-but-meaningless input.
func BenchmarkDecode_TokensTextDeep_TokenEqual_WhitespaceOnlyTextSkipsCompare(b *testing.B) {
	a := Token{ID: 1, Text: "   \t\n"}
	c := Token{ID: 1, Text: "\r\n  "}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkBool = TokenEqual(a, c)
	}
}

// UnicodeWhitespaceSkipsCompare — multi-byte whitespace forces the
// hasNonSpace core.Trim fallback; tokenSurface still resolves to
// "empty" and the compare short-circuits. Validates the slow path
// reaches the same answer as the fast path.
func BenchmarkDecode_TokensTextDeep_TokenEqual_UnicodeWhitespaceSkipsCompare(b *testing.B) {
	a := Token{ID: 1, Text: "  "}
	c := Token{ID: 1, Text: "　"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeSinkBool = TokenEqual(a, c)
	}
}
