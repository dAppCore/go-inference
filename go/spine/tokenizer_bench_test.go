// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the root-package Tokenizer wrapper + BOS-stripping
// helpers. Per AX-11 — Encode fires on every prompt entering the
// generation path; Decode fires on every detokenisation at the end
// (and again for `mlx.FilterThinkingTokens`). The BOS-strip helpers
// run on every call, so they show up in the steady-state profile of
// any session that runs lots of short prompts.
//
// Run:    go test -bench='BenchmarkTokenizerCommon' -benchtime=100ms -benchmem -run='^$' ./go

package spine

import "testing"

// Sinks defeat compiler DCE.
var (
	tokenizerBenchSinkInt32s []int32
	tokenizerBenchSinkString string
	tokenizerBenchSinkInt32  int32
	tokenizerBenchSinkBool   bool
	tokenizerBenchSinkErr    error
)

// benchFakeTokenizer is a CPU-only TokenizerImpl that returns
// pre-seeded ID/text vectors. The wrapper code is what we bench;
// the underlying impl just has to be cheap so the wrapper cost
// dominates timing.
type benchFakeTokenizer struct {
	ids        []int32
	text       string
	bos        int32
	bosText    string
	hasBOS     bool
	tokenID    int32
	tokenIDOK  bool
	idTokenStr string
}

func (f *benchFakeTokenizer) Encode(string) []int32 { return f.ids }
func (f *benchFakeTokenizer) Decode([]int32) string { return f.text }
func (f *benchFakeTokenizer) DecodeOne(int32) string {
	// Mirror Decode: the wrapper's IDToken takes whatever DecodeOne returns
	// when non-empty, so for "PlainToken" benches we return the seeded text.
	return f.text
}
func (f *benchFakeTokenizer) TokenID(string) (int32, bool) {
	return f.tokenID, f.tokenIDOK
}
func (f *benchFakeTokenizer) IDToken(id int32) string {
	if f.hasBOS && id == f.bos {
		return f.bosText
	}
	return f.idTokenStr
}
func (f *benchFakeTokenizer) BOS() int32        { return f.bos }
func (f *benchFakeTokenizer) EOS() int32        { return 2 }
func (f *benchFakeTokenizer) HasBOSToken() bool { return f.hasBOS }

// makeTokenIDs builds a synthetic id vector. The leading id is the
// BOS when withBOS=true so stripImplicitBOS exercises its fast-path.
func makeTokenIDs(count int, withBOS bool) []int32 {
	ids := make([]int32, count)
	for i := range ids {
		ids[i] = int32(i + 10)
	}
	if withBOS && count > 0 {
		ids[0] = 1 // matches benchFakeTokenizer.bos
	}
	return ids
}

// --- Encode wrapper — strips implicit BOS without cloning the result ---

func BenchmarkTokenizerCommon_Encode_100Tokens(b *testing.B) {
	ids := makeTokenIDs(100, true)
	tok := &Tokenizer{tok: &benchFakeTokenizer{ids: ids, bos: 1, bosText: "<s>", hasBOS: true}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokenizerBenchSinkInt32s, tokenizerBenchSinkErr = tok.Encode("hello world")
	}
}

func BenchmarkTokenizerCommon_Encode_1000Tokens(b *testing.B) {
	ids := makeTokenIDs(1000, true)
	tok := &Tokenizer{tok: &benchFakeTokenizer{ids: ids, bos: 1, bosText: "<s>", hasBOS: true}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokenizerBenchSinkInt32s, tokenizerBenchSinkErr = tok.Encode("hello world")
	}
}

func BenchmarkTokenizerCommon_Encode_10000Tokens(b *testing.B) {
	ids := makeTokenIDs(10000, true)
	tok := &Tokenizer{tok: &benchFakeTokenizer{ids: ids, bos: 1, bosText: "<s>", hasBOS: true}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokenizerBenchSinkInt32s, tokenizerBenchSinkErr = tok.Encode("hello world")
	}
}

// Encode when the text already carries the BOS prefix — exercises
// the early-return branch where no BOS strip is needed.
func BenchmarkTokenizerCommon_Encode_ExplicitBOSPrefix(b *testing.B) {
	ids := makeTokenIDs(1000, true)
	tok := &Tokenizer{tok: &benchFakeTokenizer{ids: ids, bos: 1, bosText: "<s>", hasBOS: true}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokenizerBenchSinkInt32s, tokenizerBenchSinkErr = tok.Encode("<s>hello world")
	}
}

// Encode against a tokenizer that doesn't carry BOS — exercises
// the "no strip" path.
func BenchmarkTokenizerCommon_Encode_NoBOS(b *testing.B) {
	ids := makeTokenIDs(1000, false)
	tok := &Tokenizer{tok: &benchFakeTokenizer{ids: ids, hasBOS: false}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokenizerBenchSinkInt32s, tokenizerBenchSinkErr = tok.Encode("hello world")
	}
}

// --- Decode wrapper — fires on every detokenisation ---

func BenchmarkTokenizerCommon_Decode_100Tokens(b *testing.B) {
	ids := makeTokenIDs(100, false)
	tok := &Tokenizer{tok: &benchFakeTokenizer{text: "decoded text"}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokenizerBenchSinkString, tokenizerBenchSinkErr = tok.Decode(ids)
	}
}

// --- TokenID — single-lookup fast path + Encode fallback ---

func BenchmarkTokenizerCommon_TokenID_DirectHit(b *testing.B) {
	tok := &Tokenizer{tok: &benchFakeTokenizer{tokenID: 42, tokenIDOK: true}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokenizerBenchSinkInt32, tokenizerBenchSinkBool = tok.TokenID("hello")
	}
}

// Fallback path — direct lookup misses, so the wrapper Encode-then-
// strip-then-len-check fallback runs. This is the slower branch and
// fires whenever the caller asks for a plain-text token without the
// model-native form (e.g. "hello" vs "▁hello").
func BenchmarkTokenizerCommon_TokenID_EncodeFallback(b *testing.B) {
	tok := &Tokenizer{tok: &benchFakeTokenizer{
		tokenID:   0,
		tokenIDOK: false,
		ids:       []int32{1, 42}, // BOS + single token
		bos:       1,
		bosText:   "<s>",
		hasBOS:    true,
	}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokenizerBenchSinkInt32, tokenizerBenchSinkBool = tok.TokenID("hello")
	}
}

// --- IDToken — fires per token in FilterThinkingTokens loop ---

func BenchmarkTokenizerCommon_IDToken_PlainToken(b *testing.B) {
	tok := &Tokenizer{tok: &benchFakeTokenizer{
		idTokenStr: "hello",
		text:       "hello", // Decode([id]) returns this
	}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokenizerBenchSinkString = tok.IDToken(42)
	}
}

func BenchmarkTokenizerCommon_IDToken_EmptyToken(b *testing.B) {
	tok := &Tokenizer{tok: &benchFakeTokenizer{idTokenStr: ""}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokenizerBenchSinkString = tok.IDToken(42)
	}
}

// SentencePiece bare-space token — IDToken returns "▁" from invVocab, the
// DecodeOne fast path returns "" (single "▁" strips to ""), the wrapper falls
// through to the `raw == "▁"` substitution and returns " ". Verifies the
// fallback substitution still fires on the no-allocation path.
func BenchmarkTokenizerCommon_IDToken_SentencePieceSpace(b *testing.B) {
	tok := &Tokenizer{tok: &benchFakeTokenizer{idTokenStr: "▁", text: ""}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokenizerBenchSinkString = tok.IDToken(42)
	}
}

// --- BOS / EOS — cheap accessors, fire across the pipeline ---

func BenchmarkTokenizerCommon_BOS(b *testing.B) {
	tok := &Tokenizer{tok: &benchFakeTokenizer{bos: 1}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokenizerBenchSinkInt32 = tok.BOS()
	}
}

func BenchmarkTokenizerCommon_EOS(b *testing.B) {
	tok := &Tokenizer{tok: &benchFakeTokenizer{}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokenizerBenchSinkInt32 = tok.EOS()
	}
}

// --- Strip helpers — internal, but the inner loop of Encode ---

func BenchmarkTokenizerCommon_StripImplicitBOS_WithBOS(b *testing.B) {
	tok := &benchFakeTokenizer{bos: 1, bosText: "<s>", hasBOS: true}
	ids := makeTokenIDs(1000, true)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokenizerBenchSinkInt32s = stripImplicitBOS(tok, ids)
	}
}

func BenchmarkTokenizerCommon_StripImplicitBOS_NoBOS(b *testing.B) {
	tok := &benchFakeTokenizer{hasBOS: false}
	ids := makeTokenIDs(1000, false)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokenizerBenchSinkInt32s = stripImplicitBOS(tok, ids)
	}
}

func BenchmarkTokenizerCommon_HasExplicitBOSPrefix_True(b *testing.B) {
	tok := &benchFakeTokenizer{bos: 1, bosText: "<s>", hasBOS: true}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokenizerBenchSinkBool = hasExplicitBOSPrefix(tok, "<s>hello world")
	}
}

func BenchmarkTokenizerCommon_HasExplicitBOSPrefix_False(b *testing.B) {
	tok := &benchFakeTokenizer{bos: 1, bosText: "<s>", hasBOS: true}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokenizerBenchSinkBool = hasExplicitBOSPrefix(tok, "hello world")
	}
}
