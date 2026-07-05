// SPDX-Licence-Identifier: EUPL-1.2

package tokenizer

import (
	"testing"
)

// Benchmark coverage for the W11-S lane: every hot tokenizer surface
// except IDToken / DecodeOne (W11-K's territory, already optimised).
// Canonical shapes: short / typical / long prompts; ASCII / SentencePiece
// / special-token boundaries; Greedy decode vs full-stream decode.

// --- Shared fixtures ---------------------------------------------------

func benchTokenizerSP(b *testing.B) *Tokenizer {
	b.Helper()
	// Hand-built tokenizer with a SentencePiece-style vocab + merges.
	// Avoids the LoadTokenizer file-IO path so bench cost is the math
	// under test, not test-fixture overhead.
	tok := &Tokenizer{
		vocab: map[string]int32{
			"<bos>":  100,
			"<eos>":  101,
			"▁":      4,
			"h":      0,
			"e":      1,
			"l":      2,
			"o":      3,
			"w":      8,
			"r":      9,
			"d":      10,
			"he":     5,
			"ll":     6,
			"▁h":     7,
			"hel":    11,
			"hello":  12,
			"▁hello": 13,
			"▁world": 14,
			"world":  15,
			" ":      16,
		},
		invVocab: map[int32]string{
			100: "<bos>", 101: "<eos>",
			0: "h", 1: "e", 2: "l", 3: "o",
			4: "▁", 5: "he", 6: "ll", 7: "▁h",
			8: "w", 9: "r", 10: "d",
			11: "hel", 12: "hello", 13: "▁hello", 14: "▁world",
			15: "world", 16: " ",
		},
		special: map[string]int32{
			"<bos>": 100, "<eos>": 101,
		},
		specialOrder: []string{"<bos>", "<eos>"},
		bosToken:     100, hasBOS: true,
		eosToken: 101, hasEOS: true,
		addPrefixSpace: true,
		mergeRanks: map[mergeKey]int{
			{a: "h", b: "e"}:     0,
			{a: "l", b: "l"}:     1,
			{a: "he", b: "l"}:    2,
			{a: "hel", b: "l"}:   3,
			{a: "hel", b: "lo"}:  4,
			{a: "▁", b: "h"}:     5,
			{a: "▁h", b: "ello"}: 6,
			{a: "▁", b: "w"}:     7,
		},
	}
	return tok
}

// --- Encode benches ---------------------------------------------------

func BenchmarkTokenizer_Encode_Short(b *testing.B) {
	tok := benchTokenizerSP(b)
	text := "hello"
	b.ReportAllocs()
	for b.Loop() {
		_ = tok.Encode(text)
	}
}

func BenchmarkTokenizer_Encode_Typical(b *testing.B) {
	tok := benchTokenizerSP(b)
	text := "hello world hello world hello world"
	b.ReportAllocs()
	for b.Loop() {
		_ = tok.Encode(text)
	}
}

func BenchmarkTokenizer_Encode_WithSpecial(b *testing.B) {
	tok := benchTokenizerSP(b)
	text := "<bos>hello world<eos>"
	b.ReportAllocs()
	for b.Loop() {
		_ = tok.Encode(text)
	}
}

func BenchmarkTokenizer_Encode_LongASCII(b *testing.B) {
	tok := benchTokenizerSP(b)
	// 16-segment prompt — exercises segment-loop + per-segment SP normalisation.
	text := "hello world hello world hello world hello world " +
		"hello world hello world hello world hello world"
	b.ReportAllocs()
	for b.Loop() {
		_ = tok.Encode(text)
	}
}

// --- Decode benches ---------------------------------------------------

func BenchmarkTokenizer_Decode_Short(b *testing.B) {
	tok := benchTokenizerSP(b)
	ids := []int32{5, 6, 3} // "he" + "ll" + "o" → "hello"
	b.ReportAllocs()
	for b.Loop() {
		_ = tok.Decode(ids)
	}
}

func BenchmarkTokenizer_Decode_Typical(b *testing.B) {
	tok := benchTokenizerSP(b)
	// 12-token stream — typical mid-stream Decode call.
	ids := []int32{13, 14, 13, 14, 13, 14, 13, 14, 13, 14, 13, 14}
	b.ReportAllocs()
	for b.Loop() {
		_ = tok.Decode(ids)
	}
}

func BenchmarkTokenizer_Decode_WithSpecials(b *testing.B) {
	tok := benchTokenizerSP(b)
	// BOS + tokens + EOS — specials skipped silently.
	ids := []int32{100, 13, 14, 101}
	b.ReportAllocs()
	for b.Loop() {
		_ = tok.Decode(ids)
	}
}

func BenchmarkTokenizer_Decode_LongStream(b *testing.B) {
	tok := benchTokenizerSP(b)
	// 64-token stream simulating an end-of-generation decode.
	ids := make([]int32, 64)
	src := []int32{13, 14, 5, 6, 3, 12, 15, 4}
	for i := range ids {
		ids[i] = src[i%len(src)]
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = tok.Decode(ids)
	}
}

// --- DecodeToken benches ----------------------------------------------

func BenchmarkTokenizer_DecodeToken_Regular(b *testing.B) {
	tok := benchTokenizerSP(b)
	b.ReportAllocs()
	for b.Loop() {
		_ = tok.DecodeToken(5) // "he"
	}
}

func BenchmarkTokenizer_DecodeToken_Special(b *testing.B) {
	tok := benchTokenizerSP(b)
	b.ReportAllocs()
	for b.Loop() {
		_ = tok.DecodeToken(100) // <bos>, returns ""
	}
}

func BenchmarkTokenizer_DecodeToken_SentencePieceSpace(b *testing.B) {
	tok := benchTokenizerSP(b)
	b.ReportAllocs()
	for b.Loop() {
		_ = tok.DecodeToken(7) // "▁h" → " h"
	}
}

// --- DecodeOne benches ------------------------------------------------
// DecodeOne fires once per emitted generation token via the root-package
// IDToken wrapper. The two dominant shapes are continuation pieces (no ▁
// marker — must stay zero-alloc) and word-leading pieces (leading ▁ — the
// ▁→space→strip round-trip is identity on a substring view, so also
// zero-alloc after the AX-11 marker-aware rewrite).

func BenchmarkTokenizer_DecodeOne_Regular(b *testing.B) {
	tok := benchTokenizerSP(b)
	b.ReportAllocs()
	for b.Loop() {
		_ = tok.DecodeOne(5) // "he" (no marker — continuation piece)
	}
}

func BenchmarkTokenizer_DecodeOne_WordBoundary(b *testing.B) {
	tok := benchTokenizerSP(b)
	b.ReportAllocs()
	for b.Loop() {
		_ = tok.DecodeOne(7) // "▁h" → "h" (leading marker stripped)
	}
}

func BenchmarkTokenizer_DecodeOne_Special(b *testing.B) {
	tok := benchTokenizerSP(b)
	b.ReportAllocs()
	for b.Loop() {
		_ = tok.DecodeOne(100) // <bos>, returns ""
	}
}

// --- Vocab probe benches ----------------------------------------------

func BenchmarkTokenizer_TokenID_Hit(b *testing.B) {
	tok := benchTokenizerSP(b)
	b.ReportAllocs()
	for b.Loop() {
		_, _ = tok.TokenID("hello")
	}
}

func BenchmarkTokenizer_TokenID_Miss(b *testing.B) {
	tok := benchTokenizerSP(b)
	b.ReportAllocs()
	for b.Loop() {
		_, _ = tok.TokenID("zzz_not_in_vocab")
	}
}

// --- bpeMerge benches (BPE inner-loop hot path) -----------------------

func BenchmarkTokenizer_bpeMerge_Short(b *testing.B) {
	tok := benchTokenizerSP(b)
	// Standard "hello" merge — common path.
	b.ReportAllocs()
	for b.Loop() {
		syms := []string{"h", "e", "l", "l", "o"}
		_ = tok.bpeMerge(syms)
	}
}

func BenchmarkTokenizer_bpeMerge_Long(b *testing.B) {
	tok := benchTokenizerSP(b)
	// 16-symbol input — exercises heap-pop loop.
	b.ReportAllocs()
	for b.Loop() {
		syms := []string{
			"▁", "h", "e", "l", "l", "o",
			"▁", "w", "o", "r", "l", "d",
			"h", "e", "l", "l",
		}
		_ = tok.bpeMerge(syms)
	}
}

// --- nextSpecialBoundary bench ----------------------------------------

func BenchmarkTokenizer_nextSpecialBoundary_NoSpecial(b *testing.B) {
	tok := benchTokenizerSP(b)
	text := "hello world hello world"
	b.ReportAllocs()
	for b.Loop() {
		_ = tok.nextSpecialBoundary(text)
	}
}

func BenchmarkTokenizer_nextSpecialBoundary_HasSpecial(b *testing.B) {
	tok := benchTokenizerSP(b)
	text := "hello world <eos> rest"
	b.ReportAllocs()
	for b.Loop() {
		_ = tok.nextSpecialBoundary(text)
	}
}

func BenchmarkTokenizer_matchSpecialToken_Hit(b *testing.B) {
	tok := benchTokenizerSP(b)
	text := "<bos>hello"
	b.ReportAllocs()
	for b.Loop() {
		_, _, _ = tok.matchSpecialToken(text)
	}
}

func BenchmarkTokenizer_matchSpecialToken_Miss(b *testing.B) {
	tok := benchTokenizerSP(b)
	text := "hello world"
	b.ReportAllocs()
	for b.Loop() {
		_, _, _ = tok.matchSpecialToken(text)
	}
}

// --- normalizeSentencePieceSegment bench ------------------------------

func BenchmarkTokenizer_normalizeSP_Short(b *testing.B) {
	tok := benchTokenizerSP(b)
	b.ReportAllocs()
	for b.Loop() {
		_ = tok.normalizeSentencePieceSegment("hello world")
	}
}

func BenchmarkTokenizer_normalizeSP_Long(b *testing.B) {
	tok := benchTokenizerSP(b)
	text := "hello world hello world hello world hello world"
	b.ReportAllocs()
	for b.Loop() {
		_ = tok.normalizeSentencePieceSegment(text)
	}
}

// --- shouldPrependBOS bench -------------------------------------------

func BenchmarkTokenizer_shouldPrependBOS_NoBOS(b *testing.B) {
	tok := benchTokenizerSP(b)
	tok.hasBOS = false
	text := "hello"
	b.ReportAllocs()
	for b.Loop() {
		_ = tok.shouldPrependBOS(text)
	}
}

func BenchmarkTokenizer_shouldPrependBOS_PrefixMatches(b *testing.B) {
	tok := benchTokenizerSP(b)
	tok.invVocab[100] = "<bos>"
	text := "<bos>hello"
	b.ReportAllocs()
	for b.Loop() {
		_ = tok.shouldPrependBOS(text)
	}
}

func BenchmarkTokenizer_shouldPrependBOS_NoMatch(b *testing.B) {
	tok := benchTokenizerSP(b)
	tok.invVocab[100] = "<bos>"
	text := "hello world"
	b.ReportAllocs()
	for b.Loop() {
		_ = tok.shouldPrependBOS(text)
	}
}

// --- IndexIn bench (no-strings replacement) ---------------------------

func BenchmarkTokenizer_IndexIn_Found(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = IndexIn("hello world this is a test string", "test")
	}
}

func BenchmarkTokenizer_IndexIn_NotFound(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = IndexIn("hello world this is a test string", "zzz")
	}
}

// --- buildGPT2ByteMaps bench (one-shot on load) -----------------------

func BenchmarkTokenizer_buildGPT2ByteMaps(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_, _ = buildGPT2ByteMaps()
	}
}

// --- decodeGPT2Bytes bench (per-stream GPT-2 decode) ------------------

func BenchmarkTokenizer_decodeGPT2Bytes(b *testing.B) {
	tok := benchTokenizerSP(b)
	tok.isGPT2BPE = true
	tok.gpt2Decoder, tok.gpt2Encoder = buildGPT2ByteMaps()
	// "Ġhello" — typical Qwen / GPT-2 byte-encoded "▁hello" equivalent.
	s := "Ġhello world"
	b.ReportAllocs()
	for b.Loop() {
		_ = tok.decodeGPT2Bytes(s)
	}
}

// BenchmarkTokenizer_decodeGPT2Bytes_ASCIIContinuation models the dominant
// per-token GPT-2/byte-level shape: a mid-word continuation piece made of
// self-mapped printable ASCII (no leading Ġ space marker). This is the case
// the zero-alloc fast path targets — every emitted continuation token for
// Qwen/GPT/Llama funnels through decodeGPT2Bytes, so a 1->0 here is per-token
// pressure relief, not per-prompt.
func BenchmarkTokenizer_decodeGPT2Bytes_ASCIIContinuation(b *testing.B) {
	tok := benchTokenizerSP(b)
	tok.isGPT2BPE = true
	tok.gpt2Decoder, tok.gpt2Encoder = buildGPT2ByteMaps()
	s := "hello" // pure self-mapped ASCII continuation piece
	b.ReportAllocs()
	for b.Loop() {
		_ = tok.decodeGPT2Bytes(s)
	}
}

// --- encodeSentencePieceSegment bench (cache-miss path) ---------------

func BenchmarkTokenizer_encodeSentencePieceSegment_CacheMiss(b *testing.B) {
	tok := benchTokenizerSP(b)
	b.ReportAllocs()
	for b.Loop() {
		// Clear cache to force the BPE walk; uses a unique key each
		// iteration's bpeCache state to keep miss-path coverage honest.
		tok.bpeCache = nil
		_ = tok.encodeSentencePieceSegment("hello world")
	}
}

func BenchmarkTokenizer_encodeSentencePieceSegment_CacheHit(b *testing.B) {
	tok := benchTokenizerSP(b)
	// Prime the cache.
	_ = tok.encodeSentencePieceSegment("hello world")
	b.ReportAllocs()
	for b.Loop() {
		_ = tok.encodeSentencePieceSegment("hello world")
	}
}

// --- encodeGPT2Segment bench (cache-miss path) ------------------------

func BenchmarkTokenizer_encodeGPT2Segment_CacheMiss(b *testing.B) {
	tok := benchTokenizerSP(b)
	tok.isGPT2BPE = true
	tok.gpt2Decoder, tok.gpt2Encoder = buildGPT2ByteMaps()
	b.ReportAllocs()
	for b.Loop() {
		tok.bpeCache = nil
		_ = tok.encodeGPT2Segment("hello world")
	}
}
