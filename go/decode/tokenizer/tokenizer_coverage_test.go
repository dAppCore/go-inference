// SPDX-Licence-Identifier: EUPL-1.2

package tokenizer

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// gpt2TokenizerJSON is a minimal GPT-2 byte-level BPE tokenizer. The presence
// of "Ġthe" in the vocab is what flips Tokenizer.isGPT2BPE on, routing Encode/
// Decode/DecodeToken/DecodeOne through the byte-level path. The vocab carries
// the byte-encoded forms of the characters in "the"/" the" (space → Ġ, U+0120)
// plus single-char leaf tokens so a bare "t"/"h"/"e" round-trips, and one merge
// ("t h" → "th") so the GPT-2 BPE merge step has work to do.
const gpt2TokenizerJSON = `{
  "model": {
    "type": "BPE",
    "vocab": {
      "t": 0,
      "h": 1,
      "e": 2,
      "th": 3,
      "the": 4,
      "Ġ": 5,
      "Ġt": 6,
      "Ġthe": 7,
      "x": 8,
      "Ġth": 9
    },
    "merges": ["Ġ t", "Ġt h", "Ġth e", "t h", "th e"],
    "byte_fallback": false
  },
  "added_tokens": [
    {"id": 200, "content": "<|endoftext|>", "special": true}
  ]
}`

// arrayMergesTokenizerJSON exercises the [["a","b"], ...] merges form (the
// alternate HuggingFace encoding) instead of the ["a b", ...] string form.
const arrayMergesTokenizerJSON = `{
  "model": {
    "type": "BPE",
    "vocab": {
      "h": 0,
      "e": 1,
      "l": 2,
      "o": 3,
      "▁": 4,
      "he": 5,
      "ll": 6
    },
    "merges": [["h", "e"], ["l", "l"]],
    "byte_fallback": false
  },
  "added_tokens": []
}`

// qwenLlamaSpecialsTokenizerJSON carries the Qwen3 / Llama-3 special tokens so
// the corresponding BOS/EOS-assignment branches in LoadTokenizer fire:
// <|im_start|> (BOS), <|im_end|> (EOS), <|begin_of_text|> (BOS), <|eot_id|>
// (EOS). The later assignments win, matching production precedence.
const qwenLlamaSpecialsTokenizerJSON = `{
  "model": {
    "type": "BPE",
    "vocab": {"h": 0, "i": 1},
    "merges": []
  },
  "added_tokens": [
    {"id": 1, "content": "<|im_start|>", "special": true},
    {"id": 2, "content": "<|im_end|>", "special": true},
    {"id": 3, "content": "<|begin_of_text|>", "special": true},
    {"id": 4, "content": "<|eot_id|>", "special": true}
  ]
}`

// gpt2WithBOSTokenizerJSON is a GPT-2 byte-level tokenizer that also defines a
// Llama-3 BOS special (<|begin_of_text|>) and a generic special token, so the
// GPT-2 encode path exercises BOS prepending and in-loop special matching.
const gpt2WithBOSTokenizerJSON = `{
  "model": {
    "type": "BPE",
    "vocab": {
      "t": 0,
      "h": 1,
      "e": 2,
      "th": 3,
      "the": 4,
      "Ġ": 5,
      "Ġthe": 7,
      "x": 8
    },
    "merges": ["t h", "th e"],
    "byte_fallback": false
  },
  "added_tokens": [
    {"id": 128000, "content": "<|begin_of_text|>", "special": true},
    {"id": 128001, "content": "<|sep|>", "special": true}
  ]
}`

// gemmaEndOfTurnTokenizerJSON defines <end_of_turn> so the Gemma EOS-assignment
// branch (which overrides any prior <eos>) fires during load.
const gemmaEndOfTurnTokenizerJSON = `{
  "model": {"type": "BPE", "vocab": {"h": 0}, "merges": []},
  "added_tokens": [
    {"id": 1, "content": "<eos>", "special": true},
    {"id": 107, "content": "<end_of_turn>", "special": true}
  ]
}`

// nonIntegerVocabJSON has a vocab whose values are strings, not integers. It
// re-marshals fine (it parsed from JSON) but fails to unmarshal into
// map[string]int32, exercising the "parse vocab" error path in LoadTokenizer.
const nonIntegerVocabJSON = `{
  "model": {
    "type": "BPE",
    "vocab": {"h": "not-an-int", "e": "also-not"},
    "merges": []
  },
  "added_tokens": []
}`

func writeTokenizerJSON(t *testing.T, body string) string {
	t.Helper()
	dir := t.TempDir()
	path := core.JoinPath(dir, "tokenizer.json")
	if err := coreio.Local.Write(path, body); err != nil {
		t.Fatalf("write tokenizer json: %v", err)
	}
	return path
}

// --- NewForDecode ---

// TestTokenizer_NewForDecode_Good builds a decode-only tokenizer from an
// inverse vocab and confirms DecodeToken/IDToken work without a full load.
func TestTokenizer_NewForDecode_Good(t *testing.T) {
	tok := NewForDecode(map[int32]string{5: "he", 6: "ll", 3: "o"})
	if tok == nil {
		t.Fatal("NewForDecode returned nil")
	}
	if got := tok.DecodeToken(5); got != "he" {
		t.Errorf("DecodeToken(5) = %q, want %q", got, "he")
	}
	if got := tok.IDToken(6); got != "ll" {
		t.Errorf("IDToken(6) = %q, want %q", got, "ll")
	}
	if got := tok.Decode([]int32{5, 6, 3}); got != "hello" {
		t.Errorf("Decode = %q, want %q", got, "hello")
	}
}

// TestTokenizer_NewForDecode_Nil_Ugly: a nil inverse vocab yields an empty,
// usable tokenizer (no panic, empty results).
func TestTokenizer_NewForDecode_Nil_Ugly(t *testing.T) {
	tok := NewForDecode(nil)
	if tok == nil {
		t.Fatal("NewForDecode(nil) returned nil")
	}
	if got := tok.DecodeToken(1); got != "" {
		t.Errorf("DecodeToken on empty = %q, want empty", got)
	}
}

// --- GPT-2 byte-level path (detection + encode + decode round-trip) ---

// TestTokenizer_GPT2_Detected_Good: a vocab containing "Ġthe" flips the
// tokenizer into GPT-2 byte-level BPE mode and builds the byte maps.
func TestTokenizer_GPT2_Detected_Good(t *testing.T) {
	tok, err := LoadTokenizer(writeTokenizerJSON(t, gpt2TokenizerJSON))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	if !tok.isGPT2BPE {
		t.Fatal("expected isGPT2BPE = true for vocab with Ġthe")
	}
	if len(tok.gpt2Encoder) != 256 || len(tok.gpt2Decoder) != 256 {
		t.Fatalf("byte maps not built: enc=%d dec=%d", len(tok.gpt2Encoder), len(tok.gpt2Decoder))
	}
}

// TestTokenizer_GPT2_EncodeDecodeRoundTrip_Good exercises encodeGPT2 +
// encodeGPT2Segment + the BPE merge step + the GPT-2 Decode branch.
// "the" → byte-encodes to "the" → merges t+h→th, th+e→the → vocab id 4.
// " the" (leading space) → "Ġthe" → id 7.
func TestTokenizer_GPT2_EncodeDecodeRoundTrip_Good(t *testing.T) {
	tok, err := LoadTokenizer(writeTokenizerJSON(t, gpt2TokenizerJSON))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}

	ids := tok.Encode("the")
	want := []int32{4}
	if len(ids) != len(want) || ids[0] != want[0] {
		t.Fatalf("Encode(\"the\") = %v, want %v", ids, want)
	}
	if dec := tok.Decode(ids); dec != "the" {
		t.Errorf("Decode(Encode(\"the\")) = %q, want %q", dec, "the")
	}

	// Leading-space form → Ġthe (id 7).
	ids2 := tok.Encode(" the")
	if len(ids2) != 1 || ids2[0] != 7 {
		t.Fatalf("Encode(\" the\") = %v, want [7]", ids2)
	}
	if dec := tok.Decode(ids2); dec != " the" {
		t.Errorf("Decode(Encode(\" the\")) = %q, want %q", dec, " the")
	}
}

// TestTokenizer_GPT2_DecodeToken_Good: single-token GPT-2 decode (DecodeToken
// and DecodeOne both route through decodeGPT2Bytes).
func TestTokenizer_GPT2_DecodeToken_Good(t *testing.T) {
	tok, err := LoadTokenizer(writeTokenizerJSON(t, gpt2TokenizerJSON))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	if got := tok.DecodeToken(7); got != " the" {
		t.Errorf("DecodeToken(7) = %q, want %q", got, " the")
	}
	if got := tok.DecodeOne(7); got != " the" {
		t.Errorf("DecodeOne(7) = %q, want %q", got, " the")
	}
}

// TestTokenizer_GPT2_DecodeSkipsSpecial_Good: the special <|endoftext|> token
// is skipped in the GPT-2 Decode path (special-skip branch).
func TestTokenizer_GPT2_DecodeSkipsSpecial_Good(t *testing.T) {
	tok, err := LoadTokenizer(writeTokenizerJSON(t, gpt2TokenizerJSON))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	// 200 = <|endoftext|> special; 4 = "the". Special is dropped.
	if got := tok.Decode([]int32{200, 4, 200}); got != "the" {
		t.Errorf("Decode with specials = %q, want %q", got, "the")
	}
	// DecodeToken on the special returns empty (not a channel marker).
	if got := tok.DecodeToken(200); got != "" {
		t.Errorf("DecodeToken(special) = %q, want empty", got)
	}
}

// TestTokenizer_GPT2_EncodeSegment_Empty_Ugly: encodeGPT2Segment on an empty
// segment short-circuits to nil.
func TestTokenizer_GPT2_EncodeSegment_Empty_Ugly(t *testing.T) {
	tok, err := LoadTokenizer(writeTokenizerJSON(t, gpt2TokenizerJSON))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	if got := tok.encodeGPT2Segment(""); got != nil {
		t.Errorf("encodeGPT2Segment(\"\") = %v, want nil", got)
	}
}

// TestTokenizer_GPT2_EncodeCaches_Good: a repeated GPT-2 segment is served from
// the BPE cache the second time (storeBPETokens + cachedBPETokens on the gpt2
// key prefix).
func TestTokenizer_GPT2_EncodeCaches_Good(t *testing.T) {
	tok, err := LoadTokenizer(writeTokenizerJSON(t, gpt2TokenizerJSON))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	first := tok.Encode("the")
	if len(tok.bpeCache) == 0 {
		t.Fatal("expected a cached GPT-2 segment after first Encode")
	}
	second := tok.Encode("the")
	if len(first) != len(second) || first[0] != second[0] {
		t.Fatalf("cached GPT-2 encode mismatch: %v vs %v", first, second)
	}
}

// --- LoadTokenizer alternate forms & special-token branches ---

// TestTokenizer_LoadTokenizer_ArrayMerges_Good: the [["a","b"]] merges form is
// parsed identically to the "a b" string form.
func TestTokenizer_LoadTokenizer_ArrayMerges_Good(t *testing.T) {
	tok, err := LoadTokenizer(writeTokenizerJSON(t, arrayMergesTokenizerJSON))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	if len(tok.merges) != 2 {
		t.Fatalf("merges = %d, want 2 (array form parsed)", len(tok.merges))
	}
	// "hello" with the same merges as the string-form fixture: ▁ + he + ll + o.
	got := tok.Encode("hello")
	want := []int32{4, 5, 6, 3}
	if len(got) != len(want) {
		t.Fatalf("Encode(\"hello\") = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

// TestTokenizer_LoadTokenizer_QwenLlamaSpecials_Good: Qwen3 / Llama-3 special
// tokens drive the BOS/EOS assignment branches. Production applies them in
// order, so the last BOS/EOS assignment wins.
func TestTokenizer_LoadTokenizer_QwenLlamaSpecials_Good(t *testing.T) {
	tok, err := LoadTokenizer(writeTokenizerJSON(t, qwenLlamaSpecialsTokenizerJSON))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	if !tok.HasBOSToken() || !tok.HasEOSToken() {
		t.Fatalf("expected BOS and EOS present, got bos=%t eos=%t", tok.HasBOSToken(), tok.HasEOSToken())
	}
	// <|begin_of_text|> (id 3) is the last BOS assignment → wins over <|im_start|>.
	if tok.BOSToken() != 3 {
		t.Errorf("BOSToken() = %d, want 3 (<|begin_of_text|> wins)", tok.BOSToken())
	}
	// <|eot_id|> (id 4) is the last EOS assignment → wins over <|im_end|>.
	if tok.EOSToken() != 4 {
		t.Errorf("EOSToken() = %d, want 4 (<|eot_id|> wins)", tok.EOSToken())
	}
}

// TestTokenizer_LoadTokenizer_NonIntegerVocab_Bad: a vocab with string values
// re-marshals but fails to parse into map[string]int32 → error.
func TestTokenizer_LoadTokenizer_NonIntegerVocab_Bad(t *testing.T) {
	_, err := LoadTokenizer(writeTokenizerJSON(t, nonIntegerVocabJSON))
	if err == nil {
		t.Error("expected error for non-integer vocab values")
	}
}

// TestTokenizer_LoadTokenizer_SpecialOrderSort_Good: specials of equal length
// sort lexicographically; this drives both the a<b and a>b comparator arms.
// Three same-length specials ("<aa>","<bb>","<cc>") guarantee both orders are
// compared during the sort.
func TestTokenizer_LoadTokenizer_SpecialOrderSort_Good(t *testing.T) {
	const body = `{
	  "model": {"type": "BPE", "vocab": {"h": 0}, "merges": []},
	  "added_tokens": [
	    {"id": 10, "content": "<cc>", "special": true},
	    {"id": 11, "content": "<aa>", "special": true},
	    {"id": 12, "content": "<bb>", "special": true}
	  ]
	}`
	tok, err := LoadTokenizer(writeTokenizerJSON(t, body))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	// Equal length → ascending lexicographic order.
	want := []string{"<aa>", "<bb>", "<cc>"}
	if len(tok.specialOrder) != len(want) {
		t.Fatalf("specialOrder = %v, want %v", tok.specialOrder, want)
	}
	for i := range want {
		if tok.specialOrder[i] != want[i] {
			t.Fatalf("specialOrder[%d] = %q, want %q", i, tok.specialOrder[i], want[i])
		}
	}
}

// --- normalizeSentencePieceSegment edge cases ---

// TestTokenizer_NormalizeSP_Empty_Ugly: an empty segment returns "" directly.
func TestTokenizer_NormalizeSP_Empty_Ugly(t *testing.T) {
	tok := &Tokenizer{addPrefixSpace: true}
	if got := tok.normalizeSentencePieceSegment(""); got != "" {
		t.Errorf("normalizeSentencePieceSegment(\"\") = %q, want empty", got)
	}
}

// TestTokenizer_NormalizeSP_AlreadyPrefixed_Good: a segment already starting
// with the ▁ leader (U+2581, bytes E2 96 81) does NOT get a second prefix.
func TestTokenizer_NormalizeSP_AlreadyPrefixed_Good(t *testing.T) {
	tok := &Tokenizer{addPrefixSpace: true}
	// "▁hi" — first rune is already ▁, so no extra prefix is added; with no
	// inner spaces the input passes through unchanged.
	in := "▁hi"
	if got := tok.normalizeSentencePieceSegment(in); got != in {
		t.Errorf("normalizeSentencePieceSegment(%q) = %q, want unchanged", in, got)
	}
}

// TestTokenizer_NormalizeSP_AlreadyPrefixedWithSpace_Good: already ▁-prefixed
// AND containing an inner space — needPrefix is false but the space still
// triggers the Builder path (space → ▁).
func TestTokenizer_NormalizeSP_AlreadyPrefixedWithSpace_Good(t *testing.T) {
	tok := &Tokenizer{addPrefixSpace: true}
	in := "▁a b"
	want := "▁a▁b"
	if got := tok.normalizeSentencePieceSegment(in); got != want {
		t.Errorf("normalizeSentencePieceSegment(%q) = %q, want %q", in, got, want)
	}
}

// TestTokenizer_NormalizeSP_NoPrefixSpace_Good: with addPrefixSpace=false the
// leading ▁ is never added; inner spaces still translate.
func TestTokenizer_NormalizeSP_NoPrefixSpace_Good(t *testing.T) {
	tok := &Tokenizer{addPrefixSpace: false}
	if got := tok.normalizeSentencePieceSegment("ab"); got != "ab" {
		t.Errorf("no-prefix plain = %q, want %q", got, "ab")
	}
	if got := tok.normalizeSentencePieceSegment("a b"); got != "a▁b" {
		t.Errorf("no-prefix with space = %q, want %q", got, "a▁b")
	}
}

// --- encodeSentencePieceSegment empty-normalisation ---

// TestTokenizer_EncodeSPSegment_EmptyResult_Ugly: a segment that normalises to
// "" (empty input, addPrefixSpace off) returns nil.
func TestTokenizer_EncodeSPSegment_EmptyResult_Ugly(t *testing.T) {
	tok := &Tokenizer{addPrefixSpace: false, vocab: map[string]int32{}, mergeRanks: map[mergeKey]int{}}
	if got := tok.encodeSentencePieceSegment(""); got != nil {
		t.Errorf("encodeSentencePieceSegment(\"\") = %v, want nil", got)
	}
}

// --- splitRunes (multi-byte, invalid, truncated) ---

// TestTokenizer_SplitRunes_MultiByte_Good covers 2/3/4-byte runes and an ASCII
// mix. splitRunes is unexported; call it directly (white-box).
func TestTokenizer_SplitRunes_MultiByte_Good(t *testing.T) {
	// 'a' (1) + 'é' (2, C3 A9) + '€' (3, E2 82 AC) + '😀' (4, F0 9F 98 80).
	got := splitRunes(nil, "aé€\U0001f600")
	want := []string{"a", "é", "€", "\U0001f600"}
	if len(got) != len(want) {
		t.Fatalf("splitRunes = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("splitRunes[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

// TestTokenizer_SplitRunes_InvalidLeadByte_Ugly: a 0x80 continuation byte with
// no leader is emitted as a single byte (the default branch).
func TestTokenizer_SplitRunes_InvalidLeadByte_Ugly(t *testing.T) {
	got := splitRunes(nil, string([]byte{0x80, 'a'}))
	if len(got) != 2 {
		t.Fatalf("splitRunes(invalid) = %v, want 2 elements", got)
	}
	if got[0] != string([]byte{0x80}) || got[1] != "a" {
		t.Errorf("splitRunes(invalid) = %q", got)
	}
}

// TestTokenizer_SplitRunes_TruncatedMultiByte_Ugly: a multi-byte leader at the
// end of the string with too few trailing bytes is clamped to the remaining
// length (the i+n > len(s) guard).
func TestTokenizer_SplitRunes_TruncatedMultiByte_Ugly(t *testing.T) {
	// 0xF0 expects a 4-byte rune but only 2 bytes remain → clamp to 2.
	got := splitRunes(nil, string([]byte{0xF0, 0x9F}))
	if len(got) != 1 {
		t.Fatalf("splitRunes(truncated) = %v, want 1 element", got)
	}
	if got[0] != string([]byte{0xF0, 0x9F}) {
		t.Errorf("splitRunes(truncated)[0] = %x, want F0 9F", got[0])
	}
}

// TestTokenizer_SplitRunes_AllLengths_Ugly walks 2- and 3-byte leaders that are
// well-formed and confirms each length switch arm is taken.
func TestTokenizer_SplitRunes_AllLengths_Ugly(t *testing.T) {
	// 2-byte (0xC0 leader, here C2 A0 = NBSP) then 3-byte (E2 80 99).
	got := splitRunes(nil, string([]byte{0xC2, 0xA0, 0xE2, 0x80, 0x99}))
	want := []string{string([]byte{0xC2, 0xA0}), string([]byte{0xE2, 0x80, 0x99})}
	if len(got) != len(want) {
		t.Fatalf("splitRunes = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("splitRunes[%d] = %x, want %x", i, got[i], want[i])
		}
	}
}

// --- storeBPETokens: oversized skip, existing-key update, LRU eviction ---

// TestTokenizer_StoreBPETokens_OversizedSkipped_Ugly: a key over the segment
// byte cap, or a token slice over the token cap, is silently not cached.
func TestTokenizer_StoreBPETokens_OversizedSkipped_Ugly(t *testing.T) {
	tok := &Tokenizer{}

	bigKey := string(make([]byte, tokenizerBPECacheMaxSegmentBytes+1))
	tok.storeBPETokens(bigKey, []int32{1})
	if len(tok.bpeCache) != 0 {
		t.Errorf("oversized key cached: %d entries, want 0", len(tok.bpeCache))
	}

	bigTokens := make([]int32, tokenizerBPECacheMaxTokens+1)
	tok.storeBPETokens("k", bigTokens)
	if len(tok.bpeCache) != 0 {
		t.Errorf("oversized token slice cached: %d entries, want 0", len(tok.bpeCache))
	}
}

// TestTokenizer_StoreBPETokens_ExistingKeyUpdated_Good: storing the same key
// twice overwrites the value without growing the LRU order list.
func TestTokenizer_StoreBPETokens_ExistingKeyUpdated_Good(t *testing.T) {
	tok := &Tokenizer{}
	tok.storeBPETokens("k", []int32{1, 2})
	tok.storeBPETokens("k", []int32{9})
	if len(tok.bpeCache) != 1 {
		t.Fatalf("cache entries = %d, want 1", len(tok.bpeCache))
	}
	if len(tok.bpeCacheOrder) != 1 {
		t.Fatalf("cache order entries = %d, want 1 (no duplicate)", len(tok.bpeCacheOrder))
	}
	got, ok := tok.cachedBPETokens("k")
	if !ok || len(got) != 1 || got[0] != 9 {
		t.Fatalf("cachedBPETokens(k) = (%v, %t), want ([9], true)", got, ok)
	}
}

// TestTokenizer_StoreBPETokens_LRUEviction_Ugly: inserting more than the cache
// limit evicts the oldest entries in FIFO order.
func TestTokenizer_StoreBPETokens_LRUEviction_Ugly(t *testing.T) {
	tok := &Tokenizer{}
	// Fill exactly to the limit.
	for i := 0; i < tokenizerBPECacheLimit; i++ {
		tok.storeBPETokens(core.Sprintf("k%d", i), []int32{int32(i)})
	}
	if len(tok.bpeCacheOrder) != tokenizerBPECacheLimit {
		t.Fatalf("order len = %d, want %d", len(tok.bpeCacheOrder), tokenizerBPECacheLimit)
	}
	// One more insertion must evict the oldest ("k0").
	tok.storeBPETokens("overflow", []int32{-1})
	if len(tok.bpeCacheOrder) != tokenizerBPECacheLimit {
		t.Fatalf("after overflow order len = %d, want %d (capped)", len(tok.bpeCacheOrder), tokenizerBPECacheLimit)
	}
	if _, ok := tok.cachedBPETokens("k0"); ok {
		t.Error("oldest entry k0 should have been evicted")
	}
	if _, ok := tok.cachedBPETokens("overflow"); !ok {
		t.Error("newest entry overflow should be present")
	}
}

// --- bpeMerge boundary-discard branches ---

// TestTokenizer_BPEMerge_StaleVersionDiscarded_Ugly: when a candidate pair has
// already been consumed by an earlier merge, the popped candidate is discarded
// (alive/next/version guards). A chain where the same left index participates
// in two candidates forces a stale pop. "a a a" with merge a+a exercises the
// overlapping-candidate discard path.
func TestTokenizer_BPEMerge_StaleVersionDiscarded_Ugly(t *testing.T) {
	tok := &Tokenizer{mergeRanks: map[mergeKey]int{
		{a: "a", b: "a"}:   0,
		{a: "aa", b: "a"}:  1,
		{a: "a", b: "aa"}:  1,
		{a: "aa", b: "aa"}: 2,
	}}
	// "a a a a" → merges collapse to "aaaa"; the middle candidates go stale.
	got := tok.bpeMerge([]string{"a", "a", "a", "a"})
	want := []string{"aaaa"}
	if len(got) != len(want) || got[0] != want[0] {
		t.Fatalf("bpeMerge = %v, want %v", got, want)
	}
}

// --- Decode / DecodeToken / DecodeOne interior-marker branches ---

// TestTokenizer_Decode_InteriorMarker_Good: a SentencePiece token whose ▁ is
// interior (not leading) splits inside Decode's bulk-write loop (the idx>0 arm).
func TestTokenizer_Decode_InteriorMarker_Good(t *testing.T) {
	// invVocab token "a▁b" → "a b"; not special.
	tok := &Tokenizer{
		invVocab: map[int32]string{1: "a▁b"},
		special:  map[string]int32{},
	}
	if got := tok.Decode([]int32{1}); got != "a b" {
		t.Errorf("Decode(interior ▁) = %q, want %q", got, "a b")
	}
}

// TestTokenizer_DecodeToken_InteriorMarkerMulti_Good: DecodeToken on a token
// with multiple ▁ markers (one leading, one interior) walks its Builder loop.
func TestTokenizer_DecodeToken_InteriorMarkerMulti_Good(t *testing.T) {
	// "▁a▁b" → " a b" (DecodeToken keeps the leading space).
	tok := &Tokenizer{
		invVocab: map[int32]string{1: "▁a▁b"},
		special:  map[string]int32{},
	}
	if got := tok.DecodeToken(1); got != " a b" {
		t.Errorf("DecodeToken(multi ▁) = %q, want %q", got, " a b")
	}
}

// TestTokenizer_DecodeToken_ChannelMarkers_Good: the reasoning-channel
// delimiters are special yet preserved (not stripped) so the parser can split
// the thinking span.
func TestTokenizer_DecodeToken_ChannelMarkers_Good(t *testing.T) {
	tok := &Tokenizer{
		invVocab: map[int32]string{
			1: channelOpenMarker,
			2: channelCloseMarker,
			3: "<eos>",
		},
		special: map[string]int32{
			channelOpenMarker:  1,
			channelCloseMarker: 2,
			"<eos>":            3,
		},
	}
	if got := tok.DecodeToken(1); got != channelOpenMarker {
		t.Errorf("DecodeToken(open) = %q, want %q", got, channelOpenMarker)
	}
	if got := tok.DecodeToken(2); got != channelCloseMarker {
		t.Errorf("DecodeToken(close) = %q, want %q", got, channelCloseMarker)
	}
	// A non-channel special still returns empty.
	if got := tok.DecodeToken(3); got != "" {
		t.Errorf("DecodeToken(<eos>) = %q, want empty", got)
	}
}

// TestTokenizer_DecodeOne_InteriorMarkerOnly_Good: DecodeOne on a token with a
// single interior ▁ (no leading marker) → text[:idx] + space + rest, no strip.
func TestTokenizer_DecodeOne_InteriorMarkerOnly_Good(t *testing.T) {
	tok := &Tokenizer{
		invVocab: map[int32]string{1: "a▁b"},
		special:  map[string]int32{},
	}
	if got := tok.DecodeOne(1); got != "a b" {
		t.Errorf("DecodeOne(interior ▁) = %q, want %q", got, "a b")
	}
}

// TestTokenizer_DecodeOne_MultipleMarkers_Good: DecodeOne on a token with two+
// markers (leading + interior) takes the general Builder branch and strips the
// single leading space.
func TestTokenizer_DecodeOne_MultipleMarkers_Good(t *testing.T) {
	tok := &Tokenizer{
		invVocab: map[int32]string{1: "▁a▁b"},
		special:  map[string]int32{},
	}
	// " a b" with leading space stripped → "a b".
	if got := tok.DecodeOne(1); got != "a b" {
		t.Errorf("DecodeOne(multi ▁) = %q, want %q", got, "a b")
	}
}

// TestTokenizer_DecodeOne_GPT2_Good: DecodeOne routes a GPT-2 tokenizer through
// decodeGPT2Bytes.
func TestTokenizer_DecodeOne_GPT2_Good(t *testing.T) {
	tok, err := LoadTokenizer(writeTokenizerJSON(t, gpt2TokenizerJSON))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	// id 4 = "the" (no leading space).
	if got := tok.DecodeOne(4); got != "the" {
		t.Errorf("DecodeOne(4) = %q, want %q", got, "the")
	}
}

// TestTokenizer_DecodeOne_Unknown_Ugly: an unknown id returns empty (the
// not-ok early return).
func TestTokenizer_DecodeOne_Unknown_Ugly(t *testing.T) {
	tok := &Tokenizer{invVocab: map[int32]string{}, special: map[string]int32{}}
	if got := tok.DecodeOne(123); got != "" {
		t.Errorf("DecodeOne(unknown) = %q, want empty", got)
	}
}

// TestTokenizer_DecodeOne_GeneralCaseNoLeadingSpace_Good drives the general
// multi-marker branch of DecodeOne where the output does NOT start with a space
// (interior markers only), so the final no-strip return is taken.
func TestTokenizer_DecodeOne_GeneralCaseNoLeadingSpace_Good(t *testing.T) {
	// "a▁b▁c" → first marker is interior (idx>0) with a following marker →
	// general Builder branch → "a b c" (no leading space to strip).
	tok := &Tokenizer{
		invVocab: map[int32]string{1: "a▁b▁c"},
		special:  map[string]int32{},
	}
	if got := tok.DecodeOne(1); got != "a b c" {
		t.Errorf("DecodeOne(\"a▁b▁c\") = %q, want %q", got, "a b c")
	}
}

// --- Gemma <end_of_turn> EOS override ---

// TestTokenizer_LoadTokenizer_GemmaEndOfTurnEOS_Good: <end_of_turn> overrides a
// prior <eos> as the generation stop token.
func TestTokenizer_LoadTokenizer_GemmaEndOfTurnEOS_Good(t *testing.T) {
	tok, err := LoadTokenizer(writeTokenizerJSON(t, gemmaEndOfTurnTokenizerJSON))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	if !tok.HasEOSToken() {
		t.Fatal("expected EOS present")
	}
	if tok.EOSToken() != 107 {
		t.Errorf("EOSToken() = %d, want 107 (<end_of_turn> overrides <eos>)", tok.EOSToken())
	}
}

// --- GPT-2 encode: BOS prepend + in-loop special match ---

// TestTokenizer_GPT2_EncodeWithBOSAndSpecial_Good drives the GPT-2 encode path
// through BOS prepending (shouldPrependBOS true) and the special-token match
// branch inside the segment loop.
func TestTokenizer_GPT2_EncodeWithBOSAndSpecial_Good(t *testing.T) {
	tok, err := LoadTokenizer(writeTokenizerJSON(t, gpt2WithBOSTokenizerJSON))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	if !tok.isGPT2BPE {
		t.Fatal("expected GPT-2 mode")
	}
	if tok.BOSToken() != 128000 {
		t.Fatalf("BOSToken() = %d, want 128000", tok.BOSToken())
	}

	// "the<|sep|>the": leading BOS prepended, then "the" (id 4), then the
	// <|sep|> special (id 128001) matched in-loop, then "the" (id 4) again.
	ids := tok.Encode("the<|sep|>the")
	want := []int32{128000, 4, 128001, 4}
	if len(ids) != len(want) {
		t.Fatalf("Encode = %v, want %v", ids, want)
	}
	for i := range want {
		if ids[i] != want[i] {
			t.Fatalf("ids[%d] = %d, want %d", i, ids[i], want[i])
		}
	}
}

// --- popDirect sift-down settle (break with left<n) ---

// TestTokenizer_HeapPopDirect_SiftSettles_Good drives popDirect's sift-down to
// the point where the swapped-in root is already smaller than its children, so
// the loop breaks with a live left child (the settle branch) rather than
// running off the bottom. Push candidates with ranks that, after the root pop,
// leave a near-minimal element at the top.
func TestTokenizer_HeapPopDirect_SiftSettles_Good(t *testing.T) {
	h := make(bpeCandidateHeap, 0, 8)
	// Ranks: 1,2,3,4,5,6,7. After popping rank 1, the last element (rank 7)
	// moves to the root and sifts down; the tree below keeps the heap property
	// such that sift-down terminates after a swap leaving a live left child.
	for _, r := range []int{1, 2, 3, 4, 5, 6, 7} {
		h.pushDirect(bpeCandidate{rank: r, left: r})
	}
	// Pop everything in ascending rank order — the staged sift-downs exercise
	// both the run-off-bottom and the settle (break) paths.
	prev := -1
	for h.Len() > 0 {
		c := h.popDirect()
		if c.rank < prev {
			t.Fatalf("heap pop out of order: got %d after %d", c.rank, prev)
		}
		prev = c.rank
	}
}
