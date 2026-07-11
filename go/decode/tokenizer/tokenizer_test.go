// SPDX-Licence-Identifier: EUPL-1.2

package tokenizer

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// minimalTokenizerJSON is a valid HuggingFace tokenizer.json with a tiny vocab.
const minimalTokenizerJSON = `{
  "model": {
    "type": "BPE",
    "vocab": {
      "h": 0,
      "e": 1,
      "l": 2,
      "o": 3,
      "▁": 4,
      "he": 5,
      "ll": 6,
      "▁h": 7
    },
    "merges": ["h e", "l l"],
    "byte_fallback": false
  },
  "added_tokens": [
    {"id": 100, "content": "<bos>", "special": true},
    {"id": 101, "content": "<eos>", "special": true}
  ]
}`

const tokenizerWithoutSpecialsJSON = `{
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
    "merges": ["h e", "l l"],
    "byte_fallback": false
  },
  "added_tokens": []
}`

const gemma4SpecialTokenizerJSON = `{
  "normalizer": {"type": "Replace", "content": "▁"},
  "pre_tokenizer": {"type": "Split", "behavior": "MergedWithPrevious"},
  "model": {
    "type": "BPE",
    "vocab": {
      "▁": 30,
      "h": 20,
      "i": 21,
      "u": 31,
      "s": 32,
      "e": 33,
      "r": 34,
      "us": 35,
      "use": 36,
      "\n": 9,
      "user": 10,
      "▁user": 11
    },
    "merges": ["u s", "us e", "use r"]
  },
  "added_tokens": [
    {"id": 2, "content": "<bos>", "special": true},
    {"id": 1, "content": "<eos>", "special": true},
    {"id": 105, "content": "<|turn>", "special": true},
    {"id": 106, "content": "<turn|>", "special": true}
  ]
}`

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

func writeTestTokenizer(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	path := core.JoinPath(dir, "tokenizer.json")
	if err := coreio.Local.Write(path, minimalTokenizerJSON); err != nil {
		t.Fatalf("write test tokenizer: %v", err)
	}
	return path
}

func writeTokenizerWithoutSpecials(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	path := core.JoinPath(dir, "tokenizer.json")
	if err := coreio.Local.Write(path, tokenizerWithoutSpecialsJSON); err != nil {
		t.Fatalf("write tokenizer without specials: %v", err)
	}
	return path
}

func writeGemma4SpecialTokenizer(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	path := core.JoinPath(dir, "tokenizer.json")
	if err := coreio.Local.Write(path, gemma4SpecialTokenizerJSON); err != nil {
		t.Fatalf("write gemma4 tokenizer: %v", err)
	}
	return path
}

func writeTokenizerJSON(t *testing.T, body string) string {
	t.Helper()
	dir := t.TempDir()
	path := core.JoinPath(dir, "tokenizer.json")
	if err := coreio.Local.Write(path, body); err != nil {
		t.Fatalf("write tokenizer json: %v", err)
	}
	return path
}

// --- IndexIn ---

// TestTokenizer_IndexIn_Good confirms IndexIn returns the byte position of a
// substring that is present.
func TestTokenizer_IndexIn_Good(t *testing.T) {
	if got := IndexIn("hello world", "world"); got != 6 {
		t.Errorf("IndexIn(\"hello world\", \"world\") = %d, want 6", got)
	}
}

// TestTokenizer_IndexIn_Bad confirms IndexIn returns -1 for a substring that
// is absent.
func TestTokenizer_IndexIn_Bad(t *testing.T) {
	if got := IndexIn("hello", "xyz"); got != -1 {
		t.Errorf("IndexIn(\"hello\", \"xyz\") = %d, want -1", got)
	}
}

// TestTokenizer_IndexIn_Ugly confirms the stdlib edge case IndexIn inherits
// from core.Index (itself strings.Index): an empty substring is considered
// present at offset 0.
func TestTokenizer_IndexIn_Ugly(t *testing.T) {
	if got := IndexIn("hello", ""); got != 0 {
		t.Errorf("IndexIn(\"hello\", \"\") = %d, want 0", got)
	}
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

// TestTokenizer_NewForDecode_Bad confirms a populated decode-only tokenizer
// still returns empty for an id absent from the supplied inverse vocabulary
// — a graceful miss, not a panic.
func TestTokenizer_NewForDecode_Bad(t *testing.T) {
	tok := NewForDecode(map[int32]string{5: "he"})
	if got := tok.DecodeToken(999); got != "" {
		t.Errorf("DecodeToken(999) = %q, want empty for id absent from invVocab", got)
	}
}

// TestTokenizer_NewForDecode_Ugly: a nil inverse vocab yields an empty,
// usable tokenizer (no panic, empty results).
func TestTokenizer_NewForDecode_Ugly(t *testing.T) {
	tok := NewForDecode(nil)
	if tok == nil {
		t.Fatal("NewForDecode(nil) returned nil")
	}
	if got := tok.DecodeToken(1); got != "" {
		t.Errorf("DecodeToken on empty = %q, want empty", got)
	}
}

// --- LoadTokenizer ---

func TestTokenizer_LoadTokenizer_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, err := LoadTokenizer(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if tok == nil {
		t.Fatal("tokenizer is nil")
	}
}

func TestTokenizer_LoadTokenizer_Bad(t *testing.T) {
	_, err := LoadTokenizer("/nonexistent/tokenizer.json")
	if err == nil {
		t.Error("expected error for missing file")
	}
}

func TestTokenizer_LoadTokenizer_Ugly(t *testing.T) {
	dir := t.TempDir()
	path := core.JoinPath(dir, "tokenizer.json")
	_ = coreio.Local.Write(path, "not json")

	_, err := LoadTokenizer(path)
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

// TestTokenizer_LoadTokenizer_EmptyFile_Ugly tests loading a tokenizer from an
// empty file. Should return a parse error, not panic.
func TestTokenizer_LoadTokenizer_EmptyFile_Ugly(t *testing.T) {
	dir := t.TempDir()
	path := core.JoinPath(dir, "tokenizer.json")
	_ = coreio.Local.Write(path, "")

	_, err := LoadTokenizer(path)
	if err == nil {
		t.Error("expected error for empty tokenizer file")
	}
}

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

func TestTokenizer_LoadTokenizer_Gemma4TurnEndIsEOS_Good(t *testing.T) {
	path := writeGemma4SpecialTokenizer(t)
	tok, err := LoadTokenizer(path)
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}

	if tok.BOSToken() != 2 {
		t.Fatalf("BOSToken() = %d, want 2", tok.BOSToken())
	}
	if tok.EOSToken() != 106 {
		t.Fatalf("EOSToken() = %d, want Gemma4 turn end 106", tok.EOSToken())
	}
}

// TestTokenizer_LoadTokenizer_GPT2Detected_Good: a vocab containing "Ġthe"
// flips the tokenizer into GPT-2 byte-level BPE mode and builds the byte maps
// during LoadTokenizer.
func TestTokenizer_LoadTokenizer_GPT2Detected_Good(t *testing.T) {
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

// --- BOSToken / EOSToken / HasBOSToken / HasEOSToken / BOS / EOS ---

func TestTokenizer_BOSToken_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)
	if got := tok.BOSToken(); got != 100 {
		t.Errorf("BOSToken() = %d, want 100", got)
	}
}

// TestTokenizer_BOSToken_Bad: a tokenizer with no BOS special defined returns
// the zero value rather than inventing an id.
func TestTokenizer_BOSToken_Bad(t *testing.T) {
	path := writeTokenizerWithoutSpecials(t)
	tok, err := LoadTokenizer(path)
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	if got := tok.BOSToken(); got != 0 {
		t.Errorf("BOSToken() = %d, want 0 (no BOS defined)", got)
	}
}

// TestTokenizer_BOSToken_Ugly: BOSToken is a bare field accessor with no
// validation — an out-of-range (negative) stored value passes through
// unchanged.
func TestTokenizer_BOSToken_Ugly(t *testing.T) {
	tok := &Tokenizer{bosToken: -1}
	if got := tok.BOSToken(); got != -1 {
		t.Errorf("BOSToken() = %d, want -1 (verbatim passthrough)", got)
	}
}

func TestTokenizer_EOSToken_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)
	if got := tok.EOSToken(); got != 101 {
		t.Errorf("EOSToken() = %d, want 101", got)
	}
}

// TestTokenizer_EOSToken_Bad: a tokenizer with no EOS special defined returns
// the zero value rather than inventing an id.
func TestTokenizer_EOSToken_Bad(t *testing.T) {
	path := writeTokenizerWithoutSpecials(t)
	tok, err := LoadTokenizer(path)
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	if got := tok.EOSToken(); got != 0 {
		t.Errorf("EOSToken() = %d, want 0 (no EOS defined)", got)
	}
}

// TestTokenizer_EOSToken_Ugly: EOSToken is a bare field accessor with no
// validation — an out-of-range (negative) stored value passes through
// unchanged.
func TestTokenizer_EOSToken_Ugly(t *testing.T) {
	tok := &Tokenizer{eosToken: -1}
	if got := tok.EOSToken(); got != -1 {
		t.Errorf("EOSToken() = %d, want -1 (verbatim passthrough)", got)
	}
}

func TestTokenizer_HasBOSToken_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)
	if !tok.HasBOSToken() {
		t.Error("HasBOSToken() = false, want true")
	}
}

func TestTokenizer_HasBOSToken_Bad(t *testing.T) {
	path := writeTokenizerWithoutSpecials(t)
	tok, err := LoadTokenizer(path)
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	if tok.HasBOSToken() {
		t.Error("HasBOSToken() = true, want false (no BOS defined)")
	}
}

// TestTokenizer_HasBOSToken_Ugly: a nil *Tokenizer is a safe caller — the
// method's own nil guard must return false rather than panic.
func TestTokenizer_HasBOSToken_Ugly(t *testing.T) {
	var tok *Tokenizer
	if tok.HasBOSToken() {
		t.Error("HasBOSToken() on nil tokenizer = true, want false")
	}
}

func TestTokenizer_HasEOSToken_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)
	if !tok.HasEOSToken() {
		t.Error("HasEOSToken() = false, want true")
	}
}

func TestTokenizer_HasEOSToken_Bad(t *testing.T) {
	path := writeTokenizerWithoutSpecials(t)
	tok, err := LoadTokenizer(path)
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	if tok.HasEOSToken() {
		t.Error("HasEOSToken() = true, want false (no EOS defined)")
	}
}

// TestTokenizer_HasEOSToken_Ugly: a nil *Tokenizer is a safe caller — the
// method's own nil guard must return false rather than panic.
func TestTokenizer_HasEOSToken_Ugly(t *testing.T) {
	var tok *Tokenizer
	if tok.HasEOSToken() {
		t.Error("HasEOSToken() on nil tokenizer = true, want false")
	}
}

func TestTokenizer_BOS_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)
	if got := tok.BOS(); got != 100 {
		t.Errorf("BOS() = %d, want 100", got)
	}
}

func TestTokenizer_BOS_Bad(t *testing.T) {
	path := writeTokenizerWithoutSpecials(t)
	tok, err := LoadTokenizer(path)
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	if got := tok.BOS(); got != 0 {
		t.Errorf("BOS() = %d, want 0 (no BOS defined)", got)
	}
}

// TestTokenizer_BOS_Ugly confirms BOS() delegates to BOSToken() verbatim, even
// for an out-of-range stored value.
func TestTokenizer_BOS_Ugly(t *testing.T) {
	tok := &Tokenizer{bosToken: -1}
	if got := tok.BOS(); got != tok.BOSToken() {
		t.Errorf("BOS() = %d, want %d (BOSToken() parity)", got, tok.BOSToken())
	}
}

func TestTokenizer_EOS_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)
	if got := tok.EOS(); got != 101 {
		t.Errorf("EOS() = %d, want 101", got)
	}
}

func TestTokenizer_EOS_Bad(t *testing.T) {
	path := writeTokenizerWithoutSpecials(t)
	tok, err := LoadTokenizer(path)
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	if got := tok.EOS(); got != 0 {
		t.Errorf("EOS() = %d, want 0 (no EOS defined)", got)
	}
}

// TestTokenizer_EOS_Ugly confirms EOS() delegates to EOSToken() verbatim, even
// for an out-of-range stored value.
func TestTokenizer_EOS_Ugly(t *testing.T) {
	tok := &Tokenizer{eosToken: -1}
	if got := tok.EOS(); got != tok.EOSToken() {
		t.Errorf("EOS() = %d, want %d (EOSToken() parity)", got, tok.EOSToken())
	}
}

// --- TokenID / IDToken ---

func TestTokenizer_TokenID_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)
	id, ok := tok.TokenID("he")
	if !ok || id != 5 {
		t.Errorf("TokenID(\"he\") = (%d, %t), want (5, true)", id, ok)
	}
}

func TestTokenizer_TokenID_Bad(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)
	id, ok := tok.TokenID("nonexistent")
	if ok || id != 0 {
		t.Errorf("TokenID(\"nonexistent\") = (%d, %t), want (0, false)", id, ok)
	}
}

// TestTokenizer_TokenID_Ugly: an empty-string lookup is a valid call (not a
// panic) and misses cleanly when "" isn't itself a vocab entry.
func TestTokenizer_TokenID_Ugly(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)
	id, ok := tok.TokenID("")
	if ok || id != 0 {
		t.Errorf("TokenID(\"\") = (%d, %t), want (0, false)", id, ok)
	}
}

func TestTokenizer_IDToken_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)
	if got := tok.IDToken(6); got != "ll" {
		t.Errorf("IDToken(6) = %q, want %q", got, "ll")
	}
}

func TestTokenizer_IDToken_Bad(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)
	if got := tok.IDToken(9999); got != "" {
		t.Errorf("IDToken(9999) = %q, want empty", got)
	}
}

// TestTokenizer_IDToken_Ugly: a negative id is a valid map lookup (not a
// panic) and misses cleanly.
func TestTokenizer_IDToken_Ugly(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)
	if got := tok.IDToken(-1); got != "" {
		t.Errorf("IDToken(-1) = %q, want empty (negative id, no panic)", got)
	}
}

// --- Encode ---

func TestTokenizer_Encode_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	tokens := tok.Encode("hello")
	if len(tokens) == 0 {
		t.Fatal("Encode returned empty tokens")
	}
	// First token should be BOS
	if tokens[0] != tok.BOSToken() {
		t.Errorf("first token = %d, want BOS (%d)", tokens[0], tok.BOSToken())
	}
	// With BPE merges ("h e" → "he", "l l" → "ll"), "hello" with ▁ prefix becomes:
	// "▁" "h" "e" "l" "l" "o" → merge "h e" → "▁" "he" "l" "l" "o"
	// → merge "l l" → "▁" "he" "ll" "o"
	// No further merges. But "▁" is not "▁h" so it stays as "▁".
	// Vocab: ▁=4, he=5, ll=6, o=3. Expected: [BOS, 4, 5, 6, 3]
	want := []int32{100, 4, 5, 6, 3}
	if len(tokens) != len(want) {
		t.Fatalf("Encode(\"hello\") = %v, want %v", tokens, want)
	}
	for i := range tokens {
		if tokens[i] != want[i] {
			t.Errorf("tokens[%d] = %d, want %d", i, tokens[i], want[i])
		}
	}
}

// TestTokenizer_Encode_Bad: text containing characters entirely absent from
// the vocabulary produces no tokens for that segment — an unmapped symbol is
// silently dropped rather than erroring.
func TestTokenizer_Encode_Bad(t *testing.T) {
	tok := &Tokenizer{
		vocab:      map[string]int32{"a": 0},
		mergeRanks: map[mergeKey]int{},
	}
	tokens := tok.Encode("z")
	if len(tokens) != 0 {
		t.Errorf("Encode(\"z\") = %v, want empty (z absent from vocab)", tokens)
	}
}

// TestTokenizer_Encode_Ugly tests encoding an empty string.
// Should return only the BOS token (no panic, no out-of-bounds).
func TestTokenizer_Encode_Ugly(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	tokens := tok.Encode("")
	// Empty input: only BOS token expected
	if len(tokens) == 0 {
		t.Fatal("Encode(\"\") returned empty slice — expected at least BOS token")
	}
	if tokens[0] != tok.BOSToken() {
		t.Errorf("first token = %d, want BOS (%d)", tokens[0], tok.BOSToken())
	}
}

func TestTokenizer_Encode_ExplicitBOSDoesNotDuplicate_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, err := LoadTokenizer(path)
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}

	tokens := tok.Encode("<bos>hello")
	want := []int32{100, 4, 5, 6, 3}
	if len(tokens) != len(want) {
		t.Fatalf("Encode(\"<bos>hello\") = %v, want %v", tokens, want)
	}
	for i := range want {
		if tokens[i] != want[i] {
			t.Fatalf("tokens[%d] = %d, want %d", i, tokens[i], want[i])
		}
	}
}

func TestTokenizer_Encode_MultiWordSentencePiece_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	tokens := tok.Encode("hello hello")
	want := []int32{100, 4, 5, 6, 3, 4, 5, 6, 3}
	if len(tokens) != len(want) {
		t.Fatalf("Encode(\"hello hello\") = %v, want %v", tokens, want)
	}
	for i := range want {
		if tokens[i] != want[i] {
			t.Fatalf("tokens[%d] = %d, want %d", i, tokens[i], want[i])
		}
	}

	if decoded := tok.Decode(tokens); decoded != "hello hello" {
		t.Fatalf("Decode(Encode(\"hello hello\")) = %q, want %q", decoded, "hello hello")
	}
}

func TestTokenizer_Encode_Gemma4DoesNotInventPrefixSpace_Good(t *testing.T) {
	path := writeGemma4SpecialTokenizer(t)
	tok, err := LoadTokenizer(path)
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}

	raw := tok.Encode("h")
	wantRaw := []int32{2, 20}
	if len(raw) != len(wantRaw) {
		t.Fatalf("Encode(\"h\") = %v, want %v", raw, wantRaw)
	}
	for i := range wantRaw {
		if raw[i] != wantRaw[i] {
			t.Fatalf("raw[%d] = %d, want %d", i, raw[i], wantRaw[i])
		}
	}

	chat := tok.Encode("<bos><|turn>user\nh<turn|>\n")
	wantChat := []int32{2, 105, 10, 9, 20, 106, 9}
	if len(chat) != len(wantChat) {
		t.Fatalf("Encode(chat) = %v, want %v", chat, wantChat)
	}
	for i := range wantChat {
		if chat[i] != wantChat[i] {
			t.Fatalf("chat[%d] = %d, want %d", i, chat[i], wantChat[i])
		}
	}
}

// TestTokenizer_Encode_NoSpecialTokensDoesNotInventBOS_Bad: when a tokenizer
// has no BOS special defined, Encode must not prepend an invented BOS id.
func TestTokenizer_Encode_NoSpecialTokensDoesNotInventBOS_Bad(t *testing.T) {
	path := writeTokenizerWithoutSpecials(t)
	tok, err := LoadTokenizer(path)
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}

	tokens := tok.Encode("hello")
	want := []int32{4, 5, 6, 3}
	if len(tokens) != len(want) {
		t.Fatalf("Encode(\"hello\") = %v, want %v", tokens, want)
	}
	for i := range want {
		if tokens[i] != want[i] {
			t.Fatalf("tokens[%d] = %d, want %d", i, tokens[i], want[i])
		}
	}
}

func TestTokenizer_EncodeCachesSentencePieceSegments_Good(t *testing.T) {
	tok := &Tokenizer{
		vocab: map[string]int32{
			"▁ab": 7,
		},
		addPrefixSpace: true,
		mergeRanks: map[mergeKey]int{
			{a: "▁", b: "a"}:  0,
			{a: "▁a", b: "b"}: 1,
		},
	}

	first := tok.Encode("ab")
	if len(first) != 1 || first[0] != 7 {
		t.Fatalf("Encode first = %v, want [7]", first)
	}
	if len(tok.bpeCache) != 1 {
		t.Fatalf("bpe cache entries = %d, want 1", len(tok.bpeCache))
	}

	first[0] = 99
	second := tok.Encode("ab")
	if len(second) != 1 || second[0] != 7 {
		t.Fatalf("Encode second = %v, want cached [7]", second)
	}
	if len(tok.bpeCache) != 1 {
		t.Fatalf("bpe cache entries after repeat = %d, want 1", len(tok.bpeCache))
	}
}

// TestTokenizer_Encode_Decode_GPT2RoundTrip_Good exercises encodeGPT2 +
// encodeGPT2Segment + the BPE merge step + the GPT-2 Decode branch.
// "the" → byte-encodes to "the" → merges t+h→th, th+e→the → vocab id 4.
// " the" (leading space) → "Ġthe" → id 7.
func TestTokenizer_Encode_Decode_GPT2RoundTrip_Good(t *testing.T) {
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

// TestTokenizer_Encode_GPT2Caches_Good: a repeated GPT-2 segment is served
// from the BPE cache the second time (storeBPETokens + cachedBPETokens on the
// gpt2 key prefix).
func TestTokenizer_Encode_GPT2Caches_Good(t *testing.T) {
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

// TestTokenizer_Encode_GPT2WithBOSAndSpecial_Good drives the GPT-2 encode path
// through BOS prepending (shouldPrependBOS true) and the special-token match
// branch inside the segment loop.
func TestTokenizer_Encode_GPT2WithBOSAndSpecial_Good(t *testing.T) {
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

func TestTokenizer_BPEMerge_Good(t *testing.T) {
	tok := &Tokenizer{
		mergeRanks: map[mergeKey]int{
			{a: "h", b: "e"}:  0,
			{a: "l", b: "l"}:  1,
			{a: "he", b: "l"}: 2,
		},
	}

	// "h" "e" "l" "l" "o" → merge "h e" (rank 0) → "he" "l" "l" "o"
	// → merge "l l" (rank 1) → "he" "ll" "o"
	// → merge "he l" does NOT match "he ll" — stops here.
	symbols := []string{"h", "e", "l", "l", "o"}
	got := tok.bpeMerge("hello", symbols)
	want := []string{"he", "ll", "o"}
	if len(got) != len(want) {
		t.Fatalf("bpeMerge = %v, want %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("bpeMerge[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestTokenizer_BPEMerge_OverlappingPairs_Good(t *testing.T) {
	tok := &Tokenizer{
		mergeRanks: map[mergeKey]int{
			{a: "a", b: "b"}:   1,
			{a: "b", b: "c"}:   0,
			{a: "bc", b: "d"}:  0,
			{a: "a", b: "bcd"}: 0,
		},
	}

	got := tok.bpeMerge("abcd", []string{"a", "b", "c", "d"})
	want := []string{"abcd"}
	if len(got) != len(want) {
		t.Fatalf("bpeMerge = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("bpeMerge[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestTokenizer_BPEMerge_LeftMostTie_Good(t *testing.T) {
	tok := &Tokenizer{
		mergeRanks: map[mergeKey]int{
			{a: "a", b: "b"}:  0,
			{a: "c", b: "d"}:  0,
			{a: "ab", b: "c"}: 0,
		},
	}

	got := tok.bpeMerge("abcd", []string{"a", "b", "c", "d"})
	want := []string{"abc", "d"}
	if len(got) != len(want) {
		t.Fatalf("bpeMerge = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("bpeMerge[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestTokenizer_BPEMerge_NoMerges_Good(t *testing.T) {
	tok := &Tokenizer{mergeRanks: map[mergeKey]int{}}
	symbols := []string{"a", "b", "c"}
	got := tok.bpeMerge("abc", symbols)
	if len(got) != 3 {
		t.Errorf("bpeMerge with no merges = %v, want [a b c]", got)
	}
}

func TestTokenizer_BPEMerge_SingleSymbol_Good(t *testing.T) {
	tok := &Tokenizer{mergeRanks: map[mergeKey]int{{a: "a", b: "b"}: 0}}
	got := tok.bpeMerge("x", []string{"x"})
	if len(got) != 1 || got[0] != "x" {
		t.Errorf("bpeMerge single = %v, want [x]", got)
	}
}

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
	got := tok.bpeMerge("aaaa", []string{"a", "a", "a", "a"})
	want := []string{"aaaa"}
	if len(got) != len(want) || got[0] != want[0] {
		t.Fatalf("bpeMerge = %v, want %v", got, want)
	}
}

func TestTokenizer_BPEMerge_NilSymbols_Ugly(t *testing.T) {
	tok := &Tokenizer{mergeRanks: map[mergeKey]int{{a: "a", b: "b"}: 0}}
	got := tok.bpeMerge("", []string{})
	if len(got) != 0 {
		t.Errorf("bpeMerge(empty) = %v, want empty", got)
	}
}

// --- Decode ---

func TestTokenizer_Decode_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	// Decode known vocab entries
	text := tok.Decode([]int32{5, 6, 3}) // "he" + "ll" + "o"
	if text != "hello" {
		t.Errorf("Decode = %q, want %q", text, "hello")
	}
}

// TestTokenizer_Decode_Bad: an id absent from the inverse vocabulary is
// skipped rather than producing garbage text or panicking.
func TestTokenizer_Decode_Bad(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	text := tok.Decode([]int32{5, 9999, 6, 3})
	if text != "hello" {
		t.Errorf("Decode(with unknown id) = %q, want %q (unknown id skipped)", text, "hello")
	}
}

// TestTokenizer_Decode_Ugly tests decoding an empty token slice.
// Should return empty string without panicking.
func TestTokenizer_Decode_Ugly(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	text := tok.Decode([]int32{})
	if text != "" {
		t.Errorf("Decode(empty) = %q, want empty string", text)
	}
}

func TestTokenizer_Decode_SpecialTokensSkipped_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	// Decoding BOS/EOS should produce empty string
	text := tok.Decode([]int32{100, 101})
	if text != "" {
		t.Errorf("Decode(BOS, EOS) = %q, want empty", text)
	}
}

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

// TestTokenizer_Decode_DecodeToken_GPT2SkipsSpecial_Good: the special
// <|endoftext|> token is skipped in the GPT-2 Decode path (special-skip
// branch), and DecodeToken returns empty for the same special.
func TestTokenizer_Decode_DecodeToken_GPT2SkipsSpecial_Good(t *testing.T) {
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

func TestTokenizer_DecodeToken_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	// "he" = token 5
	text := tok.DecodeToken(5)
	if text != "he" {
		t.Errorf("DecodeToken(5) = %q, want %q", text, "he")
	}
}

func TestTokenizer_DecodeToken_Special_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	// Special tokens should return empty
	text := tok.DecodeToken(100)
	if text != "" {
		t.Errorf("DecodeToken(BOS) = %q, want empty", text)
	}
}

func TestTokenizer_DecodeToken_SentencePieceSpace_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	// "▁h" = token 7, should decode to " h" (space prefix)
	text := tok.DecodeToken(7)
	if text != " h" {
		t.Errorf("DecodeToken(7) = %q, want %q", text, " h")
	}
}

// TestTokenizer_DecodeToken_SoloMarker_Good pins the bare-▁ decode (the
// standalone-space token) to exactly " ". This is the case the zero-alloc
// spaceString const short-circuits — the pin guards against the const ever
// drifting from the Builder path it replaced (which produced " " by writing
// a single space then an empty remainder).
func TestTokenizer_DecodeToken_SoloMarker_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	// "▁" = token 4, should decode to a single space.
	text := tok.DecodeToken(4)
	if text != " " {
		t.Errorf("DecodeToken(4) = %q, want %q", text, " ")
	}
}

func TestTokenizer_DecodeToken_Bad(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	text := tok.DecodeToken(9999)
	if text != "" {
		t.Errorf("DecodeToken(unknown) = %q, want empty", text)
	}
}

// TestTokenizer_DecodeToken_Ugly tests decoding a token ID outside vocab range.
// Should return empty string without panicking.
func TestTokenizer_DecodeToken_Ugly(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	// Use a large ID well outside any realistic vocab range
	text := tok.DecodeToken(1 << 30)
	if text != "" {
		t.Errorf("DecodeToken(huge id) = %q, want empty", text)
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

// DecodeOne mirrors Decode([]int32{id}) — verify byte-exact equivalence on
// regular, SentencePiece-prefixed, special, and unknown ids. This is the
// contract IDToken depends on for its no-allocation fast path.
func TestTokenizer_DecodeOne_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	cases := []struct {
		name string
		id   int32
	}{
		{"regular_he", 5},
		{"regular_ll", 6},
		{"sentencepiece_h", 7},
		{"special_bos", 100},
		{"special_eos", 101},
		{"unknown_high", 9999},
	}
	for _, c := range cases {
		want := tok.Decode([]int32{c.id})
		got := tok.DecodeOne(c.id)
		if got != want {
			t.Errorf("DecodeOne(%s id=%d) = %q, want %q (Decode parity)",
				c.name, c.id, got, want)
		}
	}
}

// TestTokenizer_DecodeOne_Bad: an unknown id returns empty (the not-ok early
// return) rather than panicking.
func TestTokenizer_DecodeOne_Bad(t *testing.T) {
	tok := &Tokenizer{invVocab: map[int32]string{}, special: map[string]int32{}}
	if got := tok.DecodeOne(123); got != "" {
		t.Errorf("DecodeOne(unknown) = %q, want empty", got)
	}
}

// TestTokenizer_DecodeOne_Ugly: unlike DecodeToken, DecodeOne has no
// channel-marker exception — ANY special token (even a reasoning-channel
// delimiter) decodes to empty. This asymmetry with DecodeToken is a real,
// easy-to-regress edge, so it is pinned explicitly.
func TestTokenizer_DecodeOne_Ugly(t *testing.T) {
	tok := &Tokenizer{
		invVocab: map[int32]string{1: channelOpenMarker},
		special:  map[string]int32{channelOpenMarker: 1},
	}
	if got := tok.DecodeOne(1); got != "" {
		t.Errorf("DecodeOne(channel marker) = %q, want empty (no channel exception)", got)
	}
	if got := tok.DecodeToken(1); got != channelOpenMarker {
		t.Errorf("DecodeToken(channel marker) = %q, want %q (contrast with DecodeOne)", got, channelOpenMarker)
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

func TestTokenizer_FormatGemmaPrompt_Good(t *testing.T) {
	got := FormatGemmaPrompt("What is 2+2?")
	want := "<bos><start_of_turn>user\nWhat is 2+2?<end_of_turn>\n<start_of_turn>model\n"
	if got != want {
		t.Errorf("FormatGemmaPrompt = %q, want %q", got, want)
	}
}

// TestTokenizer_FormatGemmaPrompt_Bad: an empty prompt still produces the full
// chat-template wrapper with no content between the turn markers.
func TestTokenizer_FormatGemmaPrompt_Bad(t *testing.T) {
	got := FormatGemmaPrompt("")
	want := "<bos><start_of_turn>user\n<end_of_turn>\n<start_of_turn>model\n"
	if got != want {
		t.Errorf("FormatGemmaPrompt(\"\") = %q, want %q", got, want)
	}
}

// TestTokenizer_FormatGemmaPrompt_Ugly: FormatGemmaPrompt does no escaping — a
// prompt that itself embeds turn markers passes through verbatim. Pinning this
// documents the current (unescaped) behaviour, not that it is desirable.
func TestTokenizer_FormatGemmaPrompt_Ugly(t *testing.T) {
	got := FormatGemmaPrompt("hi<end_of_turn>\n<start_of_turn>model\ninjected")
	want := "<bos><start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\ninjected<end_of_turn>\n<start_of_turn>model\n"
	if got != want {
		t.Errorf("FormatGemmaPrompt(injected) = %q, want %q", got, want)
	}
}

// --- GPT-2 byte maps ---

func TestTokenizer_BuildGPT2ByteMaps_Good(t *testing.T) {
	decoder, encoder := buildGPT2ByteMaps()

	// All 256 bytes must be mapped
	if len(encoder) != 256 {
		t.Errorf("encoder has %d entries, want 256", len(encoder))
	}
	if len(decoder) != 256 {
		t.Errorf("decoder has %d entries, want 256", len(decoder))
	}

	// Round-trip: every byte should survive encode → decode
	for b := range 256 {
		r := encoder[byte(b)]
		got := decoder[r]
		if got != byte(b) {
			t.Errorf("byte %d: encode→decode = %d, want %d", b, got, b)
		}
	}
}

func TestTokenizer_BuildGPT2ByteMaps_PrintableASCII_Good(t *testing.T) {
	_, encoder := buildGPT2ByteMaps()

	// Printable ASCII (33-126) should self-map
	for b := 33; b <= 126; b++ {
		if encoder[byte(b)] != rune(b) {
			t.Errorf("byte %d (%c): expected self-map, got %c", b, b, encoder[byte(b)])
		}
	}
}

func TestTokenizer_BuildGPT2ByteMaps_ControlChars_Good(t *testing.T) {
	_, encoder := buildGPT2ByteMaps()

	// Space (32) and control chars (0-31) should NOT self-map
	if encoder[byte(32)] == rune(32) {
		t.Error("space (32) should not self-map in GPT-2 encoding")
	}
	if encoder[byte(0)] == rune(0) {
		t.Error("null (0) should not self-map in GPT-2 encoding")
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

// TestTokenizer_GPT2_EncodeSegment_Empty_Ugly (renamed to
// EncodeGPT2Segment_Empty_Ugly): encodeGPT2Segment on an empty segment
// short-circuits to nil.
func TestTokenizer_EncodeGPT2Segment_Empty_Ugly(t *testing.T) {
	tok, err := LoadTokenizer(writeTokenizerJSON(t, gpt2TokenizerJSON))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	if got := tok.encodeGPT2Segment(""); got != nil {
		t.Errorf("encodeGPT2Segment(\"\") = %v, want nil", got)
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
	for i := range tokenizerBPECacheLimit {
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

// --- popDirect sift-down settle (break with left<n) ---

// TestTokenizer_PopDirect_SiftSettles_Good drives popDirect's sift-down to
// the point where the swapped-in root is already smaller than its children, so
// the loop breaks with a live left child (the settle branch) rather than
// running off the bottom. Push candidates with ranks that, after the root pop,
// leave a near-minimal element at the top.
func TestTokenizer_PopDirect_SiftSettles_Good(t *testing.T) {
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

// TestTokenizer_utf8EncodeRune_Good: the inlined UTF-8 encoder writes the correct
// byte sequence and length across all four code-point classes (1/2/3/4 bytes).
// Pure-Go, no tokenizer state.
func TestTokenizer_utf8EncodeRune_Good(t *testing.T) {
	cases := []struct {
		name string
		r    rune
		want []byte
	}{
		{"ascii", 'A', []byte{0x41}},
		{"two-byte", 'é', []byte{0xC3, 0xA9}},              // U+00E9
		{"three-byte", '€', []byte{0xE2, 0x82, 0xAC}},      // U+20AC
		{"four-byte", '😀', []byte{0xF0, 0x9F, 0x98, 0x80}}, // U+1F600
	}
	for _, c := range cases {
		var buf [4]byte
		n := utf8EncodeRune(buf[:], c.r)
		if n != len(c.want) {
			t.Errorf("%s: length = %d, want %d", c.name, n, len(c.want))
			continue
		}
		for i := range c.want {
			if buf[i] != c.want[i] {
				t.Errorf("%s: byte[%d] = %#x, want %#x", c.name, i, buf[i], c.want[i])
			}
		}
	}
}

// TestTokenizer_decodeGPT2Bytes_Good: GPT-2 byte-level decoding maps each Unicode
// placeholder rune back to its original byte via the decoder table, and an empty
// string short-circuits. A synthetic two-entry table proves the mapping without
// loading a real tokenizer.
func TestTokenizer_decodeGPT2Bytes_Good(t *testing.T) {
	// 'Ġ' (U+0120) is GPT-2's placeholder for a space (0x20); map 'A'→0x41.
	tok := &Tokenizer{gpt2Decoder: map[rune]byte{'Ġ': 0x20, 'A': 0x41}}

	if got := tok.decodeGPT2Bytes(""); got != "" {
		t.Errorf("decodeGPT2Bytes(\"\") = %q, want empty", got)
	}
	if got := tok.decodeGPT2Bytes("ĠA"); got != " A" {
		t.Errorf("decodeGPT2Bytes(\"ĠA\") = %q, want \" A\"", got)
	}
}

// TestTokenizer_decodeGPT2Bytes_Ugly: an unmapped rune falls through to its raw
// UTF-8 encoding (the safety-net branch), so a mix of mapped and unmapped runes
// decodes to the mapped bytes followed by the literal UTF-8 of the stray rune.
func TestTokenizer_decodeGPT2Bytes_Ugly(t *testing.T) {
	tok := &Tokenizer{gpt2Decoder: map[rune]byte{'A': 0x41}}
	// 'A' maps to 'A'; '€' is unmapped → its 3 UTF-8 bytes pass through.
	got := tok.decodeGPT2Bytes("A€")
	want := string([]byte{0x41, 0xE2, 0x82, 0xAC})
	if got != want {
		t.Errorf("decodeGPT2Bytes(\"A€\") = %q, want %q", got, want)
	}
}
