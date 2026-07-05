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

func TestTokenizer_LoadTokenizer_MissingFile_Bad(t *testing.T) {
	_, err := LoadTokenizer("/nonexistent/tokenizer.json")
	if err == nil {
		t.Error("expected error for missing file")
	}
}

func TestTokenizer_LoadTokenizer_InvalidJSON_Ugly(t *testing.T) {
	dir := t.TempDir()
	path := core.JoinPath(dir, "tokenizer.json")
	_ = coreio.Local.Write(path, "not json")

	_, err := LoadTokenizer(path)
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestTokenizer_BOSEOS_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	if tok.BOSToken() != 100 {
		t.Errorf("BOS = %d, want 100", tok.BOSToken())
	}
	if tok.EOSToken() != 101 {
		t.Errorf("EOS = %d, want 101", tok.EOSToken())
	}
}

func TestTokenizer_Gemma4TurnEndIsEOS_Good(t *testing.T) {
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

func TestTokenizer_Gemma4DoesNotInventPrefixSpace_Good(t *testing.T) {
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

func TestTokenizer_Lookups_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	if tok.BOS() != 100 {
		t.Fatalf("BOS() = %d, want 100", tok.BOS())
	}
	if tok.EOS() != 101 {
		t.Fatalf("EOS() = %d, want 101", tok.EOS())
	}
	id, ok := tok.TokenID("he")
	if !ok || id != 5 {
		t.Fatalf("TokenID(\"he\") = (%d, %t), want (5, true)", id, ok)
	}
	if tok.IDToken(6) != "ll" {
		t.Fatalf("IDToken(6) = %q, want %q", tok.IDToken(6), "ll")
	}
}

func TestTokenizer_NoSpecialTokens_DoesNotInventBOSOrEOS_Good(t *testing.T) {
	path := writeTokenizerWithoutSpecials(t)
	tok, err := LoadTokenizer(path)
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}

	if tok.HasBOSToken() {
		t.Fatal("HasBOSToken() = true, want false")
	}
	if tok.HasEOSToken() {
		t.Fatal("HasEOSToken() = true, want false")
	}
	if tok.BOSToken() != 0 {
		t.Fatalf("BOSToken() = %d, want 0 zero value", tok.BOSToken())
	}
	if tok.EOSToken() != 0 {
		t.Fatalf("EOSToken() = %d, want 0 zero value", tok.EOSToken())
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
	got := tok.bpeMerge(symbols)
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

	got := tok.bpeMerge([]string{"a", "b", "c", "d"})
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

	got := tok.bpeMerge([]string{"a", "b", "c", "d"})
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
	got := tok.bpeMerge(symbols)
	if len(got) != 3 {
		t.Errorf("bpeMerge with no merges = %v, want [a b c]", got)
	}
}

func TestTokenizer_BPEMerge_SingleSymbol_Good(t *testing.T) {
	tok := &Tokenizer{mergeRanks: map[mergeKey]int{{a: "a", b: "b"}: 0}}
	got := tok.bpeMerge([]string{"x"})
	if len(got) != 1 || got[0] != "x" {
		t.Errorf("bpeMerge single = %v, want [x]", got)
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

func TestTokenizer_Decode_SpecialTokensSkipped_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	// Decoding BOS/EOS should produce empty string
	text := tok.Decode([]int32{100, 101})
	if text != "" {
		t.Errorf("Decode(BOS, EOS) = %q, want empty", text)
	}
}

func TestTokenizer_Decode_RegularTokens_Good(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	// Decode known vocab entries
	text := tok.Decode([]int32{5, 6, 3}) // "he" + "ll" + "o"
	if text != "hello" {
		t.Errorf("Decode = %q, want %q", text, "hello")
	}
}

func TestTokenizer_DecodeToken_Regular_Good(t *testing.T) {
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

func TestTokenizer_DecodeToken_Unknown_Bad(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	text := tok.DecodeToken(9999)
	if text != "" {
		t.Errorf("DecodeToken(unknown) = %q, want empty", text)
	}
}

// DecodeOne mirrors Decode([]int32{id}) — verify byte-exact equivalence on
// regular, SentencePiece-prefixed, special, and unknown ids. This is the
// contract IDToken depends on for its no-allocation fast path.
func TestTokenizer_DecodeOne_MatchesDecodeSingle_Good(t *testing.T) {
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

func TestTokenizer_FormatGemmaPrompt_Good(t *testing.T) {
	got := FormatGemmaPrompt("What is 2+2?")
	want := "<bos><start_of_turn>user\nWhat is 2+2?<end_of_turn>\n<start_of_turn>model\n"
	if got != want {
		t.Errorf("FormatGemmaPrompt = %q, want %q", got, want)
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

// TestTokenizer_Encode_EmptyString_Ugly tests encoding an empty string.
// Should return only the BOS token (no panic, no out-of-bounds).
func TestTokenizer_Encode_EmptyString_Ugly(t *testing.T) {
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

// TestTokenizer_Decode_EmptySlice_Ugly tests decoding an empty token slice.
// Should return empty string without panicking.
func TestTokenizer_Decode_EmptySlice_Ugly(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	text := tok.Decode([]int32{})
	if text != "" {
		t.Errorf("Decode(empty) = %q, want empty string", text)
	}
}

// TestTokenizer_DecodeToken_UnknownID_Ugly tests decoding a token ID outside vocab range.
// Should return empty string without panicking.
func TestTokenizer_DecodeToken_UnknownID_Ugly(t *testing.T) {
	path := writeTestTokenizer(t)
	tok, _ := LoadTokenizer(path)

	// Use a large ID well outside any realistic vocab range
	text := tok.DecodeToken(1 << 30)
	if text != "" {
		t.Errorf("DecodeToken(huge id) = %q, want empty", text)
	}
}

// TestTokenizer_BPEMerge_NilSymbols_Ugly tests bpeMerge with an empty symbols slice.
// Should return empty slice without panicking.
func TestTokenizer_BPEMerge_NilSymbols_Ugly(t *testing.T) {
	tok := &Tokenizer{mergeRanks: map[mergeKey]int{{a: "a", b: "b"}: 0}}
	got := tok.bpeMerge([]string{})
	if len(got) != 0 {
		t.Errorf("bpeMerge(empty) = %v, want empty", got)
	}
}

// TestTokenizer_LoadTokenizer_EmptyFile_Ugly tests loading a tokenizer from an empty file.
// Should return a parse error, not panic.
func TestTokenizer_LoadTokenizer_EmptyFile_Ugly(t *testing.T) {
	dir := t.TempDir()
	path := core.JoinPath(dir, "tokenizer.json")
	_ = coreio.Local.Write(path, "")

	_, err := LoadTokenizer(path)
	if err == nil {
		t.Error("expected error for empty tokenizer file")
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
