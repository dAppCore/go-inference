// SPDX-Licence-Identifier: EUPL-1.2

package rwkv7

import (
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"
)

// mkVocabHexFile writes a tiny hex-per-line vocab fixture (id = 1-based line number) to a temp file and
// returns its path — the same on-disk shape as testdata/rwkv_vocab_v20230424.hex, small enough to hand-
// verify. Tokens: 1="a" 2="b" 3="ab" 4=" " 5="ba" — a deliberately ambiguous set (both "a"+"b" AND the
// longer "ab" are valid) to exercise greedy-longest-match.
func mkVocabHexFile(t *testing.T) string {
	t.Helper()
	toks := [][]byte{[]byte("a"), []byte("b"), []byte("ab"), []byte(" "), []byte("ba")}
	var lines string
	for _, tok := range toks {
		lines += hex.EncodeToString(tok) + "\n"
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "vocab.hex")
	if err := os.WriteFile(path, []byte(lines), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	return path
}

func TestTokenizer_LoadWorldTokenizerHex_Good(t *testing.T) {
	tok, err := LoadWorldTokenizerHex(mkVocabHexFile(t))
	if err != nil {
		t.Fatalf("LoadWorldTokenizerHex: %v", err)
	}
	if len(tok.toBytes) != 5 {
		t.Fatalf("loaded %d tokens, want 5", len(tok.toBytes))
	}
	if string(tok.toBytes[3]) != "ab" {
		t.Fatalf("id 3 = %q, want \"ab\"", tok.toBytes[3])
	}
}

func TestTokenizer_LoadWorldTokenizerHex_Bad(t *testing.T) {
	if _, err := LoadWorldTokenizerHex("/nonexistent/rwkv_vocab.hex"); err == nil {
		t.Fatal("nonexistent path accepted")
	}
}

// TestTokenizer_LoadWorldTokenizerHex_Ugly rejects a line that is not valid hex, distinct from _Bad's
// missing-file case.
func TestTokenizer_LoadWorldTokenizerHex_Ugly(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.hex")
	if err := os.WriteFile(path, []byte("00\nZZ\n"), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	if _, err := LoadWorldTokenizerHex(path); err == nil {
		t.Fatal("non-hex line accepted")
	}
}

// TestTokenizer_WorldTokenizer_Encode_Good proves GREEDY LONGEST match: "ab" must encode as the single
// token id 3, not the two tokens id 1 + id 2, even though both are valid vocabulary entries.
func TestTokenizer_WorldTokenizer_Encode_Good(t *testing.T) {
	tok, err := LoadWorldTokenizerHex(mkVocabHexFile(t))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	got := tok.Encode("ab")
	if len(got) != 1 || got[0] != 3 {
		t.Fatalf("Encode(\"ab\") = %v, want [3] (longest match, not [1 2])", got)
	}
}

// TestTokenizer_WorldTokenizer_Encode_Bad proves a byte with NO vocabulary entry at all is skipped
// (stays well-defined) rather than looping forever or panicking.
func TestTokenizer_WorldTokenizer_Encode_Bad(t *testing.T) {
	tok, err := LoadWorldTokenizerHex(mkVocabHexFile(t))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	got := tok.Encode("ac") // 'a' is vocab, 'c' is not
	if len(got) != 1 || got[0] != 1 {
		t.Fatalf("Encode(\"ac\") = %v, want [1] ('a' matched, 'c' skipped)", got)
	}
}

// TestTokenizer_WorldTokenizer_Encode_Ugly is the round-trip invariant: Decode(Encode(s)) reproduces s
// for text fully covered by the vocabulary.
func TestTokenizer_WorldTokenizer_Encode_Ugly(t *testing.T) {
	tok, err := LoadWorldTokenizerHex(mkVocabHexFile(t))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	const text = "ab ba a b"
	ids := tok.Encode(text)
	got := tok.Decode(ids)
	if got != text {
		t.Fatalf("Decode(Encode(%q)) = %q, want round-trip", text, got)
	}
}

func TestTokenizer_WorldTokenizer_Decode_Good(t *testing.T) {
	tok, err := LoadWorldTokenizerHex(mkVocabHexFile(t))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if got := tok.Decode([]int32{1, 2}); got != "ab" {
		t.Fatalf("Decode([1 2]) = %q, want \"ab\"", got)
	}
}

// TestTokenizer_WorldTokenizer_Decode_Ugly proves an out-of-range id is silently skipped rather than
// panicking or corrupting the surrounding text.
func TestTokenizer_WorldTokenizer_Decode_Ugly(t *testing.T) {
	tok, err := LoadWorldTokenizerHex(mkVocabHexFile(t))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if got := tok.Decode([]int32{1, 999, 2}); got != "ab" {
		t.Fatalf("Decode with an out-of-range id = %q, want \"ab\" (999 skipped)", got)
	}
}
