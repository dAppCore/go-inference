// SPDX-Licence-Identifier: EUPL-1.2

package bert

import (
	"testing"
)

// syntheticVocab is a tiny WordPiece vocab (id = line index) covering the
// tokeniser behaviours: specials, whole words, a "##" continuation, punctuation,
// and an out-of-vocab fallback.
const syntheticVocab = "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nthe\nquick\nbrown\nfox\nplay\n##ing\nreset\n!\n?\npassword\n"

func newTestTokenizer(t *testing.T) *Tokenizer {
	t.Helper()
	tk, err := NewTokenizer([]byte(syntheticVocab), true)
	if err != nil {
		t.Fatalf("NewTokenizer: %v", err)
	}
	return tk
}

// TestNewTokenizer_Bad_MissingSpecial rejects a vocab without the framing tokens.
func TestNewTokenizer_Bad_MissingSpecial(t *testing.T) {
	if _, err := NewTokenizer([]byte("the\nquick\nfox\n"), true); err == nil {
		t.Fatal("expected an error for a vocab missing [CLS]/[SEP]")
	}
}

// TestTokenizer_Encode_Good frames a lower-cased sentence with [CLS]/[SEP] and
// splits a "##" continuation and trailing punctuation.
func TestTokenizer_Encode_Good(t *testing.T) {
	tk := newTestTokenizer(t)
	got := tk.Encode("The quick fox playing!")
	// [CLS] the quick fox play ##ing ! [SEP]
	want := []int32{2, 5, 6, 8, 9, 10, 12, 3}
	assertIDs(t, got, want)
}

// TestTokenizer_Encode_Good_Punctuation isolates each punctuation rune as a token.
func TestTokenizer_Encode_Good_Punctuation(t *testing.T) {
	tk := newTestTokenizer(t)
	got := tk.Encode("reset password?")
	// [CLS] reset password ? [SEP]
	want := []int32{2, 11, 14, 13, 3}
	assertIDs(t, got, want)
}

// TestTokenizer_Encode_Ugly_Unknown maps an out-of-vocab word to [UNK].
func TestTokenizer_Encode_Ugly_Unknown(t *testing.T) {
	tk := newTestTokenizer(t)
	got := tk.Encode("zzzz")
	want := []int32{2, 1, 3} // [CLS] [UNK] [SEP]
	assertIDs(t, got, want)
}

// TestTokenizer_Encode_Good_Empty frames an empty string as just [CLS] [SEP].
func TestTokenizer_Encode_Good_Empty(t *testing.T) {
	tk := newTestTokenizer(t)
	got := tk.Encode("   ")
	want := []int32{2, 3}
	assertIDs(t, got, want)
}

func assertIDs(t *testing.T, got, want []int32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("token count = %d, want %d (got %v want %v)", len(got), len(want), got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token[%d] = %d, want %d (got %v want %v)", i, got[i], want[i], got, want)
		}
	}
}
