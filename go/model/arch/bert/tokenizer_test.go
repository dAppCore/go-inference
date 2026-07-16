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

func TestTokenizer_NewTokenizer_Good(t *testing.T) {
	tk, err := NewTokenizer([]byte(syntheticVocab), true)
	if err != nil {
		t.Fatalf("NewTokenizer: %v", err)
	}
	if tk.PadID() != 0 {
		t.Fatalf("PadID() = %d, want 0", tk.PadID())
	}
}

// TestTokenizer_NewTokenizer_Bad rejects a vocab without the framing tokens.
func TestTokenizer_NewTokenizer_Bad(t *testing.T) {
	if _, err := NewTokenizer([]byte("the\nquick\nfox\n"), true); err == nil {
		t.Fatal("expected an error for a vocab missing [CLS]/[SEP]")
	}
}

// TestTokenizer_NewTokenizer_Ugly proves a CRLF vocab.txt (trailing \r per
// line) is handled — the \r must not become part of the token, so the
// special-token lookups (which require an exact match) still succeed.
// Distinct from _Bad's missing-token rejection.
func TestTokenizer_NewTokenizer_Ugly(t *testing.T) {
	crlfVocab := "[PAD]\r\n[UNK]\r\n[CLS]\r\n[SEP]\r\n[MASK]\r\nthe\r\n"
	tk, err := NewTokenizer([]byte(crlfVocab), true)
	if err != nil {
		t.Fatalf("NewTokenizer rejected a CRLF vocab: %v", err)
	}
	if tk.PadID() != 0 {
		t.Fatalf("PadID() = %d, want 0 (CRLF must not corrupt the special-token ids)", tk.PadID())
	}
}

func TestTokenizer_PadID_Good(t *testing.T) {
	tk := newTestTokenizer(t)
	if got := tk.PadID(); got != 0 {
		t.Fatalf("PadID() = %d, want 0 ([PAD] is vocab line 0)", got)
	}
}

// TestTokenizer_PadID_Bad proves an uninitialised Tokenizer (never built via
// NewTokenizer) reports 0 honestly rather than panicking.
func TestTokenizer_PadID_Bad(t *testing.T) {
	tk := &Tokenizer{}
	if got := tk.PadID(); got != 0 {
		t.Fatalf("PadID() = %d, want 0 for an uninitialised Tokenizer", got)
	}
}

// TestTokenizer_PadID_Ugly proves PadID reflects wherever [PAD] actually sits
// in the vocab — not hardcoded to 0 — distinct from _Bad's zero-value case.
func TestTokenizer_PadID_Ugly(t *testing.T) {
	vocab := "[UNK]\n[CLS]\n[SEP]\n[MASK]\n[PAD]\n"
	tk, err := NewTokenizer([]byte(vocab), true)
	if err != nil {
		t.Fatalf("NewTokenizer: %v", err)
	}
	if got := tk.PadID(); got != 4 {
		t.Fatalf("PadID() = %d, want 4 ([PAD] is vocab line 4, not hardcoded to 0)", got)
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

// TestTokenizer_Encode_Ugly maps an out-of-vocab word to [UNK].
func TestTokenizer_Encode_Ugly(t *testing.T) {
	tk := newTestTokenizer(t)
	got := tk.Encode("zzzz")
	want := []int32{2, 1, 3} // [CLS] [UNK] [SEP]
	assertIDs(t, got, want)
}

// TestTokenizer_Encode_Bad proves control characters (NUL etc.) are dropped by
// cleanText rather than corrupting the surrounding tokens — a malformed-input
// case handled gracefully, distinct from _Ugly's out-of-vocab-word case.
func TestTokenizer_Encode_Bad(t *testing.T) {
	tk := newTestTokenizer(t)
	got := tk.Encode("\x00\x01reset\x00")
	want := []int32{2, 11, 3} // [CLS] reset [SEP]
	assertIDs(t, got, want)
}

// TestTokenizer_Encode_Good_Empty frames an empty string as just [CLS] [SEP].
func TestTokenizer_Encode_Good_Empty(t *testing.T) {
	tk := newTestTokenizer(t)
	got := tk.Encode("   ")
	want := []int32{2, 3}
	assertIDs(t, got, want)
}

func TestTokenizer_EncodePair_Good(t *testing.T) {
	tk := newTestTokenizer(t)
	ids, types := tk.EncodePair("reset password", "the quick fox")
	assertIDs(t, ids, []int32{2, 11, 14, 3, 5, 6, 8, 3})
	assertIDs(t, types, []int32{0, 0, 0, 0, 1, 1, 1, 1})
}

func TestTokenizer_EncodePair_Bad(t *testing.T) {
	tk := newTestTokenizer(t)
	ids, types := tk.EncodePair("zzzz", "yyyy")
	assertIDs(t, ids, []int32{2, 1, 3, 1, 3})
	assertIDs(t, types, []int32{0, 0, 0, 1, 1})
}

func TestTokenizer_EncodePair_Ugly(t *testing.T) {
	tk := newTestTokenizer(t)
	ids, types := tk.EncodePair("", "")
	assertIDs(t, ids, []int32{2, 3, 3})
	assertIDs(t, types, []int32{0, 0, 1})
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
