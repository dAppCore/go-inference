// SPDX-Licence-Identifier: EUPL-1.2

package spine

import "testing"

// Tests for the Tokenizer wrapper. The wrapper's whole job is the
// nil-guard + implicit-BOS-strip behaviour layered over a TokenizerImpl,
// so the Good/Bad/Ugly split here is: Good = live impl returns the
// expected value; Bad = nil-receiver / nil-impl guard returns the
// documented zero value; Ugly = the BOS-strip edge paths (explicit-BOS
// prefix, leading-BOS strip, no-BOS passthrough). benchFakeTokenizer
// (tokenizer_bench_test.go) is the shared in-package fake.

// --- NewTokenizer / Valid ---

func TestTokenizer_NewTokenizer_Good(t *testing.T) {
	tok := NewTokenizer(&benchFakeTokenizer{bos: 1})
	if tok == nil || !tok.Valid() {
		t.Fatalf("NewTokenizer(impl).Valid() = false, want a live wrapper")
	}
}

func TestTokenizer_NewTokenizer_Bad(t *testing.T) {
	// A wrapper built around a nil impl is not Valid — the guard the
	// root package ran against the unexported field, now exported.
	tok := NewTokenizer(nil)
	if tok == nil {
		t.Fatal("NewTokenizer(nil) returned nil pointer, want non-nil wrapper")
	}
	if tok.Valid() {
		t.Fatal("NewTokenizer(nil).Valid() = true, want false for nil impl")
	}
}

func TestTokenizer_NewTokenizer_Ugly(t *testing.T) {
	// Valid must tolerate a nil *Tokenizer receiver, not panic.
	var tok *Tokenizer
	if tok.Valid() {
		t.Fatal("(*Tokenizer)(nil).Valid() = true, want false")
	}
}

// --- Encode ---

func TestTokenizer_Encode_Good(t *testing.T) {
	// Leading BOS in the result + a BOS-bearing tokenizer → the implicit
	// BOS is stripped (text does not already carry the BOS prefix).
	tok := NewTokenizer(&benchFakeTokenizer{
		ids: []int32{1, 10, 11}, bos: 1, bosText: "<s>", hasBOS: true,
	})
	got, err := tok.Encode("hello")
	if err != nil {
		t.Fatalf("Encode error = %v, want nil", err)
	}
	if len(got) != 2 || got[0] != 10 || got[1] != 11 {
		t.Fatalf("Encode = %v, want [10 11] (leading BOS stripped)", got)
	}
}

func TestTokenizer_Encode_Bad(t *testing.T) {
	// Nil impl → documented error, no panic.
	tok := NewTokenizer(nil)
	got, err := tok.Encode("hello")
	if err == nil {
		t.Fatal("Encode on nil impl = nil error, want 'tokenizer is nil'")
	}
	if got != nil {
		t.Fatalf("Encode on nil impl = %v, want nil tokens", got)
	}
	// Nil receiver must also be guarded.
	var nilTok *Tokenizer
	if _, err := nilTok.Encode("x"); err == nil {
		t.Fatal("(*Tokenizer)(nil).Encode = nil error, want error")
	}
}

func TestTokenizer_Encode_Ugly(t *testing.T) {
	// Text already carries the explicit BOS prefix → the leading BOS is
	// KEPT (hasExplicitBOSPrefix short-circuits the strip). Distinct code
	// path from the Good case above.
	tok := NewTokenizer(&benchFakeTokenizer{
		ids: []int32{1, 10}, bos: 1, bosText: "<s>", hasBOS: true,
	})
	got, err := tok.Encode("<s>hello")
	if err != nil {
		t.Fatalf("Encode error = %v, want nil", err)
	}
	if len(got) != 2 || got[0] != 1 {
		t.Fatalf("Encode(<s>...) = %v, want leading BOS kept [1 10]", got)
	}

	// A tokenizer without a BOS token → passthrough, nothing stripped.
	noBOS := NewTokenizer(&benchFakeTokenizer{ids: []int32{5, 6}, hasBOS: false})
	got, err = noBOS.Encode("hello")
	if err != nil {
		t.Fatalf("Encode (no-BOS) error = %v, want nil", err)
	}
	if len(got) != 2 || got[0] != 5 {
		t.Fatalf("Encode (no-BOS) = %v, want passthrough [5 6]", got)
	}
}

// --- Decode ---

func TestTokenizer_Decode_Good(t *testing.T) {
	tok := NewTokenizer(&benchFakeTokenizer{text: "decoded"})
	got, err := tok.Decode([]int32{1, 2, 3})
	if err != nil {
		t.Fatalf("Decode error = %v, want nil", err)
	}
	if got != "decoded" {
		t.Fatalf("Decode = %q, want %q", got, "decoded")
	}
}

func TestTokenizer_Decode_Bad(t *testing.T) {
	tok := NewTokenizer(nil)
	got, err := tok.Decode([]int32{1})
	if err == nil {
		t.Fatal("Decode on nil impl = nil error, want error")
	}
	if got != "" {
		t.Fatalf("Decode on nil impl = %q, want empty", got)
	}
	var nilTok *Tokenizer
	if _, err := nilTok.Decode([]int32{1}); err == nil {
		t.Fatal("(*Tokenizer)(nil).Decode = nil error, want error")
	}
}

func TestTokenizer_Decode_Ugly(t *testing.T) {
	// Empty token slice — the wrapper forwards to the impl unchanged; the
	// fake returns its seeded text regardless, so we assert the wrapper
	// does not short-circuit or panic on a zero-length slice.
	tok := NewTokenizer(&benchFakeTokenizer{text: ""})
	got, err := tok.Decode(nil)
	if err != nil {
		t.Fatalf("Decode(nil) error = %v, want nil", err)
	}
	if got != "" {
		t.Fatalf("Decode(nil) = %q, want empty", got)
	}
}

// --- TokenID ---

func TestTokenizer_TokenID_Good(t *testing.T) {
	// Direct impl hit — TokenID returns the seeded id.
	tok := NewTokenizer(&benchFakeTokenizer{tokenID: 42, tokenIDOK: true})
	id, ok := tok.TokenID("hello")
	if !ok || id != 42 {
		t.Fatalf("TokenID = (%d,%v), want (42,true)", id, ok)
	}
}

func TestTokenizer_TokenID_Bad(t *testing.T) {
	// Nil impl → (0,false); nil receiver guarded too.
	tok := NewTokenizer(nil)
	if id, ok := tok.TokenID("x"); ok || id != 0 {
		t.Fatalf("TokenID on nil impl = (%d,%v), want (0,false)", id, ok)
	}
	var nilTok *Tokenizer
	if id, ok := nilTok.TokenID("x"); ok || id != 0 {
		t.Fatalf("(*Tokenizer)(nil).TokenID = (%d,%v), want (0,false)", id, ok)
	}
}

func TestTokenizer_TokenID_Ugly(t *testing.T) {
	// Direct lookup misses → Encode fallback. A single-token encode (after
	// BOS strip) resolves to that token id; a multi-token encode does not.
	single := NewTokenizer(&benchFakeTokenizer{
		tokenIDOK: false,
		ids:       []int32{1, 99}, // BOS + one real token
		bos:       1, bosText: "<s>", hasBOS: true,
	})
	if id, ok := single.TokenID("hello"); !ok || id != 99 {
		t.Fatalf("TokenID fallback (single) = (%d,%v), want (99,true)", id, ok)
	}

	multi := NewTokenizer(&benchFakeTokenizer{
		tokenIDOK: false,
		ids:       []int32{1, 7, 8}, // BOS + two real tokens → not a single token
		bos:       1, bosText: "<s>", hasBOS: true,
	})
	if id, ok := multi.TokenID("hello world"); ok || id != 0 {
		t.Fatalf("TokenID fallback (multi) = (%d,%v), want (0,false)", id, ok)
	}
}

// --- IDToken ---

func TestTokenizer_IDToken_Good(t *testing.T) {
	// DecodeOne returns a non-empty string → that wins.
	tok := NewTokenizer(&benchFakeTokenizer{idTokenStr: "hello", text: "hello"})
	if got := tok.IDToken(42); got != "hello" {
		t.Fatalf("IDToken = %q, want %q", got, "hello")
	}

	// Raw IDToken non-empty, but DecodeOne returns "" and the raw form is
	// not the bare SentencePiece space — the wrapper falls through to the
	// final `return raw`. (A token with a vocab entry but no decode form.)
	fallback := NewTokenizer(&benchFakeTokenizer{idTokenStr: "x", text: ""})
	if got := fallback.IDToken(7); got != "x" {
		t.Fatalf("IDToken (raw fallthrough) = %q, want %q", got, "x")
	}
}

func TestTokenizer_IDToken_Bad(t *testing.T) {
	// Nil impl → empty string; nil receiver guarded.
	tok := NewTokenizer(nil)
	if got := tok.IDToken(42); got != "" {
		t.Fatalf("IDToken on nil impl = %q, want empty", got)
	}
	var nilTok *Tokenizer
	if got := nilTok.IDToken(42); got != "" {
		t.Fatalf("(*Tokenizer)(nil).IDToken = %q, want empty", got)
	}
	// Raw IDToken empty → the whole wrapper returns "" before DecodeOne.
	empty := NewTokenizer(&benchFakeTokenizer{idTokenStr: ""})
	if got := empty.IDToken(42); got != "" {
		t.Fatalf("IDToken (raw empty) = %q, want empty", got)
	}
}

func TestTokenizer_IDToken_Ugly(t *testing.T) {
	// SentencePiece bare-space: raw IDToken = "▁", DecodeOne returns ""
	// (a lone "▁" strips to empty), so the wrapper falls through to the
	// `raw == "▁"` substitution and yields a literal space.
	tok := NewTokenizer(&benchFakeTokenizer{idTokenStr: "▁", text: ""})
	if got := tok.IDToken(42); got != " " {
		t.Fatalf("IDToken (▁) = %q, want a single space", got)
	}
}

// --- BOS / EOS ---

func TestTokenizer_BOS_Good(t *testing.T) {
	tok := NewTokenizer(&benchFakeTokenizer{bos: 1})
	if got := tok.BOS(); got != 1 {
		t.Fatalf("BOS = %d, want 1", got)
	}
}

func TestTokenizer_BOS_Bad(t *testing.T) {
	if got := NewTokenizer(nil).BOS(); got != 0 {
		t.Fatalf("BOS on nil impl = %d, want 0", got)
	}
	var nilTok *Tokenizer
	if got := nilTok.BOS(); got != 0 {
		t.Fatalf("(*Tokenizer)(nil).BOS = %d, want 0", got)
	}
}

func TestTokenizer_EOS_Good(t *testing.T) {
	// benchFakeTokenizer.EOS is hard-wired to 2.
	tok := NewTokenizer(&benchFakeTokenizer{})
	if got := tok.EOS(); got != 2 {
		t.Fatalf("EOS = %d, want 2", got)
	}
}

func TestTokenizer_EOS_Bad(t *testing.T) {
	if got := NewTokenizer(nil).EOS(); got != 0 {
		t.Fatalf("EOS on nil impl = %d, want 0", got)
	}
	var nilTok *Tokenizer
	if got := nilTok.EOS(); got != 0 {
		t.Fatalf("(*Tokenizer)(nil).EOS = %d, want 0", got)
	}
}
