// SPDX-Licence-Identifier: EUPL-1.2

package spine

import core "dappco.re/go"

type TokenizerImpl interface {
	Encode(string) []int32
	Decode([]int32) string
	// DecodeOne mirrors Decode([]int32{id}) semantics for a single ID
	// without forcing the caller to allocate a one-element slice header.
	// Hot path: Tokenizer.IDToken fires per emitted generation token.
	DecodeOne(int32) string
	TokenID(string) (int32, bool)
	IDToken(int32) string
	BOS() int32
	EOS() int32
	HasBOSToken() bool
}

// Tokenizer wraps a pure-Go tokenizer implementation with the API the
// root mlx package re-exports (`type Tokenizer = spine.Tokenizer`).
type Tokenizer struct {
	tok TokenizerImpl
}

// NewTokenizer wraps a TokenizerImpl in the Tokenizer API. It is the
// bring-your-own-tokenizer seam: callers build a Tokenizer from any
// implementation without reaching the unexported field.
//
//	tok := spine.NewTokenizer(myImpl)
//
// Returns *Tokenizer to match the pointer-receiver method set (Encode/Decode/…)
// and the &Tokenizer{} construction it replaces.
func NewTokenizer(impl TokenizerImpl) *Tokenizer {
	return &Tokenizer{tok: impl}
}

func stripImplicitBOS(tok TokenizerImpl, tokens []int32) []int32 {
	if tok == nil || len(tokens) == 0 {
		return tokens
	}
	if tok.HasBOSToken() && tokens[0] == tok.BOS() {
		return tokens[1:]
	}
	return tokens
}

func hasExplicitBOSPrefix(tok TokenizerImpl, text string) bool {
	if tok == nil || !tok.HasBOSToken() {
		return false
	}
	bosText := tok.IDToken(tok.BOS())
	return bosText != "" && core.HasPrefix(text, bosText)
}

func stripImplicitBOSForText(tok TokenizerImpl, text string, tokens []int32) []int32 {
	if hasExplicitBOSPrefix(tok, text) {
		return tokens
	}
	return stripImplicitBOS(tok, tokens)
}

// Valid reports whether the wrapper holds a live tokenizer implementation.
// It is the exported form of the `t == nil || t.tok == nil` guard the root
// package ran against the unexported field before the spine extraction.
func (t *Tokenizer) Valid() bool {
	return t != nil && t.tok != nil
}

// Encode converts text to token IDs without the model-internal implicit BOS token.
func (t *Tokenizer) Encode(text string) ([]int32, error) {
	if t == nil || t.tok == nil {
		return nil, core.NewError("mlx: tokenizer is nil")
	}
	return stripImplicitBOSForText(t.tok, text, t.tok.Encode(text)), nil
}

// Decode converts token IDs back to text.
func (t *Tokenizer) Decode(tokens []int32) (string, error) {
	if t == nil || t.tok == nil {
		return "", core.NewError("mlx: tokenizer is nil")
	}
	return t.tok.Decode(tokens), nil
}

// TokenID resolves a token string to its ID.
func (t *Tokenizer) TokenID(text string) (int32, bool) {
	if t == nil || t.tok == nil {
		return 0, false
	}
	if id, ok := t.tok.TokenID(text); ok {
		return id, true
	}
	// The public tokenizer API accepts plain-text tokens such as "hello",
	// while the internal tokenizer stores model-native forms like "▁hello".
	encoded := stripImplicitBOSForText(t.tok, text, t.tok.Encode(text))
	if len(encoded) == 1 {
		return encoded[0], true
	}
	return 0, false
}

// IDToken resolves a token ID to a decoded token string when possible.
func (t *Tokenizer) IDToken(id int32) string {
	if t == nil || t.tok == nil {
		return ""
	}
	raw := t.tok.IDToken(id)
	if raw == "" {
		return ""
	}
	// DecodeOne sidesteps the per-call []int32{id} heap escape that the
	// interface-boxed Decode([]int32{id}) path forced — sessionParserTokenText
	// fires this wrapper once per emitted generation token, so a 1-allocs/op
	// → 0-allocs/op flip lands as steady-state pressure relief.
	if decoded := t.tok.DecodeOne(id); decoded != "" {
		return decoded
	}
	if raw == "▁" {
		return " "
	}
	return raw
}

// BOS returns the beginning-of-sequence token ID.
func (t *Tokenizer) BOS() int32 {
	if t == nil || t.tok == nil {
		return 0
	}
	return t.tok.BOS()
}

// EOS returns the end-of-sequence token ID.
func (t *Tokenizer) EOS() int32 {
	if t == nil || t.tok == nil {
		return 0
	}
	return t.tok.EOS()
}
