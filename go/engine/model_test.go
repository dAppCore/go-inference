// SPDX-Licence-Identifier: EUPL-1.2

package engine

import (
	"testing"

	"dappco.re/go/inference/decode/tokenizer"
)

// spTok builds a decode-only SentencePiece-style tokenizer: ▁-led pieces are
// word-leading (the marker IS the space), bare pieces are continuations.
func spTok() *tokenizer.Tokenizer {
	return tokenizer.NewForDecode(map[int32]string{
		1: "▁hello",
		2: "▁world",
		3: "!",
		4: "hello",
	})
}

// TestTextModelDecodeKeepsWordBoundarySpace pins the STREAMING decode contract:
// concatenating per-token decode output must reassemble the words WITH their
// boundary spaces. The 2026-07-05 regression served "helloworld" for every
// reply because the stream decoded through DecodeOne, whose Decode-of-one
// semantics strip the ▁ boundary.
func TestTextModelDecodeKeepsWordBoundarySpace(t *testing.T) {
	m := &TextModel{tok: spTok()}
	var got string
	for _, id := range []int32{1, 2, 3} {
		got += m.decode(id)
	}
	if want := " hello world!"; got != want {
		t.Fatalf("streamed concat = %q, want %q", got, want)
	}
	if c := m.decode(4); c != "hello" {
		t.Fatalf("continuation piece = %q, want %q (no invented space)", c, "hello")
	}
}

// TestTextModelDecodeLabelStripsBoundary pins the classification contract:
// a standalone label token decodes clean, boundary space stripped — the
// Decode([]int32{id}) semantics classify wants ("▁world" → "world").
func TestTextModelDecodeLabelStripsBoundary(t *testing.T) {
	m := &TextModel{tok: spTok()}
	if got := m.decodeLabel(2); got != "world" {
		t.Fatalf("decodeLabel = %q, want %q", got, "world")
	}
}

// TestTextModelDecodeNilSafe pins the nil-model / nil-tokenizer guards both
// decode variants share.
func TestTextModelDecodeNilSafe(t *testing.T) {
	var nilModel *TextModel
	if got := nilModel.decode(1); got != "" {
		t.Fatalf("nil model decode = %q, want empty", got)
	}
	m := &TextModel{}
	if got := m.decodeLabel(1); got != "" {
		t.Fatalf("nil tok decodeLabel = %q, want empty", got)
	}
}
