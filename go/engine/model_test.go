// SPDX-Licence-Identifier: EUPL-1.2

package engine

import (
	"testing"
	"time"

	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/tokenizer"
)

// bareTokenModel is a minimal engine.TokenModel with no optional seams — used to
// prove Capabilities reports no cache modes when the engine declares none.
type bareTokenModel struct{}

func (bareTokenModel) OpenEngineSession() (Session, error) { return nil, nil }
func (bareTokenModel) Close() error                        { return nil }

// cacheModeTokenModel adds the CacheModeReporter seam so Capabilities forwards
// the declared modes.
type cacheModeTokenModel struct {
	bareTokenModel
	modes []string
}

func (m cacheModeTokenModel) SupportedCacheModes() []string { return m.modes }

// TestTextModelCapabilities_CacheModesForwarded_Good pins the capability seam:
// a TokenModel declaring cache modes surfaces them on the report's CacheModes,
// the data `generate` consults for the -kv-cache note.
func TestTextModelCapabilities_CacheModesForwarded_Good(t *testing.T) {
	m := &TextModel{tm: cacheModeTokenModel{modes: []string{"native"}}}
	report := m.Capabilities()
	if len(report.CacheModes) != 1 || report.CacheModes[0] != "native" {
		t.Fatalf("CacheModes = %v, want [native]", report.CacheModes)
	}
}

// TestTextModelCapabilities_NoReporter_Bad pins the absence: a TokenModel without
// the seam leaves CacheModes empty (no invented modes) while the base capability
// set still reports.
func TestTextModelCapabilities_NoReporter_Bad(t *testing.T) {
	m := &TextModel{tm: bareTokenModel{}}
	report := m.Capabilities()
	if len(report.CacheModes) != 0 {
		t.Fatalf("CacheModes = %v, want empty", report.CacheModes)
	}
	if len(report.Capabilities) == 0 {
		t.Fatal("base capability set is empty; expected generate/chat/classify")
	}
}

// TestTextModelSetDecodePhases pins the metrics fold: a traced budget attaches to
// the metrics snapshot the caller reads via Metrics().
func TestTextModelSetDecodePhases(t *testing.T) {
	m := &TextModel{}
	budget := &inference.DecodePhaseBudget{Tokens: 7, TotalPerToken: 2 * time.Millisecond}
	m.setDecodePhases(budget)
	got := m.Metrics().DecodePhases
	if got == nil || got.Tokens != 7 {
		t.Fatalf("Metrics().DecodePhases = %+v, want the 7-token budget", got)
	}
}

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
