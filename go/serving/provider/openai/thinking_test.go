// SPDX-Licence-Identifier: EUPL-1.2

// Tests for reasoning-channel extraction.
package openai

import (
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestOpenAI_ThinkingExtractor_Good_CapturesQwenAndChannelMarkers(t *testing.T) {
	extractor := NewThinkingExtractor()

	visible, thought := extractor.Process(inference.Token{Text: "A <thi"})
	visible2, thought2 := extractor.Process(inference.Token{Text: "nk>hidden</think> B <|channel>thought plan"})
	visible3, thought3 := extractor.Process(inference.Token{Text: "<|channel>assistant C"})
	visible4, thought4 := extractor.Flush()

	gotVisible := visible + visible2 + visible3 + visible4
	gotThought := thought + thought2 + thought3 + thought4
	if gotVisible != "A  B  C" {
		t.Fatalf("visible = %q", gotVisible)
	}
	if gotThought != "hidden plan" {
		t.Fatalf("thought = %q", gotThought)
	}
	if extractor.Content() != gotVisible || extractor.Thinking() != gotThought {
		t.Fatalf("extractor content/thought = %q/%q", extractor.Content(), extractor.Thinking())
	}
}

func TestOpenAI_ThinkingExtractor_Ugly_IncompleteChannelMarkerDoesNotHang(t *testing.T) {
	extractor := NewThinkingExtractor()
	done := make(chan struct{})
	go func() {
		extractor.Process(inference.Token{Text: "<|channel>"})
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(100 * time.Millisecond):
		t.Fatal("Process() hung on incomplete channel marker")
	}
	visible, thought := extractor.Flush()
	if visible != "<|channel>" || thought != "" {
		t.Fatalf("Flush() = %q/%q", visible, thought)
	}
}

func TestOpenAI_ThinkingExtractor_Good_Gemma4ChannelCloseSwitchesToContent(t *testing.T) {
	// Gemma4 terminates its reasoning with the <channel|> CLOSE marker —
	// distinct from gpt-oss's <|channel> OPEN. Everything after the close
	// is the visible answer and must reach content, not be swallowed as
	// thinking (which left chat-completions content empty). go-mlx #48.
	extractor := NewThinkingExtractor()

	visible, thought := extractor.Process(inference.Token{Text: "<|channel>thought\nadd two and two<channel|>4"})
	visible2, thought2 := extractor.Flush()

	gotVisible := visible + visible2
	gotThought := thought + thought2
	if gotVisible != "4" {
		t.Fatalf("visible = %q, want %q", gotVisible, "4")
	}
	if gotThought != "\nadd two and two" {
		t.Fatalf("thought = %q, want %q", gotThought, "\nadd two and two")
	}
	if extractor.Content() != "4" {
		t.Fatalf("Content() = %q, want %q", extractor.Content(), "4")
	}
}

func TestOpenAI_ThinkingExtractor_Ugly_Gemma4ChannelCloseSplitAcrossTokens(t *testing.T) {
	// The <channel|> close can straddle a streaming token boundary. The
	// safe-suffix split must hold a partial close marker back so it is
	// recognised once complete, not mis-emitted as thinking. go-mlx #48.
	extractor := NewThinkingExtractor()

	v1, th1 := extractor.Process(inference.Token{Text: "<|channel>thought\nadd<chan"})
	v2, th2 := extractor.Process(inference.Token{Text: "nel|>4"})
	v3, th3 := extractor.Flush()

	gotVisible := v1 + v2 + v3
	gotThought := th1 + th2 + th3
	if gotVisible != "4" {
		t.Fatalf("visible = %q, want %q", gotVisible, "4")
	}
	if gotThought != "\nadd" {
		t.Fatalf("thought = %q, want %q", gotThought, "\nadd")
	}
}

// TestOpenAI_ThinkingExtractor_Bad_NilReceiver pins the nil-receiver
// guard on every exported method — a caller holding a *ThinkingExtractor
// obtained from a failed constructor path must get safe zero values,
// never a panic.
func TestOpenAI_ThinkingExtractor_Bad_NilReceiver(t *testing.T) {
	var e *ThinkingExtractor

	if content, thought := e.Process(inference.Token{Text: "x"}); content != "" || thought != "" {
		t.Fatalf("nil.Process() = %q, %q, want empty/empty", content, thought)
	}
	if content, thought := e.Flush(); content != "" || thought != "" {
		t.Fatalf("nil.Flush() = %q, %q, want empty/empty", content, thought)
	}
	if got := e.Content(); got != "" {
		t.Fatalf("nil.Content() = %q, want empty", got)
	}
	if got := e.Thinking(); got != "" {
		t.Fatalf("nil.Thinking() = %q, want empty", got)
	}
}

// TestOpenAI_ThinkingExtractor_Good_MidBufferPairedMarker sends plain
// content and a complete <think>...</think> span in a SINGLE Process
// call. Every other paired-marker test in this file feeds the marker
// at the very start of a pending buffer (reached by
// consumeMarkerAtStart before the mid-buffer search ever runs) — this
// is the only path that exercises earliestReasoningStart finding a
// marker at idx>0 and the resulting pairedEndFor lookup.
func TestOpenAI_ThinkingExtractor_Good_MidBufferPairedMarker(t *testing.T) {
	extractor := NewThinkingExtractor()

	visible, thought := extractor.Process(inference.Token{Text: "Sure. <think>hidden plan</think> answer"})

	if visible != "Sure.  answer" {
		t.Fatalf("visible = %q, want %q", visible, "Sure.  answer")
	}
	if thought != "hidden plan" {
		t.Fatalf("thought = %q, want %q", thought, "hidden plan")
	}
}

// TestOpenAI_ThinkingExtractor_Ugly_PairedMarkerNeverCloses drains a
// <think> span that never receives its closing tag before Flush —
// the pending text must still surface as thinking rather than being
// dropped, and splitSafeSuffixOne's final=true fast path (return the
// whole pending string, nothing held back) must fire since there is
// no more streaming input coming.
func TestOpenAI_ThinkingExtractor_Ugly_PairedMarkerNeverCloses(t *testing.T) {
	extractor := NewThinkingExtractor()

	visible1, thought1 := extractor.Process(inference.Token{Text: "<think>partial reasoning, no close"})
	visible2, thought2 := extractor.Flush()

	if got := visible1 + visible2; got != "" {
		t.Fatalf("visible = %q, want empty (unclosed reasoning never reaches content)", got)
	}
	if got := thought1 + thought2; got != "partial reasoning, no close" {
		t.Fatalf("thought = %q, want %q", got, "partial reasoning, no close")
	}
}

// TestOpenAI_ThinkingExtractor_Ugly_PairedMarkerSuffixStraddle mirrors
// the Gemma4-close straddle test for the generic <think>/</think>
// pair, driving splitSafeSuffixOne's partial-match branch (as opposed
// to splitSafeSuffix, which handles the multi-marker open-tag search).
func TestOpenAI_ThinkingExtractor_Ugly_PairedMarkerSuffixStraddle(t *testing.T) {
	extractor := NewThinkingExtractor()

	v1, th1 := extractor.Process(inference.Token{Text: "<think>hidden</thi"})
	v2, th2 := extractor.Process(inference.Token{Text: "nk> visible"})
	v3, th3 := extractor.Flush()

	gotVisible := v1 + v2 + v3
	gotThought := th1 + th2 + th3
	if gotVisible != " visible" {
		t.Fatalf("visible = %q, want %q", gotVisible, " visible")
	}
	if gotThought != "hidden" {
		t.Fatalf("thought = %q, want %q", gotThought, "hidden")
	}
}

// TestOpenAI_ThinkingExtractor_Ugly_NonASCIIAfterChannelMarker sends a
// non-ASCII byte immediately after "<|channel>" — consumeMarkerAtStart
// must decode it via utf8Rune (rather than indexing a single byte) to
// correctly decide it is not channel-name whitespace, then correctly
// decide it is not a channel-name character either, holding the whole
// marker back as literal content once Flush forces the issue.
func TestOpenAI_ThinkingExtractor_Ugly_NonASCIIAfterChannelMarker(t *testing.T) {
	extractor := NewThinkingExtractor()

	visible1, thought1 := extractor.Process(inference.Token{Text: "pre <|channel>éfoo bar"})
	visible2, thought2 := extractor.Flush()

	gotVisible := visible1 + visible2
	gotThought := thought1 + thought2
	if gotVisible != "pre <|channel>éfoo bar" {
		t.Fatalf("visible = %q, want %q", gotVisible, "pre <|channel>éfoo bar")
	}
	if gotThought != "" {
		t.Fatalf("thought = %q, want empty (unrecognised channel name falls back to literal content)", gotThought)
	}
}

// TestOpenAI_ThinkingExtractor_Ugly_ChannelReopenWithinThought covers
// a SECOND "<|channel>" marker appearing mid-buffer while already
// inside a reasoning channel — reached via the thought-branch's own
// openIdx/closeIdx scan (drain lines ~152-159), not the top-of-loop
// consumeMarkerAtStart fast path (which only matches when the marker
// is the very first byte of pending). The reopen's channel name is
// unparseable ("!!!invalid"), so it also drives the hold-until-final
// branch (non-final Process call) and the literal-marker fallback
// once Flush forces the issue.
func TestOpenAI_ThinkingExtractor_Ugly_ChannelReopenWithinThought(t *testing.T) {
	extractor := NewThinkingExtractor()

	visible1, thought1 := extractor.Process(inference.Token{Text: "<|channel>thought stuff<|channel>!!!invalid"})
	visible2, thought2 := extractor.Flush()

	gotVisible := visible1 + visible2
	gotThought := thought1 + thought2
	if gotVisible != "" {
		t.Fatalf("visible = %q, want empty (everything here routes to thought)", gotVisible)
	}
	if gotThought != " stuff<|channel>!!!invalid" {
		t.Fatalf("thought = %q, want %q", gotThought, " stuff<|channel>!!!invalid")
	}
}

// TestOpenAI_ThinkingExtractor_Ugly_PairedMarkerFlushWithoutFollowup
// pins splitSafeSuffixOne's final=true fast path specifically (as
// opposed to PairedMarkerSuffixStraddle, which resolves the straddle
// via a second Process call instead of Flush): a partial "</think>"
// suffix is held back by one non-final Process call, then Flush —
// with no further input arriving — must emit the held-back partial
// marker as literal thought text rather than waiting forever.
func TestOpenAI_ThinkingExtractor_Ugly_PairedMarkerFlushWithoutFollowup(t *testing.T) {
	extractor := NewThinkingExtractor()

	visible1, thought1 := extractor.Process(inference.Token{Text: "<think>hidden</thi"})
	visible2, thought2 := extractor.Flush()

	gotVisible := visible1 + visible2
	gotThought := thought1 + thought2
	if gotVisible != "" {
		t.Fatalf("visible = %q, want empty", gotVisible)
	}
	if gotThought != "hidden</thi" {
		t.Fatalf("thought = %q, want %q (the held-back partial marker surfaces as literal thought text)", gotThought, "hidden</thi")
	}
}

// TestOpenAI_ThinkingExtractor_Good_ChannelReopenWithRecognisedName
// mirrors ChannelReopenWithinThought but with a VALID channel name
// after the reopen marker ("assistant" instead of "!!!invalid") — the
// only way to reach consumeMarkerAtStart's true-returning path from
// inside the thought-branch's own mid-buffer marker scan (as opposed
// to the top-of-loop fast path every other "recognised name" test in
// this file exercises, where the marker always starts at position 0).
func TestOpenAI_ThinkingExtractor_Good_ChannelReopenWithRecognisedName(t *testing.T) {
	extractor := NewThinkingExtractor()

	visible, thought := extractor.Process(inference.Token{Text: "<|channel>thought stuff<|channel>assistant answer"})

	if visible != " answer" {
		t.Fatalf("visible = %q, want %q", visible, " answer")
	}
	if thought != " stuff" {
		t.Fatalf("thought = %q, want %q", thought, " stuff")
	}
}

// TestOpenAI_ThinkingExtractor_Good_ChannelMarkerWithLeadingSpace
// covers consumeMarkerAtStart's leading-whitespace skip between
// "<|channel>" and the channel name — every other channel-marker test
// in this file places the name immediately after the marker with no
// separating space.
func TestOpenAI_ThinkingExtractor_Good_ChannelMarkerWithLeadingSpace(t *testing.T) {
	extractor := NewThinkingExtractor()

	visible, thought := extractor.Process(inference.Token{Text: "<|channel> thought plan"})

	if visible != "" {
		t.Fatalf("visible = %q, want empty", visible)
	}
	if thought != " plan" {
		t.Fatalf("thought = %q, want %q", thought, " plan")
	}
}

// TestThinking_WriteContent_Direct and TestThinking_WriteThought_Direct
// drive the empty-text no-op guard directly — drain's hot loop never
// calls these with an empty emit (every call site checks emit != ""
// first), so the guard is only reachable via a direct call.
func TestThinking_WriteContent_Direct(t *testing.T) {
	e := NewThinkingExtractor()
	var b core.Builder
	writeContent(e, &b, "")
	if b.Len() != 0 || e.content != "" {
		t.Fatalf("writeContent(empty) wrote something: builder=%q content=%q", b.String(), e.content)
	}
	writeContent(e, &b, "hi")
	if b.String() != "hi" || e.content != "hi" {
		t.Fatalf("writeContent(hi) = builder:%q content:%q, want hi/hi", b.String(), e.content)
	}
}

func TestThinking_WriteThought_Direct(t *testing.T) {
	e := NewThinkingExtractor()
	var b core.Builder
	writeThought(e, &b, "")
	if b.Len() != 0 || e.thinking != "" {
		t.Fatalf("writeThought(empty) wrote something: builder=%q thinking=%q", b.String(), e.thinking)
	}
	writeThought(e, &b, "hmm")
	if b.String() != "hmm" || e.thinking != "hmm" {
		t.Fatalf("writeThought(hmm) = builder:%q thinking:%q, want hmm/hmm", b.String(), e.thinking)
	}
}

// TestThinking_PairedEndFor_Direct drives pairedEndFor's own branches:
// every known marker start resolves to its paired end, and an unknown
// start (never produced by earliestReasoningStart in practice, but
// pairedEndFor is defensive against it) falls back to "".
func TestThinking_PairedEndFor_Direct(t *testing.T) {
	if got := pairedEndFor("<think>"); got != "</think>" {
		t.Fatalf("pairedEndFor(<think>) = %q, want </think>", got)
	}
	if got := pairedEndFor("<reasoning>"); got != "</reasoning>" {
		t.Fatalf("pairedEndFor(<reasoning>) = %q, want </reasoning>", got)
	}
	if got := pairedEndFor("<unknown-marker>"); got != "" {
		t.Fatalf("pairedEndFor(unknown) = %q, want empty", got)
	}
}

// TestThinking_Utf8Rune_Direct drives the rune-decode helper across a
// multi-byte and a single-byte input.
func TestThinking_Utf8Rune_Direct(t *testing.T) {
	if r, size := utf8Rune("é"); r != 'é' || size != 2 {
		t.Fatalf("utf8Rune(é) = %q, %d, want é, 2", r, size)
	}
	if r, size := utf8Rune("a"); r != 'a' || size != 1 {
		t.Fatalf("utf8Rune(a) = %q, %d, want a, 1", r, size)
	}
}

// TestThinkingExtractorSwallowsTurnTerminator pins the assistant-lane
// terminator contract: gemma4 MLX snapshots ship <end_of_turn> as a PLAIN
// vocab token the decode layer cannot hide, so its literal text reaches the
// extractor — it must never land in content.
func TestThinkingExtractorSwallowsTurnTerminator(t *testing.T) {
	e := NewThinkingExtractor()
	content, thought := e.Process(inference.Token{Text: "the answer is 68.\n<end_of_turn>"})
	fc, ft := e.Flush()
	content += fc
	thought += ft
	if want := "the answer is 68.\n"; content != want {
		t.Fatalf("content = %q, want %q", content, want)
	}
	if thought != "" {
		t.Fatalf("thought = %q, want empty", thought)
	}
}

// TestThinkingExtractorTurnTerminatorSplitAcrossTokens pins the streaming
// shape: a terminator arriving split across Process calls is held back (it is
// in the marker-starts holdback set) and swallowed once complete.
func TestThinkingExtractorTurnTerminatorSplitAcrossTokens(t *testing.T) {
	e := NewThinkingExtractor()
	var content string
	for _, piece := range []string{"done", "<end_of_", "turn>"} {
		c, _ := e.Process(inference.Token{Text: piece})
		content += c
	}
	fc, _ := e.Flush()
	content += fc
	if content != "done" {
		t.Fatalf("content = %q, want %q", content, "done")
	}
}

// TestThinkingExtractorTerminatorAfterChannelClose pins the interplay with the
// gemma4 thought channel: thought is captured, the close switches back to the
// assistant lane, and the trailing terminator still never reaches content.
func TestThinkingExtractorTerminatorAfterChannelClose(t *testing.T) {
	e := NewThinkingExtractor()
	content, thought := e.Process(inference.Token{Text: "<|channel>thought\nplan<channel|>final<end_of_turn>"})
	fc, ft := e.Flush()
	content += fc
	thought += ft
	if content != "final" {
		t.Fatalf("content = %q, want %q", content, "final")
	}
	if thought != "\nplan" {
		// The extractor's channel contract keeps the newline after the channel
		// name in the thought text; only the content lane matters here.
		t.Fatalf("thought = %q, want %q", thought, "\nplan")
	}
}
