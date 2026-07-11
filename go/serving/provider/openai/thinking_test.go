// SPDX-Licence-Identifier: EUPL-1.2

// Tests for reasoning-channel extraction.
package openai

import (
	"testing"
	"time"

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

// The four tests below pin the nil-receiver guard on every exported
// method individually — a caller holding a *ThinkingExtractor obtained
// from a failed constructor path must get safe zero values, never a
// panic. (Previously one combined test; split so each method has its
// own named Bad case per the package's per-symbol test convention.)

func TestThinking_ThinkingExtractor_Process_Bad(t *testing.T) {
	var e *ThinkingExtractor
	if content, thought := e.Process(inference.Token{Text: "x"}); content != "" || thought != "" {
		t.Fatalf("nil.Process() = %q, %q, want empty/empty", content, thought)
	}
}

func TestThinking_ThinkingExtractor_Flush_Bad(t *testing.T) {
	var e *ThinkingExtractor
	if content, thought := e.Flush(); content != "" || thought != "" {
		t.Fatalf("nil.Flush() = %q, %q, want empty/empty", content, thought)
	}
}

func TestThinking_ThinkingExtractor_Content_Bad(t *testing.T) {
	var e *ThinkingExtractor
	if got := e.Content(); got != "" {
		t.Fatalf("nil.Content() = %q, want empty", got)
	}
}

func TestThinking_ThinkingExtractor_Thinking_Bad(t *testing.T) {
	var e *ThinkingExtractor
	if got := e.Thinking(); got != "" {
		t.Fatalf("nil.Thinking() = %q, want empty", got)
	}
}

// TestThinking_NewThinkingExtractor_Good covers the plain construction
// path — a fresh extractor starts in the assistant (non-reasoning)
// channel, so ordinary content passes straight through.
func TestThinking_NewThinkingExtractor_Good(t *testing.T) {
	extractor := NewThinkingExtractor()
	if extractor == nil {
		t.Fatal("NewThinkingExtractor() = nil")
	}
	content, thought := extractor.Process(inference.Token{Text: "hello"})
	if content != "hello" || thought != "" {
		t.Fatalf("Process() = %q/%q, want plain content in the initial assistant channel", content, thought)
	}
}

// TestThinking_NewThinkingExtractor_Bad covers isolation — two
// independently constructed extractors must not share state.
func TestThinking_NewThinkingExtractor_Bad(t *testing.T) {
	first := NewThinkingExtractor()
	second := NewThinkingExtractor()
	first.Process(inference.Token{Text: "first"})

	if content, _ := second.Process(inference.Token{Text: "second"}); content != "second" {
		t.Fatalf("second extractor content = %q, want isolated from the first", content)
	}
	if first.Content() == second.Content() {
		t.Fatalf("extractors share content: %q", first.Content())
	}
}

// TestThinking_NewThinkingExtractor_Ugly covers the zero-activity
// edge — flushing a fresh extractor that never received a Process
// call returns empty/empty rather than panicking on unset fields.
func TestThinking_NewThinkingExtractor_Ugly(t *testing.T) {
	extractor := NewThinkingExtractor()

	content, thought := extractor.Flush()
	if content != "" || thought != "" {
		t.Fatalf("Flush() on a fresh extractor = %q/%q, want empty/empty", content, thought)
	}
}

// TestThinking_ThinkingExtractor_Process_Good covers the plain pass-
// through path — content with no reasoning markers streams straight
// to the content channel.
func TestThinking_ThinkingExtractor_Process_Good(t *testing.T) {
	extractor := NewThinkingExtractor()

	content, thought := extractor.Process(inference.Token{Text: "plain answer"})
	if content != "plain answer" || thought != "" {
		t.Fatalf("Process() = %q/%q, want plain content only", content, thought)
	}
}

// TestThinking_ThinkingExtractor_Process_Ugly covers an empty-text
// token — a no-op that must not disturb already-accumulated state.
func TestThinking_ThinkingExtractor_Process_Ugly(t *testing.T) {
	extractor := NewThinkingExtractor()
	extractor.Process(inference.Token{Text: "so far"})

	content, thought := extractor.Process(inference.Token{Text: ""})
	if content != "" || thought != "" {
		t.Fatalf("Process(empty text) = %q/%q, want no new output", content, thought)
	}
	if extractor.Content() != "so far" {
		t.Fatalf("Content() = %q, want prior state undisturbed", extractor.Content())
	}
}

// TestThinking_ThinkingExtractor_Flush_Good covers the held-back-
// partial-marker tail — a suffix that could still become a reasoning
// marker, but with no more input coming, Flush must surface it as
// literal content.
func TestThinking_ThinkingExtractor_Flush_Good(t *testing.T) {
	extractor := NewThinkingExtractor()
	extractor.Process(inference.Token{Text: "hello <thi"})

	content, thought := extractor.Flush()
	if content != "<thi" || thought != "" {
		t.Fatalf("Flush() = %q/%q, want the held-back partial marker as literal content", content, thought)
	}
}

// TestThinking_ThinkingExtractor_Flush_Ugly covers flushing a held-back
// partial close-marker suffix while inPaired — "</thi" could still
// become "</think>", so Process holds it rather than emitting it, and
// Flush must surface it as thought text (not content) once there is no
// more input coming.
func TestThinking_ThinkingExtractor_Flush_Ugly(t *testing.T) {
	extractor := NewThinkingExtractor()
	extractor.Process(inference.Token{Text: "<think>hidden</thi"})

	content, thought := extractor.Flush()
	if content != "" || thought != "</thi" {
		t.Fatalf("Flush() = %q/%q, want the held-back partial marker to surface as thought text", content, thought)
	}
}

// TestThinking_ThinkingExtractor_Content_Good covers plain accumulation
// across multiple Process calls.
func TestThinking_ThinkingExtractor_Content_Good(t *testing.T) {
	extractor := NewThinkingExtractor()
	extractor.Process(inference.Token{Text: "hello "})
	extractor.Process(inference.Token{Text: "world"})

	if got := extractor.Content(); got != "hello world" {
		t.Fatalf("Content() = %q, want %q", got, "hello world")
	}
}

// TestThinking_ThinkingExtractor_Content_Ugly covers exclusion — a
// thought span embedded in the stream must not appear in Content(),
// only in Thinking().
func TestThinking_ThinkingExtractor_Content_Ugly(t *testing.T) {
	extractor := NewThinkingExtractor()
	extractor.Process(inference.Token{Text: "before <think>hidden</think> after"})

	if got := extractor.Content(); got != "before  after" {
		t.Fatalf("Content() = %q, want the thought span excluded", got)
	}
}

// TestThinking_ThinkingExtractor_Thinking_Good covers plain
// accumulation of a single thought span.
func TestThinking_ThinkingExtractor_Thinking_Good(t *testing.T) {
	extractor := NewThinkingExtractor()
	extractor.Process(inference.Token{Text: "<think>plan</think>answer"})

	if got := extractor.Thinking(); got != "plan" {
		t.Fatalf("Thinking() = %q, want %q", got, "plan")
	}
}

// TestThinking_ThinkingExtractor_Thinking_Ugly covers exclusion — the
// visible content around a thought span must not appear in Thinking().
func TestThinking_ThinkingExtractor_Thinking_Ugly(t *testing.T) {
	extractor := NewThinkingExtractor()
	extractor.Process(inference.Token{Text: "before <think>hidden</think> after"})

	if got := extractor.Thinking(); got != "hidden" {
		t.Fatalf("Thinking() = %q, want only the thought span", got)
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
	writeContent(e, "")
	if e.content.String() != "" {
		t.Fatalf("writeContent(empty) wrote something: content=%q", e.content.String())
	}
	writeContent(e, "hi")
	if e.content.String() != "hi" {
		t.Fatalf("writeContent(hi) = content:%q, want hi", e.content.String())
	}
}

func TestThinking_WriteThought_Direct(t *testing.T) {
	e := NewThinkingExtractor()
	writeThought(e, "")
	if e.thinking.String() != "" {
		t.Fatalf("writeThought(empty) wrote something: thinking=%q", e.thinking.String())
	}
	writeThought(e, "hmm")
	if e.thinking.String() != "hmm" {
		t.Fatalf("writeThought(hmm) = thinking:%q, want hmm", e.thinking.String())
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
