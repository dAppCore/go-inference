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
