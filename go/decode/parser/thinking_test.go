// SPDX-Licence-Identifier: EUPL-1.2

package parser

import (
	"testing"
)

func TestThinking_FilterGemmaHide_Good(t *testing.T) {
	got := Filter(
		"<start_of_turn>thinking\nplan<end_of_turn>final",
		Config{Mode: Hide},
		Hint{Architecture: "gemma4_text"},
	)
	if got.Text != "final" {
		t.Fatalf("Text = %q, want final", got.Text)
	}
	if got.Reasoning != "plan" {
		t.Fatalf("Reasoning = %q, want plan", got.Reasoning)
	}
}

func TestThinking_Filter_Ugly(t *testing.T) {
	raw := "<think>secret</think>visible"
	got := Filter(raw, Config{Mode: Show}, Hint{Architecture: "qwen3"})
	if got.Text != raw {
		t.Fatalf("Text = %q, want raw passthrough", got.Text)
	}
	if got.Reasoning != "" {
		t.Fatalf("Reasoning = %q, want empty for passthrough mode", got.Reasoning)
	}
}

func TestThinking_Flush_Ugly(t *testing.T) {
	var captured []Chunk
	processor := NewProcessor(Config{
		Mode: Capture,
		Capture: func(chunk Chunk) {
			captured = append(captured, chunk)
		},
	}, Hint{Architecture: "qwen3"})

	if text := processor.Process("visible <thi"); text != "visible " {
		t.Fatalf("partial start output = %q, want visible prefix", text)
	}
	if text := processor.Process("nk>unfinished"); text != "" {
		t.Fatalf("open reasoning output = %q, want hidden reasoning", text)
	}
	if text := processor.Flush(); text != "" {
		t.Fatalf("flush output = %q, want empty while closing open reasoning", text)
	}
	if processor.Reasoning() != "unfinished" {
		t.Fatalf("reasoning = %q, want unfinished", processor.Reasoning())
	}
	if len(captured) != 1 || captured[0].Text != "unfinished" {
		t.Fatalf("captured = %+v, want unfinished block", captured)
	}

	processor = NewProcessor(Config{Mode: Hide}, Hint{Architecture: "qwen3"})
	if text := processor.Process("<thi"); text != "" {
		t.Fatalf("partial marker output = %q, want held text until flush", text)
	}
	if text := processor.Flush(); text != "<thi" {
		t.Fatalf("partial marker flush = %q, want literal partial marker", text)
	}
}

func TestThinking_NormaliseMode_Ugly(t *testing.T) {
	if mode := NormaliseMode("unknown"); mode != Show {
		t.Fatalf("NormaliseMode(unknown) = %q, want show", mode)
	}
	if mode := NormaliseMode(""); mode != Show {
		t.Fatalf("NormaliseMode(empty) = %q, want show", mode)
	}
	if mode := NormaliseMode(Capture); mode != Capture {
		t.Fatalf("NormaliseMode(capture) = %q, want capture", mode)
	}
}

// TestThinking_GemmaTurnTerminatorSwallowed_Good pins the terminator contract:
// gemma4 MLX snapshots ship <end_of_turn> as a PLAIN vocab token the tokenizer
// cannot hide, so the Processor swallows the bare terminator from visible
// output — replies must not end with a literal "<end_of_turn>".
func TestThinking_GemmaTurnTerminatorSwallowed_Good(t *testing.T) {
	got := Filter(
		"the answer is 68.\n<end_of_turn>",
		Config{Mode: Capture},
		Hint{Architecture: "gemma4"},
	)
	if want := "the answer is 68.\n"; got.Text != want {
		t.Fatalf("Text = %q, want %q", got.Text, want)
	}
}

// TestThinking_GemmaTurnTerminatorSplitStream_Good pins the streaming shape:
// the terminator arriving split across Process calls is held back (the
// holdback set includes terminators) and swallowed once complete.
func TestThinking_GemmaTurnTerminatorSplitStream_Good(t *testing.T) {
	p := NewProcessor(Config{Mode: Capture}, Hint{Architecture: "gemma4"})
	out := p.Process("done<end_of_")
	out += p.Process("turn>")
	out += p.Flush()
	if out != "done" {
		t.Fatalf("streamed = %q, want %q", out, "done")
	}
}

// TestThinking_GemmaSpanEndStillWorks_Bad guards the span interplay: inside a
// thinking span <end_of_turn> is the SPAN end and must close the span (not be
// treated as a bare terminator), leaving the visible tail intact.
func TestThinking_GemmaSpanEndStillWorks_Bad(t *testing.T) {
	got := Filter(
		"<start_of_turn>thinking\nplan<end_of_turn>final",
		Config{Mode: Capture},
		Hint{Architecture: "gemma4"},
	)
	if got.Text != "final" {
		t.Fatalf("Text = %q, want final", got.Text)
	}
	if got.Reasoning != "plan" {
		t.Fatalf("Reasoning = %q, want plan", got.Reasoning)
	}
}

// TestThinking_TerminatorOtherFamiliesUntouched_Ugly pins the scope: families
// without a turn terminator (qwen etc.) pass the literal text through — the
// swallow is gemma-specific, not a blanket rewrite.
func TestThinking_TerminatorOtherFamiliesUntouched_Ugly(t *testing.T) {
	got := Filter(
		"tail<end_of_turn>",
		Config{Mode: Capture},
		Hint{Architecture: "qwen3"},
	)
	if want := "tail<end_of_turn>"; got.Text != want {
		t.Fatalf("Text = %q, want %q", got.Text, want)
	}
}
