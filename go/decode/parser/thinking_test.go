// SPDX-Licence-Identifier: EUPL-1.2

package parser

import (
	"testing"

	core "dappco.re/go"
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

func TestThinking_Processor_Flush_Ugly(t *testing.T) {
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

// TestThinking_Filter_Good pins the terminator contract: gemma4 MLX
// snapshots ship <end_of_turn> as a PLAIN vocab token the tokenizer cannot
// hide, so Filter swallows the bare terminator from visible output —
// replies must not end with a literal "<end_of_turn>".
func TestThinking_Filter_Good(t *testing.T) {
	got := Filter(
		"the answer is 68.\n<end_of_turn>",
		Config{Mode: Capture},
		Hint{Architecture: "gemma4"},
	)
	if want := "the answer is 68.\n"; got.Text != want {
		t.Fatalf("Text = %q, want %q", got.Text, want)
	}
}

// TestThinking_Processor_Process_Ugly pins the streaming shape: the
// terminator arriving split across Process calls is held back (the holdback
// set includes terminators) and swallowed once complete.
func TestThinking_Processor_Process_Ugly(t *testing.T) {
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

// TestThinking_Filter_Bad pins the scope: families without a turn terminator
// (qwen etc.) pass the literal text through — the swallow is gemma-specific,
// not a blanket rewrite.
func TestThinking_Filter_Bad(t *testing.T) {
	got := Filter(
		"tail<end_of_turn>",
		Config{Mode: Capture},
		Hint{Architecture: "qwen3"},
	)
	if want := "tail<end_of_turn>"; got.Text != want {
		t.Fatalf("Text = %q, want %q", got.Text, want)
	}
}

// TestThinking_NewProcessor_Good pins normal construction: the mode
// normalises and the returned Processor is immediately usable.
func TestThinking_NewProcessor_Good(t *testing.T) {
	p := NewProcessor(Config{Mode: Hide}, Hint{Architecture: "qwen3"})
	if p == nil {
		t.Fatal("NewProcessor() = nil")
	}
	if text := p.Process("<think>plan</think>answer"); text != "answer" {
		t.Fatalf("Process() = %q, want answer", text)
	}
}

// TestThinking_NewProcessor_Bad pins the mode-normalisation guard: an
// unrecognised Config.Mode falls back to Show rather than a broken state.
func TestThinking_NewProcessor_Bad(t *testing.T) {
	p := NewProcessor(Config{Mode: "not-a-mode"}, Hint{Architecture: "qwen3"})
	if text := p.Process("<think>plan</think>answer"); text != "<think>plan</think>answer" {
		t.Fatalf("Process() = %q, want raw passthrough under the Show fallback", text)
	}
}

// TestThinking_NewProcessor_Ugly pins the unknown-architecture edge: a hint
// with no matching family still returns a working Processor over the
// generic marker set, never nil or a panic. Generic excludes the qwen-only
// <think> spelling (genericMarkers), so the case uses <reasoning> instead.
func TestThinking_NewProcessor_Ugly(t *testing.T) {
	p := NewProcessor(Config{Mode: Hide}, Hint{Architecture: "not-a-real-arch"})
	if text := p.Process("<reasoning>plan</reasoning>answer"); text != "answer" {
		t.Fatalf("Process() with unknown architecture = %q, want the generic marker set to still apply", text)
	}
}

// TestThinking_NormaliseMode_Good pins the explicit-mode passthrough cases.
func TestThinking_NormaliseMode_Good(t *testing.T) {
	if mode := NormaliseMode(Hide); mode != Hide {
		t.Fatalf("NormaliseMode(Hide) = %q, want Hide", mode)
	}
	if mode := NormaliseMode(Show); mode != Show {
		t.Fatalf("NormaliseMode(Show) = %q, want Show", mode)
	}
}

// TestThinking_NormaliseMode_Bad pins the default-empty case: an unset
// Config.Mode normalises to Show, matching DefaultGenerateConfig's zero
// value rather than an error.
func TestThinking_NormaliseMode_Bad(t *testing.T) {
	var mode Mode
	if got := NormaliseMode(mode); got != Show {
		t.Fatalf("NormaliseMode(zero) = %q, want Show", got)
	}
}

// TestThinking_Processor_Process_Good pins the plain no-marker path: text
// with no reasoning span passes straight through in Show mode.
func TestThinking_Processor_Process_Good(t *testing.T) {
	p := NewProcessor(Config{Mode: Show}, Hint{Architecture: "qwen3"})
	if text := p.Process("plain answer, no markers"); text != "plain answer, no markers" {
		t.Fatalf("Process() = %q, want unchanged passthrough", text)
	}
}

// TestThinking_Processor_Process_Bad pins the Hide-mode span: reasoning
// text between markers never reaches Process's return value.
func TestThinking_Processor_Process_Bad(t *testing.T) {
	p := NewProcessor(Config{Mode: Hide}, Hint{Architecture: "qwen3"})
	if text := p.Process("<think>secret plan</think>answer"); text != "answer" {
		t.Fatalf("Process() = %q, want reasoning span hidden", text)
	}
}

// TestThinking_Processor_Flush_Good pins the plain-flush path: nothing
// pending, nothing open — Flush returns empty.
func TestThinking_Processor_Flush_Good(t *testing.T) {
	p := NewProcessor(Config{Mode: Hide}, Hint{Architecture: "qwen3"})
	p.Process("<think>plan</think>answer")
	if text := p.Flush(); text != "" {
		t.Fatalf("Flush() = %q, want empty with nothing pending", text)
	}
}

// TestThinking_Processor_Flush_Bad pins the mid-span close: Flush finalises
// an open reasoning block (final=true) rather than holding it back.
func TestThinking_Processor_Flush_Bad(t *testing.T) {
	p := NewProcessor(Config{Mode: Capture}, Hint{Architecture: "qwen3"})
	p.Process("<think>unterminated plan")
	if text := p.Flush(); text != "" {
		t.Fatalf("Flush() = %q, want no visible text for an unterminated reasoning span", text)
	}
	if p.Reasoning() != "unterminated plan" {
		t.Fatalf("Reasoning() = %q, want the span content captured at flush", p.Reasoning())
	}
}

// TestThinking_Processor_Reasoning_Good pins the accumulation contract:
// every hidden reasoning span joins into one string.
func TestThinking_Processor_Reasoning_Good(t *testing.T) {
	p := NewProcessor(Config{Mode: Hide}, Hint{Architecture: "qwen3"})
	p.Process("<think>first</think>mid<think>second</think>tail")
	if got := p.Reasoning(); got != "firstsecond" {
		t.Fatalf("Reasoning() = %q, want firstsecond", got)
	}
}

// TestThinking_Processor_Reasoning_Bad pins the empty case: a Processor
// that never saw a reasoning span reports an empty Reasoning(), not nil
// dereferenced or a placeholder.
func TestThinking_Processor_Reasoning_Bad(t *testing.T) {
	p := NewProcessor(Config{Mode: Hide}, Hint{Architecture: "qwen3"})
	p.Process("no reasoning here")
	if got := p.Reasoning(); got != "" {
		t.Fatalf("Reasoning() = %q, want empty", got)
	}
}

// TestThinking_Processor_Reasoning_Ugly pins the holdback edge: a trailing
// byte that could be the start of the span-end marker is held back from
// Reasoning() until a later call confirms it either way — streaming must
// never commit an ambiguous partial match.
func TestThinking_Processor_Reasoning_Ugly(t *testing.T) {
	p := NewProcessor(Config{Mode: Hide}, Hint{Architecture: "qwen3"})
	p.Process("<think>plan<")
	if got := p.Reasoning(); got != "plan" {
		t.Fatalf("Reasoning() with an ambiguous trailing byte held back = %q, want plan", got)
	}
	p.Process("/think>tail")
	if got := p.Reasoning(); got != "plan" {
		t.Fatalf("Reasoning() once the marker resolves = %q, want plan", got)
	}
}

// TestThinking_Processor_Chunks_Good pins the per-span chunk record: each
// closed reasoning span emits one Chunk carrying its channel and model.
func TestThinking_Processor_Chunks_Good(t *testing.T) {
	p := NewProcessor(Config{Mode: Hide}, Hint{Architecture: "qwen3"})
	p.Process("<think>plan</think>answer")
	chunks := p.Chunks()
	if len(chunks) != 1 || chunks[0].Text != "plan" || chunks[0].Channel != "thinking" {
		t.Fatalf("Chunks() = %+v, want one thinking chunk with Text=plan", chunks)
	}
}

// TestThinking_Processor_Chunks_Bad pins the empty case: no reasoning span
// seen means Chunks() returns nil, not an empty-but-allocated slice.
func TestThinking_Processor_Chunks_Bad(t *testing.T) {
	p := NewProcessor(Config{Mode: Hide}, Hint{Architecture: "qwen3"})
	p.Process("plain text")
	if chunks := p.Chunks(); chunks != nil {
		t.Fatalf("Chunks() = %+v, want nil", chunks)
	}
}

// TestThinking_Processor_Chunks_Ugly pins the defensive-copy contract: the
// slice Chunks() returns is a snapshot — mutating it must not corrupt the
// Processor's internal state seen by a later call.
func TestThinking_Processor_Chunks_Ugly(t *testing.T) {
	p := NewProcessor(Config{Mode: Hide}, Hint{Architecture: "qwen3"})
	p.Process("<think>plan</think>answer")
	chunks := p.Chunks()
	chunks[0].Text = "mutated"
	if got := p.Chunks(); got[0].Text != "plan" {
		t.Fatalf("Chunks() after external mutation = %+v, want the original plan text intact", got)
	}
}

// TestThinking_FindStartTerminator_AnchorMatchesNaive_Good pins the streaming
// Processor's lead-byte anchor scans (findStart, findTerminator) byte-for-byte
// to naive min-index references, across the builtin families and adversarial
// fragment streams. These fire per token, so a divergence would corrupt every
// streamed reply — this is the byte-identity guard for the O(markers×pending)→
// O(pending) rewrite.
func TestThinking_FindStartTerminator_AnchorMatchesNaive_Good(t *testing.T) {
	naiveStart := func(p *Processor, text string) (int, thinkingMarker, bool) {
		best := -1
		var m thinkingMarker
		for _, c := range p.markers {
			idx := indexString(text, c.start)
			if idx < 0 {
				continue
			}
			if best < 0 || idx < best || idx == best && len(c.start) > len(m.start) {
				best = idx
				m = c
			}
		}
		return best, m, best >= 0
	}
	naiveTerm := func(p *Processor, text string) (int, int) {
		best, size := -1, 0
		for _, term := range p.terminators {
			idx := indexString(text, term)
			if idx < 0 {
				continue
			}
			if best < 0 || idx < best {
				best, size = idx, len(term)
			}
		}
		return best, size
	}
	hints := []Hint{{Architecture: "qwen3"}, {Architecture: "gemma4_text"}, {Architecture: "gpt-oss"}, {Architecture: ""}}
	frags := []string{
		"word ", "the ", "<", ">", "|", "\n", "<think>", "</think>", "<|channel>",
		"analysis\n", "<start_of_turn>", "thinking\n", "thought\n", "<end_of_turn>",
		"reasoning\n", "final\n", "x", "  ", "<start_of_turn>thinking\n",
	}
	st := uint64(0x2545f4914f6cdd1d)
	rnd := func() uint64 { st = st*6364136223846793005 + 1442695040888963407; return st >> 1 }
	for _, hint := range hints {
		p := NewProcessor(Config{Mode: Hide}, hint)
		for iter := 0; iter < 50000; iter++ {
			b := core.NewBuilder()
			for k := int(rnd()%9) + 1; k > 0; k-- {
				b.WriteString(frags[rnd()%uint64(len(frags))])
			}
			text := b.String()
			si, sm, sok := p.findStart(text)
			ni, nm, nok := naiveStart(p, text)
			if si != ni || sok != nok || sm.start != nm.start {
				t.Fatalf("findStart %q anchor=(%d,%q,%v) naive=(%d,%q,%v)", text, si, sm.start, sok, ni, nm.start, nok)
			}
			ti, tl := p.findTerminator(text)
			nti, ntl := naiveTerm(p, text)
			if ti != nti || tl != ntl {
				t.Fatalf("findTerminator %q anchor=(%d,%d) naive=(%d,%d)", text, ti, tl, nti, ntl)
			}
		}
	}
}

// TestProcessorDrainFinalConsumesPending pins drain(final=true)'s postcondition:
// pending is fully consumed on EVERY path (the partial-marker holdback only arms
// mid-stream), which is what lets Flush skip residue splicing. If a drain change
// ever retains bytes at final, this fails before Flush silently drops them.
func TestProcessorDrainFinalConsumesPending(t *testing.T) {
	cases := []struct {
		name  string
		feeds []string
	}{
		{"reasoning_tail_no_end_marker", []string{"<think>half finished thought"}},
		{"holdback_shaped_tail", []string{"visible <thi"}},
		{"marker_then_tail", []string{"a<think>x</think>b<thi"}},
		{"bare_terminator_tail", []string{"answer<turn|"}},
		{"empty", nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := NewProcessor(Config{Mode: Hide}, Hint{Architecture: "gemma4"})
			for _, f := range tc.feeds {
				p.Process(f)
			}
			p.Flush()
			if p.pending != "" {
				t.Fatalf("drain(true) left pending %q — Flush residue splicing was removed on this postcondition", p.pending)
			}
			if p.inReasoning {
				t.Fatalf("Flush left inReasoning set")
			}
		})
	}
}
