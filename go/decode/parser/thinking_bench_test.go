// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the streaming thinking-mode Processor — Filter,
// NewProcessor, Process, Flush, Reasoning, Chunks, NormaliseMode,
// markersForHint, longestSuffixPrefix. Per AX-11 — Processor.Process is
// the PER-TOKEN hot loop fired on every streamed chunk during
// generation (one call per generated token, possibly thousands per
// response). longestSuffixPrefix is the partial-marker held-tail check
// also paid per token. NewProcessor + markersForHint are the
// per-stream build cost paid once per response but reach into the
// registry. Filter is the batch (non-streaming) entry point.
//
// Run:    go test -bench='Benchmark_Thinking' -benchmem -run='^$' ./go/parser
//
// Stream sizes:
//   - 32-token  ≈ very short response
//   - 256-token ≈ typical chat response
//   - 2048-token ≈ long-form streamed response

package parser

import (
	"testing"

	core "dappco.re/go"
)

// Sinks defeat compiler DCE.
var (
	thinkingBenchResult    Result
	thinkingBenchProcessor *Processor
	thinkingBenchText      string
	thinkingBenchMode      Mode
	thinkingBenchMarkers   []thinkingMarker
	thinkingBenchKeep      int
	thinkingBenchChunks    []Chunk
	thinkingBenchReasoning string
)

// thinkingBenchWords builds a synthetic prose stream of `tokens` words.
func thinkingBenchWords(tokens int) string {
	out := core.NewBuilder()
	for i := 0; i < tokens; i++ {
		out.WriteString("word ")
	}
	return out.String()
}

// thinkingBenchTokens chunks a stream into per-token deliveries — the
// actual per-token Process() input shape during streaming. We split
// on whitespace and reassemble each "word " into a delivery to mirror
// the inference loop's flush rhythm.
func thinkingBenchTokens(text string) []string {
	out := make([]string, 0, 256)
	start := 0
	for i := 0; i < len(text); i++ {
		if text[i] == ' ' {
			out = append(out, text[start:i+1])
			start = i + 1
		}
	}
	if start < len(text) {
		out = append(out, text[start:])
	}
	return out
}

// thinkingBenchStream wraps a span of words inside the marker pair,
// span covering `spanFraction` of the total.
func thinkingBenchStream(tokens int, spanFraction float64, startMarker, endMarker string) string {
	span := int(float64(tokens) * spanFraction)
	if span < 1 {
		span = 1
	}
	if span > tokens {
		span = tokens
	}
	pre := (tokens - span) / 2
	post := tokens - span - pre
	out := core.NewBuilder()
	out.WriteString(thinkingBenchWords(pre))
	out.WriteString(startMarker)
	out.WriteString(thinkingBenchWords(span))
	out.WriteString(endMarker)
	out.WriteString(thinkingBenchWords(post))
	return out.String()
}

// --- Filter (batch entry point) ---

func Benchmark_Thinking_Filter_Show_Qwen(b *testing.B) {
	text := thinkingBenchStream(256, 0.50, "<think>", "</think>")
	hint := Hint{Architecture: "qwen3"}
	cfg := Config{Mode: Show}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		thinkingBenchResult = Filter(text, cfg, hint)
	}
}

func Benchmark_Thinking_Filter_Hide_Qwen(b *testing.B) {
	text := thinkingBenchStream(256, 0.50, "<think>", "</think>")
	hint := Hint{Architecture: "qwen3"}
	cfg := Config{Mode: Hide}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		thinkingBenchResult = Filter(text, cfg, hint)
	}
}

func Benchmark_Thinking_Filter_Capture_Qwen(b *testing.B) {
	text := thinkingBenchStream(256, 0.50, "<think>", "</think>")
	hint := Hint{Architecture: "qwen3"}
	cfg := Config{Mode: Capture, Capture: func(Chunk) {}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		thinkingBenchResult = Filter(text, cfg, hint)
	}
}

func Benchmark_Thinking_Filter_Hide_Gemma(b *testing.B) {
	text := thinkingBenchStream(256, 0.50, "<start_of_turn>thinking\n", "<end_of_turn>")
	hint := Hint{Architecture: "gemma4_text"}
	cfg := Config{Mode: Hide}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		thinkingBenchResult = Filter(text, cfg, hint)
	}
}

// --- NewProcessor (per-stream build cost) ---

func Benchmark_Thinking_NewProcessor_Qwen(b *testing.B) {
	hint := Hint{Architecture: "qwen3"}
	cfg := Config{Mode: Hide}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		thinkingBenchProcessor = NewProcessor(cfg, hint)
	}
}

func Benchmark_Thinking_NewProcessor_Gemma(b *testing.B) {
	hint := Hint{Architecture: "gemma4_text"}
	cfg := Config{Mode: Hide}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		thinkingBenchProcessor = NewProcessor(cfg, hint)
	}
}

// --- markersForHint (per-NewProcessor inner cost) ---

func Benchmark_Thinking_MarkersForHint_Qwen(b *testing.B) {
	hint := Hint{Architecture: "qwen3"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		thinkingBenchMarkers = markersForHint(hint)
	}
}

func Benchmark_Thinking_MarkersForHint_Gemma(b *testing.B) {
	hint := Hint{Architecture: "gemma4_text"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		thinkingBenchMarkers = markersForHint(hint)
	}
}

func Benchmark_Thinking_MarkersForHint_GPTOSS(b *testing.B) {
	hint := Hint{Architecture: "gpt-oss"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		thinkingBenchMarkers = markersForHint(hint)
	}
}

// --- NormaliseMode (cheap branch, called per NewProcessor) ---

func Benchmark_Thinking_NormaliseMode_Empty(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		thinkingBenchMode = NormaliseMode("")
	}
}

func Benchmark_Thinking_NormaliseMode_Hide(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		thinkingBenchMode = NormaliseMode(Hide)
	}
}

func Benchmark_Thinking_NormaliseMode_Capture(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		thinkingBenchMode = NormaliseMode(Capture)
	}
}

func Benchmark_Thinking_NormaliseMode_Unknown(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		thinkingBenchMode = NormaliseMode("unknown")
	}
}

// --- Process: PER-TOKEN HOT LOOP ---
// Show-mode short-circuits at the function head (the cheap path).
// Hide/Capture-mode pays the full drain() cost per call.

func Benchmark_Thinking_Process_Show_Qwen_PerToken(b *testing.B) {
	pieces := thinkingBenchTokens(thinkingBenchStream(256, 0.50, "<think>", "</think>"))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		processor := NewProcessor(Config{Mode: Show}, Hint{Architecture: "qwen3"})
		for _, piece := range pieces {
			thinkingBenchText = processor.Process(piece)
		}
		thinkingBenchText = processor.Flush()
	}
}

// Per-token streaming over various stream sizes.
var thinkingBenchStreamSizes = []int{32, 256, 2048}

func Benchmark_Thinking_Process_Hide_Qwen_PerToken(b *testing.B) {
	for _, size := range thinkingBenchStreamSizes {
		pieces := thinkingBenchTokens(thinkingBenchStream(size, 0.50, "<think>", "</think>"))
		b.Run(core.Sprintf("Tokens%d", size), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				processor := NewProcessor(Config{Mode: Hide}, Hint{Architecture: "qwen3"})
				for _, piece := range pieces {
					thinkingBenchText = processor.Process(piece)
				}
				thinkingBenchText = processor.Flush()
			}
		})
	}
}

func Benchmark_Thinking_Process_Capture_Qwen_PerToken(b *testing.B) {
	for _, size := range thinkingBenchStreamSizes {
		pieces := thinkingBenchTokens(thinkingBenchStream(size, 0.50, "<think>", "</think>"))
		b.Run(core.Sprintf("Tokens%d", size), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				processor := NewProcessor(Config{Mode: Capture, Capture: func(Chunk) {}}, Hint{Architecture: "qwen3"})
				for _, piece := range pieces {
					thinkingBenchText = processor.Process(piece)
				}
				thinkingBenchText = processor.Flush()
			}
		})
	}
}

// Vary span fraction at fixed 256-token length — covers the 10/50/90%
// reasoning-density profile.
var thinkingBenchSpanFractions = []struct {
	id   string
	frac float64
}{
	{"Span10pct", 0.10},
	{"Span50pct", 0.50},
	{"Span90pct", 0.90},
}

func Benchmark_Thinking_Process_Hide_Qwen_Span(b *testing.B) {
	for _, span := range thinkingBenchSpanFractions {
		pieces := thinkingBenchTokens(thinkingBenchStream(256, span.frac, "<think>", "</think>"))
		b.Run(span.id, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				processor := NewProcessor(Config{Mode: Hide}, Hint{Architecture: "qwen3"})
				for _, piece := range pieces {
					thinkingBenchText = processor.Process(piece)
				}
				thinkingBenchText = processor.Flush()
			}
		})
	}
}

// Gemma + gpt-oss carry the worst-case marker fan-out — markersForHint
// builds a much bigger marker set, and findStart pays per token.
func Benchmark_Thinking_Process_Hide_Gemma_PerToken(b *testing.B) {
	pieces := thinkingBenchTokens(thinkingBenchStream(256, 0.50, "<start_of_turn>thinking\n", "<end_of_turn>"))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		processor := NewProcessor(Config{Mode: Hide}, Hint{Architecture: "gemma4_text"})
		for _, piece := range pieces {
			thinkingBenchText = processor.Process(piece)
		}
		thinkingBenchText = processor.Flush()
	}
}

func Benchmark_Thinking_Process_Hide_GPTOSS_PerToken(b *testing.B) {
	pieces := thinkingBenchTokens(thinkingBenchStream(256, 0.50, "<|channel>analysis\n", "<|channel>final\n"))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		processor := NewProcessor(Config{Mode: Hide}, Hint{Architecture: "gpt-oss"})
		for _, piece := range pieces {
			thinkingBenchText = processor.Process(piece)
		}
		thinkingBenchText = processor.Flush()
	}
}

// Process pays nothing in Show mode beyond the type-switch + concat —
// exercise that fast path as a baseline.
func Benchmark_Thinking_Process_Show_Single(b *testing.B) {
	processor := NewProcessor(Config{Mode: Show}, Hint{Architecture: "qwen3"})
	piece := "word "
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		thinkingBenchText = processor.Process(piece)
	}
}

// Hide-mode single-piece call when there's no marker in flight —
// pays the pending-append + drain probe cost.
func Benchmark_Thinking_Process_Hide_NoMarker_Single(b *testing.B) {
	processor := NewProcessor(Config{Mode: Hide}, Hint{Architecture: "qwen3"})
	piece := "word "
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		thinkingBenchText = processor.Process(piece)
	}
}

// --- Flush ---

func Benchmark_Thinking_Flush_NoPending(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		processor := NewProcessor(Config{Mode: Hide}, Hint{Architecture: "qwen3"})
		b.StartTimer()
		thinkingBenchText = processor.Flush()
	}
}

func Benchmark_Thinking_Flush_OpenReasoning(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		processor := NewProcessor(Config{Mode: Hide}, Hint{Architecture: "qwen3"})
		processor.Process("<think>partial reasoning never closed")
		b.StartTimer()
		thinkingBenchText = processor.Flush()
	}
}

// --- Reasoning + Chunks accessors ---

func Benchmark_Thinking_Reasoning_Empty(b *testing.B) {
	processor := NewProcessor(Config{Mode: Hide}, Hint{Architecture: "qwen3"})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		thinkingBenchReasoning = processor.Reasoning()
	}
}

func Benchmark_Thinking_Reasoning_Populated(b *testing.B) {
	processor := NewProcessor(Config{Mode: Hide}, Hint{Architecture: "qwen3"})
	for _, piece := range thinkingBenchTokens(thinkingBenchStream(256, 0.50, "<think>", "</think>")) {
		processor.Process(piece)
	}
	processor.Flush()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		thinkingBenchReasoning = processor.Reasoning()
	}
}

func Benchmark_Thinking_Chunks_Empty(b *testing.B) {
	processor := NewProcessor(Config{Mode: Hide}, Hint{Architecture: "qwen3"})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		thinkingBenchChunks = processor.Chunks()
	}
}

func Benchmark_Thinking_Chunks_Populated(b *testing.B) {
	processor := NewProcessor(Config{Mode: Capture, Capture: func(Chunk) {}}, Hint{Architecture: "qwen3"})
	for _, piece := range thinkingBenchTokens(thinkingBenchStream(256, 0.50, "<think>", "</think>")) {
		processor.Process(piece)
	}
	processor.Flush()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		thinkingBenchChunks = processor.Chunks()
	}
}

// --- longestSuffixPrefix: per-token held-tail check inside Process() ---

func Benchmark_Thinking_LongestSuffixPrefix_NoMatch(b *testing.B) {
	text := "ordinary text with no marker prefix at the end"
	markers := []string{"<think>", "<thinking>", "<reasoning>", "<analysis>"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		thinkingBenchKeep = longestSuffixPrefix(text, markers)
	}
}

func Benchmark_Thinking_LongestSuffixPrefix_PartialMatch(b *testing.B) {
	text := "ordinary text trailing with <thi"
	markers := []string{"<think>", "<thinking>", "<reasoning>", "<analysis>"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		thinkingBenchKeep = longestSuffixPrefix(text, markers)
	}
}

func Benchmark_Thinking_LongestSuffixPrefix_LongMarkerSet(b *testing.B) {
	// Build the gemma marker fan-out as a starts-only list.
	gemma := gemmaMarkers()
	starts := make([]string, 0, len(gemma))
	for _, m := range gemma {
		starts = append(starts, m.start)
	}
	text := "ordinary text trailing with <start_of_t"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		thinkingBenchKeep = longestSuffixPrefix(text, starts)
	}
}

// AX-11: alloc budget for markersForHint. The flattened marker view +
// its parallel start-set are cached on the builtin parser at registry
// build time, so the per-stream resolve must not allocate either slice.
// Family now scans the arch + adapter keys separately (no joined-string
// Concat), so the resolve is fully zero-alloc for already-canonical
// hints. The only residual is NormaliseKey's '-' → '_' rewrite, which a
// dash-bearing arch name (gpt-oss) still pays. A regression above this
// means the per-stream view alloc has returned (each NewProcessor pays
// it again, and each thousand-token response opens a stream).
func TestAllocBudget_Thinking_MarkersForHint(t *testing.T) {
	cases := []struct {
		name string
		hint Hint
	}{
		{"Qwen", Hint{Architecture: "qwen3"}},
		{"Gemma", Hint{Architecture: "gemma4_text"}},
		{"GPTOSS", Hint{Architecture: "gpt-oss"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			avg := testing.AllocsPerRun(5, func() {
				thinkingBenchMarkers = markersForHint(tc.hint)
			})
			// Floor: 0 allocs; Family builds no Concat now. Hints
			// that carry a dash in the architecture name (gpt-oss) pay
			// one extra for the NormaliseKey '-' → '_' replace before
			// the family lookup. That dash replace is the lone cost — the
			// markersForHint view itself is zero-alloc.
			budget := 0.0
			if tc.name == "GPTOSS" {
				budget = 1.0
			}
			if avg > budget {
				t.Fatalf("markersForHint(%s) alloc budget exceeded: %.1f allocs/call (budget=%.0f)\n"+
					"This is per-stream build cost. A regression here re-allocates the\n"+
					"flat thinkingMarker view + start-set on every NewProcessor call.\n"+
					"Profile: go test -bench=Benchmark_Thinking_MarkersForHint_%s -benchmem -memprofile=/tmp/m.mem",
					tc.name, avg, budget, tc.name)
			}
		})
	}
}

// AX-11: alloc budget for NewProcessor. The marker + start-set views
// come from the cached parser; the per-stream NewProcessor must only
// allocate the Processor struct itself plus the Family-path transient.
// Streaming responses open one Processor per request — a regression
// scales per-request, not per-token.
func TestAllocBudget_Thinking_NewProcessor(t *testing.T) {
	cases := []struct {
		name string
		hint Hint
	}{
		{"Qwen", Hint{Architecture: "qwen3"}},
		{"Gemma", Hint{Architecture: "gemma4_text"}},
		{"GPTOSS", Hint{Architecture: "gpt-oss"}},
	}
	cfg := Config{Mode: Hide}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			avg := testing.AllocsPerRun(5, func() {
				thinkingBenchProcessor = NewProcessor(cfg, tc.hint)
			})
			// Floor: 1 alloc for the &Processor{} struct. The Family
			// Concat is gone. Architectures carrying a dash pay one extra
			// for NormaliseKey's '-' → '_' replace.
			budget := 1.0
			if tc.name == "GPTOSS" {
				budget = 2.0
			}
			if avg > budget {
				t.Fatalf("NewProcessor(%s) alloc budget exceeded: %.1f allocs/call (budget=%.0f)\n"+
					"This is per-stream open cost. A regression here means we re-built\n"+
					"the marker view or start-set instead of sharing the registry copy.\n"+
					"Profile: go test -bench=Benchmark_Thinking_NewProcessor_%s -benchmem -memprofile=/tmp/np.mem",
					tc.name, avg, budget, tc.name)
			}
		})
	}
}
