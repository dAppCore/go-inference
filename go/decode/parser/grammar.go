// SPDX-Licence-Identifier: EUPL-1.2

package parser

// grammar.go is the single source of the reasoning-marker grammar. Two
// streaming engines consume it — this package's Processor (the state-session
// lane) and serving/provider/openai.ThinkingExtractor (the live /v1/chat
// lane) — and before it existed each carried its own copy of the marker
// tables, so every grammar fix had to land twice (the <end_of_turn>
// terminator landed in both engines in 3825bc8; this file ends that tax).

const (
	// ChannelOpenMarker opens a named output channel — gemma4 emits
	// `<|channel>thought` for its reasoning stream, gpt-oss uses the same
	// open with its harmony channel names (analysis/final/...).
	ChannelOpenMarker = "<|channel>"
	// ChannelCloseMarker is gemma4's explicit channel close: after it the
	// remaining tokens are the visible answer. (gpt-oss instead ends a
	// channel by opening the next one.)
	ChannelCloseMarker = "<channel|>"
	// GemmaTurnTerminator is gemma4's turn-end token. MLX gemma4 snapshots
	// ship it as a PLAIN vocab token (absent from tokenizer.json
	// added_tokens), so no decode layer can hide it — both streaming engines
	// swallow the bare terminator from visible output.
	GemmaTurnTerminator = "<end_of_turn>"
)

// PairedMarker is an explicit open/close reasoning span (`<think>…</think>`).
// Kind is the reasoning channel the span's text belongs to ("thinking",
// "reasoning", "analysis") — the Processor maps it onto chunk channels; the
// openai extractor routes every kind to the thought stream.
type PairedMarker struct {
	Start, End, Kind string
}

// pairedReasoningMarkers is the one authoritative table of explicit reasoning
// spans. genericMarkers derives the Processor's view from it; the openai
// extractor consumes it directly.
var pairedReasoningMarkers = []PairedMarker{
	{Start: "<think>", End: "</think>", Kind: "thinking"},
	{Start: "<thinking>", End: "</thinking>", Kind: "thinking"},
	{Start: "<thought>", End: "</thought>", Kind: "thinking"},
	{Start: "<reasoning>", End: "</reasoning>", Kind: "reasoning"},
	{Start: "<analysis>", End: "</analysis>", Kind: "analysis"},
}

// PairedReasoningMarkers returns the explicit reasoning spans. The slice is a
// package-owned read-only view — callers must not mutate it.
//
//	for _, m := range parser.PairedReasoningMarkers() { starts = append(starts, m.Start) }
func PairedReasoningMarkers() []PairedMarker {
	return pairedReasoningMarkers
}

// IsReasoningChannel reports whether a `<|channel>NAME` names a reasoning
// channel whose text belongs in the thought stream rather than the visible
// content. "analysis" is gpt-oss harmony's reasoning channel; "final" and
// "assistant" (and anything else) are content.
//
//	if parser.IsReasoningChannel(name) { thought += text } else { content += text }
func IsReasoningChannel(name string) bool {
	switch name {
	case "thought", "thinking", "reasoning", "analysis":
		return true
	}
	return false
}
