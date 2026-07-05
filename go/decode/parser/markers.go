// SPDX-Licence-Identifier: EUPL-1.2

package parser

import "sync"

// Per-architecture marker sets are immutable lookup tables. Each call site
// (newBuiltinOutputParser, parseReasoningText, registry init) consumes them
// read-only and the only mutating consumer — newBuiltinOutputParser — copies
// via append into a fresh slice. We can therefore cache one shared backing
// slice per architecture and hand the same header back on every call.
//
// Before this cache, qwenMarkers / gemmaMarkers / gptOSSMarkers / genericMarkers
// each rebuilt their full marker set on every invocation, allocating one
// slice for the outer `[]reasoningMarker` plus one `[]string` per marker.ends
// literal (e.g. Gemma = 14 allocs / 1664 B). Per-call cost dominated short-lived
// parser construction in tests and any consumer that declined to cache a Registry.

var (
	genericMarkersOnce  sync.Once
	genericMarkersCache []reasoningMarker

	qwenMarkersOnce  sync.Once
	qwenMarkersCache []reasoningMarker

	gemmaMarkersOnce  sync.Once
	gemmaMarkersCache []reasoningMarker

	gptOSSMarkersOnce  sync.Once
	gptOSSMarkersCache []reasoningMarker
)

func genericMarkers() []reasoningMarker {
	genericMarkersOnce.Do(func() {
		// Derived from the authoritative grammar table (grammar.go); the
		// generic set excludes <think> — that spelling is the qwen family's.
		for _, m := range PairedReasoningMarkers() {
			if m.Start == "<think>" {
				continue
			}
			genericMarkersCache = append(genericMarkersCache, reasoningMarker{start: m.Start, ends: []string{m.End}, kind: m.Kind})
		}
	})
	return genericMarkersCache
}

func qwenMarkers() []reasoningMarker {
	qwenMarkersOnce.Do(func() {
		qwenMarkersCache = append([]reasoningMarker{
			{start: "<think>", ends: []string{"</think>"}, kind: "thinking"},
		}, genericMarkers()...)
	})
	return qwenMarkersCache
}

func gemmaMarkers() []reasoningMarker {
	gemmaMarkersOnce.Do(func() {
		gemmaMarkersCache = append([]reasoningMarker{
			{start: ChannelOpenMarker + "thought\n", ends: []string{ChannelCloseMarker}, kind: "thinking"},
			{start: ChannelOpenMarker + "thinking\n", ends: []string{ChannelCloseMarker}, kind: "thinking"},
			{start: ChannelOpenMarker + "reasoning\n", ends: []string{ChannelCloseMarker}, kind: "reasoning"},
			{start: ChannelOpenMarker + "analysis\n", ends: []string{ChannelCloseMarker}, kind: "analysis"},
			{start: "<start_of_turn>thinking\n", ends: []string{GemmaTurnTerminator}, kind: "thinking"},
			{start: "<start_of_turn>thought\n", ends: []string{GemmaTurnTerminator}, kind: "thinking"},
			{start: "<start_of_turn>analysis\n", ends: []string{GemmaTurnTerminator}, kind: "analysis"},
			{start: "<start_of_turn>reasoning\n", ends: []string{GemmaTurnTerminator}, kind: "reasoning"},
		}, genericMarkers()...)
	})
	return gemmaMarkersCache
}

func gptOSSMarkers() []reasoningMarker {
	gptOSSMarkersOnce.Do(func() {
		gptOSSMarkersCache = append([]reasoningMarker{
			{start: ChannelOpenMarker + "analysis\n", ends: []string{ChannelOpenMarker + "final\n", ChannelOpenMarker + "assistant\n", ChannelOpenMarker + "assistant"}, kind: "analysis"},
			{start: ChannelOpenMarker + "thought\n", ends: []string{ChannelOpenMarker + "final\n", ChannelOpenMarker + "assistant\n", ChannelOpenMarker + "assistant"}, kind: "thinking"},
			{start: ChannelOpenMarker + "reasoning\n", ends: []string{ChannelOpenMarker + "final\n", ChannelOpenMarker + "assistant\n", ChannelOpenMarker + "assistant"}, kind: "reasoning"},
			{start: ChannelOpenMarker + "analysis", ends: []string{ChannelOpenMarker + "final", ChannelOpenMarker + "assistant"}, kind: "analysis"},
			{start: ChannelOpenMarker + "thought", ends: []string{ChannelOpenMarker + "final", ChannelOpenMarker + "assistant"}, kind: "thinking"},
			{start: ChannelOpenMarker + "reasoning", ends: []string{ChannelOpenMarker + "final", ChannelOpenMarker + "assistant"}, kind: "reasoning"},
		}, genericMarkers()...)
	})
	return gptOSSMarkersCache
}
