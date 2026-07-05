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
		genericMarkersCache = []reasoningMarker{
			{start: "<thinking>", ends: []string{"</thinking>"}, kind: "thinking"},
			{start: "<thought>", ends: []string{"</thought>"}, kind: "thinking"},
			{start: "<reasoning>", ends: []string{"</reasoning>"}, kind: "reasoning"},
			{start: "<analysis>", ends: []string{"</analysis>"}, kind: "analysis"},
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
			{start: "<|channel>thought\n", ends: []string{"<channel|>"}, kind: "thinking"},
			{start: "<|channel>thinking\n", ends: []string{"<channel|>"}, kind: "thinking"},
			{start: "<|channel>reasoning\n", ends: []string{"<channel|>"}, kind: "reasoning"},
			{start: "<|channel>analysis\n", ends: []string{"<channel|>"}, kind: "analysis"},
			{start: "<start_of_turn>thinking\n", ends: []string{"<end_of_turn>"}, kind: "thinking"},
			{start: "<start_of_turn>thought\n", ends: []string{"<end_of_turn>"}, kind: "thinking"},
			{start: "<start_of_turn>analysis\n", ends: []string{"<end_of_turn>"}, kind: "analysis"},
			{start: "<start_of_turn>reasoning\n", ends: []string{"<end_of_turn>"}, kind: "reasoning"},
		}, genericMarkers()...)
	})
	return gemmaMarkersCache
}

func gptOSSMarkers() []reasoningMarker {
	gptOSSMarkersOnce.Do(func() {
		gptOSSMarkersCache = append([]reasoningMarker{
			{start: "<|channel>analysis\n", ends: []string{"<|channel>final\n", "<|channel>assistant\n", "<|channel>assistant"}, kind: "analysis"},
			{start: "<|channel>thought\n", ends: []string{"<|channel>final\n", "<|channel>assistant\n", "<|channel>assistant"}, kind: "thinking"},
			{start: "<|channel>reasoning\n", ends: []string{"<|channel>final\n", "<|channel>assistant\n", "<|channel>assistant"}, kind: "reasoning"},
			{start: "<|channel>analysis", ends: []string{"<|channel>final", "<|channel>assistant"}, kind: "analysis"},
			{start: "<|channel>thought", ends: []string{"<|channel>final", "<|channel>assistant"}, kind: "thinking"},
			{start: "<|channel>reasoning", ends: []string{"<|channel>final", "<|channel>assistant"}, kind: "reasoning"},
		}, genericMarkers()...)
	})
	return gptOSSMarkersCache
}
