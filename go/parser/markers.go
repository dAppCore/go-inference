// SPDX-Licence-Identifier: EUPL-1.2

package parser

func qwenMarkers() []reasoningMarker {
	return append([]reasoningMarker{
		{start: "<think>", ends: []string{"</think>"}, kind: "thinking"},
	}, genericMarkers()...)
}

func gemmaMarkers() []reasoningMarker {
	return append([]reasoningMarker{
		{start: "<start_of_turn>thinking\n", ends: []string{"<end_of_turn>"}, kind: "thinking"},
		{start: "<start_of_turn>thought\n", ends: []string{"<end_of_turn>"}, kind: "thinking"},
		{start: "<start_of_turn>analysis\n", ends: []string{"<end_of_turn>"}, kind: "analysis"},
		{start: "<start_of_turn>reasoning\n", ends: []string{"<end_of_turn>"}, kind: "reasoning"},
	}, genericMarkers()...)
}

func gptOSSMarkers() []reasoningMarker {
	return append([]reasoningMarker{
		{start: "<|channel>analysis\n", ends: []string{"<|channel>final\n", "<|channel>assistant\n", "<|channel>assistant"}, kind: "analysis"},
		{start: "<|channel>thought\n", ends: []string{"<|channel>final\n", "<|channel>assistant\n", "<|channel>assistant"}, kind: "thinking"},
		{start: "<|channel>reasoning\n", ends: []string{"<|channel>final\n", "<|channel>assistant\n", "<|channel>assistant"}, kind: "reasoning"},
		{start: "<|channel>analysis", ends: []string{"<|channel>final", "<|channel>assistant"}, kind: "analysis"},
		{start: "<|channel>thought", ends: []string{"<|channel>final", "<|channel>assistant"}, kind: "thinking"},
		{start: "<|channel>reasoning", ends: []string{"<|channel>final", "<|channel>assistant"}, kind: "reasoning"},
	}, genericMarkers()...)
}

func genericMarkers() []reasoningMarker {
	return []reasoningMarker{
		{start: "<thinking>", ends: []string{"</thinking>"}, kind: "thinking"},
		{start: "<thought>", ends: []string{"</thought>"}, kind: "thinking"},
		{start: "<reasoning>", ends: []string{"</reasoning>"}, kind: "reasoning"},
		{start: "<analysis>", ends: []string{"</analysis>"}, kind: "analysis"},
	}
}
