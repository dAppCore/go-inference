// SPDX-Licence-Identifier: EUPL-1.2

package parser

import (
	"dappco.re/go/inference"
)

type builtinOutputParser struct {
	id      string
	markers []reasoningMarker
	// Pre-built thinking-mode views over markers. The conversion from
	// reasoningMarker (with []ends) into a flat []thinkingMarker fires
	// every NewProcessor call on the stream-build path; both views are
	// read-only after construction so we hold them on the parser and
	// hand them out by reference. Saves a slice alloc + the per-end
	// flatten loop per stream — see thinking.go markersForHint.
	thinkingMarkers []thinkingMarker
	thinkingStarts  []string
}

func newBuiltinOutputParser(id string, markers []reasoningMarker) *builtinOutputParser {
	owned := append([]reasoningMarker(nil), markers...)
	// Pre-size to the exact total flattened end count so the build
	// loop never re-grows — GPT-OSS markers have 3 ends per start,
	// which previously forced two extra slice grows per call.
	total := 0
	for _, m := range owned {
		for _, end := range m.ends {
			if m.start == "" || end == "" {
				continue
			}
			total++
		}
	}
	thinkingMarkers := make([]thinkingMarker, 0, total)
	thinkingStarts := make([]string, 0, total)
	for _, m := range owned {
		for _, end := range m.ends {
			if m.start == "" || end == "" {
				continue
			}
			thinkingMarkers = append(thinkingMarkers, thinkingMarker{
				start:   m.start,
				end:     end,
				channel: m.kind,
				model:   id,
			})
			thinkingStarts = append(thinkingStarts, m.start)
		}
	}
	return &builtinOutputParser{
		id:              id,
		markers:         owned,
		thinkingMarkers: thinkingMarkers,
		thinkingStarts:  thinkingStarts,
	}
}

func (parser *builtinOutputParser) ParserID() string {
	if parser == nil || parser.id == "" {
		return "generic"
	}
	return parser.id
}

func (parser *builtinOutputParser) ParseReasoning(_ []inference.Token, text string) (inference.ReasoningParseResult, error) {
	if parser == nil {
		parser = newBuiltinOutputParser("generic", genericMarkers())
	}
	return parseReasoningText(text, parser.markers), nil
}

func (parser *builtinOutputParser) ParseTools(_ []inference.Token, text string) (inference.ToolParseResult, error) {
	return parseToolText(text)
}
