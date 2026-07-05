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
	// terminators are the family's bare turn-end tokens — control text that is
	// never visible content OUTSIDE a reasoning span (inside one, the span-end
	// match consumes it first). gemma4 MLX snapshots ship <end_of_turn> as a
	// PLAIN vocab token (absent from added_tokens), so the tokenizer cannot
	// hide it and every reply carried a literal trailing "<end_of_turn>" until
	// the Processor learnt to swallow it here.
	terminators []string
	// thinkingHoldback is thinkingStarts + terminators — the combined
	// partial-suffix set the streaming Processor holds back on, prebuilt so
	// NewProcessor stays alloc-free.
	thinkingHoldback []string
}

// turnTerminators maps a builtin parser id to its bare turn-terminator tokens.
// Only a family whose template ends the assistant turn with an in-vocab PLAIN
// token needs one; span-end markers stay on the reasoningMarker ends.
func turnTerminators(id string) []string {
	if id == "gemma" {
		return []string{"<end_of_turn>"}
	}
	return nil
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
	terminators := turnTerminators(id)
	holdback := thinkingStarts
	if len(terminators) > 0 {
		holdback = make([]string, 0, len(thinkingStarts)+len(terminators))
		holdback = append(holdback, thinkingStarts...)
		holdback = append(holdback, terminators...)
	}
	return &builtinOutputParser{
		id:               id,
		markers:          owned,
		thinkingMarkers:  thinkingMarkers,
		thinkingStarts:   thinkingStarts,
		terminators:      terminators,
		thinkingHoldback: holdback,
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
