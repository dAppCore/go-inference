// SPDX-Licence-Identifier: EUPL-1.2

package parser

import (
	"dappco.re/go/inference"
)

type builtinOutputParser struct {
	id      string
	markers []reasoningMarker
}

func newBuiltinOutputParser(id string, markers []reasoningMarker) *builtinOutputParser {
	return &builtinOutputParser{id: id, markers: append([]reasoningMarker(nil), markers...)}
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
