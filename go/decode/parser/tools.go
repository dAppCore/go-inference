// SPDX-Licence-Identifier: EUPL-1.2

package parser

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

var toolBlockMarkers = []toolBlockMarker{
	{start: "<tool_call>", end: "</tool_call>"},
	{start: "<tool_calls>", end: "</tool_calls>"},
	{start: "<function_call>", end: "</function_call>"},
}

func parseToolText(text string) (inference.ToolParseResult, error) {
	// Lazy-build the visible builder + calls slice. The common no-call
	// case (plain assistant prose with no tool markers) is one
	// findToolBlockStart scan + return of the original string — no
	// builder copy, no empty slice header, no fallback parse. The
	// previous shape paid a full visible.WriteString(text) + .String()
	// copy of the entire response on every no-call call.
	var (
		visible     *core.Builder
		calls       []inference.ToolCall
		foundTagged bool
		pending     = text
	)
	for pending != "" {
		idx, marker, ok := findToolBlockStart(pending)
		if !ok {
			if visible != nil {
				visible.WriteString(pending)
			}
			break
		}
		afterStart := pending[idx+len(marker.start):]
		end := indexString(afterStart, marker.end)
		if end < 0 {
			// Unclosed tagged block — every byte of `pending` is plain
			// visible content. If this is the first iteration (no
			// builder yet AND no prior successful blocks), the whole
			// `text` IS the visible string; return it directly without
			// the builder.String() copy. Adapter sites that emit
			// unclosed tool-call tags hit this branch — token streams
			// where the model emits "<tool_call>{..." then continues
			// generating prose without ever closing the tag, or where
			// the parser sees a partial flush at end-of-stream.
			if visible == nil {
				return inference.ToolParseResult{VisibleText: text, Calls: nil}, nil
			}
			visible.WriteString(pending)
			foundTagged = true
			break
		}
		foundTagged = true
		if visible == nil {
			visible = core.NewBuilder()
			visible.Grow(len(text))
		}
		visible.WriteString(pending[:idx])
		parsed, err := parseToolPayload(afterStart[:end])
		if err != nil {
			return inference.ToolParseResult{}, err
		}
		calls = append(calls, parsed...)
		pending = afterStart[end+len(marker.end):]
	}
	if !foundTagged {
		parsed, err := parseToolPayload(text)
		if err == nil && len(parsed) > 0 {
			return inference.ToolParseResult{VisibleText: "", Calls: parsed}, nil
		}
		// No tags found AND no JSON-shaped payload — the input is
		// plain prose. Return it as-is; no builder copy needed.
		return inference.ToolParseResult{VisibleText: text, Calls: nil}, nil
	}
	return inference.ToolParseResult{VisibleText: visible.String(), Calls: calls}, nil
}

func findToolBlockStart(text string) (int, toolBlockMarker, bool) {
	best := -1
	var marker toolBlockMarker
	for _, candidate := range toolBlockMarkers {
		idx := indexString(text, candidate.start)
		if idx < 0 {
			continue
		}
		if best < 0 || idx < best {
			best = idx
			marker = candidate
		}
	}
	return best, marker, best >= 0
}

type parsedToolCall struct {
	ID            string           `json:"id"`
	Type          string           `json:"type"`
	Name          string           `json:"name"`
	Arguments     core.RawMessage  `json:"arguments"`
	ArgumentsJSON string           `json:"arguments_json"`
	Function      *parsedFunction  `json:"function"`
	ToolCalls     []parsedToolCall `json:"tool_calls"`
	Calls         []parsedToolCall `json:"calls"`
}

type parsedFunction struct {
	Name      string          `json:"name"`
	Arguments core.RawMessage `json:"arguments"`
}

func parseToolPayload(payload string) ([]inference.ToolCall, error) {
	payload = core.Trim(payload)
	if payload == "" {
		return nil, nil
	}
	// Cheap shape check before reflection-decoding — a tool-call payload
	// is always JSON. If the trimmed text doesn't start with '[' or '{',
	// don't pay the encoding/json reflect walk just to discover that
	// fact (the common no-tool-calls case the streaming parser feeds us
	// is plain assistant prose).
	first := payload[0]
	if first != '[' && first != '{' {
		return nil, nil
	}
	var list []parsedToolCall
	if first == '[' {
		result := core.JSONUnmarshalString(payload, &list)
		if !result.OK {
			return nil, resultError("parser.tool", result)
		}
		return convertParsedToolCalls(list), nil
	}
	var envelope parsedToolCall
	result := core.JSONUnmarshalString(payload, &envelope)
	if !result.OK {
		return nil, resultError("parser.tool", result)
	}
	if len(envelope.ToolCalls) > 0 {
		return convertParsedToolCalls(envelope.ToolCalls), nil
	}
	if len(envelope.Calls) > 0 {
		return convertParsedToolCalls(envelope.Calls), nil
	}
	call := convertParsedToolCall(envelope)
	if call.Name == "" {
		return nil, nil
	}
	return []inference.ToolCall{call}, nil
}

func convertParsedToolCalls(input []parsedToolCall) []inference.ToolCall {
	out := make([]inference.ToolCall, 0, len(input))
	for _, parsed := range input {
		call := convertParsedToolCall(parsed)
		if call.Name != "" {
			out = append(out, call)
		}
	}
	return out
}

func convertParsedToolCall(parsed parsedToolCall) inference.ToolCall {
	name := parsed.Name
	args := parsed.Arguments
	if parsed.Function != nil {
		if parsed.Function.Name != "" {
			name = parsed.Function.Name
		}
		if len(parsed.Function.Arguments) > 0 {
			args = parsed.Function.Arguments
		}
	}
	callType := parsed.Type
	if callType == "" {
		callType = "function"
	}
	return inference.ToolCall{
		ID:            parsed.ID,
		Type:          callType,
		Name:          name,
		ArgumentsJSON: normaliseArgumentsJSON(parsed.ArgumentsJSON, args),
	}
}

// normaliseArgumentsJSON resolves the arguments surface to its JSON
// string. args arrives as a core.RawMessage (deferred-decode bytes)
// rather than `any`, so the common object/array case is the raw bytes
// verbatim — no map[string]any decode + no JSONMarshalString re-encode
// round-trip. A JSON-string-encoded argument (`"{\"id\":7}"`) is
// unquoted to its inner JSON; everything else is used as-is.
func normaliseArgumentsJSON(existing string, args core.RawMessage) string {
	if core.Trim(existing) != "" {
		return core.Trim(existing)
	}
	if len(args) == 0 {
		return ""
	}
	trimmed := core.Trim(string(args))
	if trimmed == "" || trimmed == "null" {
		return ""
	}
	// A JSON string literal carries the arguments as an embedded JSON
	// payload (`"{\"id\":7}"`); unquote it to surface the inner JSON.
	if trimmed[0] == '"' {
		var inner string
		if result := core.JSONUnmarshalString(trimmed, &inner); result.OK {
			return core.Trim(inner)
		}
	}
	return trimmed
}

func resultError(scope string, result core.Result) error {
	if err, ok := result.Value.(error); ok {
		return core.Wrap(err, scope, "parse JSON")
	}
	return core.E(scope, "parse JSON", nil)
}
