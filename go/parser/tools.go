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
	visible := core.NewBuilder()
	calls := []inference.ToolCall{}
	pending := text
	foundTagged := false
	for pending != "" {
		idx, marker, ok := findToolBlockStart(pending)
		if !ok {
			visible.WriteString(pending)
			break
		}
		foundTagged = true
		visible.WriteString(pending[:idx])
		afterStart := pending[idx+len(marker.start):]
		end := indexString(afterStart, marker.end)
		if end < 0 {
			visible.WriteString(pending[idx:])
			break
		}
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
	Arguments     any              `json:"arguments"`
	ArgumentsJSON string           `json:"arguments_json"`
	Function      *parsedFunction  `json:"function"`
	ToolCalls     []parsedToolCall `json:"tool_calls"`
	Calls         []parsedToolCall `json:"calls"`
}

type parsedFunction struct {
	Name      string `json:"name"`
	Arguments any    `json:"arguments"`
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
		if parsed.Function.Arguments != nil {
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

func normaliseArgumentsJSON(existing string, args any) string {
	if core.Trim(existing) != "" {
		return core.Trim(existing)
	}
	if args == nil {
		return ""
	}
	if raw, ok := args.(string); ok {
		return core.Trim(raw)
	}
	return core.JSONMarshalString(args)
}

func resultError(scope string, result core.Result) error {
	if err, ok := result.Value.(error); ok {
		return core.Wrap(err, scope, "parse JSON")
	}
	return core.E(scope, "parse JSON", nil)
}
