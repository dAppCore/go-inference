// SPDX-Licence-Identifier: EUPL-1.2

// gemma_tools.go parses Gemma 4's native function-call syntax —
// <|tool_call>call:NAME{arg:<|"|>val<|"|>,n:5}<tool_call|> — into structured
// inference.ToolCall values with a JSON arguments object. It is distinct from
// tools.go's parseToolText, which handles the generic <tool_call>{json}</tool_call>
// tagging other model families emit. The markers are the vocab tokens
// DecodeToken preserves (grammar.go), so the whole call span survives into the
// decoded stream instead of collapsing to a bare, ambiguous "call:NAME{…}".
package parser

import (
	"strconv"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/jsonenc"
)

// ParseGemmaToolCalls lifts every <|tool_call>…<tool_call|> span out of text,
// returning the parsed calls and the visible text with those spans removed. When
// text carries no tool-call marker it is returned unchanged with no calls — the
// one-scan fast path for the dominant plain-prose reply.
func ParseGemmaToolCalls(text string) ([]inference.ToolCall, string) {
	if indexString(text, ToolCallOpenMarker) < 0 {
		return nil, text
	}
	var calls []inference.ToolCall
	visible := core.NewBuilder()
	rest := text
	for {
		i := indexString(rest, ToolCallOpenMarker)
		if i < 0 {
			visible.WriteString(rest)
			break
		}
		visible.WriteString(rest[:i])
		after := rest[i+len(ToolCallOpenMarker):]
		c := indexString(after, ToolCallCloseMarker)
		if c < 0 {
			// Unclosed call span — keep the raw text visible rather than
			// swallowing a partial flush at end-of-stream.
			visible.WriteString(ToolCallOpenMarker)
			visible.WriteString(after)
			break
		}
		if call, ok := parseGemmaCallSpan(after[:c]); ok {
			calls = append(calls, call)
		} else {
			// Not a well-formed call — surface the raw span rather than drop it.
			visible.WriteString(ToolCallOpenMarker)
			visible.WriteString(after[:c])
			visible.WriteString(ToolCallCloseMarker)
		}
		rest = after[c+len(ToolCallCloseMarker):]
	}
	return calls, core.Trim(visible.String())
}

// parseGemmaCallSpan parses the inner "call:NAME{ARGS}" of one tool-call span.
func parseGemmaCallSpan(span string) (inference.ToolCall, bool) {
	span = core.Trim(span)
	if !core.HasPrefix(span, "call:") {
		return inference.ToolCall{}, false
	}
	span = span[len("call:"):]
	brace := indexString(span, "{")
	if brace < 0 || !core.HasSuffix(span, "}") {
		return inference.ToolCall{}, false
	}
	name := core.Trim(span[:brace])
	if name == "" {
		return inference.ToolCall{}, false
	}
	return inference.ToolCall{
		Type:          "function",
		Name:          name,
		ArgumentsJSON: gemmaArgsToJSON(span[brace+1 : len(span)-1]),
	}, true
}

// gemmaArgsToJSON turns the comma-separated key:value argument list into a JSON
// object. A value wrapped in <|"|> is a string (JSON-escaped); a bare value is a
// number / boolean / null when it parses as one, else it is treated as a string
// — mirroring the cast in the gemma4 function-calling reference.
func gemmaArgsToJSON(inner string) string {
	buf := []byte{'{'}
	first := true
	rest := core.Trim(inner)
	for rest != "" {
		colon := indexString(rest, ":")
		if colon < 0 {
			break
		}
		key := core.Trim(rest[:colon])
		rest = core.Trim(rest[colon+1:])
		var valJSON []byte
		if core.HasPrefix(rest, ToolArgQuoteMarker) {
			rest = rest[len(ToolArgQuoteMarker):]
			end := indexString(rest, ToolArgQuoteMarker)
			if end < 0 {
				valJSON = jsonenc.AppendJSONString(nil, rest)
				rest = ""
			} else {
				valJSON = jsonenc.AppendJSONString(nil, rest[:end])
				rest = rest[end+len(ToolArgQuoteMarker):]
			}
		} else {
			comma := indexString(rest, ",")
			if comma < 0 {
				valJSON = bareArgToJSON(core.Trim(rest))
				rest = ""
			} else {
				valJSON = bareArgToJSON(core.Trim(rest[:comma]))
				rest = rest[comma+1:]
			}
		}
		if key != "" {
			if !first {
				buf = append(buf, ',')
			}
			first = false
			buf = jsonenc.AppendJSONString(buf, key)
			buf = append(buf, ':')
			buf = append(buf, valJSON...)
		}
		rest = core.Trim(rest)
		if core.HasPrefix(rest, ",") {
			rest = core.Trim(rest[1:])
		}
	}
	return string(append(buf, '}'))
}

// bareArgToJSON renders an unquoted argument value: a JSON literal when it is one
// (number / true / false / null), otherwise a JSON string so the object stays
// valid regardless of what the model emitted.
func bareArgToJSON(v string) []byte {
	switch v {
	case "true", "false", "null":
		return []byte(v)
	}
	if _, err := strconv.ParseFloat(v, 64); err == nil {
		return []byte(v)
	}
	return jsonenc.AppendJSONString(nil, v)
}
