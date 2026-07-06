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
	"slices"
	"strconv"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/jsonenc"
)

// ToolDecl is one tool a caller offers the model — the engine-neutral form both
// the Anthropic and OpenAI providers convert their wire tools into, so the Gemma
// 4 declaration syntax has a single renderer (RenderGemmaToolDeclarations).
type ToolDecl struct {
	Name        string
	Description string
	Properties  map[string]ToolParam
	Required    []string
}

// ToolParam is one tool parameter's JSON-schema type + description.
type ToolParam struct {
	Type        string
	Description string
}

// RenderGemmaToolDeclarations renders tools into Gemma 4's native declaration
// syntax — one <|tool>declaration:NAME{...}<tool|> block per tool — matching the
// format the model was trained on (reference/gemma/capabilities-text-function-
// calling-gemma4.md). Empty when no tools; property order is sorted so the prompt
// is deterministic (the model does not key on declaration order).
func RenderGemmaToolDeclarations(tools []ToolDecl) string {
	if len(tools) == 0 {
		return ""
	}
	q := ToolArgQuoteMarker
	b := core.NewBuilder()
	for _, t := range tools {
		b.WriteString("<|tool>declaration:")
		b.WriteString(t.Name)
		b.WriteString("{description:")
		b.WriteString(q)
		b.WriteString(t.Description)
		b.WriteString(q)
		b.WriteString(",parameters:{properties:{")
		names := make([]string, 0, len(t.Properties))
		for name := range t.Properties {
			names = append(names, name)
		}
		slices.Sort(names)
		for i, name := range names {
			if i > 0 {
				b.WriteString(",")
			}
			p := t.Properties[name]
			b.WriteString(name)
			b.WriteString(":{description:")
			b.WriteString(q)
			b.WriteString(p.Description)
			b.WriteString(q)
			b.WriteString(",type:")
			b.WriteString(q)
			b.WriteString(gemmaSchemaType(p.Type))
			b.WriteString(q)
			b.WriteString("} ")
		}
		b.WriteString("},required:[")
		for i, r := range t.Required {
			if i > 0 {
				b.WriteString(",")
			}
			b.WriteString(q)
			b.WriteString(r)
			b.WriteString(q)
		}
		b.WriteString("],type:")
		b.WriteString(q)
		b.WriteString("OBJECT")
		b.WriteString(q)
		b.WriteString("} }<tool|>")
	}
	return b.String()
}

// RenderGemmaToolCall renders a prior assistant tool call back into the gemma4
// wire form the model both emits and reads —
// <|tool_call>call:NAME{args}<tool_call|> — so a STATELESS client that replays
// full conversation history (rather than relying on server-side KV continuity)
// still carries the call context that a following tool_result answers. It is the
// exact inverse of ParseGemmaToolCalls' gemmaArgsToJSON: string values wrap in
// ToolArgQuoteMarker, scalars stay bare, objects/arrays recurse. argumentsJSON is
// the call's arguments as a JSON object; malformed/empty yields "{}".
func RenderGemmaToolCall(name, argumentsJSON string) string {
	var args map[string]any
	if core.Trim(argumentsJSON) != "" {
		if res := core.JSONUnmarshal([]byte(argumentsJSON), &args); !res.OK {
			args = nil
		}
	}
	b := core.NewBuilder()
	b.WriteString(ToolCallOpenMarker)
	b.WriteString("call:")
	b.WriteString(name)
	b.WriteString(renderGemmaObject(args))
	b.WriteString(ToolCallCloseMarker)
	return b.String()
}

// renderGemmaObject renders a decoded JSON object as a gemma tool-arg body
// {k:v,…}; keys are sorted so the render is deterministic.
func renderGemmaObject(m map[string]any) string {
	b := core.NewBuilder()
	b.WriteString("{")
	names := make([]string, 0, len(m))
	for k := range m {
		names = append(names, k)
	}
	slices.Sort(names)
	for i, k := range names {
		if i > 0 {
			b.WriteString(",")
		}
		b.WriteString(k)
		b.WriteString(":")
		b.WriteString(renderGemmaValue(m[k]))
	}
	b.WriteString("}")
	return b.String()
}

// renderGemmaArray renders a decoded JSON array as a gemma [v,…] body.
func renderGemmaArray(a []any) string {
	b := core.NewBuilder()
	b.WriteString("[")
	for i, v := range a {
		if i > 0 {
			b.WriteString(",")
		}
		b.WriteString(renderGemmaValue(v))
	}
	b.WriteString("]")
	return b.String()
}

// renderGemmaValue renders one decoded JSON value in gemma tool-arg form: a
// string wraps in ToolArgQuoteMarker; an object/array recurses; a number/bool/
// null renders as its bare scalar (identical to the gemma form).
func renderGemmaValue(v any) string {
	switch t := v.(type) {
	case string:
		return ToolArgQuoteMarker + t + ToolArgQuoteMarker
	case map[string]any:
		return renderGemmaObject(t)
	case []any:
		return renderGemmaArray(t)
	case float64:
		return strconv.FormatFloat(t, 'g', -1, 64)
	case bool:
		if t {
			return "true"
		}
		return "false"
	default: // nil or an unexpected type
		return "null"
	}
}

// gemmaSchemaType maps a JSON-schema type name to Gemma 4's uppercase form
// (string -> STRING, …); an unknown type upper-cases as-is, empty -> STRING.
func gemmaSchemaType(t string) string {
	switch core.Lower(t) {
	case "string":
		return "STRING"
	case "integer":
		return "INTEGER"
	case "number":
		return "NUMBER"
	case "boolean":
		return "BOOLEAN"
	case "object":
		return "OBJECT"
	case "array":
		return "ARRAY"
	case "":
		return "STRING"
	default:
		return core.Upper(t)
	}
}

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

// gemmaArgsToJSON turns a tool-call's {ARGS} body into a JSON object, recursing
// into nested {objects} and [arrays]. A value wrapped in <|"|> is a string
// (JSON-escaped, so an embedded comma or brace can't break the structure); a
// bare value is a number / boolean / null when it parses as one, else a string
// — mirroring the cast in the gemma4 function-calling reference. inner is the
// already-unwrapped object body (the outer braces stripped by the caller).
func gemmaArgsToJSON(inner string) string {
	obj, _ := gemmaObjectBody(inner, 0)
	return string(obj)
}

// gemmaObjectBody parses key:value pairs from s[i] into a JSON object, stopping
// at a closing '}' (consumed) or end of input. Nested values recurse via
// gemmaValue. Returns the JSON object and the index just past the body.
func gemmaObjectBody(s string, i int) ([]byte, int) {
	buf := []byte{'{'}
	first := true
	for {
		i = gemmaSkipSep(s, i)
		if i >= len(s) || s[i] == '}' {
			if i < len(s) {
				i++ // consume '}'
			}
			break
		}
		colon := gemmaIndexByte(s, i, ':')
		if colon < 0 {
			break
		}
		key := core.Trim(s[i:colon])
		val, next := gemmaValue(s, colon+1)
		if key != "" {
			if !first {
				buf = append(buf, ',')
			}
			first = false
			buf = jsonenc.AppendJSONString(buf, key)
			buf = append(buf, ':')
			buf = append(buf, val...)
		}
		i = next
	}
	return append(buf, '}'), i
}

// gemmaArrayBody parses comma-separated values from s[i] into a JSON array,
// stopping at ']' (consumed) or end. Returns the array and the index past it.
func gemmaArrayBody(s string, i int) ([]byte, int) {
	buf := []byte{'['}
	first := true
	for {
		i = gemmaSkipSep(s, i)
		if i >= len(s) || s[i] == ']' {
			if i < len(s) {
				i++ // consume ']'
			}
			break
		}
		val, next := gemmaValue(s, i)
		if !first {
			buf = append(buf, ',')
		}
		first = false
		buf = append(buf, val...)
		i = next
	}
	return append(buf, ']'), i
}

// gemmaValue parses one value at s[i] — a <|"|>-delimited string, a {nested
// object}, an [array], or a bare scalar — returning its JSON bytes and the index
// just past the value.
func gemmaValue(s string, i int) ([]byte, int) {
	for i < len(s) && (s[i] == ' ' || s[i] == '\t') {
		i++
	}
	if i >= len(s) {
		return []byte("null"), i
	}
	if core.HasPrefix(s[i:], ToolArgQuoteMarker) {
		j := i + len(ToolArgQuoteMarker)
		end := gemmaIndexStr(s, j, ToolArgQuoteMarker)
		if end < 0 {
			return jsonenc.AppendJSONString(nil, s[j:]), len(s)
		}
		return jsonenc.AppendJSONString(nil, s[j:end]), end + len(ToolArgQuoteMarker)
	}
	switch s[i] {
	case '{':
		return gemmaObjectBody(s, i+1)
	case '[':
		return gemmaArrayBody(s, i+1)
	}
	// Bare scalar — runs to the next separator or closing bracket.
	j := i
	for j < len(s) && s[j] != ',' && s[j] != '}' && s[j] != ']' {
		j++
	}
	return bareArgToJSON(core.Trim(s[i:j])), j
}

// gemmaSkipSep skips whitespace and leading separator commas.
func gemmaSkipSep(s string, i int) int {
	for i < len(s) && (s[i] == ' ' || s[i] == '\t' || s[i] == ',') {
		i++
	}
	return i
}

// gemmaIndexByte returns the index of byte b in s at or after i, or -1.
func gemmaIndexByte(s string, i int, b byte) int {
	for ; i < len(s); i++ {
		if s[i] == b {
			return i
		}
	}
	return -1
}

// gemmaIndexStr returns the index of sub in s at or after i, or -1.
func gemmaIndexStr(s string, i int, sub string) int {
	if idx := indexString(s[i:], sub); idx >= 0 {
		return i + idx
	}
	return -1
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
