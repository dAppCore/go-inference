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

// SupportsToolSyntax reports whether architecture — a loaded model's
// inference.ModelInfo.Architecture — is a Gemma 4 checkpoint, the only family
// RenderGemmaToolDeclarations / ParseGemmaToolCalls actually round-trip tool
// calls for. Both functions speak Gemma 4's native special-token vocabulary
// (<|tool>, <tool|>, <|tool_call>, <tool_call|>, <|"|>) rather than a generic
// convention any instruction-tuned model would recognise, so declaring tools to
// an unsupported architecture would inject bytes its tokenizer has never seen
// as special tokens — the model would see ordinary text, not a tool menu, and
// no reliable tool_calls would ever come back. Callers (the OpenAI/Anthropic
// serving handlers, capability reporting) use this to gate tool declarations
// honestly instead of silently rendering a menu the model can't read.
//
//	parser.SupportsToolSyntax("gemma4_text") // true
//	parser.SupportsToolSyntax("qwen3")       // false
func SupportsToolSyntax(architecture string) bool {
	return core.Contains(core.Lower(architecture), "gemma4")
}

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
	// Reserve the whole render up front so the builder assembles in one
	// allocation instead of doubling its backing per tool, and reuse one
	// property-name scratch across every tool instead of allocating a fresh
	// sort slice per tool. Both are byte-transparent: Grow is a capacity hint
	// and the sort is order-identical whichever backing it runs on.
	b.Grow(gemmaToolDeclarationsSize(tools, q))
	var names []string
	for _, t := range tools {
		b.WriteString("<|tool>declaration:")
		b.WriteString(t.Name)
		b.WriteString("{description:")
		b.WriteString(q)
		b.WriteString(t.Description)
		b.WriteString(q)
		b.WriteString(",parameters:{properties:{")
		names = names[:0]
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

// gemmaToolDeclarationsSize reserves the builder backing for
// RenderGemmaToolDeclarations so the render assembles in one allocation instead
// of the builder's per-tool doubling growth. It mirrors that function's
// WriteString sequence; the reserve is a capacity hint (Grow), so any drift
// between this estimate and the exact output costs only a single extra
// append-grow and never changes a rendered byte. The optional-separator commas
// and the schema-type width (len(Type)+8 covers the "" -> STRING widening and
// any upper-cased custom type) are reserved generously rather than re-walked, so
// the estimate is a safe upper bound without re-calling gemmaSchemaType.
func gemmaToolDeclarationsSize(tools []ToolDecl, q string) int {
	size := 0
	for _, t := range tools {
		// "<|tool>declaration:" + name + "{description:" + q + desc + q
		size += 19 + len(t.Name) + 13 + len(q) + len(t.Description) + len(q)
		size += 25 // ",parameters:{properties:{"
		for name, p := range t.Properties {
			// sep + name + ":{description:" + q + desc + q
			size += 1 + len(name) + 14 + len(q) + len(p.Description) + len(q)
			// ",type:" + q + TYPE + q + "} "
			size += 6 + len(q) + len(p.Type) + 8 + len(q) + 2
		}
		size += 12 // "},required:["
		for _, r := range t.Required {
			size += 1 + len(q) + len(r) + len(q) // sep + q + r + q
		}
		// "],type:" + q + "OBJECT" + q + "} }<tool|>"
		size += 7 + len(q) + 6 + len(q) + 10
	}
	return size
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
	writeGemmaObject(b, args)
	b.WriteString(ToolCallCloseMarker)
	return b.String()
}

// writeGemmaObject writes a decoded JSON object as a gemma tool-arg body {k:v,…}
// into b; keys are sorted so the render is deterministic. The recursion threads
// the single caller builder rather than each level allocating its own builder +
// returning an intermediate string that the parent copies back in — byte-
// identical output, but a nested {…}/[…] no longer costs a builder alloc plus a
// String() copy per level.
func writeGemmaObject(b *core.Builder, m map[string]any) {
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
		writeGemmaValue(b, m[k])
	}
	b.WriteString("}")
}

// writeGemmaArray writes a decoded JSON array as a gemma [v,…] body into b.
func writeGemmaArray(b *core.Builder, a []any) {
	b.WriteString("[")
	for i, v := range a {
		if i > 0 {
			b.WriteString(",")
		}
		writeGemmaValue(b, v)
	}
	b.WriteString("]")
}

// writeGemmaValue writes one decoded JSON value in gemma tool-arg form into b: a
// string wraps in ToolArgQuoteMarker; an object/array recurses; a number/bool/
// null renders as its bare scalar (identical to the gemma form). The float path
// appends through a stack buffer so the scalar never materialises as its own heap
// string.
func writeGemmaValue(b *core.Builder, v any) {
	switch t := v.(type) {
	case string:
		b.WriteString(ToolArgQuoteMarker)
		b.WriteString(t)
		b.WriteString(ToolArgQuoteMarker)
	case map[string]any:
		writeGemmaObject(b, t)
	case []any:
		writeGemmaArray(b, t)
	case float64:
		var buf [32]byte
		b.Write(strconv.AppendFloat(buf[:0], t, 'g', -1, 64))
	case bool:
		if t {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	default: // nil or an unexpected type
		b.WriteString("null")
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
	// Append the whole object into one buffer sized to the input rather than
	// letting every nested value/object/array return its own freshly-allocated
	// []byte the parent then copies back in. JSON quoting/escaping can nudge the
	// size, so a drifted estimate costs at most a single append-grow; the common
	// case is one backing allocation. len(inner)+2 covers the braces.
	obj, _ := gemmaAppendObjectBody(make([]byte, 0, len(inner)+2), inner, 0)
	return string(obj)
}

// gemmaAppendObjectBody appends the JSON object for the key:value pairs in s[i…]
// onto dst, stopping at a closing '}' (consumed) or end of input, and returns the
// grown buffer plus the index just past the body. Nested values recurse via
// gemmaAppendValue, all writing into the same dst — byte-identical to the prior
// shape that returned a fresh []byte per node and copied it back in.
func gemmaAppendObjectBody(dst []byte, s string, i int) ([]byte, int) {
	dst = append(dst, '{')
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
		if key == "" {
			// Keyless pair: parse the value only to advance the index (the prior
			// shape computed val then discarded it), rolling dst back so nothing
			// is emitted.
			mark := len(dst)
			var next int
			dst, next = gemmaAppendValue(dst, s, colon+1)
			dst = dst[:mark]
			i = next
			continue
		}
		if !first {
			dst = append(dst, ',')
		}
		first = false
		dst = jsonenc.AppendJSONString(dst, key)
		dst = append(dst, ':')
		dst, i = gemmaAppendValue(dst, s, colon+1)
	}
	return append(dst, '}'), i
}

// gemmaAppendArrayBody appends the JSON array for the comma-separated values in
// s[i…] onto dst, stopping at ']' (consumed) or end, and returns the grown buffer
// and the index past it.
func gemmaAppendArrayBody(dst []byte, s string, i int) ([]byte, int) {
	dst = append(dst, '[')
	first := true
	for {
		i = gemmaSkipSep(s, i)
		if i >= len(s) || s[i] == ']' {
			if i < len(s) {
				i++ // consume ']'
			}
			break
		}
		if !first {
			dst = append(dst, ',')
		}
		first = false
		dst, i = gemmaAppendValue(dst, s, i)
	}
	return append(dst, ']'), i
}

// gemmaAppendValue appends one value at s[i] — a <|"|>-delimited string, a
// {nested object}, an [array], or a bare scalar — onto dst and returns the grown
// buffer and the index just past the value.
func gemmaAppendValue(dst []byte, s string, i int) ([]byte, int) {
	for i < len(s) && (s[i] == ' ' || s[i] == '\t') {
		i++
	}
	if i >= len(s) {
		return append(dst, "null"...), i
	}
	if core.HasPrefix(s[i:], ToolArgQuoteMarker) {
		j := i + len(ToolArgQuoteMarker)
		end := gemmaIndexStr(s, j, ToolArgQuoteMarker)
		if end < 0 {
			return jsonenc.AppendJSONString(dst, s[j:]), len(s)
		}
		return jsonenc.AppendJSONString(dst, s[j:end]), end + len(ToolArgQuoteMarker)
	}
	switch s[i] {
	case '{':
		return gemmaAppendObjectBody(dst, s, i+1)
	case '[':
		return gemmaAppendArrayBody(dst, s, i+1)
	}
	// Bare scalar — runs to the next separator or closing bracket.
	j := i
	for j < len(s) && s[j] != ',' && s[j] != '}' && s[j] != ']' {
		j++
	}
	return gemmaAppendBareArg(dst, core.Trim(s[i:j])), j
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

// gemmaAppendBareArg appends an unquoted argument value onto dst: a JSON literal
// when it is one (number / true / false / null), otherwise a JSON string so the
// object stays valid regardless of what the model emitted.
func gemmaAppendBareArg(dst []byte, v string) []byte {
	switch v {
	case "true", "false", "null":
		return append(dst, v...)
	}
	if _, err := strconv.ParseFloat(v, 64); err == nil {
		return append(dst, v...)
	}
	return jsonenc.AppendJSONString(dst, v)
}
