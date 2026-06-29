// SPDX-Licence-Identifier: EUPL-1.2

// Package parse turns a Gemma 4 model's raw text output into structured tool
// calls and a reasoning/answer split, mirroring SGLang's gemma4_detector.py and
// reasoning_parser.py so the same model serialisation decodes identically here.
//
//	calls, normal, err := parse.ParseGemma4ToolCalls(modelOutput)
//	// calls plug straight into tools.Dispatch; normal is the user-facing text.
//
//	reasoning, answer := parse.Gemma4Reasoning().Parse(modelOutput)
package parse

import (
	"strconv"

	core "dappco.re/go"
	tools "dappco.re/go/inference/tools"
)

// Gemma 4 tool-call special tokens, byte-for-byte from SGLang's
// gemma4_detector.py. A tool call is a span TOOL_CALL_START … TOOL_CALL_END
// whose inner text is `call:func_name{args}`; string values inside the args are
// wrapped in STRING_DELIM rather than JSON quotes.
const (
	toolCallStart = "<|tool_call>"
	toolCallEnd   = "<tool_call|>"
	stringDelim   = `<|"|>`
	callPrefix    = "call:"
)

// ParseGemma4ToolCalls extracts every `<|tool_call>…<tool_call|>` span from a
// Gemma 4 response, parses each span's custom key:value argument format into a
// map, and serialises that map to a JSON string for ToolCall.Arguments so the
// result drops straight into tools.Dispatch. normalText is the text before the
// first tool-call token (empty when a tool call is present but preceded by
// nothing). With no tool-call token at all, calls is empty and normalText is the
// whole input — "the model answered without tools".
//
//	calls, normal, err := parse.ParseGemma4ToolCalls(out)
//	if err != nil { return err }
//	if len(calls) == 0 { /* normal is the final answer */ }
func ParseGemma4ToolCalls(text string) (calls []tools.ToolCall, normalText string, err error) {
	calls = []tools.ToolCall{}

	// No start token: the whole text is the answer (SGLang's early return).
	if !core.Contains(text, toolCallStart) {
		return calls, text, nil
	}

	matches := extractGemma4ToolCalls(text)
	if len(matches) == 0 {
		// A start token existed but no usable span (e.g. no matching end token):
		// SGLang returns the whole text as normal_text with no calls.
		return calls, text, nil
	}

	// One ToolCall per match, count known — size once instead of regrowing the
	// backing array call-by-call.
	calls = make([]tools.ToolCall, 0, len(matches))
	for _, m := range matches {
		args := parseGemma4Args(m.args)
		calls = append(calls, tools.ToolCall{
			Name:      m.name,
			Arguments: core.JSONMarshalString(args),
		})
	}

	// Content = text before the first start token. SGLang only keeps it when the
	// token is not at position 0 (content_end > 0), otherwise normal_text is "".
	contentEnd := core.Index(text, toolCallStart)
	if contentEnd > 0 {
		normalText = text[:contentEnd]
	}
	return calls, normalText, nil
}

// gemma4Match is one extracted span: the function name and its raw (still
// unparsed) argument substring, exactly as _extract_tool_calls yields them.
type gemma4Match struct {
	name string
	args string
}

// extractGemma4ToolCalls walks the text finding TOOL_CALL_START … TOOL_CALL_END
// spans, and for each span that begins `call:name{` it slices out the function
// name and the brace-balanced argument body. This is a direct port of SGLang's
// Gemma4Detector._extract_tool_calls — same find/slice arithmetic, same skips.
//
//	matches := extractGemma4ToolCalls(`<|tool_call>call:f{a: 1}<tool_call|>`)
//	// matches[0].name == "f", matches[0].args == "a: 1"
func extractGemma4ToolCalls(text string) []gemma4Match {
	results := []gemma4Match{}
	searchFrom := 0
	for {
		start := indexFrom(text, toolCallStart, searchFrom)
		if start == -1 {
			break
		}
		end := indexFrom(text, toolCallEnd, start)
		if end == -1 {
			break
		}
		inner := text[start+len(toolCallStart) : end]
		if core.HasPrefix(inner, callPrefix) {
			brace := core.Index(inner, "{")
			if brace != -1 {
				funcName := inner[len(callPrefix):brace]
				argsContent := inner[brace+1:]
				matchIdx := findMatchingBrace(argsContent)
				argsStr := argsContent
				if matchIdx != -1 {
					argsStr = argsContent[:matchIdx]
				}
				results = append(results, gemma4Match{name: funcName, args: argsStr})
			}
		}
		searchFrom = end + len(toolCallEnd)
	}
	return results
}

// findMatchingBrace returns the index of the '}' that closes an opening '{'
// already consumed, treating any STRING_DELIM-wrapped run as opaque so braces
// inside a string don't shift the balance. It returns -1 when the braces never
// balance (an incomplete span) — matching SGLang's _find_matching_brace, which
// also returns -1 if a string delimiter run reaches the end unclosed.
//
//	findMatchingBrace("a: 1}") // 4 — the closing brace
func findMatchingBrace(text string) int {
	depth := 1
	i := 0
	n := len(text)
	dl := len(stringDelim)
	for i < n && depth > 0 {
		if i+dl <= n && text[i:i+dl] == stringDelim {
			i += dl
			next := indexFrom(text, stringDelim, i)
			if next == -1 {
				return -1
			}
			i = next + dl
			continue
		}
		switch text[i] {
		case '{':
			depth++
		case '}':
			depth--
		}
		i++
	}
	if depth == 0 {
		return i - 1
	}
	return -1
}

// parseGemma4Args parses Gemma 4's custom `key: value, …` argument format into a
// map[string]any: keys are bare up to ':'; string values are STRING_DELIM
// wrapped; values may be objects {…}, arrays […], booleans, numbers or bare
// strings. A direct port of _parse_gemma4_args, including its tolerant
// end-of-input branches (key with no value -> "", unterminated string -> rest).
//
//	parseGemma4Args(`city: <|"|>Paris<|"|>, days: 3`)
//	// map[string]any{"city": "Paris", "days": 3}
func parseGemma4Args(argsStr string) map[string]any {
	result := map[string]any{}
	if core.Trim(argsStr) == "" {
		return result
	}

	i := 0
	n := len(argsStr)
	dl := len(stringDelim)

	for i < n {
		// Skip whitespace and commas between entries.
		for i < n && isArgSep(argsStr[i]) {
			i++
		}
		if i >= n {
			break
		}

		// Key: bare text up to ':'.
		keyStart := i
		for i < n && argsStr[i] != ':' {
			i++
		}
		if i >= n {
			break
		}
		key := core.Trim(argsStr[keyStart:i])
		i++ // consume ':'

		// Value: nothing left after ':' means an empty-string value.
		if i >= n {
			result[key] = ""
			break
		}
		// Skip whitespace after ':' (not commas — a comma here is the value).
		for i < n && isSpace(argsStr[i]) {
			i++
		}
		if i >= n {
			result[key] = ""
			break
		}

		switch {
		// String: <|"|>…<|"|>.
		case i+dl <= n && argsStr[i:i+dl] == stringDelim:
			i += dl
			valStart := i
			end := indexFrom(argsStr, stringDelim, i)
			if end == -1 {
				result[key] = argsStr[valStart:] // unterminated — take the rest
				return result
			}
			result[key] = argsStr[valStart:end]
			i = end + dl

		// Nested object: {…}.
		case argsStr[i] == '{':
			objStart := i + 1
			i = skipBalanced(argsStr, i+1, '{', '}')
			result[key] = parseGemma4Args(argsStr[objStart : i-1])

		// Array: […].
		case argsStr[i] == '[':
			arrStart := i + 1
			i = skipBalanced(argsStr, i+1, '[', ']')
			result[key] = parseGemma4Array(argsStr[arrStart : i-1])

		// Bare value: number, boolean, or bare string up to , } ].
		default:
			valStart := i
			for i < n && !isValueEnd(argsStr[i]) {
				i++
			}
			result[key] = parseGemma4Value(argsStr[valStart:i])
		}
	}
	return result
}

// parseGemma4Array parses the inside of a Gemma 4 array (the text between '['
// and ']') into a slice, supporting string elements, nested objects, nested
// arrays and bare values — a port of _parse_gemma4_array.
//
//	parseGemma4Array(`1, 2, 3`)            // []any{1, 2, 3}
//	parseGemma4Array(`<|"|>a<|"|>, <|"|>b<|"|>`) // []any{"a", "b"}
func parseGemma4Array(arrStr string) []any {
	// Elements are comma-separated, so commas+1 is the exact element count for a
	// flat array and a safe upper bound otherwise — size the slice once instead
	// of regrowing it element-by-element. Guarded on commas>0 so empty and
	// single-element bodies keep the zero-alloc empty-literal start.
	items := []any{}
	if commas := countByte(arrStr, ','); commas > 0 {
		items = make([]any, 0, commas+1)
	}
	i := 0
	n := len(arrStr)
	dl := len(stringDelim)

	for i < n {
		// Skip whitespace and commas between elements.
		for i < n && isArgSep(arrStr[i]) {
			i++
		}
		if i >= n {
			break
		}

		switch {
		// String element.
		case i+dl <= n && arrStr[i:i+dl] == stringDelim:
			i += dl
			end := indexFrom(arrStr, stringDelim, i)
			if end == -1 {
				items = append(items, arrStr[i:]) // unterminated — take the rest
				return items
			}
			items = append(items, arrStr[i:end])
			i = end + dl

		// Nested object.
		case arrStr[i] == '{':
			objStart := i + 1
			i = skipBalanced(arrStr, i+1, '{', '}')
			items = append(items, parseGemma4Args(arrStr[objStart:i-1]))

		// Nested array (no string-delim handling, matching _parse_gemma4_array).
		case arrStr[i] == '[':
			subStart := i + 1
			depth := 1
			i++
			for i < n && depth > 0 {
				switch arrStr[i] {
				case '[':
					depth++
				case ']':
					depth--
				}
				i++
			}
			items = append(items, parseGemma4Array(arrStr[subStart:i-1]))

		// Bare element up to ',' or ']'.
		default:
			valStart := i
			for i < n && arrStr[i] != ',' && arrStr[i] != ']' {
				i++
			}
			items = append(items, parseGemma4Value(arrStr[valStart:i]))
		}
	}
	return items
}

// parseGemma4Value converts a single bare token (already sliced) into the right
// Go type: "true"/"false" -> bool, an integer- or float-looking token -> the
// number, otherwise the trimmed token as a string. Mirrors _parse_gemma4_value.
//
//	parseGemma4Value("true")  // true
//	parseGemma4Value("1.5")   // 1.5
//	parseGemma4Value("draft") // "draft"
func parseGemma4Value(valueStr string) any {
	valueStr = core.Trim(valueStr)
	if valueStr == "" {
		return valueStr
	}
	if valueStr == "true" {
		return true
	}
	if valueStr == "false" {
		return false
	}
	// Number: probe via the JSON number grammar (core has no float parser). A
	// token that decodes to a JSON number is kept as that number; anything else
	// (quoted text, null, a bare word) falls through to a bare string — the same
	// outcome as Python's int()/float() raising ValueError.
	if num, ok := parseNumber(valueStr); ok {
		return num
	}
	return valueStr // bare string
}

// parseNumber reports whether s is a JSON number and returns it as float64. It
// rejects non-number JSON (null, true, "quoted") so only genuine numbers are
// treated as numeric — matching _parse_gemma4_value's int()/float() guard.
//
// isJSONNumber gates on the exact RFC 8259 number grammar that encoding/json's
// scanner enforces, then strconv.ParseFloat does the very conversion json's
// number decoder runs internally — so the result is byte-identical to decoding
// into an `any`, without the decoder's per-call allocations on this per-value
// hot path (it fires once per bare numeric argument).
//
//	parseNumber("1.5") // 1.5, true
//	parseNumber("abc") // 0, false
func parseNumber(s string) (float64, bool) {
	if !isJSONNumber(s) {
		return 0, false
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		// Grammar-valid but out of float64 range (e.g. 1e400): json's number
		// decoder treats ParseFloat's error the same way — not a number.
		return 0, false
	}
	return f, true
}

// isJSONNumber reports whether s is exactly one JSON number per RFC 8259 — the
// grammar encoding/json's scanner accepts: an optional leading '-', an integer
// part that is a lone 0 or 1-9 then digits (no leading zeros), an optional
// '.'-fraction with at least one digit, and an optional e/E exponent. Gating
// strconv.ParseFloat on it keeps parseNumber's accept set identical to
// json.Unmarshal's (no leading '+', no "Inf"/"NaN", no hex, no bare ".5"/"1.").
//
//	isJSONNumber("-2.5e3") // true
//	isJSONNumber("01")     // false (leading zero)
func isJSONNumber(s string) bool {
	n := len(s)
	if n == 0 {
		return false
	}
	i := 0
	if s[i] == '-' {
		i++
	}
	// Integer part: a lone 0, or 1-9 followed by any digits.
	if i >= n {
		return false
	}
	switch {
	case s[i] == '0':
		i++
	case s[i] >= '1' && s[i] <= '9':
		i++
		for i < n && s[i] >= '0' && s[i] <= '9' {
			i++
		}
	default:
		return false
	}
	// Optional fraction: '.' then at least one digit.
	if i < n && s[i] == '.' {
		i++
		if i >= n || s[i] < '0' || s[i] > '9' {
			return false
		}
		for i < n && s[i] >= '0' && s[i] <= '9' {
			i++
		}
	}
	// Optional exponent: e/E, an optional sign, then at least one digit.
	if i < n && (s[i] == 'e' || s[i] == 'E') {
		i++
		if i < n && (s[i] == '+' || s[i] == '-') {
			i++
		}
		if i >= n || s[i] < '0' || s[i] > '9' {
			return false
		}
		for i < n && s[i] >= '0' && s[i] <= '9' {
			i++
		}
	}
	return i == n
}

// skipBalanced consumes a {…} or […] region whose opener was already passed,
// returning the index just past the matching closer. STRING_DELIM runs inside
// are skipped so delimiters of the open/close rune buried in a string don't
// count. Mirrors the object/array balance loops in _parse_gemma4_args, including
// their "delimiter run reaches end" early-out.
//
//	skipBalanced("k: 1} rest", 0, '{', '}') // index just after the '}'
func skipBalanced(s string, i int, open, close byte) int {
	n := len(s)
	dl := len(stringDelim)
	depth := 1
	for i < n && depth > 0 {
		if i+dl <= n && s[i:i+dl] == stringDelim {
			i += dl
			next := indexFrom(s, stringDelim, i)
			if next == -1 {
				return n
			}
			i = next + dl
			continue
		}
		switch s[i] {
		case open:
			depth++
		case close:
			depth--
		}
		i++
	}
	return i
}

// indexFrom is core.Index with a start offset — the offset-aware find SGLang
// relies on (Python's str.find(sub, from)). It returns the absolute index, or
// -1 if not found at or after from.
//
//	indexFrom("aXbX", "X", 2) // 3
func indexFrom(s, sub string, from int) int {
	if from < 0 {
		from = 0
	}
	if from > len(s) {
		return -1
	}
	rel := core.Index(s[from:], sub)
	if rel == -1 {
		return -1
	}
	return from + rel
}

// countByte returns how many times b occurs in s. A 0-alloc scan used to size
// an array slice from its comma count before parsing.
//
//	countByte("a,b,c", ',') // 2
func countByte(s string, b byte) int {
	count := 0
	for i := 0; i < len(s); i++ {
		if s[i] == b {
			count++
		}
	}
	return count
}

// isArgSep reports whether b separates entries/elements (space, comma, newline,
// tab) — the skip set shared by the argument and array loops.
func isArgSep(b byte) bool {
	return b == ' ' || b == ',' || b == '\n' || b == '\t'
}

// isSpace reports whether b is the post-colon whitespace skipped before a value
// (space, newline, tab — not comma, which would be the value itself).
func isSpace(b byte) bool {
	return b == ' ' || b == '\n' || b == '\t'
}

// isValueEnd reports whether b terminates a bare value (',', '}' or ']').
func isValueEnd(b byte) bool {
	return b == ',' || b == '}' || b == ']'
}
