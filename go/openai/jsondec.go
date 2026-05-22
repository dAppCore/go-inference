// SPDX-Licence-Identifier: EUPL-1.2

// Hand-rolled JSON-decoding primitives for the openai adapter's
// hot-path variant-shape unmarshallers.
//
// Some openai request fields accept either a JSON string or an array
// of strings (StopList, EmbeddingInput) — the canonical UnmarshalJSON
// shape dispatches by peeking the first non-whitespace byte and then
// recursively calls encoding/json.Unmarshal on the inner value. Each
// recursive call pays the encoder-state-machine alloc + the per-element
// string allocation cost. For a 3-stop array that's 9 allocations /
// 336 bytes per chat-completion request.
//
// parseJSONStringList walks the same string-or-array variant in a
// single pass — produces []string with one or two allocations
// regardless of element count.

package openai

import "errors"

// errInvalidJSONString is the sentinel returned for malformed string
// content in the parseJSONStringList walker. Wrapped at call sites
// via resultError-equivalent shape.
var errInvalidJSONString = errors.New("invalid JSON string content")

// parseJSONStringList walks data as either a JSON string (e.g.
// "END") or an array of JSON strings (e.g. ["END","</s>"]) and
// returns a []string with the inner values unescaped.
//
// The "null" literal returns (nil, nil). Empty or invalid data
// returns an error; otherwise the first non-whitespace byte
// determines the shape.
//
//	stops, err := parseJSONStringList([]byte(`["a","b"]`))
//	// stops == []string{"a","b"}
//
//	stops, err := parseJSONStringList([]byte(`"END"`))
//	// stops == []string{"END"}
func parseJSONStringList(data []byte) ([]string, error) {
	i := skipJSONWhitespace(data, 0)
	if i >= len(data) {
		return nil, errInvalidJSONString
	}
	c := data[i]
	if c == 'n' {
		// Possible "null" literal.
		if i+4 <= len(data) && data[i+1] == 'u' && data[i+2] == 'l' && data[i+3] == 'l' {
			return nil, nil
		}
		return nil, errInvalidJSONString
	}
	if c == '"' {
		s, _, err := parseJSONString(data, i)
		if err != nil {
			return nil, err
		}
		return []string{s}, nil
	}
	if c == '[' {
		return parseJSONStringArray(data, i+1)
	}
	return nil, errInvalidJSONString
}

// parseJSONStringArray walks data from position i (just past the '[')
// and returns the inner array of strings.
func parseJSONStringArray(data []byte, i int) ([]string, error) {
	out := []string(nil)
	// Empty-array fast path.
	j := skipJSONWhitespace(data, i)
	if j < len(data) && data[j] == ']' {
		return out, nil
	}
	for {
		i = skipJSONWhitespace(data, i)
		if i >= len(data) {
			return nil, errInvalidJSONString
		}
		if data[i] != '"' {
			return nil, errInvalidJSONString
		}
		s, next, err := parseJSONString(data, i)
		if err != nil {
			return nil, err
		}
		out = append(out, s)
		i = skipJSONWhitespace(data, next)
		if i >= len(data) {
			return nil, errInvalidJSONString
		}
		switch data[i] {
		case ',':
			i++
		case ']':
			return out, nil
		default:
			return nil, errInvalidJSONString
		}
	}
}

// parseJSONString walks a JSON string starting at data[i] (which must
// be '"') and returns the unescaped string + the index one past the
// closing '"'.
//
// The fast path (no escapes) returns a string copy of the slice
// range directly via Go's built-in string conversion. The escape
// path walks byte-by-byte and re-decodes \" \\ \b \f \n \r \t / \uXXXX
// escapes. Most chat-completion stop sequences carry no escapes —
// the fast path is the common case.
func parseJSONString(data []byte, i int) (string, int, error) {
	if i >= len(data) || data[i] != '"' {
		return "", i, errInvalidJSONString
	}
	start := i + 1
	for j := start; j < len(data); j++ {
		c := data[j]
		if c == '"' {
			return string(data[start:j]), j + 1, nil
		}
		if c == '\\' {
			return parseJSONStringEscaped(data, start, j)
		}
		if c < 0x20 {
			return "", j, errInvalidJSONString
		}
	}
	return "", i, errInvalidJSONString
}

// parseJSONStringEscaped is the slow path for strings containing
// backslash escapes. Walks the remainder character-by-character,
// emitting into a backing buffer with appended decoded bytes.
func parseJSONStringEscaped(data []byte, start, firstEscape int) (string, int, error) {
	buf := make([]byte, 0, len(data)-start)
	buf = append(buf, data[start:firstEscape]...)
	for i := firstEscape; i < len(data); {
		c := data[i]
		if c == '"' {
			return string(buf), i + 1, nil
		}
		if c == '\\' {
			if i+1 >= len(data) {
				return "", i, errInvalidJSONString
			}
			esc := data[i+1]
			switch esc {
			case '"':
				buf = append(buf, '"')
			case '\\':
				buf = append(buf, '\\')
			case '/':
				buf = append(buf, '/')
			case 'b':
				buf = append(buf, '\b')
			case 'f':
				buf = append(buf, '\f')
			case 'n':
				buf = append(buf, '\n')
			case 'r':
				buf = append(buf, '\r')
			case 't':
				buf = append(buf, '\t')
			case 'u':
				if i+6 > len(data) {
					return "", i, errInvalidJSONString
				}
				cp, ok := parseJSONUnicodeEscape(data[i+2 : i+6])
				if !ok {
					return "", i, errInvalidJSONString
				}
				// UTF-8 encode the codepoint.
				buf = appendUTF8(buf, cp)
				i += 6
				continue
			default:
				return "", i, errInvalidJSONString
			}
			i += 2
			continue
		}
		if c < 0x20 {
			return "", i, errInvalidJSONString
		}
		buf = append(buf, c)
		i++
	}
	return "", firstEscape, errInvalidJSONString
}

// parseJSONUnicodeEscape decodes a 4-hex-digit codepoint following
// the \u escape prefix.
func parseJSONUnicodeEscape(hex []byte) (rune, bool) {
	if len(hex) != 4 {
		return 0, false
	}
	var cp rune
	for _, b := range hex {
		var v rune
		switch {
		case b >= '0' && b <= '9':
			v = rune(b - '0')
		case b >= 'a' && b <= 'f':
			v = rune(b-'a') + 10
		case b >= 'A' && b <= 'F':
			v = rune(b-'A') + 10
		default:
			return 0, false
		}
		cp = cp<<4 | v
	}
	return cp, true
}

// appendUTF8 appends the UTF-8 encoding of cp to buf.
func appendUTF8(buf []byte, cp rune) []byte {
	switch {
	case cp < 0x80:
		return append(buf, byte(cp))
	case cp < 0x800:
		return append(buf, byte(0xc0|cp>>6), byte(0x80|cp&0x3f))
	case cp < 0x10000:
		return append(buf, byte(0xe0|cp>>12), byte(0x80|(cp>>6)&0x3f), byte(0x80|cp&0x3f))
	default:
		return append(buf, byte(0xf0|cp>>18), byte(0x80|(cp>>12)&0x3f), byte(0x80|(cp>>6)&0x3f), byte(0x80|cp&0x3f))
	}
}

// skipJSONWhitespace advances i past JSON whitespace bytes.
func skipJSONWhitespace(data []byte, i int) int {
	for i < len(data) {
		c := data[i]
		if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
			i++
			continue
		}
		break
	}
	return i
}
