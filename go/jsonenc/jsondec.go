// SPDX-Licence-Identifier: EUPL-1.2

// JSON-decoding primitives shared by the inference adapter
// UnmarshalJSON hot paths. The encoding/json reflect path allocates
// an encoder state machine, per-field reflect.Value boxing, and a
// per-string copy on every Unmarshal call — each adapter request
// decoder pays that floor.
//
// Provenance: lifted in W11-B from openai/jsondec.go which shipped
// in W10-M (StopList / EmbeddingInput single-pass walker). The set
// of primitives mirrors the encode side of jsonenc — ParseJSONString
// is the inverse of AppendJSONString and shares the same escape
// contract. Hand-rolled per-type field walkers (anthropic /
// openai / ollama Unmarshal*Request) call directly into these.
//
// All primitives parse the JSON spec across every branch:
//   - Whitespace: space, tab, CR, LF.
//   - Strings: \" \\ \/ \b \f \n \r \t \uXXXX (UTF-8 re-encoded).
//   - Numbers: int64 + float64 with the same shape strconv.ParseFloat
//     accepts.
//   - Literals: true / false / null.
//
// Output matches what encoding/json.Unmarshal would have produced
// for the same input.

package jsonenc

import "errors"

// ErrInvalidJSON is the sentinel returned for malformed input.
// Call sites wrap into typed result errors as appropriate.
var ErrInvalidJSON = errors.New("invalid JSON")

// ParseJSONStringList walks data as either a JSON string (e.g.
// `"END"`) or an array of JSON strings (e.g. `["END","</s>"]`) and
// returns a []string with the inner values unescaped.
//
// The "null" literal returns (nil, nil). Empty or invalid data
// returns ErrInvalidJSON; otherwise the first non-whitespace byte
// determines the shape.
//
//	stops, err := jsonenc.ParseJSONStringList([]byte(`["a","b"]`))
//	// stops == []string{"a","b"}
//
//	stops, err := jsonenc.ParseJSONStringList([]byte(`"END"`))
//	// stops == []string{"END"}
func ParseJSONStringList(data []byte) ([]string, error) {
	i := SkipJSONWhitespace(data, 0)
	if i >= len(data) {
		return nil, ErrInvalidJSON
	}
	c := data[i]
	if c == 'n' {
		// Possible "null" literal.
		if i+4 <= len(data) && data[i+1] == 'u' && data[i+2] == 'l' && data[i+3] == 'l' {
			return nil, nil
		}
		return nil, ErrInvalidJSON
	}
	if c == '"' {
		s, _, err := ParseJSONString(data, i)
		if err != nil {
			return nil, err
		}
		return []string{s}, nil
	}
	if c == '[' {
		return parseJSONStringArray(data, i+1)
	}
	return nil, ErrInvalidJSON
}

// parseJSONStringArray walks data from position i (just past the '[')
// and returns the inner array of strings.
func parseJSONStringArray(data []byte, i int) ([]string, error) {
	out := []string(nil)
	// Empty-array fast path.
	j := SkipJSONWhitespace(data, i)
	if j < len(data) && data[j] == ']' {
		return out, nil
	}
	for {
		i = SkipJSONWhitespace(data, i)
		if i >= len(data) {
			return nil, ErrInvalidJSON
		}
		if data[i] != '"' {
			return nil, ErrInvalidJSON
		}
		s, next, err := ParseJSONString(data, i)
		if err != nil {
			return nil, err
		}
		out = append(out, s)
		i = SkipJSONWhitespace(data, next)
		if i >= len(data) {
			return nil, ErrInvalidJSON
		}
		switch data[i] {
		case ',':
			i++
		case ']':
			return out, nil
		default:
			return nil, ErrInvalidJSON
		}
	}
}

// ParseJSONString walks a JSON string starting at data[i] (which must
// be '"') and returns the unescaped string + the index one past the
// closing '"'.
//
// The fast path (no escapes) returns a string copy of the slice
// range directly via Go's built-in string conversion. The escape
// path walks byte-by-byte and re-decodes \" \\ \b \f \n \r \t / \uXXXX
// escapes. Most adapter wire strings carry no escapes — the fast
// path is the common case.
//
//	value, next, err := jsonenc.ParseJSONString(data, i)
func ParseJSONString(data []byte, i int) (string, int, error) {
	if i >= len(data) || data[i] != '"' {
		return "", i, ErrInvalidJSON
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
			return "", j, ErrInvalidJSON
		}
	}
	return "", i, ErrInvalidJSON
}

// ParseJSONStringRaw is the no-copy variant of ParseJSONString —
// returns a []byte slice into data when no escapes are present, or
// allocates only when an escape forces a copy. Caller MUST treat
// the returned slice as read-only and assignable to a string via
// the standard byte-to-string conversion when persistence is needed.
//
// Hot use case: anthropic/openai field dispatch where the matched
// key path can clone the underlying string in one allocation rather
// than two.
func ParseJSONStringRaw(data []byte, i int) ([]byte, int, error) {
	if i >= len(data) || data[i] != '"' {
		return nil, i, ErrInvalidJSON
	}
	start := i + 1
	for j := start; j < len(data); j++ {
		c := data[j]
		if c == '"' {
			return data[start:j], j + 1, nil
		}
		if c == '\\' {
			s, next, err := parseJSONStringEscaped(data, start, j)
			if err != nil {
				return nil, next, err
			}
			return []byte(s), next, nil
		}
		if c < 0x20 {
			return nil, j, ErrInvalidJSON
		}
	}
	return nil, i, ErrInvalidJSON
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
				return "", i, ErrInvalidJSON
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
					return "", i, ErrInvalidJSON
				}
				cp, ok := parseJSONUnicodeEscape(data[i+2 : i+6])
				if !ok {
					return "", i, ErrInvalidJSON
				}
				// UTF-8 encode the codepoint.
				buf = appendUTF8(buf, cp)
				i += 6
				continue
			default:
				return "", i, ErrInvalidJSON
			}
			i += 2
			continue
		}
		if c < 0x20 {
			return "", i, ErrInvalidJSON
		}
		buf = append(buf, c)
		i++
	}
	return "", firstEscape, ErrInvalidJSON
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

// SkipJSONWhitespace advances i past JSON whitespace bytes — space,
// tab, CR, LF — and returns the new position.
//
//	i := jsonenc.SkipJSONWhitespace(data, 0)
func SkipJSONWhitespace(data []byte, i int) int {
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

// ParseJSONInt walks a JSON integer (possibly signed) at data[i]
// and returns the parsed int64 + the index one past the last digit.
// Accepts the same shape encoding/json accepts for an integer field
// (no leading '+', no leading zeros except the lone '0').
//
//	n, next, err := jsonenc.ParseJSONInt(data, i)
func ParseJSONInt(data []byte, i int) (int64, int, error) {
	if i >= len(data) {
		return 0, i, ErrInvalidJSON
	}
	start := i
	neg := false
	if data[i] == '-' {
		neg = true
		i++
		if i >= len(data) {
			return 0, i, ErrInvalidJSON
		}
	}
	c := data[i]
	if c < '0' || c > '9' {
		return 0, i, ErrInvalidJSON
	}
	var n int64
	for i < len(data) {
		c := data[i]
		if c < '0' || c > '9' {
			break
		}
		n = n*10 + int64(c-'0')
		i++
	}
	if neg {
		n = -n
	}
	if i == start {
		return 0, i, ErrInvalidJSON
	}
	return n, i, nil
}

// ParseJSONBool walks the literal `true` or `false` at data[i] and
// returns the value + the index one past the literal.
//
//	v, next, err := jsonenc.ParseJSONBool(data, i)
func ParseJSONBool(data []byte, i int) (bool, int, error) {
	if i+4 <= len(data) && data[i] == 't' && data[i+1] == 'r' && data[i+2] == 'u' && data[i+3] == 'e' {
		return true, i + 4, nil
	}
	if i+5 <= len(data) && data[i] == 'f' && data[i+1] == 'a' && data[i+2] == 'l' && data[i+3] == 's' && data[i+4] == 'e' {
		return false, i + 5, nil
	}
	return false, i, ErrInvalidJSON
}

// IsJSONNull reports whether data[i:] starts with the `null` literal.
// Does NOT advance i — the caller picks the new index based on
// whether they care to consume it.
//
//	if jsonenc.IsJSONNull(data, i) { i += 4; continue }
func IsJSONNull(data []byte, i int) bool {
	return i+4 <= len(data) && data[i] == 'n' && data[i+1] == 'u' && data[i+2] == 'l' && data[i+3] == 'l'
}

// SkipJSONValue walks one complete JSON value at data[i] (object,
// array, string, number, true, false, null) and returns the index
// one past the value. Caller uses it to skip an unknown / ignored
// field during single-pass dispatch.
//
//	next, err := jsonenc.SkipJSONValue(data, i)
func SkipJSONValue(data []byte, i int) (int, error) {
	i = SkipJSONWhitespace(data, i)
	if i >= len(data) {
		return i, ErrInvalidJSON
	}
	switch data[i] {
	case '{':
		return skipJSONObject(data, i+1)
	case '[':
		return skipJSONArray(data, i+1)
	case '"':
		_, next, err := ParseJSONString(data, i)
		return next, err
	case 't', 'f':
		_, next, err := ParseJSONBool(data, i)
		return next, err
	case 'n':
		if IsJSONNull(data, i) {
			return i + 4, nil
		}
		return i, ErrInvalidJSON
	}
	return skipJSONNumber(data, i)
}

// skipJSONObject skips through the object body at data[i:] starting
// just past the '{'. Returns the index one past the closing '}'.
func skipJSONObject(data []byte, i int) (int, error) {
	i = SkipJSONWhitespace(data, i)
	if i < len(data) && data[i] == '}' {
		return i + 1, nil
	}
	for {
		i = SkipJSONWhitespace(data, i)
		if i >= len(data) || data[i] != '"' {
			return i, ErrInvalidJSON
		}
		_, next, err := ParseJSONString(data, i)
		if err != nil {
			return next, err
		}
		i = SkipJSONWhitespace(data, next)
		if i >= len(data) || data[i] != ':' {
			return i, ErrInvalidJSON
		}
		i++
		next, err = SkipJSONValue(data, i)
		if err != nil {
			return next, err
		}
		i = SkipJSONWhitespace(data, next)
		if i >= len(data) {
			return i, ErrInvalidJSON
		}
		if data[i] == ',' {
			i++
			continue
		}
		if data[i] == '}' {
			return i + 1, nil
		}
		return i, ErrInvalidJSON
	}
}

// skipJSONArray skips through the array body at data[i:] starting
// just past the '['. Returns the index one past the closing ']'.
func skipJSONArray(data []byte, i int) (int, error) {
	i = SkipJSONWhitespace(data, i)
	if i < len(data) && data[i] == ']' {
		return i + 1, nil
	}
	for {
		next, err := SkipJSONValue(data, i)
		if err != nil {
			return next, err
		}
		i = SkipJSONWhitespace(data, next)
		if i >= len(data) {
			return i, ErrInvalidJSON
		}
		if data[i] == ',' {
			i++
			continue
		}
		if data[i] == ']' {
			return i + 1, nil
		}
		return i, ErrInvalidJSON
	}
}

// skipJSONNumber walks a JSON number (possibly signed, possibly
// containing '.' / 'e' / 'E') at data[i] and returns the index one
// past the last byte.
func skipJSONNumber(data []byte, i int) (int, error) {
	start := i
	if i < len(data) && data[i] == '-' {
		i++
	}
	for i < len(data) {
		c := data[i]
		if (c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-' {
			i++
			continue
		}
		break
	}
	if i == start {
		return i, ErrInvalidJSON
	}
	return i, nil
}

// MatchObjectStart skips whitespace and asserts data[i] == '{',
// returning the index one past the opening brace.
//
//	i, err := jsonenc.MatchObjectStart(data, 0)
func MatchObjectStart(data []byte, i int) (int, error) {
	i = SkipJSONWhitespace(data, i)
	if i >= len(data) || data[i] != '{' {
		return i, ErrInvalidJSON
	}
	return i + 1, nil
}

// MatchArrayStart skips whitespace and asserts data[i] == '[',
// returning the index one past the opening bracket.
//
//	i, err := jsonenc.MatchArrayStart(data, 0)
func MatchArrayStart(data []byte, i int) (int, error) {
	i = SkipJSONWhitespace(data, i)
	if i >= len(data) || data[i] != '[' {
		return i, ErrInvalidJSON
	}
	return i + 1, nil
}
