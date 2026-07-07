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

import (
	"errors"
	"strconv"
	"strings"
)

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
	// Empty-array fast path.
	j := SkipJSONWhitespace(data, i)
	if j < len(data) && data[j] == ']' {
		return nil, nil
	}
	out := make([]string, 0, CountJSONArrayElements(data, i))
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
// emitting decoded bytes into a strings.Builder.
//
// The Builder is sized once to the remaining raw length — the decoded
// string can never exceed it, so there is no geometric regrowth — and
// its String() hands the backing array straight to the result with no
// second copy. The earlier make([]byte,0,n)+string(buf) shape paid a
// heap buffer allocation AND a conversion copy (two allocs); this is
// one allocation for the whole decode (AX-11).
func parseJSONStringEscaped(data []byte, start, firstEscape int) (string, int, error) {
	var sb strings.Builder
	sb.Grow(len(data) - start)
	sb.Write(data[start:firstEscape])
	for i := firstEscape; i < len(data); {
		// Bulk-copy the run of plain bytes up to the next escape,
		// closing quote, or control byte — only the escape
		// replacements pay a per-byte write, so a long content body
		// between sparse escapes copies in one Write rather than
		// byte-by-byte.
		runStart := i
		for i < len(data) {
			c := data[i]
			if c == '"' || c == '\\' || c < 0x20 {
				break
			}
			i++
		}
		if i > runStart {
			sb.Write(data[runStart:i])
		}
		if i >= len(data) {
			break
		}
		c := data[i]
		if c == '"' {
			return sb.String(), i + 1, nil
		}
		if c < 0x20 {
			return "", i, ErrInvalidJSON
		}
		// c == '\\' — decode one escape.
		if i+1 >= len(data) {
			return "", i, ErrInvalidJSON
		}
		esc := data[i+1]
		switch esc {
		case '"':
			sb.WriteByte('"')
		case '\\':
			sb.WriteByte('\\')
		case '/':
			sb.WriteByte('/')
		case 'b':
			sb.WriteByte('\b')
		case 'f':
			sb.WriteByte('\f')
		case 'n':
			sb.WriteByte('\n')
		case 'r':
			sb.WriteByte('\r')
		case 't':
			sb.WriteByte('\t')
		case 'u':
			if i+6 > len(data) {
				return "", i, ErrInvalidJSON
			}
			cp, ok := parseJSONUnicodeEscape(data[i+2 : i+6])
			if !ok {
				return "", i, ErrInvalidJSON
			}
			// UTF-8 encode the codepoint.
			writeUTF8(&sb, cp)
			i += 6
			continue
		default:
			return "", i, ErrInvalidJSON
		}
		i += 2
	}
	return "", firstEscape, ErrInvalidJSON
}

// parseJSONUnicodeEscape decodes a 4-hex-digit codepoint following
// the \u escape prefix.
// hex is exactly the 4 bytes after `\u` — the single call site slices data[i+2 : i+6].
func parseJSONUnicodeEscape(hex []byte) (rune, bool) {
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

// writeUTF8 writes the UTF-8 encoding of cp to sb. Byte-for-byte
// identical to the previous appendUTF8 — same nibble arithmetic and
// the same (raw, un-paired) emission for surrogate-range code points;
// only the sink changed from a []byte append to the Builder.
func writeUTF8(sb *strings.Builder, cp rune) {
	switch {
	case cp < 0x80:
		sb.WriteByte(byte(cp))
	case cp < 0x800:
		sb.WriteByte(byte(0xc0 | cp>>6))
		sb.WriteByte(byte(0x80 | cp&0x3f))
	default:
		// cp <= 0xFFFF always: the 4-hex escape is the only producer, and surrogate-range
		// code points emit raw (un-paired) by design — no supplementary-plane branch.
		sb.WriteByte(byte(0xe0 | cp>>12))
		sb.WriteByte(byte(0x80 | (cp>>6)&0x3f))
		sb.WriteByte(byte(0x80 | cp&0x3f))
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
		return SkipJSONString(data, i)
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

// SkipJSONString walks a JSON string at data[i] (which must be '"')
// and returns the index one past the closing '"'. Unlike
// ParseJSONString it does NOT materialise a Go string — callers use
// it when they only need to advance past the value (object-key
// inside a SkipJSONValue path, ignored field, CountJSONArrayElements
// prescan).
//
//	next, err := jsonenc.SkipJSONString(data, i)
func SkipJSONString(data []byte, i int) (int, error) {
	if i >= len(data) || data[i] != '"' {
		return i, ErrInvalidJSON
	}
	for j := i + 1; j < len(data); j++ {
		c := data[j]
		if c == '"' {
			return j + 1, nil
		}
		if c == '\\' {
			// Escape — bump j past the escape body without decoding.
			if j+1 >= len(data) {
				return j, ErrInvalidJSON
			}
			if data[j+1] == 'u' {
				if j+6 > len(data) {
					return j, ErrInvalidJSON
				}
				j += 5
				continue
			}
			j++
			continue
		}
		if c < 0x20 {
			return j, ErrInvalidJSON
		}
	}
	return i, ErrInvalidJSON
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
		next, err := SkipJSONString(data, i)
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

// ParseJSONFloat32 walks a JSON number at data[i] and returns the
// parsed float32 + the index one past the last byte. Accepts the
// same shape encoding/json accepts for a float field (optional
// leading '-', integer, optional fraction, optional exponent).
//
//	v, next, err := jsonenc.ParseJSONFloat32(data, i)
func ParseJSONFloat32(data []byte, i int) (float32, int, error) {
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
		return 0, i, ErrInvalidJSON
	}
	// strconv.ParseFloat with bitSize 32 matches encoding/json's
	// float32 decoder. The string conversion at the strconv boundary
	// is unavoidable — pre-W11-B json.Unmarshal paid the same cost
	// via its own internal walker; the hand-roll wins from skipping
	// reflect overhead, not from defeating the stdlib's float parser.
	v, err := strconv.ParseFloat(string(data[start:i]), 32)
	if err != nil {
		return 0, i, ErrInvalidJSON
	}
	return float32(v), i, nil
}

// ParseJSONFloat64 walks a JSON number at data[i] and returns the
// parsed float64 + the index one past the last byte.
func ParseJSONFloat64(data []byte, i int) (float64, int, error) {
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
		return 0, i, ErrInvalidJSON
	}
	v, err := strconv.ParseFloat(string(data[start:i]), 64)
	if err != nil {
		return 0, i, ErrInvalidJSON
	}
	return v, i, nil
}

// CountJSONArrayElements counts the elements in the JSON array body
// starting at data[i] (just past the '['). Does NOT mutate the
// caller's index — callers use the count only for slice pre-sizing.
//
// Walks each element via SkipJSONValue so it handles nested objects
// / arrays / quoted strings (no naive comma-count footgun). A malformed
// element ends the count early, returning the leading well-formed count —
// the value is a capacity hint, so a partial count stays useful, and the
// caller's subsequent parse re-reports the malformedness.
//
//	count := jsonenc.CountJSONArrayElements(data, i)
//	out := make([]T, 0, count)
func CountJSONArrayElements(data []byte, i int) int {
	i = SkipJSONWhitespace(data, i)
	if i >= len(data) || data[i] == ']' {
		return 0
	}
	count := 0
	for {
		next, err := SkipJSONValue(data, i)
		if err != nil {
			return count
		}
		count++
		i = SkipJSONWhitespace(data, next)
		if i >= len(data) {
			return count
		}
		if data[i] == ',' {
			i = SkipJSONWhitespace(data, i+1)
			continue
		}
		return count
	}
}
