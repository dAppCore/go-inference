// SPDX-Licence-Identifier: EUPL-1.2

// filestore hand-rolled JSON: record-meta append encoder and the extractRecordURI reader/walker.
package filestore

import core "dappco.re/go"

// appendJSONField appends a "key":"value" pair (prefixed by a comma
// when not the first field) to buf. Key is ASCII-only and not
// escaped — recordMeta keys are compile-time constants.
func appendJSONField(buf []byte, key, value string, first bool) []byte {
	if !first {
		buf = append(buf, ',')
	}
	buf = append(buf, '"')
	buf = append(buf, key...)
	buf = append(buf, '"', ':')
	return appendJSONString(buf, value)
}

// appendJSONString appends a JSON-encoded string to buf — opening
// quote, escaped body, closing quote. Escapes match the subset
// recognised by extractRecordURI's jsonUnescape walker: \" \\ \b
// \f \n \r \t for the canonical mnemonic forms and \u00XX for
// other control chars (< 0x20). All bytes ≥ 0x20 outside the
// quote / backslash pair pass through verbatim — encoding/json's
// default also escapes <, >, & for HTML safety but the read path
// does not, and the on-disk record is not consumed by HTML
// contexts.
//
// The body walk batches runs of non-escape bytes into a single
// append per span, so a typical URI / Title / Kind value (no
// escapes) collapses to one append-string call rather than N
// append-byte calls. encoding/json's own writer emits the no-
// escape path the same way; the per-byte loop here was an artefact
// of the original simple shape.
func appendJSONString(buf []byte, s string) []byte {
	buf = append(buf, '"')
	start := 0
	for i := 0; i < len(s); i++ {
		c := s[i]
		// Fast-path predicate: any byte ≥ 0x20 that is neither '"'
		// nor '\\' passes through verbatim. The boolean short-
		// circuits left-to-right and the compiler emits two CMPs
		// + AND, cheaper than the previous per-byte switch dispatch.
		if c >= 0x20 && c != '"' && c != '\\' {
			continue
		}
		// Flush the verbatim span up to but not including the
		// escape byte. The span is empty on the first escape at
		// position 0; append-zero-length is a no-op.
		if start < i {
			buf = append(buf, s[start:i]...)
		}
		switch c {
		case '"':
			buf = append(buf, '\\', '"')
		case '\\':
			buf = append(buf, '\\', '\\')
		case '\b':
			buf = append(buf, '\\', 'b')
		case '\f':
			buf = append(buf, '\\', 'f')
		case '\n':
			buf = append(buf, '\\', 'n')
		case '\r':
			buf = append(buf, '\\', 'r')
		case '\t':
			buf = append(buf, '\\', 't')
		default:
			// c < 0x20 and not one of the mnemonic escapes — emit
			// \u00XX. Hex digits emitted lowercase to match the
			// jsonUnescape reader and encoding/json output.
			buf = append(buf, '\\', 'u', '0', '0', hexChar(c>>4), hexChar(c&0x0f))
		}
		start = i + 1
	}
	if start < len(s) {
		buf = append(buf, s[start:]...)
	}
	return append(buf, '"')
}

// hexChar returns the ASCII hex digit for the low nibble of v.
func hexChar(v byte) byte {
	v &= 0x0f
	if v < 10 {
		return '0' + v
	}
	return 'a' + (v - 10)
}

// extractRecordURI walks data as a top-level JSON object and returns
// the value of the "uri" key as a string, or "" if absent. The walker
// fully traverses the object (including nested arrays / objects) so
// any structural corruption — unbalanced braces, truncated value,
// trailing garbage — surfaces as an error. This replaces a full
// json.Unmarshal into recordMeta for the rebuildIndex hot path,
// dropping ~6 allocs per record at 10k scale (Tags map, Labels slice,
// Title/Kind/Track string copies). The "uri" field is encoded by
// json.Marshal of a string — URLs do not require escapes in
// practice, so the fast path returns a direct slice-to-string copy;
// the rare-but-valid escape path is handled by jsonUnescape.
func extractRecordURI(data []byte) (string, error) {
	i, err := jsonSkipWS(data, 0)
	if err != nil {
		return "", err
	}
	if data[i] != '{' {
		return "", core.NewError("state file store metadata is not a JSON object")
	}
	i++
	uri := ""
	uriSeen := false
	first := true
	for {
		i, err = jsonSkipWS(data, i)
		if err != nil {
			return "", err
		}
		if data[i] == '}' {
			i++
			break
		}
		if !first {
			if data[i] != ',' {
				return "", core.NewError("state file store metadata is missing comma")
			}
			i++
			i, err = jsonSkipWS(data, i)
			if err != nil {
				return "", err
			}
		}
		first = false
		if data[i] != '"' {
			return "", core.NewError("state file store metadata key is not a string")
		}
		keyStart := i + 1
		keyEnd, err := jsonSkipString(data, i)
		if err != nil {
			return "", err
		}
		i = keyEnd
		i, err = jsonSkipWS(data, i)
		if err != nil {
			return "", err
		}
		if data[i] != ':' {
			return "", core.NewError("state file store metadata is missing colon")
		}
		i++
		i, err = jsonSkipWS(data, i)
		if err != nil {
			return "", err
		}
		isURI := !uriSeen && keyEnd-1-keyStart == 3 &&
			data[keyStart] == 'u' && data[keyStart+1] == 'r' && data[keyStart+2] == 'i'
		if isURI {
			if data[i] != '"' {
				return "", core.NewError("state file store uri is not a string")
			}
			value, end, err := jsonReadString(data, i)
			if err != nil {
				return "", err
			}
			uri = value
			uriSeen = true
			i = end
		} else {
			end, err := jsonSkipValue(data, i)
			if err != nil {
				return "", err
			}
			i = end
		}
	}
	// Validate no trailing garbage beyond whitespace.
	for i < len(data) {
		c := data[i]
		if c != ' ' && c != '\t' && c != '\n' && c != '\r' {
			return "", core.NewError("state file store metadata has trailing data")
		}
		i++
	}
	return uri, nil
}

// jsonSkipWS advances past JSON whitespace, returning the first
// non-whitespace index or an error if end-of-data is hit. The caller
// uses the returned index to read the next significant byte.
func jsonSkipWS(data []byte, i int) (int, error) {
	for i < len(data) {
		c := data[i]
		if c != ' ' && c != '\t' && c != '\n' && c != '\r' {
			return i, nil
		}
		i++
	}
	return i, core.NewError("state file store metadata is truncated")
}

// jsonSkipString advances past a JSON string starting at data[i]
// (which must be '"') and returns the index after the closing quote.
// Handles escape sequences but does not decode them.
func jsonSkipString(data []byte, i int) (int, error) {
	if i >= len(data) || data[i] != '"' {
		return i, core.NewError("state file store metadata expects string")
	}
	i++
	for i < len(data) {
		c := data[i]
		if c == '\\' {
			if i+1 >= len(data) {
				return i, core.NewError("state file store metadata has trailing escape")
			}
			// One-byte escapes (\" \\ \/ \b \f \n \r \t) or \uXXXX —
			// either way the next single byte cannot terminate the
			// string and the wider \uXXXX is bounded by the closing
			// quote check on later iterations.
			i += 2
			continue
		}
		if c == '"' {
			return i + 1, nil
		}
		i++
	}
	return i, core.NewError("state file store metadata string is unterminated")
}

// jsonReadString reads a JSON string at data[i] (which must be '"')
// and returns its decoded value plus the index after the closing
// quote. Fast path: no escapes → direct string copy of the byte
// slice. Slow path: presence of an escape forces a per-byte decode
// into a fresh buffer. Used only for the "uri" field, where escapes
// are extremely rare in practice (URLs).
func jsonReadString(data []byte, i int) (string, int, error) {
	if i >= len(data) || data[i] != '"' {
		return "", i, core.NewError("state file store metadata expects string")
	}
	start := i + 1
	j := start
	hasEscape := false
	for j < len(data) {
		c := data[j]
		if c == '\\' {
			hasEscape = true
			if j+1 >= len(data) {
				return "", j, core.NewError("state file store metadata has trailing escape")
			}
			j += 2
			continue
		}
		if c == '"' {
			if !hasEscape {
				return string(data[start:j]), j + 1, nil
			}
			decoded, err := jsonUnescape(data[start:j])
			if err != nil {
				return "", j, err
			}
			return decoded, j + 1, nil
		}
		j++
	}
	return "", j, core.NewError("state file store metadata string is unterminated")
}

// jsonUnescape decodes the contents of a JSON string (without
// surrounding quotes) that contains at least one backslash escape.
// Handles the six single-byte escapes and \uXXXX (no surrogate-pair
// decoding — surrogate halves pass through as their raw UTF-8
// encoding, which is what encoding/json itself emits for unpaired
// surrogates). Allocated once per uri-with-escape; URIs never have
// escapes in observed corpora, so this is the cold path.
func jsonUnescape(src []byte) (string, error) {
	out := make([]byte, 0, len(src))
	for i := 0; i < len(src); i++ {
		c := src[i]
		if c != '\\' {
			out = append(out, c)
			continue
		}
		if i+1 >= len(src) {
			return "", core.NewError("state file store metadata has trailing escape")
		}
		i++
		switch src[i] {
		case '"', '\\', '/':
			out = append(out, src[i])
		case 'b':
			out = append(out, '\b')
		case 'f':
			out = append(out, '\f')
		case 'n':
			out = append(out, '\n')
		case 'r':
			out = append(out, '\r')
		case 't':
			out = append(out, '\t')
		case 'u':
			if i+4 >= len(src) {
				return "", core.NewError("state file store metadata has short \\u escape")
			}
			var r rune
			for k := 1; k <= 4; k++ {
				h := src[i+k]
				var v byte
				switch {
				case h >= '0' && h <= '9':
					v = h - '0'
				case h >= 'a' && h <= 'f':
					v = h - 'a' + 10
				case h >= 'A' && h <= 'F':
					v = h - 'A' + 10
				default:
					return "", core.NewError("state file store metadata has invalid \\u escape")
				}
				r = r<<4 | rune(v)
			}
			i += 4
			// Emit r as UTF-8. Unpaired surrogates pass through as
			// their replacement encoding — sufficient for the URI
			// field which is ASCII in every observed corpus.
			switch {
			case r < 0x80:
				out = append(out, byte(r))
			case r < 0x800:
				out = append(out, byte(0xC0|r>>6), byte(0x80|r&0x3F))
			case r < 0x10000:
				out = append(out, byte(0xE0|r>>12), byte(0x80|(r>>6)&0x3F), byte(0x80|r&0x3F))
			default:
				out = append(out, byte(0xF0|r>>18), byte(0x80|(r>>12)&0x3F), byte(0x80|(r>>6)&0x3F), byte(0x80|r&0x3F))
			}
		default:
			return "", core.NewError("state file store metadata has unknown escape")
		}
	}
	return string(out), nil
}

// jsonSkipValue advances past a single JSON value (string, number,
// boolean, null, object, array) starting at data[i] and returns the
// index of the first byte after the value. The full traversal is
// what gives rebuildIndex its structural-corruption guarantee
// without forcing the whole metadata blob through json.Unmarshal.
func jsonSkipValue(data []byte, i int) (int, error) {
	if i >= len(data) {
		return i, core.NewError("state file store metadata is truncated")
	}
	c := data[i]
	switch {
	case c == '"':
		return jsonSkipString(data, i)
	case c == '{' || c == '[':
		open := c
		var closeByte byte
		if open == '{' {
			closeByte = '}'
		} else {
			closeByte = ']'
		}
		depth := 1
		i++
		for i < len(data) && depth > 0 {
			cc := data[i]
			switch cc {
			case '"':
				end, err := jsonSkipString(data, i)
				if err != nil {
					return i, err
				}
				i = end
			case '{', '[':
				depth++
				i++
			case '}', ']':
				if cc == closeByte {
					depth--
					i++
					continue
				}
				if (open == '{' && cc == ']') || (open == '[' && cc == '}') {
					return i, core.NewError("state file store metadata has mismatched bracket")
				}
				depth--
				i++
			default:
				i++
			}
		}
		if depth != 0 {
			return i, core.NewError("state file store metadata is unbalanced")
		}
		return i, nil
	case c == 't':
		if i+4 > len(data) || data[i+1] != 'r' || data[i+2] != 'u' || data[i+3] != 'e' {
			return i, core.NewError("state file store metadata expects true")
		}
		return i + 4, nil
	case c == 'f':
		if i+5 > len(data) || data[i+1] != 'a' || data[i+2] != 'l' || data[i+3] != 's' || data[i+4] != 'e' {
			return i, core.NewError("state file store metadata expects false")
		}
		return i + 5, nil
	case c == 'n':
		if i+4 > len(data) || data[i+1] != 'u' || data[i+2] != 'l' || data[i+3] != 'l' {
			return i, core.NewError("state file store metadata expects null")
		}
		return i + 4, nil
	case c == '-' || (c >= '0' && c <= '9'):
		// Number — consume digits, sign, dot, exponent. Loose but
		// correct enough for structural validation; json.Marshal
		// emits canonical numbers so the surface is constrained.
		j := i
		if data[j] == '-' {
			j++
		}
		for j < len(data) {
			b := data[j]
			if (b >= '0' && b <= '9') || b == '.' || b == 'e' || b == 'E' || b == '+' || b == '-' {
				j++
				continue
			}
			break
		}
		if j == i {
			return i, core.NewError("state file store metadata has empty number")
		}
		return j, nil
	default:
		return i, core.NewError("state file store metadata has invalid value")
	}
}
