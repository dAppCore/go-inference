// SPDX-Licence-Identifier: EUPL-1.2

// Hand-rolled JSON-encoding primitives shared by the anthropic
// adapter's hot-path encoders. The encoding/json reflect path
// allocates an encoder state machine + a grow-doubled output buffer
// on every Marshal call. Adapter encoders that fire per-request
// (MessageRequest, MessageResponse) and per-streamed-emission pay
// that two-allocation floor before any per-field cost.
//
// These helpers compose into per-shape encoders (MessageResponse,
// MessageRequest) that land at a single buffer allocation per call —
// same minimax lift as state/filestore's encodeRecordMeta (W8-D),
// core.ParseHeaderRefs (W8-I/K), and openai's appendJSONString
// (W9-D). Kept private to the anthropic package per the W9-D/W9-E
// lane-isolation rule — a future follow-up lane will lift to a
// shared helper once both adapters land.
//
// The output is valid JSON and parseable both by encoding/json
// (round-trips into the same Go types) and by any naive JSON walker.
// All callers share the same escape contract — quote, backslash,
// b/f/n/r/t mnemonics, and \u00XX for other control chars below 0x20.
// Bytes >= 0x20 outside the quote/backslash pair pass through
// verbatim; encoding/json's default also escapes <, >, & for HTML
// safety but the anthropic adapter does not emit into HTML contexts.
package anthropic

import "strconv"

// appendJSONString appends a JSON-encoded string to buf — opening
// quote, escaped body, closing quote. Caller is responsible for
// providing the surrounding context (key, comma, etc).
//
//	buf = appendJSONString(buf, "answer")  // -> "answer"
//
// Escapes: \" \\ \b \f \n \r \t for the mnemonic forms and \u00XX
// for other bytes < 0x20. All other bytes pass through.
//
// Fast path: scan for any character requiring an escape. Anthropic
// message bodies overwhelmingly contain neither — once a hot prefix
// passes the scan, we copy the whole string verbatim in one append.
// On the rare escape-bearing path we drop back to the byte-by-byte
// walk starting from the first hit.
func appendJSONString(buf []byte, s string) []byte {
	buf = append(buf, '"')
	// Scan for the first byte that needs escaping. \" \\ and any
	// byte < 0x20 all require special handling; everything else
	// passes through.
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c == '"' || c == '\\' || c < 0x20 {
			// Bulk-copy the safe prefix, then walk the rest.
			buf = append(buf, s[:i]...)
			return appendJSONStringEscaped(buf, s[i:])
		}
	}
	// No escapes — single bulk append covers the whole body.
	buf = append(buf, s...)
	return append(buf, '"')
}

// appendJSONStringEscaped completes a string already opened with `"`
// and that has at least one byte requiring escape treatment in s[0].
// Internal helper for appendJSONString — separated out to keep the
// fast-path inlineable.
func appendJSONStringEscaped(buf []byte, s string) []byte {
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c == '"':
			buf = append(buf, '\\', '"')
		case c == '\\':
			buf = append(buf, '\\', '\\')
		case c == '\b':
			buf = append(buf, '\\', 'b')
		case c == '\f':
			buf = append(buf, '\\', 'f')
		case c == '\n':
			buf = append(buf, '\\', 'n')
		case c == '\r':
			buf = append(buf, '\\', 'r')
		case c == '\t':
			buf = append(buf, '\\', 't')
		case c < 0x20:
			buf = append(buf, '\\', 'u', '0', '0', hexChar(c>>4), hexChar(c&0x0f))
		default:
			buf = append(buf, c)
		}
	}
	return append(buf, '"')
}

// appendStringField appends a `"key":"value"` pair (optionally
// prefixed with a leading comma) to buf. Key is treated as an ASCII
// literal — anthropic-schema keys carry no escapes by construction.
//
//	buf = appendStringField(buf, "model", req.Model, false)
//	buf = appendStringField(buf, "id", id, true)  // leading comma
func appendStringField(buf []byte, key, value string, leadingComma bool) []byte {
	if leadingComma {
		buf = append(buf, ',')
	}
	buf = append(buf, '"')
	buf = append(buf, key...)
	buf = append(buf, '"', ':')
	return appendJSONString(buf, value)
}

// appendIntField appends a `"key":N` pair (optionally prefixed with a
// leading comma) where N is the base-10 representation of value.
//
//	buf = appendIntField(buf, "max_tokens", 1024, true)
func appendIntField(buf []byte, key string, value int, leadingComma bool) []byte {
	if leadingComma {
		buf = append(buf, ',')
	}
	buf = append(buf, '"')
	buf = append(buf, key...)
	buf = append(buf, '"', ':')
	return strconv.AppendInt(buf, int64(value), 10)
}

// appendFloat32Field appends a `"key":F` pair where F is rendered in
// the same 'g' format encoding/json emits for float32 (bitSize 32).
//
//	buf = appendFloat32Field(buf, "temperature", *req.Temperature, true)
func appendFloat32Field(buf []byte, key string, value float32, leadingComma bool) []byte {
	if leadingComma {
		buf = append(buf, ',')
	}
	buf = append(buf, '"')
	buf = append(buf, key...)
	buf = append(buf, '"', ':')
	return strconv.AppendFloat(buf, float64(value), 'g', -1, 32)
}

// appendBoolField appends a `"key":true|false` pair.
//
//	buf = appendBoolField(buf, "stream", req.Stream, true)
func appendBoolField(buf []byte, key string, value bool, leadingComma bool) []byte {
	if leadingComma {
		buf = append(buf, ',')
	}
	buf = append(buf, '"')
	buf = append(buf, key...)
	buf = append(buf, '"', ':')
	if value {
		return append(buf, 't', 'r', 'u', 'e')
	}
	return append(buf, 'f', 'a', 'l', 's', 'e')
}

// hexChar returns the ASCII hex digit for the low nibble of v.
func hexChar(v byte) byte {
	v &= 0x0f
	if v < 10 {
		return '0' + v
	}
	return 'a' + (v - 10)
}
