// SPDX-Licence-Identifier: EUPL-1.2

// Hand-rolled JSON-encoding primitives shared by the openai adapter's
// hot-path encoders. The encoding/json reflect path allocates an
// encoder state machine + grow-doubled output buffer per call (~5550 B
// + 4 allocs even for an empty struct, per the W7-G data); each
// adapter encoder that fires per-request or per-streamed-token pays
// that floor.
//
// These helpers compose into per-shape encoders (appendChatMessageDelta,
// appendChatCompletionChunk, etc.) that land at a single buffer
// allocation per call — same minimax lift as state/filestore's
// encodeRecordMeta (W8-D) and core.ParseHeaderRefs (W8-I/K).
//
// The output is valid JSON and parseable both by encoding/json
// (round-trips into the same Go types) and by any naive JSON walker.
// All callers share the same escape contract — quote, backslash,
// b/f/n/r/t mnemonics, and \u00XX for other control chars below 0x20.
// Bytes ≥ 0x20 outside the quote/backslash pair pass through verbatim;
// encoding/json's default also escapes <, >, & for HTML safety but
// the openai adapter does not emit into HTML contexts.
package openai

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
// The common case (no escapes — most chat content) goes through a
// bulk-copy fast path: scan to find the first byte that needs
// escaping, copy [pos, i) in one append, then emit the escape and
// continue. For escape-free strings this collapses to a single
// append(buf, s...). A char-by-char fallback handles strings with
// mixed escapes.
func appendJSONString(buf []byte, s string) []byte {
	buf = append(buf, '"')
	pos := 0
	for i := 0; i < len(s); i++ {
		c := s[i]
		// Fast path: byte requires no escaping — keep scanning.
		if c >= 0x20 && c != '"' && c != '\\' {
			continue
		}
		// Flush the run we've scanned past, then emit the escape.
		if pos < i {
			buf = append(buf, s[pos:i]...)
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
			buf = append(buf, '\\', 'u', '0', '0', hexChar(c>>4), hexChar(c&0x0f))
		}
		pos = i + 1
	}
	if pos < len(s) {
		buf = append(buf, s[pos:]...)
	}
	return append(buf, '"')
}

// appendStringField appends a `"key":"value"` pair (optionally
// prefixed with a leading comma) to buf. Key is treated as an ASCII
// literal — recordMeta-style schema keys carry no escapes by
// construction.
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
//	buf = appendIntField(buf, "index", 0, true)
func appendIntField(buf []byte, key string, value int, leadingComma bool) []byte {
	if leadingComma {
		buf = append(buf, ',')
	}
	buf = append(buf, '"')
	buf = append(buf, key...)
	buf = append(buf, '"', ':')
	return strconv.AppendInt(buf, int64(value), 10)
}

// appendInt64Field appends a `"key":N` pair for an int64.
func appendInt64Field(buf []byte, key string, value int64, leadingComma bool) []byte {
	if leadingComma {
		buf = append(buf, ',')
	}
	buf = append(buf, '"')
	buf = append(buf, key...)
	buf = append(buf, '"', ':')
	return strconv.AppendInt(buf, value, 10)
}

// appendFloat32 appends a float32 in the same shape json.Marshal
// emits — 'g' format, bitSize 32. Used by EmbeddingResponseDatum
// (per-element vector emission).
func appendFloat32(buf []byte, value float32) []byte {
	return strconv.AppendFloat(buf, float64(value), 'g', -1, 32)
}

// appendFloat64 appends a float64 in the same shape json.Marshal
// emits — 'g' format, bitSize 64.
func appendFloat64(buf []byte, value float64) []byte {
	return strconv.AppendFloat(buf, value, 'g', -1, 64)
}

// hexChar returns the ASCII hex digit for the low nibble of v.
func hexChar(v byte) byte {
	v &= 0x0f
	if v < 10 {
		return '0' + v
	}
	return 'a' + (v - 10)
}
