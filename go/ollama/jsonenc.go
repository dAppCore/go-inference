// SPDX-Licence-Identifier: EUPL-1.2

// Hand-rolled JSON-encoding primitives shared by the ollama adapter's
// hot-path encoders. The encoding/json reflect path allocates an
// encoder state machine + grow-doubled output buffer per call — every
// adapter encoder that fires per-request or per-streamed-NDJSON-chunk
// pays that floor. Ollama's wire protocol streams one ChatResponse or
// GenerateResponse JSON object per token, so the marshal hot path
// fires N times per generation.
//
// These helpers compose into per-shape encoders (Message.MarshalJSON,
// Options.MarshalJSON, ChatResponse.MarshalJSON, etc.) that land at a
// single buffer allocation per call — same minimax lift as
// state/filestore's encodeRecordMeta (W8-D), core.ParseHeaderRefs
// (W8-I/K), and the W9-D openai adapter's parallel jsonenc.go.
//
// The output is valid JSON, parseable both by encoding/json (round-
// trips into the same Go types) and by any naive JSON walker. All
// callers share the same escape contract — quote, backslash,
// b/f/n/r/t mnemonics, and \u00XX for other control chars below 0x20.
// Bytes >= 0x20 outside the quote/backslash pair pass through verbatim;
// encoding/json's default also escapes <, >, & for HTML safety but
// the ollama adapter does not emit into HTML contexts.
package ollama

import "strconv"

// appendJSONString appends a JSON-encoded string to buf — opening
// quote, escaped body, closing quote. Caller is responsible for
// providing the surrounding context (key, comma, etc).
//
//	buf = appendJSONString(buf, "answer")  // -> "answer"
//
// Escapes: \" \\ \b \f \n \r \t for the mnemonic forms and \u00XX
// for other bytes < 0x20. All other bytes pass through.
func appendJSONString(buf []byte, s string) []byte {
	buf = append(buf, '"')
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
// literal — wire-schema keys carry no escapes by construction.
//
//	buf = appendStringField(buf, "model", req.Model, false)
//	buf = appendStringField(buf, "role", "assistant", true)  // leading comma
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
//	buf = appendIntField(buf, "prompt_eval_count", 200, true)
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
//
//	buf = appendInt64Field(buf, "total_duration", 1_500_000_000, true)
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
// emits — 'g' format, bitSize 32. Used by Options sampling fields.
func appendFloat32(buf []byte, value float32) []byte {
	return strconv.AppendFloat(buf, float64(value), 'g', -1, 32)
}

// appendBoolField appends a `"key":true` or `"key":false` pair.
func appendBoolField(buf []byte, key string, value, leadingComma bool) []byte {
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
