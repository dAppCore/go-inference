// SPDX-Licence-Identifier: EUPL-1.2

// Package jsonenc provides hand-rolled JSON-encoding primitives
// shared across the inference adapter hot paths (openai, anthropic,
// ollama). The encoding/json reflect path allocates an encoder state
// machine and a grow-doubled output buffer on every Marshal call —
// each adapter encoder that fires per-request or per-streamed-token
// pays that floor. These primitives let per-shape encoders land at a
// single buffer allocation per call.
//
// Provenance: lifted in W9-Z from three byte-identical copies that
// shipped in W9-D (openai), W9-E (anthropic), and W9-G (ollama). The
// canonical fast-path uses anthropic's two-function split (W9-E) for
// AppendJSONString — a single forward scan followed by a single bulk
// append when no escape is needed; a separate tail-walker handles
// the escape-bearing case. Same minimax lift as state/filestore's
// encodeRecordMeta (W8-D) and core.ParseHeaderRefs (W8-I/K).
//
// The output is valid JSON and parseable both by encoding/json
// (round-trips into the same Go types) and by any naive JSON walker.
// All callers share the same escape contract — quote, backslash,
// b/f/n/r/t mnemonics, and \u00XX for other control chars below 0x20.
// Bytes >= 0x20 outside the quote/backslash pair pass through verbatim;
// encoding/json's default also escapes <, >, & for HTML safety but the
// adapters built on this package do not emit into HTML contexts.
//
// Encoders are exported as standalone Append* functions rather than
// MarshalJSON methods. encoding/json.Marshal validates and recopies
// the bytes returned by MarshalJSON — for top-level marshals that
// erases the win. Consumers on the hot path call the Append* entry
// points directly.
package jsonenc

import "strconv"

// AppendJSONString appends a JSON-encoded string to buf — opening
// quote, escaped body, closing quote. Caller is responsible for
// providing the surrounding context (key, comma, etc).
//
//	buf = jsonenc.AppendJSONString(buf, "answer")  // -> "answer"
//
// Escapes: \" \\ \b \f \n \r \t for the mnemonic forms and \u00XX
// for other bytes < 0x20. All other bytes pass through.
//
// Fast path: scan for any character requiring an escape. Adapter
// message bodies overwhelmingly contain neither — once a hot prefix
// passes the scan, we copy the whole string verbatim in one append.
// On the rare escape-bearing path we drop back to the byte-by-byte
// walk starting from the first hit. The split keeps the fast path
// inlineable.
func AppendJSONString(buf []byte, s string) []byte {
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
// Internal helper for AppendJSONString — separated out to keep the
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
			buf = append(buf, '\\', 'u', '0', '0', HexChar(c>>4), HexChar(c&0x0f))
		default:
			buf = append(buf, c)
		}
	}
	return append(buf, '"')
}

// AppendStringField appends a `"key":"value"` pair (optionally
// prefixed with a leading comma) to buf. Key is treated as an ASCII
// literal — wire-schema keys carry no escapes by construction.
//
//	buf = jsonenc.AppendStringField(buf, "model", req.Model, false)
//	buf = jsonenc.AppendStringField(buf, "id", id, true)  // leading comma
func AppendStringField(buf []byte, key, value string, leadingComma bool) []byte {
	if leadingComma {
		buf = append(buf, ',')
	}
	buf = append(buf, '"')
	buf = append(buf, key...)
	buf = append(buf, '"', ':')
	return AppendJSONString(buf, value)
}

// AppendIntField appends a `"key":N` pair (optionally prefixed with a
// leading comma) where N is the base-10 representation of value.
//
//	buf = jsonenc.AppendIntField(buf, "index", 0, true)
func AppendIntField(buf []byte, key string, value int, leadingComma bool) []byte {
	if leadingComma {
		buf = append(buf, ',')
	}
	buf = append(buf, '"')
	buf = append(buf, key...)
	buf = append(buf, '"', ':')
	return strconv.AppendInt(buf, int64(value), 10)
}

// AppendInt64Field appends a `"key":N` pair for an int64.
//
//	buf = jsonenc.AppendInt64Field(buf, "total_duration", 1_500_000_000, true)
func AppendInt64Field(buf []byte, key string, value int64, leadingComma bool) []byte {
	if leadingComma {
		buf = append(buf, ',')
	}
	buf = append(buf, '"')
	buf = append(buf, key...)
	buf = append(buf, '"', ':')
	return strconv.AppendInt(buf, value, 10)
}

// AppendBoolField appends a `"key":true` or `"key":false` pair.
//
//	buf = jsonenc.AppendBoolField(buf, "stream", req.Stream, true)
func AppendBoolField(buf []byte, key string, value, leadingComma bool) []byte {
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

// AppendFloat32Field appends a `"key":F` pair where F is rendered in
// the same 'g' format encoding/json emits for float32 (bitSize 32).
//
//	buf = jsonenc.AppendFloat32Field(buf, "temperature", *req.Temperature, true)
func AppendFloat32Field(buf []byte, key string, value float32, leadingComma bool) []byte {
	if leadingComma {
		buf = append(buf, ',')
	}
	buf = append(buf, '"')
	buf = append(buf, key...)
	buf = append(buf, '"', ':')
	return strconv.AppendFloat(buf, float64(value), 'g', -1, 32)
}

// AppendFloat32 appends a bare float32 value (no key, no comma) in
// the same shape json.Marshal emits — 'g' format, bitSize 32. Used
// for array-element emission (per-element embedding vectors) where
// the caller drives commas and surrounding context.
//
//	buf = jsonenc.AppendFloat32(buf, v)
func AppendFloat32(buf []byte, value float32) []byte {
	return strconv.AppendFloat(buf, float64(value), 'g', -1, 32)
}

// AppendFloat64 appends a bare float64 value in the same shape
// json.Marshal emits — 'g' format, bitSize 64.
//
//	buf = jsonenc.AppendFloat64(buf, score.Score)
func AppendFloat64(buf []byte, value float64) []byte {
	return strconv.AppendFloat(buf, value, 'g', -1, 64)
}

// HexChar returns the ASCII hex digit for the low nibble of v. Used
// by AppendJSONString's \u00XX escape branch; exported so adapter
// packages can reuse the same byte-to-hex contract when they emit
// their own escape paths (e.g. URI-encoded fields).
func HexChar(v byte) byte {
	v &= 0x0f
	if v < 10 {
		return '0' + v
	}
	return 'a' + (v - 10)
}
