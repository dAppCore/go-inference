// SPDX-Licence-Identifier: EUPL-1.2

// Hand-rolled JSON-encoding primitives shared by the ollama adapter's
// hot-path encoders. The encoding/json reflect path allocates an
// encoder state machine + grow-doubled output buffer per call — every
// adapter encoder that fires per-request or per-streamed-NDJSON-chunk
// pays that floor. Ollama's wire protocol streams one ChatResponse or
// GenerateResponse JSON object per token, so the marshal hot path
// fires N times per generation.
//
// These helpers compose into per-shape encoders (AppendChatResponse,
// AppendGenerateResponse, etc.) that land at a single buffer
// allocation per call when invoked DIRECTLY. Same minimax lift as
// state/filestore's encodeRecordMeta (W8-D), core.ParseHeaderRefs
// (W8-I/K), and the W9-D openai adapter's parallel jsonenc.go.
//
// Note: encoders are exported as standalone Append* functions, NOT as
// MarshalJSON methods. encoding/json.Marshal validates and recopies
// the bytes returned by MarshalJSON — for top-level marshals that
// erases the win. Consumers on the hot path use the Append* entry
// points directly; non-hot-path call sites can keep using
// core.JSONMarshalString. The lift is mirrored in W9-D's openai
// chunkenc.go comment ("Adding [MarshalJSON] routes encoding/json
// .Marshal through a call-and-revalidate path that ends up slower").
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
// emits — 'g' format, bitSize 32.
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

// --- Per-shape encoders for the ollama wire types ---

// appendMessage walks one Message into buf. Both fields always
// emitted (no omitempty on Role/Content per the Ollama API
// contract). Used inline by AppendChatResponse rather than as a
// MarshalJSON method — see package note above.
//
// Wire shape: {"role":"X","content":"Y"}
func appendMessage(buf []byte, msg Message) []byte {
	buf = append(buf, '{')
	buf = appendStringField(buf, "role", msg.Role, false)
	buf = appendStringField(buf, "content", msg.Content, true)
	return append(buf, '}')
}

// AppendChatResponse walks a ChatResponse into buf. Fires per
// streamed NDJSON token (server side) — one of the two hottest
// encoders in the package.
//
// Field order matches the struct declaration: model, message,
// done, prompt_eval_count, eval_count, four duration fields. All
// five count/duration fields carry omitempty semantics matching
// the reflect-path behaviour (zero-int / zero-int64 suppressed).
//
//	buf := AppendChatResponse(make([]byte, 0, chatResponseSize(resp)), resp)
func AppendChatResponse(buf []byte, resp ChatResponse) []byte {
	buf = append(buf, '{')
	buf = appendStringField(buf, "model", resp.Model, false)
	buf = append(buf, ',', '"', 'm', 'e', 's', 's', 'a', 'g', 'e', '"', ':')
	buf = appendMessage(buf, resp.Message)
	buf = appendBoolField(buf, "done", resp.Done, true)
	if resp.PromptEvalCount != 0 {
		buf = appendIntField(buf, "prompt_eval_count", resp.PromptEvalCount, true)
	}
	if resp.EvalCount != 0 {
		buf = appendIntField(buf, "eval_count", resp.EvalCount, true)
	}
	if resp.TotalDuration != 0 {
		buf = appendInt64Field(buf, "total_duration", resp.TotalDuration, true)
	}
	if resp.LoadDuration != 0 {
		buf = appendInt64Field(buf, "load_duration", resp.LoadDuration, true)
	}
	if resp.PromptEvalDuration != 0 {
		buf = appendInt64Field(buf, "prompt_eval_duration", resp.PromptEvalDuration, true)
	}
	if resp.EvalDuration != 0 {
		buf = appendInt64Field(buf, "eval_duration", resp.EvalDuration, true)
	}
	return append(buf, '}')
}

// chatResponseSize estimates the backing-buffer size for one
// ChatResponse so AppendChatResponse allocates once for the typical
// shape. Over-sizing inflates the make() allocation cost above what
// the reflect-path's tighter sizing pays; the estimate matches the
// actual wire-byte count closely.
//
// Fixed prefix: {"model":"X","message":{"role":"R","content":"C"},"done":bool}
// = 1 (open {) + 10 + len(Model) + 11 (",message":) + 24 + len(Role) + len(Content) + 13 (",done":false) + 1 (close })
// = 60 + variable bytes
func chatResponseSize(resp ChatResponse) int {
	size := 60 + len(resp.Model) + len(resp.Message.Role) + len(resp.Message.Content)
	if resp.PromptEvalCount != 0 {
		size += 25
	}
	if resp.EvalCount != 0 {
		size += 18
	}
	if resp.TotalDuration != 0 {
		size += 35
	}
	if resp.LoadDuration != 0 {
		size += 34
	}
	if resp.PromptEvalDuration != 0 {
		size += 41
	}
	if resp.EvalDuration != 0 {
		size += 34
	}
	return size
}

// AppendGenerateResponse walks a GenerateResponse into buf — the
// /api/generate per-NDJSON-token streaming shape. Same fields as
// ChatResponse minus the nested Message.
//
//	buf := AppendGenerateResponse(make([]byte, 0, generateResponseSize(resp)), resp)
func AppendGenerateResponse(buf []byte, resp GenerateResponse) []byte {
	buf = append(buf, '{')
	buf = appendStringField(buf, "model", resp.Model, false)
	buf = appendStringField(buf, "response", resp.Response, true)
	buf = appendBoolField(buf, "done", resp.Done, true)
	if resp.PromptEvalCount != 0 {
		buf = appendIntField(buf, "prompt_eval_count", resp.PromptEvalCount, true)
	}
	if resp.EvalCount != 0 {
		buf = appendIntField(buf, "eval_count", resp.EvalCount, true)
	}
	if resp.TotalDuration != 0 {
		buf = appendInt64Field(buf, "total_duration", resp.TotalDuration, true)
	}
	if resp.LoadDuration != 0 {
		buf = appendInt64Field(buf, "load_duration", resp.LoadDuration, true)
	}
	if resp.PromptEvalDuration != 0 {
		buf = appendInt64Field(buf, "prompt_eval_duration", resp.PromptEvalDuration, true)
	}
	if resp.EvalDuration != 0 {
		buf = appendInt64Field(buf, "eval_duration", resp.EvalDuration, true)
	}
	return append(buf, '}')
}

// generateResponseSize estimates the GenerateResponse buffer.
//
// Fixed prefix: {"model":"X","response":"Y","done":bool}
// = 1 + 10+len(Model) + 14+len(Response) + 13 + 1
func generateResponseSize(resp GenerateResponse) int {
	size := 39 + len(resp.Model) + len(resp.Response)
	if resp.PromptEvalCount != 0 {
		size += 25
	}
	if resp.EvalCount != 0 {
		size += 18
	}
	if resp.TotalDuration != 0 {
		size += 35
	}
	if resp.LoadDuration != 0 {
		size += 34
	}
	if resp.PromptEvalDuration != 0 {
		size += 41
	}
	if resp.EvalDuration != 0 {
		size += 34
	}
	return size
}
