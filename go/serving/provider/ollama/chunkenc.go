// SPDX-Licence-Identifier: EUPL-1.2

// Hand-rolled encoders for the Ollama wire shapes — ChatResponse,
// GenerateResponse, TagsResponse. Per-token cost matters: Ollama
// streams one ChatResponse or GenerateResponse JSON object per
// generated token on /api/chat and /api/generate respectively, so
// every per-shape encoder fires N times per generation.
//
// These encoders compose the shared jsonenc primitives at
// dappco.re/go/inference/jsonenc (W9-Z lift) and land at a single
// buffer allocation per call — same minimax lift as state/filestore's
// encodeRecordMeta (W8-D) and openai's chunkenc.go (W9-D).
//
// Note: encoders are exported as standalone Append* functions rather
// than MarshalJSON methods. encoding/json.Marshal validates and
// recopies the bytes returned by MarshalJSON — for top-level marshals
// that erases the win. Consumers on the hot path call Append* entry
// points directly; non-hot-path call sites can keep using
// core.JSONMarshalString.

package ollama

import "dappco.re/go/inference/jsonenc"

// appendMessage walks one Message into buf. Both fields always
// emitted (no omitempty on Role/Content per the Ollama API
// contract). Used inline by AppendChatResponse rather than as a
// MarshalJSON method — see package note above.
//
// Wire shape: {"role":"X","content":"Y"}
func appendMessage(buf []byte, msg Message) []byte {
	buf = append(buf, '{')
	buf = jsonenc.AppendStringField(buf, "role", msg.Role, false)
	buf = jsonenc.AppendStringField(buf, "content", msg.Content, true)
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
	buf = jsonenc.AppendStringField(buf, "model", resp.Model, false)
	buf = append(buf, ',', '"', 'm', 'e', 's', 's', 'a', 'g', 'e', '"', ':')
	buf = appendMessage(buf, resp.Message)
	buf = jsonenc.AppendBoolField(buf, "done", resp.Done, true)
	if resp.PromptEvalCount != 0 {
		buf = jsonenc.AppendIntField(buf, "prompt_eval_count", resp.PromptEvalCount, true)
	}
	if resp.EvalCount != 0 {
		buf = jsonenc.AppendIntField(buf, "eval_count", resp.EvalCount, true)
	}
	if resp.TotalDuration != 0 {
		buf = jsonenc.AppendInt64Field(buf, "total_duration", resp.TotalDuration, true)
	}
	if resp.LoadDuration != 0 {
		buf = jsonenc.AppendInt64Field(buf, "load_duration", resp.LoadDuration, true)
	}
	if resp.PromptEvalDuration != 0 {
		buf = jsonenc.AppendInt64Field(buf, "prompt_eval_duration", resp.PromptEvalDuration, true)
	}
	if resp.EvalDuration != 0 {
		buf = jsonenc.AppendInt64Field(buf, "eval_duration", resp.EvalDuration, true)
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
	buf = jsonenc.AppendStringField(buf, "model", resp.Model, false)
	buf = jsonenc.AppendStringField(buf, "response", resp.Response, true)
	buf = jsonenc.AppendBoolField(buf, "done", resp.Done, true)
	if resp.PromptEvalCount != 0 {
		buf = jsonenc.AppendIntField(buf, "prompt_eval_count", resp.PromptEvalCount, true)
	}
	if resp.EvalCount != 0 {
		buf = jsonenc.AppendIntField(buf, "eval_count", resp.EvalCount, true)
	}
	if resp.TotalDuration != 0 {
		buf = jsonenc.AppendInt64Field(buf, "total_duration", resp.TotalDuration, true)
	}
	if resp.LoadDuration != 0 {
		buf = jsonenc.AppendInt64Field(buf, "load_duration", resp.LoadDuration, true)
	}
	if resp.PromptEvalDuration != 0 {
		buf = jsonenc.AppendInt64Field(buf, "prompt_eval_duration", resp.PromptEvalDuration, true)
	}
	if resp.EvalDuration != 0 {
		buf = jsonenc.AppendInt64Field(buf, "eval_duration", resp.EvalDuration, true)
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

// appendModelTag walks one ModelTag into buf — used inline by
// AppendTagsResponse. Three of the four fields carry omitempty
// (Model, ModifiedAt, Size); Name is always emitted.
func appendModelTag(buf []byte, tag ModelTag) []byte {
	buf = append(buf, '{')
	buf = jsonenc.AppendStringField(buf, "name", tag.Name, false)
	if tag.Model != "" {
		buf = jsonenc.AppendStringField(buf, "model", tag.Model, true)
	}
	if tag.ModifiedAt != "" {
		buf = jsonenc.AppendStringField(buf, "modified_at", tag.ModifiedAt, true)
	}
	if tag.Size != 0 {
		buf = jsonenc.AppendInt64Field(buf, "size", tag.Size, true)
	}
	return append(buf, '}')
}

// AppendTagsResponse walks a TagsResponse (/api/tags). Discovery
// hot path — fires once per client startup (open-webui pings this
// on every page load) and again on every model-list refresh.
//
// A nil Models slice emits as "models":null (matching encoding/json
// semantics for nil-slice fields); an empty []ModelTag{} emits as
// "models":[]. Downstream consumers (e.g. open-webui) treat both
// forms as "no models served" interchangeably, but the wire shape
// must remain consistent with the reflect-path output for proxy
// pass-through.
//
//	buf := AppendTagsResponse(make([]byte, 0, tagsResponseSize(resp)), resp)
func AppendTagsResponse(buf []byte, resp TagsResponse) []byte {
	buf = append(buf, '{', '"', 'm', 'o', 'd', 'e', 'l', 's', '"', ':')
	if resp.Models == nil {
		return append(buf, 'n', 'u', 'l', 'l', '}')
	}
	buf = append(buf, '[')
	for i, tag := range resp.Models {
		if i > 0 {
			buf = append(buf, ',')
		}
		buf = appendModelTag(buf, tag)
	}
	return append(buf, ']', '}')
}

// tagsResponseSize estimates the TagsResponse buffer. The
// "models":null variant emits 17 bytes; the slice variant grows
// per-tag.
func tagsResponseSize(resp TagsResponse) int {
	if resp.Models == nil {
		return 17 // {"models":null}
	}
	size := 13 // {"models":[]}
	for i, tag := range resp.Models {
		if i > 0 {
			size++
		}
		// {"name":"X" = 11 fixed + name
		size += 11 + len(tag.Name)
		if tag.Model != "" {
			size += 11 + len(tag.Model)
		}
		if tag.ModifiedAt != "" {
			size += 16 + len(tag.ModifiedAt)
		}
		if tag.Size != 0 {
			size += 9 + 12 // "size":NNNNNNNNN
		}
		size++ // closing }
	}
	return size
}
