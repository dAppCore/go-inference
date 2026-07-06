// SPDX-Licence-Identifier: EUPL-1.2

package anthropic

import (
	"dappco.re/go/inference/jsonenc"
)

// Anthropic Messages streaming events — the SSE `data:` payloads a streaming
// completion emits. The HTTP handler frames each as
// `event: <type>\ndata: <payload>\n\n`; these builders produce the <payload>
// JSON in the same caller-owns-buf, hand-rolled shape as AppendMessageResponse
// (content_block_delta fires once per token, so it must stay off the reflect
// path). The spec sequence per stream is:
//
//	message_start → content_block_start → content_block_delta* →
//	content_block_stop → message_delta → message_stop
//
// (a `ping` event may interleave). Claude Code's SSE parser requires the full
// sequence — a stream that skips content_block_start/stop or drops the
// message_start wrapper fails to render.

// MessageStopPayload is the fixed `message_stop` event data — the terminal
// event of every stream.
const MessageStopPayload = `{"type":"message_stop"}`

// PingPayload is the `ping` keep-alive event data.
const PingPayload = `{"type":"ping"}`

// AppendMessageStartEvent emits the `message_start` payload — the opening
// event, wrapping the message envelope (id/model/role + empty content +
// input-token usage; output_tokens is 0 at start and accumulates into the
// closing message_delta):
//
//	{"type":"message_start","message":{<MessageResponse>}}
//
//	buf := AppendMessageStartEvent(nil, anthropic.MessageResponse{
//	    ID: id, Type: "message", Role: "assistant", Model: model,
//	    Usage: anthropic.Usage{InputTokens: promptTokens},
//	})
func AppendMessageStartEvent(buf []byte, msg MessageResponse) []byte {
	buf = append(buf, '{')
	buf = jsonenc.AppendStringField(buf, "type", "message_start", false)
	buf = append(buf, `,"message":`...)
	buf = AppendMessageResponse(buf, msg)
	return append(buf, '}')
}

// AppendContentBlockStartEvent emits the `content_block_start` payload opening
// the text block at index:
//
//	{"type":"content_block_start","index":N,"content_block":{"type":"text","text":""}}
func AppendContentBlockStartEvent(buf []byte, index int) []byte {
	buf = append(buf, '{')
	buf = jsonenc.AppendStringField(buf, "type", "content_block_start", false)
	buf = jsonenc.AppendIntField(buf, "index", index, true)
	buf = append(buf, `,"content_block":{"type":"text","text":""}`...)
	return append(buf, '}')
}

// AppendContentBlockDeltaEvent emits one `content_block_delta` payload — the
// per-token hot path:
//
//	{"type":"content_block_delta","index":N,"delta":{"type":"text_delta","text":"…"}}
//
// text is JSON-escaped via jsonenc.AppendStringField.
func AppendContentBlockDeltaEvent(buf []byte, index int, text string) []byte {
	buf = append(buf, '{')
	buf = jsonenc.AppendStringField(buf, "type", "content_block_delta", false)
	buf = jsonenc.AppendIntField(buf, "index", index, true)
	buf = append(buf, `,"delta":{`...)
	buf = jsonenc.AppendStringField(buf, "type", "text_delta", false)
	buf = jsonenc.AppendStringField(buf, "text", text, true)
	return append(buf, '}', '}')
}

// AppendContentBlockStopEvent emits the `content_block_stop` payload closing
// the block at index:
//
//	{"type":"content_block_stop","index":N}
func AppendContentBlockStopEvent(buf []byte, index int) []byte {
	buf = append(buf, '{')
	buf = jsonenc.AppendStringField(buf, "type", "content_block_stop", false)
	buf = jsonenc.AppendIntField(buf, "index", index, true)
	return append(buf, '}')
}

// AppendContentBlockStartToolUseEvent opens a tool_use content block at index —
// the model called a function. The input starts as an empty object; the
// arguments arrive as input_json_delta events the client assembles:
//
//	{"type":"content_block_start","index":N,"content_block":{"type":"tool_use","id":"…","name":"…","input":{}}}
func AppendContentBlockStartToolUseEvent(buf []byte, index int, id, name string) []byte {
	buf = append(buf, '{')
	buf = jsonenc.AppendStringField(buf, "type", "content_block_start", false)
	buf = jsonenc.AppendIntField(buf, "index", index, true)
	buf = append(buf, `,"content_block":{`...)
	buf = jsonenc.AppendStringField(buf, "type", "tool_use", false)
	buf = jsonenc.AppendStringField(buf, "id", id, true)
	buf = jsonenc.AppendStringField(buf, "name", name, true)
	buf = append(buf, `,"input":{}}`...)
	return append(buf, '}')
}

// AppendInputJSONDeltaEvent emits one tool_use arguments delta — the partial (or,
// as this engine sends it, whole) JSON of the call's input object. Claude Code
// concatenates the partial_json fragments and parses the result:
//
//	{"type":"content_block_delta","index":N,"delta":{"type":"input_json_delta","partial_json":"…"}}
func AppendInputJSONDeltaEvent(buf []byte, index int, partialJSON string) []byte {
	buf = append(buf, '{')
	buf = jsonenc.AppendStringField(buf, "type", "content_block_delta", false)
	buf = jsonenc.AppendIntField(buf, "index", index, true)
	buf = append(buf, `,"delta":{`...)
	buf = jsonenc.AppendStringField(buf, "type", "input_json_delta", false)
	buf = jsonenc.AppendStringField(buf, "partial_json", partialJSON, true)
	return append(buf, '}', '}')
}

// AppendMessageDeltaEvent emits the `message_delta` payload — the penultimate
// event carrying the terminal stop_reason + cumulative output usage:
//
//	{"type":"message_delta","delta":{"stop_reason":"…","stop_sequence":<seq|null>},"usage":{"output_tokens":N}}
//
// A non-empty stopSequence emits it as the matched sequence (stop_reason is
// then typically "stop_sequence"); empty emits null.
func AppendMessageDeltaEvent(buf []byte, stopReason, stopSequence string, outputTokens int) []byte {
	buf = append(buf, '{')
	buf = jsonenc.AppendStringField(buf, "type", "message_delta", false)
	buf = append(buf, `,"delta":{`...)
	buf = jsonenc.AppendStringField(buf, "stop_reason", stopReason, false)
	if stopSequence != "" {
		buf = jsonenc.AppendStringField(buf, "stop_sequence", stopSequence, true)
	} else {
		buf = append(buf, `,"stop_sequence":null`...)
	}
	buf = append(buf, `},"usage":{`...)
	buf = jsonenc.AppendIntField(buf, "output_tokens", outputTokens, false)
	return append(buf, '}', '}')
}
