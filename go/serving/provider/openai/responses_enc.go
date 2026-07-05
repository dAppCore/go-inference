// SPDX-Licence-Identifier: EUPL-1.2

// Hand-rolled encoders for the OpenAI Responses API wire shapes —
// Response and ResponseStreamEvent. Same W9-D shape as the chat-
// completions encoders: single-buffer emission, no reflect, the
// shared jsonenc.AppendStringField / jsonenc.AppendIntField
// primitives from dappco.re/go/inference/jsonenc (W9-Z lift).
//
// Responses is the OpenAI v1/responses endpoint — the per-token
// stream event encoder fires per generated text delta on the
// streaming path; the per-response Response encoder fires once per
// non-streaming completed call (and embeds itself inside the
// terminal "response.completed" stream event).

package openai

import "dappco.re/go/inference/jsonenc"

// appendResponseOutputText walks one ResponseOutputText into buf.
// Two ASCII string fields in canonical order.
func appendResponseOutputText(buf []byte, item ResponseOutputText) []byte {
	buf = append(buf, '{')
	buf = jsonenc.AppendStringField(buf, "type", item.Type, false)
	buf = jsonenc.AppendStringField(buf, "text", item.Text, true)
	return append(buf, '}')
}

// appendResponseOutputMessage walks one ResponseOutputMessage into
// buf. The ID field carries the omitempty tag — emit only when set.
func appendResponseOutputMessage(buf []byte, msg ResponseOutputMessage) []byte {
	buf = append(buf, '{')
	leading := false
	if msg.ID != "" {
		buf = jsonenc.AppendStringField(buf, "id", msg.ID, false)
		leading = true
	}
	buf = jsonenc.AppendStringField(buf, "type", msg.Type, leading)
	buf = jsonenc.AppendStringField(buf, "role", msg.Role, true)
	buf = append(buf, ',', '"', 'c', 'o', 'n', 't', 'e', 'n', 't', '"', ':', '[')
	for i, item := range msg.Content {
		if i > 0 {
			buf = append(buf, ',')
		}
		buf = appendResponseOutputText(buf, item)
	}
	return append(buf, ']', '}')
}

// appendResponseUsage walks a ResponseUsage into buf. Three int
// fields — input_tokens, output_tokens, total_tokens.
func appendResponseUsage(buf []byte, usage ResponseUsage) []byte {
	buf = append(buf, '{')
	buf = jsonenc.AppendIntField(buf, "input_tokens", usage.InputTokens, false)
	buf = jsonenc.AppendIntField(buf, "output_tokens", usage.OutputTokens, true)
	buf = jsonenc.AppendIntField(buf, "total_tokens", usage.TotalTokens, true)
	return append(buf, '}')
}

// appendResponse walks the full Response shape into buf. Field
// order matches the struct declaration so the wire output is byte-
// identical to encoding/json.Marshal output.
func appendResponse(buf []byte, resp Response) []byte {
	buf = append(buf, '{')
	buf = jsonenc.AppendStringField(buf, "id", resp.ID, false)
	buf = jsonenc.AppendStringField(buf, "object", resp.Object, true)
	buf = jsonenc.AppendInt64Field(buf, "created", resp.Created, true)
	buf = jsonenc.AppendStringField(buf, "model", resp.Model, true)
	buf = append(buf, ',', '"', 'o', 'u', 't', 'p', 'u', 't', '"', ':', '[')
	for i, msg := range resp.Output {
		if i > 0 {
			buf = append(buf, ',')
		}
		buf = appendResponseOutputMessage(buf, msg)
	}
	buf = append(buf, ']', ',', '"', 'u', 's', 'a', 'g', 'e', '"', ':')
	buf = appendResponseUsage(buf, resp.Usage)
	if resp.Thought != nil {
		buf = append(buf, ',', '"', 't', 'h', 'o', 'u', 'g', 'h', 't', '"', ':')
		buf = jsonenc.AppendJSONString(buf, *resp.Thought)
	}
	return append(buf, '}')
}

// responseSize estimates the backing-buffer size for one Response
// so the encoder allocates once. Conservative (slight over-shoot)
// so closing punctuation doesn't trigger a grow into the next size
// class.
func responseSize(resp Response) int {
	size := 4 // {} + slack for closing punctuation
	size += 7 + len(resp.ID)
	size += 11 + len(resp.Object)
	size += 12 + 20
	size += 10 + len(resp.Model)
	size += 12 // ,"output":[]
	for _, msg := range resp.Output {
		size += 3 // {} + separator
		if msg.ID != "" {
			size += 8 + len(msg.ID)
		}
		size += 9 + len(msg.Type)
		size += 9 + len(msg.Role)
		size += 13 // ,"content":[]
		for _, item := range msg.Content {
			size += 3 + 9 + len(item.Type) + 9 + len(item.Text)
		}
	}
	size += 62 // ,"usage":{...}
	if resp.Thought != nil {
		size += 13 + len(*resp.Thought)
	}
	return size
}

// appendResponseStreamEvent walks the ResponseStreamEvent shape
// into buf. The Response pointer + Delta + Thought are all
// omitempty — emit only the fields set on the event.
func appendResponseStreamEvent(buf []byte, event ResponseStreamEvent) []byte {
	buf = append(buf, '{')
	buf = jsonenc.AppendStringField(buf, "type", event.Type, false)
	if event.Response != nil {
		buf = append(buf, ',', '"', 'r', 'e', 's', 'p', 'o', 'n', 's', 'e', '"', ':')
		buf = appendResponse(buf, *event.Response)
	}
	if event.Delta != "" {
		buf = jsonenc.AppendStringField(buf, "delta", event.Delta, true)
	}
	if event.Thought != nil {
		buf = append(buf, ',', '"', 't', 'h', 'o', 'u', 'g', 'h', 't', '"', ':')
		buf = jsonenc.AppendJSONString(buf, *event.Thought)
	}
	return append(buf, '}')
}

// responseStreamEventSize estimates the backing-buffer size for one
// stream event so the encoder allocates once. The Response pointer
// embedding is the load-bearing case (response.completed events) —
// uses responseSize recursively.
//
// The estimate is intentionally conservative (covers the closing
// '}' and any trailing punctuation) so the typical event lands in a
// single allocator size class. Pathological escape-heavy values let
// append grow once.
func responseStreamEventSize(event ResponseStreamEvent) int {
	size := 4 // {"type":"..."} framing + closing brace + slack
	size += 8 + len(event.Type)
	if event.Response != nil {
		size += 12 + responseSize(*event.Response)
	}
	if event.Delta != "" {
		size += 11 + len(event.Delta)
	}
	if event.Thought != nil {
		size += 13 + len(*event.Thought)
	}
	return size
}
