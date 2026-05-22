// SPDX-Licence-Identifier: EUPL-1.2

// Hand-rolled encoders for the OpenAI chat-completions wire shapes
// that fire on the streaming + non-streaming serve paths.
//
// Per-token cost matters: serveStreaming emits one ChatCompletionChunk
// per content/thought delta in the SSE loop plus a priming chunk and
// a terminating chunk. Routing each through encoding/json's reflect
// path costs an encoder state machine, a grow-doubled output buffer,
// per-pointer envelope copies, and (via core.JSONMarshalString +
// core.Concat) a separate string copy for the "data: " SSE framing.
//
// These encoders collapse the same shape into a single caller-bound
// buffer and embed the SSE framing in-line — one allocation for the
// emitted frame, no intermediate string conversion. Wire output
// matches encoding/json across every branch (round-trip locked by
// TestChatCompletionChunk_MarshalJSON_RoundTrip).

package openai

import "dappco.re/go/inference/jsonenc"

// appendChatMessageDelta walks the two-field ChatMessageDelta into buf.
// Same shape and escape contract as ChatMessageDelta.MarshalJSON, but
// without the buffer-allocation hop — the chunk encoders pull it
// inline so the entire frame lands in a single backing buffer.
//
// Wire shapes (identical to encoding/json with the existing tags):
//   - empty                       -> {}
//   - role set (priming/closing)  -> {"role":"X","content":"Y"}
//   - content only                -> {"content":"Y"}
//   - both                        -> {"role":"X","content":"Y"}
func appendChatMessageDelta(buf []byte, d ChatMessageDelta) []byte {
	if d.Role == "" && d.Content == "" {
		return append(buf, '{', '}')
	}
	buf = append(buf, '{')
	if d.Role != "" {
		buf = jsonenc.AppendStringField(buf, "role", d.Role, false)
		buf = jsonenc.AppendStringField(buf, "content", d.Content, true)
	} else {
		buf = jsonenc.AppendStringField(buf, "content", d.Content, false)
	}
	return append(buf, '}')
}

// appendChatChunkChoice walks one ChatChunkChoice into buf. The
// FinishReason pointer maps to `null` (not omitted) when nil — the
// field carries no omitempty tag in the canonical shape, and the
// terminal chunk's finish_reason is the load-bearing field clients
// pivot on.
func appendChatChunkChoice(buf []byte, choice ChatChunkChoice) []byte {
	buf = append(buf, '{')
	buf = jsonenc.AppendIntField(buf, "index", choice.Index, false)
	buf = append(buf, ',', '"', 'd', 'e', 'l', 't', 'a', '"', ':')
	buf = appendChatMessageDelta(buf, choice.Delta)
	buf = append(buf, ',', '"', 'f', 'i', 'n', 'i', 's', 'h', '_', 'r', 'e', 'a', 's', 'o', 'n', '"', ':')
	if choice.FinishReason == nil {
		buf = append(buf, 'n', 'u', 'l', 'l')
	} else {
		buf = jsonenc.AppendJSONString(buf, *choice.FinishReason)
	}
	return append(buf, '}')
}

// appendChatCompletionChunk walks a ChatCompletionChunk into buf.
// Field order matches the struct declaration (id, object, created,
// model, choices, thought) — encoding/json emits in that same order
// for the canonical tag set.
func appendChatCompletionChunk(buf []byte, chunk ChatCompletionChunk) []byte {
	buf = append(buf, '{')
	buf = jsonenc.AppendStringField(buf, "id", chunk.ID, false)
	buf = jsonenc.AppendStringField(buf, "object", chunk.Object, true)
	buf = jsonenc.AppendInt64Field(buf, "created", chunk.Created, true)
	buf = jsonenc.AppendStringField(buf, "model", chunk.Model, true)
	buf = append(buf, ',', '"', 'c', 'h', 'o', 'i', 'c', 'e', 's', '"', ':', '[')
	for i, choice := range chunk.Choices {
		if i > 0 {
			buf = append(buf, ',')
		}
		buf = appendChatChunkChoice(buf, choice)
	}
	buf = append(buf, ']')
	if chunk.Thought != nil {
		buf = append(buf, ',', '"', 't', 'h', 'o', 'u', 'g', 'h', 't', '"', ':')
		buf = jsonenc.AppendJSONString(buf, *chunk.Thought)
	}
	return append(buf, '}')
}

// appendChatCompletionChunkSSE writes a complete SSE frame into buf —
// the literal `data: ` prefix, the chunk JSON body, and the trailing
// `\n\n`. Lets the streaming hot path emit the whole frame in a
// single backing buffer instead of three (JSON body + Concat scratch
// + final []byte conversion).
//
//	frame := appendChatCompletionChunkSSE(nil, chunk)
//	w.Write(frame)
func appendChatCompletionChunkSSE(buf []byte, chunk ChatCompletionChunk) []byte {
	buf = append(buf, 'd', 'a', 't', 'a', ':', ' ')
	buf = appendChatCompletionChunk(buf, chunk)
	return append(buf, '\n', '\n')
}

// chunkSSEFrameSize estimates the backing-buffer size for one SSE
// frame so the streaming path allocates once. The estimate covers
// the "data: " prefix + every fixed key + a one-byte ASCII assumption
// for the variable fields + the trailing "\n\n". Pathological
// escape-heavy content lets append grow once.
func chunkSSEFrameSize(chunk ChatCompletionChunk) int {
	// "data: " (6) + "{" (1) + closing "\n\n" (2)
	size := 6 + 1 + 2
	size += 7 + len(chunk.ID)          // "id":"...",
	size += 11 + len(chunk.Object)     // "object":"...",
	size += 12 + 20                    // "created":<int64>,
	size += 10 + len(chunk.Model)      // "model":"...",
	size += 12                         // "choices":[...]
	for _, choice := range chunk.Choices {
		size += 11 + 20                    // "index":N,
		// delta = "delta":{...} — delta wire is 30+role+content bytes worst case
		size += 9 + 32 + len(choice.Delta.Role) + len(choice.Delta.Content)
		size += 19                         // ,"finish_reason":<null|"..">
		if choice.FinishReason != nil {
			size += len(*choice.FinishReason) + 2
		} else {
			size += 4
		}
		size += 3                          // {} wrap + separator
	}
	if chunk.Thought != nil {
		size += 12 + len(*chunk.Thought)   // ,"thought":"..."
	}
	return size
}

// Note: ChatCompletionChunk does NOT carry a MarshalJSON method.
// Adding one routes encoding/json.Marshal through a call-and-revalidate
// path that ends up slower than the reflect-walked default — every
// proxy serialisation site would pay the cost. The streaming hot
// path bypasses encoding/json entirely via appendChatCompletionChunkSSE.

// appendChatMessage walks a ChatMessage into buf. Used by the
// non-streaming response encoder for the assistant message body.
func appendChatMessage(buf []byte, msg ChatMessage) []byte {
	buf = append(buf, '{')
	buf = jsonenc.AppendStringField(buf, "role", msg.Role, false)
	buf = jsonenc.AppendStringField(buf, "content", msg.Content, true)
	return append(buf, '}')
}

// appendChatChoice walks a ChatChoice (non-streaming response) into
// buf. Field order matches the struct: index, message, finish_reason.
func appendChatChoice(buf []byte, choice ChatChoice) []byte {
	buf = append(buf, '{')
	buf = jsonenc.AppendIntField(buf, "index", choice.Index, false)
	buf = append(buf, ',', '"', 'm', 'e', 's', 's', 'a', 'g', 'e', '"', ':')
	buf = appendChatMessage(buf, choice.Message)
	buf = jsonenc.AppendStringField(buf, "finish_reason", choice.FinishReason, true)
	return append(buf, '}')
}

// appendChatUsage walks a ChatUsage into buf. Three int fields in
// canonical OpenAI order.
func appendChatUsage(buf []byte, usage ChatUsage) []byte {
	buf = append(buf, '{')
	buf = jsonenc.AppendIntField(buf, "prompt_tokens", usage.PromptTokens, false)
	buf = jsonenc.AppendIntField(buf, "completion_tokens", usage.CompletionTokens, true)
	buf = jsonenc.AppendIntField(buf, "total_tokens", usage.TotalTokens, true)
	return append(buf, '}')
}

// appendChatCompletionResponse walks the non-streaming ChatCompletion
// response into buf. Field order matches the struct declaration so
// the wire shape is byte-identical to encoding/json.Marshal output.
func appendChatCompletionResponse(buf []byte, resp ChatCompletionResponse) []byte {
	buf = append(buf, '{')
	buf = jsonenc.AppendStringField(buf, "id", resp.ID, false)
	buf = jsonenc.AppendStringField(buf, "object", resp.Object, true)
	buf = jsonenc.AppendInt64Field(buf, "created", resp.Created, true)
	buf = jsonenc.AppendStringField(buf, "model", resp.Model, true)
	buf = append(buf, ',', '"', 'c', 'h', 'o', 'i', 'c', 'e', 's', '"', ':', '[')
	for i, choice := range resp.Choices {
		if i > 0 {
			buf = append(buf, ',')
		}
		buf = appendChatChoice(buf, choice)
	}
	buf = append(buf, ']', ',', '"', 'u', 's', 'a', 'g', 'e', '"', ':')
	buf = appendChatUsage(buf, resp.Usage)
	if resp.Thought != nil {
		buf = append(buf, ',', '"', 't', 'h', 'o', 'u', 'g', 'h', 't', '"', ':')
		buf = jsonenc.AppendJSONString(buf, *resp.Thought)
	}
	return append(buf, '}')
}

// appendEmbeddingResponseDatum walks one embedding-response datum
// (object, index, embedding vector) into buf. The embedding slice
// is emitted directly via strconv.AppendFloat — avoids the
// reflect-walk per-element cost that encoding/json pays.
func appendEmbeddingResponseDatum(buf []byte, datum EmbeddingResponseDatum) []byte {
	buf = append(buf, '{')
	buf = jsonenc.AppendStringField(buf, "object", datum.Object, false)
	buf = jsonenc.AppendIntField(buf, "index", datum.Index, true)
	buf = append(buf, ',', '"', 'e', 'm', 'b', 'e', 'd', 'd', 'i', 'n', 'g', '"', ':', '[')
	for i, v := range datum.Embedding {
		if i > 0 {
			buf = append(buf, ',')
		}
		buf = jsonenc.AppendFloat32(buf, v)
	}
	return append(buf, ']', '}')
}

// appendEmbeddingUsage walks an inference.EmbeddingUsage into buf.
// Two int fields — prompt_tokens, total_tokens — in canonical
// OpenAI order.
func appendEmbeddingUsage(buf []byte, prompt, total int) []byte {
	buf = append(buf, '{')
	buf = jsonenc.AppendIntField(buf, "prompt_tokens", prompt, false)
	buf = jsonenc.AppendIntField(buf, "total_tokens", total, true)
	return append(buf, '}')
}

// appendEmbeddingResponse walks the full EmbeddingResponse shape
// into buf. The per-vector embedding fan-out is the load-bearing
// cost (a 20×1024 response emits 20480 float32 values); the hand-
// rolled walk keeps the per-element path on a single buffer with
// no reflect.
func appendEmbeddingResponse(buf []byte, resp EmbeddingResponse) []byte {
	buf = append(buf, '{')
	buf = jsonenc.AppendStringField(buf, "object", resp.Object, false)
	buf = append(buf, ',', '"', 'd', 'a', 't', 'a', '"', ':', '[')
	for i, datum := range resp.Data {
		if i > 0 {
			buf = append(buf, ',')
		}
		buf = appendEmbeddingResponseDatum(buf, datum)
	}
	buf = append(buf, ']')
	buf = jsonenc.AppendStringField(buf, "model", resp.Model, true)
	buf = append(buf, ',', '"', 'u', 's', 'a', 'g', 'e', '"', ':')
	buf = appendEmbeddingUsage(buf, resp.Usage.PromptTokens, resp.Usage.TotalTokens)
	return append(buf, '}')
}

// embeddingResponseSize estimates the backing-buffer size for one
// EmbeddingResponse so the encoder allocates once. Each float32
// emits at most ~12 ASCII chars under the 'g' format (sign + 7
// significant digits + exponent + dot); empirical mean across the
// embedding ranges (~ -1..+1) is ~7.9 chars + 1 separator. The
// heuristic uses 9 — under-commits on the worst case (scientific-
// notation values) and lets append grow once.
func embeddingResponseSize(resp EmbeddingResponse) int {
	size := 2 // braces
	size += 11 + len(resp.Object)
	size += 9 // "data":[]
	for _, datum := range resp.Data {
		size += 12 + len(datum.Object) // {"object":"X"
		size += 11 + 20                // "index":N
		size += 14                     // "embedding":[]
		size += len(datum.Embedding) * 9
		size += 2 // }
	}
	size += 10 + len(resp.Model)
	size += 50 // "usage":{prompt_tokens:N,total_tokens:N}
	return size
}

// chatCompletionResponseSize estimates the backing-buffer size for
// one ChatCompletionResponse so the encoder allocates once.
func chatCompletionResponseSize(resp ChatCompletionResponse) int {
	size := 2 // braces
	size += 7 + len(resp.ID)
	size += 11 + len(resp.Object)
	size += 12 + 20
	size += 10 + len(resp.Model)
	size += 12 // "choices":[]
	for _, choice := range resp.Choices {
		// {"index":N,"message":{"role":"X","content":"Y"},"finish_reason":"Z"}
		size += 12 + 20
		size += 12 + 8 + len(choice.Message.Role) + 11 + len(choice.Message.Content) + 1
		size += 18 + len(choice.FinishReason)
		size += 2
	}
	size += 56 // "usage":{prompt_tokens:N,completion_tokens:N,total_tokens:N}
	if resp.Thought != nil {
		size += 12 + len(*resp.Thought)
	}
	return size
}
