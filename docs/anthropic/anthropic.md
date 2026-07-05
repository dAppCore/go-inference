<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# serving/provider/anthropic — Messages API native server

**Package**: `dappco.re/go/inference/serving/provider/anthropic`
**Route**: `POST /v1/messages`

## What this is

A **native** Anthropic Messages server: it decodes the Anthropic wire request,
runs it against the LOCAL engine, and emits Anthropic-native output — a
`MessageResponse` JSON body, or the Anthropic SSE event sequence when
`stream: true`. Not a proxy to Anthropic's API.

The DTOs, translation, and wire encoders live in this package (`anthropic.go`,
`anthropic_stream.go`). The HTTP handler is assembled in `serving/compat`
(`mux.go`, `anthropicMessagesHandler` + `serveAnthropicMessageStream`) and
mounted by `cmd/lem serve` (default `:36911`). Point a Claude-flavoured SDK at
the route and it gets real local inference.

## Constants

```go
const DefaultMessagesPath = "/v1/messages"
```

## DTOs (`anthropic.go`)

```go
ContentBlock     // type + text — Anthropic's typed-block content model
Message          // role + []ContentBlock
MessageRequest   // model + system + messages + max_tokens + sampler + stream + stop_sequences
Usage            // input_tokens + output_tokens
MessageResponse  // id + type + role + model + content[] + stop_reason + stop_sequence + usage
```

`MessageRequest` models: `model`, `system`, `messages`, `max_tokens`,
`temperature`, `top_p`, `min_p`, `top_k`, `stream`, `stop_sequences`. `min_p` is
the gemma4 sampling extension.

Key differences from OpenAI:

- `Message.Content` is `[]ContentBlock`, not a plain string.
- `system` is a top-level field, not a message with role=system.
- `Usage` uses `input_tokens` / `output_tokens` (vs OpenAI's `prompt_tokens` /
  `completion_tokens`).
- Stop reason is named (`end_turn` / `stop_sequence` / …), not a free string.

## InferenceMessages

```go
messages := anthropic.InferenceMessages(req)
```

Flattens each message's typed-block content to plain text (`blockText`) and
builds the `inference.Message` slice. The top-level `system` field becomes a
leading system message, so the runtime sees one uniform message list regardless
of API origin. `blockText` keeps only `type: "text"` (or untyped) blocks; other
block types are dropped at the translation boundary.

## GenerateOptions

```go
opts := anthropic.GenerateOptions(req)
for tok := range model.Chat(ctx, messages, opts...) { ... }
```

Lowers the sampler fields to `[]inference.GenerateOption`. `max_tokens` has no
default on the Anthropic side — `WithMaxTokens` is appended only when
`max_tokens > 0`.

## NewTextResponse

```go
resp := anthropic.NewTextResponse(requestID, modelName, text, metrics)
```

Builds a `MessageResponse` with a single `text` content block,
`stop_reason: "end_turn"`, and usage from the inference metrics. The
non-streaming handler uses it directly.

## Wire encoders

`AppendMessageResponse` / `AppendMessageRequest` hand-roll the response and
request JSON into a caller-owned buffer, staying off the `encoding/json` reflect
path at the HTTP-emit and client-encode boundaries. `MessageResponseSize` /
`MessageRequestSize` pre-size the buffer so the encode lands in one allocation.

## Streaming (`anthropic_stream.go`)

The streaming handler emits the full Anthropic SSE event sequence — Claude Code's
parser requires all of it:

```
message_start → content_block_start → content_block_delta* →
content_block_stop → message_delta → message_stop
```

(`ping` may interleave.) The `content_block_delta` events are the per-token hot
path (`text_delta`); `message_delta` carries the terminal `stop_reason`
(`end_turn`, or `stop_sequence` when a stop sequence matched) and the cumulative
`output_tokens`. Each event payload is built by the `Append*Event` builders in
this file. `MessageStopPayload` and `PingPayload` are the two fixed payloads.

## Related

- [../openai/openai.md](../openai/openai.md) — the parallel OpenAI Chat Completions server
- [../ollama/ollama.md](../ollama/ollama.md) — Ollama sibling
- [../inference/inference.md](../inference/inference.md) — base `Message` + `GenerateOption` types
