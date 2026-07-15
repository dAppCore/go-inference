<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# serving/provider/openai/responses.go — Responses API surface

**Package**: `dappco.re/go/inference/serving/provider/openai`
**File**: `go/serving/provider/openai/responses.go`
**Route**: `POST /v1/responses`

## What this is

The OpenAI **Responses API** (`/v1/responses`) wire types and translation. Same
pattern as Chat Completions — DTOs + an `inference.Message` adapter + a
`GenerateOption` builder. This file holds the DTOs and translation; the HTTP
handler is assembled in `serving/compat` (`mux.go`, `openAIResponsesHandler`)
over these types and mounted by `cmd/lem serve` alongside the other routes.

This is a **minimal** Responses shape, not the full typed-item variant of
OpenAI's API: input items are plain `{role, content}` messages, and
`instructions` maps to a leading system message. There are no typed multimodal
input items, tool-result items, or server-side previous-response state.

## DTOs

```go
ResponseInputMessage   // {role, content} input item
ResponseRequest        // model + input[] + instructions + sampler + stream + stop
ResponseOutputText     // {type:"output_text", text}
ResponseOutputMessage  // typed assistant message with a content[] of output_text
ResponseUsage          // input_tokens + output_tokens + total_tokens
Response               // non-streaming body (id + object + created + model + output[] + usage + thought?)
ResponseStreamEvent    // streaming event (type + response? + delta + thought?)
```

`ResponseRequest` models:

```
model, input[], instructions, temperature, top_p, min_p, top_k,
max_output_tokens, stream, stop, user
```

`min_p` is the gemma4 sampling extension. Note `max_output_tokens` (the Responses
name) maps to the same cap as chat's `max_tokens`.

Reasoning is surfaced as an optional `thought` string on `Response` /
`ResponseStreamEvent` — not as a separate token count. `ResponseUsage` carries
`input_tokens` / `output_tokens` / `total_tokens` only.

## Translation

```go
messages   := openai.ResponseMessages(req)          // input items → inference.Message
opts, err  := openai.ResponseGenerateOptions(req)   // sampler → GenerateOption
```

`ResponseMessages` prepends `instructions` as a `system` message, then maps each
`input` item to an `inference.Message`. `ResponseGenerateOptions` folds the
request into a `ChatCompletionRequest` and reuses `GenerateOptions`, so the
Responses and Chat Completions surfaces share one sampler-translation path.

## NewTextResponse

```go
resp := openai.NewTextResponse(requestID, modelName, text, metrics)
```

The minimal builder — produces a complete `Response` with one output message
containing one `output_text` segment, plus usage filled from the inference
metrics. The non-streaming handler uses it directly.

## Handler behaviour (in `serving/compat`)

- **Non-streaming** — collects tokens, splits reasoning from the visible answer,
  truncates at any stop sequence, and returns a `Response`. A non-empty reasoning
  channel is attached as `thought`.
- **Streaming** — emits `response.created`, then `response.output_text.delta`
  per visible-text delta, then `response.completed` (carrying the full
  `Response`), then `data: [DONE]`. A generation error emits `response.error`.
  Reasoning extraction runs through `decode/parser`'s processor.

## Why Responses vs Chat Completions

OpenAI introduced Responses to express things Chat Completions can't cleanly —
typed inputs, tool results as input items, server-side state. This local
implementation adopts the route and the response envelope; the input side stays
minimal (role/content messages + instructions) until a typed-item consumer needs
more.

## Related

- [openai.md](openai.md) — Chat Completions counterpart
- [services.md](services.md) — embeddings / rerank / cache / cancel handlers
- [../inference/contracts.md](../inference/contracts.md) — the reasoning parser contract behind the `thought` split
