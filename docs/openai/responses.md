<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# openai/responses.go — Responses API wire shapes

**Package**: `dappco.re/go/inference/openai`
**File**: `go/openai/responses.go`

## What this is

The OpenAI **Responses API** (`/v1/responses`) wire types — a newer, more structured alternative to Chat Completions that treats inputs as typed items and outputs as typed messages. Same translation pattern as Chat Completions: DTOs + `inference.Message` adapter + `inference.GenerateOption` builder.

This is a parity item from the 2026-05-09 vMLX gap report; vMLX exposed `/v1/responses` and CoreAgent needed the same surface for SDK compatibility.

## DTOs

```go
ResponseInputMessage   // structured input item (text / image / tool result / …)
ResponseRequest        // model + input items + sampler + tools + reasoning hints
ResponseOutputText     // typed text segment
ResponseOutputMessage  // typed assistant message with output_text array
ResponseUsage          // input_tokens + output_tokens + reasoning_tokens
Response               // non-streaming response (id + model + output[] + usage)
ResponseStreamEvent    // streaming event (event_type + payload)
```

The Responses API distinguishes **visible text** from **reasoning text** at the wire level — `ResponseUsage.ReasoningTokens` is its own count. This pairs cleanly with the `ReasoningParser` interface in `contracts.go` — backends that emit reasoning channels feed them through as separate output items.

## Translation

```go
messages := openai.ResponseMessages(req)          // flatten input items to inference.Message
opts, err := openai.ResponseGenerateOptions(req)  // sampler → GenerateOption
```

`ResponseMessages` walks `req.Input[]`, extracting text content and converting role + content per item. Tool-result items map to `Role: "tool"` messages.

`ResponseGenerateOptions` follows the same logic as `GenerateOptions` in `openai.go` — the Responses API and Chat Completions accept the same sampler set.

## NewTextResponse

```go
resp := openai.NewTextResponse(requestID, modelName, text, metrics)
```

The minimal builder — produces a complete `Response` with one output message containing one text segment. Used by the handler to serialise the simple non-streaming path. Streaming responses build `ResponseStreamEvent` chunks instead.

## Why Responses vs Chat Completions

OpenAI introduced Responses because Chat Completions can't cleanly express:

- Multi-modal inputs (image + text in the same turn)
- Tool-call results as typed input items, not assistant turns
- Reasoning tokens billed separately from output tokens
- Server-side state (response references the previous response)

Local CoreAgent inference benefits from the same shape — reasoning channels are first-class, tool results flow without role abuse, server-state can be tied to wake/sleep bundles.

## Where the handler lives

The Responses HTTP handler is currently not in this file (the Chat Completions handler in `openai.go` is the only HTTP entry). A Responses-specific handler is on the parity-plan roadmap; the DTOs are in place so once the handler lands, the SDK side already compiles.

## Related

- [openai.md](openai.md) — Chat Completions counterpart
- [services.md](services.md) — embeddings/rerank/cache/cancel handlers
- [../inference/contracts.md](../inference/contracts.md) — `ReasoningParser` for emitting reasoning channels
- `go-mlx/docs/inference/thinking.md` (planned) — reasoning parser implementation
