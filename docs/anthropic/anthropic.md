<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# anthropic/anthropic.go — Messages API wire types

**Package**: `dappco.re/go/inference/anthropic`
**File**: `go/anthropic/anthropic.go`

## What this is

The Anthropic Messages API (`/v1/messages`) wire surface. Same pattern as `openai/openai.go` but for Anthropic-compatible SDKs — DTOs + translation to `inference.Message` + `inference.GenerateOption`. No HTTP handler yet; planned alongside the Responses handler.

This is a parity item from the 2026-05-09 vMLX gap report: vMLX exposed Anthropic compatibility and CoreAgent needed the same surface for Claude-flavoured SDKs hitting local inference.

## Constants

```go
const DefaultMessagesPath = "/v1/messages"
```

## DTOs

```go
ContentBlock     // type + text — Anthropic's typed-block content model
Message          // role + []ContentBlock
MessageRequest   // model + system + messages + max_tokens + sampler + stream + stop_sequences
Usage            // input_tokens + output_tokens
MessageResponse  // id + type + role + model + content[] + stop_reason + stop_sequence + usage
```

Key differences from OpenAI:

- `Message.Content` is `[]ContentBlock`, not a plain string — supports image / tool_use / tool_result block types out of the box.
- `system` is a top-level field, not a message with role=system.
- `Usage` uses `input_tokens` / `output_tokens` (vs OpenAI's `prompt_tokens` / `completion_tokens`).
- Stop reason is named (`end_turn` / `max_tokens` / `stop_sequence` / `tool_use`), not a free string.

## InferenceMessages

```go
messages := anthropic.InferenceMessages(req)
```

Flattens the typed-block content to plain text + builds the standard `inference.Message` slice. The Anthropic top-level `system` field becomes a leading system message in the inference slice — so the runtime sees one uniform message list regardless of API origin.

`blockText` strips down to `type: "text"` blocks only; image/tool blocks are dropped at the translation boundary (no multi-modal support in the core runner yet).

## GenerateOptions

```go
opts := anthropic.GenerateOptions(req)
for tok := range model.Chat(ctx, messages, opts...) { ... }
```

Same translation as the OpenAI sibling — sampler fields lowered to `inference.GenerateOption`. `MaxTokens` is required on the Anthropic side (no default); the translation only appends `WithMaxTokens` when `MaxTokens > 0`.

## NewTextResponse

```go
resp := anthropic.NewTextResponse(requestID, modelName, text, metrics)
```

Minimal response builder — single text content block + stop_reason="end_turn" + usage filled from the inference metrics. Same convenience as `openai.NewTextResponse`; lets a handler produce a valid Anthropic-shaped response in one line.

## What's not here

- Streaming. Anthropic's streaming format (`event: message_start`, etc.) is its own thing — not yet implemented.
- Tool-use / tool-result blocks. The shape is in `ContentBlock` but the translation drops them. When tool-call parsing lands (per the parity plan), this will route through `inference.ToolParser`.
- Vision blocks. Same reason as OpenAI Responses — multi-modal is out of scope for the core runner.

## Why a separate file from openai/

Anthropic's wire shape is **different enough** that mashing them into one package would require option types or interface-based content blocks — both worse than just having two parallel files. The size budget is small (~110 lines).

## Related

- [README.md](README.md) — package overview (planned)
- [../openai/openai.md](../openai/openai.md) — the parallel OpenAI translation
- [../inference/contracts.md](../inference/contracts.md) — `ToolParser` for future tool-use routing
- `core/api` — mounts an Anthropic handler when configured (handler TBD)
