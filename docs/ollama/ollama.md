<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# ollama/ollama.go — Ollama-compatible wire types

**Package**: `dappco.re/go/inference/ollama`
**File**: `go/ollama/ollama.go`

## What this is

The Ollama-compatible API wire surface — DTOs for `/api/chat`, `/api/generate`, `/api/tags`, `/api/show` plus translation to `inference.Message` + `inference.GenerateOption`. Same pattern as the OpenAI and Anthropic sibling packages.

Used by tools and IDE plugins that talk to Ollama natively (Continue, Cody, Cline, the Codex `ollama` profile) — when this surface is mounted by core/api, those tools find a local model server transparent to "is this real Ollama or core?"

## Paths

```go
DefaultChatPath     = "/api/chat"
DefaultGeneratePath = "/api/generate"
DefaultTagsPath     = "/api/tags"
DefaultShowPath     = "/api/show"
```

## DTOs

```go
Message              // role + content (plain string, unlike Anthropic's typed blocks)
Options              // temperature + top_k + top_p + num_predict
ChatRequest          // model + messages + stream + options
GenerateRequest      // model + prompt + stream + options
ChatResponse         // model + message + done + prompt_eval_count + eval_count + durations (nanos)
GenerateResponse     // model + response (text) + done + counters + durations
ModelTag             // name + model + modified_at + size
TagsResponse         // models[]
ShowRequest          // model
ShowResponse         // license + modelfile + parameters + template + details
```

Two response timing peculiarities to know:

- Durations are **int64 nanoseconds**, not floats / seconds.
- `prompt_eval_count` = prompt tokens, `eval_count` = generated tokens (different field names from OpenAI / Anthropic).

## InferenceMessages

```go
messages := ollama.InferenceMessages(req.Messages)
```

Straight 1:1 map. Ollama's message shape matches `inference.Message` directly so the conversion is a slice rebuild.

## GenerateOptions

```go
opts := ollama.GenerateOptions(req.Options)
for tok := range model.Chat(ctx, messages, opts...) { ... }
```

Translates Ollama's sampler set. `num_predict` becomes `WithMaxTokens` — the Ollama name reflects its llama.cpp lineage.

## NewChatResponse + NewGenerateResponse

```go
chatResp := ollama.NewChatResponse(modelName, text, metrics)
genResp  := ollama.NewGenerateResponse(modelName, text, metrics)
```

Convenience builders. `Done: true` always set — they produce single-shot responses, not streaming chunks. Streaming responses build per-chunk shapes inline at the handler.

## /api/tags + /api/show

`TagsResponse` mirrors the model picker — backends that implement model listing can serve this from their inventory. `ShowResponse` carries Ollama's "model details" payload (license / template / parameters) which map onto `ModelIdentity` + `TokenizerIdentity.ChatTemplate`.

These two endpoints are read-only meta queries, no inference work — making them easy to satisfy from a backend's `Discover()` + `Inspect()` results.

## What's not here

- `/api/pull`, `/api/push`, `/api/copy`, `/api/delete` — model management. CoreAgent's model store has different semantics (State bundles vs Ollama tags). Not a wire-parity target.
- `/api/embeddings` — Ollama has it; CoreAgent serves embeddings via the OpenAI `/v1/embeddings` path instead.
- HTTP handler. As with `anthropic.go`, the wire DTOs are in place; the handler is roadmap.

## Why three sibling files, not one mega-package

The temptation is a single `wire` package with `wire.OpenAIChat`, `wire.AnthropicMessages`, `wire.OllamaChat`. We resist for three reasons:

1. **Naming friction** — `wire.MessageRequest` is ambiguous; `anthropic.MessageRequest` isn't.
2. **Import economy** — a server that only exposes the OpenAI surface shouldn't compile Anthropic + Ollama into its binary.
3. **Independent evolution** — each upstream API changes on its own clock; isolated packages let us track each without cross-touch.

## Related

- [../openai/openai.md](../openai/openai.md) — OpenAI sibling
- [../anthropic/anthropic.md](../anthropic/anthropic.md) — Anthropic sibling
- [../inference/inference.md](../inference/inference.md) — base `Message` + `GenerateOption` types
- [../inference/capability.md](../inference/capability.md) — `CapabilityOllamaCompat` declares this surface
