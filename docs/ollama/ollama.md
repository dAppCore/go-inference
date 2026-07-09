<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# serving/provider/ollama — Ollama-compatible native server

**Package**: `dappco.re/go/inference/serving/provider/ollama`
**Routes**: `/api/chat`, `/api/generate`, `/api/tags`, `/api/show`

## What this is

A **native** Ollama-compatible server — DTOs, translation, and (assembled in
`serving/compat`) the HTTP handlers for `/api/chat`, `/api/generate`,
`/api/tags`, and `/api/show`. It decodes the Ollama wire request and serves it
from the LOCAL engine, emitting Ollama-native JSON or the Ollama NDJSON stream.
Not a proxy to a real Ollama daemon.

Tools and IDE plugins that speak Ollama natively (Continue, Cody, Cline, and the
like) find a local model server transparent to "is this real Ollama or not?" The
routes are mounted by `serving/compat` (`mux.go`) and served by `cmd/lem serve`
(default `:36911` — Lethean's own port, so an Ollama install on `11434` never
collides).

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
Options              // temperature + top_k + top_p + min_p + num_predict
ChatRequest          // model + messages + stream + options
GenerateRequest      // model + prompt + stream + options
ChatResponse         // model + message + done + prompt_eval_count + eval_count + durations (nanos)
GenerateResponse     // model + response (text) + done + counters + durations
ModelTag             // name + model + modified_at + size
TagsResponse         // models[]
ShowRequest          // model
ShowResponse         // license + modelfile + parameters + template + details
```

`Options` carries `min_p` alongside the standard Ollama sampler set — the gemma4
sampling extension. Two response timing peculiarities:

- Durations are **int64 nanoseconds**, not floats / seconds.
- `prompt_eval_count` = prompt tokens, `eval_count` = generated tokens (different
  field names from OpenAI / Anthropic).

## InferenceMessages

```go
messages := ollama.InferenceMessages(req.Messages)
```

Straight 1:1 map — Ollama's message shape matches `inference.Message` directly.

## GenerateOptions

```go
opts := ollama.GenerateOptions(req.Options)
```

Translates Ollama's sampler set into one fused `GenerateOption`. `num_predict`
becomes `WithMaxTokens` (the name reflects its llama.cpp lineage). An all-zero
`Options` returns nil so callers skip a no-op option pass.

## NewChatResponse + NewGenerateResponse

```go
chatResp := ollama.NewChatResponse(modelName, text, metrics)
genResp  := ollama.NewGenerateResponse(modelName, text, metrics)
```

Convenience builders for the terminal frame. `Done: true` is always set — these
are the single-shot / stream-summary shapes, filled with `prompt_eval_count` and
`eval_count` from the metrics. Responses carry the visible answer only; any
reasoning channel is stripped (the Ollama wire has no separate thought field).

## Streaming

Both `/api/chat` and `/api/generate` stream **NDJSON** (`application/x-ndjson`):
one JSON object per generated token with `done: false`, then a terminal summary
frame carrying `done: true` and the metric counters. The per-token frames are
built by hand-rolled encoders (in `serving/compat`) to stay off the reflect path.

## /api/tags + /api/show

`/api/tags` lists the resolver's model names as `ModelTag` entries. `/api/show`
returns the model's `details` — `architecture`, `model_type`, and (when known)
`quantization` — derived from the model's `Info()`. Both are read-only meta
queries with no inference work.

## What's not here

- `/api/pull`, `/api/push`, `/api/copy`, `/api/delete` — model management. The
  model store has different semantics (State bundles vs Ollama tags); not a
  wire-parity target.
- `/api/embeddings` — embeddings are served via the OpenAI `/v1/embeddings` path
  instead.

## Why three sibling packages, not one mega-package

A single `wire` package with `wire.OpenAIChat` / `wire.AnthropicMessages` /
`wire.OllamaChat` was resisted for three reasons:

1. **Naming friction** — `wire.MessageRequest` is ambiguous; `ollama.ChatRequest`
   isn't.
2. **Import economy** — a server exposing only the Ollama surface shouldn't
   compile the OpenAI + Anthropic packages into its binary.
3. **Independent evolution** — each upstream API changes on its own clock;
   isolated packages track each without cross-touch.

## Related

- [../openai/openai.md](../openai/openai.md) — OpenAI sibling
- [../anthropic/anthropic.md](../anthropic/anthropic.md) — Anthropic sibling
- [../inference/inference.md](../inference/inference.md) — base `Message` + `GenerateOption` types
- [../inference/capability.md](../inference/capability.md) — capability report covering the Ollama surface
