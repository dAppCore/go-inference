<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# serving/provider/openai — Chat Completions native server

**Package**: `dappco.re/go/inference/serving/provider/openai`
**Route**: `POST /v1/chat/completions`

## What this is

A **native** OpenAI Chat Completions server: it decodes the OpenAI wire request,
runs the request against the LOCAL engine (`inference.TextModel`), and emits the
OpenAI-native response — a JSON `chat.completion` body, or a `text/event-stream`
of `chat.completion.chunk` SSE frames when `stream: true`. It is not a proxy to
a remote OpenAI endpoint; the only inference that happens is local.

Point any OpenAI SDK at this route and it gets real local inference with no SDK
changes. The route is mounted onto the shared multi-protocol mux by
`serving/compat` (`NewMux` / `NewModelMux`) and served by `cmd/lem serve`
(default `:36911`).

The chat-completions surface spans several files in this package:

| File | Scope |
|------|-------|
| `openai.go` | Request/response/chunk DTOs + `ChatMessageDelta.MarshalJSON` |
| `request.go` | `DecodeRequest`, `ValidateRequest`, `GenerateOptions`, `NormalizeStopSequences` |
| `handler.go` | `Handler` — the `net/http` streaming + non-streaming entry point |
| `content.go` | Multimodal content-part decoding (text + image parts) |
| `thinking.go` | `ThinkingExtractor` — reasoning-channel split |
| `stops.go` | `TruncateAtStopSequence` |
| `chunkenc.go` | Hand-rolled SSE / response encoders (off the reflect path) |
| `resolver.go` | `Resolver` implementations |

## DTOs (`openai.go`)

```go
ChatCompletionRequest    // model + messages + sampler + gemma4 extensions
ChatMessage              // role + content (string OR multimodal parts) + decoded Images
ChatTemplateKwargs       // enable_thinking + thinking_budget
ChatCompletionResponse   // non-streaming; carries optional "thought"
ChatChoice               // index + message + finish_reason
ChatUsage                // prompt_tokens + completion_tokens + total_tokens
ChatCompletionChunk      // streaming SSE chunk; carries optional "thought"
ChatChunkChoice          // streaming choice
ChatMessageDelta         // streaming delta (hand-rolled MarshalJSON)
ErrorResponse / ErrorObject
StopList                 // accepts either a JSON string or []string
```

`ChatCompletionRequest` models these fields:

```
model, messages, temperature, top_p, min_p, top_k, max_tokens,
stream, stop, user, reasoning_effort, chat_template_kwargs
```

- `min_p` is a gemma4 sampling extension (0 = disabled).
- `reasoning_effort: "none"` disables the thinking channel.
- `chat_template_kwargs` follows the vLLM/SGLang convention:
  `enable_thinking` (bool) and `thinking_budget` (int; 0/absent = unlimited).
  Unknown keys in the object are skipped by the decoder.

## Defaults

```go
DefaultTemperature = 1.0
DefaultTopP        = 0.95
DefaultTopK        = 64
DefaultMaxTokens   = 2048
```

Applied by `GenerateOptions` when the request leaves the matching optional field
nil. `min_p` has no named constant — its fallback is `0` (disabled).

## DecodeRequest + ValidateRequest (`request.go`)

```go
req, err := openai.DecodeRequest(r.Body)
err     := openai.ValidateRequest(req)
```

`DecodeRequest` reads the body and unmarshals `ChatCompletionRequest`, resolving
`StopList` (string vs array) and the polymorphic `ChatMessage.Content`.
`ValidateRequest` checks required fields and sanity bounds: `model` non-empty;
`messages` non-empty; each role one of `system`/`developer`/`user`/`assistant`/`tool`;
`temperature` in [0,2]; `top_p`, `min_p` in [0,1]; `top_k`, `max_tokens` >= 0.

## GenerateOptions

```go
opts, err := openai.GenerateOptions(req)
for tok := range model.Chat(ctx, messages, opts...) { ... }
```

Translates the wire sampler fields into `[]inference.GenerateOption`
(`WithTemperature` / `WithTopP` / `WithMinP` / `WithTopK` / `WithMaxTokens`), then
appends a thinking toggle and a thinking budget when the request carries them.
`thinkingOverride` resolves the toggle: `chat_template_kwargs.enable_thinking`
wins when present; otherwise `reasoning_effort == "none"` disables thinking; nil
means the model default is left in place.

## NormalizeStopSequences

```go
stops, err := openai.NormalizeStopSequences(req.Stop)
```

Trims each stop string and rejects empty ones. The result is used at the response
boundary — `TruncateAtStopSequence` (`stops.go`) cuts generated content at the
first matching sequence, and the streaming path stops emitting once a stop cut is
reached.

## Multimodal content (`content.go`)

`ChatMessage.Content` accepts both shapes:

```jsonc
{"role":"user","content":"plain text"}
{"role":"user","content":[
    {"type":"text","text":"What is in this image?"},
    {"type":"image_url","image_url":{"url":"data:image/png;base64,…"}}]}
```

Text parts concatenate into `Content` (newline-joined); `image_url` parts decode
into `ChatMessage.Images` and never round-trip into responses. Only base64
`data:` URLs are accepted — this is a local engine, so a remote image URL in a
prompt is **refused, not fetched** (no SSRF surface, no silent network I/O).
Caps: 16 images per request, 32 MiB per decoded image. When a request carries
images the handler requires the resolved model to satisfy `inference.VisionModel`
and accept images, else it returns 400.

## Reasoning-channel split (`thinking.go`)

`ThinkingExtractor` separates model-internal reasoning from assistant content in
the streamed token sequence, so reasoning lands in the response's `thought` field
rather than in the visible answer. It recognises:

- The gemma4 / gpt-oss channel markers: `<|channel><name>` open and the gemma4
  `<channel|>` explicit close (after which the remaining tokens are the visible
  answer).
- Paired reasoning tags: `<think>`, `<thinking>`, `<thought>`, `<reasoning>`.

Markers straddling a token boundary are held back and re-joined, so a marker
split across two tokens is never mis-emitted.

## Resolver (`resolver.go`)

```go
type Resolver interface {
    ResolveModel(ctx, name) (inference.TextModel, error)
}
```

Three implementations ship here:

| Type | Use |
|------|-----|
| `ResolverFunc` | inline closure |
| `StaticResolver` | pre-loaded `map[string]TextModel` — model-picker UI, fixed deployments |
| `BackendResolver` | lazy `inference.LoadModel(path)` — cold-load on first request, cached under a mutex |

`serving/compat.NewResolver` wraps a `BackendResolver` pinned to the `"metal"`
backend for `cmd/lem serve`.

## Handler (`handler.go`)

```go
h := openai.NewHandler(resolver)
http.Handle("/v1/chat/completions", h)
```

`Handler` serves both paths from one entry point:

- **Non-streaming** — runs `model.Chat`, drains the `ThinkingExtractor`, and
  returns a `chat.completion` body. `finish_reason` is `"stop"`, or `"length"`
  when the generated-token count reaches `max_tokens`. A non-empty reasoning
  channel is attached as `thought`.
- **Streaming** — sets `text/event-stream` and emits: a role-priming chunk, then
  one `chat.completion.chunk` per content/thought delta, then a final chunk with
  `finish_reason` (`"stop"` / `"length"` / `"error"`), then `data: [DONE]`.
  Each SSE frame is built by the hand-rolled `chunkenc.go` encoders to stay off
  the `encoding/json` reflect path on the per-token hot loop.

## Related

- [README.md](README.md) — package overview + full route set
- [responses.md](responses.md) — the `/v1/responses` surface
- [services.md](services.md) — embeddings / rerank / capabilities / cache / cancel handlers
- [../anthropic/anthropic.md](../anthropic/anthropic.md) — Anthropic Messages sibling
- [../ollama/ollama.md](../ollama/ollama.md) — Ollama sibling
- [../inference/inference.md](../inference/inference.md) — `TextModel` + `Backend` interfaces
