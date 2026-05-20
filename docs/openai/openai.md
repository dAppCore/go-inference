<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# openai/openai.go — Chat Completions wire adapter

**Package**: `dappco.re/go/inference/openai`
**File**: `go/openai/openai.go`

## What this is

The OpenAI Chat Completions wire surface, adapted onto `inference.TextModel`. Three layers in one file:

1. **DTOs** — exact request/response shapes matching the OpenAI public API.
2. **Translation** — converting between the wire shape and `inference.GenerateOption` / `inference.Message`.
3. **HTTP handler** — `Handler` that resolves a model by name and streams completions.

Drop-in compatibility with OpenAI SDKs out of the box. A consumer points the SDK at this handler's path (`POST /v1/chat/completions`) and gets back real local inference — no SDK changes.

## DTOs (wire-exact)

```go
ChatCompletionRequest    // model + messages + sampler (all *T optional)
ChatMessage              // role + content
ChatCompletionResponse   // non-streaming response
ChatChoice               // index + message + finish_reason
ChatUsage                // prompt_tokens + completion_tokens + total_tokens
ChatCompletionChunk      // streaming SSE chunk
ChatChunkChoice          // streaming choice
ChatMessageDelta         // streaming delta (custom MarshalJSON)
ErrorResponse / ErrorObject
StopList                 // accepts either string or []string in JSON
```

## Defaults

```go
DefaultTemperature = 1.0
DefaultTopP        = 0.95
DefaultTopK        = 64
DefaultMaxTokens   = 2048
```

Used when the wire request has nil optional fields.

## DecodeRequest + ValidateRequest

```go
req, err := openai.DecodeRequest(r.Body)
err     := openai.ValidateRequest(req)
```

DecodeRequest handles the StopList polymorphism (string vs array). ValidateRequest checks required fields + sanity bounds.

## GenerateOptions

```go
opts, err := openai.GenerateOptions(req)
for tok := range model.Chat(ctx, messages, opts...) { ... }
```

Translates wire-typed sampler fields into a slice of `inference.GenerateOption`. Stop sequences are normalised to token-id stops where possible; freeform stop strings flow through a different path.

## NormalizeStopSequences

```go
ids, err := openai.NormalizeStopSequences(req.Stop)
```

Resolves OpenAI's stop strings against the model tokenizer where the tokenizer is available. Falls back to string-mode stop on streaming if the tokenizer can't pre-tokenise the sequence.

## Resolver

```go
type Resolver interface {
    ResolveModel(ctx, name) (inference.TextModel, error)
}
```

Three built-in implementations:

| Type | Use |
|------|-----|
| `ResolverFunc` | inline closure |
| `StaticResolver` | pre-loaded `map[string]TextModel` — model-picker UI, fixed deployments |
| `BackendResolver` | lazy load via `inference.Backend.LoadModel(path)` — cold-load on first request |

## Handler

```go
h := openai.NewHandler(resolver)
http.Handle("/v1/chat/completions", h)
```

Serves both streaming (`stream: true` → SSE) and non-streaming responses. Channel-marker (`<|channel>`) support lets reasoning channels flow into a separate stream key when the model emits thinking tokens.

## Why this lives in `inference` not in `go-ai`

The OpenAI wire format is **inference shape**, not provider policy. Any inference backend can be a server. go-ai's outbound provider (`go-ai/providers/openai`) uses the *same DTOs* for its **client** side — that's deliberate. The router (go-ai) owns policy (rate limits, fallback, quota); the wire (this package) owns the shape both sides agree on.

## Related

- [responses.md](responses.md) — newer `/v1/responses` API surface
- [services.md](services.md) — embeddings / rerank / cache / cancel handlers
- `go-ai/docs/providers/openai.md` — client-side outbound provider
- `core/api` — mounts this handler when `inference.api.openai = true`
