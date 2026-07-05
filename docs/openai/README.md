<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# serving/provider/openai — OpenAI-compatible native server

**Package**: `dappco.re/go/inference/serving/provider/openai`

## What this package owns

Three things:

1. **Wire DTOs** for the OpenAI public API surface (Chat Completions, Responses,
   Embeddings, Rerank, Capabilities, Cache control, Cancel).
2. **Translation** between those DTOs and the `inference` runtime types
   (`Message`, `GenerateOption`, capability interfaces).
3. **HTTP handlers** that wrap an `inference.TextModel` and serve the requests
   from the LOCAL engine — decoding the OpenAI request format and emitting
   OpenAI-native JSON / SSE. These are native servers, not proxies to a remote
   vendor.

Point any OpenAI SDK at a mounted route and you get real local inference.

## File map

| File | Doc | Scope |
|------|-----|-------|
| `openai.go` + `request.go` + `handler.go` | [openai.md](openai.md) | Chat Completions — DTOs, translation, streaming + non-streaming handler |
| `content.go` | [openai.md](openai.md) | Multimodal content-part decoding (text + `data:` image parts) |
| `thinking.go` | [openai.md](openai.md) | `ThinkingExtractor` — reasoning-channel split into `thought` |
| `responses.go` | [responses.md](responses.md) | Responses API DTOs + translation |
| `services.go` | [services.md](services.md) | Embeddings / Rerank / Capabilities / Cache / Cancel handlers |
| `resolver.go` | [openai.md](openai.md) | `Resolver` implementations |
| `stops.go` / `chunkenc.go` | [openai.md](openai.md) | Stop-sequence truncation + hand-rolled wire encoders |

The chat-completions handler lives in this package (`handler.go`). The
`/v1/responses` handler is assembled in `serving/compat` (`mux.go`) over these
same DTOs; see [responses.md](responses.md).

## Route set (mounted by `serving/compat`)

`serving/compat.NewMux(resolver)` mounts the whole local-inference surface —
OpenAI, Anthropic, and Ollama — over one `Resolver`, and `cmd/lem serve` hosts it
(default `:36911`). The OpenAI routes:

```
POST /v1/chat/completions      chat (streaming + non-streaming)   openai.Handler
POST /v1/responses             Responses API (streaming + not)    compat handler
POST /v1/embeddings            embeddings                         EmbeddingsHandler
POST /v1/rerank                rerank                             RerankHandler
GET  /v1/models/capabilities   capability report (?model=X)       CapabilityHandler
GET  /v1/cache/stats           cache stats (?model=X)             CacheStatsHandler
POST /v1/cache/warm            warm cache                         CacheWarmHandler
POST /v1/cache/clear           clear cache                        CacheClearHandler
POST /v1/cancel                cancel an in-flight request        CancelHandler
```

`serving/compat` additionally mounts the Anthropic (`/v1/messages`), Ollama
(`/api/*`), and host admin routes (`/v1/health`, `/v1/runtime/wake`,
`/v1/runtime/sleep`, `/v1/cache/entries`).

## Resolver contract

Every handler takes a `Resolver` (defined in `resolver.go`) — the indirection
that maps a wire `model` field to a real `inference.TextModel`:

```go
type Resolver interface {
    ResolveModel(ctx, name) (inference.TextModel, error)
}
```

Three implementations ship in `resolver.go`:

- `ResolverFunc` — inline closure
- `StaticResolver` — pre-loaded `map[string]TextModel`
- `BackendResolver` — lazy `inference.LoadModel(path)`, cached

A custom Resolver is the right shape for quota-checked dispatch (reject when
quota exceeded), per-user model gating, or hot-swap (look up the current pin from
a config service on each request).

## Why the wire types live in `inference`, not a router

The OpenAI wire format is **inference shape**, not provider policy. Any backend
that satisfies the `inference` contracts can serve it, so the DTOs + handlers +
translation live next to the runtime. That keeps the dependency arrows pointing
only **into** `inference`: a host (`cmd/lem serve`, an embedding app, a test)
imports this package to get a drop-in HTTP surface, and this package imports
nothing above it.

go-inference is the sovereign inference repo — these servers compile and run from
go-inference alone.

## Related

- [../inference/inference.md](../inference/inference.md) — `TextModel` + `Backend` interfaces
- [../inference/contracts.md](../inference/contracts.md) — `EmbeddingModel` / `RerankModel` / `CacheService` / `CancellableModel`
- [../inference/capability.md](../inference/capability.md) — the capability report served on `/v1/models/capabilities`
- [../anthropic/anthropic.md](../anthropic/anthropic.md) — sibling Anthropic Messages server
- [../ollama/ollama.md](../ollama/ollama.md) — sibling Ollama server
