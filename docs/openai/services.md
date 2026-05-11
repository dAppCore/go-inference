<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# openai/services.go — embeddings / rerank / cache / cancel handlers

**Package**: `dappco.re/go/inference/openai`
**File**: `go/openai/services.go`

## What this is

The non-chat HTTP surface — eight handlers for the auxiliary OpenAI-compatible endpoints. Each handler probes the resolved model for the right interface (`EmbeddingModel`, `RerankModel`, `CacheService`, `CancellableModel`) and 501s if the backend doesn't support it.

Paths exposed:

```go
DefaultEmbeddingsPath   = "/v1/embeddings"
DefaultRerankPath       = "/v1/rerank"
DefaultCapabilitiesPath = "/v1/models/capabilities"
DefaultCacheStatsPath   = "/v1/cache/stats"
DefaultCacheWarmPath    = "/v1/cache/warm"
DefaultCacheClearPath   = "/v1/cache/clear"
DefaultCancelPath       = "/v1/cancel"
```

## Handlers

| Handler | Path | Backend interface needed |
|---------|------|--------------------------|
| `EmbeddingsHandler`  | `/v1/embeddings`             | `EmbeddingModel` |
| `RerankHandler`      | `/v1/rerank`                 | `RerankModel` |
| `CapabilityHandler`  | `/v1/models/capabilities`    | `CapabilityReporter` |
| `CacheStatsHandler`  | `/v1/cache/stats`            | `CacheService` |
| `CacheWarmHandler`   | `/v1/cache/warm`             | `CacheService` |
| `CacheClearHandler`  | `/v1/cache/clear`            | `CacheService` |
| `CancelHandler`      | `/v1/cancel`                 | `CancellableModel` |

Each constructed via `NewXxxHandler(resolver)` — the same `Resolver` interface used by the chat handler.

## DTOs

```go
EmbeddingRequest     // model + input + encoding_format + dimensions + normalize
EmbeddingInput       // string OR []string (custom UnmarshalJSON)
EmbeddingResponse    // object + data[] + model + usage
EmbeddingResponseDatum

RerankRequest        // model + query + documents + top_n
RerankResponse       // results[] (index + score + text)

CacheWarmRequest     // model + tokens or prompt + labels
CacheClearRequest    // labels filter
CancelRequest        // request id
```

The capability + cache-stats GET endpoints take no body — query string `?model=X` selects which loaded model to report on.

## EmbeddingInput polymorphism

OpenAI's embeddings API accepts either a single string or an array. The custom `UnmarshalJSON` on `EmbeddingInput` handles both. The Go-side always sees `[]string` — single-string inputs become a one-element slice.

## Shared handler scaffolding

```go
type serviceHandler struct{ resolver Resolver }

func (h *serviceHandler) resolve(...) (TextModel, bool)
func (h *serviceHandler) resolveCacheService(...) (CacheService, bool)
```

Each concrete handler embeds `serviceHandler` and gets the resolve helpers for free. The helper writes 4xx/5xx + JSON error responses when:

- Resolver returns "model not found"
- Model doesn't satisfy the required capability interface
- Decode / validation fails

## Why these are HTTP-shape primitives

The runtime *interfaces* (`EmbeddingModel`, `RerankModel`, `CacheService`, `CancellableModel`) live in `inference/contracts.go`. This file is **just the wire layer** on top — turning HTTP requests into runtime calls and runtime results into HTTP responses.

A non-HTTP transport (Unix socket, gRPC, MCP tool call) can use the same interfaces without involving this file. Conversely, an OpenAI-compatible server that wants the wire compatibility without going through the runtime contract can crib the DTOs here.

## What's not here

- `/v1/audio/transcriptions` — vMLX exposed it; we don't have audio runtime support yet (out of scope for the core runner)
- `/v1/images/generations` — same reason
- `/v1/files` — bundle-as-file maps onto agent memory, but the wire mapping isn't designed yet
- Speech endpoints — see `/v1/audio` note

## Related

- [openai.md](openai.md) — Chat Completions handler
- [responses.md](responses.md) — Responses API DTOs
- [../inference/contracts.md](../inference/contracts.md) — `EmbeddingModel` / `RerankModel` / `CacheService` / `CancellableModel`
- [../inference/capability.md](../inference/capability.md) — `CapabilityReport` returned by the capability handler
- `core/api` — mounts these handlers when configured
