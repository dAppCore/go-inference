<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# serving/provider/openai/services.go — embeddings / rerank / cache / cancel handlers

**Package**: `dappco.re/go/inference/serving/provider/openai`
**File**: `go/serving/provider/openai/services.go`

## What this is

The non-chat HTTP surface — seven handlers for the auxiliary OpenAI-compatible
endpoints. Each handler resolves the model, probes it for the interface the
endpoint needs (`EmbeddingModel`, `RerankModel`, `CapabilityReporter`,
`CacheService`, `CancellableModel`), and returns `501 Not Implemented` when the
backend doesn't satisfy it.

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

| Handler | Path | Method | Backend interface needed |
|---------|------|--------|--------------------------|
| `EmbeddingsHandler`  | `/v1/embeddings`             | POST | `EmbeddingModel` |
| `RerankHandler`      | `/v1/rerank`                 | POST | `RerankModel` |
| `CapabilityHandler`  | `/v1/models/capabilities`    | GET  | `CapabilityReporter` (falls back to a computed report) |
| `CacheStatsHandler`  | `/v1/cache/stats`            | GET  | `CacheService` |
| `CacheWarmHandler`   | `/v1/cache/warm`             | POST | `CacheService` |
| `CacheClearHandler`  | `/v1/cache/clear`            | POST | `CacheService` |
| `CancelHandler`      | `/v1/cancel`                 | POST | `CancellableModel` |

Each is constructed via `NewXxxHandler(resolver)` — the same `Resolver` interface
the chat handler uses. `CapabilityHandler` is the one that never 501s: when the
model isn't a `CapabilityReporter` it returns a report computed from the model's
declared interfaces via `inference.TextModelCapabilities`.

## DTOs

```go
EmbeddingRequest     // model + input + encoding_format + dimensions + user + normalize
EmbeddingInput       // string OR []string (custom UnmarshalJSON)
EmbeddingResponse    // object + data[] + model + usage
EmbeddingResponseDatum   // object + index + embedding []float32

RerankRequest        // model + query + documents + top_n
RerankResponse       // object + model + results[] (inference.RerankScore)

CacheWarmRequest     // model + prompt OR tokens ([]int32) + mode + labels
CacheClearRequest    // model + labels filter
CancelRequest        // model + id
```

The capability and cache-stats GET endpoints take no body — a `?model=X` query
string selects which loaded model to report on.

## EmbeddingInput polymorphism

OpenAI's embeddings API accepts either a single string or an array. The custom
`UnmarshalJSON` on `EmbeddingInput` handles both in a single pass. The Go side
always sees `[]string` — a single-string input becomes a one-element slice.

## Validation

Handlers reject with `400` before touching the model: embeddings require
`model` + non-empty `input`; rerank requires `model` + `query` + non-empty
`documents`; cancel requires `id`. A missing/unsupported capability returns
`501`; a resolver "model not found" returns `404`.

## Why these are HTTP-shape primitives

The runtime *interfaces* (`EmbeddingModel`, `RerankModel`, `CacheService`,
`CancellableModel`) live in `inference/contracts.go`. This file is **just the
wire layer** on top — turning HTTP requests into runtime calls and runtime
results into HTTP responses. A non-HTTP transport (Unix socket, MCP tool call)
can drive the same interfaces without this file.

## What's not here

- `/v1/audio/transcriptions`, `/v1/audio/*` — no audio runtime support yet.
- `/v1/images/generations` — same reason.
- `/v1/files` — the wire mapping onto agent memory bundles isn't designed yet.

## Related

- [openai.md](openai.md) — Chat Completions handler
- [responses.md](responses.md) — Responses API surface
- [../inference/contracts.md](../inference/contracts.md) — `EmbeddingModel` / `RerankModel` / `CacheService` / `CancellableModel`
- [../inference/capability.md](../inference/capability.md) — the capability report served by `CapabilityHandler`
