<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# openai/ — OpenAI-compatible wire types + HTTP handlers

**Package**: `dappco.re/go/inference/openai`

## What this package owns

Three things:

1. **Wire DTOs** for the OpenAI public API surface (Chat Completions, Responses, Embeddings, Rerank, Capabilities, Cache control, Cancel).
2. **Translation** between those DTOs and the `inference` package's runtime types (`Message`, `GenerateOption`, `CapabilityReport`, etc.).
3. **HTTP handlers** that wrap an `inference.TextModel` (or capability-extended variant) and serve OpenAI-compatible requests.

Drop-in compatible with any OpenAI SDK. Point the SDK at this handler's path and you get real local inference.

## File map

| File | Doc | Scope |
|------|-----|-------|
| `openai.go` | [openai.md](openai.md) | Chat Completions — DTOs + translation + Handler |
| `responses.go` | [responses.md](responses.md) | Responses API — DTOs + translation (handler TBD) |
| `services.go` | [services.md](services.md) | Embeddings / Rerank / Capabilities / Cache / Cancel handlers |

## Resolver contract

All handlers take a `Resolver` (defined in `openai.go`) — the indirection that maps a wire `model` field to a real `inference.TextModel`:

```go
type Resolver interface {
    ResolveModel(ctx, name) (inference.TextModel, error)
}
```

Three implementations ship in `openai.go`:

- `ResolverFunc` — inline closure
- `StaticResolver` — pre-loaded `map[string]TextModel`
- `BackendResolver` — lazy `inference.Backend.LoadModel(path)`

A custom Resolver is the right shape for:

- Quota-checked model dispatch (resolver rejects when quota exceeded)
- Per-user model gating
- Hot-swap (resolver looks up the current pin from config service)

## Why this package exists

The OpenAI wire format is **inference shape**, not provider policy. Any backend can serve it. Putting the DTOs + handlers + translation here gives go-mlx, go-rocm, and any future native driver an instant HTTP frontage without each one re-implementing the wire — and lets the outbound provider in `go-ai/providers/openai` use the same DTOs from the client side.

The opposite arrangement — DTOs in `go-ai` because OpenAI is "external" — would force every backend to depend on `go-ai`, which would then have to depend on every backend. The current shape keeps the dependency arrows pointing only **into** `inference`.

## Related

- [../inference/inference.md](../inference/inference.md) — `TextModel` + `Backend` interfaces
- [../inference/contracts.md](../inference/contracts.md) — `EmbeddingModel` / `RerankModel` / `CacheService` / `CancellableModel`
- [../inference/capability.md](../inference/capability.md) — `CapabilityReport` returned by `/v1/models/capabilities`
- [../anthropic/anthropic.md](../anthropic/anthropic.md) — sibling Anthropic wire types
- [../ollama/ollama.md](../ollama/ollama.md) — sibling Ollama wire types
- `go-ai/docs/providers/openai.md` (planned) — client-side outbound use of these DTOs
