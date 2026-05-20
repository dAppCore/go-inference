<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# state/identity.go — portable identity DTOs

**Package**: `dappco.re/go/inference/state`
**File**: `go/state/identity.go`
**Aliased into**: `dappco.re/go/inference` (via `identity.go` —
`inference.ModelIdentity` etc. are aliases of these types)

## What this is

Six DTOs that travel with every durable artefact in the system:

| Type | What it identifies |
|------|--------------------|
| `ModelIdentity` | which model produced/expects this — hash, arch, quant, ctx-len |
| `TokenizerIdentity` | which tokenizer + chat template — BOS/EOS/PAD ids, template hash |
| `AdapterIdentity` | which LoRA/adapter is active — hash, rank, alpha, target keys, base-model hash |
| `RuntimeIdentity` | which runtime/device produced it — backend name, device, version, cache mode |
| `SamplerConfig` | reproducible sampling — temp, top-k, top-p, repeat penalty, stop tokens |
| `StateRef` | typed reference to one external blob — kind, URI, hash, size, encoding |

Plus the envelope:

| Type | Role |
|------|------|
| `Bundle` (`StateBundle` alias) | the full state envelope a sleep emits — model + tokenizer + adapter + sampler + runtime + prompt hash + KV refs + probe refs + memvid refs + labels |

## Why these are separate from `state/agent_memory.go`

Agent memory is about lifecycle (Wake/Sleep/Fork). Identity is about
**compatibility checking** at lifecycle boundaries:

- A wake refuses to restore a Gemma-3 bundle into a Gemma-4 session
  (model arch differs).
- A wake refuses to restore an adapter-on bundle into an adapter-off
  session (`AdapterIdentity.Hash` differs).
- A wake records which runtime produced the bundle so audit can trace
  divergent results back to "this bundle came from go-rocm vs go-mlx".

`Bundle.KVRefs` / `ProbeRefs` / `MemvidRefs` are arrays of `StateRef`
because one bundle commonly fans out to multiple blobs — KV blocks are
chunked, probes are per-layer, memvid frames are sequenced.

## Why `ModelIdentity.Hash` is load-bearing

The hash is what `WakeRequest.SkipCompatibilityCheck` flips off. By
default a wake compares `req.Model.Hash` to `bundle.Model.Hash` and
rejects on mismatch — even if the architecture matches, a quantisation
re-pack or weight delta produces a different hash and would silently
corrupt KV.

Hash format is backend-defined (typically SHA-256 of safetensor index
file + adapter file), but the contract is "same hash → same weights →
KV is valid".

## SamplerConfig <-> GenerateConfig

The `state` package keeps the portable `SamplerConfig` shape. The
`inference` parent package converts to/from its richer
`GenerateConfig` (which includes `GenerateOption` plumbing) via two
free functions in `inference/identity.go`:

```go
inference.SamplerConfigFromGenerateConfig(cfg) → SamplerConfig
inference.GenerateConfigFromSamplerConfig(cfg) → GenerateConfig
```

This is deliberate — the bundle stores the **outcome** of the option
choices, not the option-function chain.

## Used by

- `state/agent_memory.go` — `Ref` carries `StateRefs []StateRef`
- `state/store.go` — chunk metadata
- `go-mlx/state_bundle.go` — bundle encode/decode
- `go-mlx/kv_snapshot.go` — snapshot/restore stores Bundle alongside KV
  blocks
- `go-ml/agent_eval.go` — eval reports embed `ModelIdentity` +
  `AdapterIdentity` for reproducibility
- `core/api` benchmark surface — bench reports carry `RuntimeIdentity`
