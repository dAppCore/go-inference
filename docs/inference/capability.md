<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# capability.go — capability reports + memory limiter

**Package**: `dappco.re/go/inference`
**File**: `go/capability.go`

## What this is

The portable shape for **"what does this backend / model support, at what maturity?"** — consumed by `agent/`, `serving/` and `eval/`. Backends that implement `CapabilityReporter` answer; consumers branch on the report without importing backend-specific packages.

Also hosts `RuntimeMemoryLimits` + `RuntimeMemoryLimiter` — the same lane for runtime allocator limits.

## Capability ID catalogue

54 stable IDs grouped by lane:

**Model / inference**: `model.load`, `generate`, `chat`, `classify`, `batch.generate`, `tokenizer`, `chat.template`, `lora.inference`, `lora.training`, `model.slice`

**Runtime / cache / scheduling**: `state.bundle`, `kv.snapshot`, `prompt.cache`, `kv.cache.planning`, `memory.planning`, `model.fit`, `runtime.discovery`, `runtime.autotune`, `model.replace`, `model.differential_load`, `model.split_inference`, `scheduler`, `request.cancel`, `cache.blocks`, `cache.disk`, `cache.warm`

**Training / eval**: `benchmark`, `evaluation`, `distillation`, `grpo`, `quantization`, `model.merge`

**Probe / research**: `probe.events`, `probe.attention`, `probe.logits`

**Query**: `query.lql`, `query.vindex`

**Wire / compat**: `responses.api`, `anthropic.messages`, `ollama.compat`, `embeddings`, `rerank`

**Parsers**: `tool.parse`, `reasoning.parse`

**Decoding**: `speculative.decode`, `prompt.lookup.decode`

**MoE / specialised quant**: `moe.routing`, `moe.lazy_experts`, `jangtq`, `codebook.vq`

**Agent memory**: `agent.memory`, `state.wake`, `state.sleep`, `state.fork`

## Groups + status

```go
type CapabilityGroup string  // "model" | "runtime" | "training" | "probe"
type CapabilityStatus string // "supported" | "experimental" | "planned" | "unsupported"
```

Group is a coarse routing dimension (a UI filter). Status is the maturity stamp.

## Capability

```go
type Capability struct {
    ID     CapabilityID
    Group  CapabilityGroup
    Status CapabilityStatus
    Detail string
    Labels map[string]string
}
```

Constructors short-cut the common shapes: `NewCapability(id, group, status, detail)` plus `SupportedCapability(id, group)`, `ExperimentalCapability(id, group, detail)`, `PlannedCapability(id, group, detail)`, and `UnsupportedCapability(id, group, detail)`. `Capability.Usable()` reports true for supported or experimental status.

## AlgorithmProfile

Richer than `Capability` — for backends that want to advertise the exact algorithm + which architectures it covers + what it requires + what it provides:

```go
type AlgorithmProfile struct {
    ID               CapabilityID
    Group            CapabilityGroup
    CapabilityStatus CapabilityStatus
    RuntimeStatus    FeatureRuntimeStatus  // native | experimental | metadata_only | planned
    Algorithm        string                // free-form: "jangtq_k", "flash_attn_v2", "paged_kv_v1"
    Detail           string
    Architectures    []string              // ["gemma4", "qwen3", "minimax_m2"]
    Requires         []CapabilityID
    Provides         []string
    Notes            []string
}
```

`profile.Capability()` lowers it to a plain `Capability` with the algorithm/architectures/requires/provides folded into labels for transport.

**Why two shapes?** `Capability` is the wire-stable contract — consumers depend on its small shape. `AlgorithmProfile` is the richer authoring shape backends use locally; lowering to Capability strips author detail to whatever the wire promises.

## CapabilityReport

```go
type CapabilityReport struct {
    Runtime       RuntimeIdentity
    Model         ModelIdentity
    Tokenizer     TokenizerIdentity
    Adapter       AdapterIdentity
    Available     bool
    Architectures []string
    Quantizations []string
    CacheModes    []string
    Capabilities  []Capability
    Labels        map[string]string
}
```

The full envelope: runtime + model + tokenizer + adapter identity, the available bit, lists of supported architectures / quantisations / cache modes, the capability array, plus free-form labels.

## CapabilityReporter

```go
type CapabilityReporter interface {
    Capabilities() CapabilityReport
}
```

Implemented by `Backend` (returns runtime-level capabilities) and by loaded `TextModel` instances (returns model-level capabilities). Consumers walk via type assertion — not every backend or model implements it. `CapabilitiesOf(value)` does the assertion for you, falling back to `BackendCapabilities` / `TextModelCapabilities` when the value doesn't implement `CapabilityReporter`. The report exposes query helpers: `Supports(id)`, `Capability(id)`, `SupportedCapabilityIDs()`, `CapabilityIDs()`.

## RuntimeMemoryLimits + RuntimeMemoryLimiter

```go
type RuntimeMemoryLimits struct {
    CacheLimitBytes          uint64
    MemoryLimitBytes         uint64
    PreviousCacheLimitBytes  uint64
    PreviousMemoryLimitBytes uint64
}

type RuntimeMemoryLimiter interface {
    SetRuntimeMemoryLimits(limits) RuntimeMemoryLimits
}

inference.SetRuntimeMemoryLimits("metal", limits)  // package-level helper
```

Zero request fields = "leave unchanged". Previous values report the prior caps so callers can restore on exit.

## Consumed by

- `engine/metal` — exposes Metal allocator limits via `RuntimeMemoryLimiter` and publishes JANG/MoE/codebook `AlgorithmProfile`s
- `engine/hip` — the AMD/ROCm engine's capability + memory-limit surface
- `serving/` — surfaces reports over HTTP for consumers to render the "what can I do" panel
- `agent/` + `eval/` — read the report to gate which scoring/eval features are available on the loaded model
