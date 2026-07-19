<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# kvcache-turboquant

TurboQuant live KV cache versus the native bf16 cache: the same model, the
same greedy prompt, loaded twice — once default and once with
`inference.WithCacheMode("turboquant:3.5")` (K codes at 4 bits, V at 3; the
engine's global attention layers hold packed Lloyd-Max centroid codes + one
f32 norm per row per head instead of bf16 rows). The program prints each
mode's decode tok/s and the global-layer KV bytes per token — the residency
win is the point. The codes are lossy by design; modes: `turboquant` (=3.5),
`turboquant:4`, `turboquant:3.5`, `turboquant:3`, `turboquant:2`.

## Run

```sh
go run ./pkg/kvcache-turboquant -model ~/models/gemma-4-e2b-4bit
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
