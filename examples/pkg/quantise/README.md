<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# quantise

Quantise a dense bf16/f32 safetensors model pack into GGUF — the same call
`lem quant -gguf <format>` wraps (go/cmd/lem/quant.go). GGUF's quantisation
schemes are named recipes (q4_k_m, q8_0, ...), not a raw bit count, so the
-bits flag here is a small convenience mapping onto the nearest recipe;
cmd/lem's own -gguf flag takes the recipe name directly for full control.

## Run

```sh
go run ./pkg/quantise -src ~/models/gemma-4-e2b-it-bf16 -out ~/models/gemma-4-e2b-it-gguf-q4
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
