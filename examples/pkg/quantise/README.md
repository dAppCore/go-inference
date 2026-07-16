<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# quantise

Quantise a dense bf16/f32 safetensors model pack into any of lem's output
formats — the same calls `lem quant` wraps (cli/quant.go). Two families:

- gguf (default): the portable interchange lane for the llama.cpp
ecosystem. GGUF's quantisation schemes are named recipes (q4_k_m,
q8_0, ...), not a raw bit count, so -bits maps onto the nearest recipe;
`lem quant --gguf` takes the recipe name directly for full control.
- gptq | awq | fp8 | nf4: the HF-ecosystem exporters — GPTQ and AutoAWQ
GEMM packing for vLLM/TGI/ExLlama-class consumers, compressed-tensors
static E4M3, and bitsandbytes NF4 blockwise. GPTQ/AWQ honour -bits and
-group; fp8 and NF4 carry their formats' own fixed shapes.

The engine's own native format (MLX group-affine, both engines load it
directly) is `lem quant`'s default lane and lives in model/quant/mlxaffine.

## Run

```sh
go run ./pkg/quantise -src ~/models/gemma-4-e2b-it-bf16 -out /tmp/e2b-q4
go run ./pkg/quantise -format gptq -src ~/models/gemma-4-e2b-it-bf16 -out /tmp/e2b-gptq
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
