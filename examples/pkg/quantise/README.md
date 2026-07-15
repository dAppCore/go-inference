# quantise

Quantise a dense bf16/f32 safetensors pack into any of lem's output formats —
the library calls `lem quant` wraps.

```sh
# GGUF (default) — the portable interchange lane (llama.cpp ecosystem)
go run ./pkg/quantise -src ~/models/gemma-4-e2b-it-bf16 -out /tmp/e2b-q4.gguf

# HF-ecosystem exporters — GPTQ / AutoAWQ / compressed-tensors fp8 / bitsandbytes NF4
go run ./pkg/quantise -format gptq -src ~/models/gemma-4-e2b-it-bf16 -out /tmp/e2b-gptq
go run ./pkg/quantise -format awq  -src ~/models/gemma-4-e2b-it-bf16 -out /tmp/e2b-awq
go run ./pkg/quantise -format fp8  -src ~/models/gemma-4-e2b-it-bf16 -out /tmp/e2b-fp8
go run ./pkg/quantise -format nf4  -src ~/models/gemma-4-e2b-it-bf16 -out /tmp/e2b-nf4
```

`-bits` maps onto the nearest GGUF recipe (gguf lane) or is used directly
(gptq/awq, with `-group`); fp8 and NF4 carry their formats' own fixed shapes.
The engine's own native format (MLX group-affine, loaded directly by both
engines) is `lem quant`'s default lane — see `model/quant/mlxaffine`.
