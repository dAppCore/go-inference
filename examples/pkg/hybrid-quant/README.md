<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# hybrid-quant

Serve a QUANTISED hybrid (Qwen 3.6 — model_type qwen3_5) with its weights kept PACKED on device. The
composed hybrid stack alternates gated-delta linear-attention layers with full-attention layers; at 27B
a dense f32 widening is ~110 GB (dead on arrival), so the loader carries every 2-D projection PACKED
(MLX affine codes + scales/biases) straight to the engine's quant matvec — affine_qmv for decode,
affine_qmm_t for prefill. This example is the lib-level acceptance for that path: LoadModel a packed
snapshot and Generate, no HTTP serve in the way. Point it at either an MLX 4-bit pack or a 1-bit pack
(Bonsai — its 1-bit codes are repacked to 2-bit at load, exact) — the same call serves both.

## Run

```sh
go run ./pkg/hybrid-quant -model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-4bit/snapshots/<snap>
go run ./pkg/hybrid-quant -model ~/models/Bonsai-27B-mlx-1bit -prompt "The colour of a clear daytime sky is"
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
