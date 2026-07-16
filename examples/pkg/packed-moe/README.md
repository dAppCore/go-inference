<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# packed-moe

The packed-expert MoE path: a quantised Mixture-of-Experts checkpoint (Qwen 3.6 — model_type
qwen3_5_moe) keeps its routed + shared experts PACKED (mlx-affine codes + scales/biases) all
the way to the matvec — buildMoE used to widen every expert to f32 at load regardless, an ~8x
blow-up on a grouped checkpoint's dominant tensor class. This example is the lib-level
acceptance for that fix: build a small synthetic quantised MoE checkpoint in memory (the same
fixture shape go/model/composed/moe_quant_test.go proves against), load it through the composed
loader TWO ways — packed, and dense-over-the-exact-dequantised-values — and compare a forward.

## Run

```sh
go run ./pkg/packed-moe
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
