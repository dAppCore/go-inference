<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# packed-moe

The packed-expert MoE convention, end to end. A Mixture-of-Experts checkpoint on disk (Mixtral's
HuggingFace layout: .block_sparse_moe.experts.{i}.{w1,w3,w2}.weight — one 2-D tensor PER expert)
does not match the engine's generic MoE assembler, which wants ONE tensor per role per layer —
the "every expert lives in one [experts·outDim, inDim] tensor" shape gpt_oss and qwen3_5_moe
already ship natively. packExperts (model/arch/mistralai/mixtral/weights.go — the same synthesis
dbrx/olmoe/qwenmoe/llama4 share) bridges the two at load time: it concatenates the N per-expert
matrices row-major into one packed tensor per role. A quantised checkpoint carries the identical
synthesis for the .scales/.biases triple (since 542d5484), so a 4-bit MoE pack loads and serves
exactly like a dense one — through the SAME factory route (model.Load, then inference.LoadModel)
any other architecture takes.

This example builds a small synthetic 4-bit Mixtral-shaped checkpoint (1 layer, 2 experts,
top-1) — self-contained, no download, no cgo — writes it to a temp directory in the REAL
per-expert on-disk layout, loads it once directly (model.Load) to show the packed tensors
packExperts synthesised, then loads it again through inference.LoadModel — the identical call
any HF snapshot on disk takes — and Generates a few greedy tokens through the metal engine. The
checkpoint's weights are synthetic noise (no training), so the generated text is not meant to
read as language: it is the receipt that the packed-MoE wiring loads and decodes end to end.

## Run

```sh
go run ./pkg/packed-moe
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
