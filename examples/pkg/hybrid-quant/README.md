# hybrid-quant

Serve a **quantised hybrid** (Qwen 3.6 — `model_type qwen3_5`) with its weights kept
**packed on device**. The composed stack alternates gated-delta linear-attention layers
with full-attention layers; at 27B a dense f32 widening is ~110 GB, so the loader carries
every 2-D projection packed (MLX affine codes + scales/biases) straight to the engine's
quant matvec — `affine_qmv` for decode, `affine_qmm_t` for prefill.

```sh
# 4-bit MLX pack
go run ./pkg/hybrid-quant -model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-4bit/snapshots/<snap>

# 1-bit pack (Bonsai) — the 1-bit codes are repacked to 2-bit at load (exact), so the
# stock b_2 kernels serve it with no b_1 kernel required
go run ./pkg/hybrid-quant -model ~/models/Bonsai-27B-mlx-1bit -prompt "The colour of a clear daytime sky is"
```

The same `inference.LoadModel` + `Generate` call serves the dense and the packed lanes —
a quantised checkpoint just keeps its projections packed. Text-only: a multimodal wrapper's
vision tensors are skipped at load, and image input keeps refusing through the existing
text-only machinery. The MLX affine format the loader reads is the one `lem quant` writes
(see `model/quant/mlxaffine`); an MTP drafter pack (`qwen3_5_mtp`) refuses cleanly — serve
it paired with its base model.
