# JetMoE routed-attention gap

The published `jetmoe/jetmoe-8b` checkpoint stores each layer's packed FFN experts as
`mlp.input_linear.weight` (`[8, 11264, 2048]`), `mlp.output_linear.weight`
(`[8, 2048, 5632]`), and `mlp.router.layer.weight` (`[8, 2048]`). `weights.go`'s
`adaptFFNWeights` can still turn those into zero-copy expert Gate/Up/Down views declaring
softmax top-2 routing with selected weights normalised to sum to one — but nothing calls it:
the generalised composed MoE seam it targeted was retired with `model/composed` (#50), and no
factory route has adopted it, so the FFN half is unwired, not loadable.

Full checkpoint loading is deliberately refused — named, at `Config.Arch()` (register.go's
Parse still recognises the model_type) — because JetMoE also applies Mixture of Experts to
attention (MoA). Its published weights contain:

- `self_attention.experts.input_linear.weight` (`[8, 2048, 2048]`)
- `self_attention.experts.output_linear.weight` (`[8, 2048, 2048]`)
- `self_attention.experts.router.layer.weight` (`[8, 2048]`)
- `self_attention.kv_proj.weight` (`[4096, 2048]`)

The missing neutral primitive is therefore a routed-attention mixer that
selects top-k expert-specific query and output projections per token, combines
their results using normalised router weights, and shares a K/V projection and
KV cache across those routes. The factory attention assembler (`model.Assemble`)
accepts one dense Q projection and one dense O projection per layer, so mapping
these tensors onto it would silently change the model — hence the named refusal
in `Config.Arch()` rather than a best-effort load.

Evidence:

- Published config and INDEX: <https://huggingface.co/jetmoe/jetmoe-8b/tree/main>
- Transformers JetMoE model documentation: <https://huggingface.co/docs/transformers/main/en/model_doc/jetmoe>
