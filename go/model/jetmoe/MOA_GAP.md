# JetMoE routed-attention gap

The JetMoE FFN half is supported through the generalised composed MoE seam. The
published `jetmoe/jetmoe-8b` checkpoint stores each layer's packed FFN experts as
`mlp.input_linear.weight` (`[8, 11264, 2048]`), `mlp.output_linear.weight`
(`[8, 2048, 5632]`), and `mlp.router.layer.weight` (`[8, 2048]`). The adapter
exposes zero-copy expert Gate/Up/Down views and declares softmax top-2 routing
with selected weights normalised to sum to one.

Full checkpoint loading is deliberately blocked because JetMoE also applies
Mixture of Experts to attention (MoA). Its published weights contain:

- `self_attention.experts.input_linear.weight` (`[8, 2048, 2048]`)
- `self_attention.experts.output_linear.weight` (`[8, 2048, 2048]`)
- `self_attention.experts.router.layer.weight` (`[8, 2048]`)
- `self_attention.kv_proj.weight` (`[4096, 2048]`)

The missing neutral primitive is therefore a routed-attention mixer that
selects top-k expert-specific query and output projections per token, combines
their results using normalised router weights, and shares a K/V projection and
KV cache across those routes. The current composed attention mixer accepts one
dense Q projection and one dense O projection, so mapping these tensors onto it
would silently change the model.

Evidence:

- Published config and INDEX: <https://huggingface.co/jetmoe/jetmoe-8b/tree/main>
- Transformers JetMoE model documentation: <https://huggingface.co/docs/transformers/main/en/model_doc/jetmoe>
