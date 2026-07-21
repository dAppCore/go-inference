# Zoo MoE real-checkpoint parity (#59)

Every factory MoE zoo arch (mixtral/dbrx/olmoe/granitemoe/qwenmoe) had load/bind/serve receipts on
SYNTHETIC fixtures only (`moe_zoo_generate_test.go`, `owned_bind_test.go`, each arch's own
`load_test.go`) — numerical parity against a REAL checkpoint had never been proven for any of them.
`engine/metal/real_checkpoint_olmoe_gpu_test.go` (`TestRealCheckpointGPU_OLMoEArgmaxParis_Good`) is
that receipt for OLMoE, house pattern per the qwen2 ArgmaxParis receipt (`real_checkpoint_gpu_test.go`,
board #24).

## The receipt

`mlx-community/OLMoE-1B-7B-0125-Instruct-4bit` loaded through the unmodified production path
(`LoadDir` -> `model.Load` -> the olmoe `ArchSpec`'s quant route -> `PrefillTokens` ->
`GenerateFromCache`), fixed prompt ids from mlx-lm's own tokenizer, greedy (temp 0) 8-token
continuation checked against mlx-lm's own answer.

Reference (mlx-lm 0.31.3):

```
python -m mlx_lm generate --model <snapshot> --prompt "The capital of France is" \
  --max-tokens 8 --temp 0 --ignore-chat-template
```

- Prompt ids (`tokenizer.encode("The capital of France is")`, `add_bos_token=false`):
  `[510, 5347, 273, 6181, 310]`
- Greedy continuation ids: `[7785, 15, 187, 187, 510, 6100, 39798, 310]` -> `" Paris.\n\nThe Louvre is"`

**Verdict: PASS.** The production `ArchSession` reproduces the reference continuation exactly, from
the first generated token. `engine/metal/capture_hidden_olmoe_oracle_test.go` corroborates this at
per-layer resolution: all 16 layers' post-block hidden-state sums track an independent mlx-lm
extraction within its band, and the embedding-table output matches within band.

## The two mechanisms (#65)

Two independent, compounding defects in the generic MoE zoo dispatch (shared by
mixtral/dbrx/olmoe/granitemoe/qwenmoe's device-router lane) corrupted every layer's output on any
checkpoint whose attention or router shape differs from gemma4's:

**QK-norm granularity.** A q_norm/k_norm weight is either PER-HEAD (length headDim — one shared
weight, RMS-normed independently per head, gemma4/mixtral's convention) or WHOLE-VECTOR (length
heads·headDim / kvHeads·headDim — ONE reduction over the full concatenated projection before the head
split, OLMoE's convention). The engine applied the per-head kernel unconditionally at every consumer,
silently reading only the first headDim elements of a whole-vector weight and broadcasting that wrong
slice across every head. The fix selects granularity by the LOADED WEIGHT'S LENGTH at load time
(`qkNormGranularity`, `engine/metal/qknorm_rope.go`) and, for a whole-vector layer, wraps that layer's
projector (`qkNormWideProjector`) so the correct single-row RMSNorm runs immediately after the Q/K
matmul — the same point every consumer (plain/paged/shared-KV decode, the batched-dense prefill fold,
the q8 KV landing) already runs the per-head norm, so no consumer's own code changes. A mismatched
weight length that matches neither shape is now a load error, never a silent truncation.

**Router combine-weight order.** A router either normalises its selected top-K weights to sum to one
(softmax over ALL experts, gather the top-K, renormalise the gathered subset — mathematically
identical to softmax over just the selected K, which is cheaper and is what the engine's kernel and
host path both already computed) or does not (softmax over ALL experts, gather the top-K WITHOUT
renormalising — OLMoE's `norm_topk_prob=false` checkpoints, whose true combine weights are smaller in
magnitude and do not sum to 1). The engine always computed the former, regardless of what the
checkpoint declared. `model.Arch.NormaliseMoETopK` — populated by every MoE arch's config parser, read
nowhere — now drives the router (`engine/metal/router.go`): the top-K SELECTION is unaffected (a
monotonic transform of the scores), but the WEIGHT computation branches on the declared policy, with
the device (GPU kernel) lane declining to the host path whenever the policy is false, since a fixed
kernel can only implement the always-renormalise order. gemma4 and mixtral both declare
`NormaliseMoETopK: true`, so their routing is byte-unchanged.

## Loader-side hardening (follow-up)

Two guards close residual gaps the #65 fix left as reasoning rather than enforcement — neither changes
today's routing for any registered arch:

- `model.Assemble`'s generic norm loader now validates a loaded q_norm/k_norm tensor's element count
  against the two shapes `qkNormGranularity` understands (per-head / whole-vector) at LOAD time,
  refusing with a typed error on any other length (`TestAssemble_QKNormShape_*`,
  `go/model/assemble_test.go`) — previously the loader applied no shape check at all; only the engine's
  later bind-time `qkNormGranularity` call caught a mismatch.
- The ICB recorder (`decode_forward_arch_icb.go` / `decode_forward_arch_icb_quant.go`) explicitly
  refuses a whole-vector q_norm/k_norm via `icbQKNormSupported` (`TestQknormRope_IcbQKNormSupported_*`,
  `go/engine/metal/qknorm_rope_test.go`), rather than relying solely on the (still true) fact that no
  registered arch's MoE-only whole-vector-QK-norm combination can reach it today.
- The router's `normalise=false` combine-weight order gained direct unit coverage independent of the
  real-checkpoint receipt above: `TestMoERouterNormaliseFalseNotRenormalised` and `TestMoERouter`'s
  `normalise=false` cases, checked against `routerRef`'s own independent softmax-over-all reference
  (`go/engine/metal/router_test.go`).
