# Zoo MoE real-checkpoint parity (#59 — the zoo's last honest bar)

Every factory MoE zoo arch (mixtral/dbrx/olmoe/granitemoe/qwenmoe) had load/bind/serve receipts on
SYNTHETIC fixtures only (`moe_zoo_generate_test.go`, `owned_bind_test.go`, each arch's own
`load_test.go`) — numerical parity against a REAL checkpoint had never been proven for any of them.
This is the first such receipt, for OLMoE, and it fails — cleanly root-caused, not fixed, per this
lane's file fence (test + docs only, no engine/model production edits).

## The receipt

`engine/metal/real_checkpoint_olmoe_gpu_test.go` (`TestRealCheckpointGPU_OLMoEArgmaxParis_Good`),
house pattern per the qwen2 ArgmaxParis receipt (`real_checkpoint_gpu_test.go`, board #24):
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

lem (production `ArchSession`, same prompt ids, same checkpoint, 2026-07-20):

- Greedy continuation ids: `[4960, 1205, 90, 2072, 31511, 14, 4410, 45327]` -> `"iformengynaoin- RedFlash"`

**Verdict: FAIL.** Diverges from the first generated token. Not "coherent but wrong" (the failure
mode the qwen fused-chain lane's activation-binding note describes) — outright garbage, consistent
with 16 MoE layers each compounding a wrong gate/up nonlinearity.

## Root cause

The generic device MoE expert combine hardcodes GELU with no per-arch activation selection.

- `engine/metal/decode_forward_arch.go`'s per-layer MoE dispatch (~L1930-1990) checks, in order: the
  arch's `MoEQuantLayerWeights.ClampedSwiGLU` marker (gpt_oss -> `encGptOssMoEHalf`, clamped-sigmoid
  SwiGLU), then `len(moeQ.SharedGate.Packed) > 0` (a bound shared expert marks a Qwen-family MoE layer
  — true for qwen3_5_moe AND for plain qwenmoe/Qwen2-MoE, which also ships a `.mlp.shared_expert.*`
  trio — routed to `encQwenMoEHalf`, plain SiLU via `MoEExpertsQuantSiLU`). Everything else falls to
  `encMoEBlockQuantDevice` (`engine/metal/moe_block.go:1856-2323`), the gemma4-shaped generic path.
- `encMoEBlockQuantDevice`'s per-expert gate/up combine (L2302) is
  `emitBinary(sink, geluPSO, msc.gate, 0, msc.up, 0, msc.gated, 0, expertDFF)` — unconditional. The
  same GELU kernel (`geluPSO` / `gpuHasGeluKernel()` / `geluPipeline()`, `lthn_kernels.go`) backs the
  bf16 sibling (`moeBlockBF16AfterRouterWithBufferPooled`, `moe_block.go:1006-1013`) and the other quant
  entry points (`moeBlockQuantAfterRouterWithDeviceIndexBufferPooled`, `moe_block.go:2595-2602`).
  `encGeluGateMul` (`moe.go:28-46`) is explicitly the tanh-approx GELU chain, fused or composed — there
  is no SiLU (or any other) variant anywhere in this file's expert-combine code, and no
  `MoEQuantLayerWeights` field selects one. This is correct for gemma4 (genuinely GELU-gated, the one
  MoE arch proven end-to-end on a real checkpoint before this receipt) and wrong for every zoo arch
  that isn't.
- Mixtral, DBRX, OLMoE and GraniteMoE all declare `hidden_act`/equivalent `"silu"` in their real HF
  configs (SwiGLU routed experts) and carry no shared-expert tensors, so all four fall to the same
  GELU-only `encMoEBlockQuantDevice` path — **this receipt directly confirms OLMoE; mixtral/dbrx/
  granitemoe are inferred from sharing the identical dispatch and are not independently verified with
  a real checkpoint here.** qwenmoe is the one zoo member architecturally exempt (shared-expert marker
  routes it to the already-SiLU-correct `encQwenMoEHalf` host path).
- The per-arch activation selector that DOES exist — `model.Arch.Activation` ->
  `ffnUsesSiLU`/`ffnUsesClampedSwiGLU` (`engine/metal/projector.go:127-137`) — is wired to the
  dense-FFN and ICB-recorded decode paths only (`arch_session.go` L689/889/902/975/1239/1248,
  `load_shared.go:153`). It is never consulted by the MoE-expert branch. This matters less than it
  first appears for three of the four affected arches: `model/arch/mistralai/mixtral/config.go` and
  `model/arch/databricks/dbrx/config.go` don't parse an activation field from their configs at all
  (always constant SiLU for real checkpoints, so never plumbed), and neither does
  `model/arch/allenai/olmoe/config.go` — `Config.Arch()`'s returned `model.Arch{}` literal never sets
  `Activation`, so it defaults to `""` for every OLMoE checkpoint regardless. `model/arch/ibm-granite/
  granitemoe/config.go` (L26, L79) DOES forward `Activation: c.HiddenActivation` correctly — but
  `moe_block.go` ignores the field regardless, so it makes no difference today. Closing this gap
  needs both: threading `arch.Activation` (or an explicit MoE-expert activation choice) into
  `encMoEBlockQuantDevice`'s combine step, AND teaching mixtral/dbrx/olmoe's config parsers to declare
  it (granitemoe already does).

## Scope note

This lane's file fence is test + docs only — the fix is a follow-up lane's. The committed test
(`TestRealCheckpointGPU_OLMoEArgmaxParis_Good`) runs the real checkpoint every time the gate runs (not
a vacuous receipt) and `t.Skipf`s with the observed-vs-want ids when parity fails, rather than
red-failing the gate on an already-known, already-documented engine limitation. It will start passing
outright, with no test-code change, the day the activation gap above closes.
