# model/composed strip inventory (#50 finale)

Base: `751d7dfdff81583751ebeb71d58ba5030f69480a` (lane/composedrip). `model/composed` = 48 files,
14,319 lines. Every registered arch is dual-routed at this base; the factory-vs-composed A/B sweep
receipt is banked by the orchestrator on the pre-delete tree.

Method: repo-wide grep split into REAL imports (`"dappco.re/go/inference/model/composed"`) vs
comment-only mentions. Classification of every engine/metal importer traced from the factory
arch-session call path (`arch_qwen_fused.go` → chain machinery) vs the composed backend hook
registrations (`composed.X = localImpl` assignments).

## The load-bearing finding

The brief's expected (a)-list is WRONG for three files. `composed_chain_backend.go` (1,377 lines)
and `composed_chain_moe.go` (447 lines) are NOT composed-model backends — they are the #18 factory
fused-chain machinery (`ComposedChainBeginDevice/RecordBegin/ReplayDevice`,
`gatedDeltaQuantChainLayerDevice`, `attnQuantChain(MoE)LayerDevice`, `resolveMoEChainWeights`,
`moeChainRecordable`), called DIRECTLY by `arch_qwen_fused.go` (`qwenChainWalk`,
`stepTokenQwenChain`, `qwenChainReady`). Deleting them breaks qwen3_5/3.6 factory serving.
Similarly `composed_bf16_backend.go` (653 lines) defines the native bf16/quant attention
front/tail seams and binds the `attn.*` hooks the factory host path consumes. Their only composed
edges are (1) `*composed.MoEMLP` in internal signatures and (2) `composed.X = …` hook assignments.

## engine/metal classification

| File | Class | Evidence | Action |
|---|---|---|---|
| composed_backend.go (71) | (a)+relocate | pure `composed.*` hook assignments; BUT its `init` also binds `attn.GatedDeltaBlockDevice/DeviceStateExport/QuantLayerDevice` (read by model/attn — the factory host fallback) under `gdBlockEnabled` | DELETE file; relocate the three `attn.*` bindings + `gdBlockEnabled` to lthn_gated_delta.go |
| composed_bf16_backend.go (653) | (b) | defines MatMulBF16WeightF32NTInto, AttnBF16/QuantFront/TailDevice, gatedDeltaBF16LayerRun — native seams; `attn.ProjBF16MatMulInto`/`attn.GatedDeltaBF16LayerDevice` read by model/attn (factory) | KEEP; strip every `composed.*` assignment; keep `attn.*` bindings whose reader survives (drop `attn.GatedDeltaBF16ChainLayerDevice`, `attn.GatedDeltaQuantChainLayerDevice`, `attn.GatedDeltaChainGeometryOK` — read only by model/composed) |
| composed_chain_backend.go (1377) | (b) FACTORY | arch_qwen_fused.go calls its chain walk + record/replay + per-layer funcs directly | KEEP; port `*composed.MoEMLP` → native `chainMoE` type; delete 3 hook assignments (L1083-85); delete now-dead bf16 chain funcs if orphaned |
| composed_chain_moe.go (447) | (b) FACTORY | `moeChainRecordable` called by arch_qwen_fused.go; `resolveMoEChainWeights` by the chain MoE tail | KEEP; retype to `chainMoE` |
| composed_quant_backend.go (157) | (b) | `MatMulQuantF32NTInto` bound to `attn.ProjQuantMatMulInto` — the factory host quant matmul (arch_gated_attn.go projQuantAttn) | KEEP func + attn binding; delete `composed.*` assignments + `MoEExpertsQuantDevice` (only consumer was the composed hook) + LTHN_COMPOSED_MOE_DEVICE lever |
| composed_hook_receipts.go (309) | (a) | wraps/counts `composed.*` hooks only; sole users composed_backend_test.go + composed_decode_census_test.go | DELETE |
| speculative_composed.go (200) | (a) | composed.LoadSpeculativePairDirs — the qwen MTP pair serve | DELETE; typed decline at speculative_model.go dispatch |
| arch_gated_attn.go (249) | (b) FACTORY | factory host gated-attention (encGatedAttnHalf); composed edge = `moe *composed.MoEMLP` field only | KEEP; retype field |
| arch_gated_delta.go (130) | (b) FACTORY | factory host gated-delta (encGatedDeltaHalf); same single field edge | KEEP; retype field |
| arch_qwen_fused.go (496) | (b) FACTORY | the fused device decode itself; builds the MoEMLP view (qwenChainMoE) | KEEP; retype to `chainMoE` |
| load.go (477) | sever | composed preference/fallback (LoadComposedDir call), isQwen35FactoryType, qwenFactoryQuantServable, loadComposedTokenModel | factory-only routing; typed sub-2-bit decline (see capabilities); delete isQwen35FactoryType + loadComposedTokenModel |
| inference_register.go (731) | sever | `composed.ChatMLDialect` ×3 (template dialect), 2 comment refs | port dialect helper to model/arch/Qwen/qwen35 (family owns its dialect); reword comments |
| speculative_model.go | sever | `spec.Composed != nil` dispatch to the composed pair arm | typed decline for qwen-hybrid targets (factory pair route pending) |
| composed_backend_test.go, composed_chain_head_test.go, composed_chain_icb_test.go, composed_chain_moe_e2e_test.go, composed_decode_census_test.go, composed_state_test.go | (a) tests | composed-session harness (import composed; two pin LTHN_QWEN_COMPOSED=1) | DELETE — the banked pre-delete A/B sweep is the receipt; factory chain coverage remains via arch-session tests + composed_chain_moe_test.go (native, no import) |
| lthn_gated_delta_test.go (1107) | SPLIT | tests ≤ L476 (GatedDeltaStepDevice ×4, GatedDeltaBlockDeviceRun ×3, BeatsHost) are composed-free native kernel tests; tests from L521 on build composed.ComposedModel/NewSession as harness (QuantLayerDevice, BF16LayerDevice, AttnFold, DevKVRealShape_AB) | KEEP the composed-free half; DELETE the composed-harness half (honest coverage note below) |
| composed_registry_test.go | (a) test | asserts LoadTokenModelDir resolves composed archs via registry — the deleted route | DELETE |
| composed_moe_fuse_test.go / _bench_test.go, composed_chain_moe_test.go, rwkv7_device_parity_test.go | KEEP | no composed import; test surviving native seams (MatMulQuantF32NTInto, chain MoE byte-identity, rwkv7 proj) | keep; reword comments |
| load_test.go | trim | TestIsQwen35FactoryType_Good/_Bad die with the func | keep TestLoadDirReactiveDispatch |

## model side

| Edge | Action |
|---|---|
| model/arch_spec.go `Composed` field | DELETE (+ doc) |
| model/load.go: Composed-only branch (L44), LoadComposedDir, mmapRetainer | DELETE (mmapRetainer's only user is LoadComposedDir) |
| Composed hooks — real loaders (import composed): mixtral, dbrx, olmoe, granitemoe, qwenmoe, llama4, qwen35 | delete hook + import; Parse/Weights factory route stays |
| Composed hooks — refusals (no import): deepseek (MLA), deepseekvl2, glmocr, dotsocr, jetmoe | delete hook. deepseek's `Config.Arch()` already refuses MLA identically — no port needed. deepseekvl2/glmocr/dotsocr refuse/serve on their Parse route. jetmoe: its tensor-shape-aware MoA refusal message degrades to a generic Assemble missing-weights error (noted, capability-neutral — it never loaded) |
| model/composed/register.go ids | generic `composed`/`hybrid`: die with the package → `model.Load: no architecture registered` (honest). qwen3_5_mtp/_text/qwen3_6_mtp: typed refusal MOVES to qwen35/register.go as a Parse-refusal spec |
| ChatMLDialect | moves to model/arch/Qwen/qwen35 (qwen-prefix ChatML declaration) |
| model/builtin/builtin.go | drop blank import; rewrite the stale route comment |
| model/conformance | composed.LoadComposed host-forward harness (L102) — rework/delete per family; skips referencing Composed hooks reworded |
| integration_test.go A/B files (olmoe, dbrx, granitemoe, qwenmoe, llama4-adjacent, starcoder2, cohere, granite, jetmoe, llama, qwen2) | DELETE where the subject is composed execution/A-B; config_test.go fixtures (smollm3, mpt, stablelm) rework to config-level assertions or delete the mixer-execution half |
| engine/hip composed_runtime.go L81 + gemma4_architecture_adapter.go L88 (`spec.Composed` FIELD reads — compile break) | composed_runtime: replace registry-hook detection with an explicit named no-ROCm-execution model_type set (HIP owns its capability table; the profile table is proven insufficient). adapter: the `spec.Composed != nil` guard becomes vacuous — delete |
| cmd/mtp-probe | DELETE directory (retired stub) |

## Capabilities composed uniquely served — decisions

| Capability | Decision |
|---|---|
| qwen3_5 35B VISION tower (composed vision_loader; factory route is text-only) | HONEST REGRESSION, no port in this lane. Image turns on qwen hybrids now hit the generic engine vision decline (engine/vision.go — composed-independent, the existing "clean 400"). Factory vision tower = board follow-up |
| qwen MTP speculative pair (`lem pair`/`-draft` on hybrid targets) | typed decline at LoadSpeculativePair: named "factory pair route pending (#50)" |
| Sub-2-bit packs (Bonsai 1-bit) — factory qmv has no <2-bit width (#24) | typed decline at LoadTokenModelDir (generic bits<2 probe, not qwen-specific) instead of a confusing missing-kernel error |
| Generic `composed`/`hybrid` model_type ids | die with the package; unregistered-arch error is the honest answer |
| qwen3_5_mtp standalone-load refusal | preserved — moves into qwen35's register (Parse refusal, same message) |
| mamba2 / rwkv7 | unaffected — native model/arch loaders, reached by name in load.go |

## Honest coverage notes

- Deleting the composed-session halves of lthn_gated_delta_test.go loses direct harness tests for
  `gatedDeltaQuantLayerDeviceHook` / bf16 layer hook / Attn fold seams; those seams stay exercised
  live by the factory arch-session + chain tests and the scoped metal gate.
- The chain record/replay machinery loses its composed-driven ICB/head tests; factory-side
  coverage = arch_session fused-chain tests.

## Board follow-ups (not this lane)

- File renames: composed_chain_backend.go / composed_chain_moe.go / composed_bf16_backend.go /
  composed_quant_backend.go / arch_* keep historic "composed" names — rename to chain_*/seam_* once
  the strip has soaked.
  - **Landed (2026-07-20, lane/chainrename):** composed_chain_backend.go → fused_chain_backend.go,
    composed_chain_moe.go → fused_chain_moe.go, composed_bf16_backend.go → bf16_seam_backend.go,
    composed_quant_backend.go → quant_seam_backend.go, plus their test companions
    composed_moe_fuse_test.go → quant_moe_fuse_test.go, composed_moe_fuse_bench_test.go →
    quant_moe_fuse_bench_test.go, composed_state_test.go → mamba2_session_state_test.go. arch_*
    (arch_gated_attn.go / arch_gated_delta.go / arch_qwen_fused.go) keep their names — already honest,
    out of scope. Names + header self-references only; zero identifier/behaviour change.
- Factory qwen vision tower port; factory MTP pair; 1-bit qmv width (#24).
- engine/hip: wire the staged Qwen36 native guard (dense_config.go) if HIP ever grows a native route.

## Post-inventory finding (examples module)

The root go.work builds `examples/` against the LIVE ./go tree (not the v0.12.0 pin the module
file names), so `examples/pkg/packed-moe` — a direct composed.LoadComposed consumer — broke with
the deletion and is retired with its subject (README row removed). A factory-route packed-MoE
example is a board follow-up. `examples/pkg/hybrid-quant` uses only the public inference.LoadModel
surface and serves through the factory unchanged (comment corrected).
