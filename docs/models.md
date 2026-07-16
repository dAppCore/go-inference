<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# Models for local inference

The engine's curated model list: what serves well locally, at what cost, in which fleet role.
Every number is dated and machine-stamped; reproduce any row with the locked grid
(`lem bench --config bench/gemma4.json`, see [bench/README.md](../bench/README.md)) rather than
trusting a stale table. Hardware honesty rule: a model must fit PHYSICAL RAM (weights + KV cache +
~2 GB headroom) — the harness hard-skips oversized loads because this Mac has OOM'd on them.

## Gemma 4 — the primary family

The house default. Fully multimodal across the family (text + image + audio + video), thinking
mode (the reasoning channel), and native MTP speculative decode via the `-assistant` drafters.
Snapshots are MLX-format from `mlx-community/`; keys below are the `lem.sh` shorthands.

Decode numbers: M3 Ultra, tg-512 greedy no-think, build `fb199ae5` (2026-07-16) — the serving
lane; sampled (temp>0) serves at greedy parity since `574c85bc`.

| key | model | HF repo (4-bit) | ctx | weights | decode | +MTP | fleet role |
|-----|-------|-----------------|----:|--------:|-------:|-----:|------------|
| `e2b` | E2B | `mlx-community/gemma-4-e2b-it-4bit` | 128K | 3.6 GB | 170.0 | **223.9 (+32%)** | fast sub-agent · Lemma v2 base |
| `e4b` | E4B | `mlx-community/gemma-4-e4b-it-4bit` | 128K | 5.1 GB | 117.5 | 135.7 (+47% w/ matched QAT draft) | sub-agent worker |
| `12b` | 12B dense (unified vision) | `mlx-community/gemma-4-12B-it-4bit` | 256K | 6.7 GB | 71.2 | 87.1 (+22%) | mid-weight generalist |
| `26b` | 26B-A4B MoE | `mlx-community/gemma-4-26b-a4b-it-4bit` | 256K | 15.3 GB | 138.9 | ✗ −73% (#372) — serve plain | big-context orchestrator (efficient MoE) |
| `31b` | 31B dense | `mlx-community/gemma-4-31B-it-4bit` | 256K | 18.4 GB | 33.1 | **46.0 (+39%)** | orchestrator (dense) |

Cross-engine anchor (E2B, tg-512, same machine): lem 168.5 · mlx-lm 158.4 · llama.cpp 144.9.

### MTP drafters (the `-assistant` repos)

`gemma-4-<target>-it-assistant[-bf16|-qat-4bit]` checkpoints are MTP speculative drafters — they
error on standalone serve and pair with their target (`lem serve --draft <assistant-path>`; serve
auto-arms a detected drafter). Two rules, both measured:

- **Match drafter quant to target quant.** E4B with its matched QAT draft gains +47%; the same
  draft cross-paired onto the non-QAT base gains only +11% (the drafter's distribution shifts and
  acceptance collapses).
- **MTP gain is workload-dependent.** The grid's counting prompt reads acceptance at its CEILING
  (60%+); creative prose measures 8–20% (roughly break-even). MTP pays on structured output —
  lists, code, tool-JSON — the agent workloads.
- The 26B MoE pair currently LOSES (−73%, the MoE verify cost — #372's open campaign). Serve the
  26B plain until that closes.

### Quant guidance

- **Plain 4-bit for speed, QAT for quality.** The mlx-community `qat-4bit` checkpoints are
  mixed-precision (every `mlp.{gate,up,down}_proj` is 8-bit): more bytes, and decode is
  weight-bandwidth-bound, so they decode slower BY THAT MASS — E2B 142.6 (−16%), E4B 92.6 (−21%),
  31B 22.4 (−32%). Not an engine defect; full physics in [bench/README.md](../bench/README.md).
- **bf16** variants are for drafters (`-assistant-bf16` dequantises to the same fused path) and
  training/self-distillation — not for serving on RAM-constrained boxes.
- 8-bit sits between: ~2× 4-bit's weight mass for a quality bump few serving workloads need.

### Task-built variants

| model | repo | use |
|-------|------|-----|
| EmbeddingGemma 300M | `google/embeddinggemma-300m` | RAG / vector search (2K ctx, 128–768 dims) |
| ShieldGemma 2 4B | `google/shieldgemma-2-4b-it` | content moderation, run beside the primary |

## Qwen 3.5 / 3.6 hybrids — the composed lane

Gated-delta linear attention interleaved with full attention (`full_attention_interval 4`),
262144-token context, served through `go/model/composed` with the device gated-delta block
(`docs/design-hybrid-recurrence.md`). Native MTP head supported (the trained checkpoint head, not
a bolt-on drafter).

| model | HF repo (4-bit) | ctx | weights | decode | status |
|-------|-----------------|----:|--------:|-------:|--------|
| Qwen3.6-27B | `mlx-community/Qwen3.6-27B-4bit` | 256K | 16.1 GB | 15.05 (2026-07-16, `3af77a87`) | serving; perf campaign #18 in flight (4.63 → 15.05 so far; target ≥40 — mlx-lm does 41.3 on this snapshot) |
| Qwen3.5-0.8B | `mlx-community/Qwen3.5-0.8B-OptiQ-4bit` | — | 0.6 GB | — | fixture/smoke model for the composed lane |
| Qwen3.6-35B-A3B | (not yet local) | — | — | — | the MoE hybrid target — pulls with the #17/#18 follow-on |

## Legacy

Gemma 3 (`1b`/`4b`/`12b`/`27b-it`) loads and serves but is the UNOPTIMISED legacy path — mlx-lm
leads on it (E2B-class gemma3-1B: ~200 vs mlx-lm ~292). Use Gemma 4; if a tool lacks the Gemma 4
architecture, update the tool rather than dropping back.

## Adding a row

A model earns a row when: (1) it loads from the HF cache by path, (2) `lem bench` produces a
clean tg-512 number on this hardware, (3) its RAM footprint and context length are verified from
the snapshot itself (not the model card), and (4) anything surprising (a quant that isn't what its
name says, a lane that declines) is written down next to the number that shows it.
