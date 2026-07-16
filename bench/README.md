<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# bench/ — the reproducible benchmark grids

`lem bench --config bench/gemma4.json --json out.json` runs the locked house
grid: per model × lane (plain, mtp), one untimed warmup then one timed greedy
tg-512, decode tok/s from the engine's own metrics. Model refs are HF cache
repo names, so the same config reproduces the grid on any box sharing the
cache convention (the metal Mac, the hip AMD box) — edit the JSON to shift
focus, never the code.

Notes baked into gemma4.json:
- Drafters are EXPLICIT (auto-detection ambiguity broke the shell sweep this
  verb replaces) and quant-matched where possible — the 2026-07-16 matrix
  measured E4B's matched QAT draft at +47% vs +11% for the same draft
  cross-paired onto the non-QAT base. e4b-4bit therefore runs plain-only.
- The 26B rows keep their mtp lane deliberately: the −70% MoE-verify loss is
  #372's open tracking number, and the grid should show it until it's fixed.
- The default prompt (counting to 800) is the house tg-N workload: maximally
  predictable output, so MTP acceptance reads at its CEILING — right for
  comparability with the reference tables, wrong for estimating real-workload
  gains (creative text measured 8-20% acceptance on the same pair, i.e. MTP
  roughly break-even there). Bench a workload by shifting the prompt in a
  config copy.
- QAT rows decode SLOWER than plain 4bit by design, not by engine defect: the
  mlx-community `qat-4bit` checkpoints are mixed-precision — every
  `mlp.{gate,up,down}_proj` carries a per-tensor 8-bit override (config.json
  `quantization`), attention/embed/head stay 4-bit, group size 64 throughout.
  Decode is weight-bandwidth-bound, so the extra bytes ARE the slowdown:
  E2B 4.33 vs 3.55 GB (+22%) → measured −16%, E4B 6.80 vs 5.15 GB (+32%) →
  −21%, 31B 28.8 vs 18.4 GB (+57%) → −32% — each at or slightly better than
  the pure byte-ratio prediction (−18/−24/−36%), which is the receipt that the
  q8 tensors ride the fast qmv path. Pick QAT for quality, plain 4bit for
  speed; a grid row comparing them measures checkpoint mass, not the engine.
