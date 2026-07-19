# TurboQuant real-KV distortion — gemma-4 e2b-4bit

RFC #41 slice S3. Date: 2026-07-19.

## Method

Real K/V cache rows captured post-RoPE from a live `engine/metal` decode
session, model `mlx-community/gemma-4-e2b-it-4bit`, via
`(*ArchSession).DumpKVRows`. Prompt: a real English paragraph (not a
synthetic token-id fixture), 141 tokens, tokenised by the checkpoint's own
tokenizer. Greedy decode (`eosID=-1`, no early stop) for 550 further tokens.
Session position at capture: 691. Measured by `turboquant.MeasureReal`
(TurboQuant Q_mse codec, rotation seed 42).

Rows are one per (attention head, cached token). gemma-4 e2b declares
`num_key_value_heads=1` (MQA) on its global-attention layers, so each
captured layer contributes 691 rows per side (K, V). `global_head_dim=512`
in config.json — d=512 throughout.

**Layer selection.** gemma-4 e2b has 35 layers, 7 typed `full_attention`
(indices 4, 9, 14, 19, 24, 29, 34), but `num_kv_shared_layers=20` means only
the first 15 layers (`model.DeriveLayers`) own a physical cache slot — layers
15-34 read an earlier layer's cache verbatim. Of the 7 nominal
full_attention layers, only 4, 9 and 14 own distinct cache; 19, 24, 29 and
34 all share layer 14's bytes. This measurement covers all 3 distinct
global-attention layers this checkpoint has — not the brief's nominal "≥4",
because a 4th distinct sample does not exist on this architecture.

## Distortion table — relative MSE (the paper's metric)

| layer | side | b=2 | b=3 | b=4 | K4/V3 pooled |
|---|---|---|---|---|---|
| 4  | K | 0.1170 | 0.03433 | 0.009412 | 0.03478 |
| 4  | V | 0.1187 | 0.03488 | 0.009582 | *(pooled with K, same row)* |
| 9  | K | 0.1172 | 0.03436 | 0.009392 | 0.03482 |
| 9  | V | 0.1183 | 0.03491 | 0.009548 | *(pooled with K, same row)* |
| 14 | K | 0.1179 | 0.03485 | 0.009679 | 0.03404 |
| 14 | V | 0.1164 | 0.03413 | 0.009336 | *(pooled with K, same row)* |
| paper oracle (arXiv 2504.19874) | — | 0.117 | 0.03 | 0.009 | — |

Bytes/row at d=512: b=2 132, b=3 196, b=4 260. Samples: 691 rows/side/layer
for the b=2/3/4 columns; the K4/V3 pooled column blends 691 K rows at b=4
with 691 V rows at b=3 (1382 rows), one pooled ΣΣ‖x-x̃‖²/ΣΣ‖x‖² per layer.

## Row MSE — absolute, not the paper's metric (included for scale only)

| layer | side | b=2 | b=3 | b=4 |
|---|---|---|---|---|
| 4  | K | 0.2525 | 0.07411 | 0.02032 |
| 4  | V | 60.76 | 17.86 | 4.906 |
| 9  | K | 0.2165 | 0.06346 | 0.01735 |
| 9  | V | 60.59 | 17.87 | 4.889 |
| 14 | K | 0.2249 | 0.06649 | 0.01847 |
| 14 | V | 59.62 | 17.48 | 4.78 |

V's row MSE runs roughly 250-300x K's at every bit width — V's raw
per-coordinate energy is far larger than K's in this checkpoint (RoPE
applies to K, not V). Relative MSE (row MSE divided by the row's own squared
norm) is the comparable figure and is the table above.

## Reading

- **b=2**: measured K 0.1170-0.1179, V 0.1164-0.1187, vs the published
  oracle 0.117 — every measured value is within -0.5% to +1.5% of the oracle.
- **b=3**: measured K 0.03433-0.03485, V 0.03413-0.03491, vs oracle 0.03 —
  every measured value is +13.8% to +16.4% above the oracle.
- **b=4**: measured K 0.009392-0.009679, V 0.009336-0.009582, vs oracle
  0.009 — every measured value is +3.7% to +7.5% above the oracle.
- K vs V, same layer and bit width, differs by at most 3.7% anywhere in this
  table. The sign is not consistent across layers: V exceeds K by 0.9-1.8%
  at layers 4 and 9; K exceeds V by 1.3-3.7% at layer 14.
- Values are stable across the 3 sampled layers (4, 9, 14 — the only
  positions in this checkpoint holding independent K/V data); no trend
  across that depth range is visible in this data.

## Recommendation

Real-KV relative MSE does not show K as more distortion-sensitive than V —
the largest measured K/V difference at any sampled layer and bit width is
3.7%, and its sign flips between layers. K4/V3 is not supported by this
measurement; a symmetric split (K3/V3 or K4/V4) is equally consistent with
the numbers above. If K4/V3 stays the live 3.5-bit default, that needs a
justification other than the relative MSE measured here.

## Reproduction

```
MLX_METALLIB_PATH=<repo>/build/dist/lib/mlx.metallib \
LEM_TQ_CAPTURE=1 \
go test -tags metal_runtime -count=1 -v -run '^TestTurboQuantRealKVCaptureReceipt$' \
  ./engine/metal/...   # from go/
```

- Capture harness: `go/engine/metal/turboquant_real_kv_capture_test.go`
  (`TestTurboQuantRealKVCaptureReceipt`, gated on `LEM_TQ_CAPTURE=1`).
- Tap: `go/engine/metal/turboquant_capture_tap.go`
  (`(*ArchSession).DumpKVRows`).
- Measurement: `go/kv/turboquant/real.go` (`MeasureReal`; live default under
  test is `RealMixedKeyBits`=4, `RealMixedValueBits`=3).
- CI fixture (32 rows/side, ~128 KB combined, subsampled from layer 4):
  `go/kv/turboquant/testdata/real_kv_keys.bin`,
  `go/kv/turboquant/testdata/real_kv_values.bin` — exercised by
  `TestMeasureReal_Ugly` (`go/kv/turboquant/real_test.go`).
- Model cache: `~/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit`.
