# HIP 12B r11 emit-to-feedback pin

Date: 2026-07-12

## Verdict

The emit-to-feedback handoff is not broken. On every recorded 12B and E2B step,
the synchronized device argmax equals the token yielded and fed into the next
forward. The next position equals the retained device-KV token count, so the
generation loop does not use a stale logits result, a different host token, or
an off-by-one KV write index.

Teacher-forcing the exact emitted prefixes reproduces every generation-time
argmax. E2B stays within the configured logit tolerance for all 29 comparisons.
The 12B crosses the `max|delta logit| / reference RMS <= 0.05` threshold at
steps 16 and 24 while retaining the same argmax. Step-16 KV localization shows
broad accumulated differences beginning at dense layer 3 and spreading across
both sliding and full-attention layers. That evidence names the remaining seam
as **12B dense carried-state accumulation versus fresh same-prefix prefill**,
but it does not identify one root-cause op. No fix is applied because the brief
forbids speculative fixes when the pin is ambiguous.

## Instrument

`GO_ROCM_HIP_FEEDBACK_RECEIPTS=1` arms one generation run. The collector is an
atomic nil-gated sibling of the r10 route collector. Armed runs synchronously
read the packed device greedy result before yield/feed and print the device
argmax, fed token, next position, and actual retained-KV token count. Armed runs
also bypass deferred multi-token unrolling so every step receives a receipt.
Unarmed generation retains the existing optimized paths and allocates no
collector.

## 12B generation receipts

Model: `/tmp/models/gemma-4-12B-it-4bit-clean`; KV: `fp16`; prompt:
`why the sky is blue`; 30 greedy tokens.

| step | device argmax | fed token | position | KV write index |
|---:|---:|---:|---:|---:|
| 0 | 107 | 107 | 6 | 6 |
| 1 | 45518 | 45518 | 7 | 7 |
| 2 | 107 | 107 | 8 | 8 |
| 3 | 45518 | 45518 | 9 | 9 |
| 4 | 45518 | 45518 | 10 | 10 |
| 5 | 45518 | 45518 | 11 | 11 |
| 6 | 45518 | 45518 | 12 | 12 |
| 7 | 45518 | 45518 | 13 | 13 |
| 8 | 45518 | 45518 | 14 | 14 |
| 9 | 45518 | 45518 | 15 | 15 |
| 10 | 45518 | 45518 | 16 | 16 |
| 11 | 45518 | 45518 | 17 | 17 |
| 12 | 45518 | 45518 | 18 | 18 |
| 13 | 45518 | 45518 | 19 | 19 |
| 14 | 45518 | 45518 | 20 | 20 |
| 15 | 45518 | 45518 | 21 | 21 |
| 16 | 45518 | 45518 | 22 | 22 |
| 17 | 45518 | 45518 | 23 | 23 |
| 18 | 45518 | 45518 | 24 | 24 |
| 19 | 45518 | 45518 | 25 | 25 |
| 20 | 45518 | 45518 | 26 | 26 |
| 21 | 45518 | 45518 | 27 | 27 |
| 22 | 45518 | 45518 | 28 | 28 |
| 23 | 45518 | 45518 | 29 | 29 |
| 24 | 45518 | 45518 | 30 | 30 |
| 25 | 45518 | 45518 | 31 | 31 |
| 26 | 45518 | 45518 | 32 | 32 |
| 27 | 45518 | 45518 | 33 | 33 |
| 28 | 45518 | 45518 | 34 | 34 |
| 29 | 45518 | 45518 | 35 | 35 |

Transcript (test duration 1.96 seconds):

```text

thought
thoughtthoughtthoughtthoughtthoughtthoughtthoughtthoughtthoughtthoughtthoughtthoughtthoughtthoughtthoughtthoughtthoughtthoughtthoughtthoughtthoughtthoughtthoughtthoughtthoughtthoughtthought
```

## E2B contrast receipts

Model: `/tmp/models/gemma-4-E2B-it-4bit-clean`; otherwise identical.

| step | device argmax | fed token | position | KV write index |
|---:|---:|---:|---:|---:|
| 0 | 236761 | 236761 | 6 | 6 |
| 1 | 108 | 108 | 7 | 7 |
| 2 | 2094 | 2094 | 8 | 8 |
| 3 | 563 | 563 | 9 | 9 |
| 4 | 496 | 496 | 10 | 10 |
| 5 | 3606 | 3606 | 11 | 11 |
| 6 | 5456 | 5456 | 12 | 12 |
| 7 | 529 | 529 | 13 | 13 |
| 8 | 1707 | 1707 | 14 | 14 |
| 9 | 236761 | 236761 | 15 | 15 |
| 10 | 108 | 108 | 16 | 16 |
| 11 | 1509 | 1509 | 17 | 17 |
| 12 | 15517 | 15517 | 18 | 18 |
| 13 | 506 | 506 | 19 | 19 |
| 14 | 2258 | 2258 | 20 | 20 |
| 15 | 529 | 529 | 21 | 21 |
| 16 | 506 | 506 | 22 | 22 |
| 17 | 7217 | 7217 | 23 | 23 |
| 18 | 236761 | 236761 | 24 | 24 |
| 19 | 108 | 108 | 25 | 25 |
| 20 | 1509 | 1509 | 26 | 26 |
| 21 | 563 | 563 | 27 | 27 |
| 22 | 496 | 496 | 28 | 28 |
| 23 | 160180 | 160180 | 29 | 29 |
| 24 | 13315 | 13315 | 30 | 30 |
| 25 | 236761 | 236761 | 31 | 31 |
| 26 | 108 | 108 | 32 | 32 |
| 27 | 1509 | 1509 | 33 | 33 |
| 28 | 64815 | 64815 | 34 | 34 |
| 29 | 2613 | 2613 | 35 | 35 |

Transcript (test duration 1.21 seconds):

```text
.

This is a simple statement of fact.

It describes the color of the sky.

It is a declarative sentence.

It asserts something
```

## Same-prefix teacher-force comparison

The existing incremental oracle was run for 29 steps with
`GO_ROCM_ORACLE_FORCE_BATCHED_PROJ=1`. Its generated token sequences exactly
match the receipt tables above, so each recompute row teacher-forces the exact
production-emitted prefix.

| model | argmax matches | first logit threshold crossing | worst reported ratio | result |
|---|---:|---:|---:|---|
| 12B | 29/29 | step 16, position 21 | 0.0569 at step 24 | ambiguous carried-state logit drift; repeated argmax unchanged |
| E2B | 29/29 | none | 0.0468 at step 19 | healthy coherent contrast |

At the 12B step-16 localization, layer 0 is identical, layers 1 and 2 remain
below the oracle's row threshold, and layer 3 is the first flagged layer. The
differences then accumulate through 45 of 48 layers (37 sliding, 8 full). This
is not the signature of a wrong position, wrong fed embedding, or one isolated
KV append operation.

## Reproduction commands

```sh
export PATH=$PATH:/usr/local/go/bin:/usr/lib/go/bin
export GO_ROCM_RUN_HIP_TESTS=1
export GO_ROCM_KERNEL_HSACO=/tmp/hip-12b-r11/build/kernels/rocm_kernels_gfx1101.hsaco
export GO_ROCM_GEN_PROMPT='why the sky is blue'
export GO_ROCM_GEN_KV_MODE=fp16
export GO_ROCM_GEN_MAX_TOKENS=30
export GO_ROCM_HIP_FEEDBACK_RECEIPTS=1
export GO_ROCM_ORACLE_MODEL_PATH=/tmp/models/gemma-4-12B-it-4bit-clean
go -C go test -v -count=1 ./engine/hip -run '^TestHIPGemma4Q4DecodeKVMode$'

export GO_ROCM_ORACLE_KV_MODE=fp16
export GO_ROCM_ORACLE_PROMPT='why the sky is blue'
export GO_ROCM_ORACLE_DECODE_STEPS=29
export GO_ROCM_ORACLE_FORCE_BATCHED_PROJ=1
export GO_ROCM_ORACLE_SKIP_FINAL_STAGE=1
export GO_ROCM_ORACLE_SKIP_LOGIT_LENS=1
go -C go test -v -count=1 ./engine/hip -run '^TestHIPGemma4Q4IncrementalDecodeOracle$'
```

Repeat both commands with
`GO_ROCM_ORACLE_MODEL_PATH=/tmp/models/gemma-4-E2B-it-4bit-clean` for the
healthy contrast.
