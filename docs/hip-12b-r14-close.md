# HIP 12B r14 pre-change declared-defaults probe receipt

Date: 2026-07-13

## Verdict

The mandatory pre-change probe is **not coherent**. Applying the model-declared
sampling values explicitly (`temperature=1`, `top_k=64`, `top_p=0.95`) is not
sufficient to recover the clean 12B pack. Per the r14 lane stop condition, no
generation-defaults implementation was attempted and #52 is not claimed
closed. The diagnostic pin must move before defaults application can proceed.

## Probe method

The `generate` CLI exposes `-temp` but no top-k or top-p flags. The allowed
direct loaded-model probe therefore drove the clean 12B model for 40 tokens
with:

```go
inference.WithMaxTokens(40)
inference.WithTemperature(1)
inference.WithTopK(64)
inference.WithTopP(0.95)
```

The model was `/tmp/models/gemma-4-12B-it-4bit-clean`; the runtime used the
pinned gfx1101 HSACO copied from the unchanged r13 kernel tree.

## Transcript

```text
=== RUN   TestHIPR14ExplicitDeclaredSamplerProbe
    hip_r14_probe_test.go:19:
            //color://style
        /111111//1111111111111111111111111
--- PASS: TestHIPR14ExplicitDeclaredSamplerProbe (2.26s)
PASS
ok  dappco.re/go/inference/engine/hip  2.319s
```

The test process passed because it is a transcript probe; the generated text
itself fails the lane's coherence requirement.

## Command

```sh
GO_ROCM_RUN_HIP_TESTS=1 \
GO_ROCM_KERNEL_HSACO=/tmp/hip-12b-r14/build/kernels/rocm_kernels_gfx1101.hsaco \
GO_ROCM_MODEL_PATH=/tmp/models/gemma-4-12B-it-4bit-clean \
go -C go test -count=1 -v ./engine/hip/ \
  -run '^TestHIPR14ExplicitDeclaredSamplerProbe$'
```

The temporary box-only probe test was not added to this worktree and is not
part of the commit.
