# HIP 12B r15 close-gate receipt

Date: 2026-07-13

## Verdict

The producer-scale bug is fixed and the default 12B gate now passes, but the
explicit top-k gate remains incoherent. This lane therefore does **not** claim
the complete r15 done-gate.

## Live probes

- 12B default CLI, 120 tokens: coherent Rayleigh-scattering explanation;
  decode `20.8 tok/s`, prefill `102 tok/s`, total `20.3 tok/s`.
- 12B explicit temperature 1, top-k 64, top-p 0.95, 120 tokens: failed
  coherence (repeated `1`/control-like output). Its packed sampler matched the
  host sampler token-for-token for identical actual logits and draws.
- 12B greedy, 40 tokens: coherent Rayleigh-scattering / step-by-step transcript;
  decode `42.5 tok/s`, prefill `108 tok/s`.
- E2B default, 120 tokens: coherent Rayleigh-scattering explanation; decode
  `56.1 tok/s`, prefill `340 tok/s`, total `55.3 tok/s`.

## Verification

```text
GO_ROCM_RUN_HIP_TESTS=1 \
GO_ROCM_KERNEL_HSACO=/tmp/hip-12b-r15/build/kernels/rocm_kernels_gfx1101.hsaco \
go -C go test -count=1 ./engine/hip/
ok dappco.re/go/inference/engine/hip 2.627s
```

`go -C go vet ./engine/hip/` reports only the four acknowledged pre-existing
`unsafe.Pointer` findings in `hip_driver_cgo.go` at lines 1255, 1290, 1317,
and 1381. No r15 file has a vet finding.

The r12 packed fixed-logit oracle remains covered by the armed suite. The new
spread receipt unit tests are green on Linux, and the actual-logit armed probe
shows packed-device and host-reference draws agree. The remaining explicit
top-k coherence failure is therefore recorded without a mitigation.

