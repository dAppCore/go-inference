# HIP 12B r13 full-vocabulary sampling receipt

Date: 2026-07-13

## Verdict

The 200x symptom was a quadratic host sort, not the LM-head projection or the
logits readback. Replacing insertion sort with the same-order radix sort moved
the 262144-candidate sampler from seconds to 7.39 ms and the 12B default
sampling probe from 0.2 to 20.8 tok/s. E2B sampling and 12B greedy meet their
coherence and speed gates. The 12B default-sampling transcript still becomes
incoherent, so #52 is **not claimed closed**.

## Softcap ownership pin

The proposed missing-softcap RED did not exist. On the host arm,
`hipRunGemma4Q4SingleTokenForwardWithStateInternal` reads final hidden state at
`hip_gemma4_q4_layer.go:993-997`, projects the complete vocabulary at
`:999-1006`, and applies `hipGemma4Q4SoftcapLogits` exactly once at `:1007-1010`.
The resulting `current.Logits` reaches `hipGemma4Q4HostSampleResult` at
`hip_tiny_model.go:1242` and `:1428`; that sampler does not cap again.

The device top-k arm owns softcap inside
`hipRunMLXQ4ProjectionSoftcapSampleKernelWithDeviceInputBufferSuppress`
(`hip_gemma4_q4_layer.go:969`). Device greedy owns it inside the fused softcap
greedy kernels (`:982-990`). Thus every arm caps once, at its result producer.

## Timing root cause and exact fix

The r10 kernel collector measured the full-vocabulary
`rocm_mlx_q4_projection` launch below 1 ms. Source tracing then identified
`sortHIPReferenceCandidates` as insertion sort over all 262144 finite logits:
typical unsorted input requires roughly 34 billion comparisons. The logits
readback is 1 MiB; it was not the multi-second phase.

The replacement radix-sorts the identical total key: softcapped score
descending, then token ID ascending. Temperature, top-p, min-p, repeat penalty,
and draw mapping are unchanged. This is not a top-k clamp or a sampling reroute.

Target-box benchmark after the final radix implementation:

```text
BenchmarkHIPGemma4Q4HostSampleResult_FullVocabulary-32
5  7394156 ns/op  12061972 B/op  9 allocs/op
```

## Runtime gates

12B default sampling, 120 tokens:

```text
The scientific reason the sky is blue is due to a phenomenon called **Raydel Scattering**.

Here is the step-by-step breakdown of how it works:

### 1. Sunlight is a mix of all colors
White light (like sunlight) travels ==11 Identity. It is actually made up of all the colors of the visible spectrum(1.1.1/~1.1.1/1.1.1/~1.1.1/1.1.1/~1.1.1/~1.1.1/~1.1.1/

decode 20.8 tok/s (120 tok / 5.719s); prefill 41 tok/s
```

Speed is sane, but the transcript is not coherent. This is the remaining
failed done-gate.

12B greedy, 40 tokens:

```text
The sky is blue because of a phenomenon called **Rayleigh scattering**.

Here is the step-by-step explanation of how it works:

### 1. Sunlight is a mixture of all

decode 42.5 tok/s (40 tok / 0.917s); prefill 110 tok/s
```

This matches the r11-merge Rayleigh / “How it works” greedy receipt.

E2B default sampling, 40 tokens:

```text
The sky appears blue primarily because of a phenomenon called **Rayleigh scattering**, which is an effect of how **sunlight interacts with the Earth's atmosphere**.

Here is a breakdown of the process

decode 32.0 tok/s (40 tok / 1.219s); prefill 108 tok/s
```

## Verification

```sh
GO_ROCM_RUN_HIP_TESTS=1 \
GO_ROCM_KERNEL_HSACO=/tmp/hip-12b-r13/build/kernels/rocm_kernels_gfx1101.hsaco \
go -C go test -count=1 ./engine/hip/ \
  -run '^TestHIPHardwarePackedTopKSamplerMatchesHostReference_Good$'
# ok dappco.re/go/inference/engine/hip 0.205s

GO_ROCM_RUN_HIP_TESTS=1 \
GO_ROCM_KERNEL_HSACO=/tmp/hip-12b-r13/build/kernels/rocm_kernels_gfx1101.hsaco \
go -C go test -count=1 ./engine/hip/
# ok dappco.re/go/inference/engine/hip 2.632s
```

`go vet ./engine/hip/` reports only the four acknowledged pre-existing
`unsafe.Pointer` findings in `hip_driver_cgo.go` at lines 1255, 1290, 1317,
and 1381. No model files changed. The repository-wide core/go audit remains
non-compliant with pre-existing findings; the r13 files add none of its banned
imports or result-shape changes.
