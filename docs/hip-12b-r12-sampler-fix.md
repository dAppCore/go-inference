# HIP 12B r12 sampler pin

Date: 2026-07-13

## Verdict

The device top-k sampler had a real correctness bug: it exponentiated raw LM-head
scores without applying Gemma 4's final logit softcap. The 262144-vocabulary
device-versus-host oracle was RED for top-k 40/64, top-p 1.0/0.95, and multiple
draws. The launch ABI now carries the softcap into
`rocm_packed_topk_sample`; the kernel applies it before temperature, top-p, and
draw mapping. The rebuilt gfx1101 oracle is GREEN across the sweep.

That fix does not clear the end-to-end done gate. The exact default CLI probe
does not call the device top-k sampler: `-temp 1` carries `TopK=0`.
`hipGemma4Q4HostSamplingRequested` accepts temperature at
`hip_tiny_model.go:2539-2545`, while
`hipGemma4Q4DeviceTopKSamplingRequested` additionally requires `TopK > 0` at
`hip_tiny_model.go:2551-2553`. `deviceCandidateSampling` is unconditionally
false at `hip_tiny_model.go:2547-2549`. The request therefore falls through to
the full-vocabulary host path, including device-hidden readback, full LM-head
projection, and full logits readback at `hip_gemma4_q4_layer.go:992-1010`.

This gate is independent of model geometry. E2B and 12B default sampled probes
take the same host path; E2B coherence is not evidence about
`rocm_packed_topk_sample`. Geometry changes the cost of the full projection but
not sampler eligibility.

## Timing receipt

Fresh clean-pack A/B, same rebuilt binary and gfx1101 HSACO, prompt
`why is the sky blue`, three output tokens:

```text
greedy:  decode 42.4 tok/s (0.047s), prefill 110 tok/s (0.182s), elapsed 1.77s
sampled: decode 0.2 tok/s (12.926s), prefill 2 tok/s (8.520s), elapsed 79.75s
```

The sampled output was `The short answer`; three tokens are too short for a
coherence verdict. The receipt localises the latency to the temperature-only,
top-k-off host route, not the corrected device top-k reduction/draw path. The
fixed-logit device oracle executes twenty reduce-and-draw cases in 0.281s on the
same box.

## Stop condition

No top-k clamp or forced 12B host/device reroute is applied. Exact top-k-off
sampling requires a full-vocabulary device distribution (or an equivalent exact
multi-stage normalisation and draw kernel); substituting top-k 40/64 would
change public sampling semantics and is the forbidden mitigation. The remaining
120-token 12B transcript, greedy byte comparison, E2B transcript, and full armed
suite are therefore not claimed as passing done gates.

## Receipts

```sh
GO_ROCM_RUN_HIP_TESTS=1 \
GO_ROCM_KERNEL_HSACO=/tmp/hip-12b-r12/build/kernels/rocm_kernels_gfx1101.hsaco \
go -C go test -count=1 ./engine/hip \
  -run '^TestHIPHardwarePackedTopKSamplerMatchesHostReference_Good$'
# ok dappco.re/go/inference/engine/hip 0.281s

GO_ROCM_KERNEL_HSACO=/tmp/hip-12b-r12/build/kernels/rocm_kernels_gfx1101.hsaco \
/tmp/hip-12b-r12/build/bin/lthn-rocm generate -temp 0 \
  -prompt 'why is the sky blue' -max-tokens 3 \
  /tmp/models/gemma-4-12B-it-4bit-clean

GO_ROCM_KERNEL_HSACO=/tmp/hip-12b-r12/build/kernels/rocm_kernels_gfx1101.hsaco \
/tmp/hip-12b-r12/build/bin/lthn-rocm generate \
  -prompt 'why is the sky blue' -max-tokens 3 \
  /tmp/models/gemma-4-12B-it-4bit-clean
```
