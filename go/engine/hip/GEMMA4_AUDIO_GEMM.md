# Gemma 4 audio rocBLAS receipt

The HIP audio tower uses the neutral `audio.GEMM` seam when the HIP driver and
rocBLAS are both available. Library loading is dynamic; a missing rocBLAS
library, unavailable driver, rejected matrix geometry, allocation failure, copy
failure, or GEMM failure falls back to the byte-identical host calculation for
that product. Portable and non-cgo builds never require rocBLAS.

## gfx1101 A/B — 12 July 2026

`TestHIPAudioGEMMHardwareParity_Good` ran the real e2b-4bit Conformer tower with
the shared 32-frame module-golden input. Both arms used the same loaded weights
and input; figures are the mean of three measured forwards after one warm
forward per arm.

| tower path | mean forward |
|---|---:|
| host float32 | 1.146884389 s |
| rocBLAS device GEMM | 0.626513030 s |

The device path was 1.83x faster at this input size, so rocBLAS remains the
default when dynamically available. Parity against the host golden reference
was cosine `1.000000000`, maximum absolute delta `0.00006484985`, across 12,288
float32 outputs, with 1,688 GEMMs accepted by rocBLAS and none rejected across
the warm and measured device forwards. The host reference separately clears
the checked-in HF golden at cosine >= 0.999.
